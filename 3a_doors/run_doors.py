#!/usr/bin/env python3
"""Place 2-D feature deltas around every perimeter in Coords/.

The normal user edits only the four settings immediately below. One coherent
component GRIM is written to output/ for every Coords/*.txt file.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "Backend"
sys.path[:0] = [str(BACKEND), str(ROOT)]

# ── USER SETTINGS ──────────────────────────────────────────────────────────
# One entry per variable encoded in the dataset filenames:
#   (minimum, maximum)  selects the available values in that interval
#   [v1, v2, ...]       selects explicit available values
TOLERANCES = {
    "gap": (0.006, 0.017),
}
SHADOW = False
DATASETS_DIR = "Datasets"
BODY_GRIM = os.path.join("..", "2b_solve_body_hpc", "results", "body.grim")
# ───────────────────────────────────────────────────────────────────────────

# ── ADVANCED SETTINGS ──────────────────────────────────────────────────────
STUDY = ""                       # required only when Datasets has >1 study
UNITS = "meters"                 # units of Coords/*.txt and the optional STL
SKIN_TOL_M = 1e-3
SKIN_PHASE_TOL_DEG = 15.0
SPREAD_TRIALS = 0                # optional diagnostic; 0 avoids extra work
SPREAD_FREQ_GHZ = 6.0
SHADOW_BIAS_OVERRIDE_M = None    # normally leave None for automatic selection
SHADOW_CAL_AZ_STEP_DEG = 10.0
SHADOW_CAL_POINTS_PER_SEGMENT = 5
# ───────────────────────────────────────────────────────────────────────────

from grid import (  # noqa: E402
    AZIMUTHS_DEG,
    ELEVATIONS_DEG,
    FREQUENCIES_GHZ,
    POLARIZATIONS,
)
from components import keep_pols, tag_component  # noqa: E402
from delta_library import DeltaLibrary, Range, tolerance_placements  # noqa: E402
from feature_sum import (  # noqa: E402
    _attitude,
    _direction,
    _load_grim,
    directions_from_aspect_roll,
    export_radar_grim,
    load_body_profile_grim,
    sum_features,
)
from frame import (  # noqa: E402
    AXIS_AZ_DEG,
    AXIS_EL_DEG,
    ROLL_DEG,
    scale_for,
    to_axis_frame,
)
from grim_io import _save_grim_npz  # noqa: E402
from line_expand import (  # noqa: E402
    C0,
    dbsm,
    perimeter_surface_deviation,
    read_perimeter_txt,
    surface_of_revolution_normal,
)
from workflow_provenance import (  # noqa: E402
    backend_source_fingerprint,
    runtime_environment_fingerprint,
    sha256_file,
)
from shadow_bias import conservative_occluder  # noqa: E402

SCALE = scale_for(UNITS)
COORDS_DIR = HERE / "Coords"
OUTPUT_DIR = HERE / "output"


def _here(path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (HERE / value).resolve()


def _pick_datasets(library):
    keys = library.keys()
    for key in keys:
        if key not in TOLERANCES:
            raise SystemExit(
                f"Dataset filenames contain variable {key!r}, but TOLERANCES "
                "does not define it."
            )
    studies = sorted({entry.study for entry in library.entries})
    selected = library
    if STUDY:
        if STUDY not in studies:
            raise SystemExit(f"STUDY={STUDY!r}; available studies are {studies}.")
        selected = DeltaLibrary(
            [entry for entry in library.entries if entry.study == STUDY],
            library.decimals,
            library.root,
            library.unindexed,
        )
    elif len(studies) > 1:
        raise SystemExit(
            f"Datasets contains multiple studies {studies}; set advanced "
            "STUDY to the one intended for this door run."
        )
    for key, specification in TOLERANCES.items():
        if key not in keys:
            raise SystemExit(
                f"TOLERANCES defines {key!r}, but filenames contain {keys}."
            )
        if isinstance(specification, tuple) and len(specification) == 2:
            wanted = Range(*specification)
        else:
            wanted = [float(value) for value in np.atleast_1d(specification)]
        try:
            selected = selected.select(**{key: wanted})
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return selected.entries, (STUDY or studies[0])


def _check_on_skin(perimeter, profile, name):
    offset = perimeter_surface_deviation(
        perimeter, profile, samples_per_segment=33
    )
    wavelength = C0 / (float(np.max(FREQUENCIES_GHZ)) * 1e9)
    phase_limit = float(SKIN_PHASE_TOL_DEG) * wavelength / 720.0
    limit = min(float(SKIN_TOL_M), phase_limit)
    if offset > limit:
        phase = 720.0 * offset / wavelength
        raise SystemExit(
            f"{name}: coordinates are {offset*1e3:.2f} mm off the body skin "
            f"({phase:.1f}° worst-case two-way phase; allowed "
            f"{limit*1e3:.3f} mm). Check BODY_GRIM, UNITS, frame, and coordinate "
            "density."
        )
    return offset


def _sample_perimeters(perimeters):
    count = max(2, int(SHADOW_CAL_POINTS_PER_SEGMENT))
    fractions = np.linspace(0.0, 1.0, count)
    points = []
    for perimeter in perimeters:
        p0 = perimeter[:, 0, :]
        vector = perimeter[:, 1, :] - p0
        points.append(
            (p0[:, None, :] + fractions[None, :, None] * vector[:, None, :])
            .reshape(-1, 3)
        )
    return np.unique(np.round(np.vstack(points), 12), axis=0)


def _calibration_directions():
    step = float(SHADOW_CAL_AZ_STEP_DEG)
    if not np.isfinite(step) or step <= 0.0 or step > 90.0:
        raise SystemExit("SHADOW_CAL_AZ_STEP_DEG must be in (0, 90].")
    azimuths = np.arange(0.0, 360.0, step)
    elevations = np.unique(np.asarray(ELEVATIONS_DEG, dtype=float))
    rotation, _axis = _attitude(AXIS_AZ_DEG, AXIS_EL_DEG, ROLL_DEG)
    directions = [
        _direction(float(azimuth), float(elevation)) @ rotation
        for azimuth in azimuths
        for elevation in elevations
    ]
    return np.asarray(directions, dtype=float)


def _automatic_occluder(stl_path, perimeters, profile):
    from occluder import read_stl

    triangles = to_axis_frame(read_stl(str(stl_path)))
    points = _sample_perimeters(perimeters)
    normals = surface_of_revolution_normal(profile)(points)
    directions = _calibration_directions()
    try:
        occluder, info = conservative_occluder(
            triangles,
            scale=SCALE,
            points=points,
            normals=normals,
            directions=directions,
            override_m=SHADOW_BIAS_OVERRIDE_M,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Shadow bias {info['mode']}: mesh default "
        f"{info['mesh_default_bias_m']*1e3:.4g} mm; selected "
        f"{info['selected_bias_m']*1e3:.4g} mm"
    )
    if info.get("changes_default_to_validation", 0):
        print(
            f"  warning: {info['changes_default_to_validation']} lit "
            "visibility samples change "
            "above the mesh default. The automatic selector did not increase "
            "the bias because those changes may be real nearby blockers."
        )
    return occluder, info


def _arrangement_spread(perimeter, entries, order_key, profile, occluder):
    if not SPREAD_TRIALS or len(entries) <= 1:
        return
    directions, _aspect, _roll = directions_from_aspect_roll(
        np.arange(30.0, 150.1, 15.0), np.arange(0.0, 360.0, 15.0)
    )
    rng = np.random.default_rng(20260725)
    peaks = []
    for _index in range(int(SPREAD_TRIALS)):
        placements = tolerance_placements(
            perimeter, entries, order="random", rng=rng, kind="delta"
        )
        result = sum_features(
            None,
            placements,
            directions,
            float(SPREAD_FREQ_GHZ),
            generatrix=profile,
            occluder=occluder,
            mode="coherent",
        )
        peaks.append(float(np.max(result["dbsm_vv"])))
    print(
        f"  arrangement uncertainty at {SPREAD_FREQ_GHZ:g} GHz: "
        f"{min(peaks):+.1f} to {max(peaks):+.1f} dBsm"
    )


def _embed_provenance(path, provenance):
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.array(source[key], copy=True) for key in source.files}
    payload["component_provenance_json"] = np.asarray(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    )
    _save_grim_npz(payload, path)


def _refuse_unexpected_grims(output_dir, expected):
    """Keep a changed coordinate inventory from silently mixing old outputs."""
    root = Path(output_dir)
    wanted = set(expected)
    stale = sorted(
        path.name for path in root.glob("*.grim") if path.name not in wanted
    )
    if stale:
        raise SystemExit(
            f"{root.name}/ contains stale GRIMs {stale}; move them out before "
            "this run."
        )


def main():
    dataset_dir = _here(DATASETS_DIR)
    body_grim = _here(BODY_GRIM)
    dataset_paths = sorted(dataset_dir.glob("*.grim"))
    coord_paths = sorted(COORDS_DIR.glob("*.txt"))
    if not dataset_paths:
        raise SystemExit(f"No .grim datasets found in {dataset_dir}.")
    if not coord_paths:
        raise SystemExit(f"No coordinate files found in {COORDS_DIR}.")
    if not body_grim.is_file():
        raise SystemExit(f"Body GRIM not found: {body_grim}")

    library = DeltaLibrary.from_dir(str(dataset_dir))
    if library.unindexed:
        details = "; ".join(
            f"{Path(path).name}: {reason}"
            for path, reason in library.unindexed[:8]
        )
        raise SystemExit(f"Unindexed datasets: {details}")
    entries, study = _pick_datasets(library)
    if not entries:
        raise SystemExit("TOLERANCES selected no datasets.")
    profile = load_body_profile_grim(str(body_grim))
    perimeters = {
        path: to_axis_frame(read_perimeter_txt(str(path), scale=SCALE))
        for path in coord_paths
    }
    offsets = {
        path: _check_on_skin(perimeter, profile, path.name)
        for path, perimeter in perimeters.items()
    }

    occluder = None
    shadow_info = {"mode": "disabled", "selected_bias_m": None}
    stl_path = None
    if SHADOW:
        stls = sorted(HERE.glob("*.stl"))
        if len(stls) != 1:
            raise SystemExit(
                f"SHADOW=True requires exactly one STL beside run_doors.py; "
                f"found {[path.name for path in stls]}."
            )
        stl_path = stls[0]
        occluder, shadow_info = _automatic_occluder(
            stl_path, list(perimeters.values()), profile
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = {f"{path.stem}.grim" for path in coord_paths}
    _refuse_unexpected_grims(OUTPUT_DIR, expected)

    order_key = next(iter(TOLERANCES))
    shared_provenance = {
        "schema": "ghost.workflow.door-component.v2",
        "study": study,
        "tolerances": TOLERANCES,
        "datasets_sha256": {
            path.name: sha256_file(path) for path in dataset_paths
        },
        "selected_datasets": sorted(Path(entry.path).name for entry in entries),
        "body_grim": body_grim.name,
        "body_grim_sha256": sha256_file(body_grim),
        "shadow": bool(SHADOW),
        "shadow_calibration": shadow_info,
        "shadow_mesh_sha256": (
            None if stl_path is None else sha256_file(stl_path)
        ),
        "grid": {
            "frequencies_ghz": [float(value) for value in FREQUENCIES_GHZ],
            "azimuths_deg": [float(value) for value in AZIMUTHS_DEG],
            "elevations_deg": [float(value) for value in ELEVATIONS_DEG],
            "polarizations": [str(value) for value in POLARIZATIONS],
        },
        "advanced": {
            "units": UNITS,
            "skin_tol_m": SKIN_TOL_M,
            "skin_phase_tol_deg": SKIN_PHASE_TOL_DEG,
        },
        "workflow_source_sha256": backend_source_fingerprint(
            str(BACKEND), {"3a_doors/run_doors.py": str(Path(__file__).resolve())}
        ),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
    }

    print(
        f"Step 3a: {len(coord_paths)} coordinate file(s), {len(entries)} "
        f"dataset(s), study {study}, shadow={SHADOW}"
    )
    for coord_path, perimeter in perimeters.items():
        placements = tolerance_placements(
            perimeter, entries, order_by=order_key, kind="delta"
        )
        destination = OUTPUT_DIR / coord_path.stem
        output = export_radar_grim(
            str(destination),
            bor_result=None,
            placements=placements,
            generatrix=profile,
            occluder=occluder,
            frequencies_ghz=FREQUENCIES_GHZ,
            azimuths_deg=AZIMUTHS_DEG,
            elevations_deg=ELEVATIONS_DEG,
            axis_az_deg=AXIS_AZ_DEG,
            axis_el_deg=AXIS_EL_DEG,
            roll_deg=ROLL_DEG,
            history=(
                f"step 3a {coord_path.stem}; tolerances={TOLERANCES}; "
                f"shadow={SHADOW}"
            ),
        )
        keep_pols(output, POLARIZATIONS)
        tag_component(
            output,
            "coherent",
            note="line-expanded door delta; common phase reference",
        )
        provenance = dict(
            shared_provenance,
            coordinate_file=coord_path.name,
            coordinate_sha256=sha256_file(coord_path),
            skin_offset_m=float(offsets[coord_path]),
        )
        _embed_provenance(output, provenance)
        payload = _load_grim(output)
        pols = [str(value) for value in payload["polarizations"]]
        peaks = {
            pol: dbsm(np.max(np.asarray(payload["rcs_power"])[..., index]))
            for index, pol in enumerate(pols)
        }
        print(
            f"  {Path(output).name}: {offsets[coord_path]*1e3:.3f} mm off skin; "
            + ", ".join(f"{pol} {value:+.1f} dBsm" for pol, value in peaks.items())
        )
        _arrangement_spread(
            perimeter, entries, order_key, profile, occluder
        )

    print(f"Wrote {len(coord_paths)} GRIM file(s) to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
