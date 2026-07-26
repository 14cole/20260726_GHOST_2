#!/usr/bin/env python3
"""Place compact 3-D feature patterns at every row of each Coords CSV.

Datasets contains reusable installed-feature-minus-clean-skin 3-D patterns.
Each Coords/<name>.csv is coherently accumulated into Outputs/<name>.grim.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "Backend"
sys.path[:0] = [str(BACKEND), str(ROOT)]

# ── USER SETTINGS ──────────────────────────────────────────────────────────
# One entry per variable encoded in the dataset filenames:
#   (minimum, maximum)  selects every available value in that interval
#   [v1, v2, ...]       selects explicit available values
TOLERANCES = {
    "scale": (1.0, 1.0),
}
SHADOW = False
DATASETS_DIR = "Datasets"
BODY_GRIM = os.path.join("..", "2b_solve_body_hpc", "results", "body.grim")
# ───────────────────────────────────────────────────────────────────────────

# ── ADVANCED SETTINGS ──────────────────────────────────────────────────────
STUDY = ""                         # required only when Datasets has >1 study
UNITS = "meters"                   # units of CSV coordinates and optional STL
DEFAULT_ROLL_REF_CAD = (0.0, 1.0, 0.0)
SKIN_TOL_M = 2e-3
SKIN_PHASE_TOL_DEG = 15.0
NORMAL_TOL_DEG = 15.0
SURFACE_CHECK = "bor"              # "bor" or "supplied" for non-BoR surfaces
SHADOW_BIAS_OVERRIDE_M = None
SHADOW_CAL_AZ_STEP_DEG = 10.0
# ───────────────────────────────────────────────────────────────────────────

from grid import (  # noqa: E402
    AZIMUTHS_DEG,
    ELEVATIONS_DEG,
    FREQUENCIES_GHZ,
    POLARIZATIONS,
)
from components import keep_pols, tag_component  # noqa: E402
from delta_library import DeltaLibrary, Range  # noqa: E402
from feature_sum import (  # noqa: E402
    _attitude,
    _direction,
    _load_grim,
    export_radar_grim,
    load_body_profile_grim,
    prepare_point_pattern,
    surface_of_revolution_distance,
)
from frame import (  # noqa: E402
    AXIS_AZ_DEG,
    AXIS_EL_DEG,
    ROLL_DEG,
    scale_for,
    to_axis_frame,
)
from grim_io import _save_grim_npz  # noqa: E402
from line_expand import C0, dbsm, surface_of_revolution_normal  # noqa: E402
from shadow_bias import conservative_occluder  # noqa: E402
from workflow_provenance import (  # noqa: E402
    backend_source_fingerprint,
    runtime_environment_fingerprint,
    sha256_file,
)

SCALE = scale_for(UNITS)
COORDS_DIR = HERE / "Coords"
OUTPUTS_DIR = HERE / "Outputs"
OUTPUT_SCHEMA = "ghost.workflow.compact-component.v2"


def _here(path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (HERE / value).resolve()


def _pick_datasets(library):
    keys = library.keys()
    if set(keys) != set(TOLERANCES):
        raise SystemExit(
            f"TOLERANCES keys {sorted(TOLERANCES)} do not match dataset "
            f"filename variables {keys}."
        )
    studies = sorted(
        {entry.study for entry in library.entries},
        key=lambda value: "" if value is None else str(value),
    )
    selected = library
    if STUDY:
        try:
            selected = selected.select(study=STUDY)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        study = STUDY
    elif len(studies) == 1:
        study = studies[0]
    else:
        raise SystemExit(
            f"Datasets contains multiple studies {studies}; set advanced "
            "STUDY to the compact-feature family intended for this run."
        )
    for key, specification in TOLERANCES.items():
        wanted = (
            Range(*specification)
            if isinstance(specification, tuple) and len(specification) == 2
            else [float(value) for value in np.atleast_1d(specification)]
        )
        try:
            selected = selected.select(**{key: wanted})
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    # A revision is a replacement solve, not another tolerance population.
    newest = {}
    for entry in selected.entries:
        point = tuple(sorted(entry.params.items()))
        if point not in newest or entry.rev > newest[point].rev:
            newest[point] = entry
    entries = sorted(
        newest.values(),
        key=lambda entry: tuple(entry.params[key] for key in sorted(keys)),
    )
    return entries, study


def _assigned_entries(entries, coordinate_count, label="coordinate file"):
    """Balanced deterministic realization of a selected tolerance population."""
    if coordinate_count < len(entries):
        raise SystemExit(
            f"{label} has {coordinate_count} coordinate(s), but TOLERANCES "
            f"selected {len(entries)} datasets. Narrow the selection or provide "
            "at least one coordinate per variant."
        )
    return [entries[index % len(entries)] for index in range(coordinate_count)]


def _normalized_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


_COLUMN_ALIASES = {
    "x": ("x",),
    "y": ("y",),
    "z": ("z",),
    "nx": ("nx", "xnormal", "normalx"),
    "ny": ("ny", "ynormal", "normaly"),
    "nz": ("nz", "znormal", "normalz"),
    "rx": ("rx", "xroll", "rollx", "xref", "refx"),
    "ry": ("ry", "yroll", "rolly", "yref", "refy"),
    "rz": ("rz", "zroll", "rollz", "zref", "refz"),
}


def read_placement_csv(path):
    """Read x/y/z plus optional outward normal and pattern clocking vectors."""
    source = _here(path)
    try:
        with source.open(newline="", encoding="utf-8-sig") as stream:
            raw_rows = [
                (number, [cell.strip() for cell in row])
                for number, row in enumerate(csv.reader(stream), 1)
                if row
                and any(cell.strip() for cell in row)
                and not row[0].lstrip().startswith("#")
            ]
    except OSError as exc:
        raise SystemExit(f"Cannot read coordinates {source}: {exc}") from exc
    if not raw_rows:
        raise SystemExit(f"{source}: coordinate CSV is empty.")

    first_number, first = raw_rows[0]
    try:
        [float(value) for value in first]
        header = None
    except ValueError:
        header = first
        raw_rows = raw_rows[1:]
    if not raw_rows:
        raise SystemExit(f"{source}: coordinate CSV has no data rows.")

    if header is None:
        width = len(first)
        if width not in (3, 6, 9):
            raise SystemExit(
                f"{source}:{first_number}: headerless rows require 3, 6, or "
                f"9 columns; found {width}."
            )
        keys = ("x", "y", "z", "nx", "ny", "nz", "rx", "ry", "rz")[:width]
        indices = {key: index for index, key in enumerate(keys)}
    else:
        normalized = [_normalized_header(value) for value in header]
        indices = {}
        for key, aliases in _COLUMN_ALIASES.items():
            matches = [
                index for index, value in enumerate(normalized)
                if value in aliases
            ]
            if len(matches) > 1:
                raise SystemExit(f"{source}: header provides {key!r} twice.")
            if matches:
                indices[key] = matches[0]
        missing = [key for key in ("x", "y", "z") if key not in indices]
        if missing:
            raise SystemExit(f"{source}: header is missing {missing}.")
        for group, label in (
            (("nx", "ny", "nz"), "normal"),
            (("rx", "ry", "rz"), "roll-reference"),
        ):
            present = [key in indices for key in group]
            if any(present) and not all(present):
                raise SystemExit(
                    f"{source}: provide all three {label} columns or none."
                )

    placements = []
    for line_number, row in raw_rows:
        if max(indices.values()) >= len(row):
            raise SystemExit(f"{source}:{line_number}: row has too few columns.")
        try:
            values = {
                key: float(row[index]) for key, index in indices.items()
            }
        except ValueError as exc:
            raise SystemExit(
                f"{source}:{line_number}: placement values must be numeric."
            ) from exc
        if not np.all(np.isfinite(list(values.values()))):
            raise SystemExit(
                f"{source}:{line_number}: placement has NaN or infinity."
            )
        placements.append({
            "line": line_number,
            "location_cad": tuple(values[key] for key in ("x", "y", "z")),
            "normal_cad": (
                tuple(values[key] for key in ("nx", "ny", "nz"))
                if "nx" in values else None
            ),
            "roll_ref_cad": (
                tuple(values[key] for key in ("rx", "ry", "rz"))
                if "rx" in values else None
            ),
        })
    return source, placements


def _unit_vector(value, label):
    vector = np.asarray(value, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if (
        vector.shape != (3,)
        or not np.all(np.isfinite(vector))
        or magnitude <= 1e-12
    ):
        raise SystemExit(f"{label} must be one finite nonzero 3-vector.")
    return vector / magnitude


def _skin_limit_m():
    skin = float(SKIN_TOL_M)
    phase = float(SKIN_PHASE_TOL_DEG)
    normal = float(NORMAL_TOL_DEG)
    if (
        not np.isfinite(skin)
        or skin < 0.0
        or not np.isfinite(phase)
        or phase < 0.0
        or not np.isfinite(normal)
        or not 0.0 <= normal <= 90.0
    ):
        raise SystemExit("Invalid advanced skin, phase, or normal tolerance.")
    wavelength = C0 / (float(np.max(FREQUENCIES_GHZ)) * 1e9)
    return min(skin, phase * wavelength / 720.0), wavelength


def _validated_point(profile, pattern, row, label):
    location = to_axis_frame(np.asarray(row["location_cad"], float) * SCALE)
    derived = None
    offset = None
    if SURFACE_CHECK == "bor":
        offset = float(
            surface_of_revolution_distance(profile, location[None, :])[0]
        )
        limit, wavelength = _skin_limit_m()
        if not np.isfinite(offset) or offset > limit:
            raise SystemExit(
                f"{label}: coordinate is {offset*1e3:.3f} mm off the BoR skin "
                f"({720.0*offset/wavelength:.1f} deg worst-case two-way phase; "
                f"allowed {limit*1e3:.3f} mm)."
            )
        derived = _unit_vector(
            surface_of_revolution_normal(profile)(location[None, :])[0],
            f"{label} derived normal",
        )
    elif SURFACE_CHECK != "supplied":
        raise SystemExit("SURFACE_CHECK must be 'bor' or 'supplied'.")

    if row["normal_cad"] is None:
        if derived is None:
            raise SystemExit(
                f"{label}: SURFACE_CHECK='supplied' requires nx,ny,nz."
            )
        normal = derived
        normal_source = "derived_bor"
    else:
        normal = _unit_vector(
            to_axis_frame(np.asarray(row["normal_cad"], float)),
            f"{label} supplied normal",
        )
        if derived is not None:
            difference = float(
                np.degrees(
                    np.arccos(np.clip(float(normal @ derived), -1.0, 1.0))
                )
            )
            if difference > float(NORMAL_TOL_DEG):
                raise SystemExit(
                    f"{label}: supplied normal is {difference:.2f} deg from "
                    f"the outward BoR skin normal; allowed {NORMAL_TOL_DEG:g}."
                )
        normal_source = "supplied"

    roll_cad = (
        row["roll_ref_cad"]
        if row["roll_ref_cad"] is not None
        else DEFAULT_ROLL_REF_CAD
    )
    roll = _unit_vector(
        to_axis_frame(np.asarray(roll_cad, float)),
        f"{label} roll reference",
    )
    if np.linalg.norm(roll - (roll @ normal) * normal) <= 1e-9:
        raise SystemExit(f"{label}: roll reference is parallel to its normal.")
    return {
        "pattern": pattern,
        "location": tuple(location),
        "aperture_normal": tuple(normal),
        "roll_ref": tuple(roll),
    }, {
        "csv_line": int(row["line"]),
        "normal_source": normal_source,
        "skin_offset_m": offset,
    }


def _calibration_directions():
    step = float(SHADOW_CAL_AZ_STEP_DEG)
    if not np.isfinite(step) or not 0.0 < step <= 90.0:
        raise SystemExit("SHADOW_CAL_AZ_STEP_DEG must be in (0, 90].")
    rotation, _axis = _attitude(AXIS_AZ_DEG, AXIS_EL_DEG, ROLL_DEG)
    return np.asarray([
        _direction(float(azimuth), float(elevation)) @ rotation
        for azimuth in np.arange(0.0, 360.0, step)
        for elevation in np.unique(np.asarray(ELEVATIONS_DEG, dtype=float))
    ])


def _automatic_occluder(points):
    from occluder import read_stl

    stls = sorted(HERE.glob("*.stl"))
    if len(stls) != 1:
        raise SystemExit(
            f"SHADOW=True requires exactly one STL beside run_place_3d.py; "
            f"found {[path.name for path in stls]}."
        )
    triangles = to_axis_frame(read_stl(str(stls[0])))
    locations = np.asarray([point["location"] for point in points], dtype=float)
    normals = np.asarray(
        [point["aperture_normal"] for point in points], dtype=float
    )
    try:
        occluder, info = conservative_occluder(
            triangles,
            scale=SCALE,
            points=locations,
            normals=normals,
            directions=_calibration_directions(),
            override_m=SHADOW_BIAS_OVERRIDE_M,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Shadow bias {info['mode']}: default "
        f"{info['mesh_default_bias_m']*1e3:.4g} mm; selected "
        f"{info['selected_bias_m']*1e3:.4g} mm"
    )
    if info.get("changes_default_to_validation", 0):
        print(
            f"  warning: {info['changes_default_to_validation']} visibility "
            "samples change above the default; the bias was not enlarged."
        )
    return occluder, info, stls[0]


def _embed_provenance(path, provenance):
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    payload["component_provenance_json"] = np.asarray(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    )
    _save_grim_npz(payload, path)


def _refuse_unexpected_grims(output_dir, expected):
    wanted = set(expected)
    stale = sorted(
        path.name for path in Path(output_dir).glob("*.grim")
        if path.name not in wanted
    )
    if stale:
        raise SystemExit(
            f"Outputs/ contains stale GRIMs {stale}; move them out before "
            "this run."
        )


def main():
    dataset_dir = _here(DATASETS_DIR)
    body_grim = _here(BODY_GRIM)
    coord_paths = sorted(COORDS_DIR.glob("*.csv"))
    if not coord_paths:
        raise SystemExit(f"No coordinate CSV files found in {COORDS_DIR}.")
    if not body_grim.is_file():
        raise SystemExit(f"Body GRIM not found: {body_grim}")

    try:
        library = DeltaLibrary.from_dir(str(dataset_dir))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if library.unindexed:
        details = "; ".join(
            f"{Path(path).name}: {reason}"
            for path, reason in library.unindexed[:8]
        )
        raise SystemExit(f"Unindexed compact datasets: {details}")
    entries, study = _pick_datasets(library)
    if not entries:
        raise SystemExit("TOLERANCES selected no compact-feature datasets.")
    prepared = {}
    for entry in entries:
        try:
            prepared[entry.path] = prepare_point_pattern(entry.path)
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"Invalid compact 3-D dataset {entry.path}: {exc}"
            ) from exc

    profile = load_body_profile_grim(str(body_grim))
    coordinate_tables = {}
    all_points = []
    all_records = {}
    for coord_path in coord_paths:
        _source, rows = read_placement_csv(coord_path)
        assigned = _assigned_entries(entries, len(rows), coord_path.name)
        points = []
        records = []
        occupied = {}
        for row, entry in zip(rows, assigned):
            label = f"{coord_path.name}:{row['line']}"
            point, record = _validated_point(
                profile, prepared[entry.path], row, label
            )
            key = tuple(np.round(np.asarray(point["location"]), 12))
            if key in occupied:
                raise SystemExit(
                    f"{label}: duplicate coordinate already used at "
                    f"line {occupied[key]}."
                )
            occupied[key] = row["line"]
            points.append(point)
            records.append(dict(
                record,
                dataset=Path(entry.path).name,
                parameters=entry.params,
            ))
        coordinate_tables[coord_path] = points
        all_points.extend(points)
        all_records[coord_path] = records

    occluder = None
    shadow_info = {"mode": "disabled", "selected_bias_m": None}
    stl_path = None
    if SHADOW:
        occluder, shadow_info, stl_path = _automatic_occluder(all_points)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    expected = {f"{path.stem}.grim" for path in coord_paths}
    _refuse_unexpected_grims(OUTPUTS_DIR, expected)
    shared = {
        "schema": OUTPUT_SCHEMA,
        "study": study,
        "tolerances": TOLERANCES,
        "selected_datasets": [Path(entry.path).name for entry in entries],
        "datasets_sha256": {
            Path(entry.path).name: sha256_file(entry.path) for entry in entries
        },
        "assignment": (
            "selected datasets sorted by filename variables and cycled in "
            "CSV row order"
        ),
        "body_grim": body_grim.name,
        "body_grim_sha256": sha256_file(body_grim),
        "shadow": bool(SHADOW),
        "shadow_calibration": shadow_info,
        "shadow_mesh_sha256": (
            None if stl_path is None else sha256_file(stl_path)
        ),
        "surface_check": SURFACE_CHECK,
        "grid": {
            "frequencies_ghz": [float(value) for value in FREQUENCIES_GHZ],
            "azimuths_deg": [float(value) for value in AZIMUTHS_DEG],
            "elevations_deg": [float(value) for value in ELEVATIONS_DEG],
            "polarizations": [str(value) for value in POLARIZATIONS],
        },
        "advanced": {
            "units": UNITS,
            "default_roll_ref_cad": DEFAULT_ROLL_REF_CAD,
            "skin_tol_m": SKIN_TOL_M,
            "skin_phase_tol_deg": SKIN_PHASE_TOL_DEG,
            "normal_tol_deg": NORMAL_TOL_DEG,
        },
        "workflow_source_sha256": backend_source_fingerprint(
            str(BACKEND),
            {"3c_add_cavity/run_place_3d.py": str(Path(__file__).resolve())},
        ),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
    }

    print(
        f"Step 3c: {len(coord_paths)} coordinate file(s), {len(entries)} "
        f"dataset variant(s), study {study}, shadow={SHADOW}"
    )
    for coord_path in coord_paths:
        points = coordinate_tables[coord_path]
        out = export_radar_grim(
            str(OUTPUTS_DIR / coord_path.stem),
            bor_result=None,
            placements=[],
            points=points,
            generatrix=profile,
            occluder=occluder,
            frequencies_ghz=FREQUENCIES_GHZ,
            azimuths_deg=AZIMUTHS_DEG,
            elevations_deg=ELEVATIONS_DEG,
            axis_az_deg=AXIS_AZ_DEG,
            axis_el_deg=AXIS_EL_DEG,
            roll_deg=ROLL_DEG,
            history=(
                f"step 3c {coord_path.stem}; {len(points)} compact placements; "
                f"tolerances={TOLERANCES}; shadow={SHADOW}"
            ),
        )
        keep_pols(out, POLARIZATIONS)
        tag_component(
            out,
            "coherent",
            note=(
                f"{len(points)} compact 3-D patterns; rotation and two-way "
                "placement phase tracked"
            ),
        )
        provenance = dict(
            shared,
            coordinate_file=coord_path.name,
            coordinate_sha256=sha256_file(coord_path),
            placement_count=len(points),
            derived_normal_count=sum(
                record["normal_source"] == "derived_bor"
                for record in all_records[coord_path]
            ),
            supplied_normal_count=sum(
                record["normal_source"] == "supplied"
                for record in all_records[coord_path]
            ),
            dataset_assignment=[
                {
                    "csv_line": record["csv_line"],
                    "dataset": record["dataset"],
                    "parameters": record["parameters"],
                }
                for record in all_records[coord_path]
            ],
        )
        _embed_provenance(out, provenance)
        payload = _load_grim(out)
        pols = [str(value) for value in payload["polarizations"]]
        peaks = {
            pol: dbsm(np.max(np.asarray(payload["rcs_power"])[..., index]))
            for index, pol in enumerate(pols)
        }
        print(
            f"  {Path(out).name}: {len(points)} placement(s); "
            + ", ".join(
                f"{pol} {value:+.1f} dBsm" for pol, value in peaks.items()
            )
        )
    print(f"Wrote {len(coord_paths)} GRIM file(s) to {OUTPUTS_DIR}.")


if __name__ == "__main__":
    main()
