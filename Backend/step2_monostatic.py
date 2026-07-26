"""Shared implementation for simplified local and SLURM BoR body runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Iterable

import numpy as np

from feature_sum import (
    geometry_input_fingerprint,
    outer_generatrix,
    save_body_grim,
    solve_vehicle_body,
)
from geometry_io import build_geometry_snapshot, parse_geometry
from workflow_provenance import (
    backend_source_fingerprint,
    runtime_environment_fingerprint,
    stable_json_fingerprint,
)


def validate_config(frequencies: Iterable[float], aspects: Iterable[float]) -> tuple[list[float], list[float]]:
    freqs = [float(value) for value in frequencies]
    angles = [float(value) for value in aspects]
    if (
        not freqs
        or not all(np.isfinite(value) and value > 0.0 for value in freqs)
        or len(set(freqs)) != len(freqs)
    ):
        raise ValueError("FREQUENCIES_GHZ must be positive, finite, and unique.")
    if (
        not angles
        or not all(np.isfinite(value) and 0.0 <= value <= 180.0 for value in angles)
        or len(set(angles)) != len(angles)
    ):
        raise ValueError("Body aspects must be unique finite values in [0, 180].")
    return freqs, angles


def discover_jobs(step_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = Path(step_dir).resolve()
    geometry_dir = root / "geometries"
    result_dir = root / "results"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for geometry in sorted(geometry_dir.glob("*.geo")):
        jobs.append(
            {
                "geometry": str(geometry.resolve()),
                "name": geometry.stem,
                "output": str((result_dir / f"{geometry.stem}.grim").resolve()),
            }
        )
    return jobs


def prepare_jobs(
    jobs: list[dict[str, Any]],
    *,
    frequencies: list[float],
    aspects: list[float],
    geometry_units: str,
    runner_path: str,
) -> list[dict[str, Any]]:
    backend = Path(__file__).resolve().parent
    source_sha = backend_source_fingerprint(
        str(backend), {"step2_runner.py": str(Path(runner_path).resolve())}
    )
    runtime_sha = runtime_environment_fingerprint()
    prepared = []
    for original in jobs:
        job = dict(original)
        spec = {
            "schema": "ghost.workflow.bor-body-unit.v1",
            "geometry_input_sha256": geometry_input_fingerprint(
                job["geometry"], geometry_units
            ),
            "solver_source_sha256": source_sha,
            "runtime_environment_sha256": runtime_sha,
            "geometry_units": str(geometry_units).strip().lower(),
            "frequencies_ghz": frequencies,
            "aspects_deg": aspects,
        }
        spec["unit_sha256"] = stable_json_fingerprint(spec)
        job["specification"] = spec
        prepared.append(job)
    return prepared


def _stored_sha(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return str(np.asarray(payload["run_solve_spec_sha256"]).reshape(()).item())
    except (OSError, KeyError, TypeError, ValueError):
        return ""


def solve_job(
    job: dict[str, Any],
    *,
    frequencies: list[float],
    aspects: list[float],
    geometry_units: str,
    workers_per_body: int,
    force: bool,
) -> tuple[str, str]:
    output = Path(job["output"])
    expected = str(job["specification"]["unit_sha256"])
    if output.exists() and not force:
        if _stored_sha(output) == expected:
            return "skipped", str(output)
        raise RuntimeError(
            f"{output} exists but does not match the current geometry or "
            "settings. Move it aside or set FORCE=True."
        )
    geometry = Path(job["geometry"])
    snapshot = build_geometry_snapshot(
        *parse_geometry(geometry.read_text(encoding="utf-8"))
    )
    profile = outer_generatrix(snapshot, geometry_units)
    bodies, _profile = solve_vehicle_body(
        str(geometry),
        frequencies,
        aspects,
        geometry_units=geometry_units,
        cfie_alpha=0.5,
        workers=int(workers_per_body),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".grim", dir=output.parent
    )
    os.close(descriptor)
    try:
        saved = save_body_grim(
            bodies,
            temporary,
            source_path=str(geometry),
            history=f"simplified step 2 body={geometry.name}",
            geometry_input_sha256=job["specification"]["geometry_input_sha256"],
            solver_source_sha256=job["specification"]["solver_source_sha256"],
            runtime_environment_sha256=job["specification"]["runtime_environment_sha256"],
            run_solve_spec_sha256=expected,
            body_profile=profile,
        )
        os.replace(saved, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return "written", str(output)


def solve_job_catching(args):
    job, kwargs = args
    try:
        status, path = solve_job(job, **kwargs)
        return "ok", status, path, job
    except Exception:
        return "error", traceback.format_exc(), "", job
