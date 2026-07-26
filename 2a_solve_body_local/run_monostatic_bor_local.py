#!/usr/bin/env python3
"""Solve every geometries/*.geo BoR body directly into results/*.grim."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "Backend"), str(ROOT)]

from frame import AXIS_AZ_DEG, AXIS_EL_DEG  # noqa: E402
from feature_sum import radar_grid_aspects  # noqa: E402
from grid import AZIMUTHS_DEG, ELEVATIONS_DEG  # noqa: E402
from step2_monostatic import (  # noqa: E402
    discover_jobs,
    prepare_jobs,
    solve_job,
    validate_config,
)

# ── USER SETTINGS ──────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
ASPECT_STEP_DEG = None         # None = exact aspects required by grid.py
GEOMETRY_UNITS = "meters"
MAX_CONCURRENT_BODIES = 1
WORKERS_PER_BODY = 4
FORCE = False
# ───────────────────────────────────────────────────────────────────────────


def _aspects():
    required = radar_grid_aspects(
        AZIMUTHS_DEG, ELEVATIONS_DEG, AXIS_AZ_DEG, AXIS_EL_DEG
    )
    if ASPECT_STEP_DEG is None:
        return list(required)
    import numpy as np
    step = float(ASPECT_STEP_DEG)
    values = np.arange(0.0, 180.0 + 0.5 * step, step)
    missing = [
        value for value in required
        if not np.any(np.isclose(values, value, rtol=0.0, atol=1e-9))
    ]
    if missing:
        raise SystemExit(
            "ASPECT_STEP_DEG omits exact grid.py aspects; use None."
        )
    return list(values)


def main():
    try:
        frequencies, aspects = validate_config(FREQUENCIES_GHZ, _aspects())
        jobs = discover_jobs(HERE)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not jobs:
        raise SystemExit("No body .geo files found in geometries/.")
    jobs = prepare_jobs(
        jobs,
        frequencies=frequencies,
        aspects=aspects,
        geometry_units=GEOMETRY_UNITS,
        runner_path=__file__,
    )
    concurrent = max(1, min(int(MAX_CONCURRENT_BODIES), len(jobs)))
    kwargs = {
        "frequencies": frequencies,
        "aspects": aspects,
        "geometry_units": GEOMETRY_UNITS,
        "workers_per_body": WORKERS_PER_BODY,
        "force": FORCE,
    }
    print(
        f"Step 2 local: {len(jobs)} body/bodies, {concurrent} concurrent; "
        "outputs -> results/"
    )
    failures = 0
    with ProcessPoolExecutor(max_workers=concurrent) as pool:
        futures = {pool.submit(solve_job, job, **kwargs): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                status, path = future.result()
                print(f"[{index}/{len(jobs)}] {status:7s} {Path(path).name}")
            except Exception as exc:
                failures += 1
                print(f"[{index}/{len(jobs)}] FAILED: {exc}")
    if failures:
        raise SystemExit(f"{failures} body solve(s) failed.")


if __name__ == "__main__":
    main()
