#!/usr/bin/env python3
"""Submit all geometries/*.geo BoR bodies to SLURM.

Workers publish final body GRIMs directly to results/. Scheduler output is
written to logs/. No staging or collection step is required.
"""

import argparse
from multiprocessing import Pool
import os
from pathlib import Path
import shlex
import shutil
import subprocess
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
    solve_job_catching,
    validate_config,
)

# ── USER SETTINGS ──────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
GEOMETRY_UNITS = "meters"
WORKERS_PER_BODY = 4
MAX_CONCURRENT_BODIES_PER_TASK = 1
FORCE = False

ARRAY_TASKS = 1
SLURM_PARTITION = "compute"
SLURM_ACCOUNT = None
SLURM_TIME = None
SLURM_MEMORY = "0"
SLURM_CPUS = None
SUBMIT = True
# ───────────────────────────────────────────────────────────────────────────


def _configuration():
    aspects = list(
        radar_grid_aspects(
            AZIMUTHS_DEG, ELEVATIONS_DEG, AXIS_AZ_DEG, AXIS_EL_DEG
        )
    )
    try:
        frequencies, aspects = validate_config(FREQUENCIES_GHZ, aspects)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(ARRAY_TASKS) < 1:
        raise SystemExit("ARRAY_TASKS must be at least 1.")
    jobs = discover_jobs(HERE)
    if not jobs:
        raise SystemExit("No body .geo files found in geometries/.")
    return frequencies, aspects, prepare_jobs(
        jobs,
        frequencies=frequencies,
        aspects=aspects,
        geometry_units=GEOMETRY_UNITS,
        runner_path=__file__,
    )


def worker(task_index):
    frequencies, aspects, jobs = _configuration()
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", ARRAY_TASKS))
    assigned = [job for index, job in enumerate(jobs) if index % count == task_index]
    if not assigned:
        print(f"Task {task_index}: no assigned bodies.")
        return
    concurrent = max(
        1, min(int(MAX_CONCURRENT_BODIES_PER_TASK), len(assigned))
    )
    kwargs = {
        "frequencies": frequencies,
        "aspects": aspects,
        "geometry_units": GEOMETRY_UNITS,
        "workers_per_body": WORKERS_PER_BODY,
        "force": FORCE,
    }
    failures = 0
    with Pool(processes=concurrent, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(
            solve_job_catching, [(job, kwargs) for job in assigned]
        ):
            kind, first, second, job = result
            if kind == "ok":
                print(f"{first:7s} {Path(second).name}", flush=True)
            else:
                failures += 1
                print(f"FAILED {job['name']}\n{first}", flush=True)
    if failures:
        raise SystemExit(f"{failures} body solve(s) failed.")


def submit():
    _frequencies, _aspects, jobs = _configuration()
    if SUBMIT and shutil.which("sbatch") is None:
        raise SystemExit("SUBMIT=True but sbatch is unavailable.")
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    wrapped = (
        f"{shlex.quote(sys.executable)} "
        f"{shlex.quote(str(Path(__file__).resolve()))} "
        "--worker ${SLURM_ARRAY_TASK_ID}"
    )
    args = [
        "sbatch",
        "--job-name=step2_bor",
        f"--array=0-{int(ARRAY_TASKS)-1}",
        "--nodes=1",
        "--ntasks=1",
        f"--partition={SLURM_PARTITION}",
        f"--chdir={HERE}",
        f"--output={logs}/%A_%a.out",
        f"--error={logs}/%A_%a.err",
    ]
    if SLURM_ACCOUNT:
        args.append(f"--account={SLURM_ACCOUNT}")
    if SLURM_TIME:
        args.append(f"--time={SLURM_TIME}")
    if SLURM_MEMORY:
        args.append(f"--mem={SLURM_MEMORY}")
    args.append(f"--cpus-per-task={int(SLURM_CPUS)}" if SLURM_CPUS else "--exclusive")
    args.extend(["--wrap", wrapped])
    print(
        f"Step 2 HPC: {len(jobs)} body/bodies; direct results -> results/, "
        "scheduler output -> logs/."
    )
    if not SUBMIT:
        print(shlex.join(args))
        return
    completed = subprocess.run(args, check=False, text=True, capture_output=True)
    if completed.returncode:
        raise SystemExit(completed.stderr or completed.stdout)
    print(completed.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    arguments = parser.parse_args()
    worker(arguments.worker) if arguments.worker is not None else submit()


if __name__ == "__main__":
    main()
