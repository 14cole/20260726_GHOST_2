#!/usr/bin/env python3
"""Submit the step-1 coupon library to SLURM.

There are no staged run directories and no collection step. Array workers
write completed files directly to results/FRD or results/OPN, while SLURM
stdout/stderr goes to hpc_logs/.
"""

import argparse
from multiprocessing import Pool
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Backend"))

from step1_monostatic import (  # noqa: E402
    _pin_blas,
    discover_jobs,
    prepare_jobs,
    solve_job_catching,
    validate_config,
)

# ── USER SETTINGS ──────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
ANGLES_DEG = np.arange(0.0, 180.1, 5.0)
POLARIZATIONS = ["TM", "TE"]
GEOMETRY_UNITS = "meters"
SOLVER_METHOD = "auto"
MAX_PANELS = 50_000
FORCE = False

ARRAY_TASKS = 1               # number of simultaneously scheduled nodes
MAX_WORKERS_PER_TASK = None   # None = allocated CPU count
BLAS_THREADS_PER_WORKER = 1
SLURM_PARTITION = "compute"
SLURM_ACCOUNT = None
SLURM_TIME = None
SLURM_MEMORY = "0"            # all node memory; use None for cluster default
SLURM_CPUS = None             # None requests an exclusive node
SUBMIT = True
# ───────────────────────────────────────────────────────────────────────────


def _configuration():
    try:
        frequencies, angles, polarizations = validate_config(
            FREQUENCIES_GHZ, ANGLES_DEG, POLARIZATIONS
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(ARRAY_TASKS) < 1:
        raise SystemExit("ARRAY_TASKS must be at least 1.")
    jobs = discover_jobs(HERE, frequencies, polarizations)
    if not jobs:
        raise SystemExit(
            "No .geo files found. Put coupons in geometries/FRD or "
            "geometries/OPN; either folder may otherwise be empty."
        )
    return angles, prepare_jobs(
        jobs,
        angles_deg=angles,
        geometry_units=GEOMETRY_UNITS,
        solver_method=SOLVER_METHOD,
        max_panels=MAX_PANELS,
        runner_path=__file__,
    )


def _validate_config() -> None:
    """Validate user settings without discovering or submitting work."""
    try:
        validate_config(FREQUENCIES_GHZ, ANGLES_DEG, POLARIZATIONS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(ARRAY_TASKS) < 1:
        raise SystemExit("ARRAY_TASKS must be at least 1.")


def _allocated_cpus() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name, "")
        if value.isdigit() and int(value) > 0:
            return int(value)
    return max(1, os.cpu_count() or 1)


def worker(task_index: int) -> None:
    angles, jobs = _configuration()
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", ARRAY_TASKS))
    if task_count < 1:
        raise SystemExit("SLURM array task count must be at least 1.")
    assigned = [
        job for index, job in enumerate(jobs)
        if index % task_count == int(task_index)
    ]
    if not assigned:
        print(f"Task {task_index}: no assigned units.")
        return
    cpus = _allocated_cpus()
    worker_limit = cpus if MAX_WORKERS_PER_TASK is None else int(MAX_WORKERS_PER_TASK)
    workers = max(1, min(cpus, worker_limit, len(assigned)))
    kwargs = {
        "angles_deg": angles,
        "geometry_units": GEOMETRY_UNITS,
        "solver_method": SOLVER_METHOD,
        "max_panels": MAX_PANELS,
        "force": FORCE,
    }
    print(
        f"Task {task_index}: {len(assigned)} unit(s), {workers} worker(s), "
        "direct output to results/{FRD,OPN}",
        flush=True,
    )
    _pin_blas(BLAS_THREADS_PER_WORKER)
    failures = 0
    arguments = [(job, kwargs) for job in assigned]
    with Pool(
        processes=workers,
        initializer=_pin_blas,
        initargs=(BLAS_THREADS_PER_WORKER,),
        maxtasksperchild=1,
    ) as pool:
        for index, result in enumerate(
            pool.imap_unordered(solve_job_catching, arguments), 1
        ):
            kind, first, second, job = result
            if kind == "ok":
                print(
                    f"[{index}/{len(assigned)}] {first:7s} "
                    f"{job['role']}/{Path(second).name}",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"[{index}/{len(assigned)}] FAILED "
                    f"{job['role']}/{Path(job['output']).name}\n{first}",
                    flush=True,
                )
    if failures:
        raise SystemExit(f"{failures} solve unit(s) failed.")


def submit() -> None:
    _angles, jobs = _configuration()
    if shutil.which("sbatch") is None and SUBMIT:
        raise SystemExit("SUBMIT=True but sbatch is not available.")
    logs = HERE / "hpc_logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
        "--worker ${SLURM_ARRAY_TASK_ID}"
    )
    args = [
        "sbatch",
        f"--job-name=step1_2d",
        f"--array=0-{int(ARRAY_TASKS) - 1}",
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
    if SLURM_CPUS:
        args.append(f"--cpus-per-task={int(SLURM_CPUS)}")
    else:
        args.append("--exclusive")
    args.extend(["--wrap", command])
    print(
        f"Step 1 HPC: {len(jobs)} solve unit(s), {ARRAY_TASKS} array task(s)\n"
        "Results write directly to results/{FRD,OPN}; logs write to hpc_logs/."
    )
    if not SUBMIT:
        print("SUBMIT=False. Command:\n" + shlex.join(args))
        return
    completed = subprocess.run(args, check=False, text=True, capture_output=True)
    if completed.returncode:
        raise SystemExit(completed.stderr or completed.stdout)
    print(completed.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    arguments = parser.parse_args()
    if arguments.worker is None:
        submit()
    else:
        worker(arguments.worker)


if __name__ == "__main__":
    main()
