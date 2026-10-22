#!/usr/bin/env python3
"""Background submission with persistent plans and a shared eight-job limit."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ablation.dispatch import Dispatcher, MAX_JOBS, file_hash, outstanding
from ablation.recorder import atomic_json
from ablation.slurm import TERMINAL, resource_options
from run_experiment import PLAN_FIELDS, make_plan

ROOT = Path(__file__).resolve().parent


def check_tools():
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Start/resume the dispatcher on the login node, outside a Slurm job")
    missing = [name for name in ("sbatch", "squeue", "sacct") if not shutil.which(name)]
    if missing:
        raise ValueError("Required Slurm commands not found: " + ", ".join(missing))


def project_lock():
    (ROOT / "plans").mkdir(exist_ok=True)
    handle = (ROOT / "plans" / ".dispatcher.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ValueError("A dispatcher is already running for this project (eight shared slots)") from None
    return handle


def current_batch():
    registry = ROOT / "plans" / "active_batch.json"
    if registry.exists():
        batch = Path(json.loads(registry.read_text())["batch"])
        state = json.loads((batch / "state.json").read_text())
        if state["phase"] != "finished":
            return batch
    return None


def prepare(args):
    options = resource_options(args.sbatch_options[1:] if args.sbatch_options[:1] == ["--"] else args.sbatch_options)
    priority = ROOT / "configs" / "training_priority.csv"
    selection = argparse.Namespace(all_configs=False, config=args.config, manifest=args.manifest.resolve(),
                                   tf=None, train_assay=None, seed=None, priority_file=priority)
    rows = make_plan(selection)
    if not rows:
        raise ValueError("Task plan is empty")
    batch = Path(tempfile.mkdtemp(prefix="batch_", dir=ROOT / "plans")).resolve()
    (batch / "slurm").mkdir()
    shutil.copyfile(priority, batch / "training_priority.csv")
    batch_id = "h301_" + uuid.uuid4().hex[:12]
    tasks, hashes, configs = [], {}, []
    # make_plan returns complete configuration blocks in the explicit order
    # from submit.sh; within each block it applies training_priority.csv.
    for row in rows:
        config = json.loads(Path(row["config_path"]).read_text())["name"]
        if config not in configs:
            configs.append(config)
        label = f"{config}_{row['tf'] or 'multi_tf'}_{row['train_assay']}_s{row['seed']}"
        tasks.append(dict(config=config, tf=row["tf"], train_assay=row["train_assay"], seed=row["seed"],
                          job_name=f"{label[:50]}_{batch_id}_{len(tasks):04d}", job_id=None, state="UNSUBMITTED"))
    for name in configs:
        directory = batch / name
        directory.mkdir()
        plan = directory / "tasks.csv"
        config_rows = [row for row, task in zip(rows, tasks) if task["config"] == name]
        with plan.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
            writer.writeheader()
            writer.writerows(config_rows)
        hashes[str(plan)] = file_hash(plan)
        for index, task in enumerate(task for task in tasks if task["config"] == name):
            task.update(plan=str(plan), index=index)
    atomic_json(batch / "state.json", dict(version=1, root=str(ROOT), batch_id=batch_id,
                created_utc=datetime.now(timezone.utc).isoformat(), created_date=datetime.now().strftime("%Y-%m-%d"),
                phase="prepared", max_jobs=MAX_JOBS, config_order=configs, sbatch_options=options,
                plan_hashes=hashes, tasks=tasks, collection=None, last_error=""))
    return batch


def background(batch, lock):
    with (batch / "launcher.log").open("a") as log:
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "submit_jobs.py"), "_work", str(batch),
                                    "--lock-fd", str(lock.fileno())], cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lock.fileno(),))
    print(f"Dispatcher PID: {process.pid}; shared training-job limit: {MAX_JOBS}")
    print(f"Batch: {batch}")
    print(f"Log: {batch / 'launcher.log'}")
    print(f"Status: {sys.executable} {ROOT / 'submit_jobs.py'} status {batch}")


def status(batch):
    state = json.loads((batch / "state.json").read_text())
    print(f"Batch: {batch}\nPhase: {state['phase']}\nLimit: {state['max_jobs']}")
    for config in state["config_order"]:
        tasks = [task for task in state["tasks"] if task["config"] == config]
        ended = [task for task in tasks if task["state"] in TERMINAL]
        failed = [task for task in ended if task["state"] != "COMPLETED"]
        waiting = sum(task["state"] == "UNSUBMITTED" for task in tasks)
        print(f"{config}: {len(ended)}/{len(tasks)} ended ({len(failed)} failed/cancelled), "
              f"{len(outstanding(tasks))} submitted/unfinished, {waiting} not yet submitted")
    if state.get("collection"):
        print(f"Collection: {state['collection']['job_id']} {state['collection']['state']}")
    if state.get("last_error"):
        print("Last error: " + state["last_error"])
    print(f"Log: {batch / 'launcher.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="Plan selected configurations and start the background dispatcher")
    start.add_argument("--manifest", type=Path, required=True)
    start.add_argument("--config", nargs="+", required=True)
    start.add_argument("sbatch_options", nargs=argparse.REMAINDER)
    for name in ("resume", "status", "_work"):
        command = sub.add_parser(name)
        command.add_argument("batch", type=Path)
        if name == "_work":
            command.add_argument("--lock-fd", type=int, required=True)
    args = parser.parse_args()
    if args.command == "status":
        status(args.batch.resolve())
        return 0
    if args.command == "_work":
        # Closing the inherited descriptor releases the shared project lock.
        with os.fdopen(args.lock_fd, "a"):
            Dispatcher(ROOT, args.batch).work()
        return 0
    check_tools()
    # The child inherits this exact open file description. Do not explicitly
    # unlock in the parent: it would also unlock the child's shared lock.
    with project_lock() as lock:
        current = current_batch()
        if args.command == "start":
            if current:
                raise ValueError(f"Unfinished batch exists: {current}. Resume it with submit_jobs.py resume {current}")
            batch = prepare(args)
            atomic_json(ROOT / "plans" / "active_batch.json", {"batch": str(batch)})
        else:
            batch = args.batch.resolve()
            if batch != current:
                raise ValueError("Resume requires the current unfinished batch from this project")
            Dispatcher(ROOT, batch)  # Check root and frozen plans before detaching.
        background(batch, lock)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
