#!/usr/bin/env python3
"""Remove only outputs of the old, aborted single-job batch. Keep all caches."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from ablation.slurm import TERMINAL, state_name

ROOT = Path(__file__).resolve().parent


def inside(path, parent):

    return path.resolve().is_relative_to(parent.resolve()) and path.resolve() != parent.resolve()


def remove_outputs(root, batch, state):
    """Delete runs by recorded job ID, never by TF or a broad results glob."""
    ids = {str(t["job_id"]) for t in state["tasks"] if t.get("job_id")}
    affected, removed = set(), []
    results = root / "results"
    targets = []
    if results.exists():
        for meta_path in results.glob("*/runs/**/run.json"):
            meta = json.loads(meta_path.read_text())
            if str(meta.get("job_id", "")) not in ids:
                continue
            run = meta_path.parent
            experiment = results / run.relative_to(results).parts[0]
            if not inside(run, experiment / "runs"):
                raise ValueError(f"Run points outside its result directory: {run}")
            targets.append((run, experiment))
    for run, experiment in targets:
        shutil.rmtree(run)
        removed.append(str(run))
        affected.add(experiment)
        parent = run.parent
        while parent != experiment and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    for experiment in affected:
        remaining = list((experiment / "runs").glob("**/run.json")) if (experiment / "runs").exists() else []
        if remaining:
            # Preserve shared provenance and refresh an existing export rather
            # than deleting results belonging to other jobs.
            if (results / "Results_all" / (experiment.name + ".csv")).exists():
                from ablation.collect import collect
                collect(root, experiment.name)
            continue
        # Do not touch an experiment containing unidentified run directories.
        if (experiment / "runs").exists() and any((experiment / "runs").iterdir()):
            continue
        for name in ("sources",):
            directory = experiment / name
            if directory.exists() and not directory.is_symlink():
                shutil.rmtree(directory)
        for name in ("config.json", "evaluation.md", ".initialise.lock"):
            path = experiment / name
            if path.is_file():
                path.unlink()
        export = results / "Results_all" / (experiment.name + ".csv")
        if export.is_file():
            export.unlink()
        if not any(experiment.iterdir()):
            experiment.rmdir()
    return removed


def main():
    registry = ROOT / "plans" / "active_batch.json"
    if not registry.exists():
        print("No old active batch remains. Caches and configuration files were preserved.")
        return 0
    batch = Path(json.loads(registry.read_text())["batch"])
    if not inside(batch, ROOT / "plans") or batch.parent.resolve() != (ROOT / "plans").resolve():
        raise ValueError("Batch is not a direct child of this project's plans directory")
    state = json.loads((batch / "state.json").read_text())
    if state.get("version") != 1 or Path(state["root"]).resolve() != ROOT:
        raise ValueError("Cleanup applies only to the old single-job batch, not a new array batch")
    pid = state.get("pid")
    if isinstance(pid, int) and pid > 1:
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if (str(ROOT / "submit_jobs.py").encode() in args and b"_work" in args
                    and str(batch).encode() in args):
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, FileNotFoundError):
            pass
    with (ROOT / "plans" / ".dispatcher.lock").open("a") as lock:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Launcher has not stopped; no files deleted")
                time.sleep(.25)
        state = json.loads((batch / "state.json").read_text())
        ids = {str(t["job_id"]) for t in state["tasks"] if t.get("job_id")}
        collection = state.get("collection") or {}
        if collection.get("job_id"):
            ids.add(str(collection["job_id"]))
        if any(not re.fullmatch(r"[0-9]+", job_id) for job_id in ids):
            raise ValueError("Unexpected old job ID; no files deleted")
        if any(t.get("state") == "SUBMITTING" and not t.get("job_id") for t in state["tasks"]):
            raise RuntimeError("An old submission has no confirmed ID; check Slurm before deleting its files")
        if ids:
            # Already-cancelled IDs may return an error. The following squeue
            # check, not scancel's return code, decides whether deletion is safe.
            subprocess.run(["scancel", *sorted(ids)], capture_output=True, text=True, timeout=30)
            deadline = time.monotonic() + 15
            while True:
                result = subprocess.run(["squeue", "--local", "--me", "--array", "--states=all",
                                         "--noheader", "--format=%i|%T"], check=True, capture_output=True,
                                        text=True, timeout=30)
                live = set()
                for line in result.stdout.splitlines():
                    parts = line.strip().split("|", 1)
                    if parts[0] in ids and (len(parts) != 2 or state_name(parts[1]) not in TERMINAL):
                        live.add(parts[0])
                if not live:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Jobs still visible in Slurm: " + ", ".join(sorted(live)) + "; no files deleted")
                time.sleep(2)
        removed = remove_outputs(ROOT, batch, state)
        shutil.rmtree(batch)
        registry.unlink()
        for path in removed:
            print("Deleted run: " + path)
        print(f"Deleted {len(removed)} run directories and the old batch folder: {batch}")
        print("Preserved: caches/, configs/caches.csv, configurations, priority list, and unrelated runs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
