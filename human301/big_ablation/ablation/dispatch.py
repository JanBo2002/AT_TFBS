"""Ordered configurations share eight outstanding training-job slots."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from .recorder import atomic_json
from .slurm import Slurm, SlurmError, TERMINAL

MAX_JOBS = 8
POLL_SECONDS = 60


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outstanding(tasks):
    return [task for task in tasks if task["state"] not in TERMINAL | {"UNSUBMITTED"}]


class Dispatcher:
    def __init__(self, root, batch, slurm=None):
        self.root = Path(root).resolve()
        self.batch = Path(batch).resolve()
        self.path = self.batch / "state.json"
        self.state = json.loads(self.path.read_text())
        self.slurm = slurm or Slurm()
        if self.state["root"] != str(self.root) or self.state["max_jobs"] != MAX_JOBS:
            raise ValueError("Batch project path or job cap does not match")
        for plan, expected in self.state["plan_hashes"].items():
            if file_hash(plan) != expected:
                raise ValueError(f"Frozen plan changed: {plan}")

    def save(self):
        self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, self.state)

    def submit(self, record, args):
        # Persist intent BEFORE sbatch: a crash after acceptance can be recovered
        # by unique job name, without resubmitting the same training task.
        record["state"] = "SUBMITTING"
        self.save()
        try:
            record["job_id"] = self.slurm.submit(args)
        except SlurmError as exc:
            raise RuntimeError(f"Submission unconfirmed for {record['job_name']}: {exc}. "
                               "Resume this batch; do not start another one.") from exc
        record["state"] = "SUBMITTED"
        self.save()

    def refresh(self, records):
        active = outstanding(records)
        if not active:
            return
        for job in active:
            if job["state"] == "SUBMITTING" and not job["job_id"]:
                job["job_id"] = self.slurm.recover(job["job_name"], self.state["created_date"])
                job["state"] = "SUBMITTED"
                self.save()
        queue = self.slurm.queue()
        missing = [job["job_id"] for job in active if job["job_id"] not in queue]
        account = self.slurm.accounting(missing) if missing else {}
        for job in active:
            job_id = job["job_id"]
            if job_id in queue:
                observed = queue[job_id]["state"]
                # Even terminal entries still in squeue retain their slot until
                # the allocation disappears. COMPLETING also retains its slot.
                job["state"] = "FINISHING" if observed in TERMINAL else observed
            else:
                observed = account.get(job_id, "UNKNOWN")
                # Absence from squeue alone is never evidence of completion.
                job["state"] = observed if observed in TERMINAL else "UNKNOWN"
            job["slurm_state"] = observed
        self.save()

    def tick(self):
        tasks = self.state["tasks"]
        collection = self.state.get("collection")
        self.refresh(tasks + ([collection] if collection else []))
        if collection:
            if collection["state"] in TERMINAL:
                self.state["phase"] = "finished"
                self.state["success"] = all(job["state"] == "COMPLETED" for job in tasks + [collection])
                self.save()
                print(f"Batch finished; success={self.state['success']}. Results: {self.root / 'results'}", flush=True)
                return True
            return False
        active = outstanding(tasks)
        if any(job["state"] == "UNKNOWN" for job in active):
            print("Waiting for Slurm accounting; unknown jobs retain their slots.", flush=True)
            return False
        for task in tasks:
            if len(active) >= MAX_JOBS:
                break
            if task["state"] != "UNSUBMITTED":
                continue
            args = [*self.state["sbatch_options"], "--no-requeue", "--export=ALL",
                    f"--job-name={task['job_name']}", f"--chdir={self.root}",
                    f"--output={self.batch / 'slurm' / (task['job_name'] + '_%j.out')}"]
            if active:
                # after: means 'has started (or was cancelled)', not 'has
                # finished'. Thus jobs overlap, while later tasks cannot start
                # ahead of an earlier pending task, even under backfill.
                args += ["--dependency=after:" + ":".join(job["job_id"] for job in active),
                         "--kill-on-invalid-dep=yes"]
            args += [str(self.root / "train_task.sbatch"), task["plan"], str(task["index"]), str(self.root)]
            self.submit(task, args)
            active.append(task)
            print(f"{task['config']} | {task['tf'] or 'multi_tf'} | {task['train_assay']} | "
                  f"seed {task['seed']} -> job {task['job_id']} ({len(active)}/{MAX_JOBS} slots)", flush=True)
        if not active and all(task["state"] in TERMINAL for task in tasks):
            collection = {"job_name": self.state["batch_id"] + "_collect", "job_id": None,
                          "state": "UNSUBMITTED"}
            self.state["collection"] = collection
            # CPU collection inherits the account, but not GPU resources or a
            # GPU partition/QoS. Site-specific CPU settings live in its sbatch.
            account_options = [opt for opt in self.state["sbatch_options"] if opt.startswith("--account=")]
            self.submit(collection, [*account_options, "--no-requeue", "--export=ALL",
                        f"--job-name={collection['job_name']}", f"--chdir={self.root}",
                        f"--output={self.batch / 'slurm' / 'collect_%j.out'}",
                        str(self.root / "collect_all.sbatch"), str(self.root)])
            print(f"All training jobs ended. CPU collection: {collection['job_id']}", flush=True)
        return False

    def work(self):
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Dispatcher stopped by signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        self.state.update(phase="running", pid=os.getpid(), last_error="")
        self.save()
        try:
            while True:
                try:
                    if self.tick():
                        return
                    self.state["last_error"] = ""
                except SlurmError as exc:
                    # Controller/accounting failures release no slots. Retry
                    # one minute later rather than submitting on stale data.
                    self.state["last_error"] = str(exc)
                    print(f"Slurm query failed; no new submissions: {exc}", flush=True)
                    self.save()
                time.sleep(POLL_SECONDS)
        except BaseException as exc:
            self.state.update(phase="paused", last_error=f"{type(exc).__name__}: {exc}")
            self.save()
            raise
