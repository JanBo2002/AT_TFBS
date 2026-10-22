"""One visible array per experiment, with eight shared released-task slots."""
from __future__ import annotations

from .array_slurm import ArraySlurm
from .dispatch import Dispatcher, MAX_JOBS
from .slurm import TERMINAL

INACTIVE = TERMINAL | {"UNSUBMITTED", "HELD"}


def active_tasks(tasks):
    return [task for task in tasks if task["state"] not in INACTIVE]


class ArrayDispatcher(Dispatcher):
    def __init__(self, root, batch, slurm=None):
        super().__init__(root, batch, slurm or ArraySlurm())
        if self.state.get("version") != 2:
            raise ValueError("This is an old single-job batch. Run cleanup_aborted_batch.py before starting the arrays.")

    def tasks_for(self, array):
        return [task for task in self.state["tasks"] if task["config"] == array["config"]]

    def assign_ids(self, array):
        for task in self.tasks_for(array):
            if not task.get("job_id"):
                task.update(job_id=f"{array['job_id']}_{task['index']}", state="HELD")
        self.save()

    def refresh(self, records):
        pending = [r for r in records if r.get("job_id") and r["state"] not in TERMINAL]
        if not pending:
            return
        queue = self.slurm.queue()
        missing = [r["job_id"] for r in pending if r["job_id"] not in queue]
        account = self.slurm.accounting(missing) if missing else {}
        for record in pending:
            job_id = record["job_id"]
            if job_id in queue:
                observed = queue[job_id]["state"]
                reason = queue[job_id].get("reason", "")
                if observed == "PENDING" and reason.startswith("JobHeld"):
                    record["state"] = "HELD"
                else:
                    record["state"] = observed
            else:
                observed = account.get(job_id, "UNKNOWN")
                record["state"] = observed if observed in TERMINAL else "UNKNOWN"
            record["slurm_state"] = observed
        self.save()

    def tick(self):
        arrays, tasks = self.state["arrays"], self.state["tasks"]
        for array in arrays:
            if array["state"] == "SUBMITTING" and not array["job_id"]:
                array.update(job_id=self.slurm.recover_array(array["job_name"], self.state["created_date"]),
                             state="SUBMITTED")
            if array["job_id"]:
                self.assign_ids(array)
        collection = self.state.get("collection")
        if collection and collection["state"] == "SUBMITTING" and not collection["job_id"]:
            collection.update(job_id=self.slurm.recover(collection["job_name"], self.state["created_date"]),
                              state="SUBMITTED")
            self.save()
        self.refresh(tasks + ([collection] if collection else []))
        if collection:
            if collection["state"] in TERMINAL:
                self.state.update(phase="finished", success=all(t["state"] == "COMPLETED" for t in tasks + [collection]))
                self.save()
                print(f"Batch finished; success={self.state['success']}", flush=True)
                return True
            return False
        active = active_tasks(tasks)
        if len(active) > MAX_JOBS:
            raise RuntimeError("More than eight released tasks observed; dispatcher paused without further releases")
        if any(t["state"] == "UNKNOWN" for t in active):
            print("Waiting for Slurm/accounting; no further tasks released.", flush=True)
            return False
        for array in arrays:
            if len(active) >= MAX_JOBS:
                break
            if not array["job_id"]:
                options = [*self.state["sbatch_options"], "--hold", "--no-requeue", "--export=ALL", "--kill-on-invalid-dep=yes",
                           f"--array=0-{array['count']-1}%{MAX_JOBS}", f"--job-name={array['job_name']}",
                           f"--chdir={self.root}", f"--output={self.batch / 'slurm' / (array['config'] + '_%A_%a.out')}",
                           str(self.root / "train_array.sbatch"), array["plan"], str(self.root)]
                self.submit(array, options)
                self.assign_ids(array)
                print(f"Visible array: {array['config']} -> {array['job_id']} (0-{array['count']-1}%8)", flush=True)
            for task in self.tasks_for(array):
                if len(active) >= MAX_JOBS:
                    break
                if task["state"] != "HELD":
                    continue
                task["state"] = "RELEASING"
                self.save()
                self.slurm.release(task["job_id"], [t["job_id"] for t in active])
                task["state"] = "SUBMITTED"
                active.append(task)
                self.save()
                print(f"Release {task['job_id']}: {task['config']} {task['tf'] or 'multi_tf'} "
                      f"{task['train_assay']} seed {task['seed']} ({len(active)}/8 slots)", flush=True)
        if not active and all(t["state"] in TERMINAL for t in tasks):
            collection = dict(job_name=self.state["batch_id"] + "_collect", job_id=None, state="UNSUBMITTED")
            self.state["collection"] = collection
            accounts = [x for x in self.state["sbatch_options"] if x.startswith("--account=")]
            self.submit(collection, [*accounts, "--no-requeue", "--export=ALL", f"--job-name={collection['job_name']}",
                        f"--chdir={self.root}", f"--output={self.batch / 'slurm' / 'collect_%j.out'}",
                        str(self.root / "collect_all.sbatch"), str(self.root)])
            print(f"All training tasks ended. CPU collection: {collection['job_id']}", flush=True)
        return False
