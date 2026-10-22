"""Submission tests use a local Slurm simulation, never a real cluster."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ablation.dispatch import Dispatcher, MAX_JOBS, file_hash, outstanding
from ablation.array_dispatch import active_tasks
from ablation.priority import read_priority
from ablation.slurm import Slurm, SlurmError, resource_options
from run_experiment import make_plan

CONFIGS = ["00_basic", "A01_kernel_LegNetMax_3_6_9_12", "A02_kernel_7_11_15_21"]
# Independent transcription of the supplied order. No duration values.
EXPECTED = """CTCF:CHS ELF3:CHS SOX2:CHS FOSL2:CHS MYF6:CHS PAX7:CHS YY1:CHS FLI1:CHS
ZBTB8A:CHS ZBED9:CHS NFKB1:CHS RFX5:CHS LEF1:CHS BATF2:CHS FLI1:GHTS MAX:CHS
ELF3:GHTS MGA:CHS CTCF:GHTS ZBED9:GHTS MAX:GHTS SOX2:GHTS GABPA:CHS LEUTX:GHTS
GABPA:GHTS NFKB1:GHTS YY1:GHTS VDR:GHTS USF3:GHTS MYF6:GHTS FOSL2:GHTS SRY:CHS
PAX7:GHTS VDR:CHS LEF1:GHTS MGA:GHTS RORB:GHTS LEUTX:CHS USF3:CHS SRY:GHTS
CREB3L3:GHTS ZBTB8A:GHTS NR1H4:CHS RFX5:GHTS NR1H4:GHTS TERF1:CHS CREB3L3:CHS
TERF1:GHTS ZNF772:GHTS ZNF696:GHTS GLI4:CHS ZFP3:CHS RORB:CHS BATF2:GHTS ZNF696:CHS
ZNF773:GHTS ZNF772:CHS ZNF773:CHS GLI4:GHTS ZFP3:GHTS""".split()
PAIRS = [(tf, "CHIP" if assay == "CHS" else assay) for tf, assay in (p.split(":") for p in EXPECTED)]


class FakeSlurm:
    def __init__(self):
        self.jobs, self.ended, self.calls = {}, {}, []
        self.fail_query = False
        self.ambiguous_once = False
        self.max_open = 0

    def submit(self, args):
        job_id = str(1000 + len(self.calls))
        name = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--job-name="))
        self.jobs[job_id] = dict(name=name, state="PENDING")
        self.calls.append(list(args))
        self.max_open = max(self.max_open, len(self.jobs))
        if self.ambiguous_once:
            self.ambiguous_once = False
            raise SlurmError("response lost after acceptance")
        return job_id

    def queue(self):
        if self.fail_query:
            raise SlurmError("controller unavailable")
        return self.jobs

    def accounting(self, ids):
        return {job_id: self.ended[job_id] for job_id in ids if job_id in self.ended}

    def recover(self, name, created_date):
        return next(job_id for job_id, job in self.jobs.items() if job["name"] == name)

    def finish(self, job_id, state="COMPLETED"):
        self.jobs.pop(job_id)
        self.ended[job_id] = state


def make_batch(root, counts=(10, 11, 3)):
    batch = root / "plans" / "batch_test"
    (batch / "slurm").mkdir(parents=True)
    tasks, hashes = [], {}
    for config, count in zip(CONFIGS, counts):
        plan = batch / (config + ".csv")
        plan.write_text("index\n" + "\n".join(str(i) for i in range(count)) + "\n")
        hashes[str(plan)] = file_hash(plan)
        for index in range(count):
            tasks.append(dict(config=config, tf=f"TF{index}", train_assay="CHIP", seed=13,
                plan=str(plan), index=index, job_name=f"test_{len(tasks)}", job_id=None, state="UNSUBMITTED"))
    state = dict(root=str(root), batch_id="test", created_date="2026-10-22", phase="prepared",
                 max_jobs=8, plan_hashes=hashes, tasks=tasks, collection=None,
                 sbatch_options=["--account=test", "--partition=gpu", "--gres=gpu:1"])
    (batch / "state.json").write_text(json.dumps(state))
    return batch


class SubmissionTests(unittest.TestCase):
    def test_all_60_priorities_and_three_complete_plans(self):
        ranks = read_priority(ROOT / "configs" / "training_priority.csv")
        self.assertEqual(list(ranks), PAIRS)
        self.assertEqual(list(ranks.values()), list(range(1, 61)))
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            cache = temp / "cache.npz"
            cache.touch()
            manifest = temp / "caches.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["tf", "assay", "cache"])
                for pair in reversed(PAIRS):
                    writer.writerow([*pair, cache])
            args = argparse.Namespace(all_configs=False, config=CONFIGS, manifest=manifest,
                tf=None, train_assay=None, seed=[13, 14], priority_file=ROOT / "configs" / "training_priority.csv")
            rows = make_plan(args)
            self.assertEqual(len(rows), 360)
            for number, name in enumerate(CONFIGS):
                block = rows[number * 120:(number + 1) * 120]
                self.assertTrue(all(Path(row["config_path"]).stem == name for row in block))
                self.assertEqual([(r["tf"], r["train_assay"], r["seed"]) for r in block],
                                 [(tf, assay, seed) for tf, assay in PAIRS for seed in (13, 14)])
            args.tf = ["CTCF", "ZFP3"]
            args.train_assay = ["GHTS"]
            filtered = make_plan(args)
            self.assertEqual([r["tf"] for r in filtered[:4]], ["CTCF", "CTCF", "ZFP3", "ZFP3"])

    def test_global_eight_slots_staging_and_final_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = make_batch(root)
            fake = FakeSlurm()
            dispatcher = Dispatcher(root, batch, fake)
            self.assertFalse(dispatcher.tick())
            self.assertEqual(len(fake.calls), 8)
            self.assertTrue(all(t["config"] == CONFIGS[0] for t in outstanding(dispatcher.state["tasks"])))
            for job_id, state in [("1000", "COMPLETED"), ("1001", "FAILED"), ("1002", "CANCELLED")]:
                fake.finish(job_id, state)
            dispatcher.tick()
            active = outstanding(dispatcher.state["tasks"])
            self.assertEqual(len(active), 8)
            self.assertEqual(sum(t["config"] == CONFIGS[0] for t in active), 7)
            self.assertEqual(sum(t["config"] == CONFIGS[1] for t in active), 1)
            self.assertIn("--dependency=after:1003:1004:1005:1006:1007:1008:1009", fake.calls[10])
            # Resume: existing submitted tasks occupy slots, without resubmission.
            dispatcher = Dispatcher(root, batch, fake)
            dispatcher.tick()
            self.assertEqual(len(fake.calls), 11)
            while dispatcher.state.get("collection") is None:
                for job_id in list(fake.jobs)[:4]:
                    fake.finish(job_id)
                dispatcher.tick()
                self.assertLessEqual(len(outstanding(dispatcher.state["tasks"])), MAX_JOBS)
            tasks = dispatcher.state["tasks"]
            self.assertEqual([t["job_id"] for t in tasks], [str(1000 + i) for i in range(24)])
            self.assertEqual(fake.max_open, 8)
            self.assertEqual(len(fake.calls), 25)
            self.assertTrue(fake.calls[-1][-2].endswith("collect_all.sbatch"))
            self.assertIn("--account=test", fake.calls[-1])
            self.assertNotIn("--partition=gpu", fake.calls[-1])
            self.assertNotIn("--gres=gpu:1", fake.calls[-1])
            fake.finish(dispatcher.state["collection"]["job_id"])
            self.assertTrue(dispatcher.tick())
            self.assertEqual(dispatcher.state["phase"], "finished")
            self.assertFalse(dispatcher.state["success"])  # earlier failure/cancellation retained

    def test_five_remaining_tasks_leave_three_slots_for_next_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = FakeSlurm()
            dispatcher = Dispatcher(root, make_batch(root, counts=(5, 8, 2)), fake)
            dispatcher.tick()
            active = outstanding(dispatcher.state["tasks"])
            self.assertEqual([t["config"] for t in active], [CONFIGS[0]] * 5 + [CONFIGS[1]] * 3)
            self.assertEqual(fake.max_open, 8)

    def test_completing_unknown_and_query_failures_release_no_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = FakeSlurm()
            dispatcher = Dispatcher(root, make_batch(root), fake)
            dispatcher.tick()
            fake.jobs["1000"]["state"] = "COMPLETING"
            dispatcher.tick()
            self.assertEqual(len(fake.calls), 8)
            fake.jobs["1000"]["state"] = "COMPLETED"
            dispatcher.tick()
            self.assertEqual(len(fake.calls), 8)
            fake.jobs.pop("1000")  # accounting has not caught up yet
            dispatcher.tick()
            self.assertEqual(len(fake.calls), 8)
            self.assertEqual(dispatcher.state["tasks"][0]["state"], "UNKNOWN")
            fake.fail_query = True
            with self.assertRaises(SlurmError):
                dispatcher.tick()
            self.assertEqual(len(fake.calls), 8)
            fake.fail_query = False
            fake.ended["1000"] = "COMPLETED"
            dispatcher.tick()
            self.assertEqual(len(fake.calls), 9)

    def test_lost_receipt_recovers_without_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = make_batch(root)
            fake = FakeSlurm()
            fake.ambiguous_once = True
            with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                Dispatcher(root, batch, fake).tick()
            self.assertEqual(len(fake.calls), 1)
            resumed = Dispatcher(root, batch, fake)
            resumed.tick()
            self.assertEqual(len(fake.calls), 8)
            self.assertEqual(resumed.state["tasks"][0]["job_id"], "1000")

    def test_resource_options_cannot_add_arrays_or_other_dependencies(self):
        self.assertEqual(resource_options(["--account", "a", "--mem=40G"]), ["--account=a", "--mem=40G"])
        for option in ("--array=0-59%8", "--dependency=afterany:1", "--requeue", "--wrap=echo hi", "--clusters=x"):
            with self.assertRaises(ValueError):
                resource_options([option])

    def test_slurm_accounting_parser_preserves_cancelled_state(self):
        slurm = Slurm()
        with patch.object(slurm, "command", return_value="12|CANCELLED by 123\n13|OUT_OF_MEMORY+\n12.batch|FAILED\n"):
            self.assertEqual(slurm.accounting(["12", "13"]), {"12": "CANCELLED", "13": "OUT_OF_MEMORY"})
        with patch.object(slurm, "command", return_value="123;cluster\n"):
            self.assertEqual(slurm.submit([]), "123")

    def test_background_launcher_and_shared_lock_with_mock_commands(self):
        with tempfile.TemporaryDirectory(prefix="mock_slurm_") as temp:
            temp = Path(temp)
            package = temp / "big_ablation"
            shutil.copytree(ROOT, package, ignore=shutil.ignore_patterns("__pycache__", "results", "plans"))
            tools = temp / "bin"
            tools.mkdir()
            driver = tools / "driver.py"
            driver.write_text('''#!/usr/bin/env python3
import fcntl,json,os,sys
from pathlib import Path
command=Path(sys.argv[0]).name
database=Path(os.environ["FAKE_SLURM_DB"])
with database.with_suffix(".lock").open("a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    jobs=json.loads(database.read_text()) if database.exists() else []
    if command=="sbatch":
        name=next(a.split("=",1)[1] for a in sys.argv[1:] if a.startswith("--job-name="))
        base=str(100+len(jobs))
        size=int(next(a.split("=",1)[1] for a in sys.argv[1:] if a.startswith("--array=")).split("%",1)[0].split("-",1)[1])+1
        for index in range(size):
            jobs.append(dict(id=base+"_"+str(index),name=name,state="PENDING",reason="JobHeldUser",args=sys.argv[1:],
                             leaked_array="SBATCH_ARRAY_INX" in os.environ))
        database.write_text(json.dumps(jobs))
        print(base)
    elif command=="squeue":
        for job in jobs: print(job["id"]+"|"+job["state"]+"|"+job["reason"]+"|"+job["name"])
    elif command=="sacct":
        for job in jobs: print(job["id"]+"|"+job["state"])
    elif command=="scontrol" and sys.argv[1]=="release":
        job=next(j for j in jobs if j["id"]==sys.argv[2])
        job.update(state="RUNNING",reason="None")
        database.write_text(json.dumps(jobs))
''')
            driver.chmod(0o755)
            for name in ("sbatch", "squeue", "sacct", "scontrol"):
                (tools / name).symlink_to(driver.name)
            manifest = package / "configs" / "caches.csv"
            cache = temp / "cache.npz"
            cache.touch()
            with manifest.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["tf", "assay", "cache"])
                writer.writerows([tf, assay, str(cache)] for tf, assay in reversed(PAIRS))
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                       ABLATION_PYTHON=sys.executable, FAKE_SLURM_DB=str(temp / "jobs.json"), SBATCH_ARRAY_INX="0-99")
            env.pop("SLURM_JOB_ID", None)
            pid = None
            state_path = None
            try:
                launch = subprocess.run(["bash", str(package / "submit.sh"), str(manifest), "--account=test"],
                                        env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(launch.returncode, 0, launch.stderr)
                pid = int(launch.stdout.split("Dispatcher PID: ", 1)[1].split(";", 1)[0])
                batch = Path(json.loads((package / "plans" / "active_batch.json").read_text())["batch"])
                state_path = batch / "state.json"
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    state = json.loads(state_path.read_text())
                    if len(active_tasks(state["tasks"])) == 8:
                        break
                    time.sleep(.05)
                self.assertEqual(sum(bool(t["job_id"]) for t in state["tasks"]), 60,
                                 (batch / "launcher.log").read_text())
                self.assertEqual(len(active_tasks(state["tasks"])),8)
                self.assertEqual(state["config_order"], CONFIGS)
                self.assertEqual(len(state["tasks"]), 180)
                self.assertTrue(all(t["config"] == CONFIGS[0] for t in active_tasks(state["tasks"])))
                self.assertEqual([(t["tf"], t["train_assay"]) for t in state["tasks"][:8]], PAIRS[:8])
                self.assertTrue((batch / "training_priority.csv").is_file())
                duplicate = subprocess.run(["bash", str(package / "submit_00_basic.sh"), str(manifest)],
                                           env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(duplicate.returncode, 2)
                self.assertIn("already running", duplicate.stderr)
                jobs = json.loads((temp / "jobs.json").read_text())
                self.assertEqual(len(jobs), 60)
                self.assertEqual(sum(job["state"]=="RUNNING" for job in jobs),8)
                self.assertTrue(all(not job["leaked_array"] for job in jobs))
            finally:
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 5
                    while state_path and time.monotonic() < deadline:
                        if json.loads(state_path.read_text())["phase"] == "paused":
                            break
                        time.sleep(.05)
                    else:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
