"""Array scheduling and selective cleanup, using mock Slurm and temporary files."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ablation.array_dispatch import ArrayDispatcher, active_tasks
from ablation.array_slurm import ArraySlurm
from ablation.dispatch import file_hash
from ablation.slurm import SlurmError, TERMINAL
import cleanup_aborted_batch as cleanup

CONFIGS = ["00_basic", "A01_kernel_LegNetMax_3_6_9_12", "A02_kernel_7_11_15_21"]


class MockArrays:
    def __init__(self):
        self.jobs, self.ended, self.calls, self.releases = {}, {}, [], []
        self.max_active = 0
        self.fail_query = self.uncertain_array = self.uncertain_release = False

    def submit(self, args):
        base = str(2000 + len(self.calls))
        name = next(a.split("=", 1)[1] for a in args if a.startswith("--job-name="))
        array_arg = next((a for a in args if a.startswith("--array=")), None)
        self.calls.append(list(args))
        if array_arg:
            assert "--hold" in args and array_arg.endswith("%8")
            count = int(array_arg.split("-", 3)[-1].split("%", 1)[0]) + 1
            for i in range(count):
                self.jobs[f"{base}_{i}"] = dict(state="PENDING", reason="JobHeldUser", name=name)
        else:
            self.jobs[base] = dict(state="RUNNING", reason="None", name=name)
        if self.uncertain_array:
            self.uncertain_array = False
            raise SlurmError("lost sbatch receipt")
        return base

    def queue(self):
        if self.fail_query:
            raise SlurmError("controller unavailable")
        return self.jobs

    def accounting(self, ids):
        return {job_id: self.ended[job_id] for job_id in ids if job_id in self.ended}

    def recover_array(self, name, created_date):
        return next(job_id.split("_", 1)[0] for job_id, job in self.jobs.items() if job["name"] == name)

    def release(self, task_id, preceding):
        assert all(job_id != task_id for job_id in preceding)
        self.jobs[task_id].update(state="RUNNING", reason="None")
        self.releases.append((task_id, list(preceding)))
        count = sum(job["reason"] != "JobHeldUser" and job["state"] not in TERMINAL for job in self.jobs.values())
        self.max_active = max(self.max_active, count)
        assert count <= 8
        if self.uncertain_release:
            self.uncertain_release = False
            raise SlurmError("release applied but response lost")

    def finish(self, job_id, state="COMPLETED"):
        self.jobs.pop(job_id)
        self.ended[job_id] = state


def batch(root, counts=(10, 11, 3)):
    directory = root / "plans" / "batch_array_test"
    (directory / "slurm").mkdir(parents=True)
    tasks, arrays, hashes = [], [], {}
    for name, count in zip(CONFIGS, counts):
        plan = directory / (name + ".csv")
        plan.write_text("index\n" + "\n".join(str(i) for i in range(count)))
        hashes[str(plan)] = file_hash(plan)
        arrays.append(dict(config=name, plan=str(plan), count=count, job_name="test_" + name,
                           job_id=None, state="UNSUBMITTED"))
        for i in range(count):
            tasks.append(dict(config=name, plan=str(plan), index=i, tf=f"TF{i}", train_assay="CHIP", seed=13,
                              job_id=None, state="UNSUBMITTED"))
    state = dict(version=2, root=str(root), batch_id="test", max_jobs=8, phase="prepared",
                 created_date="2026-10-22", config_order=CONFIGS, plan_hashes=hashes, arrays=arrays,
                 tasks=tasks, sbatch_options=["--account=test"], collection=None)
    (directory / "state.json").write_text(json.dumps(state))
    return directory


class ArrayTests(unittest.TestCase):
    def test_all_remaining_tasks_visible_and_global_eight_across_transitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mock = MockArrays()
            dispatcher = ArrayDispatcher(root, batch(root), mock)
            dispatcher.tick()
            self.assertEqual(len(mock.calls), 1)
            self.assertEqual(len(mock.jobs), 10)  # full first array, not only eight tasks
            self.assertEqual(len(active_tasks(dispatcher.state["tasks"])), 8)
            self.assertEqual(mock.releases[-1][1], [f"2000_{i}" for i in range(7)])
            for i, state in ((0, "COMPLETED"), (1, "FAILED"), (2, "CANCELLED")):
                mock.finish(f"2000_{i}", state)
            dispatcher.tick()
            self.assertEqual(len(mock.calls), 2)
            active = active_tasks(dispatcher.state["tasks"])
            self.assertEqual([sum(t["config"] == name for t in active) for name in CONFIGS], [7, 1, 0])
            self.assertEqual(sum(j["reason"] == "JobHeldUser" for j in mock.jobs.values()), 10)
            # Restart sees existing held/released elements and creates no duplicate array.
            dispatcher = ArrayDispatcher(root, dispatcher.batch, mock)
            dispatcher.tick()
            self.assertEqual(len(mock.calls), 2)
            while dispatcher.state.get("collection") is None:
                active = active_tasks(dispatcher.state["tasks"])
                for task in active[:4]:
                    mock.finish(task["job_id"])
                dispatcher.tick()
                self.assertLessEqual(len(active_tasks(dispatcher.state["tasks"])), 8)
            expected = [f"{2000+n}_{i}" for n, size in enumerate((10, 11, 3)) for i in range(size)]
            self.assertEqual([job_id for job_id, _ in mock.releases], expected)
            self.assertEqual(mock.max_active, 8)
            self.assertEqual(len(mock.calls), 4)  # three arrays + CPU collection
            mock.finish(dispatcher.state["collection"]["job_id"])
            self.assertTrue(dispatcher.tick())
            self.assertFalse(dispatcher.state["success"])

    def test_five_basic_plus_three_next_and_no_third_array(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mock = MockArrays()
            dispatcher = ArrayDispatcher(root, batch(root, (5, 8, 2)), mock)
            dispatcher.tick()
            self.assertEqual(len(mock.calls), 2)
            self.assertEqual([t["config"] for t in active_tasks(dispatcher.state["tasks"])],
                             [CONFIGS[0]] * 5 + [CONFIGS[1]] * 3)

    def test_unknown_completing_and_query_errors_never_release_extra_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mock = MockArrays()
            dispatcher = ArrayDispatcher(root, batch(root), mock)
            dispatcher.tick()
            mock.jobs["2000_0"].update(state="COMPLETING", reason="None")
            dispatcher.tick()
            self.assertEqual(len(mock.releases), 8)
            mock.jobs["2000_0"].update(state="COMPLETED", reason="None")
            dispatcher.tick()
            self.assertEqual(len(mock.releases), 9)  # do not wait for MinJobAge/purging
            mock.jobs.pop("2000_1")
            dispatcher.tick()
            self.assertEqual(len(mock.releases), 9)
            mock.fail_query = True
            with self.assertRaises(SlurmError):
                dispatcher.tick()
            self.assertEqual(len(mock.releases), 9)
            mock.fail_query = False
            mock.ended["2000_1"] = "COMPLETED"
            dispatcher.tick()
            self.assertEqual(len(mock.releases), 10)

    def test_uncertain_array_submission_and_release_resume_idempotently(self):
        for failure in ("uncertain_array", "uncertain_release"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                mock = MockArrays()
                setattr(mock, failure, True)
                directory = batch(root)
                with self.assertRaises((SlurmError, RuntimeError)):
                    ArrayDispatcher(root, directory, mock).tick()
                resumed = ArrayDispatcher(root, directory, mock)
                resumed.tick()
                self.assertEqual(len(mock.calls), 1)
                self.assertEqual(len(mock.releases), 8)
                self.assertEqual(len({job_id for job_id, _ in mock.releases}), 8)

    def test_command_adapter_expands_array_ids_and_uses_element_dependencies(self):
        slurm = ArraySlurm()
        with patch.object(slurm, "command", return_value="123_0|PENDING|JobHeldUser|basic\n123_1|RUNNING|None|basic\n"):
            self.assertEqual(set(slurm.queue()), {"123_0", "123_1"})
        with patch.object(slurm, "command", return_value="123_0|COMPLETED\n123_1|CANCELLED by 12\n123_1.batch|FAILED\n"):
            self.assertEqual(slurm.accounting(["123_0", "123_1"]), {"123_0": "COMPLETED", "123_1": "CANCELLED"})
        with patch.object(slurm, "command", return_value="") as command:
            slurm.release("123_1", ["123_0"])
            self.assertEqual(command.call_args_list[0].args[0],
                             ["scontrol", "update", "JobId=123_1", "Dependency=after:123_0"])
            self.assertEqual(command.call_args_list[1].args[0], ["scontrol", "release", "123_1"])


class CleanupTests(unittest.TestCase):
    def test_collection_none_cleanup_is_scoped_and_preserves_caches_and_other_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "plans" / "batch_old"
            directory.mkdir(parents=True)
            state = dict(version=1, root=str(root), pid=None, tasks=[dict(job_id="71958")], collection=None)
            (directory / "state.json").write_text(json.dumps(state))
            registry = root / "plans" / "active_batch.json"
            registry.write_text(json.dumps(dict(batch=str(directory))))
            cache = root / "caches" / "dataset.npz"
            cache.parent.mkdir()
            cache.write_bytes(b"original cache")
            manifest = root / "configs" / "caches.csv"
            manifest.parent.mkdir()
            manifest.write_text("original mapping")
            runs = root / "results" / "00_basic" / "runs"
            for name, job_id in (("aborted", "71958"), ("keep", "71843")):
                run = runs / name
                run.mkdir(parents=True)
                (run / "run.json").write_text(json.dumps(dict(job_id=job_id)))
                (run / "checkpoint.pt").write_bytes(b"checkpoint")
            calls = []
            def command(args, **kwargs):
                calls.append(args)
                return subprocess.CompletedProcess(args, 0, "71958|CANCELLED\n" if args[0] == "squeue" else "", "")
            with patch.object(cleanup, "ROOT", root), patch.object(cleanup.subprocess, "run", side_effect=command):
                self.assertEqual(cleanup.main(), 0)
                self.assertEqual(cleanup.main(), 0)  # safe second invocation
            self.assertFalse((runs / "aborted").exists())
            self.assertTrue((runs / "keep" / "checkpoint.pt").is_file())
            self.assertEqual(cache.read_bytes(), b"original cache")
            self.assertEqual(manifest.read_text(), "original mapping")
            self.assertFalse(directory.exists())
            self.assertFalse(registry.exists())
            self.assertEqual(calls[0], ["scancel", "71958"])

    def test_last_owned_run_removes_generated_provenance_and_keeps_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiment = root / "results" / "00_basic"
            run = experiment / "runs" / "CTCF" / "train_CHIP" / "seed_13" / "run1"
            run.mkdir(parents=True)
            (run / "run.json").write_text('{"job_id":"71958"}')
            (experiment / "sources" / "codehash").mkdir(parents=True)
            (experiment / "sources" / "codehash" / "model.py").write_text("snapshot")
            (experiment / "config.json").write_text("{}")
            (experiment / ".initialise.lock").touch()
            export_dir = root / "results" / "Results_all"
            export_dir.mkdir()
            (export_dir / "schema.json").write_text("schema")
            (export_dir / "00_basic.csv").write_text("generated")
            cleanup.remove_outputs(root, root / "plans" / "batch_old", dict(tasks=[dict(job_id="71958")]))
            self.assertFalse(experiment.exists())
            self.assertFalse((export_dir / "00_basic.csv").exists())
            self.assertEqual((export_dir / "schema.json").read_text(), "schema")


if __name__ == "__main__":
    unittest.main()
