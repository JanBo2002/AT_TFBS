"""Run every experiment through the same four shared Python modules."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from data import discover, read_manifest, read_priority, order_tasks
from training import (
    RunRecorder, atomic_json, metadata, package_hash, RunContext, digest,
    recipe_digest, invoke, load_config, collect, verify,
)

SCRIPT_VERSION = '5.2.1-task-profiles-default-long'


# Slurm access and shared eight-job dispatcher
# --------------------------------------------

TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "OUT_OF_MEMORY", "DEADLINE"}


class SlurmError(RuntimeError):
    pass


def state_name(value):
    return value.strip().split()[0].rstrip("+*").upper() if value.strip() else "UNKNOWN"


def resource_options(options):
    """Normalize resource options; the dispatcher owns job count and dependencies."""
    allowed = {"--account", "--qos", "--partition", "--time", "--cpus-per-task",
               "--mem", "--mem-per-cpu", "--mem-per-gpu", "--gres", "--gpus",
               "--gpus-per-node", "--gpus-per-task", "--constraint", "--reservation",
               "--nodelist", "--exclude", "--licenses", "--hint", "--nice"}
    result = []
    index = 0
    while index < len(options):
        token = options[index]
        key, separator, value = token.partition("=")
        if key == "--exclusive" and not separator:
            result.append(key)
        else:
            if key not in allowed:
                raise ValueError(f"Unsupported sbatch option {key}; use long resource/account options")
            if not separator:
                index += 1
                if index >= len(options) or options[index].startswith("--"):
                    raise ValueError(f"Missing value for {key}")
                value = options[index]
            if not value:
                raise ValueError(f"Empty value for {key}")
            result.append(f"{key}={value}")
        index += 1
    return result


class Slurm:
    def command(self, args):
        # Inherited SBATCH_ARRAY_INX or SBATCH_DEPENDENCY must not bypass the cap
        # or order. The activated Python/CUDA environment is otherwise retained.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SBATCH_") and k not in {"SLURM_CLUSTERS", "SLURM_FEDERATION"}}
        env["LC_ALL"] = "C"
        try:
            result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=45)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmError(f"{args[0]}: {exc}") from exc
        if result.returncode:
            raise SlurmError(f"{args[0]} failed: {result.stderr.strip() or result.stdout.strip()}")
        return result.stdout

    def submit(self, args):
        output = self.command(["sbatch", "--parsable", *args]).strip()
        if not re.fullmatch(r"[0-9]+(?:;[^\s;]+)?", output):
            raise SlurmError(f"Unrecognized sbatch receipt: {output!r}")
        return output.split(";", 1)[0]

    def queue(self):
        output = self.command(["squeue", "--local", "--me", "--states=all", "--noheader",
                               "--format=%i|%T|%j"])
        result = {}
        for line in output.splitlines():
            parts = line.strip().split("|", 2)
            if len(parts) == 3 and parts[0].isdigit():
                result[parts[0]] = {"state": state_name(parts[1]), "name": parts[2].strip()}
        return result

    def accounting(self, ids):
        output = self.command(["sacct", "--local", "--allocations", "--noheader", "--parsable2",
                               "--jobs=" + ",".join(ids), "--format=JobIDRaw,State%32"])
        result = {}
        for line in output.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 2 and parts[0] in ids:
                result[parts[0]] = state_name(parts[1])
        return result

    def recover(self, name, created_date):
        matches = {job_id for job_id, job in self.queue().items() if job["name"] == name}
        output = self.command(["sacct", "--local", "--allocations", "--noheader", "--parsable2",
                               f"--name={name}", f"--starttime={created_date}",
                               "--format=JobIDRaw,State%32,JobName%80"])
        for line in output.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[0].isdigit() and parts[2].strip() == name:
                matches.add(parts[0])
        if len(matches) != 1:
            raise RuntimeError(f"Submission receipt missing for {name}: found {len(matches)} jobs. "
                               "The batch is paused to avoid duplicate submissions. Check the Slurm logs, "
                               "then resume this batch once the job is visible.")
        return matches.pop()

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

    def collect_finished_configs(self, tasks):
        """Write each configuration's CSV as soon as its own jobs have ended."""
        summaries = self.state.setdefault("configuration_summaries", {})
        for name in dict.fromkeys(task["config"] for task in tasks):
            selected = [task for task in tasks if task["config"] == name]
            if name in summaries or not all(task["state"] in TERMINAL for task in selected):
                continue
            experiment = self.root / "results" / name
            if not (experiment / "config.json").is_file():
                # A job may fail before Python starts, leaving no result files.
                summaries[name] = {"status": "no_results"}
            else:
                try:
                    output = collect(self.root, name)
                    summaries[name] = {"status": "completed", "csv": str(output)}
                    print(f"Configuration completed; summary available: {output}", flush=True)
                except Exception as exc:
                    summaries[name] = {"status": "failed", "error": str(exc)}
                    print(f"Summary failed for {name}: {exc}", flush=True)
            self.save()

    def tick(self):
        tasks = self.state["tasks"]
        collection = self.state.get("collection")
        self.refresh(tasks + ([collection] if collection else []))
        self.collect_finished_configs(tasks)
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


# Submission, status and resume
# -----------------------------

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
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "run_experiment.py"), "submit", "_work", str(batch),
                                    "--lock-fd", str(lock.fileno())], cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lock.fileno(),))
    print(f"Dispatcher PID: {process.pid}; shared training-job limit: {MAX_JOBS}")
    print(f"Batch: {batch}")
    print(f"Log: {batch / 'launcher.log'}")
    print(f"Status: {sys.executable} {ROOT / 'run_experiment.py'} submit status {batch}")


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

def submission_main(argv=None):
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
    args = parser.parse_args(argv)
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
                raise ValueError(f"Unfinished batch exists: {current}. Resume it with run_experiment.py submit resume {current}")
            batch = prepare(args)
            atomic_json(ROOT / "plans" / "active_batch.json", {"batch": str(batch)})
        else:
            batch = args.batch.resolve()
            if batch != current:
                raise ValueError("Resume requires the current unfinished batch from this project")
            Dispatcher(ROOT, batch)  # Check root and frozen plans before detaching.
        background(batch, lock)
    return 0


# Experiment planning and execution
# ---------------------------------

ROOT = Path(__file__).resolve().parent


sys.path.insert(0,str(ROOT))


PLAN_FIELDS=["config_path","config_hash","tf","tf_ids_json","training_scope","train_assay","seed","cache_paths_json"]


def resolve_configs(args):
    paths = sorted((ROOT/"configs").glob("*.json")) if args.all_configs else args.config
    if not paths: raise ValueError("Select --config or --all-configs")
    result=[]
    for raw in paths:
        path=Path(raw)
        if not path.is_file():
            path=ROOT/"configs"/(str(raw) if str(raw).endswith(".json") else str(raw)+".json")
        path=path.resolve()
        config=load_config(path)
        if config["name"] in [c[1]["name"] for c in result]:
            raise ValueError(f"Duplicate experiment name {config['name']}")
        result.append((path,config))
    return result


def make_plan(args):
    configs=resolve_configs(args)
    manifest=read_manifest(args.manifest)
    tfs=list(dict.fromkeys(r["tf"] for r in manifest))
    if args.tf:
        absent=set(args.tf)-set(tfs)
        if absent: raise ValueError(f"TFs not in manifest: {sorted(absent)}")
        tfs=[tf for tf in tfs if tf in args.tf]
    lookup={(r["tf"],r["assay"]):r["cache"] for r in manifest}
    rows=[]
    for path,config in configs:
        selected_tfs = config.get("tfs", tfs)
        priority = getattr(args, "priority_file", ROOT/"configs"/"training_priority.csv")
        profile_pairs = None
        # Resolve the default here without adding a field to the stored JSON.
        if "tfs" not in config:
            profile = config.get("task_profile", "long")
            filename = "training_priority.csv" if profile == "long" else "training_priority_short.csv"
            priority = ROOT / "configs" / filename
            profile_pairs = read_priority(priority)
            selected_tfs = list(dict.fromkeys(tf for tf, assay in profile_pairs))
        absent = set(selected_tfs) - {row["tf"] for row in manifest}
        if absent:
            raise ValueError(f"Configured TFs not in manifest: {sorted(absent)}")
        selected_tfs = [tf for tf in tfs if tf in selected_tfs]
        if "planner" in config["workflow"]:
            custom = invoke(config["workflow"]["planner"],config,manifest,
                tf_filter=selected_tfs if "tfs" in config or "task_profile" in config else args.tf,
                train_assays=args.train_assay,seeds=args.seed)
            for row in custom:
                if set(row) != set(PLAN_FIELDS)-{"config_path","config_hash"}:
                    raise ValueError("Custom planner returned an invalid task schema")
                rows.append(dict(row,config_path=str(path),config_hash=digest(config)))
            continue
        config_rows=[]
        for tf in selected_tfs:
            assays=args.train_assay or config["train_assays"]
            if set(assays)-set(config["train_assays"]):
                raise ValueError("Requested source assay is not enabled in the configuration")
            needed=set(assays)|set(config["test_assays"])
            missing=[a for a in needed if (tf,a) not in lookup]
            if missing: raise ValueError(f"Missing caches for {tf}: {missing}")
            cache_paths={a:str(lookup[(tf,a)]) for a in needed}
            for assay in assays:
                if profile_pairs is not None and (tf, assay) not in profile_pairs:
                    continue
                for seed in args.seed or config["seeds"]:
                    config_rows.append({"config_path":str(path),"config_hash":digest(config),"tf":tf,
                        "tf_ids_json":json.dumps([tf]),"training_scope":"single_tf",
                        "train_assay":assay,"seed":seed,"cache_paths_json":json.dumps(cache_paths,sort_keys=True)})
        rows.extend(order_tasks(config_rows, priority))
    return rows


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        self.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def execute(row,args):
    config_path=Path(row["config_path"])
    config=load_config(config_path)
    if digest(config)!=row["config_hash"]:
        raise ValueError("Configuration changed since planning; create a new job plan")
    code_hash=package_hash(ROOT)
    experiment=ROOT/"results"/config["name"]
    experiment.mkdir(parents=True,exist_ok=True)
    with (experiment/".initialise.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        stored=experiment/"config.json"
        if stored.exists() and recipe_digest(json.loads(stored.read_text()))!=recipe_digest(config):
            raise ValueError(f"Experiment {config['name']} already has another configuration. Assign a new experiment name.")
        atomic_json(stored,config)
        write_versions(experiment,code_hash)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_"+uuid.uuid4().hex[:8]
    seed=int(row["seed"])
    tf=row["tf"]
    assay=row["train_assay"]
    tf_ids=tuple(json.loads(row["tf_ids_json"]))
    group_name=tf or "TFGROUP_"+digest(tf_ids)[:12]
    run_dir=experiment/"runs"/group_name/f"train_{assay}"/f"seed_{seed:02d}"
    run_dir.parent.mkdir(parents=True,exist_ok=True)
    paths={a:Path(p) for a,p in json.loads(row["cache_paths_json"]).items()}
    meta=metadata(config,code_hash,run_id,tf,assay,seed,paths.get(assay,""))
    meta.update(training_scope=row["training_scope"],tf_ids_json=json.dumps(tf_ids))
    recorder=RunRecorder(meta,run_dir)
    context=RunContext(config,config_path,ROOT,run_dir,tf,assay,seed,paths,recorder,
        device=args.device,workers=args.workers,tf_ids=tf_ids)
    atomic_json(run_dir/"config.json",config)
    with (run_dir/"train.log").open("w",buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print(f"Starting {config['name']} | {tf} | {assay} | seed {seed}")
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                invoke(config["workflow"],context)
                recorder.finish()
                print(f"Completed: {run_dir}")
            except BaseException as exc:
                traceback.print_exc()
                recorder.finish(status="failed",error=f"{type(exc).__name__}: {exc}")
                raise
    return config["name"]


def select_group(parser):
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",nargs="+",help="Config path or name(s) from configs/")
    group.add_argument("--all-configs",action="store_true")


def plan_options(parser):
    select_group(parser)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--tf",nargs="+",help="Optional subset; task order follows the priority list")
    parser.add_argument("--priority-file",type=Path,default=ROOT/"configs"/"training_priority.csv")
    parser.add_argument("--train-assay",nargs="+",choices=["CHIP","GHTS"])
    parser.add_argument("--seed",nargs="+",type=int,help="Default: seeds specified in each configuration")


def runtime_options(parser):
    parser.add_argument("--device",default="auto")
    parser.add_argument("--workers",type=int,default=None,help="Explicit override of the original worker setting")


# Version recording
# -----------------

def write_versions(experiment, code_hash):
    """Record script versions without copying Python files into results."""
    lines = [f"code_sha256: {code_hash}"]
    for filename in ("run_experiment.py", "model.py", "data.py", "training.py"):
        path = ROOT / filename
        match = re.search(r"^SCRIPT_VERSION\s*=\s*['\"]([^'\"]+)", path.read_text(), re.MULTILINE)
        lines.append(f"{filename}: {match.group(1) if match else 'unversioned'}")
    destination = experiment / "versions.txt"
    content = "\n".join(lines) + "\n"
    if destination.exists() and destination.read_text() != content:
        raise ValueError("Experiment already uses another code version; choose a new experiment name")
    destination.write_text(content)



# Command-line interface
# ----------------------

def main():
    if sys.argv[1:2] == ["submit"]:
        return submission_main(sys.argv[2:])
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("list",help="List the three experiment configurations")
    manifest=sub.add_parser("manifest",help="Discover caches; ambiguous TF/assay names are rejected")
    manifest.add_argument("--cache-root",type=Path,required=True)
    manifest.add_argument("--output",type=Path,default=ROOT/"configs"/"caches.csv")
    plan=sub.add_parser("plan",help="Write a reproducible task table for SLURM")
    plan_options(plan)
    plan.add_argument("--output",type=Path,required=True)
    run=sub.add_parser("run",help="Execute selected tasks sequentially, then collect results")
    plan_options(run)
    runtime_options(run)
    task=sub.add_parser("task",help="Execute exactly one row of a task table")
    task.add_argument("--plan",type=Path,required=True)
    task.add_argument("--index",type=int,required=True)
    runtime_options(task)
    collect_parser=sub.add_parser("collect",help="Export standard CSVs and evaluation.md")
    select_group(collect_parser)
    verify_parser=sub.add_parser("verify-basic",help="Check model, optimizer, dataset and inference parity against the frozen original")
    verify_parser.add_argument("--reference", type=Path, required=True, help="Original motif64 script used for parity checks")
    verify_parser.add_argument("--output",type=Path,default=ROOT/"verification.json")
    args=parser.parse_args()
    if hasattr(args,"workers") and args.workers is not None and args.workers<0:
        parser.error("--workers must be nonnegative")
    if hasattr(args,"seed") and args.seed and (len(args.seed)!=len(set(args.seed)) or any(s<0 for s in args.seed)):
        parser.error("--seed values must be unique and nonnegative")
    if args.command=="list":
        for p in sorted((ROOT/"configs").glob("*.json")):
            c=load_config(p)
            print(c["name"],c["predictor"]["params"].get("motif_kernels"),"seeds",c["seeds"])
    elif args.command=="manifest":
        discover(args.cache_root,args.output)
        print(args.output.resolve())
    elif args.command=="plan":
        rows=make_plan(args)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=PLAN_FIELDS)
            writer.writeheader();writer.writerows(rows)
        print(f"{len(rows)} tasks -> {args.output.resolve()}")
    elif args.command=="task":
        with args.plan.open(newline="") as handle: rows=list(csv.DictReader(handle))
        if not 0<=args.index<len(rows): raise IndexError("Task index is out of bounds")
        execute(rows[args.index],args)
    elif args.command=="run":
        rows=make_plan(args)
        remaining=Counter(row["config_path"] for row in rows)
        failed=0
        for row in rows:
            name=load_config(Path(row["config_path"]))["name"]
            try: execute(row,args)
            except Exception:
                failed+=1
                print(f"Task failed: {row['tf']} {row['train_assay']} seed {row['seed']}",file=sys.stderr)
            remaining[row["config_path"]]-=1
            if remaining[row["config_path"]]==0 and (ROOT/"results"/name/"config.json").is_file():
                collect(ROOT,name)
        if failed: return 1
    elif args.command=="collect":
        for path,config in resolve_configs(args):
            if (ROOT/"results"/config["name"]/"config.json").exists():
                collect(ROOT,config["name"])
    elif args.command=="verify-basic":
        verify(ROOT,args.output,args.reference)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
