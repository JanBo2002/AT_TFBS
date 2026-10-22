#!/usr/bin/env python3
"""Human301 ablations: configuration, execution, SLURM tasks and CSV collection."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import os
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from ablation.providers import discover, read_manifest
from ablation.priority import order_tasks
from ablation.recorder import RunRecorder, atomic_json, metadata, package_hash
from ablation.specs import RunContext, digest, recipe_digest, invoke, load_config


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
        if "planner" in config["workflow"]:
            custom = invoke(config["workflow"]["planner"],config,manifest,
                tf_filter=args.tf,train_assays=args.train_assay,seeds=args.seed)
            for row in custom:
                if set(row) != set(PLAN_FIELDS)-{"config_path","config_hash"}:
                    raise ValueError("Custom planner returned an invalid task schema")
                rows.append(dict(row,config_path=str(path),config_hash=digest(config)))
            continue
        config_rows=[]
        for tf in tfs:
            assays=args.train_assay or config["train_assays"]
            if set(assays)-set(config["train_assays"]):
                raise ValueError("Requested source assay is not enabled in the configuration")
            needed=set(assays)|set(config["test_assays"])
            missing=[a for a in needed if (tf,a) not in lookup]
            if missing: raise ValueError(f"Missing caches for {tf}: {missing}")
            cache_paths={a:str(lookup[(tf,a)]) for a in needed}
            for assay in assays:
                for seed in args.seed or config["seeds"]:
                    config_rows.append({"config_path":str(path),"config_hash":digest(config),"tf":tf,
                        "tf_ids_json":json.dumps([tf]),"training_scope":"single_tf",
                        "train_assay":assay,"seed":seed,"cache_paths_json":json.dumps(cache_paths,sort_keys=True)})
        rows.extend(order_tasks(config_rows,getattr(args,"priority_file",ROOT/"configs"/"training_priority.csv")))
    return rows


def snapshot_code(destination, code_hash):
    target=destination/"sources"/code_hash
    if (target/"COMPLETE").exists(): return
    target.mkdir(parents=True,exist_ok=True)
    for path in ROOT.rglob("*.py"):
        rel=path.relative_to(ROOT)
        if "results" in rel.parts or "__pycache__" in rel.parts: continue
        output=target/rel
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_name(output.name+f".{os.getpid()}.tmp")
        shutil.copyfile(path,tmp)
        tmp.replace(output)
    (target/"COMPLETE").write_text(code_hash+"\n")


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
        snapshot_code(experiment,code_hash)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_"+uuid.uuid4().hex[:8]
    seed=int(row["seed"])
    tf=row["tf"]
    assay=row["train_assay"]
    tf_ids=tuple(json.loads(row["tf_ids_json"]))
    group_name=tf or "TFGROUP_"+digest(tf_ids)[:12]
    run_dir=experiment/"runs"/group_name/f"train_{assay}"/f"seed_{seed:02d}"/run_id
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


def main():
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
    collect=sub.add_parser("collect",help="Export standard CSVs and evaluation.md")
    select_group(collect)
    verify=sub.add_parser("verify-basic",help="Check model, optimizer, dataset and inference parity against the frozen original")
    verify.add_argument("--output",type=Path,default=ROOT/"verification.json")
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
        from ablation.collect import collect
        rows=make_plan(args)
        completed=[]
        failed=0
        for row in rows:
            name=load_config(Path(row["config_path"]))["name"]
            if name not in completed: completed.append(name)
            try: execute(row,args)
            except Exception:
                failed+=1
                print(f"Task failed: {row['tf']} {row['train_assay']} seed {row['seed']}",file=sys.stderr)
        for name in completed: collect(ROOT,name)
        if failed: return 1
    elif args.command=="collect":
        from ablation.collect import collect
        for path,config in resolve_configs(args):
            if (ROOT/"results"/config["name"]/"config.json").exists():
                collect(ROOT,config["name"])
    elif args.command=="verify-basic":
        from ablation.verification import verify
        verify(ROOT,args.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
