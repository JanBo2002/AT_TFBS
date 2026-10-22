"""Per-run logging; workers never append to a shared experiment CSV."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .schema import COLUMNS, METRICS, SCHEMA_VERSION
from .specs import digest, recipe_digest, json_text


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    if hasattr(value, "item"): return clean(value.item())
    return value


def atomic_json(path: Path, payload):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(clean(payload), indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def package_hash(root: Path):
    result = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "results" in path.relative_to(root).parts or "__pycache__" in path.parts:
            continue
        result.update(str(path.relative_to(root)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


class RunRecorder:
    def __init__(self, context_metadata, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.meta = dict(context_metadata, status="running", started_at=now())
        self.base = {k:v for k,v in self.meta.items() if k in COLUMNS}
        self.base.update(schema_version=SCHEMA_VERSION, included_in_summary=False)
        self.epoch_rows = []
        self.fit_loop_seconds = None
        self.update()

    def update(self, **values):
        self.meta.update(clean(values))
        if hasattr(self, "base"):
            for key in {"device","device_name","torch_version","train_cache_sha256","split_hash"} & values.keys():
                self.base[key] = clean(values[key])
        atomic_json(self.run_dir / "run.json", self.meta)

    def record(self, row_type, **values):
        if row_type not in {"run","epoch","evaluation","analysis"}:
            raise ValueError(f"Invalid per-run row type {row_type}")
        unknown = set(values) - set(COLUMNS)
        if unknown:
            raise ValueError(f"Unknown result columns: {sorted(unknown)}; use extra_json or analysis metric fields")
        row = clean(dict(self.base, row_type=row_type, **values))
        with (self.run_dir / "records.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False, default=str) + "\n")
        if row_type == "epoch":
            self.epoch_rows.append(row)
        return row

    def epoch(self, original):
        row = {
            "epoch": original["epoch"], "status":"completed",
            "epoch_s": original["elapsed_seconds"],
            "train_s": original["train_seconds"], "validation_s":original["validation_seconds"],
            "mining_s":original["mining_seconds"],
            "learning_rate_used":original["learning_rate_used"],
            "learning_rate_next":original["learning_rate_next"],
            "train_loss":original["train"]["loss"]["total"],
            "val_loss":original["val"]["loss"]["total"],
            "negative_pool_coverage":original["sampling"].get("negative_pool_coverage"),
            "sampling_hard_count":original["sampling"].get("hard_negative"),
            "sampling_cyclic_count":original["sampling"].get("cyclic_negative"),
            "extra_json":json_text({"sampling":original["sampling"],"train_shift":original["train_shift"]}),
        }
        for split in ("train","val"):
            for metric,value in original[split]["classification"].items():
                if metric in METRICS: row[f"{split}_{metric}"] = value
        self.record("epoch", **row)

    def fit_loop_finished(self, seconds, history):
        self.fit_loop_seconds = seconds
        self.update(fit_loop_s=seconds, epochs_run=len(history))

    def finish(self, status="completed", error=""):
        import numpy as np
        training = self.run_dir / "training"
        config_file = training / "configuration.json"
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        arguments = config.get("arguments",self.meta.get("resolved_training_arguments",{}))
        history = self.epoch_rows
        durations = [r["epoch_s"] for r in history if r.get("epoch_s") is not None]
        eval_files = list((self.run_dir / "evaluations").glob("*/*/metrics.json"))
        evaluations = [json.loads(p.read_text()) for p in eval_files]
        inference_s = sum(e.get("inference_s",0) for e in evaluations)
        eval_wall = sum(e.get("evaluation_wall_s",0) for e in evaluations)
        checkpoint_file = training / "prediction_metadata.json"
        checkpoint = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else {}
        device = self.meta.get("device", "")
        torch_version = ""
        device_name = device
        peak_memory = None
        try:
            import torch
            torch_version = str(torch.__version__)
            if device.startswith("cuda") and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(torch.device(device))
                peak_memory = torch.cuda.max_memory_allocated(torch.device(device))
        except ImportError:
            pass
        requested = arguments.get("epochs")
        train_wall = self.meta.get("training_wall_s")
        fit_loop = self.fit_loop_seconds
        train_total = sum(r.get("train_s") or 0 for r in history) if history else None
        val_total = sum(r.get("validation_s") or 0 for r in history) if history else None
        mining_total = sum(r.get("mining_s") or 0 for r in history) if history else None
        row = {
            "status":status, "error":error, "finished_at":now(),
            "training_wall_s":train_wall, "fit_loop_s":fit_loop,
            "train_total_s":train_total, "validation_total_s":val_total, "mining_total_s":mining_total,
            "setup_s":self.meta.get("setup_s"),
            "post_fit_s":None if train_wall is None or fit_loop is None else max(0,train_wall-fit_loop-(self.meta.get("setup_s") or 0)),
            "epochs_requested":requested, "epochs_run":len(history) if history or self.meta.get("config",{}).get("fitting") else None,
            "early_stopped":None if requested is None else len(history)<requested,
            "best_epoch":checkpoint.get("epoch"),
            "checkpoint_objective":checkpoint.get("checkpoint_objective"),
            "validation_selection_score":checkpoint.get("validation_score"),
            "checkpoint_path":checkpoint.get("checkpoint_path"),
            "checkpoint_sha256":checkpoint.get("checkpoint_sha256"),
            "threshold":checkpoint.get("classification_threshold"),
            "threshold_strategy":arguments.get("threshold_strategy"),
            "parameter_count":config.get("parameter_count"),
            "trainable_parameter_count":checkpoint.get("trainable_parameter_count"),
            "input_length":config.get("model",{}).get("sequence_length"),
            "model_config_json":json_text(config.get("model",{})),
            "training_config_json":json_text(arguments),
            "inference_config_json":json_text(self.meta.get("config",{}).get("inference",{})),
            "batch_size":arguments.get("batch_size"), "eval_batch_size":arguments.get("eval_batch_size"),
            "num_workers":arguments.get("num_workers"), "negatives_per_positive":arguments.get("negatives_per_positive"),
            "pos_weight":config.get("class_counts",{}).get("pos_weight"),
            "weight_decay":arguments.get("weight_decay"), "learning_rate_initial":arguments.get("learning_rate"),
            "patience":arguments.get("patience"), "min_checkpoint_epoch":arguments.get("min_checkpoint_epoch"),
            "train_cache_sha256":config.get("cache_sha256"),
            "samples_per_training_epoch":config.get("class_counts",{}).get("samples_per_training_epoch"),
            "amp_enabled":checkpoint.get("amp_enabled"), "torch_version":torch_version,
            "device_name":device_name, "gpu_count":int(device.startswith("cuda")),
            "gpu_peak_memory_bytes":peak_memory, "inference_s":inference_s,
            "evaluation_wall_s":eval_wall, "total_wall_s":time.perf_counter()-self.started,
        }
        for stem,source in (("train","train"),("val","val"),("test","test")):
            indices = config.get("splits",{}).get(source,[])
            counts = self.meta.get("split_counts",{}).get(source,{})
            row[f"{stem}_positive"] = counts.get("positive")
            row[f"{stem}_negative"] = counts.get("negative")
        kernels = config.get("model",{}).get("motif_kernels",[])
        channels = config.get("model",{}).get("embed_channels",0)
        row["kernels_json"] = json_text(kernels)
        if kernels:
            b,r = divmod(channels,len(kernels))
            row["channels_per_branch_json"] = json_text([b+(i<r) for i in range(len(kernels))])
        if durations:
            for name,fn in (("mean",np.mean),("median",np.median),("min",np.min),("max",np.max)):
                row[f"epoch_{name}_s"] = float(fn(durations))
            row["epoch_std_s"] = float(np.std(durations,ddof=1)) if len(durations)>1 else None
            row["epoch_p95_s"] = float(np.percentile(durations,95))
        # Selected-checkpoint validation metrics are distinct from the per-epoch
        # metrics at the provisional 0.5 threshold. Preserve both in the CSV.
        val_file=training/"validation_metrics.json"
        if val_file.exists():
            best_val=json.loads(val_file.read_text())
            row["val_loss"]=best_val.get("loss",{}).get("total")
            for name,value in best_val.get("classification",{}).items():
                if name in METRICS: row[f"val_{name}"]=value
        best_history=next((r for r in history if r.get("epoch")==checkpoint.get("epoch")),None)
        if best_history:
            row["train_loss"]=best_history.get("train_loss")
            for name in METRICS:
                if f"train_{name}" in best_history:
                    row[f"train_{name}"]=best_history[f"train_{name}"]
        row.update(self.meta.get("run_metrics", {}))
        self.record("run", **row)
        self.update(status=status,error=error,finished_at=row["finished_at"],total_wall_s=row["total_wall_s"])


def metadata(config, code_hash, run_id, tf, train_assay, seed, train_cache):
    return {
        "experiment_id":config["name"], "reference_id":config["reference"],
        "config_hash":recipe_digest(config), "code_hash":code_hash, "config":config,
        "run_id":run_id, "model_id":run_id, "training_scope":"single_tf",
        "tf":tf, "tf_ids_json":json_text([tf]), "train_assay":train_assay,
        "seed":seed, "train_cache_path":str(train_cache), "hostname":socket.gethostname(),
        "job_id":os.environ.get("SLURM_ARRAY_JOB_ID",os.environ.get("SLURM_JOB_ID","")),
        "array_task_id":os.environ.get("SLURM_ARRAY_TASK_ID",""),
        "python_version":platform.python_version(),
    }
