"""Shared training, inference, metrics and result export for all experiments."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import socket
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from scipy.stats import pearsonr, spearmanr

from data import STORED_LENGTH, evaluation_crop_starts, load_cache, sha256_file, seed_everything, canonical_chrom, json_default, TFBindingDataset, CyclicBalancedNegativeSampler, build_or_load_cache, make_splits, draw_shifts, cache_geometry, make_fixed_balanced_evaluation_split, make_loader, resolve_device
from model import HumanTFBindingModel, ModelConfig, count_parameters

SCRIPT_VERSION = '5.1.0-input-channel-dropout'


# Configuration loading and extension points
# ------------------------------------------

def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def recipe_digest(config: dict[str, Any]) -> str:
    # Seed/assay lists select tasks; their actual values are recorded per run.
    # Adding repetitions does not create another model/training recipe.
    return digest({k:v for k,v in config.items() if k not in {"description","seeds","train_assays","test_assays"}})


def factory(spec: dict[str, Any]) -> Callable:
    module_name, sep, attr = spec["factory"].partition(":")
    if not sep or not module_name or not attr:
        raise ValueError("A factory must use the form 'package.module:function'")
    legacy_modules = {
        "ablation.providers": "data", "ablation.builders": "model",
        "ablation.fitting": "training", "ablation.inference": "training",
        "ablation.workflows": "training", "ablation.planning": "training",
    }
    module_name = legacy_modules.get(module_name, module_name)
    fn = getattr(importlib.import_module(module_name), attr)
    if not callable(fn):
        raise TypeError(f"Factory is not callable: {spec['factory']}")
    return fn


def invoke(spec: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
    return factory(spec)(*args, **copy.deepcopy(spec.get("params", {})), **kwargs)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {"schema_version", "name", "reference", "workflow", "data", "predictor",
                "fitting", "inference", "analyses", "seeds", "train_assays", "test_assays"}
    allowed = required | {"description"}
    if required - config.keys() or config.keys() - allowed:
        raise ValueError(f"Invalid configuration keys: missing={required-config.keys()}, unknown={config.keys()-allowed}")
    if config["schema_version"] != "1.0":
        raise ValueError("Unsupported configuration schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["name"]):
        raise ValueError("Experiment names must be safe single directory names")
    for key in ("workflow", "data", "predictor", "inference"):
        validate_component(config[key], key)
    if config["fitting"] is not None:
        validate_component(config["fitting"], "fitting")
    for spec in config["analyses"]:
        validate_component(spec, "analysis")
    if not config["seeds"] or any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate seeds")
    for key in ("train_assays", "test_assays"):
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError(f"{key} must be nonempty and unique")
        if any(a not in {"CHIP", "GHTS"} for a in config[key]):
            raise ValueError("Human301 recipes support CHIP and GHTS")
    return config


def validate_component(spec: Any, label: str) -> None:
    allowed = {"factory","params","planner"} if label == "workflow" else {"factory","params"}
    if not isinstance(spec, dict) or not {"factory","params"} <= set(spec) or set(spec)-allowed:
        raise ValueError(f"{label}: expected exactly factory and params")
    if not isinstance(spec["factory"], str) or ":" not in spec["factory"] or not isinstance(spec["params"], dict):
        raise ValueError(f"Invalid {label} component")
    if "planner" in spec:
        validate_component(spec["planner"], "planner")


@dataclass
class RunContext:
    config: dict[str, Any]
    config_path: Path
    root: Path
    run_dir: Path
    tf: str
    train_assay: str
    seed: int
    cache_paths: dict[str, Path]
    recorder: Any
    device: str = "auto"
    workers: int | None = None
    tf_ids: tuple[str, ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)


class Predictor(Protocol):
    """Custom workflows may adapt neural nets, scanners, or external estimators.

    Scores need not be probabilities. Brier/log-loss must only be emitted when
    probabilities exist. Models requiring gradients can expose extra methods.
    """
    def predict_scores(self, batch: Any) -> Any: ...


class Workflow(Protocol):
    def __call__(self, context: RunContext, **params: Any) -> None: ...


# CSV schema and run recording
# ----------------------------

SCHEMA_VERSION = "1.0"


METRICS = [
    "auprc", "auroc", "mcc", "f1", "accuracy", "balanced_accuracy", "precision",
    "recall", "specificity", "negative_predictive_value", "true_positive",
    "false_positive", "true_negative", "false_negative", "threshold", "brier_score",
    "log_loss", "sample_count", "positive_count", "negative_count", "pr_baseline",
]


TIMES = [
    "training_wall_s", "fit_loop_s", "train_total_s", "validation_total_s", "mining_total_s",
    "setup_s", "post_fit_s", "epoch_mean_s", "epoch_median_s", "epoch_std_s",
    "epoch_min_s", "epoch_max_s", "epoch_p95_s", "epoch_s", "train_s", "validation_s",
    "mining_s", "inference_s", "evaluation_wall_s", "total_wall_s", "samples_per_s",
]


COLUMNS = [
    "schema_version", "row_type", "experiment_id", "reference_id", "config_hash", "code_hash",
    "run_id", "model_id", "training_scope", "tf", "tf_ids_json", "train_assay", "test_assay",
    "direction", "seed", "epoch", "evaluation_id", "status", "included_in_summary",
    "summary_level", "summary_source", "statistic", "aggregation_unit", "n_records",
    "n_runs", "n_tfs", "n_seeds", "started_at", "finished_at", "hostname", "job_id",
    "array_task_id", "device", "device_name", "gpu_count", "torch_version", "python_version",
    "amp_enabled", "gpu_peak_memory_bytes", "parameter_count", "trainable_parameter_count",
    "input_length", "kernels_json", "channels_per_branch_json", "model_config_json",
    "training_config_json", "inference_config_json", "epochs_requested", "epochs_run",
    "best_epoch", "early_stopped", "learning_rate_used", "learning_rate_next", "batch_size",
    "eval_batch_size", "num_workers", "negatives_per_positive", "pos_weight", "weight_decay",
    "learning_rate_initial", "patience", "min_checkpoint_epoch", "checkpoint_objective",
    "checkpoint_path", "checkpoint_sha256", "threshold_strategy", "validation_selection_score",
    "train_cache_path", "train_cache_sha256", "test_cache_path", "test_cache_sha256", "split_hash",
    "train_positive", "train_negative", "val_positive", "val_negative", "test_positive",
    "test_negative", "samples_per_training_epoch", "sampling_hard_count", "sampling_cyclic_count",
    "negative_pool_coverage", "evaluation_crops_json", "evaluation_rc", "forward_views",
    "train_loss", "val_loss", "classification_loss", "regression_loss", "total_loss",
] + METRICS + [f"train_{m}" for m in METRICS] + [f"val_{m}" for m in METRICS] + TIMES + [
    "delta_auprc", "ratio_auprc", "log2_ratio_auprc", "win_fraction", "paired_reference_run_id",
    "metric_name", "metric_value", "metric_unit", "analysis_name", "error", "extra_json",
]


assert len(COLUMNS) == len(set(COLUMNS))


SUM_FIELDS = set(TIMES) - {"epoch_mean_s","epoch_median_s","epoch_std_s","epoch_min_s",
    "epoch_max_s","epoch_p95_s","samples_per_s"}


SUM_FIELDS |= {"epochs_run", "sample_count", "positive_count", "negative_count",
    "true_positive", "false_positive", "true_negative", "false_negative"}


NUMERIC = set(METRICS + TIMES + [f"train_{m}" for m in METRICS] + [f"val_{m}" for m in METRICS]) | {
    "parameter_count", "trainable_parameter_count", "epochs_requested", "epochs_run", "best_epoch",
    "gpu_peak_memory_bytes", "train_loss", "val_loss", "classification_loss", "regression_loss",
    "total_loss", "delta_auprc", "ratio_auprc", "log2_ratio_auprc", "win_fraction", "metric_value",
}

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


# Training defaults
# -----------------

DEFAULTS = {'dataset_root': None,
 'rebuild_cache': False,
 'fe_transform': 'none',
 'train_chromosomes': ('1', '2', '3', '4'),
 'test_chromosomes': ('5',),
 'val_fraction': 0.1,
 'val_block_size': 100000,
 'val_milp_time_limit': 120.0,
 'min_split_samples_per_class': 2,
 'epochs': 50,
 'batch_size': 88,
 'num_workers': 4,
 'input_length': 256,
 'embed_channels': 64,
 'encoder_channels': (80, 96, 128),
 'heads': 4,
 'ffn_channels': 256,
 'negatives_per_positive': 10,
 'hard_negative_fraction': 0.0,
 'hard_negative_start_epoch': 6,
 'hard_negative_refresh': 5,
 'eval_batch_size': 128,
 'eval_crops': 'multi',
 'eval_rc': True,
 'skip_test': False,
 'learning_rate': 0.0003,
 'min_learning_rate': 1e-06,
 'weight_decay': 0.0001,
 'adam_beta1': 0.9,
 'adam_beta2': 0.999,
 'adam_eps': 1e-08,
 'dropout': 0.1,
 'activation': 'relu',
 'gradient_clip': 1.0,
 'patience': 10,
 'min_delta': 0.0001,
 'min_checkpoint_epoch': 1,
 'seed': 13,
 'device': 'auto',
 'amp': True,
 'reverse_complement_probability': 0.5,
 'train_shift_max': 23,
 'train_shift_distribution': 'uniform',
 'eval_shift': 'fixed',
 'eval_shift_max': None,
 'balanced_evaluation': False,
 'max_eval_positive_reduction': 1.0,
 'task_mode': 'classification',
 'classification_readout': 'center',
 'regression_mode': 'profile',
 'regression_weight': 0.3,
 'checkpoint_objective': 'auto',
 'export_full_regression_profiles': False,
 'export_bigwig': False,
 'export_target_bigwig': False,
 'bigwig_output': None,
 'target_bigwig_output': None,
 'bigwig_aggregation': 'center-weighted',
 'bigwig_chrom_prefix': 'Chr',
 'chrom_sizes': None,
 'threshold_strategy': 'val-mcc',
 'classification_threshold': 0.5,
 'regression_positive_only': False,
 'smooth_l1_beta': 0.5,
 'allow_negative_regression_output': False}


def training_namespace():
    return argparse.Namespace(**copy.deepcopy(DEFAULTS))


# Losses and metrics
# ------------------

TaskMode = Literal["classification", "regression", "multitask"]


RegressionMode = Literal["profile", "center", "max", "mean"]


CheckpointObjective = Literal[
    "auto",
    "classification-auprc",
    "regression-loss",
    "regression-profile-mae",
    "regression-profile-pearson",
]


BigWigAggregation = Literal["mean", "center-weighted", "max"]


@dataclass
class TrainingConfig:
    task_mode: TaskMode = "multitask"
    regression_mode: RegressionMode = "profile"
    regression_weight: float = 1.0
    regression_positive_only: bool = False
    smooth_l1_beta: float = 0.5


def scalar_from_track(
    track: torch.Tensor,
    center_indices: torch.Tensor,
    mode: Literal["center", "max", "mean"],
) -> torch.Tensor:
    if mode == "center":
        return HumanTFBindingModel.gather_positions(track, center_indices)
    if mode == "max":
        return track.max(dim=1).values
    if mode == "mean":
        return track.mean(dim=1)
    raise ValueError(f"Unknown scalar track mode: {mode}")


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    classification_criterion: nn.Module,
    training_config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = batch["label"]
    classification_loss = classification_criterion(
        outputs["classification_logit"], labels
    )

    if training_config.task_mode == "classification":
        # Keep the regression head in the architecture for later multitask runs,
        # but do not include it in the optimization objective.
        regression_loss = classification_loss.new_zeros(())
        total = classification_loss
    else:
        prediction = outputs["regression_track"]
        target = batch["fe_profile"]
        valid_mask = batch["valid_mask"]
        mode = training_config.regression_mode

        if mode == "profile":
            elementwise = F.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
                beta=training_config.smooth_l1_beta,
            )
            mask = valid_mask
            if training_config.regression_positive_only:
                mask = mask * labels[:, None]
            denominator = mask.sum().clamp_min(1.0)
            regression_loss = (elementwise * mask).sum() / denominator
        else:
            scalar_prediction = scalar_from_track(prediction, batch["center_index"], mode)
            scalar_target = scalar_from_track(target, batch["center_index"], mode)
            elementwise = F.smooth_l1_loss(
                scalar_prediction,
                scalar_target,
                reduction="none",
                beta=training_config.smooth_l1_beta,
            )
            if training_config.regression_positive_only:
                denominator = labels.sum().clamp_min(1.0)
                regression_loss = (elementwise * labels).sum() / denominator
            else:
                regression_loss = elementwise.mean()

        if training_config.task_mode == "regression":
            # Regression-only training: the classification head remains in the model,
            # but classification_loss is excluded from the optimization objective.
            total = regression_loss
        else:
            total = classification_loss + training_config.regression_weight * regression_loss
    return total, {
        "classification": classification_loss.detach(),
        "regression": regression_loss.detach(),
        "total": total.detach(),
    }

def safe_correlation(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    result = function(y_true, y_pred)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / max(1, tn + fp)
    negative_predictive_value = tn / max(1, tn + fn)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
    }
    if np.unique(labels).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["auprc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def select_classification_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    strategy: str,
    fixed_threshold: float = 0.5,
) -> tuple[float, float]:
    """Choose a binary threshold using validation predictions only.

    The sweep is exact over all unique predicted probabilities and is O(n log n).
    The test chromosome is never used to select the threshold.
    """
    if strategy == "fixed":
        return float(fixed_threshold), float("nan")
    if np.unique(labels).size != 2:
        raise ValueError("Threshold selection requires both classes in validation data")

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_prob = probabilities[order]
    sorted_label = labels[order].astype(np.int64)
    cumulative_tp = np.cumsum(sorted_label)
    cumulative_fp = np.cumsum(1 - sorted_label)
    group_ends = np.flatnonzero(
        np.r_[sorted_prob[1:] != sorted_prob[:-1], True]
    )

    tp = cumulative_tp[group_ends].astype(np.float64)
    fp = cumulative_fp[group_ends].astype(np.float64)
    positives = float(sorted_label.sum())
    negatives = float(sorted_label.size - sorted_label.sum())
    fn = positives - tp
    tn = negatives - fp
    thresholds = sorted_prob[group_ends].astype(np.float64)

    if strategy == "val-f1":
        scores = 2.0 * tp / np.maximum(1.0, 2.0 * tp + fp + fn)
    elif strategy == "val-balanced-accuracy":
        sensitivity = tp / np.maximum(1.0, tp + fn)
        specificity = tn / np.maximum(1.0, tn + fp)
        scores = 0.5 * (sensitivity + specificity)
    elif strategy == "val-mcc":
        numerator = tp * tn - fp * fn
        denominator = np.sqrt(
            np.maximum(1.0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
        scores = numerator / denominator
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    best_score = float(np.nanmax(scores))
    best_indices = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-12, atol=1e-12))
    # Prefer the equally good threshold closest to 0.5 for stable behavior.
    best_index = int(best_indices[np.argmin(np.abs(thresholds[best_indices] - 0.5))])
    return float(thresholds[best_index]), best_score


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    difference = predictions - targets
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "pearson": safe_correlation(pearsonr, targets, predictions),
        "spearman": safe_correlation(spearmanr, targets, predictions),
    }


def regression_metrics_or_nan(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Regression metrics that remain well-defined for an empty subgroup."""
    if targets.size == 0:
        return {
            "n_samples": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    metrics = regression_metrics(targets, predictions)
    metrics["n_samples"] = float(targets.size)
    return metrics


def new_profile_accumulator() -> dict[str, float]:
    return {
        "n_values": 0.0,
        "sum_target": 0.0,
        "sum_prediction": 0.0,
        "sum_target_sq": 0.0,
        "sum_prediction_sq": 0.0,
        "sum_product": 0.0,
        "sum_abs_error": 0.0,
        "sum_sq_error": 0.0,
    }


def update_profile_accumulator(
    accumulator: dict[str, float],
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Accumulate profile metrics without storing every 1024-position profile."""
    valid = mask > 0
    if not bool(valid.any()):
        return
    target_values = target[valid].detach().to(torch.float64)
    prediction_values = prediction[valid].detach().to(torch.float64)
    difference = prediction_values - target_values
    accumulator["n_values"] += float(target_values.numel())
    accumulator["sum_target"] += float(target_values.sum().cpu())
    accumulator["sum_prediction"] += float(prediction_values.sum().cpu())
    accumulator["sum_target_sq"] += float((target_values * target_values).sum().cpu())
    accumulator["sum_prediction_sq"] += float(
        (prediction_values * prediction_values).sum().cpu()
    )
    accumulator["sum_product"] += float((target_values * prediction_values).sum().cpu())
    accumulator["sum_abs_error"] += float(difference.abs().sum().cpu())
    accumulator["sum_sq_error"] += float((difference * difference).sum().cpu())


def finalize_profile_accumulator(accumulator: dict[str, float]) -> dict[str, float]:
    n = accumulator["n_values"]
    if n <= 0:
        return {
            "n_values": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
        }
    mean_target = accumulator["sum_target"] / n
    mean_prediction = accumulator["sum_prediction"] / n
    covariance = accumulator["sum_product"] / n - mean_target * mean_prediction
    variance_target = accumulator["sum_target_sq"] / n - mean_target * mean_target
    variance_prediction = (
        accumulator["sum_prediction_sq"] / n - mean_prediction * mean_prediction
    )
    denominator = math.sqrt(max(0.0, variance_target) * max(0.0, variance_prediction))
    pearson = covariance / denominator if denominator > 0 else float("nan")
    return {
        "n_values": float(n),
        "mae": accumulator["sum_abs_error"] / n,
        "rmse": math.sqrt(accumulator["sum_sq_error"] / n),
        "pearson": float(pearson),
    }


def validation_selection_score(
    result: EpochResult,
    task_mode: TaskMode,
    objective: CheckpointObjective,
) -> tuple[float, str]:
    """Return a score where larger is always better for checkpoint selection."""
    resolved = objective
    if resolved == "auto":
        resolved = (
            "regression-profile-pearson"
            if task_mode == "regression"
            else "classification-auprc"
        )

    if resolved == "classification-auprc":
        score = result.classification["auprc"]
    elif resolved == "regression-loss":
        score = -result.losses["regression"]
    elif resolved == "regression-profile-mae":
        score = -result.profile_regression["mae"]
    elif resolved == "regression-profile-pearson":
        score = result.profile_regression["pearson"]
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"Unknown checkpoint objective: {resolved}")

    if not math.isfinite(score):
        # The regression loss is normally finite from the first epoch and is a
        # robust fallback when a correlation is initially undefined.
        fallback = result.losses["regression"]
        if math.isfinite(fallback):
            return -fallback, f"{resolved} (fallback: regression-loss)"
        return -result.losses["total"], f"{resolved} (fallback: total-loss)"
    return float(score), resolved


# Training and evaluation steps
# -----------------------------

def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


@dataclass
class EpochResult:
    losses: dict[str, float]
    classification: dict[str, float]
    regression: dict[str, float]
    regression_positive: dict[str, float]
    regression_negative: dict[str, float]
    profile_regression: dict[str, float]
    profile_regression_positive: dict[str, float]
    profile_regression_negative: dict[str, float]
    profile_mae: float
    labels: np.ndarray
    probabilities: np.ndarray
    regression_targets: np.ndarray
    regression_predictions: np.ndarray
    sample_ids: list[str]
    chromosomes: list[str]
    window_starts: np.ndarray
    center_indices: np.ndarray
    full_regression_targets: np.ndarray
    full_regression_predictions: np.ndarray


def classification_forward(
    model: HumanTFBindingModel,
    sequence: torch.Tensor,
    center_indices: torch.Tensor,
    eval_crops: str = "multi",
    eval_rc: bool = True,
) -> dict[str, torch.Tensor]:
    if sequence.shape[-1] != STORED_LENGTH:
        return model(sequence, center_indices)
    length = model.config.sequence_length
    starts = evaluation_crop_starts(STORED_LENGTH, length, eval_crops)
    centers = torch.full((sequence.shape[0],), length // 2, dtype=torch.long, device=sequence.device)
    scores = []
    for start in starts:
        crop = sequence[:, :, start:start + length]
        scores.append(torch.sigmoid(model(crop, centers)["classification_logit"].float()))
        if eval_rc:
            reverse = crop[:, [3, 2, 1, 0], :].flip(-1)
            reverse_centers = length - 1 - centers
            scores.append(torch.sigmoid(model(reverse, reverse_centers)["classification_logit"].float()))
    probabilities = torch.stack(scores).mean(dim=0)
    return {
        "classification_logit": torch.logit(probabilities, eps=torch.finfo(torch.float32).eps),
        "classification_probability": probabilities,
    }


@torch.inference_mode()
def score_training_negatives(model, loader, device, eval_crops, eval_rc, use_amp=False):
    """Return scores in the training-negative pool order; no validation/test access."""
    model.eval()
    scores = []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = classification_forward(model, batch["sequence"], batch["center_index"], eval_crops, eval_rc)
        scores.append(outputs["classification_probability"].cpu().numpy())
    return np.concatenate(scores)


def run_epoch(
    model: HumanTFBindingModel,
    loader: DataLoader,
    device: torch.device,
    classification_criterion: nn.Module,
    training_config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    gradient_clip: float,
    classification_threshold: float = 0.5,
    collect_full_profiles: bool = False,
    eval_crops: str = "multi",
    eval_rc: bool = True,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    loss_sums = {"classification": 0.0, "regression": 0.0, "total": 0.0}
    sample_count = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    chromosomes_all: list[str] = []
    window_starts_all: list[np.ndarray] = []
    center_indices_all: list[np.ndarray] = []
    full_targets_all: list[np.ndarray] = []
    full_predictions_all: list[np.ndarray] = []
    profile_stats_all = new_profile_accumulator()
    profile_stats_positive = new_profile_accumulator()
    profile_stats_negative = new_profile_accumulator()

    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            batch_size = int(batch["label"].shape[0])
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=amp_device_type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = classification_forward(
                    model, batch["sequence"], batch["center_index"], eval_crops, eval_rc
                )
                loss, components = compute_multitask_loss(
                    outputs,
                    batch,
                    classification_criterion,
                    training_config,
                )

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

            sample_count += batch_size
            for key in loss_sums:
                loss_sums[key] += float(components[key]) * batch_size

            probabilities = outputs.get("classification_probability")
            if probabilities is None:
                probabilities = torch.sigmoid(outputs["classification_logit"].float())
            labels_all.append(batch["label"].detach().cpu().numpy())
            probabilities_all.append(probabilities.detach().cpu().numpy())

            if "fe_profile" in batch:
                # Scalar regression metrics are reported at the center for profile mode;
                # otherwise they follow the configured scalar target.
                scalar_mode: Literal["center", "max", "mean"] = (
                    "center"
                    if training_config.regression_mode == "profile"
                    else training_config.regression_mode
                )
                scalar_target = scalar_from_track(
                    batch["fe_profile"], batch["center_index"], scalar_mode
                )
                scalar_prediction = scalar_from_track(
                    outputs["regression_track"], batch["center_index"], scalar_mode
                )
                targets_all.append(scalar_target.detach().cpu().numpy())
                predictions_all.append(scalar_prediction.detach().cpu().numpy())
            sample_ids.extend(list(raw_batch["sample_id"]))
            chromosomes_all.extend(list(raw_batch["chromosome"]))
            window_starts_all.append(batch["window_start"].detach().cpu().numpy())
            center_indices_all.append(batch["center_index"].detach().cpu().numpy())
            if "fe_profile" in batch:
                if collect_full_profiles:
                    full_targets_all.append(
                        batch["fe_profile"].detach().to(torch.float32).cpu().numpy()
                    )
                    full_predictions_all.append(
                        outputs["regression_track"].detach().to(torch.float32).cpu().numpy()
                    )

                update_profile_accumulator(
                    profile_stats_all,
                    batch["fe_profile"],
                    outputs["regression_track"],
                    batch["valid_mask"],
                )
                update_profile_accumulator(
                    profile_stats_positive,
                    batch["fe_profile"],
                    outputs["regression_track"],
                    batch["valid_mask"] * batch["label"][:, None],
                )
                update_profile_accumulator(
                    profile_stats_negative,
                    batch["fe_profile"],
                    outputs["regression_track"],
                    batch["valid_mask"] * (1.0 - batch["label"][:, None]),
                )

    labels_np = np.concatenate(labels_all).astype(np.int64)
    probabilities_np = np.concatenate(probabilities_all)
    targets_np = np.concatenate(targets_all) if targets_all else np.empty(0, dtype=np.float32)
    predictions_np = np.concatenate(predictions_all) if predictions_all else np.empty(0, dtype=np.float32)
    window_starts_np = np.concatenate(window_starts_all).astype(np.int64, copy=False)
    center_indices_np = np.concatenate(center_indices_all).astype(np.int64, copy=False)
    if collect_full_profiles and full_targets_all:
        full_targets_np = np.concatenate(full_targets_all, axis=0).astype(
            np.float32, copy=False
        )
        full_predictions_np = np.concatenate(full_predictions_all, axis=0).astype(
            np.float32, copy=False
        )
    else:
        full_targets_np = np.empty((0, 0), dtype=np.float32)
        full_predictions_np = np.empty((0, 0), dtype=np.float32)
    positive_mask = labels_np == 1
    negative_mask = labels_np == 0
    profile_all = finalize_profile_accumulator(profile_stats_all)
    profile_positive = finalize_profile_accumulator(profile_stats_positive)
    profile_negative = finalize_profile_accumulator(profile_stats_negative)
    return EpochResult(
        losses={key: value / max(1, sample_count) for key, value in loss_sums.items()},
        classification=classification_metrics(
            labels_np, probabilities_np, threshold=classification_threshold
        ),
        regression=regression_metrics_or_nan(targets_np, predictions_np),
        regression_positive=regression_metrics_or_nan(
            targets_np[positive_mask] if targets_np.size else targets_np,
            predictions_np[positive_mask] if predictions_np.size else predictions_np
        ),
        regression_negative=regression_metrics_or_nan(
            targets_np[negative_mask] if targets_np.size else targets_np,
            predictions_np[negative_mask] if predictions_np.size else predictions_np
        ),
        profile_regression=profile_all,
        profile_regression_positive=profile_positive,
        profile_regression_negative=profile_negative,
        profile_mae=profile_all["mae"],
        labels=labels_np,
        probabilities=probabilities_np,
        regression_targets=targets_np,
        regression_predictions=predictions_np,
        sample_ids=sample_ids,
        chromosomes=chromosomes_all,
        window_starts=window_starts_np,
        center_indices=center_indices_np,
        full_regression_targets=full_targets_np,
        full_regression_predictions=full_predictions_np,
    )


# Prediction files and figures
# ----------------------------

def result_summary(result: EpochResult) -> dict[str, Any]:
    if result.regression_targets.size == 0:
        return {"loss": result.losses, "classification": result.classification}
    return {
        "loss": result.losses,
        "classification": result.classification,
        "regression_scalar_all": result.regression,
        "regression_scalar_positive": result.regression_positive,
        "regression_scalar_negative": result.regression_negative,
        "regression_profile_all": result.profile_regression,
        "regression_profile_positive": result.profile_regression_positive,
        "regression_profile_negative": result.profile_regression_negative,
    }


def print_result(
    prefix: str, result: EpochResult, task_mode: TaskMode = "multitask"
) -> None:
    c = result.classification
    r = result.regression
    profile = result.profile_regression

    if task_mode == "regression":
        message = (
            f"{prefix}: loss={result.losses['total']:.4f} "
            f"FE-loss={result.losses['regression']:.4f} "
            f"FE-scalar-MAE={r['mae']:.4f} FE-scalar-Pearson={r['pearson']:.4f} "
            f"FE-profile-MAE={profile['mae']:.4f} "
            f"FE-profile-Pearson={profile['pearson']:.4f}"
        )
    else:
        message = (
            f"{prefix}: loss={result.losses['total']:.4f} "
            f"AUROC={c['auroc']:.4f} AUPRC={c['auprc']:.4f} "
            f"BalAcc={c['balanced_accuracy']:.4f} F1={c['f1']:.4f} "
            f"MCC={c['mcc']:.4f}"
        )
        if task_mode == "multitask":
            message += (
                f" FE-scalar-MAE={r['mae']:.4f} FE-scalar-Pearson={r['pearson']:.4f} "
                f"FE-profile-MAE={profile['mae']:.4f} "
                f"FE-profile-Pearson={profile['pearson']:.4f}"
            )
    print(message)


def write_predictions(
    path: Path, result: EpochResult, classification_threshold: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predicted_labels = (result.probabilities >= classification_threshold).astype(np.int64)
    has_fe = result.regression_targets.size == result.labels.size
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        columns = ["sample_id", "true_label", "binding_probability", "predicted_binary_label",
                   "classification_threshold", "correct"]
        if has_fe:
            columns += ["fe_target_scalar", "fe_prediction_scalar"]
        writer.writerow(columns)
        for i, (sample_id, label, probability, predicted_label) in enumerate(zip(
            result.sample_ids, result.labels, result.probabilities, predicted_labels, strict=True
        )):
            row = [sample_id, int(label), f"{float(probability):.8g}", int(predicted_label),
                   f"{float(classification_threshold):.8g}", int(int(label) == int(predicted_label))]
            if has_fe:
                row += [f"{float(result.regression_targets[i]):.8g}",
                        f"{float(result.regression_predictions[i]):.8g}"]
            writer.writerow(row)


def write_regression_predictions(path: Path, result: EpochResult) -> None:
    """Write one scalar FE target/prediction per test window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "true_label",
                "fe_target_scalar",
                "fe_prediction_scalar",
                "absolute_error",
                "squared_error",
            ]
        )
        for sample_id, label, target, prediction in zip(
            result.sample_ids,
            result.labels,
            result.regression_targets,
            result.regression_predictions,
            strict=True,
        ):
            error = float(prediction) - float(target)
            writer.writerow(
                [
                    sample_id,
                    int(label),
                    f"{float(target):.8g}",
                    f"{float(prediction):.8g}",
                    f"{abs(error):.8g}",
                    f"{error * error:.8g}",
                ]
            )


def write_full_regression_profiles(
    npz_path: Path,
    index_path: Path,
    result: EpochResult,
) -> None:
    """Write complete 1024-position FE target/prediction profiles for each window."""
    if result.full_regression_predictions.size == 0:
        raise ValueError(
            "No full profiles were collected. Run the final test epoch with "
            "collect_full_profiles=True."
        )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        sample_ids=np.asarray(result.sample_ids),
        chromosomes=np.asarray(result.chromosomes),
        window_starts=result.window_starts,
        center_indices=result.center_indices,
        genomic_centers=result.window_starts + result.center_indices,
        labels=result.labels,
        binding_probabilities=result.probabilities,
        fe_targets=result.full_regression_targets,
        fe_predictions=result.full_regression_predictions,
    )

    with index_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "profile_index",
                "sample_id",
                "chromosome",
                "window_start_0based",
                "center_index_0based",
                "genomic_center_0based",
                "true_label",
                "binding_probability",
            ]
        )
        for index, values in enumerate(
            zip(
                result.sample_ids,
                result.chromosomes,
                result.window_starts,
                result.center_indices,
                result.labels,
                result.probabilities,
                strict=True,
            )
        ):
            sample_id, chromosome, start, center, label, probability = values
            writer.writerow(
                [
                    index,
                    sample_id,
                    chromosome,
                    int(start),
                    int(center),
                    int(start) + int(center),
                    int(label),
                    f"{float(probability):.8g}",
                ]
            )


def _read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(
                    f"Invalid chromosome-size line {line_number} in {path}: {raw_line.rstrip()}"
                )
            try:
                size = int(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid chromosome size at {path}:{line_number}: {fields[1]}"
                ) from exc
            if size <= 0:
                raise ValueError(
                    f"Chromosome size must be positive at {path}:{line_number}"
                )
            sizes[fields[0]] = size
    if not sizes:
        raise ValueError(f"No chromosome sizes found in {path}")
    return sizes


def _natural_chrom_key(name: str) -> tuple[int, int | str]:
    value = canonical_chrom(name)
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _output_chromosome_name(chromosome: str, prefix: str) -> str:
    canonical = canonical_chrom(chromosome)
    return f"{prefix}{canonical}" if prefix else canonical


def _lookup_chromosome_size(
    sizes: dict[str, int],
    canonical_name: str,
    output_name: str,
) -> int | None:
    candidates = [
        output_name,
        canonical_name,
        f"Chr{canonical_name}",
        f"chr{canonical_name}",
        f"chromosome{canonical_name}",
    ]
    for candidate in candidates:
        if candidate in sizes:
            return sizes[candidate]
    lower_sizes = {name.lower(): size for name, size in sizes.items()}
    for candidate in candidates:
        if candidate.lower() in lower_sizes:
            return lower_sizes[candidate.lower()]
    return None


def _aggregate_profiles_for_chromosome(
    window_starts: np.ndarray,
    profiles: np.ndarray,
    aggregation: BigWigAggregation,
) -> tuple[np.ndarray, np.ndarray]:
    """Map overlapping window profiles to one value per covered genomic base."""
    if profiles.ndim != 2 or profiles.shape[0] != window_starts.size:
        raise ValueError("Profile/window-start dimensions do not match")
    if profiles.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    profile_length = int(profiles.shape[1])
    offsets = np.arange(profile_length, dtype=np.int64)
    positions = (window_starts[:, None] + offsets[None, :]).reshape(-1)
    values = profiles.astype(np.float64, copy=False).reshape(-1)
    valid = np.isfinite(values) & (positions >= 0)
    positions = positions[valid]
    values = values[valid]
    if positions.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    order = np.argsort(positions, kind="mergesort")
    positions = positions[order]
    values = values[order]
    group_starts = np.r_[0, np.flatnonzero(positions[1:] != positions[:-1]) + 1]
    unique_positions = positions[group_starts]

    if aggregation == "max":
        aggregated = np.maximum.reduceat(values, group_starts)
    else:
        if aggregation == "mean":
            weights = np.ones(profile_length, dtype=np.float64)
        elif aggregation == "center-weighted":
            center = (profile_length - 1) / 2.0
            distance = np.abs(np.arange(profile_length, dtype=np.float64) - center)
            weights = 1.0 - distance / (center + 1.0)
            # Keep a small non-zero edge weight so singly covered edge bases remain visible.
            weights = np.maximum(weights, 1.0 / profile_length)
        else:  # pragma: no cover - argparse restricts choices
            raise ValueError(f"Unknown BigWig aggregation: {aggregation}")

        repeated_weights = np.broadcast_to(weights, profiles.shape).reshape(-1)[valid]
        repeated_weights = repeated_weights[order]
        weighted_sum = np.add.reduceat(values * repeated_weights, group_starts)
        weight_sum = np.add.reduceat(repeated_weights, group_starts)
        aggregated = weighted_sum / np.maximum(weight_sum, np.finfo(np.float64).eps)

    return unique_positions.astype(np.int64, copy=False), aggregated.astype(np.float32)


def write_regression_bigwig(
    path: Path,
    result: EpochResult,
    profiles: np.ndarray,
    aggregation: BigWigAggregation,
    chromosome_prefix: str,
    chrom_sizes_path: Path | None,
) -> dict[str, int]:
    """Write overlapping per-window profiles as an IGV-compatible BigWig track."""
    try:
        import pyBigWig  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "BigWig export requires pyBigWig. Install it in the active environment with: "
            "python -m pip install pyBigWig"
        ) from exc

    if profiles.size == 0:
        raise ValueError(
            "No full profiles were collected. Enable --export-bigwig or "
            "--export-full-regression-profiles for the final test pass."
        )
    if profiles.ndim != 2 or profiles.shape[0] != len(result.chromosomes):
        raise ValueError("BigWig profiles do not match the number of test windows")

    supplied_sizes = (
        _read_chrom_sizes(chrom_sizes_path.resolve())
        if chrom_sizes_path is not None
        else None
    )
    chromosome_array = np.asarray(result.chromosomes, dtype=str)
    output_data: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    for canonical_name in sorted(set(chromosome_array), key=_natural_chrom_key):
        mask = chromosome_array == canonical_name
        positions, values = _aggregate_profiles_for_chromosome(
            result.window_starts[mask], profiles[mask], aggregation
        )
        if positions.size == 0:
            continue

        output_name = _output_chromosome_name(canonical_name, chromosome_prefix)
        inferred_size = int(positions[-1]) + 1
        if supplied_sizes is None:
            chromosome_size = inferred_size
        else:
            found_size = _lookup_chromosome_size(
                supplied_sizes, canonical_name, output_name
            )
            if found_size is None:
                raise ValueError(
                    f"No chromosome size found for {output_name!r} in {chrom_sizes_path}"
                )
            chromosome_size = int(found_size)
            if inferred_size > chromosome_size:
                raise ValueError(
                    f"Predictions reach coordinate {inferred_size - 1} on {output_name}, "
                    f"but the supplied chromosome size is only {chromosome_size}."
                )

        keep = positions < chromosome_size
        output_data[output_name] = (positions[keep], values[keep], chromosome_size)

    if not output_data:
        raise ValueError("No covered genomic positions were available for BigWig export")

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_names = sorted(output_data, key=_natural_chrom_key)
    header = [(name, output_data[name][2]) for name in ordered_names]
    bigwig = pyBigWig.open(str(path), "w")
    try:
        bigwig.addHeader(header)
        for chromosome in ordered_names:
            positions, values, _ = output_data[chromosome]
            if positions.size == 0:
                continue
            # Fixed-step blocks are efficient, but a new block is required at every gap.
            boundaries = np.r_[
                0,
                np.flatnonzero(np.diff(positions) != 1) + 1,
                positions.size,
            ]
            for block_start, block_end in zip(boundaries[:-1], boundaries[1:], strict=True):
                # Chunk very long contiguous runs to avoid constructing huge Python lists.
                for chunk_start in range(int(block_start), int(block_end), 1_000_000):
                    chunk_end = min(chunk_start + 1_000_000, int(block_end))
                    bigwig.addEntries(
                        chromosome,
                        int(positions[chunk_start]),
                        values=values[chunk_start:chunk_end].astype(float).tolist(),
                        span=1,
                        step=1,
                    )
    finally:
        bigwig.close()

    return {
        chromosome: int(output_data[chromosome][0].size)
        for chromosome in ordered_names
    }


def write_regression_summary(path: Path, result: EpochResult, regression_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "MACS3 FE regression result",
        "==========================",
        f"Regression mode used for training: {regression_mode}",
        "For profile mode, scalar metrics below refer to the center position.",
        "",
    ]
    groups = [
        ("Scalar metrics, all windows", result.regression),
        ("Scalar metrics, positive windows", result.regression_positive),
        ("Scalar metrics, negative windows", result.regression_negative),
        ("Profile metrics, all valid bases", result.profile_regression),
        ("Profile metrics, positive valid bases", result.profile_regression_positive),
        ("Profile metrics, negative valid bases", result.profile_regression_negative),
    ]
    for title, metrics in groups:
        lines.append(title)
        lines.append("-" * len(title))
        for key, value in metrics.items():
            if isinstance(value, float) and math.isfinite(value):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_multitask_summary(
    path: Path,
    result: EpochResult,
    threshold_strategy: str,
    regression_mode: str,
    regression_weight: float,
) -> None:
    c = result.classification
    r = result.regression
    p = result.profile_regression
    lines = [
        "Joint TF-binding classification and MACS3 FE regression",
        "========================================================",
        f"Regression loss weight: {regression_weight}",
        f"Regression mode: {regression_mode}",
        f"Threshold strategy: {threshold_strategy}",
        "",
        "Classification",
        f"AUROC: {c['auroc']:.6f}",
        f"AUPRC: {c['auprc']:.6f}",
        f"Balanced accuracy: {c['balanced_accuracy']:.6f}",
        f"F1: {c['f1']:.6f}",
        f"MCC: {c['mcc']:.6f}",
        "",
        "FE regression (scalar readout)",
        f"MAE: {r['mae']:.6f}",
        f"RMSE: {r['rmse']:.6f}",
        f"Pearson: {r['pearson']:.6f}",
        f"Spearman: {r['spearman']:.6f}",
        "",
        "FE regression (full profile)",
        f"MAE: {p['mae']:.6f}",
        f"RMSE: {p['rmse']:.6f}",
        f"Pearson: {p['pearson']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_matrix(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["", "predicted_0", "predicted_1"])
        writer.writerow(
            [
                "true_0",
                int(metrics["true_negative"]),
                int(metrics["false_positive"]),
            ]
        )
        writer.writerow(
            [
                "true_1",
                int(metrics["false_negative"]),
                int(metrics["true_positive"]),
            ]
        )


def write_binary_summary(path: Path, metrics: dict[str, float], threshold_strategy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Binary TF-binding classification result",
        "=======================================",
        f"Threshold strategy: {threshold_strategy}",
        f"Decision threshold: {metrics['threshold']:.8f}",
        f"AUROC: {metrics['auroc']:.6f}",
        f"AUPRC: {metrics['auprc']:.6f}",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall / sensitivity: {metrics['recall']:.6f}",
        f"Specificity: {metrics['specificity']:.6f}",
        f"F1: {metrics['f1']:.6f}",
        f"MCC: {metrics['mcc']:.6f}",
        "",
        "Confusion matrix counts",
        f"TN: {int(metrics['true_negative'])}",
        f"FP: {int(metrics['false_positive'])}",
        f"FN: {int(metrics['false_negative'])}",
        f"TP: {int(metrics['true_positive'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_classification_curves(
    output_dir: Path,
    result: EpochResult,
    task_mode: TaskMode,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Write ROC and precision-recall curves for the final test predictions."""
    if np.unique(result.labels).size != 2:
        raise ValueError("ROC and precision-recall curves require both classes")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Automatic ROC/PR plots require matplotlib. Install it with: "
            "python -m pip install matplotlib"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    chromosomes = ", ".join(sorted(set(result.chromosomes))) or "test split"
    diagnostic = task_mode == "regression"
    title_suffix = " (diagnostic only: untrained classification head)" if diagnostic else ""

    fpr, tpr, _ = roc_curve(result.labels, result.probabilities)
    auroc = roc_auc_score(result.labels, result.probabilities)
    roc_path = output_dir / "roc_curve.png"
    plt.figure(figsize=(6.4, 5.4))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve - test {chromosomes}{title_suffix}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(result.labels, result.probabilities)
    auprc = average_precision_score(result.labels, result.probabilities)
    prevalence = float(np.mean(result.labels))
    pr_path = output_dir / "precision_recall_curve.png"
    plt.figure(figsize=(6.4, 5.4))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc:.4f}")
    plt.plot(
        [0.0, 1.0],
        [prevalence, prevalence],
        linestyle="--",
        label=f"Positive prevalence = {prevalence:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall curve - test {chromosomes}{title_suffix}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    if diagnostic:
        print(
            "WARNING: ROC/PR plots were written for a regression-only checkpoint. "
            "Its classification head was not trained, so these curves are diagnostic only."
        )
    return roc_path, pr_path


# Training loop
# -------------

def train_command(args: argparse.Namespace, *, recorder=None, model_factory=None) -> int:
    function_started = time.perf_counter()
    if not 0.0 <= args.reverse_complement_probability <= 1.0:
        raise ValueError("reverse-complement-probability must be in [0, 1]")
    if not 0.0 <= args.classification_threshold <= 1.0:
        raise ValueError("classification-threshold must be in [0, 1]")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative and initial LR > 0")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate cannot exceed learning-rate")
    if not 0.0 < args.adam_beta1 < 1.0 or not 0.0 < args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must lie strictly between 0 and 1")
    if args.regression_weight < 0:
        raise ValueError("regression-weight must be non-negative")
    if args.min_checkpoint_epoch < 1 or args.min_checkpoint_epoch > args.epochs:
        raise ValueError("min-checkpoint-epoch must be between 1 and epochs")
    if args.export_bigwig and args.regression_mode != "profile":
        raise ValueError("--export-bigwig requires --regression-mode profile")
    if args.export_target_bigwig and args.regression_mode != "profile":
        raise ValueError("--export-target-bigwig requires --regression-mode profile")
    if args.val_fraction <= 0:
        raise ValueError("Training requires --val-fraction > 0 for validation and threshold selection")
    if args.negatives_per_positive < 1 or args.eval_batch_size < 1:
        raise ValueError("Sampling ratio and evaluation batch size must be positive")
    if args.hard_negative_start_epoch < 2 or args.hard_negative_refresh < 1:
        raise ValueError("Hard-negative mining needs a warmup epoch and a positive refresh interval")
    seed_everything(args.seed)
    dataset_root = args.dataset_root.resolve() if args.dataset_root is not None else args.cache_path.resolve().parent
    cache_path = (
        args.cache_path.resolve()
        if args.cache_path is not None
        else dataset_root / "tf_binding_cache.npz"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"Existing dataset cache not found: {cache_path}")
    cache = build_or_load_cache(
        dataset_root,
        cache_path,
        fe_transform=args.fe_transform,
        rebuild=args.rebuild_cache,
    )
    sequence_length, context_per_side = cache_geometry(cache)
    if sequence_length != 256:
        raise ValueError(
            f"This requested architecture expects 256 bp, but the dataset contains {sequence_length} bp"
        )
    human301 = "split" in cache
    if human301:
        if args.task_mode != "classification":
            raise ValueError("The Human301 BED cache has binary labels and no FE targets; use --task-mode classification")
        if args.export_full_regression_profiles or args.export_bigwig or args.export_target_bigwig:
            raise ValueError("The BED cache has no FE targets or trained FE head; regression exports are unavailable")
        audit = cache["summary"].get("sequence_audit", {})
        conflicts = [key for key, value in audit.items() if isinstance(value, dict)
                     and value.get("identical_or_RC_identical_sequences_with_both_labels", 0) > 0]
        if conflicts:
            raise ValueError("Conflicting sequence labels in the existing cache: " + ", ".join(conflicts))
        overlaps = [f"{key}={value}" for key, value in audit.items() if isinstance(value, int) and value > 0]
        if overlaps:
            print("WARNING: Shared sequences or reverse complements between cached splits: "
                  + ", ".join(overlaps) + ". The datasets remain unchanged.", file=sys.stderr)
        print("Window geometry: 256 bp from 301 bp; crop starts 0..45; shifts -22..+23; fresh uniform draw per access")
        print(f"Evaluation: crops {evaluation_crop_starts(301, 256, args.eval_crops)}, reverse complement={args.eval_rc}; all samples")
    else:
        if args.train_shift_max < 0:
            raise ValueError("--train-shift-max must not be negative")
        if args.eval_shift_max is None:
            args.eval_shift_max = args.train_shift_max
        if args.eval_shift_max < 0:
            raise ValueError("--eval-shift-max must not be negative")
        if args.train_shift_max > context_per_side:
            raise ValueError(
                f"--train-shift-max {args.train_shift_max} exceeds the context of this dataset "
                f"({context_per_side} bases per side); regenerate the dataset with a larger "
                "--context-per-side or lower the shift"
            )
        if args.eval_shift == "fixed" and args.eval_shift_max > context_per_side:
            raise ValueError(
                f"--eval-shift-max {args.eval_shift_max} exceeds the context of this dataset "
                f"({context_per_side} bases per side)"
            )
        print(
            f"Window geometry: crop {sequence_length} bp from {sequence_length + 2 * context_per_side} bp "
            f"context; train shift +/-{args.train_shift_max} ({args.train_shift_distribution}); "
            f"eval shift {args.eval_shift}"
            + (f" +/-{args.eval_shift_max}" if args.eval_shift == "fixed" else "")
        )

    splits = make_splits(
        cache,
        train_chromosomes=args.train_chromosomes,
        test_chromosomes=args.test_chromosomes,
        val_fraction=args.val_fraction,
        val_block_size=args.val_block_size,
        min_split_samples_per_class=args.min_split_samples_per_class,
        balanced_evaluation=args.balanced_evaluation,
        max_eval_positive_reduction=args.max_eval_positive_reduction,
        val_milp_time_limit=args.val_milp_time_limit,
    )
    if human301:
        args.train_chromosomes = tuple(sorted(set(cache["chromosomes"][splits["train"]].astype(str))))
        args.test_chromosomes = tuple(sorted(set(cache["chromosomes"][splits["test"]].astype(str))))
    evaluation_balance: dict[str, Any] = {"scope": "all_samples", "undersampled": False}
    if args.balanced_evaluation:
        # These selections are made exactly once. They remain unchanged across
        # all epochs so validation scores and the final test are comparable.
        splits["val"], evaluation_balance["validation"] = make_fixed_balanced_evaluation_split(
            cache,
            splits["val"],
            seed=args.seed + 10_001,
            split_name="validation",
            max_positive_reduction=args.max_eval_positive_reduction,
        )
        splits["test"], evaluation_balance["test"] = make_fixed_balanced_evaluation_split(
            cache,
            splits["test"],
            seed=args.seed + 20_003,
            split_name="test",
            max_positive_reduction=args.max_eval_positive_reduction,
        )
    labels = cache["labels"].astype(np.int64)
    for split_name, indices in splits.items():
        split_labels = labels[indices]
        positives = int(split_labels.sum())
        negatives = int(indices.size - positives)
        print(
            f"{split_name:>5}: n={indices.size:,}, positive={positives:,}, "
            f"negative={negatives:,}"
        )

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if recorder is not None:
        split_hash = hashlib.sha256()
        for name, indices in splits.items():
            split_hash.update(name.encode())
            split_hash.update(indices.tobytes())
        recorder.update(device=str(device), split_hash=split_hash.hexdigest(),
            split_counts={name:{"positive":int(labels[indices].sum()),
                "negative":int(len(indices)-labels[indices].sum())} for name,indices in splits.items()})

    model_config = ModelConfig(
        sequence_length=sequence_length,
        bottleneck_length=sequence_length // 4,
        motif_kernels=tuple(args.motif_kernels),
        embed_channels=args.embed_channels,
        encoder_channels=tuple(args.encoder_channels),
        transformer_heads=args.heads,
        transformer_ffn=args.ffn_channels,
        dropout=args.dropout,
        channel_dropout=getattr(args, "channel_dropout", 0.0),
        activation=args.activation,
        classification_readout=args.classification_readout,
        regression_nonnegative=not args.allow_negative_regression_output,
    )
    model = (model_factory(model_config) if model_factory else HumanTFBindingModel(model_config)).to(device)
    print(f"Hidden activation: {model_config.activation.upper()}")
    total_parameters, trainable_parameters = count_parameters(model)
    if recorder is not None:
        recorder.update(trainable_parameter_count=trainable_parameters)
    print(
        f"Parameters: {total_parameters:,} total; {trainable_parameters:,} trainable"
    )

    print(f"Architecture: motif kernels {model_config.motif_kernels}; encoder 128/64/64; "
          "original five-block bottleneck at 64; decoder 64/128/256")

    train_dataset = TFBindingDataset(
        cache,
        splits["train"],
        reverse_complement_probability=args.reverse_complement_probability,
        shift_max=args.train_shift_max,
        shift_distribution=args.train_shift_distribution,
        seed=args.seed,
        training=True,
    )
    if human301:
        evaluation_shift = {
            "mode": "fixed_crop_average", "crop_starts": list(evaluation_crop_starts(301, 256, args.eval_crops)),
            "reverse_complement": args.eval_rc, "aggregation": "mean_probability",
        }
        val_dataset = TFBindingDataset(cache, splits["val"])
        test_dataset = TFBindingDataset(cache, splits["test"])
    else:
        # Val/Test bekommen je Fenster einen festen Shift (oder 0), damit die
        # Bewertung ueber Epochen und Laeufe hinweg dieselben Ausschnitte sieht.
        evaluation_shift: dict[str, Any] = {
            "mode": args.eval_shift,
            "shift_max": args.eval_shift_max if args.eval_shift == "fixed" else 0,
            "distribution": args.train_shift_distribution,
            "seed": args.seed + 30_007,
        }
        eval_rng = np.random.default_rng(evaluation_shift["seed"])
        fixed_val_shifts = draw_shifts(
            eval_rng, int(splits["val"].size), evaluation_shift["shift_max"], args.train_shift_distribution
        )
        fixed_test_shifts = draw_shifts(
            eval_rng, int(splits["test"].size), evaluation_shift["shift_max"], args.train_shift_distribution
        )
        evaluation_shift["validation_shifts"] = fixed_val_shifts.tolist()
        evaluation_shift["test_shifts"] = fixed_test_shifts.tolist()
        val_dataset = TFBindingDataset(cache, splits["val"], 0.0, fixed_shifts=fixed_val_shifts)
        test_dataset = TFBindingDataset(cache, splits["test"], 0.0, fixed_shifts=fixed_test_shifts)
    train_sampler = CyclicBalancedNegativeSampler(
        train_dataset, seed=args.seed, negatives_per_positive=args.negatives_per_positive,
        batch_size=args.batch_size, hard_negative_fraction=args.hard_negative_fraction,
    )
    required_epochs = train_sampler.min_epochs_for_positive_coverage
    if args.epochs < required_epochs:
        print(
            f"WARNING: {required_epochs} epochs are needed to expose every "
            f"training positive at least once; only {args.epochs} were requested.",
            file=sys.stderr,
        )
    if required_epochs > args.min_checkpoint_epoch:
        args.min_checkpoint_epoch = min(args.epochs, required_epochs)
        print(
            "Checkpoint selection and early stopping start at epoch "
            f"{args.min_checkpoint_epoch} so all training positives can be traversed."
        )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_dataset, args.eval_batch_size, False, args.num_workers, device
    )
    test_loader = make_loader(
        test_dataset, args.eval_batch_size, False, args.num_workers, device
    )

    mining_loader = None
    if train_sampler.hard_count:
        mining_indices = splits["train"][train_sampler.pool.positions]
        mining_dataset = TFBindingDataset(cache, mining_indices)
        mining_loader = make_loader(mining_dataset, args.eval_batch_size, False, args.num_workers, device)

    train_labels = labels[splits["train"]]
    positive_count = int(train_labels.sum())
    negative_count = int(train_labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("The training split must contain both classes")
    # Weight the presented 1:k distribution, rather than the full 1:100 pool.
    pos_weight = torch.tensor(float(args.negatives_per_positive), dtype=torch.float32, device=device)
    classification_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    training_config = TrainingConfig(
        task_mode=args.task_mode,
        regression_mode=args.regression_mode,
        regression_weight=args.regression_weight,
        regression_positive_only=args.regression_positive_only,
        smooth_l1_beta=args.smooth_l1_beta,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_learning_rate
    )
    amp_enabled = args.amp and device.type == "cuda"

    if not amp_enabled:
        scaler = None
    elif hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty; existing runs are not overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    epochs_without_improvement = 0

    config_payload = {
        "script_version": SCRIPT_VERSION,
        "model": asdict(model_config),
        "parameter_count": total_parameters,
        "cache_sha256": sha256_file(cache_path),
        "dataset": cache.get("summary"),
        "training": asdict(training_config),
        "arguments": {key: value for key, value in vars(args).items() if key != "function"},
        "splits": {key: value.tolist() for key, value in splits.items()},
        "class_counts": {
            "train_positive": positive_count,
            "train_negative": negative_count,
            "sampling_scope": "global_positive_negative",
            "samples_per_training_epoch": train_sampler.samples_per_epoch,
            "negatives_per_positive": args.negatives_per_positive,
            "pos_weight": float(pos_weight.item()),
        },
        "evaluation_balance": evaluation_balance,
        "window_geometry": {
            "crop_length": sequence_length,
            "context_per_side": context_per_side,
            "context_length": int(cache["sequences"].shape[1]),
            "context_left": context_per_side,
            "context_right": int(cache["sequences"].shape[1]) - sequence_length - context_per_side,
            "classification_readout_position": "crop_center",
        },
        "train_shift": {
            "shift_max": args.train_shift_max,
            "distribution": args.train_shift_distribution,
            "redrawn_every_epoch": True,
            "redrawn_per_access": human301,
            "crop_start_range_inclusive": [0, 45] if human301 else None,
            "shift_range_inclusive": [-22, 23] if human301 else None,
        },
        "evaluation_shift": evaluation_shift,
    }
    with (output_dir / "configuration.json").open("wt", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, default=json_default)
        handle.write("\n")

    if recorder is not None:
        recorder.update(setup_s=time.perf_counter()-function_started,
                        train_cache_sha256=config_payload["cache_sha256"])
    fit_loop_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        mining_started = time.perf_counter()
        if mining_loader is not None and epoch >= args.hard_negative_start_epoch and (epoch - args.hard_negative_start_epoch) % args.hard_negative_refresh == 0:
            mining_scores = score_training_negatives(model, mining_loader, device, args.eval_crops, args.eval_rc, amp_enabled)
            train_sampler.update_hard_negatives(mining_scores)
            print(f"Epoch {epoch:03d}: refreshed hard-negative candidates from training only")
        mining_seconds = time.perf_counter() - mining_started
        train_dataset.set_epoch(epoch)
        train_started = time.perf_counter()
        train_result = run_epoch(
            model,
            train_loader,
            device,
            classification_criterion,
            training_config,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=amp_enabled,
            gradient_clip=args.gradient_clip,
            eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        )
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        val_result = run_epoch(
            model,
            val_loader,
            device,
            classification_criterion,
            training_config,
            optimizer=None,
            scaler=None,
            use_amp=amp_enabled,
            gradient_clip=args.gradient_clip,
            eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        )
        validation_seconds = time.perf_counter() - val_started
        print(f"  negatives seen={train_sampler.last_summary['negative_pool_coverage']:.1%}; "
              f"hard negatives={train_sampler.last_summary['hard_negative']:,}")
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        learning_rate_next = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - start_time
        print_result(f"Epoch {epoch:03d} train", train_result, training_config.task_mode)
        print_result(f"Epoch {epoch:03d} val  ", val_result, training_config.task_mode)
        print(
            f"  learning_rate_used={learning_rate_used:.3e}; "
            f"learning_rate_next={learning_rate_next:.3e}; elapsed={elapsed:.1f}s"
        )

        row = {
            "epoch": epoch,
            "learning_rate_used": learning_rate_used,
            "learning_rate_next": learning_rate_next,
            "elapsed_seconds": elapsed,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "mining_seconds": mining_seconds,
            "train": result_summary(train_result),
            "val": result_summary(val_result),
            "sampling": train_sampler.last_summary,
            "train_shift": train_dataset.shift_summary(),
        }
        history.append(row)
        if recorder is not None:
            recorder.epoch(row)
        with (output_dir / "history.json").open("wt", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, default=json_default)
            handle.write("\n")

        score, objective_used = validation_selection_score(
            val_result,
            training_config.task_mode,
            args.checkpoint_objective,
        )
        if epoch < args.min_checkpoint_epoch:
            print(
                f"  checkpoint selection starts at epoch "
                f"{args.min_checkpoint_epoch}; current checkpoint skipped"
            )
            continue
        if score > best_score + args.min_delta:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "script_version": SCRIPT_VERSION,
                    "cache_sha256": config_payload["cache_sha256"],
                    "model_config": asdict(model_config),
                    "training_config": asdict(training_config),
                    "epoch": epoch,
                    "validation_score": score,
                    "checkpoint_objective": objective_used,
                },
                checkpoint_path,
            )
            print(
                f"  saved new best checkpoint: {checkpoint_path} "
                f"(objective={objective_used}, score={score:.6f})"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if recorder is not None:
        recorder.fit_loop_finished(time.perf_counter() - fit_loop_started, history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # Select the binary decision threshold exclusively on the held-out
    # validation split. The test chromosome remains untouched until final use.
    best_val_result = run_epoch(
        model,
        val_loader,
        device,
        classification_criterion,
        training_config,
        optimizer=None,
        scaler=None,
        use_amp=amp_enabled,
        gradient_clip=args.gradient_clip,
        eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        classification_threshold=0.5,
    )
    if training_config.task_mode == "regression":
        classification_threshold = 0.5
        threshold_score = float("nan")
        threshold_strategy_used = "not-selected-regression-only"
        print(
            "Regression-only mode: binary threshold optimization is skipped; "
            "classification outputs are diagnostic only."
        )
    else:
        classification_threshold, threshold_score = select_classification_threshold(
            best_val_result.labels,
            best_val_result.probabilities,
            strategy=args.threshold_strategy,
            fixed_threshold=args.classification_threshold,
        )
        threshold_strategy_used = args.threshold_strategy
        print(
            f"Binary threshold: {classification_threshold:.6f} "
            f"(strategy={args.threshold_strategy}, validation_score={threshold_score:.6f})"
        )

    best_val_result.classification = classification_metrics(
        best_val_result.labels, best_val_result.probabilities, threshold=classification_threshold
    )
    write_predictions(output_dir / "validation_predictions.tsv", best_val_result, classification_threshold)
    with (output_dir / "validation_metrics.json").open("wt", encoding="utf-8") as handle:
        json.dump(result_summary(best_val_result), handle, indent=2, default=json_default)
        handle.write("\n")
    checkpoint["classification_threshold"] = classification_threshold
    checkpoint["evaluation_crop_mode"] = args.eval_crops
    checkpoint["evaluation_crop_starts"] = list(evaluation_crop_starts(301, 256, args.eval_crops))
    checkpoint["evaluation_reverse_complement"] = args.eval_rc
    checkpoint["negatives_per_positive"] = args.negatives_per_positive
    checkpoint["amp_enabled"] = amp_enabled
    torch.save(checkpoint, output_dir / "model_for_prediction.pt")
    if args.skip_test:
        print(f"Validation-only run completed; frozen model: {output_dir / 'model_for_prediction.pt'}")
        return 0

    collect_full_profiles = (
        args.export_full_regression_profiles
        or args.export_bigwig
        or args.export_target_bigwig
    )
    test_result = run_epoch(
        model,
        test_loader,
        device,
        classification_criterion,
        training_config,
        optimizer=None,
        scaler=None,
        use_amp=amp_enabled,
        gradient_clip=args.gradient_clip,
        eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        classification_threshold=classification_threshold,
        collect_full_profiles=collect_full_profiles,
    )
    print_result("TEST " + ",".join(args.test_chromosomes), test_result, training_config.task_mode)
    test_summary = result_summary(test_result)
    test_summary["checkpoint_epoch"] = int(checkpoint["epoch"])
    test_summary["threshold_selection"] = {
        "strategy": threshold_strategy_used,
        "threshold": classification_threshold,
        "validation_objective_score": threshold_score,
    }
    with (output_dir / "test_metrics.json").open("wt", encoding="utf-8") as handle:
        json.dump(test_summary, handle, indent=2, default=json_default)
        handle.write("\n")
    write_predictions(
        output_dir / "binary_classification_results.tsv",
        test_result,
        classification_threshold,
    )
    # Keep the old generic filename as an identical convenience export.
    write_predictions(
        output_dir / "test_predictions.tsv",
        test_result,
        classification_threshold,
    )
    write_confusion_matrix(
        output_dir / "binary_confusion_matrix.tsv", test_result.classification
    )
    write_binary_summary(
        output_dir / "binary_classification_summary.txt",
        test_result.classification,
        threshold_strategy_used,
    )
    roc_plot_path, pr_plot_path = write_classification_curves(
        output_dir, test_result, training_config.task_mode
    )
    with (output_dir / "binary_classification_metrics.json").open(
        "wt", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "threshold_strategy": threshold_strategy_used,
                "threshold": classification_threshold,
                "validation_objective_score": threshold_score,
                "test_metrics": test_result.classification,
            },
            handle,
            indent=2,
            default=json_default,
        )
        handle.write("\n")
    if "fe_profiles" in cache:
        write_regression_predictions(
            output_dir / "regression_results.tsv", test_result
        )
        if args.export_full_regression_profiles:
            if training_config.task_mode == "classification":
                print(
                    "WARNING: Full FE profiles are being exported from a classification-only "
                    "checkpoint. The regression head received no training signal, so these FE "
                    "predictions are not biologically meaningful. Use a multitask checkpoint "
                    "for interpretable FE predictions."
                )
            write_full_regression_profiles(
                output_dir / "regression_profiles.npz",
                output_dir / "regression_profile_index.tsv",
                test_result,
            )

        if args.export_bigwig or args.export_target_bigwig:
            if training_config.task_mode == "classification":
                print(
                    "WARNING: BigWig export is using a classification-only checkpoint. "
                    "The regression track did not receive a regression training signal."
                )
            if args.export_bigwig:
                predicted_bigwig_path = (
                    args.bigwig_output.resolve()
                    if args.bigwig_output is not None
                    else output_dir / "predicted_fe_test.bw"
                )
                coverage = write_regression_bigwig(
                    predicted_bigwig_path,
                    test_result,
                    test_result.full_regression_predictions,
                    aggregation=args.bigwig_aggregation,
                    chromosome_prefix=args.bigwig_chrom_prefix,
                    chrom_sizes_path=args.chrom_sizes,
                )
                print(
                    f"Predicted FE BigWig: {predicted_bigwig_path} "
                    f"({sum(coverage.values()):,} covered bases)"
                )
            if args.export_target_bigwig:
                target_bigwig_path = (
                    args.target_bigwig_output.resolve()
                    if args.target_bigwig_output is not None
                    else output_dir / "observed_fe_test.bw"
                )
                target_coverage = write_regression_bigwig(
                    target_bigwig_path,
                    test_result,
                    test_result.full_regression_targets,
                    aggregation=args.bigwig_aggregation,
                    chromosome_prefix=args.bigwig_chrom_prefix,
                    chrom_sizes_path=args.chrom_sizes,
                )
                print(
                    f"Observed FE BigWig: {target_bigwig_path} "
                    f"({sum(target_coverage.values()):,} covered bases)"
                )
        with (output_dir / "regression_metrics.json").open(
            "wt", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "regression_mode": training_config.regression_mode,
                    "regression_weight": training_config.regression_weight,
                    "scalar_all": test_result.regression,
                    "scalar_positive": test_result.regression_positive,
                    "scalar_negative": test_result.regression_negative,
                    "profile_all": test_result.profile_regression,
                    "profile_positive": test_result.profile_regression_positive,
                    "profile_negative": test_result.profile_regression_negative,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
        write_regression_summary(
            output_dir / "regression_summary.txt",
            test_result,
            training_config.regression_mode,
        )
        write_multitask_summary(
            output_dir / "multitask_summary.txt",
            test_result,
            threshold_strategy_used,
            training_config.regression_mode,
            training_config.regression_weight,
        )
    print(f"Outputs written to {output_dir}")
    print("Binary predictions: binary_classification_results.tsv")
    print("Binary metrics: binary_classification_metrics.json")
    print(f"ROC plot: {roc_plot_path.name}")
    print(f"Precision-recall plot: {pr_plot_path.name}")
    if "fe_profiles" in cache:
        print("Regression predictions: regression_results.tsv")
        if args.export_full_regression_profiles:
            print("Full FE profiles: regression_profiles.npz")
            print("Full FE profile index: regression_profile_index.tsv")
        if args.export_bigwig:
            print("Predicted FE BigWig export completed.")
        if args.export_target_bigwig:
            print("Observed FE BigWig export completed.")
        print("Regression metrics: regression_metrics.json")
        print("Joint summary: multitask_summary.txt")
    print("Confusion matrix: binary_confusion_matrix.tsv")
    return 0


# Fit strategy, inference and workflow
# ------------------------------------

def baseline(context, data, **params):
    args = training_namespace()
    unknown = set(params) - set(vars(args))
    reserved = {"dataset_root", "cache_path", "output_dir", "skip_test", "rebuild_cache", "seed", "device"}
    if unknown or reserved & params.keys():
        raise ValueError(f"Invalid/reserved fitting settings: {unknown | (reserved & params.keys())}")
    for key, value in params.items():
        setattr(args, key, value)
    model_params = context.config["predictor"]["params"]
    known_model_keys = {f.name for f in fields(ModelConfig)}
    translations = {"transformer_heads":"heads", "transformer_ffn":"ffn_channels", "sequence_length":"input_length"}
    for key,value in model_params.items():
        if key in known_model_keys:
            setattr(args, translations.get(key,key), value)
    args.motif_kernels = tuple(model_params.get("motif_kernels", (9,12)))
    args.allow_negative_regression_output = not model_params.get("regression_nonnegative",True)
    args.cache_path = Path(data[context.train_assay])
    args.output_dir = context.run_dir / "training"
    args.seed = context.seed
    args.device = context.device
    if context.workers is not None:
        args.num_workers = context.workers
    # Use the original validation-only endpoint, then the common inference
    # component evaluates both assays exactly once using the frozen checkpoint.
    args.skip_test = True
    context.recorder.update(resolved_training_arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    make_model = lambda config: invoke(context.config["predictor"], context, model_config=config)
    code = train_command(args, recorder=context.recorder, model_factory=make_model)
    if code:
        raise RuntimeError(f"Training returned status {code}")
    checkpoint = args.output_dir / "model_for_prediction.pt"
    import torch
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    atomic_json(args.output_dir / "prediction_metadata.json", {
        "checkpoint_path":str(checkpoint), "checkpoint_sha256":file_hash(checkpoint),
        "epoch":frozen["epoch"], "classification_threshold":frozen["classification_threshold"],
        "checkpoint_objective":frozen["checkpoint_objective"],
        "validation_score":frozen["validation_score"], "amp_enabled":frozen["amp_enabled"],
        "trainable_parameter_count":context.recorder.meta.get("trainable_parameter_count"),
    })
    context.artifacts["checkpoint"] = checkpoint
    return checkpoint

def fixed_views(context, data, checkpoint, eval_crops="multi", eval_rc=True):
    if checkpoint is None:
        raise ValueError("fixed_views requires a trained checkpoint")
    device = resolve_device(context.device)
    frozen = torch.load(checkpoint, map_location=device, weights_only=True)
    source_hash = context.recorder.meta.get("train_cache_sha256")
    if not source_hash or frozen["cache_sha256"] != source_hash:
        raise ValueError("Source checkpoint/cache provenance mismatch")
    if (eval_crops,eval_rc) != (frozen["evaluation_crop_mode"], frozen["evaluation_reverse_complement"]):
        raise ValueError("Inference must match the source-validation settings frozen in this recipe")
    config = ModelConfig(**frozen["model_config"])
    model = invoke(context.config["predictor"], context, model_config=config).to(device)
    model.load_state_dict(frozen["model_state"], strict=True)
    parameter_count, trainable = count_parameters(model)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(frozen["negatives_per_positive"]),device=device))
    training_config = TrainingConfig(**frozen["training_config"])
    training_args = context.recorder.meta["resolved_training_arguments"]
    workers = training_args["num_workers"]
    batch_size = training_args["eval_batch_size"]
    amp = bool(frozen["amp_enabled"]) and device.type == "cuda"
    checkpoint_hash = file_hash(checkpoint)
    starts = evaluation_crop_starts(301,config.sequence_length,eval_crops)
    for target in context.config["test_assays"]:
        if target not in data:
            raise ValueError(f"No {target} cache for {context.tf}")
        wall_start = time.perf_counter()
        output = context.run_dir / "evaluations" / "standard" / target
        output.mkdir(parents=True, exist_ok=False)
        cache_path = Path(data[target])
        cache = load_cache(cache_path)
        indices = np.flatnonzero(cache["split"] == 2)
        cache_hash = file_hash(cache_path)
        loader = make_loader(TFBindingDataset(cache,indices),batch_size,False,workers,device)
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = run_epoch(model,loader,device,criterion,training_config,
            optimizer=None,scaler=None,use_amp=amp,gradient_clip=training_args["gradient_clip"],
            classification_threshold=frozen["classification_threshold"],eval_crops=eval_crops,eval_rc=eval_rc)
        if device.type == "cuda": torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-started
        metrics = dict(result.classification)
        positive = int(result.labels.sum())
        metrics.update(sample_count=len(result.labels),positive_count=positive,negative_count=len(result.labels)-positive,
            pr_baseline=float(result.labels.mean()),brier_score=float(brier_score_loss(result.labels,result.probabilities)),
            log_loss=float(log_loss(result.labels,np.asarray(result.probabilities,dtype=np.float64),labels=[0,1])))
        write_predictions(output/"predictions.tsv",result,frozen["classification_threshold"])
        write_confusion_matrix(output/"confusion_matrix.tsv",metrics)
        write_binary_summary(output/"summary.txt",metrics,"frozen-source-validation")
        write_classification_curves(output,result,"classification")
        payload = {
            "experiment_id":context.config["name"],"tf":context.tf,"train_assay":context.train_assay,
            "test_assay":target,"seed":context.seed,"evaluation_id":"standard","status":"completed",
            "checkpoint_path":str(checkpoint),"checkpoint_sha256":checkpoint_hash,"checkpoint_epoch":int(frozen["epoch"]),
            "threshold_strategy":training_args["threshold_strategy"],"threshold":frozen["classification_threshold"],
            "train_cache_sha256":source_hash,"test_cache_sha256":cache_hash,"test_metrics":metrics,
            "losses":result.losses,"inference_s":elapsed,"evaluation_wall_s":time.perf_counter()-wall_start,
            "inference_config":{"crop_starts":starts,"reverse_complement":eval_rc,"aggregation":"mean_probability"},
        }
        atomic_json(output/"metrics.json",payload)
        context.recorder.record("evaluation",status="completed",test_assay=target,
            direction=f"{context.train_assay}_to_{target}",evaluation_id="standard",device=str(device),
            **metrics,classification_loss=result.losses["classification"],regression_loss=result.losses["regression"],
            total_loss=result.losses["total"],inference_s=elapsed,evaluation_wall_s=payload["evaluation_wall_s"],
            samples_per_s=len(result.labels)/elapsed,checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_hash,
            best_epoch=int(frozen["epoch"]),threshold_strategy=training_args["threshold_strategy"],
            parameter_count=parameter_count,trainable_parameter_count=trainable,
            test_cache_path=str(cache_path),test_cache_sha256=cache_hash,
            split_hash=hashlib.sha256(cache_hash.encode()+indices.tobytes()).hexdigest(),
            evaluation_crops_json=json_text(starts),evaluation_rc=eval_rc,forward_views=len(starts)*(2 if eval_rc else 1),
            model_config_json=json_text(asdict(config)),inference_config_json=json_text(payload["inference_config"]),
            test_positive=positive,test_negative=len(result.labels)-positive)
        print(f"{context.tf} {context.train_assay}->{target}: AUPRC={metrics['auprc']:.6f}; {elapsed:.2f}s")

def supervised(context):
    data = invoke(context.config["data"], context)
    if context.config["fitting"] is not None:
        started = time.perf_counter()
        checkpoint = invoke(context.config["fitting"], context, data)
        context.recorder.update(training_wall_s=time.perf_counter()-started)
    else:
        checkpoint = context.artifacts.get("checkpoint")
    invoke(context.config["inference"], context, data, checkpoint)
    for analysis in context.config["analyses"]:
        invoke(analysis, context, data)


# Optional multi-TF planning
# --------------------------

def multi_tf(config,manifest,tf_filter=None,train_assays=None,seeds=None):
    """One task per source assay and seed, containing every selected TF.

    Select this planner together with a multi-TF workflow/data/fit factory. This
    only plans the tasks; no multi-TF learning algorithm is implemented here.
    """
    tfs=list(dict.fromkeys(r["tf"] for r in manifest))
    if tf_filter: tfs=[t for t in tfs if t in tf_filter]
    lookup={(r["tf"],r["assay"]):r["cache"] for r in manifest}
    rows=[]
    for source in train_assays or config["train_assays"]:
        paths={}
        for tf in tfs:
            for assay in set(config["test_assays"])|{source}:
                if (tf,assay) not in lookup: raise ValueError(f"Missing {tf}/{assay}")
                paths[f"{tf}|{assay}"]=str(lookup[(tf,assay)])
        for seed in seeds or config["seeds"]:
            rows.append({"tf":"","tf_ids_json":json.dumps(tfs),"training_scope":"multi_tf",
                "train_assay":source,"seed":seed,"cache_paths_json":json.dumps(paths,sort_keys=True)})
    return rows


# CSV and experiment summaries
# ----------------------------

def number(value):
    if value in (None, ""): return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError): return None


def truth(value):
    return value is True or str(value).lower() == "true"


def tf_set(rows):
    names=set()
    for row in rows:
        if row.get("tf"):
            names.add(row["tf"])
        elif row.get("tf_ids_json"):
            names.update(json.loads(row["tf_ids_json"]))
    return names


def summaries(records, level, source, keys):
    groups = defaultdict(list)
    for row in records:
        if row["row_type"] == source and truth(row.get("included_in_summary")):
            groups[tuple(row.get(k,"") for k in keys)].append(row)
    result = []
    for group_key, rows in groups.items():
        common = {k:rows[0].get(k) for k in ("schema_version","experiment_id","reference_id","config_hash","code_hash")}
        common.update(dict(zip(keys,group_key)))
        common.update(row_type="summary",status="completed",included_in_summary=True,
            summary_level=level,summary_source=source,n_records=len(rows),
            n_runs=len({r["run_id"] for r in rows}),n_tfs=len(tf_set(rows)),
            n_seeds=len({r.get("seed") for r in rows if r.get("seed") is not None}))
        for statistic in ("mean","std","median","q25","q75","min","max","sum"):
            row = dict(common,statistic=statistic)
            row["aggregation_unit"] = "unique_runs" if statistic == "sum" else ("tf_mean_over_seeds" if level=="experiment" else "seed")
            for field in NUMERIC:
                if statistic == "sum" and field not in SUM_FIELDS: continue
                values = [number(r.get(field)) for r in rows]
                values = [v for v in values if v is not None]
                if not values: continue
                if field=="win_fraction":
                    by_tf_delta=defaultdict(list)
                    for r in rows:
                        delta=number(r.get("delta_auprc"))
                        if delta is not None: by_tf_delta[r.get("tf",r["run_id"])].append(delta)
                    values=[float(np.mean(v)>0) for v in by_tf_delta.values()]
                    if not values: continue
                elif level == "experiment" and statistic != "sum":
                    by_tf = defaultdict(list)
                    for r in rows:
                        value = number(r.get(field))
                        if value is not None: by_tf[r.get("tf",r["run_id"])].append(value)
                    values = [float(np.mean(v)) for v in by_tf.values()]
                if statistic == "std":
                    value = float(np.std(values,ddof=1)) if len(values)>1 else None
                elif statistic == "mean": value = float(np.mean(values))
                elif statistic == "median": value = float(np.median(values))
                elif statistic == "q25": value = float(np.percentile(values,25))
                elif statistic == "q75": value = float(np.percentile(values,75))
                elif statistic == "min": value = min(values)
                elif statistic == "max": value = max(values)
                else: value = sum(values)
                row[field] = value
            result.append(row)
    return result


def pair_key(row):
    return tuple(str(row.get(k,"")) for k in ("tf","train_assay","test_assay","seed","evaluation_id",
        "test_cache_sha256","split_hash","evaluation_crops_json","evaluation_rc"))


def add_reference_comparisons(rows, root, reference_id, experiment_id):
    reference_file = root/"results"/"Results_all"/f"{reference_id}.csv"
    if experiment_id == reference_id:
        baseline = [r for r in rows if r["row_type"]=="evaluation" and truth(r.get("included_in_summary"))]
    elif reference_file.exists():
        with reference_file.open(newline="") as handle:
            baseline = [r for r in csv.DictReader(handle) if r["row_type"]=="evaluation" and truth(r.get("included_in_summary"))]
    else:
        return
    lookup = {pair_key(r):r for r in baseline}
    for row in rows:
        if row["row_type"] != "evaluation": continue
        ref = lookup.get(pair_key(row))
        if ref is None: continue
        value, reference = number(row.get("auprc")),number(ref.get("auprc"))
        if value is None or reference is None: continue
        row["paired_reference_run_id"] = ref["run_id"]
        row["delta_auprc"] = value-reference
        row["win_fraction"] = float(value>reference)
        if reference>0:
            row["ratio_auprc"] = value/reference
            if value>0: row["log2_ratio_auprc"] = math.log2(value/reference)


def collect(root: Path, experiment_id: str):
    experiment = root/"results"/experiment_id
    config = json.loads((experiment/"config.json").read_text())
    destination = root/"results"/"Results_all"
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/f".{experiment_id}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _collect(root,experiment,config,destination)


def _collect(root, experiment, config, destination):
    rows,metadata = [],{}
    for path in sorted((experiment/"runs").rglob("run.json")):
        meta = json.loads(path.read_text())
        metadata[meta["run_id"]] = meta
        records_path = path.parent/"records.jsonl"
        if records_path.exists():
            for line in records_path.read_text().splitlines():
                if line.strip(): rows.append(json.loads(line))
        if not any(r["run_id"]==meta["run_id"] and r["row_type"]=="run" for r in rows):
            row = {k:v for k,v in meta.items() if k in COLUMNS}
            row.update(schema_version=SCHEMA_VERSION,row_type="run",epochs_run=meta.get("epochs_run",0),
                status=meta["status"],error=meta.get("error",""))
            rows.append(row)
    if not rows:
        raise ValueError(f"No recorded runs for {config['name']}")
    # A repeated attempt with the same TF/source/seed is not an additional seed.
    # Pick the latest completed attempt, retain every attempt in the raw CSV.
    selected = {}
    for meta in sorted(metadata.values(),key=lambda m:m["started_at"]):
        if meta["status"] != "completed": continue
        key=(meta.get("training_scope","single_tf"),meta.get("tf"),meta.get("tf_ids_json"),meta["train_assay"],meta["seed"],meta["config_hash"])
        selected[key]=meta["run_id"]
    active = set(selected.values())
    code_hashes = {metadata[r]["code_hash"] for r in active}
    config_hashes = {metadata[r]["config_hash"] for r in active}
    if len(code_hashes)>1 or len(config_hashes)>1:
        raise ValueError("Selected runs mix code/config versions. Use a distinct experiment name for a changed implementation.")
    for row in rows:
        row["included_in_summary"] = row["run_id"] in active and row.get("status")=="completed"
    # Enrich evaluation rows with source-run descriptors without copying runtime
    # totals onto every target-assay row.
    run_lookup = {r["run_id"]:r for r in rows if r["row_type"]=="run"}
    descriptive = {"epochs_requested","epochs_run","best_epoch","parameter_count","trainable_parameter_count",
        "kernels_json","channels_per_branch_json","input_length","batch_size","eval_batch_size","num_workers",
        "learning_rate_initial","weight_decay","patience","min_checkpoint_epoch","negatives_per_positive",
        "pos_weight","checkpoint_objective","validation_selection_score","amp_enabled","torch_version",
        "device_name","train_positive","train_negative","val_positive","val_negative"}
    for row in rows:
        if row["row_type"]=="evaluation":
            source = run_lookup.get(row["run_id"],{})
            for key in descriptive:
                if row.get(key) is None: row[key]=source.get(key)
    add_reference_comparisons(rows,root,config["reference"],config["name"])
    aggregated = []
    aggregated += summaries(rows,"tf","run",["tf","tf_ids_json","training_scope","train_assay"])
    aggregated += summaries(rows,"tf","evaluation",["tf","train_assay","test_assay","direction","evaluation_id"])
    aggregated += summaries(rows,"experiment","run",["train_assay"])
    aggregated += summaries(rows,"experiment","run",[])
    aggregated += summaries(rows,"experiment","evaluation",["train_assay","test_assay","direction","evaluation_id"])
    all_rows = rows+aggregated
    chosen = [metadata[r] for r in active]
    if chosen:
        started=min(datetime.fromisoformat(m["started_at"]) for m in chosen)
        finished=max(datetime.fromisoformat(m["finished_at"]) for m in chosen)
        all_rows.append({"schema_version":SCHEMA_VERSION,"row_type":"summary","experiment_id":config["name"],
            "reference_id":config["reference"],"summary_level":"experiment","summary_source":"run",
            "statistic":"span","aggregation_unit":"calendar_span","status":"completed","included_in_summary":True,
            "metric_name":"experiment_elapsed_wall_s","metric_value":(finished-started).total_seconds(),
            "metric_unit":"s","n_runs":len(chosen)})
    csv_path=destination/f"{config['name']}.csv"
    tmp=csv_path.with_name(csv_path.name+f".{os.getpid()}.tmp")
    with tmp.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS,extrasaction="raise")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k:("" if v is None else v) for k,v in row.items()})
    tmp.replace(csv_path)
    write_report(experiment,config,rows,aggregated,csv_path)
    atomic_json(destination/"schema.json",{"schema_version":SCHEMA_VERSION,"columns":COLUMNS,
        "row_types":["run","epoch","evaluation","analysis","summary"],
        "time_unit":"seconds","memory_unit":"bytes","auprc_definition":"sklearn average_precision_score"})
    print(f"Collected {config['name']}: {len(rows)} raw rows, {len(aggregated)} summary rows -> {csv_path}")
    return csv_path


def fmt(value,digits=4):
    n=number(value)
    return "—" if n is None else f"{n:.{digits}f}"


def write_report(experiment,config,rows,aggregated,csv_path):
    runs=[r for r in rows if r["row_type"]=="run"]
    selected=[r for r in runs if truth(r.get("included_in_summary"))]
    lines=[f"# {config['name']}","",config.get("description",""),"",
        f"Reference: `{config['reference']}`. Selected completed runs: {len(selected)}. Recorded attempts: {len(runs)}.",
        f"Standardized CSV: `{csv_path.relative_to(experiment.parent.parent)}`.","",
        "## Test metrics", "", "Seeds are averaged per TF before aggregation across TFs.","",
        "| Direction | TFs | Runs | AUPRC mean | AUPRC median | AUROC mean | MCC mean | Δ AUPRC mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    means=[r for r in aggregated if r["summary_level"]=="experiment" and r["summary_source"]=="evaluation" and r["statistic"]=="mean"]
    for row in means:
        median=next(r for r in aggregated if r["summary_level"]=="experiment" and r["summary_source"]=="evaluation" and r["statistic"]=="median" and r.get("direction")==row["direction"])
        lines.append(f"| {row['direction']} | {row['n_tfs']} | {row['n_runs']} | {fmt(row.get('auprc'))} | {fmt(median.get('auprc'))} | {fmt(row.get('auroc'))} | {fmt(row.get('mcc'))} | {fmt(row.get('delta_auprc'))} |")
    lines += ["","## Runtime per TF and source assay","",
        "Training totals count each trained model once. Epoch time includes training, validation, mining and epoch overhead before checkpoint/report writing.","",
        "| TF | Source | Seeds | Epochs mean | Epoch seconds mean | Training seconds sum | Inference seconds sum |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in [r for r in aggregated if r["summary_level"]=="tf" and r["summary_source"]=="run" and r["statistic"]=="mean"]:
        summed=next(r for r in aggregated if r["summary_level"]=="tf" and r["summary_source"]=="run" and r["statistic"]=="sum" and r["tf"]==row["tf"] and r["train_assay"]==row["train_assay"])
        lines.append(f"| {row['tf']} | {row['train_assay']} | {row['n_seeds']} | {fmt(row.get('epochs_run'),1)} | {fmt(row.get('epoch_mean_s'),2)} | {fmt(summed.get('training_wall_s'),2)} | {fmt(summed.get('inference_s'),2)} |")
    total=next((r for r in aggregated if r["summary_level"]=="experiment" and r["summary_source"]=="run" and not r.get("train_assay") and r["statistic"]=="sum"),{})
    lines += ["",f"Summed training time: {fmt(total.get('training_wall_s'),2)} s. Summed job time including inference: {fmt(total.get('total_wall_s'),2)} s.",
        "Calendar span is recorded separately in the CSV and can include gaps between job submissions.","",
        "## Attempts", "", "| TF | Source | Seed | Run | Status | Included |", "|---|---|---:|---|---|---|"]
    for row in runs:
        lines.append(f"| {row.get('tf','')} | {row.get('train_assay','')} | {row.get('seed','')} | {row['run_id']} | {row.get('status','')} | {row.get('included_in_summary',False)} |")
    lines += ["","## Interpretation","",
        "AUPRC is average precision. MCC/F1 use the threshold selected on source validation only.",
        "Raw `evaluation` rows contain individual TF/source/target/seed results. `run` rows contain runtime totals; `epoch` rows contain the complete epoch trajectory.",
        "Repeated attempts are retained; only the latest completed attempt per TF/source/seed contributes to summaries.",""]
    path=experiment/"evaluation.md"
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)


# Numerical checks with an external reference
# -------------------------------------------

def reference_module(root, reference_path=None):
    import data as cache_tools
    sys.modules.setdefault("hum301_data", cache_tools)
    name="_hum301_frozen_reference"
    spec=importlib.util.spec_from_file_location(name,Path(reference_path) if reference_path else root/"reference"/"hum301_motif64.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def synthetic_cache(path: Path, seed=101):
    """Tiny valid cache for software checks; never scientific experiment data."""
    rng=np.random.default_rng(seed)
    labels=np.array([1,1]+[0]*20+[1,1,0,0]+[1,1,0,0],dtype=np.uint8)
    split=np.array([0]*22+[1]*4+[2]*4,dtype=np.uint8)
    n=len(labels)
    starts=np.arange(n,dtype=np.int64)*1000
    np.savez_compressed(path,sequences=rng.integers(0,4,size=(n,301),dtype=np.uint8),labels=labels,
        split=split,chromosomes=np.array([f"chr{int(s)+1}" for s in split]),starts=starts,ends=starts+301,
        sample_ids=np.array([f"synthetic_{seed}_{i}" for i in range(n)]),
        summary_json=np.array(json.dumps({"format_version":"at301-v1","sequence_audit":{},"splits":{}})))


def same_state(left,right):
    a,b=left.state_dict(),right.state_dict()
    if a.keys()!=b.keys(): raise AssertionError("State-dict keys changed")
    for name in a:
        if not torch.equal(a[name],b[name]):
            raise AssertionError(f"State mismatch: {name}; max difference {float((a[name]-b[name]).abs().max())}")


def same_outputs(a,b):
    if a.keys()!=b.keys(): raise AssertionError("Output names changed")
    for key in a:
        if not torch.equal(a[key],b[key]): raise AssertionError(f"Output mismatch: {key}")


def verify(root: Path, output: Path, reference_path=None):
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(2)
    reference=reference_module(root, reference_path)
    checks=[]
    try:
        original=reference.build_parser().parse_args(["train","--cache","CACHE","--output-dir","OUTPUT"])
        current=vars(training_namespace())
        for key,value in current.items():
            if getattr(original,key)!=value: raise AssertionError(f"Changed default {key}")
        checks.append("all original argparse training defaults preserved")
        for seed in (1,13,42):
            reference.seed_everything(seed)
            old=reference.HumanTFBindingModel(reference.ModelConfig())
            seed_everything(seed)
            new=HumanTFBindingModel(ModelConfig())
            same_state(old,new)
        checks.append("state-dict names, initial weights and buffers bitwise equal for seeds 1/13/42")
        old.eval();new.eval()
        generator=torch.Generator().manual_seed(99)
        codes=torch.randint(0,4,(2,256),generator=generator)
        sequence=torch.nn.functional.one_hot(codes,4).transpose(1,2).float()
        centers=torch.tensor([128,127])
        with torch.no_grad(): same_outputs(old(sequence,centers),new(sequence,centers))
        checks.append("all forward outputs bitwise equal, including center indices 128 and 127")
        full=torch.nn.functional.one_hot(torch.randint(0,4,(2,301),generator=generator),4).transpose(1,2).float()
        for crops in ("center","multi"):
            for rc in (False,True):
                with torch.no_grad():
                    a=reference.classification_forward(old,full,centers,crops,rc)
                    b=classification_forward(new,full,centers,crops,rc)
                same_outputs(a,b)
        checks.append("center/multi crop and RC probability averaging bitwise equal")
        labels=torch.tensor([0.,1.])
        for epoch in range(3):
            opt_old=torch.optim.AdamW(old.parameters(),lr=3e-4,weight_decay=1e-4)
            opt_new=torch.optim.AdamW(new.parameters(),lr=3e-4,weight_decay=1e-4)
            old.train();new.train()
            for model,opt in ((old,opt_old),(new,opt_new)):
                opt.zero_grad(set_to_none=True)
                torch.manual_seed(1000+epoch)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(sequence,centers)["classification_logit"],labels,pos_weight=torch.tensor(10.))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
            same_state(old,new)
        checks.append("dropout, gradients, clipping, AdamW updates and normalization buffers bitwise equal")
        with tempfile.TemporaryDirectory(prefix="hum301_verify_") as temp:
            temp=Path(temp)
            cache_path=temp/"dataset.npz"
            synthetic_cache(cache_path)
            cache=load_cache(cache_path)
            indices=np.flatnonzero(cache["split"]==0)
            old_data=reference.TFBindingDataset(cache,indices,0.5,seed=13,training=True)
            new_data=TFBindingDataset(cache,indices,0.5,seed=13,training=True)
            for epoch in (0,1,5):
                old_data.set_epoch(epoch);new_data.set_epoch(epoch)
                for index in (0,1,2,10,0):
                    a,b=old_data[index],new_data[index]
                    for key in a:
                        if torch.is_tensor(a[key]):
                            if not torch.equal(a[key],b[key]): raise AssertionError(f"Dataset mismatch {key}")
                        elif a[key]!=b[key]: raise AssertionError(f"Dataset mismatch {key}")
            checks.append("fresh random crops, RC augmentation and per-worker RNG match")
            old_sampler=reference.CyclicBalancedNegativeSampler(old_data,seed=13,negatives_per_positive=10,batch_size=88)
            new_sampler=CyclicBalancedNegativeSampler(new_data,seed=13,negatives_per_positive=10,batch_size=88)
            for epoch in range(5):
                a,b=list(old_sampler),list(new_sampler)
                if a!=b: raise AssertionError("Sampler order changed")
            checks.append("cyclic negative selection and batch order match over five epochs")
            # A full training trajectory catches scheduler/checkpoint/threshold
            # regressions that a forward-only smoke check cannot detect.
            original=reference.build_parser().parse_args(["train","--cache",str(cache_path),"--output-dir",str(temp/"old"),
                "--epochs","2","--num-workers","0","--eval-batch-size","8","--device","cpu","--no-amp","--skip-test"])
            new_args=copy.deepcopy(original)
            new_args.output_dir=temp/"new"
            new_args.motif_kernels=(9,12)
            class Recorder:
                def update(self,**kwargs): pass
                def epoch(self,row): pass
                def fit_loop_finished(self,*args): pass
            buffer=io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
                    reference.train_command(original)
                    train_command(new_args,recorder=Recorder())
            except Exception:
                print(buffer.getvalue()[-5000:])
                raise
            a=torch.load(temp/"old"/"model_for_prediction.pt",map_location="cpu",weights_only=True)
            b=torch.load(temp/"new"/"model_for_prediction.pt",map_location="cpu",weights_only=True)
            for key in a["model_state"]:
                if not torch.equal(a["model_state"][key],b["model_state"][key]): raise AssertionError(f"Training state differs: {key}")
            for key in ("epoch","validation_score","classification_threshold","checkpoint_objective","evaluation_crop_mode","evaluation_reverse_complement"):
                if a[key]!=b[key]: raise AssertionError(f"Checkpoint metadata differs: {key}")
            if (temp/"old"/"validation_predictions.tsv").read_bytes()!=(temp/"new"/"validation_predictions.tsv").read_bytes():
                raise AssertionError("Validation predictions differ")
            old_history=json.loads((temp/"old"/"history.json").read_text())
            new_history=json.loads((temp/"new"/"history.json").read_text())
            for old_row,new_row in zip(old_history,new_history):
                for key in ("train","val","sampling","train_shift","learning_rate_used","learning_rate_next"):
                    if old_row[key]!=new_row[key]: raise AssertionError(f"Epoch history differs: {key}")
            checks.append("two full training epochs: weights, LR schedule, metrics, selected checkpoint and validation-MCC threshold bitwise equal")
        variants=[]
        for path in sorted((root/"configs").glob("*.json")):
            c=json.loads(path.read_text())
            params=c["predictor"]["params"]
            params={k:tuple(v) if k in {"motif_kernels","encoder_channels"} else v for k,v in params.items()}
            model=HumanTFBindingModel(ModelConfig(**params)).eval()
            with torch.no_grad(): result=model(sequence,centers)
            if result["classification_track"].shape!=(2,256): raise AssertionError("Invalid track shape")
            widths=[branch.out_channels for branch in model.embedder.initial.branches]
            budget=params.get("embed_channels",64)
            expected=[budget//len(widths)+int(i<budget%len(widths)) for i in range(len(widths))]
            if widths!=expected: raise AssertionError("Input channel allocation differs from configuration")
            variants.append({"name":c["name"],"kernels":list(model.config.motif_kernels),"channels":widths,
                "channel_dropout":model.config.channel_dropout,"bottleneck_dropout":model.config.dropout,
                "parameters":sum(p.numel() for p in model.parameters())})
        checks.append("all configured variants forward successfully with their specified input channel allocation")
        report={"status":"passed","device":"cpu","torch_version":str(torch.__version__),
            "reference_sha256":file_hash(Path(reference_path) if reference_path else root/"reference"/"hum301_motif64.py"),
            "data_dependency_sha256":file_hash(root/"data.py"),
            "checks":checks,"variants":variants,
            "limits":"Synthetic regression tests; no scientific TF training data were available here. GPU/ROCm full-run reproducibility depends on the original environment."}
        atomic_json(output,report)
        print(json.dumps(report,indent=2))
        return report
    finally:
        torch.set_num_threads(previous_threads)
