"""Frozen source-selected threshold; evaluate source and target assay separately."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, log_loss

from .data import TFBindingDataset
from .epoch import run_epoch
from .hum301_data import evaluation_crop_starts, load_cache
from .loaders import make_loader, resolve_device
from .losses import TrainingConfig
from .model import ModelConfig
from .recorder import atomic_json, file_hash
from .reporting import write_predictions, write_confusion_matrix, write_binary_summary, write_classification_curves
from .specs import invoke, json_text
from .utils import count_parameters


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
