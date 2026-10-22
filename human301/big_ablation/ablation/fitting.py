"""The first fit strategy preserves the reference optimization algorithm."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from .defaults import training_namespace
from .model import ModelConfig
from .specs import invoke
from .training import train_command
from .recorder import atomic_json, file_hash


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
