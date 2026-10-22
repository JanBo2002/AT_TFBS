"""Complete predictor builders; no common encoder/decoder is mandatory."""
from __future__ import annotations

from .model import HumanTFBindingModel, ModelConfig
from dataclasses import fields


def maa(context, model_config=None, **params):
    """Build the original MAA model with configurable parallel input kernels."""
    unknown = set(params) - {f.name for f in fields(ModelConfig)}
    if unknown:
        raise ValueError(f"Unknown MAA predictor settings: {sorted(unknown)}")
    if model_config is None:
        params = dict(params)
        for key in ("motif_kernels", "encoder_channels"):
            if key in params:
                params[key] = tuple(params[key])
        model_config = ModelConfig(**params)
    return HumanTFBindingModel(model_config)
