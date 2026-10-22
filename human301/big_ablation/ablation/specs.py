"""Small, open extension points. Factories are ordinary trusted Python code."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


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
            raise ValueError("These three Human301 recipes support CHIP and GHTS")
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
