#!/usr/bin/env python3
"""Run the existing local Human301 model on ChIP and GHTS, with both test sets.

The model and hum301_data.py stay unchanged. Each TF has two independently
trained models, one seed, and four evaluations with frozen source thresholds.
Only data preparation runs in the CPU array; training and inference use GPUs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/data/projects/SFB_A03/jan/AT_TFBS")
TFS = ("CREB3L3", "TERF1", "SRY", "BATF2", "CTCF", "NFKB1", "MAX", "MGA", "MYF6", "ELF3", "ZBED9")
ASSAYS = ("CHIP", "GHTS")
MODEL_SCRIPT = "hum301_motif64_local.py"
DATA_SCRIPT = "hum301_data.py"
FORMAT = "hum301-crossassay-v1"
METRICS = ("auprc", "auroc", "mcc", "balanced_accuracy", "f1", "precision", "recall", "specificity")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise ValueError("Missing files:\n" + "\n".join(missing))


def bed_paths(input_root, tf):
    root = Path(input_root)
    return {
        "train_positive_bed": root / "Train" / tf / "positives.bed",
        "train_negative_bed": root / "Train" / tf / "random.bed",
        "test_positive_bed": root / "Test" / tf / "positives.bed",
        "test_negative_bed": root / "Test" / tf / "random.bed",
    }


def plan(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id must contain only letters, digits, underscores or hyphens")
    root = args.project_root.resolve()
    code = root / "human301"
    handoff = root / "raw_data/clara/chip/student_handoff_30tf_increment"
    inputs = {"CHIP": str(handoff / "Input_data/CHS"), "GHTS": str(handoff / "GHTS")}
    genome = root / "raw_data/general_data/genome/hg38.fa"
    required = [code / MODEL_SCRIPT, code / DATA_SCRIPT, genome, Path(str(genome) + ".fai")]
    for tf in TFS:
        for assay in ASSAYS:
            required.extend(bed_paths(inputs[assay], tf).values())
    require_files(required)
    if args.manifest.exists():
        raise ValueError(f"Workflow already exists: {args.manifest}")
    runs = [{"tf": tf, "train_assay": assay, "seed": args.seed,
             "run_dir": str(code / "runs" / tf / args.run_id / f"{assay}_seed{args.seed}")}
            for tf in TFS for assay in ASSAYS]
    if any(Path(row["run_dir"]).exists() for row in runs):
        raise ValueError("A result directory already exists; choose a new run-id")
    payload = {"format": FORMAT, "run_id": args.run_id, "script_dir": str(code),
               "data_root": str(root / "raw_data/human"), "input_roots": inputs,
               "genome_fasta": str(genome), "validation_chromosomes": ["chr5", "chr7"],
               "model_script": MODEL_SCRIPT, "data_script": DATA_SCRIPT,
               "script_sha256": {name: sha256(code / name) for name in (MODEL_SCRIPT, DATA_SCRIPT)},
               "tfs": list(TFS), "seed": args.seed, "runs": runs}
    write_json(args.manifest, payload)
    print(f"Workflow: {args.manifest}")
    print(f"{len(TFS)} CPU data tasks; {len(runs)} GPU training tasks; {2 * len(runs)} test evaluations.")


def load_manifest(path):
    manifest = read_json(path)
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported workflow manifest")
    return manifest


def check_scripts(manifest):
    code = Path(manifest["script_dir"])
    for name, expected in manifest["script_sha256"].items():
        if sha256(code / name) != expected:
            raise ValueError(f"Script changed after submission: {code / name}")


def data_state_path(manifest_path, tf):
    return Path(manifest_path).parent / "datasets" / f"{tf}.json"


def import_data(manifest):
    sys.path.insert(0, manifest["script_dir"])
    import hum301_data
    return hum301_data


def summary_matches(summary, source_hashes, val_chromosomes):
    if summary.get("genome_build") != "hg38" or summary.get("stored_length") != 301:
        return False
    if set(summary.get("splits", {}).get("validation", {}).get("chromosomes", [])) != set(val_chromosomes):
        return False
    return all(summary.get("sources", {}).get(key, {}).get("sha256") == value
               for key, value in source_hashes.items())


def prepare_cache(manifest, tf, assay, reference_hashes, data_module):
    beds = bed_paths(manifest["input_roots"][assay], tf)
    hashes = {**reference_hashes, **{key: sha256(path) for key, path in beds.items()}}
    tf_root = Path(manifest["data_root"]) / tf
    preferred = tf_root / assay / "dataset.npz"
    candidates = []
    if preferred.is_file():
        candidates.append(preferred)
    # Reuse an existing cache only if the actual BED and reference contents,
    # stored length and validation chromosomes match. No legacy name is assumed.
    if tf_root.is_dir():
        for summary_path in sorted(tf_root.rglob("dataset.summary.json")):
            try:
                summary = read_json(summary_path)
            except (OSError, ValueError):
                continue
            cache_path = summary_path.with_name("dataset.npz")
            if cache_path.is_file() and cache_path not in candidates and summary_matches(
                    summary, hashes, manifest["validation_chromosomes"]):
                candidates.append(cache_path)
    selected = None
    summary = None
    for candidate in candidates:
        cache = data_module.load_cache(candidate)
        candidate_summary = cache["summary"]
        del cache
        if summary_matches(candidate_summary, hashes, manifest["validation_chromosomes"]):
            selected, summary = candidate, candidate_summary
            print(f"{tf}/{assay}: reusing {candidate}", flush=True)
            break
        if candidate == preferred:
            raise ValueError(f"{preferred} belongs to different data; nothing overwritten")
    if selected is None:
        command = [sys.executable, "-u", str(Path(manifest["script_dir"]) / DATA_SCRIPT), "prepare",
                   "--genome-fasta", manifest["genome_fasta"], "--genome-build", "hg38",
                   "--val-chromosomes", ",".join(manifest["validation_chromosomes"]),
                   "--output", str(preferred)]
        for key, path in beds.items():
            command.extend(("--" + key.replace("_", "-"), str(path)))
        print(f"{tf}/{assay}: preparing {preferred}", flush=True)
        subprocess.run(command, check=True)
        cache = data_module.load_cache(preferred)
        summary = cache["summary"]
        del cache
        if not summary_matches(summary, hashes, manifest["validation_chromosomes"]):
            raise ValueError(f"Prepared cache does not match inputs: {preferred}")
        selected = preferred
    print(json.dumps(summary["splits"], indent=2), flush=True)
    return {"cache": str(selected.resolve()), "cache_sha256": sha256(selected), "summary": summary}


def check_cross_splits(datasets):
    for source in ASSAYS:
        splits = datasets[source]["summary"]["splits"]
        used = set(splits["train"]["chromosomes"]) | set(splits["validation"]["chromosomes"])
        for target in ASSAYS:
            held_out = set(datasets[target]["summary"]["splits"]["test"]["chromosomes"])
            overlap = used & held_out
            if overlap:
                raise ValueError(f"{source} train/validation overlaps {target} test chromosomes: {sorted(overlap)}")


def prepare(args):
    manifest = load_manifest(args.manifest)
    check_scripts(manifest)
    if not 0 <= args.task_id < len(manifest["tfs"]):
        raise ValueError("CPU task-id must be 0..10")
    tf = manifest["tfs"][args.task_id]
    genome = Path(manifest["genome_fasta"])
    reference_hashes = {"genome_fasta": sha256(genome), "genome_fai": sha256(str(genome) + ".fai")}
    data_module = import_data(manifest)
    datasets = {assay: prepare_cache(manifest, tf, assay, reference_hashes, data_module) for assay in ASSAYS}
    check_cross_splits(datasets)
    write_json(data_state_path(args.manifest, tf), {"tf": tf, "datasets": datasets})
    print(f"{tf}: CPU preparation completed for CHIP and GHTS; cross-test chromosomes are held out.")


def training_command(manifest, row, cache, device="cuda"):
    return [sys.executable, "-u", str(Path(manifest["script_dir"]) / MODEL_SCRIPT), "train",
            "--cache", cache, "--output-dir", row["run_dir"],
            "--input-length", "256", "--embed-channels", "64", "--encoder-channels", "80", "96", "128",
            "--heads", "4", "--ffn-channels", "256", "--classification-readout", "local-logsumexp",
            "--task-mode", "classification", "--checkpoint-objective", "classification-auprc",
            "--threshold-strategy", "val-mcc", "--negatives-per-positive", "10", "--batch-size", "88",
            "--hard-negative-fraction", "0", "--epochs", "100", "--patience", "20",
            "--learning-rate", "0.0003", "--min-learning-rate", "0.000001",
            "--weight-decay", "0.0001", "--dropout", "0.1", "--activation", "relu",
            "--reverse-complement-probability", "0.5", "--eval-crops", "multi", "--eval-rc",
            "--no-balanced-evaluation", "--eval-batch-size", "128", "--num-workers", "2",
            "--seed", str(row["seed"]), "--device", device, "--skip-test"]


def task_row(manifest, task_id):
    if not 0 <= task_id < len(manifest["runs"]):
        raise ValueError("GPU task-id must be 0..21")
    return manifest["runs"][task_id]


def train(args):
    manifest = load_manifest(args.manifest)
    check_scripts(manifest)
    row = task_row(manifest, args.task_id)
    datasets = read_json(data_state_path(args.manifest, row["tf"]))["datasets"]
    check_cross_splits(datasets)
    source = row["train_assay"]
    require_files([datasets[assay]["cache"] for assay in ASSAYS])
    command = training_command(manifest, row, datasets[source]["cache"], args.device)
    print(f"TF={row['tf']} train={source} seed={row['seed']} model={MODEL_SCRIPT}", flush=True)
    print(shlex.join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    training_seconds = time.perf_counter() - started
    directory = Path(row["run_dir"])
    require_files([directory / "model_for_prediction.pt", directory / "validation_metrics.json", directory / "history.json"])
    write_json(directory / "training_timing.json", {"training_wall_seconds": training_seconds,
               "scope": "training process including setup, epoch training/validation and final source validation; tests excluded"})
    # Tests run once each, after model/threshold selection is finished. Each
    # subprocess releases its cached sequences and model before the next test.
    targets = [source, next(assay for assay in ASSAYS if assay != source)]
    for target in targets:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "evaluate",
                   "--manifest", str(args.manifest.resolve()), "--task-id", str(args.task_id),
                   "--test-assay", target, "--device", args.device]
        print(f"{row['tf']}: {source} -> {target}", flush=True)
        subprocess.run(command, check=True)
    print(f"Completed {row['tf']}/{source}: one model, both fixed test sets.", flush=True)


def import_model(manifest):
    sys.path.insert(0, manifest["script_dir"])
    name = "hum301_crossassay_local_model"
    spec = importlib.util.spec_from_file_location(name, Path(manifest["script_dir"]) / MODEL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(args):
    started = time.perf_counter()
    manifest = load_manifest(args.manifest)
    check_scripts(manifest)
    row = task_row(manifest, args.task_id)
    source, target = row["train_assay"], args.test_assay
    datasets = read_json(data_state_path(args.manifest, row["tf"]))["datasets"]
    check_cross_splits(datasets)
    directory = Path(row["run_dir"])
    checkpoint_path = directory / "model_for_prediction.pt"
    output = directory / f"{source}_to_{target}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Evaluation output already exists: {output}")
    module = import_model(manifest)
    device = module.resolve_device(args.device)
    checkpoint = module.torch.load(checkpoint_path, map_location=device, weights_only=True)
    # The checkpoint must belong to the declared TRAINING cache. The target
    # cache is intentionally different for a cross-assay test.
    if checkpoint.get("cache_sha256") != datasets[source]["cache_sha256"]:
        raise ValueError("Checkpoint does not belong to the declared training cache")
    if sha256(datasets[target]["cache"]) != datasets[target]["cache_sha256"]:
        raise ValueError("Test cache changed after CPU preparation")
    required = ("classification_threshold", "evaluation_crop_mode", "evaluation_reverse_complement")
    if any(field not in checkpoint for field in required):
        raise ValueError("Use model_for_prediction.pt with frozen validation threshold and inference settings")
    if checkpoint["model_config"]["classification_readout"] != "local-logsumexp":
        raise ValueError("The checkpoint is not the requested local model")
    training_config = module.TrainingConfig(**checkpoint["training_config"])
    if training_config.task_mode != "classification":
        raise ValueError("This workflow requires the unchanged classification setup")
    threshold = float(checkpoint["classification_threshold"])
    val = read_json(directory / "validation_metrics.json")["classification"]
    if not math.isfinite(threshold) or abs(float(val["threshold"]) - threshold) > 1e-10:
        raise ValueError("Frozen threshold differs from source-validation threshold")
    cache = module.load_cache(datasets[target]["cache"])
    indices = module.np.flatnonzero(cache["split"] == 2)
    dataset = module.TFBindingDataset(cache, indices)
    loader = module.make_loader(dataset, 128, False, 2, device)
    model = module.HumanTFBindingModel(module.ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = module.nn.BCEWithLogitsLoss(pos_weight=module.torch.tensor(
        float(checkpoint["negatives_per_positive"]), device=device))
    inference_started = time.perf_counter()
    result = module.run_epoch(
        model, loader, device, criterion, training_config, optimizer=None, scaler=None,
        use_amp=bool(checkpoint.get("amp_enabled", False)) and device.type == "cuda",
        gradient_clip=1.0, classification_threshold=threshold,
        eval_crops=checkpoint["evaluation_crop_mode"], eval_rc=checkpoint["evaluation_reverse_complement"],
    )
    inference_seconds = time.perf_counter() - inference_started
    if len(result.labels) != len(indices):
        raise ValueError("Test evaluation did not include every held-out sample")
    output.mkdir(parents=True, exist_ok=True)
    module.write_predictions(output / "test_predictions.tsv", result, threshold)
    module.write_confusion_matrix(output / "binary_confusion_matrix.tsv", result.classification)
    module.write_binary_summary(output / "binary_classification_summary.txt", result.classification, "frozen-source-validation-MCC")
    module.write_classification_curves(output, result, "classification")
    payload = {"tf": row["tf"], "seed": row["seed"], "train_assay": source, "test_assay": target,
               "direction": f"{source}_to_{target}", "checkpoint": str(checkpoint_path),
               "checkpoint_epoch": int(checkpoint["epoch"]), "threshold": threshold,
               "threshold_strategy": "val-mcc", "threshold_source": f"{source} validation only",
               "training_cache": datasets[source]["cache"], "training_cache_sha256": datasets[source]["cache_sha256"],
               "test_cache": datasets[target]["cache"], "test_cache_sha256": datasets[target]["cache_sha256"],
               "evaluation_scope": "all fixed test samples; no target-data training or threshold tuning",
               "evaluation_crop_mode": checkpoint["evaluation_crop_mode"],
               "evaluation_reverse_complement": checkpoint["evaluation_reverse_complement"],
               "test_positive": int((result.labels == 1).sum()), "test_negative": int((result.labels == 0).sum()),
               "inference_seconds": inference_seconds, "evaluation_wall_seconds": time.perf_counter() - started,
               "test_metrics": result.classification}
    write_json(output / "binary_classification_metrics.json", payload)
    module.print_result(f"TEST {row['tf']} {source} -> {target}", result, "classification")


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite value in result")
    return value


def write_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    manifest = load_manifest(args.manifest)
    rows, training_rows = [], []
    parameter_counts = set()
    test_signatures = {}
    for run in manifest["runs"]:
        directory = Path(run["run_dir"])
        config = read_json(directory / "configuration.json")
        history = read_json(directory / "history.json")
        val = read_json(directory / "validation_metrics.json")["classification"]
        timing = read_json(directory / "training_timing.json")
        arguments = config["arguments"]
        expected = {"classification_readout": "local-logsumexp", "seed": manifest["seed"],
                    "negatives_per_positive": 10, "batch_size": 88, "hard_negative_fraction": 0.0,
                    "epochs": 100, "patience": 20, "eval_crops": "multi", "eval_rc": True,
                    "skip_test": True, "task_mode": "classification"}
        if any(arguments.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unexpected training settings in {directory}")
        if config["evaluation_balance"].get("undersampled") is not False or not history:
            raise ValueError(f"Missing history or undersampled validation in {directory}")
        parameter_counts.add(int(config["parameter_count"]))
        training = {"tf": run["tf"], "train_assay": run["train_assay"], "seed": run["seed"],
                    "epochs_run": len(history),
                    "epoch_training_validation_minutes": sum(finite(item["elapsed_seconds"]) for item in history) / 60,
                    "training_wall_minutes": finite(timing["training_wall_seconds"]) / 60,
                    "validation_auprc_selected": finite(val["auprc"]),
                    "validation_auprc_last_epoch": finite(history[-1]["val"]["classification"]["auprc"]),
                    "parameter_count": int(config["parameter_count"]), "run_dir": str(directory)}
        training_rows.append(training)
        threshold = finite(val["threshold"])
        state = read_json(data_state_path(args.manifest, run["tf"]))["datasets"]
        for target in ASSAYS:
            direction = f"{run['train_assay']}_to_{target}"
            path = directory / direction / "binary_classification_metrics.json"
            result = read_json(path)
            source_cache_sha = state[run["train_assay"]]["cache_sha256"]
            if (result["tf"] != run["tf"] or result["seed"] != run["seed"] or result["direction"] != direction
                    or result["train_assay"] != run["train_assay"] or result["test_assay"] != target
                    or result["training_cache_sha256"] != source_cache_sha or config["cache_sha256"] != source_cache_sha
                    or abs(finite(result["threshold"]) - threshold) > 1e-10
                    or result["threshold_strategy"] != "val-mcc"):
                raise ValueError(f"Incompatible model/threshold/source in {path}")
            test = state[target]["summary"]["splits"]["test"]
            signature = (result["test_cache_sha256"], int(result["test_positive"]), int(result["test_negative"]))
            expected_signature = (state[target]["cache_sha256"], test["positive"], test["negative"])
            key = (run["tf"], target)
            if signature != expected_signature or key in test_signatures and test_signatures[key] != signature:
                raise ValueError(f"Models used different test sets or omitted samples: {path}")
            test_signatures[key] = signature
            metrics = result["test_metrics"]
            if abs(finite(metrics["threshold"]) - threshold) > 1e-10:
                raise ValueError(f"Test metrics used a different threshold: {path}")
            if int(result["checkpoint_epoch"]) not in [item["epoch"] for item in history]:
                raise ValueError(f"Selected epoch is absent from history: {path}")
            row = {"tf": run["tf"], "train_assay": run["train_assay"], "test_assay": target,
                   "direction": direction, "seed": run["seed"],
                   **{name: finite(metrics[name]) for name in METRICS}, "threshold": threshold,
                   "test_positive": signature[1], "test_negative": signature[2],
                   "pr_baseline": signature[1] / (signature[1] + signature[2]),
                   "checkpoint_epoch": int(result["checkpoint_epoch"]),
                   **{key: value for key, value in training.items() if key not in ("tf", "train_assay", "seed")},
                   "inference_minutes": finite(result["inference_seconds"]) / 60,
                   "evaluation_wall_minutes": finite(result["evaluation_wall_seconds"]) / 60,
                   "metrics_file": str(path)}
            for name in ("true_positive", "false_positive", "true_negative", "false_negative"):
                row[name] = int(metrics[name])
            rows.append(row)
    if len(rows) != 4 * len(manifest["tfs"]) or len(parameter_counts) != 1:
        raise ValueError("Expected four evaluations per TF with the same local architecture")
    directions = tuple(f"{source}_to_{target}" for source in ASSAYS for target in ASSAYS)
    matrix = []
    for tf in manifest["tfs"]:
        by_direction = {row["direction"]: row for row in rows if row["tf"] == tf}
        if set(by_direction) != set(directions):
            raise ValueError(f"Missing or duplicate direction for {tf}")
        matrix.append({"tf": tf, **{direction: by_direction[direction]["auprc"] for direction in directions}})
    output = Path(args.manifest).parent / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Summary already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output / "results.tsv", rows)
    write_tsv(output / "auprc_by_tf.tsv", matrix)
    write_tsv(output / "training_runtime.tsv", training_rows)
    write_json(output / "summary.json", {"run_id": manifest["run_id"], "seed": manifest["seed"],
               "model": MODEL_SCRIPT, "n_models": len(training_rows), "n_test_evaluations": len(rows),
               "parameter_count": next(iter(parameter_counts)), "auprc_by_tf": matrix,
               "macro_equal_tf_weight": {direction: sum(row[direction] for row in matrix) / len(matrix) for direction in directions},
               "total_training_wall_hours": sum(row["training_wall_minutes"] for row in training_rows) / 60,
               "total_epoch_training_validation_hours": sum(row["epoch_training_validation_minutes"] for row in training_rows) / 60,
               "total_test_evaluation_wall_hours": sum(row["evaluation_wall_minutes"] for row in rows) / 60,
               "runtime_note": "Sums of processes, not array elapsed time; training counted once per model. Queue and CPU data preparation excluded.",
               "training": training_rows, "evaluations": rows})
    print(f"Summary: {output}")
    print("TF\t" + "\t".join(directions))
    for row in matrix:
        print(row["tf"] + "\t" + "\t".join(f"{row[key]:.4f}" for key in directions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    planning.add_argument("--run-id", required=True)
    planning.add_argument("--seed", type=int, default=1)
    planning.add_argument("--manifest", type=Path, required=True)
    planning.set_defaults(function=plan)
    for name, function in (("prepare", prepare), ("train", train), ("evaluate", evaluate), ("summarize", summarize)):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name != "summarize":
            command.add_argument("--task-id", type=int, required=True)
        if name in ("train", "evaluate"):
            command.add_argument("--device", default="cuda")
        if name == "evaluate":
            command.add_argument("--test-assay", choices=ASSAYS, required=True)
        command.set_defaults(function=function)
    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
