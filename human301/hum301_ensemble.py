#!/usr/bin/env python3
"""Average existing seed predictions; choose the ensemble threshold on validation.

Reads the full validation_predictions.tsv and test_predictions.tsv from each
completed run. No model training, genome access, cache generation or resampling.
Rows are aligned by sample_id; class labels and inference provenance must match.
Equal weights are fixed before looking at test scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from hum301_motif64 import classification_metrics, select_classification_threshold


def read_predictions(path: Path) -> dict:
    ids, labels, probabilities, thresholds = [], [], [], []
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "true_label", "binding_probability", "classification_threshold"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing prediction columns in {path}")
        for row in reader:
            sample_id = row["sample_id"]
            if not sample_id or sample_id in seen:
                raise ValueError(f"Empty or duplicate sample_id in {path}: {sample_id!r}")
            seen.add(sample_id)
            if row["true_label"] not in ("0", "1"):
                raise ValueError(f"Invalid binary label in {path}")
            probability = float(row["binding_probability"])
            threshold = float(row["classification_threshold"])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid binding probability in {path}")
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid classification threshold in {path}")
            ids.append(sample_id)
            labels.append(int(row["true_label"]))
            probabilities.append(probability)
            thresholds.append(threshold)
    if not ids or set(labels) != {0, 1}:
        raise ValueError(f"Predictions need both classes in {path}")
    if len(set(thresholds)) != 1:
        raise ValueError(f"Multiple classification thresholds in {path}")
    return {
        "ids": ids,
        "labels": np.asarray(labels, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "threshold": thresholds[0],
    }


def read_configuration(run_dir: Path) -> dict:
    with (run_dir / "configuration.json").open(encoding="utf-8") as handle:
        configuration = json.load(handle)
    arguments = configuration["arguments"]
    signature = {
        "cache_sha256": configuration["cache_sha256"],
        "input_length": configuration["model"]["sequence_length"],
        "eval_crops": arguments["eval_crops"],
        "eval_rc": arguments["eval_rc"],
        "evaluation_undersampled": configuration["evaluation_balance"]["undersampled"],
    }
    if not signature["cache_sha256"] or signature["evaluation_undersampled"]:
        raise ValueError(f"Use completed runs with the full evaluation split: {run_dir}")
    return signature


def align_and_average(tables: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    reference = tables[0]
    ids = reference["ids"]
    labels = reference["labels"]
    reference_ids = set(ids)
    aligned = []
    for table in tables:
        if set(table["ids"]) != reference_ids:
            raise ValueError("Ensemble members have different sample_id sets")
        positions = {sample_id: i for i, sample_id in enumerate(table["ids"])}
        order = np.asarray([positions[sample_id] for sample_id in ids])
        if not np.array_equal(table["labels"][order], labels):
            raise ValueError("Ensemble members disagree on class labels")
        aligned.append(table["probabilities"][order])
    return ids, labels, np.mean(np.stack(aligned), axis=0)


def write_predictions(path: Path, ids: list[str], labels: np.ndarray,
                      probabilities: np.ndarray, threshold: float) -> None:
    predicted = (probabilities >= threshold).astype(np.int64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample_id", "true_label", "binding_probability", "predicted_binary_label",
                         "classification_threshold", "correct"])
        for sample_id, label, probability, prediction in zip(ids, labels, probabilities, predicted, strict=True):
            writer.writerow([sample_id, int(label), f"{probability:.17g}", int(prediction),
                             f"{threshold:.17g}", int(label == prediction)])


def run_ensemble(args: argparse.Namespace) -> int:
    run_dirs = [path.resolve() for path in args.run_dirs]
    if len(run_dirs) < 2 or len(set(run_dirs)) != len(run_dirs):
        raise ValueError("Provide at least two distinct completed run directories")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Ensemble output directory must be new or empty")
    signatures = [read_configuration(path) for path in run_dirs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Cache or inference settings differ between ensemble members")

    validation_tables = [read_predictions(path / "validation_predictions.tsv") for path in run_dirs]
    val_ids, val_labels, val_scores = align_and_average(validation_tables)
    threshold, threshold_score = select_classification_threshold(val_labels, val_scores, strategy="val-mcc")
    validation_metrics = classification_metrics(val_labels, val_scores, threshold)
    # Test files are read only after the validation-only threshold is frozen.
    test_tables = [read_predictions(path / "test_predictions.tsv") for path in run_dirs]
    test_ids, test_labels, test_scores = align_and_average(test_tables)
    if set(val_ids) & set(test_ids):
        raise ValueError("Validation and test sample_id sets overlap")
    test_metrics = classification_metrics(test_labels, test_scores, threshold)

    members = []
    for path, validation, test in zip(run_dirs, validation_tables, test_tables, strict=True):
        if validation["threshold"] != test["threshold"]:
            raise ValueError(f"Validation/test thresholds differ for {path}")
        members.append({
            "run_dir": str(path),
            "threshold": validation["threshold"],
            "validation_metrics": classification_metrics(validation["labels"], validation["probabilities"], validation["threshold"]),
            "test_metrics": classification_metrics(test["labels"], test["probabilities"], validation["threshold"]),
        })
    best_member = max(members, key=lambda member: member["validation_metrics"]["auprc"])
    payload = {
        "ensemble_method": "arithmetic_mean_of_binding_probabilities",
        "weights": [1 / len(members)] * len(members),
        "inference_provenance": signatures[0],
        "threshold_strategy": "val-mcc",
        "threshold": threshold,
        "validation_objective_score": threshold_score,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_sample_count": len(val_ids),
        "test_sample_count": len(test_ids),
        "members": members,
        "best_member_selected_on_validation": best_member["run_dir"],
        "mean_individual_test_auprc": float(np.mean([member["test_metrics"]["auprc"] for member in members])),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ensemble_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_predictions(args.output_dir / "validation_predictions.tsv", val_ids, val_labels, val_scores, threshold)
    write_predictions(args.output_dir / "test_predictions.tsv", test_ids, test_labels, test_scores, threshold)
    with (args.output_dir / "ensemble_comparison.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "validation_auprc", "test_auprc", "test_auroc", "test_mcc", "threshold"])
        for member in members:
            val, test = member["validation_metrics"], member["test_metrics"]
            writer.writerow([Path(member["run_dir"]).name, val["auprc"], test["auprc"], test["auroc"], test["mcc"], member["threshold"]])
        writer.writerow(["ensemble", validation_metrics["auprc"], test_metrics["auprc"], test_metrics["auroc"], test_metrics["mcc"], threshold])

    for member in members:
        print(f"{Path(member['run_dir']).name}: val AUPRC={member['validation_metrics']['auprc']:.4f}; "
              f"test AUPRC={member['test_metrics']['auprc']:.4f}")
    print(f"Ensemble: val AUPRC={validation_metrics['auprc']:.4f}; "
          f"test AUPRC={test_metrics['auprc']:.4f}; AUROC={test_metrics['auroc']:.4f}; MCC={test_metrics['mcc']:.4f}")
    print(f"Validation-selected threshold: {threshold:.8g}")
    print(f"Outputs written to {args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                        help="Completed training output directories with validation/test predictions.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    try:
        return run_ensemble(build_parser().parse_args())
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
