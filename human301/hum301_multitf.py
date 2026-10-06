#!/usr/bin/env python3
"""CPU data preparation, existing Human301 model runs, and a paired TF report.

The existing hum301_data.py, hum301_motif64.py, and hum301_motif64_local.py
remain unchanged. CREB3L3 uses completed job 68001, seeds 1 and 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data/projects/SFB_A03/jan/AT_TFBS")
TFS = ("TERF1", "SRY", "BATF2", "CTCF", "NFKB1", "MAX", "MGA", "MYF6", "ELF2", "ZBED9")
SEEDS = (1, 2)
VARIANTS = {
    "base": ("hum301_motif64.py", "center"),
    "local": ("hum301_motif64_local.py", "local-logsumexp"),
}
METRICS = ("auprc", "auroc", "mcc", "balanced_accuracy", "f1", "precision", "recall", "specificity")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


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
        raise ValueError("run-id must contain only letters, digits, underscores, or hyphens")
    root = args.project_root.resolve()
    code = root / "human301"
    inputs = root / "raw_data/clara/chip/student_handoff_30tf_increment/Input_data/CHS"
    genome = root / "raw_data/general_data/genome/hg38.fa"
    required = [code / "hum301_data.py", genome, Path(str(genome) + ".fai")]
    required += [code / script for script, _ in VARIANTS.values()]
    caches = {}
    for tf in TFS:
        required.extend(bed_paths(inputs, tf).values())
        legacy = root / "raw_data/human" / tf / "hum301_20261005_185818_tdUK1h/dataset.npz"
        caches[tf] = str(legacy if tf in ("TERF1", "SRY") and legacy.is_file() else
                         root / "raw_data/human" / tf / "hum301_multitf_chr5_chr7/dataset.npz")
    runs = []
    for tf in TFS:
        for variant in VARIANTS:
            for seed in SEEDS:
                runs.append({"tf": tf, "variant": variant, "seed": seed, "source": "new",
                             "cache": caches[tf],
                             "run_dir": str(code / "runs" / tf / f"{args.run_id}_{variant}_seed{seed}")})
    for variant in VARIANTS:
        old_variant = "base" if variant == "base" else "local_readout"
        for seed in SEEDS:
            run_dir = code / "runs/CREB3L3" / f"68001_ablation_{old_variant}_seed{seed}"
            required += [run_dir / name for name in
                         ("configuration.json", "validation_metrics.json", "binary_classification_metrics.json", "history.json")]
            runs.append({"tf": "CREB3L3", "variant": variant, "seed": seed,
                         "source": "existing_68001", "run_dir": str(run_dir)})
    require_files(required)
    if args.manifest.exists():
        raise ValueError(f"Workflow already exists: {args.manifest}")
    if any(Path(row["run_dir"]).exists() for row in runs if row["source"] == "new"):
        raise ValueError("A new result path already exists; use a new run-id")
    payload = {"format": "hum301-multitf-v1", "run_id": args.run_id,
               "script_dir": str(code), "input_root": str(inputs), "genome_fasta": str(genome),
               "validation_chromosomes": ["chr5", "chr7"], "tfs": list(TFS),
               "seeds": list(SEEDS), "caches": caches, "runs": runs}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Workflow: {args.manifest}")
    print("10 CPU data tasks; 40 GPU runs; 4 existing CREB3L3 runs included in evaluation.")


def load_manifest(path):
    manifest = read_json(path)
    if manifest.get("format") != "hum301-multitf-v1":
        raise ValueError("Unsupported workflow manifest")
    return manifest


def prepare(args):
    manifest = load_manifest(args.manifest)
    if not 0 <= args.task_id < len(manifest["tfs"]):
        raise ValueError("CPU task-id must be 0..9")
    tf = manifest["tfs"][args.task_id]
    cache_path = Path(manifest["caches"][tf])
    beds = bed_paths(manifest["input_root"], tf)
    genome = Path(manifest["genome_fasta"])
    sources = {**beds, "genome_fasta": genome, "genome_fai": Path(str(genome) + ".fai")}
    require_files(sources.values())
    if cache_path.is_file():
        # Reuse only an intact cache from these exact BED/reference contents.
        sys.path.insert(0, manifest["script_dir"])
        from hum301_data import load_cache
        cache = load_cache(cache_path)
        summary = cache["summary"]
        if summary.get("genome_build") != "hg38" or summary.get("stored_length") != 301:
            raise ValueError(f"{tf}: existing cache is not an hg38 Human301 cache")
        chromosomes = summary["splits"]["validation"]["chromosomes"]
        if set(chromosomes) != set(manifest["validation_chromosomes"]):
            raise ValueError(f"{tf}: existing validation chromosomes differ from chr5,chr7")
        for key, path in sources.items():
            if summary.get("sources", {}).get(key, {}).get("sha256") != sha256(path):
                raise ValueError(f"{tf}: existing cache does not match {path}; no files overwritten")
        print(f"{tf}: reusing {cache_path}")
        print(json.dumps(summary["splits"], indent=2))
        return
    command = [sys.executable, "-u", str(Path(manifest["script_dir"]) / "hum301_data.py"), "prepare",
               "--genome-fasta", str(genome), "--genome-build", "hg38",
               "--val-chromosomes", ",".join(manifest["validation_chromosomes"]),
               "--output", str(cache_path)]
    for key, path in beds.items():
        command += ["--" + key.replace("_", "-"), str(path)]
    print(f"{tf}: preparing {cache_path}", flush=True)
    subprocess.run(command, check=True)


def model_command(manifest, row):
    script, readout = VARIANTS[row["variant"]]
    return [sys.executable, "-u", str(Path(manifest["script_dir"]) / script), "train",
            "--cache", row["cache"], "--output-dir", row["run_dir"],
            "--input-length", "256", "--embed-channels", "64", "--encoder-channels", "80", "96", "128",
            "--heads", "4", "--ffn-channels", "256", "--classification-readout", readout,
            "--task-mode", "classification", "--checkpoint-objective", "classification-auprc",
            "--threshold-strategy", "val-mcc", "--negatives-per-positive", "10", "--batch-size", "88",
            "--hard-negative-fraction", "0", "--epochs", "100", "--patience", "20",
            "--learning-rate", "0.0003", "--min-learning-rate", "0.000001",
            "--weight-decay", "0.0001", "--dropout", "0.1", "--activation", "relu",
            "--reverse-complement-probability", "0.5", "--eval-crops", "multi", "--eval-rc",
            "--no-balanced-evaluation", "--eval-batch-size", "128", "--num-workers", "2",
            "--seed", str(row["seed"]), "--device", "cuda"]


def train(args):
    manifest = load_manifest(args.manifest)
    rows = [row for row in manifest["runs"] if row["source"] == "new"]
    if not 0 <= args.task_id < len(rows):
        raise ValueError("GPU task-id must be 0..39")
    row = rows[args.task_id]
    require_files([row["cache"], Path(manifest["script_dir"]) / VARIANTS[row["variant"]][0]])
    command = model_command(manifest, row)
    print(f"TF={row['tf']} variant={row['variant']} seed={row['seed']}", flush=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric: {name}")
    return result


def read_run(row):
    directory = Path(row["run_dir"])
    config = read_json(directory / "configuration.json")
    binary = read_json(directory / "binary_classification_metrics.json")
    val = read_json(directory / "validation_metrics.json")["classification"]
    test = binary["test_metrics"]
    history = read_json(directory / "history.json")
    arguments = config["arguments"]
    readout = VARIANTS[row["variant"]][1]
    expected = {"seed": row["seed"], "epochs": 100, "patience": 20, "batch_size": 88,
                "negatives_per_positive": 10, "hard_negative_fraction": 0.0,
                "eval_crops": "multi", "eval_rc": True, "classification_readout": readout,
                "dropout": 0.1, "weight_decay": 0.0001}
    if any(arguments.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected model/settings in {directory}")
    if config.get("evaluation_balance", {}).get("undersampled") is not False:
        raise ValueError(f"Evaluation is not the full held-out dataset: {directory}")
    if binary["threshold_strategy"] != "val-mcc":
        raise ValueError(f"Threshold was not chosen by validation MCC: {directory}")
    threshold = finite(binary["threshold"], "threshold")
    if not 0 <= threshold <= 1 or abs(finite(val["threshold"], "val threshold") - threshold) > 1e-10:
        raise ValueError(f"Validation/test thresholds differ: {directory}")
    if abs(finite(test["threshold"], "test threshold") - threshold) > 1e-10:
        raise ValueError(f"Test threshold differs: {directory}")
    if not history or int(binary["checkpoint_epoch"]) not in [item["epoch"] for item in history]:
        raise ValueError(f"Missing selected checkpoint in training history: {directory}")
    output = {key: row[key] for key in ("tf", "variant", "seed", "source", "run_dir")}
    output.update(checkpoint_epoch=int(binary["checkpoint_epoch"]), epochs_run=len(history), threshold=threshold,
                  training_minutes=sum(finite(item["elapsed_seconds"], "elapsed_seconds") for item in history) / 60,
                  parameter_count=int(config["parameter_count"]), cache_sha256=config["cache_sha256"])
    for split, metrics in (("val", val), ("test", test)):
        output.update({f"{split}_{name}": finite(metrics[name], name) for name in METRICS})
        for name in ("true_positive", "false_positive", "true_negative", "false_negative"):
            count = finite(metrics[name], name)
            if count < 0 or count != int(count):
                raise ValueError(f"Invalid confusion-matrix count in {directory}")
            output[f"{split}_{name}"] = int(count)
        output[f"{split}_positive"] = output[f"{split}_true_positive"] + output[f"{split}_false_negative"]
        output[f"{split}_negative"] = output[f"{split}_true_negative"] + output[f"{split}_false_positive"]
        if min(output[f"{split}_positive"], output[f"{split}_negative"]) <= 0:
            raise ValueError(f"Evaluation needs both classes: {directory}")
    signature = (config["cache_sha256"], json.dumps(config["splits"], sort_keys=True),
                 output["val_positive"], output["val_negative"], output["test_positive"], output["test_negative"])
    return output, signature


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    manifest = load_manifest(args.manifest)
    runs, signatures = [], {}
    errors = []
    for row in manifest["runs"]:
        try:
            result, signature = read_run(row)
            if row["tf"] in signatures and signatures[row["tf"]] != signature:
                raise ValueError(f"{row['tf']}: models/seeds were evaluated on different data")
            signatures[row["tf"]] = signature
            runs.append(result)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            errors.append(f"{row['run_dir']}: {exc}")
    if errors:
        raise ValueError("Incomplete/incompatible results; no partial summary written:\n" + "\n".join(errors))
    groups, comparisons = [], []
    for tf in [*manifest["tfs"], "CREB3L3"]:
        pair = {}
        for variant in VARIANTS:
            members = [row for row in runs if row["tf"] == tf and row["variant"] == variant]
            if sorted(row["seed"] for row in members) != list(SEEDS):
                raise ValueError(f"{tf}/{variant}: expected exactly seeds 1,2")
            group = {"tf": tf, "variant": variant, "n_seeds": len(members),
                     "test_positive": members[0]["test_positive"], "test_negative": members[0]["test_negative"]}
            group["random_ap_baseline"] = group["test_positive"] / (group["test_positive"] + group["test_negative"])
            for split in ("val", "test"):
                for name in METRICS:
                    values = [row[f"{split}_{name}"] for row in members]
                    group[f"{split}_{name}_mean"] = statistics.mean(values)
                    group[f"{split}_{name}_sd"] = statistics.stdev(values)
            group["checkpoint_epochs"] = ",".join(str(row["checkpoint_epoch"]) for row in sorted(members, key=lambda row: row["seed"]))
            group["training_minutes_mean"] = statistics.mean(row["training_minutes"] for row in members)
            groups.append(group)
            pair[variant] = group
        comparison = {"tf": tf, "base_val_auprc": pair["base"]["val_auprc_mean"],
                      "local_val_auprc": pair["local"]["val_auprc_mean"],
                      "delta_val_auprc_local_minus_base": pair["local"]["val_auprc_mean"] - pair["base"]["val_auprc_mean"],
                      "base_test_auprc": pair["base"]["test_auprc_mean"],
                      "local_test_auprc": pair["local"]["test_auprc_mean"],
                      "delta_test_auprc_local_minus_base": pair["local"]["test_auprc_mean"] - pair["base"]["test_auprc_mean"],
                      "delta_test_mcc_local_minus_base": pair["local"]["test_mcc_mean"] - pair["base"]["test_mcc_mean"]}
        for seed in SEEDS:
            values = {row["variant"]: row["test_auprc"] for row in runs if row["tf"] == tf and row["seed"] == seed}
            comparison[f"delta_test_auprc_seed{seed}"] = values["local"] - values["base"]
        comparison["validation_preference"] = ("local" if comparison["delta_val_auprc_local_minus_base"] > 0 else
                                               "base" if comparison["delta_val_auprc_local_minus_base"] < 0 else "tie")
        comparisons.append(comparison)
    macro = {variant: {f"{split}_{name}": statistics.mean(group[f"{split}_{name}_mean"] for group in groups if group["variant"] == variant)
                       for split in ("val", "test") for name in METRICS} for variant in VARIANTS}
    lines = ["# Human301: Basis gegen lokale Auswertung", "",
             f"Run-ID: `{manifest['run_id']}`. 44 Einzelruns, 11 TFs, jeweils Seeds 1 und 2.",
             "CREB3L3 stammt aus Job 68001; die anderen zehn TFs aus diesem Workflow.", "",
             "Beide Modelle: 9/12-bp-Filter, originale fünf Bottleneck-Blöcke und drei Decoder; "
             "1:10, Batch 88, maximal 100 Epochen, Patience 20, kein Mining.",
             "Validierung: chr5/chr7 aus Train; vollständige feste Testdaten. Schwellen: Validierungs-MCC.",
             "AP ist Average Precision. ± zeigt die Standardabweichung zwischen den zwei Seeds, kein Konfidenzintervall.", "",
             "| TF | Variante | Val-AP | Test-AP ± SD | Test-AUROC | Test-MCC | Positive / Negative im Test |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        lines.append(f"| {group['tf']} | {group['variant']} | {group['val_auprc_mean']:.4f} | "
                     f"{group['test_auprc_mean']:.4f} ± {group['test_auprc_sd']:.4f} | "
                     f"{group['test_auroc_mean']:.4f} | {group['test_mcc_mean']:.4f} | "
                     f"{group['test_positive']} / {group['test_negative']} |")
    lines += ["", "| TF | Δ Test-AP: lokal − Basis | Δ Test-MCC | Präferenz anhand Val-AP |",
              "|---|---:|---:|---|"]
    for row in comparisons:
        lines.append(f"| {row['tf']} | {row['delta_test_auprc_local_minus_base']:+.4f} | "
                     f"{row['delta_test_mcc_local_minus_base']:+.4f} | {row['validation_preference']} |")
    lines += ["", "Über alle elf TFs, jeder TF gleich gewichtet:", ""]
    for variant in VARIANTS:
        lines.append(f"- {variant}: Val-AP {macro[variant]['val_auprc']:.4f}; "
                     f"Test-AP {macro[variant]['test_auprc']:.4f}; Test-MCC {macro[variant]['test_mcc']:.4f}.")
    lines += ["", "Diese Mittelwerte sind keine gemeinsam neu berechnete AP über gemischte TF-Datensätze.",
              "Training_minutes umfasst die protokollierten Trainings- und Validierungsepochen, ohne abschließende Testauswertung.",
              "Die Auswertung sammelt vorhandene Metriken; sie trainiert, resampelt und optimiert keine Schwellen."]
    output = args.output_dir or args.manifest.parent / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Summary output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output / "run_results.tsv", runs)
    write_tsv(output / "summary_by_tf.tsv", groups)
    write_tsv(output / "comparison_by_tf.tsv", comparisons)
    (output / "summary.json").write_text(json.dumps({"run_id": manifest["run_id"], "runs": runs, "groups": groups,
                                                     "comparisons": comparisons, "macro_equal_tf_weight": macro}, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Evaluation written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    planning.add_argument("--run-id", required=True)
    planning.add_argument("--manifest", type=Path, required=True)
    planning.set_defaults(function=plan)
    for name, function in (("prepare", prepare), ("train", train), ("summarize", summarize)):
        child = sub.add_parser(name)
        child.add_argument("--manifest", type=Path, required=True)
        if name in ("prepare", "train"):
            child.add_argument("--task-id", type=int, required=True)
        else:
            child.add_argument("--output-dir", type=Path)
        child.set_defaults(function=function)
    args = parser.parse_args()
    try:
        args.function(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
