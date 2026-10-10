#!/usr/bin/env bash
#SBATCH --job-name=hum301-kernel-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/human301
# Alle drei Dateien neben hum301_data.py und hum301_crossassay.py ablegen.
# Start: bash hum301_motif_kernel_submit.sh
# Der Launcher legt den Laufordner an und reicht diese Datei fuer die
# CPU-Datenpruefung, das GPU-Array und die CPU-Zusammenfassung getrennt ein.
# Logs und Ergebnisse liegen gemeinsam unter runs/motif_kernel/<RUN_ID>/.

set -euo pipefail
: "${HUM301_KERNEL_STAGE:?Bitte mit bash hum301_motif_kernel_submit.sh starten}"
if [[ "$HUM301_KERNEL_STAGE" == "plan" ]]; then
    PYTHON_BIN=python3
else
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate at_env
    PYTHON_BIN=python
fi
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -u - <<'PY'
import argparse
import gc
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("/data/projects/SFB_A03/jan/AT_TFBS")
CODE = PROJECT / "human301"
SELF = CODE / "hum301_motif_kernel.sbatch"
LAUNCHER = CODE / "hum301_motif_kernel_submit.sh"
MODEL_SCRIPT = "hum301_motif_kernel.py"
SEED = 1
ASSAYS = ("CHIP", "GHTS")

# Ein einzelner Eingangskernel pro TF. Werte = PWM-Laengen aus der vorherigen
# Tabelle, einschliesslich Flanken; keine minimalen biologischen Bindungskerne.
# Zum manuellen Aendern der Groessen nur diese Tabelle bearbeiten.
KERNELS = {
    "BATF2": 13, "CREB3L3": 14, "CTCF": 20, "ELF3": 13, "FLI1": 12,
    "FOSL2": 12, "GABPA": 12, "GLI4": 23, "LEF1": 12, "LEUTX": 8,
    "MAX": 11, "MGA": 19, "MYF6": 14, "NFKB1": 12, "NR1H4": 20,
    "PAX7": 12, "RFX5": 19, "RORB": 17, "SOX2": 12, "SRY": 12,
    "TERF1": 8, "USF3": 12, "VDR": 16, "YY1": 12, "ZBED9": 20,
    "ZBTB8A": 20, "ZFP3": 23, "ZNF696": 23, "ZNF772": 21, "ZNF773": 21,
}

# Die 19 neuen TFs und die vier TFs aus den letzten Fuenf-Seed-Laeufen nutzen
# weiterhin ihre ArchThalia-ML-Caches. Die uebrigen sieben nutzen ihre
# bisherigen Handoff-Caches, damit der Vergleich dieselben Daten verwendet.
LEGACY_TFS = {"SRY", "BATF2", "NFKB1", "MGA", "MYF6", "ELF3", "ZBED9"}

sys.path.insert(0, str(CODE))
import hum301_crossassay as workflow

# Nur die Modellwahl in diesem Prozess anpassen; die vorhandene Datei bleibt
# unveraendert. Trainingsargumente und Test-Evaluation werden wiederverwendet.
workflow.MODEL_SCRIPT = MODEL_SCRIPT
STAGE = os.environ["HUM301_KERNEL_STAGE"]


def cache_paths(tf):
    human = PROJECT / "raw_data/human"
    if tf not in LEGACY_TFS:
        root = human / "ArchThalia-ML" / tf
        return {assay: root / assay / "dataset.npz" for assay in ASSAYS}
    root = human / tf
    chip_directory = ("hum301_20261005_185818_tdUK1h" if tf == "SRY"
                      else "hum301_multitf_chr5_chr7")
    return {"CHIP": root / chip_directory / "dataset.npz",
            "GHTS": root / "GHTS/dataset.npz"}


def plan():
    # Nur kleine Metadaten auf dem Login-Node; die Caches werden erst im
    # CPU-Job geladen und geprueft. Kein Kopieren von Daten oder Python-Code.
    workflow.require_files([CODE / MODEL_SCRIPT, CODE / "hum301_data.py",
                            CODE / "hum301_crossassay.py", SELF, LAUNCHER])
    run_directory = Path(os.environ["HUM301_KERNEL_RUN_DIR"])
    run_id = run_directory.name
    manifest_path = run_directory / "workflow.json"
    runs = [{"tf": tf, "train_assay": assay, "seed": SEED,
             "motif_kernel": kernel,
             "run_dir": str(run_directory / "models" / tf / f"{assay}_seed{SEED}_k{kernel}")}
            for tf, kernel in KERNELS.items() for assay in ASSAYS]
    if run_directory.exists():
        raise ValueError("Ergebnisordner existiert bereits")
    workflow.require_files([path for tf in KERNELS for path in cache_paths(tf).values()])
    run_directory.mkdir(parents=True, exist_ok=False)
    for relative in ("datasets", "evaluation", "logs/data", "logs/train", "logs/summary"):
        (run_directory / relative).mkdir(parents=True)
    for row in runs:
        Path(row["run_dir"]).mkdir(parents=True)
    manifest = {"format": workflow.FORMAT, "run_id": run_id, "script_dir": str(CODE),
                "data_root": str(PROJECT / "raw_data/human"), "model_script": MODEL_SCRIPT,
                "data_script": "hum301_data.py", "seed": SEED, "tfs": list(KERNELS),
                "runs": runs, "motif_kernel_by_tf": KERNELS,
                "input_caches": {tf: {assay: str(path) for assay, path in cache_paths(tf).items()}
                                 for tf in KERNELS},
                "kernel_basis": "published PWM widths, including flanks; see motif table",
                "script_sha256": {name: workflow.sha256(CODE / name) for name in
                                  (MODEL_SCRIPT, "hum301_data.py", "hum301_crossassay.py",
                                   SELF.name, LAUNCHER.name)}}
    workflow.write_json(manifest_path, manifest)
    print(f"Laufordner: {run_directory}")
    print(f"{len(KERNELS)} TFs; {len(runs)} Modelle; {2 * len(runs)} Testauswertungen; Seed {SEED}")


def prepare_existing(manifest_path):
    import hum301_data as data
    manifest = workflow.load_manifest(manifest_path)
    workflow.check_scripts(manifest)
    workflow.require_files([path for assays in manifest["input_caches"].values()
                            for path in assays.values()])
    for tf in manifest["tfs"]:
        datasets = {}
        for assay, path in manifest["input_caches"][tf].items():
            cache = data.load_cache(path)
            summary = cache["summary"]
            if cache["sequences"].shape[1] != 301 or summary.get("genome_build") != "hg38":
                raise ValueError(f"Kein passender Human301-Cache: {path}")
            datasets[assay] = {"cache": str(path), "cache_sha256": workflow.sha256(path),
                               "summary": summary}
            kernel = manifest["motif_kernel_by_tf"][tf]
            print(f"{tf}/{assay}: vorhandener Cache {path}; Kernel={kernel} bp", flush=True)
            del cache
            gc.collect()
        workflow.check_cross_splits(datasets)
        workflow.write_json(workflow.data_state_path(manifest_path, tf), {"tf": tf, "datasets": datasets})
    print(f"CPU-Pruefung abgeschlossen: {2 * len(manifest['tfs'])} vorhandene Caches.")


def train_and_test(manifest_path):
    manifest = workflow.load_manifest(manifest_path)
    workflow.check_scripts(manifest)
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    row = workflow.task_row(manifest, task_id)
    datasets = workflow.read_json(workflow.data_state_path(manifest_path, row["tf"]))["datasets"]
    workflow.check_cross_splits(datasets)
    for state in datasets.values():
        if workflow.sha256(state["cache"]) != state["cache_sha256"]:
            raise ValueError("Ein Cache wurde nach der CPU-Pruefung geaendert")
    directory = Path(row["run_dir"])
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Ergebnisordner ist nicht leer: {directory}")
    source = row["train_assay"]
    command = workflow.training_command(manifest, row, datasets[source]["cache"], "cuda")
    command += ["--motif-kernel", str(row["motif_kernel"])]
    print(f"TF={row['tf']} train={source} seed={row['seed']} kernel={row['motif_kernel']} bp", flush=True)
    print(shlex.join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    training_seconds = time.perf_counter() - started
    workflow.require_files([directory / name for name in
                            ("model_for_prediction.pt", "validation_metrics.json", "history.json")])
    workflow.write_json(directory / "training_timing.json", {
        "training_wall_seconds": training_seconds,
        "scope": "training process including training/validation and final source validation; tests excluded"})
    # Ein trainiertes Modell, zwei Tests; Schwelle nur aus der Source-Validierung.
    for target in (source, next(assay for assay in ASSAYS if assay != source)):
        workflow.evaluate(argparse.Namespace(manifest=manifest_path, task_id=task_id,
                                             test_assay=target, device="cuda"))
    print(f"Fertig: {row['tf']}/{source}, Kernel {row['motif_kernel']} bp", flush=True)


def summarize(manifest_path):
    manifest = workflow.load_manifest(manifest_path)
    workflow.check_scripts(manifest)
    rows, training_rows = [], []
    expected_arguments = {"classification_readout": "local-logsumexp", "seed": SEED,
        "negatives_per_positive": 10, "batch_size": 88, "hard_negative_fraction": 0.0,
        "epochs": 100, "patience": 20, "eval_crops": "multi", "eval_rc": True,
        "skip_test": True, "task_mode": "classification", "balanced_evaluation": False}
    test_signatures = {}
    for run in manifest["runs"]:
        directory = Path(run["run_dir"])
        config = workflow.read_json(directory / "configuration.json")
        history = workflow.read_json(directory / "history.json")
        val = workflow.read_json(directory / "validation_metrics.json")["classification"]
        timing = workflow.read_json(directory / "training_timing.json")
        expected = {**expected_arguments, "motif_kernel": run["motif_kernel"]}
        if any(config["arguments"].get(key) != value for key, value in expected.items()):
            raise ValueError(f"Abweichende Trainingsparameter: {directory}")
        if config["model"]["motif_kernel"] != run["motif_kernel"] or not history:
            raise ValueError(f"Kernel oder History unpassend: {directory}")
        state = workflow.read_json(workflow.data_state_path(manifest_path, run["tf"]))["datasets"]
        if config["cache_sha256"] != state[run["train_assay"]]["cache_sha256"]:
            raise ValueError(f"Trainings-Cache unpassend: {directory}")
        training = {"tf": run["tf"], "train_assay": run["train_assay"], "seed": run["seed"],
            "motif_kernel_bp": run["motif_kernel"], "parameter_count": int(config["parameter_count"]),
            "epochs_run": len(history),
            "epoch_training_validation_minutes": sum(item["elapsed_seconds"] for item in history) / 60,
            "training_wall_minutes": timing["training_wall_seconds"] / 60,
            "validation_auprc_selected": val["auprc"],
            "validation_auprc_last_epoch": history[-1]["val"]["classification"]["auprc"],
            "run_dir": str(directory)}
        training_rows.append(training)
        for target in ASSAYS:
            direction = f"{run['train_assay']}_to_{target}"
            path = directory / direction / "binary_classification_metrics.json"
            result = workflow.read_json(path)
            test = state[target]["summary"]["splits"]["test"]
            signature = (result["test_cache_sha256"], result["test_positive"], result["test_negative"])
            expected_signature = (state[target]["cache_sha256"], test["positive"], test["negative"])
            key = (run["tf"], target)
            if signature != expected_signature or key in test_signatures and test_signatures[key] != signature:
                raise ValueError(f"Andere oder unvollstaendige Testdaten: {path}")
            test_signatures[key] = signature
            if (result["tf"] != run["tf"] or result["seed"] != SEED or result["direction"] != direction
                    or result["training_cache_sha256"] != state[run["train_assay"]]["cache_sha256"]
                    or abs(result["threshold"] - val["threshold"]) > 1e-10
                    or result["threshold_strategy"] != "val-mcc"):
                raise ValueError(f"Modell oder Source-Schwelle unpassend: {path}")
            metrics = result["test_metrics"]
            rows.append({**training, "test_assay": target, "direction": direction,
                **{name: workflow.finite(metrics[name]) for name in workflow.METRICS},
                "threshold": result["threshold"], "checkpoint_epoch": result["checkpoint_epoch"],
                "test_positive": signature[1], "test_negative": signature[2],
                "pr_baseline": signature[1] / (signature[1] + signature[2]),
                **{name: int(metrics[name]) for name in
                   ("true_positive", "false_positive", "true_negative", "false_negative")},
                "inference_minutes": result["inference_seconds"] / 60,
                "evaluation_wall_minutes": result["evaluation_wall_seconds"] / 60,
                "metrics_file": str(path)})
    if len(training_rows) != 60 or len(rows) != 120:
        raise ValueError("Erwartet: 60 Modelle und 120 Testauswertungen")
    directions = tuple(f"{source}_to_{target}" for source in ASSAYS for target in ASSAYS)
    matrix = []
    for tf, kernel in manifest["motif_kernel_by_tf"].items():
        by_direction = {row["direction"]: row for row in rows if row["tf"] == tf}
        if set(by_direction) != set(directions):
            raise ValueError(f"Fehlende Richtungen fuer {tf}")
        matrix.append({"tf": tf, "motif_kernel_bp": kernel,
                       **{direction: by_direction[direction]["auprc"] for direction in directions}})
    output = manifest_path.parent / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Auswertung existiert bereits: {output}")
    output.mkdir(parents=True, exist_ok=True)
    workflow.write_tsv(output / "results.tsv", rows)
    workflow.write_tsv(output / "auprc_by_tf.tsv", matrix)
    workflow.write_tsv(output / "training_runtime.tsv", training_rows)
    workflow.write_json(output / "summary.json", {
        "run_id": manifest["run_id"], "seed": SEED, "model": MODEL_SCRIPT,
        "n_models": len(training_rows), "n_test_evaluations": len(rows),
        "motif_kernel_by_tf": manifest["motif_kernel_by_tf"], "auprc_by_tf": matrix,
        "macro_equal_tf_weight": {d: sum(row[d] for row in matrix) / len(matrix) for d in directions},
        "total_training_wall_hours": sum(row["training_wall_minutes"] for row in training_rows) / 60,
        "total_epoch_training_validation_hours": sum(row["epoch_training_validation_minutes"] for row in training_rows) / 60,
        "total_test_evaluation_wall_hours": sum(row["evaluation_wall_minutes"] for row in rows) / 60,
        "runtime_note": "Summed process times, not array elapsed time; training counted once per model; CPU preparation and queue excluded.",
        "training": training_rows, "evaluations": rows})
    print(f"Auswertung: {output}")
    print("TF\tKernel_bp\t" + "\t".join(directions))
    for row in matrix:
        print(row["tf"] + "\t" + str(row["motif_kernel_bp"]) + "\t" +
              "\t".join(f"{row[d]:.4f}" for d in directions))


if STAGE == "plan":
    plan()
elif STAGE in ("data", "train", "summary"):
    manifest_path = Path(os.environ["HUM301_KERNEL_MANIFEST"])
    {"data": prepare_existing, "train": train_and_test, "summary": summarize}[STAGE](manifest_path)
else:
    raise ValueError(f"Unbekannte Stage: {STAGE}")
PY
