#!/usr/bin/env bash
#SBATCH --job-name=m33_rshift_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jbossenz/AT_TFBS/m33_rshift/slurm/%x-%j.out
#SBATCH --error=/home/jbossenz/AT_TFBS/m33_rshift/slurm/%x-%j.err

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/software/conda/envs/relion-5.0/bin/python}"
PROJECT_DIR=/home/jbossenz/AT_TFBS/m33_rshift
SLURM_DIR="$PROJECT_DIR/slurm"
SELF="$SLURM_DIR/m33_rshift_all.sbatch"
DATA_PARENT="$PROJECT_DIR/m33_rshift_data"
RAW=/home/jbossenz/AT_TFBS/raw_data
GENERAL="$RAW/general_data"
DATA_SCRIPT="$PROJECT_DIR/data_m33_rshift.py"

# Reihenfolge = Array-Index in beiden Stufen.
TF_NAMES=(
    AP1_WILDTYPE_GUO2025
    MYC3_air_jablock_2h_chipseq_Yin2023
    MYC2_ja_2h_chipseq_Yin2023
)
# Variante -> Skriptdatei. base = m33_rshift.py (alle drei Schalter auf Basis).
VARIANT_NAMES=(
    base
    relpos
    res4bp
    attnpool
    relpos_res4bp
    relpos_attnpool
    res4bp_attnpool
    relpos_res4bp_attnpool
)
VARIANT_SCRIPTS=(
    "$PROJECT_DIR/m33_rshift.py"
    "$PROJECT_DIR/model_3_3_relpos.py"
    "$PROJECT_DIR/model_3_3_res4bp.py"
    "$PROJECT_DIR/model_3_3_attnpool.py"
    "$PROJECT_DIR/model_3_3_relpos_res4bp.py"
    "$PROJECT_DIR/model_3_3_relpos_attnpool.py"
    "$PROJECT_DIR/model_3_3_res4bp_attnpool.py"
    "$PROJECT_DIR/model_3_3_relpos_res4bp_attnpool.py"
)

# Datenerzeugung
CONTEXT_PER_SIDE=196
DATA_SEED=42
# Training (fuer alle Varianten identisch)
SEED=1
EPOCHS=50
PATIENCE=8
BATCH_SIZE=8
TRAIN_SHIFT_MAX=128
TRAIN_SHIFT_DISTRIBUTION=uniform
EVAL_SHIFT=fixed
CV_CONCURRENCY=4            # gleichzeitig laufende GPU-Tasks (%N im Array)
STAGE_LOGS=(--output="$SLURM_DIR/%x-%A_%a.out" --error="$SLURM_DIR/%x-%A_%a.err")
DATA_RESOURCES=(--cpus-per-task=4 --mem=32G --time=06:00:00 "${STAGE_LOGS[@]}")
CV_RESOURCES=(--cpus-per-task=4 --mem=32G --time=24:00:00 --gres=gpu:1 "${STAGE_LOGS[@]}")

unset PYTHONHOME PYTHONPATH || true
export MPLBACKEND=Agg OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

check_files() {
    local missing=0 path
    for path in "$PYTHON_BIN" "$DATA_SCRIPT" "$PROJECT_DIR/gc_matching.py" "${VARIANT_SCRIPTS[@]}"; do
        [[ -e "$path" ]] || { echo "Fehlt: $path" >&2; missing=1; }
    done
    [[ "$missing" == 0 ]]
}

# ===========================================================================
# Launcher: sbatch m33_rshift_all.sbatch [all|data-only|cv-only]
# Erkennung ueber STAGE (leer = Launcher); laeuft als kleiner Slurm-Job oder
# direkt per bash.
# ===========================================================================
if [[ -z "${STAGE:-}" ]]; then
    MODE="${1:-all}"
    command -v sbatch >/dev/null || { echo "sbatch nicht gefunden." >&2; exit 1; }
    [[ -f "$SELF" ]] || { echo "Diese Datei muss unter $SELF liegen." >&2; exit 1; }
    check_files || exit 1
    mkdir -p "$SLURM_DIR"
    grep -H "^SCRIPT_VERSION\|^MATCHER_VERSION\|^ARCH_VARIANT_NAME" \
        "$DATA_SCRIPT" "$PROJECT_DIR/gc_matching.py" "${VARIANT_SCRIPTS[@]}" || true

    submit_data() {
        sbatch --parsable --export=ALL,STAGE=data --job-name=m33_rshift_data \
            --array=0-2 "${DATA_RESOURCES[@]}" "$SELF"
    }
    submit_cv() {
        local dependency=()
        [[ -n "${1:-}" ]] && dependency=(--dependency=afterok:"$1")
        sbatch --parsable --export=ALL,STAGE=cv --job-name=m33_rshift_cv \
            --array="0-119%$CV_CONCURRENCY" "${CV_RESOURCES[@]}" "${dependency[@]}" "$SELF"
    }
    case "$MODE" in
        all)
            DATA_JOB=$(submit_data); echo "Datensatz-Job: $DATA_JOB"
            CV_JOB=$(submit_cv "$DATA_JOB"); echo "CV-Job (wartet auf $DATA_JOB): $CV_JOB"
            ;;
        data-only)
            DATA_JOB=$(submit_data); echo "Datensatz-Job: $DATA_JOB"
            ;;
        cv-only)
            for tf in "${TF_NAMES[@]}"; do
                [[ -f "$DATA_PARENT/$tf/.complete" ]] || { echo "Datensatz nicht fertig: $tf" >&2; exit 1; }
            done
            CV_JOB=$(submit_cv ""); echo "CV-Job: $CV_JOB"
            ;;
        *)
            echo "Unbekannter Modus: $MODE (all | data-only | cv-only)" >&2; exit 1
            ;;
    esac
    exit 0
fi

# ===========================================================================
# Stufe "data": ein Array-Task je TF (CPU). Generator 7.1: 1024-bp-Fenster plus
# 196 Basen Umfeld je Seite, Positive auf dem Summit, GC-gematchte Negative.
# Fertige TFs werden uebersprungen; FORCE_REGENERATE=1 erzwingt die Neuerzeugung.
# ===========================================================================
stage_data() {
    local TASK_ID="${SLURM_ARRAY_TASK_ID:?Nur als Slurm-Array starten}"
    local TF="${TF_NAMES[$TASK_ID]}"
    local PREFIX="$RAW/$TF/signal/$TF.pooled_replicates.macs3"
    local DATASET_ROOT="$DATA_PARENT/$TF"
    local FE_BW="$PREFIX.FE.bw"
    local PVAL_BW="$PREFIX.pval.bw"
    local SUMMIT_BW="$PREFIX.pval.called_summits.bw"
    local TSS_PLUS="$GENERAL/AT_TSS/clustered_TSS_Le_2022_sup2_wt_plus.bed"
    local TSS_MINUS="$GENERAL/AT_TSS/clustered_TSS_Le_2022_sup2_wt_minus.bed"
    local GFF="$GENERAL/Arabidopsis_thaliana.TAIR10.63.gff3.gz"
    local FASTA="$GENERAL/genome/TAIR10.1.atlas.fa"
    local FAI="$GENERAL/genome/TAIR10.1.atlas.fa.fai"
    local path

    for path in "$PYTHON_BIN" "$DATA_SCRIPT" "$PROJECT_DIR/gc_matching.py" \
                "$FE_BW" "$PVAL_BW" "$SUMMIT_BW" "$TSS_PLUS" "$TSS_MINUS" \
                "$GFF" "$FASTA" "$FAI"; do
        [[ -e "$path" ]] || { echo "Fehlt: $path" >&2; exit 1; }
    done

    "$PYTHON_BIN" - <<'PY'
import pyBigWig, pysam, numpy, matplotlib  # noqa: F401  (Abhaengigkeiten des Generators)
print("Generator-Abhaengigkeiten vorhanden")
PY

    if [[ -f "$DATASET_ROOT/.complete" && "${FORCE_REGENERATE:-0}" != "1" ]]; then
        echo "Datensatz fuer $TF ist bereits fertig: $DATASET_ROOT"
        return 0
    fi

    mkdir -p "$DATA_PARENT"
    rm -f "$DATASET_ROOT/.complete"

    # Die Kandidatenpools des Matchers (mehrere GB SQLite) landen sonst in /tmp
    # des Knotens; TMPDIR leitet sie auf das Projektlaufwerk um.
    export TMPDIR="$DATA_PARENT/.tmp_$TF"
    mkdir -p "$TMPDIR"
    trap 'rm -rf "$TMPDIR"' EXIT

    echo "TF $TF | Ausgabe $DATASET_ROOT | Umfeld $CONTEXT_PER_SIDE bp je Seite"
    "$PYTHON_BIN" "$DATA_SCRIPT" \
        --gene-annotation "$GFF" \
        --fe-bigwig "$FE_BW" \
        --pval-bigwig "$PVAL_BW" \
        --summit-bigwig "$SUMMIT_BW" \
        --tss-plus "$TSS_PLUS" \
        --tss-minus "$TSS_MINUS" \
        --fasta "$FASTA" \
        --fai "$FAI" \
        --output-dir "$DATASET_ROOT" \
        --context-per-side "$CONTEXT_PER_SIDE" \
        --seed "$DATA_SEED" \
        --overwrite \
        2>&1 | tee "$DATA_PARENT/generation_$TF.log"

    # Plausibilitaet: die sechs Gruppen-Tabellen vorhanden, Umfeld wie erwartet.
    "$PYTHON_BIN" - "$DATASET_ROOT" "$CONTEXT_PER_SIDE" <<'PY'
import csv, sys
from pathlib import Path
root, context = Path(sys.argv[1]), int(sys.argv[2])
files = [root / f"metadata_{label}_{pool}.tsv"
         for label in ("positive", "negative") for pool in ("tss", "genic", "noncoding")]
missing = [str(p) for p in files if not p.is_file()]
if missing:
    raise SystemExit(f"Metadaten-Tabellen fehlen: {missing}")
total = 0
for path in files:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit(f"Leere Tabelle: {path}")
    bad = [r for r in rows if int(r["context_per_side"]) != context]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} Fenster mit falschem Umfeld")
    total += len(rows)
    print(f"{path.name}: {len(rows):,} Fenster")
print(f"Gesamt {total:,} Fenster mit Umfeld {context} bp je Seite")
PY

    touch "$DATASET_ROOT/.complete"
    echo "Datensatz fertig: $DATASET_ROOT"
}

# ===========================================================================
# Stufe "cv": 8 Varianten x 3 TFs x 5 Testchromosomen (GPU).
# Array-ID = Variante * 15 + TF * 5 + (Testchromosom - 1); die fuenf Folds einer
# Kombination liegen hintereinander. Der Cache wird je TF einmal gebaut und von
# allen Varianten geteilt (identischer Datenpfad in allen acht Skripten).
# ===========================================================================
stage_cv() {
    local TASK_ID="${SLURM_ARRAY_TASK_ID:?Nur als Slurm-Array starten}"
    local VARIANT_INDEX=$((TASK_ID / 15))
    local REMAINDER=$((TASK_ID % 15))
    local TF="${TF_NAMES[$((REMAINDER / 5))]}"
    local TEST_CHR=$((REMAINDER % 5 + 1))
    local VARIANT="${VARIANT_NAMES[$VARIANT_INDEX]}"
    local TRAIN_SCRIPT="${VARIANT_SCRIPTS[$VARIANT_INDEX]}"
    local TRAIN_CHR=() chr
    for chr in 1 2 3 4 5; do
        [[ "$chr" == "$TEST_CHR" ]] || TRAIN_CHR+=("$chr")
    done
    local TRAIN_CHR_CSV
    TRAIN_CHR_CSV=$(IFS=,; echo "${TRAIN_CHR[*]}")

    local DATASET_ROOT="$DATA_PARENT/$TF"
    local BASE_OUTPUT="$PROJECT_DIR/cv_rshift_seed_${SEED}_${SLURM_ARRAY_JOB_ID:?}"
    local TF_DIR="$BASE_OUTPUT/$TF"
    local COMBO_DIR="$TF_DIR/$VARIANT"
    local RUN_DIR="$COMBO_DIR/test_chr_$TEST_CHR"
    local CACHE_PATH="$TF_DIR/cache.npz"

    check_files || exit 1
    [[ -f "$DATASET_ROOT/.complete" ]] || {
        echo "Datensatz fuer $TF ist nicht fertig (kein .complete): $DATASET_ROOT" >&2
        exit 1
    }
    command -v flock >/dev/null || { echo "flock fehlt." >&2; exit 1; }
    mkdir -p "$RUN_DIR"
    rm -f "$RUN_DIR/.complete"
    echo "TF $TF | Variante $VARIANT ($TRAIN_SCRIPT) | Seed $SEED | Testchromosom $TEST_CHR | Training $TRAIN_CHR_CSV"
    echo "Ergebnisse: $RUN_DIR"

    "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch erkennt keine GPU")
print("GPU:", torch.cuda.get_device_name(0))
PY

    # Pro TF baut genau ein Array-Task den Cache; die uebrigen warten darauf.
    (
        flock -x 9
        if [[ ! -s "$CACHE_PATH" ]]; then
            "$PYTHON_BIN" "${VARIANT_SCRIPTS[0]}" inspect \
                --dataset-root "$DATASET_ROOT" --cache-path "$CACHE_PATH" \
                --train-chromosomes 1,2,3,4 --test-chromosomes 5
        fi
    ) 9>"$TF_DIR/.cache.lock"

    "$PYTHON_BIN" "$TRAIN_SCRIPT" train \
        --dataset-root "$DATASET_ROOT" --cache-path "$CACHE_PATH" \
        --output-dir "$RUN_DIR" \
        --train-chromosomes "$TRAIN_CHR_CSV" --test-chromosomes "$TEST_CHR" \
        --device cuda --batch-size "$BATCH_SIZE" --num-workers 0 \
        --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
        --task-mode classification --activation relu \
        --checkpoint-objective classification-auprc \
        --threshold-strategy val-mcc \
        --reverse-complement-probability 0.5 \
        --train-shift-max "$TRAIN_SHIFT_MAX" \
        --train-shift-distribution "$TRAIN_SHIFT_DISTRIBUTION" \
        --eval-shift "$EVAL_SHIFT" \
        --balanced-evaluation --no-amp \
        2>&1 | tee "$RUN_DIR/training.log"

    [[ -f "$RUN_DIR/binary_classification_metrics.json" ]] || {
        echo "Metrik-Datei fehlt: $RUN_DIR/binary_classification_metrics.json" >&2
        exit 1
    }
    touch "$RUN_DIR/.complete"

    # Der letzte erfolgreiche Fold jeder TF/Varianten-Kombination fasst sie zusammen.
    (
        flock -x 8
        for chr in 1 2 3 4 5; do
            if [[ ! -f "$COMBO_DIR/test_chr_$chr/.complete" ]]; then
                echo "Weitere Folds von $TF/$VARIANT laufen noch; Auswertung folgt."
                exit 0
            fi
        done

        "$PYTHON_BIN" - "$COMBO_DIR" "$TF" "$VARIANT" "$SEED" <<'PY'
import csv
import json
import math
import statistics
import sys
from pathlib import Path

base, tf, variant, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
metrics = ("auroc", "auprc", "balanced_accuracy", "f1", "mcc",
           "precision", "recall", "specificity")
numbers = ("checkpoint_epoch", "epochs_run", "training_minutes",
           "threshold", "validation_mcc", *metrics)
columns = ("tf", "variant", "seed", "train_chromosomes", "test_chromosome",
           "test_positives_dropped", *numbers)
rows = []

for chromosome in range(1, 6):
    run = base / f"test_chr_{chromosome}"
    payload = json.loads((run / "binary_classification_metrics.json").read_text())
    row = {
        "tf": tf,
        "variant": variant,
        "seed": seed,
        "train_chromosomes": ",".join(str(c) for c in range(1, 6) if c != chromosome),
        "test_chromosome": chromosome,
        "checkpoint_epoch": payload["checkpoint_epoch"],
        "threshold": payload["threshold"],
        "validation_mcc": payload["validation_objective_score"],
        "test_positives_dropped": 0,
    }
    row.update({name: payload["test_metrics"][name] for name in metrics})
    config_path = run / "configuration.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        balance = config.get("evaluation_balance", {}).get("test", {})
        row["test_positives_dropped"] = balance.get("positives_dropped_total", 0)
    history_path = run / "history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text())
        row["epochs_run"] = len(history)
        row["training_minutes"] = sum(
            float(epoch.get("elapsed_seconds", 0)) for epoch in history
        ) / 60
    rows.append(row)

with (base / "run_results.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for name in numbers:
    values = [float(row[name]) for row in rows
              if name in row and math.isfinite(float(row[name]))]
    if values:
        summary[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

with (base / "summary_overall.tsv").open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(("metric", "mean", "std", "min", "max"))
    for name, values in summary.items():
        writer.writerow((name, *(f"{values[key]:.6f}"
                                 for key in ("mean", "std", "min", "max"))))

architecture = {}
config_path = base / "test_chr_1" / "configuration.json"
if config_path.is_file():
    architecture = json.loads(config_path.read_text()).get("architecture_variant", {})

(base / "summary.json").write_text(json.dumps({
    "tf": tf,
    "variant": variant,
    "architecture_variant": architecture,
    "seed": seed,
    "activation": "relu",
    "reverse_complement_probability": 0.5,
    "runs": rows,
    "summary_overall": summary,
}, indent=2) + "\n")
print("Fuenf Folds aggregiert:", base)
PY
        touch "$COMBO_DIR/.complete"
    ) 8>"$COMBO_DIR/.aggregation.lock"

    # Sind alle 24 Kombinationen fertig, schreibt der letzte Task die Gesamtuebersicht.
    (
        flock -x 7
        local tf variant
        for tf in "${TF_NAMES[@]}"; do
            for variant in "${VARIANT_NAMES[@]}"; do
                [[ -f "$BASE_OUTPUT/$tf/$variant/.complete" ]] || exit 0
            done
        done

        "$PYTHON_BIN" - "$BASE_OUTPUT" "${TF_NAMES[@]}" -- "${VARIANT_NAMES[@]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

base = Path(sys.argv[1])
split = sys.argv.index("--")
tfs, variants = sys.argv[2:split], sys.argv[split + 1:]
metrics = ("auroc", "auprc", "mcc", "balanced_accuracy")

rows = []
for tf in tfs:
    for variant in variants:
        summary = json.loads((base / tf / variant / "summary.json").read_text())
        row = {"tf": tf, "variant": variant, "folds": len(summary["runs"]),
               "parameters": summary.get("architecture_variant", {}).get("parameters_total", "")}
        for name in metrics:
            row[f"{name}_mean"] = summary["summary_overall"][name]["mean"]
            row[f"{name}_std"] = summary["summary_overall"][name]["std"]
        rows.append(row)

columns = ["tf", "variant", "folds", "parameters"]
for name in metrics:
    columns += [f"{name}_mean", f"{name}_std"]
with (base / "variants_by_tf.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, columns, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})

# Rangliste ueber die TFs: Mittel je Variante, Abstand zur Basis, Rang je Metrik.
ranking = []
for variant in variants:
    entry = {"variant": variant}
    for name in metrics:
        values = [r[f"{name}_mean"] for r in rows if r["variant"] == variant]
        entry[f"{name}_mean_over_tfs"] = statistics.mean(values)
        deltas = []
        for tf in tfs:
            own = next(r for r in rows if r["tf"] == tf and r["variant"] == variant)
            ref = next(r for r in rows if r["tf"] == tf and r["variant"] == "base")
            deltas.append(own[f"{name}_mean"] - ref[f"{name}_mean"])
        entry[f"{name}_delta_vs_base"] = statistics.mean(deltas)
    ranking.append(entry)
for name in metrics:
    ordered = sorted(ranking, key=lambda e: e[f"{name}_mean_over_tfs"], reverse=True)
    for rank, entry in enumerate(ordered, start=1):
        entry[f"{name}_rank"] = rank
for entry in ranking:
    entry["mean_rank"] = statistics.mean(entry[f"{name}_rank"] for name in metrics)
ranking.sort(key=lambda e: (e["mean_rank"], -e["auprc_mean_over_tfs"]))

columns = ["variant", "mean_rank"]
for name in metrics:
    columns += [f"{name}_mean_over_tfs", f"{name}_delta_vs_base", f"{name}_rank"]
with (base / "variant_ranking.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, columns, delimiter="\t")
    writer.writeheader()
    for entry in ranking:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in entry.items()})
print("Gesamtuebersicht geschrieben:", base / "variants_by_tf.tsv", base / "variant_ranking.tsv")
PY
    ) 7>"$BASE_OUTPUT/.overall.lock"
}

case "$STAGE" in
    data) stage_data ;;
    cv)   stage_cv ;;
    *)    echo "Unbekannte Stufe STAGE=$STAGE (erwartet data oder cv)." >&2; exit 1 ;;
esac
