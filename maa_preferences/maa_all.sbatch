#!/usr/bin/env bash
#SBATCH --job-name=maa_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jbossenz/AT_TFBS/maa_preferences/slurm/%x-%j.out
#SBATCH --error=/home/jbossenz/AT_TFBS/maa_preferences/slurm/%x-%j.err

# Ablage: /home/jbossenz/AT_TFBS/maa_preferences/maa_all.sbatch
# Vor dem ersten sbatch: mkdir -p /home/jbossenz/AT_TFBS/maa_preferences/slurm
# Start: sbatch /home/jbossenz/AT_TFBS/maa_preferences/maa_all.sbatch [all|data-only|cv-only]
# all: 3 CPU-Tasks (Datensatz + Cache), danach 15 GPU-Tasks (3 TFs x 5 Folds).

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/software/conda/envs/relion-5.0/bin/python}"
PROJECT_DIR=/home/jbossenz/AT_TFBS/maa_preferences
SELF="$PROJECT_DIR/maa_all.sbatch"
SLURM_DIR="$PROJECT_DIR/slurm"
DATA_PARENT="$PROJECT_DIR/maa_data"
RAW=/home/jbossenz/AT_TFBS/raw_data
GENERAL="$RAW/general_data"
GENE_ANNOTATION="$GENERAL/Arabidopsis_thaliana.TAIR10.63.gff3.gz"
TSS_PLUS="$GENERAL/AT_TSS/clustered_TSS_Le_2022_sup2_wt_plus.bed"
TSS_MINUS="$GENERAL/AT_TSS/clustered_TSS_Le_2022_sup2_wt_minus.bed"
NEGATIVE_LOCATION_PREFERENCE="${NEGATIVE_LOCATION_PREFERENCE:-gene-or-tss}"
NEGATIVE_PREFERENCE_DISTANCE="${NEGATIVE_PREFERENCE_DISTANCE:-1000}"
DATA_SCRIPT="$PROJECT_DIR/data_maa.py"
TRAIN_SCRIPT="$PROJECT_DIR/maa.py"
MATCH_SCRIPT="$PROJECT_DIR/gc_matching.py"

# Reihenfolge = TF-Index im Daten- und Trainingsarray.
TF_NAMES=(
    AP1_WILDTYPE_GUO2025
    MYC3_air_jablock_2h_chipseq_Yin2023
    MYC2_ja_2h_chipseq_Yin2023
)
CHROMOSOMES=1,2,3,4,5
CONTEXT_PER_SIDE=196
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_SHIFT_MAX=128
TRAIN_SHIFT_DISTRIBUTION=uniform
EVAL_SHIFT=fixed
CV_CONCURRENCY="${CV_CONCURRENCY:-4}"

TF_COUNT=${#TF_NAMES[@]}
DATA_MAX=$((TF_COUNT - 1))
CV_MAX=$((TF_COUNT * 5 - 1))
STAGE_LOGS=(--output="$SLURM_DIR/%x-%A_%a.out" --error="$SLURM_DIR/%x-%A_%a.err")
# Genomweiter 1-bp-Kandidatenpool; CPU-Stufe bekommt bis zu 24 h.
DATA_RESOURCES=(--cpus-per-task=4 --mem=32G --time=24:00:00 "${STAGE_LOGS[@]}")
CV_RESOURCES=(--cpus-per-task=4 --mem=32G --time=24:00:00 --gres=gpu:1 "${STAGE_LOGS[@]}")

unset PYTHONHOME PYTHONPATH || true
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

fail() { echo "$*" >&2; exit 1; }

check_files() {
    local path
    [[ -x "$PYTHON_BIN" ]] || fail "Python fehlt oder ist nicht ausfuehrbar: $PYTHON_BIN"
    for path in "$SELF" "$DATA_SCRIPT" "$TRAIN_SCRIPT" "$MATCH_SCRIPT"; do
        [[ -f "$path" ]] || fail "Fehlt: $path"
    done
    command -v flock >/dev/null || fail "flock fehlt."
    case "$NEGATIVE_LOCATION_PREFERENCE" in gene-or-tss|gene|tss|none) ;;
        *) fail "NEGATIVE_LOCATION_PREFERENCE: gene-or-tss | gene | tss | none" ;; esac
    [[ "$NEGATIVE_PREFERENCE_DISTANCE" =~ ^[0-9]+$ ]] || fail "NEGATIVE_PREFERENCE_DISTANCE muss >= 0 sein."
}

check_raw_files() {
    local tf="$1" path
    local prefix="$RAW/$tf/signal/$tf.pooled_replicates.macs3"
    for path in "$prefix.FE.bw" "$prefix.pval.bw" "$prefix.pval.called_summits.bw" \
                "$GENERAL/genome/TAIR10.1.atlas.fa" "$GENERAL/genome/TAIR10.1.atlas.fa.fai"; do
        [[ -f "$path" ]] || fail "Fehlt: $path"
    done
    if [[ "$NEGATIVE_LOCATION_PREFERENCE" == gene || "$NEGATIVE_LOCATION_PREFERENCE" == gene-or-tss ]]; then
        [[ -f "$GENE_ANNOTATION" ]] || fail "Fehlt: $GENE_ANNOTATION"
    fi
    if [[ "$NEGATIVE_LOCATION_PREFERENCE" == tss || "$NEGATIVE_LOCATION_PREFERENCE" == gene-or-tss ]]; then
        for path in "$TSS_PLUS" "$TSS_MINUS"; do [[ -f "$path" ]] || fail "Fehlt: $path"; done
    fi
}

dataset_is_current() {
    "$PYTHON_BIN" - "$1" "$DATA_SCRIPT" "$MATCH_SCRIPT" "$NEGATIVE_LOCATION_PREFERENCE" "$NEGATIVE_PREFERENCE_DISTANCE" \
        "$GENE_ANNOTATION" "$TSS_PLUS" "$TSS_MINUS" <<'PY'
import ast
import json
import sys
from pathlib import Path

root, data_script, match_script = map(Path, sys.argv[1:4])
mode, distance = sys.argv[4], int(sys.argv[5])
annotations = [Path(value) for value in sys.argv[6:]]
def version(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Version fehlt: {path}")
try:
    payload = json.loads((root / "summary.json").read_text())
    profile = payload.get("negative_location_preference", {})
    current = (payload.get("script_version") == version(data_script, "SCRIPT_VERSION")
               and payload.get("gc_matcher_version") == version(match_script, "MATCHER_VERSION")
               and profile.get("mode") == mode and profile.get("distance_bp") == distance)
    required = ([annotations[0]] if mode in {"gene", "gene-or-tss"} else [])
    if mode in {"tss", "gene-or-tss"}:
        required.extend(annotations[1:])
    complete_mtime = (root / ".complete").stat().st_mtime_ns
    current &= all(path.is_file() and path.stat().st_mtime_ns <= complete_mtime
                   for path in (data_script, match_script, *required))
    if mode in {"gene", "gene-or-tss"}:
        current &= profile.get("gene_annotation") == str(annotations[0])
    if mode in {"tss", "gene-or-tss"}:
        current &= profile.get("tss_plus", {}).get("path") == str(annotations[1])
        current &= profile.get("tss_minus", {}).get("path") == str(annotations[2])
except (OSError, ValueError, TypeError):
    current = False
raise SystemExit(0 if current else 1)
PY
}

validate_dataset() {
    "$PYTHON_BIN" - "$1" "$CONTEXT_PER_SIDE" "$NEGATIVE_LOCATION_PREFERENCE" "$NEGATIVE_PREFERENCE_DISTANCE" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

root, context = Path(sys.argv[1]), int(sys.argv[2])
if not (root / ".complete").is_file():
    raise SystemExit(f"Datensatz nicht fertig: {root}")
required = {"sample_id", "label", "chromosome", "window_length", "context_per_side",
            "context_length", "window_data_file"}
counts = Counter()
for name, expected_label in (("positive", 1), ("negative", 0)):
    path = root / f"metadata_{name}.tsv"
    if not path.is_file():
        raise SystemExit(f"Metadaten fehlen: {path}")
    count = 0
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"{path}: benoetigte Metadaten-Spalten fehlen")
        for row in reader:
            if (int(row["label"]) != expected_label or int(row["window_length"]) != 1024
                    or int(row["context_per_side"]) != context
                    or int(row["context_length"]) != 1024 + 2 * context):
                raise SystemExit(f"{path}: falsches Label oder falsche Fenstergeometrie")
            counts[(row["chromosome"], name)] += 1
            count += 1
    if not count:
        raise SystemExit(f"Leere Klasse: {path}")
    print(f"{path.name}: {count:,} Fenster")
summary = json.loads((root / "summary.json").read_text())
if summary.get("selection", {}).get("gc_matching_scope") != "pooled":
    raise SystemExit(f"{root}: erwartet wird ein Datensatz mit gemeinsamem GC-Matching")
profile = summary.get("negative_location_preference", {})
if profile.get("mode") != sys.argv[3] or profile.get("distance_bp") != int(sys.argv[4]):
    raise SystemExit(f"{root}: falsche Lagepraeferenz; zuerst all oder data-only starten")
print("Lagepraeferenz:", profile["mode"], profile["distance_bp"], "bp; Anteil bevorzugter Negativer:",
      profile.get("preferred_selected_fraction"))
for chrom in sorted({chrom for chrom, _ in counts}):
    print(f"{chrom}: {counts[(chrom, 'positive')]:,} Positive / {counts[(chrom, 'negative')]:,} Negative")
PY
}

ensure_cache() {
    local root="$1"
    local cache="$root/maa_cache.npz"
    (
        flock -x 8
        if [[ ! -s "$cache" || "$root/metadata_positive.tsv" -nt "$cache" \
                || "$root/metadata_negative.tsv" -nt "$cache" || "$TRAIN_SCRIPT" -nt "$cache" ]]; then
            # inspect mit val-fraction 0 baut den Cache ohne MILP-Validierungssplit.
            # Die eigentlichen Folds waehlen danach ihren eigenen Split.
            "$PYTHON_BIN" "$TRAIN_SCRIPT" inspect \
                --dataset-root "$root" --cache-path "$cache" \
                --train-chromosomes 1,2,3,4 --test-chromosomes 5 \
                --val-fraction 0 --rebuild-cache \
                2>&1 | tee "$root/cache_build.log"
        fi
    ) 8>"$root/.cache.lock"
    [[ -s "$cache" ]] || fail "Cache fehlt: $cache"
}

submit_job() {
    local reply job
    reply=$(sbatch --parsable "$@")
    job=${reply%%;*}  # --parsable kann JOBID;CLUSTER liefern.
    [[ "$job" =~ ^[0-9]+$ ]] || fail "Unerwartete sbatch-Antwort: $reply"
    printf '%s\n' "$job"
}

# ---------------------------------------------------------------------------
# Launcher. all wartet mit dem GPU-Array auf ALLE drei erfolgreichen CPU-Tasks.
# ---------------------------------------------------------------------------
if [[ -z "${STAGE:-}" ]]; then
    MODE="${1:-all}"
    case "$MODE" in all|data-only|cv-only) ;; *) fail "Modus: all | data-only | cv-only" ;; esac
    command -v sbatch >/dev/null || fail "sbatch nicht gefunden."
    check_files
    [[ "$CV_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || fail "CV_CONCURRENCY muss positiv sein."
    mkdir -p "$SLURM_DIR" "$DATA_PARENT"
    echo "Python: $PYTHON_BIN"
    echo "Projekt: $PROJECT_DIR | $TF_COUNT TFs | $((TF_COUNT * 5)) Trainingslaeufe | Seed $SEED"

    submit_data() {
        submit_job --export=ALL,STAGE=data --job-name=maa_data \
            --array="0-$DATA_MAX" "${DATA_RESOURCES[@]}" "$SELF"
    }
    submit_cv() {
        local dependency=()
        [[ -z "${1:-}" ]] || dependency=(--dependency="afterok:$1")
        submit_job --export=ALL,STAGE=cv --job-name=maa_cv \
            --array="0-$CV_MAX%$CV_CONCURRENCY" "${CV_RESOURCES[@]}" "${dependency[@]}" "$SELF"
    }
    case "$MODE" in
        all|data-only)
            for tf in "${TF_NAMES[@]}"; do check_raw_files "$tf"; done
            DATA_JOB=$(submit_data)
            echo "Datensatz-Job: $DATA_JOB (3 TFs, inklusive Cache)"
            if [[ "$MODE" == all ]]; then
                CV_JOB=$(submit_cv "$DATA_JOB")
                echo "CV-Job: $CV_JOB (15 Laeufe, wartet auf $DATA_JOB)"
            fi
            ;;
        cv-only)
            for tf in "${TF_NAMES[@]}"; do validate_dataset "$DATA_PARENT/$tf"; done
            CV_JOB=$(submit_cv "")
            echo "CV-Job: $CV_JOB (15 Laeufe auf vorhandenen Datensaetzen)"
            ;;
    esac
    exit 0
fi

# ---------------------------------------------------------------------------
# CPU-Stufe: alle Summits / genomweite negative Starts, gemeinsames GC-Matching.
# TSS/GFF nur fuer weiche Negativprioritaet; keine positiven Abstandsregeln.
# ---------------------------------------------------------------------------
stage_data() {
    local task="${SLURM_ARRAY_TASK_ID:?Nur als Slurm-Array starten}"
    [[ "$task" =~ ^[0-9]+$ ]] && ((task >= 0 && task < TF_COUNT)) || fail "Ungueltiger Daten-Task: $task"
    local TF="${TF_NAMES[$task]}"
    local PREFIX="$RAW/$TF/signal/$TF.pooled_replicates.macs3"
    local ROOT="$DATA_PARENT/$TF"
    check_files
    check_raw_files "$TF"
    mkdir -p "$DATA_PARENT/.locks" "$DATA_PARENT/.tmp"
    # Gleichzeitige Launcher duerfen denselben TF-Datensatz nicht ueberschreiben.
    exec 9>"$DATA_PARENT/.locks/$TF.lock"
    flock -x 9

    if [[ -f "$ROOT/.complete" && "${FORCE_REGENERATE:-0}" != 1 ]] && dataset_is_current "$ROOT"; then
        echo "Fertiger Datensatz wird verwendet: $ROOT"
    else
        echo "Datensatz wird mit aktueller Lagepraeferenz neu erzeugt: $NEGATIVE_LOCATION_PREFERENCE, $NEGATIVE_PREFERENCE_DISTANCE bp"
        "$PYTHON_BIN" - <<'PY'
import pyBigWig, pysam, numpy, matplotlib
print("Generator-Abhaengigkeiten vorhanden")
PY
        TASK_TMP_DIR=$(mktemp -d "$DATA_PARENT/.tmp/${TF}.${SLURM_JOB_ID:-local}.XXXXXX")
        export TMPDIR="$TASK_TMP_DIR"
        trap 'if [[ -n "${TASK_TMP_DIR:-}" && -d "$TASK_TMP_DIR" ]]; then rm -rf -- "$TASK_TMP_DIR"; fi' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        echo "TF $TF | Ausgabe $ROOT | Kontext $CONTEXT_PER_SIDE bp je Seite"
        "$PYTHON_BIN" "$DATA_SCRIPT" \
            --fe-bigwig "$PREFIX.FE.bw" \
            --pval-bigwig "$PREFIX.pval.bw" \
            --summit-bigwig "$PREFIX.pval.called_summits.bw" \
            --fasta "$GENERAL/genome/TAIR10.1.atlas.fa" \
            --fai "$GENERAL/genome/TAIR10.1.atlas.fa.fai" \
            --chromosomes "$CHROMOSOMES" \
            --negative-location-preference "$NEGATIVE_LOCATION_PREFERENCE" \
            --negative-preference-distance "$NEGATIVE_PREFERENCE_DISTANCE" \
            --gene-annotation "$GENE_ANNOTATION" --tss-plus "$TSS_PLUS" --tss-minus "$TSS_MINUS" \
            --output-dir "$ROOT" \
            --context-per-side "$CONTEXT_PER_SIDE" \
            --tmp-dir "$TASK_TMP_DIR" \
            --overwrite \
            2>&1 | tee "$DATA_PARENT/generation_$TF.log"
    fi
    validate_dataset "$ROOT"
    ensure_cache "$ROOT"
    echo "Datensatz und Cache fertig: $ROOT"
    exec 9>&-
}

# ---------------------------------------------------------------------------
# GPU-Stufe: Task-ID = TF-Index * 5 + (Testchromosom - 1), insgesamt 0..14.
# Je TF/Chromosom genau ein maa.py-Lauf; keine Architekturvarianten.
# ---------------------------------------------------------------------------
stage_cv() {
    local task="${SLURM_ARRAY_TASK_ID:?Nur als Slurm-Array starten}"
    [[ "$task" =~ ^[0-9]+$ ]] && ((task >= 0 && task <= CV_MAX)) || fail "Ungueltiger CV-Task: $task"
    local TF="${TF_NAMES[$((task / 5))]}"
    local TEST_CHR=$((task % 5 + 1))
    local TRAIN_CHR_CSV="" chr
    for chr in 1 2 3 4 5; do
        [[ "$chr" == "$TEST_CHR" ]] || TRAIN_CHR_CSV+="${TRAIN_CHR_CSV:+,}$chr"
    done
    local ROOT="$DATA_PARENT/$TF"
    local BASE_OUTPUT="$PROJECT_DIR/cv_maa_seed_${SEED}_${SLURM_ARRAY_JOB_ID:?}"
    local TF_DIR="$BASE_OUTPUT/$TF"
    local RUN_DIR="$TF_DIR/test_chr_$TEST_CHR"
    local CACHE_PATH="$ROOT/maa_cache.npz"
    check_files
    [[ -f "$ROOT/.complete" ]] || fail "Datensatz nicht fertig: $ROOT"
    mkdir -p "$RUN_DIR"
    rm -f "$RUN_DIR/.complete"
    echo "TF $TF | Modell maa | Seed $SEED | Test $TEST_CHR | Training $TRAIN_CHR_CSV"
    echo "Ergebnisse: $RUN_DIR"
    "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch erkennt keine GPU")
print("GPU:", torch.cuda.get_device_name(0))
PY
    ensure_cache "$ROOT"
    "$PYTHON_BIN" "$TRAIN_SCRIPT" train \
        --dataset-root "$ROOT" --cache-path "$CACHE_PATH" --output-dir "$RUN_DIR" \
        --train-chromosomes "$TRAIN_CHR_CSV" --test-chromosomes "$TEST_CHR" \
        --device cuda --batch-size "$BATCH_SIZE" --num-workers 0 \
        --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
        --task-mode classification --activation relu \
        --checkpoint-objective classification-auprc --threshold-strategy val-mcc \
        --reverse-complement-probability 0.5 \
        --train-shift-max "$TRAIN_SHIFT_MAX" --train-shift-distribution "$TRAIN_SHIFT_DISTRIBUTION" \
        --eval-shift "$EVAL_SHIFT" --balanced-evaluation --no-amp \
        2>&1 | tee "$RUN_DIR/training.log"
    [[ -f "$RUN_DIR/binary_classification_metrics.json" ]] || fail "Metrik-Datei fehlt: $RUN_DIR"
    touch "$RUN_DIR/.complete"

    # Der letzte erfolgreiche Fold je TF schreibt dessen Zusammenfassung.
    (
        flock -x 8
        for chr in 1 2 3 4 5; do
            [[ -f "$TF_DIR/test_chr_$chr/.complete" ]] || exit 0
        done
        "$PYTHON_BIN" - "$TF_DIR" "$TF" "$SEED" <<'PY'
import csv
import json
import math
import statistics
import sys
from pathlib import Path

base, tf, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
metrics = ("auroc", "auprc", "balanced_accuracy", "f1", "mcc", "precision", "recall", "specificity")
numbers = ("checkpoint_epoch", "epochs_run", "training_minutes", "threshold", "validation_mcc", *metrics)
columns = ("tf", "model", "seed", "train_chromosomes", "test_chromosome", "test_positives_dropped", *numbers)
rows = []
for chromosome in range(1, 6):
    run = base / f"test_chr_{chromosome}"
    payload = json.loads((run / "binary_classification_metrics.json").read_text())
    config = json.loads((run / "configuration.json").read_text())
    history = json.loads((run / "history.json").read_text())
    row = {
        "tf": tf, "model": "maa", "seed": seed,
        "train_chromosomes": ",".join(str(c) for c in range(1, 6) if c != chromosome),
        "test_chromosome": chromosome, "checkpoint_epoch": payload["checkpoint_epoch"],
        "threshold": payload["threshold"], "validation_mcc": payload["validation_objective_score"],
        "test_positives_dropped": config.get("evaluation_balance", {}).get("test", {}).get("positives_dropped_total", 0),
        "epochs_run": len(history),
        "training_minutes": sum(float(epoch.get("elapsed_seconds", 0)) for epoch in history) / 60,
        **{name: payload["test_metrics"][name] for name in metrics},
    }
    rows.append(row)
with (base / "run_results.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
summary = {}
for name in numbers:
    values = [float(row[name]) for row in rows if math.isfinite(float(row[name]))]
    if values:
        summary[name] = {"mean": statistics.mean(values),
                         "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                         "min": min(values), "max": max(values)}
with (base / "summary_overall.tsv").open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(("metric", "mean", "std", "min", "max"))
    for name, values in summary.items():
        writer.writerow((name, *(f"{values[key]:.6f}" for key in ("mean", "std", "min", "max"))))
(base / "summary.json").write_text(json.dumps({
    "tf": tf, "model": "maa", "seed": seed, "task_mode": "classification",
    "runs": rows, "summary_overall": summary,
}, indent=2) + "\n")
print("Fuenf Folds aggregiert:", base)
PY
        touch "$TF_DIR/.complete"
    ) 8>"$TF_DIR/.aggregation.lock"

    # Nach allen 15 Laeufen: gemeinsame Ergebnistabelle und Mittel je TF.
    (
        flock -x 7
        local tf
        for tf in "${TF_NAMES[@]}"; do
            [[ -f "$BASE_OUTPUT/$tf/.complete" ]] || exit 0
        done
        "$PYTHON_BIN" - "$BASE_OUTPUT" "${TF_NAMES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

base, tfs = Path(sys.argv[1]), sys.argv[2:]
metrics = ("auroc", "auprc", "mcc", "balanced_accuracy", "f1", "precision", "recall", "specificity")
all_runs, summaries = [], []
for tf in tfs:
    payload = json.loads((base / tf / "summary.json").read_text())
    all_runs.extend(payload["runs"])
    row = {"tf": tf, "model": "maa", "seed": payload["seed"], "folds": len(payload["runs"])}
    for name in metrics:
        stats = payload["summary_overall"].get(name, {})
        row[f"{name}_mean"] = stats.get("mean", "")
        row[f"{name}_std"] = stats.get("std", "")
    summaries.append(row)
with (base / "run_results_all.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, list(all_runs[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(all_runs)
with (base / "tf_summary.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, list(summaries[0]), delimiter="\t")
    writer.writeheader()
    for row in summaries:
        writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()})
(base / "summary.json").write_text(json.dumps({"model": "maa", "runs_total": len(all_runs),
                                             "tf_summaries": summaries}, indent=2) + "\n")
print("Gesamtuebersicht:", base / "run_results_all.tsv", base / "tf_summary.tsv")
PY
        touch "$BASE_OUTPUT/.complete"
    ) 7>"$BASE_OUTPUT/.overall.lock"
}

case "$STAGE" in
    data) stage_data ;;
    cv) stage_cv ;;
    *) fail "Unbekannte Stufe STAGE=$STAGE (data oder cv erwartet)." ;;
esac
