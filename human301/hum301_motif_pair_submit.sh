#!/usr/bin/env bash
# Auf dem Login-Node starten: bash hum301_motif_pair_submit.sh
# Vorhandene Daten; 30 TFs, Seed 1, 60 Hauptmodelle und 10 zusaetzliche Kontrollen.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Bitte auf dem Login-Node mit bash hum301_motif_pair_submit.sh starten." >&2
    exit 1
fi

PROJECT="/data/projects/SFB_A03/jan/AT_TFBS"
SCRIPT_DIR="$PROJECT/human301"
WORKER="$SCRIPT_DIR/hum301_motif_pair.sbatch"
RUN_ID="motif_pair_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$SCRIPT_DIR/runs/motif_pair/$RUN_ID"
MANIFEST="$RUN_DIR/workflow.json"

command -v sbatch >/dev/null
command -v python3 >/dev/null
for file in hum301_motif_pair.py hum301_motif_pair.sbatch hum301_motif_pair_submit.sh hum301_data.py hum301_crossassay.py; do
    [[ -f "$SCRIPT_DIR/$file" ]] || { echo "Datei fehlt: $SCRIPT_DIR/$file" >&2; exit 1; }
done

# Die Planung braucht nur Python-Standardbibliotheken. Caches prueft der CPU-Job.
# Alle Log-Verzeichnisse muessen bereits vor dem Einreichen existieren.
HUM301_PAIR_STAGE=plan HUM301_PAIR_RUN_DIR="$RUN_DIR" bash "$WORKER"
N_MODELS="$(python3 - "$MANIFEST" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(len(json.load(handle)["runs"]))
PY
)"
[[ "$N_MODELS" =~ ^[1-9][0-9]*$ ]] || { echo "Ungueltige Modellanzahl: $N_MODELS" >&2; exit 1; }
LAST_TASK=$((N_MODELS - 1))

submit_job() {
    local response job
    response="$(sbatch --parsable "$@")"
    job="${response%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Unerwartete sbatch-Antwort: $response" >&2; return 1; }
    printf '%s\n' "$job"
}

DATA_JOB="$(submit_job \
    --job-name=hum301-pair-data --ntasks=1 --cpus-per-task=4 \
    --mem=32G --time=24:00:00 \
    --output="$RUN_DIR/logs/data/hum301-pair-data-%j.out" \
    --error="$RUN_DIR/logs/data/hum301-pair-data-%j.err" \
    --export="ALL,HUM301_PAIR_STAGE=data,HUM301_PAIR_MANIFEST=$MANIFEST" \
    "$WORKER")"
printf 'stage\tjob_id\ndata\t%s\n' "$DATA_JOB" > "$RUN_DIR/jobs.tsv"
printf 'CPU-Datenpruefung: %s\n' "$DATA_JOB"

TRAIN_JOB="$(submit_job \
    --dependency="afterok:$DATA_JOB" --array="0-$LAST_TASK" \
    --job-name=hum301-pair-train --ntasks=1 --cpus-per-task=4 \
    --mem=32G --gres=gpu:1 --time=48:00:00 \
    --output="$RUN_DIR/logs/train/hum301-pair-train-%A_%a.out" \
    --error="$RUN_DIR/logs/train/hum301-pair-train-%A_%a.err" \
    --export="ALL,HUM301_PAIR_STAGE=train,HUM301_PAIR_MANIFEST=$MANIFEST" \
    "$WORKER")"
printf 'train\t%s\n' "$TRAIN_JOB" >> "$RUN_DIR/jobs.tsv"
printf 'GPU-Array: %s (%s Modelle, %s Testauswertungen)\n' "$TRAIN_JOB" "$N_MODELS" "$((2 * N_MODELS))"

SUMMARY_JOB="$(submit_job \
    --dependency="afterok:$TRAIN_JOB" \
    --job-name=hum301-pair-summary --ntasks=1 --cpus-per-task=2 \
    --mem=8G --time=01:00:00 \
    --output="$RUN_DIR/logs/summary/hum301-pair-summary-%j.out" \
    --error="$RUN_DIR/logs/summary/hum301-pair-summary-%j.err" \
    --export="ALL,HUM301_PAIR_STAGE=summary,HUM301_PAIR_MANIFEST=$MANIFEST" \
    "$WORKER")"
printf 'summary\t%s\n' "$SUMMARY_JOB" >> "$RUN_DIR/jobs.tsv"
printf 'CPU-Zusammenfassung: %s\nAuswertung: %s\nLogs: %s\n' \
    "$SUMMARY_JOB" "$RUN_DIR/evaluation" "$RUN_DIR/logs"
