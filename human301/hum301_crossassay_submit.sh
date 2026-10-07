#!/usr/bin/env bash
# Start with bash on the login node; computation is submitted with sbatch.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Start bash hum301_crossassay_submit.sh on the login node, outside a Slurm job." >&2
    exit 1
fi

PROJECT="/data/projects/SFB_A03/jan/AT_TFBS"
SCRIPT_DIR="$PROJECT/human301"
SEED=1
RUN_ID="crossassay_$(date +%Y%m%d_%H%M%S)_$$"
MANIFEST="$SCRIPT_DIR/runs/crossassay/$RUN_ID/workflow.json"

command -v sbatch >/dev/null
for file in hum301_crossassay.py hum301_crossassay_data.sbatch hum301_crossassay_train.sbatch hum301_crossassay_summary.sbatch; do
    [[ -f "$SCRIPT_DIR/$file" ]] || { echo "Missing $SCRIPT_DIR/$file" >&2; exit 1; }
done
python3 "$SCRIPT_DIR/hum301_crossassay.py" plan --project-root "$PROJECT" \
    --run-id "$RUN_ID" --seed "$SEED" --manifest "$MANIFEST"

EXPORT="ALL,HUM301_CROSSASSAY_MANIFEST=$MANIFEST"
DATA_RESPONSE="$(sbatch --parsable --export="$EXPORT" "$SCRIPT_DIR/hum301_crossassay_data.sbatch")"
DATA_JOB="${DATA_RESPONSE%%;*}"
[[ "$DATA_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid data job ID: $DATA_RESPONSE" >&2; exit 1; }
printf 'CPU data array: %s (11 TFs, CHIP and GHTS)\n' "$DATA_JOB"

TRAIN_RESPONSE="$(sbatch --parsable --dependency="afterok:$DATA_JOB" --export="$EXPORT" "$SCRIPT_DIR/hum301_crossassay_train.sbatch")"
TRAIN_JOB="${TRAIN_RESPONSE%%;*}"
[[ "$TRAIN_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid training job ID: $TRAIN_RESPONSE" >&2; exit 1; }
printf 'GPU array: %s (22 models, each tested on CHIP and GHTS)\n' "$TRAIN_JOB"

SUMMARY_RESPONSE="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --export="$EXPORT" "$SCRIPT_DIR/hum301_crossassay_summary.sbatch")"
SUMMARY_JOB="${SUMMARY_RESPONSE%%;*}"
[[ "$SUMMARY_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid summary job ID: $SUMMARY_RESPONSE" >&2; exit 1; }
printf 'CPU summary job: %s\nRun ID: %s\nEvaluation: %s\n' \
    "$SUMMARY_JOB" "$RUN_ID" "$SCRIPT_DIR/runs/crossassay/$RUN_ID/evaluation"
