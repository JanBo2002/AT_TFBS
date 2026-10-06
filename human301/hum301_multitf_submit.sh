#!/usr/bin/env bash
# Run with bash on the login node. This launcher submits the compute jobs.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run bash hum301_multitf_submit.sh on the login node, outside a Slurm job." >&2
    exit 1
fi

PROJECT="/data/projects/SFB_A03/jan/AT_TFBS"
SCRIPT_DIR="$PROJECT/human301"
RUN_ID="multitf_$(date +%Y%m%d_%H%M%S)_$$"
MANIFEST="$SCRIPT_DIR/runs/multitf/$RUN_ID/workflow.json"

command -v sbatch >/dev/null
for file in hum301_multitf.py hum301_multitf_data.sbatch hum301_multitf_train.sbatch hum301_multitf_summary.sbatch; do
    [[ -f "$SCRIPT_DIR/$file" ]] || { echo "Missing $SCRIPT_DIR/$file" >&2; exit 1; }
done

python3 "$SCRIPT_DIR/hum301_multitf.py" plan --project-root "$PROJECT" \
    --run-id "$RUN_ID" --manifest "$MANIFEST"

EXPORT="ALL,HUM301_WORKFLOW_MANIFEST=$MANIFEST"
DATA_RESPONSE="$(sbatch --parsable --export="$EXPORT" "$SCRIPT_DIR/hum301_multitf_data.sbatch")"
DATA_JOB="${DATA_RESPONSE%%;*}"
[[ "$DATA_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid data job ID: $DATA_RESPONSE" >&2; exit 1; }
printf 'CPU data array: %s\n' "$DATA_JOB"

TRAIN_RESPONSE="$(sbatch --parsable --dependency="afterok:$DATA_JOB" --export="$EXPORT" "$SCRIPT_DIR/hum301_multitf_train.sbatch")"
TRAIN_JOB="${TRAIN_RESPONSE%%;*}"
[[ "$TRAIN_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid training job ID: $TRAIN_RESPONSE" >&2; exit 1; }
printf 'GPU training array: %s (after CPU data array %s)\n' "$TRAIN_JOB" "$DATA_JOB"

SUMMARY_RESPONSE="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --export="$EXPORT" "$SCRIPT_DIR/hum301_multitf_summary.sbatch")"
SUMMARY_JOB="${SUMMARY_RESPONSE%%;*}"
[[ "$SUMMARY_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid summary job ID: $SUMMARY_RESPONSE" >&2; exit 1; }
printf 'CPU summary job: %s (after GPU training array %s)\n' "$SUMMARY_JOB" "$TRAIN_JOB"
printf 'Run ID: %s\nEvaluation: %s\n' "$RUN_ID" "$SCRIPT_DIR/runs/multitf/$RUN_ID/evaluation"
