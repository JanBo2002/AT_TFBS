#!/usr/bin/env bash
# Start with bash on the login node; computation is submitted with sbatch.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Start bash hum301_otherTFs_submit.sh on the login node, outside a Slurm job." >&2
    exit 1
fi

PROJECT="/data/projects/SFB_A03/jan/AT_TFBS"
SCRIPT_DIR="$PROJECT/human301"
INPUT_ROOT="/data/projects/SFB_A03/clara/ArchThalia-ML"
SEED=1
RUN_ID="crossassay_archthalia_$(date +%Y%m%d_%H%M%S)_$$"
MANIFEST="$SCRIPT_DIR/runs/crossassay/$RUN_ID/workflow.json"

command -v sbatch >/dev/null
for file in hum301_crossassay.py hum301_crossassay_data.sbatch hum301_crossassay_train.sbatch hum301_crossassay_summary.sbatch; do
    [[ -f "$SCRIPT_DIR/$file" ]] || { echo "Missing $SCRIPT_DIR/$file" >&2; exit 1; }
done
python3 "$SCRIPT_DIR/hum301_crossassay.py" plan --project-root "$PROJECT" \
    --other-tfs --input-root "$INPUT_ROOT" --run-id "$RUN_ID" --seed "$SEED" --manifest "$MANIFEST"

read -r DATA_TASKS TRAIN_TASKS <<< "$(python3 - "$MANIFEST" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    workflow = json.load(handle)
print(len(workflow["tfs"]), len(workflow["runs"]))
PY
)"
DATA_LAST=$((DATA_TASKS - 1))
TRAIN_LAST=$((TRAIN_TASKS - 1))

EXPORT="ALL,HUM301_CROSSASSAY_MANIFEST=$MANIFEST"
DATA_RESPONSE="$(sbatch --parsable --array="0-$DATA_LAST" --export="$EXPORT" "$SCRIPT_DIR/hum301_crossassay_data.sbatch")"
DATA_JOB="${DATA_RESPONSE%%;*}"
[[ "$DATA_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid data job ID: $DATA_RESPONSE" >&2; exit 1; }
printf 'CPU cache array: %s (%s TFs; pack existing CHS/GHTS FASTAs)\n' "$DATA_JOB" "$DATA_TASKS"

TRAIN_RESPONSE="$(sbatch --parsable --array="0-$TRAIN_LAST" --dependency="afterok:$DATA_JOB" --export="$EXPORT" "$SCRIPT_DIR/hum301_crossassay_train.sbatch")"
TRAIN_JOB="${TRAIN_RESPONSE%%;*}"
[[ "$TRAIN_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid training job ID: $TRAIN_RESPONSE" >&2; exit 1; }
printf 'GPU array: %s (%s models, %s tests; each model tested on CHIP and GHTS)\n' \
    "$TRAIN_JOB" "$TRAIN_TASKS" "$((2 * TRAIN_TASKS))"

SUMMARY_RESPONSE="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --export="$EXPORT" "$SCRIPT_DIR/hum301_crossassay_summary.sbatch")"
SUMMARY_JOB="${SUMMARY_RESPONSE%%;*}"
[[ "$SUMMARY_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid summary job ID: $SUMMARY_RESPONSE" >&2; exit 1; }
printf 'CPU summary job: %s\nRun ID: %s\nEvaluation: %s\n' \
    "$SUMMARY_JOB" "$RUN_ID" "$SCRIPT_DIR/runs/crossassay/$RUN_ID/evaluation"
