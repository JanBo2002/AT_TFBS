#!/usr/bin/env bash
#SBATCH --job-name=hum301_ablation
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
set -euo pipefail
ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"
if [[ $# -ne 2 || ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "Submit with submit.sh; the array task ID, plan and project root are required" >&2
    exit 2
fi
ABLATION_ROOT="$2"
"$ABLATION_PYTHON" "$ABLATION_ROOT/run_experiment.py" task --plan "$1" --index "$SLURM_ARRAY_TASK_ID"
