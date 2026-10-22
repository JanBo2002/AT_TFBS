#!/usr/bin/env bash
#SBATCH --job-name=hum301_ablation
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
set -euo pipefail
ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"
if [[ $# -ne 3 || ! "$2" =~ ^[0-9]+$ ]]; then
    echo "Usage: sbatch train_task.sbatch /absolute/tasks.csv task_index /absolute/big_ablation" >&2
    exit 2
fi
# Slurm spools this file: its own directory is not the project directory.
ABLATION_ROOT="$3"
"$ABLATION_PYTHON" "$ABLATION_ROOT/run_experiment.py" task --plan "$1" --index "$2"
