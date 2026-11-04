#!/usr/bin/env bash
# Version: 1.0.3

#SBATCH --job-name=prediction_overlap
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if (( $# < 7 )); then
    echo "Fehler: Dieses Script über submit_prediction_overlap.sh starten." >&2
    exit 2
fi

python_bin=$1
analysis_script=$2
own_root=$3
reference_root=$4
output_dir=$5
seed=$6
shift 6

tfs=("$@")
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID fehlt}

if [[ ! "$task_id" =~ ^[0-9]+$ ]]; then
    echo "Ungültige Array-Task-ID: $task_id" >&2
    exit 2
fi

if (( task_id >= ${#tfs[@]} )); then
    echo "Array-Task-ID außerhalb der TF-Liste: $task_id" >&2
    exit 2
fi

tf=${tfs[$task_id]}

echo "TF: $tf"
echo "Job: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-unbekannt}}"
echo "Task: $task_id"
echo "Ausgabe: $output_dir"

exec "$python_bin" -u "$analysis_script" \
    --tf "$tf" \
    --own-root "$own_root" \
    --reference-root "$reference_root" \
    --output-dir "$output_dir" \
    --seed "$seed"
