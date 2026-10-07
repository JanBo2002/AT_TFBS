#!/usr/bin/env bash
#SBATCH --job-name=hum301_cross_data
#SBATCH --array=0-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/human301
#SBATCH --output=/data/projects/SFB_A03/jan/AT_TFBS/human301/hum301-crossassay-data-%A_%a.out
#SBATCH --error=/data/projects/SFB_A03/jan/AT_TFBS/human301/hum301-crossassay-data-%A_%a.err

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate at_env

MANIFEST="${HUM301_CROSSASSAY_MANIFEST:?Start with bash hum301_crossassay_submit.sh}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}"
python -u /data/projects/SFB_A03/jan/AT_TFBS/human301/hum301_crossassay.py \
    prepare --manifest "$MANIFEST" --task-id "$TASK_ID"
