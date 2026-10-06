#!/usr/bin/env bash
#SBATCH --job-name=hum301_multi_train
#SBATCH --array=0-39
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/human301
#SBATCH --output=/data/projects/SFB_A03/jan/AT_TFBS/human301/hum301-multitf-train-%A_%a.out
#SBATCH --error=/data/projects/SFB_A03/jan/AT_TFBS/human301/hum301-multitf-train-%A_%a.err

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate at_env

MANIFEST="${HUM301_WORKFLOW_MANIFEST:?Start with bash hum301_multitf_submit.sh}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}"
python -u /data/projects/SFB_A03/jan/AT_TFBS/human301/hum301_multitf.py \
    train --manifest "$MANIFEST" --task-id "$TASK_ID"
