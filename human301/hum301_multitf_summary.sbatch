#!/usr/bin/env bash
#SBATCH --job-name=hum301_multi_summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/human301
#SBATCH --output=/data/projects/SFB_A03/jan/AT_TFBS/human301/hum301-multitf-summary-%j.out
#SBATCH --error=/data/projects/SFB_A03/jan/AT_TFBS/human301/hum301-multitf-summary-%j.err

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate at_env

MANIFEST="${HUM301_WORKFLOW_MANIFEST:?Start with bash hum301_multitf_submit.sh}"
python -u /data/projects/SFB_A03/jan/AT_TFBS/human301/hum301_multitf.py \
    summarize --manifest "$MANIFEST"
