#!/usr/bin/env bash
#SBATCH --job-name=CREB3L3_ensemble
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/human301
#SBATCH --output=/data/projects/SFB_A03/jan/AT_TFBS/human301/CREB3L3-ensemble-%j.out
#SBATCH --error=/data/projects/SFB_A03/jan/AT_TFBS/human301/CREB3L3-ensemble-%j.err

set -eo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate at_env

RUNS_ROOT="/data/projects/SFB_A03/jan/AT_TFBS/human301/runs/CREB3L3"

# Evaluate the existing three 9/12-bp seed runs from job 67995.
# No dependency on the new ablation jobs and no new training.
python -u "/data/projects/SFB_A03/jan/AT_TFBS/human301/hum301_ensemble.py" \
    --run-dirs \
    "$RUNS_ROOT/67995_motif64_9_12_epochs100_seed1" \
    "$RUNS_ROOT/67995_motif64_9_12_epochs100_seed2" \
    "$RUNS_ROOT/67995_motif64_9_12_epochs100_seed3" \
    --output-dir "$RUNS_ROOT/${SLURM_JOB_ID}_ensemble_existing_seed123"
