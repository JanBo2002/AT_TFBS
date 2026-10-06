#!/usr/bin/env bash
#SBATCH --job-name=CREB3L3_ablations
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/human301
#SBATCH --output=/data/projects/SFB_A03/jan/AT_TFBS/human301/CREB3L3-ablations-%A_%a.out
#SBATCH --error=/data/projects/SFB_A03/jan/AT_TFBS/human301/CREB3L3-ablations-%A_%a.err

set -eo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate at_env

TF="CREB3L3"
DATA_RUN_ID="hum301_20261005_185818_tdUK1h"

# Each group of three tasks uses seeds 1, 2, 3.
#  0-2:  base              -> 9/12, 64 -> 80/96/128
#  3-5:  channels96        -> 9/12, 96 -> 96/112/128; compare with base
#  6-8:  weight_decay     -> base + weight decay 1e-3 only
#  9-11: dropout          -> base + dropout 0.2 only
# 12-14: local_readout    -> base + central 64-bp LogSumExp only
# 15-17: channels128      -> 9/12, 128 -> 128/128/128; wide control
# 18-20: multi32          -> 9/12/15/21, 128 -> 128/128/128
# Compare multi32 with channels128: all channel widths match.
# Bottleneck in every run: original five blocks, 128 channels, 64 positions,
# four attention heads, FFN 256. All three decoder blocks remain.
case "${SLURM_ARRAY_TASK_ID:-}" in
    [0-9]|1[0-9]|20) ;;
    *)
        echo "Submit with sbatch; array task must be 0 through 20." >&2
        exit 1
        ;;
esac

VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3 + 1))
MODEL_SCRIPT="hum301_motif64.py"
EMBED_CHANNELS=64
ENCODER_CHANNELS=(80 96 128)
WEIGHT_DECAY="1e-4"
DROPOUT="0.1"
READOUT="center"

case "$VARIANT_INDEX" in
    0) VARIANT="base" ;;
    1)
        VARIANT="channels96"
        EMBED_CHANNELS=96
        ENCODER_CHANNELS=(96 112 128)
        ;;
    2)
        VARIANT="weight_decay"
        WEIGHT_DECAY="1e-3"
        ;;
    3)
        VARIANT="dropout"
        DROPOUT="0.2"
        ;;
    4)
        VARIANT="local_readout"
        MODEL_SCRIPT="hum301_motif64_local.py"
        READOUT="local-logsumexp"
        ;;
    5)
        VARIANT="channels128"
        EMBED_CHANNELS=128
        ENCODER_CHANNELS=(128 128 128)
        ;;
    6)
        VARIANT="multi32"
        MODEL_SCRIPT="hum301_motif64_multi.py"
        EMBED_CHANNELS=128
        ENCODER_CHANNELS=(128 128 128)
        ;;
esac

python -u "/data/projects/SFB_A03/jan/AT_TFBS/human301/$MODEL_SCRIPT" train \
    --cache "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/$TF/$DATA_RUN_ID/dataset.npz" \
    --output-dir "/data/projects/SFB_A03/jan/AT_TFBS/human301/runs/$TF/${SLURM_ARRAY_JOB_ID}_ablation_${VARIANT}_seed${SEED}" \
    --device cuda \
    --seed "$SEED" \
    --input-length 256 \
    --embed-channels "$EMBED_CHANNELS" \
    --encoder-channels "${ENCODER_CHANNELS[@]}" \
    --heads 4 \
    --ffn-channels 256 \
    --dropout "$DROPOUT" \
    --weight-decay "$WEIGHT_DECAY" \
    --classification-readout "$READOUT" \
    --negatives-per-positive 10 \
    --batch-size 88 \
    --rc-probability 0.5 \
    --eval-rc \
    --hard-negative-fraction 0 \
    --eval-batch-size 128 \
    --eval-crops multi \
    --epochs 100 \
    --patience 20 \
    --num-workers 2
