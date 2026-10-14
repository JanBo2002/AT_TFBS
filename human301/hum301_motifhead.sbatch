#!/usr/bin/env bash
#SBATCH --job-name=hum301-head-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/human301
#SBATCH --output=hum301-motifhead-data-%j.out
#SBATCH --error=hum301-motifhead-data-%j.err

# Diese Datei und hum301_motif64_motifhead.py neben den vorhandenen Skripten ablegen.
# Start: sbatch hum301_motifhead.sbatch
# CPU-Pruefung -> 18 geordnete GPU-Arrays -> CPU-Auswertung mit α-Suche.
# Pro TF laufen CHIP und GHTS parallel; der naechste TF wartet auf beide Tests.
# Alle Ergebnisse liegen unter runs/motifhead/<RUN_ID>/.
# Optionale Einstellungen: HUM301_HEAD_LOSS_WEIGHT, HUM301_HEAD_ALPHA_STEP, HUM301_HEAD_SEED.

set -euo pipefail
trap 'head_exit_status=$?; printf "Error: Motivkopf-Job fehlgeschlagen (Exit %s).\n" "$head_exit_status" >&2; exit "$head_exit_status"' ERR
: "${SLURM_JOB_ID:?Error: Bitte direkt mit sbatch hum301_motifhead.sbatch starten}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate at_env
export PYTHONUNBUFFERED=1

head_code_directory="/data/projects/SFB_A03/jan/AT_TFBS/human301"
head_model="$head_code_directory/hum301_motif64_motifhead.py"
if [[ ! -f "$head_model" ]]; then
    printf 'Error: Modellskript fehlt: %s\n' "$head_model" >&2
    exit 1
fi

case "${HUM301_HEAD_STAGE:-setup}" in
    setup)
        head_run_id="motifhead_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
        head_manifest="$head_code_directory/runs/motifhead/$head_run_id/workflow.json"
        python -u "$head_model" workflow-plan --manifest "$head_manifest" \
            --project-root "$(dirname "$head_code_directory")" \
            --seed "${HUM301_HEAD_SEED:-1}" \
            --motif-loss-weight "${HUM301_HEAD_LOSS_WEIGHT:-1.0}" \
            --alpha-step "${HUM301_HEAD_ALPHA_STEP:-0.01}"
        # Slurm hat die CPU-Logs vor Skriptstart geoeffnet; der geoeffnete Dateideskriptor
        # schreibt nach dem Verschieben weiter in den jetzt vorhandenen Laufordner.
        for head_extension in out err; do
            head_log="$head_code_directory/hum301-motifhead-data-${SLURM_JOB_ID}.$head_extension"
            if [[ -f "$head_log" ]]; then
                mv -- "$head_log" "$(dirname "$head_manifest")/logs/data/"
            fi
        done
        python -u "$head_model" workflow-prepare --manifest "$head_manifest"
        python -u "$head_model" workflow-submit --manifest "$head_manifest"
        ;;
    train)
        : "${HUM301_HEAD_MANIFEST:?Error: Manifest fehlt}"
        : "${HUM301_HEAD_TF_INDEX:?Error: TF-Index fehlt}"
        : "${SLURM_ARRAY_TASK_ID:?Error: Array-Task-ID fehlt}"
        if [[ ! "$HUM301_HEAD_TF_INDEX" =~ ^([0-9]|1[0-7])$ ]] || \
           [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]; then
            printf 'Error: Ungueltige TF- oder Array-Task-ID.\n' >&2
            exit 1
        fi
        head_task_id=$((2 * HUM301_HEAD_TF_INDEX + SLURM_ARRAY_TASK_ID))
        python -u "$head_model" workflow-train --manifest "$HUM301_HEAD_MANIFEST" \
            --task-id "$head_task_id" --device cuda
        ;;
    summary)
        : "${HUM301_HEAD_MANIFEST:?Error: Manifest fehlt}"
        python -u "$head_model" workflow-summarize --manifest "$HUM301_HEAD_MANIFEST"
        ;;
    *)
        printf 'Error: Unbekannte Stage: %s\n' "$HUM301_HEAD_STAGE" >&2
        exit 1
        ;;
esac
