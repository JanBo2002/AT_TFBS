#!/usr/bin/env bash
# Auf dem Login-Node starten: bash hum301_motifhead_submit.sh
# Der CPU-Job prueft die Daten und reicht danach ein GPU-Array mit 36 Tasks
# sowie die abschliessende CPU-Auswertung mit der Alpha-Suche ein.
# Die bestehenden Modell-, Daten- und Cross-Assay-Skripte bleiben unveraendert.

set -euo pipefail
trap 'head_exit_status=$?; printf "Error: Motivkopf-Launcher fehlgeschlagen (Exit %s).\n" "$head_exit_status" >&2; exit "$head_exit_status"' ERR

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf 'Error: Bitte auf dem Login-Node mit bash hum301_motifhead_submit.sh starten.\n' >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    printf 'Error: sbatch ist nicht verfuegbar.\n' >&2
    exit 1
fi

head_script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
head_worker="$head_script_directory/hum301_motifhead.sbatch"
head_parallel_models="${HUM301_HEAD_MAX_PARALLEL:-6}"
if [[ ! "$head_parallel_models" =~ ^[0-9]+$ ]] || \
   (( head_parallel_models < 1 || head_parallel_models > 36 )); then
    printf 'Error: HUM301_HEAD_MAX_PARALLEL muss zwischen 1 und 36 liegen.\n' >&2
    exit 1
fi
for head_required in hum301_motif64_motifhead.py hum301_motifhead.sbatch hum301_data.py hum301_crossassay.py; do
    if [[ ! -f "$head_script_directory/$head_required" ]]; then
        printf 'Error: Datei fehlt: %s/%s\n' "$head_script_directory" "$head_required" >&2
        exit 1
    fi
done

# Auf dem Login-Node wird nur sbatch aufgerufen; at_env aktiviert der CPU-Job.
head_response="$(sbatch --parsable \
    --job-name=hum301-head-data --ntasks=1 --cpus-per-task=4 \
    --mem=32G --time=04:00:00 --chdir="$head_script_directory" \
    --output="$head_script_directory/hum301-motifhead-data-%j.out" \
    --error="$head_script_directory/hum301-motifhead-data-%j.err" \
    --export="ALL,HUM301_HEAD_STAGE=setup,HUM301_HEAD_MAX_PARALLEL=$head_parallel_models" \
    "$head_worker")"
head_job_id="${head_response%%;*}"
if [[ ! "$head_job_id" =~ ^[0-9]+$ ]]; then
    printf 'Error: Ungueltige sbatch-Antwort: %s\n' "$head_response" >&2
    exit 1
fi
printf 'CPU-Datenpruefung: %s\n' "$head_job_id"
printf 'Danach: 36 Modelle, bis zu %s parallel; Tasknummern in TF-Reihenfolge, keine TF-Dependencies.\n' "$head_parallel_models"
printf 'Anschliessend: Alpha-Auswertung nach allen 72 Tests.\n'
printf 'Ergebnisse und Logs: %s/runs/motifhead/<RUN_ID>/\n' "$head_script_directory"
