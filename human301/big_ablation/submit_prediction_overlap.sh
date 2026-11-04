#!/usr/bin/env bash
# Version: 1.0.3

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

analysis_script="$script_dir/compare_prediction_overlap.py"
worker_script="$script_dir/run_prediction_overlap.sbatch"

own_root=${OWN_ROOT:-/data/projects/SFB_A03/jan/AT_TFBS/human301/big_ablation/results}
reference_root=${REFERENCE_ROOT:-/data/projects/SFB_A03/clara/ArchThalia-ML/30tf_output}
output_dir=${OUTPUT_DIR:-"$own_root/prediction_overlap"}

seed=${SEED:-13}
max_parallel=${MAX_PARALLEL:-4}

if [[ ! "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL muss eine positive ganze Zahl sein." >&2
    exit 2
fi

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "SEED muss eine nichtnegative ganze Zahl sein." >&2
    exit 2
fi

for file in "$analysis_script" "$worker_script"; do
    if [[ ! -f "$file" ]]; then
        echo "Script fehlt: $file" >&2
        exit 2
    fi
done

for directory in "$own_root" "$reference_root"; do
    if [[ ! -d "$directory" ]]; then
        echo "Eingabe-Hauptordner fehlt: $directory" >&2
        exit 2
    fi
done

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch wurde nicht gefunden." >&2
    exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin=$(command -v "$PYTHON_BIN") || {
        echo "Python nicht gefunden: $PYTHON_BIN" >&2
        exit 2
    }
else
    conda_sh=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}

    if [[ ! -f "$conda_sh" ]]; then
        echo "Conda-Aktivierung nicht gefunden: $conda_sh" >&2
        echo "Alternativ PYTHON_BIN auf einen passenden Interpreter setzen." >&2
        exit 2
    fi

    source "$conda_sh"
    conda activate "${CONDA_ENV:-at_env}"
    python_bin=$(command -v python)
fi

echo "Python: $python_bin"

"$python_bin" "$analysis_script" --self-test

mapfile -t available_tfs < <(
    "$python_bin" "$analysis_script" --list-tfs
)

if (( ${#available_tfs[@]} == 0 )); then
    echo "Keine TFs gefunden." >&2
    exit 2
fi

declare -A known_tfs=()
declare -A seen_tfs=()

for tf in "${available_tfs[@]}"; do
    known_tfs["$tf"]=1
done

if (( $# > 0 )); then
    tfs=("$@")
else
    tfs=("${available_tfs[@]}")
fi

for tf in "${tfs[@]}"; do
    if [[ -z "${known_tfs[$tf]:-}" ]]; then
        echo "Unbekannter TF: $tf" >&2
        exit 2
    fi
    if [[ -n "${seen_tfs[$tf]:-}" ]]; then
        echo "TF mehrfach angegeben: $tf" >&2
        exit 2
    fi
    seen_tfs["$tf"]=1
done

# Prüft alle zugelassenen Eingabepfade und aktualisiert missing_files.csv.
# Fehlende Dateien verhindern die Einreichung nicht.
"$python_bin" "$analysis_script" \
    --check-files \
    --own-root "$own_root" \
    --reference-root "$reference_root" \
    --output-dir "$output_dir" \
    --seed "$seed"

mkdir -p "$output_dir/logs"

last_index=$((${#tfs[@]} - 1))

job_id=$(sbatch \
    --parsable \
    --export=ALL \
    --array="0-${last_index}%${max_parallel}" \
    --output="$output_dir/logs/overlap_%A_%a.log" \
    "$worker_script" \
    "$python_bin" \
    "$analysis_script" \
    "$own_root" \
    "$reference_root" \
    "$output_dir" \
    "$seed" \
    "${tfs[@]}")

echo "Eingereicht: $job_id"
echo "TF-Tasks: ${#tfs[@]}, maximal gleichzeitig: $max_parallel"
echo "Je Task: alle zugelassenen Modellvergleiche für einen TF."
echo "Ergebnisse: $output_dir"
