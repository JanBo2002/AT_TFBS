#!/usr/bin/env bash
# Auf dem Login-Knoten starten. Python ruft sbatch selbst auf.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate at_env
elif [[ "${CONDA_DEFAULT_ENV:-}" != "at_env" ]]; then
    echo "Bitte at_env aktivieren; ~/miniforge3/etc/profile.d/conda.sh fehlt." >&2
    exit 1
fi
exec python3 "$task_dir/pwm_auswahl.py" submit "$@"
