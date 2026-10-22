#!/usr/bin/env bash
#SBATCH --job-name=hum301_collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch collect_all.sbatch /absolute/path/to/big_ablation" >&2
    exit 2
fi
ABLATION_ROOT="$1"
ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"
"$ABLATION_PYTHON" "$ABLATION_ROOT/run_experiment.py" collect --all-configs
