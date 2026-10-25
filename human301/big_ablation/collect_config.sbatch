#!/usr/bin/env bash
#SBATCH --job-name=hum301_collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
if [[ $# -ne 2 || ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Aufruf: collect_config.sbatch /absolut/big_ablation KONFIG" >&2
    exit 2
fi
ABLATION_ROOT="$1"
if [[ ! -f "$ABLATION_ROOT/configs/$2.json" ]]; then
    echo "Konfiguration fehlt: $ABLATION_ROOT/configs/$2.json" >&2
    exit 2
fi
"${ABLATION_PYTHON:-python3}" "$ABLATION_ROOT/run_experiment.py" collect --config "$2"
