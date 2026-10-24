#!/usr/bin/env bash
set -euo pipefail
ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash submit_A02_kernel_7_11_15_21.sh configs/caches.csv [sbatch resource/account options...]" >&2
    exit 2
fi
ABLATION_MANIFEST="$1"
shift
exec "$ABLATION_PYTHON" "$ABLATION_ROOT/run_experiment.py" submit start \
    --manifest "$ABLATION_MANIFEST" --config A02_kernel_7_11_15_21 -- "$@"
