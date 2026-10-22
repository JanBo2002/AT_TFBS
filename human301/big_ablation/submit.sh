#!/usr/bin/env bash
# Ordered configurations, sharing eight unfinished training-job slots.
set -euo pipefail
ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash submit.sh configs/caches.csv [resource/account sbatch options...]" >&2
    exit 2
fi
ABLATION_MANIFEST="$1"
shift
exec "$ABLATION_PYTHON" "$ABLATION_ROOT/submit_jobs.py" start \
    --manifest "$ABLATION_MANIFEST" \
    --config 00_basic A01_kernel_LegNetMax_3_6_9_12 A02_kernel_7_11_15_21 -- "$@"
