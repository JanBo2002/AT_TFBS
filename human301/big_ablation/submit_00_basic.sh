#!/usr/bin/env bash
set -euo pipefail
ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash submit_00_basic.sh configs/caches.csv [sbatch resource/account options...]" >&2
    exit 2
fi
ABLATION_MANIFEST="$1"
shift
exec "$ABLATION_PYTHON" "$ABLATION_ROOT/submit_jobs.py" start \
    --manifest "$ABLATION_MANIFEST" --config 00_basic -- "$@"
