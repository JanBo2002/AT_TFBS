#!/usr/bin/env bash
# Ordered configurations, sharing eight unfinished training-job slots.
set -euo pipefail
ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"
# Explicit config selection uses the existing native array launcher (eight slots).
# Legacy manifest/resource-only calls below retain their original behavior.
if [[ "${1:-}" == "--config" ]]; then
    shift
    exec bash "$ABLATION_ROOT/submit_array.sh" "$@"
fi
if [[ "${1:-}" == *.csv && "${2:-}" == "--config" ]]; then
    ABLATION_MANIFEST="$1"
    shift 2
    exec bash "$ABLATION_ROOT/submit_array.sh" "$ABLATION_MANIFEST" "$@"
fi
if [[ $# -lt 1 ]]; then
    echo "Usage: bash submit.sh --config CONFIG [CONFIG ...] [--account=NAME]" >&2
    echo "Legacy: bash submit.sh configs/caches.csv [resource/account sbatch options...]" >&2
    exit 2
fi
ABLATION_MANIFEST="$1"
shift
exec "$ABLATION_PYTHON" "$ABLATION_ROOT/run_experiment.py" submit start \
    --manifest "$ABLATION_MANIFEST" \
    --config 00_basic A01_kernel_LegNetMax_3_6_9_12 A02_kernel_7_11_15_21 -- "$@"
