#!/usr/bin/env bash
#SBATCH --job-name=hum301_collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Aufruf: collect_config.sbatch /absolut/big_ablation KONFIG [STATUS_JSON]" >&2
    exit 2
fi
ABLATION_ROOT="$1"
if [[ ! -f "$ABLATION_ROOT/configs/$2.json" ]]; then
    echo "Konfiguration fehlt: $ABLATION_ROOT/configs/$2.json" >&2
    exit 2
fi
"${ABLATION_PYTHON:-python3}" - "$ABLATION_ROOT" "$2" "${3:-}" <<'HUM301_COLLECT_FALLBACK'
import json
from pathlib import Path
import subprocess
import sys


def main(root, name, completion):
    root = Path(root).resolve()
    if completion:
        path = Path(completion)
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
        outputs = (root / 'results' / 'Results_all' / (name + '.csv'),
                   root / 'results' / name / 'evaluation.md')
        if state.get('summary_status') == 'completed' and all(output.is_file() for output in outputs):
            print(f'{name}: Zusammenfassung wurde bereits direkt im letzten Lauf erstellt.')
            return 0
    return subprocess.run([sys.executable, str(root / 'run_experiment.py'), 'collect', '--config', name]).returncode


raise SystemExit(main(*sys.argv[1:]))
HUM301_COLLECT_FALLBACK
