#!/usr/bin/env bash
#SBATCH --job-name=hum301_array
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
set -euo pipefail
if [[ $# -ne 2 || ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "Dieses Skript wird als SLURM-Jobarray mit Aufgabenplan und Projektpfad gestartet." >&2
    exit 2
fi
ABLATION_PLAN="$1"
ABLATION_ROOT="$2"
"${ABLATION_PYTHON:-python3}" - "$ABLATION_ROOT" "$ABLATION_PLAN" "$SLURM_ARRAY_TASK_ID" <<'HUM301_FINISH_CONFIG'
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(root, plan, index):
    root, plan, index = Path(root).resolve(), Path(plan).resolve(), int(index)
    array_id = os.environ.get('SLURM_ARRAY_JOB_ID', '')
    if not array_id.isdigit():
        raise RuntimeError('SLURM_ARRAY_JOB_ID fehlt; als natives Jobarray starten.')
    with plan.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows):
        raise IndexError('Task-Index liegt außerhalb des Aufgabenplans.')
    name = Path(rows[index]['config_path']).stem
    expected = {str(i) for i, row in enumerate(rows) if Path(row['config_path']).stem == name}
    result = subprocess.run([sys.executable, str(root / 'run_experiment.py'), 'task',
                             '--plan', str(plan), '--index', str(index)])
    status = result.returncode
    completion = plan.parent / f'completion_{array_id}_{name}.json'
    with completion.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(completion.read_text()) if completion.exists() else {
            'array_id': array_id, 'config': name, 'task_results': {}, 'summary_status': 'waiting'}
        state['task_results'][str(index)] = status
        atomic_json(completion, state)
        if expected <= state['task_results'].keys() and state['summary_status'] != 'completed':
            print(f'{name}: letzter Lauf beendet; Zusammenfassung wird jetzt erstellt.', flush=True)
            collected = subprocess.run([sys.executable, str(root / 'run_experiment.py'),
                                        'collect', '--config', name])
            outputs = (root / 'results' / 'Results_all' / (name + '.csv'),
                       root / 'results' / name / 'evaluation.md')
            success = collected.returncode == 0 and all(path.is_file() for path in outputs)
            state['summary_status'] = 'completed' if success else 'failed'
            state['summary_returncode'] = collected.returncode if success else collected.returncode or 1
            atomic_json(completion, state)
            if not success:
                print(f'{name}: sofortige Zusammenfassung fehlgeschlagen; CPU-Sammeljob bleibt als Absicherung.',
                      file=sys.stderr, flush=True)
                if status == 0:
                    status = state['summary_returncode']
    return status if status >= 0 else 128 - status


raise SystemExit(main(*sys.argv[1:]))
HUM301_FINISH_CONFIG
