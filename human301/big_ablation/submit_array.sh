#!/usr/bin/env bash
set -euo pipefail
ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ABLATION_PYTHON:-python3}" - "$ABLATION_ROOT" "$@" <<'HUM301_NATIVE_ARRAY'
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(arguments, env):
    result = subprocess.run(arguments, capture_output=True, text=True, env=env, timeout=90)
    require(result.returncode == 0, f'{arguments[0]} fehlgeschlagen: ' + (result.stderr.strip() or result.stdout.strip()))
    return result.stdout.strip()


def receipt(output):
    require(re.fullmatch(r'[0-9]+(?:;[^\s;]+)?', output) is not None,
            f'SLURM-Job-ID nicht eindeutig bestätigt: {output!r}. Vor erneutem Start die Queue prüfen.')
    return output.split(';', 1)[0]


def main():
    root = Path(sys.argv[1]).resolve()
    arguments = sys.argv[2:]
    require(arguments, 'Aufruf: bash submit_array.sh KONFIG [KONFIG ...] [--account=NAME]')
    require(not os.environ.get('SLURM_JOB_ID'), 'Den Launcher auf dem Login-Knoten starten.')
    manifest = root / 'configs' / 'caches.csv'
    # Keep the existing manifest-first calls from older launchers working.
    if arguments[0].lower().endswith('.csv'):
        manifest = Path(arguments.pop(0)).expanduser().resolve()
    configs = []
    config_specs = {}
    account_options = []
    for value in arguments:
        if value.startswith('--account=') and value.split('=', 1)[1]:
            account_options.append(value)
        else:
            require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is not None,
                    'Konfigurationsnamen aus configs/ und optional --account=NAME angeben.')
            config_path = root / 'configs' / (value + '.json')
            require(config_path.is_file(), f'Konfiguration fehlt: {config_path}')
            config_specs[value] = json.loads(config_path.read_text())
            require(config_specs[value]['name'] == value,
                    f'Konfigurationsname und Dateiname stimmen nicht überein: {config_path}')
            configs.append(value)
    require(configs and len(configs) == len(set(configs)), 'Konfigurationen müssen angegeben und eindeutig sein.')
    require(len(account_options) <= 1, 'Nur einen Account angeben.')
    for executable in ('sbatch', 'squeue'):
        require(shutil.which(executable), f'{executable} fehlt.')
    require(manifest.is_file(), f'Cache-Manifest fehlt: {manifest}')
    for name in configs:
        require(not (root / 'results' / name).exists(), f'results/{name} existiert bereits; kein Überschreiben vorhandener Läufe.')
    for filename in ('run_experiment.py', 'train_array.sbatch', 'collect_config.sbatch'):
        require((root / filename).is_file(), f'{filename} fehlt.')
    plans = root / 'plans'
    plans.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('SBATCH_') and key not in {'SLURM_CLUSTERS', 'SLURM_FEDERATION'}}
    env.update(LC_ALL='C', ABLATION_PYTHON=sys.executable)
    with (plans / '.dispatcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Der alte Hintergrund-Launcher oder ein anderer Start läuft noch.') from None
        queue = run(['squeue', '--me', '--noheader', '--format=%i|%j|%Z'], env)
        for line in queue.splitlines():
            fields = line.split('|', 2)
            if len(fields) == 3:
                job, name, directory = (field.strip() for field in fields)
                if directory and Path(directory).resolve() == root and (name.startswith('h301_array_') or re.match(r'A\d+_', name)):
                    raise RuntimeError(f'Für dieses Projekt ist noch ein Trainingsjob/Array aktiv: {job}. '
                                       'Mehrere Konfigurationen gemeinsam in einem Aufruf angeben.')
        batch = Path(tempfile.mkdtemp(prefix='batch_array_', dir=plans)).resolve()
        (batch / 'slurm').mkdir()
        plan = batch / 'tasks.csv'
        # Reuse the existing planner and all existing model/training settings.
        run([sys.executable, str(root / 'run_experiment.py'), 'plan', '--config', *configs,
             '--manifest', str(manifest), '--output', str(plan)], env)
        with plan.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        require(rows, 'Der Aufgabenplan enthält keine Trainingsläufe.')
        for row in rows:
            require(Path(row['config_path']).stem in config_specs, 'Unbekannte Konfiguration im Aufgabenplan.')
        if all(config_specs[name].get('task_profile', 'long') for name in configs):
            # Profile-controlled arrays follow TF/assay priority across configs.
            profiles = {config_specs[name].get('task_profile', 'long') for name in configs}
            filename = 'training_priority_short.csv' if profiles == {'short'} else 'training_priority.csv'
            with (root / 'configs' / filename).open(newline='') as handle:
                ranks = {(row['tf'], {'CHS': 'CHIP'}.get(row['assay'], row['assay'])): int(row['priority'])
                         for row in csv.DictReader(handle)}
            require(all((row['tf'], row['train_assay']) in ranks for row in rows),
                    'Der Aufgabenplan enthält eine Kombination ohne Profilpriorität.')
            config_rank = {name: index for index, name in enumerate(configs)}
            rows.sort(key=lambda row: (ranks[(row['tf'], row['train_assay'])],
                                       config_rank[Path(row['config_path']).stem], int(row['seed'])))
            with plan.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        indices = {name: [] for name in configs}
        seen = set()
        for index, row in enumerate(rows):
            config = json.loads(Path(row['config_path']).read_text())['name']
            require(config in indices and int(row['seed']) in config_specs[config]['seeds'],
                    'Unerwartete Konfiguration oder Seed im Aufgabenplan.')
            identity = (config, row['tf'], row['train_assay'], int(row['seed']))
            require(identity not in seen, 'Doppelter Trainingslauf im Aufgabenplan.')
            seen.add(identity)
            indices[config].append(index)
        require(all(indices.values()), 'Eine Konfiguration hat keine Trainingsaufgaben.')
        shutil.copyfile(root / 'configs/training_priority.csv', batch / 'training_priority.csv')
        if any(spec.get('task_profile') == 'short' for spec in config_specs.values()):
            shutil.copyfile(root / 'configs/training_priority_short.csv', batch / 'training_priority_short.csv')
        state = dict(root=str(root), configs=configs, mode='native_array', array_id=None,
                     created_at=datetime.now(timezone.utc).isoformat(), plan=str(plan),
                     task_count=len(rows), max_parallel=8, collectors={})
        state_path = batch / 'submission.json'
        atomic_json(state_path, state)
        array_id = receipt(run(['sbatch', '--parsable', *account_options,
            f'--array=0-{len(rows)-1}%8', '--no-requeue', '--export=ALL',
            '--job-name=h301_array_' + '_'.join(configs), f'--chdir={root}',
            f'--output={batch / "slurm" / "train_%A_%a.out"}',
            f'--error={batch / "slurm" / "train_%A_%a.err"}',
            str(root / 'train_array.sbatch'), str(plan), str(root)], env))
        state['array_id'] = array_id
        atomic_json(state_path, state)
        print(f'Jobarray {array_id}: {len(rows)} Trainingsaufgaben, maximal 8 gleichzeitig.', flush=True)
        print(f'Aufgabenplan und Logs: {batch}', flush=True)
        # Only CPU result collection has a dependency; training tasks have none.
        for name, selected in indices.items():
            dependency = 'afterany:' + ':'.join(f'{array_id}_{index}' for index in selected)
            try:
                job = receipt(run(['sbatch', '--parsable', *account_options, '--no-requeue', '--export=ALL',
                    '--dependency=' + dependency, '--kill-on-invalid-dep=yes',
                    '--job-name=h301_collect_' + name, f'--chdir={root}',
                    f'--output={batch / "slurm" / ("collect_" + name + "_%j.out")}',
                    str(root / 'collect_config.sbatch'), str(root), name], env))
                state['collectors'][name] = job
            except RuntimeError as exc:
                state['collection_error'] = str(exc)
                atomic_json(state_path, state)
                print(f'Das Trainingsarray {array_id} ist bereits eingereicht. Sammlung für {name} fehlgeschlagen: {exc}', file=sys.stderr)
                print(f'Nach dem Training: python run_experiment.py collect --config {name}', file=sys.stderr)
                return 1
            atomic_json(state_path, state)
            print(f'{name}: {len(selected)} Aufgaben; automatische CPU-Zusammenfassung als Job {job}.', flush=True)
        return 0


try:
    raise SystemExit(main())
except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
    print(f'Abgebrochen: {exc}', file=sys.stderr)
    raise SystemExit(1)
HUM301_NATIVE_ARRAY
