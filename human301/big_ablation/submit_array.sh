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
import time
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


def queue_jobs(root, env):
    rows = []
    for line in run(['squeue', '--me', '--noheader', '--format=%i|%j|%Z'], env).splitlines():
        fields = [field.strip() for field in line.split('|', 2)]
        if len(fields) == 3 and fields[2] and Path(fields[2]).resolve() == root:
            parent = fields[0].split('_', 1)[0]
            require(parent.isdigit(), f'Job-ID nicht eindeutig: {fields[0]}')
            rows.append((parent, fields[1]))
    return list(dict.fromkeys(rows))


def run_directory(root, name, row):
    require(row['tf'] and row['training_scope'] == 'single_tf',
            'Wiederaufnahme ist für die vorhandenen Single-TF-Konfigurationen vorgesehen.')
    path = root / 'results' / name / 'runs' / row['tf'] / ('train_' + row['train_assay']) / f"seed_{int(row['seed']):02d}"
    require(not (root / 'results' / name / 'runs').is_symlink(),
            f'Kein automatischer Neustart eines verlinkten runs-Ordners: {name}')
    require(path.resolve().is_relative_to((root / 'results' / name / 'runs').resolve()),
            f'Run-Pfad außerhalb der Konfiguration: {path}')
    require(not path.is_symlink(), f'Kein automatischer Neustart eines verlinkten Run-Ordners: {path}')
    return path


def completed(path, row, config):
    try:
        meta = json.loads((path / 'run.json').read_text())
    except (OSError, ValueError):
        return False
    if meta.get('status') == 'completed':
        require(meta.get('tf') == row['tf'] and meta.get('train_assay') == row['train_assay']
                and int(meta.get('seed', -1)) == int(row['seed']), f'Widersprüchliche Run-Metadaten: {path}')
        require((path / 'records.jsonl').is_file(), f'Abgeschlossener Run ohne records.jsonl: {path}')
        for assay in config['test_assays']:
            evaluation = path / 'evaluations' / 'standard' / assay
            try:
                metrics = json.loads((evaluation / 'metrics.json').read_text())
            except (OSError, ValueError):
                raise RuntimeError(f'Abgeschlossener Run mit fehlender/unlesbarer Auswertung: {evaluation}') from None
            require(metrics.get('status') == 'completed' and (evaluation / 'predictions.tsv').is_file(),
                    f'Abgeschlossener Run mit unvollständiger Auswertung: {evaluation}')
        return True
    return False


def main():
    root = Path(sys.argv[1]).resolve()
    arguments = sys.argv[2:]
    require(arguments, 'Aufruf: bash submit_array.sh [--restart] KONFIG [KONFIG ...] [--account=NAME]')
    require(not os.environ.get('SLURM_JOB_ID'), 'Den Launcher auf dem Login-Knoten starten.')
    manifest = root / 'configs' / 'caches.csv'
    # Keep the existing manifest-first calls from older launchers working.
    if arguments[0].lower().endswith('.csv'):
        manifest = Path(arguments.pop(0)).expanduser().resolve()
    configs = []
    config_specs = {}
    account_options = []
    restart = False
    for value in arguments:
        if value == '--restart':
            restart = True
        elif value.startswith('--account=') and value.split('=', 1)[1]:
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
    if restart:
        require(shutil.which('scancel'), 'scancel fehlt.')
    for executable in ('sbatch', 'squeue'):
        require(shutil.which(executable), f'{executable} fehlt.')
    require(manifest.is_file(), f'Cache-Manifest fehlt: {manifest}')
    if not restart:
        for name in configs:
            require(not (root / 'results' / name).exists(), f'results/{name} existiert bereits; für fehlende/unvollständige Runs --restart verwenden.')
    for filename in ('run_experiment.py', 'train_array.sbatch', 'collect_config.sbatch'):
        require((root / filename).is_file(), f'{filename} fehlt.')
    require('HUM301_FINISH_CONFIG' in (root / 'train_array.sbatch').read_text(),
            'Für die sofortige Zusammenfassung wird train_array.sbatch aus array_queue_immediate_summary.zip benötigt.')
    require('HUM301_COLLECT_FALLBACK' in (root / 'collect_config.sbatch').read_text(),
            'collect_config.sbatch aus array_queue_immediate_summary.zip verwenden.')
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
        requested = set(configs)
        previous_arrays = {}
        for previous_path in plans.glob('*/submission.json'):
            try:
                previous = json.loads(previous_path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(previous, dict) and previous.get('array_id') is not None:
                previous_arrays[str(previous['array_id'])] = set(previous.get('configs', []))
        stop = set()
        for job, name in queue_jobs(root, env):
            if name.startswith('h301_array_') or re.match(r'A\d+_', name):
                owned = previous_arrays.get(job)
                if owned is None:
                    owned = {config for config in configs if name == 'h301_array_' + config}
                require(restart and owned and owned <= requested,
                        f'Trainingsarray {job} ({name}) ist noch aktiv. Ein Neustart ersetzt nur vollständig ausgewählte Configs; andere Trainings werden nicht abgebrochen.')
                stop.add(job)
            elif name.startswith('h301_collect_') and name[len('h301_collect_'):] in requested:
                require(restart, f'Für diese Config ist noch Sammeljob {job} aktiv; --restart verwenden.')
                stop.add(job)
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
        planned_identities = set()
        for row in rows:
            name = Path(row['config_path']).stem
            require(int(row['seed']) in config_specs[name]['seeds'], 'Unerwarteter Seed im Aufgabenplan.')
            identity = (name, row['tf'], row['train_assay'], int(row['seed']))
            require(identity not in planned_identities, 'Doppelter Trainingslauf im Aufgabenplan.')
            planned_identities.add(identity)
        # Configs follow argument order; the existing planner defines TF/assay
        # priority within each config. There are no barriers between config blocks.
        config_rank = {name: index for index, name in enumerate(configs)}
        rows.sort(key=lambda row: config_rank[Path(row['config_path']).stem])
        if restart:
            for row in rows:
                name = Path(row['config_path']).stem
                run_directory(root, name, row)
                experiment = root / 'results' / name
                require(not experiment.is_symlink(), f'Kein automatischer Neustart eines verlinkten Ergebnisordners: {experiment}')
                completed(run_directory(root, name, row), row, config_specs[name])
            if stop:
                print('Beende bisherige Jobs der ausgewählten Configs: ' + ', '.join(sorted(stop, key=int)), flush=True)
                cancelled = subprocess.run(['scancel', *sorted(stop, key=int)], capture_output=True,
                                           text=True, env=env, timeout=90)
                deadline = time.monotonic() + 60
                while any(job in stop for job, name in queue_jobs(root, env)):
                    require(time.monotonic() < deadline, 'Jobs sind noch nicht vollständig beendet. Ergebnisse wurden noch nicht umgeordnet; später denselben Aufruf wiederholen. ' + cancelled.stderr.strip())
                    time.sleep(1)
        kept = {name: 0 for name in configs}
        pending = []
        archives = []
        for row in rows:
            name = Path(row['config_path']).stem
            if restart:
                path = run_directory(root, name, row)
                if completed(path, row, config_specs[name]):
                    kept[name] += 1
                    continue
                if path.exists():
                    archives.append((path, batch / 'aborted_runs' / name / path.relative_to(root / 'results' / name / 'runs')))
            pending.append(row)
        # Validate all completed metadata before moving any incomplete output.
        for source, destination in archives:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        if restart:
            shutil.copyfile(plan, batch / 'tasks.requested.csv')
            for name in {Path(row['config_path']).stem for row in pending}:
                for source in (root / 'results' / 'Results_all' / (name + '.csv'), root / 'results' / name / 'evaluation.md'):
                    if source.is_file():
                        destination = batch / 'previous_summaries' / name / source.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source), str(destination))
        rows = pending
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
        shutil.copyfile(root / 'configs/training_priority.csv', batch / 'training_priority.csv')
        if any(spec.get('task_profile') == 'short' for spec in config_specs.values()):
            shutil.copyfile(root / 'configs/training_priority_short.csv', batch / 'training_priority_short.csv')
        active_configs = [name for name in configs if indices[name]]
        state = dict(root=str(root), configs=active_configs, requested_configs=configs, mode='native_array', array_id=None,
                     created_at=datetime.now(timezone.utc).isoformat(), plan=str(plan),
                     task_count=len(rows), max_parallel=8, collectors={}, restart=restart,
                     replaced_jobs=sorted(stop, key=int), kept_runs=kept,
                     archived_runs=[str(destination) for source, destination in archives])
        state_path = batch / 'submission.json'
        atomic_json(state_path, state)
        for name in configs:
            if kept[name]:
                print(f'{name}: {kept[name]} abgeschlossene Runs erhalten; {len(indices[name])} Runs einzureichen.', flush=True)
            if not indices[name] and (root / 'results' / name / 'config.json').is_file():
                run([sys.executable, str(root / 'run_experiment.py'), 'collect', '--config', name], env)
        if not rows:
            print('Alle ausgewählten Runs sind bereits abgeschlossen. Keine Trainingsjobs eingereicht.', flush=True)
            return 0
        label = active_configs[0] if len(active_configs) == 1 else active_configs[0] + '_to_' + active_configs[-1]
        array_id = receipt(run(['sbatch', '--parsable', *account_options,
            f'--array=0-{len(rows)-1}%8', '--no-requeue', '--export=ALL',
            '--job-name=h301_array_' + label, f'--chdir={root}',
            f'--output={batch / "slurm" / "train_%A_%a.out"}',
            f'--error={batch / "slurm" / "train_%A_%a.err"}',
            str(root / 'train_array.sbatch'), str(plan), str(root)], env))
        state['array_id'] = array_id
        atomic_json(state_path, state)
        print(f'Jobarray {array_id}: {len(rows)} Trainingsaufgaben, maximal 8 gleichzeitig.', flush=True)
        print('Planreihenfolge: ' + ' -> '.join(active_configs), flush=True)
        print(f'Aufgabenplan und Logs: {batch}', flush=True)
        # Normal collection runs in the last finishing task. Retain a CPU fallback
        # for tasks that were cancelled or could not write their completion state.
        for name, selected in indices.items():
            if not selected:
                continue
            dependency = 'afterany:' + ':'.join(f'{array_id}_{index}' for index in selected)
            try:
                job = receipt(run(['sbatch', '--parsable', *account_options, '--no-requeue', '--export=ALL',
                    '--dependency=' + dependency, '--kill-on-invalid-dep=yes',
                    '--job-name=h301_collect_' + name, f'--chdir={root}',
                    f'--output={batch / "slurm" / ("collect_" + name + "_%j.out")}',
                    str(root / 'collect_config.sbatch'), str(root), name,
                    str(batch / f'completion_{array_id}_{name}.json')], env))
                state['collectors'][name] = job
            except RuntimeError as exc:
                state['collection_error'] = str(exc)
                atomic_json(state_path, state)
                print(f'Das Trainingsarray {array_id} ist bereits eingereicht. Sammlung für {name} fehlgeschlagen: {exc}', file=sys.stderr)
                print(f'Nach dem Training: python run_experiment.py collect --config {name}', file=sys.stderr)
                return 1
            atomic_json(state_path, state)
            print(f'{name}: {len(selected)} Aufgaben; Zusammenfassung im letzten Lauf, CPU-Absicherung als Job {job}.', flush=True)
        return 0


try:
    raise SystemExit(main())
except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
    print(f'Abgebrochen: {exc}', file=sys.stderr)
    raise SystemExit(1)
HUM301_NATIVE_ARRAY
