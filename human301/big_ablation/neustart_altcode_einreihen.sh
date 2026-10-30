#!/usr/bin/env bash
# Bereits gesicherten alten Code jetzt hinter der bestehenden Queue einreihen.
# Aufruf: bash neustart_altcode_einreihen.sh results/Code_snapshots/<Sicherung>
set -euo pipefail
HUM301_ENQUEUE_SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
exec "${ABLATION_PYTHON:-python3}" - "$HUM301_ENQUEUE_SCRIPT" "$@" <<'HUM301_ENQUEUE_FROZEN'
import argparse
import ast
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import zipfile

controller = Path(sys.argv[1]).resolve()
env = {key: value for key, value in os.environ.items()
       if not key.startswith('SBATCH_') and key not in {
           'SLURM_CLUSTERS', 'SLURM_FEDERATION', 'ABLATION_ALLOW_CODE_VERSION_MISMATCH'}}
env.update(LC_ALL='C', ABLATION_PYTHON=sys.executable)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command, cwd, *, capture=True):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            capture_output=capture, timeout=180)
    require(result.returncode == 0, f'{command[0]} fehlgeschlagen: '
            + (result.stderr or result.stdout or '').strip())
    return result.stdout.strip() if capture else ''


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def code_hash(root):
    result = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        if 'results' in path.relative_to(root).parts or '__pycache__' in path.parts:
            continue
        result.update(str(path.relative_to(root)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def queue(root):
    rows = []
    output = run(['squeue', '--me', '--array', '--noheader', '--format=%i|%j|%T|%Z|%r'], root)
    for line in output.splitlines():
        fields = [part.strip() for part in line.split('|', 4)]
        require(len(fields) == 5, f'Unklare Queue-Zeile: {line!r}')
        job, name, status, directory, reason = fields
        parent = job.split('_', 1)[0]
        require(parent.isdigit(), 'Unklare Job-ID: ' + job)
        rows.append(dict(job=job, parent=parent, name=name, state=status,
                         directory=str(Path(directory).resolve()) if directory else '', reason=reason))
    return rows


def training_job(name):
    return name.startswith(('h301_array_', 'hum301_array')) or re.match(r'A\d+_', name)


def native_helpers(frozen):
    # Reuse the saved launcher's own result checks and result-path rules.
    source = (frozen / 'submit_array.sh').read_text()
    marker = "<<'HUM301_NATIVE_ARRAY'\n"
    require(source.count(marker) == 1, 'Gesicherter nativer Array-Launcher nicht erkannt.')
    body = source.split(marker, 1)[1].rsplit('\nHUM301_NATIVE_ARRAY', 1)[0]
    names = {'run_directory', 'completed'}
    definitions = [node for node in ast.parse(body).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in definitions} == names, 'Ergebnisprüfung fehlt im gesicherten Launcher.')
    namespace = dict(Path=Path, json=json, require=require)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<saved result checks>', 'exec'), namespace)
    return namespace['run_directory'], namespace['completed']


def check_versions(root, configs, expected):
    for name in configs:
        experiment = root / 'results' / name
        require(not experiment.is_symlink(), f'Verlinkter Config-Ergebnisordner: {experiment}')
        versions = experiment / 'versions.txt'
        if versions.exists():
            hashes = re.findall(r'^code_sha256:\s*(\S+)', versions.read_text(), re.M)
            require(not hashes or hashes[-1] == expected, f'{name}: vorhandene Code-Kennung weicht von der Sicherung ab.')
        for path in (experiment / 'runs').glob('**/run.json'):
            meta = json.loads(path.read_text())
            if meta.get('status') in {'running', 'completed'} and meta.get('code_hash'):
                require(meta['code_hash'] == expected, f'Abweichende Code-Kennung: {path}')


def restore(snapshot):
    archive = Path(snapshot['archive'])
    require(file_hash(archive) == snapshot['archive_hash'], 'Gesichertes Codearchiv wurde verändert.')
    root = Path(snapshot['root']).resolve()
    token = archive.parent.name
    frozen = root.with_name(root.name + '_frozen_' + token)
    external = root.with_name(root.name + '_frozen_external_' + token)
    frozen.mkdir(exist_ok=True)
    external.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            relative = PurePosixPath(entry.filename)
            require(not relative.is_absolute() and '..' not in relative.parts
                    and len(relative.parts) > 1 and relative.parts[0] in {'project', 'external'},
                    'Ungültiger Archivpfad: ' + entry.filename)
            require(not entry.is_dir(), 'Unerwarteter Verzeichniseintrag im Archiv.')
            base = frozen if relative.parts[0] == 'project' else external
            target = base.joinpath(*relative.parts[1:])
            require(not any(parent.is_symlink() for parent in [target, *target.parents]
                            if parent.is_relative_to(base)), 'Verlinkter Pfad im Codearchiv: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read(entry)
            if target.exists():
                require(target.read_bytes() == content, 'Vorhandene gefrorene Datei wurde verändert: ' + str(target))
            else:
                target.write_bytes(content)
    require(code_hash(frozen) == snapshot['source_hash'], 'Gefrorener Python-Code stimmt nicht mit der Sicherung überein.')
    for name, expected in snapshot.get('external_hashes', {}).items():
        require(file_hash(external / name) == expected, 'Externe Factory weicht von der Sicherung ab: ' + name)
    results = frozen / 'results'
    if not results.exists():
        results.symlink_to(root / 'results', target_is_directory=True)
    require(results.is_symlink() and results.resolve() == (root / 'results').resolve(), 'Falscher Ergebnisordner im alten Code.')
    for directory in root.iterdir():
        if (directory.is_dir() and directory.name not in {'results', 'plans', 'configs', '__pycache__'}
                and not (frozen / directory.name).exists()):
            (frozen / directory.name).symlink_to(directory, target_is_directory=True)
    return frozen, external


def use_frozen_environment(state):
    paths = [str(Path(state['external_root']) / name) for name in state.get('external_paths', [])]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)


def verify_saved_files(state):
    require(file_hash(Path(state['archive'])) == state['archive_hash'], 'Gesichertes Codearchiv wurde verändert.')
    require(code_hash(Path(state['frozen_root'])) == state['source_hash'], 'Gefrorener Python-Code wurde verändert.')
    for path, expected in state['script_hashes'].items():
        require(file_hash(Path(path)) == expected, 'Gesichertes Jobskript wurde verändert: ' + path)
    for name, expected in state.get('external_hashes', {}).items():
        require(file_hash(Path(state['external_root']) / name) == expected, 'Gesicherte externe Factory wurde verändert: ' + name)
    check_versions(Path(state['queue_root']), state['configs'], state['source_hash'])


def read_plan(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_plan(path, fields, rows):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def job_script(path, resource_source, saved_controller, mode, state_path, *extra):
    # Keep the saved CPU job's resource header. No additional Python module is created.
    header = []
    for line in resource_source.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            break
        header.append(line)
    require(header and header[0].startswith('#!'), 'Ungültiger gesicherter CPU-Job-Header.')
    body = '\n'.join(header) + '\nset -euo pipefail\nexec bash ' + shlex.quote(str(saved_controller))
    body += ' ' + mode + ' ' + shlex.quote(str(state_path))
    body += ''.join(' ' + shlex.quote(str(value)) for value in extra) + '\n'
    path.write_text(body)
    run(['bash', '-n', str(path)], path.parent)


def submit(args):
    require(not os.environ.get('SLURM_JOB_ID'), 'Auf dem Login-Knoten mit bash ausführen.')
    for program in ('sbatch', 'squeue', 'scontrol', 'scancel'):
        require(shutil.which(program), program + ' fehlt.')
    selected = args.snapshot.expanduser().resolve()
    directory = selected if selected.is_dir() else selected.parent
    manifest = directory / 'manifest.json'
    snapshot = json.loads(manifest.read_text())
    root = Path(snapshot['root']).resolve()
    batch = root / 'plans' / ('frozen_restart_' + directory.name)
    state_path = batch / 'submission.json'
    if args.status:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            print('Status: ' + state['phase'])
            print('Vorbereitung: ' + str(state.get('prepare_id')))
            print('Trainingsarray: ' + str(state.get('array_id')))
            print('Sammler: ' + ', '.join(state.get('collectors', {}).values()))
            print('Logs: ' + str(batch / 'slurm'))
        else:
            print('Für diese Sicherung ist noch kein Neustart eingereiht.')
        return
    require(not args.account or re.fullmatch(r'[A-Za-z0-9_.-]+', args.account), 'Ungültiger Accountname.')
    (root / 'plans').mkdir(exist_ok=True)
    with (root / 'plans' / '.dispatcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('phase') in {'queued', 'preparing', 'prepared', 'collecting', 'collected', 'collection_failed'}:
                print('Diese Sicherung wurde bereits eingereiht; keine doppelten Jobs.')
                print('Trainingsarray: ' + str(state.get('array_id')))
                print('Status/Logs: ' + str(state_path))
                return
            raise RuntimeError('Eine frühere Einreichung ist unvollständig. Erst Status und Queue prüfen: ' + str(state_path))
        frozen, external = restore(snapshot)
        check_versions(root, snapshot['configs'], snapshot['source_hash'])
        run_directory, completed = native_helpers(frozen)
        specifications = {name: json.loads((frozen / 'configs' / (name + '.json')).read_text())
                          for name in snapshot['configs']}
        fields = snapshot['fields']
        rows, identities = [], set()
        kept = {name: 0 for name in snapshot['configs']}
        for original in snapshot['rows']:
            row = dict(original)
            name = Path(row['config_path']).stem
            require(name in specifications and digest(specifications[name]) == row['config_hash'], 'Config oder Config-Hash weicht von der Sicherung ab.')
            identity = (name, row['tf'], row['train_assay'], int(row['seed']))
            require(identity not in identities, 'Doppelter Lauf im gesicherten Plan.')
            identities.add(identity)
            row['config_path'] = str(frozen / 'configs' / (name + '.json'))
            if completed(run_directory(frozen, name, row), row, specifications[name]):
                kept[name] += 1
            else:
                rows.append(row)
        for filename, marker in [('train_array.sbatch', 'HUM301_FINISH_CONFIG'),
                                 ('collect_config.sbatch', 'HUM301_COLLECT_FALLBACK')]:
            require(marker in (frozen / filename).read_text(), f'Gesichertes {filename} wird nicht unterstützt.')
        before = queue(root)
        require(not any(job['parent'] in snapshot['held_jobs'] and job['state'] == 'PENDING' for job in before),
                'Es sind noch gehaltene alte Pending-Tasks vorhanden; keine zweite Einreichung darüberlegen.')
        waiting_for = sorted({job['parent'] for job in before
                              if training_job(job['name']) or job['name'].startswith('h301_collect_')
                              or job['parent'] in snapshot['held_jobs']
                              or (job['directory'] == str(root) and job['name'].startswith('h301_prepare'))}, key=int)
        batch.mkdir()
        (batch / 'slurm').mkdir()
        saved_controller = batch / 'queue_controller.sh'
        shutil.copyfile(controller, saved_controller)
        candidates, plan = batch / 'tasks.requested.csv', batch / 'tasks.csv'
        write_plan(candidates, fields, rows)
        write_plan(plan, fields, rows)
        state = dict(schema=1, mode='native_array', frozen_queued_restart=True, phase='submitting',
                     root=str(frozen), queue_root=str(root), restart=True, replaced_jobs=[],
                     frozen_root=str(frozen), external_root=str(external),
                     archive=snapshot['archive'], archive_hash=snapshot['archive_hash'],
                     source_hash=snapshot['source_hash'], configs=snapshot['configs'],
                     requested_configs=snapshot['configs'], external_paths=snapshot.get('external_paths', []),
                     external_hashes=snapshot.get('external_hashes', {}),
                     fields=fields, plan=str(plan), candidates=str(candidates),
                     candidates_hash=file_hash(candidates), task_count=len(rows), max_parallel=8,
                     kept_runs=kept, wait_for_jobs=waiting_for, array_id=None, prepare_id=None,
                     collectors={}, collection_results={}, archived_runs=[],
                     created_at=datetime.now(timezone.utc).isoformat(),
                     script_hashes={str(frozen / filename): file_hash(frozen / filename)
                                    for filename in ('submit_array.sh', 'train_array.sbatch', 'collect_config.sbatch')})
        atomic_json(state_path, state)
        use_frozen_environment(state)
        prepare_script = batch / 'prepare.sbatch'
        job_script(prepare_script, frozen / 'collect_config.sbatch', saved_controller, '_prepare', state_path)
        account = ['--account=' + args.account] if args.account else []
        common = ['--parsable', *account, '--no-requeue', '--export=ALL', '--chdir=' + str(root)]
        submitted = []

        def sbatch(options, script, *parameters):
            output = run(['sbatch', *common, *options, str(script), *map(str, parameters)], root)
            require(re.fullmatch(r'[0-9]+(?:;[^\s;]+)?', output), 'Slurm-Job-ID nicht eindeutig bestätigt: ' + repr(output))
            identifier = output.split(';', 1)[0]
            submitted.append(identifier)
            return identifier

        try:
            dependency = ['--dependency=afterany:' + ':'.join(waiting_for), '--kill-on-invalid-dep=yes'] if waiting_for else []
            state['prepare_id'] = sbatch(['--hold', *dependency, '--cpus-per-task=1', '--mem=2G', '--time=00:15:00',
                '--job-name=h301_prepare_frozen', '--output=' + str(batch / 'slurm' / 'prepare_%j.out')], prepare_script)
            atomic_json(state_path, state)
            if rows:
                state['array_id'] = sbatch(['--dependency=afterok:' + state['prepare_id'], '--kill-on-invalid-dep=yes',
                    f'--array=0-{len(rows)-1}%8', '--job-name=h301_array_frozen_' + snapshot['source_hash'][:12],
                    '--output=' + str(batch / 'slurm' / 'train_%A_%a.out'),
                    '--error=' + str(batch / 'slurm' / 'train_%A_%a.err')], frozen / 'train_array.sbatch', plan, frozen)
                atomic_json(state_path, state)
            prerequisites = [state['prepare_id']] + ([state['array_id']] if state['array_id'] else [])
            for name in state['configs']:
                collect_script = batch / ('collect_' + name + '.sbatch')
                job_script(collect_script, frozen / 'collect_config.sbatch', saved_controller,
                           '_collect', state_path, name)
                state['collectors'][name] = sbatch(['--dependency=afterany:' + ':'.join(prerequisites),
                    '--kill-on-invalid-dep=yes', '--job-name=h301_collect_' + name,
                    '--output=' + str(batch / 'slurm' / ('collect_' + name + '_%j.out'))], collect_script)
                atomic_json(state_path, state)
            state['phase'] = 'queued'
            atomic_json(state_path, state)
            # Only the newly created preparation job is released. Existing jobs
            # and all existing ArrayTaskThrottle values remain untouched.
            run(['scontrol', 'release', state['prepare_id']], root)
        except Exception as error:
            state['phase'], state['error'] = 'submission_failed', str(error)
            atomic_json(state_path, state)
            # The preparation job is held until the complete graph is recorded.
            # Clean up only our own confirmed pending jobs if submission fails.
            if submitted:
                try:
                    run(['scancel', '--quiet', '--state=PENDING', *submitted], root)
                except Exception:
                    pass
            raise
        print('Alter Neustart jetzt eingereiht; gespeicherter Code-Hash: ' + state['source_hash'])
        print('Vorbereitungsjob: ' + state['prepare_id'])
        print('Trainingsarray: ' + str(state['array_id']) + '; maximal 8 gleichzeitig.')
        print('Wartet auf bisherige Jobs: ' + (', '.join(waiting_for) or 'keine'))
        print('Ergebnisse werden erst danach erneut geprüft und unvollständige Versuche archiviert.')
        print('Logs/Status: ' + str(batch))


def prepare(state_path):
    state = json.loads(state_path.read_text())
    root, frozen = Path(state['queue_root']), Path(state['frozen_root'])
    require(os.environ.get('SLURM_JOB_ID') == state['prepare_id'], 'Vorbereitung nur im zugehörigen Slurm-Job ausführen.')
    with (root / 'plans' / '.dispatcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(state['phase'] == 'queued', 'Vorbereitung bereits ausgeführt oder unvollständiger Einreichungsstatus.')
        live = queue(root)
        require(not any(job['parent'] in state['wait_for_jobs'] for job in live), 'Vorgängerjobs stehen noch in der Queue.')
        if state['array_id']:
            array_jobs = [job for job in live if job['parent'] == state['array_id']]
            require(array_jobs and all(job['state'] == 'PENDING' for job in array_jobs), 'Eigenes Trainingsarray fehlt oder ist unerwartet bereits gestartet.')
        verify_saved_files(state)
        require(file_hash(Path(state['candidates'])) == state['candidates_hash'], 'Gesicherter Aufgabenplan wurde verändert.')
        fields, rows = read_plan(Path(state['candidates']))
        run_directory, completed = native_helpers(frozen)
        specs = {name: json.loads((frozen / 'configs' / (name + '.json')).read_text()) for name in state['configs']}
        pending, archives = [], []
        kept = dict(state['kept_runs'])
        for row in rows:
            name = Path(row['config_path']).stem
            require(digest(specs[name]) == row['config_hash'], 'Gesicherte Config wurde verändert: ' + name)
            path = run_directory(frozen, name, row)
            if completed(path, row, specs[name]):
                kept[name] += 1
            else:
                pending.append(row)
                if path.exists():
                    archives.append((path, state_path.parent / 'aborted_runs' / name /
                                     path.relative_to(frozen / 'results' / name / 'runs')))
        state['phase'] = 'preparing'
        atomic_json(state_path, state)
        # All predecessors are now terminal, and our own array is blocked by
        # afterok on this job. Only now may incomplete result folders be moved.
        for source, destination in archives:
            require(not destination.exists(), 'Archivziel existiert bereits: ' + str(destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        for name in {Path(row['config_path']).stem for row in pending}:
            for source in (root / 'results' / 'Results_all' / (name + '.csv'), root / 'results' / name / 'evaluation.md'):
                if source.is_file():
                    target = state_path.parent / 'previous_summaries' / name / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
        write_plan(Path(state['plan']), fields, pending)
        # Compact the plan before any array task may start. Cancel only the
        # unused trailing slots of OUR still-pending array; never an old task.
        unused = [f'{state["array_id"]}_{index}' for index in range(len(pending), len(rows))]
        if unused:
            run(['scancel', '--quiet', '--state=PENDING', *unused], root)
            require(not any(job['job'] in unused for job in queue(root)), 'Nicht benötigte eigene Pending-Tasks wurden nicht abgemeldet.')
        state.update(phase='prepared', task_count=len(pending), kept_runs=kept,
                     plan_hash=file_hash(Path(state['plan'])),
                     cancelled_unused_tasks=unused,
                     archived_runs=[str(destination) for source, destination in archives])
        atomic_json(state_path, state)
        print(f'Alter Code bestätigt. {len(pending)} fehlende/unvollständige Läufe bleiben; '
              f'{sum(kept.values())} abgeschlossene Läufe übernommen.', flush=True)


def collect(state_path, name):
    state = json.loads(state_path.read_text())
    require(name in state['configs'] and os.environ.get('SLURM_JOB_ID') == state['collectors'][name],
            'Sammlung nur im zugehörigen Slurm-Job ausführen.')
    require(state['phase'] in {'prepared', 'collecting', 'collection_failed'},
            'Vorbereitung war nicht erfolgreich oder Sammlung bereits abgeschlossen.')
    verify_saved_files(state)
    use_frozen_environment(state)
    frozen = Path(state['frozen_root'])
    completion = state_path.parent / f'completion_{state["array_id"]}_{name}.json'
    error = ''
    try:
        run(['bash', str(frozen / 'collect_config.sbatch'), str(frozen), name, str(completion)], frozen, capture=False)
    except Exception as caught:
        error = str(caught)
    with (state_path.parent / '.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        state['collection_results'][name] = error or 'completed'
        finished = len(state['collection_results']) == len(state['configs'])
        failures = any(value != 'completed' for value in state['collection_results'].values())
        state['phase'] = 'collection_failed' if failures else ('collected' if finished else 'collecting')
        atomic_json(state_path, state)
    require(not error, error)
    print(name + ': Zusammenfassung mit dem gespeicherten Code erstellt.', flush=True)


try:
    if len(sys.argv) >= 4 and sys.argv[2] in {'_prepare', '_collect'}:
        state_path = Path(sys.argv[3]).resolve()
        if sys.argv[2] == '_prepare':
            prepare(state_path)
        else:
            require(len(sys.argv) == 5, 'Für die Sammlung muss eine Config angegeben werden.')
            collect(state_path, sys.argv[4])
    else:
        parser = argparse.ArgumentParser(description='Vorhandene Code-Sicherung hinter aktuellen Jobs einreihen.')
        parser.add_argument('snapshot', type=Path, help='Sicherungsordner oder dessen neustart.sh/manifest.json')
        parser.add_argument('--account')
        parser.add_argument('--status', action='store_true')
        submit(parser.parse_args(sys.argv[2:]))
except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
    print('Abgebrochen: ' + str(error), file=sys.stderr)
    raise SystemExit(1)
HUM301_ENQUEUE_FROZEN
