#!/usr/bin/env bash
#SBATCH --job-name=h301_array_retry
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
set -euo pipefail
ABLATION_SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
exec "${ABLATION_PYTHON:-python3}" - "$ABLATION_SCRIPT" "$@" <<'PYTHON_RESTART'
"""One launcher: inspect old array tasks and restart only selected missing work.

bash wiederanlauf_array.sh 72554 A04 --check
bash wiederanlauf_array.sh 72554 A04

Missing and incomplete terminal tasks are restarted automatically, from epoch 1.
The old folder of each incomplete run is deleted immediately before restarting.
Running jobs and fully completed runs are skipped. --check never deletes files.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
            'PREEMPTED', 'BOOT_FAIL', 'OUT_OF_MEMORY', 'DEADLINE', 'REVOKED',
            'SPECIAL_EXIT'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def package_hash(root):
    result = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        if 'results' not in path.relative_to(root).parts and '__pycache__' not in path.parts:
            result.update(str(path.relative_to(root)).encode())
            result.update(path.read_bytes())
    return result.hexdigest()


def run(arguments):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('SBATCH_') and k not in {'SLURM_CLUSTERS', 'SLURM_FEDERATION'}}
    env.update(LC_ALL='C', ABLATION_PYTHON=sys.executable, SLURM_BITSTR_LEN='0')
    answer = subprocess.run(arguments, env=env, text=True, capture_output=True, timeout=90)
    require(answer.returncode == 0, f'{arguments[0]} fehlgeschlagen: {answer.stderr.strip() or answer.stdout.strip()}')
    return answer.stdout.strip()


def job_id(answer):
    require(re.fullmatch(r'[0-9]+(?:;[^\s;]+)?', answer),
            f'Job-ID unklar: {answer!r}. Queue pruefen, nicht blind erneut starten.')
    return answer.split(';')[0]


def queue():
    rows = []
    for line in run(['squeue', '--me', '--array', '--states=all', '--noheader',
                     '--format=%F|%K|%T|%j|%Z']).splitlines():
        fields = [v.strip() for v in line.split('|', 4)]
        require(len(fields) == 5, f'Unlesbare squeue-Zeile: {line}')
        array, index, state, name, directory = fields
        if state not in TERMINAL:
            rows.append(dict(array=array, index=index, state=state, name=name, directory=directory))
    return rows


def location(root, config, row):
    require(row['training_scope'] == 'single_tf' and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['tf']),
            'Dieser Wiederanlauf erwartet einzelne TFs mit sicheren Verzeichnisnamen.')
    require(row['train_assay'] in {'CHIP', 'GHTS'}, 'Unbekannter Trainingsassay.')
    return root / 'results' / config['name'] / 'runs' / row['tf'] / ('train_' + row['train_assay']) / f"seed_{int(row['seed']):02d}"


def delete_incomplete(root, config, row, path):
    expected = location(root, config, row)
    base = root / 'results' / config['name'] / 'runs'
    require(path == expected and len(path.relative_to(base).parts) == 3,
            'Loeschziel ist kein einzelner geplanter Seed-Ordner.')
    current = path
    while current != root:
        require(not current.is_symlink(), 'Symlink im Loeschpfad; Lauf wird nicht geloescht: ' + str(current))
        current = current.parent
    require(path.is_dir() and path.resolve().is_relative_to(base.resolve()),
            'Loeschziel liegt ausserhalb des vorgesehenen Laufordners.')
    shutil.rmtree(path)
    print(f'Unvollstaendiger Lauf geloescht: {path}', flush=True)


def completed(path, config, row, code_hash):
    """Completion markers + checkpoint hash + BOTH evaluations; no torch import."""
    try:
        meta = read_json(path / 'run.json')
        require(meta.get('status') == 'completed', 'run.json meldet ' + str(meta.get('status')))
        recipe = {k: v for k, v in config.items() if k not in {'description', 'seeds', 'train_assays', 'test_assays'}}
        require(meta['config_hash'] == digest(recipe) and meta['code_hash'] == code_hash,
                'Code-/Config-Hash in run.json stimmt nicht.')
        require((meta['tf'], meta['train_assay'], int(meta['seed'])) ==
                (row['tf'], row['train_assay'], int(row['seed'])), 'Falsche Laufzuordnung.')
        require(read_json(path / 'config.json') == config, 'Gespeicherte Config weicht ab.')
        checkpoint = path / 'training/model_for_prediction.pt'
        saved = read_json(path / 'training/prediction_metadata.json')
        require(checkpoint.stat().st_size > 0 and file_hash(checkpoint) == saved['checkpoint_sha256'],
                'Vorhersage-Checkpoint fehlt oder Pruefsumme stimmt nicht.')
        require((path / 'training/best_model.pt').stat().st_size > 0, 'Best-Checkpoint fehlt.')
        require(bool(read_json(path / 'training/history.json')), 'Trainingsverlauf fehlt.')
        records = [json.loads(line) for line in (path / 'records.jsonl').read_text().splitlines() if line.strip()]
        records = [r for r in records if r.get('run_id') == meta['run_id']]
        require(any(r.get('row_type') == 'run' and r.get('status') == 'completed' for r in records),
                'Abschlussdatensatz in records.jsonl fehlt.')
        for assay in config['test_assays']:
            folder = path / 'evaluations/standard' / assay
            result = read_json(folder / 'metrics.json')
            require(result.get('status') == 'completed' and result.get('test_assay') == assay and
                    result.get('tf') == row['tf'] and result.get('train_assay') == row['train_assay'] and
                    int(result.get('seed', -1)) == int(row['seed']) and
                    result.get('checkpoint_sha256') == saved['checkpoint_sha256'] and
                    result['test_metrics']['sample_count'] > 0, assay + ': Auswertung unvollstaendig.')
            require(any(r.get('row_type') == 'evaluation' and r.get('status') == 'completed' and
                        r.get('test_assay') == assay for r in records), assay + ': Auswertungsdatensatz fehlt.')
            for filename in ('predictions.tsv', 'summary.txt', 'confusion_matrix.tsv'):
                require((folder / filename).stat().st_size > 0, assay + ': ' + filename + ' fehlt/leer.')
            with (folder / 'predictions.tsv').open('rb') as handle:
                require(handle.readline().startswith(b'sample_id\ttrue_label\t'), assay + ': Ungueltiger TSV-Kopf.')
                handle.seek(-1, 2)
                require(handle.read() == b'\n', assay + ': Vorhersage-Datei endet mitten in einer Zeile.')
        return True, 'Training und beide Testauswertungen abgeschlossen'
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return False, str(exc)


def worker(root, batch):
    state = read_json(batch / 'submission.json')
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    require(index in state['selected_indices'], 'Aufgabe nicht fuer diesen Wiederanlauf ausgewaehlt.')
    with Path(state['plan']).open(newline='') as handle:
        row = list(csv.DictReader(handle))[index]
    config = read_json(row['config_path'])
    require(Path(state['root']).resolve() == root and state['configs'] == [config['name']],
            'Projekt oder Config passt nicht zum Wiederanlauf.')
    require(digest(config) == row['config_hash'] and package_hash(root) == state['code_hash'],
            'Code oder Config seit Einreichung veraendert.')
    path = location(root, config, row)
    with (batch / f'task_{index}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if completed(path, config, row, state['code_hash'])[0]:
            print(f'{index}: bereits vollstaendig, uebersprungen.')
            return 0
        if path.exists() or path.is_symlink():
            require(state.get('cleanup') == 'delete', 'Kein Loeschauftrag im Einreichungsbeleg.')
            delete_incomplete(root, config, row, path)
        return subprocess.call([sys.executable, str(root / 'run_experiment.py'), 'task',
                                '--plan', state['plan'], '--index', str(index)])


def main():
    script = Path(sys.argv[1]).resolve()
    if sys.argv[2:3] == ['--worker']:
        return worker(Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('array_id', type=int)
    parser.add_argument('config', nargs='?', default='A04')
    parser.add_argument('--check', action='store_true', help='Nur Bericht, keine Einreichung')
    # Accept the earlier invocation as an alias; restarting is now the default.
    parser.add_argument('--restart-incomplete', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--account')
    args = parser.parse_args(sys.argv[2:])
    require(not os.environ.get('SLURM_JOB_ID'), 'Den Launcher auf dem Login-Knoten starten.')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.config), 'Ungueltiger Config-Name.')
    root, original = script.parent, str(args.array_id)
    require((root / 'run_experiment.py').is_file(), 'Skript direkt in big_ablation ablegen.')
    plans = root / 'plans'
    with (plans / '.dispatcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = [(p, read_json(p)) for p in sorted(plans.glob('*/submission.json'))]
        matches = [(p, s) for p, s in states if str(s.get('array_id')) == original]
        require(len(matches) == 1, 'Urspruengliches Array nicht eindeutig in plans/*/submission.json gefunden.')
        source, initial = matches[0]
        plan = Path(initial['plan']).resolve()
        require(plan.is_file() and plan.is_relative_to(plans.resolve()), 'Urspruenglicher Aufgabenplan fehlt/liegt ausserhalb von plans/.')
        with plan.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        config = read_json(root / 'configs' / (args.config + '.json'))
        require(config['name'] == args.config, 'Config-Name passt nicht zum Dateinamen.')
        selected_rows = [(i, row) for i, row in enumerate(rows) if Path(row['config_path']).stem == args.config]
        require(len(selected_rows) == 60, 'Genau 60 urspruengliche Aufgaben fuer diese Config erwartet.')
        require(all(digest(config) == row['config_hash'] and read_json(row['config_path']) == config
                    for _, row in selected_rows), 'Config wurde seit dem urspruenglichen Plan geaendert.')
        code_hash = package_hash(root)
        versions = root / 'results' / args.config / 'versions.txt'
        if versions.exists():
            saved_hash = re.search(r'^code_sha256:\s*(\w+)', versions.read_text(), re.MULTILINE)
            require(saved_hash and saved_hash.group(1) == code_hash, 'Python-Code weicht von versions.txt ab; kein Mischen von Versionen.')
        related = [original] + sorted({str(s['array_id']) for _, s in states
                   if s.get('retry_of') == original and s.get('configs') == [args.config]
                   and str(s.get('array_id', '')).isdigit()}, key=int)
        live = queue()
        require(not any(j['name'] == 'h301_array_retry_' + args.config and j['array'] not in related
                        and j['directory'] and Path(j['directory']).resolve() == root for j in live),
                'Ein Wiederanlauf ohne passenden Einreichungsbeleg ist aktiv. Erst dessen Job-ID pruefen.')
        accounting = {}
        for line in run(['sacct', '--allocations', '--array', '--noheader', '--parsable2',
                         '--jobs=' + ','.join(related), '--format=JobID%80,State%40,ExitCode,Start']).splitlines():
            fields = line.split('|')
            if len(fields) >= 4 and re.fullmatch(r'\d+_\d+', fields[0].strip()):
                accounting[fields[0].strip()] = [v.strip() for v in fields[1:4]]
        report, retry = [], []
        for index, row in selected_rows:
            path = location(root, config, row)
            active = next((j for j in live if j['array'] in related and j['index'] == str(index)), None)
            history = [(a, accounting[f'{a}_{index}']) for a in related if f'{a}_{index}' in accounting]
            slurm, exitcode, started = history[-1][1] if history else ['UNKNOWN', '', 'Unknown']
            slurm = slurm or 'UNKNOWN'
            status, reason = 'AKTIV', active['state'] if active else ''
            if not active:
                ok, reason = completed(path, config, row, code_hash)
                if ok:
                    status = 'FERTIG'
                elif slurm.split()[0].rstrip('+*') not in TERMINAL:
                    status, reason = 'UNKLAR', 'Kein sicher bestaetigtes Jobende; ' + reason
                else:
                    begun = path.exists() or started not in {'', 'Unknown', 'None', 'N/A'} or slurm.startswith('COMPLETED')
                    status = 'UNVOLLSTAENDIG' if begun else 'NICHT_GESTARTET'
                    retry.append(index)
            log_batch = next((p.parent for p, s in states if history and str(s.get('array_id')) == history[-1][0]), source.parent)
            errors = ''
            log = log_batch / 'slurm' / f'train_{history[-1][0] if history else original}_{index}.err'
            if status == 'UNVOLLSTAENDIG' and log.is_file():
                with log.open('rb') as handle:
                    handle.seek(max(0, log.stat().st_size - 4096))
                    errors = ' '.join(handle.read().decode(errors='replace').split())[-500:]
            report.append(dict(index=index, tf=row['tf'], train_assay=row['train_assay'], status=status,
                               slurm_state=slurm, exit_code=exitcode, start=started, reason=reason, error_tail=errors))
            print(f"{index:3} {row['tf']:12} {row['train_assay']:4} {status:18} SLURM={slurm} | {reason}")
        batch = Path(tempfile.mkdtemp(prefix=f'retry_{args.config}_', dir=plans)).resolve()
        with (batch / 'audit.tsv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[0]), delimiter='\t')
            writer.writeheader(); writer.writerows(report)
        print('Bilanz:', dict(Counter(r['status'] for r in report)))
        print('Pruefbericht:', batch / 'audit.tsv')
        if args.check or not retry:
            print('Keine Jobs eingereicht.' if args.check else 'Keine sicher neu einzureichenden Aufgaben.')
            return 0
        current = queue()
        require(not any(j['array'] in related and j['index'].isdigit() and int(j['index']) in retry for j in current),
                'Ausgewaehlte Aufgabe ist inzwischen aktiv. Launcher erneut pruefen lassen.')
        counts = Counter(j['array'] for j in current if j['directory'] and Path(j['directory']).resolve() == root
                         and (j['name'].startswith('h301_array_') or re.match(r'A\d+_', j['name'])))
        limits = {str(s['array_id']): int(s.get('max_parallel', 8)) for _, s in states if s.get('array_id')}
        occupied = sum(min(count, limits.get(array, 8)) for array, count in counts.items())
        parallel = min(len(retry), 8 - occupied)
        require(parallel > 0, 'Acht Trainingsplaetze sind bereits belegt/vorgemerkt. Bericht erstellt; spaeter erneut starten.')
        (batch / 'slurm').mkdir()
        state = dict(root=str(root), configs=[args.config], retry_of=original, plan=str(plan), array_id=None,
                     selected_indices=retry, cleanup='delete', code_hash=code_hash,
                     max_parallel=parallel, collectors={}, status='prepared')
        save_json(batch / 'submission.json', state)
        account = ['--account=' + args.account] if args.account else []
        new = job_id(run(['sbatch', '--parsable', '--hold', *account, '--no-requeue', '--export=ALL',
                          '--array=' + ','.join(map(str, retry)) + f'%{parallel}',
                          '--job-name=h301_array_retry_' + args.config, '--chdir=' + str(root),
                          '--output=' + str(batch / 'slurm/train_%A_%a.out'),
                          '--error=' + str(batch / 'slurm/train_%A_%a.err'),
                          str(script), '--worker', str(root), str(batch)]))
        state.update(array_id=new, status='held')
        save_json(batch / 'submission.json', state)
        # Finished tasks may already have disappeared from the controller.
        # Only still-active tasks of THIS config belong in the new dependency.
        current = queue()
        own_indices = {str(i) for i, _ in selected_rows}
        waiting_for = sorted({f"{j['array']}_{j['index']}" for j in current
                              if j['array'] in related and j['index'] in own_indices})
        dependency = 'afterany:' + ':'.join(waiting_for + [new])
        old_collector = str(initial.get('collectors', {}).get(args.config, ''))
        try:
            existing = next((j for j in current if j['array'] == old_collector and j['name'] == 'h301_collect_' + args.config
                             and j['directory'] and Path(j['directory']).resolve() == root and j['state'] == 'PENDING'), None)
            if existing:
                run(['scontrol', 'update', 'JobId=' + old_collector, 'Dependency=' + dependency])
                collector = old_collector
            else:
                collector = job_id(run(['sbatch', '--parsable', *account, '--export=ALL', '--no-requeue',
                     '--dependency=' + dependency, '--kill-on-invalid-dep=yes', '--chdir=' + str(root),
                     '--job-name=h301_collect_' + args.config,
                     '--output=' + str(batch / 'slurm/collect_%j.out'),
                     str(root / 'collect_config.sbatch'), str(root), args.config]))
            state['collectors'][args.config] = collector
            print('Zusammenfassungsjob wartet auch auf den Wiederanlauf:', collector)
        except RuntimeError as exc:
            state['collection_error'] = str(exc)
            print(f'Zusammenfassungsjob nicht aktualisiert: {exc}', file=sys.stderr)
            print(f'Nach allen Laeufen: python run_experiment.py collect --config {args.config}', file=sys.stderr)
        save_json(batch / 'submission.json', state)
        run(['scontrol', 'release', new])
        state['status'] = 'submitted'
        save_json(batch / 'submission.json', state)
        print(f'Jobarray {new}: {len(retry)} Aufgaben, maximal {parallel} weitere Trainings gleichzeitig.')
        print('Urspruengliche Indizes:', ','.join(map(str, retry)))
        return 0


try:
    raise SystemExit(main())
except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
    print('Abgebrochen:', exc, file=sys.stderr)
    raise SystemExit(1)
PYTHON_RESTART
