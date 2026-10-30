#!/usr/bin/env bash
# Queue four arrays now; validate/plan only after both data jobs succeed.
# Uses the existing train_array.sbatch and collect_config.sbatch unchanged.
# Run with bash on the login node after activating at_env.
# bash submit_split_vergleiche.sh --afterok 74021 74022
# Optional split-specific PWM banks:
#   --pwm-root-v2 /existing/bank_v2 --pwm-root-v3 /existing/bank_v3
set -euo pipefail
SPLIT_LAUNCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ABLATION_PYTHON:-python3}" - "$SPLIT_LAUNCH_ROOT" "$@" <<'SPLIT_COMPARISONS_PY'
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
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
from types import SimpleNamespace


ORDER = [
    ('A14_kernel_3_6_9_12_128c_do0p2_100epo_split_v2', 'split_v2', False),
    ('A15_PWM03_30TF_split_v2', 'split_v2', True),
    ('A14_kernel_3_6_9_12_128c_do0p2_100epo_split_v3', 'split_v3', False),
    ('A15_PWM03_30TF_split_v3', 'split_v3', True),
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_json(path, payload):
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def launcher_lock(root):
    path = root / 'plans' / '.split_comparisons.lock'
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Dieser Vier-Config-Starter läuft bereits.') from None
        yield


def check_target_jobs(root, env):
    """Refuse to change a config that already belongs to a queued array."""
    result = subprocess.run(['squeue', '--me', '--noheader', '--format=%i|%j|%Z'],
                            capture_output=True, text=True, env=env, timeout=45)
    require(result.returncode == 0, 'Queue-Abfrage fehlgeschlagen: '+result.stderr.strip())
    targets = {name for name, _, _ in ORDER}
    queued = set()
    names = []
    for line in result.stdout.splitlines():
        parts = [v.strip() for v in line.split('|', 2)]
        if len(parts) == 3 and parts[2] and Path(parts[2]).resolve() == root:
            queued.add(parts[0].split('_', 1)[0])
            names.append(parts[1])
    for name in targets:
        require('h301_array_'+name not in names and 'h301_collect_'+name not in names,
                f'Für {name} ist schon ein Job eingereicht. Kein automatischer Doppelstart.')
    for path in (root / 'plans').glob('*/submission.json'):
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if str(state.get('array_id')) in queued:
            selected = set(state.get('configs', [])) & targets
            require(not selected, 'Bereits eingereichte Ziel-Configs: '+', '.join(sorted(selected)))


def preflight(root, args, frozen_paths=None):
    sys.path.insert(0, str(root))
    try:
        import data
        import model
        import training
        import run_experiment as runner
    except ImportError as exc:
        raise RuntimeError(f'Projektmodule nicht importierbar: {exc}. Zuerst at_env aktivieren.') from exc
    configs, originals = {}, {}
    native = root / 'submit_array.sh'
    require(native.is_file(), f'Fehlt: {native}')
    native_source = native.read_text()
    require(all(marker in native_source for marker in
                ('HUM301_NATIVE_ARRAY', 'prior_jobs', '--dependency=afterany:', '%8', 'collect_config.sbatch')),
            'submit_array.sh muss der vorhandene native Starter mit acht Slots, Warteschlange und Config-Collectoren sein.')
    for filename, marker in [('train_array.sbatch', 'HUM301_FINISH_CONFIG'),
                             ('collect_config.sbatch', 'HUM301_COLLECT_FALLBACK')]:
        path = root / filename
        require(path.is_file() and marker in path.read_text(), f'{filename}: aktuelle Zusammenfassungsfunktion fehlt.')
    require(callable(getattr(data, 'validate_pwm_split_provenance', None)),
            'data.py mit der vorhandenen Split-/PWM-Provenienzprüfung verwenden.')
    require(callable(getattr(model, '_load_pwm_bank', None)), 'model.py enthält keinen kompatiblen PWM-Lader.')
    for name, split, pwm in ORDER:
        path = Path(frozen_paths[name]) if frozen_paths else root / 'configs' / (name+'.json')
        require(path.is_file(), f'Config fehlt: {path}')
        cfg = training.load_config(path)
        originals[name] = deepcopy(cfg)
        require(cfg['name'] == name, f'Name/Dateiname passen nicht zusammen: {path}')
        require(data.config_split_variant(cfg) == split, f'Falscher Split in {name}')
        require(cfg['reference'] == '00_basic_'+split, f'Falsche Basic-Referenz in {name}')
        reference = root / 'configs' / (cfg['reference']+'.json')
        require(reference.is_file(), f'Passende Basic-Config fehlt: {reference}')
        require(data.config_split_variant(training.load_config(reference)) == split,
                f'Die Referenz verwendet einen anderen Split: {reference}')
        require(cfg['seeds'] == [13], f'{name}: erwartet genau Seed 13')
        require(set(cfg['train_assays']) == set(cfg['test_assays']) == {'CHIP', 'GHTS'},
                f'{name}: CHIP und GHTS müssen für Training und Test aktiviert sein.')
        require(cfg.get('task_profile', 'long') != 'short', f'{name}: short würde die TF-Auswahl beschränken.')
        expected = {'motif_kernels': [7] if pwm else [3, 6, 9, 12],
                    'embed_channels': 64 if pwm else 128,
                    'dropout': 0.0 if pwm else 0.1,
                    'channel_dropout': 0.0 if pwm else 0.2}
        for key, value in expected.items():
            require(cfg['predictor']['params'].get(key) == value, f'{name}: unerwarteter Wert für {key}')
        require(cfg['fitting']['params']['epochs'] == 100, f'{name}: erwartet maximal 100 Epochen')
        if pwm:
            require(cfg['predictor']['params'].get('pwm_count') == 3, f'{name}: erwartet drei PWMs')
            override = args.pwm_root_v2 if split == 'split_v2' else args.pwm_root_v3
            if override is not None:
                cfg['predictor']['params']['pwm_root'] = str(override.expanduser().resolve())
        for role in ('workflow', 'data', 'predictor', 'fitting', 'inference'):
            if cfg[role] is not None:
                training.factory(cfg[role])
        configs[name] = cfg

    selection = argparse.Namespace(all_configs=False, config=[str(frozen_paths[name]) if frozen_paths else name for name, _, _ in ORDER],
                                   manifest=args.manifest, tf=None, train_assay=None, seed=None,
                                   priority_file=root/'configs'/'training_priority.csv')
    print('Prüfe Aufgabenplan und vorhandene Split-Caches; es startet kein Training ...', flush=True)
    # make_plan uses the project's own source-hash and split-definition checks.
    # Memoize source reads within this one preflight, shared by the four configs.
    from functools import lru_cache
    source_hash, source_summary = data.sha256_file, data.cache_summary
    try:
        data.sha256_file = lru_cache(maxsize=None)(source_hash)
        data.cache_summary = lru_cache(maxsize=None)(source_summary)
        rows = runner.make_plan(selection)
    finally:
        data.sha256_file, data.cache_summary = source_hash, source_summary
    grouped = {name: [] for name, _, _ in ORDER}
    for row in rows:
        name = Path(row['config_path']).stem
        require(name in grouped, 'Aufgabenplan enthält eine andere Config.')
        grouped[name].append(row)
    tf_set = None
    for name, split, pwm in ORDER:
        selected = grouped[name]
        identities = {(r['tf'], r['train_assay'], int(r['seed'])) for r in selected}
        tfs = {r['tf'] for r in selected}
        require(len(selected) == len(identities) == 60 and len(tfs) == 30,
                f'{name}: erwartet 60 eindeutige Trainings für 30 TFs, gefunden {len(selected)} / {len(tfs)}.')
        require(identities == {(tf, assay, 13) for tf in tfs for assay in ('CHIP', 'GHTS')},
                f'{name}: unvollständige TF-/Assay-Auswahl.')
        require(all(r['training_scope'] == 'single_tf' for r in selected), 'Erwartet Single-TF-Läufe.')
        tf_set = tfs if tf_set is None else tf_set
        require(tfs == tf_set, 'Die vier Configs wählen unterschiedliche TFs aus.')
        if pwm:
            cfg = configs[name]
            p = cfg['predictor']['params']
            print(f'Prüfe drei PWMs je TF/Assay für {split}: {p["pwm_root"]}', flush=True)
            errors = []
            for row in selected:
                context = SimpleNamespace(config=cfg, tf=row['tf'], train_assay=row['train_assay'], artifacts={})
                try:
                    model._load_pwm_bank(context, p['pwm_root'], 3, p['sequence_length'], p['embed_channels'])
                except (ValueError, OSError, KeyError) as exc:
                    errors.append(f'{row["tf"]}/{row["train_assay"]}: {exc}')
            require(not errors, f'{name}: PWM-Vorprüfung fehlgeschlagen ({len(errors)}/60).\n'+
                    '\n'.join(errors[:3])+
                    '\nDie aktuelle Version verlangt passende PWMs und split_provenance.json je Split. '+
                    'Vorhandene korrekte Ordner mit --pwm-root-v2 und --pwm-root-v3 angeben. '+
                    'Die Herkunftsdateien müssen die tatsächliche Erzeugung/Auswahl dokumentieren.')
        print(f'OK: {name}: 30 TFs, 60 Trainings, Seed 13, {split}', flush=True)
    return configs, originals, rows


def slurm_submit(arguments, env):
    result = subprocess.run(['sbatch', '--parsable', *arguments], capture_output=True, text=True, env=env, timeout=90)
    require(result.returncode == 0, 'sbatch fehlgeschlagen: '+(result.stderr.strip() or result.stdout.strip()))
    output = result.stdout.strip()
    require(re.fullmatch(r'[0-9]+(?:;[^\s;]+)?', output) is not None,
            'Job-ID nicht eindeutig bestätigt. Vor erneutem Aufruf die Queue und start.json prüfen.')
    return output.split(';', 1)[0]


def older_jobs(root, env):
    result = subprocess.run(['squeue', '--me', '--noheader', '--format=%i|%j|%Z'], capture_output=True, text=True, env=env, timeout=45)
    require(result.returncode == 0, 'Queue-Abfrage fehlgeschlagen: '+result.stderr.strip())
    jobs = set()
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split('|', 2)]
        if len(parts) != 3 or not parts[2] or Path(parts[2]).resolve() != root:
            continue
        job, name = parts[0].split('_', 1)[0], parts[1]
        require(name != 'h301_slots', 'Eine alte h301_slots-Steuerung läuft noch; zuerst deren Zustand prüfen.')
        if name.startswith(('h301_array_', 'hum301_array', 'h301_collect_')) or re.match(r'A\d+_', name):
            require(job.isdigit(), f'Job-ID nicht eindeutig: {job}')
            jobs.add(job)
    return sorted(jobs, key=int)


def prepare_batch(batch):
    """CPU job: validate completed split caches and write all four frozen plans."""
    state = json.loads((batch/'start.json').read_text())
    root = Path(state['root'])
    os.chdir(root)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    args = SimpleNamespace(manifest=Path(state['manifest']), pwm_root_v2=None, pwm_root_v3=None)
    configs, _, rows = preflight(root, args, state['frozen_configs'])
    import run_experiment as runner
    for name, _, _ in ORDER:
        directory = Path(state['batches'][name])
        selected = [r for r in rows if Path(r['config_path']).stem == name]
        require(len(selected) == 60, f'{name}: Aufgabenanzahl hat sich geändert.')
        path = directory/'tasks.csv'
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=runner.PLAN_FIELDS)
            writer.writeheader()
            writer.writerows(selected)
    atomic_json(batch/'pruefung_erfolgreich.json', {'checked_at': datetime.now(timezone.utc).isoformat(),
                'configurations': [n for n, _, _ in ORDER], 'trainings_per_config': 60, 'trainings_total': len(rows)})
    print('Alle vier Aufgabenpläne geprüft und geschrieben. Die GPU-Abhängigkeiten können freigegeben werden.', flush=True)
    return 0


def main():
    root = Path(sys.argv[1]).resolve()
    if sys.argv[2:3] == ['--_prepare']:
        return prepare_batch(Path(sys.argv[3]).resolve())
    parser = argparse.ArgumentParser(description='Vier Arrays jetzt einreichen, erst nach beiden Datenjobs prüfen und starten.')
    parser.add_argument('--afterok', nargs='+', default=['74021', '74022'], metavar='JOBID', help='Erfolgreich abzuwartende Datenjobs; Standard: 74021 74022')
    parser.add_argument('--check', action='store_true', help='Vollständige Daten-/PWM-Prüfung jetzt, ohne Jobs einzureichen; erst bei fertigen Daten sinnvoll')
    parser.add_argument('--manifest', type=Path, default=root/'configs'/'caches.csv')
    parser.add_argument('--pwm-root-v2', type=Path, help='Vorhandene PWM-Bank für split_v2; wird nur in der eingefrorenen Config verwendet')
    parser.add_argument('--pwm-root-v3', type=Path, help='Vorhandene PWM-Bank für split_v3; wird nur in der eingefrorenen Config verwendet')
    parser.add_argument('--account', help='Optionaler SLURM-Account; keine Partition wird vorgegeben')
    args = parser.parse_args(sys.argv[2:])
    args.manifest = args.manifest.expanduser().resolve()
    require(not os.environ.get('SLURM_JOB_ID'), 'Mit bash auf dem Login-Knoten starten, nicht mit sbatch.')
    require(args.manifest.is_file(), f'Manifest fehlt: {args.manifest}')
    require(args.afterok and all(re.fullmatch(r'[0-9]+', v) for v in args.afterok), 'Datenjob-IDs müssen numerisch sein.')
    if args.check:
        preflight(root, args)
        print('Prüfung erfolgreich: 240 Trainings; keine Jobs eingereicht, keine Configs geändert.')
        return 0
    env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_') and k not in {'SLURM_CLUSTERS', 'SLURM_FEDERATION'}}
    env.update(LC_ALL='C', ABLATION_PYTHON=sys.executable)
    for executable in ('bash', 'sbatch', 'squeue'):
        require(shutil.which(executable), f'{executable} fehlt.')
    for filename, marker in [('train_array.sbatch', 'HUM301_FINISH_CONFIG'), ('collect_config.sbatch', 'HUM301_COLLECT_FALLBACK')]:
        path = root/filename
        require(path.is_file() and marker in path.read_text(), f'{filename}: aktuelle Zusammenfassungsfunktion fehlt.')
    require((root/'configs'/'training_priority.csv').is_file(), 'configs/training_priority.csv fehlt.')
    configs = {}
    for name, split, pwm in ORDER:
        path = root/'configs'/(name+'.json')
        require(path.is_file(), f'Config fehlt: {path}')
        cfg = json.loads(path.read_text())
        require(cfg['name'] == name and cfg['data']['params'].get('split_variant') == split,
                f'Config-Name oder Split passt nicht: {path}')
        require(cfg['reference'] == '00_basic_'+split, f'Falsche Referenz: {path}')
        require((root/'configs'/(cfg['reference']+'.json')).is_file(), f'Basic-Referenz fehlt: {cfg["reference"]}')
        require(cfg['seeds'] == [13] and cfg.get('task_profile', 'long') != 'short', f'Falscher Seed oder short-Profil: {name}')
        require(not (root/'results'/name).exists(), f'results/{name} existiert bereits; Wiederaufnahme gesondert prüfen.')
        override = args.pwm_root_v2 if split == 'split_v2' else args.pwm_root_v3
        if pwm and override is not None:
            cfg['predictor']['params']['pwm_root'] = str(override.expanduser().resolve())
        configs[name] = cfg
    plans = root/'plans'
    plans.mkdir(exist_ok=True)
    with launcher_lock(root):
        check_target_jobs(root, env)
        wait_for = older_jobs(root, env)
        batch = Path(tempfile.mkdtemp(prefix='split_vergleiche_', dir=plans)).resolve()
        source = root/'submit_split_vergleiche.sh'
        require(source.is_file(), 'Starter als submit_split_vergleiche.sh im big_ablation-Ordner speichern.')
        frozen_launcher = batch/'submit_split_vergleiche.sh'
        shutil.copy2(source, frozen_launcher)
        state = {'root': str(root), 'manifest': str(args.manifest), 'created_at': datetime.now(timezone.utc).isoformat(),
                 'afterok_data_jobs': args.afterok, 'wait_for_jobs': wait_for, 'phase': 'submitting',
                 'frozen_configs': {}, 'batches': {}, 'prepare_job': None, 'arrays': {}, 'collectors': {}}
        for name, _, _ in ORDER:
            directory = Path(tempfile.mkdtemp(prefix='batch_array_', dir=plans)).resolve()
            (directory/'slurm').mkdir()
            frozen = directory/(name+'.json')
            atomic_json(frozen, configs[name])
            state['frozen_configs'][name] = str(frozen)
            state['batches'][name] = str(directory)
            shutil.copy2(root/'configs'/'training_priority.csv', directory/'training_priority.csv')
            atomic_json(directory/'submission.json', dict(root=str(root), configs=[name], requested_configs=[name],
                        mode='native_array', array_id=None, created_at=state['created_at'],
                        plan=str(directory/'tasks.csv'), task_count=60, max_parallel=8, collectors={},
                        restart=False, replaced_jobs=[], kept_runs={name: 0}, archived_runs=[], wait_for_jobs=wait_for))
        receipt = batch/'start.json'
        atomic_json(receipt, state)
        account = ['--account='+args.account] if args.account else []
        import shlex
        preparation = ['--no-requeue', '--export=ALL', '--dependency=afterok:'+':'.join(args.afterok),
                       '--kill-on-invalid-dep=yes', '--job-name=h301_check_split_models', '--cpus-per-task=1',
                       '--mem=16G', '--time=02:00:00', '--chdir='+str(root), '--output='+str(batch/'pruefung_%j.out'),
                       '--wrap='+shlex.join(['bash', str(frozen_launcher), '--_prepare', str(batch)])]
        try:
            # Persist submission intent. If a receipt is lost, never auto-retry.
            state['current_submission'] = 'prepare'
            atomic_json(receipt, state)
            prepare_id = slurm_submit([*account, *preparation], env)
            state['prepare_job'] = prepare_id
            atomic_json(receipt, state)
            previous_collector = None
            for name, _, _ in ORDER:
                directory = Path(state['batches'][name])
                prior = [previous_collector] if previous_collector else wait_for
                dependency = 'afterok:'+prepare_id
                if prior:
                    dependency += ',afterany:'+':'.join(prior)
                state['current_submission'] = 'array:'+name
                atomic_json(receipt, state)
                array_id = slurm_submit([*account, '--array=0-59%8', '--no-requeue', '--export=ALL',
                    '--dependency='+dependency, '--kill-on-invalid-dep=yes', '--job-name=h301_array_'+name,
                    '--chdir='+str(root), '--output='+str(directory/'slurm'/'train_%A_%a.out'),
                    '--error='+str(directory/'slurm'/'train_%A_%a.err'),
                    str(root/'train_array.sbatch'), str(directory/'tasks.csv'), str(root)], env)
                state['arrays'][name] = array_id
                atomic_json(receipt, state)
                submission = json.loads((directory/'submission.json').read_text())
                submission.update(array_id=array_id, prepare_job=prepare_id, dependency=dependency)
                atomic_json(directory/'submission.json', submission)
                state['current_submission'] = 'collect:'+name
                atomic_json(receipt, state)
                collector_id = slurm_submit([*account, '--no-requeue', '--export=ALL',
                    '--dependency=afterok:'+prepare_id+',afterany:'+array_id,
                    '--kill-on-invalid-dep=yes', '--job-name=h301_collect_'+name, '--chdir='+str(root),
                    '--output='+str(directory/'slurm'/('collect_'+name+'_%j.out')),
                    str(root/'collect_config.sbatch'), str(root), name,
                    str(directory/f'completion_{array_id}_{name}.json')], env)
                state['collectors'][name] = collector_id
                atomic_json(receipt, state)
                submission['collectors'][name] = collector_id
                atomic_json(directory/'submission.json', submission)
                previous_collector = collector_id
                print(f'{name}: Array {array_id} (60 Trainings, %8), Zusammenfassung {collector_id}', flush=True)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            state.update(phase='stopped', error=str(exc))
            atomic_json(receipt, state)
            raise RuntimeError(f'{exc}\nBisherige Einreichungen: {receipt}. Vor erneutem Start die Queue prüfen; vorhandene Jobs werden nicht storniert.') from exc
        state.update(phase='submitted', current_submission=None)
        atomic_json(receipt, state)
        print(f'\nCPU-Prüfjob {prepare_id}: wartet auf Erfolg von '+', '.join(args.afterok)+'.', flush=True)
        print('Alle vier Arrays sind bereits eingereicht. Erst nach erfolgreicher Prüfung starten sie nacheinander; '+
              'die nächste Config wartet auf die vorherige Zusammenfassung.', flush=True)
        print('Kein dauerhafter Steuerprozess. Bestehende Jobs und ArrayTaskThrottle-Werte bleiben unverändert.', flush=True)
        print(f'Protokoll und Job-IDs: {receipt}', flush=True)
    return 0


try:
    raise SystemExit(main())
except (RuntimeError, ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
    print('Abgebrochen: '+str(exc), file=sys.stderr)
    raise SystemExit(1)
SPLIT_COMPARISONS_PY
