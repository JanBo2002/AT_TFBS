"""Install the uploaded-version update without replacing normal configs or running jobs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_jobs(target):
    if not shutil.which('squeue'):
        return
    result = subprocess.run(['squeue', '--me', '--noheader', '--format=%i|%j|%Z'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Queue konnte nicht geprüft werden: ' + result.stderr.strip())
    active = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split('|', 2)]
        if len(fields) == 3 and fields[2] and Path(fields[2]).resolve() == target:
            active.append(fields[0] + ' (' + fields[1] + ')')
    if active:
        raise RuntimeError('Eigene Projektjobs sind noch aktiv oder warten: ' + ', '.join(active)
                           + '. Diese müssen erst mit der bisherigen Codeversion enden.')


def atomic_copy(source, target):
    fd, temporary = tempfile.mkstemp(prefix=target.name + '.', dir=target.parent)
    os.close(fd)
    try:
        mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else stat.S_IMODE(source.stat().st_mode)
        shutil.copy2(source, temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install(target, dry_run=False):
    target = Path(target).expanduser().resolve()
    if target == ROOT or not (target / 'configs' / 'caches.csv').is_file():
        raise ValueError('Als Ziel den bestehenden big_ablation-Ordner mit configs/caches.csv angeben.')
    expected = json.loads((ROOT / 'source_checksums.json').read_text())
    changed = []
    for name, original_sha in expected.items():
        current = target / name
        if not current.is_file():
            raise FileNotFoundError(current)
        actual = sha(current)
        if actual == sha(ROOT / name):
            continue
        allowed = original_sha if isinstance(original_sha, list) else [original_sha]
        if actual not in allowed:
            raise ValueError(f'{name} wurde seit dem Upload verändert. Installation abgebrochen; '
                             'die vorhandenen Änderungen werden nicht überschrieben.')
        changed.append(name)
    base = json.loads((target / 'configs' / '00_basic.json').read_text())
    if base['name'] != '00_basic' or base['reference'] != '00_basic':
        raise ValueError('00_basic muss eine selbstreferenzierte Basic-Config sein.')
    if base.get('data', {}).get('params', {}).get('split_variant', 'original') != 'original':
        raise ValueError('Die vorhandene 00_basic-Config muss den ursprünglichen Split verwenden.')
    new_configs = []
    for variant in ('split_v2', 'split_v3'):
        recipe = copy.deepcopy(base)
        recipe['name'] += '_' + variant
        recipe['reference'] += '_' + variant
        recipe['description'] = base.get('description', '') + ' Chromosome split: ' + variant + '.'
        recipe['data']['params']['split_variant'] = variant
        destination = target / 'configs' / (recipe['name'] + '.json')
        if destination.exists():
            if json.loads(destination.read_text()) != recipe:
                raise FileExistsError(f'Abweichende Config existiert bereits: {destination}')
        else:
            new_configs.append((destination, recipe))
    if not changed and not new_configs:
        print('Diese Version ist bereits installiert.')
        return
    check_jobs(target)
    print('Ziel:', target)
    print('Programmdateien:', ', '.join(changed))
    print('Neue Basic-Configs:', ', '.join(p.name for p, _ in new_configs))
    if dry_run:
        print('Prüfung erfolgreich; keine Änderungen ausgeführt.')
        return
    backup_parent = target.parent / 'split_update_backups'
    backup_parent.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=backup_parent))
    for name in changed:
        shutil.copy2(target / name, backup / name)
    (backup / 'source_checksums.json').write_text(json.dumps({n:sha(target/n) for n in changed}, indent=2)+'\n')
    copied = []
    created = []
    try:
        for name in changed:
            atomic_copy(ROOT / name, target / name)
            copied.append(name)
        for path, recipe in new_configs:
            with path.open('x') as handle:
                created.append(path)
                json.dump(recipe, handle, indent=2); handle.write('\n')
    except BaseException:
        for name in copied:
            atomic_copy(backup / name, target / name)
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print('Installiert. Bestehende normale Configs und Daten bleiben erhalten.')
    print('Sicherung:', backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        install(args.target, args.dry_run)
    except Exception as error:
        parser.exit(1, f'Abgebrochen: {error}\n')


if __name__ == '__main__':
    main()
