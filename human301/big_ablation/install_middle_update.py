"""Installiert das Middle-Profil-Update, ohne bestehende Configs oder Ergebnisse anzufassen.

Ersetzt training.py, run_experiment.py und submit_array.sh (nur wenn sie noch dem
Stand vom Upload entsprechen), legt extend_to_middle.py und
configs/training_priority_middle.csv an und sichert die ersetzten Dateien.

    python install_middle_update.py --target /pfad/zu/big_ablation [--dry-run]
"""
from __future__ import annotations

import argparse
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
PROGRAM_FILES = ("training.py", "run_experiment.py", "submit_array.sh")
NEW_FILES = ("extend_to_middle.py", "configs/training_priority_middle.csv")


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
    for name in PROGRAM_FILES:
        current = target / name
        if not current.is_file():
            raise FileNotFoundError(current)
        actual = sha(current)
        if actual == sha(ROOT / name):
            continue
        if actual != expected[name]:
            raise ValueError(f'{name} wurde seit dem Upload verändert. Installation abgebrochen; '
                             'die vorhandenen Änderungen werden nicht überschrieben.')
        changed.append(name)
    created = []
    for name in NEW_FILES:
        destination = target / name
        if destination.exists():
            if sha(destination) != sha(ROOT / name):
                raise FileExistsError(f'Abweichende Datei existiert bereits: {destination}')
        else:
            created.append(name)
    if not changed and not created:
        print('Diese Version ist bereits installiert.')
        return
    check_jobs(target)
    print('Ziel:', target)
    print('Programmdateien:', ', '.join(changed) or '-')
    print('Neue Dateien:', ', '.join(created) or '-')
    if dry_run:
        print('Prüfung erfolgreich; keine Änderungen ausgeführt.')
        return
    backup_parent = target.parent / 'middle_update_backups'
    backup_parent.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=backup_parent))
    for name in changed:
        shutil.copy2(target / name, backup / name)
    (backup / 'source_checksums.json').write_text(json.dumps({n: sha(target / n) for n in changed}, indent=2) + '\n')
    copied, written = [], []
    try:
        for name in changed:
            atomic_copy(ROOT / name, target / name)
            copied.append(name)
        for name in created:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            atomic_copy(ROOT / name, target / name)
            written.append(name)
    except BaseException:
        for name in copied:
            atomic_copy(backup / name, target / name)
        for name in written:
            (target / name).unlink(missing_ok=True)
        raise
    print('Installiert. Bestehende Configs und Ergebnisse bleiben unverändert.')
    print('Sicherung:', backup)
    print('Hinweis: Der Code-Hash des Projekts ändert sich. Läufe, die in einen bestehenden results-Ordner '
          'ergänzt werden, brauchen ABLATION_ALLOW_CODE_VERSION_MISMATCH=1 (extend_to_middle.py setzt das '
          'im angezeigten Befehl). Neue Configs sind davon nicht betroffen.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        install(args.target, args.dry_run)
    except Exception as error:
        parser.exit(1, f'Abgebrochen: {error}\n')


if __name__ == '__main__':
    main()
