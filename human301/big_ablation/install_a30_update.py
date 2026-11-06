"""Installiert das Update für die Experimente A30–A35 (model.py, data.py, training.py, Configs).

Ersetzt nur Programmdateien, die noch genau dem erwarteten Stand entsprechen
(model.py und data.py: Original vom 4.11.2026; training.py: Middle-Profil-Version),
legt die sechs Configs an und sichert die ersetzten Dateien. Reihenfolge der
Ersetzung: model.py, data.py, zuletzt training.py – so ist jeder Zwischenstand
mit sich selbst verträglich, falls genau in dem Moment ein Job startet.

    python install_a30_update.py --target /pfad/zu/big_ablation [--dry-run] [--while-jobs-running]

Ohne --while-jobs-running wird abgebrochen, solange eigene Projektjobs laufen
oder warten: Jobs, die nach der Installation starten, laufen dann mit dem neuen
Code (rechnerisch identisch, verifiziert), tragen aber einen anderen Code-Hash
in versions.txt ihres Experiments ein.
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
# Reihenfolge ist Absicht (siehe Modulbeschreibung).
PROGRAM_FILES = ("model.py", "data.py", "training.py")
EXPECTED = {
    "model.py": ["b8766dbbe7dfbe1b7f29ac45939933a2b29f6bdd7e56ef9da316b1e1ec04b497"],
    "data.py": ["1bae52b69c2d3a05bdb7b8098898ce0847c7fb967e34f31bdb48e326d0e2df50"],
    "training.py": ["312697726b98b03b5dff763a8892a0383beab1022adccf1c8155bce733870383"],
}
REQUIRED_PRESENT = ("run_experiment.py", "submit_array.sh", "configs/training_priority_middle.csv")
NEW_CONFIGS = tuple(sorted(p.name for p in (ROOT / "configs").glob("A3*.json")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_jobs(target):
    if not shutil.which("squeue"):
        return []
    result = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%j|%Z"],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Queue konnte nicht geprüft werden: " + result.stderr.strip())
    active = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split("|", 2)]
        if len(fields) == 3 and fields[2] and Path(fields[2]).resolve() == target:
            active.append(fields[0] + " (" + fields[1] + ")")
    return active


def atomic_copy(source, target):
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    os.close(fd)
    try:
        mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else stat.S_IMODE(source.stat().st_mode)
        shutil.copy2(source, temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install(target, dry_run=False, while_jobs_running=False):
    target = Path(target).expanduser().resolve()
    if target == ROOT or not (target / "configs" / "caches.csv").is_file():
        raise ValueError("Als Ziel den bestehenden big_ablation-Ordner mit configs/caches.csv angeben.")
    for name in REQUIRED_PRESENT:
        if not (target / name).is_file():
            raise FileNotFoundError(f"{target / name} fehlt; zuerst das Middle-Profil-Update installieren.")
    if "profile_priority_file" not in (target / "run_experiment.py").read_text():
        raise ValueError("run_experiment.py ist noch nicht die Middle-Profil-Version.")
    changed = []
    for name in PROGRAM_FILES:
        current = target / name
        if not current.is_file():
            raise FileNotFoundError(current)
        actual = sha(current)
        if actual == sha(ROOT / name):
            continue
        if actual not in EXPECTED[name]:
            raise ValueError(f"{name} entspricht weder dem erwarteten Stand noch dieser Version. Installation "
                             "abgebrochen; vorhandene Änderungen werden nicht überschrieben.")
        changed.append(name)
    created = []
    for name in NEW_CONFIGS:
        destination = target / "configs" / name
        if destination.exists():
            if json.loads(destination.read_text()) != json.loads((ROOT / "configs" / name).read_text()):
                raise FileExistsError(f"Abweichende Config existiert bereits: {destination}")
        else:
            created.append(name)
    if not changed and not created:
        print("Diese Version ist bereits installiert.")
        return
    jobs = active_jobs(target)
    if jobs and changed and not while_jobs_running:
        raise RuntimeError("Eigene Projektjobs sind aktiv oder warten: " + ", ".join(jobs)
                           + ". Entweder abwarten oder bewusst --while-jobs-running angeben (siehe Beschreibung).")
    print("Ziel:", target)
    print("Programmdateien:", ", ".join(changed) or "-")
    print("Neue Configs:", ", ".join(created) or "-")
    if jobs:
        print("Hinweis: laufende/wartende Jobs:", ", ".join(jobs))
    if dry_run:
        print("Prüfung erfolgreich; keine Änderungen ausgeführt.")
        return
    backup_parent = target.parent / "a30_update_backups"
    backup_parent.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=backup_parent))
    for name in changed:
        shutil.copy2(target / name, backup / name)
    (backup / "replaced_checksums.json").write_text(json.dumps({n: sha(target / n) for n in changed}, indent=2) + "\n")
    copied, written = [], []
    try:
        for name in changed:  # model.py, data.py, dann training.py
            atomic_copy(ROOT / name, target / name)
            copied.append(name)
        for name in created:
            atomic_copy(ROOT / "configs" / name, target / "configs" / name)
            written.append(name)
    except BaseException:
        for name in reversed(copied):
            atomic_copy(backup / name, target / name)
        for name in written:
            (target / "configs" / name).unlink(missing_ok=True)
        raise
    print("Installiert. Bestehende Configs und Ergebnisse bleiben unverändert.")
    print("Sicherung:", backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--while-jobs-running", action="store_true",
                        help="Auch bei laufenden/wartenden Projektjobs installieren (Code-Hash-Wechsel in deren versions.txt)")
    args = parser.parse_args()
    try:
        install(args.target, args.dry_run, args.while_jobs_running)
    except Exception as error:
        parser.exit(1, f"Abgebrochen: {error}\n")


if __name__ == "__main__":
    main()
