#!/usr/bin/env python3
"""Rank original PWMs with MEX metadata; select diverse representatives using Tomtom.

Python >= 3.8, standard library only. External dependency: MEME Suite Tomtom.
See README.md for the exact methods, thresholds, limitations and references.
"""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager

VERSION = '1.0.0'
BASE = Path('/data/projects/SFB_A03/jan/AT_TFBS/human301')
ASSIGNMENT_ROOT = Path('/data/projects/SFB_A03/clara/ArchThalia-ML/Input_data/best_20_motif_CHS_GHTS')
SCRIPT = Path(__file__).resolve()
STATE = BASE / 'pwm_diversity_priority.json'
TF_NAMES = '''BATF2 CREB3L3 CTCF ELF3 FLI1 FOSL2 GABPA GLI4 LEF1 LEUTX
MAX MGA MYF6 NFKB1 NR1H4 PAX7 RFX5 RORB SOX2 SRY TERF1 USF3 VDR YY1
ZBED9 ZBTB8A ZFP3 ZNF696 ZNF772 ZNF773'''.split()
METADATA_FIELDS = ['Motif', 'TF', 'Overall rank', 'Tool', 'Train data',
                   'Train dataset ID', 'Train experiment ID', 'Approved']
SUMMARY_FIELDS = ['TF', 'Assay', 'Status', 'Input_files', 'Unique_matrices',
                  'Eligible', 'Missing_metadata_or_rank', 'Not_approved',
                  'Above_rank_limit', 'Representatives', 'Selected', 'Requested', 'Message']
DECISION_FIELDS = ['TF', 'Assay', 'Matrix_ID', 'Original_header', 'Original_file',
                   'Width', 'Alias_IDs', 'MEX_overall_rank', 'Rank_motif_ID',
                   'Approved', 'Eligible', 'Status', 'Representative', 'Group',
                   'Selected_order', 'Similarity_rule', 'Evidence_query',
                   'Evidence_target', 'Tomtom_E', 'Tomtom_p', 'Tomtom_q',
                   'Overlap', 'Offset', 'Orientation', 'Copied_to']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(value):
    value = value.strip().lstrip('>').strip()
    for suffix in ('.pwm', '.ppm', '.pcm', '.pfm'):
        if value.lower().endswith(suffix):
            return value[:-len(suffix)]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.json_', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_tsv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.tsv_', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t',
                                    extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def pwm_files(folder):
    if not folder.is_dir():
        return []
    result = []
    def walk_error(error):
        raise error
    for root, dirs, files in os.walk(str(folder), onerror=walk_error):
        dirs[:] = sorted(x for x in dirs if x not in {'.ipynb_checkpoints', '__MACOSX'})
        result.extend(Path(root) / name for name in sorted(files)
                      if name.lower().endswith('.pwm') and not name.startswith('._'))
    return sorted(result)


def read_pwm(path):
    header, rows = None, []
    with path.open(encoding='utf-8-sig') as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if header is not None:
                    raise ValueError('Mehrere Motive in einer PWM-Datei: ' + str(path))
                header = canonical_id(line)
                continue
            try:
                row = tuple(Decimal(value) for value in line.split())
            except InvalidOperation as error:
                raise ValueError('{}:{}: keine Zahlenzeile'.format(path, number)) from error
            if len(row) != 4 or any(value.is_nan() for value in row):
                raise ValueError('{}:{}: vier Zahlen in A,C,G,T-Reihenfolge erwartet'.format(path, number))
            rows.append(row)
    if not rows:
        raise ValueError('Leere PWM: ' + str(path))
    return header or canonical_id(path.name), tuple(rows)


def reverse_key(key):
    return tuple(tuple(reversed(row)) for row in reversed(key))


def restore_ppm(key, args, path):
    """Invert W=ln((p+epsilon)/0.25), validate, then remove rounding drift."""
    ppm, max_error = [], 0.0
    for index, row in enumerate(key, 1):
        try:
            probabilities = [0.25 * math.exp(float(value)) - args.pwm_pseudocount
                             for value in row]
        except (OverflowError, ValueError) as error:
            raise ValueError('Unpassende Log-Odds in ' + str(path)) from error
        if any(not math.isfinite(value) or value < -1e-9 for value in probabilities):
            raise ValueError('{}: Position {} passt nicht zu ln-Log-Odds, '
                             'Hintergrund 0.25 und Pseudocount {}'.format(
                                 path, index, args.pwm_pseudocount))
        probabilities = [max(0.0, value) for value in probabilities]
        total = sum(probabilities)
        error = abs(total - 1.0)
        if total <= 0 or error > args.row_tolerance:
            raise ValueError('{}: Position {}, rekonstruierte Zeilensumme {:.12g}; '
                             'erwartet 1 +/- {}. Formatannahme pruefen.'.format(
                                 path, index, total, args.row_tolerance))
        ppm.append(tuple(value / total for value in probabilities))
        max_error = max(max_error, error)
    return tuple(ppm), max_error


@dataclass
class Motif:
    uid: str
    path: Path
    header: str
    key: tuple
    ppm: tuple
    row_error: float
    aliases: dict = field(default_factory=dict)
    rank: object = None
    rank_id: str = ''
    approved: str = ''
    reason: str = ''


def load_metadata(path):
    lookup = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        missing = set(METADATA_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Fehlende MEX-Spalten: ' + ', '.join(sorted(missing)))
        for row in reader:
            key = canonical_id(row['Motif'])
            if key in lookup and lookup[key] != row:
                raise ValueError('Mehrdeutige MEX-Motif-ID: ' + key)
            lookup[key] = row
    return lookup


def assign_rank(motif, metadata, tf, args):
    matched, ranked, acceptable = [], [], []
    for alias in sorted(motif.aliases):
        row = metadata.get(alias)
        if row is None:
            continue
        if row['TF'] != tf:
            raise ValueError('TF-Konflikt in MEX fuer ' + alias)
        matched.append(row)
        try:
            rank = int(row['Overall rank'])
        except (ValueError, TypeError):
            continue
        if rank <= 0:
            continue
        ranked.append((rank, alias, row))
        approved = row['Approved'].strip().lower() == 'yes'
        if approved or args.include_unapproved:
            acceptable.append((rank, alias, row))
    if not matched:
        motif.reason = 'excluded_missing_metadata'
        return
    if not ranked:
        motif.reason = 'excluded_missing_rank'
        return
    chosen = min(acceptable or ranked, key=lambda item: (item[0], item[1]))
    motif.rank, motif.rank_id, row = chosen
    motif.approved = row['Approved']
    if not acceptable:
        motif.reason = 'excluded_not_approved'
    elif args.max_overall_rank is not None and motif.rank > args.max_overall_rank:
        motif.reason = 'excluded_above_rank_limit'


def load_group(tf, assay, metadata, args):
    files = pwm_files(args.input_root / tf / assay)
    by_key = {}
    for path in files:
        header, key = read_pwm(path)
        if key not in by_key:
            ppm, error = restore_ppm(key, args, path)
            by_key[key] = Motif('M{:04d}'.format(len(by_key) + 1), path, header,
                                key, ppm, error)
        by_key[key].aliases.setdefault(header, set()).add(str(path))
    motifs = list(by_key.values())
    # Recover IDs that disappeared when identical matrices were deduplicated.
    if args.assignment_root is not None:
        assigned = args.assignment_root / assay / tf
        if not assigned.is_dir() and files:
            logging.warning('Keine urspruenglichen Zuordnungen: %s; nur erhaltene IDs nutzbar.', assigned)
        for path in pwm_files(assigned):
            header, key = read_pwm(path)
            motif = by_key.get(key)
            if motif is not None:
                motif.aliases.setdefault(header, set()).add(str(path))
    for motif in motifs:
        assign_rank(motif, metadata, tf, args)
    return files, motifs


def find_tomtom(explicit):
    candidates = ([explicit] if explicit else [
        shutil.which('tomtom'),
        str(Path.home() / 'miniforge3/envs/at_env/bin/tomtom'),
        str(Path.home() / 'miniforge3/envs/pwm_meme/bin/tomtom')])
    for candidate in candidates:
        if not candidate:
            continue
        located = shutil.which(str(candidate))
        if located and os.access(located, os.X_OK):
            return str(Path(located).resolve())
    raise RuntimeError('Tomtom nicht gefunden. Einmalig z.B.:\n'
                       'source ~/miniforge3/etc/profile.d/conda.sh\n'
                       'conda create -n pwm_meme -c conda-forge -c bioconda meme=5.5.8\n'
                       'Danach erneut starten oder --tomtom /absoluter/pfad/tomtom angeben.')


def command(args, check=True, timeout=45):
    result = subprocess.run([str(x) for x in args], text=True, capture_output=True,
                            timeout=timeout)
    if check and result.returncode:
        raise RuntimeError('{}: {}'.format(shlex.join([str(x) for x in args]),
                                           (result.stderr or result.stdout).strip()))
    return result


def preflight(args):
    args.input_root = args.input_root.expanduser().resolve()
    args.metadata = args.metadata.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.assignment_root is not None:
        args.assignment_root = args.assignment_root.expanduser().resolve()
    if not args.input_root.is_dir():
        raise RuntimeError('PWM-Eingabeordner fehlt: ' + str(args.input_root))
    if not args.metadata.is_file():
        raise RuntimeError('MEX-Metadaten fehlen: ' + str(args.metadata))
    if args.output == args.input_root or args.input_root in args.output.parents:
        raise RuntimeError('Ausgabe muss ausserhalb des PWM-Eingabeordners liegen.')
    if args.assignment_root is not None and (args.output == args.assignment_root
                                             or args.assignment_root in args.output.parents):
        raise RuntimeError('Ausgabe darf nicht innerhalb der Originalzuordnungen liegen.')
    if not any(pwm_files(args.input_root / tf / assay)
               for tf in args.tfs for assay in args.assays):
        raise RuntimeError('In den gewaehlten TF-/Assay-Ordnern wurden keine PWMs gefunden.')
    args.tomtom = find_tomtom(args.tomtom)
    version = command([args.tomtom, '-version']).stdout.strip()
    if not version:
        raise RuntimeError('Tomtom liefert keine Versionsnummer.')
    metadata = load_metadata(args.metadata)
    return metadata, version


def write_meme(path, motifs):
    with path.open('w', encoding='ascii') as handle:
        handle.write('MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n'
                     'Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n')
        for motif in motifs:
            handle.write('MOTIF {}\nletter-probability matrix: alength= 4 w= {} nsites= 1\n'.format(
                motif.uid, len(motif.ppm)))
            for row in motif.ppm:
                handle.write(' '.join(format(value, '.17g') for value in row) + '\n')
            handle.write('\n')


def normal_field(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def parse_tomtom(path, motifs):
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.startswith('#'):
                if not rows and normal_field(line.split('\t')[0]) == 'queryid':
                    rows.append(line.lstrip('#'))
                continue
            rows.append(line)
    if not rows:
        raise ValueError('Leere Tomtom-Ausgabe: ' + str(path))
    known = {motif.uid for motif in motifs}
    matches = {}
    for row in csv.DictReader(rows, delimiter='\t'):
        row = {normal_field(key): value for key, value in row.items() if key is not None}
        required = {'queryid', 'targetid', 'evalue', 'pvalue', 'overlap', 'optimaloffset', 'orientation'}
        if not required <= row.keys():
            raise ValueError('Unbekannte Tomtom-TSV-Spalten: ' + ','.join(row))
        query, target = row['queryid'], row['targetid']
        if query not in known or target not in known:
            raise ValueError('Unbekannte Motif-ID in Tomtom-Ausgabe')
        evalue, pvalue = float(row['evalue']), float(row['pvalue'])
        if not math.isfinite(evalue) or not math.isfinite(pvalue):
            raise ValueError('Nicht endlicher Tomtom-p-/E-Wert fuer ' + query + '/' + target)
        match = {'query': query, 'target': target, 'evalue': evalue, 'pvalue': pvalue,
                 'qvalue': row.get('qvalue', ''), 'overlap': int(row['overlap']),
                 'offset': int(row['optimaloffset']), 'orientation': row['orientation']}
        if (query, target) not in matches or evalue < matches[(query, target)]['evalue']:
            matches[(query, target)] = match
    missing = [(a, b) for a in known for b in known if (a, b) not in matches]
    if missing:
        raise ValueError('Tomtom meldet nicht alle Paarvergleiche ({} fehlen, z.B. {}). '
                         'Sie werden nicht still als unaehnlich interpretiert.'.format(len(missing), missing[:3]))
    return matches


def run_tomtom(folder, motifs, args):
    meme = folder / 'motifs_for_comparison.meme'
    write_meme(meme, motifs)
    arguments = [args.tomtom, '-text', '-dist', 'ed', '-evalue', '-thresh', '1000000',
                 '-min-overlap', str(args.min_overlap), '-motif-pseudo', '0',
                 str(meme), str(meme)]
    atomic_json(folder / 'tomtom_command.json', {'argv': arguments})
    with (folder / 'tomtom.tsv').open('w') as stdout, (folder / 'tomtom.stderr.log').open('w') as stderr:
        subprocess.run(arguments, stdout=stdout, stderr=stderr, check=True,
                       timeout=args.tomtom_timeout)
    return parse_tomtom(folder / 'tomtom.tsv', motifs)


def similarity(seed, candidate, matches, args):
    if seed.key == candidate.key:
        return {'rule': 'exact_matrix', 'orientation': '+'}
    if seed.key == reverse_key(candidate.key):
        return {'rule': 'exact_reverse_complement', 'orientation': '-'}
    directions = [(seed.uid, candidate.uid)]
    if args.direction == 'either':
        directions.append((candidate.uid, seed.uid))
    hits = []
    for pair in directions:
        match = matches[pair]
        minimum = min(args.min_overlap, len(seed.key), len(candidate.key))
        if match['evalue'] <= args.evalue and match['overlap'] >= minimum:
            hits.append(match)
    if not hits:
        return None
    return dict(min(hits, key=lambda match: (match['evalue'], match['pvalue'], match['query'])),
                rule='tomtom_E')


def select_motifs(tf, assay, motifs, matches, args):
    leaders, decisions = [], []
    ordered = sorted(motifs, key=lambda motif: (
        motif.rank is None, motif.rank if motif.rank is not None else math.inf,
        motif.rank_id, motif.header, str(motif.path)))
    for motif in ordered:
        row = {'TF': tf, 'Assay': assay, 'Matrix_ID': motif.uid,
               'Original_header': motif.header, 'Original_file': str(motif.path),
               'Width': len(motif.key), 'Alias_IDs': len(motif.aliases),
               'MEX_overall_rank': motif.rank if motif.rank is not None else '',
               'Rank_motif_ID': motif.rank_id, 'Approved': motif.approved,
               'Eligible': not bool(motif.reason), 'Status': motif.reason}
        if motif.reason:
            decisions.append(row)
            continue
        for group, seed in enumerate(leaders, 1):
            match = similarity(seed, motif, matches, args)
            if match is not None:
                row.update(Status='redundant', Representative=seed.uid, Group=group,
                           Similarity_rule=match['rule'], Evidence_query=match.get('query', seed.uid),
                           Evidence_target=match.get('target', motif.uid),
                           Tomtom_E=match.get('evalue', ''), Tomtom_p=match.get('pvalue', ''),
                           Tomtom_q=match.get('qvalue', ''), Overlap=match.get('overlap', len(motif.key)),
                           Offset=match.get('offset', 0), Orientation=match['orientation'])
                break
        else:
            leaders.append(motif)
            group = len(leaders)
            selected = group <= args.top_k
            row.update(Status='selected' if selected else 'additional_representative',
                       Representative=motif.uid, Group=group,
                       Selected_order=group if selected else '')
        decisions.append(row)
    return leaders, decisions


def alias_rows(tf, assay, motifs, metadata):
    result = []
    for motif in motifs:
        for alias, paths in sorted(motif.aliases.items()):
            entry = metadata.get(alias, {})
            result.append(dict(entry, TF=tf, Assay=assay, Matrix_ID=motif.uid,
                               Alias_ID=alias, Source_files=' | '.join(sorted(paths))))
    return result


def group_fingerprint(tf, assay, motifs, args, context):
    data = {'settings': context, 'TF': tf, 'Assay': assay, 'motifs': [
        {'path': str(motif.path), 'sha256': sha256(motif.path), 'rank': motif.rank,
         'rank_id': motif.rank_id, 'reason': motif.reason,
         'aliases': {key: sorted(value) for key, value in motif.aliases.items()}}
        for motif in motifs]}
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def process_group(tf, assay, metadata, args, context):
    files, motifs = load_group(tf, assay, metadata, args)
    folder = args.output / tf / assay
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = group_fingerprint(tf, assay, motifs, args, context)
    completion = folder / 'completion.json'
    if completion.is_file():
        previous = json.loads(completion.read_text())
        if previous['fingerprint'] != fingerprint:
            raise RuntimeError('Eingaben/Parameter geaendert fuer {} / {}. '
                               'Bitte neues --output verwenden.'.format(tf, assay))
        for record in previous.get('output_files', []):
            path = folder / record['path']
            if not path.is_file() or sha256(path) != record['sha256']:
                raise RuntimeError('Ergebnisdatei fehlt oder wurde geaendert: ' + str(path))
        logging.info('%s/%s: vorhandenes, unveraendertes Ergebnis wiederverwendet.', tf, assay)
        return previous['summary'], previous['selected']
    matches = run_tomtom(folder, motifs, args) if motifs else {}
    leaders, decisions = select_motifs(tf, assay, motifs, matches, args)
    selected_dir = folder / 'selected'
    selected_dir.mkdir(exist_ok=True)
    expected_names = set()
    by_uid = {motif.uid: motif for motif in motifs}
    for row in decisions:
        if row['Status'] != 'selected':
            continue
        motif = by_uid[row['Matrix_ID']]
        name = '{:02d}_{}_{}_{}.pwm'.format(row['Selected_order'], tf, assay, motif.uid)
        expected_names.add(name)
        destination = selected_dir / name
        # Copy original bytes, never the reconstructed comparison matrix.
        shutil.copyfile(str(motif.path), str(destination))
        if sha256(destination) != sha256(motif.path):
            raise RuntimeError('Kopierpruefung fehlgeschlagen: ' + str(destination))
        row['Copied_to'] = str(destination)
    # Only files created by an interrupted run of this script may be present.
    for existing in selected_dir.iterdir():
        if existing.name in expected_names:
            continue
        if re.fullmatch(r'\d+_' + re.escape(tf) + '_' + re.escape(assay) + r'_M\d+\.pwm', existing.name):
            existing.unlink()
        else:
            raise RuntimeError('Unerwartete Datei im Auswahlordner: ' + str(existing))
    atomic_tsv(folder / 'ranking.tsv', DECISION_FIELDS, decisions)
    atomic_tsv(folder / 'aliases.tsv', ['TF', 'Assay', 'Matrix_ID', 'Alias_ID',
               'Source_files'] + [name for name in METADATA_FIELDS if name != 'TF'],
               alias_rows(tf, assay, motifs, metadata))
    eligible = [motif for motif in motifs if not motif.reason]
    chosen = [row for row in decisions if row['Status'] == 'selected']
    message = ''
    if not motifs:
        status, message = 'NO_PWM', 'Keine Eingabe-PWMs gefunden.'
    elif len(chosen) < args.top_k:
        status = 'FEWER_THAN_REQUESTED'
        message = 'Nur {} geeignete, untereinander nicht stark aehnliche Repraesentanten.'.format(len(chosen))
    else:
        status = 'OK'
    summary = dict(TF=tf, Assay=assay, Status=status, Input_files=len(files),
                   Unique_matrices=len(motifs), Eligible=len(eligible),
                   Missing_metadata_or_rank=sum(motif.reason in {
                       'excluded_missing_metadata', 'excluded_missing_rank'} for motif in motifs),
                   Not_approved=sum(motif.reason == 'excluded_not_approved' for motif in motifs),
                   Above_rank_limit=sum(motif.reason == 'excluded_above_rank_limit' for motif in motifs),
                   Representatives=len(leaders), Selected=len(chosen), Requested=args.top_k,
                   Message=message)
    output_files = []
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path != completion:
            output_files.append({'path': str(path.relative_to(folder)), 'sha256': sha256(path)})
    atomic_json(completion, {'fingerprint': fingerprint, 'completed_utc': now(),
                            'max_reconstructed_row_error': max((m.row_error for m in motifs), default=0),
                            'summary': summary, 'selected': chosen, 'output_files': output_files})
    logging.info('%s/%s: %d Matrizen, %d geeignet, %d Repraesentanten, %d ausgewaehlt. %s',
                 tf, assay, len(motifs), len(eligible), len(leaders), len(chosen), message)
    return summary, chosen


def analysis_context(args, version):
    return {'script_version': VERSION, 'script_sha256': sha256(SCRIPT),
            'input_root': str(args.input_root),
            'assignment_root': str(args.assignment_root) if args.assignment_root else None,
            'metadata_sha256': sha256(args.metadata), 'tomtom_version': version,
            'tomtom_sha256': sha256(args.tomtom), 'distance': 'ed',
            'evalue_threshold': args.evalue, 'direction': args.direction,
            'minimum_overlap': args.min_overlap, 'top_k': args.top_k,
            'pwm_log_base': 'e', 'background_ACGT': [0.25] * 4,
            'pwm_pseudocount': args.pwm_pseudocount, 'row_tolerance': args.row_tolerance,
            'tomtom_motif_pseudocount': 0,
            'approved_only': not args.include_unapproved,
            'max_overall_rank': args.max_overall_rank}


def run_analysis(args):
    metadata, version = preflight(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.analysis.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Ein anderer Lauf bearbeitet bereits ' + str(args.output)) from error
        marker = args.output / 'run.json'
        unexpected = [p for p in args.output.iterdir() if p.name != '.analysis.lock']
        if unexpected and not marker.exists():
            raise RuntimeError('Nicht leerer Ausgabeordner ohne run.json: ' + str(args.output))
        context = analysis_context(args, version)
        if marker.exists() and json.loads(marker.read_text()).get('settings') != context:
            raise RuntimeError('Andere Parameter/Software/Metadaten im Ausgabeordner. '
                               'Bitte einen neuen --output-Pfad verwenden.')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout),
                                      logging.FileHandler(args.output / 'summary.log', encoding='utf-8')],
                            force=True)
        atomic_json(marker, {'settings': context, 'started_utc': now(),
                             'python': sys.version, 'host': socket.gethostname(),
                             'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                             'input_root': str(args.input_root), 'metadata': str(args.metadata),
                             'assignment_root': str(args.assignment_root) if args.assignment_root else None,
                             'tomtom': args.tomtom})
        logging.info('PWM-Auswahl: Top-%d je TF/Assay; Tomtom %s; E <= %g; Richtung %s.',
                     args.top_k, version, args.evalue, args.direction)
        summaries, selected, failures = [], [], 0
        for tf in args.tfs:
            for assay in args.assays:
                try:
                    summary, chosen = process_group(tf, assay, metadata, args, context)
                    summaries.append(summary)
                    selected.extend(chosen)
                except Exception as error:
                    failures += 1
                    logging.exception('%s/%s fehlgeschlagen: %s', tf, assay, error)
                    summaries.append(dict(TF=tf, Assay=assay, Status='ERROR',
                                          Requested=args.top_k, Message=str(error)))
                atomic_tsv(args.output / 'summary.tsv', SUMMARY_FIELDS, summaries)
                atomic_tsv(args.output / 'selected.tsv', DECISION_FIELDS, selected)
        total = sum(int(row.get('Unique_matrices', 0)) for row in summaries)
        logging.info('Fertig: %d TF/Assay-Gruppen, %d Matrizen, %d ausgewaehlt, %d Fehler.',
                     len(summaries), total, len(selected), failures)
        logging.info('Zusammenfassung: %s', args.output / 'summary.tsv')
        if failures:
            raise RuntimeError('{} Gruppen fehlgeschlagen; erfolgreiche Gruppen bleiben erhalten.'.format(failures))


# Queue management: only this user's pending jobs; retain their dependency
# expressions and ArrayTaskThrottle. No controller job or polling daemon.
@contextmanager
def queue_lock(path):
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def queue_snapshot():
    output = command(['squeue', '--local', '--noheader', '--array',
                      '--user=' + str(os.getuid()), '--format=%i|%t|%Q|%r']).stdout
    jobs = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split('|', 3)]
        if len(parts) != 4 or not parts[2].isdigit():
            raise RuntimeError('Unerwartete squeue-Ausgabe: ' + line)
        job_id, state, priority, reason = parts
        jobs[job_id] = (state, int(priority), reason)
    return jobs


def queue_change(action, ids):
    for start in range(0, len(ids), 100):
        result = command(['scontrol', action, ','.join(ids[start:start + 100])], check=False)
        if result.returncode:
            print('Slurm-Hinweis:', (result.stderr or result.stdout).strip(), flush=True)


def release_held(state_path, state):
    ids = state.get('held_jobs', [])
    if not ids:
        return
    before = queue_snapshot()
    ours = [job_id for job_id in ids if job_id in before
            and before[job_id][1] == 0 and before[job_id][2] == 'JobHeldUser']
    queue_change('release', ours)
    after = queue_snapshot()
    remaining = [job_id for job_id in ids if job_id in after
                 and after[job_id][1] == 0 and after[job_id][2] == 'JobHeldUser']
    state['held_jobs'] = remaining
    atomic_json(state_path, state)
    if remaining:
        raise RuntimeError('Noch gehalten: {}. Wiederherstellen mit: python {} release'.format(
            ','.join(remaining), SCRIPT))
    print('Eigene wartende Jobs wieder freigegeben: {}'.format(len(ours)), flush=True)


def worker_arguments(args):
    result = [str(Path(sys.executable).resolve()), '-u', str(SCRIPT), 'run',
              '--input-root', str(args.input_root), '--metadata', str(args.metadata),
              '--output', str(args.output), '--top-k', str(args.top_k),
              '--evalue', str(args.evalue), '--min-overlap', str(args.min_overlap),
              '--direction', args.direction, '--pwm-pseudocount', str(args.pwm_pseudocount),
              '--row-tolerance', str(args.row_tolerance), '--tomtom', args.tomtom,
              '--tomtom-timeout', str(args.tomtom_timeout),
              '--tfs', ','.join(args.tfs), '--assays', ','.join(args.assays)]
    if args.assignment_root is None:
        result.append('--no-alias-scan')
    else:
        result.extend(['--assignment-root', str(args.assignment_root)])
    if args.max_overall_rank is not None:
        result.extend(['--max-overall-rank', str(args.max_overall_rank)])
    if args.include_unapproved:
        result.append('--include-unapproved')
    return result


def submit(args):
    _, version = preflight(args)
    for tool in ('sbatch', 'squeue', 'scontrol', 'scancel'):
        if shutil.which(tool) is None:
            raise RuntimeError(tool + ' fehlt; bitte auf dem Slurm-Login-Knoten starten.')
    if not BASE.is_dir():
        raise RuntimeError('Projektordner fehlt: ' + str(BASE))
    state_path = args.state_file.expanduser().resolve()
    with queue_lock(state_path):
        jobs = queue_snapshot()
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if previous.get('held_jobs') or previous.get('job_id') in jobs:
                raise RuntimeError('Vorheriger PWM-Auswahljob aktiv oder Holds offen. '
                                   'Pruefen bzw. nach Abbruch ausfuehren: python {} release'.format(SCRIPT))
        pending = [] if args.normal_queue else [
            job_id for job_id, (state, priority, reason) in jobs.items()
            if state == 'PD' and priority > 0 and not reason.startswith('JobHeld')]
        if any(not re.fullmatch(r'\d+(?:_\d+)?', job_id) for job_id in pending):
            raise RuntimeError('Nicht unterstuetzte Job-ID; keine Warteschlangenaenderung vorgenommen.')
        worker = worker_arguments(args) + ['--priority-state', str(state_path)]
        log = BASE / 'pwm_diverse_%j.log'
        result = command(['sbatch', '--parsable', '--hold', '--export=ALL',
                          '--job-name=pwm_diverse', '--nodes=1', '--ntasks=1',
                          '--cpus-per-task=1', '--mem=4G', '--time=' + args.time,
                          '--chdir=' + str(BASE), '--output=' + str(log),
                          '--wrap=exec ' + shlex.join(worker)])
        job_id = result.stdout.strip().split(';', 1)[0]
        if not job_id.isdigit():
            raise RuntimeError('Unerwartete sbatch-Antwort: ' + result.stdout)
        state = {'job_id': job_id, 'held_jobs': pending, 'status': 'preparing',
                 'created_utc': now(), 'script': str(SCRIPT), 'output': str(args.output)}
        print('Job {}. Falls vor dem Start abgebrochen: python {} release --state-file {}'.format(
            job_id, SCRIPT, state_path), flush=True)
        try:
            atomic_json(state_path, state)
            queue_change('hold', pending)
            after = queue_snapshot()
            not_held = [job_id for job_id in pending if job_id in after
                        and after[job_id][0] == 'PD' and after[job_id][1] > 0]
            if not_held:
                raise RuntimeError('Hold fehlgeschlagen fuer ' + ','.join(not_held))
            raced = [job_id for job_id in pending if job_id in after
                     and after[job_id][0] != 'PD' and after[job_id][1] == 0
                     and after[job_id][2] == 'JobHeldUser']
            queue_change('release', raced)
            if raced:
                checked = queue_snapshot()
                if any(job_id in checked and checked[job_id][1] == 0
                       and checked[job_id][2] == 'JobHeldUser' for job_id in raced):
                    raise RuntimeError('Ein gleichzeitig gestarteter Task blieb gehalten.')
            state['status'] = 'waiting'
            atomic_json(state_path, state)
            command(['scontrol', 'release', job_id])
        except BaseException:
            try:
                command(['scancel', job_id])
            finally:
                release_held(state_path, state)
            raise
        print('Submitted batch job ' + job_id)
        print('Tomtom: {} ({})'.format(args.tomtom, version))
        print('Zurueckgestellte eigene wartende Jobs/Tasks: {}'.format(len(pending)))
        print('Slurm-Log: ' + str(BASE / ('pwm_diverse_' + job_id + '.log')))
        print('Ergebnisse: ' + str(args.output))


def comma_list(value):
    items = [x.strip() for x in value.split(',') if x.strip()]
    if not items or len(items) != len(set(items)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in items):
        raise argparse.ArgumentTypeError('Eindeutige, kommaseparierte Namen erwartet.')
    return items


def make_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--input-root', type=Path, default=BASE / 'pwms')
    common.add_argument('--metadata', type=Path, default=SCRIPT.parent / 'MEX_metadata_30TF.tsv')
    common.add_argument('--assignment-root', type=Path, default=ASSIGNMENT_ROOT)
    common.add_argument('--no-alias-scan', action='store_true', help='Keine Originalzuordnungen erneut lesen.')
    common.add_argument('--output', type=Path, default=BASE / 'pwms_diverse')
    common.add_argument('--top-k', type=int, default=4)
    common.add_argument('--evalue', type=float, default=0.05)
    common.add_argument('--min-overlap', type=int, default=5)
    common.add_argument('--direction', choices=['either', 'seed'], default='either',
                        help='either: E<=Schwelle in mindestens einer Richtung; seed: nur Repraesentant als Query.')
    common.add_argument('--include-unapproved', action='store_true', help='Auch MEX Approved!=yes fuer Auswahl zulassen.')
    common.add_argument('--max-overall-rank', type=int, default=None)
    common.add_argument('--pwm-pseudocount', type=float, default=1e-5)
    common.add_argument('--row-tolerance', type=float, default=5e-6)
    common.add_argument('--tomtom', default=None, help='Pfad zur Tomtom-Executable.')
    common.add_argument('--tomtom-timeout', type=int, default=1800, help='Maximale Sekunden pro TF/Assay.')
    common.add_argument('--tfs', type=comma_list, default=TF_NAMES)
    common.add_argument('--assays', type=comma_list, default=['CHS', 'GHTS'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=VERSION)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', parents=[common], help='Pfade, Metadaten und Tomtom pruefen; keine Jobs aendern.')
    direct = sub.add_parser('run', parents=[common], help='Analyse im aktuellen Prozess ausfuehren.')
    direct.add_argument('--priority-state', type=Path, default=None, help=argparse.SUPPRESS)
    batch = sub.add_parser('submit', parents=[common], help='Als priorisierten CPU-Slurm-Job einreichen.')
    batch.add_argument('--state-file', type=Path, default=STATE)
    batch.add_argument('--normal-queue', action='store_true', help='Normal einreihen; andere Jobs nicht zurueckstellen.')
    batch.add_argument('--time', default='02:00:00')
    release = sub.add_parser('release', help='Nach Abbruch vor Jobstart die eigenen Holds aufheben.')
    release.add_argument('--state-file', type=Path, default=STATE)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.command == 'release':
        state_path = args.state_file.expanduser().resolve()
        with queue_lock(state_path):
            if not state_path.exists():
                print('Keine gespeicherten Holds vorhanden.')
                return
            release_held(state_path, json.loads(state_path.read_text()))
        return
    # Once the allocation starts, restore the queue before dependency checks or analysis.
    if getattr(args, 'priority_state', None) is not None:
        path = args.priority_state.expanduser().resolve()
        with queue_lock(path):
            state = json.loads(path.read_text())
            if os.environ.get('SLURM_JOB_ID') != state['job_id']:
                raise RuntimeError('Job-ID passt nicht zur gespeicherten Prioritaetsdatei.')
            release_held(path, state)
            state['status'] = 'started'
            atomic_json(path, state)
    if args.no_alias_scan:
        args.assignment_root = None
    if args.top_k < 1 or args.min_overlap < 1 or args.evalue <= 0 or not math.isfinite(args.evalue):
        parser.error('top-k/min-overlap und E-Wert-Schwelle muessen positiv sein.')
    if args.pwm_pseudocount < 0 or not math.isfinite(args.pwm_pseudocount) or args.row_tolerance <= 0:
        parser.error('Ungueltiger PWM-Pseudocount oder ungueltige Zeilentoleranz.')
    if args.max_overall_rank is not None and args.max_overall_rank < 1:
        parser.error('max-overall-rank muss positiv sein.')
    if args.tomtom_timeout < 1:
        parser.error('tomtom-timeout muss positiv sein.')
    if set(args.assays) - {'CHS', 'GHTS'}:
        parser.error('Zulaessige Assays: CHS,GHTS')
    if args.command == 'check':
        _, version = preflight(args)
        print('Pruefung erfolgreich. Eingabe: {}\nMetadaten: {}\nTomtom: {} ({})\nAusgabe: {}'.format(
            args.input_root, args.metadata, args.tomtom, version, args.output))
    elif args.command == 'submit':
        submit(args)
    else:
        run_analysis(args)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('ABGEBROCHEN: {}'.format(error), file=sys.stderr, flush=True)
        sys.exit(1)
