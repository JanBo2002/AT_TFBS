#!/usr/bin/env python3
"""Compare three Basic+CCR weighting rules using existing cross-assay scores.

Input: prediction paths recorded in comparison_summary.csv, 00_basic,
output_ccr, seed 13, CHS_to_GHTS and GHTS_to_CHS, all 30 TFs.

No model training, score normalization, percentiles or calibration.
All mixtures use the ORIGINAL scores B and C:
  fixed:                s = alpha*B + (1-alpha)*C
  ap_weighted:          w = sigmoid(beta*(AP_B_valid - AP_C_valid))
  ap_sequence_weighted: w = sigmoid(beta*(AP_B_valid - AP_C_valid)
                                   + gamma*(B-C))
  For the latter two:   s = w*B + (1-w)*C.
Sequence-dependent weighting here favors the higher POSITIVE score.

Each repeat uses a deterministic genomic-block split, approximately 50:50.
Whole 5-Mb blocks are assigned consistently across TFs. Windows crossing a
block boundary are excluded and counted, so overlapping retained windows
cannot occur on opposite sides. Labels are never used to assign blocks.
Both subsets must contain positives AND negatives; otherwise that TF/repeat
is reported as an error. There is no fallback to a random sequence split.

For each scenario/repeat, parameters are chosen by maximum UNWEIGHTED MEAN
validation AP across available TFs. Alpha/beta/gamma are shared across TFs;
validation AP differences are TF-specific. Evaluation labels are not used
in parameter selection. Evaluation scores are used only to apply the rule.
Exact numerical ties prefer larger alpha, smaller beta, and then (for
sequence weighting) smaller gamma before smaller beta. Tolerance: 1e-12.

Defaults: alpha=0,0.1,...,1; beta=0,5,10,15,20,30;
gamma=0,1,2,5,10,20; 3 repeats; split seeds 13,14,15.

Outputs (only two CSVs; saved after every completed evaluation pair):
  basic_ccr_simulation_by_tf.csv   -- one TF/scenario/repeat per row
  basic_ccr_simulation_summary.csv -- method means, parameters, repeat means
The summary's repeat=ALL first averages within TF across repeats, then
across TFs. Missing/invalid cases have blank AP, never zero. All five
methods/baselines are evaluated on exactly the same rows within a pair.

IMPORTANT: these are simulated validation/evaluation splits of an already
examined test dataset, not a new independent test or reconstructed epochs.
Requires numpy, pandas, scipy and scikit-learn (existing at_env).
Example: python -u basic_ccr_simulation.py --repeats 1
"""

import argparse
import csv
import hashlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score


RESULTS = Path('/data/projects/SFB_A03/jan/AT_TFBS/human301/big_ablation/results')
DEFAULT_SUMMARY = RESULTS / 'prediction_overlap' / 'comparison_summary.csv'
SCENARIOS = ('CHS_to_GHTS', 'GHTS_to_CHS')
TFS = (
    'BATF2', 'CREB3L3', 'CTCF', 'ELF3', 'FLI1', 'FOSL2', 'GABPA', 'GLI4',
    'LEF1', 'LEUTX', 'MAX', 'MGA', 'MYF6', 'NFKB1', 'NR1H4', 'PAX7',
    'RFX5', 'RORB', 'SOX2', 'SRY', 'TERF1', 'USF3', 'VDR', 'YY1', 'ZBED9',
    'ZBTB8A', 'ZFP3', 'ZNF696', 'ZNF772', 'ZNF773',
)
BETAS = (0., 5., 10., 15., 20., 30.)
GAMMAS = (0., 1., 2., 5., 10., 20.)
RULES = ('fixed', 'ap_weighted', 'ap_sequence_weighted')
METHODS = ('basic', 'ccr') + RULES
DETAIL_FIELDS = [
    'scenario', 'repeat', 'split_seed', 'TF', 'status', 'n_sequences',
    'n_boundary_excluded', 'n_valid', 'n_eval', 'n_positive_valid',
    'n_negative_valid', 'n_positive_eval', 'n_negative_eval',
] + ['AP_' + m + '_valid' for m in METHODS] + [
    'AP_' + m + '_eval' for m in METHODS
] + [
    'alpha_fixed', 'beta_ap_weighted', 'beta_ap_sequence_weighted',
    'gamma_ap_sequence_weighted', 'w_basic_ap_weighted',
    'n_TFs_for_selection', 'block_bp', 'validation_fraction', 'message',
]
SUMMARY_FIELDS = [
    'scenario', 'repeat', 'method', 'status', 'n_TFs', 'n_TF_runs',
    'n_repeats', 'AP_valid_mean', 'AP_eval_mean',
    'delta_eval_vs_basic', 'delta_eval_vs_ccr', 'alpha', 'beta', 'gamma',
    'message',
]


@dataclass(frozen=True)
class Candidate:
    method: str
    alpha: float = 0.
    beta: float = 0.
    gamma: float = 0.


CANDIDATES = (
    tuple(Candidate('fixed', alpha=a / 10.) for a in range(10, -1, -1))
    + tuple(Candidate('ap_weighted', beta=b) for b in BETAS)
    + tuple(Candidate('ap_sequence_weighted', beta=b, gamma=g)
            for g in GAMMAS for b in BETAS)
)


def ap(y, scores):
    return float(average_precision_score(y, scores))


def labels_scores(frame, label, score, source):
    y = pd.to_numeric(frame[label], errors='raise').to_numpy()
    p = pd.to_numeric(frame[score], errors='raise').to_numpy(dtype=np.float64)
    if not np.isin(y, [0, 1]).all():
        raise ValueError(source + ': labels must be 0 or 1')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(source + ': scores must be finite and in [0,1]')
    return y.astype(np.uint8), p


def coord_index(chrom, start, end, source):
    if chrom.isna().any() or chrom.str.contains(r'\s|^$', regex=True).any():
        raise ValueError(source + ': invalid chromosome')
    if (start < 0).any() or (end <= start).any():
        raise ValueError(source + ': invalid interval')
    index = pd.MultiIndex.from_arrays([chrom, start, end])
    if not index.is_unique:
        raise ValueError(source + ': duplicate genomic coordinates')
    return index


def load_pair(own_file, reference_file):
    own = pd.read_csv(
        own_file, sep='\t', usecols=['sample_id', 'true_label', 'binding_probability'],
        dtype={'sample_id': 'string'},
    )
    coords = own.sample_id.str.extract(r'(?:^|:)([^:\s]+):(\d+)-(\d+)$')
    if coords.isna().any().any():
        raise ValueError('Basic: sample_id must end in chrom:start-end')
    oi = coord_index(coords[0], coords[1].astype(np.int64),
                     coords[2].astype(np.int64), 'Basic')
    y, b = labels_scores(own, 'true_label', 'binding_probability', 'Basic')
    del own, coords
    ref = pd.read_csv(
        reference_file,
        usecols=['bed_chrom', 'bed_start', 'bed_end', 'label', 'pred_proba'],
        dtype={'bed_chrom': 'string', 'bed_start': np.int64, 'bed_end': np.int64},
    )
    ri = coord_index(ref.bed_chrom, ref.bed_start, ref.bed_end, 'CCR')
    ry, c = labels_scores(ref, 'label', 'pred_proba', 'CCR')
    if len(oi) != len(ri):
        raise ValueError('Different sequence counts in Basic and CCR')
    order = ri.get_indexer(oi)
    if (order < 0).any():
        raise ValueError('Different genomic coordinates in Basic and CCR')
    if not np.array_equal(y, ry[order]):
        raise ValueError('Labels disagree after genomic alignment')
    if np.unique(y).size != 2:
        raise ValueError('Positive and negative labels are required')
    chrom = oi.get_level_values(0).to_numpy()
    start = oi.get_level_values(1).to_numpy(dtype=np.int64)
    end = oi.get_level_values(2).to_numpy(dtype=np.int64)
    return y, b, c[order], chrom, start, end


def load_manifest(path):
    data = pd.read_csv(path)
    required = {
        'own_model', 'reference_model', 'scenario', 'TF', 'seed', 'status',
        'own_file', 'reference_file', 'n_sequences', 'n_positive', 'n_negative',
        'ap_own', 'ap_reference',
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError('Summary is missing columns: ' + ', '.join(sorted(missing)))
    data = data[data.own_model.eq('00_basic') & data.reference_model.eq('output_ccr')
                & data.scenario.isin(SCENARIOS)]
    if data.duplicated(['TF', 'scenario']).any():
        raise ValueError('Duplicate Basic/CCR TF/scenario rows in summary')
    if set(data.TF) - set(TFS):
        raise ValueError('Unexpected TFs: ' + str(sorted(set(data.TF) - set(TFS))))
    return {(r['scenario'], r['TF']): r for r in data.to_dict('records')}


def paths_and_stamp(record):
    if record is None:
        raise FileNotFoundError('Basic/CCR pair absent from summary')
    if record['status'] != 'ok':
        raise ValueError('Previous analysis status: ' + str(record['status']))
    if int(record['seed']) != 13:
        raise ValueError('Expected prediction seed 13')
    paths = (Path(record['own_file']), Path(record['reference_file']))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing prediction file(s): ' + ' | '.join(missing))
    stamp = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    return paths, stamp


def validate_originals(y, b, c, record):
    actual = (len(y), int(y.sum()), int(len(y) - y.sum()))
    expected = tuple(int(record[k]) for k in ('n_sequences', 'n_positive', 'n_negative'))
    if actual != expected:
        raise ValueError('Sequence/class counts differ from previous analysis')
    for value, key in ((ap(y, b), 'ap_own'), (ap(y, c), 'ap_reference')):
        if not np.isclose(value, float(record[key]), rtol=1e-6, atol=1e-8):
            raise ValueError('Original AP differs from previous analysis: {} vs {}'.format(
                value, record[key]))


def make_blocks(chrom, start, end, block_bp):
    blocks = start // block_bp
    inside = blocks == (end - 1) // block_bp
    codes, names = pd.factorize(pd.MultiIndex.from_arrays([chrom, blocks]), sort=False)
    return codes, list(names), inside


def split_masks(codes, names, inside, seed, fraction):
    # Cryptographic hash avoids Python's process-randomized hash and makes
    # the assignment identical across TFs and separate runs.
    assignments = []
    for chrom, block in names:
        key = '{}\0{}\0{}'.format(seed, chrom, int(block)).encode('utf-8')
        number = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big')
        assignments.append(number / float(2 ** 64) < fraction)
    valid = np.asarray(assignments, dtype=bool)[codes] & inside
    evaluation = inside & ~valid
    return valid, evaluation


def split_counts(y, valid, evaluation, inside):
    for name, mask in (('validation', valid), ('evaluation', evaluation)):
        positive = int(y[mask].sum())
        negative = int(mask.sum()) - positive
        if positive == 0 or negative == 0:
            raise ValueError('Genomic {} split lacks a class ({} positives, {} negatives)'.format(
                name, positive, negative))
    return {
        'n_sequences': len(y), 'n_boundary_excluded': int((~inside).sum()),
        'n_valid': int(valid.sum()), 'n_eval': int(evaluation.sum()),
        'n_positive_valid': int(y[valid].sum()),
        'n_negative_valid': int(valid.sum() - y[valid].sum()),
        'n_positive_eval': int(y[evaluation].sum()),
        'n_negative_eval': int(evaluation.sum() - y[evaluation].sum()),
    }


def scores(candidate, b, c, delta_ap):
    if candidate.method == 'fixed':
        w = candidate.alpha
        if w == 1.:
            return b
        if w == 0.:
            return c
    elif candidate.method == 'ap_weighted' or candidate.gamma == 0.:
        w = float(expit(candidate.beta * delta_ap))
    else:
        w = expit(candidate.beta * delta_ap + candidate.gamma * (b - c))
    return w * b + (1. - w) * c


def validation_search(y, b, c):
    ab, ac = ap(y, b), ap(y, c)
    delta = ab - ac
    cache = {('linear', 1.): ab, ('linear', 0.): ac}
    values = []
    for candidate in CANDIDATES:
        if candidate.method == 'fixed':
            key = ('linear', candidate.alpha)
        elif candidate.method == 'ap_weighted' or candidate.gamma == 0.:
            key = ('linear', float(expit(candidate.beta * delta)))
        else:
            key = ('dynamic', candidate.beta * delta, candidate.gamma)
        if key not in cache:
            cache[key] = ap(y, scores(candidate, b, c, delta))
        values.append(cache[key])
    return ab, ac, tuple(values)


def select_parameters(infos):
    """Selection depends ONLY on candidate validation AP, never evaluation AP."""
    if not infos:
        return {}
    means = np.asarray([info['candidate_APs'] for info in infos]).mean(axis=0)
    chosen = {}
    for method in RULES:
        indices = [i for i, c in enumerate(CANDIDATES) if c.method == method]
        best = max(means[i] for i in indices)
        tied = [i for i in indices if means[i] >= best - 1e-12]
        if method == 'fixed':
            index = max(tied, key=lambda i: CANDIDATES[i].alpha)
        elif method == 'ap_weighted':
            index = min(tied, key=lambda i: CANDIDATES[i].beta)
        else:
            index = min(tied, key=lambda i: (CANDIDATES[i].gamma, CANDIDATES[i].beta))
        chosen[method] = index
    return chosen


def atomic_csv(path, fields, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=str(path.parent), prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def params_for_summary(method, detail):
    if method == 'fixed':
        return {'alpha': detail['alpha_fixed']}
    if method == 'ap_weighted':
        return {'beta': detail['beta_ap_weighted']}
    if method == 'ap_sequence_weighted':
        return {'beta': detail['beta_ap_sequence_weighted'],
                'gamma': detail['gamma_ap_sequence_weighted']}
    return {}


def summary_rows(details, repeats):
    result = []
    for scenario in SCENARIOS:
        for repeat in range(1, repeats + 1):
            group = [r for r in details if r['scenario'] == scenario and r['repeat'] == repeat]
            complete = [r for r in group if r['status'] == 'ok']
            selectable = [r for r in group if 'alpha_fixed' in r]
            for method in METHODS:
                row = {'scenario': scenario, 'repeat': repeat, 'method': method,
                       'status': 'ok' if len(complete) == len(TFS) else 'partial',
                       'n_TFs': len(complete), 'n_TF_runs': len(complete), 'n_repeats': 1,
                       'message': 'Unweighted TF mean; {}/{} evaluated; {} used for selection'.format(
                           len(complete), len(TFS), len(selectable))}
                if selectable:
                    row.update(params_for_summary(method, selectable[0]))
                if complete:
                    row.update({
                        'AP_valid_mean': float(np.mean([r['AP_' + method + '_valid'] for r in complete])),
                        'AP_eval_mean': float(np.mean([r['AP_' + method + '_eval'] for r in complete])),
                        'delta_eval_vs_basic': float(np.mean([
                            r['AP_' + method + '_eval'] - r['AP_basic_eval'] for r in complete])),
                        'delta_eval_vs_ccr': float(np.mean([
                            r['AP_' + method + '_eval'] - r['AP_ccr_eval'] for r in complete])),
                    })
                result.append(row)
        complete = [r for r in details if r['scenario'] == scenario and r['status'] == 'ok']
        groups = {tf: [r for r in complete if r['TF'] == tf] for tf in TFS}
        groups = {tf: rows for tf, rows in groups.items() if rows}
        for method in METHODS:
            row = {'scenario': scenario, 'repeat': 'ALL', 'method': method,
                   'status': 'ok' if len(complete) == len(TFS) * repeats else 'partial',
                   'n_TFs': len(groups), 'n_TF_runs': len(complete),
                   'n_repeats': len({r['repeat'] for r in complete}),
                   'message': 'Mean over repeats within each TF, then unweighted TF mean; parameters are in individual-repeat rows; exploratory'}
            if groups:
                value = lambda column: float(np.mean([
                    np.mean([r[column] for r in rows]) for rows in groups.values()]))
                row.update({'AP_valid_mean': value('AP_' + method + '_valid'),
                            'AP_eval_mean': value('AP_' + method + '_eval'),
                            'delta_eval_vs_basic': value('AP_' + method + '_eval') - value('AP_basic_eval'),
                            'delta_eval_vs_ccr': value('AP_' + method + '_eval') - value('AP_ccr_eval')})
            result.append(row)
    return result


def save_outputs(outdir, details, repeats):
    atomic_csv(outdir / 'basic_ccr_simulation_by_tf.csv', DETAIL_FIELDS, details)
    atomic_csv(outdir / 'basic_ccr_simulation_summary.csv', SUMMARY_FIELDS,
               summary_rows(details, repeats))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument('--outdir', type=Path, default=None,
                        help='Default: folder containing comparison_summary.csv')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--split-seed', type=int, default=13)
    parser.add_argument('--validation-fraction', type=float, default=.5,
                        help='Expected fraction of genomic BLOCKS used for validation')
    parser.add_argument('--block-bp', type=int, default=5_000_000)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.block_bp < 1 or not 0. < args.validation_fraction < 1.:
        parser.error('Require repeats>=1, block-bp>=1, 0<validation-fraction<1')
    outdir = args.outdir or args.summary.parent
    outputs = [outdir / 'basic_ccr_simulation_by_tf.csv',
               outdir / 'basic_ccr_simulation_summary.csv']
    if args.summary.resolve() in [p.resolve() for p in outputs]:
        parser.error('Output must differ from input summary')
    try:
        manifest = load_manifest(args.summary)
        outdir.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        print('Cannot initialize: ' + str(exc), file=sys.stderr)
        return 1
    started = time.monotonic()
    keys = [(s, tf) for s in SCENARIOS for tf in TFS]
    errors = []
    for key in keys:
        try:
            paths_and_stamp(manifest.get(key))
        except (ValueError, OSError, TypeError) as exc:
            errors.append((key, str(exc)))
    print('Preflight: {} expected TF/scenario pairs; {} available; {} unavailable/invalid'.format(
        len(keys), len(keys) - len(errors), len(errors)), flush=True)
    for key, message in errors:
        print('{} {}: {}'.format(*key, message), file=sys.stderr, flush=True)
    print('Exploratory pseudo-validation/evaluation; {} block splits; {} candidate settings per split'.format(
        args.repeats, len(CANDIDATES)), flush=True)
    infos, details, stamps = {}, [], {}
    exception_types = (ValueError, TypeError, OSError, pd.errors.ParserError, OverflowError)

    # First pass: ONLY validation labels influence parameter selection.
    for number, key in enumerate(keys, 1):
        scenario, tf = key
        print('[validation {}/{}] {} {}'.format(number, len(keys), scenario, tf), flush=True)
        rows = []
        for repeat in range(1, args.repeats + 1):
            rows.append({'scenario': scenario, 'TF': tf, 'repeat': repeat,
                         'split_seed': args.split_seed + repeat - 1,
                         'block_bp': args.block_bp,
                         'validation_fraction': args.validation_fraction,
                         'status': 'pending_evaluation', 'message': ''})
        details.extend(rows)
        try:
            record = manifest.get(key)
            paths, stamp = paths_and_stamp(record)
            y, b, c, chrom, start, end = load_pair(*paths)
            validate_originals(y, b, c, record)
            if paths_and_stamp(record)[1] != stamp:
                raise ValueError('Prediction file changed while reading')
            stamps[key] = stamp
            codes, names, inside = make_blocks(chrom, start, end, args.block_bp)
            for row in rows:
                try:
                    valid, evaluation = split_masks(codes, names, inside, row['split_seed'],
                                                    args.validation_fraction)
                    row.update(split_counts(y, valid, evaluation, inside))
                    ab, ac, values = validation_search(y[valid], b[valid], c[valid])
                    row.update(AP_basic_valid=ab, AP_ccr_valid=ac)
                    infos[(scenario, tf, row['repeat'])] = {'candidate_APs': values, 'row': row}
                except exception_types as exc:
                    row.update(status='error', message=str(exc))
                    print('SKIP {} {} repeat {}: {}'.format(scenario, tf, row['repeat'], exc),
                          file=sys.stderr, flush=True)
            del y, b, c, chrom, start, end, codes, names, inside
        except exception_types as exc:
            for row in rows:
                row.update(status='missing' if isinstance(exc, FileNotFoundError) else 'error',
                           message=str(exc))
            print('SKIP {} {}: {}'.format(scenario, tf, exc), file=sys.stderr, flush=True)
        save_outputs(outdir, details, args.repeats)

    # One common choice per method, scenario and repeat, using macro AP.
    for scenario in SCENARIOS:
        for repeat in range(1, args.repeats + 1):
            cohort = [v for (s, tf, r), v in infos.items() if s == scenario and r == repeat]
            selected = select_parameters(cohort)
            if not selected:
                continue
            choices = {method: CANDIDATES[i] for method, i in selected.items()}
            print('Selected {} repeat {} ({} TFs): {}'.format(scenario, repeat, len(cohort), choices),
                  flush=True)
            for info in cohort:
                row = info['row']
                for method, index in selected.items():
                    row['AP_' + method + '_valid'] = info['candidate_APs'][index]
                row.update(alpha_fixed=choices['fixed'].alpha,
                           beta_ap_weighted=choices['ap_weighted'].beta,
                           beta_ap_sequence_weighted=choices['ap_sequence_weighted'].beta,
                           gamma_ap_sequence_weighted=choices['ap_sequence_weighted'].gamma,
                           w_basic_ap_weighted=float(expit(choices['ap_weighted'].beta
                               * (row['AP_basic_valid'] - row['AP_ccr_valid']))),
                           n_TFs_for_selection=len(cohort))
                info['choices'] = choices
    save_outputs(outdir, details, args.repeats)

    # Second pass: freeze all parameters and evaluate on the other blocks.
    # Re-read one pair at a time, avoiding large prediction cache files.
    for number, key in enumerate(keys, 1):
        scenario, tf = key
        pair_infos = [infos[(scenario, tf, r)] for r in range(1, args.repeats + 1)
                      if (scenario, tf, r) in infos]
        if not pair_infos:
            continue
        print('[evaluation {}/{}] {} {}'.format(number, len(keys), scenario, tf), flush=True)
        try:
            paths, stamp = paths_and_stamp(manifest[key])
            if stamp != stamps[key]:
                raise ValueError('Prediction file changed after validation; restart with stable input files')
            y, b, c, chrom, start, end = load_pair(*paths)
            if paths_and_stamp(manifest[key])[1] != stamp:
                raise ValueError('Prediction file changed while reading')
            codes, names, inside = make_blocks(chrom, start, end, args.block_bp)
            for info in pair_infos:
                row = info['row']
                valid, evaluation = split_masks(codes, names, inside, row['split_seed'],
                                                args.validation_fraction)
                counts = split_counts(y, valid, evaluation, inside)
                if any(row[k] != v for k, v in counts.items()):
                    raise ValueError('Split changed after validation')
                ey, eb, ec = y[evaluation], b[evaluation], c[evaluation]
                values = {'AP_basic_eval': ap(ey, eb), 'AP_ccr_eval': ap(ey, ec)}
                delta = row['AP_basic_valid'] - row['AP_ccr_valid']
                for method in RULES:
                    values['AP_' + method + '_eval'] = ap(
                        ey, scores(info['choices'][method], eb, ec, delta))
                row.update(values)
                row['status'] = 'ok'
            del y, b, c, chrom, start, end, codes, names, inside
        except exception_types as exc:
            for info in pair_infos:
                info['row'].update(status='error', message=str(exc))
                for method in METHODS:
                    info['row'].pop('AP_' + method + '_eval', None)
            print('EVALUATION ERROR {} {}: {}'.format(scenario, tf, exc), file=sys.stderr, flush=True)
        save_outputs(outdir, details, args.repeats)

    summary = summary_rows(details, args.repeats)
    print('\nMean AP on held-out blocks (repeat=ALL):', flush=True)
    for row in summary:
        if row['repeat'] == 'ALL':
            print('{} {:22s} AP={} delta_vs_basic={} TFs={}'.format(
                row['scenario'], row['method'],
                '{:.6f}'.format(row['AP_eval_mean']) if 'AP_eval_mean' in row else 'NA',
                '{:+.6f}'.format(row['delta_eval_vs_basic']) if 'delta_eval_vs_basic' in row else 'NA',
                row['n_TFs']), flush=True)
    failures = sum(row['status'] != 'ok' for row in details)
    print('\nCompleted {}/{} TF/scenario/repeat cases in {:.1f} minutes.\n{}\n{}'.format(
        len(details) - failures, len(details), (time.monotonic() - started) / 60., *outputs), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
