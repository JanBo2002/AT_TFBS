#!/usr/bin/env python3
"""Export real Human301 GC histograms and plot within/across-assay comparisons.

Reads existing caches only. Default paths and expected counts/means come from
the supplied cross-assay reports (data 68191, training 68192, summary 68193).
Output: gc_histograms.csv.gz plus one train/test figure per TF, as PNG and PDF.
Run inside at_env; requires numpy and matplotlib. No model or genome loading.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

DATA_ROOT = Path('/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human')
CACHE_SPECS = json.loads(r'''{
  "BATF2": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/BATF2/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 11216,
          "negative": 1121600
        },
        "validation": {
          "positive": 2891,
          "negative": 289100
        },
        "test": {
          "positive": 12258,
          "negative": 1225800
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5444586163098752,
          "negative": 0.4972220895872531
        },
        "validation": {
          "positive": 0.5122774876541578,
          "negative": 0.4769405797118104
        },
        "test": {
          "positive": 0.5256280618239776,
          "negative": 0.4855516717267563
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/BATF2/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 543,
          "negative": 54300
        },
        "validation": {
          "positive": 171,
          "negative": 17100
        },
        "test": {
          "positive": 775,
          "negative": 77500
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5114821890807169,
          "negative": 0.5114766615884437
        },
        "validation": {
          "positive": 0.496771758991441,
          "negative": 0.4968984476695614
        },
        "test": {
          "positive": 0.48829493087557607,
          "negative": 0.4882858428892938
        }
      }
    }
  },
  "CREB3L3": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/CREB3L3/hum301_20261005_185818_tdUK1h/dataset.npz",
      "counts": {
        "train": {
          "positive": 363,
          "negative": 36300
        },
        "validation": {
          "positive": 124,
          "negative": 12400
        },
        "test": {
          "positive": 398,
          "negative": 39800
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.507399577167019,
          "negative": 0.5074101022303982
        },
        "validation": {
          "positive": 0.483388704318937,
          "negative": 0.4834037080698746
        },
        "test": {
          "positive": 0.48591796190253594,
          "negative": 0.485920048748727
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/CREB3L3/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 1425,
          "negative": 142500
        },
        "validation": {
          "positive": 415,
          "negative": 41500
        },
        "test": {
          "positive": 1592,
          "negative": 159200
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5522497845348607,
          "negative": 0.5521887043189367
        },
        "validation": {
          "positive": 0.5357921519966911,
          "negative": 0.5358044270103671
        },
        "test": {
          "positive": 0.5416951868979449,
          "negative": 0.5416610669627206
        }
      }
    }
  },
  "CTCF": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/CTCF/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 44203,
          "negative": 4420300
        },
        "validation": {
          "positive": 12912,
          "negative": 1291200
        },
        "test": {
          "positive": 51547,
          "negative": 5154700
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.50709394808894,
          "negative": 0.4475993902489894
        },
        "validation": {
          "positive": 0.47435309604087167,
          "negative": 0.4031380013750118
        },
        "test": {
          "positive": 0.48748589085585664,
          "negative": 0.44758607488298746
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/CTCF/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 10821,
          "negative": 1082100
        },
        "validation": {
          "positive": 2562,
          "negative": 256200
        },
        "test": {
          "positive": 11114,
          "negative": 1111400
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.540023229103248,
          "negative": 0.511726647551626
        },
        "validation": {
          "positive": 0.5144742609205329,
          "negative": 0.49500532961945737
        },
        "test": {
          "positive": 0.5216412231158769,
          "negative": 0.502316873692574
        }
      }
    }
  },
  "ELF3": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/ELF3/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 31860,
          "negative": 3186000
        },
        "validation": {
          "positive": 9263,
          "negative": 926300
        },
        "test": {
          "positive": 37515,
          "negative": 3751500
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.4783118835937125,
          "negative": 0.4468054674416519
        },
        "validation": {
          "positive": 0.452851239982456,
          "negative": 0.43368110114078695
        },
        "test": {
          "positive": 0.46277104307778544,
          "negative": 0.4393956012279473
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/ELF3/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 15765,
          "negative": 1576500
        },
        "validation": {
          "positive": 4348,
          "negative": 434800
        },
        "test": {
          "positive": 18174,
          "negative": 1817400
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.4732093369848373,
          "negative": 0.4730417121067
        },
        "validation": {
          "positive": 0.4587490316390737,
          "negative": 0.4587153294599113
        },
        "test": {
          "positive": 0.46467974584553084,
          "negative": 0.4646341749211297
        }
      }
    }
  },
  "MAX": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/MAX/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 9882,
          "negative": 988200
        },
        "validation": {
          "positive": 2094,
          "negative": 209400
        },
        "test": {
          "positive": 9289,
          "negative": 928900
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.6380146190160169,
          "negative": 0.5310544155251233
        },
        "validation": {
          "positive": 0.6304089837440939,
          "negative": 0.532734152633533
        },
        "test": {
          "positive": 0.6359989256037846,
          "negative": 0.5362769774845323
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/MAX/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 11373,
          "negative": 1137300
        },
        "validation": {
          "positive": 2994,
          "negative": 299400
        },
        "test": {
          "positive": 12534,
          "negative": 1253400
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5846014199950168,
          "negative": 0.5241706489666469
        },
        "validation": {
          "positive": 0.5695266501996241,
          "negative": 0.5158282123493942
        },
        "test": {
          "positive": 0.5688225037863788,
          "negative": 0.5187472347639669
        }
      }
    }
  },
  "MGA": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/MGA/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 8442,
          "negative": 844200
        },
        "validation": {
          "positive": 1995,
          "negative": 199500
        },
        "test": {
          "positive": 8313,
          "negative": 831300
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.6023265258897728,
          "negative": 0.5343370593638358
        },
        "validation": {
          "positive": 0.591014080050625,
          "negative": 0.5263069134630597
        },
        "test": {
          "positive": 0.5994412053117247,
          "negative": 0.5335921722091604
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/MGA/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 2097,
          "negative": 209700
        },
        "validation": {
          "positive": 609,
          "negative": 60900
        },
        "test": {
          "positive": 2556,
          "negative": 255600
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.4712041479530578,
          "negative": 0.47119761342338445
        },
        "validation": {
          "positive": 0.462085891439028,
          "negative": 0.46215390406363027
        },
        "test": {
          "positive": 0.4665147950580312,
          "negative": 0.46651404551339043
        }
      }
    }
  },
  "MYF6": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/MYF6/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 31904,
          "negative": 3190400
        },
        "validation": {
          "positive": 7791,
          "negative": 779100
        },
        "test": {
          "positive": 33204,
          "negative": 3320400
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5395560643725195,
          "negative": 0.45377264892684693
        },
        "validation": {
          "positive": 0.5203111521045451,
          "negative": 0.4518379116204872
        },
        "test": {
          "positive": 0.5296587310258821,
          "negative": 0.45770218114056627
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/MYF6/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 6784,
          "negative": 678400
        },
        "validation": {
          "positive": 1894,
          "negative": 189400
        },
        "test": {
          "positive": 7995,
          "negative": 799500
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.48778265593964004,
          "negative": 0.4877795222685388
        },
        "validation": {
          "positive": 0.46944363561096947,
          "negative": 0.4694394257789066
        },
        "test": {
          "positive": 0.47464756835148214,
          "negative": 0.4746456028373215
        }
      }
    }
  },
  "NFKB1": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/NFKB1/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 15326,
          "negative": 1532600
        },
        "validation": {
          "positive": 4067,
          "negative": 406700
        },
        "test": {
          "positive": 16856,
          "negative": 1685600
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5459926739482078,
          "negative": 0.4965053761809238
        },
        "validation": {
          "positive": 0.5207982244252621,
          "negative": 0.4841426537392365
        },
        "test": {
          "positive": 0.5309141534230937,
          "negative": 0.490509600966246
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/NFKB1/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 7607,
          "negative": 760700
        },
        "validation": {
          "positive": 2200,
          "negative": 220000
        },
        "test": {
          "positive": 8969,
          "negative": 896900
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5197721804580236,
          "negative": 0.5197310310882572
        },
        "validation": {
          "positive": 0.5068287526427061,
          "negative": 0.5067236484445786
        },
        "test": {
          "positive": 0.5064598660057956,
          "negative": 0.5064417267450194
        }
      }
    }
  },
  "SRY": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/SRY/hum301_20261005_185818_tdUK1h/dataset.npz",
      "counts": {
        "train": {
          "positive": 3319,
          "negative": 331900
        },
        "validation": {
          "positive": 999,
          "negative": 99900
        },
        "test": {
          "positive": 3678,
          "negative": 367800
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.4581284239839282,
          "negative": 0.45812299866168704
        },
        "validation": {
          "positive": 0.43432136455392273,
          "negative": 0.43432319362551924
        },
        "test": {
          "positive": 0.44087227819539365,
          "negative": 0.44083662578427174
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/SRY/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 1547,
          "negative": 154700
        },
        "validation": {
          "positive": 419,
          "negative": 41900
        },
        "test": {
          "positive": 1852,
          "negative": 185200
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.4498439686392153,
          "negative": 0.44984075920171285
        },
        "validation": {
          "positive": 0.45439624481640356,
          "negative": 0.45439228030669443
        },
        "test": {
          "positive": 0.4506755738610679,
          "negative": 0.4506755738610679
        }
      }
    }
  },
  "TERF1": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/TERF1/hum301_20261005_185818_tdUK1h/dataset.npz",
      "counts": {
        "train": {
          "positive": 709,
          "negative": 70900
        },
        "validation": {
          "positive": 186,
          "negative": 18600
        },
        "test": {
          "positive": 664,
          "negative": 66400
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.6359619322521544,
          "negative": 0.6281450173141714
        },
        "validation": {
          "positive": 0.6246918872575287,
          "negative": 0.6148751473582681
        },
        "test": {
          "positive": 0.6251501020694071,
          "negative": 0.6185894408197574
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/TERF1/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 583,
          "negative": 58300
        },
        "validation": {
          "positive": 155,
          "negative": 15500
        },
        "test": {
          "positive": 695,
          "negative": 69500
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5183182888225715,
          "negative": 0.5182918573309095
        },
        "validation": {
          "positive": 0.504881706140821,
          "negative": 0.5049203729503805
        },
        "test": {
          "positive": 0.5170189945929567,
          "negative": 0.5170202920719903
        }
      }
    }
  },
  "ZBED9": {
    "CHS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/ZBED9/hum301_multitf_chr5_chr7/dataset.npz",
      "counts": {
        "train": {
          "positive": 24298,
          "negative": 2429800
        },
        "validation": {
          "positive": 5723,
          "negative": 572300
        },
        "test": {
          "positive": 24442,
          "negative": 2444200
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.5230800715033787,
          "negative": 0.4867510156968473
        },
        "validation": {
          "positive": 0.5080966642149792,
          "negative": 0.47617955873107465
        },
        "test": {
          "positive": 0.5155414363544478,
          "negative": 0.4795007830592785
        }
      }
    },
    "GHTS": {
      "path": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/human/ZBED9/GHTS/dataset.npz",
      "counts": {
        "train": {
          "positive": 6979,
          "negative": 697900
        },
        "validation": {
          "positive": 1795,
          "negative": 179500
        },
        "test": {
          "positive": 7744,
          "negative": 774400
        }
      },
      "mean_gc": {
        "train": {
          "positive": 0.49464185402637034,
          "negative": 0.4946391428676156
        },
        "validation": {
          "positive": 0.47941032213883156,
          "negative": 0.4794069351002692
        },
        "test": {
          "positive": 0.48543422750610915,
          "negative": 0.48543309062765994
        }
      }
    }
  }
}''')
SPLITS = ('train', 'validation', 'test')
CLASSES = ('negative', 'positive')
EDGES = np.linspace(0.0, 1.0, 51)
FIELDS = ('TF', 'assay', 'split', 'class', 'bin_left_gc_percent',
          'bin_right_gc_percent', 'count', 'total_sequences',
          'gc_defined_sequences', 'undefined_gc_sequences',
          'sequences_with_N', 'mean_gc_percent', 'source_cache')


def read_array(archive, name):
    with archive.open(name + '.npy') as handle:
        return np.lib.format.read_array(handle, allow_pickle=False)


def collect_histograms(path, chunk_size=65536):
    """Stream the large DNA array; keep only labels/splits and 50-bin counts."""
    groups = {(s, c): {'hist': np.zeros(50, dtype=np.int64), 'total': 0,
                     'defined': 0, 'with_n': 0, 'sum_gc': 0.0}
              for s in SPLITS for c in CLASSES}
    with zipfile.ZipFile(path) as archive:
        labels, splits = read_array(archive, 'labels'), read_array(archive, 'split')
        if labels.ndim != 1 or splits.shape != labels.shape:
            raise ValueError(f'{path}: invalid labels/split shape')
        if not np.isin(labels, [0, 1]).all() or not np.isin(splits, [0, 1, 2]).all():
            raise ValueError(f'{path}: invalid class/split codes')
        n = len(labels)
        with archive.open('sequences.npy') as handle:
            version = np.lib.format.read_magic(handle)
            readers = {(1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0}
            if version not in readers:
                raise ValueError(f'{path}: unsupported NPY version {version}')
            shape, fortran, dtype = readers[version](handle)
            if shape != (n, 301) or fortran or dtype != np.dtype('uint8'):
                raise ValueError(f'{path}: expected C-order uint8 DNA, shape ({n}, 301)')
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                raw = handle.read((stop - start) * 301)
                if len(raw) != (stop - start) * 301:
                    raise ValueError(f'{path}: truncated sequences array')
                dna = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 301)
                if np.any(dna > 4):
                    raise ValueError(f'{path}: unexpected DNA code (expected A/C/G/T/N = 0/1/2/3/4)')
                valid_bases = (dna != 4).sum(axis=1)
                cg_bases = ((dna == 1) | (dna == 2)).sum(axis=1)
                gc = np.divide(cg_bases, valid_bases, out=np.zeros(len(dna)),
                               where=valid_bases > 0)
                for code, split in enumerate(SPLITS):
                    for label, class_name in enumerate(CLASSES):
                        mask = (splits[start:stop] == code) & (labels[start:stop] == label)
                        defined = mask & (valid_bases > 0)
                        group = groups[split, class_name]
                        group['total'] += int(mask.sum())
                        group['defined'] += int(defined.sum())
                        group['with_n'] += int((mask & (valid_bases < 301)).sum())
                        group['sum_gc'] += float(gc[defined].sum(dtype=np.float64))
                        group['hist'] += np.histogram(gc[defined], bins=EDGES)[0]
            if handle.read(1):
                raise ValueError(f'{path}: unexpected trailing sequence bytes')
    for group in groups.values():
        if int(group['hist'].sum()) != group['defined']:
            raise ValueError(f'{path}: histogram count mismatch')
    return groups


def validate_report(groups, spec, path):
    for split in SPLITS:
        for class_name in CLASSES:
            group = groups[split, class_name]
            expected = spec['counts'][split][class_name]
            if group['total'] != expected:
                raise ValueError(f'{path}: {split}/{class_name}: {group["total"]} sequences, expected {expected} from the reports')
            if not group['defined']:
                raise ValueError(f'{path}: {split}/{class_name}: no defined GC values')
            mean = group['sum_gc'] / group['defined']
            expected_mean = spec['mean_gc'][split][class_name]
            if not math.isclose(mean, expected_mean, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f'{path}: {split}/{class_name}: GC mean {mean:.10f} differs from reported {expected_mean:.10f}')


def rows_for(tf, assay, groups, path):
    for (split, class_name), group in groups.items():
        mean = 100 * group['sum_gc'] / group['defined'] if group['defined'] else ''
        for i, count in enumerate(group['hist']):
            yield dict(TF=tf, assay=assay, split=split, **{'class': class_name},
                       bin_left_gc_percent=float(100 * EDGES[i]),
                       bin_right_gc_percent=float(100 * EDGES[i + 1]),
                       count=int(count), total_sequences=group['total'],
                       gc_defined_sequences=group['defined'],
                       undefined_gc_sequences=group['total'] - group['defined'],
                       sequences_with_N=group['with_n'], mean_gc_percent=mean,
                       source_cache=str(path))


def write_rows(path, rows):
    with gzip.open(path, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as handle:
        return list(csv.DictReader(handle))


def plot_histograms(rows, output_dir, tfs, requested_splits=('train', 'test')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    grouped = {}
    for row in rows:
        key = row['TF'], row['assay'], row['split'], row['class']
        grouped.setdefault(key, []).append(row)
    for key, group in grouped.items():
        group.sort(key=lambda row: float(row['bin_left_gc_percent']))
        if len(group) != 50:
            raise ValueError(f'{key}: expected 50 common GC bins')
        if not np.allclose([float(r['bin_left_gc_percent']) for r in group], 100 * EDGES[:-1]):
            raise ValueError(f'{key}: inconsistent GC bin edges')
        total = int(group[0]['gc_defined_sequences'])
        if total <= 0 or sum(int(r['count']) for r in group) != total:
            raise ValueError(f'{key}: invalid histogram totals')
    panels = [
        ('CHS: positive vs negative', [('CHS', 'positive', 'Positive'), ('CHS', 'negative', 'Negative')]),
        ('GHTS: positive vs negative', [('GHTS', 'positive', 'Positive'), ('GHTS', 'negative', 'Negative')]),
        ('Positive sequences: CHS vs GHTS', [('CHS', 'positive', 'CHS'), ('GHTS', 'positive', 'GHTS')]),
        ('Negative sequences: CHS vs GHTS', [('CHS', 'negative', 'CHS'), ('GHTS', 'negative', 'GHTS')]),
    ]
    for tf in tfs:
        for split in requested_splits:
            keys = [(tf, a, split, c) for a in ('CHS', 'GHTS') for c in CLASSES]
            if any(key not in grouped for key in keys):
                raise ValueError(f'{tf}/{split}: both classes and assays are required')
            peak = max(100 * max(int(r['count']) for r in grouped[key]) /
                       int(grouped[key][0]['gc_defined_sequences']) for key in keys)
            ylim = max(5, 5 * math.ceil(peak * 1.16 / 5))
            fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True, sharey=True)
            fig.subplots_adjust(left=.08, right=.98, bottom=.12, top=.84, hspace=.67, wspace=.22)
            for ax, (title, comparisons) in zip(axes.flat, panels):
                for color, (assay, class_name, label) in zip(('#256D85', '#C96B18'), comparisons):
                    group = grouped[tf, assay, split, class_name]
                    total = int(group[0]['gc_defined_sequences'])
                    values = np.array([int(r['count']) for r in group], dtype=float) / total * 100
                    ax.stairs(values, EDGES * 100, color=color, linewidth=2.2,
                              label=f'{label} (n={total:,})')
                ax.set_title(title, fontsize=16, pad=32)
                ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2,
                          fontsize=12, frameon=False, borderaxespad=0)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, ylim)
                ax.set_xticks([0, 20, 40, 60, 80, 100])
                ax.tick_params(labelsize=12)
                ax.grid(axis='y', alpha=.18)
                ax.spines[['top', 'right']].set_visible(False)
            for ax in axes[:, 0]:
                ax.set_ylabel('Sequences in bin (%)', fontsize=14)
            for ax in axes[1, :]:
                ax.set_xlabel('GC content (%)', fontsize=14)
            fig.suptitle(f'{tf}: {split} sequences, 301 bp', fontsize=19, y=.985)
            fig.text(.5, .018, 'Common 2-percentage-point bins. Each class normalized separately. GC denominator excludes N bases.',
                     ha='center', fontsize=11)
            for suffix in ('png', 'pdf'):
                fig.savefig(output_dir / f'{tf}_gc_{split}.{suffix}', dpi=240, facecolor='white')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT,
                        help='Relocate the recorded cache paths beneath this Human301 data root.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'gc_distributions')
    parser.add_argument('--histograms', type=Path, help='Replot an existing gc_histograms.csv.gz without reading caches.')
    parser.add_argument('--tfs', nargs='+', choices=list(CACHE_SPECS), default=list(CACHE_SPECS))
    args = parser.parse_args()
    if args.histograms:
        rows = load_rows(args.histograms)
    else:
        jobs = [(tf, assay, spec, args.data_root / Path(spec['path']).relative_to(DATA_ROOT))
                for tf in args.tfs for assay, spec in CACHE_SPECS[tf].items()]
        missing = [str(path) for _, _, _, path in jobs if not path.is_file()]
        if missing:
            raise FileNotFoundError('Reported caches not found:\n' + '\n'.join(missing))
        rows = []
        for tf, assay, spec, path in jobs:
            print(f'{tf}/{assay}: reading {path}', flush=True)
            groups = collect_histograms(path)
            validate_report(groups, spec, path)
            rows.extend(rows_for(tf, assay, groups, path))
            print(f'{tf}/{assay}: counts and GC means match the reports', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = args.output_dir / 'gc_histograms.csv.gz'
    if not args.histograms or args.histograms.resolve() != histogram_path.resolve():
        write_rows(histogram_path, rows)
    plot_histograms(rows, args.output_dir, args.tfs)
    print(f'Export: {histogram_path}', flush=True)
    print(f'Figures: {args.output_dir}', flush=True)
    print('GC matching alone does not establish overall dataset quality.', flush=True)


if __name__ == '__main__':
    main()
