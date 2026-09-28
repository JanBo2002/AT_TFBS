#!/usr/bin/env python3
"""Convert dominant CAGE TSS positions from the Le et al. workbook to two IGV BED tracks.

The input workbook stores dominant_ctss as a 1-based genomic coordinate.
BED uses 0-based, half-open coordinates, so a 1-bp TSS feature is written as:
    BED start = dominant_ctss - 1
    BED end   = dominant_ctss

By default, the script reads the wild-type sheet ("wt") and creates separate
plus- and minus-strand BED files.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


def chromosome_sort_key(chrom: str) -> tuple[int, int | str]:
    """Natural TAIR10 chromosome order: Chr1..Chr5, then ChrC, ChrM, then others."""
    match = re.fullmatch(r"Chr(\d+)", chrom, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))

    special = {"chrc": 0, "chrm": 1}
    lowered = chrom.lower()
    if lowered in special:
        return (1, special[lowered])

    return (2, lowered)


def clean_number(value: Any) -> float | None:
    """Return a finite float or None."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert dominant CAGE TSS positions from an XLSX sheet into two "
            "1-bp BED tracks for IGV, split by strand."
        )
    )
    parser.add_argument("xlsx", type=Path, help="Input XLSX file")
    parser.add_argument(
        "--sheet",
        default="wt",
        help='Worksheet containing dominant_ctss data (default: "wt")',
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix. Default: <input_directory>/<input_stem>_<sheet>. "
            "The script adds _plus.bed and _minus.bed."
        ),
    )
    parser.add_argument(
        "--no-track-line",
        action="store_true",
        help="Do not write an IGV/UCSC track header line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.xlsx

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    prefix = args.prefix
    if prefix is None:
        prefix = input_path.with_name(f"{input_path.stem}_{args.sheet}")

    plus_path = Path(f"{prefix}_plus.bed")
    minus_path = Path(f"{prefix}_minus.bed")
    plus_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = load_workbook(input_path, read_only=True, data_only=True)
    if args.sheet not in workbook.sheetnames:
        available = ", ".join(workbook.sheetnames)
        raise ValueError(f'Worksheet "{args.sheet}" not found. Available: {available}')

    worksheet = workbook[args.sheet]
    rows = worksheet.iter_rows(values_only=True)

    try:
        header_row = next(rows)
    except StopIteration as exc:
        raise ValueError(f'Worksheet "{args.sheet}" is empty') from exc

    headers = {str(value).strip(): index for index, value in enumerate(header_row) if value is not None}
    required = {"consensus.cluster", "chr", "strand", "dominant_ctss"}
    missing = sorted(required - headers.keys())
    if missing:
        raise ValueError(
            f'Worksheet "{args.sheet}" lacks required columns: {", ".join(missing)}. '
            "Use a sample-specific sheet such as wt, ddm1, ibm1, etc.; "
            'the sheet "All" has no dominant_ctss column.'
        )

    tpm_column = headers.get("tpm.dominant_ctss")
    records: dict[str, list[tuple[str, int, int, str, int, str, float | None]]] = {
        "+": [],
        "-": [],
    }

    skipped = 0
    for row_number, row in enumerate(rows, start=2):
        chrom_value = row[headers["chr"]]
        strand_value = row[headers["strand"]]
        position_value = row[headers["dominant_ctss"]]
        cluster_value = row[headers["consensus.cluster"]]

        if chrom_value is None or strand_value not in {"+", "-"} or position_value is None:
            skipped += 1
            continue

        try:
            position_1based = int(position_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid dominant_ctss at worksheet row {row_number}: {position_value!r}"
            ) from exc

        if position_1based < 1:
            raise ValueError(
                f"dominant_ctss must be >= 1 at worksheet row {row_number}: {position_1based}"
            )

        chrom = str(chrom_value).strip()
        strand = str(strand_value)
        bed_start = position_1based - 1
        bed_end = position_1based
        name = f"CAGE_TSS_{cluster_value}"

        # BED score must be an integer from 0 to 1000. We avoid implying a
        # cross-file expression normalization and therefore use score 0.
        score = 0
        dominant_tpm = clean_number(row[tpm_column]) if tpm_column is not None else None

        records[strand].append(
            (chrom, bed_start, bed_end, name, score, strand, dominant_tpm)
        )

    workbook.close()

    for strand in ("+", "-"):
        records[strand].sort(key=lambda record: (chromosome_sort_key(record[0]), record[1], record[2], record[3]))

    track_lines = {
        "+": (
            f'track name="CAGE_{args.sheet}_TSS_plus" '
            f'description="Le CAGE dominant TSS, {args.sheet}, plus strand, TAIR10" '
            'visibility=2 color=0,90,180'
        ),
        "-": (
            f'track name="CAGE_{args.sheet}_TSS_minus" '
            f'description="Le CAGE dominant TSS, {args.sheet}, minus strand, TAIR10" '
            'visibility=2 color=180,50,50'
        ),
    }

    for strand, output_path in (("+", plus_path), ("-", minus_path)):
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            if not args.no_track_line:
                handle.write(track_lines[strand] + "\n")

            for chrom, start, end, name, score, feature_strand, dominant_tpm in records[strand]:
                # BED6 plus one optional annotation column. IGV displays the first
                # six columns normally; the seventh retains the measured dominant
                # CTSS TPM for inspection without changing feature width.
                tpm_text = "." if dominant_tpm is None else f"{dominant_tpm:.6g}"
                handle.write(
                    f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{feature_strand}\t{tpm_text}\n"
                )

    print(f"Worksheet: {args.sheet}")
    print(f"Plus-strand TSS:  {len(records['+']):,} -> {plus_path}")
    print(f"Minus-strand TSS: {len(records['-']):,} -> {minus_path}")
    if skipped:
        print(f"Skipped incomplete rows: {skipped:,}")


if __name__ == "__main__":
    main()
