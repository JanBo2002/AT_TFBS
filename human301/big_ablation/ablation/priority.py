"""Submission order only; no runtime estimates or training changes."""
from __future__ import annotations

import csv
from pathlib import Path


def read_priority(path: Path) -> dict[tuple[str, str], int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["priority", "tf", "assay"]:
            raise ValueError("Priority header must be exactly: priority,tf,assay")
        ranks = {}
        for expected, row in enumerate(reader, 1):
            assay = {"CHS": "CHIP"}.get(row["assay"], row["assay"])
            key = (row["tf"], assay)
            if int(row["priority"]) != expected or not row["tf"]:
                raise ValueError("Priority numbers must be consecutive, starting at 1")
            if assay not in {"CHIP", "GHTS"} or key in ranks:
                raise ValueError(f"Invalid or duplicate priority entry: {key}")
            ranks[key] = expected
    if not ranks:
        raise ValueError("Priority list is empty")
    return ranks


def order_tasks(rows: list[dict], path: Path) -> list[dict]:
    ranks = read_priority(path)
    # Stable sort: repetitions retain seed order. Unknown pairs follow the list
    # in manifest order. Custom multi-TF planners keep their own task order.
    return sorted(rows, key=lambda row: ranks.get((row["tf"], row["train_assay"]), len(ranks) + 1))
