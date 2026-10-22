"""Human301 manifest and provider; future representations use another factory."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    path = path.resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["tf", "assay", "cache"]:
            raise ValueError("Manifest header must be exactly: tf,assay,cache")
        rows = []
        keys = set()
        for row in reader:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["tf"]):
                raise ValueError(f"Invalid TF name: {row['tf']}")
            assay = row["assay"].upper()
            assay = {"CHS": "CHIP", "CHIPSEQ": "CHIP", "GHT-SELEX": "GHTS"}.get(assay, assay)
            if assay not in {"CHIP", "GHTS"}:
                raise ValueError(f"Unknown assay {assay}")
            cache = Path(row["cache"]).expanduser()
            cache = cache.resolve() if cache.is_absolute() else (path.parent / cache).resolve()
            key = (row["tf"], assay)
            if key in keys:
                raise ValueError(f"Duplicate manifest entry: {key}")
            if not cache.is_file():
                raise FileNotFoundError(cache)
            keys.add(key)
            rows.append({"tf": row["tf"], "assay": assay, "cache": cache})
    if not rows:
        raise ValueError("Manifest has no caches")
    return rows


def discover(root: Path, destination: Path) -> None:
    """Infer TF/assay only from recognizable Human301 cache paths or sources.

    Ambiguity is an error, not an opportunity to guess a TF assignment.
    """
    import numpy as np
    from .hum301_data import load_cache
    candidates = []
    for path in sorted(root.resolve().rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as raw:
                if not {"split", "summary_json", "sequences"}.issubset(raw.files):
                    continue
                summary = json.loads(str(raw["summary_json"].item()))
        except (OSError, ValueError):
            continue
        texts = [str(path.relative_to(root.resolve()))]
        texts += [str(v.get("path", "")) for v in summary.get("sources", {}).values() if isinstance(v, dict)]
        assignments = set()
        for text in texts:
            # Known layouts: TF/CHIP/dataset.npz, CHIP/TF/dataset.npz,
            # TF_CHIP.npz, or BED names containing those two tokens.
            tokens = [t for t in re.split(r"[/_.\\-]+", text) if t]
            for i, token in enumerate(tokens):
                assay = {"CHIP": "CHIP", "CHS": "CHIP", "GHTS": "GHTS"}.get(token.upper())
                if assay:
                    for adjacent in (i-1, i+1):
                        if 0 <= adjacent < len(tokens):
                            tf = tokens[adjacent]
                            if re.fullmatch(r"[A-Z][A-Z0-9]{1,15}", tf) and tf not in {"CHIP","CHS","GHTS","DNA","TRAIN","TEST","TFBS"}:
                                assignments.add((tf, assay))
        if len(assignments) != 1:
            raise ValueError(f"Cannot unambiguously assign {path}. Enter it manually in a tf,assay,cache manifest.")
        tf, assay = assignments.pop()
        candidates.append({"tf": tf, "assay": assay, "cache": str(path.resolve())})
    if not candidates:
        raise ValueError(f"No Human301 caches found below {root}")
    keys = [(r["tf"],r["assay"]) for r in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("Multiple caches for the same TF/assay; choose one explicitly in the manifest")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tf","assay","cache"])
        writer.writeheader()
        writer.writerows(candidates)


def human301(context):
    # The current fitting module deliberately uses the unchanged original
    # Dataset/Sampler. A different data factory can return another provider.
    return dict(context.cache_paths)
