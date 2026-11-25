#!/usr/bin/env python3
"""N1-Vorabprüfung der echten Quelldaten (nur lesen).

Prüft für alle TF/Assay-Zeilen der Middle-Liste, was prepare-negatives prüft, bricht aber nie ab:
Dateien, doppelte Fenster, Chromosomen, Überlappungen, Shade-Abstände, Basen, Genom, Vorrat je Positivem.
Schreibt nichts in big_ablation; Ergebnis: Bildschirm und n1_auswertung/logs/diagnose.json.

Aufruf aus .../human301:  python n1_diagnose.py            (optional: --genom /pfad/hg38.fa)
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HUMAN301 = Path.cwd()
BA = HUMAN301 / "big_ablation"
if not (BA / "data.py").is_file():
    sys.exit("Bitte aus dem Ordner human301 starten (dort liegt big_ablation/).")
sys.path.insert(0, str(BA))
import numpy as np  # noqa: E402
import data as D  # noqa: E402

ROOTS = [os.environ.get("N1_HANDOFF", "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/clara/chip/"
                        "student_handoff_30tf_increment/Input_data"),
         os.environ.get("N1_ARCHTHALIA", "/data/projects/SFB_A03/clara/ArchThalia-ML")]
SUPPLEMENTS = [os.environ.get("N1_ERGAENZUNG", "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/archipelago_ergaenzung")]
MIX = {"random": 5, "aliens": 5, "shades": 1}
POSITIVE_PROBES = 300
FULL_FETCH_LIMIT = 3_000_000


def bases(sequences):
    """(all-N rows, rows with other characters than ACGTN/acgtn, rows with lowercase)."""
    if not sequences:
        return 0, 0, 0
    raw = np.frombuffer("".join(sequences).encode("ascii", "replace"), dtype=np.uint8).reshape(len(sequences), -1)
    codes = D.ENCODE[raw]
    return (int(((codes == 4).all(axis=1)).sum()), int((codes == 255).any(axis=1).sum()),
            int(((raw >= 97) & (raw <= 122)).any(axis=1).sum()))


def tolerant(sequences):
    raw = np.frombuffer("".join(sequences).encode("ascii", "replace"), dtype=np.uint8)
    codes = D.ENCODE[raw].reshape(len(sequences), D.STORED_LENGTH)
    codes[codes == 255] = 4
    return codes


def diagnose(row, genome):
    tf, assay = row["tf"], row["assay"]
    result = {"tf": tf, "assay": assay, "cache": str(row["cache"]), "files": {}, "befunde": [], "hinweise": []}
    source = Path(row["cache"]).resolve()
    summary = D.cache_summary(source)
    parent = D._cache_arrays(source, ("labels", "split", "chromosomes", "starts", "ends"))
    chromosomes = D._Chromosomes()
    known = set(parent["chromosomes"].tolist())
    side = {"Train": set(parent["chromosomes"][parent["split"] != 2].tolist()),
            "Test": set(parent["chromosomes"][parent["split"] == 2].tolist())}
    if side["Train"] & side["Test"]:
        result["befunde"].append(f"Original-Cache: Train und Test teilen Chromosomen {sorted(side['Train'] & side['Test'])}")
    val = sorted(set(parent["chromosomes"][parent["split"] == 1].tolist()))
    result["validierungs_chromosomen"] = val
    try:
        plans = D.plan_typed_negatives(tf, assay, parent, summary, ROOTS, SUPPLEMENTS, chromosomes)
    except ValueError as error:
        result["befunde"].append(f"Dateisuche: {error}")
        return result
    try:
        genome_path, reader = genome.choose(summary)
        result["genom"] = str(genome_path)
    except ValueError as error:
        genome_path, reader = None, None
        result["genom"] = None
        result["genom_fehler"] = str(error)
    canonical = {D.canonical_chrom(name): name for name in known}
    counts = {code: {"positive": int(np.sum((parent["split"] == code) & (parent["labels"] == 1))),
                     "random": int(np.sum((parent["split"] == code) & (parent["labels"] == 0)))} for code in range(3)}
    file_chroms = {"Train": set(), "Test": set()}
    for split_name, codes in D.NEGATIVE_SOURCE_SPLITS:
        member = np.isin(parent["split"], codes)
        positive = member & (parent["labels"] == 1)
        pos_keys = chromosomes.keys(parent["chromosomes"][positive], parent["starts"][positive])
        rnd_keys = chromosomes.keys(parent["chromosomes"][member & (parent["labels"] == 0)],
                                    parent["starts"][member & (parent["labels"] == 0)])
        other_side = side["Test" if split_name == "Train" else "Train"]
        for kind in D.NEGATIVE_VARIANTS["ras"]:
            info = plans[split_name]["files"][kind]
            name = f"{split_name} {kind}"
            entry = {"path": info["path"], "format": info["format"], "root": info["root"]}
            result["files"][name] = entry
            try:
                chroms, starts, sequences = D.read_typed_windows(info["path"], info["format"])
            except ValueError as error:
                result["befunde"].append(f"{name}: Datei nicht lesbar: {error}")
                continue
            n = int(chroms.size)
            names = set(chroms.tolist())
            file_chroms[split_name] |= names
            keys = chromosomes.keys(chroms, starts)
            unique, first = np.unique(keys, return_index=True)
            first_mask = np.zeros(n, dtype=bool)
            first_mask[first] = True
            renamed = sorted(x for x in names - known if D.canonical_chrom(x) in canonical)
            other = np.isin(chroms, sorted(other_side)) if other_side else np.zeros(n, dtype=bool)
            missing = ~np.isin(chroms, sorted(known))
            overlap = D._near_mask(keys, pos_keys, D.STORED_LENGTH)
            same_random = np.isin(keys, rnd_keys)
            entry.update(zeilen=n, doppelt=n - int(unique.size), umbenannt=renamed[:5],
                         andere_aufteilung=int((other & first_mask).sum()),
                         chromosomen_der_anderen_aufteilung=sorted(set(chroms[other].tolist()))[:10],
                         nicht_im_cache=int((missing & first_mask).sum()),
                         chromosomen_nicht_im_cache=sorted(set(chroms[missing].tolist()))[:10],
                         ueberlappt_positive=int((overlap & first_mask).sum()),
                         auch_random=int((same_random & first_mask).sum()))
            if n - unique.size:
                example = int(first[np.argmax(np.unique(keys, return_counts=True)[1] > 1)])
                entry["doppelt_beispiel"] = f"{chroms[example]}:{starts[example]}"
            if kind == "shades":
                near = D._near_mask(keys[first_mask], pos_keys, D.SHADE_DISTANCE_LIMIT)
                entry["nah_an_positiven"] = float(near.mean()) if near.size else 0.0
                ref = np.sort(pos_keys)
                if ref.size:
                    k = keys[first_mask]
                    idx = np.searchsorted(ref, k)
                    left = np.abs(k - ref[np.clip(idx - 1, 0, ref.size - 1)])
                    right = np.abs(ref[np.clip(idx, 0, ref.size - 1)] - k)
                    dist = np.minimum(np.where(idx > 0, left, np.iinfo(np.int64).max),
                                      np.where(idx < ref.size, right, np.iinfo(np.int64).max))
                    dist = dist[dist < (1 << 39)]
                    if dist.size:
                        entry["abstand_bp"] = {q: int(np.percentile(dist, p)) for q, p in
                                               (("min", 0), ("median", 50), ("p95", 95), ("max", 100))}
            # bases: FASTA rows from the file, BED rows from the genome (when known)
            if sequences is None and reader is not None:
                picked = np.flatnonzero(first_mask & ~missing)
                if picked.size > FULL_FETCH_LIMIT:
                    picked = picked[np.linspace(0, picked.size - 1, FULL_FETCH_LIMIT).astype(int)]
                try:
                    fetched = [reader.fetch(str(chroms[i]), int(starts[i]), int(starts[i]) + D.STORED_LENGTH)
                               for i in picked]
                    entry["basen_aus"] = f"Genom ({len(fetched):,} Fenster)"
                    all_n, invalid, lower = bases(fetched)
                except ValueError as error:
                    result["befunde"].append(f"{name}: Fenster nicht im Genom: {error}")
                    all_n = invalid = lower = None
            elif sequences is not None:
                entry["basen_aus"] = "FASTA"
                all_n, invalid, lower = bases([s for s, k in zip(sequences, first_mask) if k])
                if reader is not None:
                    probe = np.flatnonzero(first_mask & ~missing)
                    probe = probe[np.unique(np.linspace(0, probe.size - 1, min(probe.size, 2000)).astype(int))] \
                        if probe.size else probe
                    try:
                        a = tolerant([sequences[i] for i in probe])
                        b = tolerant([reader.fetch(str(chroms[i]), int(starts[i]), int(starts[i]) + D.STORED_LENGTH)
                                      for i in probe])
                        both = (a != 4) & (b != 4)
                        mismatch = ((a != b) & both).sum(axis=1) / np.maximum(both.sum(axis=1), 1)
                        entry["fasta_wie_genom"] = float(np.mean(mismatch < 0.01)) if probe.size else 1.0
                    except ValueError as error:
                        result["befunde"].append(f"{name}: FASTA-Fenster nicht im Genom: {error}")
            else:
                all_n = invalid = lower = None
            entry.update(nur_N=all_n, andere_zeichen=invalid, kleinbuchstaben=lower)
            keep = first_mask & ~other & ~missing & ~overlap
            split = np.full(n, 2, dtype=np.uint8) if split_name == "Test" else np.isin(chroms, val).astype(np.uint8)
            for code in codes:
                counts[code][kind] = counts[code].get(kind, 0) + int(np.sum(keep & (split == code)))
            # what the installed check (N1) would stop at
            if renamed:
                result["befunde"].append(f"{name}: andere Chromosomen-Namen ({renamed[0]})")
            if entry["doppelt"]:
                result["befunde"].append(f"{name}: {entry['doppelt']} doppelte Fenster (z. B. {entry['doppelt_beispiel']})")
            if entry["andere_aufteilung"]:
                result["befunde"].append(f"{name}: {entry['andere_aufteilung']} Fenster auf Chromosomen der anderen "
                                         f"Aufteilung {entry['chromosomen_der_anderen_aufteilung']}")
            if entry["ueberlappt_positive"]:
                result["befunde"].append(f"{name}: {entry['ueberlappt_positive']} Fenster überlappen Positive")
            if kind == "shades" and entry.get("nah_an_positiven", 1.0) < D.SHADE_NEAR_FRACTION:
                result["befunde"].append(f"{name}: nur {entry['nah_an_positiven']:.1%} innerhalb "
                                         f"{D.SHADE_DISTANCE_LIMIT} bp eines Positiven")
            if all_n:
                result["befunde"].append(f"{name}: {all_n} Fenster nur aus N")
            if invalid:
                result["befunde"].append(f"{name}: {invalid} Fenster mit anderen Zeichen als ACGTN")
            if entry.get("fasta_wie_genom", 1.0) < 0.95:
                result["befunde"].append(f"{name}: nur {entry['fasta_wie_genom']:.1%} der FASTA-Fenster passen zum Genom")
            if entry["nicht_im_cache"]:
                result["hinweise"].append(f"{name}: {entry['nicht_im_cache']} Fenster auf Chromosomen ohne Original-"
                                          f"zeilen {entry['chromosomen_nicht_im_cache']}")
            if entry["auch_random"]:
                result["hinweise"].append(f"{name}: {entry['auch_random']} Fenster sind auch random")
    shared = (side["Train"] | file_chroms["Train"]) & (side["Test"] | file_chroms["Test"])
    if shared and not (side["Train"] & side["Test"]):
        result["befunde"].append(f"Train- und Test-Dateien teilen Chromosomen {sorted(shared)}")
    if reader is not None:
        member = np.flatnonzero(parent["labels"] == 1)
        probe = member[np.unique(np.linspace(0, member.size - 1, min(member.size, POSITIVE_PROBES)).astype(int))]
        cached = D._cache_arrays(source, ("sequences",))["sequences"][probe]
        try:
            fetched = tolerant([reader.fetch(str(parent["chromosomes"][i]), int(parent["starts"][i]),
                                             int(parent["ends"][i])) for i in probe])
            exact = int(np.sum((fetched == cached).all(axis=1)))
            both = (fetched != 4) & (cached != 4)
            close = int(np.sum(((fetched != cached) & both).sum(axis=1) / np.maximum(both.sum(axis=1), 1) < 0.01))
            result["positive_gegen_genom"] = {"geprueft": int(probe.size), "identisch": exact, "fast_gleich": close}
            if exact < probe.size:
                result["befunde"].append(f"Positive gegen Genom: {exact}/{probe.size} identisch, {close} fast gleich")
        except ValueError as error:
            result["befunde"].append(f"Positive nicht im Genom: {error}")
        del cached
    elif any(f.get("format") == "bed" for f in result["files"].values()):
        result["befunde"].append(f"Kein Genom für BED-Dateien: {result.get('genom_fehler')}")
    result["zaehlung"] = {D.SPLIT_NAMES[c]: v for c, v in counts.items()}
    train = counts[0]
    result["vorrat_je_positivem_train"] = {k: (train.get(k, 0) / train["positive"] if train["positive"] else 0.0)
                                           for k in MIX}
    for kind, need in MIX.items():
        if train.get(kind, 0) < need * train["positive"]:
            result["befunde"].append(f"Train-Vorrat {kind}: {train.get(kind, 0):,} nach Bereinigung, gebraucht "
                                     f"{need} x {train['positive']:,} = {need * train['positive']:,} je Epoche")
    test = counts[2]
    negatives = sum(v for k, v in test.items() if k != "positive")
    result["zufallsniveau_test"] = {"gesamt": test["positive"] / max(test["positive"] + negatives, 1),
                                    **{f"nur_{k}": test["positive"] / max(test["positive"] + test.get(k, 0), 1)
                                       for k in MIX}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--genom", help="Referenzgenom (FASTA mit .fai), falls nicht im Cache vermerkt")
    parser.add_argument("--tf", action="append", help="nur diese TFs (mehrfach möglich)")
    args = parser.parse_args()
    manifest = BA / "configs" / "caches.csv"
    rows = D._negative_rows(manifest, args.tf, None if args.tf else BA / "configs" / "training_priority_middle.csv")
    recorded = {}
    for row in D.read_manifest(manifest):
        entry = D.cache_summary(row["cache"]).get("sources", {}).get("genome_fasta")
        if isinstance(entry, dict) and entry.get("path"):
            recorded.setdefault(str(Path(entry["path"]).resolve()), entry.get("sha256"))
    genome = D._Genome(args.genom, recorded)
    print(f"N1-Vorabprüfung: {len(rows)} TF/Assay-Zeilen; Genome in den Caches: {sorted(recorded) or 'keine'}",
          flush=True)
    results, started = [], time.time()
    for number, row in enumerate(rows, 1):
        try:
            result = diagnose(row, genome)
        except Exception as error:  # noqa: BLE001 - a report must never stop
            result = {"tf": row["tf"], "assay": row["assay"], "befunde": [f"Abbruch der Prüfung: {error!r}"],
                      "hinweise": [], "traceback": traceback.format_exc()}
        results.append(result)
        ratio = result.get("vorrat_je_positivem_train", {})
        pos = result.get("zaehlung", {}).get("train", {}).get("positive")
        line = (f"[{number:2}/{len(rows)}] {row['tf']}/{row['assay']}: "
                + (f"Train {pos:,} Positive, je Positivem random {ratio['random']:.2f} | aliens {ratio['aliens']:.2f}"
                   f" | shades {ratio['shades']:.2f}" if ratio else "keine Zählung")
                + f" | {'ohne Befund' if not result['befunde'] else str(len(result['befunde'])) + ' Befund(e)'}")
        print(line, flush=True)
        for text in result["befunde"]:
            print(f"      ! {text}", flush=True)
        for text in result["hinweise"]:
            print(f"      - {text}", flush=True)
    genome.close()
    out = HUMAN301 / "n1_auswertung" / "logs" / "diagnose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"roots": ROOTS, "supplements": SUPPLEMENTS, "results": results}, indent=1,
                              default=str) + "\n")
    clean = sum(1 for r in results if not r["befunde"])
    print(f"\n{clean} von {len(results)} Zeilen ohne Befund; Dauer {time.time() - started:.0f} s; Details: {out}")


if __name__ == "__main__":
    main()
