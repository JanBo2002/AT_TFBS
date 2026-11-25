#!/usr/bin/env python3
"""Überblick: Für welche TFs gibt es neben random auch aliens- und shades-Negative (CHS und GHTS, Train und Test)?

Durchsucht <Wurzel>/<CHS|GHTS>/<Train|Test>/<TF>/ und zählt je Datei die Einträge:
BED = Zeilen ohne Kopf- und Kommentarzeilen, FASTA = Header-Zeilen (">"); .gz wird mitgelesen.
Standard-Wurzeln: die Handoff-Daten und für GLI4, ZFP3, ZNF696 und ZNF772 zusätzlich ArchThalia-ML.

Aufruf (am besten in human301, nicht in big_ablation):
    python negative_uebersicht.py                 Standard-Wurzeln
    python negative_uebersicht.py WURZEL [...]    eigene Wurzeln, alle TFs darin
Ausgabe: Übersicht auf dem Bildschirm und negative_uebersicht.csv (eine Zeile je Quelle, Assay, Split und TF).
"""
from __future__ import annotations

import csv
import gzip
import statistics
import sys
from pathlib import Path

HANDOFF = Path("/data/projects/SFB_A03/jan/AT_TFBS/raw_data/clara/chip/student_handoff_30tf_increment/Input_data")
ARCHTHALIA = Path("/data/projects/SFB_A03/clara/ArchThalia-ML")
SONDER_TFS = {"GLI4", "ZFP3", "ZNF696", "ZNF772"}
TYPEN = ("positives", "random", "aliens", "shades")
KUERZEL = {"positives": "p", "random": "r", "aliens": "a", "shades": "s"}
ASSAYS = ("CHS", "GHTS")
SPLITS = ("Train", "Test")
FASTA_ENDUNGEN = (".fa", ".fasta", ".fna")


def grundname(path: Path) -> str:
    name = path.name.lower()
    return name[:-3] if name.endswith(".gz") else name


def zaehlen(path: Path) -> int:
    oeffnen = gzip.open if path.name.lower().endswith(".gz") else open
    with oeffnen(path, "rt", errors="replace") as handle:
        if grundname(path).endswith(FASTA_ENDUNGEN):
            return sum(1 for zeile in handle if zeile.startswith(">"))
        return sum(1 for zeile in handle
                   if zeile.strip() and not zeile.startswith(("#", "track", "browser")))


def typ(path: Path) -> str | None:
    stamm = grundname(path).split(".")[0]
    return stamm if stamm in TYPEN else None


def bevorzugt(dateien: list[Path], assay: str) -> Path:
    """Gibt es einen Typ als BED und als FASTA, zählt bei CHS die BED-, bei GHTS die FASTA-Datei."""
    fasta = [f for f in dateien if grundname(f).endswith(FASTA_ENDUNGEN)]
    andere = [f for f in dateien if f not in fasta]
    reihenfolge = (fasta + andere) if assay == "GHTS" else (andere + fasta)
    return reihenfolge[0]


def tf_ordner(wurzel: Path, assay: str, split: str, nur: set[str] | None):
    ordner = wurzel / assay / split
    if not ordner.is_dir():
        return []
    return [p for p in sorted(ordner.iterdir()) if p.is_dir() and (nur is None or p.name in nur)]


def erfassen(wurzel: Path, quelle: str, nur: set[str] | None) -> list[dict]:
    zeilen = []
    for assay in ASSAYS:
        for split in SPLITS:
            for ordner in tf_ordner(wurzel, assay, split, nur):
                nach_typ, weitere = {}, []
                for datei in sorted(p for p in ordner.iterdir() if p.is_file()):
                    t = typ(datei)
                    if t is None:
                        weitere.append(datei.name)
                    else:
                        nach_typ.setdefault(t, []).append(datei)
                zeile = {"quelle": quelle, "assay": assay, "split": split, "tf": ordner.name}
                for t in TYPEN:
                    dateien = nach_typ.get(t, [])
                    zeile[f"n_{t}"] = zaehlen(bevorzugt(dateien, assay)) if dateien else ""
                n_pos = zeile["n_positives"]
                for t in TYPEN[1:]:
                    n = zeile[f"n_{t}"]
                    zeile[f"{t}_je_positiv"] = round(n / n_pos, 1) if n != "" and n_pos else ""
                zeile["dateien"] = " ".join(f.name for t in TYPEN for f in nach_typ.get(t, []))
                zeile["weitere_dateien"] = " ".join(weitere)
                zeilen.append(zeile)
    return zeilen


def main(argumente: list[str]) -> int:
    if argumente:
        wurzeln = [(Path(a), Path(a).name, None) for a in argumente]
    else:
        wurzeln = [(HANDOFF, "Handoff", None), (ARCHTHALIA, "ArchThalia-ML", SONDER_TFS)]
    zeilen = []
    for wurzel, quelle, nur in wurzeln:
        if not wurzel.is_dir():
            print(f"Nicht gefunden, übersprungen: {wurzel}")
            continue
        zeilen += erfassen(wurzel, quelle, nur)
    if not zeilen:
        print("Keine TF-Ordner gefunden.")
        return 1
    ziel = Path("negative_uebersicht.csv")
    with ziel.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(zeilen[0]))
        writer.writeheader()
        writer.writerows(zeilen)

    # Matrix: welche Typen je TF und Assay/Split vorliegen (über alle Quellen zusammengefasst)
    spalten = [(a, s) for a in ASSAYS for s in SPLITS]
    vorhanden: dict[tuple[str, str, str], set[str]] = {}
    for z in zeilen:
        typen = {t for t in TYPEN if z[f"n_{t}"] not in ("", 0)}
        vorhanden.setdefault((z["tf"], z["assay"], z["split"]), set()).update(typen)
    tfs = sorted({z["tf"] for z in zeilen})
    print("Vorhandene Typen je TF (p = positives, r = random, a = aliens, s = shades, – = Ordner fehlt):")
    print(f"{'TF':10s}" + "".join(f"{a + '-' + s:>13s}" for a, s in spalten))
    for tf in tfs:
        felder = []
        for a, s in spalten:
            typen = vorhanden.get((tf, a, s))
            felder.append("–" if typen is None else "".join(KUERZEL[t] for t in TYPEN if t in typen) or "leer")
        print(f"{tf:10s}" + "".join(f"{f:>13s}" for f in felder))

    print("\nZusammenfassung:")
    for a, s in spalten:
        mit = {t: sorted(tf for tf in tfs if t in vorhanden.get((tf, a, s), set())) for t in TYPEN}
        gesamt = sum(1 for tf in tfs if (tf, a, s) in vorhanden)
        verh = {}
        for t in TYPEN[1:]:
            werte = [z[f"{t}_je_positiv"] for z in zeilen
                     if z["assay"] == a and z["split"] == s and z[f"{t}_je_positiv"] != ""]
            verh[t] = f" (Median {statistics.median(werte):g} je Positivem)" if werte else ""
        beide = sorted(set(mit["aliens"]) & set(mit["shades"]))
        print(f"  {a} {s}: {gesamt} TFs | random {len(mit['random'])}{verh['random']} | "
              f"aliens {len(mit['aliens'])}{verh['aliens']} | shades {len(mit['shades'])}{verh['shades']} | "
              f"beide {len(beide)}")
        for t in ("aliens", "shades"):
            if mit[t] and len(mit[t]) < gesamt:
                print(f"    mit {t}: {', '.join(mit[t])}")
    weitere = sorted({(z["tf"], z["assay"], z["split"], z["weitere_dateien"]) for z in zeilen if z["weitere_dateien"]})
    if weitere:
        print("\nWeitere Dateien (kein positives/random/aliens/shades):")
        for tf, a, s, namen in weitere[:20]:
            print(f"  {tf} {a} {s}: {namen}")
        if len(weitere) > 20:
            print(f"  … und {len(weitere) - 20} weitere Ordner, siehe CSV")
    doppelt = sorted({z["tf"] for z in zeilen if z["quelle"] != zeilen[0]["quelle"]}
                     & {z["tf"] for z in zeilen if z["quelle"] == zeilen[0]["quelle"]})
    if doppelt:
        print(f"\nIn beiden Quellen vorhanden (in der Matrix zusammengefasst): {', '.join(doppelt)}")
    print(f"\nDetails je TF: {ziel.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
