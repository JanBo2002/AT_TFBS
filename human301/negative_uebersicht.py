#!/usr/bin/env python3
"""Überblick: Welche Negativ-Typen gibt es je TF (CHS und GHTS, Train und Test), und wie viele Sequenzen?

Durchsucht <Wurzel>/<CHS|GHTS>/<Train|Test>/<TF>/ und zählt je Datei die Einträge:
BED = Zeilen ohne Kopf- und Kommentarzeilen, FASTA = Header-Zeilen (">"); .gz wird mitgelesen.
Typen: positives, random, aliens, shades, foreigns; dazu jeweils die Variante <typ>_no_NF, falls vorhanden.
Standard: alle TFs der Handoff-Daten. Jeder deiner TFs, der dort fehlt, wird unter ArchThalia-ML gesucht. Die
TF-Liste kommt aus big_ablation/configs/caches.csv. Ohne diese Datei werden alle TFs aus ArchThalia-ML gelesen,
die nicht in den Handoff-Daten stehen.

Aufruf (am besten in human301, nicht in big_ablation):
    python negative_uebersicht.py                 Standard
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
TYPEN = ("positives", "random", "aliens", "shades", "foreigns")
NEGATIVE = TYPEN[1:]
KUERZEL = {"positives": "p", "random": "r", "aliens": "a", "shades": "s", "foreigns": "f"}
ASSAYS = ("CHS", "GHTS")
SPLITS = ("Train", "Test")
FASTA_ENDUNGEN = (".fa", ".fasta", ".fna")
NO_NF = "_no_nf"


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


def typ(path: Path) -> tuple[str, str] | None:
    """(Typ, Variante) aus dem Dateinamen, z. B. positives_no_NF.fa -> ("positives", "_no_nf")."""
    stamm = grundname(path).split(".")[0]
    variante = ""
    if stamm.endswith(NO_NF):
        stamm, variante = stamm[: -len(NO_NF)], NO_NF
    return (stamm, variante) if stamm in TYPEN else None


def bevorzugt(dateien: list[Path], assay: str) -> Path:
    """Gibt es einen Typ als BED und als FASTA, zählt bei CHS die BED-, bei GHTS die FASTA-Datei."""
    fasta = [f for f in dateien if grundname(f).endswith(FASTA_ENDUNGEN)]
    andere = [f for f in dateien if f not in fasta]
    return ((fasta + andere) if assay == "GHTS" else (andere + fasta))[0]


def eigene_tfs() -> tuple[list[str] | None, Path | None]:
    for basis in (Path.cwd(), Path(__file__).resolve().parent):
        pfad = basis / "big_ablation" / "configs" / "caches.csv"
        if pfad.is_file():
            with pfad.open(newline="") as handle:
                return sorted({z["tf"] for z in csv.DictReader(handle)}), pfad
    return None, None


def erfassen(wurzel: Path, quelle: str, nur: set[str] | None = None, ohne: set[str] = frozenset()) -> list[dict]:
    zeilen = []
    for assay in ASSAYS:
        for split in SPLITS:
            ordner_split = wurzel / assay / split
            if not ordner_split.is_dir():
                continue
            for ordner in sorted(p for p in ordner_split.iterdir() if p.is_dir()):
                if (nur is not None and ordner.name not in nur) or ordner.name in ohne:
                    continue
                nach_typ, weitere = {}, []
                for datei in sorted(p for p in ordner.iterdir() if p.is_file()):
                    erkannt = typ(datei)
                    if erkannt is None:
                        weitere.append(datei.name)
                    else:
                        nach_typ.setdefault(erkannt, []).append(datei)
                zeile = {"quelle": quelle, "assay": assay, "split": split, "tf": ordner.name}
                for t in TYPEN:
                    dateien = nach_typ.get((t, ""), [])
                    zeile[f"n_{t}"] = zaehlen(bevorzugt(dateien, assay)) if dateien else ""
                n_pos = zeile["n_positives"]
                for t in NEGATIVE:
                    n = zeile[f"n_{t}"]
                    zeile[f"{t}_je_positiv"] = round(n / n_pos, 1) if n != "" and n_pos else ""
                for t in TYPEN:
                    dateien = nach_typ.get((t, NO_NF), [])
                    zeile[f"n_{t}_no_NF"] = zaehlen(bevorzugt(dateien, assay)) if dateien else ""
                zeile["dateien"] = " ".join(f.name for key in sorted(nach_typ) for f in nach_typ[key])
                zeile["weitere_dateien"] = " ".join(weitere)
                zeilen.append(zeile)
    return zeilen


def median_text(werte: list[float], einheit: str = " je Positivem") -> str:
    return f" (Median {statistics.median(werte):g}{einheit})" if werte else ""


def main(argumente: list[str]) -> int:
    liste = None
    if argumente:
        zeilen = []
        for a in argumente:
            wurzel = Path(a)
            if wurzel.is_dir():
                zeilen += erfassen(wurzel, wurzel.name)
            else:
                print(f"Nicht gefunden, übersprungen: {wurzel}")
    else:
        liste, listenpfad = eigene_tfs()
        if liste:
            print(f"TF-Liste: {len(liste)} TFs aus {listenpfad}")
        else:
            print("big_ablation/configs/caches.csv nicht gefunden: ArchThalia-ML wird für alle TFs gelesen, "
                  "die nicht in den Handoff-Daten stehen.")
        zeilen = erfassen(HANDOFF, "Handoff") if HANDOFF.is_dir() else []
        if not HANDOFF.is_dir():
            print(f"Nicht gefunden: {HANDOFF}")
        im_handoff = {z["tf"] for z in zeilen}
        if ARCHTHALIA.is_dir():
            zeilen += erfassen(ARCHTHALIA, "ArchThalia-ML",
                               nur=(set(liste) - im_handoff) if liste else None, ohne=im_handoff)
        else:
            print(f"Nicht gefunden: {ARCHTHALIA}")
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
    quellen: dict[str, set[str]] = {}
    for z in zeilen:
        typen = {t for t in TYPEN if z[f"n_{t}"] not in ("", 0)}
        vorhanden.setdefault((z["tf"], z["assay"], z["split"]), set()).update(typen)
        quellen.setdefault(z["tf"], set()).add(z["quelle"])
    tfs = sorted(quellen)
    print("\nVorhandene Typen je TF (p = positives, r = random, a = aliens, s = shades, f = foreigns, "
          "– = Ordner fehlt; * = Quelle ArchThalia-ML):")
    print(f"{'TF':11s}" + "".join(f"{a + '-' + s:>13s}" for a, s in spalten))
    for tf in tfs:
        felder = []
        for a, s in spalten:
            typen = vorhanden.get((tf, a, s))
            felder.append("–" if typen is None else "".join(KUERZEL[t] for t in TYPEN if t in typen) or "leer")
        marke = "*" if "ArchThalia-ML" in quellen[tf] else ""
        print(f"{tf + marke:11s}" + "".join(f"{f:>13s}" for f in felder))

    print("\nZusammenfassung (Negative je Positivem):")
    for a, s in spalten:
        sel = [z for z in zeilen if z["assay"] == a and z["split"] == s]
        gesamt = len({z["tf"] for z in sel})
        teile = []
        for t in NEGATIVE:
            mit = sorted({z["tf"] for z in sel if z[f"n_{t}"] not in ("", 0)})
            werte = [z[f"{t}_je_positiv"] for z in sel if z[f"{t}_je_positiv"] != ""]
            teile.append(f"{t} {len(mit)}{median_text(werte, '')}")
        print(f"  {a} {s}: {gesamt} TFs | " + " | ".join(teile))
        for t in ("aliens", "shades", "foreigns"):
            mit = sorted({z["tf"] for z in sel if z[f"n_{t}"] not in ("", 0)})
            if mit and len(mit) < gesamt:
                print(f"    mit {t}: {', '.join(mit)}")
        mit_nonf = [z for z in sel if any(z[f"n_{t}_no_NF"] != "" for t in TYPEN)]
        if mit_nonf:
            vergleich = []
            for t in TYPEN:
                quoten = [z[f"n_{t}_no_NF"] / z[f"n_{t}"] for z in mit_nonf
                          if z[f"n_{t}_no_NF"] != "" and z[f"n_{t}"] not in ("", 0)]
                if quoten:
                    vergleich.append(f"{t} {statistics.median(quoten):.3g}×")
            print(f"    _no_NF-Fassungen bei {len({z['tf'] for z in mit_nonf})} TFs; Einträge gegenüber der "
                  f"normalen Datei (Median): {', '.join(vergleich)}")

    weitere = sorted({(z["tf"], z["assay"], z["split"], z["weitere_dateien"]) for z in zeilen if z["weitere_dateien"]})
    if weitere:
        print("\nWeitere Dateien (kein bekannter Typ):")
        for tf, a, s, namen in weitere[:20]:
            print(f"  {tf} {a} {s}: {namen}")
        if len(weitere) > 20:
            print(f"  … und {len(weitere) - 20} weitere Ordner, siehe CSV")
    if liste:
        fehlen = sorted(set(liste) - set(tfs))
        print(f"\nVon deinen {len(liste)} TFs gefunden: {len(liste) - len(fehlen)}"
              + (f"; nirgends gefunden: {', '.join(fehlen)}" if fehlen else ""))
        fremd = sorted(set(tfs) - set(liste))
        if fremd:
            print(f"Nicht in deiner Liste, trotzdem gezählt: {', '.join(fremd)}")
    print(f"\nDetails je TF: {ziel.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
