#!/usr/bin/env python3
"""Lädt fehlende Negativ-Dateien (aliens/foreigns, shades) aus dem offiziellen ArChIPelago-Datensatz auf Zenodo und
legt sie neben die vorhandenen Dateien des TFs. Aus datasets.zip (7,4 GB) werden per Teil-Download nur die
benötigten Dateien gelesen; eine Sicherheitsgrenze bricht ab, falls mehr als 500 MB anfallen würden.

Ablauf
 1. Wie negative_uebersicht.py: deine TFs aus big_ablation/configs/caches.csv, die Handoff-Daten und für TFs, die
    dort fehlen, ArchThalia-ML. Fehlend heißt: ein Ordner Assay/Split/TF mit positives, aber ohne aliens/foreigns
    oder ohne shades.
 2. Zenodo: die Versionen des Datensatzes, neueste zuerst. Gesucht werden im Zip (nur .bed/.bed.gz)
    aliens = foreigns, shades = shades_addshift (Train) bzw. shades (Test), wie im Projekt-Notebook.
 3. Prüfung: positives im Zip müssen dieselben Fenster enthalten wie deine positives (Koordinaten und Anzahl).
    Nur dann wird abgelegt; sonst kommt die nächste Version dran. Mehrdeutige Treffer im Zip werden übersprungen.
 4. Ablage im Ordner des TFs. Name nach dem Schema dort: Handoff aliens.bed, ArchThalia-ML foreigns.bed,
    jeweils shades.bed. Vorhandene Dateien werden nie überschrieben (atomares Anlegen). Ist der Ordner nicht
    beschreibbar oder ist --alles-nach-ersatz gesetzt, landet die Datei in gleicher Struktur unter --ersatz.
 5. Mit --genom hg38.fa (mit .fai daneben) zusätzlich als FASTA: eine Zeile je Sequenz, Kopf chr:start-end wie
    im Notebook.
 Protokoll: ergaenzung_herkunft.csv im aktuellen Ordner (Ziel, Zenodo-Version, Quelldatei, Zeilen, SHA-256),
 je Datei sofort geschrieben.
 Rückgabewert: 0 alles ergänzt, 2 etwas offen geblieben, 1 Abbruch vor dem Start.

Aufruf (am besten in human301, nicht in big_ablation):
    python negative_ergaenzen.py --nur-zeigen      zeigt, was fehlt und was geladen würde; schreibt nichts
    python negative_ergaenzen.py                   lädt und legt ab
    Optionen: --genom PFAD, --record ID, --zip PFAD_ODER_URL, --ersatz PFAD, --alles-nach-ersatz, --anzahl-reicht
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.client
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

HANDOFF = Path("/data/projects/SFB_A03/jan/AT_TFBS/raw_data/clara/chip/student_handoff_30tf_increment/Input_data")
ARCHTHALIA = Path("/data/projects/SFB_A03/clara/ArchThalia-ML")
ERSATZ = Path("/data/projects/SFB_A03/jan/AT_TFBS/raw_data/archipelago_ergaenzung")
KONZEPT = "10515307"                           # Zenodo-DOI 10.5281/zenodo.10515307, alle Versionen
BEKANNTE_VERSIONEN = ["15555251", "10515308"]  # v4 (Mai 2025), 2024.1 (Jan 2024), falls die API nicht antwortet
DATEI = "datasets.zip"
MAX_BYTES = 500_000_000
ASSAYS = ("CHS", "GHTS")
SPLITS = ("Train", "Test")
FASTA_ENDUNGEN = (".fa", ".fasta", ".fna")
PLATTFORM = {"chs": "CHS", "chip": "CHS", "chip-seq": "CHS", "chipseq": "CHS",
             "ghts": "GHTS", "afs": "GHTS", "ght-selex": "GHTS", "ghtselex": "GHTS"}
QUELLNAMEN = {("aliens", "Train"): ["foreigns", "aliens"], ("aliens", "Test"): ["foreigns", "aliens"],
              ("shades", "Train"): ["shades_addshift", "shades"], ("shades", "Test"): ["shades", "shades_addshift"]}
KENNUNG = "negative_ergaenzen/1.1"
PROTOKOLL = Path("ergaenzung_herkunft.csv")
FELDER = ["ziel", "quelle", "datei_im_zip", "zeilen", "sha256", "zeitpunkt"]


# ------------------------------------------------------------------ lokale Daten
def grundname(path: Path) -> str:
    name = path.name.lower()
    return name[:-3] if name.endswith(".gz") else name


def typ_von(path: Path) -> str | None:
    stamm = grundname(path).split(".")[0]
    for endung in ("_no_nf", "_addshift"):
        if stamm.endswith(endung):
            stamm = stamm[: -len(endung)]
    stamm = {"foreigns": "aliens"}.get(stamm, stamm)
    return stamm if stamm in ("positives", "random", "aliens", "shades") else None


def eigene_tfs() -> list[str] | None:
    for basis in (Path.cwd(), Path(__file__).resolve().parent):
        pfad = basis / "big_ablation" / "configs" / "caches.csv"
        if pfad.is_file():
            with pfad.open(newline="") as handle:
                return sorted({z["tf"] for z in csv.DictReader(handle) if z.get("tf")})
    return None


def ordner_erfassen(wurzel: Path, quelle: str, nur=None, ohne=frozenset()) -> dict:
    gefunden = {}
    for assay in ASSAYS:
        for split in SPLITS:
            basis = wurzel / assay / split
            if not basis.is_dir():
                continue
            for ordner in sorted(p for p in basis.iterdir() if p.is_dir()):
                if (nur is not None and ordner.name not in nur) or ordner.name in ohne:
                    continue
                dateien = [p for p in ordner.iterdir() if p.is_file()]
                typen = {t for p in dateien if (t := typ_von(p))}
                positives = sorted((p for p in dateien if typ_von(p) == "positives" and "_no_nf" not in p.name.lower()),
                                   key=lambda p: (not grundname(p).endswith(".bed"), p.name))
                gefunden[(ordner.name, assay, split)] = {"ordner": ordner, "quelle": quelle, "typen": typen,
                                                        "positives": positives[0] if positives else None}
    return gefunden


def lokale_daten() -> dict:
    liste = eigene_tfs()
    daten = ordner_erfassen(HANDOFF, "Handoff") if HANDOFF.is_dir() else {}
    im_handoff = {tf for tf, _, _ in daten}
    if ARCHTHALIA.is_dir():
        daten.update(ordner_erfassen(ARCHTHALIA, "ArchThalia-ML",
                                     nur=(set(liste) - im_handoff) if liste is not None else None, ohne=im_handoff))
    return daten


KOORD = re.compile(r"(chr[0-9A-Za-z_]+):(\d+)-(\d+)")


def bed_fenster(zeilen) -> tuple[list | None, int]:
    """(chrom, start, ende) je Datenzeile; None, wenn eine Zeile nicht lesbar ist."""
    fenster, lesbar = [], True
    for zeile in zeilen:
        if not zeile.strip() or zeile.startswith(("#", "track", "browser")):
            continue
        felder = zeile.split()
        if len(felder) >= 3 and felder[1].isdigit() and felder[2].isdigit():
            fenster.append((felder[0], int(felder[1]), int(felder[2])))
        else:
            lesbar = False
            fenster.append(None)
    return (fenster if lesbar else None), len(fenster)


def fenster_lokal(pfad: Path) -> tuple[list | None, int]:
    oeffnen = gzip.open if pfad.name.lower().endswith(".gz") else open
    with oeffnen(pfad, "rt", errors="replace") as handle:
        if not grundname(pfad).endswith(FASTA_ENDUNGEN):
            return bed_fenster(handle)
        fenster, lesbar = [], True
        for zeile in handle:
            if zeile.startswith(">"):
                treffer = KOORD.search(zeile)
                if treffer:
                    fenster.append((treffer.group(1), int(treffer.group(2)), int(treffer.group(3))))
                else:
                    lesbar = False
                    fenster.append(None)
        return (fenster if lesbar else None), len(fenster)


# ------------------------------------------------------------------ Zenodo: Teil-Downloads aus der Zip
def anfrage(url: str, bereich: str | None = None, versuche: int = 5):
    for versuch in range(versuche):
        kopf = {"User-Agent": KENNUNG}
        if bereich:
            kopf["Range"] = bereich
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=kopf), timeout=120)
        except urllib.error.HTTPError as fehler:
            if fehler.code == 429 and versuch < versuche - 1:          # Zenodo-Ratenlimit: warten
                warte = fehler.headers.get("Retry-After", "60")
                time.sleep(int(warte) if str(warte).isdigit() else 60)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if versuch < versuche - 1:
                time.sleep(5 * (versuch + 1))
                continue
            raise
    raise RuntimeError(f"Keine Antwort von {url}")


class Fernzugriff(io.RawIOBase):
    """Lesbare, suchbare Datei über HTTP-Bereichsanfragen; lädt nur, was zipfile tatsächlich liest."""

    def __init__(self, url: str):
        self.url, self.pos, self.geladen = url, 0, 0
        with anfrage(url, "bytes=0-0") as antwort:
            treffer = re.match(r"bytes 0-0/(\d+)", antwort.headers.get("Content-Range", ""))
            if antwort.status != 206 or not treffer:
                raise RuntimeError("Der Server erlaubt keine Teil-Downloads; nur der ganze Download (7,4 GB) ginge. "
                                   "Lade datasets.zip dann selbst und gib sie mit --zip an.")
            antwort.read(1)
            self.groesse = int(treffer.group(1))
            self.ziel = antwort.geturl()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.groesse + offset}[whence]
        return self.pos

    def readall(self):
        """zipfile bis Python 3.12 liest das Archivende ohne Längenangabe; erlaubt ist das nur für einen kleinen Rest."""
        rest = self.groesse - self.pos
        if rest > 2 * 1024 * 1024:
            raise RuntimeError("Unbegrenztes Lesen der entfernten Datei ist gesperrt")
        teile = []
        while self.pos < self.groesse:
            puffer = bytearray(self.groesse - self.pos)
            n = self.readinto(puffer)
            if not n:
                break
            teile.append(bytes(puffer[:n]))
        return b"".join(teile)

    def readinto(self, puffer):
        if self.pos >= self.groesse:
            return 0
        ende = min(self.pos + len(puffer), self.groesse) - 1
        soll = ende - self.pos + 1
        for versuch in range(4):
            try:
                with anfrage(self.ziel, f"bytes={self.pos}-{ende}") as antwort:
                    treffer = re.match(r"bytes (\d+)-(\d+)/", antwort.headers.get("Content-Range", ""))
                    if (antwort.status != 206 or not treffer or int(treffer.group(1)) != self.pos
                            or int(treffer.group(2)) > ende):
                        raise RuntimeError("Unerwartete Antwort auf eine Teil-Anfrage; Abbruch, damit nicht die "
                                           "ganze Datei geladen wird.")
                    daten = antwort.read(soll)
                if not daten:
                    raise ConnectionError("leere Antwort")
                break
            except urllib.error.HTTPError as fehler:            # abgelaufener Weiterleitungs-Link: neu holen
                if fehler.code in (403, 410) and versuch < 3:
                    with anfrage(self.url, "bytes=0-0") as antwort:
                        self.ziel = antwort.geturl()
                    continue
                raise
            except (http.client.HTTPException, ConnectionError, TimeoutError, urllib.error.URLError):
                if versuch < 3:
                    time.sleep(5 * (versuch + 1))
                    continue
                raise
        n = len(daten)
        puffer[:n] = daten
        self.pos += n
        self.geladen += n
        if self.geladen > MAX_BYTES:
            raise RuntimeError(f"Sicherheitsgrenze von {MAX_BYTES / 1e6:.0f} MB überschritten; Abbruch")
        return n


def versionen() -> list[tuple[str, str]]:
    """(Record-ID, Bezeichnung), neueste zuerst."""
    letzter = None
    for url in (f"https://zenodo.org/api/records/{BEKANNTE_VERSIONEN[0]}/versions?size=25",
                f"https://zenodo.org/api/records/{BEKANNTE_VERSIONEN[0]}/versions",
                f"https://zenodo.org/api/records/{KONZEPT}/versions"):
        try:
            with anfrage(url, versuche=2) as antwort:
                treffer = json.load(antwort)["hits"]["hits"]
            liste = [(str(h["id"]), f"{h.get('metadata', {}).get('version', '?')} vom "
                                    f"{h.get('metadata', {}).get('publication_date', '?')}") for h in treffer]
            liste.sort(key=lambda x: int(x[0]), reverse=True)
            if liste:
                bekannt = {rid for rid, _ in liste}
                return liste + [(rid, "bekannte Version") for rid in BEKANNTE_VERSIONEN if rid not in bekannt]
        except Exception as fehler:  # noqa: BLE001 - nächste Variante, dann bekannte Versionen
            letzter = fehler
    print(f"Zenodo-Versionsliste nicht abrufbar ({letzter}); nutze die bekannten Versionen.")
    return [(rid, "bekannte Version") for rid in BEKANNTE_VERSIONEN]


def zip_oeffnen(quelle: str):
    if Path(quelle).is_file():
        return zipfile.ZipFile(quelle), None
    roh = Fernzugriff(quelle)
    return zipfile.ZipFile(io.BufferedReader(roh, buffer_size=1024 * 1024)), roh


def eindeutig(werte: set):
    return next(iter(werte)) if len(werte) == 1 else None


def tf_aus(teile: list[str], tfs: set[str]) -> str | None:
    """Genauer Ordnername; nur wenn keiner passt, Name gefolgt von '.' (z. B. GLI4.THC_0193). Mehrdeutig: None."""
    genau = {t for t in teile if t in tfs}
    if genau:
        return eindeutig(genau)
    return eindeutig({k for t in teile for k in tfs if t.startswith(k + ".")})


def zip_inhalt(zf: zipfile.ZipFile, tfs: set[str]) -> dict:
    """(TF, Assay, Split, Stamm) -> Liste der Namen im Zip (nur .bed und .bed.gz)."""
    index = {}
    for name in zf.namelist():
        teile = name.split("/")
        datei = teile[-1].lower()
        if not (datei.endswith(".bed") or datei.endswith(".bed.gz")):
            continue
        ordner = teile[:-1]
        assay = eindeutig({PLATTFORM[t.lower()] for t in ordner if t.lower() in PLATTFORM})
        split = eindeutig({t.capitalize() for t in ordner if t.lower() in ("train", "test")})
        tf = tf_aus(ordner, tfs)
        if assay and split and tf:
            index.setdefault((tf, assay, split, datei.split(".")[0]), []).append(name)
    return index


def lesen(zf: zipfile.ZipFile, name: str) -> str:
    daten = zf.read(name)
    if name.lower().endswith(".gz"):
        daten = gzip.decompress(daten)
    return daten.decode("utf-8")


# ------------------------------------------------------------------ FASTA aus dem Genom (optional)
class Genom:
    def __init__(self, pfad: Path):
        fai = Path(str(pfad) + ".fai")
        if not fai.is_file():
            raise FileNotFoundError(f"Index fehlt: {fai} (samtools faidx {pfad})")
        self.handle = pfad.open("rb")
        self.index = {}
        for zeile in fai.read_text().splitlines():
            name, laenge, offset, basen, breite = zeile.split("\t")[:5]
            self.index[name] = (int(laenge), int(offset), int(basen), int(breite))

    def sequenz(self, chrom: str, start: int, ende: int) -> str:
        laenge, offset, basen, breite = self.index[chrom]
        if not 0 <= start < ende <= laenge:
            raise ValueError(f"Bereich außerhalb von {chrom}: {start}-{ende}")
        anfang = offset + (start // basen) * breite + start % basen
        schluss = offset + ((ende - 1) // basen) * breite + (ende - 1) % basen
        self.handle.seek(anfang)
        return self.handle.read(schluss - anfang + 1).decode().replace("\n", "").replace("\r", "")


def fasta_aus(text: str, genom: Genom) -> tuple[str, int]:
    """FASTA zu einer BED-Datei; Fehler (Chromosom fehlt, Bereich außerhalb) brechen nur diese Datei ab."""
    fenster, _ = bed_fenster(text.splitlines())
    if fenster is None:
        raise ValueError("BED-Zeilen nicht lesbar")
    stuecke = []
    for c, s, e in fenster:
        if c not in genom.index:
            raise ValueError(f"Chromosom {c} fehlt im Genom")
        stuecke.append(f">{c}:{s}-{e}\n{genom.sequenz(c, s, e)}\n")
    return "".join(stuecke), len(stuecke)


# ------------------------------------------------------------------ Ablage
def neu_anlegen(ziel: Path, text: str) -> bool:
    """Legt ziel nur an, wenn es noch nicht existiert (atomar per Hardlink); False, wenn es schon da ist."""
    fd, zwischen = tempfile.mkstemp(dir=ziel.parent, prefix=f".{ziel.name}.", suffix=".unfertig")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(zwischen, 0o644)
        try:
            os.link(zwischen, ziel)
        except FileExistsError:
            return False
        except OSError:                                  # Dateisystem ohne Hardlinks
            try:
                with open(ziel, "x") as handle:
                    handle.write(text)
                os.chmod(ziel, 0o644)
            except FileExistsError:
                return False
        return True
    finally:
        os.unlink(zwischen)


def beschreibbar(ordner: Path) -> bool:
    return os.access(ordner, os.W_OK | os.X_OK)


def protokollieren(eintrag: dict) -> None:
    """Sofort je Datei anhängen, damit nach einem Abbruch nichts unprotokolliert liegt."""
    neu = not PROTOKOLL.exists()
    with PROTOKOLL.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FELDER)
        if neu:
            writer.writeheader()
        writer.writerow(eintrag)


def vergleich(info: dict, text: str, anzahl_reicht: bool) -> tuple[bool, str]:
    lokal, n_lokal = fenster_lokal(info["positives"])
    fern, n_fern = bed_fenster(text.splitlines())
    kopf = f"positives {n_lokal} lokal, {n_fern} im Zip – "
    if n_lokal == 0 or n_fern == 0:
        return False, kopf + "leer, nicht vergleichbar"
    if lokal is not None and fern is not None:
        gleich = n_lokal == n_fern and set(lokal) == set(fern) and len(set(lokal)) == len(set(fern))
        return gleich, kopf + ("gleich (Koordinaten)" if gleich else "VERSCHIEDEN (Koordinaten), diese Version passt nicht")
    if anzahl_reicht:
        return n_lokal == n_fern, kopf + ("nur Anzahl geprüft: gleich" if n_lokal == n_fern else "VERSCHIEDEN (Anzahl)")
    return False, kopf + "Koordinaten nicht lesbar; mit --anzahl-reicht genügt die gleiche Anzahl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fehlende aliens/shades aus dem ArChIPelago-Datensatz ergänzen")
    parser.add_argument("--nur-zeigen", action="store_true", help="nichts schreiben, nur anzeigen")
    parser.add_argument("--genom", type=Path, help="hg38-FASTA mit .fai: zusätzlich FASTA-Dateien erzeugen")
    parser.add_argument("--record", help="bestimmte Zenodo-Record-ID statt automatischer Wahl")
    parser.add_argument("--zip", help="lokale datasets.zip oder eine andere URL")
    parser.add_argument("--ersatz", type=Path, default=ERSATZ, help="Ablage, falls ein Zielordner nicht beschreibbar ist")
    parser.add_argument("--alles-nach-ersatz", action="store_true",
                        help="nichts neben die vorhandenen Dateien legen, alles in gleicher Struktur unter --ersatz")
    parser.add_argument("--anzahl-reicht", action="store_true",
                        help="sind deine Koordinaten nicht lesbar, genügt die gleiche Anzahl Positiver")
    args = parser.parse_args()

    daten = lokale_daten()
    if not daten:
        print("Keine lokalen Daten gefunden.")
        return 1
    fehlend = [(tf, assay, split, typ) for (tf, assay, split), info in sorted(daten.items())
               if info["positives"] is not None for typ in ("aliens", "shades") if typ not in info["typen"]]
    if not fehlend:
        print("Nichts fehlt: Jeder Ordner mit positives hat aliens/foreigns und shades.")
        return 0
    print(f"Fehlend: {len(fehlend)} Dateien")
    for tf, assay, split, typ in fehlend:
        print(f"  {tf:8s} {assay:4s} {split:5s} {typ}  ({daten[(tf, assay, split)]['ordner']})")
    if not args.nur_zeigen:
        if not os.access(Path.cwd(), os.W_OK) or (PROTOKOLL.exists() and not os.access(PROTOKOLL, os.W_OK)):
            print(f"Abbruch: Das Protokoll {PROTOKOLL.resolve()} ist nicht beschreibbar. Starte im Ordner human301.")
            return 1
    try:
        genom = Genom(args.genom) if args.genom else None
    except FileNotFoundError as fehler:
        print(f"Abbruch: {fehler}")
        return 1

    quellen = ([(args.zip, "angegebene Datei")] if args.zip else
               [(f"https://zenodo.org/records/{rid}/files/{DATEI}?download=1", f"Record {rid}, {txt}")
                for rid, txt in ([(args.record, "angegeben")] if args.record else versionen())])
    tfs = {tf for tf, _, _, _ in fehlend}
    offen, abgelegt, unvollstaendig = list(fehlend), 0, []
    for url, bezeichnung in quellen:
        if not offen:
            break
        print(f"\nQuelle: {bezeichnung}")
        roh = None
        try:
            zf, roh = zip_oeffnen(url)
            index = zip_inhalt(zf, tfs)
        except Exception as fehler:  # noqa: BLE001 - nächste Version versuchen
            print(f"  nicht lesbar: {fehler}")
            continue
        if not index:
            namen = zf.namelist()
            print(f"  Keine passenden Dateien erkannt. Die Zip enthält {len(namen)} Einträge, zum Beispiel:")
            for name in namen[:25]:
                print(f"    {name}")
            continue
        geprueft, noch_offen = {}, []
        for tf, assay, split, typ in offen:
            info = daten[(tf, assay, split)]
            try:
                kandidaten = next((index[(tf, assay, split, s)] for s in QUELLNAMEN[(typ, split)]
                                   if (tf, assay, split, s) in index), None)
                pos_namen = index.get((tf, assay, split, "positives"))
                if not kandidaten or not pos_namen:
                    print(f"  {tf} {assay} {split} {typ}: nicht in dieser Version")
                    noch_offen.append((tf, assay, split, typ))
                    continue
                if len(kandidaten) > 1 or len(pos_namen) > 1:
                    doppelt = kandidaten if len(kandidaten) > 1 else pos_namen
                    print(f"  {tf} {assay} {split} {typ}: mehrdeutig im Zip ({', '.join(doppelt)}), übersprungen")
                    noch_offen.append((tf, assay, split, typ))
                    continue
                name = kandidaten[0]
                if (tf, assay, split) not in geprueft:
                    gleich, meldung = vergleich(info, lesen(zf, pos_namen[0]), args.anzahl_reicht)
                    geprueft[(tf, assay, split)] = gleich
                    print(f"  {tf} {assay} {split}: {meldung}")
                if not geprueft[(tf, assay, split)]:
                    noch_offen.append((tf, assay, split, typ))
                    continue
                zielname = {"aliens": "aliens" if info["quelle"] == "Handoff" else "foreigns", "shades": "shades"}[typ]
                ordner = info["ordner"]
                if args.alles_nach_ersatz or not beschreibbar(ordner):
                    ordner = args.ersatz / assay / split / tf
                ziel = ordner / f"{zielname}.bed"
                print(f"  -> {ziel}  (aus {name})" + ("  [existiert schon]" if ziel.exists() else ""))
                if args.nur_zeigen:
                    continue
                text = lesen(zf, name)
                ordner.mkdir(parents=True, exist_ok=True)
                if not neu_anlegen(ziel, text):
                    print("     existiert schon, nicht überschrieben")
                else:
                    protokollieren({"ziel": str(ziel), "quelle": bezeichnung, "datei_im_zip": name,
                                    "zeilen": sum(1 for z in text.splitlines() if z.strip()),
                                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    "zeitpunkt": datetime.now().isoformat(timespec="seconds")})
                    abgelegt += 1
            except Exception as fehler:  # noqa: BLE001 - diese Datei offen lassen, weiter mit den übrigen
                print(f"  {tf} {assay} {split} {typ}: FEHLER {fehler}")
                noch_offen.append((tf, assay, split, typ))
                continue
            if genom:
                ziel_fa = ordner / f"{zielname}.fa"
                try:
                    inhalt, anzahl = fasta_aus(text, genom)
                    if not neu_anlegen(ziel_fa, inhalt):
                        print(f"     {ziel_fa.name} existiert schon, nicht überschrieben")
                        continue
                    protokollieren({"ziel": str(ziel_fa), "quelle": bezeichnung, "datei_im_zip": name,
                                    "zeilen": anzahl, "sha256": hashlib.sha256(inhalt.encode()).hexdigest(),
                                    "zeitpunkt": datetime.now().isoformat(timespec="seconds")})
                    abgelegt += 1
                    print(f"     dazu {ziel_fa.name} ({anzahl} Sequenzen)")
                except Exception as fehler:  # noqa: BLE001
                    print(f"     FASTA nicht erzeugt: {fehler} (die BED-Datei ist abgelegt)")
                    unvollstaendig.append(f"{ziel_fa}")
        offen = noch_offen
        if roh is not None:
            print(f"  geladen aus dieser Quelle: {roh.geladen / 1e6:.1f} MB")
    if abgelegt:
        print(f"\nAbgelegt: {abgelegt} Dateien; Protokoll: {PROTOKOLL.resolve()}")
    if offen:
        print(f"\nNicht ergänzt ({len(offen)}): " + ", ".join(f"{tf} {a} {s} {t}" for tf, a, s, t in offen))
    if unvollstaendig:
        print(f"FASTA fehlt ({len(unvollstaendig)}): " + ", ".join(unvollstaendig))
    if args.nur_zeigen:
        print("\nNur angezeigt, nichts geschrieben. Ohne --nur-zeigen starten, um abzulegen.")
        return 2 if offen else 0
    return 2 if offen or unvollstaendig else 0


if __name__ == "__main__":
    sys.exit(main())
