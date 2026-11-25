#!/usr/bin/env bash
# Lädt die Codebook-Daten zu PBM, SMiLE-seq und HT-SELEX (MEX) von Zenodo, prüft und entpackt sie und schreibt ein Inventar zum Hochladen.
#SBATCH --job-name=codebook_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=codebook_download-%j.out
#SBATCH --error=codebook_download-%j.out

set -o pipefail

ZIEL="${1:-${ZIEL:-/data/projects/SFB_A03/jan/AT_TFBS/raw_data/codebook}}"
OHNE_DOWNLOAD="${OHNE_DOWNLOAD:-0}"
MIN_FREI_GB="${MIN_FREI_GB:-}"

MEX="https://zenodo.org/records/15667805/files"
MEX_HTS="https://zenodo.org/records/15620252/files"
META=MEX_metadata_complete_dataset.tsv
DATEIEN=(
  "MEX_README.txt|$MEX/README?download=1|3646815063fa54186f5c96b47f4d324b"
  "$META|$MEX/metadata_complete_dataset.tsv?download=1|599e323ba3d79fa7f684baf711b1718a"
  "MEX.PBM.tar|$MEX/MEX.PBM.tar?download=1|9385a3e156e3f59df681cc95ef98432c"
  "MEX.SMS.tar|$MEX/MEX.SMS.tar?download=1|93b85d6db55f67a48da995630e2b629b"
  "MEX.HTS.TAR|$MEX_HTS/MEX.HTS.TAR?download=1|35169cdb118b7d33831d7b7567ea865b"
)
TFS="BATF2 CREB3L3 CTCF ELF3 FLI1 FOSL2 GABPA GLI4 LEF1 LEUTX MAX MGA MYF6 NFKB1 NR1H4 PAX7 RFX5 RORB SOX2 SRY TERF1 USF3 VDR YY1 ZBED9 ZBTB8A ZFP3 ZNF696 ZNF772 ZNF773"
# Typ | Archiv | 1 = ganz entpacken, 0 = nur Dateien deiner TFs | TFs mit gelungenen Experimenten laut Codebook-Übersicht
TYPEN=(
  "PBM|MEX.PBM.tar|1|CREB3L3 FLI1 FOSL2 GABPA LEF1 LEUTX MAX MYF6 NR1H4 PAX7 RORB SOX2 SRY TERF1 USF3 VDR YY1 ZBED9"
  "SMS|MEX.SMS.tar|1|CREB3L3 CTCF ELF3 FOSL2 LEF1 MAX NFKB1 PAX7 RORB SOX2 SRY USF3 VDR YY1 ZBTB8A"
  "HTS|MEX.HTS.TAR|0|BATF2 CREB3L3 CTCF ELF3 FLI1 FOSL2 GABPA GLI4 LEF1 LEUTX MAX MGA MYF6 NFKB1 NR1H4 PAX7 RFX5 RORB SOX2 SRY TERF1 USF3 VDR YY1 ZBED9 ZBTB8A ZFP3 ZNF696"
)
TAB="$(printf '\t')"

FEHLER=0
fehler() { echo "Error: $*" >&2; FEHLER=$((FEHLER + 1)); }
trap 'echo "Error: Abbruch durch ein Signal (Zeitlimit oder scancel). Skript noch einmal starten, es macht dort weiter" >&2; exit 143' TERM INT
schritt() { printf '\n== %s (%s)\n' "$*" "$(date '+%H:%M:%S')"; }

if command -v wget > /dev/null 2>&1; then
  LADER=wget
elif command -v curl > /dev/null 2>&1; then
  LADER=curl
else
  LADER=""
fi

hole() {
  if [ "$LADER" = "wget" ]; then
    wget -c -nv --timeout=60 --tries=2 -O "$2" "$1"
  else
    curl -L -sS --fail -C - --connect-timeout 60 -o "$2" "$1"
  fi
}

netz_da() {
  if [ "$LADER" = "wget" ]; then
    wget -q --timeout=20 --tries=1 -O /dev/null "https://zenodo.org/"
  else
    curl -L -sS --fail --connect-timeout 20 -o /dev/null "https://zenodo.org/" 2> /dev/null
  fi
}

lade() {
  local versuch
  for versuch in 1 2 3; do
    hole "$1" "$2" && return 0
    netz_da || return 2
    if [ "$versuch" -lt 3 ]; then
      echo "Versuch $versuch für $2 fehlgeschlagen, nächster Versuch in 60 Sekunden"
      sleep 60
    fi
  done
  return 1
}

md5_von() { md5sum "$1" | cut -d' ' -f1; }

lies() {
  case "$1" in
    *.gz) gzip -cd "$1" ;;
    *) cat "$1" ;;
  esac
}

packe_aus() {
  case "$1" in
    *.zip|*.ZIP)
      if command -v unzip > /dev/null 2>&1; then
        unzip -q -o "$1" -d "$2" < /dev/null
      else
        python3 -m zipfile -e "$1" "$2" < /dev/null
      fi
      ;;
    *)
      tar -xf "$1" -C "$2"
      ;;
  esac
}

lesbar() {
  if [ -r "$1" ]; then
    return 0
  fi
  fehler "$1 lässt sich nicht lesen"
  return 1
}

inhalt_von() {
  LC_ALL=C tar --quoting-style=literal -tvf "$1" |
    awk '$1 !~ /^[dl]/ {g = $3; p = $0; for (i = 1; i <= 5; i++) sub(/^[^ ]+ +/, "", p); print g "\t" p}'
}

# Eingabe: Bytes und Pfad. Ausgabe: TF, PBM-Kennung, Merkmale aus dem Pfad, Bytes, Pfad
ZUORDNUNG='
  BEGIN {n = split(tfs, a, " "); for (i = 1; i <= n; i++) tf[a[i]] = 1
         while ((getline z < "listen/kennung_tf.txt") > 0) {split(z, b, "\t"); ktf[b[1]] = b[2]}
         while ((getline z < "listen/kennung_design.txt") > 0) {split(z, b, "\t"); kde[b[1]] = b[2]}}
  {p = $2
   if (!archive && tolower(p) ~ /\.(zip|tar|tgz|tar\.gz)$/) next
   m = split(p, t, /[^A-Za-z0-9]+/); f = ""; id = "-"; mk = ""; split("", da)
   for (i = 1; i <= m; i++) {
     if (f == "" && (t[i] in tf)) f = t[i]
     if (t[i] ~ /^PBM[0-9]+$/) id = t[i]
     if (t[i] ~ /^(ME|HK|SD|SDQN|QNZS|Train|Val|Validation|Test|IVT|Lys|GFPIVT|C[0-9]+|Cycle[0-9]+|R[0-9]+)$/ && !(t[i] in da)) {
       da[t[i]] = 1; mk = mk (mk == "" ? "" : ",") t[i]
     }
   }
   if (f == "" && (id in ktf)) f = ktf[id]
   if ((id in kde) && !(kde[id] in da)) mk = mk (mk == "" ? "" : ",") kde[id]
   if (mk == "") mk = "-"
   if (f != "") print f "\t" id "\t" mk "\t" $1 "\t" p}'

for werkzeug in tar md5sum gzip awk find sort df head cut tr wc grep du mv touch date uniq; do
  command -v "$werkzeug" > /dev/null 2>&1 || fehler "das Programm $werkzeug fehlt auf $(hostname)"
done
if [ "$OHNE_DOWNLOAD" != "1" ] && [ -z "$LADER" ]; then
  fehler "weder wget noch curl ist auf $(hostname) vorhanden"
fi
if [ "$FEHLER" -gt 0 ]; then
  exit 1
fi

mkdir -p "$ZIEL/listen" || { fehler "der Zielordner $ZIEL lässt sich nicht anlegen"; exit 1; }
cd "$ZIEL" || { fehler "der Wechsel nach $ZIEL ist nicht möglich"; exit 1; }
echo "Zielordner: $(pwd) auf $(hostname)"

bedarf=2
[ -s MEX.PBM.tar ] || bedarf=$((bedarf + 12))
[ -s MEX.SMS.tar ] || bedarf=$((bedarf + 3))
[ -s MEX.HTS.TAR ] || bedarf=$((bedarf + 43))
[ -e entpackt/MEX.PBM/.fertig ] || bedarf=$((bedarf + 12))
[ -e entpackt/MEX.SMS/.fertig ] || bedarf=$((bedarf + 3))
[ -e entpackt/MEX.HTS/.fertig ] || bedarf=$((bedarf + 8))
if [ -n "$MIN_FREI_GB" ]; then
  bedarf=$MIN_FREI_GB
fi
frei_gb=$(df -Pk . | awk 'NR == 2 {print int($4 / 1048576)}')
if [ "${frei_gb:-0}" -lt "$bedarf" ]; then
  fehler "nur ${frei_gb:-0} GB frei in $ZIEL, gebraucht werden etwa $bedarf GB. Anderen Ordner als erstes Argument angeben oder die Grenze mit MIN_FREI_GB=0 abschalten"
  exit 1
fi

schritt "Download"
if [ "$OHNE_DOWNLOAD" = "1" ]; then
  echo "Übersprungen (OHNE_DOWNLOAD=1). Die Dateien müssen schon in $ZIEL liegen."
else
  for eintrag in "${DATEIEN[@]}"; do
    IFS='|' read -r name url soll <<< "$eintrag"
    if [ -s "$name" ] && [ "$(md5_von "$name")" = "$soll" ]; then
      echo "$name liegt schon vollständig vor"
      continue
    fi
    echo "Lade $name"
    lade "$url" "$name"
    status=$?
    if [ "$status" -eq 2 ]; then
      fehler "Zenodo ist von $(hostname) aus nicht erreichbar. Auf dem Login-Knoten starten: nohup bash codebook_download.sh > codebook_download.log 2>&1 &"
      echo "Hat auch der Login-Knoten keinen Zugang: die Dateien woanders laden, nach $ZIEL kopieren und mit OHNE_DOWNLOAD=1 starten."
      exit 1
    elif [ "$status" -ne 0 ]; then
      fehler "$name ließ sich nicht laden, obwohl Zenodo erreichbar ist. Skript später noch einmal starten, es setzt den Download fort"
      exit 1
    fi
  done
fi

schritt "Prüfsummen"
: > inv_0_md5.txt
ABWEICHEND=" "
for eintrag in "${DATEIEN[@]}"; do
  IFS='|' read -r name url soll <<< "$eintrag"
  if [ ! -s "$name" ]; then
    fehler "$name fehlt in $ZIEL"
    printf '%s\tfehlt\t%s\tfehlt\n' "$name" "$soll" >> inv_0_md5.txt
    continue
  fi
  ist=$(md5_von "$name")
  if [ "$ist" = "$soll" ]; then
    echo "$name: stimmt"
    printf '%s\t%s\t%s\tstimmt\n' "$name" "$ist" "$soll" >> inv_0_md5.txt
  else
    fehler "$name: Prüfsumme $ist statt $soll. Die Datei wird trotzdem verwendet. Meldet das Entpacken dafür einen Fehler, ist der Download beschädigt. Läuft es durch, liegt bei Zenodo vermutlich eine neuere Fassung"
    printf '%s\t%s\t%s\tweicht ab\n' "$name" "$ist" "$soll" >> inv_0_md5.txt
    ABWEICHEND="$ABWEICHEND$name "
  fi
done

schritt "Inventar 1: Beschreibung und Metadaten (README, Kopf der Metadaten, Zeilen je TF und Plattform, Zeilen deiner TFs)"
: > listen/kennung_tf.txt
: > listen/kennung_design.txt
{
  echo "### MEX_README.txt"
  if [ -s MEX_README.txt ]; then cat MEX_README.txt; else echo "fehlt"; fi
  echo
  if [ -s "$META" ]; then
    echo "### $META: $(wc -l < "$META") Zeilen, Kopfzeile und erste drei Zeilen"
    head -n 4 "$META" | cut -c1-1500
    echo
    echo "### Zeilen je TF und Plattform (gezählt werden Zeilen, in denen der TF-Name und die Plattform-Kennung als eigenes Wort stehen)"
    awk -v tfs="$TFS" '
      BEGIN {n = split(tfs, a, " "); for (i = 1; i <= n; i++) tf[a[i]] = 1
             k = split("CHS GHTS AFS HTS SMS PBM", pl, " ")}
      NR > 1 {m = split($0, t, /[^A-Za-z0-9]+/); f = ""; split("", da)
              for (i = 1; i <= m; i++) {if (f == "" && (t[i] in tf)) f = t[i]
                                        if (t[i] ~ /^PBM[0-9]+$/) da["PBM"] = 1; else da[t[i]] = 1}
              if (f == "") next
              z[f]++
              for (j = 1; j <= k; j++) if (pl[j] in da) c[f, pl[j]]++}
      END {printf "TF\tZeilen"; for (j = 1; j <= k; j++) printf "\t%s", pl[j]; printf "\n"
           for (i = 1; i <= n; i++) {printf "%s\t%d", a[i], z[a[i]]; for (j = 1; j <= k; j++) printf "\t%d", c[a[i], pl[j]]; printf "\n"}}' "$META"
    for plattform in PBM SMS HTS; do
      echo
      echo "### Zeilen deiner TFs mit $plattform (Kopfzeile, dann höchstens 400 Zeilen)"
      awk -v tfs="$TFS" -v pl="$plattform" '
        BEGIN {n = split(tfs, a, " "); for (i = 1; i <= n; i++) tf[a[i]] = 1}
        NR == 1 {print; next}
        {m = split($0, t, /[^A-Za-z0-9]+/); f = 0; p = 0
         for (i = 1; i <= m; i++) {if (t[i] in tf) f = 1; if (t[i] == pl || (pl == "PBM" && t[i] ~ /^PBM[0-9]+$/)) p = 1}
         if (f && p) print}' "$META" | head -n 401 | cut -c1-1500
    done
    awk -v tfs="$TFS" '
      BEGIN {n = split(tfs, a, " "); for (i = 1; i <= n; i++) tf[a[i]] = 1}
      NR > 1 {m = split($0, t, /[^A-Za-z0-9]+/); f = ""; d = ""
              for (i = 1; i <= m; i++) {if (f == "" && (t[i] in tf)) f = t[i]; if (t[i] == "ME" || t[i] == "HK") d = t[i]}
              for (i = 1; i <= m; i++) if (t[i] ~ /^PBM[0-9]+$/) {if (f != "") print t[i] "\t" f > "listen/kennung_tf.txt"
                                                                if (d != "") print t[i] "\t" d > "listen/kennung_design.txt"}}' "$META"
  else
    echo "### $META fehlt"
  fi
} > inv_1_metadaten.txt
sort -u -o listen/kennung_tf.txt listen/kennung_tf.txt
sort -u -o listen/kennung_design.txt listen/kennung_design.txt
doppelt=$(cut -f1 listen/kennung_tf.txt | uniq -d | tr '\n' ' ')
if [ -n "$doppelt" ]; then
  echo "Hinweis: Diese PBM-Kennungen stehen in den Metadaten bei mehreren deiner TFs: $doppelt" | tee -a inv_1_metadaten.txt
fi
if [ ! -s "$META" ]; then
  fehler "ohne $META lassen sich Dateien, die nur eine PBM-Kennung im Namen tragen, keinem TF zuordnen"
fi

for eintrag in "${TYPEN[@]}"; do
  IFS='|' read -r typ archiv ganz erwartet <<< "$eintrag"
  ordner="entpackt/MEX.$typ"
  schritt "$typ: Inhalt von $archiv lesen und entpacken"
  for liste in inhalt dateien eigene archive; do
    : > "listen/${typ}_$liste.txt"
  done
  archiv_ok=1
  if [ ! -s "$archiv" ]; then
    fehler "$archiv fehlt, $typ wird nicht entpackt"
    archiv_ok=0
  elif ! inhalt_von "$archiv" > "listen/${typ}_inhalt.txt" || [ ! -s "listen/${typ}_inhalt.txt" ]; then
    archiv_ok=0
    case "$ABWEICHEND" in
      *" $archiv "*)
        beiseite="$archiv.defekt_$(date '+%Y%m%d_%H%M%S')"
        mv -n "$archiv" "$beiseite"
        fehler "$archiv ist beschädigt und heißt jetzt $beiseite. Skript noch einmal starten, dann wird die Datei neu geladen"
        ;;
      *)
        fehler "$archiv lässt sich nicht lesen, obwohl die Prüfsumme stimmt"
        ;;
    esac
  else
    echo "Dateien im Archiv: $(wc -l < "listen/${typ}_inhalt.txt")"
  fi
  if [ -e "$ordner/.fertig" ]; then
    echo "$archiv ist schon entpackt"
  elif [ "$archiv_ok" = "1" ]; then
    mkdir -p "$ordner"
    if [ "$ganz" = "1" ]; then
      echo "Entpacke $archiv vollständig nach $ordner"
      tar -xf "$archiv" -C "$ordner"
      status=$?
    else
      awk -F'\t' -v tfs="$TFS" -v archive=1 "$ZUORDNUNG" "listen/${typ}_inhalt.txt" | cut -f5 > "listen/${typ}_auswahl.txt"
      eigene=$(wc -l < "listen/${typ}_auswahl.txt")
      awk -F'\t' '{p = $2; m = split(p, t, /[^A-Za-z0-9]+/); k = 0
                   for (i = 1; i <= m; i++) if (tolower(t[i]) ~ /^(control|controls|ctrl|input|zero|cycle0|c0|unselected|naive)$/) k = 1
                   if (!k && $1 < 20971520 && tolower(p) !~ /\.(fastq|fq|fasta|fa)(\.gz)?$/) k = 1
                   if (k) print p}' "listen/${typ}_inhalt.txt" >> "listen/${typ}_auswahl.txt"
      sort -u -o "listen/${typ}_auswahl.txt" "listen/${typ}_auswahl.txt"
      if [ "$eigene" -eq 0 ]; then
        fehler "in $archiv trägt keine Datei den Namen eines deiner TFs. Das Archiv wird deshalb vollständig entpackt"
        tar -xf "$archiv" -C "$ordner"
        status=$?
      else
        echo "Entpacke aus $archiv $eigene Dateien deiner TFs, dazu $(( $(wc -l < "listen/${typ}_auswahl.txt") - eigene )) kleine oder Kontroll-Dateien"
        tar -xf "$archiv" -C "$ordner" -T "listen/${typ}_auswahl.txt"
        status=$?
      fi
    fi
    if [ "$status" -eq 0 ]; then
      touch "$ordner/.fertig"
    else
      fehler "$archiv lässt sich nicht vollständig entpacken, obwohl das Archiv lesbar ist. Meist ist der Speicherplatz in $ZIEL voll"
    fi
  fi
  if [ -e "$ordner/.fertig" ]; then
    archiv_fehler="|"
    for _ in 1 2 3; do
      find "$ordner" -type f \( -iname "*.zip" -o -iname "*.tar" -o -iname "*.tgz" -o -iname "*.tar.gz" \) 2> /dev/null | sort > "listen/${typ}_archive.txt"
      neu=0
      while IFS= read -r innen; do
        case "$archiv_fehler" in *"|$innen|"*) continue ;; esac
        if [ -e "${innen}_entpackt/.fertig" ]; then
          continue
        fi
        neu=$((neu + 1))
        mkdir -p "${innen}_entpackt"
        if packe_aus "$innen" "${innen}_entpackt"; then
          touch "${innen}_entpackt/.fertig"
        else
          archiv_fehler="$archiv_fehler$innen|"
          fehler "$innen lässt sich nicht entpacken"
        fi
      done < "listen/${typ}_archive.txt"
      if [ "$neu" -eq 0 ]; then
        break
      fi
    done
    echo "Archive innerhalb des Pakets: $(wc -l < "listen/${typ}_archive.txt")"
  fi
  find "$ordner" -type f ! -name ".fertig" -printf '%s\t%p\n' 2> /dev/null | sort -t "$TAB" -k2 > "listen/${typ}_dateien.txt"
  awk -F'\t' -v tfs="$TFS" -v archive=0 "$ZUORDNUNG" "listen/${typ}_dateien.txt" | sort -t "$TAB" -k1,1 -k2,2 -k5,5 > "listen/${typ}_eigene.txt"
  echo "Entpackte Dateien: $(wc -l < "listen/${typ}_dateien.txt"), davon deinen TFs zugeordnet: $(wc -l < "listen/${typ}_eigene.txt")"
  fehlend=""
  for tf in $erwartet; do
    if ! awk -F'\t' -v t="$tf" '$1 == t {f = 1} END {exit !f}' "listen/${typ}_eigene.txt"; then
      fehlend="$fehlend $tf"
    fi
  done
  if [ -n "$fehlend" ]; then
    fehler "$typ: keine Datei gefunden für$fehlend. Laut Codebook-Übersicht gibt es dazu gelungene Experimente, die Zuordnung prüfe ich im Inventar"
  fi
done

schritt "Inventar 2: Inhalt der Archive (Ordner, Dateiendungen, alle Dateien)"
{
  for eintrag in "${TYPEN[@]}"; do
    IFS='|' read -r typ archiv ganz erwartet <<< "$eintrag"
    liste="listen/${typ}_inhalt.txt"
    echo "######## $typ: $archiv, $(wc -l < "$liste") Dateien, $(awk -F'\t' '{s += $1} END {printf "%.1f", s / 1073741824}' "$liste") GB"
    echo "### Ordner (Dateien, MB, Ordner)"
    awk -F'\t' '{p = $2; if (p ~ /\//) sub(/\/[^\/]*$/, "", p); else p = "."; n[p]++; s[p] += $1}
                END {for (p in n) printf "%d\t%.1f\t%s\n", n[p], s[p] / 1048576, p}' "$liste" | sort -t "$TAB" -k3
    echo "### Dateiendungen (Dateien, Endung)"
    awk -F'\t' '{d = $2; sub(/^.*\//, "", d); k = split(d, x, ".")
                 if (k == 1) e = "(ohne)"; else if (x[k] == "gz" && k > 2) e = x[k - 1] "." x[k]; else e = x[k]
                 n[e]++}
                END {for (e in n) printf "%d\t%s\n", n[e], e}' "$liste" | sort -nr | head -n 30
    echo "### Alle Dateien (Bytes, Pfad), höchstens 30000"
    head -n 30000 "$liste"
    echo
  done
} > inv_2_archive.txt

schritt "Inventar 3: Dateien deiner 30 TFs"
{
  echo "### Dateien je TF und Typ (Anzahl, MB). Zugeordnet über den TF-Namen im Pfad oder bei PBM über die Kennung aus den Metadaten"
  printf 'TF\tPBM\tPBM_MB\tSMS\tSMS_MB\tHTS\tHTS_MB\n'
  for tf in $TFS; do
    printf '%s' "$tf"
    for typ in PBM SMS HTS; do
      awk -F'\t' -v t="$tf" '$1 == t {n++; s += $4} END {printf "\t%d\t%.1f", n, s / 1048576}' "listen/${typ}_eigene.txt"
    done
    printf '\n'
  done
  for typ in PBM SMS HTS; do
    echo
    echo "### $typ: alle entpackten Dateien deiner TFs (TF, PBM-Kennung, Merkmale aus dem Pfad, Bytes, Pfad)"
    cat "listen/${typ}_eigene.txt"
  done
} > inv_3_eigene_tfs.txt

schritt "Inventar 4: Inhalt einzelner Dateien (erste Zeilen)"
{
  awk -F'\t' '$1 == "LEF1"' listen/PBM_eigene.txt | head -n 4
  awk -F'\t' '$1 == "GABPA"' listen/PBM_eigene.txt | head -n 2
  awk -F'\t' '$1 == "LEF1" || $1 == "MAX" || $1 == "YY1"' listen/SMS_eigene.txt | head -n 3
  awk -F'\t' '$1 == "GABPA"' listen/HTS_eigene.txt | head -n 2
  awk -F'\t' '$1 == "LEF1"' listen/HTS_eigene.txt | head -n 1
} | cut -f5 > listen/vorschau.txt
for typ in PBM SMS HTS; do
  if ! grep -q "MEX\.$typ/" listen/vorschau.txt; then
    head -n 2 "listen/${typ}_eigene.txt" | cut -f5 >> listen/vorschau.txt
  fi
done
while IFS= read -r datei; do
  lesbar "$datei" || continue
  echo "### $datei"
  case "$datei" in
    *MEX.PBM/*)
      echo "Zeilen: $(lies "$datei" | wc -l)"
      echo "Spalten je Zeile in den ersten 5000 Zeilen (Spalten: Zeilen):"
      lies "$datei" | head -n 5000 | awk -F'\t' '{c[NF]++} END {for (k in c) printf "  %s: %d\n", k, c[k]}'
      ;;
  esac
  echo "Erste acht Zeilen:"
  lies "$datei" | head -n 8 | cut -c1-500
  echo
done < listen/vorschau.txt > inv_4_inhalt.txt

schritt "Inventar 5: Kennzahlen der PBM-Dateien deiner TFs"
printf 'TF\tPBM\tMerkmale\tZeilen\tMittel\tStdAbw\tMin\tMax\tab_Mittel+4StdAbw\tIBIS_SD_Positive\tab_4\tSonde_Laenge\tLinker_Laenge\tLinker_verschieden\tcontrol\tflag\tDatei\n' > inv_5_pbm_kennzahlen.txt
MIT_SPALTEN=0
while IFS="$TAB" read -r tf id merkmale _ datei; do
  lesbar "$datei" || continue
  werte=$(lies "$datei" | awk -F'\t' '
    NR == 1 {for (i = 1; i <= NF; i++) {k = tolower($i); gsub(/^#|"|\r/, "", k); h[k] = i}
             ws = h["mean_signal_intensity"]; ps = h["pbm_sequence"]; ls = h["linker_sequence"]; cs = h["control"]; fs = h["flag"]
             next}
    !ws || !ps {n++; next}
    {sub(/\r$/, "")
     n++; w = $ws + 0; s += w; q += w * w; v[n] = w
     if (n == 1 || w < mn) mn = w
     if (n == 1 || w > mx) mx = w
     l = length($ps); if (lmin == "" || l < lmin) lmin = l; if (l > lmax) lmax = l
     if (ls) {k = length($ls); if (kmin == "" || k < kmin) kmin = k; if (k > kmax) kmax = k; lk[$ls] = 1}
     if (cs && $cs !~ /^(0|false|False|FALSE)?$/) c++
     if (fs && $fs !~ /^(0|false|False|FALSE)?$/) fl++}
    END {if (!ws || !ps) {printf "%d\tSpalten fehlen", n; exit}
         if (n == 0) {printf "0\tkeine Datenzeilen"; exit}
         m = s / n; var = q / n - m * m; if (var < 0) var = 0; sd = sqrt(var)
         for (i = 1; i <= n; i++) {if (v[i] >= m + 4 * sd) a++; if (v[i] >= 4) z++}
         ib = a; if (ib < 50) ib = 50; if (ib > 1300) ib = 1300; if (ib > n) ib = n
         nl = 0; for (x in lk) nl++
         printf "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%s-%s\t%s-%s\t%d\t%d\t%d", n, m, sd, mn, mx, a, ib, z, lmin, lmax, (ls ? kmin : "?"), (ls ? kmax : "?"), nl, c, fl}')
  case "$werte" in
    *"Spalten fehlen"*|*"keine Datenzeilen"*) ;;
    *) MIT_SPALTEN=$((MIT_SPALTEN + 1)) ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\n' "$tf" "$id" "$merkmale" "$werte" "$datei" >> inv_5_pbm_kennzahlen.txt
done < listen/PBM_eigene.txt
echo "PBM-Dateien mit den erwarteten Spalten: $MIT_SPALTEN"
if [ -s listen/PBM_eigene.txt ] && [ "$MIT_SPALTEN" -eq 0 ]; then
  fehler "in keiner PBM-Datei deiner TFs stehen die Spalten mean_signal_intensity und pbm_sequence. Das Format weicht von den IBIS-Dateien ab, siehe inv_4_inhalt.txt"
fi

schritt "Inventar 6: Kennzahlen der Read-Dateien deiner TFs (SMiLE-seq und HT-SELEX)"
printf 'TF\tTyp\tMerkmale\tFormat\tReads\tLaenge\thaeufigste_Laenge\tReads_mit_N\terster_Read\tDatei\n' > inv_6_reads_kennzahlen.txt
for typ in SMS HTS; do
  while IFS="$TAB" read -r tf _ merkmale _ datei; do
    lesbar "$datei" || continue
    werte=$(lies "$datei" | awk '
      function erfasse(s,   l) {
        n++; l = length(s)
        if (n == 1) erste = s
        if (lmin == "" || l < lmin) lmin = l
        if (l > lmax) lmax = l
        if (n <= 100000) h[l]++
        if (s ~ /N/) mitn++
      }
      NR == 1 {c = substr($0, 1, 1); fmt = (c == "@") ? "fastq" : ((c == ">") ? "fasta" : "zeilen")}
      {sub(/\r$/, "")}
      fmt == "fastq" && NR % 4 == 2 {erfasse($0)}
      fmt == "fasta" && substr($0, 1, 1) != ">" {erfasse($0)}
      fmt == "zeilen" {erfasse($0)}
      END {best = 0; haeufig = "-"
           for (l in h) if (h[l] > best) {best = h[l]; haeufig = l}
           printf "%s\t%d\t%s-%s\t%s\t%d\t%s", fmt, n, lmin, lmax, haeufig, mitn, substr(erste, 1, 80)}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$tf" "$typ" "$merkmale" "$werte" "$datei" >> inv_6_reads_kennzahlen.txt
  done < "listen/${typ}_eigene.txt"
done

schritt "Ergebnis"
echo "Dateien je TF (PBM/SMS/HTS):"
for tf in $TFS; do
  printf '%s=%s/%s/%s ' "$tf" \
    "$(awk -F'\t' -v t="$tf" '$1 == t' listen/PBM_eigene.txt | wc -l)" \
    "$(awk -F'\t' -v t="$tf" '$1 == t' listen/SMS_eigene.txt | wc -l)" \
    "$(awk -F'\t' -v t="$tf" '$1 == t' listen/HTS_eigene.txt | wc -l)"
done
printf '\n'
LISTEN="inv_0_md5.txt inv_1_metadaten.txt inv_2_archive.txt inv_3_eigene_tfs.txt inv_4_inhalt.txt inv_5_pbm_kennzahlen.txt inv_6_reads_kennzahlen.txt"
for datei in $LISTEN; do
  if [ -s "$datei" ]; then
    printf '%-28s %8s Zeilen\n' "$datei" "$(wc -l < "$datei")"
  else
    fehler "$datei ist leer"
  fi
done
if tar -czf codebook_inventar.tar.gz $LISTEN; then
  echo "Alle Listen zusammen: $(pwd)/codebook_inventar.tar.gz ($(du -h codebook_inventar.tar.gz | cut -f1))"
else
  fehler "die Sammeldatei codebook_inventar.tar.gz ließ sich nicht schreiben"
fi
if [ "$FEHLER" -gt 0 ]; then
  echo "Fertig mit $FEHLER Fehlermeldung(en), siehe die Zeilen mit 'Error' oben."
  exit 1
fi
echo "Fertig ohne Fehler. Bitte codebook_inventar.tar.gz hochladen."
