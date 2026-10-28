# Auswahl unterschiedlicher PWMs nach MEX-Qualitaetsrang

Version 1.1.0 liest die bereits gesammelten Original-PWMs, ordnet ihnen den
veroeffentlichten MEX-Rang zu und waehlt standardmaessig zehn PWMs je TF und
Assay. Dafuer wird die Tomtom-E-Grenze schrittweise gesenkt, sodass weniger
Motive zusammengefasst werden. Wenn noetig, hat die gewuenschte Anzahl Vorrang
vor Unterschiedlichkeit; eine Auffuellung wird ausdruecklich gekennzeichnet.
Zehn werden erreicht, sofern mindestens zehn verwendbare, zugelassene und
gerankte numerisch unterschiedliche Matrizen vorhanden sind. Es werden keine
neuen Motive erzeugt und keine Rangaussagen erfunden.

## Start auf deinem Cluster

Lege `PWM_Auswahl.zip` in
`/data/projects/SFB_A03/jan/AT_TFBS/human301` ab und fuehre dort aus:

```bash
cd /data/projects/SFB_A03/jan/AT_TFBS/human301
unzip PWM_Auswahl.zip
bash pwm_auswahl/submit_pwm_auswahl.sh
```

Der Launcher aktiviert `at_env` ueber
`~/miniforge3/etc/profile.d/conda.sh`. Das Python-Programm verwendet die
Standardbibliothek. Zusaetzlich ist die Tomtom-Executable aus MEME Suite
erforderlich. Gesucht wird zuerst im PATH, dann in
`~/miniforge3/envs/at_env/bin/`.

Falls Tomtom fehlt, endet die Vorpruefung vor der Slurm-Einreichung. Eine
Installation der getesteten Version direkt in `at_env` ist moeglich mit:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate at_env
conda install --freeze-installed -c conda-forge -c bioconda meme=5.5.8
bash pwm_auswahl/submit_pwm_auswahl.sh
```

Eine vorhandene andere Tomtom-Installation kann mit `--tomtom /pfad/tomtom`
verwendet werden. Version und Hash der ausgefuehrten Binary werden dokumentiert.

Nur pruefen, noch nichts einreichen:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate at_env
python pwm_auswahl/pwm_auswahl.py check
```

## Verwendete Pfade

| Zweck | Vorgabe |
|---|---|
| Bereits gesammelte PWMs | `/data/projects/SFB_A03/jan/AT_TFBS/human301/pwms/<TF>/<CHS oder GHTS>/` |
| Urspruengliche Zuordnungen fuer Alias-Abgleich | `/data/projects/SFB_A03/clara/ArchThalia-ML/Input_data/best_20_motif_CHS_GHTS/<CHS oder GHTS>/<TF>/` |
| Qualitaetsmetadaten | `pwm_auswahl/MEX_metadata_30TF.tsv` aus diesem Paket |
| Ergebnisse | `/data/projects/SFB_A03/jan/AT_TFBS/human301/pwms_ranked10/` |
| Slurm-Log | `/data/projects/SFB_A03/jan/AT_TFBS/human301/pwm_diverse_<JOBID>.log` |

Alle 30 TFs aus deiner Liste und beide Assays werden verarbeitet. Die Zuordnung
zu CHS oder GHTS stammt ausschliesslich aus dem jeweiligen Eingabeordner. Auch
Motive mit beispielsweise `HTS`, `AFS` oder `PBM` in ihrer Herkunfts-ID bleiben
in dem Ordner, dem sie zuvor zugewiesen wurden.

Die mitgelieferten Metadaten enthalten die 29.711 Zeilen fuer diese 30 TFs aus
deiner `complete_metadata_motif_15-06-2025.tsv`. Spalten und Werte wurden
unveraendert uebernommen; lediglich andere TFs wurden ausgelassen. Verglichen
und ausgewaehlt werden ausschliesslich Matrizen, die tatsaechlich unter
`pwms/<TF>/<Assay>` liegen.

## Exakter Ablauf

### 1. Dateien und Matrizen einlesen

Pro TF/Assay werden `.pwm`-Dateien rekursiv gelesen; Notebook-Checkpoints und
Mac-Metadaten werden ignoriert. Erwartet wird ein optionaler `>Motif-ID`-Header,
danach eine Position pro Zeile und vier Werte in der Reihenfolge A, C, G, T.
Mehrere Motive in einer Datei oder falsche Spaltenzahlen fuehren zu einem Fehler.

Numerisch identische Matrizen werden zusammengefasst. Der Vergleich verwendet
exakte Dezimalzahlen: Schreibweisen wie `1.0` und `1e0` sind gleich, waehrend
kleine echte Zahlenunterschiede erhalten bleiben. Die Originaldateien bleiben
an ihrem Ort unveraendert.

### 2. Originale MEX-Qualitaetsraenge zuordnen

Der Header wird mit `Motif` aus den MEX-Metadaten abgeglichen. Die Endungen
`.pwm`, `.ppm`, `.pcm` und `.pfm` werden dabei vereinheitlicht. Insbesondere
ChIPMunk-Motive stehen in deiner Tabelle oft mit `.pcm`.

Beim vorherigen Einsammeln konnten identische Matrizen unter verschiedenen
Motiv-IDs zusammenfallen. Deshalb liest das Programm auch die urspruenglichen
Zuordnungsordner noch einmal. Wenn eine dortige Matrix numerisch exakt einer
gesammelten Matrix entspricht, wird ihre ID als Alias dokumentiert. Ein Alias
aus einem anderen TF/Assay-Ordner wird dabei nicht hinzugefuegt.

Unter den passenden Alias-Eintraegen wird der kleinste positive `Overall rank`
verwendet, standardmaessig nur aus Eintraegen mit `Approved=yes`. Alle Alias-IDs,
Quellpfade und Metadateneintraege stehen in `aliases.tsv`. Der Original-Header
der spaeter kopierten Datei kann deshalb von der fuer den Rang verwendeten
Alias-ID abweichen; beide stehen separat in der Ergebnistabelle.

Ein fehlender Rang wird nicht geschaetzt. Solche Matrizen werden als
`excluded_missing_metadata` oder `excluded_missing_rank` dokumentiert und
nicht als Vertreter ausgewaehlt. Nicht freigegebene Metadateneintraege erhalten
`excluded_not_approved`; optional koennen sie zugelassen werden.

MEX-Raenge sind keine AP-Werte und keine absoluten Qualitaetsmasse. Der
Standard filtert nach `Approved=yes` und sortiert nach Rang; er setzt keine
zusaetzliche willkuerliche Rangobergrenze. Wer eine strengere Grenze will,
kann `--max-overall-rank N` setzen. Eine hohe Rangnummer wird nicht dadurch
zu einem guten Motiv, dass das Motiv anders aussieht. Die empirische Eignung
fuer dein CNN muss separat bewertet werden.

### 3. Vergleichsformat fuer Tomtom herstellen

Tomtom erwartet Wahrscheinlichkeitsmatrizen. Fuer deine beiden angehaengten
Beispiele passt die folgende Darstellung der Log-Odds:

`W[i,b] = ln((p[i,b] + 0.00001) / 0.25)`

Das Programm invertiert sie ausschliesslich fuer den Vergleich:

`p[i,b] = 0.25 * exp(W[i,b]) - 0.00001`

Die Spaltenreihenfolge ist A,C,G,T. Vor jeder Verwendung wird geprueft:

* keine NaN-Werte;
* keine rekonstruierten Wahrscheinlichkeiten unter `-1e-9`;
* Abweichung der Zeilensumme von 1 hoechstens `0.0021` (0,21 Prozent).

Winzige negative Rundungsreste werden auf null gesetzt und gueltige Zeilen auf
Summe 1 normiert. Die Vorgabe `5e-6` der ersten Version war fuer die gesamten
Exporte zu streng: In deinem Log wurden bei HOMER Summen von 0.999 und 1.001
und bei ExplaiNN kleinere Abweichungen gemeldet. Diese werden jetzt zugelassen.
Groessere Fehler werden weiterhin gemeldet, statt eine andere Logbasis
oder einen anderen Hintergrund zu erraten. Der angenommene vorgeschaltete
Pseudocount ist mit `--pwm-pseudocount` einstellbar.

Die Wahrscheinlichkeitsmatrizen stehen in `motifs_for_comparison.meme`.
Tomtom erhaelt `-motif-pseudo 0`, damit es keinen weiteren Pseudocount addiert.
`nsites=1` ist ein Formatwert und bei diesem Null-Pseudocount keine angenommene
biologische Stichprobengroesse. Motive erhalten kurze, eindeutige IDs wie
`M0001`; das vermeidet Probleme mit den langen Original-IDs und Gleichheitszeichen.

### 4. Motive mit dem Originalprogramm Tomtom vergleichen

Fuer jeden TF und Assay separat wird die gesamte Sammlung eindeutiger Matrizen
als Query- und Target-Datei eingesetzt, einschliesslich der Matrizen, die wegen
fehlender Metadaten oder Approval spaeter nicht als Vertreter infrage kommen.
Der statistische Referenzpool ist damit die dokumentierte vollstaendige
TF/Assay-Sammlung. Alias-IDs werden darin nicht als weitere Matrizen gezaehlt.

Die wesentlichen Tomtom-Optionen sind:

```text
-text -dist ed -evalue -thresh 1000000 -min-overlap 5 -motif-pseudo 0
```

`ed` ist die euklidische Distanz zwischen den vierdimensionalen
Wahrscheinlichkeitsvektoren ausgerichteter Motivpositionen. Tomtom sucht
Ausverschiebungen ohne innere Luecken und beide relativen DNA-Orientierungen.
Ein Reverse Complement kehrt die Positionsreihenfolge um und vertauscht
A mit T sowie C mit G. Nicht ueberlappende Positionen gehen in Tomtoms
vollstaendige Bewertung ein; `-incomplete-scores` wird nicht aktiviert.

Tomtom bestimmt die statistische Signifikanz eines Motivvergleichs mit seinem
publizierten Nullmodell. Das Skript berechnet keine eigenen p-Werte. Es
speichert Tomtoms p-, E- und q-Werte, Ueberlappung, Offset und Orientierung.
Der sehr hohe Ausgabe-Schwellenwert 1.000.000 dient dazu, auch schwache
Vergleiche zu protokollieren; er ist NICHT die Aehnlichkeitsschwelle.
Die Ausgabe wird auf Vollstaendigkeit aller gerichteten Vergleiche geprueft.

### 5. Wann gelten zwei PWMs als zu aehnlich?

Zuerst werden exakt gleiche Matrizen und exakte Reverse Complements als
aequivalent erkannt, auch wenn ein kurzes oder wenig informatives Motiv
keinen signifikanten Tomtom-E-Wert erzeugen sollte.

Fuer alle anderen Paare gilt pro gepruefter E-Grenze `E*`:

`min(E(A -> B), E(B -> A)) <= E*`

fuer mindestens einen von Tomtom gemeldeten Vergleich mit

`Overlap >= min(5, Laenge(A), Laenge(B))`.

MEME-ChIP verwendet `E <= 0.05` zur Gruppierung stark aehnlicher Motive.
Fuer deine gewuenschte groessere Auswahl startet diese Version standardmaessig
schon bei `E* = 0.0001` und senkt die Grenze bei Bedarf weiter. Kleinere
E-Grenzen erkennen WENIGER Paare als redundant; es duerfen dann auch
statistisch oder biologisch aehnliche Motivvarianten nebeneinander bleiben.
Die zusaetzliche Mindestueberlappung von fuenf Positionen orientiert
sich an RSAT matrix-clustering; fuer kuerzere Motive verwendet Tomtom die
Laenge des kuerzeren Motivs als effektives Minimum.

Tomtom-p-/E-Werte koennen zwischen A->B und B->A abweichen. Die Verwendung
beider Richtungen ist eine ausdrueckliche, staerker gegen Redundanz gerichtete
Anpassung dieser Pipeline. Fuer den gerichteten Vergleich eines Vertreters
gegen einen Kandidaten kann `--direction seed` verwendet werden.

Ein E-Wert ist kein prozentualer Sequenz- oder Matrixidentitaetswert und kein
garantierter Abstand biologischer Bindespezifitaeten. Er haengt auch von
Groesse und Zusammensetzung des Target-Pools ab. Der q-Wert ist dokumentiert,
wird aber nicht als zusaetzlicher Auswahlfilter verwendet.

### 6. Qualitaet und Unterschiedlichkeit zusammenfuehren

Die geeigneten Kandidaten werden nach aufsteigendem MEX-Rang sortiert.
Gleichstaende werden deterministisch ueber die Rang-ID, den Header und den
Dateipfad aufgeloest. Auf jeder Stufe wird dieselbe sortierte Liste erneut
von vorne ausgewertet.

Der erste Kandidat wird Vertreter der ersten Gruppe. Danach wird jeder
Kandidat mit allen bisherigen Vertretern verglichen. Ist er nach der obigen
Regel zu einem davon aehnlich, wird er diesem hoeher gerankten Vertreter
zugeordnet. Falls mehrere passen, entscheidet der erste, also am besten
gerankte Vertreter. Sonst eroeffnet der Kandidat eine neue Gruppe.

Es gibt kein transitives Zusammenlegen: Wenn A aehnlich zu B ist und B zu C,
aber A nicht zu C, koennen A und C beide Vertreter bleiben. Damit haengt die
Redundanzentscheidung unmittelbar an einem Vertreter.

Sind mindestens `top-k` Vertreter vorhanden, werden die ersten `top-k`
ausgewaehlt. Andernfalls wird die E-Grenze gesenkt. Die vorgegebenen Stufen
unterhalb des Startwerts sind `1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10,
1e-12, 1e-16, 1e-20, 1e-30, 1e-50, 1e-100, 1e-200, 1e-300`; uebersprungen
werden alle Stufen, die nicht kleiner als der Startwert sind. Gewaehlt wird die
erste gepruefte Stufe, die die Anzahl erreicht. Es wird nicht angenommen,
dass die Anzahl greedy gewaehlter Vertreter bei jeder Stufe monoton ansteigt.

Reicht auch das nicht, folgt eine ausdrueckliche `exact_only`-Stufe: Nur
exakte Matrixgleichheit und exakte Reverse Complements werden noch gefiltert.
Das ist kein E-Wert von null; die statistische Tomtom-Filterung ist auf dieser
Stufe ausgeschaltet. So werden auch numerisch als E=0 ausgegebene, aber nicht
identische Motive wieder zugelassen.

Falls dann noch weniger als zehn Vertreter uebrig sind, werden die am besten
gerankten restlichen Originalmatrizen aufgefuellt. Dabei koennen insbesondere
Reverse-Complement-Varianten bereits ausgewaehlter Matrizen hinzukommen.
Solche Eintraege stehen als `Selection_reason=rank_fill` in den Tabellen.
Es werden weiterhin nur numerisch unterschiedliche Eingabematrizen verwendet;
Dateien werden nicht dupliziert und keine PWMs neu erzeugt. Die finale Auswahl
wird nach MEX-Rang sortiert und von 1 bis `top-k` nummeriert.

Fehlen insgesamt ausreichend verwendbare, zugelassene und gerankte Matrizen,
meldet das Programm `INSUFFICIENT_RANKED` und `Minimum_met=False`. Fehlende
Metadaten werden auch fuer die Mindestanzahl nicht geraten. Mit
`--selection-mode fixed` kann das alte Verhalten mit fester Grenze und ohne
Auffuellung verwendet werden.

Das Verfahren uebernimmt die qualitaetsgeordnete Seed-Idee von MEME-ChIP.
Die Rangquelle MEX, die symmetrische Paarentscheidung, die Mindestueberlappung
und die adaptive Zielanzahl sind hier explizit dokumentierte Anpassungen. Der
zusaetzliche schwache Gruppen-Merge von MEME-ChIP wird nicht angewendet.

### 7. Originaldateien und Entscheidungstabellen ausgeben

Ausgewaehlte Dateien werden bytegleich aus `pwms` kopiert. Ihre Namen sind
kurz und sortierbar; ihr Inhalt einschliesslich Header bleibt original.
Die aus Log-Odds rekonstruierten Vergleichsmatrizen ersetzen diese Dateien
nicht. SHA-256-Pruefungen kontrollieren die Kopien.

## Ausgaben

Unter `human301/pwms_ranked10/` entstehen:

| Datei/Pfad | Inhalt |
|---|---|
| `summary.tsv` | Eine Zeile pro TF/Assay: Zahlen, wirksame E-Grenze, Rank_fill und Minimum_met |
| `summary.log` | Lesbares laufendes Gesamtprotokoll |
| `selected.tsv` | Alle ausgewaehlten PWMs mit Originalpfad, Rang, Rang-ID und Kopierziel |
| `run.json` | Parameter, Software-/Metadaten-Hashes, Versionen, Job-ID und Eingabepfade |
| `<TF>/<Assay>/ranking.tsv` | Entscheidung fuer jede eindeutige Matrix; bei Redundanz auch Vertreter und Aehnlichkeitsbeleg |
| `<TF>/<Assay>/aliases.tsv` | Zuordnung deduplizierter Matrizen zu originalen IDs und MEX-Metadaten |
| `<TF>/<Assay>/tomtom.tsv` | Alle gerichteten Tomtom-Vergleiche einschliesslich p/E/q, Offset und Orientierung |
| `<TF>/<Assay>/tomtom.stderr.log` | Tomtom-Meldungen und moegliche Fehler |
| `<TF>/<Assay>/motifs_for_comparison.meme` | Rekonstruierte Wahrscheinlichkeitsmatrizen mit kurzen IDs |
| `<TF>/<Assay>/selected/` | Bytegleiche Original-PWMs, maximal `top-k` |
| `<TF>/<Assay>/selection_policy.json` | Alle Grenzwertversuche und finale Auswahlregel |
| `<TF>/<Assay>/threshold_trials.tsv` | Vertreteranzahl auf jeder geprueften E-Stufe |
| `<TF>/<Assay>/completion.json` | Eingabe-Fingerprint und gepruefte Ergebnisdateien fuer Wiederaufnahme |

Die Zusammenfassungen werden nach jeder bearbeiteten TF/Assay-Gruppe
aktualisiert. Fehlende PWMs werden als `NO_PWM`, zu wenige gerankte Matrizen
als `INSUFFICIENT_RANKED`, gekennzeichnete Auffuellung als `OK_RANK_FILLED`
und Verarbeitungsfehler als `ERROR` ausgewiesen. `FEWER_THAN_REQUESTED` tritt
nur im festen Modus auf. Andere Gruppen werden bei einem Gruppenfehler weiter
bearbeitet; der Job endet dann mit Exitcode 1.

Ein erneuter Aufruf mit identischen Daten/Parametern prueft die vorhandenen
Ergebnisse und verwendet sie wieder. Fuer geaenderte Schwellenwerte oder eine
andere Auswahlzahl ist ein neuer `--output`-Pfad zu verwenden. Das verhindert,
dass Ergebnisse verschiedener Einstellungen vermischt werden.

## Beispiele fuer andere Einstellungen

Fuenfzehn statt zehn PWMs; eigener Ausgabeordner:

```bash
bash pwm_auswahl/submit_pwm_auswahl.sh --top-k 15 \
  --output /data/projects/SFB_A03/jan/AT_TFBS/human301/pwms_ranked15
```

Nur den gerichteten Seed-Vergleich verwenden:

```bash
bash pwm_auswahl/submit_pwm_auswahl.sh --direction seed \
  --output /data/projects/SFB_A03/jan/AT_TFBS/human301/pwms_ranked10_seed
```

Eine feste E-Grenze testen, ohne automatische Absenkung oder Auffuellung:

```bash
bash pwm_auswahl/submit_pwm_auswahl.sh --top-k 10 \
  --selection-mode fixed --evalue 1e-8 \
  --output /data/projects/SFB_A03/jan/AT_TFBS/human301/pwms_fixed_1e8
```

Eine selbst gewaehlte maximale Rangnummer begrenzt die zugelassenen Motive.
Der Parameter heisst `--max-overall-rank`; eine universell sinnvolle Zahl ist
aus den vorhandenen Raengen allein nicht ableitbar. Ein kleinerer E-Wert
als Aehnlichkeitsschwelle macht das Aussortieren strenger: Es werden dann
WENIGER Paare als redundant erkannt, also tendenziell MEHR Vertreter behalten.

## Slurm und Vorrang

Der Launcher reicht genau einen CPU-Job ein: 1 CPU, 4 GiB RAM, maximal zwei
Stunden, keine fest vorgegebene Partition und keine angeforderte GPU.

Wie beim Sammeljob werden die aktuell wartenden eigenen Jobs/Array-Tasks
voruebergehend auf Hold gesetzt. Der Auswahljob wird danach freigegeben und
kann den naechsten passenden freien User-Slot erhalten. Laufende Jobs laufen
weiter. Beim Start des Auswahljobs werden die gespeicherten Holds sofort
aufgehoben, noch vor den Analyse- und Abhaengigkeitspruefungen. Vorher bereits
gehaltene Jobs und die ArrayTaskThrottle werden nicht geaendert. Auch bestehende
Dependency-Ausdruecke bleiben erhalten. Freie Clusterressourcen und weitere
Scheduler-Regeln bleiben Voraussetzung fuer einen Start.

Wenn du den Auswahljob VOR seinem Start abbrichst oder der Einreichevorgang
hart unterbrochen wurde:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate at_env
python pwm_auswahl/pwm_auswahl.py release
```

Die Wiederherstellungsinformationen liegen in
`human301/pwm_diversity_priority.json`. Mit `--normal-queue` kannst du beim
Einreichen die Hold-basierte Bevorzugung abschalten.

## Gepruefte Beispiele und praktische Grenzen

Mit echtem Tomtom 5.5.8 und den beiden BATF2-Anhaengen wurden 19 eindeutige
CHS- und 20 GHTS-Matrizen verarbeitet. Version 1.1.0 waehlt je zehn PWMs aus:
CHS bei E=1e-16 (15 Vertreter), GHTS bei E=1e-10 (14 Vertreter). In beiden
Faellen war keine Auffuellung erforderlich. Die ausgewaehlten Dateien sind
numerisch verschieden und bytegleich zu ihren Originalen; die finalen Listen
sind nach aufsteigendem MEX-Rang sortiert. Das sind Ergebnisse fuer die Anhaenge,
noch keine Ergebnisse fuer die vollstaendige Sammlung auf deinem Cluster.

Zum Vergleich: Die alte feste Grenze E=0.05 liess in diesen Beispielen nur je
zwei Vertreter uebrig. Auch die neue Auswahl von zehn Matrizen bedeutet nicht,
dass zehn unabhaengige biologische Bindespezifitaeten vorhanden sind.

22 Checks deckten unter anderem die adaptive Grenzwertwahl, nicht monotone
greedy Vertreterzahlen, transparente Rang-Auffuellung, unzureichende
Metadaten, Rundungstoleranz, Reverse Complements, Ausgabe-Reihenfolge und
Slurm-Wiederfreigabe inklusive Rennsituationen und Einreichungsfehlern ab.
Der Slurm-Ablauf wurde simuliert; auf deinem Cluster wurde nichts eingereicht.

Die Qualitaetsquelle ist der veroeffentlichte MEX-Overall-Rang. Das Skript
berechnet keine AP auf deinen Sequenzen und lernt keine neuen Motive. Aus dem
Vorhandensein eines MEX-Rangs folgt nicht, dass dessen Datengrundlage auf einen
bestimmten Source-Assay beschraenkt war. Fuer eine strikt assaygetrennte
Benchmark-Auswahl muss die erlaubte Datenbasis des Rankings geprueft werden.

## Quellen

* Gupta et al. (2007), **Quantifying similarity between motifs**,
  Genome Biology 8:R24. DOI: https://doi.org/10.1186/gb-2007-8-2-r24
* Machanick & Bailey (2011), **MEME-ChIP: motif analysis of large DNA datasets**,
  Bioinformatics 27:1696-1697. DOI: https://doi.org/10.1093/bioinformatics/btr189
* MEME-ChIP-Dokumentation, Optionen `-group-thresh` und `-group-weak` sowie
  dokumentierter Seed-Gruppierungsablauf:
  https://meme-suite.org/meme/doc/meme-chip.html
  und https://meme-suite.org/meme/doc/examples/memechip_example_output_files/meme-chip.html?man_type=web
* Tomtom-Dokumentation, Distanzen, Mindestueberlappung, Pseudocounts und
  Reverse Complements: https://meme-suite.org/meme/doc/tomtom.html
* RSAT matrix-clustering (2017), publizierte Mindest-Alignmentbreite von 5:
  https://doi.org/10.1093/nar/gkx314
* Bioconda-MEME-Paket: https://bioconda.github.io/recipes/meme/README.html

Methoden und Abhaengigkeiten geprueft am 28.10.2026.
