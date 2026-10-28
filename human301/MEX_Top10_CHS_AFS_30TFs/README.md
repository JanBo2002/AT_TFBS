# MEX-PWMs: zehn pro TF und Quellassay

Stand: 28.10.2026. Das Paket enthält 600 PWMs für 30 menschliche TFs: jeweils zehn aus CHS und zehn aus AFS/GHT-SELEX pro TF.

## Herkunft und Kennzeichnung

- 565 Dateien sind unveränderte veröffentlichte Original-PWMs aus dem MEX-ArChIPelago-Export: 266 aus CHS und 299 aus AFS.
- 35 Dateien sind eigene Umrechnungen veröffentlichter MEX-Rohmatrizen: 34 aus CHS und eine aus AFS.
- Alle Ergänzungen tragen den Dateinamensuffix __SELF_CONVERTED.pwm. Die Originaldateinamen bleiben unverändert.
- In selection.tsv und motifs.json unterscheidet PWM_origin die Werte published_original und self_converted. conversion_id und raw_matrix_path dokumentieren die Umrechnung.
- Die 565 Originaldateien sind bytegleich zur vorherigen Fassung. Bei ihnen wurden weder Header noch Matrixgewichte verändert.

Die zusätzlichen Motive wurden nicht neu entdeckt. Ihre Motiv-IDs stammen aus der zuvor ausgewählten, nach Quellassay-AUPRC geordneten Top-10-Liste. Wiederhergestellt wurden nur die logarithmierten Gewichte aus den veröffentlichten PCM/PPM-Dateien mit derselben zuvor dokumentierten eigenen Umrechnung. Die 35 Original-Rohmatrizen sind unter raw_matrices/ enthalten.

## Auswahl und Reihenfolge

Für CHS gilt Approved=yes und Train data=CHS. Für AFS gilt Approved=yes und Train data=AFS.Lys, AFS.IVT oder AFS.GFPIVT. Klassisches HT-SELEX, PBM, SMS und der jeweils andere Quellassay wurden zum Auffüllen nicht verwendet.

Alle 565 bisher enthaltenen Original-PWMs bleiben ausgewählt. Nur Gruppen mit weniger als zehn Originaldateien wurden ergänzt. Aus den noch nicht enthaltenen Motiv-IDs der früheren Top-10-Auswahl wurden die höchstplatzierten genommen. Die Zahl zehn ist eine festgelegte Auswahlgröße.

selection_index führt innerhalb jeder TF/Assay-Gruppe zuerst die Originaldateien und danach die Ergänzungen. Er ist deshalb kein gemeinsamer AUPRC-Rang aller Dateien. rank_among_all_source_candidates und previous_selection_index behalten die vorherige Ranginformation; source_only_mean_AUPRC enthält den zugehörigen MEX-Wert.

Die frühere Rangfolge verwendete ausschließlich offizielle MEX-Quellassay-Werte auf freigegebenen Val-Datensätzen: chipseq_pwmeval_PR für CHS beziehungsweise affiseq_*_pwmeval_PR für AFS. Mehrere Peaksets/Zyklen wurden innerhalb eines Experiments gemittelt, dann Experimente innerhalb eines Subtyps, dann die vorhandenen AFS-Subtypen gleichgewichtet. Gleiche Scores wurden anhand der ursprünglichen Motiv-ID aufgelöst. Overall rank wurde nicht zur Auswahl oder zum Auflösen von Gleichständen verwendet.

Die AUPRC-Werte der 35 Ergänzungen wurden erneut aus dem vollständigen MEX-Metrikarchiv ausgelesen und gegen die freigegebenen Val-Datensatzidentitäten abgeglichen. Das vollständige Archivmitglied wurde einschließlich CRC geprüft. Diese Werte bewerten die MEX-Motivmodelle, nicht das eigene CNN und nicht eine erneute Evaluation der hier umgerechneten Log-odds.

## Eigene Umrechnung der 35 Ergänzungen

Alphabet und Spaltenreihenfolge: A,C,G,T. Eine Position pro Zeile. Hintergrund: 0.25 pro Base. Natürlicher Logarithmus. epsilon=0.0001.

1. f = raw / Zeilensumme(raw)
2. p = (f + epsilon) / (1 + 4*epsilon)
3. w = ln(p / 0.25)

Diese Formel gilt für die 33 PPMs und die zwei zunächst normalisierten PCMs der Ergänzungen. Sie ist die zuvor verwendete eigene Aufbereitung. Es wird nicht behauptet, dass die Autoren alle Original-PWMs auf dieselbe Weise erzeugten oder dass diese 35 Gewichte einem nicht gefundenen offiziellen Log-odds-Export entsprechen.

Die Originaldateien behalten ihre veröffentlichte Skala. Die Ergänzungen verwenden die oben dokumentierte Glättung. Dieser Unterschied bleibt ausdrücklich sichtbar. Es wurden keine Motive gekürzt, gepolstert, auf eine Kernelbreite zugeschnitten oder an die Varianz zufälliger Filter angepasst.

## Dateien

- CHS/<TF>/*.pwm und AFS/<TF>/*.pwm: jeweils genau zehn PWMs, einschließlich der gekennzeichneten Ergänzungen.
- selection.tsv: Herkunft, Ranginformation, Quellassay-AUPRC, Motiv-ID, Quellexperiment, Breite, Dateipfade und SHA256 aller 600 PWMs.
- coverage.tsv: verfügbare Originaldateien sowie ausgewählte Originale und eigene Umrechnungen pro TF/Assay.
- motifs.json: dieselben Gewichte als numerische Matrizen der Form (Motivbreite,4), mit Herkunftskennzeichnung. Die .pwm-Dateien sind für den genauen Dateiinhalt maßgeblich.
- raw_matrices/: unveränderte veröffentlichte PCM/PPM-Dateien für die 35 Ergänzungen.
- conversion.json: vollständige eigene Umrechnung und Prüfprotokoll der erneut ausgelesenen Benchmarkwerte.
- selected_source_AUPRC_metrics.tsv: die für alle 600 ausgewählten Motiv-IDs verwendeten Quellassay-AUPRC-Messwerte.
- previous_selection_unavailable.tsv: die 97 ursprünglich ausgewählten Motiv-IDs ohne Original-PWM im geprüften Export; zusätzliche Spalten zeigen, welche jetzt als eigene Umrechnung enthalten sind. Die fehlende Originaldatei wird dadurch nicht zu einer verfügbaren Originaldatei.
- sample_byte_checks.tsv: der bestehende Bytevergleich der 40 Dateien aus den beiden BATF2-Beispielarchiven.
- sources.tsv: veröffentlichte Quellen und Prüfsummen.
- original_export_README.txt: unveränderte README der Autoren, die sich auf deren Originalexport bezieht.

## Einordnung

Das Fehlen einer Original-PWM im begrenzten Top-20-Export ist kein Beleg für die Ablehnung eines Motivs. Alle ergänzten Motiv-IDs sind in den verwendeten MEX-Metadaten Approved=yes. Warum jedes einzelne nicht im Originalexport enthalten ist, wurde nicht als nachgewiesener Ausschlussgrund interpretiert.

Die Auswahl ist kein Nachweis von zehn unabhängigen Bindemodi. Motivähnlichkeit und Redundanz wurden nicht zusätzlich gefiltert. Eine Verbesserung durch das Auffüllen muss im Modellvergleich geprüft werden.

Die MEX-Freigabe Approved=yes berücksichtigt teilweise plattformübergreifende Konsistenz. Die MEX-eigenen Datenaufteilungen wurden nicht auf Sequenzüberlappung mit deinem konkreten IBIS-Train/Test-Split geprüft.

Quelle der Original-PWMs: https://zenodo.org/records/15120481
Quelle der Rohmatrizen, Metadaten und Benchmarkwerte: https://zenodo.org/records/15667805
Dokumentation der Autoren: https://github.com/autosome-ru/MEX-ArChIPelago
