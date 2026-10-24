# Human301: gemeinsame Skripte für alle Konfigurationen

Alle Experimente verwenden dieselben vier Python-Dateien. Für eine neue
Kombination vorhandener Einstellungen wird nur eine JSON-Datei unter `configs/`
angelegt. Weder beim Planen noch beim Trainieren werden Python-Dateien pro
Konfiguration erzeugt oder in Ergebnisordner kopiert.

| Datei | Inhalt |
|---|---|
| `model.py` | Modellbausteine, Modellkonfiguration und gesamter Vorwärtslauf |
| `data.py` | Caches, Datensätze, Sampling, Augmentation und DataLoader |
| `training.py` | Training, Inferenz, Metriken, Grafiken und standardisierter Ergebnisexport |
| `run_experiment.py` | Konfiguration laden, Aufgaben planen, starten und mit SLURM verwalten |

Zusammengehörige Funktionen stehen in beschrifteten Abschnitten dieser Dateien.
Die SLURM-Dateien und die einzelnen Shell-Starter bleiben erhalten; sie rufen alle
dieselben Python-Dateien auf.

Die vier Dateien ersetzen den bisherigen Ordner `ablation/` und die Datei
`submit_jobs.py`. Beim Übernehmen werden Code und mitgelieferte Konfigurationen
gemeinsam ersetzt. Dein ausgefülltes `configs/caches.csv` bleibt verwendbar.
Bestehende Ergebnisdateien werden durch dieses Paket nicht verschoben oder
gelöscht.

## Konfigurationen

| JSON-Datei unter `configs/` | Eingangskernel | Filter je Kernel | Zusätzliches Channel-Dropout |
|---|---|---|---|
| `00_basic.json` | 9 / 12 | 32 / 32 | 0 |
| `A01_kernel_LegNetMax_3_6_9_12.json` | 3 / 6 / 9 / 12 | 16 / 16 / 16 / 16 | 0 |
| `A02_kernel_7_11_15_21.json` | 7 / 11 / 15 / 21 | 16 / 16 / 16 / 16 | 0 |
| `A03.json` | 3 / 6 / 9 / 12 | 32 / 32 / 32 / 32 | 0 |
| `A04.json` | 3 / 6 / 9 / 12 | 32 / 32 / 32 / 32 | 0,1 |

Die ursprünglichen drei Konfigurationen unterscheiden sich nur im Namen und in den
Eingangskerneln. A01 verwendet die Eingangskernel von LegNet-Max und weiterhin
deinen bisherigen Encoder, Bottleneck und Decoder.

A03 und A04 basieren auf A01 und verbreitern den Embedder auf 128 Eingangskanäle,
also 32 pro Kernel. Encoder-Kanäle 80/96/128, Bottleneck und übrige Einstellungen
bleiben erhalten. Das vorhandene Bottleneck-Dropout `dropout: 0.1` bleibt in
beiden Varianten aktiv.

`channel_dropout` ist eine zusätzliche, getrennte Einstellung. A04 verwendet
`nn.Dropout1d(0.1)` direkt auf dem Embedder-Ausgang vor dem ersten Encoder-Block
und vor der ersten Skip-Verbindung. Dabei wird je Beispiel ein vollständiger
Kanal über alle Sequenzpositionen mit Wahrscheinlichkeit 0,1 deaktiviert; die
übrigen Kanäle werden mit 1/0,9 skaliert. Bei der Auswertung ist es deaktiviert.
A03 und die bisherigen Varianten verwenden dort die Identitätsfunktion, die
keine Zufallszahlen verbraucht. Die beiden neuen Varianten verwenden dieselben
gemeinsamen Python-Dateien.

Der Ablauf ist für jede Konfiguration gleich: `workflow` wählt den gemeinsamen
Ablauf, `data` den Datenzugriff, `predictor` das Modell, `fitting` das Training und
`inference` die Auswertung. Die jeweiligen `params` enthalten die Einstellungen.
Die Einträge verweisen auf Funktionen in den vier gemeinsamen Dateien, zum
Beispiel `model:maa` und `training:baseline`.

Für eine weitere Kernelkombination kopierst du eine JSON-Datei, vergibst unter
`name` deine gewünschte Bezeichnung und änderst
`predictor.params.motif_kernels`. Kanalzahlen, Dropout und Readout stehen ebenfalls
unter `predictor.params`; Epochen, Batchgröße, Lernrate und Sampling unter
`fitting.params`. Seeds stehen unter `seeds`.

Einstellungen wählen bereits implementierte Berechnungen aus. Eine neue
mathematische Methode wird einmal in der passenden gemeinsamen Datei ergänzt
und anschließend über die Konfiguration gewählt.

## Referenz und Training

Basic basiert weiterhin auf `hum301_motif64(1).py`, Version
`4.1.0-human301-full-maa-motif64`, mit Center-Readout. Alle JSON-Dateien behalten
Seed 13, maximal 50 Epochen, Patience 10, Batchgröße 88, zehn Negative pro
Positivem, AdamW und die bisherige Crop-/RC-Auswertung bei.

Der optionale `local-logsumexp`-Readout aus `hum301_motif64_local(7).py` ist im
gemeinsamen Modell verfügbar. Er wird über
`predictor.params.classification_readout` ausgewählt. Alle mitgelieferten
Konfigurationen verwenden weiterhin `center`.

Die Funktionen für Regression und FE-Exporte bleiben im gemeinsamen Code
enthalten. Der mitgelieferte Human301-Ablauf wertet die binäre Klassifikation auf
beiden Assays aus. Zusätzliche im Experimentplan beschriebene Verfahren müssen
weiterhin implementiert werden; ihre Implementierung entsteht nicht automatisch
aus einem Konfigurationseintrag.

## Start

Aktiviere deine bisherige Trainingsumgebung und wechsle in `hum301/big_ablation/`.

```bash
python run_experiment.py list

python run_experiment.py manifest \
  --cache-root /PFAD/ZU/DEINEN/CACHES \
  --output configs/caches.csv

bash submit.sh configs/caches.csv
```

Ein vorhandenes `configs/caches.csv` kann direkt weiterverwendet werden. Die
Datei ordnet jeder Kombination aus TF und Assay ihren Cache zu. Mehrdeutige
automatische Zuordnungen werden abgebrochen. Alternativ dient
`configs/caches.example.csv` als Vorlage.

`submit.sh` arbeitet Basic, A01 und A02 mit insgesamt maximal acht offenen
Trainingsjobs ab. Freie Plätze werden mit der nächsten Konfiguration aufgefüllt.
Die Reihenfolge innerhalb jeder Konfiguration steht unverändert und ohne
Zeitangaben in `configs/training_priority.csv`.

Die bisherigen Einzelstarter bleiben nutzbar:

```bash
bash submit_00_basic.sh configs/caches.csv
bash submit_A01_kernel_LegNetMax_3_6_9_12.sh configs/caches.csv
bash submit_A02_kernel_7_11_15_21.sh configs/caches.csv
```

Für A03 und A04 gibt es zwei weitere Einzelstarter. Jeder startet mit dem
vollständigen Manifest der 30 TFs genau 60 Trainingsläufe: 30 TFs × zwei
Trainings-Assays × Seed 13. Jeder Checkpoint wird auf beiden Test-Assays
ausgewertet. Die Einzelstarter werden für getrennte Durchläufe nacheinander
verwendet; die gemeinsame Sperre verhindert zwei parallele Launcher im Projekt.

```bash
bash submit_A03.sh configs/caches.csv
bash submit_A04.sh configs/caches.csv
```

Um beide automatisch in einem gemeinsamen Durchlauf mit insgesamt acht Plätzen
abzuarbeiten, verwendet derselbe Starter beide Konfigurationen:

```bash
python run_experiment.py submit start \
  --manifest configs/caches.csv --config A03 A04
```

Das reicht 120 Trainingsläufe in der Reihenfolge A03 → A04 ein. Freie Plätze
werden wie bisher mit der nächsten Konfiguration aufgefüllt. `submit.sh` und
die bisherigen Einzelstarter wählen weiterhin ausschließlich die bisherigen
drei Konfigurationen aus.

Status und Fortsetzen werden nun ebenfalls vom gemeinsamen Starter übernommen:

```bash
python run_experiment.py submit status plans/batch_XXXXXXXX
python run_experiment.py submit resume plans/batch_XXXXXXXX
```

Ein einzelner Versuch kann auch direkt ausgeführt werden:

```bash
python run_experiment.py run \
  --config 00_basic --manifest configs/caches.csv \
  --tf CREB3L3 --train-assay CHIP
```

## Ergebnisse

Die Ausgaben liegen unter `results/<Konfiguration>/`:

| Pfad | Inhalt |
|---|---|
| `config.json` | Experimentkonfiguration |
| `versions.txt` | Namen, Versionen und gemeinsame Codekennung der vier Skripte |
| `evaluation.md` | Zusammenfassung der Konfiguration |
| `runs/<TF>/train_<Assay>/seed_13/train.log` | Protokoll eines Trainings |
| `runs/<TF>/train_<Assay>/seed_13/run.json` | Status und Laufmetadaten |
| `runs/<TF>/train_<Assay>/seed_13/training/` | Checkpoints, Verlauf und Validierung |
| `runs/<TF>/train_<Assay>/seed_13/evaluations/standard/<Assay>/` | Einzelne Testauswertung |

Unter dem Seed liegt keine weitere Lauf-ID-Ebene. Eine vorhandene Kombination
aus Konfiguration, TF, Trainings-Assay und Seed wird nicht überschrieben.
Eine interne Laufkennung bleibt für die Zuordnung der CSV-Zeilen erhalten.

Die standardisierte CSV liegt unter
`results/Results_all/<Konfiguration>.csv`. CSV und `evaluation.md` werden erzeugt,
sobald die Jobs der betreffenden Konfiguration beendet sind. Einzelne
Testauswertungen stehen schon nach dem jeweiligen Training zur Verfügung.
Die abschließende gemeinsame Sammlung aktualisiert die Referenzvergleiche.

Ein Zwischenstand lässt sich jederzeit mit demselben gemeinsamen Code sammeln:

```bash
python run_experiment.py collect --config 00_basic
```

Das CSV-Schema bleibt unverändert; Einzelheiten stehen in `SCHEMA.md`.

## Numerischen Vergleich wiederholen

Die Referenzdatei wird nicht erneut mitgeliefert. Mit deiner vorhandenen Datei
lässt sich der Basic-Vergleich erneut ausführen:

```bash
python run_experiment.py verify-basic \
  --reference /PFAD/ZU/hum301_motif64.py
```

Dieser Vergleich verwendet synthetische Daten auf CPU. Er prüft unter anderem
Initialisierung, Vorhersagen, Crop-/RC-Auswertung, Sampling und zwei vollständige
Trainingsepochen. Ein vollständiger wissenschaftlicher Lauf auf deinen TF-Daten
und GPU-/ROCm-Reproduzierbarkeit werden dadurch nicht nachgewiesen.

`VALIDATION.md` dokumentiert die beim Umbau ausgeführten Prüfungen.
