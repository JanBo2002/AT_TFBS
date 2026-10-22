# Human301: big_ablation

Dieses Paket wird unter `hum301/big_ablation/` abgelegt. Es verwendet die bereits
erzeugten Human301-NPZ-Caches und verändert sie nicht. Alle benötigten Python-
Module einschließlich der vorhandenen Cache-Hilfsfunktionen sind enthalten.

## Die drei Konfigurationen

| Datei unter `configs/` | Eingangskernel | Filter je Zweig |
|---|---|---|
| `00_basic.json` | 9 / 12 bp | 32 / 32 |
| `A01_kernel_LegNetMax_3_6_9_12.json` | 3 / 6 / 9 / 12 bp | 16 / 16 / 16 / 16 |
| `A02_kernel_7_11_15_21.json` | 7 / 11 / 15 / 21 bp | 16 / 16 / 16 / 16 |

Die Konfigurationen unterscheiden sich ausschließlich in Name und Eingangskerneln.
A01 übernimmt die Eingangskernel von LegNet-Max; das übrige Modell bleibt die
bisherige MAA-Architektur. Es ist keine vollständige LegNet-Max-Implementierung.

Die Referenz für `00_basic` ist die hochgeladene `hum301_motif64(1).py`, Version
`4.1.0-human301-full-maa-motif64`: Center-Readout, 256 bp aus 301 bp, Encoder
80/96/128, unveränderter Fünf-Block-Bottleneck und Decoder. Auch die vorhandene
untrainierte FE-Regressionsausgabe bleibt für numerische Vergleichbarkeit erhalten.

Einstellungen der ersten drei Versuche: Seed **13**, beide Quell-Assays, 50
maximale Epochen, Patience 10, Batch 88, zehn Negative pro Positivem, pos_weight 10,
AdamW lr 3e-4/wd 1e-4, Cosine bis 1e-6, RC-Augmentation 0.5, AMP auf GPU. Die
Validierung und der Test mitteln wie bisher Wahrscheinlichkeiten über Crop-Starts
0/22/23/45 und beide Stränge. Checkpoint-Auswahl: Quell-Val-AUPRC; Schwelle:
Quell-Val-MCC. Sämtliche Einstellungen stehen in den Konfigurationsdateien und den
aufgelösten Laufkonfigurationen.

## Start

Zuerst die bestehende Trainingsumgebung aktivieren, in der bereits die bisherigen
Skripte liefen. Benötigt werden Python >=3.10, PyTorch und die bisherigen
NumPy/SciPy/scikit-learn/Matplotlib-Abhängigkeiten. CUDA-/ROCm-PyTorch beibehalten.

```bash
cd hum301/big_ablation
python run_experiment.py list
python run_experiment.py verify-basic
```

Die mitgelieferte `verification.json` dokumentiert die hier ausgeführten
CPU-Regressionstests. `verify-basic` führt sie in deiner Umgebung erneut aus,
einschließlich zweier vollständiger Trainings-Epochen auf einem erzeugten kleinen
Testdatensatz. Das ist ein Softwarevergleich und ersetzt den erneuten Lauf auf
deinen echten TF-Daten nicht.

Anschließend einmal die vorhandenen Caches zuordnen:

```bash
python run_experiment.py manifest \
  --cache-root /PFAD/ZU/DEN/VORHANDENEN/CACHES \
  --output configs/caches.csv
```

Die Erkennung verwendet erkennbare TF-/Assay-Paare in Cache-Pfaden oder den
gespeicherten Quelldateipfaden, z. B. `CREB3L3/CHIP/dataset.npz`. Eine mehrdeutige
Zuordnung wird abgebrochen. Alternativ `configs/caches.example.csv` als Vorlage
verwenden und **eine Zeile je TF und Assay** eintragen. Relative Cache-Pfade werden
relativ zur Manifestdatei interpretiert. `caches.example.csv` enthält Platzhalter
und ist kein startfertiges Manifest.

Alle drei Konfigurationen auf allen TFs im Manifest starten:

```bash
bash submit.sh configs/caches.csv
```

Der gemeinsame Launcher arbeitet die Konfigurationen in dieser Reihenfolge ab:
`00_basic` → `A01_kernel_LegNetMax_3_6_9_12` → `A02_kernel_7_11_15_21`.
Er verwaltet **acht Plätze insgesamt**. Zuerst füllt Basic die Plätze; solange
noch Basic-Aufgaben einzureichen sind, haben diese Vorrang. Sobald alle Basic-
Aufgaben eingereicht sind und weniger als acht davon offen sind, füllt A01 die
freien Plätze. Das Gleiche gilt anschließend für A02. Beispiel: fünf verbleibende
Basic-Jobs und drei A01-Jobs. Die nächste Konfiguration wartet also nicht auf den
letzten langen Job der vorherigen.

Ein Job trainiert genau ein Modell für eine Konfiguration/TF/Quell-Assay/Seed-
Kombination und testet diesen Checkpoint auf beiden Assays. Bei 30 TFs und einem
Seed sind es 60 Trainingsjobs je Konfiguration, insgesamt 180. Für die globale
Grenze zählen auch wartende und noch aufräumende Jobs als belegt. Freie Plätze
werden alle 60 Sekunden geprüft. Tatsächlich gleichzeitig laufende Jobs hängen
von den verfügbaren Cluster-Ressourcen ab.

Jede Konfiguration hat außerdem ein eigenes Submit-Skript für einen separaten
Durchlauf:

```bash
bash submit_00_basic.sh configs/caches.csv
bash submit_A01_kernel_LegNetMax_3_6_9_12.sh configs/caches.csv
bash submit_A02_kernel_7_11_15_21.sh configs/caches.csv
```

Für die automatische Folge **nur `submit.sh`** starten. Die drei Einzelbefehle
sind Alternativen und jeweils auf acht Plätze begrenzt. Eine gemeinsame Sperre
verhindert einen zweiten gleichzeitigen Launcher im selben Projekt.

Die vollständige vorgegebene Reihenfolge steht in
`configs/training_priority.csv`: **60 Einträge, ohne Zeitangaben**. Sie wird in
jeder Konfiguration erneut verwendet, von `CTCF/CHS` bis `ZFP3/GHTS`. `CHS` wird
intern wie bisher `CHIP` zugeordnet. Bei mehreren Seeds stehen deren Wiederholungen
direkt hinter dem jeweiligen TF/Assay-Paar. Neue, noch nicht gelistete Paare folgen
am Ende in Manifest-Reihenfolge. Gruppierte Multi-TF-Planner behalten ihre eigene
Reihenfolge.

Die Einreichung erfolgt als Einzeljobs mit einer Start-Abhängigkeit zu früheren
offenen Jobs (`after`). Dadurch können die Trainings parallel
laufen, während ein späterer Job keinen früheren wartenden Job überholt. Die
Priorität bezieht sich auf den SLURM-Jobstart. Die Initialisierung innerhalb des
Python-Prozesses gehört zum jeweiligen Job. Fehlgeschlagene oder abgebrochene
Trainings werden protokolliert und nicht automatisch wiederholt.

Der Launcher läuft nach dem Start im Hintergrund auf dem Login-Knoten weiter.
Er gibt seinen PID und den erzeugten Batch-Ordner aus. Darin werden für jede
Konfiguration `tasks.csv`, eine Kopie der Prioritätsliste, `state.json`,
`launcher.log` und die SLURM-Ausgaben gespeichert. `state.json` enthält die
Job-IDs und ermöglicht das Fortsetzen ohne erneutes Einreichen bekannter Jobs:

```bash
python submit_jobs.py status plans/batch_XXXXXXXX
tail -f plans/batch_XXXXXXXX/launcher.log
# Nach einem gestoppten Launcher oder einem Neustart des Login-Knotens:
python submit_jobs.py resume plans/batch_XXXXXXXX
```

`batch_XXXXXXXX` durch den ausgegebenen Ordner ersetzen. Ein Terminal-Disconnect
beendet den abgekoppelten Prozess nicht. Nach einem Neustart des Login-Knotens
muss `resume` aufgerufen werden. Bereits eingereichte SLURM-Jobs bleiben dabei
erhalten. Ein unklarer Jobstatus gibt keinen Platz frei; der Launcher wartet auf
SLURM/Accounting. Bei einer verlorenen Submit-Bestätigung wird der Job anhand
seines eindeutigen Namens gesucht. Ist das nicht eindeutig möglich, pausiert der
Launcher und schreibt den Grund ins Protokoll.

Nach dem Ende aller Trainings wird einmal `collect_all.sbatch` als CPU-Job
eingereicht. Dieser erzeugt die gemeinsamen CSVs und Auswertungsdateien auch dann,
wenn einzelne Trainings fehlgeschlagen sind. Der Batch-Status enthält zusätzlich
die SLURM-Fehler, auch wenn ein Job vor dem Start des Python-Trainings scheiterte.

Partition, Account und QoS sind absichtlich nicht festgelegt. Falls dein Cluster
sie benötigt, lassen sich Ressourcen-/Account-Optionen in Langform anhängen,
beispielsweise `bash submit.sh configs/caches.csv --account=... --qos=...`.
Der Account wird auch an den CPU-Sammeljob weitergegeben; dessen Partition/QoS
und weitere standortspezifische CPU-Einstellungen gehören in `collect_all.sbatch`.
Ressourcen der Trainingsjobs stehen in `train_task.sbatch` und lassen sich ebenfalls
überschreiben, z. B. mit `--mem=40G`. Array-/Dependency-Optionen sind dem Launcher
vorbehalten. Umgebungsvariablen mit Präfix `SBATCH_` werden beim Einreichen entfernt,
damit sie die feste Grenze und Reihenfolge nicht übersteuern. Der Python-Pfad
lässt sich über `ABLATION_PYTHON` setzen; ansonsten wird `python3` aus der aktivierten
Umgebung verwendet. Für Launcher und Fortsetzen werden `sbatch`, `squeue` und
`sacct` einschließlich funktionierendem Job-Accounting benötigt.

Manuell beziehungsweise ohne SLURM:

```bash
python run_experiment.py run --all-configs --manifest configs/caches.csv
```

Dieser Aufruf arbeitet die Tasks nacheinander ab. Für einen einzelnen Versuch:

```bash
python run_experiment.py run \
  --config 00_basic \
  --manifest configs/caches.csv \
  --tf CREB3L3 \
  --train-assay CHIP
```

Andere Seeds werden in den Konfigurationen unter `seeds` eingetragen oder beim
Start mit `--seed 1 2 3` ausgewählt. `--workers` und `--device` sind ausdrücklich
protokollierte Laufzeit-Overrides. Für direkte Vergleiche dieselben Einstellungen
wie beim früheren Lauf verwenden. Auch beim sequenziellen Start folgt die
TF-/Assay-Reihenfolge der gespeicherten Prioritätsliste. Diese bestimmt nur die
Job-Reihenfolge; Modell, Trainingsdaten, Sampling und Hyperparameter bleiben
unverändert. Für eine andere Liste gibt es beim Python-Planer `--priority-file`.

## Ausgaben

| Pfad relativ zu `big_ablation/` | Inhalt |
|---|---|
| `configs/` | Experimentdefinitionen, Cache-Zuordnung und `training_priority.csv` |
| `results/00_basic/` | Detaillierte Basic-Ergebnisse |
| `results/A01_kernel_LegNetMax_3_6_9_12/` | Detaillierte A01-Ergebnisse |
| `results/A02_kernel_7_11_15_21/` | Detaillierte A02-Ergebnisse |
| `results/Results_all/<Experimentname>.csv` | Einheitliche CSV pro Experiment |
| `results/Results_all/schema.json` | Fester CSV-Header und Schemaversion |
| `plans/` | Tasktabellen je Konfiguration, Prioritätskopie, Launcher-Status und SLURM-Protokolle |

Jeder Experimentordner enthält eine eingefrorene `config.json`, eine automatisch
erzeugte `evaluation.md`, den verwendeten Python-Code unter `sources/<code_hash>/`
und die einzelnen Läufe:

```text
results/<Experiment>/runs/<TF>/train_<ASSAY>/seed_<SEED>/<run_id>/
```

Im Lauf liegen `run.json`, `config.json`, `train.log`, `records.jsonl`,
`training/` mit Checkpoints/History/Validierung sowie
`evaluations/standard/CHIP/` und `evaluations/standard/GHTS/` mit Vorhersagen,
Metriken, Konfusionsmatrix und PR-/ROC-Plots. Cache-Inhalte werden nicht kopiert.

Der bisherige Trainingscode endet nach Quell-Validierung und Speicherung des
eingefrorenen Checkpoints (`skip_test`, bereits im Original vorhanden). Danach
führt die gemeinsame Inferenzkomponente beide Tests durch. Dadurch werden
Quell-Test und Ziel-Test jeweils einmal ausgewertet. Gewichtslernen,
Checkpoint-Auswahl und Quell-Validierungsschwelle bleiben identisch.

Ein belegtes Experimentlabel wird nicht mit einer anderen Konfiguration
des Modells oder Trainings überschrieben. Zusätzliche Seeds und die Auswahl
weiterer Quell-/Test-Assays sind Planungseinstellungen und können ergänzt werden.
Wiederholungen derselben Konfiguration erhalten neue Run-IDs.
Abgebrochene Versuche bleiben sichtbar. Zusammenfassungen berücksichtigen nur
den neuesten vollständig beendeten Versuch je TF/Quell-Assay/Seed. Unterschiedliche
Codeversionen werden nicht unbemerkt zusammengefasst.

Ergebnisse bei Bedarf erneut sammeln, etwa nach weiteren Seeds:

```bash
python run_experiment.py collect --all-configs
```

Dabei wird `00_basic` zuerst gesammelt, damit die anderen CSVs passende
Referenzdifferenzen enthalten. Nach einer späteren Änderung der Referenz erneut
alle Experimente sammeln. Ohne passende Referenz bleiben Vergleichsspalten leer.

## CSV und Laufzeiten

`SCHEMA.md` beschreibt die Bedeutung des gemeinsamen Headers. Die Zeilentypen
sind `run`, `epoch`, `evaluation`, `analysis` und `summary`. Nicht zutreffende
Werte sind leer. Die CSV enthält unter anderem komplette Epochentrajektorien,
Klassifikationsmetriken, Daten-/Codeprovenienz und Laufzeiten. Zeitangaben sind
Sekunden, Speicherangaben Bytes. Die zentralen CSVs werden nur vom Sammelprozess
geschrieben; die Trainingsjobs schreiben ausschließlich ihre eigenen Laufdaten.

TF-Zusammenfassungen mitteln über Seeds, Gesamtauswertungen zunächst über Seeds
je TF und danach gleichgewichtet über TFs. AUPRC bedeutet Average Precision.
Trainingszeiten werden ausschließlich in `run`-Zeilen erfasst und nicht in den
beiden Testzeilen doppelt gezählt. Summierte Jobzeit und Kalenderzeitspanne stehen
getrennt im Export. Schwellen und Modellwahl verwenden ausschließlich Quell-Val.

## Erweiterungen

Die Fabriken in den Konfigurationen sind normale, vertrauenswürdige Python-
Importpfade (`paket.modul:funktion`). Sie müssen keine MAA-Architektur erzeugen.
Neue Verfahren kommen als kleine eigene Module hinzu; es gibt keine zentrale
Liste aller zukünftigen Methodenparameter.

| Änderung | Erweiterungspunkt |
|---|---|
| Andere MAA-Kernel oder bekannte Modellparameter | `predictor.params` |
| Neue Modellarchitektur, anderer Bottleneck oder RC-Netz | `predictor.factory` |
| Andere Features, Targets oder Sampling-Daten | `data.factory` und ggf. passender Fitter |
| SAM, zusätzliche Verluste, Deep Supervision, EMA/SWA | `fitting.factory` |
| TTA, Ensembles oder andere Inferenzverfahren | `inference.factory` |
| Attributionen oder gruppierte Diagnosen | `analyses` |
| Vortraining/Feintuning, Teacher/Student, klassische Baselines | `workflow.factory` |
| Ein gemeinsames Training für mehrere TFs | optional `workflow.planner` |

Ein neuer Workflow erhält einen `RunContext` mit Konfiguration, Cache-Pfaden,
TF-Zuordnung, Run-Ordner, Recorder und einem gemeinsamen Artefaktverzeichnis.
Er kann mehrere Phasen ausführen und mehrere Checkpoints referenzieren.
Für Multi-TF-Training steht `ablation.planning:multi_tf` als Taskplaner bereit;
die tatsächliche Multi-TF-Lernmethode wird separat implementiert. Der Planer
verhindert, dass dafür automatisch 30 unabhängige Modelle angelegt werden.

Eigene Methoden melden standardisierte Felder an `context.recorder.record(...)`.
Zusätzliche Analysewerte verwenden `analysis_name`, `metric_name`, `metric_value`,
`metric_unit` beziehungsweise `extra_json`. Mit
`context.recorder.update(run_metrics={...})` lassen sich methodenspezifische
Laufbeschreibungen ergänzen. Ein gemeinsames Modell hat eine Model-ID; seine
TF-Auswertungen referenzieren diese ID, während die Trainingszeit einmal erfasst
wird. Physische Positionsauflösung, mehrfache Zielmasken und interne Merkmale
werden vom jeweiligen Modell-/Datenadapter geführt, nicht vom Runner festgelegt.

Diese erste Lieferung implementiert die drei gewünschten Kernelkonfigurationen.
Die übrigen Methoden aus dem Experimentplan haben Erweiterungspunkte, sind aber
keine bereits implementierten oder geprüften Trainingsverfahren.

## Prüfungen

```bash
python run_experiment.py verify-basic
python -m unittest discover -s tests -v
```

Die Regressionstests prüfen die Basic-Gewichte, alle Modelloutputs, Crop-/RC-
Mittelung, Augmentation und Sampling gegen die unveränderte Originaldatei.
Zusätzlich werden zwei vollständige Trainings-Epochen verglichen. Die
Integrationstests trainieren alle drei Varianten auf kleinen erzeugten Caches,
prüfen beide Test-Assays, identische CSV-Header, Referenzvergleiche, korrekte
Laufzeitsummen und die Behandlung wiederholter Versuche. Die erzeugten
Test-Caches und Test-Ergebnisordner werden danach entfernt.

Die Submit-Tests verwenden simulierte SLURM-Kommandos. Sie prüfen alle 60
Prioritäten, die drei getrennten Pläne, die globale Acht-Job-Grenze, das Auffüllen
mit der nächsten Konfiguration, die gemeinsame Launcher-Sperre und das Fortsetzen
nach einer verlorenen Submit-Bestätigung. Auf einem echten Cluster wurden diese
Skripte hier nicht ausgeführt.
