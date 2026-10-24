# Gemeinsames CSV-Schema 1.0

Jede Experiment-CSV besitzt denselben festen Header aus dem Abschnitt zum CSV-Schema in `training.py`.
Nicht verfügbare Werte sind leer, nicht künstlich null. Alle Zeiten sind Sekunden,
alle Speicherangaben Bytes. JSON-Spalten sind reguläre CSV-Felder und müssen mit
einem CSV-Parser eingelesen werden.

| `row_type` | Granularität | Verwendung |
|---|---|---|
| `run` | Ein physischer Trainings-/Modelllauf | Laufzeiten, Konfiguration, Epochenzahl, Checkpoint, Ressourcen |
| `epoch` | Eine Epoche eines Laufs | LR, Loss, Training/Val-Metriken, Sampling, genaue Epochendauer |
| `evaluation` | Ein Modell auf einem TF/Ziel-Assay | Testmetriken, Schwelle, Inferenzaufwand, Referenzvergleich |
| `analysis` | Ein zusätzlicher Analysewert | `analysis_name`, `metric_name`, `metric_value`, `metric_unit` |
| `summary` | Zusammenfassung einer Gruppe | Quelle, Gruppierung, Statistik und Stichprobenzahlen |

## Schlüssel und Provenienz

`experiment_id` entspricht exakt dem gewählten Ordner-/Konfigurationsnamen.
`run_id` unterscheidet Wiederholungen; `model_id` bezeichnet das zugrunde liegende
physische Modell. `training_scope` ist bei den ersten drei Versuchen `single_tf`;
andere Workflows können z. B. `multi_tf` verwenden. `tf_ids_json` hält die Menge
der Trainings-TFs fest. TF-Testzeilen tragen den jeweils ausgewerteten `tf`.

`train_assay`, `test_assay`, `direction`, `seed`, `epoch` und `evaluation_id`
bestimmen die betrachtete Situation. `status` und `included_in_summary` verhindern,
dass unvollständige oder wiederholte Versuche als zusätzliche Seeds eingehen.

`config_hash`, `code_hash`, Cache-SHA256, `split_hash`, `checkpoint_sha256`,
Versionen, Host und SLURM-Kennungen erlauben die Rückverfolgung.
`config_hash` kennzeichnet das Modell-/Verfahrensrezept ohne Beschreibung und
ohne die reinen Task-Auswahllisten für Seeds und Assays. Die tatsächlich gewählten
Seeds/Assays stehen in jeder Laufzeile; die gesamte Konfigurationsdatei wird
zusätzlich pro Lauf gespeichert. Der Jobplan prüft den Hash der vollständigen Datei.
`model_config_json`, `training_config_json`, `inference_config_json` und die
aufgelösten Dateien im Laufordner halten die tatsächlichen Einstellungen fest.

## Metriken

Enthalten sind AUPRC (= `average_precision_score`), AUROC, MCC, F1, Accuracy,
Balanced Accuracy, Precision, Recall, Spezifität, negativer Vorhersagewert,
TP/FP/TN/FN, Brier Score, Log-Loss, Schwelle, Klassenzahlen und PR-Baseline.
`train_...` und `val_...` enthalten entsprechende Metriken je Epoche, soweit sie
bereits im unveränderten Trainingsablauf berechnet werden. Brier/Log-Loss werden
bei der endgültigen Inferenz ergänzt; sie benötigen Wahrscheinlichkeiten und
bleiben bei klassischen Scores ohne Kalibrierung leer.

Loss-Felder umfassen Trainings-/Validierungs-Loss und finale Klassifikations-,
Regressions- und Gesamtverluste. Der FE-Kopf der Basic-Konfiguration ist wie
bisher vorhanden, erhält aber kein Trainingsziel.
In `run`-Zeilen beziehen sich `val_...`-Metriken auf den ausgewählten Checkpoint
mit der endgültigen Quell-Validierungsschwelle. `train_...` stammt aus dessen
Trainingsepoche. Die vollständige Entwicklung bei der bisherigen vorläufigen
Schwelle 0.5 bleibt in den `epoch`-Zeilen erhalten.

`delta_auprc`, `ratio_auprc`, `log2_ratio_auprc` und `paired_reference_run_id`
werden nur bei passender Referenz mit identischem TF/Assay/Seed, Testcache,
Testsplit und Inferenzansichten ergänzt. `win_fraction` beschreibt in der
Experimentzusammenfassung den Anteil der TFs mit positiver mittlerer Differenz.

## Zeiten und Ressourcen

- `training_wall_s`: Zeit des Fit-Aufrufs einschließlich Laden, Lernen,
  Quell-Validierung zur Schwellenwahl und Checkpoint-Export; ohne die beiden Tests.
- `fit_loop_s`: gesamte Zeit der Epochenschleife einschließlich Checkpoints und
  Verlaufsexport.
- `train_total_s`, `validation_total_s`, `mining_total_s`: Summen der jeweiligen
  Epochenphasen. Die spätere Schwellenvalidierung ist in `post_fit_s` enthalten.
- `setup_s`: Vorbereitung bis zum Beginn der Epochenschleife.
- `post_fit_s`: Rest des Fit-Aufrufs nach Setup und Epochenschleife.
- `epoch_s`: Training, Validierung, Mining und kleine Epochenverwaltung bis zur
  ursprünglichen Zeitmessstelle; ohne anschließendes Checkpoint-/History-Schreiben.
- `epoch_mean_s`, `epoch_median_s`, `epoch_std_s`, `epoch_min_s`, `epoch_max_s`,
  `epoch_p95_s`: Beschreibung der einzelnen Epochendauern.
- `inference_s`: `run_epoch` beim Test einschließlich Laden der Batches,
  Vorwärtsläufen und Metrikberechnung. Auf GPU werden die Zeitgrenzen synchronisiert.
- `evaluation_wall_s`: zusätzlich Testcache-Laden, Modellprovenienz und Export;
  ohne die abschließende Speicherung der Timing-JSON selbst.
- `total_wall_s`: gesamter Run einschließlich Vorbereitungs-, Fit- und Testphasen.
- `samples_per_s`: Testsequenzen pro Sekunde; `forward_views` beschreibt den
  zusätzlichen TTA-Aufwand.
- `gpu_peak_memory_bytes`: maximal durch PyTorch belegter GPU-Speicher während
  dieses Laufs, nicht die gesamte Gerätespeicherbelegung.

Trainingssummen stehen in `run`-Zeilen und werden nicht in jede Testzeile kopiert.
Die Summe aller Jobzeiten ist bei paralleler Ausführung größer als die verstrichene
Kalenderzeit. Letztere wird als eigener Summary-Wert
`metric_name=experiment_elapsed_wall_s` erfasst und kann Pausen zwischen Jobs
einschließen. Es wird kein pauschaler GPU-Stundenwert für gemeinsam belegte GPUs
erfunden.

## Zusammenfassungen

`summary_level` ist `tf` oder `experiment`; `summary_source` verweist auf `run`
oder `evaluation`. `statistic` ist mean/std/median/q25/q75/min/max/sum.
Standardabweichungen verwenden ddof=1 und bleiben bei einem Wert leer.

Bei TF-Zusammenfassungen bilden die Seeds die Stichprobe. Bei
Experimentzusammenfassungen werden zuerst die Seeds je TF gemittelt und dann die
TF-Mittelwerte zusammengefasst. Dadurch erhalten TFs mit mehr Seeds kein größeres
Gewicht. Summen beziehen sich auf die einzelnen ausgewählten Modelle bzw.
Auswertungen und sind ausschließlich für Zeiten und Zählgrößen definiert;
AUPRC/MCC/F1 werden niemals summiert. Es gibt keine vermeintlich gepoolte AUPRC
aus gemittelten TF-Ergebnissen.

`n_records`, `n_runs`, `n_tfs`, `n_seeds` und `aggregation_unit` legen die genaue
Bezugsgröße offen. Bei Änderungen des Schemas wird die Schemaversion zentral
angehoben; einzelne Experimente erfinden keine eigenen CSV-Header.
