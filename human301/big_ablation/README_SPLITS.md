# Human301: drei Chromosomensplits

Bestehende Configs bleiben gültig. Ohne `data.params.split_variant` wird exakt
der vorhandene Originalcache mit seiner bisherigen Aufteilung verwendet.
Auch `00_basic.json` liegt unverändert im Paket.

## Installation

Das Paket beruht auf den sechs hochgeladenen Dateien und dem dazugehörigen
`run_experiment(4).py`. `run_experiment.py` ist mit enthalten, damit der Jobplan
bereits die tatsächlich ausgewählten Cachepfade festhält.

Entpacke das ZIP in einen eigenen Ordner. Von dort:

```bash
python install_split_update.py --target /data/projects/SFB_A03/jan/AT_TFBS/human301/big_ablation
```

Der Installer prüft die Versionen der vorhandenen Skripte, erstellt eine
Sicherung außerhalb von `big_ablation` und installiert die sechs Programmdateien
und die zwei zusätzlichen Basic-Configs. Vorhandene normale Configs, Caches und
Ergebnisse werden dabei erhalten. Mit `--dry-run` wird nur geprüft.
Bei aktiven oder wartenden eigenen SLURM-Jobs im Projekt wird die Installation
abgebrochen: Diese Jobs müssen erst mit ihrer bisherigen Codeversion enden.
Bei inzwischen geänderten Skripten wird ebenfalls abgebrochen, damit solche
Änderungen nicht überschrieben werden.

## Daten für jeden TF vorbereiten

Im Projektverzeichnis, in deiner `at_env`:

```bash
python data.py prepare-splits --manifest configs/caches.csv --report plans/split_preparation.json
```

Dieser eine Aufruf bearbeitet jeden TF und jeden verfügbaren Assay im bisherigen
Manifest und erzeugt `split_v2` und `split_v3`. Die bereits vorhandenen Caches
enthalten alle ursprünglichen Sequenzen einschließlich Chromosomen und Labels.
Es werden nur die `split`-Zuordnung und die zugehörigen Statistiken neu erzeugt.
Sequenzen, Labels, Koordinaten, Zeilenreihenfolge und Sample-IDs bleiben erhalten.
Die IDs bleiben stabile Quellkennungen; ein alter Split-Präfix in einer ID ist
keine aktuelle Splitzuordnung. Dafür ist das `split`-Array maßgeblich.

Beispiel:

```text
caches/ArchThalia-ML/CREB3L3/CHIP/dataset.npz
caches/ArchThalia-ML/CREB3L3/CHIP/splits/split_v2/dataset.npz
caches/ArchThalia-ML/CREB3L3/CHIP/splits/split_v3/dataset.npz
```

Das Originalmanifest `configs/caches.csv` bleibt erhalten. Bereits korrekt
erzeugte Varianten werden geprüft und wiederverwendet. Eine andere Quelldatei
oder eine abweichende Variantendatei führt zu einem Fehler. `--overwrite` ersetzt
nur Variantencaches und darf nur verwendet werden, wenn keine Runs sie verwenden.
Alle Teilmengen müssen positive und negative Beispiele enthalten; ein TF mit
unzureichender Chromosomenabdeckung wird mit einem Fehler gemeldet. Es gibt kein
stilles Ausweichen auf einen anderen Split. Nicht zugeordnete Chromosomen wie
chrX werden ebenfalls gemeldet.

Die pro TF/Assay tatsächlichen Klassenanzahlen stehen im Bericht und in der
jeweiligen `dataset.summary.json`. Die Auswahl erfolgt für alle TFs und beide
Assays nach denselben Chromosomenlisten:

| Variante | Training | Validierung | Test |
|---|---|---|---|
| original | Vorhandener Cache | Vorhandener Cache | Vorhandener Cache |
| split_v2 | 4,6,8,10,12,14,18,20,22 | 2,16 | 1,3,5,7,9,11,13,15,17,19,21 |
| split_v3 | 2,5,6,10,11,13,15,18,20 | 1,8 | 3,4,7,9,12,14,16,17,19,21,22 |

Für die vier hochgeladenen FASTAs ergeben sich folgende positive Anzahlen
(jeweils hundertmal so viele Negative):

| Variante | Training | Validierung | Test |
|---|---:|---:|---:|
| Original bei Validierung chr5/chr7 | 543 | 171 | 775 |
| split_v2 | 605 | 170 | 714 |
| split_v3 | 540 | 174 | 775 |

Bei anderen TFs variieren die Anzahlen. Die neue gemischte Gruppierung wurde
anhand der Klassenanzahlen dieser Uploads ausbalanciert, ohne Modellergebnisse
für die Auswahl zu verwenden.

## In einer Config auswählen

Im bestehenden `data`-Abschnitt:

```json
"data": {
  "factory": "ablation.providers:human301",
  "params": {
    "split_variant": "split_v3"
  }
}
```

Zulässig sind `original`, `split_v2`, `split_v3`. Ohne Angabe bleibt die bisherige
Aufteilung erhalten. Die früheren Config-Werte werden als Aliase weiterhin erkannt; neue Configs erhalten die Namen `split_v2` und `split_v3`. Verwende für neue Varianten eigene Config- und
Experimentnamen, z.B. `BESTE_CONFIG_split_v3`, und als Referenz `00_basic_split_v3`.
Die normalen Ergebnisordner bleiben dadurch erhalten. Ein Versuch, im selben
Experimentnamen eine andere Config zu verwenden, wird wie bisher abgelehnt.

## Kopien deiner besten Configs erzeugen

Ersetze den Platzhalter durch den tatsächlichen Dateinamen:

```bash
python data.py make-split-configs --config configs/DEINE_BESTE_CONFIG.json
```

Das erzeugt zwei Kopien mit den Suffixen `_split_v2` und `_split_v3`. Auch die
referenzierte Basic-Config wird bei Bedarf kopiert, und die Referenzen werden
entsprechend angepasst. Modellparameter, Training, Inferenz, Seeds und TF-Auswahl
werden aus deinen vorhandenen Configs übernommen. Die Ausgangsconfigs werden
nicht verändert. Abweichende bereits vorhandene Kopien werden nicht überschrieben.
Mehrere Ausgangsconfigs können hinter `--config` stehen. Mit `--variants split_v3`
kannst du nur eine Variante erzeugen.

## Starten

Die beiden zusätzlichen Basic-Läufe:

```bash
bash submit_array.sh 00_basic_split_v2 00_basic_split_v3
```

Für deine Varianten ebenfalls deren neue Config-Namen angeben, z.B.:

```bash
bash submit.sh --config 00_basic_split_v2 DEINE_BESTE_CONFIG_split_v2 00_basic_split_v3 DEINE_BESTE_CONFIG_split_v3
```

Der native Array-Launcher bereitet die ausgewählten neuen Splitvarianten für
alle TFs im Manifest automatisch vor, bevor er den Aufgabenplan erzeugt und
Jobs einreicht. Der vorbereitende Datenaufruf benötigt keinen GPU-Job.
Ein fehlgeschlagener Datenaufruf verhindert die Einreichung des neuen Arrays.
Die bestehende Begrenzung auf acht Trainingsjobs, Queue-Abhängigkeiten und
sofortige Zusammenfassung pro abgeschlossener Config bleiben erhalten.
Der frühere Aufruf von `submit.sh` mit Manifest und Ressourcenoptionen bleibt
als Legacy-Aufruf erhalten.

Ergebnisse liegen z.B. unter `results/00_basic_split_v3/` und
`results/Results_all/00_basic_split_v3.csv`. Die CSV-Spalten bleiben erhalten.
Die aktuellen Cachepfade und Quellprüfsummen werden wie bisher mitprotokolliert;
zusätzlich steht die Variante in der Run-Konfiguration und in `run.json`.
Neue Modelle werden von Grund auf trainiert. Bestehende Checkpoints dürfen für
die neuen Splits nicht fortgesetzt werden.

## PWM-Configs

PWMs und ihr Ranking müssen für den jeweiligen Split erneut aus zulässigen
Quelldaten ermittelt werden. Das Paket berechnet keine neuen PWMs; die dafür
verwendeten Entdeckungs-/Rankingskripte gehören nicht zu den hochgeladenen Dateien.
Für `split_v2`/`split_v3` verlangen der integrierte PWM-Loader und der Human301-Provider
bei `predictor.params.pwm_root` deshalb einen Herkunftsdatensatz je TF/Quellassay:

```text
PWM_ROOT/TF/CHS/split_provenance.json
PWM_ROOT/TF/GHTS/split_provenance.json
```

Beispiel für nachweislich auf dem gemischten Split neu ermittelte PWMs:

```json
{
  "split_variant": "split_v3",
  "fit_chromosomes": [2, 5, 6, 10, 11, 13, 15, 18, 20],
  "selection_chromosomes": [1, 8]
}
```

Trage ausschließlich die tatsächlich verwendeten Chromosomen ein. Das Erstellen
dieser JSON-Datei ersetzt keine Neuberechnung. PWM-Fitting darf nur
Trainingschromosomen verwenden, Auswahl/Ranking nur Training oder Validierung
des Quellassays. Für den ursprünglichen Split ändert sich das PWM-Verhalten nicht.
PWM-Configs mit anderen Parameternamen als `pwm_root` oder eigenem Datenprovider
müssen ihre eigene Herkunftsprüfung gewährleisten.

## Prüfung

```bash
python -m unittest discover -s tests -v
```

Die Tests prüfen die Datenvorbereitung für mehrere TFs und beide Assays,
unveränderte Quelldateien und Labels, idempotente Wiederholung, Config-Kopien,
Referenzen, Jobplanpfade und die PWM-Herkunftsprüfung. Wenn die ursprünglichen
FASTA-Uploads neben dem Paket im Ordner `upload` liegen, werden auch deren
vollständige Sequenzen durch die neue Cachevorbereitung geprüft.
Die Tests laden die relevanten Daten-/Config-/Planungsfunktionen ohne
Neuralnetz-Imports und benötigen deshalb kein PyTorch. Ein vollständiges
Training wurde hier nicht ausgeführt; das erfolgt in deiner `at_env` auf dem Cluster.
