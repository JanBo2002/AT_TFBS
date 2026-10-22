# Update: sichtbare SLURM-Arrays

Dieses Update wird direkt in den vorhandenen Ordner `big_ablation` entpackt.
Es enthält keine Caches, keine neue Cache-Zuordnung und keine Modell- oder
Trainingskonfigurationen. Die vorhandene `configs/caches.csv` bleibt erhalten.

## Alten abgebrochenen Durchlauf bereinigen

```bash
python cleanup_aborted_batch.py
```

Das Skript liest die Job-IDs des alten Durchlaufs aus
`plans/active_batch.json` und dessen `state.json`. Es stoppt gegebenenfalls den
zugehörigen Hintergrundprozess, bricht ausschließlich dessen Jobs ab und prüft,
dass sie beendet sind. `collection: null` wird berücksichtigt.

Gelöscht werden die über `run.json` diesen Jobs zugeordneten Laufordner und der
alte Batch-Ordner einschließlich Taskplänen und SLURM-Logs. Wenn danach keine
anderen Läufe eines betroffenen Experiments verbleiben, werden auch dessen
erzeugte Codekopien, Laufkonfiguration und Auswertung entfernt. Bei verbleibenden
anderen Läufen bleiben deren gemeinsame Dateien erhalten; ein vorhandener
CSV-Export wird neu gesammelt.

`caches/`, `configs/`, die Prioritätsliste und fremde/ältere Läufe bleiben
erhalten. Insbesondere wird der ältere Lauf `71843_52` außerhalb dieses neuen
Batchs nicht abgebrochen oder gelöscht. Ein zweiter Aufruf nach erfolgreicher
Bereinigung verändert nichts. Nach Start eines neuen Array-Batchs verweigert
dieses spezielle Bereinigungsskript das Löschen.

## Alle drei Konfigurationen starten

```bash
bash submit.sh configs/caches.csv
```

Reihenfolge:

1. `00_basic`
2. `A01_kernel_LegNetMax_3_6_9_12`
3. `A02_kernel_7_11_15_21`

Für jede Konfiguration wird genau ein vollständiges SLURM-Array eingereicht.
Zu Beginn sind somit alle 60 Basic-Aufgaben im Scheduler sichtbar, z. B.
als `ARRAYID_[8-59%8]` plus die bereits freigegebenen Elemente. Erst wenn Basic
weniger als acht verbleibende Aufgaben hat, wird das vollständige A01-Array
eingereicht. Freie Plätze werden dann mit A01 aufgefüllt, z. B. 5 Basic + 3 A01.
Der Übergang zu A02 erfolgt genauso.

Die gemeinsame Grenze ist acht freigegebene, noch nicht beendete Trainingsjobs,
einschließlich wartender und aufräumender Jobs. Andere bereits laufende Jobs
außerhalb dieses Batchs werden nicht mitgezählt. Zusätzlich steht an jedem
Array `%8`; die gemeinsame Grenze wird durch gezielte Freigabe einzelner
Array-Elemente eingehalten.

Noch nicht freigegebene Elemente stehen mit `JobHeldUser` in `squeue`. Das ist
beabsichtigt. Der Launcher gibt sie anhand von `configs/training_priority.csv`
in der gespeicherten Reihenfolge frei. Start-Abhängigkeiten (`after`) zwischen
den freigegebenen Elementen verhindern, dass spätere Aufgaben frühere wartende
Aufgaben überholen. Die Freigabe wird ungefähr alle 60 Sekunden aktualisiert.
Die Zahl tatsächlich laufender Jobs hängt von Cluster-Ressourcen und Limits ab.

Nach Ende aller drei Arrays wird einmal der CPU-Sammeljob eingereicht.
Fehlgeschlagene oder abgebrochene Aufgaben werden nicht automatisch wiederholt.

## Einzelstarter und Status

Für einen einzelnen Durchlauf alternativ genau eines dieser Skripte verwenden:

```bash
bash submit_00_basic.sh configs/caches.csv
bash submit_A01_kernel_LegNetMax_3_6_9_12.sh configs/caches.csv
bash submit_A02_kernel_7_11_15_21.sh configs/caches.csv
```

Für die automatische Folge ist nur `submit.sh` nötig. Die gemeinsame Sperre
verhindert mehrere Launcher im selben Projekt. Keine Partition ist festgelegt.
Ressourcen-/Account-Optionen lassen sich anhängen, z. B. `--account=...` oder
`--mem=40G`. Die CPU-Sammeljob-Einstellungen stehen in `collect_all.sbatch`.

```bash
squeue -u "$USER"
# Alle Array-Elemente einzeln anzeigen:
squeue -r -u "$USER"
```

Der Steuerprozess bleibt auf dem Login-Knoten im Hintergrund aktiv und nennt
beim Start seinen Batch-Ordner. Status und Log:

```bash
python submit_jobs.py status plans/batch_array_XXXXXXXX
tail -f plans/batch_array_XXXXXXXX/launcher.log
```

Nach einem Neustart des Login-Knotens den vorhandenen Batch fortsetzen:

```bash
python submit_jobs.py resume plans/batch_array_XXXXXXXX
```

Dabei werden vorhandene Arrays und bereits freigegebene Aufgaben übernommen.
Vorausgesetzt werden `sbatch`, `squeue`, `sacct`, `scontrol`, funktionierendes
SLURM-Accounting und die aktivierte Trainingsumgebung.

Die Array- und Bereinigungstests verwenden simuliertes SLURM und temporäre
Ergebnisordner. Sie führen keine Jobs auf deinem Cluster aus.
