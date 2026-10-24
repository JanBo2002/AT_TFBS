# Prüfung des Umbaus

Geprüft am 24. Oktober 2026 auf CPU mit PyTorch 2.14.1+cpu.

## Numerischer Vergleich

Der Basic-Vergleich mit der unveränderten `hum301_motif64(1).py` besteht:

- Gleiche Initialgewichte und Puffer für Seeds 1, 13 und 42.
- Bitgleiche Vorhersagen einschließlich unterschiedlicher Center-Indizes.
- Gleiche Wahrscheinlichkeitsmittelung für Center-/Multi-Crop und beide Stränge.
- Gleiche Dropout-, Gradienten-, Clipping- und AdamW-Schritte.
- Gleiche Crops, RC-Augmentation und zyklische Auswahl der Negativbeispiele.
- Zwei vollständige synthetische Trainingsepochen mit gleichen Gewichten,
  Lernraten, Metriken, ausgewähltem Checkpoint und Validierungs-MCC-Schwellenwert.

Center- und optionaler `local-logsumexp`-Readout stimmen bei identischen Seeds und
Eingaben außerdem bitgenau mit `hum301_motif64_local(7).py` überein.

Für die ursprünglichen drei Konfigurationen wurden außerdem Initialgewichte,
Puffer und Vorhersagen im Trainings- und Auswertungsmodus sowie ein vollständiger
AdamW-Schritt mit der zuvor ausgelieferten Paketversion verglichen. Sie stimmen
auf CPU bitgenau überein.

Die neue Modelloption ist `channel_dropout`, standardmäßig 0. Sie ergänzt
Channel-Dropout nach dem Embedder, ohne die bisherigen Verlustfunktionen,
Optimierer, Scheduler oder das Sampling zu ändern. Der Syntaxbaum der
Trainingsschleife ist bis auf die Übergabe dieser einen Modelloption unverändert.
Die ursprünglichen drei JSON-Dateien, `data.py`, `run_experiment.py` und alle
bisherigen Shell- und SLURM-Dateien sind bytegleich zur vorherigen Paketversion.

## Neue Konfigurationen

| Konfiguration | Kernel | Kanäle je Kernel | Channel-Dropout | Bottleneck-Dropout |
|---|---|---|---|---|
| A03 | 3 / 6 / 9 / 12 | 32 / 32 / 32 / 32 | 0 | 0,1 |
| A04 | 3 / 6 / 9 / 12 | 32 / 32 / 32 / 32 | 0,1 | 0,1 |

Beide Varianten besitzen 1.221.653 Parameter. Der Test für A04 prüft, dass
tatsächlich vollständige Kanäle über alle Positionen deaktiviert werden und die
beibehaltenen Werte um 1/0,9 skaliert werden. Im Auswertungsmodus ist diese
Operation die Identität. Die Einstellung wird bis in den gespeicherten Checkpoint
übernommen.

## Integration und SLURM

20 Integrationstests bestehen. Sie prüfen unter anderem:

- Alle drei Konfigurationen unterscheiden sich nur in Name und Eingangskerneln.
- Alle fünf Varianten trainieren mit beiden Quell-Assays jeweils zwei Epochen
  und erzeugen beide Ziel-Assay-Auswertungen.
- Einheitliche CSV-Spalten, TF-/Seed-Zusammenfassungen und Referenzdifferenzen.
- Trainingszeiten werden pro Training und nicht pro Test-Assay gezählt.
- Vier gemeinsame Python-Dateien auch nach dem Training; keine Python-Kopien
  in Ergebnissen, `versions.txt` je Konfiguration und direkte Seed-Ordner.
- Bestehende Kombinationen aus Konfiguration, TF, Trainings-Assay und Seed
  werden nicht überschrieben.
- Vollständige Prioritätsliste mit 60 Einträgen und korrekte Aufgabenreihenfolge.
- Beide neuen Launcher wählen die richtige Konfiguration und planen mit dem
  vollständigen Manifest jeweils 60 Läufe in der gespeicherten Reihenfolge,
  mit Seed 13 und beiden Trainings-Assays.
- Höchstens acht offene Trainingsjobs, Auffüllen mit der nächsten Konfiguration,
  Wiederaufnahme und Schutz vor doppelten Einreichungen.
- Fertige Konfigurationen werden bereits vor dem Ende des gesamten Batches
  zusammengefasst.
- SLURM-Skriptpfade, Hintergrundstart und gemeinsame Launcher-Sperre.

Die Bash-Syntax aller Shell- und SLURM-Dateien und die Syntax der vier Python-
Dateien sind ebenfalls geprüft.

## Grenzen

Die Tests verwenden synthetische Sequenzen und simulierte SLURM-Kommandos. Es
wurde kein wissenschaftliches Training auf deinen echten TF-Daten und kein
GPU-/ROCm-Lauf ausgeführt.
