# Tribolium Serosa Ultrack Motion Workflow

Dieses Repository dokumentiert einen Workflow zur Segmentierung, zum Tracking und zur Bewegungsanalyse einer 2D-Zeitreihe der extraembryonalen Membranen von *Tribolium*. Der Schwerpunkt liegt auf der Bewegung der Serosa, besonders im spaeten Bereich der Zeitreihe vor dem Aufreissen des Gewebes.

## Kurzueberblick

Die Zeitreihe umfasst 571 Frames. Zunaechst wurden rohbildbasierte Segmentierungs- und Ultrack-Ansaetze getestet. Diese Raw-only-Varianten dienten als methodische Vergleichsbasis, fuehrten aber zu starker Fragmentierung und waren visuell nicht ausreichend stabil.

Stabilere Ergebnisse wurden mit labelbasierten Multi-Input-Ansaetzen erreicht. Dabei wurden mehrere Cellpose-/CPSAM-Segmentierungsvarianten als Kandidatenquellen an Ultrack uebergeben.

Fuer die Originalframes 540--570 wurden mehrere Tracking-Varianten visuell und quantitativ verglichen. Dieses Fenster ist biologisch relevant, da dort die Serosa-Bewegung deutlich zunimmt und sich das Gewebe dem spaeteren Aufreissen naehert.

## Wichtigste Ergebnisse

- Raw-only-, Threshold-, Watershed- und reine Konturansaetze waren fuer die vorliegenden Daten nicht ausreichend stabil.
- Labelbasierte Multi-Input-Ansaetze lieferten kohaerentere Tracking-Ergebnisse.
- Der Full-Frame-Lauf mit automatisch erzeugten Cellpose-Varianten war statistisch nahezu identisch zu einem Multi-Input-Lauf mit projektspezifischen Referenzlabels.
- Fuer Frames 540--570 blieb `multi_existing_labels_540_570` der visuell kohaerenteste Referenzlauf.
- Der unabhaengige Lauf `independent_multi_cpsam_candidates_540_570` ist methodisch wichtig, weil er ohne projektspezifische Referenzlabels und ohne projektspezifisches Cellpose-Modell auskommt.
- Trackbasierte Bewegungsfelder und Raw-image Optical Flow zeigen aehnliche grobe Bewegungsstrukturen, unterscheiden sich aber lokal und bei Ausreissern.
- Divergenz, Curl, Gradient Magnitude und Strain Magnitude liefern erste explorative Hinweise auf lokale Expansion, Rotation und Deformation.

## Repository-Struktur

```text
configs/          YAML-Konfigurationen der dokumentierten Ultrack-Runs
scripts/          Skripte fuer Tracking, Export, Bewegungsfelder und Plots
documentation/    Finaler Workflow-Bericht als PDF
results/          Tabellen, Summaries und ausgewaehlte CSV-Ergebnisse
plots/            Finale Plots, Quiver-Bilder und Heatmaps
assets/videos/    Komprimierte Videos fuer Tracking und Bewegungsfelder
```

## Datenhinweis

Die originalen Rohbilder und grossen Zwischendaten sind nicht Bestandteil dieses Repositories. Nicht enthalten sind Rohbild-TIFFs, vollstaendige Cellpose-Maskenordner, Ultrack-Datenbanken, Zarr-/NPY-/NPZ-Zwischendateien, Conda-Umgebungen und temporaere Testausgaben.

Um den kompletten Workflow neu aus den Rohdaten auszufuehren, muessen die Originaldaten lokal verfuegbar sein und Pfade in den YAML-Konfigurationen gegebenenfalls angepasst werden.

## Zentrale Ergebnisdateien

```text
results/tables/run_summary_comparison.csv
results/tables/full_frame_analysis_comparison.csv
results/tables/velocity_field_comparison.csv
results/velocity_fields_540_570/
results/optical_flow_540_570/
results/velocity_fields_full/
results/derivatives/
```

## Videos

Full-Frame Tracking Overlays:

```text
assets/videos/full_frame_tracks/multi_cellpose_variants_no_baseline_tracks_overlay_compressed.mp4
assets/videos/full_frame_tracks/multi_cellpose_only_tracks_overlay_compressed.mp4
```

Bewegungsfeld-Videos fuer Frames 540--570:

```text
assets/videos/motion_fields/multi_existing_labels_540_570_scaled_quiver.mp4
assets/videos/motion_fields/independent_multi_cpsam_candidates_scaled_quiver.mp4
assets/videos/motion_fields/raw_optical_flow_scaled_quiver.mp4
```

Bei Quiver-Videos und Quiver-Plots wurden die Pfeillaengen zur besseren Sichtbarkeit visuell skaliert. Die numerischen Geschwindigkeitswerte in den CSV-Dateien bleiben unveraendert in px/Frame.

## Wichtige Skripte

```text
scripts/run_ultrack_pipeline.py
scripts/export_ultrack_results.py
scripts/collect_run_summaries.py
scripts/collect_full_frame_analysis.py
scripts/compute_track_velocity_field_540_570.py
scripts/compute_raw_optical_flow_540_570.py
scripts/compute_full_frame_velocity_field.py
scripts/compute_velocity_derivatives.py
scripts/make_final_analysis_outputs.py
scripts/plot_velocity_quiver_scaled.py
scripts/view_velocity_field_napari.py
```


Zusaetzlich ist ein fokussierter Full-Frame-Tracking-Vergleich enthalten:

```text
documentation/full_frame_tracking_comparison.pdf
```

## Interpretation

Globale Trackstatistiken sind hilfreich, aber nicht allein ausreichend. Deshalb wurden Statistik, visuelle Kontrolle, Bewegungsfelder und derivative Groessen gemeinsam betrachtet.

Die Analyse klaert die mechanische Ursache des Aufreissens nicht abschliessend. Sie liefert aber eine reproduzierbare Grundlage, um lokale Bewegung, Expansion, Rotation und Deformation der Serosa vor dem Aufreissen weiter zu untersuchen.

## Finaler Bericht

```text
documentation/tribolium_ultrack_workflow_protokoll_final.pdf
```

