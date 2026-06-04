# Expert annotations

This folder holds JSON annotations used by
`af_explain.evaluation.clinical_analysis.clinician_concordance`.

## Format

One file per ECG record, named `<record_id>.json`:

```json
{
  "record_id": "A00001",
  "fs": 300,
  "rois": [
    {"start_s": 1.20, "end_s": 1.50, "label": "P_wave"},
    {"start_s": 4.00, "end_s": 6.00, "label": "RR_irregularity"},
    {"start_s": 12.50, "end_s": 13.00, "label": "ectopic_beat"}
  ],
  "annotator": "Laura Piñero Roig",
  "annotated_at": "2026-06-XX"
}
```

## Suggested annotation workflow

1. Pick 30–50 records stratified by class (~10 per class).
2. Open each in the Streamlit demo or any ECG viewer (PhysioBank ATM, LightWAVE).
3. Mark the time windows where you would point a colleague to support
   the diagnosis (or rule it out).
4. Save the JSON in this folder.

A small starter set (3–5 records) is enough to validate the
concordance computation. The repo's clinical contribution gains weight
proportional to the number of annotated records.

## Suggested ROI labels (extend as needed)

| Label                | Meaning                                            |
|----------------------|----------------------------------------------------|
| `P_wave`             | Visible P wave preceding a QRS                     |
| `absent_P`           | Window where P wave is conspicuously absent        |
| `RR_irregularity`    | Sustained irregular RR intervals                   |
| `fibrillatory_wave`  | Fine fibrillatory ("f") waves in baseline          |
| `ectopic_beat`       | Premature atrial or ventricular complex            |
| `baseline_wander`    | Strong low-frequency drift                         |
| `noise_artifact`     | Motion / EMG artifact                              |
