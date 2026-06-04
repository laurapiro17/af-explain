# af-explain

**Explainable deep learning for atrial fibrillation detection on single-lead ECG.**

[![CI](https://github.com/laurapiro17/af-explain/actions/workflows/ci.yml/badge.svg)](https://github.com/laurapiro17/af-explain/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PhysioNet/CinC 2017](https://img.shields.io/badge/data-PhysioNet%2FCinC%202017-orange)](https://physionet.org/content/challenge-2017/1.0.0/)

> A reproducible 1-D ResNet for AFib detection trained on the PhysioNet/CinC
> Challenge 2017 single-lead dataset, paired with two explainability views
> (Grad-CAM-1D and Integrated Gradients) and a four-axis clinical analysis
> (subgroup robustness, calibration, failure-mode taxonomy, clinician-model
> concordance) written from a medical-student perspective.

---

## Question

**Background.** Atrial fibrillation is the most common sustained
arrhythmia and a leading cause of cardioembolic stroke. Deep-learning
classifiers now reach cardiologist-level accuracy on single-lead ECG,
yet three gaps block clinical and regulatory adoption: aggregate metrics
mask systematic subgroup failures, post-hoc saliency maps are rarely
audited against the morphological features clinicians actually use, and
misclassifications are reported as numbers rather than as clinically
meaningful failure modes.

**Question.** For a 1-D ResNet trained on the PhysioNet/CinC 2017
single-lead ECG corpus, does the model

1. **maintain its atrial-fibrillation sensitivity** across stratifying
   variables such as signal quality and record length;
2. **produce Grad-CAM and Integrated-Gradients saliency** that overlaps
   with clinician-annotated regions of interest (absent P waves,
   irregular RR intervals, fibrillatory baseline); and
3. **commit errors organisable into a clinically grounded taxonomy of
   failure modes** (false-negative AFib, atrial flutter mistaken for
   AFib, noise read as rhythm, low-amplitude AF read as sinus) rather
   than appearing as undifferentiated noise in a confusion matrix?

**Why this matters.** Subgroup audit is a prerequisite for FDA/EMA
submission of AI-based ECG tools; clinician–model concordance is a
prerequisite for bedside trust; a clinical failure-mode taxonomy is what
tells the next researcher where to invest data-collection and
model-improvement effort. Together they turn an accurate classifier into
a deployable one — and the three analyses are far cheaper to produce
jointly than sequentially.

> Canonical source: `RESEARCH_QUESTION` in
> [`src/af_explain/evaluation/clinical_analysis.py`](src/af_explain/evaluation/clinical_analysis.py).

## Contribution

This is not a SOTA chase. Three things make this repo useful:

1. **End-to-end reproducibility.** Public dataset, deterministic split,
   pinned dependencies, Docker, CI. `git clone && docker build && train`
   reproduces the numbers.
2. **Two explainability views, side-by-side.** Coarse-grained Grad-CAM-1D
   (which time window) plus sample-level Integrated Gradients (which sample,
   signed) — readable through a Streamlit demo.
3. **A clinical analysis layer.** The
   [`evaluation/clinical_analysis.py`](src/af_explain/evaluation/clinical_analysis.py)
   module turns the model output into four reports a cardiologist would
   actually read: subgroup sensitivity, calibration, failure-mode taxonomy,
   and clinician-model concordance.

## Data

| Property         | Value                                                       |
|------------------|-------------------------------------------------------------|
| Dataset          | PhysioNet/CinC Challenge 2017 — training set                |
| Records          | 8,528 single-lead ECGs                                      |
| Sampling rate    | 300 Hz                                                      |
| Duration         | 9 – 60 s (we pad/crop to 30 s)                              |
| Classes          | Normal · AFib · Other · Noisy                               |
| Splits           | 70 / 15 / 15 stratified, seed 42                            |
| Licence          | ODC-BY 1.0 — citation required                              |

Download with `./scripts/download_data.sh` (≈ 200 MB).

## Methods

```
raw ECG ─► bandpass 0.5-40 Hz ─► z-score ─► segment 30 s @ 300 Hz
                                                      │
                                                      ▼
                                           1-D ResNet (≈ 3 M params)
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            ▼                         ▼                         ▼
                     Grad-CAM-1D            Integrated Gradients        Clinical analysis
                  (1-s window saliency)     (per-sample, signed)        (4 reports below)
```

**Clinical analyses** (see [`clinical_analysis.py`](src/af_explain/evaluation/clinical_analysis.py)):

| # | Analysis                  | Question it answers                                                           |
|---|---------------------------|-------------------------------------------------------------------------------|
| 1 | Subgroup robustness       | Does AFib sensitivity drop for any subgroup (signal quality, length, etc.)?   |
| 2 | Calibration               | Are the predicted probabilities trustworthy? ECE + temperature scaling.       |
| 3 | Failure-mode taxonomy     | When the model fails, *how* does it fail (FN-AFib, flutter confusion, …)?    |
| 4 | Clinician concordance     | Do Grad-CAM peaks overlap with expert-annotated P waves / RR irregularity?    |

## Project layout

```
af-explain/
├── src/af_explain/
│   ├── data/                # download · preprocess · PyTorch Dataset
│   ├── models/              # 1-D ResNet
│   ├── training/            # Lightning module + CLI
│   ├── explain/             # Grad-CAM-1D · Integrated Gradients
│   ├── evaluation/          # clinical_analysis.py  ← the contribution
│   └── app/                 # Streamlit demo
├── configs/                 # Hydra configs (data, model, train)
├── tests/                   # pytest — runs in CI on every push
├── scripts/                 # download_data.sh · train.sh
├── docs/methods.qmd         # Quarto methods paper draft
├── Dockerfile               # reproducible runtime
└── .github/workflows/ci.yml # lint · format · type-check · tests
```

## Quickstart

```bash
git clone https://github.com/laurapiro17/af-explain.git
cd af-explain
uv sync --all-extras --dev          # install (or: pip install -e ".[app,dev]")
./scripts/download_data.sh          # PhysioNet 2017 → data/raw/
af-train --epochs 50                # train
streamlit run src/af_explain/app/streamlit_app.py
```

Or with Docker:

```bash
docker build -t af-explain .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs af-explain
```

## Reproducibility

- Deterministic split seeded by `--seed 42`.
- All dependencies pinned in `uv.lock`.
- CI runs the test suite on Python 3.11 and 3.12.
- Training is fully `lightning.Trainer(deterministic=True)`.

## Status

| Component                                | Status       |
|------------------------------------------|--------------|
| Data + preprocessing                     | ✅ implemented |
| 1-D ResNet model                         | ✅ implemented |
| PyTorch Lightning training loop          | ✅ implemented |
| Grad-CAM-1D                              | ✅ implemented |
| Integrated Gradients                     | ✅ implemented |
| Clinical analysis (subgroup, calibration, taxonomy) | ✅ implemented |
| Clinician-model concordance              | 🟡 interface ready, awaits expert annotations |
| Streamlit demo                           | ✅ implemented |
| Trained checkpoint                       | ⏳ pending GPU/training run |
| HuggingFace Spaces deployment            | ⏳ pending checkpoint  |
| Quarto methods paper                     | 🟡 outline drafted, results pending |

## Citation

If you use this repository, please cite the dataset (mandatory under ODC-BY)
and, optionally, the repository itself:

```bibtex
@misc{pinero2026afexplain,
  author       = {Piñero Roig, Laura},
  title        = {af-explain: Explainable Atrial Fibrillation Detection on Single-Lead ECG},
  year         = {2026},
  url          = {https://github.com/laurapiro17/af-explain}
}

@inproceedings{clifford2017af,
  author    = {Clifford, GD and Liu, C and Moody, B and Lehman, L and Silva, I
               and Li, Q and Johnson, AEW and Mark, RG},
  title     = {AF Classification from a Short Single Lead ECG Recording:
               the PhysioNet/Computing in Cardiology Challenge 2017},
  booktitle = {Computing in Cardiology},
  volume    = {44},
  year      = {2017},
  doi       = {10.22489/CinC.2017.065-469}
}
```

## Author

**Laura Piñero Roig** — Medical Student (Universitat de Barcelona) ·
[ORCID 0009-0008-3390-4029](https://orcid.org/0009-0008-3390-4029) ·
[GitHub @laurapiro17](https://github.com/laurapiro17)

## Licence

[MIT](LICENSE) — code only. The PhysioNet 2017 dataset is ODC-BY 1.0;
see [PhysioNet's data use agreement](https://physionet.org/content/challenge-2017/1.0.0/#files).
