---
title: af-explain
emoji: ❤️
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
short_description: Explainable AFib detection on single-lead ECG (Grad-CAM + IG)
tags:
  - medical
  - ecg
  - atrial-fibrillation
  - explainable-ai
  - pytorch
  - cardiology
---

# af-explain — Hugging Face Space

Interactive demo of [`af-explain`](https://github.com/laurapiro17/af-explain):
a 1-D ResNet that classifies single-lead ECG into Normal / AFib / Other /
Noisy and produces two complementary explanations:

- **Grad-CAM-1D** — which 1-second windows of the ECG drove the prediction.
- **Integrated Gradients** — per-sample, signed contribution of every point.

## How to use

1. Upload a single-lead ECG as `.npy` (1-D array, 300 Hz) or `.csv`
   (single column).
2. See the class probabilities + saliency overlays.
3. Switch the *target class* in the sidebar to ask "what would the
   model attend to *if* it were calling X here?"

## Clinical caveat

These explanations are post-hoc approximations of what the model attends
to. They do **not** imply causation and should never replace a
cardiologist's reading. The model was trained on the PhysioNet/CinC 2017
public dataset, not a regulated clinical pipeline.

## Citation

If you use this Space, please cite the underlying dataset:

> Clifford GD, Liu C, Moody B, et al. *AF Classification from a Short
> Single Lead ECG Recording: the PhysioNet/Computing in Cardiology
> Challenge 2017*. CinC 44, 2017.
> [doi:10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469)

Code: [github.com/laurapiro17/af-explain](https://github.com/laurapiro17/af-explain)
