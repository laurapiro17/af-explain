# Notebooks

Exploratory notebooks. Always treat the source modules in `src/af_explain/`
as the source of truth — copy stable code out of notebooks into the package
as soon as it works.

Recommended order:

1. **`01_data_exploration.ipynb`** — class balance, one example per class
   (raw + preprocessed), record-length distribution. Run before anything
   else: catches dataset surprises.
2. **`02_baseline_model.ipynb`** — overfit a 50-record subset for 5 epochs.
   If it doesn't overfit, the pipeline has a bug. Skip the long training
   run until this passes.
3. **`03_explainability.ipynb`** — Grad-CAM + Integrated Gradients on one
   correctly-classified record per class. Requires a trained checkpoint
   at `outputs/checkpoints/best.ckpt`.
4. **`04_clinical_interpretation.ipynb`** — the contribution notebook.
   Runs subgroup robustness, calibration, failure-mode taxonomy on the
   held-out test set, and clinician concordance against
   `annotations/*.json`. Outputs feed the Results section of
   `docs/methods.qmd`.

To register the kernel:

```bash
uv run python -m ipykernel install --user --name af-explain --display-name "Python 3 (af-explain)"
```
