# Notebooks

Exploratory notebooks. Always treat the source modules in `src/af_explain/`
as the source of truth — copy stable code out of notebooks into the package
as soon as it works.

Suggested notebook order:

1. `01_data_exploration.ipynb` — load the dataset, inspect class balance,
   plot one record per class.
2. `02_baseline_model.ipynb` — sanity-check the model on a tiny subset
   (10 minutes of training).
3. `03_explainability.ipynb` — Grad-CAM and IG on representative records.
4. `04_clinical_interpretation.ipynb` — run the four clinical analyses
   and discuss the failure modes that surface.
