# Model artifacts

On first run of `setup_project.py`, a trained scikit-learn Pipeline will be saved as `staywise_pipeline.joblib` and real validation comparison and independent-test evaluation measurements will be saved as `evaluation.json`.

These files are **not pre-generated** in this source distribution. This is intentional: validation results must be produced by actually training on the original dataset, not invented or shown as fake demos.

To retrain: `python setup_project.py --force`.
