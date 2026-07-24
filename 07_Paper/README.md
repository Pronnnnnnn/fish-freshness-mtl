# Manuscript Notes

Target venue: ICELTICs 2026, IEEE two-column format, 6 pages including
figures, tables, and references. Official template:
https://iceltics.usk.ac.id/paper-template/

## Section-to-output mapping

| Section | Source | Notes |
|---|---|---|
| Related work | literature comparison table | condense to the handful of directly comparable results, not an exhaustive survey |
| Data | `02_Manifests/manifest_full.csv` | report totals per species and per freshness level, not the full 24-cell breakdown |
| Method - architecture | one diagram of the shared backbone + two heads | a single figure combining backbone and head layout |
| Method - loss weighting | EW/UW/DWA equations from `04_Src/loss_weighting.py` | equations only, no derivation |
| Results - main table | `06_Results/metrics/multitask_test_results.csv`, aggregated mean +/- std | 5 models x {accuracy, F1, joint accuracy} at minimum; MCC/Kappa/QWK can be mentioned in text if space is tight |
| Results - efficiency | `params` / `inference_ms` columns | one sentence, or two extra columns on the main table |
| Results - interpretability | 2-3 best examples from `06_Results/figures/gradcam/` | a small selection, not the full species x freshness grid |

## Out of scope for the manuscript

Schedule, detailed hyperparameter justification, per-seed variance tables,
full 24-cell class distribution, and hardware specifications belong in
supplementary material only.
