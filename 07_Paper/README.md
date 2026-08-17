# Manuscript Notes

Target venue: ICELTICs 2026, IEEE two-column format, 6 pages including
figures, tables, and references. Official template:
https://iceltics.usk.ac.id/paper-template/

`SPEC.md` at the repository root defines the experimental design the
manuscript reports on.

## Section-to-output mapping

| Section | Source | Notes |
|---|---|---|
| Related work | literature comparison table | condense to the handful of directly comparable results, not an exhaustive survey |
| Data | `02_Manifests/manifest_full.csv` | totals per species and per freshness level, not the full 24-cell breakdown |
| Method - splitting | `02_Manifests/split_manifest.csv`, group statistics from `summarize_time_groups` | one or two sentences: bursts share a timestamp, so the split is by time group; worth stating explicitly, since prior work on this dataset does not report controlling for it |
| Method - architecture | one diagram covering the shared backbone and the head variants | a single figure spanning A/B, C, and D |
| Method - loss weighting | EW/UW/DWA equations from `04_Src/loss_weighting.py` | equations only, no derivation |
| Results - main table | `test_results_long.csv`, aggregated per model | 6 configurations x {accuracy, macro F1, joint accuracy}; MCC/Kappa/QWK in text if space is tight |
| Results - comparison | `comparison_stage1/2/3_*.csv` | report the sign-consistency verdict with mean +/- std of the paired differences; do not phrase it as a significance test |
| Results - efficiency | `comparison_efficiency.csv` | one sentence, or two extra columns on the main table; the baseline is A + B combined, not either alone |
| Results - convergence | `figures/curves/<run>.png` + `training_summary.csv` | at most 1 representative curve plus one sentence on epoch range and training time; not all 18 |
| Results - interpretability | 2-3 examples from `figures/gradcam/` + `gradcam_provenance.txt` | a small selection from the 48; state which seed and that it was the validation median |

## Points worth making explicitly

- The time-group split is a methodological contribution in its own right:
  under an image-level split on this dataset, 33.7% of test images have a
  same-burst twin in training, so any published figure that did not control
  for it is optimistic by an unknown margin. This reframes comparisons with
  prior work on FFE.
- Model C exists to test the problem formulation (one 24-way label) against
  the two-head one; it is a baseline, not a competing proposal.
- A null result on the weighting strategies is a result. If the paired
  differences do not hold their sign, report the three as
  indistinguishable at this sample size rather than ranking them.

## Out of scope for the manuscript

Schedule, detailed hyperparameter justification, per-seed tables, the full
24-cell class distribution, hardware specifications, and all 18 training
curves belong in supplementary material.
