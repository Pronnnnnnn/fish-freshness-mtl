# Leakage Ablation

Quantifies how much the reported metrics move when the split is made at the
image level instead of the time-group level.

The main experiment splits by time group because frames sharing a camera
timestamp are a burst of the same fish eye; under an image-level split,
33.7% of test images have a same-burst twin in training. That fact alone
says leakage exists. This folder measures what it is worth in metric points,
so the paper can state the size of the effect rather than only its presence.

Model A (species) and Model B (freshness) are retrained on an image-level
split under identical hyperparameters and seeds -- the split file is the only
thing that differs. Model C and the D variants are not retrained; the two
single-task models are enough to size the effect on both tasks.

```
01_uw_weights/    learned log-variances from the main D-UW checkpoints
02_manifests/     the image-level split used here
03_checkpoints/   6 ablation checkpoints (not committed)
04_results/       metrics and the comparison table
```

Nothing here writes to `02_Manifests/`, `05_Checkpoints/`, or `06_Results/`;
those hold the main results.

## Reading the comparison

The two splits have **different test sets**, so a difference between them
mixes the leakage effect with the difference in test composition. Treat the
cross-split numbers as indicative of magnitude, not as a controlled
measurement. The within-model contrast in `08_leakage_ablation_comparison`
(contaminated vs clean portions of the same test set, scored by the same
model) is the cleaner estimate and is the one to lead with.
