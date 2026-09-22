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

Three estimates, in decreasing order of how much weight they carry:

1. **Within-model, per class** (`within_model_leakage_summary.csv`) -- the
   controlled one. Same model, same class, varying only whether an image had
   a same-burst twin in training. Lead with this.
2. **Within-model, pooled** (`within_model_leakage_effect.csv`) -- the same
   contrast without holding class fixed. Confounded here, and kept only to
   show why the per-class version is needed.
3. **Cross-split** (`leakage_effect_comparison.csv`) -- image-level against
   group-level. The two splits have different test sets, so this mixes
   leakage with test composition. Indicative of magnitude only.

## Findings

**Freshness: leakage inflates macro-recall by 12.0 points**
(+0.1199, std 0.0236; positive in all 3 classes x 3 seeds, 9 of 9 cells).
Images with a twin in training are recognised 12 points better than images
without, with the class held fixed. The metric is per-class recall
macro-averaged, so state it as recall, not accuracy.

**Species: no inflation detectable.** The per-class figure is negative
(-0.0247), which leakage cannot produce. The clean portion scores 1.000 on
most classes -- the task is saturated, leaving no headroom for leakage to
show. What surfaces instead is that burst images are intrinsically harder:
a photographer takes several frames precisely when one will not do. The
largest drop is on *Oreochromis niloticus*, the genus twin of
*O. mossambicus* and the confusion the study flags as most likely. Do not
report this as leakage hurting species performance.

Both effects corroborate the main experiment, where fixing the split left
species F1 unchanged (0.9894 to 0.9942) and cut freshness F1 sharply
(0.786 to 0.734).

Composition is why the pooled estimate misleads: bursts are not spread
evenly over the classes. Three species -- *Chanos chanos*, *Eleutheronema
tetradactylum*, *Johnius trachycephalus* -- never appear in the leaked
portion at all, and its freshness mix is 54% Highly Fresh against 34% in the
clean portion.

## UW learned weights

`01_uw_weights/uw_learned_weights.csv` records what Uncertainty Weighting
converged to. The effective weight ratio between the two tasks is 0.955,
0.958, 0.957 across the three seeds -- within 4.5% of equal, and stable to
0.003 between seeds. UW settled on near-equal weights although nothing
forced it to, which is the mechanism behind the three weighting strategies
being indistinguishable: not that their metrics happened to land close
together, but that UW ends up where EW starts.

Both log-variances moved negative from their zero initialisation, so both
weights rose above 1 (1.27 to 1.65): UW scaled both losses up together
rather than separating the tasks. Freshness, the harder task, ended with
marginally the larger weight -- the opposite of the "downweight uncertain
tasks" reading of UW, though the margin is small enough to be noise around
equality.
