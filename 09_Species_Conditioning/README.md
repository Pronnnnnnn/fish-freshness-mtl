# Species Conditioning

Tests whether giving the freshness head explicit access to the species
prediction closes the gap between the multi-task models and Model C on the
freshness task.

The main experiment left an unexplained result: Model C, which predicts the
species/freshness pair as one 24-way label, beat the two-head models on
freshness. One reading is that the flat formulation lets freshness be judged
*conditional on species* -- plausible, since the organoleptic standard the
study cites defines a fresh pupil as species-specific, so what "fresh" looks
like is not the same across species. The two-head models have no such
channel: their freshness head sees only the shared features.

D-Cond adds that channel. Its freshness head receives the backbone features
concatenated with the species head's softmax output, so it can condition on
which species it is looking at. Everything else matches D-EW.

If conditioning is what Model C was exploiting, D-Cond should recover part
of that advantage. If it does not, the flat model's edge comes from
something else, and that is equally worth reporting.

```
01_checkpoints/   3 checkpoints (not committed)
02_results/       metrics, paired comparison, per-species breakdown
03_timing/        inference time on the same T4 as the main measurement
```

## Result

Conditioning helps, and it helps only where it should.

Against D-EW, which it matches in everything but the conditioning channel,
D-Cond is ahead on freshness on all three seeds: macro F1 +0.0243, QWK
+0.0271, joint accuracy +0.0233. Species is unchanged -- mean difference
4.6e-05, sign mixed -- which is what the detach predicts and is worth stating
as a check rather than a finding: had species moved, the freshness gain could
not have been attributed to conditioning.

Against Model C the gap is no longer detectable. Freshness F1, QWK, and joint
accuracy are all sign-inconsistent across seeds, so neither model is reported
as ahead; D-Cond is meanwhile ahead of Model C on species on all three seeds
(+0.0048). The flat model's freshness advantage over the two-head models was
therefore at least partly the conditioning it gets for free from predicting
the pair jointly -- it does not require the flat formulation, and it can be
had without the flat formulation's cost on species.

## Where the gain lands

Freshness accuracy per species, against the number of training images each
species has (`02_results/dcond_per_species.csv`). Seven of the eight species
fall in almost perfect order: the gain over D-EW runs from +0.0741 for
Johnius Trachycephalus, the rarest at 168 training images, down through zero
to -0.0137 for Oreochromis Niloticus at 540. Spearman correlation over those
seven is -0.99.

The eighth breaks it. Upeneus Moluccensis has the most training images of any
species (558) and still gains +0.0580, which drags the correlation over all
eight to -0.48. Its D-EW baseline explains why: at 0.7101 it is the weakest
species in the table despite the data, so it is intrinsically the hardest
freshness call, not the least-observed one. Ranking the species by baseline
accuracy instead of by training-set size correlates more strongly with the
gain (-0.79), and the two predictors are nearly independent of each other
(+0.26).

The more defensible reading is therefore the broader one: conditioning helps
where the unconditioned freshness head was weakest, and scarcity of training
images is one route to being weak rather than the mechanism itself. Both
correlations are descriptive only -- eight species, and test sets as small as
36 images, where a 0.05 difference is under two images.

## Cost

24 parameters, 5,872,819 against D-EW's 5,872,795, a 0.0004% increase; the
conditioning layer is 8x3 weights. Latency on the same Tesla T4 as the main
measurement is 11.71 ms against D-EW's 10.69, a difference that sits inside
the seed-to-seed spread of the measurement itself (see
`06_Results/metrics/inference_timings.csv`, where D-EW alone ranges 9.54 to
12.00 ms across seeds).

## Design note

The species probabilities entering the freshness head are detached. Without
that, the freshness loss would also train the species head, D-Cond's species
head would no longer be comparable to D-EW's, and any change in freshness
could not be attributed to conditioning alone. The detach is load-bearing,
not a tidiness choice.

Writes only under `09_Species_Conditioning/`. The main results in
`05_Checkpoints/`, `06_Results/`, and the ablation in `08_Leakage_Ablation/`
are untouched.
