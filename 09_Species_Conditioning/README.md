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

## Design note

The species probabilities entering the freshness head are detached. Without
that, the freshness loss would also train the species head, D-Cond's species
head would no longer be comparable to D-EW's, and any change in freshness
could not be attributed to conditioning alone. The detach is load-bearing,
not a tidiness choice.

Writes only under `09_Species_Conditioning/`. The main results in
`05_Checkpoints/`, `06_Results/`, and the ablation in `08_Leakage_Ablation/`
are untouched.
