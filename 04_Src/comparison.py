"""Model comparison via paired per-seed differences.

Every configuration is trained under the same three seeds and evaluated on
the same test set, so for a given seed two configurations form a matched
pair. Comparing those pairs directly keeps the pairing information that
contrasting two independent means would discard.

No numeric threshold is applied. With n=3 a rule such as "mean difference
must exceed its standard deviation" has no theoretical basis and reads as a
significance test without being one. A configuration is instead reported as
a candidate winner only when its paired difference carries the same sign on
all three seeds, with the mean and standard deviation of the differences
reported alongside for magnitude. Formal tests (paired t-test) are not used:
n=3 has no meaningful power.
"""
import pandas as pd

RESULT_COLUMNS = ["model", "seed", "task", "metric", "value"]


def to_long_format(rows: list[dict]) -> pd.DataFrame:
    """Normalises per-run metric dicts into one row per model/seed/task/metric."""
    records = []
    for row in rows:
        model, seed = row["model"], row["seed"]
        for task, metrics in row["tasks"].items():
            for metric, value in metrics.items():
                records.append(
                    {"model": model, "seed": seed, "task": task, "metric": metric, "value": value}
                )
    return pd.DataFrame(records, columns=RESULT_COLUMNS)


def _series(long_df: pd.DataFrame, model: str, task: str, metric: str) -> pd.Series:
    """Metric values for one configuration, indexed by seed."""
    sel = long_df[
        (long_df["model"] == model) & (long_df["task"] == task) & (long_df["metric"] == metric)
    ]
    if sel.empty:
        raise KeyError(f"no rows for model={model!r} task={task!r} metric={metric!r}")
    return sel.set_index("seed")["value"].sort_index()


def paired_difference(
    long_df: pd.DataFrame, model_a: str, model_b: str, task: str, metric: str
) -> dict:
    """Per-seed differences (model_b - model_a) on one metric.

    `consistent_sign` is True when model_b beats model_a on every seed, or
    loses on every seed -- the reporting criterion. It is the only verdict
    produced; no threshold is applied to the magnitude.
    """
    a = _series(long_df, model_a, task, metric)
    b = _series(long_df, model_b, task, metric)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError(f"{model_a} and {model_b} share no seeds")

    diffs = (b.loc[common] - a.loc[common]).sort_index()
    positive = (diffs > 0).all()
    negative = (diffs < 0).all()

    return {
        "task": task,
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "n_seeds": len(common),
        "per_seed_differences": diffs.to_dict(),
        "mean_difference": float(diffs.mean()),
        "std_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan"),
        "consistent_sign": bool(positive or negative),
        "direction": "b_better" if positive else ("a_better" if negative else "mixed"),
    }


def compare_many(
    long_df: pd.DataFrame, pairs: list[tuple[str, str]], task: str, metric: str
) -> pd.DataFrame:
    """paired_difference over several model pairs, one row each."""
    rows = []
    for model_a, model_b in pairs:
        d = paired_difference(long_df, model_a, model_b, task, metric)
        d["per_seed_differences"] = str(d["per_seed_differences"])
        rows.append(d)
    return pd.DataFrame(rows)


def stl_vs_mtl(long_df: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Stage 1 -- does folding the two tasks together cost either of them?

    Model A against each MTL variant on species, Model B against each on
    freshness: 6 comparison points.
    """
    mtl = ["ModelD_EW", "ModelD_UW", "ModelD_DWA"]
    species = compare_many(long_df, [("ModelA_species", m) for m in mtl], "species", metric)
    freshness = compare_many(long_df, [("ModelB_freshness", m) for m in mtl], "freshness", metric)
    return pd.concat([species, freshness], ignore_index=True)


def flat_vs_mtl(long_df: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Stage 2 -- flat 24-class formulation against the two-head one."""
    mtl = ["ModelD_EW", "ModelD_UW", "ModelD_DWA"]
    pairs = [("ModelC_flat24", m) for m in mtl]
    frames = [
        compare_many(long_df, pairs, "species", metric),
        compare_many(long_df, pairs, "freshness", metric),
        compare_many(long_df, pairs, "joint", "joint_accuracy"),
    ]
    return pd.concat(frames, ignore_index=True)


def strategy_pairs(long_df: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Stage 3 -- the three weighting strategies against each other."""
    pairs = [
        ("ModelD_EW", "ModelD_UW"),
        ("ModelD_EW", "ModelD_DWA"),
        ("ModelD_UW", "ModelD_DWA"),
    ]
    return compare_many(long_df, pairs, task, metric)


def contaminated_vs_clean(
    test_df: pd.DataFrame,
    train_time_groups: set,
    y_true,
    y_pred,
    metric_fn,
) -> dict:
    """Scores one model separately on the leaked and unleaked parts of its test set.

    A cross-split comparison confounds leakage with the two splits having
    different test sets. Here the model, its training, and the metric are
    held fixed, and only the rows differ: test images whose burst also
    appears in training, against those where it does not. The gap is the
    leakage effect with nothing else moving.

    Class composition can still differ between the two portions, so the
    per-portion class counts are returned for inspection.
    """
    import numpy as np

    is_contaminated = test_df["time_group"].isin(train_time_groups).to_numpy()
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if is_contaminated.sum() == 0 or (~is_contaminated).sum() == 0:
        raise ValueError("one portion is empty; this split has no leakage to measure")

    contaminated = metric_fn(y_true[is_contaminated], y_pred[is_contaminated])
    clean = metric_fn(y_true[~is_contaminated], y_pred[~is_contaminated])

    return {
        "n_contaminated": int(is_contaminated.sum()),
        "n_clean": int((~is_contaminated).sum()),
        "score_contaminated": float(contaminated),
        "score_clean": float(clean),
        "inflation": float(contaminated - clean),
        "classes_contaminated": test_df.loc[is_contaminated, "combined_class"].nunique(),
        "classes_clean": test_df.loc[~is_contaminated, "combined_class"].nunique(),
    }


def contaminated_vs_clean_per_class(
    test_df: pd.DataFrame,
    train_time_groups: set,
    y_true,
    y_pred,
    label_column: str,
) -> pd.DataFrame:
    """Per-class accuracy on the leaked vs unleaked parts of one test set.

    The pooled version of this comparison is confounded whenever the two
    portions differ in composition, which they do here: bursts are not spread
    evenly over the classes, so three species never appear in the leaked
    portion at all and the freshness mix is skewed toward one level. Comparing
    within each class removes that, since each row holds the class fixed and
    varies only whether the image had a twin in training.

    Classes missing from either portion are returned with NaN so they are
    visibly excluded rather than silently averaged in.
    """
    import numpy as np

    work = test_df.reset_index(drop=True).copy()
    work["_correct"] = np.asarray(y_true) == np.asarray(y_pred)
    work["_contaminated"] = work["time_group"].isin(train_time_groups)

    rows = []
    for label, group in work.groupby(label_column):
        contaminated = group[group["_contaminated"]]
        clean = group[~group["_contaminated"]]
        rows.append(
            {
                "class": label,
                "n_contaminated": len(contaminated),
                "n_clean": len(clean),
                "accuracy_contaminated": contaminated["_correct"].mean() if len(contaminated) else float("nan"),
                "accuracy_clean": clean["_correct"].mean() if len(clean) else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    out["inflation"] = out["accuracy_contaminated"] - out["accuracy_clean"]
    out["comparable"] = out["n_contaminated"].gt(0) & out["n_clean"].gt(0)
    return out


def efficiency_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Parameters and inference time per model, plus speedup over A+B.

    The realistic alternative to one multi-task model is running the two
    single-task models in sequence, so the baseline is their sum.
    """
    eff = long_df[long_df["task"] == "efficiency"]
    wide = eff.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")

    baseline_params = wide.loc["ModelA_species", "params"] + wide.loc["ModelB_freshness", "params"]
    baseline_time = (
        wide.loc["ModelA_species", "inference_ms"] + wide.loc["ModelB_freshness", "inference_ms"]
    )

    wide["params_vs_A_plus_B"] = baseline_params - wide["params"]
    wide["params_savings_pct"] = (baseline_params - wide["params"]) / baseline_params * 100
    wide["speedup_vs_A_plus_B"] = baseline_time / wide["inference_ms"]
    return wide.round(4)
