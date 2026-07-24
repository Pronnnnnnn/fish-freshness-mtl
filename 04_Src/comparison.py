"""Model comparison procedure (two stages):

1. Single-task vs multi-task, per task, per loss-weighting strategy.
2. Head-to-head ranking of the three loss-weighting strategies.

With only 3 seeds per model, a formal significance test has no real
power, so a difference is only called meaningful if it exceeds the
larger of the two configurations' seed-to-seed standard deviations --
otherwise the configurations are treated as equivalent.
"""
import pandas as pd


def exceeds_seed_variation(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """True if the two means differ by more than the larger of the two stds."""
    return abs(mean_a - mean_b) > max(std_a, std_b)


def compare_stl_vs_mtl(stl_mean: float, stl_std: float, mtl_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """One row per MTL strategy: STL mean/std vs that strategy's mean/std on `metric_col`."""
    summary = mtl_df.groupby("model")[metric_col].agg(["mean", "std"])
    rows = []
    for strategy, row in summary.iterrows():
        rows.append(
            {
                "strategy": strategy,
                "stl_mean": stl_mean,
                "stl_std": stl_std,
                "mtl_mean": row["mean"],
                "mtl_std": row["std"],
                "delta": row["mean"] - stl_mean,
                "exceeds_seed_variation": exceeds_seed_variation(stl_mean, stl_std, row["mean"], row["std"]),
            }
        )
    return pd.DataFrame(rows)


def compare_efficiency(stl_value: float, mtl_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Same shape as compare_stl_vs_mtl but for a value with no seed variance to test
    (parameter count and inference time are near-deterministic given a fixed architecture)."""
    summary = mtl_df.groupby("model")[value_col].mean()
    return pd.DataFrame(
        {
            "strategy": summary.index,
            "stl_value": stl_value,
            "mtl_value": summary.values,
            "savings": stl_value - summary.values,
            "savings_pct": (stl_value - summary.values) / stl_value * 100,
        }
    )


def compare_strategies(multitask_df: pd.DataFrame, metric_col: str) -> dict:
    """Ranks EW/UW/DWA on one metric; flags whether the leader's margin
    over the runner-up exceeds seed variation."""
    summary = multitask_df.groupby("model")[metric_col].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    )
    best, runner_up = summary.index[0], summary.index[1]
    best_mean, best_std = summary.loc[best]
    runner_mean, runner_std = summary.loc[runner_up]
    return {
        "metric": metric_col,
        "best_strategy": best,
        "best_mean": best_mean,
        "runner_up": runner_up,
        "runner_up_mean": runner_mean,
        "difference": best_mean - runner_mean,
        "exceeds_seed_variation": exceeds_seed_variation(best_mean, best_std, runner_mean, runner_std),
    }
