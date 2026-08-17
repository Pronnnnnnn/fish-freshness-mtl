"""Stratified train/val/test split at the time-group level.

Splitting on individual images leaks: a burst of frames shot in the same
second shows the same fish eye, so frames landing on opposite sides of the
split let the model recognise test images it effectively trained on. The
split therefore operates on time_group, keeping every frame of a burst in
one subset. A group never spans two class combinations, so stratification
on combined_class still applies -- it is just applied to groups.

The split is drawn once, written to 02_Manifests/split_manifest.csv, and
read back by every run. Its random state (SPLIT_RANDOM_STATE) is separate
from the training seeds on purpose: if the two shared a value, changing the
training seed would silently reshuffle the subsets and make every
cross-model comparison invalid.

Group sizes vary, so image-level proportions land near 70/15/15 rather than
exactly on it. That is expected and is not corrected for.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_RANDOM_STATE = 2026
SPLIT_MANIFEST_PATH = "02_Manifests/split_manifest.csv"
EXPECTED_TOTAL_IMAGES = 4390
EXPECTED_COMBINATIONS = 24


def _group_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per time_group, carrying the class it belongs to."""
    table = df.groupby("time_group", as_index=False)["combined_class"].first()
    n_classes = df.groupby("time_group")["combined_class"].nunique()
    straddling = n_classes[n_classes > 1]
    if not straddling.empty:
        raise ValueError(
            f"{len(straddling)} time_group(s) span more than one class combination; "
            "the group key must include the folder name"
        )
    return table


def stratified_group_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = SPLIT_RANDOM_STATE,
) -> pd.DataFrame:
    """Returns the manifest with a `subset` column of train/val/test."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    groups = _group_table(df)

    train_groups, rest_groups = train_test_split(
        groups,
        train_size=train_size,
        stratify=groups["combined_class"],
        random_state=random_state,
    )
    relative_val_size = val_size / (val_size + test_size)
    val_groups, test_groups = train_test_split(
        rest_groups,
        train_size=relative_val_size,
        stratify=rest_groups["combined_class"],
        random_state=random_state,
    )

    assignment = {}
    for subset, table in (
        ("train", train_groups),
        ("val", val_groups),
        ("test", test_groups),
    ):
        assignment.update(dict.fromkeys(table["time_group"], subset))

    out = df.copy()
    out["subset"] = out["time_group"].map(assignment)
    return out


def verify_split(split_df: pd.DataFrame) -> dict:
    """Raises if any of the three mandatory guarantees is violated."""
    spanning = split_df.groupby("time_group")["subset"].nunique()
    leaked = spanning[spanning > 1]
    if not leaked.empty:
        raise ValueError(
            f"{len(leaked)} time_group(s) appear in more than one subset -- "
            "this is the leakage the group split exists to prevent"
        )

    per_subset = split_df.groupby("subset")["combined_class"].nunique()
    missing = per_subset[per_subset < EXPECTED_COMBINATIONS]
    if not missing.empty or len(per_subset) != 3:
        raise ValueError(
            f"expected all {EXPECTED_COMBINATIONS} class combinations in each of the "
            f"3 subsets, got: {per_subset.to_dict()}"
        )

    if len(split_df) != EXPECTED_TOTAL_IMAGES:
        raise ValueError(f"expected {EXPECTED_TOTAL_IMAGES} images, got {len(split_df)}")

    counts = split_df["subset"].value_counts()
    return {
        "images": counts.to_dict(),
        "image_pct": (counts / len(split_df) * 100).round(1).to_dict(),
        "groups": split_df.groupby("subset")["time_group"].nunique().to_dict(),
    }


def save_split(split_df: pd.DataFrame, path: str | Path = SPLIT_MANIFEST_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)


def load_split(path: str | Path = SPLIT_MANIFEST_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads the saved split back as (train_df, val_df, test_df)."""
    split_df = pd.read_csv(path)
    return (
        split_df[split_df["subset"] == "train"].reset_index(drop=True),
        split_df[split_df["subset"] == "val"].reset_index(drop=True),
        split_df[split_df["subset"] == "test"].reset_index(drop=True),
    )
