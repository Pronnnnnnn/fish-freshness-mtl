"""Builds the image-level split used as the leakage ablation's control.

This split deliberately ignores time_group and stratifies over individual
images, which is what lets bursts straddle the subsets. That is the whole
point: it reproduces the flawed protocol so its cost can be measured.

Everything else matches the main split -- 70/15/15, stratified on
combined_class, random_state 2026 -- so the split rule is the only thing
that differs between the two.

Never writes to 02_Manifests/; the main split lives there.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from split_utils import SPLIT_RANDOM_STATE

OUT_SPLIT = "08_Leakage_Ablation/02_manifests/split_manifest_imagelevel.csv"
OUT_DIAGNOSTICS = "08_Leakage_Ablation/02_manifests/imagelevel_split_diagnostics.txt"


def stratified_image_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = SPLIT_RANDOM_STATE,
) -> pd.DataFrame:
    """Splits per image row, with no regard for time_group."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    train_df, rest_df = train_test_split(
        df, train_size=train_size, stratify=df["combined_class"], random_state=random_state
    )
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        rest_df,
        train_size=relative_val_size,
        stratify=rest_df["combined_class"],
        random_state=random_state,
    )

    out = df.copy()
    subset = pd.Series("test", index=df.index)
    subset.loc[train_df.index] = "train"
    subset.loc[val_df.index] = "val"
    out["subset"] = subset
    return out


def diagnose(split_df: pd.DataFrame) -> str:
    """Reports the contamination this split produces. Quoted in the paper."""
    total = len(split_df)
    lines = [f"total images: {total}", "", "per subset:"]

    counts = split_df["subset"].value_counts()
    for subset in ["train", "val", "test"]:
        n = int(counts.get(subset, 0))
        lines.append(f"  {subset:<6} {n:>5} images ({n / total * 100:.1f}%)")

    spanning = split_df.groupby("time_group")["subset"].nunique()
    split_groups = spanning[spanning > 1]
    affected = split_df[split_df["time_group"].isin(split_groups.index)]

    lines += [
        "",
        f"time groups total: {split_df['time_group'].nunique()}",
        f"time groups spanning >1 subset: {len(split_groups)}",
        f"images in a spanning group: {len(affected)} ({len(affected) / total * 100:.1f}%)",
    ]

    # The figure that matters: test images whose burst also appears in training.
    train_groups = set(split_df.loc[split_df["subset"] == "train", "time_group"])
    test_df = split_df[split_df["subset"] == "test"]
    contaminated = test_df[test_df["time_group"].isin(train_groups)]
    lines += [
        "",
        f"test images with a same-burst twin in train: {len(contaminated)} "
        f"({len(contaminated) / len(test_df) * 100:.1f}% of the test set)",
    ]

    lines += ["", "class combinations represented per subset:"]
    for subset in ["train", "val", "test"]:
        n = split_df[split_df["subset"] == subset]["combined_class"].nunique()
        lines.append(f"  {subset:<6} {n}/24")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="02_Manifests/manifest_full.csv")
    parser.add_argument("--out", default=OUT_SPLIT)
    parser.add_argument("--diagnostics", default=OUT_DIAGNOSTICS)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    split_df = stratified_image_split(df)

    report = diagnose(split_df)
    print(report)

    for path, writer in ((args.out, None), (args.diagnostics, report)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.out, index=False)
    Path(args.diagnostics).write_text(report + "\n", encoding="utf-8")
    print(f"\nsaved split to {args.out}")
    print(f"saved diagnostics to {args.diagnostics}")


if __name__ == "__main__":
    main()
