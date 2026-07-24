"""Stratified train/val/test split on the combined species x freshness label.

Every image carries exactly one species and one freshness label, giving
24 populated combinations, so a plain stratified split on `combined_class`
is equivalent to a multilabel stratified split here. The split runs before
any class balancing or augmentation to prevent samples derived from the
same source image from ending up in different subsets.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    train_df, rest_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df["combined_class"],
        random_state=random_state,
    )

    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        rest_df,
        train_size=relative_val_size,
        stratify=rest_df["combined_class"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
