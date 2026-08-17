"""Build a labeled image manifest from the raw FFE directory layout.

The raw dataset encodes both labels in each folder name, e.g.
"Chanos Chanos - Fresh". One folder ships with a double-space typo
("Nibea Albiflora -  Highly Fresh"); whitespace is collapsed before
parsing so it does not produce a spurious 25th class.
"""
import re
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Freshness is ordinal; encode by storage progression, not alphabetically,
# so downstream ordinal metrics (e.g. quadratic weighted kappa) are meaningful.
FRESHNESS_ORDER = ["Highly Fresh", "Fresh", "Not Fresh"]
NUM_FRESHNESS = len(FRESHNESS_ORDER)

# Android camera filenames: IMG_YYYYMMDD_HHMMSS, sometimes with a suffix.
# Matched as a prefix so anything trailing the seconds field is ignored.
TIMESTAMP_PREFIX = re.compile(r"^(IMG_\d{8}_\d{6})")


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Split "Species - Freshness" into (species, freshness)."""
    normalized = re.sub(r"\s+", " ", folder_name).strip()
    species, _, freshness = normalized.partition(" - ")
    if not freshness:
        raise ValueError(f"cannot parse species/freshness from folder name: {folder_name!r}")
    return species.strip(), freshness.strip()


def time_group_key(folder_name: str, file_name: str) -> str:
    """Group key for images shot in the same burst (same folder, same second).

    Bursts of the same fish eye taken seconds apart are near-duplicates, so
    they must not straddle train/val/test. Files whose timestamp is malformed
    (3 in this dataset, five-digit seconds fields) fall back to their full
    filename, which puts each in a group of its own rather than failing.
    """
    stem = file_name.rsplit(".", 1)[0]
    match = TIMESTAMP_PREFIX.match(stem)
    return f"{folder_name}/{match.group(1)}" if match else f"{folder_name}/{file_name}"


def build_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Scan dataset_root and return one row per image.

    Columns: filepath (relative to dataset_root, portable across
    environments), species, freshness, species_idx, freshness_idx,
    combined_class, combined_idx, time_group.
    """
    dataset_root = Path(dataset_root)
    rows = []
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        species, freshness = parse_folder_name(folder.name)
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                relative_path = file.relative_to(dataset_root)
                rows.append(
                    {
                        "filepath": relative_path.as_posix(),
                        "species": species,
                        "freshness": freshness,
                        "time_group": time_group_key(folder.name, file.name),
                    }
                )

    df = pd.DataFrame(rows)

    species_list = sorted(df["species"].unique())
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    freshness_to_idx = {f: i for i, f in enumerate(FRESHNESS_ORDER)}

    df["species_idx"] = df["species"].map(species_to_idx)
    df["freshness_idx"] = df["freshness"].map(freshness_to_idx)
    df["combined_class"] = df["species"] + " | " + df["freshness"]
    # Freshness occupies the units digit so its ordinal order survives the
    # round trip through the flat 24-class label (see combined_to_tasks).
    df["combined_idx"] = df["species_idx"] * NUM_FRESHNESS + df["freshness_idx"]

    return df


def combined_to_tasks(combined_idx):
    """Inverse of the flat 24-class encoding -> (species_idx, freshness_idx)."""
    return combined_idx // NUM_FRESHNESS, combined_idx % NUM_FRESHNESS


def summarize_time_groups(df: pd.DataFrame) -> dict:
    """Burst-group statistics used to sanity-check the manifest."""
    counts = df["time_group"].value_counts()
    multi = counts[counts > 1]
    images_in_multi = int(multi.sum())
    return {
        "total_images": len(df),
        "unique_groups": int(counts.size),
        "multi_image_groups": int(multi.size),
        "images_in_multi_image_groups": images_in_multi,
        "pct_images_in_multi_image_groups": round(images_in_multi / len(df) * 100, 1),
        "mean_images_per_multi_group": round(images_in_multi / multi.size, 2),
    }


def summarize_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Species x freshness image counts, with row/column totals."""
    pivot = df.pivot_table(
        index="species", columns="freshness", values="filepath", aggfunc="count", fill_value=0
    )
    pivot = pivot[[c for c in FRESHNESS_ORDER if c in pivot.columns]]
    pivot["Total"] = pivot.sum(axis=1)
    pivot.loc["Total"] = pivot.sum(axis=0)
    return pivot
