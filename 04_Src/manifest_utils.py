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


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Split "Species - Freshness" into (species, freshness)."""
    normalized = re.sub(r"\s+", " ", folder_name).strip()
    species, _, freshness = normalized.partition(" - ")
    if not freshness:
        raise ValueError(f"cannot parse species/freshness from folder name: {folder_name!r}")
    return species.strip(), freshness.strip()


def build_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Scan dataset_root and return one row per image.

    Columns: filepath (relative to dataset_root, portable across
    environments), species, freshness, species_idx, freshness_idx,
    combined_class.
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
                    {"filepath": relative_path.as_posix(), "species": species, "freshness": freshness}
                )

    df = pd.DataFrame(rows)

    species_list = sorted(df["species"].unique())
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    freshness_to_idx = {f: i for i, f in enumerate(FRESHNESS_ORDER)}

    df["species_idx"] = df["species"].map(species_to_idx)
    df["freshness_idx"] = df["freshness"].map(freshness_to_idx)
    df["combined_class"] = df["species"] + " | " + df["freshness"]

    return df


def summarize_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Species x freshness image counts, with row/column totals."""
    pivot = df.pivot_table(
        index="species", columns="freshness", values="filepath", aggfunc="count", fill_value=0
    )
    pivot = pivot[[c for c in FRESHNESS_ORDER if c in pivot.columns]]
    pivot["Total"] = pivot.sum(axis=1)
    pivot.loc["Total"] = pivot.sum(axis=0)
    return pivot
