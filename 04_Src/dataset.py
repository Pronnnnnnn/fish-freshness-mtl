"""Dataset, transforms, and class-balanced sampler for the FFE manifest.

Class balancing is applied only to the training split, via a
WeightedRandomSampler over the 24-class combination rather than offline
oversampling -- no extra files are generated and nothing is lost when a
Colab session ends. Augmentation is applied online per draw so a given
source image yields a different variant each epoch. Validation/test
transforms only resize and normalize, so evaluation reflects the true
class distribution.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class FishEyeDataset(Dataset):
    """Reads (image, species_idx, freshness_idx, combined_idx) from a manifest.

    combined_idx is the flat 24-class label, carried alongside the two task
    labels so the same Dataset serves the single-task, flat, and multi-task
    models without a separate loader.
    """

    def __init__(self, manifest_df: pd.DataFrame, dataset_root: str | Path, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.dataset_root / row["filepath"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        species_idx = torch.tensor(row["species_idx"], dtype=torch.long)
        freshness_idx = torch.tensor(row["freshness_idx"], dtype=torch.long)
        combined_idx = torch.tensor(row["combined_idx"], dtype=torch.long)
        return image, species_idx, freshness_idx, combined_idx


def make_balanced_sampler(
    manifest_df: pd.DataFrame, generator: torch.Generator | None = None
) -> WeightedRandomSampler:
    """Inverse-frequency sampler over the 24-class combination.

    Takes a seeded generator so the draw sequence is reproducible per run.
    """
    class_counts = manifest_df["combined_class"].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = manifest_df["combined_class"].map(class_weights).values
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(manifest_df),
        replacement=True,
        generator=generator,
    )
