"""Backbone and head definitions.

All model variants share the same EfficientNetV2-B0 backbone (ImageNet
pretrained, fine-tuned end-to-end) and the same head design: global
average pooling -> dropout(0.2) -> one linear layer per task. The only
difference between variants is the number of heads and how their losses
are combined (see loss_weighting.py).

EfficientNetV2-B0 is loaded via `timm`; torchvision only ships the
V2-S/M/L variants.
"""
import timm
import torch
import torch.nn as nn

NUM_SPECIES = 8
NUM_FRESHNESS = 3
NUM_COMBINED = NUM_SPECIES * NUM_FRESHNESS
BACKBONE_NAME = "tf_efficientnetv2_b0"


def build_backbone(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (backbone, num_features); backbone forward() yields pooled features."""
    backbone = timm.create_model(
        BACKBONE_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    return backbone, backbone.num_features


class _Head(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(pooled_features))


class SingleTaskModel(nn.Module):
    """Single backbone, single head -- species-only or freshness-only classifier."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone, num_features = build_backbone(pretrained)
        self.head = _Head(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(x)
        return self.head(pooled)


class FlatCombinedModel(nn.Module):
    """One head over all 24 species x freshness combinations.

    A baseline for the problem formulation itself: it predicts the pair
    jointly instead of factorising it into two heads. Its predictions are
    mapped back to (species, freshness) before evaluation so every metric is
    computed exactly as it is for the multi-task models.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone, num_features = build_backbone(pretrained)
        self.head = _Head(num_features, NUM_COMBINED)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(x)
        return self.head(pooled)


class MultiTaskModel(nn.Module):
    """Shared backbone with two parallel heads (species, freshness)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone, num_features = build_backbone(pretrained)
        self.species_head = _Head(num_features, NUM_SPECIES)
        self.freshness_head = _Head(num_features, NUM_FRESHNESS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.backbone(x)
        species_logits = self.species_head(pooled)
        freshness_logits = self.freshness_head(pooled)
        return species_logits, freshness_logits
