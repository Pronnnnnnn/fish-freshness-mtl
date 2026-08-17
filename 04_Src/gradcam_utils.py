"""Grad-CAM (Selvaraju et al., ICCV 2017), computed per task head.

Strictly post-hoc: it inspects the chosen model, and never informs model
selection or any hyperparameter. Gradients are taken from raw logits, not
probabilities, and the model stays in eval() so dropout is off and the maps
are deterministic.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import MultiTaskModel

GRADCAM_SAMPLE_SEED = 2026
IMAGES_PER_COMBINATION = 2


class SingleHeadWrapper(nn.Module):
    """Exposes one head of a MultiTaskModel as a single-output classifier,
    which is what GradCAM expects to backpropagate through."""

    def __init__(self, mtl_model: MultiTaskModel, head: str):
        super().__init__()
        assert head in ("species", "freshness")
        self.mtl_model = mtl_model
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        species_logits, freshness_logits = self.mtl_model(x)
        return species_logits if self.head == "species" else freshness_logits


def get_target_layer(mtl_model: MultiTaskModel) -> nn.Module:
    """Last spatial feature map before global average pooling."""
    return mtl_model.backbone.conv_head


def select_median_seed(validation_joint_accuracy: dict[int, float]) -> int:
    """Picks the seed whose validation joint accuracy is the middle of three.

    Deliberately the median rather than the best, and measured on validation
    rather than test: the test set stays untouched by any selection decision,
    and a median instance represents typical behaviour instead of a
    favourable draw.
    """
    if not validation_joint_accuracy:
        raise ValueError("no seeds supplied")
    ordered = sorted(validation_joint_accuracy.items(), key=lambda kv: kv[1])
    return ordered[len(ordered) // 2][0]


def sample_images(test_df: pd.DataFrame, per_combination: int = IMAGES_PER_COMBINATION,
                  random_state: int = GRADCAM_SAMPLE_SEED) -> pd.DataFrame:
    """Stratified random sample, `per_combination` images per class combination.

    Fixed in advance and drawn at random rather than hand-picked, so the
    panel cannot be curated toward flattering examples.
    """
    return (
        test_df.groupby("combined_class", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_combination, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )


def compute_gradcam(
    mtl_model: MultiTaskModel,
    image_tensor: torch.Tensor,  # normalized, shape (3, 224, 224)
    image_rgb_float: np.ndarray,  # unnormalized, HWC, values in [0, 1], for overlay
    head: str,
    target_class: int | None = None,
) -> np.ndarray:
    """Returns an RGB uint8 image with the Grad-CAM heatmap overlaid.

    If target_class is None, uses the head's own top predicted class.
    """
    wrapped = SingleHeadWrapper(mtl_model, head)
    wrapped.eval()
    target_layers = [get_target_layer(mtl_model)]

    device = next(mtl_model.parameters()).device
    input_batch = image_tensor.unsqueeze(0).to(device)
    if target_class is None:
        with torch.no_grad():
            target_class = int(wrapped(input_batch).argmax(dim=1).item())

    with GradCAM(model=wrapped, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_batch, targets=[ClassifierOutputTarget(target_class)])[0]

    return show_cam_on_image(image_rgb_float, grayscale_cam, use_rgb=True)


def denormalize_for_display(image_tensor: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization -> HWC float array in [0, 1] for CAM overlay."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = (image_tensor * std + mean).clamp(0, 1)
    return denorm.permute(1, 2, 0).numpy()
