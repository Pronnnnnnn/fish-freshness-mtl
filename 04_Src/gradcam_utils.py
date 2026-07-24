"""Grad-CAM (Selvaraju et al., ICCV 2017) computed separately per task head,
so each head's attention can be inspected independently -- in particular,
whether either head relies on background rather than the eye region.
"""
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import MultiTaskModel


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
    """Last conv layer before global pooling -- the backbone's `conv_head`."""
    return mtl_model.backbone.conv_head


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
