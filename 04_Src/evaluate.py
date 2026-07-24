"""Inference helpers: run a trained checkpoint over a held-out split and
measure parameter count / per-image inference latency.
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FishEyeDataset, eval_transform
from models import MultiTaskModel, SingleTaskModel


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_inference_time_ms(model: torch.nn.Module, device: str, num_warmup=10, num_runs=100) -> float:
    """Average single-image inference time in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)

    for _ in range(num_warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / num_runs) * 1000.0


def load_single_task_model(checkpoint_path: str, num_classes: int, device: str) -> torch.nn.Module:
    model = SingleTaskModel(num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_multitask_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    model = MultiTaskModel(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_single_task(model, test_df, dataset_root, task: str, device: str, batch_size: int = 32):
    ds = FishEyeDataset(test_df, dataset_root, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []
    for images, species_idx, freshness_idx in loader:
        labels = species_idx if task == "species" else freshness_idx
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.append(labels.numpy())
        y_pred.append(preds)
    return np.concatenate(y_true), np.concatenate(y_pred)


@torch.no_grad()
def predict_multitask(model, test_df, dataset_root, device: str, batch_size: int = 32):
    ds = FishEyeDataset(test_df, dataset_root, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    species_true, species_pred, freshness_true, freshness_pred = [], [], [], []
    for images, species_idx, freshness_idx in loader:
        sp_logits, fr_logits = model(images.to(device))
        species_true.append(species_idx.numpy())
        species_pred.append(sp_logits.argmax(dim=1).cpu().numpy())
        freshness_true.append(freshness_idx.numpy())
        freshness_pred.append(fr_logits.argmax(dim=1).cpu().numpy())

    return (
        np.concatenate(species_true),
        np.concatenate(species_pred),
        np.concatenate(freshness_true),
        np.concatenate(freshness_pred),
    )
