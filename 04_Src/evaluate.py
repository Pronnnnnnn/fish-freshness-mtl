"""Inference helpers: run a trained checkpoint over a held-out split and
measure parameter count / per-image inference latency.

Raw test logits are saved alongside the metrics so probabilities or any
later analysis can be derived without re-running inference.
"""
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FishEyeDataset, eval_transform
from manifest_utils import combined_to_tasks
from models import FlatCombinedModel, MultiTaskModel, SingleTaskModel


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_inference_time_ms(
    model: torch.nn.Module, device: str, num_warmup: int = 20, num_runs: int = 100
) -> dict:
    """Single-image forward-pass latency, as {mean_ms, std_ms, n}.

    Times one forward pass only, on a tensor already resident in GPU memory:
    file loading, preprocessing, and host-to-device transfer are excluded.
    Each iteration is timed individually so a standard deviation can be
    reported, and synchronised on both sides -- without that, what gets timed
    is kernel scheduling rather than completion.
    """
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)

    for _ in range(num_warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    timings = np.asarray(timings)
    return {
        "mean_ms": float(timings.mean()),
        "std_ms": float(timings.std(ddof=1)),
        "n": int(num_runs),
    }


def save_test_logits(logits: np.ndarray, run_name: str, results_dir: str | Path) -> Path:
    """Persists raw test logits so probabilities need no second inference pass."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run_name}_test_logits.npy"
    np.save(path, logits)
    return path


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


def load_flat24_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    model = FlatCombinedModel(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_flat24(model, test_df, dataset_root, device: str, batch_size: int = 32):
    """Returns (species_true, species_pred, freshness_true, freshness_pred, logits).

    The 24-way prediction is mapped back to the two tasks so its metrics are
    computed exactly as the multi-task models' are.
    """
    ds = FishEyeDataset(test_df, dataset_root, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    combined_true, combined_pred, all_logits = [], [], []
    for images, species_idx, freshness_idx, combined_idx in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu().numpy())
        combined_true.append(combined_idx.numpy())
        combined_pred.append(logits.argmax(dim=1).cpu().numpy())

    combined_true = np.concatenate(combined_true)
    combined_pred = np.concatenate(combined_pred)
    species_true, freshness_true = combined_to_tasks(combined_true)
    species_pred, freshness_pred = combined_to_tasks(combined_pred)
    return (
        species_true,
        species_pred,
        freshness_true,
        freshness_pred,
        np.concatenate(all_logits),
    )


@torch.no_grad()
def predict_single_task(model, test_df, dataset_root, task: str, device: str, batch_size: int = 32):
    ds = FishEyeDataset(test_df, dataset_root, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    y_true, y_pred, all_logits = [], [], []
    for images, species_idx, freshness_idx, combined_idx in loader:
        labels = species_idx if task == "species" else freshness_idx
        logits = model(images.to(device))
        all_logits.append(logits.cpu().numpy())
        y_true.append(labels.numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(all_logits)


@torch.no_grad()
def predict_multitask(model, test_df, dataset_root, device: str, batch_size: int = 32):
    ds = FishEyeDataset(test_df, dataset_root, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    species_true, species_pred, freshness_true, freshness_pred = [], [], [], []
    species_logits, freshness_logits = [], []
    for images, species_idx, freshness_idx, combined_idx in loader:
        sp_logits, fr_logits = model(images.to(device))
        species_logits.append(sp_logits.cpu().numpy())
        freshness_logits.append(fr_logits.cpu().numpy())
        species_true.append(species_idx.numpy())
        species_pred.append(sp_logits.argmax(dim=1).cpu().numpy())
        freshness_true.append(freshness_idx.numpy())
        freshness_pred.append(fr_logits.argmax(dim=1).cpu().numpy())

    # Heads have different widths (8 vs 3), so the two logit blocks are
    # concatenated side by side; columns 0:8 are species, 8:11 freshness.
    logits = np.concatenate(
        [np.concatenate(species_logits), np.concatenate(freshness_logits)], axis=1
    )
    return (
        np.concatenate(species_true),
        np.concatenate(species_pred),
        np.concatenate(freshness_true),
        np.concatenate(freshness_pred),
        logits,
    )
