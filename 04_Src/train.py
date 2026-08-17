"""Training loop for single-task and multi-task models.

AdamW with cosine annealing, a fixed learning rate across all models
(fairness across loss-weighting strategies), full fine-tuning (no
frozen layers), cross-entropy per task. The best checkpoint and the
early-stopping decision are both driven by validation macro F1 -- for
the multi-task model, the mean of the two heads' F1 -- not validation
loss.
"""
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import FishEyeDataset, eval_transform, make_balanced_sampler, train_transform
from loss_weighting import DynamicWeightAveraging, build_loss_strategy
from manifest_utils import combined_to_tasks
from models import FlatCombinedModel, MultiTaskModel, SingleTaskModel


def set_seed(seed: int) -> None:
    """Seeds every randomness source a run touches.

    Only affects weight initialisation, sampler draws, and DataLoader
    ordering. The data split is drawn separately (see split_utils) and is
    deliberately out of reach of this seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Gives each DataLoader worker a distinct but reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class TrainConfig:
    max_epochs: int = 100
    patience: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _make_loaders(train_df, val_df, dataset_root, cfg: TrainConfig, seed: int):
    train_ds = FishEyeDataset(train_df, dataset_root, transform=train_transform)
    val_ds = FishEyeDataset(val_df, dataset_root, transform=eval_transform)

    # Seeded generators so sampler draws and batch ordering repeat exactly
    # for a given seed.
    sampler_generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)

    sampler = make_balanced_sampler(train_df, generator=sampler_generator)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        generator=loader_generator,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=_worker_init_fn,
    )
    return train_loader, val_loader


def train_single_task(
    task: str,  # "species" or "freshness"
    train_df,
    val_df,
    dataset_root,
    checkpoint_path,
    seed: int,
    cfg: TrainConfig = TrainConfig(),
    run_name: str = "",
    verbose: bool = True,
):
    assert task in ("species", "freshness")
    set_seed(seed)
    log_prefix = f"[{run_name}] " if run_name else ""

    num_classes = 8 if task == "species" else 3
    train_loader, val_loader = _make_loaders(train_df, val_df, dataset_root, cfg, seed)

    model = SingleTaskModel(num_classes=num_classes, pretrained=True).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -float("inf")
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        for images, species_idx, freshness_idx, combined_idx in train_loader:
            labels = (species_idx if task == "species" else freshness_idx).to(cfg.device)
            images = images.to(cfg.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()

        scheduler.step()
        n_train = len(train_loader.dataset)
        train_loss = train_loss_sum / n_train
        train_accuracy = train_correct / n_train

        model.eval()
        val_loss_sum = 0.0
        val_true, val_pred = [], []
        with torch.no_grad():
            for images, species_idx, freshness_idx, combined_idx in val_loader:
                labels = (species_idx if task == "species" else freshness_idx).to(cfg.device)
                images = images.to(cfg.device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_true.append(labels.cpu().numpy())
                val_pred.append(logits.argmax(dim=1).cpu().numpy())
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        val_accuracy = (val_true == val_pred).mean()
        val_f1 = f1_score(val_true, val_pred, average="macro")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_macro": val_f1,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            marker = " (best, checkpoint saved)"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_f1_macro": val_f1},
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            marker = f" (no improvement, {epochs_without_improvement}/{cfg.patience})"

        if verbose:
            print(
                f"{log_prefix}epoch {epoch}: train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.4f} val_f1={val_f1:.4f}{marker}"
            )

        if epochs_without_improvement >= cfg.patience:
            if verbose:
                print(f"{log_prefix}early stopping at epoch {epoch} (best val_f1={best_val_f1:.4f})")
            break

    elapsed = time.time() - start_time
    if verbose:
        print(f"{log_prefix}done in {elapsed / 60:.1f} min, best val_f1={best_val_f1:.4f}")
    return {"history": history, "best_val_f1": best_val_f1, "training_time_sec": elapsed}


def train_flat24(
    train_df,
    val_df,
    dataset_root,
    checkpoint_path,
    seed: int,
    cfg: TrainConfig = TrainConfig(),
    run_name: str = "",
    verbose: bool = True,
):
    """Trains the flat 24-class baseline.

    Monitored on the unweighted mean of species and freshness macro F1
    *after* mapping predictions back to the two tasks -- not on 24-class F1
    -- so its checkpoint is selected on the same quantity as the multi-task
    models it will be compared against.
    """
    set_seed(seed)
    log_prefix = f"[{run_name}] " if run_name else ""

    train_loader, val_loader = _make_loaders(train_df, val_df, dataset_root, cfg, seed)

    model = FlatCombinedModel(pretrained=True).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -float("inf")
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        for images, species_idx, freshness_idx, combined_idx in train_loader:
            images = images.to(cfg.device)
            combined_idx = combined_idx.to(cfg.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, combined_idx)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == combined_idx).sum().item()

        scheduler.step()
        n_train = len(train_loader.dataset)
        train_loss = train_loss_sum / n_train
        train_accuracy_combined = train_correct / n_train

        model.eval()
        val_loss_sum = 0.0
        combined_true, combined_pred = [], []
        with torch.no_grad():
            for images, species_idx, freshness_idx, combined_idx in val_loader:
                images = images.to(cfg.device)
                combined_idx = combined_idx.to(cfg.device)
                logits = model(images)
                loss = criterion(logits, combined_idx)
                val_loss_sum += loss.item() * images.size(0)
                combined_true.append(combined_idx.cpu().numpy())
                combined_pred.append(logits.argmax(dim=1).cpu().numpy())
        val_loss = val_loss_sum / len(val_loader.dataset)
        combined_true = np.concatenate(combined_true)
        combined_pred = np.concatenate(combined_pred)

        species_true, freshness_true = combined_to_tasks(combined_true)
        species_pred, freshness_pred = combined_to_tasks(combined_pred)
        val_f1_species = f1_score(species_true, species_pred, average="macro")
        val_f1_freshness = f1_score(freshness_true, freshness_pred, average="macro")
        val_f1_mean = (val_f1_species + val_f1_freshness) / 2

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy_combined": train_accuracy_combined,
                "val_loss": val_loss,
                "val_accuracy_combined": float((combined_true == combined_pred).mean()),
                "val_accuracy_species": float((species_true == species_pred).mean()),
                "val_accuracy_freshness": float((freshness_true == freshness_pred).mean()),
                "val_f1_species": val_f1_species,
                "val_f1_freshness": val_f1_freshness,
                "val_f1_mean": val_f1_mean,
            }
        )

        if val_f1_mean > best_val_f1:
            best_val_f1 = val_f1_mean
            epochs_without_improvement = 0
            marker = " (best, checkpoint saved)"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_f1_mean": val_f1_mean},
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            marker = f" (no improvement, {epochs_without_improvement}/{cfg.patience})"

        if verbose:
            print(
                f"{log_prefix}epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1_species={val_f1_species:.4f} "
                f"val_f1_freshness={val_f1_freshness:.4f} val_f1_mean={val_f1_mean:.4f}{marker}"
            )

        if epochs_without_improvement >= cfg.patience:
            if verbose:
                print(f"{log_prefix}early stopping at epoch {epoch} (best val_f1_mean={best_val_f1:.4f})")
            break

    elapsed = time.time() - start_time
    if verbose:
        print(f"{log_prefix}done in {elapsed / 60:.1f} min, best val_f1_mean={best_val_f1:.4f}")
    return {"history": history, "best_val_f1": best_val_f1, "training_time_sec": elapsed}


def train_multitask(
    loss_strategy_name: str,  # "EW", "UW", or "DWA"
    train_df,
    val_df,
    dataset_root,
    checkpoint_path,
    seed: int,
    cfg: TrainConfig = TrainConfig(),
    run_name: str = "",
    verbose: bool = True,
):
    set_seed(seed)
    log_prefix = f"[{run_name}] " if run_name else ""

    train_loader, val_loader = _make_loaders(train_df, val_df, dataset_root, cfg, seed)

    model = MultiTaskModel(pretrained=True).to(cfg.device)
    loss_strategy = build_loss_strategy(loss_strategy_name).to(cfg.device)
    criterion = nn.CrossEntropyLoss()

    # UW's log-variance parameters are excluded from weight decay: its
    # objective already carries an s1 + s2 regulariser, and decaying them on
    # top of that pulls both toward zero from two directions at once, driving
    # the task weights toward uniform for reasons that have nothing to do with
    # the uncertainty mechanism. Left in, "UW behaves like EW" would be an
    # artefact of the optimiser rather than a finding. Empty for EW and DWA,
    # which carry no learnable parameters.
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "weight_decay": cfg.weight_decay},
            {"params": list(loss_strategy.parameters()), "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_f1 = -float("inf")
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_species_loss_sum = 0.0
        train_freshness_loss_sum = 0.0
        train_species_correct = 0
        train_freshness_correct = 0

        for images, species_idx, freshness_idx, combined_idx in train_loader:
            images = images.to(cfg.device)
            species_idx = species_idx.to(cfg.device)
            freshness_idx = freshness_idx.to(cfg.device)

            optimizer.zero_grad()
            species_logits, freshness_logits = model(images)
            loss_species = criterion(species_logits, species_idx)
            loss_freshness = criterion(freshness_logits, freshness_idx)
            total_loss = loss_strategy(loss_species, loss_freshness)
            total_loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss_sum += total_loss.item() * batch_size
            train_species_loss_sum += loss_species.item() * batch_size
            train_freshness_loss_sum += loss_freshness.item() * batch_size
            train_species_correct += (species_logits.argmax(dim=1) == species_idx).sum().item()
            train_freshness_correct += (freshness_logits.argmax(dim=1) == freshness_idx).sum().item()

        scheduler.step()
        n_train = len(train_loader.dataset)
        train_loss = train_loss_sum / n_train
        train_accuracy_species = train_species_correct / n_train
        train_accuracy_freshness = train_freshness_correct / n_train

        if isinstance(loss_strategy, DynamicWeightAveraging):
            loss_strategy.update_epoch_losses(
                train_species_loss_sum / n_train, train_freshness_loss_sum / n_train
            )

        model.eval()
        val_loss_sum = 0.0
        species_true, species_pred, freshness_true, freshness_pred = [], [], [], []
        with torch.no_grad():
            for images, species_idx, freshness_idx, combined_idx in val_loader:
                images = images.to(cfg.device)
                species_idx = species_idx.to(cfg.device)
                freshness_idx = freshness_idx.to(cfg.device)
                species_logits, freshness_logits = model(images)
                loss_species = criterion(species_logits, species_idx)
                loss_freshness = criterion(freshness_logits, freshness_idx)
                total_loss = loss_strategy(loss_species, loss_freshness)
                val_loss_sum += total_loss.item() * images.size(0)
                species_true.append(species_idx.cpu().numpy())
                species_pred.append(species_logits.argmax(dim=1).cpu().numpy())
                freshness_true.append(freshness_idx.cpu().numpy())
                freshness_pred.append(freshness_logits.argmax(dim=1).cpu().numpy())
        val_loss = val_loss_sum / len(val_loader.dataset)
        species_true = np.concatenate(species_true)
        species_pred = np.concatenate(species_pred)
        freshness_true = np.concatenate(freshness_true)
        freshness_pred = np.concatenate(freshness_pred)

        val_accuracy_species = (species_true == species_pred).mean()
        val_accuracy_freshness = (freshness_true == freshness_pred).mean()
        val_f1_species = f1_score(species_true, species_pred, average="macro")
        val_f1_freshness = f1_score(freshness_true, freshness_pred, average="macro")
        val_f1_mean = (val_f1_species + val_f1_freshness) / 2

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy_species": train_accuracy_species,
                "train_accuracy_freshness": train_accuracy_freshness,
                "val_loss": val_loss,
                "val_accuracy_species": val_accuracy_species,
                "val_accuracy_freshness": val_accuracy_freshness,
                "val_f1_species": val_f1_species,
                "val_f1_freshness": val_f1_freshness,
                "val_f1_mean": val_f1_mean,
            }
        )

        if val_f1_mean > best_val_f1:
            best_val_f1 = val_f1_mean
            epochs_without_improvement = 0
            marker = " (best, checkpoint saved)"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "loss_strategy_state": loss_strategy.state_dict(),
                    "val_f1_mean": val_f1_mean,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            marker = f" (no improvement, {epochs_without_improvement}/{cfg.patience})"

        if verbose:
            print(
                f"{log_prefix}epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1_species={val_f1_species:.4f} "
                f"val_f1_freshness={val_f1_freshness:.4f} val_f1_mean={val_f1_mean:.4f}{marker}"
            )

        if epochs_without_improvement >= cfg.patience:
            if verbose:
                print(f"{log_prefix}early stopping at epoch {epoch} (best val_f1_mean={best_val_f1:.4f})")
            break

    elapsed = time.time() - start_time
    if verbose:
        print(f"{log_prefix}done in {elapsed / 60:.1f} min, best val_f1_mean={best_val_f1:.4f}")
    return {"history": history, "best_val_f1": best_val_f1, "training_time_sec": elapsed}
