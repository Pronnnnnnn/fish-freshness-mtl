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
from models import MultiTaskModel, SingleTaskModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    max_epochs: int = 100
    patience: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _make_loaders(train_df, val_df, dataset_root, cfg: TrainConfig):
    train_ds = FishEyeDataset(train_df, dataset_root, transform=train_transform)
    val_ds = FishEyeDataset(val_df, dataset_root, transform=eval_transform)
    sampler = make_balanced_sampler(train_df)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
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
):
    assert task in ("species", "freshness")
    set_seed(seed)

    num_classes = 8 if task == "species" else 3
    train_loader, val_loader = _make_loaders(train_df, val_df, dataset_root, cfg)

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
        for images, species_idx, freshness_idx in train_loader:
            labels = (species_idx if task == "species" else freshness_idx).to(cfg.device)
            images = images.to(cfg.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)

        scheduler.step()
        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        val_true, val_pred = [], []
        with torch.no_grad():
            for images, species_idx, freshness_idx in val_loader:
                labels = (species_idx if task == "species" else freshness_idx).to(cfg.device)
                images = images.to(cfg.device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_true.append(labels.cpu().numpy())
                val_pred.append(logits.argmax(dim=1).cpu().numpy())
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_f1 = f1_score(np.concatenate(val_true), np.concatenate(val_pred), average="macro")

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1_macro": val_f1}
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_f1_macro": val_f1},
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    elapsed = time.time() - start_time
    return {"history": history, "best_val_f1": best_val_f1, "training_time_sec": elapsed}


def train_multitask(
    loss_strategy_name: str,  # "EW", "UW", or "DWA"
    train_df,
    val_df,
    dataset_root,
    checkpoint_path,
    seed: int,
    cfg: TrainConfig = TrainConfig(),
):
    set_seed(seed)

    train_loader, val_loader = _make_loaders(train_df, val_df, dataset_root, cfg)

    model = MultiTaskModel(pretrained=True).to(cfg.device)
    loss_strategy = build_loss_strategy(loss_strategy_name).to(cfg.device)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters()) + list(loss_strategy.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
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

        for images, species_idx, freshness_idx in train_loader:
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

        scheduler.step()
        n_train = len(train_loader.dataset)
        train_loss = train_loss_sum / n_train

        if isinstance(loss_strategy, DynamicWeightAveraging):
            loss_strategy.update_epoch_losses(
                train_species_loss_sum / n_train, train_freshness_loss_sum / n_train
            )

        model.eval()
        val_loss_sum = 0.0
        species_true, species_pred, freshness_true, freshness_pred = [], [], [], []
        with torch.no_grad():
            for images, species_idx, freshness_idx in val_loader:
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

        val_f1_species = f1_score(
            np.concatenate(species_true), np.concatenate(species_pred), average="macro"
        )
        val_f1_freshness = f1_score(
            np.concatenate(freshness_true), np.concatenate(freshness_pred), average="macro"
        )
        val_f1_mean = (val_f1_species + val_f1_freshness) / 2

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_species": val_f1_species,
                "val_f1_freshness": val_f1_freshness,
                "val_f1_mean": val_f1_mean,
            }
        )

        if val_f1_mean > best_val_f1:
            best_val_f1 = val_f1_mean
            epochs_without_improvement = 0
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
            if epochs_without_improvement >= cfg.patience:
                break

    elapsed = time.time() - start_time
    return {"history": history, "best_val_f1": best_val_f1, "training_time_sec": elapsed}
