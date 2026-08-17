"""Training-curve plots for overfitting inspection.

Renders train/val loss and train/val accuracy side by side from a
`history` list (as returned by train_single_task / train_multitask).
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_single_task_curves(history: list[dict], title: str, save_path: str | None = None):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_multitask_curves(history: list[dict], title: str, save_path: str | None = None):
    """Loss plus per-task accuracy curves.

    Covers both the two-head models and the flat 24-class one. The flat model
    trains on a single combined label, so it has no per-task training
    accuracy -- only its validation curves are per task, and the training
    panel falls back to its combined accuracy.
    """
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    is_flat = "train_accuracy_combined" in df.columns
    for ax, task in zip(axes[1:], ["species", "freshness"]):
        train_col = "train_accuracy_combined" if is_flat else f"train_accuracy_{task}"
        train_label = "train (combined)" if is_flat else "train"
        ax.plot(df["epoch"], df[train_col], label=train_label)
        ax.plot(df["epoch"], df[f"val_accuracy_{task}"], label="val")
        ax.set_title(f"{task.capitalize()} Accuracy"); ax.set_xlabel("epoch"); ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
