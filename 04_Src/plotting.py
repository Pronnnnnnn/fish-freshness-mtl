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
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy_species"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy_species"], label="val")
    axes[1].set_title("Species Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(df["epoch"], df["train_accuracy_freshness"], label="train")
    axes[2].plot(df["epoch"], df["val_accuracy_freshness"], label="val")
    axes[2].set_title("Freshness Accuracy"); axes[2].set_xlabel("epoch"); axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
