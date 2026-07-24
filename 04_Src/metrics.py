"""Evaluation metrics.

Per-task: accuracy, macro precision/recall/F1 (robust to class
imbalance), and MCC. The nominal species label additionally gets
Cohen's Kappa; the ordinal freshness label gets quadratic weighted
kappa in its place. Joint accuracy measures both labels predicted
correctly on the same image simultaneously.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def joint_accuracy(
    species_true: np.ndarray,
    species_pred: np.ndarray,
    freshness_true: np.ndarray,
    freshness_pred: np.ndarray,
) -> float:
    """Fraction of samples where both species and freshness are predicted correctly."""
    both_correct = (species_true == species_pred) & (freshness_true == freshness_pred)
    return float(np.mean(both_correct))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, ordinal: bool = False) -> dict:
    """Accuracy, macro precision/recall/F1, and MCC for one task.

    Nominal labels (ordinal=False) additionally get Cohen's Kappa;
    ordinal labels get quadratic weighted kappa in its place.
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if ordinal:
        result["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    else:
        result["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    return result


def evaluate_multitask(
    species_true: np.ndarray,
    species_pred: np.ndarray,
    freshness_true: np.ndarray,
    freshness_pred: np.ndarray,
) -> dict:
    """Full metric report for a joint species/freshness prediction."""
    return {
        "species": classification_metrics(species_true, species_pred, ordinal=False),
        "freshness": classification_metrics(freshness_true, freshness_pred, ordinal=True),
        "joint_accuracy": joint_accuracy(species_true, species_pred, freshness_true, freshness_pred),
    }


def task_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> np.ndarray:
    """Confusion matrix for one task, rows/columns ordered by `labels` if given."""
    return confusion_matrix(y_true, y_pred, labels=labels)
