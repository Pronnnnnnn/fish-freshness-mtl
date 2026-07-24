"""Evaluation metrics.

Per-task: accuracy, macro F1 (robust to class imbalance), MCC, Cohen's
Kappa. Freshness additionally gets quadratic weighted kappa since it is
ordinal. Joint accuracy measures both labels predicted correctly on the
same image simultaneously.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
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
    """Accuracy, macro F1, MCC, Cohen's Kappa, and (if ordinal) QWK for one task."""
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    if ordinal:
        result["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
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
