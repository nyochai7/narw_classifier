"""Threshold-aware metrics for the FP/FN trade-off analysis.

These complement the per-epoch metrics (AUROC, AP, accuracy@0.5) already logged
by the LightningModule. They answer "at what operating point would we deploy
this model, and how many false positives / negatives does that imply?"
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def recall_at_fpr(probs: np.ndarray, labels: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Recall (TPR) at the smallest operating point with ``fpr <= target_fpr``.

    Returns ``(recall, threshold)``. If no operating point reaches the target FPR,
    returns the closest one. Useful in conservation contexts where false-positive
    budgets are explicit (e.g. recall at 1% FPR, 5% FPR).
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    fpr, tpr, thresholds = roc_curve(labels, probs)
    # Among operating points with fpr <= target, pick the one with the highest TPR
    # (multiple points can share the same FPR; we want the most permissive).
    mask = fpr <= target_fpr
    if not mask.any():
        idx = int(np.argmin(fpr))
    else:
        masked_tpr = np.where(mask, tpr, -np.inf)
        idx = int(np.argmax(masked_tpr))
    return float(tpr[idx]), float(thresholds[idx])


def confusion_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Return the 2x2 confusion matrix ``[[tn, fp], [fn, tp]]`` at ``threshold``."""
    preds = (probs >= threshold).astype(np.int64)
    return confusion_matrix(labels, preds, labels=[0, 1])


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sweep ``thresholds`` and return precision / recall / FPR / accuracy / F1 at each.

    Handy for a single table in the W&B report. If ``thresholds`` is None, uses a
    21-point linear sweep from 0.0 to 1.0.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)
    accuracy = np.zeros_like(thresholds)
    f1 = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        tn, fp, fn, tp = confusion_at_threshold(probs, labels, t).ravel()
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy[i] = (tp + tn) / (tp + tn + fp + fn)
        denom = precision[i] + recall[i]
        f1[i] = (2 * precision[i] * recall[i] / denom) if denom > 0 else 0.0
    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
        "f1": f1,
    }
