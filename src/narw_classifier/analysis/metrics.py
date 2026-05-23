"""Threshold-aware metrics for the FP/FN trade-off analysis.

These complement the streaming per-epoch metrics (AUROC, AP, accuracy@0.5) and
provide the answers we actually care about in a conservation context:

- "How many calls do we miss / how many false alarms do we trigger at a chosen
  operating threshold?" (TP / FP / FN / TN at threshold 0.5)
- "What's our recall budget at 1% / 5% false-positive-rate constraints?"
  (``recall_at_fpr``)
- "What's the best F1 we can reach across thresholds and where?"
  (``best_f1_threshold_sweep``)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def recall_at_fpr(probs: np.ndarray, labels: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Recall (TPR) at the most permissive operating point with ``fpr <= target_fpr``.

    Returns ``(recall, threshold)``. If no operating point reaches the target FPR,
    returns the lowest-FPR point instead.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    fpr, tpr, thresholds = roc_curve(labels, probs)
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


def confusion_metrics_at_threshold(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return TP/FP/FN/TN + precision/recall/F1/FPR at ``threshold``."""
    tn, fp, fn, tp = confusion_at_threshold(probs, labels, threshold).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


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


def best_f1_from_sweep(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return ``(best_f1, threshold)`` over a 101-point sweep."""
    sweep = threshold_sweep(probs, labels, thresholds=np.linspace(0.0, 1.0, 101))
    idx = int(np.argmax(sweep["f1"]))
    return float(sweep["f1"][idx]), float(sweep["thresholds"][idx])
