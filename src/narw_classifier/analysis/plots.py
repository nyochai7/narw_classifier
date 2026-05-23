"""Matplotlib figures for the FP/FN analysis: PR curve, ROC curve, confusion matrix."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .metrics import confusion_at_threshold


def plot_pr_curve(probs: np.ndarray, labels: np.ndarray) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=1.5)
    ax.axhline(
        labels.mean(), ls="--", color="grey", lw=0.8, label=f"baseline (P={labels.mean():.2f})"
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=f"PR curve  (AP = {ap:.3f})",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_roc_curve(probs: np.ndarray, labels: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=1.5)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, label="random")
    ax.set(
        xlabel="FPR",
        ylabel="Recall (TPR)",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=f"ROC curve  (AUROC = {auc:.3f})",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> plt.Figure:
    cm = confusion_at_threshold(probs, labels, threshold)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["actual 0", "actual 1"])
    ax.set_title(f"Confusion matrix  (threshold = {threshold:.2f})")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig
