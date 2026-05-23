"""High-level 'produce all analysis artifacts for one run' helper.

Used by both the :class:`EvalArtifactsCallback` (in-training) and the
``02_error_analysis.ipynb`` notebook (post-hoc).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from .metrics import recall_at_fpr
from .plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from .predictions import save_predictions


def compute_summary(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Headline numbers for the W&B summary table / report."""
    rec_1, thr_1 = recall_at_fpr(probs, labels, target_fpr=0.01)
    rec_5, thr_5 = recall_at_fpr(probs, labels, target_fpr=0.05)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "accuracy_t0.5": float(accuracy_score(labels, (probs >= 0.5).astype(int))),
        "recall_at_1pct_fpr": rec_1,
        "threshold_at_1pct_fpr": thr_1,
        "recall_at_5pct_fpr": rec_5,
        "threshold_at_5pct_fpr": thr_5,
        "positive_rate": float(labels.mean()),
        "n_samples": int(labels.size),
    }


def save_run_artifacts(
    out_dir: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    files: list[str] | None = None,
) -> dict[str, float]:
    """Save predictions.npz + PR/ROC/confusion PNGs + summary.json under ``out_dir``.

    Returns the summary dict so the caller can also log scalars to W&B.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(out_dir / "predictions.npz", probs, labels, files)

    plot_pr_curve(probs, labels).savefig(out_dir / "pr_curve.png", dpi=150)
    plot_roc_curve(probs, labels).savefig(out_dir / "roc_curve.png", dpi=150)
    plot_confusion_matrix(probs, labels, threshold=0.5).savefig(
        out_dir / "confusion_matrix_t0.5.png", dpi=150
    )

    summary = compute_summary(probs, labels)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
