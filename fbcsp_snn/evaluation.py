"""Accuracy, confusion matrix, and result-summary utilities.

All plot functions save figures to disk and close them immediately to avoid
matplotlib memory leaks (per the project coding guidelines).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

matplotlib.use("Agg")  # non-interactive; safe on HPC and Windows
import matplotlib.pyplot as plt
import seaborn as sns

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground-truth and predicted class indices.  Any integer encoding is
        accepted (0-indexed or 1-indexed), as long as both arrays use the same
        convention.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """
    return float(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute (optionally row-normalised) confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted labels.
    n_classes : Optional[int]
        If given, force the matrix to size ``(n_classes, n_classes)`` so that
        classes absent from *y_pred* still appear as zero columns.
    normalize : bool
        If ``True`` (default), each row is divided by its sum so that the
        matrix shows recall per class.

    Returns
    -------
    np.ndarray
        Confusion matrix, shape ``(n_classes, n_classes)``.
    """
    labels = list(range(n_classes)) if n_classes is not None else None
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums > 0)
    return cm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted labels (0-indexed, aligned with *class_names*).
    class_names : List[str]
        Human-readable class labels (e.g. ``["feet", "left_hand", ...]``).
    save_path : Path
        Destination file.  Parent directory is created if absent.
    title : str
        Figure title.
    normalize : bool
        Row-normalise the matrix (shows recall per class).
    """
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=len(class_names),
                                  normalize=normalize)
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        ax=ax,
        vmin=0.0,
        vmax=1.0 if normalize else None,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", save_path)


# ---------------------------------------------------------------------------
# Per-fold summary
# ---------------------------------------------------------------------------

def log_fold_summary(
    fold_results: list,
    subject_id: int,
) -> float:
    """Log a table of per-fold val accuracies and return the mean.

    Parameters
    ----------
    fold_results : list of FoldResult
        Results from :func:`fbcsp_snn.training.train_fold`.
    subject_id : int
        Subject number (for display only).

    Returns
    -------
    float
        Mean best val accuracy across folds.
    """
    accs = [r.best_val_acc for r in fold_results]
    logger.info("Subject %d  CV fold summary:", subject_id)
    logger.info("  %-8s  %-10s  %-10s", "Fold", "BestValAcc", "BestEpoch")
    logger.info("  " + "-" * 35)
    for i, r in enumerate(fold_results):
        logger.info(
            "  %-8d  %-10.1f  %-10d",
            i + 1, r.best_val_acc * 100, r.best_epoch + 1,
        )
    logger.info("  " + "-" * 35)
    mean_acc = float(np.mean(accs))
    logger.info("  %-8s  %-10.1f", "Mean", mean_acc * 100)
    return mean_acc
