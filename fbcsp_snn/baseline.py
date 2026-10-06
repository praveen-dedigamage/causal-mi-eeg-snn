"""Classical FBCSP baseline classifiers.

Extracts log-variance features from z-normalised CSP projections and
evaluates LDA and SVM classifiers on the same train/val/test splits as
the SNN.  These results serve as an upper-bound reference: if LDA/SVM
can't beat the SNN's accuracy, the bottleneck is in feature extraction,
not the spiking model.

Typical usage (called from pipeline._run_single_fold after z-norm):

    from fbcsp_snn.baseline import extract_logvar, run_baseline_classifiers
    feat_tr  = extract_logvar(X_norm_tr)
    feat_val = extract_logvar(X_norm_val)
    feat_te  = extract_logvar(X_norm_te)
    bl = run_baseline_classifiers(feat_tr, y_tr, feat_val, y_val, feat_te, y_te)

Baseline tuning (optional Tier 0, 2026-07-06)
------------------------------------------------
SVM hyperparameters are selected via a small grid search using the *val*
split already computed every fold for the SNN's checkpoint selection —
mirroring how the SNN uses val, rather than leaving SVM fixed at untuned
defaults. LDA uses shrinkage regularisation (``solver='lsqr',
shrinkage='auto'``) instead of the previous unregularised ``solver='svd'``.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fbcsp_snn import setup_logger

logger = setup_logger(__name__)

# Small grid searched over the existing val split — kept intentionally
# modest (9 combinations) since this runs every fold on top of everything
# else; not a full hyperparameter search.
_SVM_C_GRID = (0.1, 1.0, 10.0)
_SVM_GAMMA_GRID = ("scale", 0.01, 0.1)


def extract_logvar(X_norm: np.ndarray) -> np.ndarray:
    """Compute log-variance features from z-normalised CSP projections.

    Parameters
    ----------
    X_norm : np.ndarray
        Z-normalised CSP projections, shape ``(n_trials, n_features, n_samples)``.

    Returns
    -------
    np.ndarray
        Log-variance features, shape ``(n_trials, n_features)``.
    """
    var = np.var(X_norm, axis=-1)          # (n_trials, n_features)
    var = np.clip(var, 1e-10, None)        # guard against log(0)
    return np.log(var).astype(np.float32)


def run_baseline_classifiers(
    feat_tr:  np.ndarray,
    y_tr:     np.ndarray,
    feat_val: np.ndarray,
    y_val:    np.ndarray,
    feat_te:  np.ndarray,
    y_te:     np.ndarray,
) -> dict[str, float]:
    """Train LDA and SVM on log-variance features and evaluate on all splits.

    Both classifiers are fit on ``feat_tr / y_tr`` only. SVM hyperparameters
    (``C``, ``gamma``) are selected via a small grid search scored on
    ``feat_val`` / ``y_val`` — the same val split the SNN already uses for
    checkpoint selection — rather than left at fixed defaults. No leakage:
    test is only ever scored once, after the best combination is chosen.

    Parameters
    ----------
    feat_tr : np.ndarray
        Training log-variance features, shape ``(n_trials, n_features)``.
    y_tr : np.ndarray
        Training class labels, 0-indexed, shape ``(n_trials,)``.
    feat_val : np.ndarray
        Validation features, shape ``(n_trials, n_features)``.
    y_val : np.ndarray
        Validation labels, 0-indexed.
    feat_te : np.ndarray
        Test features, shape ``(n_trials, n_features)``.
    y_te : np.ndarray
        Test labels, 0-indexed.

    Returns
    -------
    dict[str, float]
        ``val_acc_lda``, ``test_acc_lda``, ``val_acc_svm``, ``test_acc_svm``
        (accuracies in ``[0, 1]``), plus ``svm_best_c`` and
        ``svm_best_gamma`` recording which grid combination was selected
        (logged for reproducibility, not required by any existing consumer).
    """
    results: dict = {}

    # ---- LDA ------------------------------------------------------------
    # Standard choice for FBCSP in MI-BCI (same as MNE-Python default
    # pipeline). solver='lsqr' + shrinkage='auto' (Ledoit-Wolf) regularises
    # the covariance estimate — matters more as n_features grows relative to
    # n_trials (e.g. Schirrmeister2017's 128 channels) than the previous
    # unregularised solver='svd'.
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(feat_tr, y_tr)
    results["val_acc_lda"]  = float(lda.score(feat_val, y_val))
    results["test_acc_lda"] = float(lda.score(feat_te,  y_te))
    logger.info(
        "Baseline LDA   val=%.1f%%  test=%.1f%%",
        results["val_acc_lda"] * 100, results["test_acc_lda"] * 100,
    )

    # ---- SVM (RBF), tuned on val -----------------------------------------
    # RBF-SVM with StandardScaler is consistently competitive on CSP
    # features. C/gamma are selected by the combination that maximises val
    # accuracy (mirrors how val already selects the SNN's best checkpoint),
    # instead of a single fixed default.
    best_val_acc = -1.0
    best_c, best_gamma = _SVM_C_GRID[0], _SVM_GAMMA_GRID[0]
    best_svm = None
    for c, gamma in itertools.product(_SVM_C_GRID, _SVM_GAMMA_GRID):
        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=c, kernel="rbf", gamma=gamma, random_state=42)),
        ])
        svm.fit(feat_tr, y_tr)
        val_acc = float(svm.score(feat_val, y_val))
        if val_acc > best_val_acc:
            best_val_acc, best_c, best_gamma, best_svm = val_acc, c, gamma, svm

    results["val_acc_svm"]  = best_val_acc
    results["test_acc_svm"] = float(best_svm.score(feat_te, y_te))
    results["svm_best_c"]     = best_c
    results["svm_best_gamma"] = best_gamma
    logger.info(
        "Baseline SVM   val=%.1f%%  test=%.1f%%  (best C=%s gamma=%s, grid of %d)",
        results["val_acc_svm"] * 100, results["test_acc_svm"] * 100,
        best_c, best_gamma, len(_SVM_C_GRID) * len(_SVM_GAMMA_GRID),
    )

    return results
