"""Fairness-controlled classical baselines: LDA/SVM trained on the FULL
z-normalised CSP time series -- the same continuous signal the SNN's
spike encoder receives -- instead of a single log-variance scalar per
feature (as in ``baseline.py``).

Motivation: the log-variance baseline collapses each trial's full
time-domain signal to one number per feature before classification even
begins, while the SNN processes the entire T_s-timestep sequence. This
module removes that information asymmetry by giving LDA/SVM the same
pre-encoding time series, flattened into one vector per trial.

Numerical note on why LDA needs a PCA step but SVM doesn't
------------------------------------------------------------
Flattened dimensionality (n_features x T_s) reaches the hundreds of
thousands (e.g. ~288K for BNCI2014-001, ~576K for Schirrmeister2017).
Shrinkage LDA (solver='lsqr', shrinkage='auto', matching baseline.py's
convention) estimates a shared covariance matrix of shape (p, p) --
infeasible to even store at this p (hundreds of GB). We therefore first
project onto the top principal components of the TRAINING partition only
(no leakage), keeping min(n_train - 1, PCA_CAP) components -- this is
lossless up to the data's actual rank (bounded by n_train), not a
hand-crafted summary statistic, so LDA still sees everything the data
matrix can express. Linear SVM (liblinear/LinearSVC) has no such
constraint -- it operates directly on the raw flattened vectors, exactly
matching "give SVM the same time samples the SNN receives" literally.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fbcsp_snn import setup_logger

logger = setup_logger(__name__)

# Linear SVM has one hyperparameter (C) instead of RBF's (C, gamma) grid --
# widened range since there's no gamma dimension to compensate.
_SVM_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)

# Upper bound on retained PCA components even when n_train - 1 is larger
# (keeps the LDA fit fast; n_train is already the binding constraint in
# every dataset here, so this cap is rarely if ever hit).
_PCA_CAP = 200

# Matches MIBIFSelector's own default exactly (fbcsp_snn/mibif.py) -- same
# adaptive-threshold criterion, applied here to flattened time-series
# dimensions instead of post-encoding spike counts.
_MI_FRACTION = 0.1


def select_by_mutual_information(
    feat_tr: np.ndarray,
    y_tr: np.ndarray,
    mi_fraction: float = _MI_FRACTION,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Supervised feature selection for LDA/SVM, mirroring MIBIFSelector's
    adaptive-threshold logic exactly (same ``mutual_info_classif`` call,
    same ``mi_fraction`` default) -- applied to flattened time-series
    dimensions rather than post-encoding spike counts.

    Without this, LDA relies solely on PCA for dimensionality reduction,
    which is unsupervised: it maximises variance without ever looking at
    the class labels, and could discard exactly the class-discriminative
    directions if they don't dominate overall variance. This gives LDA/SVM
    an analogous, properly *supervised* reduction step instead -- the
    fairness check should not be confounded by giving the SNN a
    label-aware selection step (MIBIF) while giving the baselines only a
    label-blind one (PCA).

    Fit on the training partition only -- no leakage.

    Parameters
    ----------
    feat_tr : np.ndarray
        Flattened training features, shape ``(n_trials, n_features * n_samples)``.
    y_tr : np.ndarray
        0-indexed training labels.
    mi_fraction : float
        Adaptive threshold as a fraction of the maximum MI score (default
        matches MIBIFSelector's own default, 0.1).
    random_state : int
        Passed to ``mutual_info_classif`` for reproducibility.

    Returns
    -------
    selected_indices : np.ndarray
        Sorted indices of the selected dimensions.
    mi_scores : np.ndarray
        Raw MI score for every dimension (for logging/diagnostics).
    """
    mi_scores = mutual_info_classif(
        feat_tr, y_tr, discrete_features=False, random_state=random_state
    )
    threshold = mi_fraction * mi_scores.max()
    selected_indices = np.sort(np.where(mi_scores >= threshold)[0]).astype(np.int64)
    logger.info(
        "Fair-baseline MI selection [mi_fraction=%.3f]: %d -> %d dimensions "
        "kept (%.2f%%)  score threshold=%.6f  range=[%.6f, %.6f]",
        mi_fraction, len(mi_scores), len(selected_indices),
        100.0 * len(selected_indices) / len(mi_scores),
        threshold, mi_scores.min(), mi_scores.max(),
    )
    return selected_indices, mi_scores


def flatten_timeseries(X_norm: np.ndarray) -> np.ndarray:
    """Flatten z-normalised CSP time series into one vector per trial.

    Parameters
    ----------
    X_norm : np.ndarray
        Shape ``(n_trials, n_features, n_samples)`` -- the same array the
        adaptive-threshold encoder would receive.

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_features * n_samples)``, float32.
    """
    n_trials = X_norm.shape[0]
    return X_norm.reshape(n_trials, -1).astype(np.float32)


def run_fair_baseline_classifiers(
    feat_tr:  np.ndarray,
    y_tr:     np.ndarray,
    feat_val: np.ndarray,
    y_val:    np.ndarray,
    feat_te:  np.ndarray,
    y_te:     np.ndarray,
) -> dict:
    """Train LDA (via PCA) and linear SVM on flattened time-series features.

    Mirrors ``baseline.run_baseline_classifiers``'s structure and dict-key
    convention (with an ``_fullts`` suffix), so results line up directly
    against the log-variance baseline and the SNN.

    Parameters
    ----------
    feat_tr, feat_val, feat_te : np.ndarray
        Flattened time-series features, shape ``(n_trials, n_features *
        n_samples)``, as returned by :func:`flatten_timeseries`.
    y_tr, y_val, y_te : np.ndarray
        0-indexed class labels.

    Returns
    -------
    dict
        ``val_acc_lda_fullts``, ``test_acc_lda_fullts``,
        ``lda_pca_components``, ``lda_pca_explained_variance``,
        ``val_acc_svm_fullts``, ``test_acc_svm_fullts``,
        ``svm_fullts_best_c``, ``mi_selected_dims``, ``mi_total_dims``.
    """
    results: dict = {}
    n_train = feat_tr.shape[0]

    # ---- Supervised feature selection (MIBIF-equivalent for LDA/SVM) -----
    # Fit on TRAIN only -- no leakage. Without this, LDA's dimensionality
    # reduction would come entirely from PCA, which is unsupervised and
    # could discard the class-discriminative directions; SVM would get no
    # reduction at all. This gives both classifiers a selection step
    # analogous to what MIBIF already gives the SNN.
    mi_idx, _ = select_by_mutual_information(feat_tr, y_tr)
    feat_tr_sel  = feat_tr[:,  mi_idx]
    feat_val_sel = feat_val[:, mi_idx]
    feat_te_sel  = feat_te[:,  mi_idx]
    results["mi_selected_dims"] = int(len(mi_idx))
    results["mi_total_dims"]    = int(feat_tr.shape[1])

    # ---- LDA (PCA-projected for numerical feasibility) -------------------
    n_components = min(n_train - 1, feat_tr_sel.shape[1], _PCA_CAP)
    pca = PCA(n_components=n_components, random_state=42)
    feat_tr_pca  = pca.fit_transform(feat_tr_sel)   # fit on TRAIN only -- no leakage
    feat_val_pca = pca.transform(feat_val_sel)
    feat_te_pca  = pca.transform(feat_te_sel)
    explained = float(pca.explained_variance_ratio_.sum())

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(feat_tr_pca, y_tr)
    results["val_acc_lda_fullts"]  = float(lda.score(feat_val_pca, y_val))
    results["test_acc_lda_fullts"] = float(lda.score(feat_te_pca,  y_te))
    results["lda_pca_components"]  = n_components
    results["lda_pca_explained_variance"] = round(explained, 4)
    logger.info(
        "Fair baseline LDA (full time-series, MI-selected %d/%d dims, "
        "%d PCA comps, %.1f%% var)  val=%.1f%%  test=%.1f%%",
        len(mi_idx), feat_tr.shape[1], n_components, explained * 100,
        results["val_acc_lda_fullts"] * 100, results["test_acc_lda_fullts"] * 100,
    )

    # ---- Linear SVM (on MI-selected dimensions, no PCA needed) -----------
    best_val_acc = -1.0
    best_c = _SVM_C_GRID[0]
    best_svm = None
    for c in _SVM_C_GRID:
        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=c, random_state=42, max_iter=20000)),
        ])
        svm.fit(feat_tr_sel, y_tr)
        val_acc = float(svm.score(feat_val_sel, y_val))
        if val_acc > best_val_acc:
            best_val_acc, best_c, best_svm = val_acc, c, svm

    results["val_acc_svm_fullts"]  = best_val_acc
    results["test_acc_svm_fullts"] = float(best_svm.score(feat_te_sel, y_te))
    results["svm_fullts_best_c"]   = best_c
    logger.info(
        "Fair baseline SVM (full time-series, MI-selected %d/%d dims, "
        "linear)  val=%.1f%%  test=%.1f%%  (best C=%s, grid of %d)",
        len(mi_idx), feat_tr.shape[1],
        results["val_acc_svm_fullts"] * 100, results["test_acc_svm_fullts"] * 100,
        best_c, len(_SVM_C_GRID),
    )

    return results
