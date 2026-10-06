"""3-fold CV test: full pipeline from raw EEG to per-fold val accuracy and test accuracy.

Each fold fits ALL preprocessing steps (band selection, filter bank, CSP,
z-normalisation, spike encoding, MIBIF) on its own training split — no
data leakage from val/test into any fitting step.

Timing note
-----------
The SNN forward pass loops over T timesteps in Python.  At 0.83 ms/step on a
3060 GPU the full 1001-step sequence takes ~1.7 s per batch.  This test uses
``T_MAX=100`` to keep total runtime under ~5 minutes.  Production training uses
the full sequence; all code paths are identical.

Run from the project root::

    python tests/test_cv_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.datasets import load_moabb
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.evaluation import (
    compute_accuracy,
    log_fold_summary,
    plot_confusion_matrix,
)
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank
from fbcsp_snn.training import FoldResult, evaluate_model, train_fold

logger = setup_logger("test_cv_pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET      = "BNCI2014_001"
SUBJECT_ID   = 1
SFREQ        = 250
N_CLASSES    = 4
N_BANDS      = 6
M            = 2
LAMBDA_R     = 0.0001
HIDDEN       = 64
POP_PER_CLS  = 20
BETA         = 0.95
DROPOUT      = 0.5
N_FOLDS      = 3
MAX_EPOCHS   = 50
PATIENCE     = 20       # tighter for the 50-epoch test
WARMUP       = 10
BATCH_SIZE   = 32
T_MAX        = 100      # timestep cap; remove for full-length training
TAU_VR       = 10.0
SPIKE_PROB   = 0.7
FEAT_PCT     = 50.0     # keep top 50% by MI
LOG_EVERY    = 10

CLASS_NAMES  = ["feet", "left_hand", "right_hand", "tongue"]
RESULTS_DIR  = Path("Results") / f"Subject_{SUBJECT_ID}"


def sep(title: str) -> None:
    logger.info("=" * 62)
    logger.info("  %s", title)
    logger.info("=" * 62)


# ===========================================================================
# 1. Load data
# ===========================================================================
sep("Step 1: Load MOABB data")

X_train, y_train, X_test, y_test = load_moabb(
    DATASET, SUBJECT_ID, n_classes=N_CLASSES
)
y_train_0 = y_train - 1    # convert to 0-indexed for model
y_test_0  = y_test  - 1

logger.info("X_train: %s  X_test: %s", X_train.shape, X_test.shape)
logger.info("y_train classes: %s", np.unique(y_train).tolist())


# ===========================================================================
# 2. 3-fold CV
# ===========================================================================
sep(f"Step 2: {N_FOLDS}-fold CV  (epochs={MAX_EPOCHS}, T_MAX={T_MAX})")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results: list[FoldResult] = []
fold_artifacts: list[dict] = []   # preprocessing objects + model per fold

cv_start = time.perf_counter()

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    fold_start = time.perf_counter()
    logger.info("")
    logger.info("--- Fold %d / %d ---", fold_idx + 1, N_FOLDS)

    X_f_tr  = X_train[tr_idx];   y_f_tr  = y_train[tr_idx]
    X_f_val = X_train[val_idx];  y_f_val = y_train[val_idx]

    y_f_tr_0  = y_f_tr  - 1
    y_f_val_0 = y_f_val - 1

    # ---- 2a. Fixed frequency bands ----
    bands = [(4, 8), (8, 14), (12, 18), (16, 24), (20, 30), (26, 40)]
    logger.info("Bands: %s", bands)

    # ---- 2b. Filter bank ----
    X_bands_tr  = apply_filter_bank(X_f_tr,  bands, SFREQ, order=4)
    X_bands_val = apply_filter_bank(X_f_val, bands, SFREQ, order=4)

    # ---- 2c. Pairwise CSP ----
    csp = PairwiseCSP(m=M, lambda_r=LAMBDA_R)
    csp.fit(X_bands_tr, y_f_tr)
    proj_tr  = csp.transform(X_bands_tr)
    proj_val = csp.transform(X_bands_val)

    # ---- 2d. Concatenate projections → (n_trials, total_features, n_samples) ----
    pairs = sorted(proj_tr.keys())
    X_concat_tr  = np.concatenate([proj_tr[p]  for p in pairs], axis=1)
    X_concat_val = np.concatenate([proj_val[p] for p in pairs], axis=1)

    total_features = X_concat_tr.shape[1]   # 144
    logger.info(
        "CSP concat: train %s  val %s  (features=%d)",
        X_concat_tr.shape, X_concat_val.shape, total_features,
    )

    # ---- 2e. Z-normalisation (fit on training, apply to val) ----
    znorm = ZNormaliser()
    X_norm_tr  = znorm.fit_transform(X_concat_tr)
    X_norm_val = znorm.transform(X_concat_val)

    # ---- 2f. Spike encoding ----
    def _to_spikes(X_norm: np.ndarray) -> torch.Tensor:
        # X_norm: (n_trials, n_features, n_samples) float32
        t = torch.from_numpy(X_norm).to(DEVICE).permute(2, 0, 1)  # (T, B, F)
        return encode_tensor(t, base_thresh=0.001, adapt_inc=0.6, decay=0.95)

    spikes_tr  = _to_spikes(X_norm_tr)   # (1001, n_tr,  144)
    spikes_val = _to_spikes(X_norm_val)  # (1001, n_val, 144)

    logger.info(
        "Spikes: train %s  val %s  firing_rate %.3f",
        tuple(spikes_tr.shape), tuple(spikes_val.shape),
        spikes_tr.mean().item(),
    )

    # ---- 2g. MIBIF feature selection (fit on training spikes) ----
    mibif = MIBIFSelector(feature_percentile=FEAT_PCT, random_state=42)
    spikes_tr_sel  = mibif.fit_transform(spikes_tr,  y_f_tr_0)
    spikes_val_sel = mibif.transform(spikes_val)

    n_selected = spikes_tr_sel.shape[2]
    logger.info(
        "After MIBIF: train %s  val %s  (features %d -> %d)",
        tuple(spikes_tr_sel.shape), tuple(spikes_val_sel.shape),
        total_features, n_selected,
    )

    # ---- 2h. Train ----
    model = SNNClassifier(
        n_input=n_selected,
        n_hidden=HIDDEN,
        n_classes=N_CLASSES,
        population_per_class=POP_PER_CLS,
        beta=BETA,
        dropout_prob=DROPOUT,
    ).to(DEVICE)

    result = train_fold(
        spikes_train=spikes_tr_sel,
        y_train=y_f_tr_0,
        spikes_val=spikes_val_sel,
        y_val=y_f_val_0,
        model=model,
        n_classes=N_CLASSES,
        population_per_class=POP_PER_CLS,
        spike_prob=SPIKE_PROB,
        lr=1e-3,
        weight_decay=0.1,
        epochs=MAX_EPOCHS,
        patience=PATIENCE,
        warmup=WARMUP,
        tau_vr=TAU_VR,
        batch_size=BATCH_SIZE,
        max_time_steps=T_MAX,
        device=DEVICE,
        fold_dir=RESULTS_DIR / f"fold_{fold_idx}",
        log_every=LOG_EVERY,
    )

    elapsed = time.perf_counter() - fold_start
    logger.info(
        "Fold %d/%d  best_val_acc: %.1f%%  (epoch %d)  wall: %.1fs",
        fold_idx + 1, N_FOLDS,
        result.best_val_acc * 100,
        result.best_epoch + 1,
        elapsed,
    )

    fold_results.append(result)
    fold_artifacts.append({
        "bands": bands, "csp": csp, "znorm": znorm, "mibif": mibif, "model": model,
    })

cv_elapsed = time.perf_counter() - cv_start
logger.info("")
logger.info("CV total wall time: %.1f s  (%.1f s/fold)", cv_elapsed, cv_elapsed / N_FOLDS)


# ===========================================================================
# 3. Per-fold summary
# ===========================================================================
sep("Step 3: CV fold summary")

mean_val_acc = log_fold_summary(fold_results, SUBJECT_ID)


# ===========================================================================
# 4. Test-set evaluation using the best-fold model
# ===========================================================================
sep("Step 4: Test-set evaluation (best fold's preprocessing)")

best_fold_idx = int(np.argmax([r.best_val_acc for r in fold_results]))
logger.info(
    "Best fold: %d  (val_acc %.1f%%)",
    best_fold_idx + 1, fold_results[best_fold_idx].best_val_acc * 100,
)

art = fold_artifacts[best_fold_idx]

# Apply best fold's preprocessing chain to the held-out test set
X_test_bands   = apply_filter_bank(X_test, art["bands"], SFREQ, order=4)
proj_test      = art["csp"].transform(X_test_bands)
pairs          = sorted(proj_test.keys())
X_test_concat  = np.concatenate([proj_test[p] for p in pairs], axis=1)
X_test_norm    = art["znorm"].transform(X_test_concat)

spikes_test    = _to_spikes(X_test_norm)
spikes_test_sel = art["mibif"].transform(spikes_test)

logger.info("Test spike tensor: %s", tuple(spikes_test_sel.shape))

test_acc, test_preds = evaluate_model(
    art["model"], spikes_test_sel, y_test_0, DEVICE, batch_size=BATCH_SIZE
)

logger.info(
    "Test accuracy (Subject %d, T_MAX=%d): %.1f%%",
    SUBJECT_ID, T_MAX, test_acc * 100,
)

# Confusion matrix
plot_confusion_matrix(
    y_true=y_test_0,
    y_pred=test_preds,
    class_names=CLASS_NAMES,
    save_path=RESULTS_DIR / "confusion_matrix_test.png",
    title=f"Subject {SUBJECT_ID} — Test Set (T_MAX={T_MAX})",
)


# ===========================================================================
# 5. Assertions and final summary
# ===========================================================================
sep("Step 5: Assertions")

assert len(fold_results) == N_FOLDS, "Wrong number of fold results"
for i, r in enumerate(fold_results):
    assert 0.0 <= r.best_val_acc <= 1.0, f"Fold {i} val_acc out of range"
    assert len(r.train_loss_history) > 0, f"Fold {i} has no loss history"
    assert len(r.val_acc_history) == len(r.train_loss_history)
    assert all(np.isfinite(r.train_loss_history)), f"Fold {i} has non-finite loss"

assert 0.0 <= test_acc <= 1.0, "Test accuracy out of range"
assert len(test_preds) == len(y_test_0), "Prediction count mismatch"

logger.info("")
logger.info("All assertions passed.")
logger.info("")
logger.info("=" * 62)
logger.info("  FINAL RESULTS  (Subject %d, T_MAX=%d, %d-fold CV)",
            SUBJECT_ID, T_MAX, N_FOLDS)
logger.info("=" * 62)
logger.info("  Mean CV val accuracy : %.1f%%", mean_val_acc * 100)
logger.info("  Test accuracy        : %.1f%%", test_acc * 100)
logger.info("  Note: T_MAX=%d (%.1f%% of full 1001-step sequence)",
            T_MAX, 100 * T_MAX / 1001)
logger.info("=" * 62)
