"""End-to-end test script: data load → band selection → filter bank → CSP.

Run from the project root::

    python tests/test_band_csp.py

Logs the array shapes at every major step and asserts key invariants so that
silent shape bugs are caught immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fbcsp_snn import setup_logger
from fbcsp_snn.datasets import load_moabb
from fbcsp_snn.preprocessing import PairwiseCSP, apply_filter_bank

logger = setup_logger("test_band_csp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET = "BNCI2014_001"
SUBJECT_ID = 1
SFREQ = 250        # Hz
N_CHANNELS = 22
N_BANDS = 6
M = 2              # CSP filters per end  →  2m = 4 filters per (band, pair)
LAMBDA_R = 0.0001
N_CLASSES = 4
N_PAIRS = N_CLASSES * (N_CLASSES - 1) // 2   # = 6


def _section(title: str) -> None:
    logger.info("─" * 60)
    logger.info("  %s", title)
    logger.info("─" * 60)


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------
_section("Step 1: Load MOABB data")

X_train, y_train, X_test, y_test = load_moabb(DATASET, SUBJECT_ID, n_classes=N_CLASSES)

n_trials_train, n_ch, n_samples = X_train.shape

logger.info("X_train : %s  dtype=%s", X_train.shape, X_train.dtype)
logger.info("y_train : %s  classes=%s", y_train.shape, np.unique(y_train).tolist())
logger.info("X_test  : %s", X_test.shape)
logger.info("y_test  : %s", y_test.shape)

assert X_train.ndim == 3, "Expected 3-D X_train"
assert n_ch == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {n_ch}"
assert set(np.unique(y_train).tolist()) == set(range(1, N_CLASSES + 1)), \
    "Labels must be 1-indexed"


# ---------------------------------------------------------------------------
# Step 2 — Fixed six-band overlapping filter bank
# ---------------------------------------------------------------------------
_section("Step 2: Fixed frequency bands")

selected_bands = [(4, 8), (8, 14), (12, 18), (16, 24), (20, 30), (26, 40)]

logger.info("Using %d fixed overlapping bands:", len(selected_bands))
for i, (lo, hi) in enumerate(selected_bands):
    logger.info("  Band %d: %.1f–%.1f Hz", i, lo, hi)

assert len(selected_bands) == N_BANDS, \
    f"Expected {N_BANDS} bands, got {len(selected_bands)}"


# ---------------------------------------------------------------------------
# Step 3 — Bandpass filter bank
# ---------------------------------------------------------------------------
_section("Step 3: Bandpass filter bank")

X_bands_train = apply_filter_bank(X_train, selected_bands, sfreq=SFREQ, order=4)
X_bands_test  = apply_filter_bank(X_test,  selected_bands, sfreq=SFREQ, order=4)

logger.info("Per-band arrays (train):")
for b, (arr, (lo, hi)) in enumerate(zip(X_bands_train, selected_bands)):
    logger.info("  Band %d (%.1f–%.1f Hz): %s  dtype=%s", b, lo, hi, arr.shape, arr.dtype)

assert len(X_bands_train) == N_BANDS
for arr in X_bands_train:
    assert arr.shape == X_train.shape, \
        f"Filter bank output shape mismatch: {arr.shape} vs {X_train.shape}"

# Conceptual concatenated shape matches pipeline architecture documentation
concat_shape = (n_trials_train, N_CHANNELS * N_BANDS, n_samples)
logger.info("Conceptual concat shape: %s", concat_shape)


# ---------------------------------------------------------------------------
# Step 4 — Pairwise CSP (fit on training data only)
# ---------------------------------------------------------------------------
_section("Step 4: Pairwise CSP — fit")

csp = PairwiseCSP(m=M, lambda_r=LAMBDA_R)
csp.fit(X_bands_train, y_train)

logger.info("Fitted pairs    : %s", csp.pairs_)
logger.info("Number of filters stored: %d  (expect %d × %d = %d)",
            len(csp.filters_),
            N_BANDS, N_PAIRS, N_BANDS * N_PAIRS)

assert len(csp.filters_) == N_BANDS * N_PAIRS
for key, W in csp.filters_.items():
    b_idx, pair = key
    assert W.shape == (N_CHANNELS, 2 * M), \
        f"Filter shape {W.shape} at key {key}; expected ({N_CHANNELS}, {2 * M})"

logger.info("All filter shapes: (%d, %d)  ✓", N_CHANNELS, 2 * M)


# ---------------------------------------------------------------------------
# Step 5 — CSP transform (train and test)
# ---------------------------------------------------------------------------
_section("Step 5: Pairwise CSP — transform")

projections_train = csp.transform(X_bands_train)
projections_test  = csp.transform(X_bands_test)

expected_features = 2 * M * N_BANDS  # 4 filters × 6 bands = 24

logger.info("Expected features per pair: 2×m×n_bands = 2×%d×%d = %d",
            M, N_BANDS, expected_features)

logger.info("Projected shapes (train):")
for pair, proj in sorted(projections_train.items()):
    logger.info("  Pair %s → %s  dtype=%s", pair, proj.shape, proj.dtype)
    assert proj.shape == (n_trials_train, expected_features, n_samples), \
        f"Unexpected projection shape {proj.shape} for pair {pair}"

logger.info("Projected shapes (test):")
for pair, proj in sorted(projections_test.items()):
    logger.info("  Pair %s → %s", pair, proj.shape)
    assert proj.shape[1] == expected_features and proj.shape[2] == n_samples


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_section("Summary")

logger.info("Raw EEG (train)       : %s", X_train.shape)
logger.info("Filter bank per band  : %d × %s", N_BANDS, X_bands_train[0].shape)
logger.info("CSP pairs             : %d", N_PAIRS)
logger.info("Features per pair     : %d  (2×%d filters × %d bands)",
            expected_features, M, N_BANDS)
logger.info("Total CSP features    : %d  (%d pairs × %d features)",
            N_PAIRS * expected_features, N_PAIRS, expected_features)
logger.info("")
logger.info("All assertions passed — band selection and CSP pipeline OK.")
