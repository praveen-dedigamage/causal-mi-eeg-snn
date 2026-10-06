"""Integration test: CSP projections -> spike encoding -> SNN -> Van Rossum loss.

Pipeline under test
-------------------
1. Load Subject 1 training data (MOABB, cached)
2. Band selection + filter bank + PairwiseCSP (training data only)
3. Adaptive-threshold spike encoding (JIT compiled)
4. 2-layer LIF SNN forward pass
5. Van Rossum loss vs. population-coded target spikes
6. Backward pass — verify loss is finite and gradients exist

Run from project root::

    python tests/test_spike_snn.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.datasets import load_moabb
from fbcsp_snn.encoding import encode_csp_projections
from fbcsp_snn.losses import make_target_spikes, van_rossum_loss
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import PairwiseCSP, apply_filter_bank

logger = setup_logger("test_spike_snn")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET      = "BNCI2014_001"
SUBJECT_ID   = 1
SFREQ        = 250
N_CLASSES    = 4
N_BANDS      = 6
M            = 2           # CSP filters per end
LAMBDA_R     = 0.0001
HIDDEN       = 64
POP_PER_CLS  = 20
BETA         = 0.95
DROPOUT      = 0.5
TAU          = 10.0        # Van Rossum kernel time constant (timesteps)
SPIKE_PROB   = 0.7
BATCH_SIZE   = 16          # smaller batch keeps the forward-pass test fast


def sep(title: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


# ===========================================================================
# 1. Data
# ===========================================================================
sep("Step 1: Load data")

X_train, y_train, X_test, y_test = load_moabb(
    DATASET, SUBJECT_ID, n_classes=N_CLASSES
)
logger.info("X_train: %s  y_train: %s", X_train.shape, y_train.shape)


# ===========================================================================
# 2. Band selection + filter bank + CSP  (train split only)
# ===========================================================================
sep("Step 2: Filter bank -> PairwiseCSP")

bands = [(4, 8), (8, 14), (12, 18), (16, 24), (20, 30), (26, 40)]
logger.info("Bands: %s", bands)

X_bands_train = apply_filter_bank(X_train, bands, sfreq=SFREQ, order=4)
X_bands_test  = apply_filter_bank(X_test,  bands, sfreq=SFREQ, order=4)

csp = PairwiseCSP(m=M, lambda_r=LAMBDA_R)
csp.fit(X_bands_train, y_train)

proj_train = csp.transform(X_bands_train)   # {pair: (288, 24, 1001)}
proj_test  = csp.transform(X_bands_test)

n_features_per_pair = 2 * M * N_BANDS       # 24
n_pairs = len(csp.pairs_)                   # 6
total_features = n_pairs * n_features_per_pair  # 144
logger.info(
    "CSP projections per pair: %s  total features: %d",
    list(proj_train.values())[0].shape,
    total_features,
)


# ===========================================================================
# 3. Spike encoding
# ===========================================================================
sep("Step 3: Adaptive-threshold spike encoding")

t0 = time.perf_counter()
spikes_train = encode_csp_projections(
    proj_train,
    base_thresh=0.001,
    adapt_inc=0.6,
    decay=0.95,
    device=DEVICE,
)
enc_time = time.perf_counter() - t0

T, B, F = spikes_train.shape
logger.info("Spike tensor (train): T=%d  B=%d  F=%d", T, B, F)
logger.info("Encoding time: %.3f s", enc_time)
logger.info("Mean firing rate: %.4f  (spikes / (T*B*F))", spikes_train.mean().item())
logger.info("Min: %.1f  Max: %.1f  dtype: %s  device: %s",
            spikes_train.min().item(), spikes_train.max().item(),
            spikes_train.dtype, spikes_train.device)

assert spikes_train.shape == (1001, 288, total_features), \
    f"Unexpected spike shape: {spikes_train.shape}"
assert spikes_train.min() >= 0.0 and spikes_train.max() <= 1.0, \
    "Spike values must be binary (0 or 1)"


# ===========================================================================
# 4. SNN forward pass (small batch, full time)
# ===========================================================================
sep("Step 4: SNN forward pass")

model = SNNClassifier(
    n_input=total_features,
    n_hidden=HIDDEN,
    n_classes=N_CLASSES,
    population_per_class=POP_PER_CLS,
    beta=BETA,
    dropout_prob=DROPOUT,
).to(DEVICE)

model.train()

# Take a small batch for the forward-pass / gradient test
batch_spikes = spikes_train[:, :BATCH_SIZE, :]   # (T=1001, 16, 144)
logger.info("Model input batch: %s  on %s", tuple(batch_spikes.shape), batch_spikes.device)

t0 = time.perf_counter()
spk_out, mem_out = model(batch_spikes)
fwd_time = time.perf_counter() - t0

logger.info("spk_out: %s  dtype: %s", tuple(spk_out.shape), spk_out.dtype)
logger.info("mem_out: %s  dtype: %s", tuple(mem_out.shape), mem_out.dtype)
logger.info("SNN forward time: %.3f s", fwd_time)
logger.info("Output firing rate: %.4f", spk_out.mean().item())

assert spk_out.shape == (T, BATCH_SIZE, N_CLASSES * POP_PER_CLS), \
    f"Unexpected spk_out shape: {spk_out.shape}"
assert mem_out.shape == spk_out.shape, "mem_out shape must match spk_out"


# ===========================================================================
# 5. Van Rossum loss
# ===========================================================================
sep("Step 5: Van Rossum loss")

# Labels for the batch: y_train is 1-indexed, convert to 0-indexed for model
y_batch = torch.from_numpy(y_train[:BATCH_SIZE] - 1).long().to(DEVICE)  # 0-indexed

spk_target = make_target_spikes(
    y=y_batch,
    n_classes=N_CLASSES,
    population_per_class=POP_PER_CLS,
    T=T,
    spike_prob=SPIKE_PROB,
).to(DEVICE)

logger.info("Target spikes: %s  mean rate: %.4f",
            tuple(spk_target.shape), spk_target.mean().item())
logger.info("Target class composition: %s",
            {c.item(): (y_batch == c).sum().item()
             for c in torch.unique(y_batch)})

loss = van_rossum_loss(spk_out, spk_target, tau=TAU, dt=1.0)

logger.info("Van Rossum loss: %.6f", loss.item())

assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


# ===========================================================================
# 6. Backward pass
# ===========================================================================
sep("Step 6: Backward pass")

t0 = time.perf_counter()
loss.backward()
bwd_time = time.perf_counter() - t0

logger.info("Backward pass time: %.3f s", bwd_time)

# Check that all trainable parameters received gradients
grad_norms: dict[str, float] = {}
missing_grad: list[str] = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            missing_grad.append(name)
        else:
            grad_norms[name] = param.grad.norm().item()

if missing_grad:
    logger.warning("Parameters without gradients: %s", missing_grad)
else:
    logger.info("All %d parameter tensors have gradients.", len(grad_norms))

logger.info("Gradient norms:")
for name, norm in grad_norms.items():
    logger.info("  %-35s  ||grad|| = %.6f", name, norm)

assert not missing_grad, f"Missing gradients on: {missing_grad}"
assert all(torch.isfinite(torch.tensor(v)) for v in grad_norms.values()), \
    "Non-finite gradient detected"


# ===========================================================================
# 7. Decode (sanity check, no loss)
# ===========================================================================
sep("Step 7: WTA decoding sanity check")

with torch.no_grad():
    pred = model.decode(spk_out)

logger.info("Predictions: %s", pred.tolist())
logger.info("True labels : %s", y_batch.tolist())
acc = (pred == y_batch).float().mean().item()
logger.info("Batch accuracy (random init, expect ~25%%): %.1f%%", acc * 100)

assert pred.shape == (BATCH_SIZE,), f"Unexpected pred shape: {pred.shape}"


# ===========================================================================
# Summary
# ===========================================================================
sep("Summary")
logger.info("Spike tensor shape         : %s  (T, B, F)", tuple(spikes_train.shape))
logger.info("SNN output shape           : %s  (T, B, n_output)", tuple(spk_out.shape))
logger.info("Loss value                 : %.6f  (finite: %s)",
            loss.item(), torch.isfinite(loss).item())
logger.info("Gradients present          : %d / %d parameter tensors",
            len(grad_norms), len(grad_norms) + len(missing_grad))
logger.info("Encoding time              : %.3f s", enc_time)
logger.info("Forward time               : %.3f s", fwd_time)
logger.info("Backward time              : %.3f s", bwd_time)
logger.info("")
logger.info("All assertions passed -- encoding, model, and loss pipeline OK.")
