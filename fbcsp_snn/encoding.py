"""Spike encoding with JIT-compiled inner loops.

Three encoders are provided:

Delta encoder (default, ``encoder_type='delta'``)
-------------------------------------------------
For each timestep *t* and feature *f*:

1. Compute ``|x[t,f] - x[t-1,f]|``
2. If delta > threshold[f]: emit a spike (1), else 0
3. After each step: ``threshold *= decay``
4. On spike: ``threshold += adapt_inc``

ADM encoder (``encoder_type='adm'``)
-------------------------------------
Asynchronous Delta Modulation — reference-tracking with ON/OFF polarity.
For each timestep *t* and feature *f*:

1. Compute ``diff = x[t,f] - v_ref[f]``
2. ON spike  if ``diff >  threshold[f]`` → ``v_ref[f] += threshold[f]``
3. OFF spike if ``diff < -threshold[f]`` → ``v_ref[f] -= threshold[f]``
4. After each step: ``threshold *= decay``
5. On any spike: ``threshold += adapt_inc``

ADM output shape is ``(T, batch, 2*n_features)`` — ON channels first, OFF
channels second.  The doubled feature dimension flows through MIBIF and the
SNN automatically via ``spikes.shape[2]``.

Fixed-threshold encoder (``encoder_type='fixed'``, ablation only)
-------------------------------------------------------------------
Same delta rule as the adaptive encoder, but the threshold never changes —
no ``adapt_inc``/``decay`` dynamics. Isolates whether adaptivity itself is
responsible for the adaptive-threshold encoder's contribution, or whether any
delta-based binary encoder would do:

1. Compute ``|x[t,f] - x[t-1,f]|``
2. If delta > ``base_thresh``: emit a spike (1), else 0

Both loops are JIT-compiled via ``@torch.jit.script``; inner operations are
vectorised across batch and feature dimensions.

Input  → ``encode_csp_projections`` / ``encode_tensor``:  dict or tensor
Output → binary spike tensor ``(n_timesteps, n_trials, n_features[*2])``
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from fbcsp_snn import DEVICE, setup_logger


logger: logging.Logger = setup_logger(__name__)

# Type alias shared with preprocessing
_Pair = Tuple[int, int]


# ---------------------------------------------------------------------------
# JIT-compiled encoder kernel
# ---------------------------------------------------------------------------

@torch.jit.script
def _adaptive_threshold_encode_jit(
    x: torch.Tensor,
    base_thresh: float,
    adapt_inc: float,
    decay: float,
) -> torch.Tensor:
    """JIT-compiled adaptive threshold spike encoder.

    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape ``(T, batch, n_features)``.  Must be float.
    base_thresh : float
        Initial threshold value applied uniformly to every feature.
    adapt_inc : float
        Amount added to the threshold whenever a spike is emitted.
    decay : float
        Multiplicative decay applied to the threshold at every timestep.

    Returns
    -------
    torch.Tensor
        Binary spike tensor, same shape as *x*.  dtype matches *x*.
    """
    T: int = x.shape[0]
    batch: int = x.shape[1]
    n_feat: int = x.shape[2]

    spikes = torch.zeros_like(x)
    threshold = torch.full(
        (batch, n_feat),
        base_thresh,
        dtype=x.dtype,
        device=x.device,
    )

    for t in range(1, T):
        delta = torch.abs(x[t] - x[t - 1])          # (batch, n_feat)
        fired = (delta > threshold).to(x.dtype)       # (batch, n_feat)
        spikes[t] = fired
        # Decay threshold every step; increment on spike
        threshold = threshold * decay + fired * adapt_inc

    return spikes


@torch.jit.script
def _fixed_threshold_encode_jit(
    x: torch.Tensor,
    base_thresh: float,
) -> torch.Tensor:
    """JIT-compiled fixed-threshold delta encoder (ablation).

    Same delta rule as :func:`_adaptive_threshold_encode_jit` but the
    threshold is held constant at *base_thresh* for the entire trial — no
    adaptive increment or decay. Isolates whether threshold adaptivity
    itself matters, independent of the underlying delta-encoding scheme.

    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape ``(T, batch, n_features)``.  Must be float.
    base_thresh : float
        Constant threshold applied at every timestep.

    Returns
    -------
    torch.Tensor
        Binary spike tensor, same shape as *x*.  dtype matches *x*.
    """
    T: int = x.shape[0]

    spikes = torch.zeros_like(x)

    for t in range(1, T):
        delta = torch.abs(x[t] - x[t - 1])          # (batch, n_feat)
        spikes[t] = (delta > base_thresh).to(x.dtype)

    return spikes


@torch.jit.script
def _adaptive_threshold_encode_noisy_jit(
    x: torch.Tensor,
    init_threshold: torch.Tensor,
    adapt_inc: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled adaptive-threshold encoder with per-feature threshold,
    adapt_inc, and decay (B15 Tier B — comparator offset + adaptation-rate
    noise).

    Identical adaptation dynamics to :func:`_adaptive_threshold_encode_jit`,
    but all three parameters are per-feature tensors instead of scalars
    shared across every feature — models per-channel comparator offset
    variability *and* per-channel analog time-constant mismatch in the
    adaptation circuitry (the same class of non-ideality as LIF `beta`
    variation, applied to the encoder's own adaptation dynamics). The
    existing production kernel is left untouched; this is a separate,
    additive path used only for reliability/noise-robustness testing.

    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape ``(T, batch, n_features)``.  Must be float.
    init_threshold : torch.Tensor
        Per-feature initial threshold, shape ``(n_features,)``, already
        perturbed by the caller (e.g. via Gaussian offset noise).
    adapt_inc : torch.Tensor
        Per-feature threshold increment on spike, shape ``(n_features,)``.
    decay : torch.Tensor
        Per-feature multiplicative threshold decay, shape ``(n_features,)``.

    Returns
    -------
    torch.Tensor
        Binary spike tensor, same shape as *x*.  dtype matches *x*.
    """
    T: int = x.shape[0]
    batch: int = x.shape[1]

    spikes = torch.zeros_like(x)
    threshold = init_threshold.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(batch, -1).clone()
    adapt_inc_b = adapt_inc.to(dtype=x.dtype, device=x.device).unsqueeze(0)   # (1, n_feat), broadcasts
    decay_b = decay.to(dtype=x.dtype, device=x.device).unsqueeze(0)          # (1, n_feat), broadcasts

    for t in range(1, T):
        delta = torch.abs(x[t] - x[t - 1])          # (batch, n_feat)
        fired = (delta > threshold).to(x.dtype)       # (batch, n_feat)
        spikes[t] = fired
        threshold = threshold * decay_b + fired * adapt_inc_b

    return spikes


@torch.jit.script
def _adm_encode_jit(
    x: torch.Tensor,
    base_thresh: float,
    adapt_inc: float,
    decay: float,
) -> torch.Tensor:
    """JIT-compiled Asynchronous Delta Modulation (ADM) encoder.

    Maintains a reference voltage ``v_ref`` per (batch, feature). Emits an
    ON spike when ``x[t] > v_ref + threshold`` and an OFF spike when
    ``x[t] < v_ref - threshold``.  On each spike the reference steps by
    ``±threshold``; the threshold decays every step and increments on any spike.

    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape ``(T, batch, n_features)``.  Must be float.
    base_thresh : float
        Initial threshold value.
    adapt_inc : float
        Amount added to threshold on each spike.
    decay : float
        Multiplicative threshold decay per timestep.

    Returns
    -------
    torch.Tensor
        Binary spike tensor, shape ``(T, batch, 2*n_features)``.
        Channels ``[0 : n_features]`` are ON spikes;
        channels ``[n_features : 2*n_features]`` are OFF spikes.
    """
    T: int = x.shape[0]
    batch: int = x.shape[1]
    n_feat: int = x.shape[2]

    on_spikes = torch.zeros_like(x)
    off_spikes = torch.zeros_like(x)

    v_ref = x[0].clone()                                        # (batch, n_feat)
    threshold = torch.full(
        (batch, n_feat), base_thresh, dtype=x.dtype, device=x.device
    )

    for t in range(T):
        diff = x[t] - v_ref                                     # (batch, n_feat)
        on_fire  = (diff >  threshold).to(x.dtype)
        off_fire = (diff < -threshold).to(x.dtype)

        on_spikes[t]  = on_fire
        off_spikes[t] = off_fire

        # Step reference toward the signal
        v_ref = v_ref + on_fire * threshold - off_fire * threshold
        # Decay threshold every step; increment on any spike
        threshold = threshold * decay + (on_fire + off_fire) * adapt_inc

    return torch.cat([on_spikes, off_spikes], dim=2)


def adm_reconstruction_rmse(
    x: np.ndarray,
    base_thresh: float = 0.001,
    adapt_inc: float = 0.6,
    decay: float = 0.95,
) -> float:
    """Compute normalised RMSE between original signal and ADM reconstruction.

    Runs the ADM algorithm on a single ``(T, n_features)`` array and tracks
    ``v_ref`` at every timestep.  The reconstruction is ``v_ref`` itself.
    RMSE is normalised by the signal RMS so the result is dimensionless.

    Parameters
    ----------
    x : np.ndarray
        Single-trial signal, shape ``(T, n_features)``.
    base_thresh, adapt_inc, decay : float
        ADM hyperparameters (must match those used during encoding).

    Returns
    -------
    float
        Normalised RMSE in ``[0, 1]``.  Values below 0.05 satisfy the ≤5%
        target from the publication plan.
    """
    T, n_feat = x.shape
    x = x.astype(np.float32)

    v_ref = x[0].copy()
    threshold = np.full(n_feat, base_thresh, dtype=np.float32)
    v_ref_hist = np.empty_like(x)
    v_ref_hist[0] = v_ref

    for t in range(1, T):
        diff = x[t] - v_ref
        on_fire  = (diff >  threshold).astype(np.float32)
        off_fire = (diff < -threshold).astype(np.float32)
        v_ref = v_ref + on_fire * threshold - off_fire * threshold
        threshold = threshold * decay + (on_fire + off_fire) * adapt_inc
        v_ref_hist[t] = v_ref

    signal_rms = float(np.sqrt(np.mean(x ** 2)))
    if signal_rms < 1e-10:
        return 0.0
    error_rms = float(np.sqrt(np.mean((x - v_ref_hist) ** 2)))
    return error_rms / signal_rms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_csp_projections(
    projections: Dict[_Pair, np.ndarray],
    base_thresh: float = 0.001,
    adapt_inc: float = 0.6,
    decay: float = 0.95,
    encoder_type: str = "delta",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode a dict of CSP-projected time-series into binary spike trains.

    All class-pair projections are concatenated along the feature axis before
    encoding so that the single returned tensor covers the full feature set.
    Pair ordering is alphabetically sorted for reproducibility.

    Parameters
    ----------
    projections : Dict[Tuple[int, int], np.ndarray]
        Output of :meth:`PairwiseCSP.transform`.  Maps each class pair to an
        array of shape ``(n_trials, n_features_per_pair, n_samples)``.
    base_thresh : float
        Initial adaptive threshold.
    adapt_inc : float
        Threshold increment per emitted spike.
    decay : float
        Per-timestep threshold decay factor (< 1).
    encoder_type : str
        ``'delta'`` — adaptive-threshold delta encoder (default).
        ``'adm'``   — ON/OFF ADM encoder; output features doubled.
        ``'fixed'`` — ablation: constant-threshold delta encoder, no
        adaptation.
    device : Optional[torch.device]
        Target device.  Defaults to :data:`fbcsp_snn.DEVICE`.

    Returns
    -------
    torch.Tensor
        Binary spike tensor, shape ``(n_samples, n_trials, total_features)``
        for ``'delta'``, or ``(n_samples, n_trials, 2*total_features)`` for
        ``'adm'``.  dtype ``torch.float32``, on *device*.
    """
    if device is None:
        device = DEVICE

    # Sort pairs for deterministic feature ordering
    pairs = sorted(projections.keys())
    arrays = [projections[p] for p in pairs]

    # Concatenate along feature dim → (n_trials, total_features, n_samples)
    X = np.concatenate(arrays, axis=1).astype(np.float32)

    # Move to device and permute → (n_samples, n_trials, total_features) = (T, B, F)
    X_t = torch.from_numpy(X).to(device).permute(2, 0, 1)

    spikes = encode_tensor(X_t, base_thresh, adapt_inc, decay, encoder_type)

    firing_rate = spikes.mean().item()
    logger.info(
        "Spike encoding (%s) — input: %s -> spikes: %s  mean firing rate: %.4f",
        encoder_type,
        tuple(X_t.shape),
        tuple(spikes.shape),
        firing_rate,
    )
    return spikes


def encode_tensor(
    X: torch.Tensor,
    base_thresh: float = 0.001,
    adapt_inc: float = 0.6,
    decay: float = 0.95,
    encoder_type: str = "delta",
) -> torch.Tensor:
    """Encode a pre-arranged tensor ``(T, batch, features)`` to spikes.

    Convenience wrapper for use when data is already a tensor in the correct
    layout (e.g. inside the training loop after z-normalisation).

    Parameters
    ----------
    X : torch.Tensor
        Input signal ``(T, batch, n_features)``.
    base_thresh, adapt_inc, decay : float
        Encoding hyperparameters.
    encoder_type : str
        ``'delta'`` (default), ``'adm'``, or ``'fixed'`` (ablation: constant
        threshold, no adaptation).  ADM output has shape
        ``(T, batch, 2*n_features)``.

    Returns
    -------
    torch.Tensor
        Binary spikes, same shape as *X* for ``'delta'``/``'fixed'``, or
        ``(T, batch, 2*n_features)`` for ``'adm'``.
    """
    if encoder_type == "adm":
        return _adm_encode_jit(X, base_thresh, adapt_inc, decay)
    if encoder_type == "fixed":
        return _fixed_threshold_encode_jit(X, base_thresh)
    return _adaptive_threshold_encode_jit(X, base_thresh, adapt_inc, decay)


def _noisy_per_feature(
    nominal: float,
    n_features: int,
    sigma_frac: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: Optional[int],
    clamp_min: float = 1e-6,
    clamp_max: Optional[float] = None,
) -> torch.Tensor:
    """Build a per-feature tensor of *nominal* perturbed by Gaussian noise."""
    if sigma_frac == 0.0:
        return torch.full((n_features,), nominal, dtype=dtype, device=device)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = torch.randn(n_features, generator=gen, device=device, dtype=dtype)
    values = nominal + noise * sigma_frac * nominal
    if clamp_max is not None:
        return values.clamp(min=clamp_min, max=clamp_max)
    return values.clamp(min=clamp_min)


def encode_tensor_with_threshold_noise(
    X: torch.Tensor,
    base_thresh: float = 0.001,
    adapt_inc: float = 0.6,
    decay: float = 0.95,
    sigma_frac: float = 0.0,
    adapt_inc_sigma_frac: float = 0.0,
    decay_sigma_frac: float = 0.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Adaptive-threshold encoding with per-feature analog non-ideality noise
    (B15 Tier B).

    Models three independent sources, each optional via its own
    ``sigma_frac``:

    - **Comparator offset** (``sigma_frac``): each feature's initial
      threshold perturbed around ``base_thresh``, instead of every feature
      sharing exactly the same value.
    - **Adaptation-rate mismatch** (``adapt_inc_sigma_frac``,
      ``decay_sigma_frac``): each feature's threshold-adaptation dynamics
      (how fast it rises on a spike, how fast it decays between spikes)
      perturbed around ``adapt_inc``/``decay`` — the same class of analog
      time-constant mismatch already tested for the LIF neuron's ``beta``
      (:func:`fbcsp_snn.quantization.inject_beta_noise`), applied here to the
      encoder's own adaptation circuitry instead.

    All three default to ``0.0`` (noise-free), reproducing
    :func:`encode_tensor` with ``encoder_type='delta'`` exactly. Only the
    delta encoder supports this (ADM/fixed are unaffected by this function).

    Parameters
    ----------
    X : torch.Tensor
        Input signal ``(T, batch, n_features)``.
    base_thresh, adapt_inc, decay : float
        Nominal encoding hyperparameters (same meaning as
        :func:`encode_tensor`).
    sigma_frac : float
        Per-feature threshold noise std as a fraction of *base_thresh*.
    adapt_inc_sigma_frac : float
        Per-feature ``adapt_inc`` noise std as a fraction of *adapt_inc*.
    decay_sigma_frac : float
        Per-feature ``decay`` noise std as a fraction of *decay*.
    seed : Optional[int]
        Base RNG seed; offset per parameter so threshold/adapt_inc/decay
        don't all get identical noise draws.

    Returns
    -------
    torch.Tensor
        Binary spikes, same shape as *X*.
    """
    n_features = X.shape[2]
    thresh_seed = seed
    adapt_seed = None if seed is None else seed + 1
    decay_seed = None if seed is None else seed + 2

    init_threshold = _noisy_per_feature(
        base_thresh, n_features, sigma_frac, X.dtype, X.device, thresh_seed,
    )
    adapt_inc_t = _noisy_per_feature(
        adapt_inc, n_features, adapt_inc_sigma_frac, X.dtype, X.device, adapt_seed,
    )
    # decay is a multiplicative factor in (0, 1) — clamp to stay a valid decay.
    decay_t = _noisy_per_feature(
        decay, n_features, decay_sigma_frac, X.dtype, X.device, decay_seed,
        clamp_min=1e-3, clamp_max=0.999,
    )
    return _adaptive_threshold_encode_noisy_jit(X, init_threshold, adapt_inc_t, decay_t)
