"""Analog hardware non-ideality injection for the reliability sweep (B15).

Real analog circuits don't fail from having fewer bits — they fail from
thermal noise, device-to-device mismatch, and comparator offset drift. The
``inject_*_noise`` functions below add Gaussian noise sized as a fraction of
the nominal value, modelling specific physical non-idealities:

- :func:`inject_weight_noise_tensor` / :func:`inject_weight_noise_array` —
  conductance variation in an analog crossbar array (CSP or SNN weights).
- :func:`inject_beta_noise` — RC time-constant mismatch across fabricated
  leaky-integrator circuits (per-neuron membrane decay).
- :func:`inject_ea_whitener_noise` — Euclidean Alignment whitener eigenvalue
  drift.
- :func:`inject_znorm_noise` — z-normalisation reference/gain drift.

All noise functions return a modified copy and accept a ``seed`` for
reproducibility; callers doing Monte Carlo repetition (B15 Tier C) should
vary the seed across repeats and aggregate mean/std, not rely on any single
noisy draw.

Digital post-training quantisation (INT8 whole-model, CSP-bit PTQ, and the
joint CSP+SNN sweep) previously lived in this module too, but was retired
2026-07-11 in favour of this analog noise model, which represents the
paper's neuromorphic hardware target more directly.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fbcsp_snn import setup_logger
from fbcsp_snn.model import SNNClassifier

logger: logging.Logger = setup_logger(__name__)


def inject_csp_filter_noise(
    filters: Dict,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> Dict:
    """Return a copy of the CSP filter dict with Gaussian noise injected.

    Models analog crossbar conductance variation, per filter matrix.

    Parameters
    ----------
    filters : Dict
        Mapping ``(band_idx, pair) -> np.ndarray``, as stored in
        :attr:`PairwiseCSP.filters_`.
    sigma_frac : float
        Per-matrix noise std as a fraction of that matrix's peak magnitude.
    seed : Optional[int]
        Base RNG seed; incremented per matrix so different (band, pair)
        filters don't get identical noise patterns.

    Returns
    -------
    Dict
        New dict with noise-injected filter matrices.
    """
    noisy: Dict = {}
    for i, (key, W) in enumerate(filters.items()):
        layer_seed = None if seed is None else seed + i
        noisy[key] = inject_weight_noise_array(W, sigma_frac, seed=layer_seed)

    logger.info(
        "Weight-noise CSP: injected sigma_frac=%.4f into %d filter matrices",
        sigma_frac, len(filters),
    )
    return noisy


def inject_ea_whitener_noise(
    ea_whiteners: Dict,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> Dict:
    """Return a copy of the Euclidean Alignment whitener dict with noise injected.

    Perturbs each whitener's own eigenvalues (its per-direction whitening
    gain) by a relative amount, rather than its raw matrix entries.

    This is a deliberate redesign, not the original approach: an earlier
    version perturbed ``R^{-1/2}``'s raw entries directly via
    :func:`inject_weight_noise_array`, which scales noise by one *global*
    peak magnitude across the whole matrix. ``R^{-1/2}``'s entries can span
    a wide dynamic range — directions with a small original covariance
    eigenvalue get a large inverse-square-root gain — so noise scaled to
    the global peak is wildly disproportionate for the matrix's smaller,
    well-conditioned entries, in the same way (and same underlying flaw)
    that produced the confirmed :func:`~fbcsp_snn.preprocessing.
    bandpass_filter_noisy` instability bug: observed empirically as this sweep collapsing to chance level
    almost immediately and its spike-event count exploding ~4x, matching
    the same failure signature. Since ``R^{-1/2}`` is symmetric
    positive-definite by construction (``PairwiseCSP._compute_ea_whitener``
    builds it via eigendecomposition), perturbing its own eigenvalues by a
    relative amount and reconstructing keeps the result symmetric
    positive-definite — a valid whitening transform — for any noise
    magnitude, the same "stable by construction" property as the filter fix.

    Parameters
    ----------
    ea_whiteners : Dict
        Mapping ``band_idx -> np.ndarray``, as stored in
        :attr:`PairwiseCSP.ea_whiteners_`.
    sigma_frac : float
        Relative noise std applied independently to each eigenvalue of the
        whitener: each eigenvalue is redrawn from
        ``N(nominal, (sigma_frac * nominal)^2)``, clamped to stay positive.
    seed : Optional[int]
        Base RNG seed; incremented per band.

    Returns
    -------
    Dict
        New dict with noise-injected whitener matrices.
    """
    from scipy.linalg import eigh

    noisy: Dict = {}
    for i, (key, R_invsqrt) in enumerate(ea_whiteners.items()):
        if sigma_frac == 0.0:
            noisy[key] = R_invsqrt.copy()
            continue
        band_seed = None if seed is None else seed + i
        rng = np.random.default_rng(band_seed)
        vals, vecs = eigh(R_invsqrt)   # symmetric PD by construction
        vals = np.maximum(vals, 1e-10)
        noisy_vals = vals * (1.0 + sigma_frac * rng.standard_normal(vals.shape))
        noisy_vals = np.maximum(noisy_vals, 1e-10)   # keep positive-definite
        noisy[key] = ((vecs * noisy_vals) @ vecs.T).astype(R_invsqrt.dtype)

    logger.info(
        "Eigenvalue-noise EA whitener: injected sigma_frac=%.4f into %d matrices",
        sigma_frac, len(ea_whiteners),
    )
    return noisy


def inject_znorm_noise(
    mean: np.ndarray,
    std: np.ndarray,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to z-normalisation mean/std (analog reference drift).

    Models imprecision in the analog reference voltage (mean, typically a
    subtraction stage) and gain (1/std, typically a multiplicative stage) of
    a physical z-normalisation circuit.

    Parameters
    ----------
    mean : np.ndarray
        Per-feature mean, shape ``(n_features,)``, as stored in
        :attr:`fbcsp_snn.preprocessing.ZNormaliser.mean_`.
    std : np.ndarray
        Per-feature std, shape ``(n_features,)``, as stored in
        :attr:`fbcsp_snn.preprocessing.ZNormaliser.std_`.
    sigma_frac : float
        Noise std as a fraction of each array's peak magnitude.
    seed : Optional[int]
        Base RNG seed; ``std``'s noise uses ``seed + 1`` so mean/std don't
        get identical noise draws.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Noisy ``(mean, std)``. ``std`` is floored at ``1e-8`` (matching
        `ZNormaliser.fit`'s own epsilon) to avoid division by zero/negative
        scale downstream.
    """
    std_seed = None if seed is None else seed + 1
    mean_noisy = inject_weight_noise_array(mean, sigma_frac, seed=seed)
    std_noisy = inject_weight_noise_array(std, sigma_frac, seed=std_seed)
    std_noisy = np.maximum(std_noisy, 1e-8)

    logger.info(
        "Weight-noise Z-norm: injected sigma_frac=%.4f into mean/std (%d features)",
        sigma_frac, mean.shape[0],
    )
    return mean_noisy, std_noisy


# ---------------------------------------------------------------------------
# Analog non-ideality injection (B15 Tier B)
# ---------------------------------------------------------------------------

def inject_weight_noise_tensor(
    x: torch.Tensor,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise to a weight tensor (conductance variation).

    Parameters
    ----------
    x : torch.Tensor
        Weight tensor, any shape.
    sigma_frac : float
        Noise standard deviation as a fraction of ``max(|x|)`` (e.g. ``0.05``
        = noise std is 5% of the tensor's peak magnitude).
    seed : Optional[int]
        RNG seed for reproducibility across Monte Carlo repeats.

    Returns
    -------
    torch.Tensor
        Noisy copy of *x*, same shape and dtype.  *x* is not modified.
    """
    abs_max = x.abs().max().item()
    if abs_max == 0.0 or sigma_frac == 0.0:
        return x.clone()
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return x + noise * (sigma_frac * abs_max)


def inject_weight_noise_array(
    x: np.ndarray,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Numpy equivalent of :func:`inject_weight_noise_tensor`.

    Parameters
    ----------
    x : np.ndarray
        Weight array, any shape.
    sigma_frac : float
        Noise standard deviation as a fraction of ``max(|x|)``.
    seed : Optional[int]
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy copy of *x*, same shape and dtype.
    """
    abs_max = float(np.abs(x).max())
    if abs_max == 0.0 or sigma_frac == 0.0:
        return x.copy()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape).astype(x.dtype)
    return (x + noise * (sigma_frac * abs_max)).astype(x.dtype)


def inject_model_weight_noise(
    model: SNNClassifier,
    sigma_frac: float,
    seed: Optional[int] = None,
    include_bias: bool = True,
) -> SNNClassifier:
    """Return a deep copy of *model* with Gaussian noise added to Linear
    weights (and, by default, biases).

    Biases are included here by default: unlike digital bit-precision
    (where "INT8 biases add hardware complexity for minimal benefit" was
    once the rationale for excluding them), that reasoning doesn't carry
    over to analog noise injection, since biases are just as
    physically-realised (offset currents/voltages) as weights and subject to
    the same device mismatch. Set ``include_bias=False`` to restrict noise
    to weights only.

    Parameters
    ----------
    model : SNNClassifier
        Source model.  Not modified.
    sigma_frac : float
        Per-tensor noise std as a fraction of that tensor's peak magnitude.
    seed : Optional[int]
        Base RNG seed. Each weight/bias tensor gets its own offset so
        multi-layer models don't get identical noise patterns.
    include_bias : bool
        If ``True`` (default), also perturb ``Linear.bias``. If ``False``,
        weights only.

    Returns
    -------
    SNNClassifier
        Deep-copied, noise-injected model.
    """
    model_q = copy.deepcopy(model)
    model_q.eval()

    draw = 0
    with torch.no_grad():
        for module in model_q.modules():
            if isinstance(module, nn.Linear):
                w_seed = None if seed is None else seed + draw
                draw += 1
                noisy_w = inject_weight_noise_tensor(
                    module.weight.data, sigma_frac, seed=w_seed
                )
                module.weight.data.copy_(noisy_w)

                if include_bias and module.bias is not None:
                    b_seed = None if seed is None else seed + draw
                    draw += 1
                    noisy_b = inject_weight_noise_tensor(
                        module.bias.data, sigma_frac, seed=b_seed
                    )
                    module.bias.data.copy_(noisy_b)

    return model_q


def inject_beta_noise(
    model: SNNClassifier,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> SNNClassifier:
    """Perturb each LIF neuron's membrane decay (beta) with per-neuron noise.

    Models RC time-constant mismatch across fabricated analog leaky
    integrators — a source of real-world non-ideality that weight noise
    alone never touches.

    Parameters
    ----------
    model : SNNClassifier
        Source model.  Not modified.
    sigma_frac : float
        Noise std as a fraction of the nominal beta value (e.g. ``0.05`` =
        5% per-neuron variation).
    seed : Optional[int]
        RNG seed for reproducibility.

    Returns
    -------
    SNNClassifier
        Deep-copied model with per-neuron beta perturbed on both LIF layers,
        clamped to ``[0.01, 0.999]`` to stay in a physically valid decay range.

    Note
    ----
    Assumes snnTorch's ``Leaky.beta`` accepts a tensor shaped to the neuron
    count, which is documented snnTorch behaviour for heterogeneous/learnable
    beta. Verified on Puhti (GPU): the per-neuron tensor assignment itself
    works, but the noise tensor originally generated on the default (CPU)
    device while ``lif.beta`` lives on the model's device — fixed by moving
    the noise tensor to ``nominal.device`` before combining.
    """
    model_q = copy.deepcopy(model)
    model_q.eval()
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    with torch.no_grad():
        for lif, n_neurons in (
            (model_q.lif1, model_q.n_hidden),
            (model_q.lif2, model_q.n_output),
        ):
            nominal = lif.beta if torch.is_tensor(lif.beta) else torch.full(
                (n_neurons,), float(lif.beta)
            )
            noise = torch.randn(n_neurons, generator=gen).to(nominal.device)
            noise = noise * sigma_frac * nominal
            lif.beta = (nominal + noise).clamp(0.01, 0.999)

    return model_q
