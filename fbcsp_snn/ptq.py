"""Digital post-training quantisation (PTQ) -- uniform bit-width sweep.

Separate from :mod:`fbcsp_snn.quantization`, which on this branch models
*analog* non-idealities (thermal noise, device mismatch, comparator drift).
The two describe different hardware targets and must not be conflated:

* **Analog** front end: has no bit-width at all; precision is set by component
  tolerance. Use the ``inject_*_noise`` functions there.
* **Digital** implementation (MCU/FPGA, or a digital neuromorphic core such as
  Loihi with its 9-bit synaptic weights): precision *is* a bit-width. Use this
  module.

Every inference-time parameter is covered, so a result reported at N bits
contains no full-precision parameter anywhere:

``ea``      Euclidean-Alignment whiteners, K matrices of (N_ch x N_ch)
``csp``     CSP spatial filters, (K x pairs) matrices of (N_ch x 2m)
``znorm``   z-normalisation mean_ / std_
``snn_w``   nn.Linear.weight for both layers
``snn_b``   nn.Linear.bias for both layers

The EA whiteners matter disproportionately under binary classification: they
are sized K x N_ch^2 and carry no class-pair factor, so when the task drops to
two classes they become the largest front-end parameter block (~8x the CSP
filters at 64 channels). They are also the most fragile: built from
eigenvalue ** -0.5, giving ~100x dynamic range versus ~11x for CSP filters.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from fbcsp_snn import setup_logger
from fbcsp_snn.model import SNNClassifier

logger: logging.Logger = setup_logger(__name__)

_INT8_MAX: int = 127   # symmetric INT8 range: [-127, 127]


# ---------------------------------------------------------------------------
# Core quant/dequant
# ---------------------------------------------------------------------------

def quantize_tensor_symmetric(
    x: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, float]:
    """Simulate symmetric per-tensor quantisation and return dequantised result.

    Parameters
    ----------
    x : torch.Tensor
        Floating-point weight tensor (any shape).
    bits : int
        Bit-width.  Default ``8`` → range ``[-127, 127]``.

    Returns
    -------
    dequantised : torch.Tensor
        Tensor with the same shape and dtype as *x* but rounded to the nearest
        representable INT*bits* value.
    scale : float
        Quantisation scale (``max(|x|) / q_max``).  Zero if *x* is all-zero.
    """
    q_max = float(2 ** (bits - 1) - 1)   # 127 for INT8
    abs_max = x.abs().max().item()
    if abs_max == 0.0:
        return x.clone(), 1.0

    scale = abs_max / q_max
    q = x.div(scale).round().clamp(-q_max, q_max)
    return q.mul(scale), scale


def quantize_array_symmetric(
    x: np.ndarray,
    bits: int = 8,
) -> Tuple[np.ndarray, float]:
    """Numpy equivalent of :func:`quantize_tensor_symmetric`.

    Parameters
    ----------
    x : np.ndarray
        Array to quantise (any shape).
    bits : int
        Bit-width.

    Returns
    -------
    dequantised : np.ndarray
        Rounded array, same shape and dtype as *x*.
    scale : float
        Quantisation scale.
    """
    q_max = float(2 ** (bits - 1) - 1)
    abs_max = float(np.abs(x).max())
    if abs_max == 0.0:
        return x.copy(), 1.0

    scale = abs_max / q_max
    q = np.round(x / scale).clip(-q_max, q_max)
    return (q * scale).astype(x.dtype), scale


# ---------------------------------------------------------------------------
# Model quantisation
# ---------------------------------------------------------------------------

def quantize_model(
    model: SNNClassifier,
    bits: int = 8,
) -> SNNClassifier:
    """Return a deep copy of *model* with INT8-simulated ``Linear`` weights.

    Only ``weight`` parameters of ``torch.nn.Linear`` layers are quantised;
    biases and LIF-neuron parameters (``beta``, ``threshold``, …) are left
    at full precision.

    Parameters
    ----------
    model : SNNClassifier
        Source model.  Not modified.
    bits : int
        Quantisation bit-width.

    Returns
    -------
    SNNClassifier
        Deep-copied model whose linear weights have been rounded to the
        nearest INT*bits* value and dequantised back to float.
    """
    model_q = copy.deepcopy(model)
    model_q.eval()

    scale_log: list[str] = []
    with torch.no_grad():
        for name, module in model_q.named_modules():
            if isinstance(module, nn.Linear):
                dq, scale = quantize_tensor_symmetric(module.weight.data, bits)
                module.weight.data.copy_(dq)
                scale_log.append(f"{name}.weight scale={scale:.6f}")

    logger.info(
        "INT%d model: quantised %d Linear weight tensors — %s",
        bits, len(scale_log), "  ".join(scale_log),
    )
    return model_q


# ---------------------------------------------------------------------------
# CSP filter quantisation
# ---------------------------------------------------------------------------

def quantize_csp_filters(
    filters: Dict,
    bits: int = 8,
) -> Dict:
    """Return a copy of the CSP filter dict with INT8-simulated filter matrices.

    Parameters
    ----------
    filters : Dict
        Mapping ``(band_idx, pair) -> np.ndarray`` of shape
        ``(n_channels, 2 * m)``, as stored in :attr:`PairwiseCSP.filters_`.
    bits : int
        Quantisation bit-width.

    Returns
    -------
    Dict
        New dict with quantised (dequantised) filter matrices.
    """
    quantised: Dict = {}
    total_scale = 0.0

    for key, W in filters.items():
        Wq, scale = quantize_array_symmetric(W, bits)
        quantised[key] = Wq
        total_scale += scale

    logger.info(
        "INT%d CSP: quantised %d filter matrices  (mean scale %.6f)",
        bits, len(filters), total_scale / max(len(filters), 1),
    )
    return quantised


# ---------------------------------------------------------------------------
# Euclidean-Alignment whitener quantisation
# ---------------------------------------------------------------------------

def quantize_ea_whiteners(
    whiteners: Dict,
    bits: int = 8,
) -> Dict:
    """Quantise the per-band Euclidean-Alignment whiteners ``R^{-1/2}``.

    One dense ``(n_channels, n_channels)`` matrix is stored per frequency band
    and applied to every trial at inference, so these are inference-time
    parameters exactly like the CSP filters.  They are quantised here with a
    **separate scale per band**: each whitener has its own spectrum, and a
    shared scale would waste range on the band with the smallest entries.

    These matrices are built from ``eigenvalue ** -0.5``, so their dynamic
    range is wide and the inverse square root amplifies small eigenvalues.
    Expect this stage to be among the first to degrade at low bit-widths.

    Parameters
    ----------
    whiteners : Dict
        Mapping ``band_idx -> np.ndarray`` of shape
        ``(n_channels, n_channels)``, as stored in
        :attr:`PairwiseCSP.ea_whiteners_`.
    bits : int
        Quantisation bit-width.

    Returns
    -------
    Dict
        New dict with quantised (dequantised) whitener matrices.
    """
    quantised: Dict = {}
    scales: list[float] = []

    for band_idx, R in whiteners.items():
        Rq, scale = quantize_array_symmetric(R, bits)
        quantised[band_idx] = Rq
        scales.append(scale)

    logger.info(
        "INT%d EA: quantised %d whitener matrices  (mean scale %.6g)",
        bits, len(whiteners), float(np.mean(scales)) if scales else 0.0,
    )
    return quantised


# ---------------------------------------------------------------------------
# z-normalisation statistic quantisation
# ---------------------------------------------------------------------------

def quantize_znorm(
    znorm,
    bits: int = 8,
    eps: float = 1e-8,
):
    """Return a copy of *znorm* with quantised ``mean_`` and ``std_``.

    Both vectors are fitted on the training fold and applied unchanged at
    inference, so they are stored parameters and belong in a whole-pipeline
    quantisation sweep.

    ``mean_`` and ``std_`` are quantised with independent scales.  After
    quantisation ``std_`` is clamped to *eps*: at low bit-widths a small
    standard deviation can round to zero, which would otherwise divide by
    zero in :meth:`ZNormaliser.transform`.

    Note
    ----
    A symmetric quantiser is used for consistency with the other stages even
    though ``std_`` is strictly positive, which costs one sign bit of range.
    An unsigned quantiser would be marginally more efficient in real hardware.

    Parameters
    ----------
    znorm : ZNormaliser
        Fitted normaliser.  Not modified.
    bits : int
        Quantisation bit-width.
    eps : float
        Floor applied to the quantised standard deviations.

    Returns
    -------
    ZNormaliser
        Deep-copied normaliser with quantised statistics.
    """
    znorm_q = copy.deepcopy(znorm)

    mean_q, mean_scale = quantize_array_symmetric(znorm_q.mean_, bits)
    std_q, std_scale = quantize_array_symmetric(znorm_q.std_, bits)

    n_collapsed = int(np.sum(std_q < eps))
    std_q = np.maximum(std_q, eps)

    znorm_q.mean_ = mean_q
    znorm_q.std_ = std_q

    logger.info(
        "INT%d z-norm: quantised mean (scale %.6g) and std (scale %.6g)"
        "%s",
        bits, mean_scale, std_scale,
        f"  [{n_collapsed} std values floored to {eps:g}]" if n_collapsed else "",
    )
    return znorm_q


# ---------------------------------------------------------------------------
# Full-model quantisation (weights *and* biases)
# ---------------------------------------------------------------------------

def quantize_model_full(
    model: SNNClassifier,
    bits: int = 8,
) -> SNNClassifier:
    """Quantise every ``Linear`` weight **and bias** in *model*.

    :func:`quantize_model` deliberately leaves biases at full precision, which
    is standard practice for INT8 inference.  For a whole-pipeline bit-width
    sweep that exemption is not defensible: a pipeline described as "4-bit"
    must not retain FP32 parameters anywhere.  Weights and biases are given
    independent per-tensor scales, as their dynamic ranges differ.

    Parameters
    ----------
    model : SNNClassifier
        Source model.  Not modified.
    bits : int
        Quantisation bit-width.

    Returns
    -------
    SNNClassifier
        Deep copy with all linear weights and biases quantised.
    """
    model_q = copy.deepcopy(model)
    model_q.eval()

    n_w = n_b = 0
    with torch.no_grad():
        for _, module in model_q.named_modules():
            if isinstance(module, nn.Linear):
                dq, _ = quantize_tensor_symmetric(module.weight.data, bits)
                module.weight.data.copy_(dq)
                n_w += 1
                if module.bias is not None:
                    dqb, _ = quantize_tensor_symmetric(module.bias.data, bits)
                    module.bias.data.copy_(dqb)
                    n_b += 1

    logger.info(
        "INT%d model: quantised %d weight tensors and %d bias tensors",
        bits, n_w, n_b,
    )
    return model_q


# ---------------------------------------------------------------------------
# Accuracy-loss summary
# ---------------------------------------------------------------------------

def quantization_report(
    fp32_acc: float,
    int8_acc: float,
    label: str = "test",
) -> None:
    """Log a one-line FP32 vs INT8 accuracy comparison.

    Parameters
    ----------
    fp32_acc : float
        Full-precision accuracy.
    int8_acc : float
        INT8-simulated accuracy.
    label : str
        Split name (``"val"`` or ``"test"``).
    """
    delta = (int8_acc - fp32_acc) * 100.0
    sign = "+" if delta >= 0 else ""
    logger.info(
        "Quantisation report (%s):  FP32 %.2f%%  INT8 %.2f%%  delta %s%.2f%%",
        label, fp32_acc * 100, int8_acc * 100, sign, delta,
    )
