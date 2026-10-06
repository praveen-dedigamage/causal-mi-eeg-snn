"""Van Rossum loss with FFT-based exponential spike-train convolution.

Van Rossum distance
-------------------
The Van Rossum distance between two spike trains *s1* and *s2* is:

    D(s1, s2) = (1 / 2tau) * integral[ (f1(t) - f2(t))^2 dt ]

where *f* is the convolution of the spike train with a causal exponential
kernel:

    h[k] = alpha * (1 - alpha)^k,   alpha = dt / tau,   k >= 0

In discrete time with MSE normalisation this simplifies to:

    L = MSE(h * spk_out, h * spk_target)

FFT convolution
---------------
Sequential IIR filtering would be O(T) but is not parallelisable across the
batch/output dimensions.  Instead we use FFT convolution (O(T log T)) via
``torch.fft.rfft`` / ``irfft``, which is fully differentiable and runs in
one vectorised kernel call per split.

The output of the backward pass flows through ``irfft → rfft`` back to the
spike tensor.  snnTorch's surrogate gradient handles differentiation through
the discontinuous LIF firing function.

Target spike generation
-----------------------
:func:`make_target_spikes` samples Bernoulli spikes for the correct-class
population at *spike_prob* per timestep and zero elsewhere.

Cross-entropy ablation
-----------------------
:func:`cross_entropy_spike_loss` is an alternative to :func:`van_rossum_loss`
for ablation studies: standard softmax cross-entropy on population-summed
spike counts (the same reshape/sum readout used by
``SNNClassifier.decode``), instead of a spike-timing-sensitive distance.
Isolates whether the Van Rossum loss's timing sensitivity matters,
independent of the SNN architecture and population-coded readout.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Kernel and FFT helpers
# ---------------------------------------------------------------------------

def _build_exp_kernel(
    T: int,
    tau: float,
    dt: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a causal exponential kernel of length *T*.

    Parameters
    ----------
    T : int
        Kernel length (equal to the signal length so it covers the full window).
    tau : float
        Time constant in timesteps.
    dt : float
        Timestep size (typically 1.0 for discrete spike trains).
    device, dtype :
        Target device and floating-point type.

    Returns
    -------
    torch.Tensor
        Kernel ``h``, shape ``(T,)``.  Values are ``alpha * (1-alpha)^k``.
    """
    alpha = dt / tau
    k = torch.arange(T, device=device, dtype=dtype)
    return alpha * (1.0 - alpha) ** k


def _fft_convolve_causal(
    signal: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Causal linear convolution via FFT along the last dimension.

    Parameters
    ----------
    signal : torch.Tensor
        Shape ``(..., T)``.  Batch dimensions are supported.
    kernel : torch.Tensor
        Shape ``(K,)``.  Broadcast over all leading dimensions.

    Returns
    -------
    torch.Tensor
        Filtered signal, shape ``(..., T)``.  Only the first *T* output
        samples are returned, giving a causal result.
    """
    T = signal.shape[-1]
    K = kernel.shape[0]
    N_lin = T + K - 1                          # minimum linear convolution length
    N_fft = 1 << int(N_lin - 1).bit_length()  # next power-of-2 >= N_lin

    sig_f = torch.fft.rfft(signal, n=N_fft, dim=-1)   # (..., N_fft//2+1)
    ker_f = torch.fft.rfft(kernel, n=N_fft)            # (N_fft//2+1,)

    out = torch.fft.irfft(sig_f * ker_f, n=N_fft, dim=-1)  # (..., N_fft)
    return out[..., :T]                                      # (..., T) causal trim


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def van_rossum_loss(
    spk_out: torch.Tensor,
    spk_target: torch.Tensor,
    tau: float = 10.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Van Rossum loss between model output and target spike trains.

    Both spike trains are independently convolved with the same causal
    exponential kernel before computing MSE.  Because ``torch.fft`` operations
    are differentiable, gradients flow back through the convolution into
    *spk_out*.

    Parameters
    ----------
    spk_out : torch.Tensor
        Model output spikes, shape ``(T, batch, n_output)``.  Produced by the
        LIF layer; surrogate gradients make this differentiable.
    spk_target : torch.Tensor
        Target spike trains, shape ``(T, batch, n_output)``.  Should be
        detached (no gradient needed).
    tau : float
        Exponential kernel time constant in timesteps.  Larger values smooth
        the comparison over longer windows.
    dt : float
        Timestep size.  For raw discrete spike trains use 1.0.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    T = spk_out.shape[0]
    kernel = _build_exp_kernel(T, tau, dt, spk_out.device, spk_out.dtype)

    # Permute to (..., T) for the FFT convolution: (batch, n_output, T)
    out_t = spk_out.permute(1, 2, 0)
    tgt_t = spk_target.permute(1, 2, 0)

    filtered_out = _fft_convolve_causal(out_t, kernel)    # (batch, n_output, T)
    filtered_tgt = _fft_convolve_causal(tgt_t, kernel)    # (batch, n_output, T)

    return F.mse_loss(filtered_out, filtered_tgt)


# ---------------------------------------------------------------------------
# Target spike generation
# ---------------------------------------------------------------------------

def make_target_spikes(
    y: torch.Tensor,
    n_classes: int,
    population_per_class: int,
    T: int,
    spike_prob: float = 0.7,
) -> torch.Tensor:
    """Generate population-coded target spike trains.

    Neurons belonging to the correct-class population fire independently with
    probability *spike_prob* at each timestep.  All other output neurons are
    silent.

    Parameters
    ----------
    y : torch.Tensor
        Ground-truth class labels, shape ``(batch,)``.  **0-indexed.**
    n_classes : int
        Number of motor imagery classes.
    population_per_class : int
        Output neurons per class population.
    T : int
        Number of timesteps.
    spike_prob : float
        Per-timestep spike probability for correct-class neurons.

    Returns
    -------
    torch.Tensor
        Target spikes, shape ``(T, batch, n_classes * population_per_class)``,
        dtype ``torch.float32``, on the same device as *y*.
    """
    batch = y.shape[0]
    n_output = n_classes * population_per_class

    # pop_class_idx[o] = class that output neuron o belongs to
    pop_class_idx = torch.arange(n_classes, device=y.device).repeat_interleave(
        population_per_class
    )  # (n_output,)

    # mask[b, o] = 1 if neuron o belongs to the correct class for sample b
    mask = (pop_class_idx.unsqueeze(0) == y.unsqueeze(1)).float()  # (batch, n_output)

    # Sample Bernoulli independently at every timestep
    probs = (mask * spike_prob).unsqueeze(0).expand(T, -1, -1)  # (T, batch, n_output)
    return torch.bernoulli(probs)


# ---------------------------------------------------------------------------
# Cross-entropy ablation
# ---------------------------------------------------------------------------

def cross_entropy_spike_loss(
    spk_out: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    population_per_class: int,
) -> torch.Tensor:
    """Cross-entropy loss on population-summed spike counts (ablation).

    Alternative to :func:`van_rossum_loss` for ablation studies. Sums output
    spikes over time and within each class population (the same readout
    ``SNNClassifier.decode`` uses for prediction), then applies standard
    softmax cross-entropy against the true class label — no target spike
    trains are needed. Isolates whether the Van Rossum loss's spike-timing
    sensitivity contributes to the pipeline's performance, independent of
    the SNN architecture and population-coded readout, which are unchanged.

    Parameters
    ----------
    spk_out : torch.Tensor
        Model output spikes, shape ``(T, batch, n_output)`` where
        ``n_output = n_classes * population_per_class``. Differentiable via
        snnTorch's surrogate gradient.
    y : torch.Tensor
        Ground-truth class labels, shape ``(batch,)``, **0-indexed**, dtype
        ``torch.long``.
    n_classes : int
        Number of motor imagery classes.
    population_per_class : int
        Output neurons per class population.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss.
    """
    batch = spk_out.shape[1]
    spike_counts = spk_out.sum(dim=0)                                   # (batch, n_output)
    spike_counts = spike_counts.view(batch, n_classes, population_per_class)
    class_scores = spike_counts.sum(dim=-1)                             # (batch, n_classes)
    return F.cross_entropy(class_scores, y)
