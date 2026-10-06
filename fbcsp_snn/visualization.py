"""All plotting functions for the FBCSP-SNN pipeline.

Every function saves a figure to disk and calls ``plt.close(fig)``
immediately to prevent memory leaks.  All functions accept a ``save_path``
and return ``None``.

Functions
---------
plot_band_selection      Fisher ratio curve with selected-band highlights
plot_confusion_matrix    Normalised or raw confusion matrix heatmap
plot_spike_propagation   Feature × time spike-density heatmap
plot_neuron_traces       Membrane potential + spike overlay for output neurons
plot_weight_histograms   FP32 vs INT8-sim weight distributions per Linear layer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)

# Consistent palette across plots
_BAND_PALETTE = plt.cm.Set2.colors   # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------

def plot_band_selection(
    fisher_freqs: np.ndarray,
    fisher_curve: np.ndarray,
    selected_bands: List[Tuple[float, float]],
    save_path: Path,
    title: str = "Adaptive Band Selection — Fisher Discriminant Ratio",
) -> None:
    """Plot Fisher ratio vs. frequency with selected bands highlighted.

    Parameters
    ----------
    fisher_freqs : np.ndarray
        Frequency axis from Welch PSD, shape ``(n_freqs,)``.
    fisher_curve : np.ndarray
        Fisher discriminant ratio at each frequency bin, shape ``(n_freqs,)``.
    selected_bands : List[Tuple[float, float]]
        Ordered list of ``(lo, hi)`` band boundaries that were selected.
    save_path : Path
        Destination PNG file.  Parent directory is created if absent.
    title : str
        Figure title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4))

    # Fisher ratio curve (log-y to show dynamic range)
    ax.semilogy(fisher_freqs, fisher_curve + 1e-12, color="steelblue",
                lw=1.5, zorder=3, label="Fisher ratio")

    # Highlight selected bands
    for i, (lo, hi) in enumerate(selected_bands):
        colour = _BAND_PALETTE[i % len(_BAND_PALETTE)]
        ax.axvspan(lo, hi, alpha=0.22, color=colour, zorder=1)
        ax.axvline(lo, color=colour, lw=0.8, ls="--", alpha=0.6, zorder=2)
        ax.axvline(hi, color=colour, lw=0.8, ls="--", alpha=0.6, zorder=2)
        mid = (lo + hi) / 2.0
        ypos = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else fisher_curve.max() * 0.02
        ax.text(
            mid, fisher_curve.max() * 1.2,
            f"B{i+1}\n{lo:.0f}–{hi:.0f} Hz",
            ha="center", va="bottom", fontsize=7.5,
            color=colour, fontweight="bold",
        )

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Fisher ratio (log scale)", fontsize=11)
    ax.set_title(title, fontsize=12)
    # When static bands are used, fisher_freqs is a dummy [0.0] — derive
    # xlim from the band boundaries instead to avoid a singular transform.
    if len(fisher_freqs) > 1 and fisher_freqs[-1] > fisher_freqs[0]:
        ax.set_xlim(fisher_freqs[0], fisher_freqs[-1])
    elif selected_bands:
        ax.set_xlim(selected_bands[0][0] - 2, selected_bands[-1][1] + 2)
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Band selection plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

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
        Ground-truth and predicted class indices (0-indexed).
    class_names : List[str]
        Human-readable label for each class.
    save_path : Path
        Destination PNG.
    title : str
        Figure title.
    normalize : bool
        Row-normalise to show recall per class (default ``True``).
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    cm = sk_cm(y_true, y_pred, labels=list(range(n))).astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums > 0)

    fmt = ".2f" if normalize else "d"
    fig, ax = plt.subplots(figsize=(max(4, n), max(3.5, n)))
    sns.heatmap(
        cm, annot=True, fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
        vmin=0.0, vmax=1.0 if normalize else None,
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", save_path)


# ---------------------------------------------------------------------------
# Spike propagation
# ---------------------------------------------------------------------------

def plot_spike_propagation(
    spikes: torch.Tensor,
    save_path: Path,
    title: str = "Spike Propagation",
    n_trials: int = 4,
    n_features: int = 72,
    t_max: Optional[int] = None,
) -> None:
    """Plot feature × time spike-density heatmap for a few trials.

    Each panel shows one trial.  X-axis is timestep, Y-axis is feature index.
    Colour encodes whether a spike occurred (binary).

    Parameters
    ----------
    spikes : torch.Tensor
        Binary spike tensor ``(T, batch, n_features)``.
    save_path : Path
        Destination PNG.
    title : str
        Figure supertitle.
    n_trials : int
        Number of trials to display (first *n_trials* trials).
    n_features : int
        Number of feature channels to display.
    t_max : Optional[int]
        Truncate time axis to this many steps for readability.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    spk_np = spikes.cpu().numpy()   # (T, B, F)
    T, B, F = spk_np.shape

    n_trials  = min(n_trials, B)
    n_features = min(n_features, F)
    T_show    = min(t_max, T) if t_max else T

    fig, axes = plt.subplots(
        1, n_trials,
        figsize=(3.5 * n_trials, 3.5),
        sharey=True,
    )
    if n_trials == 1:
        axes = [axes]

    for col, ax in enumerate(axes):
        # (T_show, n_features) → transpose to (n_features, T_show) for imshow
        mat = spk_np[:T_show, col, :n_features].T
        ax.imshow(
            mat, aspect="auto", origin="lower",
            cmap="binary", vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax.set_xlabel("Timestep", fontsize=9)
        ax.set_title(f"Trial {col}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Feature index", fontsize=9)

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Spike propagation plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Neuron traces (membrane potential + spikes)
# ---------------------------------------------------------------------------

def plot_neuron_traces(
    spk_out: torch.Tensor,
    mem_out: torch.Tensor,
    save_path: Path,
    title: str = "Output LIF Neuron Traces",
    n_neurons: int = 6,
    trial_idx: int = 0,
    t_max: Optional[int] = None,
) -> None:
    """Plot membrane potential traces with spike marks for output neurons.

    Parameters
    ----------
    spk_out : torch.Tensor
        Output spike tensor ``(T, batch, n_output)``.
    mem_out : torch.Tensor
        Membrane potential tensor ``(T, batch, n_output)``.
    save_path : Path
        Destination PNG.
    title : str
        Figure title.
    n_neurons : int
        Number of output neurons to display.
    trial_idx : int
        Which trial (batch index) to visualise.
    t_max : Optional[int]
        Truncate display to this many timesteps.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    spk_np = spk_out.cpu().numpy()   # (T, B, n_out)
    mem_np = mem_out.cpu().numpy()   # (T, B, n_out)

    T, B, n_out = mem_np.shape
    n_neurons = min(n_neurons, n_out)
    T_show    = min(t_max, T) if t_max else T
    t         = np.arange(T_show)

    fig, axes = plt.subplots(
        n_neurons, 1,
        figsize=(12, 1.6 * n_neurons),
        sharex=True,
    )
    if n_neurons == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mem = mem_np[:T_show, trial_idx, i]
        spk = spk_np[:T_show, trial_idx, i]

        ax.plot(t, mem, color="steelblue", lw=0.8, label="mem")
        ax.axhline(1.0, color="grey", lw=0.5, ls="--", alpha=0.6)   # threshold

        spike_times = np.where(spk > 0.5)[0]
        if spike_times.size:
            ax.vlines(spike_times, ymin=ax.get_ylim()[0], ymax=1.05,
                      color="crimson", lw=0.8, alpha=0.8, label="spike")

        ax.set_ylabel(f"n{i}", fontsize=8, rotation=0, labelpad=20)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Timestep", fontsize=10)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Neuron traces plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Weight histograms
# ---------------------------------------------------------------------------

def plot_weight_histograms(
    model: torch.nn.Module,
    save_path: Path,
    quantized_model: Optional[torch.nn.Module] = None,
    title: str = "Weight Distributions",
    bins: int = 60,
) -> None:
    """Plot per-layer weight distributions, optionally overlaying INT8-sim.

    Parameters
    ----------
    model : torch.nn.Module
        FP32 model.
    save_path : Path
        Destination PNG.
    quantized_model : Optional[torch.nn.Module]
        INT8-simulated model.  If provided, its weights are overlaid in orange.
    title : str
        Figure title.
    bins : int
        Number of histogram bins.
    """
    import torch.nn as nn

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect (name, fp32_weights, [int8_weights]) for each Linear layer
    linear_layers: list[tuple] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w_fp32 = module.weight.data.cpu().numpy().ravel()
            w_int8 = None
            if quantized_model is not None:
                for qname, qmodule in quantized_model.named_modules():
                    if qname == name and isinstance(qmodule, nn.Linear):
                        w_int8 = qmodule.weight.data.cpu().numpy().ravel()
                        break
            linear_layers.append((name, w_fp32, w_int8))

    if not linear_layers:
        logger.warning("No Linear layers found in model; skipping histogram plot.")
        return

    n_layers = len(linear_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5.5 * n_layers, 4), squeeze=False)

    for col, (name, w_fp32, w_int8) in enumerate(linear_layers):
        ax = axes[0, col]
        ax.hist(w_fp32, bins=bins, alpha=0.72, color="steelblue",
                label=f"FP32  (std={w_fp32.std():.4f})")
        if w_int8 is not None:
            ax.hist(w_int8, bins=bins, alpha=0.55, color="darkorange",
                    label=f"INT8  (std={w_int8.std():.4f})")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Weight value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.4)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Weight histogram plot saved: %s", save_path)
