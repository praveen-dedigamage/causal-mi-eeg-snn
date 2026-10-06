"""Training loop with AdamW, AMP, early stopping, and best-model checkpointing.

Design
------
- Mini-batch SGD over trials; trials are shuffled at every epoch.
- AMP (``torch.amp.autocast`` + ``GradScaler``) is enabled on CUDA and
  disabled on CPU (autocast over CPU uses bfloat16 but has no throughput
  benefit on non-Apple hardware).
- Early stopping monitors *validation accuracy* with a configurable patience
  and warmup period so the model has time to escape random-init noise before
  stopping kicks in.
- The best model state (by val accuracy) is kept in memory and optionally
  saved to ``fold_dir/best_model.pt``.  At the end of training the model is
  restored to that best state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
try:
    from torch.amp import GradScaler, autocast   # torch >= 2.2
except ImportError:
    from torch.cuda.amp import GradScaler        # torch 2.1.x
    from torch import autocast                   # type: ignore[assignment]

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.losses import (
    cross_entropy_spike_loss,
    make_target_spikes,
    van_rossum_loss,
)
from fbcsp_snn.model import SNNClassifier

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Metrics produced by one training fold.

    Attributes
    ----------
    best_val_acc : float
        Best validation accuracy seen during training.
    best_epoch : int
        0-indexed epoch at which best val accuracy was achieved.
    stopped_epoch : int
        0-indexed epoch at which training stopped (may equal ``epochs - 1``
        if early stopping never triggered).
    train_loss_history : List[float]
        Mean training loss per epoch.
    val_acc_history : List[float]
        Validation accuracy per epoch.
    """

    best_val_acc: float = 0.0
    best_epoch: int = 0
    stopped_epoch: int = 0
    train_loss_history: List[float] = field(default_factory=list)
    val_acc_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor validation accuracy and signal when training should stop.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before stopping.
    warmup : int
        Minimum number of epochs before early stopping can trigger.

    Attributes
    ----------
    best_acc : float
        Best validation accuracy seen so far.
    best_epoch : int
        Epoch (0-indexed) at which best accuracy occurred.
    counter : int
        Epochs since last improvement.
    """

    def __init__(self, patience: int = 100, warmup: int = 100) -> None:
        self.patience = patience
        self.warmup = warmup
        self.best_acc: float = -1.0
        self.best_epoch: int = 0
        self.counter: int = 0

    def update(self, val_acc: float, epoch: int) -> Tuple[bool, bool]:
        """Update state with the latest validation accuracy.

        Parameters
        ----------
        val_acc : float
            Validation accuracy for the current epoch.
        epoch : int
            Current epoch index (0-indexed).

        Returns
        -------
        improved : bool
            ``True`` if *val_acc* is a new best.
        should_stop : bool
            ``True`` if training should be terminated.
        """
        improved = val_acc > self.best_acc
        if improved:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        should_stop = (epoch >= self.warmup) and (self.counter >= self.patience)
        return improved, should_stop


# ---------------------------------------------------------------------------
# Evaluation helper (no gradient)
# ---------------------------------------------------------------------------

def evaluate_model(
    model: SNNClassifier,
    spikes: torch.Tensor,
    y_0idx: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, np.ndarray]:
    """Run inference and compute accuracy.

    Parameters
    ----------
    model : SNNClassifier
        Model to evaluate.  Must already be on *device*.
    spikes : torch.Tensor
        Spike tensor ``(T, n_trials, n_features)``.  Can be on any device;
        batches are moved to *device* inside the loop.
    y_0idx : np.ndarray
        True labels, shape ``(n_trials,)``.  **0-indexed.**
    device : torch.device
        Inference device.
    batch_size : int
        Number of trials per inference batch.

    Returns
    -------
    accuracy : float
        Fraction of correctly classified trials.
    predictions : np.ndarray
        Predicted class indices (0-indexed), shape ``(n_trials,)``.
    """
    model.eval()
    n_trials = spikes.shape[1]
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            batch = spikes[:, start : start + batch_size, :].to(device)
            spk_out, _ = model(batch)
            preds = model.decode(spk_out)
            all_preds.append(preds.cpu().numpy())

    predictions = np.concatenate(all_preds)
    accuracy = float((predictions == y_0idx).mean())
    return accuracy, predictions


def evaluate_model_with_events(
    model: SNNClassifier,
    spikes: torch.Tensor,
    y_0idx: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, np.ndarray, float]:
    """Run inference, compute accuracy, and count total spike events per trial.

    Extends :func:`evaluate_model` with end-to-end spike/event counting —
    input-encoder spikes (the *spikes* argument itself) + hidden-layer
    spikes (exposed via ``model(..., return_hidden=True)``) + output-layer
    spikes, summed over all timesteps and neurons, averaged per trial (B15).

    This is the real, measured firing-rate number that should back any
    energy estimate. Previously no such measurement existed anywhere in the
    codebase — the paper's energy equation substituted the input encoder's
    firing rate for the whole network's activity because the hidden layer's
    spikes were never exposed by ``SNNClassifier.forward()`` (fixed
    alongside this function — see ``return_hidden`` in ``model.py``).

    Parameters
    ----------
    model : SNNClassifier
        Model to evaluate.  Must already be on *device*.
    spikes : torch.Tensor
        Input spike tensor ``(T, n_trials, n_features)``.
    y_0idx : np.ndarray
        True labels, shape ``(n_trials,)``.  **0-indexed.**
    device : torch.device
        Inference device.
    batch_size : int
        Number of trials per inference batch.

    Returns
    -------
    accuracy : float
        Fraction of correctly classified trials.
    predictions : np.ndarray
        Predicted class indices (0-indexed), shape ``(n_trials,)``.
    mean_events_per_trial : float
        Average total spike count (input + hidden + output layers, summed
        over all timesteps and neurons) per single-trial classification.
    """
    model.eval()
    n_trials = spikes.shape[1]
    all_preds: list[np.ndarray] = []
    total_events = 0.0

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            batch = spikes[:, start : start + batch_size, :].to(device)
            spk_out, _, spk_hidden = model(batch, return_hidden=True)
            preds = model.decode(spk_out)
            all_preds.append(preds.cpu().numpy())

            total_events += (
                batch.sum().item()
                + spk_hidden.sum().item()
                + spk_out.sum().item()
            )

    predictions = np.concatenate(all_preds)
    accuracy = float((predictions == y_0idx).mean())
    mean_events_per_trial = total_events / n_trials
    return accuracy, predictions, mean_events_per_trial


def evaluate_model_with_event_breakdown(
    model: SNNClassifier,
    spikes: torch.Tensor,
    y_0idx: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Like :func:`evaluate_model_with_events`, but reports events *per layer*
    instead of a single combined total.

    A single combined total treats every spike as equally costly, but real
    energy accumulates per-*synapse*: an input spike drives ``n_hidden``
    downstream synapses via ``fc1``, a hidden spike drives ``n_output``
    synapses via ``fc2``. Computing energy properly needs the input and
    hidden counts separately so each can be weighted by its own layer's
    fan-out — that's what this function is for (see
    ``paper_energy.py``). Kept as a separate function rather than changing
    :func:`evaluate_model_with_events`'s return signature, since that
    function already has call sites elsewhere (``reliability.py``) relying
    on its 3-tuple shape.

    Parameters
    ----------
    model : SNNClassifier
        Model to evaluate.  Must already be on *device*.
    spikes : torch.Tensor
        Input spike tensor ``(T, n_trials, n_features)``.
    y_0idx : np.ndarray
        True labels, shape ``(n_trials,)``.  **0-indexed.**
    device : torch.device
        Inference device.
    batch_size : int
        Number of trials per inference batch.

    Returns
    -------
    accuracy : float
        Fraction of correctly classified trials.
    predictions : np.ndarray
        Predicted class indices (0-indexed), shape ``(n_trials,)``.
    events : Dict[str, float]
        Mean events per trial, keyed ``"input"``, ``"hidden"``, ``"output"``,
        and ``"total"`` (sum of the three — matches
        :func:`evaluate_model_with_events`'s combined value exactly).
    """
    model.eval()
    n_trials = spikes.shape[1]
    all_preds: list[np.ndarray] = []
    total_input = 0.0
    total_hidden = 0.0
    total_output = 0.0

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            batch = spikes[:, start : start + batch_size, :].to(device)
            spk_out, _, spk_hidden = model(batch, return_hidden=True)
            preds = model.decode(spk_out)
            all_preds.append(preds.cpu().numpy())

            total_input += batch.sum().item()
            total_hidden += spk_hidden.sum().item()
            total_output += spk_out.sum().item()

    predictions = np.concatenate(all_preds)
    accuracy = float((predictions == y_0idx).mean())
    events = {
        "input":  total_input / n_trials,
        "hidden": total_hidden / n_trials,
        "output": total_output / n_trials,
        "total":  (total_input + total_hidden + total_output) / n_trials,
    }
    return accuracy, predictions, events


# ---------------------------------------------------------------------------
# Full fold training
# ---------------------------------------------------------------------------

def train_fold(
    spikes_train: torch.Tensor,
    y_train: np.ndarray,
    spikes_val: torch.Tensor,
    y_val: np.ndarray,
    model: SNNClassifier,
    n_classes: int,
    population_per_class: int,
    *,
    spike_prob: float = 0.7,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    epochs: int = 1000,
    patience: int = 100,
    warmup: int = 100,
    tau_vr: float = 10.0,
    loss_type: str = "van_rossum",
    batch_size: int = 32,
    max_time_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    fold_dir: Optional[Path] = None,
    log_every: int = 10,
    use_amp: bool = False,
) -> FoldResult:
    """Train an SNN on one CV fold and return performance metrics.

    Parameters
    ----------
    spikes_train : torch.Tensor
        Training spike tensor ``(T, n_train, n_features)``.
    y_train : np.ndarray
        Training labels, shape ``(n_train,)``.  **0-indexed.**
    spikes_val : torch.Tensor
        Validation spike tensor ``(T, n_val, n_features)``.
    y_val : np.ndarray
        Validation labels, shape ``(n_val,)``.  **0-indexed.**
    model : SNNClassifier
        Freshly initialised model (will be moved to *device* if not already).
    n_classes : int
        Number of MI classes.
    population_per_class : int
        Output population size per class.
    spike_prob : float
        Target spiking probability for the correct class in Van Rossum targets.
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight-decay coefficient.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience (epochs without val-acc improvement).
    warmup : int
        Minimum epochs before early stopping activates.
    tau_vr : float
        Van Rossum loss time constant (timesteps). Unused when
        ``loss_type='cross_entropy'``.
    loss_type : str
        ``'van_rossum'`` (default) or ``'cross_entropy'`` (ablation: softmax
        cross-entropy on population-summed spike counts, no target spike
        trains needed — see :func:`fbcsp_snn.losses.cross_entropy_spike_loss`).
    batch_size : int
        Mini-batch size (number of trials per gradient step).
    max_time_steps : Optional[int]
        If given, truncate spikes to the first *max_time_steps* timesteps.
        Useful during development to reduce per-epoch wall time.
    device : Optional[torch.device]
        Training device.  Defaults to :data:`fbcsp_snn.DEVICE`.
    fold_dir : Optional[Path]
        Directory to save ``best_model.pt``.  Created if absent.
    log_every : int
        Log metrics every this many epochs.

    Returns
    -------
    FoldResult
        Metrics for the fold.  The model is restored to its best-val-accuracy
        state before returning.
    """
    if device is None:
        device = DEVICE

    # Optional time truncation (development / test speedup only)
    if max_time_steps is not None:
        spikes_train = spikes_train[:max_time_steps]
        spikes_val   = spikes_val[:max_time_steps]

    T = spikes_train.shape[0]
    n_train = spikes_train.shape[1]

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Mixed precision is OFF by default. AMP keeps FP32 master weights, so
    # the saved checkpoint is FP32 either way -- but it computes the
    # forward/backward in float16, which means an "FP32 baseline" would
    # have been produced by 16-bit arithmetic. For a study whose subject
    # IS numerical precision, that is a distinction worth not having to
    # defend. It also costs little here: the workload is launch-overhead
    # bound (GPU measured at 17-21 % utilisation), and AMP accelerates
    # compute, not kernel launches.
    use_amp = bool(use_amp) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger.info("AMP %s (checkpoint weights are FP32 regardless)",
                "ENABLED - forward/backward in float16" if use_amp
                else "disabled - full FP32 arithmetic")

    es = EarlyStopping(patience=patience, warmup=warmup)
    best_state: dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    y_tensor = torch.from_numpy(y_train)   # stays on CPU; moved per batch

    train_loss_history: list[float] = []
    val_acc_history: list[float] = []
    stopped_epoch = epochs - 1

    for epoch in range(epochs):
        # ---- training ----
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            batch_spikes = spikes_train[:, idx, :].to(device)   # (T, B, F)
            batch_y      = y_tensor[idx].long().to(device)       # (B,)

            if loss_type != "cross_entropy":
                target = make_target_spikes(
                    batch_y, n_classes, population_per_class, T, spike_prob
                ).to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                spk_out, _ = model(batch_spikes)
                if loss_type == "cross_entropy":
                    loss = cross_entropy_spike_loss(
                        spk_out, batch_y, n_classes, population_per_class
                    )
                else:
                    loss = van_rossum_loss(spk_out, target, tau=tau_vr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        train_loss_history.append(mean_loss)

        # ---- validation ----
        val_acc, _ = evaluate_model(model, spikes_val, y_val, device, batch_size)
        val_acc_history.append(val_acc)

        improved, should_stop = es.update(val_acc, epoch)

        if improved:
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            if fold_dir is not None:
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, fold_dir / "best_model.pt")

        if (epoch + 1) % log_every == 0 or should_stop:
            logger.info(
                "  epoch %3d/%d  loss %.5f  val_acc %.3f  best %.3f",
                epoch + 1, epochs, mean_loss, val_acc, es.best_acc,
            )

        if should_stop:
            logger.info(
                "  Early stop at epoch %d  (best epoch %d  val_acc %.3f)",
                epoch + 1, es.best_epoch + 1, es.best_acc,
            )
            stopped_epoch = epoch
            break

    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return FoldResult(
        best_val_acc=es.best_acc,
        best_epoch=es.best_epoch,
        stopped_epoch=stopped_epoch,
        train_loss_history=train_loss_history,
        val_acc_history=val_acc_history,
    )
