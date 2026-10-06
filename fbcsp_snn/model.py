"""2-layer LIF Spiking Neural Network with population-coded output.

Architecture
------------
::

    Input (T, batch, n_input)
      for t in 0 .. T-1:
        cur1 = fc1(x[t])                       # Linear(n_input, n_hidden)
        cur1 = drop1(cur1)                     # Dropout
        spk1, mem1 = lif1(cur1, mem1)          # LIF
        cur2 = fc2(spk1)                       # Linear(n_hidden, n_output)
        cur2 = drop2(cur2)                     # Dropout
        spk2, mem2 = lif2(cur2, mem2)          # LIF
      stack spk2, mem2 → (T, batch, n_output)

Output neurons are arranged in *n_classes* populations of *population_per_class*
neurons each.  During inference, spikes are summed over time and over each
population; the class with the highest total vote wins (winner-take-all).

``torch.compile`` is guarded behind a Triton availability check: Triton is
Linux-only and CUDA Graphs conflict with snnTorch's ``init_leaky()``, so when
compiling the model use ``mode="default"`` only.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

from fbcsp_snn import DEVICE, setup_logger

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Fused LIF step (drop-in replacement for snntorch.Leaky)
# ---------------------------------------------------------------------------

class _FastSpike(torch.autograd.Function):
    """Heaviside spike with the fast-sigmoid surrogate gradient.

    Reproduces ``snntorch.surrogate.fast_sigmoid(slope=k)`` exactly:
    forward is ``1[U >= threshold]``; backward multiplies by
    ``(1 + k|U - threshold|)^-2``.
    """

    @staticmethod
    def forward(ctx, mem: torch.Tensor, threshold: float, slope: float):
        ctx.save_for_backward(mem)
        ctx.threshold = threshold
        ctx.slope = slope
        return (mem > threshold).to(mem.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + ctx.slope * (mem - ctx.threshold).abs()) ** 2
        return grad_output * sg, None, None


def fast_lif_step(
    cur: torch.Tensor,
    mem: torch.Tensor,
    spk: torch.Tensor,
    beta: float,
    threshold: float = 1.0,
    slope: float = 25.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One reset-by-subtraction LIF step, numerically identical to ``snn.Leaky``.

    ``snntorch.Leaky`` costs roughly 3.5x this implementation per call because
    of its per-call Python-side state handling; since the LIF dominates the
    timestep loop (measured: 1.94 s of a 3.92 s forward pass at T=2561, versus
    0.17 s for both Linear layers), replacing it speeds the whole network's
    forward+backward by ~1.9x.

    The reset term **must** be detached.  ``snntorch``'s ``mem_reset`` returns
    ``spike_grad(mem - threshold).clone().detach()``, so gradients do not flow
    through the reset.  Letting them flow instead changes ``fc1``/``fc2``
    gradients by >50% and trains to a different solution, while leaving the
    forward pass bit-identical -- i.e. the error is invisible unless gradients
    are compared directly.

    Parameters
    ----------
    cur : torch.Tensor
        Input current for this timestep, shape ``(batch, n_neurons)``.
    mem : torch.Tensor
        Membrane potential carried from the previous timestep.
    spk : torch.Tensor
        Spikes emitted at the previous timestep (drives the reset).
    beta : float
        Membrane decay, clamped to ``[0, 1]`` as ``snn.Leaky`` does.
    threshold : float
        Firing threshold.
    slope : float
        Fast-sigmoid surrogate slope ``k``.

    Returns
    -------
    spk : torch.Tensor
        Spikes emitted this timestep.
    mem : torch.Tensor
        Updated membrane potential.
    """
    mem = min(max(beta, 0.0), 1.0) * mem + cur - spk.detach() * threshold
    return _FastSpike.apply(mem, threshold, slope), mem


class SNNClassifier(nn.Module):
    """Two-layer LIF SNN with population-coded output.

    Parameters
    ----------
    n_input : int
        Number of input features (= total CSP features after optional MIBIF
        selection).
    n_hidden : int
        Hidden layer width.
    n_classes : int
        Number of motor imagery classes.
    population_per_class : int
        Output neurons allocated per class.
    beta : float
        LIF membrane potential decay factor (shared across both layers).
    dropout_prob : float
        Dropout probability applied after each linear layer.

    Attributes
    ----------
    n_output : int
        Total output neurons (= ``n_classes * population_per_class``).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 64,
        n_classes: int = 4,
        population_per_class: int = 20,
        beta: float = 0.95,
        dropout_prob: float = 0.5,
        fast_lif: bool = False,
    ) -> None:
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.population_per_class = population_per_class
        self.n_output = n_classes * population_per_class

        # fast_lif swaps snntorch.Leaky for the fused step in this module.
        # OFF by default: it is ~1.3x faster and verified equivalent to float32
        # epsilon (forward, loss and fc2 gradients bit-identical; fc1 gradients
        # agree to 1.4e-07 relative), but "equivalent to 1.4e-07" is not the
        # same as "unchanged", and every published result so far was produced
        # with snnTorch. The far larger speedup (2.5x, from running a subject's
        # folds concurrently on one GPU) costs no change to the computation at
        # all, so this is not a trade worth making by default.
        # Enable deliberately, and only for runs that are self-consistent.
        self.fast_lif = fast_lif
        self.beta_val = beta

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Layer 1: Linear → Dropout → LIF
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.drop1 = nn.Dropout(p=dropout_prob)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Layer 2: Linear → Dropout → LIF
        self.fc2 = nn.Linear(n_hidden, self.n_output)
        self.drop2 = nn.Dropout(p=dropout_prob)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SNNClassifier — input: %d  hidden: %d  classes: %d  "
            "pop/class: %d  output: %d  trainable params: %d",
            n_input, n_hidden, n_classes, population_per_class,
            self.n_output, param_count,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Run SNN simulation over all timesteps.

        Parameters
        ----------
        x : torch.Tensor
            Input spike tensor, shape ``(T, batch, n_input)``.
        return_hidden : bool
            If ``True``, also return the hidden-layer spike train (needed
            for event/spike counting — B15). Default ``False`` preserves the
            original 2-tuple return so every existing caller (training loop,
            ``evaluate_model``, tests) is unaffected.

        Returns
        -------
        spk_out : torch.Tensor
            Output spike trains, shape ``(T, batch, n_output)``.
        mem_out : torch.Tensor
            Output membrane potential traces, shape ``(T, batch, n_output)``.
        spk_hidden : torch.Tensor
            Only returned when ``return_hidden=True``. Hidden-layer spike
            trains, shape ``(T, batch, n_hidden)``.
        """
        T = x.shape[0]
        # snn.Leaky's init_leaky() returns a size-0 placeholder that the layer
        # lazily broadcasts on its first call; the fused path has no such
        # machinery, so shape the state explicitly. It also carries the
        # previous spikes itself (snn.Leaky keeps them internally). Zeros
        # reproduce snnTorch's first-step behaviour exactly.
        if self.fast_lif:
            B = x.shape[1]
            mem1 = torch.zeros(B, self.n_hidden, device=x.device, dtype=x.dtype)
            mem2 = torch.zeros(B, self.n_output, device=x.device, dtype=x.dtype)
            spk1_p = torch.zeros(B, self.n_hidden, device=x.device, dtype=x.dtype)
            spk2_p = torch.zeros(B, self.n_output, device=x.device, dtype=x.dtype)
        else:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk1_p = spk2_p = None

        spk_out_list: list[torch.Tensor] = []
        mem_out_list: list[torch.Tensor] = []
        spk_hidden_list: list[torch.Tensor] = []

        for t in range(T):
            # Layer 1
            cur1 = self.drop1(self.fc1(x[t]))
            if self.fast_lif:
                spk1, mem1 = fast_lif_step(cur1, mem1, spk1_p, self.beta_val)
                spk1_p = spk1
            else:
                spk1, mem1 = self.lif1(cur1, mem1)
            # Layer 2
            cur2 = self.drop2(self.fc2(spk1))
            if self.fast_lif:
                spk2, mem2 = fast_lif_step(cur2, mem2, spk2_p, self.beta_val)
                spk2_p = spk2
            else:
                spk2, mem2 = self.lif2(cur2, mem2)

            spk_out_list.append(spk2)
            mem_out_list.append(mem2)
            if return_hidden:
                spk_hidden_list.append(spk1)

        spk_out = torch.stack(spk_out_list, dim=0)   # (T, batch, n_output)
        mem_out = torch.stack(mem_out_list, dim=0)   # (T, batch, n_output)
        if return_hidden:
            spk_hidden = torch.stack(spk_hidden_list, dim=0)   # (T, batch, n_hidden)
            return spk_out, mem_out, spk_hidden
        return spk_out, mem_out

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, spk_out: torch.Tensor) -> torch.Tensor:
        """Winner-take-all decoding from output spike trains.

        1. Sum spikes over time → ``(batch, n_output)``
        2. Reshape to ``(batch, n_classes, population_per_class)``
        3. Sum over population → ``(batch, n_classes)``
        4. Argmax → ``(batch,)`` predicted class (0-indexed)

        Parameters
        ----------
        spk_out : torch.Tensor
            Shape ``(T, batch, n_output)``.

        Returns
        -------
        torch.Tensor
            Predicted class indices, shape ``(batch,)``, dtype ``torch.long``.
        """
        spike_sums = spk_out.sum(dim=0)                              # (batch, n_output)
        spike_sums = spike_sums.view(-1, self.n_classes,
                                     self.population_per_class)     # (batch, C, pop)
        class_votes = spike_sums.sum(dim=-1)                        # (batch, C)
        return class_votes.argmax(dim=-1)                           # (batch,)


class ANNClassifier(nn.Module):
    """Non-spiking twin of :class:`SNNClassifier` -- identical architecture
    and identical leaky-integration recurrence, but the LIF neuron's
    spike-and-reset nonlinearity is replaced by a continuous ReLU applied
    to the same leaky-integrated membrane potential (no threshold, no
    reset, no binary output).

    Every other parameter is unchanged from ``SNNClassifier``: same
    ``n_input``/``n_hidden``/``n_output`` sizes, same dropout, same
    ``beta`` decay applied to the membrane recurrence, and the exact same
    ``forward``/``decode`` signatures -- so it drops directly into
    :func:`fbcsp_snn.training.train_fold` and :func:`evaluate_model`
    unchanged, with either ``loss_type='van_rossum'`` (Van Rossum distance
    is just an exponentially-filtered MSE, so it works unmodified on
    continuous output) or ``loss_type='cross_entropy'``. This isolates
    whether the spiking mechanism itself contributes anything, holding
    architecture, optimizer, training protocol, and input features fixed.

    Parameters
    ----------
    n_input, n_hidden, n_classes, population_per_class, beta, dropout_prob
        Identical meaning to :class:`SNNClassifier`.

    Attributes
    ----------
    n_output : int
        Total output neurons (= ``n_classes * population_per_class``).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 64,
        n_classes: int = 4,
        population_per_class: int = 20,
        beta: float = 0.95,
        dropout_prob: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.population_per_class = population_per_class
        self.n_output = n_classes * population_per_class
        self.beta = beta

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.drop1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(n_hidden, self.n_output)
        self.drop2 = nn.Dropout(p=dropout_prob)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ANNClassifier (non-spiking twin) — input: %d  hidden: %d  "
            "classes: %d  pop/class: %d  output: %d  trainable params: %d",
            n_input, n_hidden, n_classes, population_per_class,
            self.n_output, param_count,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Run the leaky-integrator simulation over all timesteps.

        Mirrors :meth:`SNNClassifier.forward` exactly (same signature,
        same return shapes) so every existing caller works unchanged --
        ``act_out``/``act_hidden`` stand in for ``spk_out``/``spk_hidden``
        but are continuous-valued rather than binary.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``(T, batch, n_input)`` -- the same
            spike-encoded sequence ``SNNClassifier`` receives.
        return_hidden : bool
            If ``True``, also return the hidden-layer activation trace.

        Returns
        -------
        act_out : torch.Tensor
            Output activation trace, shape ``(T, batch, n_output)``.
        mem_out : torch.Tensor
            Identical to ``act_out`` (kept only so the 2-tuple return
            shape matches ``SNNClassifier.forward`` exactly).
        act_hidden : torch.Tensor
            Only returned when ``return_hidden=True``. Hidden-layer
            activation trace, shape ``(T, batch, n_hidden)``.
        """
        T, batch, _ = x.shape
        mem1 = torch.zeros(batch, self.n_hidden, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros(batch, self.n_output, device=x.device, dtype=x.dtype)

        act_out_list: list[torch.Tensor] = []
        act_hidden_list: list[torch.Tensor] = []

        for t in range(T):
            cur1 = self.drop1(self.fc1(x[t]))
            mem1 = self.beta * mem1 + cur1
            act1 = torch.relu(mem1)

            cur2 = self.drop2(self.fc2(act1))
            mem2 = self.beta * mem2 + cur2
            act2 = torch.relu(mem2)

            act_out_list.append(act2)
            if return_hidden:
                act_hidden_list.append(act1)

        act_out = torch.stack(act_out_list, dim=0)   # (T, batch, n_output)
        if return_hidden:
            act_hidden = torch.stack(act_hidden_list, dim=0)   # (T, batch, n_hidden)
            return act_out, act_out, act_hidden
        return act_out, act_out

    def decode(self, act_out: torch.Tensor) -> torch.Tensor:
        """Winner-take-all decoding -- identical logic to
        :meth:`SNNClassifier.decode`, just summing continuous activations
        instead of spike counts.
        """
        activation_sums = act_out.sum(dim=0)
        activation_sums = activation_sums.view(
            -1, self.n_classes, self.population_per_class
        )
        class_votes = activation_sums.sum(dim=-1)
        return class_votes.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Optional torch.compile guard (Linux / Triton only)
# ---------------------------------------------------------------------------

def maybe_compile(model: nn.Module) -> nn.Module:
    """Wrap *model* in ``torch.compile`` only when Triton is available.

    Triton is Linux-only; CUDA Graphs conflict with snnTorch's
    ``init_leaky()``, so ``mode="default"`` is used (not ``"reduce-overhead"``).

    Parameters
    ----------
    model : nn.Module
        Model to (optionally) compile.

    Returns
    -------
    nn.Module
        Compiled model or the original if Triton is unavailable.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        logger.info("torch.compile skipped (Triton not available)")
        return model

    # Triton PTX codegen is unreliable on Volta (V100, sm_70) with
    # PyTorch <= 2.1. Only compile on Ampere+ (sm_80+).
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.info(
                "torch.compile skipped (GPU is sm_%d0, Ampere sm_80+ required "
                "for stable Triton PTX codegen)",
                major,
            )
            return model

    compiled = torch.compile(model, mode="default")
    logger.info("torch.compile applied (Triton available, sm_80+)")
    return compiled
