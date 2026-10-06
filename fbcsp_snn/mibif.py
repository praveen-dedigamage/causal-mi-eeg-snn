"""Mutual Information Best Individual Feature (MIBIF) selection.

Spike counts (sum over T) are used as the feature representation for MI
estimation.  Raw per-timestep spikes are too noisy; collapsing to total
spike count per trial per feature gives a stable, scalar representation
that is well-suited to ``mutual_info_classif``.

The selector is fitted exclusively on training-split data so that no
information about val/test labels leaks into feature selection.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)


class MIBIFSelector:
    """Select features by mutual information with class labels.

    Two selection modes (mutually exclusive — ``mi_fraction`` takes priority):

    **Adaptive mode** (``mi_fraction`` is set):
        Keep every feature whose MI score exceeds ``mi_fraction × max_MI_score``.
        The number of selected features adapts automatically per fold/subject
        based on the shape of the MI score distribution.  Analogous to
        ``min_fisher_fraction`` in band selection.

    **Percentile mode** (``mi_fraction`` is None, default):
        Keep the top ``feature_percentile``% of features by MI score.
        Fixed feature count across all folds/subjects.

    Parameters
    ----------
    feature_percentile : float
        Percentage of features to keep in percentile mode (e.g. ``50.0``).
        Ignored when ``mi_fraction`` is set.
    mi_fraction : float or None
        Adaptive threshold as a fraction of the maximum MI score.
        Features with ``MI_score >= mi_fraction * max(MI_scores)`` are kept.
        ``None`` disables adaptive mode (uses ``feature_percentile`` instead).
    random_state : int
        Seed passed to ``mutual_info_classif`` for reproducibility.

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Sorted indices of the selected features, shape ``(n_selected,)``.
    mi_scores_ : np.ndarray
        Raw MI score for every feature, shape ``(n_features,)``.
    n_features_in_ : int
        Total number of input features before selection.
    """

    def __init__(
        self,
        feature_percentile: float = 50.0,
        mi_fraction: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.feature_percentile = feature_percentile
        self.mi_fraction = mi_fraction
        self.random_state = random_state
        self.selected_indices_: np.ndarray = np.array([], dtype=np.int64)
        self.mi_scores_: np.ndarray = np.array([])
        self.n_features_in_: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, spikes: torch.Tensor, y: np.ndarray) -> "MIBIFSelector":
        """Compute MI scores and select top features from training spikes.

        Parameters
        ----------
        spikes : torch.Tensor
            Binary spike tensor, shape ``(T, n_trials, n_features)``.
            Must be on any device; internally moved to CPU for sklearn.
        y : np.ndarray
            Class labels, shape ``(n_trials,)``.  **0-indexed.**

        Returns
        -------
        MIBIFSelector
            Self (for chaining).
        """
        # Spike counts: sum over T → (n_trials, n_features)
        spike_counts = spikes.sum(dim=0).cpu().numpy().astype(np.float64)
        self.n_features_in_ = spike_counts.shape[1]

        logger.info(
            "MIBIF: computing MI for %d features × %d trials ...",
            self.n_features_in_, len(y),
        )
        self.mi_scores_ = mutual_info_classif(
            spike_counts, y, discrete_features=False,
            random_state=self.random_state,
        )

        max_score = self.mi_scores_.max()

        if self.mi_fraction is not None:
            # Adaptive mode: threshold = mi_fraction × max MI score
            threshold = self.mi_fraction * max_score
            mode_str = f"adaptive mi_fraction={self.mi_fraction:.3f}"
        else:
            # Percentile mode: fixed top-K%
            threshold = np.percentile(self.mi_scores_, 100.0 - self.feature_percentile)
            mode_str = f"percentile={self.feature_percentile:.1f}%"

        self.selected_indices_ = np.sort(
            np.where(self.mi_scores_ >= threshold)[0]
        ).astype(np.int64)

        kept_pct = 100.0 * len(self.selected_indices_) / self.n_features_in_
        logger.info(
            "MIBIF [%s]: %d → %d features kept (%.1f%%)  "
            "score threshold=%.5f  range=[%.5f, %.5f]",
            mode_str,
            self.n_features_in_,
            len(self.selected_indices_),
            kept_pct,
            threshold,
            self.mi_scores_.min(),
            max_score,
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, spikes: torch.Tensor) -> torch.Tensor:
        """Select features from a spike tensor.

        Parameters
        ----------
        spikes : torch.Tensor
            Shape ``(T, n_trials, n_features_in)``.

        Returns
        -------
        torch.Tensor
            Shape ``(T, n_trials, n_selected)``, same device as input.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.selected_indices_.size == 0:
            raise RuntimeError("Call fit() before transform().")
        idx = torch.from_numpy(self.selected_indices_).to(spikes.device)
        return spikes[:, :, idx]

    def fit_transform(
        self,
        spikes: torch.Tensor,
        y: np.ndarray,
    ) -> torch.Tensor:
        """Fit and immediately transform the same spike tensor.

        Parameters
        ----------
        spikes : torch.Tensor
            Shape ``(T, n_trials, n_features)``.
        y : np.ndarray
            0-indexed class labels.

        Returns
        -------
        torch.Tensor
            Shape ``(T, n_trials, n_selected)``.
        """
        return self.fit(spikes, y).transform(spikes)
