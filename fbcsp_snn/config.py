"""Config dataclass and argparse CLI for the FBCSP-SNN pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    """All pipeline hyperparameters and runtime options.

    Parameters
    ----------
    mode : str
        Pipeline mode: ``"train"``, ``"infer"``, or ``"aggregate"``.
    source : str
        Data source: ``"moabb"`` or ``"hdf5"``.
    moabb_dataset : str
        MOABB dataset name (e.g. ``"BNCI2014_001"``).
    subject_id : int
        Subject index (1-indexed).
    data_path : Optional[str]
        Path to HDF5 .mat file (only used when ``source="hdf5"``).
    fold : Optional[int]
        Single fold to run (``None`` → run all folds).
    n_folds : int
        Number of CV folds.
    freq_bands : List[Tuple[float, float]]
        Six overlapping frequency bands for the filter bank.
    bandwidth : float
        Candidate band width in Hz.
    band_step : float
        Candidate band step in Hz.
    band_range : Tuple[float, float]
        Frequency range for candidate bands.
    csp_components_per_band : int
        Total CSP filters per band (half from each end when
        ``csp_dual_end=True``; all from one end when ``False``).
    lambda_r : float
        CSP covariance regularisation.
    base_thresh : float
        Spike encoding base threshold.
    adapt_inc : float
        Spike encoding adaptive increment.
    decay : float
        Spike encoding threshold decay.
    hidden_neurons : int
        Hidden layer size of the SNN.
    population_per_class : int
        Output population neurons per class.
    beta : float
        LIF membrane potential decay factor.
    dropout_prob : float
        Dropout probability.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    epochs : int
        Maximum training epochs.
    early_stopping_patience : int
        Early stopping patience (epochs).
    early_stopping_warmup : int
        Minimum epochs before early stopping kicks in.
    spiking_prob : float
        Target class spike probability for Van Rossum loss.
    feature_selection_method : str
        Feature selection method (``"mibif"`` or ``"none"``).
    feature_percentile : float
        Percentile of features to keep.
    results_dir : str
        Root directory for output artifacts.
    n_classes : Optional[int]
        Number of classes (auto-detected from dataset if ``None``).
    """

    # Runtime
    mode: str = "train"
    source: str = "moabb"
    moabb_dataset: str = "BNCI2014_001"
    subject_id: int = 1
    data_path: Optional[str] = None
    fold: Optional[int] = None
    n_folds: int = 10
    val_fraction: float = 0.2
    seed: int = 42
    use_amp: bool = False

    # Band selection — fixed 6-band overlapping filter bank
    freq_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(4, 8), (8, 14), (12, 18), (16, 24), (20, 30), (26, 40)]
    )
    filter_type: str = "butterworth"   # 'butterworth' or 'bessel'

    # CSP
    csp_components_per_band: int = 8
    lambda_r: float = 0.0001
    euclidean_alignment: bool = True
    riemannian_mean: bool = True
    csp_ledoit_wolf: bool = False
    csp_dual_end: bool = True   # ablation: False = single-end (standard) CSP

    # Encoding
    encoder_type: str = "delta"   # 'delta', 'adm', or 'fixed' (ablation)
    base_thresh: float = 0.001
    adapt_inc: float = 0.6
    decay: float = 0.95

    # Model
    hidden_neurons: int = 64
    population_per_class: int = 20
    beta: float = 0.95
    dropout_prob: float = 0.5

    # Training
    lr: float = 1e-3
    weight_decay: float = 0.1
    epochs: int = 1000
    early_stopping_patience: int = 100
    early_stopping_warmup: int = 100
    spiking_prob: float = 0.7
    loss_type: str = "van_rossum"   # 'van_rossum' or 'cross_entropy' (ablation)
    tau_vr: float = 10.0
    train_batch_size: int = 64

    # Feature selection
    feature_selection_method: str = "mibif"
    feature_percentile: float = 50.0
    mi_fraction: Optional[float] = None   # adaptive mode: keep MI >= mi_fraction*max_MI

    # I/O
    results_dir: str = "Results"
    n_classes: Optional[int] = None

    # Reliability-only: Monte Carlo noise-robustness sweep (B15)
    reliability_severities: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3]
    )
    reliability_n_repeats: int = 20


# ---------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------

def _parse_freq_bands(value: str) -> List[Tuple[float, float]]:
    """Parse a string like ``"[(4,10),(10,14),(14,30)]"`` into a list of tuples."""
    import ast
    parsed = ast.literal_eval(value)
    return [(float(lo), float(hi)) for lo, hi in parsed]


def _parse_band_range(value: str) -> Tuple[float, float]:
    """Parse a string like ``"(4.0,30.0)"`` into a ``(lo, hi)`` tuple."""
    import ast
    lo, hi = ast.literal_eval(value)
    return (float(lo), float(hi))


def _parse_severities(value: str) -> List[float]:
    """Parse a string like ``"[0.0,0.05,0.1,0.2,0.3]"`` into a list of floats."""
    import ast
    parsed = ast.literal_eval(value)
    return [float(v) for v in parsed]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FBCSP-SNN Motor Imagery EEG Classifier",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- shared arguments ----
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--source", choices=["moabb", "hdf5"], default="moabb")
    shared.add_argument("--moabb-dataset", default="BNCI2014_001")
    shared.add_argument("--subject-id", type=int, default=1)
    shared.add_argument("--data-path", default=None)
    shared.add_argument("--results-dir", default="Results")
    shared.add_argument("--n-classes", type=int, default=None)
    shared.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for reproducible runs.")
    shared.add_argument("--amp", dest="use_amp", action="store_true",
                        help="Enable mixed-precision training (float16 compute). "
                             "Off by default so the FP32 baseline is computed in "
                             "full FP32, which matters for the quantisation study.")

    # ---- train ----
    train_p = sub.add_parser("train", parents=[shared], help="Train the pipeline.")
    train_p.add_argument("--n-folds", type=int, default=10)
    train_p.add_argument("--val-fraction", type=float, default=0.2,
                         help="Validation fraction per fold (default 0.2 = 80/20 split).")
    train_p.add_argument("--fold", type=int, default=None,
                         help="Run only this fold (0-indexed).")
    train_p.add_argument("--freq-bands", type=_parse_freq_bands,
                         default=[(4, 8), (8, 14), (12, 18), (16, 24), (20, 30), (26, 40)],
                         help="Six overlapping frequency bands in Hz, "
                              "e.g. '[(4,8),(8,14),(12,18),(16,24),(20,30),(26,40)]'")
    train_p.add_argument("--filter-type", type=str, default="butterworth",
                         choices=["butterworth", "bessel"],
                         help="Causal bandpass filter type (default: butterworth).")
    train_p.add_argument("--csp-components-per-band", type=int, default=4)
    train_p.add_argument("--lambda-r", type=float, default=0.0001)
    train_p.add_argument("--euclidean-alignment", action="store_true", default=True)
    train_p.add_argument("--no-euclidean-alignment", dest="euclidean_alignment",
                         action="store_false")
    train_p.add_argument("--riemannian-mean", action="store_true", default=True)
    train_p.add_argument("--no-riemannian-mean", dest="riemannian_mean",
                         action="store_false")
    train_p.add_argument("--csp-ledoit-wolf", dest="csp_ledoit_wolf",
                         action="store_true", default=False,
                         help="Use Ledoit-Wolf shrinkage for CSP covariance "
                              "estimation instead of fixed Tikhonov regularisation.")
    train_p.add_argument("--csp-dual-end", dest="csp_dual_end",
                         action="store_true", default=True,
                         help="Take m eigenvectors from both ends of the CSP "
                              "eigenspectrum (default, 2m filters/band/pair).")
    train_p.add_argument("--csp-single-end", dest="csp_dual_end",
                         action="store_false",
                         help="Ablation: take m eigenvectors from the "
                              "largest-eigenvalue end only (standard "
                              "single-end CSP, m filters/band/pair).")
    train_p.add_argument("--encoder-type", type=str, default="delta",
                         choices=["delta", "adm", "fixed"],
                         help="Spike encoder: 'delta' (adaptive threshold, default), "
                              "'adm' (ON/OFF polarity; doubles feature dimension), or "
                              "'fixed' (ablation: constant threshold, no adaptation).")
    train_p.add_argument("--base-thresh", type=float, default=0.001)
    train_p.add_argument("--adapt-inc", type=float, default=0.6)
    train_p.add_argument("--decay", type=float, default=0.95)
    train_p.add_argument("--hidden-neurons", type=int, default=64)
    train_p.add_argument("--population-per-class", type=int, default=20)
    train_p.add_argument("--beta", type=float, default=0.95)
    train_p.add_argument("--dropout-prob", type=float, default=0.5)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=0.1)
    train_p.add_argument("--epochs", type=int, default=1000)
    train_p.add_argument("--early-stopping-patience", type=int, default=100)
    train_p.add_argument("--early-stopping-warmup", type=int, default=100)
    train_p.add_argument("--spiking-prob", type=float, default=0.7)
    train_p.add_argument("--loss-type", type=str, default="van_rossum",
                         choices=["van_rossum", "cross_entropy"],
                         help="Training loss: 'van_rossum' (default, spike-train "
                              "MSE) or 'cross_entropy' (ablation: softmax "
                              "cross-entropy on population spike counts).")
    train_p.add_argument("--tau-vr", type=float, default=10.0,
                         help="Van Rossum kernel time constant in timesteps "
                              "(default 10.0). Unused when --loss-type=cross_entropy.")
    train_p.add_argument("--train-batch-size", type=int, default=64,
                         help="Mini-batch size for SNN training (default 64).")
    train_p.add_argument("--feature-selection-method",
                         choices=["mibif", "none"], default="mibif")
    train_p.add_argument("--feature-percentile", type=float, default=50.0)
    train_p.add_argument("--mi-fraction", type=float, default=None,
                         help="Adaptive MIBIF threshold: keep features with "
                              "MI >= mi_fraction * max_MI. When set, overrides "
                              "--feature-percentile. Try 0.05-0.3.")

    # ---- infer ----
    infer_p = sub.add_parser("infer", parents=[shared], help="Run inference.")
    infer_p.add_argument("--fold", type=int, required=True)
    infer_p.add_argument("--n-folds", type=int, default=10)

    # ---- reliability ----
    rel_p = sub.add_parser(
        "reliability", parents=[shared],
        help="Run Monte Carlo hardware-noise reliability sweep on a saved fold (B15).",
    )
    rel_p.add_argument("--fold", type=int, required=True)
    rel_p.add_argument("--reliability-severities", type=_parse_severities,
                       default=[0.0, 0.05, 0.1, 0.2, 0.3],
                       help="Noise severities to sweep, as sigma_frac values, "
                            "e.g. '[0.0,0.05,0.1,0.2,0.3]'.")
    rel_p.add_argument("--reliability-n-repeats", type=int, default=20,
                       help="Monte Carlo repeats per severity level (default 20).")

    # ---- aggregate ----
    agg_p = sub.add_parser("aggregate", parents=[shared],
                            help="Aggregate fold results.")
    agg_p.add_argument("--n-folds", type=int, default=10)

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Convert parsed ``argparse.Namespace`` to a :class:`Config` instance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    Config
        Populated config object.
    """
    kwargs: dict = {}
    mapping = {
        "mode": "mode",
        "source": "source",
        "moabb_dataset": "moabb_dataset",
        "subject_id": "subject_id",
        "data_path": "data_path",
        "results_dir": "results_dir",
        "n_classes": "n_classes",
    }
    for attr, field_name in mapping.items():
        if hasattr(args, attr):
            kwargs[field_name] = getattr(args, attr)

    # mode-specific fields (all optional — Config has defaults)
    optional_fields = [
        "n_folds", "fold", "val_fraction", "freq_bands", "filter_type",
        "csp_components_per_band", "lambda_r", "euclidean_alignment", "riemannian_mean", "csp_ledoit_wolf",
        "csp_dual_end",
        "encoder_type", "base_thresh", "adapt_inc", "decay",
        "hidden_neurons", "population_per_class", "beta", "dropout_prob",
        "lr", "weight_decay", "epochs", "early_stopping_patience",
        "early_stopping_warmup", "spiking_prob", "loss_type", "tau_vr",
        "train_batch_size",
        "feature_selection_method", "feature_percentile", "mi_fraction",
        "reliability_severities", "reliability_n_repeats",
        "seed", "use_amp",
    ]
    for f in optional_fields:
        if hasattr(args, f):
            kwargs[f] = getattr(args, f)

    return Config(**kwargs)
