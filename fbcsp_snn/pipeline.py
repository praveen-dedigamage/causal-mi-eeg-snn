"""Top-level pipeline orchestrators: run_train, run_infer, run_aggregate.

Each function consumes a :class:`~fbcsp_snn.config.Config` object and
runs the corresponding pipeline phase end-to-end, persisting all artifacts
under ``Results/Subject_<N>/fold_<K>/``.

Artifact layout per fold
------------------------
::

    Results/Subject_N/
      fold_K/
        best_model.pt           — PyTorch state dict (FP32 best val)
        csp_filters.pkl         — PairwiseCSP instance (pickle)
        znorm.pkl               — ZNormaliser instance (pickle)
        mibif.pkl               — MIBIFSelector instance (pickle)
        pipeline_params.json    — bands, metrics, hyperparams (per fold)
        spike_propagation.png   — spike raster for 4 training trials
        neuron_traces.png       — output LIF membrane + spike overlay
        weight_histograms.png   — FP32 weight distributions
        confusion_fp32.png      — normalised confusion matrix (FP32)
      summary.csv               — per-fold metrics (written by aggregate)
      confusion_aggregate.png   — summed confusion matrix over all folds

Digital post-training quantisation (INT8 whole-model, CSP-bit PTQ, and the
joint CSP+SNN sweep) was retired 2026-07-11 in favour of the hardware-
realism reliability sweep (``reliability.py``), which models analog noise
directly rather than static digital bit-rounding.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from fbcsp_snn import DEVICE, set_global_seed, setup_logger
from fbcsp_snn.baseline import extract_logvar, run_baseline_classifiers
from fbcsp_snn.config import Config
from fbcsp_snn.datasets import DATASET_REGISTRY, get_n_classes, load_moabb
from fbcsp_snn.data import load_hdf5
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.evaluation import compute_accuracy, compute_confusion_matrix
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import SNNClassifier, maybe_compile
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank
from fbcsp_snn.training import (
    evaluate_model,
    evaluate_model_with_event_breakdown,
    train_fold,
)
from fbcsp_snn.visualization import (
    plot_confusion_matrix,
    plot_neuron_traces,
    plot_spike_propagation,
    plot_weight_histograms,
)

logger: logging.Logger = setup_logger(__name__)

# Default class names used when not specified; overridden by dataset registry.
_BNCI2014_CLASSES = ["feet", "left_hand", "right_hand", "tongue"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_names(cfg: Config, n_classes: int) -> list[str]:
    """Return human-readable class labels."""
    if cfg.moabb_dataset == "BNCI2014_001":
        return _BNCI2014_CLASSES[:n_classes]
    return [f"class_{i}" for i in range(n_classes)]


def _load_raw(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw EEG and return (X_train, y_train, X_test, y_test)."""
    if cfg.source == "moabb":
        return load_moabb(cfg.moabb_dataset, cfg.subject_id, cfg.n_classes)
    if cfg.source == "hdf5":
        if cfg.data_path is None:
            logger.error("--data-path required for --source hdf5")
            sys.exit(1)
        X, y = load_hdf5(cfg.data_path)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(sss.split(X, y))
        return X[tr], y[tr], X[te], y[te]
    logger.error("Unknown source: %s", cfg.source)
    sys.exit(1)


def _sfreq(cfg: Config) -> float:
    """Return sampling frequency for the configured dataset."""
    if cfg.source == "moabb" and cfg.moabb_dataset in DATASET_REGISTRY:
        return float(DATASET_REGISTRY[cfg.moabb_dataset]["sfreq"])
    return 250.0


def _spikes_from_concat(
    X_concat: np.ndarray,
    cfg: Config,
) -> torch.Tensor:
    """Encode a concatenated CSP projection array into spikes.

    Parameters
    ----------
    X_concat : np.ndarray
        Shape ``(n_trials, n_features, n_samples)``, float32.
    cfg : Config
        Pipeline config (encoding hyperparameters).

    Returns
    -------
    torch.Tensor
        Binary spikes ``(T, n_trials, n_features)``.
    """
    t = torch.from_numpy(X_concat).to(DEVICE).permute(2, 0, 1)  # (T, B, F)
    return encode_tensor(t, cfg.base_thresh, cfg.adapt_inc, cfg.decay, cfg.encoder_type)


def _concat_projections(proj: dict) -> np.ndarray:
    """Concatenate CSP projections from all pairs along the feature axis."""
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def _run_single_fold(
    fold_idx: int,
    X_f_tr: np.ndarray,
    y_f_tr: np.ndarray,
    X_f_val: np.ndarray,
    y_f_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
    sfreq: float,
    n_classes: int,
    fold_dir: Path,
) -> dict:
    """Run one CV fold end-to-end and return a metrics dict.

    All preprocessing is fit on the training split only.

    Parameters
    ----------
    fold_idx : int
        0-indexed fold number.
    X_f_tr, y_f_tr : np.ndarray
        Training EEG and labels (1-indexed).
    X_f_val, y_f_val : np.ndarray
        Validation EEG and labels (1-indexed).
    X_test, y_test : np.ndarray
        Held-out test EEG and labels (1-indexed).
    cfg : Config
        Full pipeline config.
    sfreq : float
        Sampling frequency.
    n_classes : int
        Number of MI classes.
    fold_dir : Path
        Directory to save all fold artifacts.

    Returns
    -------
    dict
        Per-fold metrics including val/test FP32 and INT8 accuracies.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    y_f_tr_0  = y_f_tr  - 1
    y_f_val_0 = y_f_val - 1
    y_test_0  = y_test  - 1

    m = cfg.csp_components_per_band // 2   # filters per end

    # ---- Frequency bands ----
    bands = cfg.freq_bands
    logger.info("Fold %d  bands: %s", fold_idx, bands)

    # ---- Filter bank ----
    X_bands_tr  = apply_filter_bank(X_f_tr,  bands, sfreq, order=4, filter_type=cfg.filter_type)
    X_bands_val = apply_filter_bank(X_f_val, bands, sfreq, order=4, filter_type=cfg.filter_type)
    X_bands_te  = apply_filter_bank(X_test,  bands, sfreq, order=4, filter_type=cfg.filter_type)

    # ---- Pre-CSP crop (dataset-specific, e.g. PhysionetMI imagery window) ----
    # Filtering on the full epoch avoids edge effects; cropping afterwards gives
    # CSP clean signal from only the imagery period (no fixation contamination).
    from fbcsp_snn.datasets import DATASET_REGISTRY
    crop_s = DATASET_REGISTRY.get(cfg.moabb_dataset, {}).get("crop_s", None)
    if crop_s is not None:
        c0 = int(crop_s[0] * sfreq)
        c1 = int(crop_s[1] * sfreq) + 1   # +1 so tmax second is included
        X_bands_tr  = [X[:, :, c0:c1] for X in X_bands_tr]
        X_bands_val = [X[:, :, c0:c1] for X in X_bands_val]
        X_bands_te  = [X[:, :, c0:c1] for X in X_bands_te]
        logger.info(
            "Pre-CSP crop %.1f–%.1f s → %d samples/trial  (sfreq=%.0f Hz)",
            crop_s[0], crop_s[1], c1 - c0, sfreq,
        )

    X_bands_csp, y_csp = X_bands_tr, y_f_tr

    # ---- Pairwise CSP ----
    csp = PairwiseCSP(m=m, lambda_r=cfg.lambda_r,
                      euclidean_alignment=cfg.euclidean_alignment,
                      riemannian_mean=cfg.riemannian_mean,
                      ledoit_wolf=cfg.csp_ledoit_wolf,
                      dual_end=cfg.csp_dual_end)
    csp.fit(X_bands_csp, y_csp)

    proj_tr  = csp.transform(X_bands_tr)
    proj_val = csp.transform(X_bands_val)
    proj_te  = csp.transform(X_bands_te)

    X_concat_tr  = _concat_projections(proj_tr)
    X_concat_val = _concat_projections(proj_val)
    X_concat_te  = _concat_projections(proj_te)

    # ---- Z-normalisation ----
    znorm = ZNormaliser()
    X_norm_tr  = znorm.fit_transform(X_concat_tr)
    X_norm_val = znorm.transform(X_concat_val)
    X_norm_te  = znorm.transform(X_concat_te)

    # ---- Classical baseline (log-var + LDA / SVM) --------------------------
    # Runs on the same z-normalised features as the SNN, before spike encoding.
    # Results are stored in pipeline_params.json for direct comparison.
    logger.info("Fold %d  running classical baselines (LDA, SVM) …", fold_idx)
    bl_feat_tr  = extract_logvar(X_norm_tr)
    bl_feat_val = extract_logvar(X_norm_val)
    bl_feat_te  = extract_logvar(X_norm_te)
    baseline_results = run_baseline_classifiers(
        bl_feat_tr,  y_f_tr_0,
        bl_feat_val, y_f_val_0,
        bl_feat_te,  y_test_0,
    )

    # ---- Spike encoding ----
    spikes_tr  = _spikes_from_concat(X_norm_tr,  cfg)
    spikes_val = _spikes_from_concat(X_norm_val, cfg)
    spikes_te  = _spikes_from_concat(X_norm_te,  cfg)

    # ---- MIBIF ----
    mibif: Optional[MIBIFSelector] = None
    if cfg.feature_selection_method == "mibif":
        mibif = MIBIFSelector(
            feature_percentile=cfg.feature_percentile,
            mi_fraction=cfg.mi_fraction,
            random_state=42,
        )
        spikes_tr  = mibif.fit_transform(spikes_tr,  y_f_tr_0)
        spikes_val = mibif.transform(spikes_val)
        spikes_te  = mibif.transform(spikes_te)

    n_input = spikes_tr.shape[2]
    n_timesteps = spikes_tr.shape[0]
    logger.info(
        "Fold %d  n_input_features: %d  n_timesteps: %d",
        fold_idx, n_input, n_timesteps,
    )

    # ---- Spike propagation plot (a few training trials) ----
    plot_spike_propagation(
        spikes_tr,
        save_path=fold_dir / "spike_propagation.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} — Spike Propagation",
        n_trials=4, n_features=min(n_input, 72), t_max=200,
    )

    # ---- Train ----
    model = maybe_compile(SNNClassifier(
        n_input=n_input,
        n_hidden=cfg.hidden_neurons,
        n_classes=n_classes,
        population_per_class=cfg.population_per_class,
        beta=cfg.beta,
        dropout_prob=cfg.dropout_prob,
    ).to(DEVICE))

    result = train_fold(
        spikes_train=spikes_tr,
        y_train=y_f_tr_0,
        spikes_val=spikes_val,
        y_val=y_f_val_0,
        model=model,
        n_classes=n_classes,
        population_per_class=cfg.population_per_class,
        spike_prob=cfg.spiking_prob,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        patience=cfg.early_stopping_patience,
        warmup=cfg.early_stopping_warmup,
        tau_vr=cfg.tau_vr,
        loss_type=cfg.loss_type,
        batch_size=cfg.train_batch_size,
        device=DEVICE,
        fold_dir=fold_dir,
        log_every=max(1, cfg.epochs // 20),
        use_amp=cfg.use_amp,
    )

    # ---- FP32 evaluate ----
    # Test-set evaluation also counts spike events per layer (input/hidden/
    # output) per trial (B15) — the measured firing rates that should back
    # any energy estimate, replacing the previous unmeasured proxy. Per-layer
    # (not just combined) so paper_energy.py can weight each layer's
    # spikes by its own fan-out rather than treating every spike as equally
    # costly.
    val_acc_fp32,  val_preds_fp32  = evaluate_model(model, spikes_val, y_f_val_0,  DEVICE)
    test_acc_fp32, test_preds_fp32, event_counts = evaluate_model_with_event_breakdown(
        model, spikes_te, y_test_0, DEVICE
    )
    mean_events_per_trial = event_counts["total"]

    # ---- Neuron traces (one test-set batch) ----
    model.eval()
    with torch.no_grad():
        spk_out_vis, mem_out_vis = model(spikes_te[:, :8, :].to(DEVICE))
    plot_neuron_traces(
        spk_out_vis, mem_out_vis,
        save_path=fold_dir / "neuron_traces.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} — Output Neuron Traces",
        n_neurons=min(6, n_classes * cfg.population_per_class),
        t_max=200,
    )

    # ---- Weight histograms ----
    plot_weight_histograms(
        model,
        save_path=fold_dir / "weight_histograms.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} — Weight Distributions",
    )

    # ---- Confusion matrix ----
    class_names = _class_names(cfg, n_classes)
    plot_confusion_matrix(
        y_test_0, test_preds_fp32, class_names,
        save_path=fold_dir / "confusion_fp32.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} FP32 — Test",
    )

    # ---- Persist preprocessing objects ----
    with open(fold_dir / "csp_filters.pkl", "wb") as f:
        pickle.dump(csp, f, protocol=4)
    with open(fold_dir / "znorm.pkl", "wb") as f:
        pickle.dump(znorm, f, protocol=4)
    if mibif is not None:
        with open(fold_dir / "mibif.pkl", "wb") as f:
            pickle.dump(mibif, f, protocol=4)

    # ---- pipeline_params.json ----
    params = {
        "subject_id":         cfg.subject_id,
        "seed":               cfg.seed,
        "use_amp":            cfg.use_amp,
        "fold":               fold_idx,
        "dataset":            cfg.moabb_dataset,
        "n_classes":          n_classes,
        "n_input_features":   n_input,
        "n_timesteps":        n_timesteps,
        "bands":              [[float(lo), float(hi)] for lo, hi in bands],
        "filter_type":        cfg.filter_type,
        "encoder_type":       cfg.encoder_type,
        "euclidean_alignment": cfg.euclidean_alignment,
        "riemannian_mean":    cfg.riemannian_mean,
        "csp_dual_end":       cfg.csp_dual_end,
        "csp_ledoit_wolf":    cfg.csp_ledoit_wolf,
        "csp_m":              m,
        "lambda_r":           cfg.lambda_r,
        "hidden_neurons":     cfg.hidden_neurons,
        "population_per_class": cfg.population_per_class,
        "beta":               cfg.beta,
        "loss_type":          cfg.loss_type,
        "tau_vr":             cfg.tau_vr,
        "train_batch_size":   cfg.train_batch_size,
        "feature_method":     cfg.feature_selection_method,
        "feature_percentile": cfg.feature_percentile,
        "mi_fraction":        cfg.mi_fraction,
        "n_features_selected": n_input * 2 if mibif is None else len(mibif.selected_indices_),
        "best_val_acc_fp32":  round(result.best_val_acc, 6),
        "best_epoch":         result.best_epoch,
        "stopped_epoch":      result.stopped_epoch,
        "val_acc_fp32":       round(val_acc_fp32, 6),
        "test_acc_fp32":      round(test_acc_fp32, 6),
        # Measured end-to-end spike events per trial (B15) — replaces the
        # unmeasured "0.15 spikes/neuron/timestep" proxy previously used in
        # the paper's energy estimate (that number was actually the input
        # encoder's rate, not a whole-network measurement). Broken down per
        # layer (not just the combined total) so paper_energy.py can
        # weight each layer's spikes by its own fan-out.
        "mean_events_per_trial": round(mean_events_per_trial, 3),
        "mean_input_events_per_trial":  round(event_counts["input"], 3),
        "mean_hidden_events_per_trial": round(event_counts["hidden"], 3),
        "mean_output_events_per_trial": round(event_counts["output"], 3),
        # Classical baselines (log-var features, same z-norm, no spike encoding)
        "val_acc_lda":        round(baseline_results["val_acc_lda"],  6),
        "test_acc_lda":       round(baseline_results["test_acc_lda"], 6),
        "val_acc_svm":        round(baseline_results["val_acc_svm"],  6),
        "test_acc_svm":       round(baseline_results["test_acc_svm"], 6),
        "svm_best_c":         baseline_results["svm_best_c"],
        "svm_best_gamma":     baseline_results["svm_best_gamma"],
    }
    with open(fold_dir / "pipeline_params.json", "w") as f:
        json.dump(params, f, indent=2)

    logger.info(
        "Fold %d saved to %s  (FP32 %.1f%%)",
        fold_idx, fold_dir, test_acc_fp32 * 100,
    )
    return params


# ---------------------------------------------------------------------------
# run_train
# ---------------------------------------------------------------------------

def run_train(cfg: Config) -> None:
    """Run the full training pipeline for one subject.

    Executes n_folds CV folds (or a single specified fold).  For each fold:
    band selection → filter bank → CSP → z-norm → spike encoding → MIBIF →
    SNN training → FP32 + INT8 evaluation → artifact save.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration.
    """
    set_global_seed(cfg.seed)
    t_start = time.perf_counter()

    # Auto-detect n_classes
    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    logger.info(
        "run_train  subject=%d  dataset=%s  n_classes=%d  n_folds=%d",
        cfg.subject_id, cfg.moabb_dataset, n_classes, cfg.n_folds,
    )

    X_train, y_train, X_test, y_test = _load_raw(cfg)
    sfreq = _sfreq(cfg)

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Determine which folds to run
    # Split session-1 (X_train) only into train/val folds.
    # X_test (session 2) is held out entirely and never touched by the splitter.
    # Default val_fraction=0.2 → StratifiedKFold (80/20, ~58 val trials).
    # Custom val_fraction (e.g. 0.3) → StratifiedShuffleSplit for flexibility.
    if cfg.val_fraction == round(1.0 / cfg.n_folds, 10):
        splitter = StratifiedKFold(
            n_splits=cfg.n_folds, shuffle=True, random_state=42
        )
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=cfg.n_folds, test_size=cfg.val_fraction, random_state=42
        )

    fold_metrics: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        splitter.split(X_train, y_train)
    ):
        if cfg.fold is not None and fold_idx != cfg.fold:
            continue

        logger.info("")
        logger.info("=" * 60)
        logger.info("  Fold %d / %d", fold_idx, cfg.n_folds - 1)
        logger.info("=" * 60)

        fold_dir = subject_dir / f"fold_{fold_idx}"

        metrics = _run_single_fold(
            fold_idx=fold_idx,
            X_f_tr=X_train[tr_idx],
            y_f_tr=y_train[tr_idx],
            X_f_val=X_train[val_idx],
            y_f_val=y_train[val_idx],
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            sfreq=sfreq,
            n_classes=n_classes,
            fold_dir=fold_dir,
        )
        fold_metrics.append(metrics)

    elapsed = time.perf_counter() - t_start
    logger.info("")
    logger.info(
        "run_train complete — %d folds  wall time %.1f s",
        len(fold_metrics), elapsed,
    )

    if fold_metrics:
        def _fmean(key: str) -> float:
            vals = [m[key] for m in fold_metrics if key in m]
            return float(np.mean(vals)) if vals else float("nan")

        logger.info(
            "Subject %d  mean test  FP32 %.1f%%",
            cfg.subject_id,
            _fmean("test_acc_fp32") * 100,
        )

    # Auto-aggregate when all folds were run in one shot (no --fold flag).
    # Skipped in SLURM array mode where each fold is a separate job.
    if cfg.fold is None and len(fold_metrics) == cfg.n_folds:
        logger.info("All folds complete — running aggregate …")
        run_aggregate(cfg)


# ---------------------------------------------------------------------------
# run_infer
# ---------------------------------------------------------------------------

def run_infer(cfg: Config) -> None:
    """Run inference for a single fold using saved artifacts.

    Loads the saved model, CSP, znorm, and MIBIF from ``fold_<K>/``, applies
    the full preprocessing chain to the test set, and logs the test accuracy.

    Parameters
    ----------
    cfg : Config
        Must have ``fold`` set.
    """
    if cfg.fold is None:
        logger.error("--fold is required for infer mode")
        sys.exit(1)

    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    fold_dir = subject_dir / f"fold_{cfg.fold}"

    params_path = fold_dir / "pipeline_params.json"
    if not params_path.exists():
        logger.error("pipeline_params.json not found at %s — run train first", fold_dir)
        sys.exit(1)

    with open(params_path) as f:
        params = json.load(f)

    bands   = [tuple(b) for b in params["bands"]]
    n_input = params["n_input_features"]
    m       = params["csp_m"]
    sfreq   = _sfreq(cfg)

    logger.info(
        "run_infer  subject=%d  fold=%d  bands=%s  n_input=%d",
        cfg.subject_id, cfg.fold, bands, n_input,
    )

    _, _, X_test, y_test = _load_raw(cfg)
    y_test_0 = y_test - 1

    # Load preprocessing objects
    with open(fold_dir / "csp_filters.pkl", "rb") as f:
        csp: PairwiseCSP = pickle.load(f)
    with open(fold_dir / "znorm.pkl", "rb") as f:
        znorm: ZNormaliser = pickle.load(f)

    mibif: Optional[MIBIFSelector] = None
    mibif_path = fold_dir / "mibif.pkl"
    if mibif_path.exists():
        with open(mibif_path, "rb") as f:
            mibif = pickle.load(f)

    # Preprocessing chain — restore training-time settings from saved params
    filter_type      = params.get("filter_type",  "butterworth")
    cfg.encoder_type = params.get("encoder_type", cfg.encoder_type)
    X_bands = apply_filter_bank(X_test, bands, sfreq, order=4, filter_type=filter_type)
    proj = csp.transform(X_bands)
    X_concat = _concat_projections(proj)
    X_norm = znorm.transform(X_concat)
    spikes = _spikes_from_concat(X_norm, cfg)
    if mibif is not None:
        spikes = mibif.transform(spikes)

    # Load model
    model = SNNClassifier(
        n_input=n_input,
        n_hidden=params.get("hidden_neurons", cfg.hidden_neurons),
        n_classes=n_classes,
        population_per_class=params.get("population_per_class", cfg.population_per_class),
        beta=params.get("beta", cfg.beta),
        dropout_prob=0.0,   # no dropout at inference
    ).to(DEVICE)
    state = torch.load(fold_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)

    # FP32 inference
    test_acc_fp32, test_preds_fp32 = evaluate_model(model, spikes, y_test_0, DEVICE)

    # Confusion matrix
    class_names = _class_names(cfg, n_classes)
    plot_confusion_matrix(
        y_test_0, test_preds_fp32, class_names,
        save_path=fold_dir / "infer_confusion_fp32.png",
        title=f"Subject {cfg.subject_id} Fold {cfg.fold} FP32 — Inference",
    )

    logger.info(
        "Fold %d inference — FP32 %.1f%%",
        cfg.fold, test_acc_fp32 * 100,
    )


# ---------------------------------------------------------------------------
# run_aggregate
# ---------------------------------------------------------------------------

def run_aggregate(cfg: Config) -> None:
    """Collect per-fold JSON artifacts and produce summary CSV + plots.

    Reads ``fold_K/pipeline_params.json`` for K in ``0 … n_folds-1``,
    writes ``summary.csv``, and saves an aggregated FP32 confusion matrix
    (sum over all folds).

    Parameters
    ----------
    cfg : Config
        Must have ``subject_id``, ``n_folds``, ``results_dir`` set.
    """
    import csv

    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    logger.info("run_aggregate  subject=%d  n_folds=%d", cfg.subject_id, cfg.n_folds)

    rows: list[dict] = []
    missing: list[int] = []

    for fold_idx in range(cfg.n_folds):
        p = subject_dir / f"fold_{fold_idx}" / "pipeline_params.json"
        if not p.exists():
            logger.warning("Missing fold %d  (%s)", fold_idx, p)
            missing.append(fold_idx)
            continue
        with open(p) as f:
            rows.append(json.load(f))

    if not rows:
        logger.error("No fold results found under %s", subject_dir)
        sys.exit(1)

    if missing:
        logger.warning("Aggregating %d / %d folds (missing: %s)",
                       len(rows), cfg.n_folds, missing)

    # ---- Summary CSV ----
    csv_path = subject_dir / "summary.csv"
    fieldnames = [
        "fold", "best_val_acc_fp32", "best_epoch", "stopped_epoch",
        "val_acc_fp32",
        "test_acc_fp32", "mean_events_per_trial",
        "mean_input_events_per_trial", "mean_hidden_events_per_trial",
        "mean_output_events_per_trial", "n_timesteps",
        "val_acc_lda", "test_acc_lda", "val_acc_svm", "test_acc_svm",
        "svm_best_c", "svm_best_gamma",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Append mean row
    def _col(key: str) -> list[float]:
        return [r[key] for r in rows if key in r]

    mean_row = {
        "fold":                "mean",
        "best_val_acc_fp32":   round(float(np.mean(_col("best_val_acc_fp32"))), 6),
        "best_epoch":          "",
        "stopped_epoch":       "",
        "val_acc_fp32":        round(float(np.mean(_col("val_acc_fp32"))),  6),
        "test_acc_fp32":       round(float(np.mean(_col("test_acc_fp32"))), 6),
        "mean_events_per_trial": round(float(np.mean(_col("mean_events_per_trial"))), 3) if _col("mean_events_per_trial") else "",
        "mean_input_events_per_trial":  round(float(np.mean(_col("mean_input_events_per_trial"))), 3) if _col("mean_input_events_per_trial") else "",
        "mean_hidden_events_per_trial": round(float(np.mean(_col("mean_hidden_events_per_trial"))), 3) if _col("mean_hidden_events_per_trial") else "",
        "mean_output_events_per_trial": round(float(np.mean(_col("mean_output_events_per_trial"))), 3) if _col("mean_output_events_per_trial") else "",
        "n_timesteps":         round(float(np.mean(_col("n_timesteps"))), 1) if _col("n_timesteps") else "",
        "val_acc_lda":         round(float(np.mean(_col("val_acc_lda"))),   6) if _col("val_acc_lda")  else "",
        "test_acc_lda":        round(float(np.mean(_col("test_acc_lda"))),  6) if _col("test_acc_lda") else "",
        "val_acc_svm":         round(float(np.mean(_col("val_acc_svm"))),   6) if _col("val_acc_svm")  else "",
        "test_acc_svm":        round(float(np.mean(_col("test_acc_svm"))),  6) if _col("test_acc_svm") else "",
        # svm_best_c / svm_best_gamma are per-fold categorical hyperparameter
        # choices (gamma may be a string like "scale") — not meaningful to
        # average, so the mean row leaves them blank; see per-fold rows.
        "svm_best_c":          "",
        "svm_best_gamma":      "",
    }
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(mean_row)

    logger.info("Summary CSV written: %s", csv_path)

    # Log table to stdout
    logger.info("")
    logger.info(
        "  %-6s  %-9s  %-8s  %-8s",
        "Fold", "FP32", "LDA", "SVM",
    )
    logger.info("  " + "-" * 40)
    for r in rows:
        logger.info(
            "  %-6s  %-9.1f  %-8.1f  %-8.1f",
            r["fold"],
            r["test_acc_fp32"]       * 100,
            r.get("test_acc_lda", 0) * 100,
            r.get("test_acc_svm", 0) * 100,
        )
    logger.info("  " + "-" * 40)
    logger.info(
        "  %-6s  %-9.1f  %-8.1f  %-8.1f",
        "MEAN",
        mean_row["test_acc_fp32"]  * 100,                                         # type: ignore[operator]
        (mean_row["test_acc_lda"]  * 100) if mean_row.get("test_acc_lda") else 0.0,
        (mean_row["test_acc_svm"]  * 100) if mean_row.get("test_acc_svm") else 0.0,
    )

    # ---- Aggregated confusion matrix ----
    # Re-load test predictions by re-running inference on each fold's saved model.
    # This is lightweight since we just decode; no retraining.
    class_names = _class_names(cfg, n_classes)
    cm_fp32_sum = np.zeros((n_classes, n_classes), dtype=float)

    _, _, X_test, y_test = _load_raw(cfg)
    y_test_0 = y_test - 1
    sfreq = _sfreq(cfg)

    for r in rows:
        fold_idx = r["fold"]
        fold_dir = subject_dir / f"fold_{fold_idx}"
        params   = r

        bands   = [tuple(b) for b in params["bands"]]
        n_input = params["n_input_features"]

        # Load preprocessing
        try:
            with open(fold_dir / "csp_filters.pkl", "rb") as f:
                csp: PairwiseCSP = pickle.load(f)
            with open(fold_dir / "znorm.pkl", "rb") as f:
                znorm: ZNormaliser = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Preprocessing pickle missing for fold %s — skip", fold_idx)
            continue

        mibif_path = fold_dir / "mibif.pkl"
        mibif: Optional[MIBIFSelector] = None
        if mibif_path.exists():
            with open(mibif_path, "rb") as f:
                mibif = pickle.load(f)

        ft = params.get("filter_type", "butterworth")
        cfg.encoder_type = params.get("encoder_type", cfg.encoder_type)
        X_bands = apply_filter_bank(X_test, bands, sfreq, order=4, filter_type=ft)
        proj    = csp.transform(X_bands)
        X_concat = _concat_projections(proj)
        X_norm  = znorm.transform(X_concat)
        spikes  = _spikes_from_concat(X_norm, cfg)
        if mibif is not None:
            spikes = mibif.transform(spikes)

        model_path = fold_dir / "best_model.pt"
        if not model_path.exists():
            logger.warning("best_model.pt missing for fold %s — skip", fold_idx)
            continue

        model = SNNClassifier(
            n_input=n_input,
            n_hidden=params.get("hidden_neurons", cfg.hidden_neurons),
            n_classes=n_classes,
            population_per_class=params.get("population_per_class", cfg.population_per_class),
            beta=params.get("beta", cfg.beta),
            dropout_prob=0.0,
        ).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)

        acc_fp32, preds_fp32 = evaluate_model(model, spikes, y_test_0, DEVICE)
        cm_fp32_sum += compute_confusion_matrix(
            y_test_0, preds_fp32, n_classes=n_classes, normalize=False
        )

    # Row-normalise the summed matrix
    def _row_norm(cm: np.ndarray) -> np.ndarray:
        row_sums = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sums, where=row_sums > 0)

    _save_cm_from_array(
        _row_norm(cm_fp32_sum), class_names,
        save_path=subject_dir / "confusion_aggregate_fp32.png",
        title=f"Subject {cfg.subject_id} — Aggregated FP32 ({len(rows)} folds)",
    )

    logger.info("Aggregation complete for Subject %d.", cfg.subject_id)


def _save_cm_from_array(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str,
) -> None:
    """Save a pre-computed normalised confusion matrix as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4, n), max(3.5, n)))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
        vmin=0.0, vmax=1.0,
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Aggregated confusion matrix saved: %s", save_path)
