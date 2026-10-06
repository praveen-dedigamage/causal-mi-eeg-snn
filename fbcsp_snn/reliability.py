"""Pipeline reliability / hardware-realism testing (B15).

Runs Monte Carlo noise-robustness sweeps on an already-trained fold's saved
checkpoint -- no retraining. Tests every stage of the pipeline that has a
continuous-valued coefficient realisable as a physical analog component:

- Filter bank (Butterworth/Bessel ``sos`` coefficients)
- Euclidean Alignment whitener
- Pairwise CSP spatial filters
- Z-normalisation mean/std
- Spike encoder (threshold, adaptation rate)
- SNN weights and biases
- LIF membrane decay (beta)

...at multiple severities, each repeated N times with fresh random draws,
reporting mean +/- std accuracy and mean spike-events-per-trial per
condition. A joint sweep also perturbs every source simultaneously, since a
real chip has every stage imperfect at once -- testing sources in isolation
doesn't represent the actual deployment scenario (the same principle
already applied to CSP+SNN quantisation in ``pipeline.py``'s joint-PTQ
block). MIBIF feature selection and the Van Rossum loss are deliberately
excluded: MIBIF is a fixed routing decision at inference (no continuous
coefficient to perturb), and the loss function has no inference-time
circuit at all (training-only).

This is distinct from -- and answers a different question than -- the
deterministic bit-quantization sweep already baked into training (see
``pipeline.py``'s PTQ and joint-PTQ blocks). Bit quantization asks "how many
bits does this need"; this module asks "does this survive the kind of noise
an actual analog circuit has" (thermal noise, device mismatch, comparator
offset, RC time-constant variation) -- the deterministic bit-rounding used
elsewhere models none of that.

Usage::

    python main.py reliability --subject-id 1 --fold 0 \\
        --reliability-severities "[0.0,0.05,0.1,0.2,0.3]" \\
        --reliability-n-repeats 20
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.config import Config
from fbcsp_snn.datasets import get_n_classes
from fbcsp_snn.encoding import encode_tensor_with_threshold_noise
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import (
    PairwiseCSP,
    ZNormaliser,
    apply_filter_bank,
    apply_filter_bank_noisy,
)
from fbcsp_snn.quantization import (
    inject_beta_noise,
    inject_csp_filter_noise,
    inject_ea_whitener_noise,
    inject_model_weight_noise,
    inject_znorm_noise,
)
from fbcsp_snn.training import evaluate_model_with_events

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Monte Carlo aggregation (B15 Tier C)
# ---------------------------------------------------------------------------

def monte_carlo_eval(
    eval_fn: Callable[[int], Tuple[float, float]],
    n_repeats: int,
) -> Dict[str, float]:
    """Repeat a noisy evaluation with distinct seeds; aggregate mean +/- std.

    Since analog noise is stochastic, a single deterministic run (as the
    existing bit-quantization sweep does) doesn't characterise robustness --
    every noisy condition needs multiple independent draws.

    Parameters
    ----------
    eval_fn : Callable[[int], Tuple[float, float]]
        Takes a seed, returns ``(accuracy, mean_events_per_trial)`` for one
        noisy draw.
    n_repeats : int
        Number of Monte Carlo repeats with distinct seeds.

    Returns
    -------
    Dict[str, float]
        ``acc_mean``, ``acc_std``, ``events_mean``, ``events_std``.
    """
    accs: List[float] = []
    events: List[float] = []
    for seed in range(n_repeats):
        acc, ev = eval_fn(seed)
        accs.append(acc)
        events.append(ev)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "events_mean": float(np.mean(events)),
        "events_std": float(np.std(events)),
    }


# ---------------------------------------------------------------------------
# Full reliability sweep for one saved fold
# ---------------------------------------------------------------------------

def run_reliability(cfg: Config) -> None:
    """Run the full noise-robustness sweep for one saved fold (B15).

    Loads the fold's saved CSP/znorm/MIBIF/model artifacts (same loading
    logic as ``run_infer``) and, for each of eight noise sources plus one
    joint (all-sources-at-once) condition, at each severity in
    ``cfg.reliability_severities``, runs ``cfg.reliability_n_repeats`` Monte
    Carlo evaluations and aggregates mean +/- std accuracy and mean
    events/trial. Results are saved to ``fold_dir/reliability_results.json``.
    No retraining occurs.

    Sweeps (result dict keys): ``csp_weight_noise``, ``snn_weight_noise``
    (weights + biases), ``beta_noise``, ``filter_bank_noise`` (Butterworth/
    Bessel ``sos`` coefficients), ``ea_whitener_noise``, ``znorm_noise``,
    ``encoder_threshold_noise``, ``encoder_adaptation_noise`` (``adapt_inc``
    + ``decay``), and ``joint_noise_all_sources`` (every source above,
    simultaneously, at matched severity — the actual deployment scenario,
    since testing sources in isolation doesn't represent a real chip where
    every stage is imperfect at once).

    Parameters
    ----------
    cfg : Config
        Must have ``fold`` set; ``reliability_severities`` and
        ``reliability_n_repeats`` control the sweep grid.
    """
    if cfg.fold is None:
        logger.error("--fold is required for reliability mode")
        raise SystemExit(1)

    # Local import: reliability.py reuses pipeline.py's internal data-loading
    # helpers rather than duplicating them. One-directional (pipeline.py does
    # not import from here), so no circular-import risk.
    from fbcsp_snn.pipeline import (
        _concat_projections,
        _load_raw,
        _sfreq,
        _spikes_from_concat,
    )

    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    fold_dir = subject_dir / f"fold_{cfg.fold}"

    params_path = fold_dir / "pipeline_params.json"
    if not params_path.exists():
        logger.error("pipeline_params.json not found at %s -- run train first", fold_dir)
        raise SystemExit(1)
    with open(params_path) as f:
        params = json.load(f)

    bands = [tuple(b) for b in params["bands"]]
    n_input = params["n_input_features"]
    sfreq = _sfreq(cfg)

    logger.info(
        "run_reliability  subject=%d  fold=%d  severities=%s  n_repeats=%d",
        cfg.subject_id, cfg.fold, cfg.reliability_severities, cfg.reliability_n_repeats,
    )

    _, _, X_test, y_test = _load_raw(cfg)
    y_test_0 = y_test - 1

    with open(fold_dir / "csp_filters.pkl", "rb") as f:
        csp: PairwiseCSP = pickle.load(f)
    with open(fold_dir / "znorm.pkl", "rb") as f:
        znorm: ZNormaliser = pickle.load(f)

    mibif: Optional[MIBIFSelector] = None
    mibif_path = fold_dir / "mibif.pkl"
    if mibif_path.exists():
        with open(mibif_path, "rb") as f:
            mibif = pickle.load(f)

    filter_type = params.get("filter_type", "butterworth")
    cfg.encoder_type = params.get("encoder_type", cfg.encoder_type)

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

    saved_csp_filters = csp.filters_
    saved_ea_whiteners = dict(csp.ea_whiteners_)
    saved_znorm_mean = znorm.mean_.copy()
    saved_znorm_std = znorm.std_.copy()

    def _encode_test(
        *,
        csp_filters: Optional[dict] = None,
        ea_whiteners: Optional[dict] = None,
        znorm_mean: Optional[np.ndarray] = None,
        znorm_std: Optional[np.ndarray] = None,
        filter_sigma_frac: float = 0.0,
        filter_seed: Optional[int] = None,
        encoder_thresh_sigma: float = 0.0,
        encoder_adapt_sigma: float = 0.0,
        encoder_decay_sigma: float = 0.0,
        encoder_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Re-run the preprocessing chain on test data, with optional noise at
        any stage. Every parameter defaults to the clean/saved value — call
        with no arguments to reproduce the noise-free pipeline exactly.
        """
        csp.filters_ = csp_filters if csp_filters is not None else saved_csp_filters
        csp.ea_whiteners_ = ea_whiteners if ea_whiteners is not None else saved_ea_whiteners
        znorm.mean_ = znorm_mean if znorm_mean is not None else saved_znorm_mean
        znorm.std_ = znorm_std if znorm_std is not None else saved_znorm_std

        if filter_sigma_frac > 0.0:
            X_bands = apply_filter_bank_noisy(
                X_test, bands, sfreq, order=4, filter_type=filter_type,
                sigma_frac=filter_sigma_frac, seed=filter_seed,
            )
        else:
            X_bands = apply_filter_bank(X_test, bands, sfreq, order=4, filter_type=filter_type)

        proj = csp.transform(X_bands)
        X_concat = _concat_projections(proj)
        X_norm = znorm.transform(X_concat)

        if encoder_thresh_sigma > 0.0 or encoder_adapt_sigma > 0.0 or encoder_decay_sigma > 0.0:
            t = torch.from_numpy(X_norm).to(DEVICE).permute(2, 0, 1)   # (T, B, F)
            spikes = encode_tensor_with_threshold_noise(
                t, cfg.base_thresh, cfg.adapt_inc, cfg.decay,
                sigma_frac=encoder_thresh_sigma,
                adapt_inc_sigma_frac=encoder_adapt_sigma,
                decay_sigma_frac=encoder_decay_sigma,
                seed=encoder_seed,
            )
        else:
            spikes = _spikes_from_concat(X_norm, cfg)

        if mibif is not None:
            spikes = mibif.transform(spikes)
        return spikes

    # ---- Clean (noise-free) baseline ----
    spikes_clean = _encode_test()
    clean_acc, _, clean_events = evaluate_model_with_events(
        model, spikes_clean, y_test_0, DEVICE
    )
    logger.info("Clean baseline -- acc: %.4f  events/trial: %.1f", clean_acc, clean_events)

    severities = cfg.reliability_severities
    n_repeats = cfg.reliability_n_repeats
    results: Dict[str, object] = {
        "clean": {"acc": clean_acc, "events_per_trial": clean_events},
        "n_repeats": n_repeats,
        "severities": severities,
    }

    def _sweep(name: str, eval_at_sigma: Callable[[float, int], Tuple[float, float]]) -> None:
        sweep_results: Dict[str, dict] = {}
        for sigma in severities:
            if sigma == 0.0:
                sweep_results[str(sigma)] = {
                    "acc_mean": clean_acc, "acc_std": 0.0,
                    "events_mean": clean_events, "events_std": 0.0,
                }
                continue
            stats = monte_carlo_eval(lambda seed, s=sigma: eval_at_sigma(s, seed), n_repeats)
            sweep_results[str(sigma)] = stats
            logger.info(
                "%s  sigma=%.3f  acc %.4f+-%.4f  events %.1f+-%.1f",
                name, sigma, stats["acc_mean"], stats["acc_std"],
                stats["events_mean"], stats["events_std"],
            )
        results[name] = sweep_results

    # ---- CSP weight noise (conductance variation in an analog crossbar) ----
    def _eval_csp_noise(sigma: float, seed: int) -> Tuple[float, float]:
        noisy_filters = inject_csp_filter_noise(saved_csp_filters, sigma, seed=seed)
        spikes_noisy = _encode_test(csp_filters=noisy_filters)
        acc, _, ev = evaluate_model_with_events(model, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("csp_weight_noise", _eval_csp_noise)

    # ---- SNN weight + bias noise (conductance variation in the classifier) ----
    def _eval_snn_noise(sigma: float, seed: int) -> Tuple[float, float]:
        model_noisy = inject_model_weight_noise(model, sigma, seed=seed)
        acc, _, ev = evaluate_model_with_events(model_noisy, spikes_clean, y_test_0, DEVICE)
        return acc, ev
    _sweep("snn_weight_noise", _eval_snn_noise)

    # ---- Beta (leak-rate) variation (RC time-constant mismatch) ----
    def _eval_beta_noise(sigma: float, seed: int) -> Tuple[float, float]:
        model_noisy = inject_beta_noise(model, sigma, seed=seed)
        acc, _, ev = evaluate_model_with_events(model_noisy, spikes_clean, y_test_0, DEVICE)
        return acc, ev
    _sweep("beta_noise", _eval_beta_noise)

    # ---- Filter bank coefficient noise (analog Gm-C component variation) ----
    # Closes the most important scope gap: the paper's hardware argument
    # specifically leans on this stage being Gm-C-circuit realisable, but
    # that claim was never stress-tested until now.
    def _eval_filter_noise(sigma: float, seed: int) -> Tuple[float, float]:
        spikes_noisy = _encode_test(filter_sigma_frac=sigma, filter_seed=seed)
        acc, _, ev = evaluate_model_with_events(model, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("filter_bank_noise", _eval_filter_noise)

    # ---- Euclidean Alignment whitener noise (same class as CSP weights) ----
    # No-ops gracefully if this fold's CSP was fit with euclidean_alignment=
    # False (saved_ea_whiteners is then empty — nothing to perturb, sweep
    # still runs but shows no change, not an error).
    def _eval_ea_noise(sigma: float, seed: int) -> Tuple[float, float]:
        noisy_ea = inject_ea_whitener_noise(saved_ea_whiteners, sigma, seed=seed)
        spikes_noisy = _encode_test(ea_whiteners=noisy_ea)
        acc, _, ev = evaluate_model_with_events(model, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("ea_whitener_noise", _eval_ea_noise)

    # ---- Z-normalisation mean/std noise (analog reference/gain drift) ----
    def _eval_znorm_noise(sigma: float, seed: int) -> Tuple[float, float]:
        noisy_mean, noisy_std = inject_znorm_noise(
            saved_znorm_mean, saved_znorm_std, sigma, seed=seed
        )
        spikes_noisy = _encode_test(znorm_mean=noisy_mean, znorm_std=noisy_std)
        acc, _, ev = evaluate_model_with_events(model, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("znorm_noise", _eval_znorm_noise)

    # ---- Spike encoder: comparator/threshold offset noise ----
    def _eval_encoder_threshold_noise(sigma: float, seed: int) -> Tuple[float, float]:
        spikes_noisy = _encode_test(encoder_thresh_sigma=sigma, encoder_seed=seed)
        acc, _, ev = evaluate_model_with_events(model, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("encoder_threshold_noise", _eval_encoder_threshold_noise)

    # ---- Spike encoder: adaptation-rate mismatch (adapt_inc + decay) ----
    # Same analog-time-constant reasoning as LIF beta variation, applied to
    # the encoder's own adaptation circuitry instead of the neuron's.
    def _eval_encoder_adaptation_noise(sigma: float, seed: int) -> Tuple[float, float]:
        spikes_noisy = _encode_test(
            encoder_adapt_sigma=sigma, encoder_decay_sigma=sigma, encoder_seed=seed,
        )
        acc, _, ev = evaluate_model_with_events(model, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("encoder_adaptation_noise", _eval_encoder_adaptation_noise)

    # ---- Joint noise: every source simultaneously at matched severity -----
    # A real chip has every stage imperfect at once -- testing sources in
    # isolation (as every sweep above does) doesn't represent the actual
    # deployment scenario. Same principle as pipeline.py's joint CSP+SNN
    # quantisation grid, extended to all seven noise sources here.
    def _eval_joint_noise(sigma: float, seed: int) -> Tuple[float, float]:
        noisy_csp_filters = inject_csp_filter_noise(saved_csp_filters, sigma, seed=seed)
        noisy_ea = inject_ea_whitener_noise(saved_ea_whiteners, sigma, seed=seed + 1000)
        noisy_mean, noisy_std = inject_znorm_noise(
            saved_znorm_mean, saved_znorm_std, sigma, seed=seed + 2000
        )
        spikes_noisy = _encode_test(
            csp_filters=noisy_csp_filters,
            ea_whiteners=noisy_ea,
            znorm_mean=noisy_mean, znorm_std=noisy_std,
            filter_sigma_frac=sigma, filter_seed=seed + 3000,
            encoder_thresh_sigma=sigma, encoder_adapt_sigma=sigma,
            encoder_decay_sigma=sigma, encoder_seed=seed + 4000,
        )
        model_noisy = inject_model_weight_noise(model, sigma, seed=seed + 5000)
        model_noisy = inject_beta_noise(model_noisy, sigma, seed=seed + 6000)
        acc, _, ev = evaluate_model_with_events(model_noisy, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("joint_noise_all_sources", _eval_joint_noise)

    # Restore clean state (defensive — nothing reads csp/znorm after this,
    # but keeps the pattern consistent in case of future refactoring).
    csp.filters_ = saved_csp_filters
    csp.ea_whiteners_ = saved_ea_whiteners
    znorm.mean_ = saved_znorm_mean
    znorm.std_ = saved_znorm_std

    out_path = fold_dir / "reliability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Reliability sweep saved to %s", out_path)
