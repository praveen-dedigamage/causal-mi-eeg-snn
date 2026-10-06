#!/usr/bin/env python
"""Fuse the EA whitener into the CSP filters, then re-run the bit sweep.

Motivation
----------
The per-group sweep shows the Euclidean-Alignment whitener is the stage that
fails first under quantisation: at 4 bits it takes accuracy to chance while
every other group is untouched.  That is a property of its numbers, not of the
idea -- the whitener is built from ``eigenvalue ** -0.5`` and so spans a wide
dynamic range, which a fixed-point grid represents badly.

But the whitener never needs to be a separate stored stage.  EA and CSP are
consecutive linear maps on the same signal::

    proj[t,f,s] = sum_c W[c,f] * (R X)[t,c,s]
                = sum_k (R^T W)[k,f] * X[t,k,s]

so ``W_eff = R^T W`` reproduces the pipeline exactly with one matrix instead of
two.  (``R`` here is the whitener ``R^-1/2``, symmetric by construction, but
``R^T W`` is used so the identity does not depend on that.)

Two consequences, both measured here:

1. **Storage.**  Per band the front end drops from ``n_ch^2 + n_ch*2m`` to
   ``n_ch*2m`` values.  At 13 channels that is 273 -> 104; at Cho2017's 64
   channels, 4608 -> 512, a 9x reduction.
2. **Quantisation.**  The split form rounds twice, once into the whitener's
   wide grid and once into the filters'.  The fused form rounds once, into a
   single matrix whose dynamic range is that of the product rather than of
   ``eigenvalue ** -0.5``.

The experiment is post-hoc: it rebuilds the projection from saved fold
artifacts and never retrains, so a difference in accuracy can only come from
the representation.

Usage
-----
::

    python fusion_experiment.py --results-dir Results_bnci2015 \\
        --subjects 1 2 3 --n-folds 5 --output-dir Results_fusion
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from fbcsp_snn import DEVICE, setup_logger, set_global_seed
from fbcsp_snn.config import Config
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.pipeline import _concat_projections, _load_raw, _spikes_from_concat, _sfreq
from fbcsp_snn.preprocessing import apply_filter_bank
from fbcsp_snn.ptq import (
    quantize_array_symmetric,
    quantize_csp_filters,
    quantize_ea_whiteners,
)
from fbcsp_snn.training import evaluate_model

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse_csp(csp):
    """Return a copy of *csp* with EA folded into the spatial filters.

    Parameters
    ----------
    csp : PairwiseCSP
        Fitted object carrying ``ea_whiteners_`` and ``filters_``.

    Returns
    -------
    PairwiseCSP
        Copy whose ``filters_[(b, pair)]`` equals ``R_b^T W`` and whose EA
        stage is disabled, so ``transform()`` applies the fused matrix
        directly to the unwhitened band signal.
    """
    fused = copy.deepcopy(csp)
    whiteners = getattr(csp, "ea_whiteners_", None) or {}
    if not whiteners:
        raise ValueError("This fold has no EA whiteners; nothing to fuse.")

    for (b_idx, pair), W in csp.filters_.items():
        R = whiteners[b_idx]                      # (n_ch, n_ch)
        fused.filters_[(b_idx, pair)] = R.T @ W   # (n_ch, 2m)

    # Disable the now-redundant stage. Clearing the dict alone would suffice
    # (transform() checks membership), but the flag is cleared too so the
    # object cannot be mistaken for one that still whitens.
    fused.ea_whiteners_ = {}
    fused.euclidean_alignment = False
    return fused


def parameter_counts(csp) -> Dict[str, int]:
    """Count stored front-end values in the split and fused forms."""
    whiteners = getattr(csp, "ea_whiteners_", None) or {}
    n_ea = int(sum(w.size for w in whiteners.values()))
    n_filt = int(sum(W.size for W in csp.filters_.values()))
    return {
        "ea_values": n_ea,
        "csp_values": n_filt,
        "split_total": n_ea + n_filt,
        "fused_total": n_filt,      # fusion leaves the filter shape unchanged
        "reduction_x": round((n_ea + n_filt) / n_filt, 3) if n_filt else float("nan"),
    }


def dynamic_ranges(csp, fused) -> Dict[str, float]:
    """Ratio of largest to smallest non-negligible magnitude in each matrix set.

    This is the quantity a fixed-point grid struggles with: a symmetric
    per-tensor scale is set by ``max|w|``, so a wide spread leaves the small
    entries with very few levels.
    """
    def spread(mats) -> float:
        vals = np.concatenate([np.abs(m).ravel() for m in mats])
        vals = vals[vals > 0]
        if vals.size == 0:
            return float("nan")
        return float(vals.max() / np.percentile(vals, 1))

    whiteners = list((getattr(csp, "ea_whiteners_", None) or {}).values())
    return {
        "ea_range": spread(whiteners) if whiteners else float("nan"),
        "csp_range": spread(list(csp.filters_.values())),
        "fused_range": spread(list(fused.filters_.values())),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_pipeline(csp, znorm, mibif, X_bands, y_test_0, params, cfg,
                  n_classes: int) -> Tuple[float, np.ndarray]:
    """Push test data through a given front end and the trained classifier."""
    proj = csp.transform(X_bands)
    X_concat = _concat_projections(proj)
    X_norm = znorm.transform(X_concat)
    spikes = _spikes_from_concat(X_norm, cfg)
    if mibif is not None:
        spikes = mibif.transform(spikes)

    model = SNNClassifier(
        n_input=params["n_input_features"],
        n_hidden=params.get("hidden_neurons", cfg.hidden_neurons),
        n_classes=n_classes,
        population_per_class=params.get("population_per_class",
                                        cfg.population_per_class),
        beta=params.get("beta", cfg.beta),
        dropout_prob=0.0,
    ).to(DEVICE)
    model.load_state_dict(torch.load(params["_model_path"], map_location=DEVICE))
    model.eval()

    acc, _ = evaluate_model(model, spikes, y_test_0, DEVICE)
    return float(acc), X_concat


def evaluate_fold(fold_dir: Path, X_bands, y_test_0, params, cfg,
                  n_classes: int, bit_list: List[int]) -> List[Dict]:
    """Compare split vs fused front ends at every bit-width for one fold."""
    with open(fold_dir / "csp_filters.pkl", "rb") as f:
        csp = pickle.load(f)
    with open(fold_dir / "znorm.pkl", "rb") as f:
        znorm = pickle.load(f)
    mibif = None
    if (fold_dir / "mibif.pkl").exists():
        with open(fold_dir / "mibif.pkl", "rb") as f:
            mibif = pickle.load(f)

    params = dict(params)
    params["_model_path"] = fold_dir / "best_model.pt"

    fused = fuse_csp(csp)
    rows: List[Dict] = []

    # --- FP32 equivalence: the fusion must be an identity, not an approximation
    acc_split, proj_split = _run_pipeline(csp, znorm, mibif, X_bands,
                                          y_test_0, params, cfg, n_classes)
    acc_fused, proj_fused = _run_pipeline(fused, znorm, mibif, X_bands,
                                          y_test_0, params, cfg, n_classes)
    denom = float(np.abs(proj_split).max()) or 1.0
    rel_err = float(np.abs(proj_split - proj_fused).max() / denom)

    counts = parameter_counts(csp)
    ranges = dynamic_ranges(csp, fused)

    rows.append({
        "condition": "fp32", "bits": "fp32",
        "acc_split": round(acc_split * 100, 4),
        "acc_fused": round(acc_fused * 100, 4),
        "projection_rel_err": rel_err,
        **counts, **ranges,
    })

    # --- quantised: split rounds twice (EA then CSP), fused rounds once
    for b in bit_list:
        split_q = copy.deepcopy(csp)
        split_q.ea_whiteners_ = quantize_ea_whiteners(csp.ea_whiteners_, bits=b)
        split_q.filters_ = quantize_csp_filters(csp.filters_, bits=b)

        fused_q = copy.deepcopy(fused)
        fused_q.filters_ = {k: quantize_array_symmetric(v, bits=b)[0]
                            for k, v in fused.filters_.items()}

        a_split, _ = _run_pipeline(split_q, znorm, mibif, X_bands,
                                   y_test_0, params, cfg, n_classes)
        a_fused, _ = _run_pipeline(fused_q, znorm, mibif, X_bands,
                                   y_test_0, params, cfg, n_classes)
        rows.append({
            "condition": f"frontend@{b}bit", "bits": b,
            "acc_split": round(a_split * 100, 4),
            "acc_fused": round(a_fused * 100, 4),
            "projection_rel_err": None,
            **counts, **ranges,
        })
        logger.info("    %2d-bit   split %6.2f%%   fused %6.2f%%   (%+.2f)",
                    b, a_split * 100, a_fused * 100,
                    (a_fused - a_split) * 100)

    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the fusion comparison over the requested subjects and folds."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="Results_bnci2015")
    ap.add_argument("--dataset", default="BNCI2015_001")
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8, 16, 32])
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="Results_fusion")
    args = ap.parse_args()

    set_global_seed(args.seed)
    logging.getLogger("moabb").setLevel(logging.ERROR)
    logging.getLogger("mne").setLevel(logging.ERROR)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    all_rows: List[Dict] = []
    for subject in args.subjects:
        subject_dir = results_dir / f"Subject_{subject}"
        if not subject_dir.is_dir():
            logger.warning("S%d: %s missing - skipped", subject, subject_dir)
            continue

        cfg = Config()
        cfg.source = "moabb"
        cfg.moabb_dataset = args.dataset
        cfg.subject_id = subject
        cfg.n_classes = args.n_classes
        _, _, X_test, y_test = _load_raw(cfg)
        y_test_0 = y_test - 1

        for fold in range(args.n_folds):
            fold_dir = subject_dir / f"fold_{fold}"
            params_path = fold_dir / "pipeline_params.json"
            if not params_path.exists():
                continue
            with open(params_path) as f:
                params = json.load(f)

            bands = [tuple(b) for b in params["bands"]]
            cfg.freq_bands = bands
            X_bands = apply_filter_bank(
                X_test, bands, _sfreq(cfg), order=4,
                filter_type=params.get("filter_type", cfg.filter_type),
            )

            logger.info("S%d fold %d", subject, fold)
            for row in evaluate_fold(fold_dir, X_bands, y_test_0, params,
                                     cfg, args.n_classes, args.bits):
                all_rows.append({"subject": subject, "fold": fold, **row})

    if not all_rows:
        logger.error("No folds evaluated.")
        return

    import csv as _csv
    csv_path = out_dir / f"fusion_{args.dataset}.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s", csv_path)

    # ---- summary -------------------------------------------------------
    fp32 = [r for r in all_rows if r["bits"] == "fp32"]
    max_rel_err = max(r["projection_rel_err"] for r in fp32)
    summary: Dict = {
        "dataset": args.dataset,
        "n_folds_evaluated": len(fp32),
        "fp32_equivalence": {
            "max_projection_rel_err": max_rel_err,
            "identity_holds": bool(max_rel_err < 1e-6),
            "note": "fusion is algebraically exact; a large value means the "
                    "transform order assumed here does not match the pipeline",
        },
        "storage": {
            "split_values_per_fold": fp32[0]["split_total"],
            "fused_values_per_fold": fp32[0]["fused_total"],
            "reduction_x": fp32[0]["reduction_x"],
        },
        "dynamic_range": {
            "ea": fp32[0]["ea_range"],
            "csp": fp32[0]["csp_range"],
            "fused": fp32[0]["fused_range"],
        },
        "accuracy_by_bits": {},
    }
    for b in args.bits:
        rows_b = [r for r in all_rows if r["bits"] == b]
        if rows_b:
            summary["accuracy_by_bits"][f"{b}bit"] = {
                "split_mean": round(float(np.mean([r["acc_split"] for r in rows_b])), 3),
                "fused_mean": round(float(np.mean([r["acc_fused"] for r in rows_b])), 3),
                "delta": round(float(np.mean([r["acc_fused"] - r["acc_split"]
                                              for r in rows_b])), 3),
            }
    summary["accuracy_by_bits"]["fp32"] = {
        "split_mean": round(float(np.mean([r["acc_split"] for r in fp32])), 3),
        "fused_mean": round(float(np.mean([r["acc_fused"] for r in fp32])), 3),
        "delta": round(float(np.mean([r["acc_fused"] - r["acc_split"] for r in fp32])), 3),
    }

    json_path = out_dir / f"fusion_{args.dataset}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", json_path)

    print("\n" + "=" * 66)
    print(f"  EA/CSP fusion - {args.dataset}  ({len(fp32)} folds)")
    print("=" * 66)
    print(f"  FP32 identity holds : {summary['fp32_equivalence']['identity_holds']} "
          f"(max rel err {max_rel_err:.2e})")
    print(f"  Front-end storage   : {summary['storage']['split_values_per_fold']} "
          f"-> {summary['storage']['fused_values_per_fold']} values "
          f"({summary['storage']['reduction_x']}x)")
    print(f"  Dynamic range       : EA {summary['dynamic_range']['ea']:.1f}x   "
          f"CSP {summary['dynamic_range']['csp']:.1f}x   "
          f"fused {summary['dynamic_range']['fused']:.1f}x")
    print("  " + "-" * 62)
    print(f"  {'bits':>6}  {'split %':>9}  {'fused %':>9}  {'delta':>8}")
    for k, v in summary["accuracy_by_bits"].items():
        print(f"  {k:>6}  {v['split_mean']:>9.2f}  {v['fused_mean']:>9.2f}  "
              f"{v['delta']:>+8.2f}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
