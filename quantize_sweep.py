"""Whole-pipeline weight-quantisation sweep.

Evaluates saved fold artifacts at a range of bit-widths, quantising **every
inference-time parameter in the pipeline** to the same width, so that a result
reported as "4-bit" contains no full-precision parameters anywhere.

Parameter groups quantised
--------------------------
======================  ==========================================  ==========================
Group                   Object                                       Size
======================  ==========================================  ==========================
``ea``                  ``PairwiseCSP.ea_whiteners_``                ``K`` matrices, ``N_ch x N_ch``
``csp``                 ``PairwiseCSP.filters_``                     ``K x pairs`` matrices, ``N_ch x 2m``
``znorm``               ``ZNormaliser.mean_`` / ``.std_``            two length-``F0`` vectors
``snn_w``               ``nn.Linear.weight`` (both layers)           ``H x F`` and ``PC x H``
``snn_b``               ``nn.Linear.bias`` (both layers)             ``H`` and ``PC``
======================  ==========================================  ==========================

Deliberately excluded
---------------------
* **Filter-bank coefficients.**  In the target hardware the bank is an analog
  Gm-C circuit, which has no bit-width at all -- its precision is set by
  component tolerance, which the noise-injection analysis covers separately.
* **MIBIF indices.**  Integer selection indices, not arithmetic operands.
* **Membrane potential and encoder threshold state.**  These are dynamic
  *state*, not stored parameters.  Real neuromorphic silicon does hold them in
  fixed point (Loihi uses integer membrane potentials), so a complete hardware
  study would also sweep state precision -- that is activation quantisation, a
  separate axis from the weight quantisation measured here.

Why the EA whiteners matter
---------------------------
They are sized ``K x N_ch^2`` and so do not shrink when the task becomes
binary, whereas the CSP filters lose their class-pair multiplier.  For a
64-channel binary dataset the whiteners are roughly **8x** the CSP filter
storage, making them the largest front-end parameter block.  Omitting them
would leave the dominant matrix at full precision.

Usage
-----
    # BNCI2015-001 (primary), all 12 subjects, uniform sweep
    python quantize_sweep.py --source moabb --moabb-dataset BNCI2015_001 \\
        --results-dir Results_bnci2015 --subjects 1 2 3 4 5 6 7 8 9 10 11 12 \\
        --n-folds 5 --bits 4 6 8 16 32 --output-dir Results_quant

    # Which stage breaks first: quantise one group at a time
    python quantize_sweep.py ... --mode per-group
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.config import Config
from fbcsp_snn.datasets import get_n_classes
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import apply_filter_bank
from fbcsp_snn.pipeline import _concat_projections, _load_raw, _spikes_from_concat, _sfreq
from fbcsp_snn.ptq import (
    quantize_csp_filters,
    quantize_ea_whiteners,
    quantize_model_full,
    quantize_znorm,
)
from fbcsp_snn.training import evaluate_model
from fusion_experiment import fuse_csp

logger: logging.Logger = setup_logger(__name__)

GROUPS: List[str] = ["ea", "csp", "znorm", "snn_w", "snn_b"]

# With the whitener folded into the spatial filters there is no separately
# stored EA matrix, so it is not a quantisable group.
GROUPS_FUSED: List[str] = ["csp", "znorm", "snn_w", "snn_b"]


# ---------------------------------------------------------------------------
# Single fold evaluation at one bit-width
# ---------------------------------------------------------------------------

def evaluate_fold_at_bits(
    fold_dir: Path,
    X_bands: List[np.ndarray],
    y_test_0: np.ndarray,
    params: Dict,
    cfg: Config,
    n_classes: int,
    bits: Optional[int],
    groups: List[str],
    fuse: bool = False,
) -> float:
    """Quantise the requested groups to *bits* and return test accuracy.

    Parameters
    ----------
    fold_dir : Path
        Directory holding this fold's saved artifacts.
    X_bands : List[np.ndarray]
        Band-pass filtered test data, computed once and reused: the filter
        coefficients are never quantised, so this stage is invariant to *bits*.
    y_test_0 : np.ndarray
        Zero-indexed test labels.
    params : Dict
        Contents of ``pipeline_params.json``.
    cfg : Config
        Config carrying encoder settings.
    n_classes : int
        Number of classes.
    bits : Optional[int]
        Bit-width, or ``None`` for an unquantised FP32 reference.
    groups : List[str]
        Which parameter groups to quantise.

    Returns
    -------
    float
        Test accuracy in ``[0, 1]``.
    """
    with open(fold_dir / "csp_filters.pkl", "rb") as f:
        csp = pickle.load(f)
    with open(fold_dir / "znorm.pkl", "rb") as f:
        znorm = pickle.load(f)

    mibif = None
    mibif_path = fold_dir / "mibif.pkl"
    if mibif_path.exists():
        with open(mibif_path, "rb") as f:
            mibif = pickle.load(f)

    # ---- optional front-end fusion --------------------------------------
    # Compose the EA whitener into the spatial filters before quantising, so
    # the measurement describes a pipeline that stores one matrix per band
    # rather than two. The substitution is algebraically exact, so the FP32
    # reference is unchanged; only the quantisation grid differs.
    if fuse:
        csp = fuse_csp(csp)

    # ---- front-end parameter quantisation -------------------------------
    if bits is not None:
        csp = copy.deepcopy(csp)
        if "ea" in groups and getattr(csp, "ea_whiteners_", None):
            csp.ea_whiteners_ = quantize_ea_whiteners(csp.ea_whiteners_, bits=bits)
        if "csp" in groups:
            csp.filters_ = quantize_csp_filters(csp.filters_, bits=bits)
        if "znorm" in groups:
            znorm = quantize_znorm(znorm, bits=bits)

    # ---- preprocessing chain (must be re-run: front end changed) --------
    proj = csp.transform(X_bands)
    X_concat = _concat_projections(proj)
    X_norm = znorm.transform(X_concat)
    spikes = _spikes_from_concat(X_norm, cfg)
    if mibif is not None:
        spikes = mibif.transform(spikes)

    # ---- classifier -----------------------------------------------------
    model = SNNClassifier(
        n_input=params["n_input_features"],
        n_hidden=params.get("hidden_neurons", cfg.hidden_neurons),
        n_classes=n_classes,
        population_per_class=params.get("population_per_class", cfg.population_per_class),
        beta=params.get("beta", cfg.beta),
        dropout_prob=0.0,
    ).to(DEVICE)
    model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location=DEVICE))

    if bits is not None and ("snn_w" in groups or "snn_b" in groups):
        if "snn_w" in groups and "snn_b" in groups:
            model = quantize_model_full(model, bits=bits)
        else:
            # Isolate one of the two; quantize_model_full does both at once.
            import torch.nn as nn
            model = copy.deepcopy(model)
            from fbcsp_snn.ptq import quantize_tensor_symmetric
            with torch.no_grad():
                for _, mod in model.named_modules():
                    if isinstance(mod, nn.Linear):
                        if "snn_w" in groups:
                            dq, _ = quantize_tensor_symmetric(mod.weight.data, bits)
                            mod.weight.data.copy_(dq)
                        if "snn_b" in groups and mod.bias is not None:
                            dqb, _ = quantize_tensor_symmetric(mod.bias.data, bits)
                            mod.bias.data.copy_(dqb)
    model.eval()

    acc, _ = evaluate_model(model, spikes, y_test_0, DEVICE)
    return float(acc)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def sweep_subject(
    cfg: Config,
    subject_id: int,
    n_folds: int,
    bit_list: List[int],
    mode: str,
    fuse: bool = False,
) -> List[Dict]:
    """Run the bit-width sweep for one subject, returning one row per (fold, condition)."""
    cfg.subject_id = subject_id
    subject_dir = Path(cfg.results_dir) / f"Subject_{subject_id}"
    if not subject_dir.exists():
        logger.warning("Subject %d: %s not found — skipped", subject_id, subject_dir)
        return []

    # Raw data loads once per subject (the expensive step).
    _, _, X_test, y_test = _load_raw(cfg)
    y_test_0 = y_test - 1
    n_classes = cfg.n_classes  # type: ignore[assignment]

    rows: List[Dict] = []
    for fold in range(n_folds):
        fold_dir = subject_dir / f"fold_{fold}"
        params_path = fold_dir / "pipeline_params.json"
        if not params_path.exists():
            logger.warning("S%d fold %d: no pipeline_params.json — skipped", subject_id, fold)
            continue
        with open(params_path) as f:
            params = json.load(f)

        bands = [tuple(b) for b in params["bands"]]
        cfg.encoder_type = params.get("encoder_type", "delta")
        filter_type = params.get("filter_type", "butterworth")

        # Filter bank is never quantised, so compute it once per fold.
        X_bands = apply_filter_bank(
            X_test, bands, _sfreq(cfg), order=4, filter_type=filter_type
        )

        def _run(bits: Optional[int], groups: List[str], label: str) -> None:
            acc = evaluate_fold_at_bits(
                fold_dir, X_bands, y_test_0, params, cfg, n_classes, bits, groups,
                fuse=fuse,
            )
            rows.append({
                "subject": subject_id, "fold": fold,
                "condition": label, "bits": bits if bits is not None else "fp32",
                "test_acc": round(acc * 100, 4),
            })
            logger.info("S%-2d fold %d  %-22s  %.2f%%", subject_id, fold, label, acc * 100)

        _run(None, [], "fp32_reference")

        groups_all = GROUPS_FUSED if fuse else GROUPS
        if mode == "uniform":
            for b in bit_list:
                _run(b, groups_all, f"all@{b}bit")
        else:  # per-group isolation
            for b in bit_list:
                for g in groups_all:
                    _run(b, [g], f"{g}@{b}bit")

    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="moabb")
    p.add_argument("--moabb-dataset", default="BNCI2015_001")
    p.add_argument("--results-dir", required=True,
                   help="Directory containing Subject_*/fold_*/ artifacts.")
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8, 16, 32])
    p.add_argument("--mode", choices=["uniform", "per-group"], default="uniform",
                   help="'uniform' quantises every group to the same width; "
                        "'per-group' isolates one group at a time.")
    p.add_argument("--fuse", action="store_true",
                   help="Compose the EA whitener into the spatial filters "
                        "before quantising, so the measurement describes a "
                        "front end that stores one matrix per band instead of "
                        "two. Exact in FP32; there is then no separate 'ea' "
                        "group. Output filenames gain a '_fused' suffix.")
    p.add_argument("--output-dir", default="Results_quant")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = Config()
    cfg.source = args.source
    cfg.moabb_dataset = args.moabb_dataset
    cfg.results_dir = args.results_dir
    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)

    logger.info(
        "Quantisation sweep — dataset=%s  mode=%s  bits=%s  subjects=%s",
        args.moabb_dataset, args.mode, args.bits, args.subjects,
    )
    logger.info("Groups quantised: %s  (filter-bank coefficients excluded by design)",
                ", ".join(GROUPS_FUSED if args.fuse else GROUPS))
    if args.fuse:
        logger.info("Front end FUSED: storing R^-1/2 W per band, not both factors")

    all_rows: List[Dict] = []
    for sid in args.subjects:
        all_rows.extend(sweep_subject(cfg, sid, args.n_folds, args.bits,
                                      args.mode, fuse=args.fuse))

    if not all_rows:
        logger.error("No results produced — check --results-dir and --subjects")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _suffix = "_fused" if args.fuse else ""
    csv_path = out_dir / f"quant_sweep_{args.moabb_dataset}_{args.mode}{_suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    logger.info("Wrote %d rows -> %s", len(all_rows), csv_path)

    # Per-condition summary across subjects (mean of per-subject fold means).
    print(f"\n{'='*64}\n  {args.moabb_dataset} — {args.mode} sweep\n{'='*64}")
    print(f"  {'condition':<24}{'mean %':>9}{'SD':>8}{'n_subj':>8}")
    print("  " + "-" * 54)
    conditions = sorted({r["condition"] for r in all_rows},
                        key=lambda c: (c != "fp32_reference", c))
    for cond in conditions:
        per_subj = {}
        for r in (r for r in all_rows if r["condition"] == cond):
            per_subj.setdefault(r["subject"], []).append(r["test_acc"])
        means = [float(np.mean(v)) for v in per_subj.values()]
        print(f"  {cond:<24}{np.mean(means):>9.2f}{np.std(means):>8.2f}{len(means):>8d}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
