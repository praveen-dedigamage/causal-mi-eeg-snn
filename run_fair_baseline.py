"""Fairness-controlled baseline for one already-trained fold: reuse the
fold's SAVED (fitted) CSP filters and z-norm stats to re-derive the full
z-normalised time series -- the same signal the SNN's spike encoder
receives -- then fit LDA/SVM on that instead of log-variance.

No SNN retraining, no GPU: this only re-applies already-fitted transforms
(filter bank -> saved CSP -> saved z-norm) to reconstruct the same
train/val/test split the original training run used for this exact fold,
then hands the flattened result to fair_baseline.run_fair_baseline_classifiers.

Usage:
    python run_fair_baseline.py --results-dir Results_verify \\
        --subject-id 1 --fold 0 --moabb-dataset BNCI2014_001
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold

from fbcsp_snn.datasets import DATASET_REGISTRY, load_moabb
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank
from fbcsp_snn.fair_baseline import flatten_timeseries, run_fair_baseline_classifiers


def _concat_projections(proj: dict) -> np.ndarray:
    """Matches fbcsp_snn/pipeline.py's _concat_projections exactly."""
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True,
                     help="Existing results dir from a COMPLETED training "
                          "run (must contain Subject_<id>/fold_<n>/ with "
                          "pipeline_params.json, csp_filters.pkl, znorm.pkl).")
    ap.add_argument("--subject-id", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--moabb-dataset", required=True,
                     help="Must match the dataset the fold was trained on "
                          "(e.g. BNCI2014_001, Schirrmeister2017, Cho2017, "
                          "BNCI2015_001).")
    ap.add_argument("--n-folds", type=int, default=5,
                     help="Every real training run in this project uses "
                          "n_folds=5 with the default val_fraction (=1/5), "
                          "i.e. plain StratifiedKFold -- matches that "
                          "unconditionally (see pipeline.py:451-463).")
    args = ap.parse_args()

    fold_dir = os.path.join(args.results_dir, f"Subject_{args.subject_id}",
                             f"fold_{args.fold}")
    print(f"Loading saved fold artifacts from {fold_dir} ...")
    with open(os.path.join(fold_dir, "pipeline_params.json")) as f:
        params = json.load(f)
    with open(os.path.join(fold_dir, "csp_filters.pkl"), "rb") as f:
        csp: PairwiseCSP = pickle.load(f)
    with open(os.path.join(fold_dir, "znorm.pkl"), "rb") as f:
        znorm: ZNormaliser = pickle.load(f)

    bands = [tuple(b) for b in params["bands"]]
    sfreq = float(DATASET_REGISTRY[args.moabb_dataset]["sfreq"])
    print(f"  bands={bands}  sfreq={sfreq}  n_folds={args.n_folds}")

    print(f"Loading raw MOABB data: {args.moabb_dataset}, "
          f"subject {args.subject_id} ...")
    X_train, y_train, X_test, y_test = load_moabb(
        args.moabb_dataset, args.subject_id
    )

    # Reproduce the EXACT train/val split pipeline.py used for this fold:
    # StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42) on
    # (X_train, y_train), taking the `fold_idx`-th split. X_test is the
    # same held-out set for every fold and is never touched by the splitter.
    splitter = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=42
    )
    for fold_idx, (tr_idx, val_idx) in enumerate(
        splitter.split(X_train, y_train)
    ):
        if fold_idx == args.fold:
            break
    else:
        raise ValueError(f"Fold {args.fold} not found (n_folds={args.n_folds})")

    X_f_tr,  y_f_tr  = X_train[tr_idx],  y_train[tr_idx]
    X_f_val, y_f_val = X_train[val_idx], y_train[val_idx]

    print("Applying filter bank + SAVED (fitted) CSP + SAVED (fitted) "
          "z-norm to train/val/test ...")
    Xb_tr  = apply_filter_bank(X_f_tr,  bands, sfreq)
    Xb_val = apply_filter_bank(X_f_val, bands, sfreq)
    Xb_te  = apply_filter_bank(X_test,  bands, sfreq)

    proj_tr  = csp.transform(Xb_tr)
    proj_val = csp.transform(Xb_val)
    proj_te  = csp.transform(Xb_te)

    X_concat_tr  = _concat_projections(proj_tr).astype(np.float32)
    X_concat_val = _concat_projections(proj_val).astype(np.float32)
    X_concat_te  = _concat_projections(proj_te).astype(np.float32)
    del Xb_tr, Xb_val, Xb_te, proj_tr, proj_val, proj_te

    X_norm_tr  = znorm.transform(X_concat_tr)
    X_norm_val = znorm.transform(X_concat_val)
    X_norm_te  = znorm.transform(X_concat_te)
    del X_concat_tr, X_concat_val, X_concat_te

    print(f"Flattening: {X_norm_tr.shape} -> "
          f"({X_norm_tr.shape[0]}, {X_norm_tr.shape[1] * X_norm_tr.shape[2]})")
    feat_tr  = flatten_timeseries(X_norm_tr)
    feat_val = flatten_timeseries(X_norm_val)
    feat_te  = flatten_timeseries(X_norm_te)
    del X_norm_tr, X_norm_val, X_norm_te

    y_f_tr_0  = y_f_tr  - 1
    y_f_val_0 = y_f_val - 1
    y_test_0  = y_test  - 1

    print("Fitting fair baseline (PCA+LDA, linear SVM) ...")
    results = run_fair_baseline_classifiers(
        feat_tr,  y_f_tr_0,
        feat_val, y_f_val_0,
        feat_te,  y_test_0,
    )

    out_path = os.path.join(fold_dir, "fair_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
