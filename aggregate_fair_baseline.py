"""Aggregate fair_baseline_results.json across folds/subjects into a
per-subject mean +/- SD table, matching the convention of the existing
log-variance baseline tables in the paper.

Usage:
    python aggregate_fair_baseline.py --results-dir Results_verify \\
        --subjects 1 2 3 4 5 6 7 8 9 --n-folds 5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    print("-" * 68)
    print(f"{'Subj':<6}{'LDA (full-ts)':<20}{'SVM (full-ts)':<20}{'Folds':<6}")
    print("-" * 68)

    per_subject_lda: dict[int, float] = {}
    per_subject_svm: dict[int, float] = {}
    missing: list[str] = []

    for subj in args.subjects:
        lda_vals, svm_vals = [], []
        for fold in range(args.n_folds):
            path = os.path.join(
                args.results_dir, f"Subject_{subj}", f"fold_{fold}",
                "fair_baseline_results.json",
            )
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                r = json.load(f)
            lda_vals.append(r["test_acc_lda_fullts"] * 100)
            svm_vals.append(r["test_acc_svm_fullts"] * 100)

        if not lda_vals:
            print(f"S{subj:<5}{'(no data)':<20}{'(no data)':<20}{0:<6}")
            continue

        lda_mean, lda_sd = np.mean(lda_vals), np.std(lda_vals, ddof=0)
        svm_mean, svm_sd = np.mean(svm_vals), np.std(svm_vals, ddof=0)
        per_subject_lda[subj] = lda_mean
        per_subject_svm[subj] = svm_mean
        print(f"S{subj:<5}{f'{lda_mean:.1f}+/-{lda_sd:.1f}':<20}"
              f"{f'{svm_mean:.1f}+/-{svm_sd:.1f}':<20}{len(lda_vals):<6}")

    print("-" * 68)
    if per_subject_lda:
        all_lda = np.array(list(per_subject_lda.values()))
        all_svm = np.array(list(per_subject_svm.values()))
        print(f"{'Mean':<6}{f'{all_lda.mean():.1f}+/-{all_lda.std(ddof=0):.1f}':<20}"
              f"{f'{all_svm.mean():.1f}+/-{all_svm.std(ddof=0):.1f}':<20}"
              f"{len(per_subject_lda) * args.n_folds:<6}")
    print("-" * 68)

    if missing:
        print(f"\n{len(missing)} fold(s) missing fair_baseline_results.json "
              f"(array job may still be running, or failed for these tasks):")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
