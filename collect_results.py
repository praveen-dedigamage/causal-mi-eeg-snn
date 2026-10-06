#!/usr/bin/env python
"""Consolidate every run artifact into a single evidence bundle.

Training writes one ``pipeline_params.json`` per fold and the quantisation
sweep writes one CSV per mode.  For analysis (and for a paper that has to be
defensible months later) those need to live in one self-describing file
alongside the provenance needed to reproduce them.

This script reads the artifacts, never the logs, so what it records is exactly
what the pipeline saved.  It is read-only with respect to the run.

Usage
-----
::

    python collect_results.py \\
        --results-dir Results_bnci2015 \\
        --quant-dir   Results_quant \\
        --dataset     BNCI2015_001 \\
        --output      results_bundle_BNCI2015_001.json \\
        --markdown    results_summary_BNCI2015_001.md

The JSON bundle is canonical: it holds every per-fold record, every sweep row,
derived summaries, and integrity checks.  The Markdown file is a readable view
of the same numbers and is regenerated from the bundle, never edited by hand.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import socket
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    # Preferred: the project's shared logger, for output consistent with runs.
    from fbcsp_snn import setup_logger

    logger = setup_logger(__name__)
except ImportError:  # pragma: no cover - depends on environment
    # Deliberate fallback. This script only reads JSON and CSV, so it must not
    # require torch/snntorch: it has to be runnable on a login node, or on a
    # laptop that only has the downloaded artifacts and no ML stack.
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

# Accuracy fields inside pipeline_params.json that are reported per subject.
ACC_FIELDS = (
    "test_acc_fp32",
    "test_acc_lda",
    "test_acc_svm",
    # Written by run_fair_baseline.py: the same classifiers refit on the full
    # time series the spike encoder receives, rather than on log-variance.
    "test_acc_lda_fullts",
    "test_acc_svm_fullts",
)


# ---------------------------------------------------------------- provenance


def _git(*args: str) -> Optional[str]:
    """Run a git command, returning ``None`` if git or the repo is unavailable.

    Parameters
    ----------
    *args : str
        Arguments passed to ``git``.

    Returns
    -------
    str or None
        Stripped stdout, or ``None`` when git fails for any reason.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _lib_version(module: str) -> Optional[str]:
    """Return an installed library's version string, or ``None``.

    Imported lazily and failing soft, because this script is deliberately
    runnable on a login node where torch may not be importable. A bundle
    collected there records ``None`` rather than refusing to build.

    The versions matter: snnTorch changed surrogate-gradient internals and
    ``init_leaky()`` behaviour across releases, so a bundle that does not name
    the version cannot be reproduced from. ``requirements.txt`` alone is not
    enough, since a range like ``snntorch>=0.9`` admits several incompatible
    releases.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"torch"``.

    Returns
    -------
    str or None
        ``module.__version__``, or ``None`` if the import or attribute fails.
    """
    try:
        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:
        return None


def build_provenance(dataset: str) -> Dict[str, Any]:
    """Capture who/where/when produced the bundle.

    Parameters
    ----------
    dataset : str
        MOABB dataset identifier the results belong to.

    Returns
    -------
    dict
        Provenance record embedded at the top of the bundle.
    """
    return {
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": _lib_version("torch"),
        "snntorch": _lib_version("snntorch"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


# ------------------------------------------------------------------ loading


def load_folds(results_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read every ``pipeline_params.json`` under ``results_dir``.

    Parameters
    ----------
    results_dir : Path
        Directory holding ``Subject_<N>/fold_<K>/`` trees.

    Returns
    -------
    records : list of dict
        One record per fold, with ``subject`` and ``fold`` prepended.
    problems : list of str
        Human-readable notes about anything missing or unreadable.
    """
    records: List[Dict[str, Any]] = []
    problems: List[str] = []

    if not results_dir.is_dir():
        return records, [f"results dir not found: {results_dir}"]

    for sub_dir in sorted(
        results_dir.glob("Subject_*"), key=lambda p: int(p.name.split("_")[1])
    ):
        subject = int(sub_dir.name.split("_")[1])
        fold_dirs = sorted(
            sub_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1])
        )
        if not fold_dirs:
            problems.append(f"S{subject}: no fold_* directories")
            continue

        for fold_dir in fold_dirs:
            fold = int(fold_dir.name.split("_")[1])
            path = fold_dir / "pipeline_params.json"
            if not path.exists():
                problems.append(f"S{subject} fold {fold}: pipeline_params.json missing")
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    params = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"S{subject} fold {fold}: unreadable ({exc})")
                continue

            # Fairness-controlled baselines live in a separate file, written
            # by a later job. Merge them in when present so every accuracy for
            # this fold travels together; absence is normal, not an error.
            fair_path = fold_dir / "fair_baseline_results.json"
            if fair_path.exists():
                try:
                    with open(fair_path, encoding="utf-8") as f:
                        params.update(json.load(f))
                except (OSError, json.JSONDecodeError) as exc:
                    problems.append(
                        f"S{subject} fold {fold}: fair_baseline unreadable ({exc})")

            records.append({"subject": subject, "fold": fold, **params})

    return records, problems


def load_quant(quant_dir: Path, dataset: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read the sweep CSVs, tagging each row with the mode it came from.

    Parameters
    ----------
    quant_dir : Path
        Directory containing ``quant_sweep_<dataset>_<mode>.csv`` files.
    dataset : str
        Dataset identifier used in the filenames.

    Returns
    -------
    rows : list of dict
        Sweep rows with an added ``mode`` key.
    problems : list of str
        Notes about missing files.
    """
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []

    if not quant_dir.is_dir():
        return rows, [f"quant dir not found: {quant_dir} (sweep may not have run yet)"]

    matches = sorted(quant_dir.glob(f"quant_sweep_{dataset}_*.csv"))
    if not matches:
        problems.append(f"no quant_sweep_{dataset}_*.csv in {quant_dir}")

    for path in matches:
        mode = path.stem.replace(f"quant_sweep_{dataset}_", "")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append(
                        {
                            "mode": mode,
                            "subject": int(row["subject"]),
                            "fold": int(row["fold"]),
                            "condition": row["condition"],
                            "bits": row["bits"],
                            "test_acc": float(row["test_acc"]),
                        }
                    )
        except (OSError, csv.Error, KeyError, ValueError) as exc:
            problems.append(f"{path.name}: unreadable ({exc})")

    return rows, problems


# ---------------------------------------------------------------- summarising


def _mean_sd(values: Iterable[float]) -> Dict[str, Optional[float]]:
    """Mean and *sample* standard deviation (ddof=1) of ``values``.

    Sample rather than population SD, matching how per-subject spread is
    normally reported; with n=1 the SD is ``None`` rather than 0.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": round(statistics.mean(vals), 4),
        "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
        "n": len(vals),
    }


def summarise_folds(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-fold records to per-subject and cross-subject level.

    The cross-subject figure averages *subject means*, not raw folds, so a
    subject with missing folds cannot skew the headline number.
    """
    by_subject: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_subject.setdefault(r["subject"], []).append(r)

    per_subject: Dict[str, Any] = {}
    for subject, recs in sorted(by_subject.items()):
        entry: Dict[str, Any] = {"n_folds": len(recs)}
        for field in ACC_FIELDS:
            # stored as fractions; report percentages
            entry[field] = _mean_sd(
                [r[field] * 100 for r in recs if r.get(field) is not None]
            )
        entry["n_features_selected"] = _mean_sd(
            [r["n_features_selected"] for r in recs if r.get("n_features_selected")]
        )
        entry["mean_events_per_trial"] = _mean_sd(
            [r["mean_events_per_trial"] for r in recs if r.get("mean_events_per_trial")]
        )
        per_subject[str(subject)] = entry

    across = {
        field: _mean_sd(
            [
                per_subject[s][field]["mean"]
                for s in per_subject
                if per_subject[s][field]["mean"] is not None
            ]
        )
        for field in ACC_FIELDS
    }

    return {
        "per_subject": per_subject,
        "across_subjects": across,
        "significance": paired_tests(per_subject),
        "note": (
            "across_subjects averages subject means (not raw folds); "
            "sd is sample sd (ddof=1) over subjects"
        ),
    }


def paired_tests(per_subject: Dict[str, Any]) -> Dict[str, Any]:
    """Paired comparisons between the SNN and each classical baseline.

    Pairing is by subject, using each subject's mean over folds, because folds
    within a subject are not independent observations -- they share the same
    recording and differ only in which trials were held out for validation.
    Treating 60 folds as 60 samples would inflate significance roughly
    fivefold.

    Reports both a paired t-test and a Wilcoxon signed-rank test: the t-test
    assumes approximately normal differences, which 12 subjects cannot
    establish, so the rank-based test is the safer of the two to quote when
    they disagree.

    Parameters
    ----------
    per_subject : dict
        Output of :func:`summarise_folds`'s ``per_subject`` field.

    Returns
    -------
    dict
        One entry per comparison, or a note if SciPy is unavailable.
    """
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover - depends on environment
        return {
            "note": "SciPy not available; run with the PyTorch module loaded "
                    "(module load python-pytorch/2.10) to compute these."
        }

    subjects = sorted(per_subject, key=int)

    def series(field: str) -> List[Optional[float]]:
        return [per_subject[s][field]["mean"] for s in subjects]

    out: Dict[str, Any] = {}
    snn = series("test_acc_fp32")
    for label, field in (("snn_vs_lda", "test_acc_lda"),
                         ("snn_vs_svm", "test_acc_svm"),
                         ("snn_vs_lda_fullts", "test_acc_lda_fullts"),
                         ("snn_vs_svm_fullts", "test_acc_svm_fullts")):
        other = series(field)
        pairs = [(a, b) for a, b in zip(snn, other)
                 if a is not None and b is not None]
        if len(pairs) < 3:
            out[label] = {"n": len(pairs), "note": "too few subjects to test"}
            continue
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        diffs = [x - y for x, y in zip(a, b)]

        t_p = float(stats.ttest_rel(a, b).pvalue)
        # Wilcoxon is undefined when every difference is zero.
        try:
            w_p = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            w_p = float("nan")

        out[label] = {
            "n_subjects": len(pairs),
            "mean_difference": round(float(statistics.mean(diffs)), 4),
            "snn_wins": sum(1 for d in diffs if d > 0),
            "ties": sum(1 for d in diffs if d == 0),
            "paired_t_p": t_p,
            "wilcoxon_p": w_p,
            "significant_at_0.05": bool(min(t_p, w_p) < 0.05)
                                   if w_p == w_p else bool(t_p < 0.05),
        }
    return out


def summarise_quant(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate sweep rows per condition, with deltas against fp32.

    For every condition the mean is taken over subject means, and the delta is
    computed against each subject's own fp32 reference before averaging, so a
    strong subject does not dominate the reported degradation.
    """
    out: Dict[str, Any] = {}

    for mode in sorted({r["mode"] for r in rows}):
        mrows = [r for r in rows if r["mode"] == mode]

        # subject -> condition -> [accs]
        nested: Dict[int, Dict[str, List[float]]] = {}
        for r in mrows:
            nested.setdefault(r["subject"], {}).setdefault(r["condition"], []).append(
                r["test_acc"]
            )

        subj_means = {
            s: {c: statistics.mean(v) for c, v in conds.items()}
            for s, conds in nested.items()
        }

        conditions = sorted(
            {c for conds in subj_means.values() for c in conds},
            key=lambda c: (c != "fp32_reference", c),
        )

        per_condition: Dict[str, Any] = {}
        for cond in conditions:
            accs = [m[cond] for m in subj_means.values() if cond in m]
            deltas = [
                m[cond] - m["fp32_reference"]
                for m in subj_means.values()
                if cond in m and "fp32_reference" in m
            ]
            per_condition[cond] = {
                "acc": _mean_sd(accs),
                "delta_vs_fp32": _mean_sd(deltas),
            }
        out[mode] = per_condition

    return out


def integrity_checks(
    fold_records: List[Dict[str, Any]],
    quant_rows: List[Dict[str, Any]],
    expected_subjects: Optional[int],
    expected_folds: int,
    dataset: str,
) -> List[Dict[str, Any]]:
    """Assertions that must hold if the harness is sound.

    Two classes of check. The first guards against pointing the collector at
    the wrong artifacts (wrong dataset, mixed seeds). The second guards the
    sweep itself: 32-bit quantisation is lossless, so if ``all@32bit`` differs
    from ``fp32_reference`` the harness is measuring something other than what
    it claims.
    """
    checks: List[Dict[str, Any]] = []

    subjects = sorted({r["subject"] for r in fold_records})

    if fold_records:
        # Guards against collecting a different dataset's Results_ directory,
        # which otherwise produces a plausible-looking but wrong table.
        found = sorted({str(r.get("dataset")) for r in fold_records})
        checks.append(
            {
                "check": f"all folds are dataset {dataset}",
                "passed": found == [dataset],
                "detail": f"found {found}",
            }
        )

        # A missing seed must not pass: artifacts written before global seeding
        # existed record no seed at all, and those runs are not reproducible.
        seeds = sorted({r.get("seed") for r in fold_records}, key=lambda s: (s is None, s))
        checks.append(
            {
                "check": "single recorded seed across all folds",
                "passed": len(seeds) == 1 and seeds[0] is not None,
                "detail": f"seed(s): {seeds}"
                + ("  [None = run predates global seeding]" if None in seeds else ""),
            }
        )

        # The paper states training is full FP32; this keeps that honest.
        amp = sorted({bool(r.get("use_amp")) for r in fold_records})
        checks.append(
            {
                "check": "AMP disabled (FP32 reference is genuine)",
                "passed": amp == [False],
                "detail": f"use_amp values: {amp}",
            }
        )

        classes = sorted({r.get("n_classes") for r in fold_records})
        checks.append(
            {
                "check": "consistent n_classes",
                "passed": len(classes) == 1,
                "detail": f"n_classes: {classes}",
            }
        )

    if expected_subjects is not None:
        checks.append(
            {
                "check": "all subjects present",
                "passed": len(subjects) == expected_subjects,
                "detail": f"found {len(subjects)} of {expected_subjects}: {subjects}",
            }
        )

    short = [
        s
        for s in subjects
        if len([r for r in fold_records if r["subject"] == s]) != expected_folds
    ]
    checks.append(
        {
            "check": f"every subject has {expected_folds} folds",
            "passed": not short,
            "detail": "all complete" if not short else f"incomplete subjects: {short}",
        }
    )

    # 32-bit quantisation must reproduce fp32 exactly (to rounding).
    pairs: List[Tuple[int, int, float, float]] = []
    index = {(r["subject"], r["fold"], r["condition"]): r["test_acc"] for r in quant_rows}
    for (subject, fold, cond), acc in index.items():
        if cond == "all@32bit":
            ref = index.get((subject, fold, "fp32_reference"))
            if ref is not None and abs(acc - ref) > 1e-6:
                pairs.append((subject, fold, acc, ref))
    if any(r["condition"] == "all@32bit" for r in quant_rows):
        checks.append(
            {
                "check": "all@32bit == fp32_reference (32-bit is lossless)",
                "passed": not pairs,
                "detail": "exact match"
                if not pairs
                else f"{len(pairs)} mismatches, e.g. {pairs[:3]}",
            }
        )

    return checks


# ------------------------------------------------------------------ markdown


def render_markdown(bundle: Dict[str, Any]) -> str:
    """Render the bundle as a readable report."""
    p = bundle["provenance"]
    L: List[str] = [
        f"# Results — {p['dataset']}",
        "",
        f"Generated {p['generated_at_utc']} on `{p['hostname']}` "
        f"({p['machine']}), commit `{(p['git_commit'] or 'unknown')[:8]}`"
        f"{' **(dirty tree)**' if p['git_dirty'] else ''}.",
        "",
        f"Python {p['python']}, torch {p.get('torch') or 'unrecorded'}, "
        f"snnTorch {p.get('snntorch') or 'unrecorded'}.",
        "",
        "Regenerated by `collect_results.py` — do not edit by hand.",
        "",
        "## Integrity checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for c in bundle["integrity_checks"]:
        L.append(
            f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |"
        )

    folds = bundle["summary"]["training"]
    L += ["", "## Accuracy by subject (%, mean +/- sd over folds)", "",
          "Log-var: classical decoders on the conventional summary. "
          "Full-ts: the same classifiers on the time series the SNN receives.",
          "",
          "| Subject | Folds | SNN | LDA | SVM | LDA full-ts | SVM full-ts | Features |",
          "|---|---|---|---|---|---|---|---|"]

    def _fmt(d: Dict[str, Any]) -> str:
        if d.get("mean") is None:
            return "--"
        return f"{d['mean']:.1f}" + (f" ± {d['sd']:.1f}" if d.get("sd") else "")

    for subject, e in folds["per_subject"].items():
        L.append(
            f"| {subject} | {e['n_folds']} | {_fmt(e['test_acc_fp32'])} | "
            f"{_fmt(e['test_acc_lda'])} | {_fmt(e['test_acc_svm'])} | "
            f"{_fmt(e['test_acc_lda_fullts'])} | {_fmt(e['test_acc_svm_fullts'])} | "
            f"{_fmt(e['n_features_selected'])} |"
        )
    a = folds["across_subjects"]
    L.append(
        f"| **Mean** | | **{_fmt(a['test_acc_fp32'])}** | "
        f"**{_fmt(a['test_acc_lda'])}** | **{_fmt(a['test_acc_svm'])}** | "
        f"**{_fmt(a['test_acc_lda_fullts'])}** | "
        f"**{_fmt(a['test_acc_svm_fullts'])}** | |"
    )

    sig = folds.get("significance", {})
    if sig and "note" not in sig:
        L += ["", "## Paired comparisons (by subject, SNN vs baseline)", "",
              "| Comparison | n | Mean diff | SNN wins | paired t | Wilcoxon | p<0.05 |",
              "|---|---|---|---|---|---|---|"]
        for name, v in sig.items():
            if "note" in v:
                continue
            L.append(
                f"| {name.replace('_', ' ')} | {v['n_subjects']} | "
                f"{v['mean_difference']:+.2f} | {v['snn_wins']}/{v['n_subjects']} | "
                f"{v['paired_t_p']:.4f} | {v['wilcoxon_p']:.4f} | "
                f"{'yes' if v['significant_at_0.05'] else 'no'} |"
            )
        L += ["", "Paired by subject, not by fold: folds within a subject share "
                  "a recording and are not independent.", ""]
    elif sig:
        L += ["", f"_{sig['note']}_", ""]

    for mode, conds in bundle["summary"]["quantisation"].items():
        L += ["", f"## Quantisation sweep — {mode}", "",
              "| Condition | Accuracy (%) | Delta vs fp32 |", "|---|---|---|"]
        for cond, v in conds.items():
            L.append(f"| `{cond}` | {_fmt(v['acc'])} | {_fmt(v['delta_vs_fp32'])} |")

    if bundle["problems"]:
        L += ["", "## Problems", ""]
        L += [f"- {p}" for p in bundle["problems"]]

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- main


def main() -> None:
    """Collect artifacts into a bundle and optionally a Markdown report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="Results_bnci2015")
    ap.add_argument("--quant-dir", default="Results_quant")
    ap.add_argument("--dataset", default="BNCI2015_001")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--expect-subjects", type=int, default=None,
                    help="fail the completeness check if fewer subjects are found")
    ap.add_argument("--output", default=None, help="JSON bundle path")
    ap.add_argument("--markdown", default=None, help="Markdown report path")
    args = ap.parse_args()

    out_json = Path(args.output or f"results_bundle_{args.dataset}.json")
    out_md = Path(args.markdown or f"results_summary_{args.dataset}.md")

    fold_records, fold_problems = load_folds(Path(args.results_dir))
    quant_rows, quant_problems = load_quant(Path(args.quant_dir), args.dataset)
    problems = fold_problems + quant_problems

    logger.info("Loaded %d fold records, %d sweep rows", len(fold_records), len(quant_rows))
    for p in problems:
        logger.warning("%s", p)

    bundle: Dict[str, Any] = {
        "provenance": build_provenance(args.dataset),
        "summary": {
            "training": summarise_folds(fold_records),
            "quantisation": summarise_quant(quant_rows),
        },
        "integrity_checks": integrity_checks(
            fold_records, quant_rows, args.expect_subjects, args.n_folds, args.dataset
        ),
        "problems": problems,
        "folds": sorted(fold_records, key=lambda r: (r["subject"], r["fold"])),
        "quantisation_rows": sorted(
            quant_rows, key=lambda r: (r["mode"], r["subject"], r["fold"], r["condition"])
        ),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=False)
    logger.info("Wrote %s (%.1f KB)", out_json, out_json.stat().st_size / 1024)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(bundle))
    logger.info("Wrote %s", out_md)

    failed = [c for c in bundle["integrity_checks"] if not c["passed"]]
    for c in failed:
        logger.error("INTEGRITY FAIL — %s: %s", c["check"], c["detail"])
    logger.info(
        "%d/%d integrity checks passed",
        len(bundle["integrity_checks"]) - len(failed),
        len(bundle["integrity_checks"]),
    )


if __name__ == "__main__":
    main()
