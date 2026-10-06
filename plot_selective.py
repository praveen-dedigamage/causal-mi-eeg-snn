#!/usr/bin/env python
"""Two-panel accuracy-against-bit-width figure, built from evidence bundles.

Draws both datasets side by side on a shared vertical scale, and adds the
uniform (all-groups-at-once) curve in black, so the per-group lines can be read
against the whole-pipeline result they are meant to explain.

It reads ``results_bundle_*.json`` rather than the sweep CSVs. The bundle
carries the same rows under ``quantisation_rows`` plus the integrity checks, so
a figure cannot be built from a run that failed them.

Aggregation matches the rest of the analysis: the mean over subject means, not
over raw folds.

Usage
-----
::

    python plot_selective.py --bundles bundle_A.json bundle_B.json \
        --labels "BNCI2015-001" "BNCI2014-002" --out figures/selective_both
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP_STYLE: Dict[str, Tuple[str, str, str]] = {
    "ea":    ("EA whitener",  "#d62728", "o"),
    "csp":   ("EA+CSP",       "#1f77b4", "s"),
    "znorm": ("$z$-norm",     "#2ca02c", "^"),
    "snn_w": ("SNN weights",  "#9467bd", "D"),
    "snn_b": ("SNN biases",   "#ff7f0e", "v"),
    "all":   ("all at once",  "#000000", "*"),
}


def aggregate(rows: List[dict]) -> Tuple[Dict[str, Dict[int, float]], Optional[float]]:
    """Mean accuracy per (group, bits), averaging subject means."""
    nested: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        nested[int(r["subject"])][r["condition"]].append(float(r["test_acc"]))

    per_cond: Dict[str, List[float]] = defaultdict(list)
    for conds in nested.values():
        for cond, vals in conds.items():
            per_cond[cond].append(statistics.mean(vals))

    curves: Dict[str, Dict[int, float]] = defaultdict(dict)
    fp32: Optional[float] = None
    for cond, vals in per_cond.items():
        mean = statistics.mean(vals)
        if cond == "fp32_reference":
            fp32 = mean
            continue
        if "@" not in cond:
            continue
        group, bit_str = cond.split("@", 1)
        curves[group][int(bit_str.replace("bit", ""))] = mean
    return dict(curves), fp32


def load_bundle(path: Path) -> Tuple[Dict[str, Dict[int, float]], Optional[float]]:
    """Read a bundle and merge its per-group and uniform sweeps."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    failed = [c["check"] for c in bundle.get("integrity_checks", []) if not c["passed"]]
    if failed:
        raise SystemExit(f"{path.name}: integrity checks failed: {failed}")
    rows = bundle["quantisation_rows"]
    curves, fp32 = aggregate([r for r in rows if r["mode"].startswith("per-group")])
    uni, _ = aggregate([r for r in rows if r["mode"].startswith("uniform")])
    if "all" in uni:
        curves["all"] = uni["all"]
    return curves, fp32


def draw_panel(ax, curves, fp32, title: str, chance: float, show_ylabel: bool,
               fs: float = 1.0) -> None:
    """Draw one dataset onto an axis.

    ``fs`` scales every font size, so a figure drawn at the width it will be
    printed at keeps its labels legible.
    """
    if fp32 is not None:
        ax.axhline(fp32, color="0.35", lw=0.9, ls="--", zorder=1)
        ax.text(1.0, fp32 + 0.8, f"full precision ({fp32:.1f}%)",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6 * fs, color="0.35")
    ax.axhline(chance, color="0.65", lw=0.9, ls=":", zorder=1)
    # Above the line, not below: below it sits outside the y-limit and the
    # axis spine clips the descenders.
    ax.text(0.01, chance + 0.4, "chance", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6 * fs, color="0.55")

    all_bits = sorted({b for c in curves.values() for b in c})
    xpos = {b: i for i, b in enumerate(all_bits)}
    for group, (label, colour, marker) in GROUP_STYLE.items():
        if group not in curves:
            continue
        pts = sorted(curves[group].items())
        ax.plot([xpos[b] for b, _ in pts], [a for _, a in pts],
                marker=marker, ms=3.4, lw=1.2, color=colour, label=label,
                zorder=4 if group == "all" else 3)

    ax.set_xticks(list(xpos.values()))
    # Unit on the ticks rather than in an axis title: it reads the same and
    # saves a whole text row of figure height.
    ax.set_xticklabels([f"{b} bits" for b in all_bits], fontsize=7 * fs)
    ax.set_xlim(-0.25, len(all_bits) - 0.75)
    if show_ylabel:
        ax.set_ylabel("Accuracy (%)", fontsize=7 * fs)
    ax.set_title(title, fontsize=5.5 * fs)
    ax.tick_params(labelsize=7 * fs)
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chance", type=float, default=50.0)
    ap.add_argument("--ylim", nargs=2, type=float, default=None)
    # Draw at the width the figure is printed at, so no rescaling shrinks the
    # labels: 	extwidth in the paper is 347.1pt = 4.80in.
    ap.add_argument("--figsize", nargs=2, type=float, default=(5.6, 2.05))
    ap.add_argument("--fontscale", type=float, default=1.0)
    args = ap.parse_args()
    if len(args.bundles) != len(args.labels):
        raise SystemExit("--bundles and --labels must be the same length")

    panels = [load_bundle(Path(b)) for b in args.bundles]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=tuple(args.figsize), sharey=True)
    if n == 1:
        axes = [axes]

    for i, ((curves, fp32), label, ax) in enumerate(zip(panels, args.labels, axes)):
        draw_panel(ax, curves, fp32, label, args.chance, show_ylabel=(i == 0),
                   fs=args.fontscale)
        print(f"{label}: groups {sorted(curves)}  fp32 {fp32:.2f}%")

    lo = min(min(c.values()) for cu, _ in panels for c in cu.values())
    hi = max([f for _, f in panels if f] + [max(c.values()) for cu, _ in panels for c in cu.values()])
    # Keep the chance line inside the axes: it is the reference the reader
    # judges the 4-bit collapse against, and a label drawn below the axis
    # collides with the tick labels.
    axes[0].set_ylim(*(args.ylim if args.ylim else (min(lo, args.chance) - 3, hi + 5)))

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=5.2 * args.fontscale,
                    frameon=False, loc="lower right", ncol=2,
                    # lifted clear of the chance line at the axes floor
                    bbox_to_anchor=(1.0, 0.10),
                    handlelength=1.2, handletextpad=0.4,
                    columnspacing=0.8, labelspacing=0.2, borderaxespad=0.2)

    # Tight outer margins so the panels claim the full text width, with a
    # wide gutter so the two datasets read as separate plots.
    fig.tight_layout(pad=0.2, w_pad=1.2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
        print(f"wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
