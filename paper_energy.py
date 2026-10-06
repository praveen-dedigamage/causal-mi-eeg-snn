"""Energy estimate of the inference pipeline (Sect. 4.3 of the paper).

Recomputes the paper's per-trial energy and average power from the evidence
bundles. The measured inputs are taken from the bundles as the pipeline wrote
them -- the number of selected features and the input and hidden spike counts per
trial, averaged over all folds -- and combined with the published per-stage
figures the paper cites:

* Gm-C filter bank: 128.7 nW per filter (Gallegos-Ramirez et al., 2014), one
  filter per band per channel, always on for the whole trial window.
* Spike encoder: 800 nW per encoded feature (Narayanan et al., 2023), applied
  only to the features MIBIF keeps.
* LIF classifier: 23.6 pJ per synaptic operation (Davies et al., 2018), with
  n_in x H operations for input spikes and n_hid x C*P for hidden spikes.
* Spatial projection: 15 fJ per multiply-accumulate (a representative
  in-memory-compute figure); it is negligible and is reported for completeness.

Usage::

    python paper_energy.py --bundles evidence/bundle_2band_BNCI2015_001.json \\
        evidence/bundle_2band_BNCI2014_002.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

GMC_W_PER_FILTER = 128.7e-9     # Gallegos-Ramirez et al., IEEE MWSCAS 2014
ENC_W_PER_FEATURE = 0.8e-6      # Narayanan et al., IEEE BioCAS 2023 (upper bound)
SYNOP_J = 23.6e-12              # Davies et al., IEEE Micro 2018
CROSSBAR_J_PER_MAC = 15e-15     # representative in-memory-compute figure
TRIAL_S = 5.0                   # analysis window (s)
N_SAMPLES = 2561                # samples per trial window at 512 Hz
N_BANDS = 2                     # K, the two-band filter bank
CSP_FILTERS_PER_PAIR = 8        # 2m with m = 4
HIDDEN = 64                     # H
POP_PER_CLASS = 20              # P
N_CLASSES = 2                   # C
N_CHANNELS = {"BNCI2015_001": 13, "BNCI2014_002": 15}


def fold_mean(bundle: Dict, field: str) -> float:
    """Mean of one per-fold field over every fold in a bundle."""
    return statistics.mean(f[field] for f in bundle["folds"] if field in f)


def energy(bundle: Dict) -> Dict[str, float]:
    """Per-trial energy (uJ) of each stage, the total, and the average power (uW).

    Parameters
    ----------
    bundle : Dict
        An evidence bundle written by ``collect_results.py``.

    Returns
    -------
    Dict[str, float]
        Selected features, synaptic operations, per-stage energies, total energy
        and average power over the trial window.
    """
    dataset = bundle["provenance"]["dataset"]
    n_ch = N_CHANNELS[dataset]
    n_features = fold_mean(bundle, "n_features_selected")
    sops = (fold_mean(bundle, "mean_input_events_per_trial") * HIDDEN
            + fold_mean(bundle, "mean_hidden_events_per_trial") * N_CLASSES * POP_PER_CLASS)
    stages = {
        "filter_bank_uJ": GMC_W_PER_FILTER * N_BANDS * n_ch * TRIAL_S * 1e6,
        # the hardware needs a whole number of encoders: the mean feature count, rounded
        "encoder_uJ": ENC_W_PER_FEATURE * round(n_features) * TRIAL_S * 1e6,
        "classifier_uJ": sops * SYNOP_J * 1e6,
        "projection_uJ": N_SAMPLES * N_BANDS * n_ch * CSP_FILTERS_PER_PAIR * CROSSBAR_J_PER_MAC * 1e6,
    }
    total = sum(stages.values())
    return {"dataset": dataset, "channels": n_ch, "features": n_features, "sops_million": sops / 1e6,
            **stages, "total_uJ": total, "power_uW": total / TRIAL_S}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundles", nargs="+", required=True, help="evidence bundle JSON files")
    args = ap.parse_args()
    rows: List[Dict] = [energy(json.loads(Path(p).read_text(encoding="utf-8"))) for p in args.bundles]

    print("%-34s" % "" + "".join("%16s" % r["dataset"] for r in rows))
    for label, key, fmt in (("EEG channels", "channels", "%16d"),
                            ("selected features (mean)", "features", "%16.1f"),
                            ("synaptic operations (million)", "sops_million", "%16.2f"),
                            ("Gm-C filter bank (uJ)", "filter_bank_uJ", "%16.1f"),
                            ("spike encoder (uJ)", "encoder_uJ", "%16.0f"),
                            ("LIF classifier (uJ)", "classifier_uJ", "%16.1f"),
                            ("spatial projection (uJ)", "projection_uJ", "%16.4f"),
                            ("total per trial (uJ)", "total_uJ", "%16.0f"),
                            ("average power (uW)", "power_uW", "%16.0f")):
        print("%-34s" % label + "".join(fmt % r[key] for r in rows))


if __name__ == "__main__":
    main()
