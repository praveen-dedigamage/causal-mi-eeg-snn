# causal-mi-eeg-snn

Code and evidence for the paper

> Ruchira Dedigamage, Marja-Leena Linne and Tuomo Mäki-Marttunen.
> **An Energy-Efficient Machine Learning Pipeline for Causal Motor Imagery EEG
> Decoding with Spiking Neural Networks.** In *Complex Networks & Their
> Applications XV* (COMPLEX NETWORKS 2026), Studies in Computational
> Intelligence, Springer, 2026 (to appear).

The pipeline decodes motor imagery from EEG causally, one sample at a time, so
that every inference stage can map onto event-driven, low-power hardware:

1. a causal two-band Butterworth filter bank (6–15 and 12–32 Hz),
2. Euclidean alignment and pairwise dual-end common spatial patterns (CSP),
3. z-normalisation with statistics from the training fold,
4. an adaptive-threshold delta encoder that turns each feature into a spike train,
5. mutual-information feature selection (MIBIF), and
6. a two-layer leaky integrate-and-fire (LIF) network trained with surrogate
   gradients and a Van Rossum distance loss, read out by population
   winner-take-all.

Everything that is fitted (alignment, spatial filters, normalisation, feature
selection and the network) is fitted inside each cross-validation training fold.

## Contents

| Path | What it is |
|---|---|
| `fbcsp_snn/` | The pipeline: preprocessing, encoding, feature selection, model, training, evaluation and quantisation |
| `main.py` | Command-line entry point (`train`, `infer`, `aggregate`) |
| `roihu/` | SLURM job scripts for CSC's Roihu cluster; setup notes in `roihu/SETUP.md` |
| `quantize_sweep.py`, `fusion_experiment.py` | Post-training bit-width sweeps of Fig. 2 (`fusion_experiment.py` folds the alignment into the spatial filters) |
| `run_fair_baseline.py`, `aggregate_fair_baseline.py` | Classical baselines given the same time series as the network (Table 2) |
| `collect_results.py` | Collects every fold into one evidence bundle and runs integrity checks |
| `plot_selective.py` | Fig. 2 from the evidence bundles |
| `paper_energy.py` | Energy and power estimate of Sect. 4.3 from the evidence bundles |
| `evidence/` | The evidence bundles behind the paper's results (see below) |
| `tests/` | Component checks of the pipeline on the four-class BNCI2014-001 dataset |

## Installation

Python 3.9 or newer:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The paper's runs used Python 3.9 with CSC's `python-pytorch/2.10` module on
NVIDIA GH200 nodes. The code has also been run on CPU with Python 3.13,
PyTorch 2.13, snnTorch 1.0, MOABB 1.5 and MNE 1.12.

## Data

Both datasets are public and are downloaded by [MOABB](https://moabb.neurotechx.com)
on first use:

| Dataset | Subjects | Channels | Sampling rate | Task |
|---|---|---|---|---|
| BNCI2015-001 | 12 | 13 | 512 Hz | right hand vs. both feet |
| BNCI2014-002 | 14 | 15 | 512 Hz | right hand vs. both feet |

They are cached under `$MNE_DATA` (by default `~/mne_data`).

## Reproducing the paper

### From the evidence bundles (no GPU needed)

`evidence/` holds one bundle per dataset from the runs reported in the paper:
every fold's test accuracy for the network and the classical baselines, the
number of selected features, spike counts per layer, the bit-width sweeps, a set
of integrity checks and the run's provenance. The `summary_*.md` files are
readable versions of the same.

Fig. 2:

```bash
python plot_selective.py \
    --bundles evidence/bundle_2band_BNCI2015_001.json evidence/bundle_2band_BNCI2014_002.json \
    --labels "BNCI2015-001" "BNCI2014-002" --figsize 4.8 1.3 --fontscale 1.18 \
    --out figures/selective_both
```

The energy and power estimates of Sect. 4.3:

```bash
python paper_energy.py \
    --bundles evidence/bundle_2band_BNCI2015_001.json evidence/bundle_2band_BNCI2014_002.json
```

### Full re-run on a SLURM cluster

Set up once as described in `roihu/SETUP.md`, then prepare both datasets:

```bash
sbatch roihu/00b_prepare_data.sh
DATASET=BNCI2014_002 sbatch roihu/00b_prepare_data.sh
```

The paper uses the two-band filter bank, which is **not** the scripts' default
(an earlier six-band bank), and writes to separate folders. Run exactly:

```bash
# BNCI2015-001: 12 subjects x 5 folds
FREQ_BANDS="[(6,15),(12,32)]" RESULTS_DIR=Results_bnci2015_2band \
    sbatch --array=1-12 roihu/01_train_array.sh

# BNCI2014-002: 14 subjects x 5 folds
DATASET=BNCI2014_002 FREQ_BANDS="[(6,15),(12,32)]" RESULTS_DIR=Results_bnci2014_002_2band \
    sbatch --array=1-14 roihu/01_train_array.sh
```

When training has finished, run the bit-width sweeps (with the alignment folded
into the spatial filters, `FUSE=1`) and the baselines:

```bash
DATASET=BNCI2015_001 RESULTS_DIR=Results_bnci2015_2band OUT_DIR=Results_quant_2band FUSE=1 \
    sbatch roihu/02_quant_sweep.sh
DATASET=BNCI2014_002 RESULTS_DIR=Results_bnci2014_002_2band OUT_DIR=Results_quant_2band FUSE=1 \
    sbatch roihu/02_quant_sweep.sh

DATASET=BNCI2015_001 RESULTS_DIR=Results_bnci2015_2band sbatch --array=1-12 roihu/04_fair_baseline.sh
DATASET=BNCI2014_002 RESULTS_DIR=Results_bnci2014_002_2band sbatch --array=1-14 roihu/04_fair_baseline.sh
```

Then collect each dataset into a bundle, and use it as above:

```bash
python collect_results.py --results-dir Results_bnci2015_2band --quant-dir Results_quant_2band \
    --dataset BNCI2015_001 --n-folds 5 --expect-subjects 12 \
    --output bundle_2band_BNCI2015_001.json --markdown summary_2band_BNCI2015_001.md
python collect_results.py --results-dir Results_bnci2014_002_2band --quant-dir Results_quant_2band \
    --dataset BNCI2014_002 --n-folds 5 --expect-subjects 14 \
    --output bundle_2band_BNCI2014_002.json --markdown summary_2band_BNCI2014_002.md
```

### A single fold on any machine

```bash
python main.py train --source moabb --moabb-dataset BNCI2015_001 --subject-id 1 \
    --fold 0 --n-folds 5 --freq-bands "[(6,15),(12,32)]" \
    --csp-components-per-band 8 --hidden-neurons 64 --population-per-class 20 \
    --beta 0.95 --dropout-prob 0.5 --lr 1e-3 --weight-decay 0.1 --epochs 1000 \
    --early-stopping-patience 100 --early-stopping-warmup 100 --spiking-prob 0.7 \
    --feature-selection-method mibif --mi-fraction 0.1 --seed 42 \
    --results-dir Results_bnci2015_2band
```

These are the settings of the paper's runs (`roihu/01_train_array.sh`). A fold
takes tens of minutes on a single CPU core.

## Reproducibility

* Cross-validation splits are deterministic, and every random number generator
  is seeded with 42.
* Training the network still draws random numbers (dropout masks, the target
  spike trains of the Van Rossum loss and the shuffling of batches), and
  PyTorch's CPU and GPU generators give different streams from the same seed. A
  run on another device therefore gives different per-fold accuracies.
* The bundles record their provenance. They were produced on 23 August 2026
  from commit `0c0bba3` of the development repository, from a working tree with
  local changes (`git_dirty`). Apart from comments, the pipeline, training,
  quantisation and baseline code here is identical to that commit; the figure
  script was finalised afterwards, and the energy estimate is packaged here as
  `paper_energy.py`.

## Citation

If you use this code, please cite the paper; `CITATION.cff` carries the same
information.

```bibtex
@inproceedings{dedigamage2026energy,
  author    = {Dedigamage, Ruchira and Linne, Marja-Leena and M{\"a}ki-Marttunen, Tuomo},
  title     = {An Energy-Efficient Machine Learning Pipeline for Causal Motor
               Imagery {EEG} Decoding with Spiking Neural Networks},
  booktitle = {Complex Networks \& Their Applications XV},
  series    = {Studies in Computational Intelligence},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

## License

MIT; see `LICENSE`.
