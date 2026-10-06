#!/bin/bash
# ============================================================================
#  Roihu — whole-pipeline bit-width quantisation sweep
#
#  Run AFTER the training array has finished. To chain automatically:
#      TRAIN_ID=$(sbatch --parsable roihu/01_train_array.sh)
#      sbatch --dependency=afterok:${TRAIN_ID} roihu/02_quant_sweep.sh
#
#  Or standalone once artifacts exist:
#      sbatch roihu/02_quant_sweep.sh
#
#  This is post-training quantisation: it loads saved fold artifacts and
#  re-evaluates, with no retraining. One subject at a time, so it is far
#  cheaper than training — but it is not free, because quantising the front
#  end (EA whiteners / CSP / z-norm) changes the spikes, so the encoder must
#  be re-run for every bit-width. The band-pass filtering is cached per fold,
#  since filter coefficients are deliberately excluded from the sweep.
# ============================================================================
#SBATCH --job-name=fbcsp_snn_quant
# Account comes from SBATCH_ACCOUNT in the environment; add
#     export SBATCH_ACCOUNT=project_XXXXXXX
# to ~/.bashrc. Deliberately not an #SBATCH line: a committed
# project id has to be re-substituted after every pull, and that
# conflicted on four separate occasions.
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/quant_%j.out
#SBATCH --error=logs/quant_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# DATASET / RESULTS_DIR / OUT_DIR / N_FOLDS / MNE_DATA / PYTORCH_MODULE.
#     DATASET=Cho2017 SUBJECTS="$(seq 1 52)" sbatch roihu/02_quant_sweep.sh
source roihu/env.sh

case "${DATASET}" in
    Cho2017)      _default_subjects="$(seq 1 52 | tr '\n' ' ')" ;;
    BNCI2014_002) _default_subjects="$(seq 1 14 | tr '\n' ' ')" ;;
    *)            _default_subjects="1 2 3 4 5 6 7 8 9 10 11 12" ;;
esac
SUBJECTS="${SUBJECTS:-${_default_subjects}}"
BITS="${BITS:-4 6 8 16 32}"

# FUSE=1 composes the EA whitener into the spatial filters before quantising,
# measuring a front end that stores one matrix per band instead of two. Exact
# in FP32, so the reference is unchanged. Output CSVs gain a '_fused' suffix,
# so a fused run cannot overwrite a split one.
FUSE_FLAG=""
if [ "${FUSE:-0}" = "1" ]; then
    FUSE_FLAG="--fuse"
fi

module purge
module load "${PYTORCH_MODULE}"

echo "=============================================="
echo "  Quantisation sweep — ${DATASET}"
echo "  results  : ${RESULTS_DIR}"
echo "  subjects : ${SUBJECTS}"
echo "  bits     : ${BITS}"
echo "  node     : $(hostname)  arch: $(uname -m)"
echo "=============================================="

# ---- uniform: every parameter group at the same width ----------------------
# The headline result. A row reported at N bits has no full-precision
# parameter anywhere in the pipeline: EA whiteners, CSP filters, z-norm
# mean/std, SNN weights AND SNN biases are all quantised together.
srun python -u quantize_sweep.py \
    --source moabb \
    --moabb-dataset "${DATASET}" \
    --results-dir "${RESULTS_DIR}" \
    --subjects ${SUBJECTS} \
    --n-folds "${N_FOLDS}" \
    --bits ${BITS} \
    --mode uniform ${FUSE_FLAG} \
    --output-dir "${OUT_DIR}"

# ---- per-group: which stage degrades first ---------------------------------
# Quantises one group at a time, everything else left at FP32. Local
# measurement predicts the EA whitener fails first: on a real fold it took
# 20.4 % relative RMS error at 4 bits vs 15.2 % for CSP, because it is built
# from eigenvalue**-0.5 and so spans ~100x dynamic range (vs ~11x for CSP).
# The classifier, by contrast, was insensitive down to 4 bits.
srun python -u quantize_sweep.py \
    --source moabb \
    --moabb-dataset "${DATASET}" \
    --results-dir "${RESULTS_DIR}" \
    --subjects ${SUBJECTS} \
    --n-folds "${N_FOLDS}" \
    --bits ${BITS} \
    --mode per-group ${FUSE_FLAG} \
    --output-dir "${OUT_DIR}"

echo "finished: $(date)"
echo "CSVs in ${OUT_DIR}/"
