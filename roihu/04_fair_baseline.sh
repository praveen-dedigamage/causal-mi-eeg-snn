#!/bin/bash
# ============================================================================
#  Roihu — fairness-controlled baseline
#
#      sbatch roihu/04_fair_baseline.sh                    # 12 subjects
#      DATASET=Cho2017 sbatch --array=1-52 roihu/04_fair_baseline.sh
#
#  Refits LDA and SVM on the SAME full z-normalised time series the spike
#  encoder receives, rather than on the log-variance summary classical FBCSP
#  pipelines use. This is what makes the comparison in the paper's accuracy
#  table information-matched: if the SNN's performance came from being handed
#  more data rather than from temporal processing, these baselines should
#  close the gap.
#
#  Reuses each fold's saved CSP filters and z-norm statistics, so nothing is
#  refitted on the front end and no SNN is retrained. Writes
#  fair_baseline_results.json into each fold directory -- a new file that no
#  other job reads or writes, so this is safe to run alongside the sweep or
#  the fusion experiment.
#
#  No GPU is used. It runs on a GPU partition regardless because the --user
#  environment was built on the ARM GPU login node and cannot be imported on
#  Roihu's x86 CPU nodes.
#
#  Afterwards, aggregate (light enough for the login node):
#      python aggregate_fair_baseline.py --results-dir Results_bnci2015 \
#          --subjects 1 2 3 4 5 6 7 8 9 10 11 12 --n-folds 5
# ============================================================================
#SBATCH --job-name=fbcsp_fair
# Account comes from SBATCH_ACCOUNT in the environment; add
#     export SBATCH_ACCOUNT=project_XXXXXXX
# to ~/.bashrc. Deliberately not an #SBATCH line: a committed
# project id has to be re-substituted after every pull, and that
# conflicted on four separate occasions.
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-12                       # one task per subject
#SBATCH --output=logs/fair_%A_%a.out
#SBATCH --error=logs/fair_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# DATASET / RESULTS_DIR / N_FOLDS / MNE_DATA / PYTORCH_MODULE
source roihu/env.sh

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}

module purge
module load "${PYTORCH_MODULE}"

echo "=============================================="
echo "  Fairness-controlled baseline — ${DATASET}"
echo "  subject  : ${SUBJECT_ID}"
echo "  results  : ${RESULTS_DIR}"
echo "  node     : $(hostname)  arch: $(uname -m)"
echo "=============================================="

# Folds run serially: each fits LDA and SVM on a wide, short design matrix
# (features x samples flattened), which is memory-hungry rather than
# parallelisable, so running them concurrently would contend for RAM without
# improving throughput.
status=0
for FOLD_IDX in $(seq 0 $(( N_FOLDS - 1 ))); do
    echo "--- subject ${SUBJECT_ID} fold ${FOLD_IDX} ---"
    if ! srun python -u run_fair_baseline.py \
            --results-dir "${RESULTS_DIR}" \
            --moabb-dataset "${DATASET}" \
            --subject-id "${SUBJECT_ID}" \
            --fold "${FOLD_IDX}" \
            --n-folds "${N_FOLDS}"; then
        echo "!! fold ${FOLD_IDX} FAILED"
        status=1
    fi
done

echo "finished: $(date)  exit=${status}"
exit ${status}
