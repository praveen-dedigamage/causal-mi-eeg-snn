#!/bin/bash
# ============================================================================
#  Roihu — BNCI2015-001 training array
#  ONE TASK PER SUBJECT (12 tasks); the subject's 5 folds run CONCURRENTLY
#  on the single reserved GH200.
#
#      sbatch roihu/01_train_array.sh
#
#  Why fold-packing instead of 60 one-fold tasks
#  ---------------------------------------------
#  This workload leaves the GPU mostly idle: the LIF loop is sequential over
#  T=2561 timesteps on a 4k-parameter network, so each kernel is tiny and the
#  GPU sat at 17-21 % utilisation locally. Measured on an RTX 3060:
#
#      1 process alone      8.04 s/iter            0.124 iter/s
#      3 processes at once  9.53 s/iter each       0.315 iter/s  (2.53x)
#
#  i.e. three concurrent trainings each slow down only ~18 % while total
#  throughput rises 2.5x. Packing a subject's 5 folds onto one GPU therefore
#  turns ~11.5 h of serial work into roughly 3-4 h of wall clock, and bills
#  12 GPU allocations instead of 60.
#
#  Memory is not the constraint: each process used ~1.5 GiB, and a reserved
#  GH200 provides 95 GiB HBM3 (+122 GiB LPDDR5).
#
#  Partitions verified against CSC docs (batch-job-partitions, Aug 2026):
#      gputest 15 min | gpumedium 36 h (used here) | gpularge whole nodes
#  GPUs are requested as --gres=gpu:gh200:N.
# ============================================================================
#SBATCH --job-name=fbcsp_snn_train
# Account comes from SBATCH_ACCOUNT in the environment; add
#     export SBATCH_ACCOUNT=project_XXXXXXX
# to ~/.bashrc. Deliberately not an #SBATCH line: a committed
# project id has to be re-substituted after every pull, and that
# conflicted on four separate occasions.
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=40                 # 5 concurrent folds; 72 cores available per GPU
#SBATCH --mem=120G
#SBATCH --time=08:00:00                    # ~3-4 h expected; margin for the slowest subject
#SBATCH --array=1-12                       # one task per subject
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# DATASET / RESULTS_DIR / N_FOLDS / SEED / MNE_DATA / PYTORCH_MODULE.
# Override at submit time, e.g. for the 52-subject Cho2017 run:
#     DATASET=Cho2017 sbatch --array=1-52 --time=12:00:00 roihu/01_train_array.sh
# (--array and --time must be sbatch flags: #SBATCH lines cannot read variables.)
source roihu/env.sh

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}

module purge
module load "${PYTORCH_MODULE}"

# Keep each fold's thread pool small: 5 processes share the CPUs, and BLAS
# over-subscription would slow all of them.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-40} / N_FOLDS ))
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "=============================================="
echo "  Roihu — subject ${SUBJECT_ID}, ${N_FOLDS} folds concurrently"
echo "  node    : $(hostname)   arch: $(uname -m)"
echo "  threads : ${OMP_NUM_THREADS} per fold"
echo "  started : $(date)"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Launch all folds on the same GPU and wait for them together.
pids=()
for FOLD_IDX in $(seq 0 $(( N_FOLDS - 1 ))); do
    python -u main.py train \
        --source moabb \
        --moabb-dataset "${DATASET}" \
        --subject-id "${SUBJECT_ID}" \
        --fold "${FOLD_IDX}" \
        --n-folds "${N_FOLDS}" \
        --freq-bands "${FREQ_BANDS}" \
        --csp-components-per-band 8 \
        --hidden-neurons 64 \
        --population-per-class 20 \
        --beta 0.95 \
        --dropout-prob 0.5 \
        --lr 1e-3 \
        --weight-decay 0.1 \
        --epochs 1000 \
        --early-stopping-patience 100 \
        --early-stopping-warmup 100 \
        --spiking-prob 0.7 \
        --feature-selection-method mibif \
        --mi-fraction 0.1 \
        --seed "${SEED}" \
        --results-dir "${RESULTS_DIR}" \
        > "logs/s${SUBJECT_ID}_f${FOLD_IDX}.log" 2>&1 &
    pids+=($!)
    echo "  launched fold ${FOLD_IDX}  pid ${pids[-1]}"
    sleep 5     # stagger startup so the folds do not race on the data cache
done

# Fail the task if any fold fails, rather than exiting 0 on a partial result.
status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "!! fold ${i} FAILED (see logs/s${SUBJECT_ID}_f${i}.log)"
        status=1
    fi
done

echo "finished: $(date)  exit=${status}"
exit ${status}
