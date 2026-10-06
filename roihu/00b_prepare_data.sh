#!/bin/bash
# ============================================================================
#  Roihu setup, part 2 — dataset download + epoching, as a BATCH JOB
#
#      sbatch roihu/00b_prepare_data.sh
#
#  Why this is not done on the login node
#  -------------------------------------
#  It looks like "just a download", but MOABB's _load_raw() also epochs the
#  data: 12 subjects x 400 trials x 13 channels x 2561 samples. That exceeds
#  the login-node budget CSC enforces ("one-core jobs that finish in minutes
#  and require less than 1 GiB of memory", terminated without warning
#  otherwise), so it belongs in a job.
#
#  Why this must be done ONCE, before the training array
#  ----------------------------------------------------
#  The training array runs 5 folds concurrently per subject. Without a warm
#  cache they would all try to populate the same ~/mne_data directory at the
#  same time and race each other. This job fills the cache serially first.
#
#  Runs on gputest (15 min limit) rather than a CPU partition for one
#  non-obvious reason: the Python environment was built with --user on the ARM
#  GPU login node, so it is aarch64 and only valid on GPU nodes. Roihu's CPU
#  nodes are x86 and would not be able to import it.
#
#  If this job fails with a network/DNS error, the compute nodes have no
#  outbound internet. In that case run the same Python on the LOGIN node
#  instead, but with --download-only, which skips the expensive epoching and
#  stays within the login-node budget.
# ============================================================================
#SBATCH --job-name=fbcsp_prepdata
# Account comes from SBATCH_ACCOUNT in the environment; add
#     export SBATCH_ACCOUNT=project_XXXXXXX
# to ~/.bashrc. Deliberately not an #SBATCH line: a committed
# project id has to be re-substituted after every pull, and that
# conflicted on four separate occasions.
# gpumedium, not gputest: gputest caps at 15 minutes, which suits
# BNCI2015-001 (12 subjects at 13 channels, ~2.5 min) but not Cho2017
# (52 subjects at 64 channels, hours). Override --time at submit time.
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepdata_%j.out
#SBATCH --error=logs/prepdata_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# DATASET / MNE_DATA / PYTORCH_MODULE. For the 52-subject, 64-channel set:
#     DATASET=Cho2017 sbatch --time=03:00:00 roihu/00b_prepare_data.sh
source roihu/env.sh

case "${DATASET}" in
    Cho2017) N_SUBJECTS="${N_SUBJECTS:-52}" ;;
    *)       N_SUBJECTS="${N_SUBJECTS:-12}" ;;
esac

module purge
module load "${PYTORCH_MODULE}"

echo "=== node: $(hostname)  arch: $(uname -m) ==="
echo "=== caching ${DATASET}, ${N_SUBJECTS} subjects -> ${MNE_DATA} ==="
df -h "${MNE_DATA}" | tail -1

srun python -u - <<PY
import warnings, logging, time
warnings.filterwarnings("ignore")
for n in ("moabb", "mne"):
    logging.getLogger(n).setLevel(logging.ERROR)

from fbcsp_snn.config import Config
from fbcsp_snn.pipeline import _load_raw

t0 = time.time()
for sid in range(1, ${N_SUBJECTS} + 1):
    cfg = Config()
    cfg.source = "moabb"
    cfg.moabb_dataset = "${DATASET}"
    cfg.subject_id = sid
    cfg.n_classes = 2
    Xtr, ytr, Xte, yte = _load_raw(cfg)
    print(f"  S{sid:<3} train {Xtr.shape}  test {Xte.shape}  "
          f"[{time.time()-t0:.0f}s elapsed]", flush=True)

print(f"\nall ${N_SUBJECTS} subjects cached in {time.time()-t0:.0f}s")
PY

echo "=== cache size ==="
du -sh "${MNE_DATA}"

echo "=== done: $(date) ==="
echo "Next:  sbatch roihu/preflight.sh"
