#!/bin/bash
# ============================================================================
#  Roihu preflight — verify the environment on a real GPU node before
#  committing 60 array tasks to the queue.
#
#      sbatch roihu/preflight.sh
#
#  Uses the 15-minute `gputest` partition, so a wrong module name or an x86/ARM
#  mismatch surfaces in minutes instead of after a long queue wait followed by
#  60 simultaneous failures.
# ============================================================================
#SBATCH --job-name=fbcsp_preflight
# Account comes from SBATCH_ACCOUNT in the environment; add
#     export SBATCH_ACCOUNT=project_XXXXXXX
# to ~/.bashrc. Deliberately not an #SBATCH line: a committed
# project id has to be re-substituted after every pull, and that
# conflicted on four separate occasions.
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

PYTORCH_MODULE="${PYTORCH_MODULE:-python-pytorch/2.10}"
module purge
module load "${PYTORCH_MODULE}"

echo "=== node ==="
hostname; uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== python / torch ==="
python -u - <<'PY'
import platform, sys
print("machine   :", platform.machine(), "(expect aarch64 on Roihu GPU nodes)")
print("python    :", sys.version.split()[0])
import torch
print("torch     :", torch.__version__)
print("cuda      :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not visible inside the job"
print("device    :", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
print("matmul    :", tuple((x @ x).shape), "ok")
PY

echo
echo "=== project imports ==="
python -u - <<'PY'
from fbcsp_snn import DEVICE, set_global_seed
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.ptq import (quantize_ea_whiteners, quantize_znorm,
                           quantize_model_full, quantize_csp_filters)
from fbcsp_snn.training import evaluate_model
import snntorch, moabb, mne
print("DEVICE    :", DEVICE)
print("snntorch  :", snntorch.__version__)
print("moabb     :", moabb.__version__)
print("mne       :", mne.__version__)
set_global_seed(42)
m = SNNClassifier(n_input=22, n_hidden=64, n_classes=2,
                  population_per_class=20, beta=0.95, dropout_prob=0.0)
print("model ok  :", sum(p.numel() for p in m.parameters() if p.requires_grad), "params")
PY

echo
echo "=== dataset cache (must be pre-fetched by 00_setup_roihu.sh) ==="
python -u - <<'PY'
import warnings, logging
warnings.filterwarnings("ignore")
for n in ("moabb", "mne"): logging.getLogger(n).setLevel(logging.ERROR)
from fbcsp_snn.config import Config
from fbcsp_snn.pipeline import _load_raw
cfg = Config(); cfg.source="moabb"; cfg.moabb_dataset="BNCI2015_001"
cfg.subject_id=1; cfg.n_classes=2
Xtr, ytr, Xte, yte = _load_raw(cfg)
print("S1 train :", Xtr.shape, " test:", Xte.shape)
print("If this triggered a download, run 00_setup_roihu.sh first --")
print("60 array tasks racing on the same cache will fail.")
PY

echo
echo "=== PREFLIGHT OK — safe to sbatch 01_train_array.sh ==="
