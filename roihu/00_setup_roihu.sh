#!/bin/bash
# ============================================================================
#  Roihu setup, part 1 — LOGIN NODE (light work only)
#
#      ssh <user>@roihu-gpu.csc.fi        (or the web terminal)
#      bash roihu/00_setup_roihu.sh
#
#  CSC's usage policy limits login nodes to "one-core jobs that finish in
#  minutes and require less than 1 GiB of memory at maximum", and states that
#  programs breaking that "will be terminated without warning". Everything
#  here stays inside that budget:
#
#      arch check        trivial
#      module load       trivial
#      pip install       explicitly permitted login-node work
#
#  The dataset download and epoching is NOT done here -- it exceeds 1 GiB and
#  takes longer than a couple of minutes. It runs as a batch job instead:
#      sbatch roihu/00b_prepare_data.sh
#
#  IMPORTANT — architecture split:
#  Roihu GPU nodes are GH200 Grace Hopper, so their CPU side is ARM (aarch64).
#  CSC states software built on GPU nodes only works on GPU nodes, "and this
#  also applies to Python environments". Build here, on roihu-gpu, never on
#  roihu-cpu, and never copy an x86 environment in.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== architecture check ==="
ARCH="$(uname -m)"
echo "uname -m = ${ARCH}"
if [[ "${ARCH}" != "aarch64" ]]; then
    echo "!! Expected aarch64 (Roihu GPU login node)."
    echo "!! You appear to be on ${ARCH} -- probably roihu-cpu.csc.fi."
    echo "!! Reconnect to roihu-gpu.csc.fi and re-run, or the environment will"
    echo "!! not work inside GPU jobs."
    exit 1
fi

# Verified on roihu-gpu via `module spider python-pytorch` (Aug 2026).
PYTORCH_MODULE="${PYTORCH_MODULE:-python-pytorch/2.10}"

echo
echo "=== loading ${PYTORCH_MODULE} ==="
module purge
module load "${PYTORCH_MODULE}"

# Deliberately does NOT call torch.cuda.is_available(): login nodes generally
# expose no GPU, so a False here would be meaningless and alarming. CUDA is
# verified on an actual GPU node by roihu/preflight.sh.
python -c "
import torch, platform
print('machine :', platform.machine(), '(expect aarch64)')
print('torch   :', torch.__version__)
"

echo
echo "=== installing project deps (snntorch, moabb, mne) ==="
# --user installs into ~/.local, which is architecture-specific. Because this
# runs on the ARM GPU login node, the result is valid for GPU jobs only.
pip install --user --quiet snntorch moabb mne
python -c "
import snntorch, moabb, mne
print('snntorch:', snntorch.__version__)
print('moabb   :', moabb.__version__)
print('mne     :', mne.__version__)
"

echo
echo "=== part 1 complete ==="
echo "Next, as a BATCH job (not on this login node):"
echo "    sbatch roihu/00b_prepare_data.sh     # downloads + epochs all 12 subjects"
echo "    sbatch roihu/preflight.sh            # 15-min environment check"
