# ============================================================================
#  Shared environment for all Roihu jobs.  Sourced, not executed.
#
#  Why this exists: the MOABB/MNE cache defaults to ~/mne_data, i.e. $HOME.
#  BNCI2015-001 (12 subjects, 13 channels) fits there at ~1.6 GB, but Cho2017
#  is 52 subjects at 64 channels and would very likely exhaust a CSC home
#  quota part-way through a download -- leaving a half-populated cache that
#  the training array then races on, failing much later and confusingly.
#  Everything therefore lives on scratch.
#
#  PROJECT_ROOT is derived from the submit directory rather than hard-coded,
#  so this works for any user/project without editing:
#      /scratch/<project>/<user>/fbcsp   ->   /scratch/<project>/<user>
# ============================================================================

_submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${_submit_dir}")}"

# Set unconditionally, not with ${MNE_DATA:-...}. sbatch exports the
# submitting shell's environment by default, so a stale MNE_DATA inherited
# from a login shell would otherwise win over the value intended here -- and
# a job that silently reads the wrong cache path fails at the first subject
# load, an hour after anything looked wrong.
export MNE_DATA="${PROJECT_ROOT}/mne_data"

# MOABB resolves the dataset-specific key FIRST and only falls back to
# MNE_DATA, so setting MNE_DATA alone is not sufficient. MNE also persists
# both keys to ~/.mne/mne-python.json on first download, and that file
# outlives any directory move.
export MNE_DATASETS_BNCI_PATH="${MNE_DATA}"

if [ ! -d "${MNE_DATA}" ]; then
    echo "!! MNE_DATA does not exist: ${MNE_DATA}"
    echo "!! Run 'sbatch roihu/00b_prepare_data.sh' before any job that reads it."
    exit 1
fi

# Verified on roihu-gpu via `module spider python-pytorch` (Aug 2026).
PYTORCH_MODULE="${PYTORCH_MODULE:-python-pytorch/2.10}"

# Dataset selection. Override per job:  DATASET=Cho2017 sbatch ...
DATASET="${DATASET:-BNCI2015_001}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-42}"

# Results land in a dataset-specific directory so two datasets cannot
# overwrite each other's fold artifacts.
case "${DATASET}" in
    BNCI2015_001) _tag="bnci2015" ;;
    Cho2017)      _tag="cho2017"  ;;
    *)            _tag="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')" ;;
esac
RESULTS_DIR="${RESULTS_DIR:-Results_${_tag}}"
OUT_DIR="${OUT_DIR:-Results_quant}"

# Filter bank. The default is the six-band bank every published result used.
# Override for band-count experiments, and pair it with RESULTS_DIR so the
# fold artifacts cannot overwrite the six-band ones:
#     FREQ_BANDS="[(6,15),(12,32)]" RESULTS_DIR=Results_bnci2015_2band sbatch roihu/01_train_array.sh
# No spaces inside the list: it is passed to argparse as a single token.
FREQ_BANDS="${FREQ_BANDS:-[(4,8),(8,14),(12,18),(16,24),(20,30),(26,40)]}"

echo "--- roihu/env.sh ---"
echo "  PROJECT_ROOT : ${PROJECT_ROOT}"
echo "  MNE_DATA     : ${MNE_DATA}"
echo "  DATASET      : ${DATASET}"
echo "  RESULTS_DIR  : ${RESULTS_DIR}"
echo "  N_FOLDS      : ${N_FOLDS}   SEED: ${SEED}"
echo "  FREQ_BANDS   : ${FREQ_BANDS}"
echo "--------------------"
