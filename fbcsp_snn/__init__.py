"""fbcsp_snn package — device detection, logger setup, CUDA config."""

import io
import logging
import sys

import torch

# Disable TorchDynamo/Inductor entirely.
# On Volta GPUs (V100, sm_70) with PyTorch 2.1.x, Triton PTX codegen fails
# and inductor tries to write large cached kernels to /tmp, which exhausts
# the node's tmpfs.  Eager mode is correct and fast enough for our small SNN.
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(name: str = "fbcsp_snn", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout.

    Parameters
    ----------
    name : str
        Logger name (default ``"fbcsp_snn"``).
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    # Force UTF-8 on Windows where the default console encoding is cp1252
    stream: io.TextIOWrapper | logging.StreamHandler
    if hasattr(sys.stdout, "buffer"):
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    else:
        stream = sys.stdout
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        _log = setup_logger()
        _log.info("GPU: %s  (%.1f GB VRAM)", props.name, props.total_memory / 1e9)
        # Prefer TF32 on Ampere+ for matmul throughput (no-op on V100/Volta)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Auto-tune cuDNN kernels for fixed input sizes (free speedup per fold)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        setup_logger().info("CUDA not available — running on CPU")
    return device


DEVICE: torch.device = _select_device()


# ---------------------------------------------------------------------------
# Global seeding (reproducibility)
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG that affects a training run.

    The scikit-learn components in this pipeline (CV splits, MIBIF, SVM) are
    already fixed via ``random_state=42``, so the *data partitions* were
    always reproducible.  The stochastic parts of SNN training were not:
    weight initialisation, dropout masks, and the Van Rossum target spike
    trains (sampled at ``spiking_prob``) all draw from torch's global RNG.
    Without this call, two runs of the same configuration give different
    weights and slightly different accuracies.

    Parameters
    ----------
    seed : int
        Seed applied to ``random``, ``numpy``, and ``torch`` (CPU and all
        CUDA devices).
    deterministic : bool
        If True, also disable cuDNN autotuning and select deterministic
        kernels.  This costs some throughput but makes GPU runs repeatable.
        ``cudnn.benchmark`` is enabled by :func:`_select_device` for speed,
        so it is explicitly turned off here.

    Note
    ----
    Full bit-exact determinism on CUDA additionally requires
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the environment; it is set here
    defensively before any cuBLAS handle is created.
    """
    import random

    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    setup_logger().info(
        "Global seed set to %d (deterministic=%s)", seed, deterministic
    )
