"""HDF5 / legacy .mat file loader.

Loads EEG data from ``.mat`` files saved in HDF5 format (MATLAB v7.3+).
Expected layout inside the file::

    /data    — float array (n_trials, n_channels, n_samples)
    /labels  — int array   (n_trials,)  [1-indexed]

Parameters match the ``Config`` dataclass but this module can also be used
standalone by passing ``data_path`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)


def load_hdf5(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load EEG data from an HDF5 ``.mat`` file.

    Parameters
    ----------
    data_path : str
        Path to the ``.mat`` / ``.h5`` file.

    Returns
    -------
    X : np.ndarray
        EEG data, shape ``(n_trials, n_channels, n_samples)``, dtype ``float32``.
    y : np.ndarray
        Class labels, shape ``(n_trials,)``, dtype ``int64``, **1-indexed**.

    Raises
    ------
    FileNotFoundError
        If ``data_path`` does not exist.
    KeyError
        If expected datasets (``/data``, ``/labels``) are missing.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading HDF5 file: %s", path)

    with h5py.File(path, "r") as f:
        required_keys = {"data", "labels"}
        missing = required_keys - set(f.keys())
        if missing:
            raise KeyError(
                f"HDF5 file missing datasets: {missing}. "
                f"Available keys: {list(f.keys())}"
            )

        X = np.array(f["data"], dtype=np.float32)
        y = np.array(f["labels"], dtype=np.int64).ravel()

    logger.info(
        "Loaded  X: %s  y: %s  (classes: %s)",
        X.shape,
        y.shape,
        np.unique(y).tolist(),
    )
    return X, y
