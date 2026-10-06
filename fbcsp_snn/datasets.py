"""MOABB dataset registry and loader.

Supported datasets
------------------
BNCI2014_001   — BCI Competition IV 2a, 9 subjects, 4 classes, 22 ch, 250 Hz
PhysionetMI    — 4-class, 109 subjects, 64 ch, 160 Hz
Cho2017        — 2-class, 52 subjects, 64 ch, 512 Hz
BNCI2015_001   — 2-class, 12 subjects, 13 ch, 512 Hz

All loaders return ``(X_train, y_train, X_test, y_test)`` where:
- ``X`` shape is ``(n_trials, n_channels, n_samples)``
- ``y`` is 1-indexed integer labels
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps dataset name → (n_classes, sfreq, n_channels, description)
DATASET_REGISTRY: Dict[str, Dict] = {
    "BNCI2014_001": {
        "n_classes": 4,
        "sfreq": 250,
        "n_channels": 22,
        "description": "BCI Competition IV 2a (4-class MI, 9 subjects)",
        "moabb_cls": "BNCI2014_001",
    },
    "PhysionetMI": {
        "n_classes": 4,
        "sfreq": 160,
        "n_channels": 64,
        "description": "PhysioNet Motor Imagery (4-class, 109 subjects)",
        "moabb_cls": "PhysionetMI",
        # Explicit event names prevent MOABB from including T0/rest as a 5th class.
        # PhysionetMI imagined runs: 4,8,12 → left_hand/right_hand; 6,10,14 → hands/feet.
        # Executed runs:             3,7,11 → left_hand/right_hand; 5,9,13  → hands/feet.
        # Load both imagined AND executed (12 runs total, ~360 trials/subject) so that
        # each class has enough samples for CSP (64 channels require >>64 trials/class).
        # MOABB default (imagined=True, executed=False) only loads ~90 trials on some
        # versions, causing rank-deficient covariances and near-chance accuracy.
        #
        # Epoch window: load the full 3-second trial (fixation 0–2 s, imagery 2–3 s).
        # crop_s crops the filter-bank output BEFORE CSP fitting so that CSP sees only
        # the 1-second imagery window — but filtering is done on the full epoch so there
        # are no edge effects from applying bandpass filters to a 161-sample signal.
        "events": ["left_hand", "right_hand", "hands", "feet"],
        "dataset_kwargs": {"imagined": True, "executed": True},
        "crop_s": (2.0, 3.0),
    },
    "Cho2017": {
        "n_classes": 2,
        "sfreq": 512,
        "n_channels": 64,
        "description": "Cho 2017 (2-class MI, 52 subjects)",
        "moabb_cls": "Cho2017",
    },
    "Schirrmeister2017": {
        "n_classes": 4,
        "sfreq": 500,
        "n_channels": 128,
        "description": "High Gamma Dataset (4-class MI, 14 subjects, 500 Hz)",
        "moabb_cls": "Schirrmeister2017",
        # 4 classes: right_hand, left_hand, rest, feet (~963 trials/subject).
        # Single session → StratifiedShuffleSplit 80/20.
        # ~240 trials/class gives full-rank 128×128 covariances for CSP.
    },
    "BNCI2014_002": {
        "n_classes": 2,
        "sfreq": 512,
        "n_channels": 15,
        "description": "BNCI 2014-002 (2-class MI, 14 subjects)",
        "moabb_cls": "BNCI2014_002",
        # MOABB reports a single session holding eight runs, but the dataset
        # ships a designed split: S**T.mat is five training runs, S**E.mat is
        # three evaluation runs, and the evaluation runs were recorded with
        # feedback. A random 80/20 draw would mix the two and discard the
        # authors' protocol, so split on run name instead.
        "run_split": {"train": "train", "test": "test"},
    },
    "BNCI2015_001": {
        "n_classes": 2,
        "sfreq": 512,
        "n_channels": 13,
        "description": "BNCI 2015-001 (2-class MI, 12 subjects)",
        "moabb_cls": "BNCI2015_001",
    },
}


_moabb_offline_patch_applied = False


def _patch_moabb_offline_cache() -> None:
    """Patch MOABB's legacy BNCI ``data_path()`` to skip its network
    round-trip when the requested file is already cached locally.

    ``moabb.datasets.download.data_dl()`` (what BNCI's ``data_path()`` calls)
    invokes ``pooch.retrieve()`` even when the local file's hash already
    matches -- pooch's HTTP downloader still attempts to reach the remote
    host (e.g. ``lampx.tugraz.at`` for BNCI2014-001) to verify/fetch
    metadata before deciding to skip the download. On clusters where that
    specific host is unreachable (firewalled login/compute nodes), this
    hangs for a long time instead of failing fast, even though the cached
    file on disk is already complete and correct.

    This patches ``moabb.datasets.bnci.base.data_path`` (and the reference
    already bound into ``bnci_2014``/``bnci_2015``'s own module namespaces,
    since ``from .base import data_path`` binds a separate name at import
    time) to return the local path directly, with NO network call,
    whenever that path already exists on disk -- computed via the exact
    same ``get_dataset_path()`` / ``_url_to_local_path()`` logic MOABB
    itself uses, so the resolved path is guaranteed identical. Falls back
    to MOABB's real, network-touching implementation whenever the file is
    not yet cached, so genuinely missing data still downloads normally.
    """
    global _moabb_offline_patch_applied
    if _moabb_offline_patch_applied:
        return

    import importlib
    import os.path as osp

    try:
        import moabb.datasets.bnci.base as _bnci_base
        from moabb.datasets.download import get_dataset_path
        from mne.utils import _url_to_local_path
    except ImportError:
        return  # MOABB/MNE not installed -- nothing to patch

    _original_data_path = _bnci_base.data_path

    def _offline_first_data_path(url, path=None, force_update=False,
                                  update_path=None, verbose=None):
        try:
            resolved_path = get_dataset_path("BNCI", path)
            destination = _url_to_local_path(
                url, osp.join(resolved_path, "MNE-bnci-data")
            )
            if osp.isfile(destination) and not force_update:
                return [destination]
        except Exception:
            pass  # any resolution issue -> fall through to the real path
        return _original_data_path(url, path, force_update, update_path, verbose)

    _bnci_base.data_path = _offline_first_data_path

    for mod_name in ("bnci_2014", "bnci_2015"):
        try:
            mod = importlib.import_module(f"moabb.datasets.bnci.{mod_name}")
            if getattr(mod, "data_path", None) is _original_data_path:
                mod.data_path = _offline_first_data_path
        except ImportError:
            continue

    _moabb_offline_patch_applied = True
    logger.info("Patched MOABB BNCI data_path() for offline-cache-first loading")


def get_n_classes(dataset_name: str) -> int:
    """Return the number of classes for *dataset_name*.

    Parameters
    ----------
    dataset_name : str
        Key into :data:`DATASET_REGISTRY`.

    Returns
    -------
    int
        Number of motor imagery classes.

    Raises
    ------
    ValueError
        If *dataset_name* is not in the registry.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]["n_classes"]


# ---------------------------------------------------------------------------
# MOABB loader
# ---------------------------------------------------------------------------

def load_moabb(
    dataset_name: str,
    subject_id: int,
    n_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from MOABB and return train/test splits.

    For datasets with two sessions (BNCI2014_001, BNCI2015_001) session 1 is
    used as training data and session 2 as test data.  For single-session
    datasets (PhysionetMI, Cho2017) a stratified 80/20 split is applied.

    Parameters
    ----------
    dataset_name : str
        Name of the MOABB dataset (key in :data:`DATASET_REGISTRY`).
    subject_id : int
        Subject to load (1-indexed, as used by MOABB).
    n_classes : Optional[int]
        Override the number of classes.  Defaults to the registry value.

    Returns
    -------
    X_train : np.ndarray
        Training EEG, shape ``(n_train, n_channels, n_samples)``, float32.
    y_train : np.ndarray
        Training labels, shape ``(n_train,)``, int64, **1-indexed**.
    X_test : np.ndarray
        Test EEG, shape ``(n_test, n_channels, n_samples)``, float32.
    y_test : np.ndarray
        Test labels, shape ``(n_test,)``, int64, **1-indexed**.

    Raises
    ------
    ValueError
        If *dataset_name* is not in the registry.
    ImportError
        If MOABB or MNE is not installed.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[dataset_name]
    effective_n_classes = n_classes if n_classes is not None else info["n_classes"]

    logger.info(
        "Loading %s — subject %d  (%d classes)",
        dataset_name,
        subject_id,
        effective_n_classes,
    )

    try:
        import moabb
        moabb.set_log_level("WARNING")
    except ImportError as exc:
        raise ImportError("MOABB is not installed. Run: pip install moabb") from exc

    _patch_moabb_offline_cache()

    # Dynamically import the dataset class from MOABB
    moabb_cls_name = info["moabb_cls"]
    try:
        import importlib
        # MOABB datasets live in moabb.datasets.*
        for module_path in (
            f"moabb.datasets.{moabb_cls_name.lower()}",
            "moabb.datasets",
        ):
            try:
                mod = importlib.import_module(module_path)
                if hasattr(mod, moabb_cls_name):
                    DatasetCls = getattr(mod, moabb_cls_name)
                    break
            except ModuleNotFoundError:
                continue
        else:
            # Fallback: search all moabb.datasets submodules
            import moabb.datasets as _ds_pkg
            DatasetCls = None
            for attr in dir(_ds_pkg):
                if attr == moabb_cls_name:
                    DatasetCls = getattr(_ds_pkg, attr)
                    break
            if DatasetCls is None:
                raise ImportError(
                    f"Could not find MOABB class '{moabb_cls_name}'."
                )
    except Exception as exc:
        raise ImportError(
            f"Failed to import MOABB dataset class '{moabb_cls_name}': {exc}"
        ) from exc

    # Pass any dataset-specific constructor kwargs (e.g. imagined/executed for PhysionetMI)
    dataset_kwargs = info.get("dataset_kwargs", {})
    if dataset_kwargs:
        logger.info("Creating %s with kwargs: %s", moabb_cls_name, dataset_kwargs)
    dataset = DatasetCls(**dataset_kwargs)

    # Use MOABB paradigm to epoch the data
    try:
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise ImportError(
            "MOABB paradigm not available. Ensure moabb>=1.1 is installed."
        ) from exc

    events = info.get("events", None)
    paradigm_kwargs = info.get("paradigm_kwargs", {})
    if paradigm_kwargs:
        logger.info("Using paradigm kwargs for %s: %s", dataset_name, paradigm_kwargs)
    if events is not None:
        paradigm = MotorImagery(events=events, n_classes=effective_n_classes, **paradigm_kwargs)
        logger.info("Using explicit event list for %s: %s", dataset_name, events)
    else:
        paradigm = MotorImagery(n_classes=effective_n_classes, **paradigm_kwargs)

    logger.info("Fetching epochs via MOABB paradigm (this may download data)...")
    X_all, y_all, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=[subject_id],
        return_epochs=False,
    )
    # X_all: (n_trials, n_channels, n_samples)  — already numpy float64
    X_all = X_all.astype(np.float32)

    # Convert string labels to 1-indexed integers
    unique_labels = sorted(set(y_all))
    label_map = {lbl: idx + 1 for idx, lbl in enumerate(unique_labels)}
    y_int = np.array([label_map[l] for l in y_all], dtype=np.int64)

    logger.info(
        "Epochs loaded — X: %s  y: %s  label_map: %s",
        X_all.shape,
        y_int.shape,
        label_map,
    )

    # Split by run when the dataset carries a designed train/evaluation
    # protocol that MOABB flattens into one session (see BNCI2014_002).
    run_split = info.get("run_split")
    if run_split is not None and "run" in metadata.columns:
        runs = metadata["run"].astype(str).values
        train_mask = np.array([run_split["train"] in r for r in runs])
        test_mask = np.array([run_split["test"] in r for r in runs])
        overlap = int(np.sum(train_mask & test_mask))
        if overlap or not train_mask.any() or not test_mask.any():
            raise ValueError(
                f"{dataset_name}: run_split {run_split} does not partition the "
                f"runs cleanly (train={train_mask.sum()}, test={test_mask.sum()}, "
                f"both={overlap}); runs seen: {sorted(set(runs))}"
            )
        X_train, y_train = X_all[train_mask], y_int[train_mask]
        X_test, y_test = X_all[test_mask], y_int[test_mask]
        logger.info(
            "Run split (%s) — train: %s  test: %s  runs: %s",
            run_split, X_train.shape, X_test.shape, sorted(set(runs)),
        )
        _log_split_summary(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test

    # Split by session when session info is available
    if "session" in metadata.columns:
        sessions = metadata["session"].values
        unique_sessions = sorted(set(sessions))
        logger.info("Sessions found: %s", unique_sessions)

        if len(unique_sessions) >= 2:
            # Session 1 → train, Session 2 → test
            s1, s2 = unique_sessions[0], unique_sessions[1]
            train_mask = sessions == s1
            test_mask = sessions == s2
            X_train, y_train = X_all[train_mask], y_int[train_mask]
            X_test, y_test = X_all[test_mask], y_int[test_mask]
            logger.info(
                "Session split — train: %s  test: %s",
                X_train.shape,
                X_test.shape,
            )
        else:
            X_train, y_train, X_test, y_test = _stratified_split(X_all, y_int)
    else:
        X_train, y_train, X_test, y_test = _stratified_split(X_all, y_int)

    _log_split_summary(X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """80/20 stratified split for single-session datasets."""
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(sss.split(X, y))
    logger.info(
        "Stratified 80/20 split — train: %d  test: %d",
        len(train_idx),
        len(test_idx),
    )
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _log_split_summary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Log shapes and class distributions for both splits."""
    def _dist(y: np.ndarray) -> str:
        vals, counts = np.unique(y, return_counts=True)
        return "  ".join(f"cls{v}:{c}" for v, c in zip(vals, counts))

    logger.info("Train  X: %s  y: %s  [%s]", X_train.shape, y_train.shape, _dist(y_train))
    logger.info("Test   X: %s  y: %s  [%s]", X_test.shape, y_test.shape, _dist(y_test))
