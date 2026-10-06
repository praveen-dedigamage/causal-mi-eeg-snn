"""CLI entry point for the FBCSP-SNN Motor Imagery EEG Classifier.

Usage
-----
    # Train all folds for subject 1 (adaptive bands, MOABB)
    python main.py train --source moabb --moabb-dataset BNCI2014_001 --subject-id 1

    # Train with static bands
    python main.py train --source moabb --moabb-dataset BNCI2014_001 --subject-id 1 \\
        --no-adaptive-bands --freq-bands "[(4,10),(10,14),(14,30)]"

    # Train a single fold
    python main.py train --source moabb --moabb-dataset BNCI2014_001 --subject-id 1 \\
        --fold 0

    # Inference on a saved fold
    python main.py infer --source moabb --moabb-dataset BNCI2014_001 --subject-id 1 \\
        --fold 0

    # Aggregate results after all folds complete
    python main.py aggregate --source moabb --moabb-dataset BNCI2014_001 --subject-id 1 \\
        --n-folds 10

    # Reliability / hardware-noise sweep on a saved fold (B15, no retraining)
    python main.py reliability --source moabb --moabb-dataset BNCI2014_001 \\
        --subject-id 1 --fold 0 --reliability-n-repeats 20
"""

from __future__ import annotations

import sys

from fbcsp_snn import setup_logger
from fbcsp_snn.config import build_parser, config_from_args

logger = setup_logger("main")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    logger.info(
        "Mode: %s  |  Dataset: %s  |  Subject: %d",
        cfg.mode, cfg.moabb_dataset, cfg.subject_id,
    )

    if cfg.mode == "train":
        from fbcsp_snn.pipeline import run_train
        run_train(cfg)

    elif cfg.mode == "infer":
        from fbcsp_snn.pipeline import run_infer
        run_infer(cfg)

    elif cfg.mode == "aggregate":
        from fbcsp_snn.pipeline import run_aggregate
        run_aggregate(cfg)

    elif cfg.mode == "reliability":
        from fbcsp_snn.reliability import run_reliability
        run_reliability(cfg)

    else:
        logger.error("Unknown mode: %s", cfg.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
