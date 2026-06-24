"""
src/utils/logger.py — Centralised logging configuration for the Fraud Detection System.

Extracted from train.py: the logging.basicConfig block and Optuna/warning filters
that were previously at module level are now set up here and called once at startup.
"""

from __future__ import annotations

import logging
import warnings


"""
def get_logger(name: str) -> logging.Logger:
   
   # Configure root logger with a consistent format used across the whole project.
  #  Also silences Optuna's per-trial chatter and sklearn UserWarnings.
   # Call once at the top of any entry-point script (scripts/train.py, etc.).
    
    if logging.getLogger().handlers:
        return
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep Optuna quiet
    warnings.filterwarnings("ignore", category=UserWarning)

    return logging.getLogger(name)
"""
def get_logger(name: str) -> logging.Logger:
    if logging.getLogger().handlers:
        return logging.getLogger(name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # silence optuna logs only if optuna is installed (training env)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass

    warnings.filterwarnings("ignore", category=UserWarning)

    return logging.getLogger(name)
