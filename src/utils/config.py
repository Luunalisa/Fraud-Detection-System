"""
src/utils/config.py — Configuration loading and model versioning helpers.

Extracted from train.py:
  - load_config()    → reads config.yaml (or any YAML path) into a dict.
  - _model_version() → auto-increments a semantic patch tag beside the model file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load and return a YAML config file as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _model_version(base_path: str) -> str:
    """
    Auto-increment a semantic patch version stored next to the model file.
    Creates  <model_path>.version  if it does not exist.
    Returns  'v<major>.<minor>.<patch>'  string.
    """
    version_file = Path(base_path).with_suffix(".version")
    if version_file.exists():
        parts = version_file.read_text().strip().lstrip("v").split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        patch += 1
    else:
        major, minor, patch = 1, 0, 0
    tag = f"v{major}.{minor}.{patch}"
    version_file.write_text(tag)
    return tag
