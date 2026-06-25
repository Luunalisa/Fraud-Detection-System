
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _model_version(base_path: str) -> str:

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
