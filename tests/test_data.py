"""
tests/test_data.py — Tests for data loading and split integrity.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_splits_no_overlap():
    """Train / val / test indices must be mutually exclusive."""
    # Replace with a call to your actual prepare_dataset() once data is wired in.
    from sklearn.model_selection import train_test_split
    idx = np.arange(1000)
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
    val_idx,   test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    assert len(set(train_idx) & set(val_idx))  == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx)   & set(test_idx)) == 0


def test_splits_cover_all_samples():
    from sklearn.model_selection import train_test_split
    idx = np.arange(1000)
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
    val_idx,   test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    combined = set(train_idx) | set(val_idx) | set(test_idx)
    assert combined == set(idx)
