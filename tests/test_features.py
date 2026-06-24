"""
tests/test_features.py — Tests for drift detection and feature utilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.helpers import compute_psi, psi_report


def test_psi_identical_distributions():
    """PSI of a distribution against itself should be ~0."""
    arr = np.random.default_rng(0).standard_normal(500)
    psi = compute_psi(arr, arr)
    assert psi < 1e-6


def test_psi_very_different_distributions():
    """PSI of two non-overlapping distributions should exceed 0.20."""
    ref  = np.zeros(500)
    comp = np.ones(500)
    psi  = compute_psi(ref, comp)
    assert psi >= 0.20


def test_psi_constant_feature_returns_zero():
    """Constant feature (min == max) must return 0 without error."""
    arr = np.ones(100)
    assert compute_psi(arr, arr) == 0.0


def test_psi_report_keys_match_feature_names():
    rng   = np.random.default_rng(1)
    ref   = rng.standard_normal((300, 4))
    comp  = rng.standard_normal((300, 4))
    names = ["a", "b", "c", "d"]
    report = psi_report(ref, comp, feature_names=names)
    assert set(report.keys()) == set(names)


def test_psi_report_wrong_feature_names_raises():
    ref  = np.ones((100, 3))
    comp = np.ones((100, 3))
    with pytest.raises(ValueError):
        psi_report(ref, comp, feature_names=["only_two_names", "here"])
