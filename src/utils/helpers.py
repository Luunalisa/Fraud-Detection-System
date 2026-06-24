"""
src/utils/helpers.py — Population Stability Index (PSI) for data drift detection.

Moved from src/drift.py — all logic is preserved exactly.

PSI interpretation (industry standard):
  PSI < 0.10  → No significant change
  PSI < 0.20  → Minor shift, monitor
  PSI >= 0.20 → Major shift, investigate / retrain

Used by the training pipeline to compare feature distributions between:
  - training set vs validation set
  - training set vs test set
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EPSILON = 1e-8   # avoid log(0)


def compute_psi(
    reference: np.ndarray,
    comparison: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the PSI for a single feature.

    Parameters
    ----------
    reference  : 1-D array from the reference distribution (e.g. training set).
    comparison : 1-D array from the comparison distribution (e.g. validation set).
    n_bins     : Number of equal-width buckets (default 10).

    Returns
    -------
    psi : float — Population Stability Index value.
    """
    # Determine bin edges from the reference distribution
    min_val = min(reference.min(), comparison.min())
    max_val = max(reference.max(), comparison.max())

    # Avoid degenerate bins when min == max (constant feature)
    if max_val == min_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0]  -= _EPSILON   # ensure the minimum value falls inside the first bin
    bins[-1] += _EPSILON   # ensure the maximum value falls inside the last bin

    ref_counts,  _ = np.histogram(reference,  bins=bins)
    comp_counts, _ = np.histogram(comparison, bins=bins)

    # Convert to proportions, guarding against empty buckets
    ref_pct  = ref_counts  / (len(reference)  + _EPSILON)
    comp_pct = comp_counts / (len(comparison) + _EPSILON)

    # Clip to epsilon so log is always defined
    ref_pct  = np.clip(ref_pct,  _EPSILON, None)
    comp_pct = np.clip(comp_pct, _EPSILON, None)

    psi_values = (comp_pct - ref_pct) * np.log(comp_pct / ref_pct)
    return float(psi_values.sum())


def psi_report(
    X_reference: np.ndarray,
    X_comparison: np.ndarray,
    label: str = "drift",
    feature_names: Sequence[str] | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute PSI for every feature column and return a {feature_name: psi} dict.

    Parameters
    ----------
    X_reference   : 2-D reference array (n_samples, n_features).
    X_comparison  : 2-D comparison array.
    label         : Human-readable tag for log messages.
    feature_names : Optional list of feature names (used as dict keys).
    n_bins        : Passed through to compute_psi.

    Returns
    -------
    report : dict mapping feature name → PSI value.
    """
    n_features = X_reference.shape[1]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) "
            f"does not match number of columns ({n_features})."
        )

    report: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        psi_val = compute_psi(X_reference[:, i], X_comparison[:, i], n_bins=n_bins)
        report[name] = round(psi_val, 6)

    # Summary log
    high_drift = {k: v for k, v in report.items() if v >= 0.20}
    mild_drift = {k: v for k, v in report.items() if 0.10 <= v < 0.20}

    logger.info(
        "[%s] PSI summary — features: %d | high drift (≥0.20): %d | mild (0.10–0.20): %d",
        label, n_features, len(high_drift), len(mild_drift),
    )
    if high_drift:
        logger.warning("[%s] High-drift features: %s", label, high_drift)
    if mild_drift:
        logger.info("[%s] Mild-drift features: %s", label, mild_drift)

    return report
