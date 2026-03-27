"""
Probe evaluation utilities: AUC, cross-framework transfer, within/across-pair comparisons.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc_score(y_true, y_score)


def within_pair_auc(
    probe,
    representations: dict[int, dict],  # {pair_id: {"jailbroken": arr, "benign": arr}}
) -> dict[int, float]:
    """
    Compute AUC separately within each matched objective pair.
    Returns {pair_id: auc}.
    """
    raise NotImplementedError


def across_pair_auc(
    probe,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """AUC across all pairs (standard evaluation)."""
    return compute_auc(y, probe.predict_proba(X)[:, 1] if hasattr(probe, "predict_proba") else probe(X))


def transfer_auc(
    source_framework: str,
    target_framework: str,
    representations: dict,
    labels: dict,
) -> float:
    """
    Train probe on source_framework representations, evaluate on target_framework.
    Returns AUC on target.
    """
    raise NotImplementedError
