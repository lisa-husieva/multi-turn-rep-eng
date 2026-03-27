"""
Part 4: Convergence analysis.

For JBB objectives where all three frameworks produced at least one successful
jailbreak, extracts full-context (k=k_max) representations of those jailbroken
final responses and compares them directly.

Visualizes using 2D PCA with points colored by framework and by objective pair.
"""

import numpy as np
from sklearn.decomposition import PCA


def run_convergence_analysis(
    representations_jsonl: str,
    model: str,
    layer: int,
    min_frameworks: int = 3,
) -> dict:
    """
    Returns dict with:
      pca_coords: array (n_jailbroken, 2)
      frameworks: list[str]
      pair_ids: list[int]
    for objectives where all min_frameworks produced a jailbreak.
    """
    raise NotImplementedError


def plot_convergence(convergence_result: dict, output_path: str) -> None:
    """2D PCA scatter colored by framework and by objective pair."""
    raise NotImplementedError
