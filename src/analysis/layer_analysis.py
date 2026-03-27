"""
Part 5: Layer-by-layer analysis.

For each layer, trains a probe on full-context representations and computes AUC,
separately per attack framework and per model, and separately for across-pair and
within-pair settings. Plots AUC as a function of layer for each condition.
"""

import numpy as np


def run_layer_sweep(
    representations_jsonl: str,
    model: str,
    framework: str,
    aggregation: str = "mean_pool",
) -> dict:
    """
    Returns {layer: {"across_pair_auc": float, "within_pair_auc": float}}
    for all available layers.
    """
    raise NotImplementedError


def plot_layer_auc(layer_results: dict[str, dict], output_path: str) -> None:
    """Plot AUC vs. layer for each (framework, condition) combination."""
    raise NotImplementedError
