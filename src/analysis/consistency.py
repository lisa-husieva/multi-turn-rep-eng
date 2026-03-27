"""
Part 6: MLP vs. LAT consistency under topic control.

Computes Pearson correlation between per-conversation MLP drift and LAT drift,
within matched pairs and under both extraction designs, across all models.

Experiment 1 found 0.31 correlation without topic control. This analysis asks
whether topic control and/or Design B increases this.
"""

import numpy as np
from scipy.stats import pearsonr


def compute_drift(scores: list[float]) -> float:
    """Drift = score at k=1 minus score at k=k_max (or t=1 minus t=t_max)."""
    return scores[0] - scores[-1]


def run_consistency_analysis(
    representations_jsonl: str,
    probe,
    reading_vector: np.ndarray,
    model: str,
    framework: str,
    layer: int,
    design: str,  # "A" or "B"
    aggregation: str = "mean_pool",
) -> dict:
    """
    Returns {
        "pearson_r": float,
        "p_value": float,
        "mlp_drifts": list[float],
        "lat_drifts": list[float],
    }
    """
    raise NotImplementedError
