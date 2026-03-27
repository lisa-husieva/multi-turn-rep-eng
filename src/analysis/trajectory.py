"""
Part 2: r_t trajectory analysis.

Using Design B extractions, plots probe score and LAT projection for each
turn's own response r_t across turns 1 to n, separately for jailbroken and
benign conversations, per attack framework, and per model.
"""

import numpy as np


def compute_trajectories(
    representations_jsonl: str,
    probe,
    reading_vector: np.ndarray,
    model: str,
    framework: str,
    layer: int,
    aggregation: str = "mean_pool",
) -> dict:
    """
    Returns trajectories dict keyed by (conversation_id, verdict):
      {conv_id: {"verdict": str, "turns": [t, ...], "probe_scores": [...], "lat_scores": [...]}}
    """
    raise NotImplementedError


def plot_mean_trajectories(trajectories: dict, output_path: str) -> None:
    """Plot mean probe score and LAT projection across turns, split by verdict."""
    raise NotImplementedError
