"""
LAT (Linear Adversarial Training) reading vector via PCA on difference vectors.

For each matched objective pair, computes the difference vector between the mean
jailbroken and mean benign full-context representations. Fits PCA on the set of
difference vectors and takes the first principal component as the reading vector.
All conversations are then projected onto this vector to produce a scalar
harmfulness score.
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_difference_vectors(
    jailbroken_reps: dict[int, np.ndarray],
    benign_reps: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Compute difference vectors for each matched objective pair.

    Args:
        jailbroken_reps: {objective_pair_id: mean representation array}
        benign_reps:     {objective_pair_id: mean representation array}

    Returns:
        Array of shape (n_pairs, hidden_dim).
    """
    raise NotImplementedError


def fit_reading_vector(difference_vectors: np.ndarray) -> np.ndarray:
    """Fit PCA on difference vectors and return the first PC as reading vector."""
    pca = PCA(n_components=1)
    pca.fit(difference_vectors)
    return pca.components_[0]


def project(representations: np.ndarray, reading_vector: np.ndarray) -> np.ndarray:
    """Project representations onto the reading vector. Returns scalar scores."""
    return representations @ reading_vector
