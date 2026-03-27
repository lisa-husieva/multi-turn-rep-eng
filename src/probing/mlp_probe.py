"""
MLP probe for harmful/benign classification.

Supports both mean-pooled (response-level) and token-level training modes,
enabling direct comparison with Bullwinkel et al.'s token-level approach.

Built on scikit-learn MLPClassifier for simplicity; outputs probability scores
(P(harmful)) used in trajectory and transfer analyses.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_probe(hidden_layer_sizes: tuple = (256,), max_iter: int = 500) -> Pipeline:
    """Return an sklearn Pipeline with standard scaling and MLP classifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)),
    ])


def train_probe(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Pipeline:
    """Train and return a probe on the provided representations and binary labels."""
    probe = build_probe(**kwargs)
    probe.fit(X_train, y_train)
    return probe


def score_probe(probe: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return P(harmful) for each sample in X."""
    return probe.predict_proba(X)[:, 1]
