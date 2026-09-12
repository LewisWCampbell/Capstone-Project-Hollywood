"""Clustering quality metrics for the Movie Genome Dashboard."""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def compute_metrics(embedding: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering quality metrics for non-outlier points."""
    valid = labels != -1
    n_valid = valid.sum()
    n_labels = len(set(labels[valid])) if n_valid > 0 else 0

    if n_valid < 2 or n_labels < 2:
        return {
            'silhouette': -1,
            'davies_bouldin': -1,
            'calinski_harabasz': -1,
            'n_clusters': 0,
            'outlier_pct': (~valid).mean() if len(valid) > 0 else 1.0,
        }

    sample_size = min(2000, n_valid)
    return {
        'silhouette': float(silhouette_score(
            embedding[valid], labels[valid],
            sample_size=sample_size, random_state=42
        )),
        'davies_bouldin': float(davies_bouldin_score(
            embedding[valid], labels[valid]
        )),
        'calinski_harabasz': float(calinski_harabasz_score(
            embedding[valid], labels[valid]
        )),
        'n_clusters': n_labels,
        'outlier_pct': float((~valid).mean()),
    }
