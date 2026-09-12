"""Pipeline wrapper for the Movie Genome Clustering Dashboard.

Refactored from Project_HOLLYWOOD.ipynb cells 29/31/36/42/44/46/48.
Each function is a standalone pipeline stage that accepts parameters and returns results.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'


def preprocess_features(feature_matrix: pd.DataFrame, genre_data: pd.DataFrame = None, include_genres=True):
    """Split features into continuous and binary views, scale, impute."""
    X = feature_matrix.copy()

    # Identify binary vs continuous columns
    binary_cols = [c for c in X.columns if X[c].nunique() <= 2]
    continuous_cols = [c for c in X.columns if c not in binary_cols]

    # Drop constant features
    constant = [c for c in X.columns if X[c].std() == 0]
    X = X.drop(columns=constant, errors='ignore')
    binary_cols = [c for c in binary_cols if c in X.columns]
    continuous_cols = [c for c in continuous_cols if c in X.columns]

    # Impute missing values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_values = imputer.fit_transform(X)
    X = pd.DataFrame(X_values, columns=X.columns, index=X.index)

    # Scale continuous features to [0, 1]
    if continuous_cols:
        scaler = MinMaxScaler()
        X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

    continuous_data = X[continuous_cols].values.astype(np.float32)
    binary_data = X[binary_cols].values.astype(np.float32)
    all_data = X.values.astype(np.float32)
    feature_names = list(X.columns)
    tt_codes = list(X.index)

    return {
        'X': all_data,
        'continuous_data': continuous_data,
        'binary_data': binary_data,
        'feature_names': feature_names,
        'tt_codes': tt_codes,
        'continuous_cols': list(continuous_cols),
        'binary_cols': list(binary_cols),
    }


def apply_idf_weighting(continuous_data, X):
    """Apply IDF + variance weighting to continuous features, then L2 normalize."""
    n_films = continuous_data.shape[0]

    feature_presence = (continuous_data > 0).sum(axis=0)
    idf_weights = np.log(n_films / (feature_presence + 1))
    variance_weights = np.var(continuous_data, axis=0)
    combined_weights = idf_weights * np.sqrt(variance_weights)

    weighted_continuous = continuous_data * combined_weights

    # Build full weighted matrix
    X_weighted = X.copy()
    X_weighted[:, :continuous_data.shape[1]] = weighted_continuous

    # L2 normalize
    norms = np.linalg.norm(X_weighted, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_weighted = X_weighted / norms

    return {
        'X_weighted': X_weighted,
        'weighted_continuous': weighted_continuous,
        'idf_weights': idf_weights,
        'variance_weights': variance_weights,
        'combined_weights': combined_weights,
    }


def compute_ensemble_distance(weighted_continuous, binary_data, w_continuous=0.75, w_genre=0.25):
    """Compute ensemble distance matrix from continuous (correlation) and binary (Jaccard)."""
    dist_continuous = pairwise_distances(weighted_continuous, metric='correlation')
    dist_genre = pairwise_distances(binary_data, metric='jaccard')

    ensemble_dist = w_continuous * dist_continuous + w_genre * dist_genre
    ensemble_dist = (ensemble_dist + ensemble_dist.T) / 2
    np.fill_diagonal(ensemble_dist, 0)

    return ensemble_dist


def run_umap(data, n_components=60, n_neighbors=30, min_dist=0.0, metric='correlation', is_precomputed=False):
    """Fit UMAP embeddings (cluster dim + 3D viz)."""
    import umap.umap_ as umap

    actual_metric = 'precomputed' if is_precomputed else metric
    reducer_cluster = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors,
        min_dist=min_dist, metric=actual_metric, random_state=42,
    )
    embedding_cluster = reducer_cluster.fit_transform(data)

    reducer_3d = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors,
        min_dist=min_dist, metric=actual_metric, random_state=42,
    )
    embedding_3d = reducer_3d.fit_transform(data)
    embedding_2d = embedding_3d[:, :2]

    return {
        'embedding_cluster': embedding_cluster,
        'embedding_3d': embedding_3d,
        'embedding_2d': embedding_2d,
    }


def run_hdbscan(embedding, min_cluster_size=40, min_samples=10, epsilon=0.0):
    """Run HDBSCAN clustering on UMAP embedding."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    outlier_scores = clusterer.outlier_scores_

    return {
        'clusterer': clusterer,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'outlier_scores': outlier_scores,
    }


def compute_soft_clusters(clusterer, n_clusters):
    """Compute soft membership vectors and secondary rail assignments."""
    import hdbscan

    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    primary_rail = np.argmax(soft_clusters, axis=1)

    threshold = 0.15
    secondary_rail = []
    for row in soft_clusters:
        sorted_idx = np.argsort(row)[::-1]
        if len(sorted_idx) > 1 and row[sorted_idx[1]] > threshold:
            secondary_rail.append(int(sorted_idx[1]))
        else:
            secondary_rail.append(-1)

    return {
        'soft_clusters': soft_clusters,
        'primary_rail': primary_rail,
        'secondary_rail': np.array(secondary_rail),
    }


def run_sub_clustering(embedding, cluster_labels, n_clusters):
    """Recursive sub-clustering per cluster (Spotify pattern)."""
    import umap.umap_ as umap
    import hdbscan

    sub_clusters = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_embedding = embedding[mask]
        if cluster_embedding.shape[0] < 30:
            continue

        sub_reducer = umap.UMAP(
            n_components=10, n_neighbors=15,
            min_dist=0.0, metric='euclidean', random_state=42,
        )
        sub_embedding = sub_reducer.fit_transform(cluster_embedding)

        sub_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5, min_samples=3,
            metric='euclidean', prediction_data=True,
        )
        sub_labels = sub_clusterer.fit_predict(sub_embedding)
        sub_clusters[cluster_id] = sub_labels

    return sub_clusters


def compute_outlier_tiers(cluster_labels, outlier_scores, embedding, n_clusters):
    """Three-tier outlier classification."""
    outlier_mask = cluster_labels == -1

    if outlier_mask.sum() == 0:
        return {
            'tier1_mask': np.zeros(len(cluster_labels), dtype=bool),
            'tier2_mask': np.zeros(len(cluster_labels), dtype=bool),
            'tier3_mask': np.zeros(len(cluster_labels), dtype=bool),
        }

    p60 = np.percentile(outlier_scores, 60)
    p85 = np.percentile(outlier_scores, 85)

    tier1_mask = outlier_mask & (outlier_scores < p60)
    tier2_mask = outlier_mask & (outlier_scores >= p60) & (outlier_scores < p85)
    tier3_mask = outlier_mask & (outlier_scores >= p85)

    return {
        'tier1_mask': tier1_mask,
        'tier2_mask': tier2_mask,
        'tier3_mask': tier3_mask,
    }


def run_full_pipeline(feature_matrix, params, progress_callback=None):
    """Run the entire pipeline end-to-end. Returns artifacts dict."""
    steps = [
        'Preprocessing', 'IDF Weighting', 'UMAP Embedding (60D)',
        'UMAP Embedding (3D)', 'HDBSCAN Clustering', 'Soft Membership',
        'Sub-Clustering', 'Outlier Tiers',
    ]

    def update(step_idx, msg=""):
        if progress_callback:
            progress_callback((step_idx + 1) / len(steps), f"Step {step_idx + 1}/{len(steps)}: {steps[step_idx]} {msg}")

    # Step 1: Preprocess
    update(0)
    prep = preprocess_features(feature_matrix)

    # Step 2: IDF Weighting
    update(1)
    if params.get('apply_idf', True):
        idf_result = apply_idf_weighting(prep['continuous_data'], prep['X'])
        data_for_umap = idf_result['X_weighted']
    else:
        idf_result = {'X_weighted': prep['X'], 'idf_weights': None, 'variance_weights': None, 'combined_weights': None}
        data_for_umap = prep['X']

    # Step 3: UMAP
    update(2)
    is_precomputed = params.get('use_ensemble', False)
    if is_precomputed:
        ensemble = compute_ensemble_distance(
            idf_result.get('weighted_continuous', prep['continuous_data']),
            prep['binary_data'],
            params.get('w_continuous', 0.75),
            params.get('w_genre', 0.25),
        )
        data_for_umap = ensemble

    umap_result = run_umap(
        data_for_umap,
        n_components=params.get('n_components', 60),
        n_neighbors=params.get('n_neighbors', 30),
        min_dist=params.get('min_dist', 0.0),
        metric=params.get('metric', 'correlation'),
        is_precomputed=is_precomputed,
    )
    update(3)

    # Step 4: HDBSCAN
    update(4)
    hdb_result = run_hdbscan(
        umap_result['embedding_cluster'],
        min_cluster_size=params.get('min_cluster_size', 40),
        min_samples=params.get('min_samples', 10),
        epsilon=params.get('epsilon', 0.0),
    )

    # Step 5: Soft Membership
    update(5)
    soft_result = compute_soft_clusters(hdb_result['clusterer'], hdb_result['n_clusters'])

    # Step 6: Sub-Clustering
    update(6)
    sub_clusters = run_sub_clustering(
        umap_result['embedding_cluster'],
        hdb_result['cluster_labels'],
        hdb_result['n_clusters'],
    )

    # Step 7: Outlier Tiers
    update(7)
    tier_result = compute_outlier_tiers(
        hdb_result['cluster_labels'],
        hdb_result['outlier_scores'],
        umap_result['embedding_cluster'],
        hdb_result['n_clusters'],
    )

    # Compute centroids
    centroids_original = {}
    centroids_umap = {}
    for cid in range(hdb_result['n_clusters']):
        mask = hdb_result['cluster_labels'] == cid
        centroids_original[cid] = prep['X'][mask].mean(axis=0)
        centroids_umap[cid] = umap_result['embedding_cluster'][mask].mean(axis=0)

    artifacts = {
        **prep,
        'X_weighted': idf_result['X_weighted'],
        'idf_weights': idf_result.get('idf_weights'),
        'variance_weights': idf_result.get('variance_weights'),
        'combined_weights': idf_result.get('combined_weights'),
        **umap_result,
        **hdb_result,
        'soft_clusters': soft_result['soft_clusters'],
        'sub_clusters': sub_clusters,
        **tier_result,
        'centroids_original': centroids_original,
        'centroids_umap': centroids_umap,
        'cluster_names': {},
        'sub_cluster_names': {},
        'config': {
            **params,
            'timestamp': datetime.now().isoformat(),
        },
    }

    return artifacts


def save_artifacts(artifacts, run_id=None):
    """Save pipeline artifacts and append to run history."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Save pkl
    joblib.dump(artifacts, RESULTS_DIR / 'pipeline_artifacts.pkl', compress=3)

    # Append to run history
    from utils.metrics import compute_metrics
    metrics = compute_metrics(
        artifacts['embedding_cluster'],
        artifacts['cluster_labels'],
    )

    run_entry = {
        'run_id': run_id or datetime.now().strftime('%Y%m%d_%H%M%S'),
        **artifacts.get('config', {}),
        **metrics,
    }

    history_path = RESULTS_DIR / 'run_history.jsonl'
    with open(history_path, 'a') as f:
        f.write(json.dumps(run_entry) + '\n')

    return run_entry
