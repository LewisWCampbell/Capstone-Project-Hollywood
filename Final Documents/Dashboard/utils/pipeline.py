"""Pipeline wrapper for the Movie Genome Clustering Dashboard.

Aligned with PROJECT_HOLLYWOOD_FINAL.ipynb methodology:
  - Z-score (StandardScaler) + IDF (TfidfTransformer) on genome features
  - Binary genre/decade bumps passed through unchanged
  - UMAP defaults: n_components=25, n_neighbors=80, min_dist=0.1, metric=correlation
  - HDBSCAN defaults: min_cluster_size=15, min_samples=3, epsilon=0.10, eom
  - Sub-clustering via HDBSCAN condensed tree extraction
  - XGBoost outlier recovery with confidence tiers (≥50% / 30-50% / <30%)
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer

PROJECT_DIR = Path(__file__).parent.parent.parent  # Final Documents/
PIPELINE_DATA_DIR = PROJECT_DIR / 'Data for Pipeline'
OUTPUT_DATA_DIR = PROJECT_DIR / 'Outputs for Dashboard'
EXTRA_DATA_DIR = PIPELINE_DATA_DIR / 'extra_data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'  # Dashboard/results/
BASE_DIR = PROJECT_DIR  # legacy alias


def preprocess_features(feature_matrix: pd.DataFrame):
    """Split features into genome (continuous) and binary bump views.

    Matches FINAL notebook cell 17: genome_cols are everything that isn't
    prefixed with genre_ or decade_. Constant features are dropped.
    """
    X_df = feature_matrix.copy()

    # Identify feature types by prefix (matches notebook convention)
    genome_cols = [c for c in X_df.columns
                   if not c.startswith('genre_') and not c.startswith('decade_')]
    genre_cols = [c for c in X_df.columns if c.startswith('genre_')]
    decade_cols = [c for c in X_df.columns if c.startswith('decade_')]

    # Drop constant features
    constant_cols = [c for c in X_df.columns if X_df[c].nunique() <= 1]
    if constant_cols:
        X_df = X_df.drop(columns=constant_cols)
        genome_cols = [c for c in genome_cols if c not in constant_cols]
        genre_cols = [c for c in genre_cols if c not in constant_cols]
        decade_cols = [c for c in decade_cols if c not in constant_cols]

    # Fill NaN with 0 (matches notebook fillna(0))
    X_df = X_df.fillna(0)

    X = X_df.values.astype(np.float32)
    feature_names = X_df.columns.tolist()
    tt_codes = list(X_df.index)

    # Extract genome and bump sub-matrices by column index
    genome_idx = [feature_names.index(c) for c in genome_cols]
    bump_idx = [feature_names.index(c) for c in genre_cols + decade_cols]

    X_genome = X[:, genome_idx]    # (n_movies, ~248) continuous 0-5
    X_bumps = X[:, bump_idx]       # (n_movies, ~34) binary 0/1

    return {
        'X': X,
        'X_genome': X_genome,
        'X_bumps': X_bumps,
        'feature_names': feature_names,
        'tt_codes': tt_codes,
        'genome_cols': genome_cols,
        'genre_cols': genre_cols,
        'decade_cols': decade_cols,
    }


def apply_idf_weighting(X_genome, X_bumps):
    """Z-score normalise genome features, then apply pre-computed IDF weights.

    Matches FINAL notebook cell 20:
      Step 1: Compute IDF weights from RAW genomes (preserves sparsity pattern)
      Step 2: Z-score normalise genomes (StandardScaler: mean=0, std=1)
      Step 3: Multiply z-scored genomes by pre-computed IDF weights
      Step 4: Concatenate with raw binary bumps (unchanged)

    Returns the full weighted matrix and the IDF weights vector.
    """
    # Step 1: Compute IDF weights from raw genomes (before z-scoring)
    # IDF depends on the sparsity pattern (zero vs nonzero). Z-scoring would
    # shift means to zero, destroying sparsity and making all IDF weights ≈ 1.
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf.fit(X_genome)
    idf_weights = tfidf.idf_  # 248 weights: rare tags → high, common → low

    # Step 2: Z-score normalise genome features
    scaler = StandardScaler()
    X_genome_zscore = scaler.fit_transform(X_genome).astype(np.float32)

    # Step 3: Apply pre-computed IDF weights to z-scored genomes
    X_genome_weighted = (X_genome_zscore * idf_weights[np.newaxis, :]).astype(np.float32)

    # Step 4: Concatenate z-scored+IDF genomes with raw binary bumps
    X_weighted = np.hstack([X_genome_weighted, X_bumps]).astype(np.float32)

    return {
        'X_weighted': X_weighted,
        'X_genome_weighted': X_genome_weighted,
        'idf_weights': idf_weights,
    }


def run_umap(data, n_components=25, n_neighbors=80, min_dist=0.1,
             metric='correlation', is_precomputed=False):
    """Fit UMAP embeddings (cluster dim + 3D viz).

    Defaults match FINAL notebook: n_components=25, n_neighbors=80,
    min_dist=0.1, metric=correlation.
    """
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


def run_hdbscan(embedding, min_cluster_size=15, min_samples=3, epsilon=0.10):
    """Run HDBSCAN clustering on UMAP embedding.

    Defaults match FINAL notebook: min_cluster_size=15, min_samples=3,
    epsilon=0.10, eom selection, euclidean metric.
    """
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


def run_sub_clustering(clusterer, cluster_labels, n_clusters, embedding_cluster):
    """Extract sub-clusters from HDBSCAN's condensed tree.

    Matches FINAL notebook cell 37: uses the condensed tree hierarchy rather
    than running a separate UMAP+HDBSCAN per cluster.
    """
    from sklearn.metrics import silhouette_score as _sil_score

    # Helper: recursively collect all point indices under a tree node
    def _get_all_points(node_id, ct_df, n_pts):
        points = set()
        for _, row in ct_df[ct_df['parent'] == node_id].iterrows():
            child = int(row['child'])
            if child < n_pts:
                points.add(child)
            else:
                points.update(_get_all_points(child, ct_df, n_pts))
        return points

    # Extract condensed tree
    ct_df = clusterer.condensed_tree_.to_pandas()
    n_pts = len(cluster_labels)

    # Pre-compute point sets for every cluster node in the tree
    _all_nodes = sorted(
        set(ct_df['parent'].unique()) |
        set(ct_df[ct_df['child'] >= n_pts]['child'].unique())
    )
    _all_nodes = [n for n in _all_nodes if n >= n_pts]
    _node_pts = {n: _get_all_points(n, ct_df, n_pts) for n in _all_nodes}

    # Map each EOM cluster to its condensed tree node
    _cluster_to_node = {}
    for _cid in range(n_clusters):
        _cid_pts = set(np.where(cluster_labels == _cid)[0])
        for _node, _pts in _node_pts.items():
            if _pts == _cid_pts:
                _cluster_to_node[_cid] = _node
                break

    # Extract sub-clusters from child nodes
    sub_clusters = {}
    sub_cluster_silhouettes = {}

    for _cid, _node in _cluster_to_node.items():
        _child_nodes = ct_df[
            (ct_df['parent'] == _node) & (ct_df['child'] >= n_pts)
        ]['child'].tolist()

        if len(_child_nodes) >= 2:
            _cid_indices = np.where(cluster_labels == _cid)[0]
            _sub_labels = np.full(len(_cid_indices), -1, dtype=int)

            for _sub_idx, _sub_node in enumerate(_child_nodes):
                _sub_pts = _node_pts[_sub_node]
                for _i, _gi in enumerate(_cid_indices):
                    if _gi in _sub_pts:
                        _sub_labels[_i] = _sub_idx

            sub_clusters[_cid] = _sub_labels.tolist()

            # Compute per-sub-cluster silhouette if enough sub-clusters
            _n_sub = len(set(_sub_labels) - {-1})
            _assigned = _sub_labels >= 0
            if _n_sub >= 2 and _assigned.sum() >= _n_sub + 1:
                try:
                    _overall_sil = _sil_score(
                        embedding_cluster[_cid_indices[_assigned]],
                        _sub_labels[_assigned],
                        sample_size=min(1000, int(_assigned.sum())),
                        random_state=42
                    )
                    sub_cluster_silhouettes[_cid] = {
                        '_overall': round(float(_overall_sil), 4)
                    }
                except Exception:
                    sub_cluster_silhouettes[_cid] = {'_overall': None}

    return sub_clusters, sub_cluster_silhouettes


def run_xgboost_validation(X_weighted, cluster_labels, feature_names):
    """Train XGBoost classifier and generate outlier recovery suggestions.

    Matches FINAL notebook cells 50-55:
      - 5-fold CV on clustered films
      - Predict recovery for outliers with confidence ≥ 50%
      - SHAP feature importance per cluster
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    RECOVERY_THRESHOLD = 0.5

    # Separate clustered vs outlier films
    clustered_mask = cluster_labels >= 0
    outlier_mask = cluster_labels == -1

    X_train = X_weighted[clustered_mask]
    y_train = cluster_labels[clustered_mask]
    X_outliers = X_weighted[outlier_mask]
    outlier_indices = np.where(outlier_mask)[0]

    # Encode labels for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    # Train XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_clf.fit(X_train, y_encoded)

    # Cross-validation
    cv_scores = cross_val_score(xgb_clf, X_train, y_encoded, cv=5, scoring='accuracy')
    xgb_cv_accuracy = float(cv_scores.mean())
    xgb_cv_std = float(cv_scores.std())

    # Outlier recovery
    recovery_suggestions = {}
    if X_outliers.shape[0] > 0:
        outlier_proba = xgb_clf.predict_proba(X_outliers)
        outlier_best_class = np.argmax(outlier_proba, axis=1)
        outlier_best_conf = np.max(outlier_proba, axis=1)
        outlier_best_label = le.inverse_transform(outlier_best_class)

        for idx, (gi, pred_label, conf) in enumerate(zip(
            outlier_indices, outlier_best_label, outlier_best_conf
        )):
            recovery_suggestions[int(gi)] = {
                'suggested_cluster': int(pred_label),
                'confidence': float(conf),
            }

    # SHAP feature importance per cluster
    cluster_shap_features = {}
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_clf)
        shap_values = explainer.shap_values(X_train)

        for class_idx in range(n_classes):
            cluster_id = int(le.inverse_transform([class_idx])[0])
            if isinstance(shap_values, list):
                class_shap = np.abs(shap_values[class_idx]).mean(axis=0)
            else:
                class_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
            top_idx = np.argsort(class_shap)[::-1][:10]
            cluster_shap_features[cluster_id] = [
                (feature_names[i], float(class_shap[i])) for i in top_idx
            ]
    except Exception:
        pass  # SHAP is optional; continue without it

    return {
        'xgb_cv_accuracy': xgb_cv_accuracy,
        'xgb_cv_std': xgb_cv_std,
        'recovery_suggestions': recovery_suggestions,
        'cluster_shap_features': cluster_shap_features,
    }


def run_full_pipeline(feature_matrix, params, progress_callback=None):
    """Run the entire pipeline end-to-end. Returns artifacts dict.

    Pipeline stages (matching FINAL notebook):
      1. Preprocess: split genome vs binary bump features
      2. IDF weighting: z-score genomes → apply TfidfTransformer IDF weights
      3. UMAP: reduce to n_components for clustering + 3D for viz
      4. HDBSCAN: density-based clustering
      5. Sub-clustering: condensed tree extraction
      6. XGBoost: validation + outlier recovery
    """
    steps = [
        'Preprocessing', 'Z-Score + IDF Weighting',
        'UMAP Embedding', 'HDBSCAN Clustering',
        'Sub-Clustering (Condensed Tree)', 'XGBoost Validation + Recovery',
    ]

    def update(step_idx, msg=""):
        if progress_callback:
            progress_callback(
                (step_idx + 1) / len(steps),
                f"Step {step_idx + 1}/{len(steps)}: {steps[step_idx]} {msg}"
            )

    # Step 1: Preprocess — split features by type
    update(0)
    prep = preprocess_features(feature_matrix)

    # Step 2: Z-Score + IDF Weighting
    update(1)
    if params.get('apply_idf', True):
        idf_result = apply_idf_weighting(prep['X_genome'], prep['X_bumps'])
        data_for_umap = idf_result['X_weighted']
        idf_weights = idf_result['idf_weights']
    else:
        # No IDF — just concatenate raw genome + bumps
        data_for_umap = np.hstack([prep['X_genome'], prep['X_bumps']]).astype(np.float32)
        idf_result = {'X_weighted': data_for_umap, 'idf_weights': None}
        idf_weights = None

    # Step 3: UMAP
    update(2)
    umap_result = run_umap(
        data_for_umap,
        n_components=params.get('n_components', 25),
        n_neighbors=params.get('n_neighbors', 80),
        min_dist=params.get('min_dist', 0.1),
        metric=params.get('metric', 'correlation'),
    )

    # Step 4: HDBSCAN
    update(3)
    hdb_result = run_hdbscan(
        umap_result['embedding_cluster'],
        min_cluster_size=params.get('min_cluster_size', 15),
        min_samples=params.get('min_samples', 3),
        epsilon=params.get('epsilon', 0.10),
    )

    # Step 5: Sub-Clustering (condensed tree)
    update(4)
    sub_clusters, sub_cluster_silhouettes = run_sub_clustering(
        hdb_result['clusterer'],
        hdb_result['cluster_labels'],
        hdb_result['n_clusters'],
        umap_result['embedding_cluster'],
    )

    # Step 6: XGBoost Validation + Outlier Recovery
    update(5)
    xgb_result = run_xgboost_validation(
        idf_result['X_weighted'],
        hdb_result['cluster_labels'],
        prep['feature_names'],
    )

    # Compute centroids (distance to centroid + rank in cluster)
    centroids_umap = {}
    for cid in range(hdb_result['n_clusters']):
        mask = hdb_result['cluster_labels'] == cid
        centroids_umap[cid] = umap_result['embedding_cluster'][mask].mean(axis=0)

    artifacts = {
        'X': prep['X'],
        'X_weighted': idf_result['X_weighted'],
        'feature_names': prep['feature_names'],
        'tt_codes': prep['tt_codes'],
        'genome_cols': prep['genome_cols'],
        'genre_cols': prep['genre_cols'],
        'decade_cols': prep['decade_cols'],
        'idf_weights': idf_weights,
        **umap_result,
        **hdb_result,
        'sub_clusters': sub_clusters,
        'sub_cluster_silhouettes': sub_cluster_silhouettes,
        'centroids_umap': centroids_umap,
        'cluster_names': {},
        'sub_cluster_names': {},
        'sub_cluster_shap_features': {},
        **xgb_result,
        'config': {
            'PIPELINE': 'zscore_idf_xgboost_dashboard',
            'PREPROCESSING': 'Z-score (StandardScaler) + IDF (TfidfTransformer) on genome features',
            'UMAP_N_COMPONENTS': params.get('n_components', 25),
            'UMAP_N_NEIGHBORS': params.get('n_neighbors', 80),
            'UMAP_MIN_DIST': params.get('min_dist', 0.1),
            'UMAP_METRIC': params.get('metric', 'correlation'),
            'HDBSCAN_MIN_CLUSTER_SIZE': params.get('min_cluster_size', 15),
            'HDBSCAN_MIN_SAMPLES': params.get('min_samples', 3),
            'HDBSCAN_EPSILON': params.get('epsilon', 0.10),
            'HDBSCAN_SELECTION_METHOD': 'eom',
            'INCLUDE_GENRE_FEATURES': params.get('include_genres', True),
            'INCLUDE_DECADE_FEATURE': params.get('include_decades', True),
            'apply_idf': params.get('apply_idf', True),
            'timestamp': datetime.now().isoformat(),
        },
    }

    return artifacts


def save_artifacts(artifacts, run_id=None):
    """Save pipeline artifacts as Parquet + JSON and append to run history.

    Matches the artifact format produced by FINAL notebook cell 64.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    artifacts_dir = RESULTS_DIR / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)

    tt_codes = artifacts.get('tt_codes', [])
    feature_names = artifacts.get('feature_names', [])

    # ── Parquet: feature matrices ────────────────────────────────────────
    if 'X_weighted' in artifacts:
        df_w = pd.DataFrame(artifacts['X_weighted'], columns=feature_names, index=tt_codes)
        df_w.index.name = 'tt_code'
        df_w.to_parquet(artifacts_dir / 'X_weighted.parquet', engine='pyarrow', compression='snappy')

    if 'X' in artifacts:
        df_r = pd.DataFrame(artifacts['X'], columns=feature_names, index=tt_codes)
        df_r.index.name = 'tt_code'
        df_r.to_parquet(artifacts_dir / 'X_raw.parquet', engine='pyarrow', compression='snappy')

    # ── Parquet: embeddings ──────────────────────────────────────────────
    for key, name in [('embedding_cluster', 'embedding_cluster'),
                      ('embedding_3d', 'embedding_3d'),
                      ('embedding_2d', 'embedding_2d')]:
        if key in artifacts:
            arr = artifacts[key]
            cols = [f'umap_{i}' for i in range(arr.shape[1])]
            df_e = pd.DataFrame(arr, columns=cols, index=tt_codes)
            df_e.index.name = 'tt_code'
            df_e.to_parquet(artifacts_dir / f'{name}.parquet', engine='pyarrow', compression='snappy')

    # ── Parquet: cluster labels ──────────────────────────────────────────
    if 'cluster_labels' in artifacts:
        cluster_labels = artifacts['cluster_labels']
        n_clusters = artifacts.get('n_clusters', 0)
        embedding = artifacts.get('embedding_cluster')
        centroids = artifacts.get('centroids_umap', {})

        # Compute distance to centroid and rank in cluster
        dist_to_centroid = np.zeros(len(cluster_labels), dtype=np.float32)
        rank_in_cluster = np.zeros(len(cluster_labels), dtype=int)
        for cid in range(n_clusters):
            mask = cluster_labels == cid
            if cid in centroids and embedding is not None:
                dists = np.linalg.norm(
                    embedding[mask] - centroids[cid][np.newaxis, :], axis=1
                )
                dist_to_centroid[mask] = dists
                ranks = np.argsort(dists) + 1
                rank_in_cluster[mask] = ranks

        df_l = pd.DataFrame({
            'tt_code': tt_codes,
            'cluster_label': cluster_labels,
            'probability': artifacts['clusterer'].probabilities_
                           if 'clusterer' in artifacts else np.zeros(len(tt_codes)),
            'outlier_score': artifacts.get('outlier_scores', np.zeros(len(tt_codes))),
            'distance_to_centroid': dist_to_centroid,
            'rank_in_cluster': rank_in_cluster,
        })
        df_l.to_parquet(artifacts_dir / 'cluster_labels.parquet', engine='pyarrow', compression='snappy')

    # ── Parquet: IDF weights ─────────────────────────────────────────────
    if artifacts.get('idf_weights') is not None:
        genome_cols = artifacts.get('genome_cols', [])
        idf_w = artifacts['idf_weights']
        if len(genome_cols) == len(idf_w):
            df_idf = pd.DataFrame({'feature': genome_cols, 'idf_weight': idf_w})
        else:
            df_idf = pd.DataFrame({'feature': [f'genome_{i}' for i in range(len(idf_w))],
                                    'idf_weight': idf_w})
        df_idf.to_parquet(artifacts_dir / 'idf_weights.parquet', engine='pyarrow', compression='snappy')

    # ── Parquet: Jordan metadata ─────────────────────────────────────────
    if 'jordan_df' in artifacts:
        artifacts['jordan_df'].to_parquet(
            artifacts_dir / 'jordan_metadata.parquet',
            engine='pyarrow', compression='snappy'
        )

    # ── JSON: metadata and nested structures ─────────────────────────────
    def _save_json(obj, name):
        with open(artifacts_dir / name, 'w') as f:
            json.dump(obj, f, indent=2, default=str)

    if 'cluster_names' in artifacts:
        _save_json({str(k): v for k, v in artifacts['cluster_names'].items()}, 'cluster_names.json')
    if 'cluster_shap_features' in artifacts:
        _save_json({str(k): v for k, v in artifacts['cluster_shap_features'].items()}, 'cluster_shap_features.json')
    if 'recovery_suggestions' in artifacts:
        _save_json({str(k): v for k, v in artifacts['recovery_suggestions'].items()}, 'recovery_suggestions.json')
    if 'title_lookup' in artifacts:
        _save_json(artifacts['title_lookup'], 'title_lookup.json')

    # Sub-cluster artifacts
    if 'sub_clusters' in artifacts:
        _save_json({str(k): v for k, v in artifacts['sub_clusters'].items()}, 'sub_clusters.json')
    if 'sub_cluster_silhouettes' in artifacts:
        _save_json({str(k): v for k, v in artifacts['sub_cluster_silhouettes'].items()}, 'sub_cluster_silhouettes.json')
    if 'sub_cluster_names' in artifacts:
        _save_json({str(k): v for k, v in artifacts['sub_cluster_names'].items()}, 'sub_cluster_names.json')
    if 'sub_cluster_shap_features' in artifacts:
        _save_json({str(k): v for k, v in artifacts['sub_cluster_shap_features'].items()}, 'sub_cluster_shap_features.json')

    _save_json({
        'feature_names': feature_names,
        'genome_cols': artifacts.get('genome_cols', []),
        'genre_cols': artifacts.get('genre_cols', []),
        'decade_cols': artifacts.get('decade_cols', []),
        'tt_codes': tt_codes,
    }, 'feature_columns.json')

    _save_json({
        'xgb_cv_accuracy': float(artifacts.get('xgb_cv_accuracy', 0)),
        'xgb_cv_std': float(artifacts.get('xgb_cv_std', 0)),
        'n_clusters': int(artifacts.get('n_clusters', 0)),
    }, 'validation_metrics.json')

    if 'config' in artifacts:
        _save_json(artifacts['config'], 'config.json')

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
