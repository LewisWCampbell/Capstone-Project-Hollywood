"""Page 2 — Pipeline Runner: Run/re-run pipeline with parameter controls.

Defaults aligned with PROJECT_HOLLYWOOD_FINAL.ipynb:
  UMAP:    n_components=25, n_neighbors=80, min_dist=0.1, metric=correlation
  HDBSCAN: min_cluster_size=15, min_samples=3, epsilon=0.10
  Preprocessing: Z-score (StandardScaler) + IDF (TfidfTransformer) on genomes
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from utils.data import (
    load_pipeline_artifacts, build_feature_matrix, BASE_DIR, RESULTS_DIR,
)
from utils.metrics import compute_metrics

st.set_page_config(page_title="Pipeline Runner", layout="wide")
st.title("Pipeline Runner")
st.markdown(
    "Run or re-run the **z-score + IDF + UMAP + HDBSCAN** clustering pipeline "
    "with adjustable parameters. Defaults match the FINAL notebook."
)

# --- Parameter Sidebar ---
with st.sidebar:
    st.header("Parameters")

    st.subheader("Preprocessing")
    include_genres = st.checkbox(
        "Include Genre Features", value=True,
        help="Appends binary one-hot flags for genres (Action, Comedy, etc.)."
    )
    include_decades = st.checkbox(
        "Include Decade Features", value=True,
        help="Appends binary one-hot flags for release decade (1980s, 1990s, etc.)."
    )
    idf_on = st.checkbox(
        "Apply Z-Score + IDF weighting", value=True,
        help="Z-score normalise genome features (StandardScaler), then apply "
             "TfidfTransformer IDF weights. Binary bumps pass through unchanged."
    )

    st.subheader("UMAP")
    n_comp = st.slider("n_components", 10, 100, 25, step=5,
                        help="FINAL notebook: 25")
    n_neigh = st.slider("n_neighbors", 5, 200, 80, step=5,
                         help="FINAL notebook: 80")
    min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, step=0.05,
                          help="FINAL notebook: 0.1")
    metric = st.selectbox("metric", ["correlation", "euclidean", "cosine"],
                          index=0,
                          help="FINAL notebook: correlation")

    st.subheader("HDBSCAN")
    mcs = st.slider("min_cluster_size", 5, 100, 15,
                     help="FINAL notebook: 15")
    ms = st.slider("min_samples", 1, 30, 3,
                    help="FINAL notebook: 3")
    eps = st.slider("cluster_selection_epsilon", 0.0, 1.0, 0.10, step=0.05,
                     help="FINAL notebook: 0.10")

# --- Show Last Run Results ---
arts = load_pipeline_artifacts()
history_path = RESULTS_DIR / 'run_history.jsonl'

if arts is not None:
    st.subheader("Latest Pipeline Results")
    config = arts.get('config', {})

    metrics = compute_metrics(arts['embedding_cluster'], arts['cluster_labels'])

    # Load previous run for delta comparison
    prev_metrics = None
    if history_path.exists():
        lines = history_path.read_text().strip().split('\n')
        if len(lines) >= 2:
            prev_metrics = json.loads(lines[-2])

    col1, col2, col3, col4 = st.columns(4)
    sil = metrics['silhouette']
    db = metrics['davies_bouldin']
    nc = metrics['n_clusters']
    op = metrics['outlier_pct']

    col1.metric("Silhouette", f"{sil:.3f}",
                f"{sil - prev_metrics['silhouette']:.3f}" if prev_metrics else None)
    col2.metric("DB Index", f"{db:.3f}",
                f"{db - prev_metrics['davies_bouldin']:.3f}" if prev_metrics else None,
                delta_color="inverse")
    col3.metric("N Clusters", f"{nc}",
                f"{nc - prev_metrics['n_clusters']}" if prev_metrics else None)
    col4.metric("Outlier Rate", f"{op:.1%}",
                f"{(op - prev_metrics['outlier_pct']):.1%}" if prev_metrics else None,
                delta_color="inverse")

    with st.expander("Pipeline Configuration"):
        st.json(config)
else:
    st.info("No pipeline results found. Run the notebook or click 'Run Pipeline' below.")

st.markdown("---")

# --- Run Pipeline ---
st.subheader("Run Pipeline")
st.warning("Running the pipeline takes several minutes (UMAP + HDBSCAN on ~8,000 films).")

if st.button("Run Pipeline", type="primary"):
    with st.status("Running pipeline...", expanded=True) as status_widget:
        progress_bar = st.progress(0)
        log_area = st.empty()

        def progress_callback(pct, msg):
            progress_bar.progress(pct)
            log_area.text(msg)

        try:
            # Build feature matrix from CSVs
            log_area.text("Loading feature matrix from CSVs...")
            matrix_df, tt_codes_list, feat_names = build_feature_matrix(
                include_genres=include_genres
            )

            params = {
                'include_genres': include_genres,
                'include_decades': include_decades,
                'apply_idf': idf_on,
                'n_components': n_comp,
                'n_neighbors': n_neigh,
                'min_dist': min_dist,
                'metric': metric,
                'min_cluster_size': mcs,
                'min_samples': ms,
                'epsilon': eps,
            }

            from utils.pipeline import run_full_pipeline, save_artifacts
            from utils.llm import name_cluster

            artifacts = run_full_pipeline(matrix_df, params, progress_callback)

            # --- Auto-Naming Step ---
            log_area.text("Auto-naming clusters via Ollama...")
            n_clusters = artifacts['n_clusters']
            cluster_labels = artifacts['cluster_labels']
            tt_codes = np.array(artifacts['tt_codes'])
            X = artifacts['X']
            feat_names = artifacts['feature_names']

            title_lookup = {}
            if 'omdb' in st.session_state:
                title_lookup = {
                    tt: st.session_state.omdb.get(tt, {}).get('Title', tt)
                    for tt in tt_codes
                }

            for cid in range(n_clusters):
                progress_bar.progress(0.90 + (0.10 * (cid / max(1, n_clusters))))
                log_area.text(f"Naming cluster {cid + 1} of {n_clusters}...")

                mask = cluster_labels == cid
                sample_tts = tt_codes[mask][:6].tolist()

                result = name_cluster(
                    cid, sample_tts, X, cluster_labels, feat_names, title_lookup
                )
                artifacts['cluster_names'][cid] = result

            progress_bar.progress(1.0)
            log_area.text("Saving artifacts...")

            run_entry = save_artifacts(artifacts)

            status_widget.update(label="Pipeline complete!", state="complete")
            st.success(f"Done! {artifacts['n_clusters']} clusters found. Results saved.")
            st.json(run_entry)

            # Clear cache so pages reload new data
            st.cache_resource.clear()
            st.cache_data.clear()

        except Exception as e:
            status_widget.update(label="Pipeline failed", state="error")
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# --- Run Log ---
if history_path.exists():
    st.markdown("---")
    st.subheader("Run History")
    lines = history_path.read_text().strip().split('\n')
    runs = [json.loads(line) for line in lines if line.strip()]
    if runs:
        run_df = pd.DataFrame(runs)
        display_cols = [
            c for c in [
                'run_id', 'timestamp',
                'UMAP_N_COMPONENTS', 'UMAP_N_NEIGHBORS',
                'HDBSCAN_MIN_CLUSTER_SIZE', 'HDBSCAN_EPSILON',
                'silhouette', 'davies_bouldin', 'n_clusters', 'outlier_pct',
                # Legacy column names from old runs
                'n_components', 'n_neighbors', 'min_cluster_size', 'epsilon',
            ] if c in run_df.columns
        ]
        st.dataframe(run_df[display_cols], use_container_width=True)
