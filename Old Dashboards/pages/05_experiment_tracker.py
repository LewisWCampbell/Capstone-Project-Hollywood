"""Page 5 — Experiment Tracker: Compare runs, metric history, parameter sensitivity."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from utils.data import RESULTS_DIR
from utils.plots import metric_line_chart

st.set_page_config(page_title="Experiment Tracker", layout="wide")
st.title("Experiment Tracker")
st.markdown("Compare pipeline runs across parameter configurations.")

history_path = RESULTS_DIR / 'run_history.jsonl'

if not history_path.exists() or history_path.stat().st_size == 0:
    st.info("No run history found. Use the **Pipeline Runner** page to execute runs, "
            "or run the notebook to generate the first set of artifacts.")
    st.stop()

# --- Load Run History ---
lines = history_path.read_text().strip().split('\n')
runs = [json.loads(line) for line in lines if line.strip()]
run_df = pd.DataFrame(runs)
run_df['run_index'] = range(len(run_df))

# --- Run History Table ---
st.subheader("Run History")

display_cols = [c for c in [
    'run_id', 'timestamp', 'n_components', 'n_neighbors',
    'min_cluster_size', 'min_samples', 'epsilon', 'metric',
    'silhouette', 'davies_bouldin', 'calinski_harabasz',
    'n_clusters', 'outlier_pct',
] if c in run_df.columns]

# Highlight best silhouette
if 'silhouette' in run_df.columns:
    best_idx = run_df['silhouette'].idxmax()
    st.caption(f"Best silhouette: run {run_df.loc[best_idx, 'run_id']} ({run_df.loc[best_idx, 'silhouette']:.4f})")

st.dataframe(run_df[display_cols], use_container_width=True, hide_index=True)

st.markdown("---")

# --- Metric History Charts ---
st.subheader("Metric History")
metric_cols = [c for c in ['silhouette', 'outlier_pct', 'n_clusters', 'davies_bouldin'] if c in run_df.columns]

if len(run_df) >= 2:
    tabs = st.tabs(metric_cols)
    for tab, mcol in zip(tabs, metric_cols):
        with tab:
            fig = metric_line_chart(run_df, 'run_index', mcol, title=f"{mcol} over runs")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Need at least 2 runs to show trend charts.")

st.markdown("---")

# --- Parameter Sensitivity ---
st.subheader("Parameter Sensitivity")
param_cols = [c for c in [
    'n_components', 'n_neighbors', 'min_cluster_size', 'min_samples',
    'epsilon', 'min_dist',
] if c in run_df.columns]

if param_cols and metric_cols:
    col1, col2 = st.columns(2)
    sel_metric = col1.selectbox("Metric:", metric_cols)
    sel_param = col2.selectbox("Parameter:", param_cols)

    import plotly.express as px
    fig = px.scatter(
        run_df, x=sel_param, y=sel_metric,
        hover_data=['run_id'], trendline='lowess',
        title=f"{sel_metric} vs {sel_param}",
    )
    fig.update_layout(height=400, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a",)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Run Comparison ---
st.subheader("Run Comparison")
if len(run_df) >= 2:
    run_ids = run_df['run_id'].tolist()
    col1, col2 = st.columns(2)
    run_a = col1.selectbox("Run A:", run_ids, index=len(run_ids) - 2)
    run_b = col2.selectbox("Run B:", run_ids, index=len(run_ids) - 1)

    if run_a != run_b:
        row_a = run_df[run_df['run_id'] == run_a].iloc[0]
        row_b = run_df[run_df['run_id'] == run_b].iloc[0]

        compare_cols = [c for c in display_cols if c not in ('run_id', 'timestamp')]
        comp_data = []
        for c in compare_cols:
            va = row_a.get(c, 'N/A')
            vb = row_b.get(c, 'N/A')
            try:
                diff = float(vb) - float(va)
                arrow = "+" if diff > 0 else ""
                comp_data.append({'Parameter': c, run_a: va, run_b: vb, 'Delta': f"{arrow}{diff:.4f}"})
            except (ValueError, TypeError):
                comp_data.append({'Parameter': c, run_a: va, run_b: vb, 'Delta': '-'})

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
else:
    st.caption("Need at least 2 runs to compare.")

st.markdown("---")

# --- Export ---
st.subheader("Export")
col1, col2 = st.columns(2)
csv_data = run_df.to_csv(index=False)
col1.download_button("Download run history (CSV)", csv_data, "run_history.csv", "text/csv")

if 'silhouette' in run_df.columns:
    best_row = run_df.loc[run_df['silhouette'].idxmax()].to_dict()
    col2.download_button("Download best config (JSON)",
                         json.dumps(best_row, indent=2, default=str),
                         "best_config.json", "application/json")
