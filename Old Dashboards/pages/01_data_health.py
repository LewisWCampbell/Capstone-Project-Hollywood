"""Page 1 — Data Health: Sparsity, coverage, feature diagnostics."""

import streamlit as st
import numpy as np
import pandas as pd
from utils.data import (
    load_pipeline_artifacts, load_taxonomy_categorized,
    load_genres, get_feature_matrix_from_artifacts, build_title_lookup,
)
from utils.plots import sparsity_heatmap, distribution_histogram

st.set_page_config(page_title="Data Health", layout="wide")
st.title("Data Health")
st.markdown("Validate the genome before running the pipeline. Catch annotation gaps, sparsity anomalies, and feature distribution issues.")

# --- Load Data ---
X, feature_names, tt_codes, continuous_cols, binary_cols = get_feature_matrix_from_artifacts()
tax_cat = load_taxonomy_categorized()
n_films, n_features = X.shape

# --- Corpus Summary Cards ---
sparsity = (X == 0).mean()
mean_annotations = (X > 0).sum(axis=1).mean()
n_continuous = len(continuous_cols) if continuous_cols else n_features
n_binary = len(binary_cols) if binary_cols else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Films", f"{n_films:,}")
col2.metric("Features", f"{n_features} ({n_continuous}C + {n_binary}B)")
col3.metric("Sparsity", f"{sparsity:.1%}")
col4.metric("Mean annotations / film", f"{mean_annotations:.0f}")

st.markdown("---")

# --- Sparsity Heatmap ---
st.subheader("Sparsity Heatmap")
view_mode = st.radio("Show features:", ["All", "Continuous only", "Binary only"], horizontal=True)

if view_mode == "Continuous only" and continuous_cols:
    col_idx = [feature_names.index(c) for c in continuous_cols if c in feature_names]
    heatmap_data = X[:, col_idx]
    heatmap_names = [feature_names[i] for i in col_idx]
elif view_mode == "Binary only" and binary_cols:
    col_idx = [feature_names.index(c) for c in binary_cols if c in feature_names]
    heatmap_data = X[:, col_idx]
    heatmap_names = [feature_names[i] for i in col_idx]
else:
    heatmap_data = X
    heatmap_names = feature_names

fig = sparsity_heatmap(heatmap_data, heatmap_names, tt_codes)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Feature Distribution Panel ---
st.subheader("Feature Distribution")
selected_feature = st.selectbox("Select a feature:", feature_names)
feat_idx = feature_names.index(selected_feature)
feat_values = X[:, feat_idx]
pct_nonzero = (feat_values > 0).mean()

col1, col2 = st.columns([3, 1])
with col1:
    fig = distribution_histogram(feat_values, selected_feature)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.metric("Non-zero %", f"{pct_nonzero:.1%}")
    st.metric("Mean", f"{feat_values.mean():.3f}")
    st.metric("Std", f"{feat_values.std():.3f}")
    if pct_nonzero < 0.05:
        st.warning("< 5% non-zero — candidate for removal")
    if feat_values.std() < 0.01:
        st.warning("Near-zero variance")

st.markdown("---")

# --- IDF Weight Preview ---
st.subheader("IDF Weight Preview")
arts = load_pipeline_artifacts()
if arts and arts.get('combined_weights') is not None:
    weights = arts['combined_weights']
    cont_names = arts.get('continuous_cols', feature_names[:len(weights)])
    weight_df = pd.DataFrame({
        'Feature': cont_names[:len(weights)],
        'Combined Weight': weights,
    }).sort_values('Combined Weight', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 30 highest-weighted features**")
        top30 = weight_df.head(30)
        import plotly.express as px
        fig = px.bar(top30, x='Combined Weight', y='Feature', orientation='h',
                     title="Highest IDF x Variance Weight")
        fig.update_layout(height=600, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Bottom 30 lowest-weighted features**")
        bot30 = weight_df.tail(30)
        fig = px.bar(bot30, x='Combined Weight', y='Feature', orientation='h',
                     title="Lowest IDF x Variance Weight")
        fig.update_layout(height=600, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run the pipeline with IDF weighting enabled to see weight previews.")

st.markdown("---")

# --- Per-Film Annotation Density ---
st.subheader("Per-Film Annotation Density")
annotations_per_film = (X > 0).sum(axis=1)
import plotly.express as px
fig = px.histogram(annotations_per_film, nbins=50,
                   labels={"value": "Non-zero features per film", "count": "Number of films"},
                   title="Annotation Density Distribution")
fig.update_layout(height=400, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

low_density = (annotations_per_film < np.percentile(annotations_per_film, 5)).sum()
st.caption(f"{low_density} films have fewer than {np.percentile(annotations_per_film, 5):.0f} non-zero features (bottom 5%) — high outlier risk.")

st.markdown("---")

# --- Variance Table ---
st.subheader("Variance Table")
presence_pct = (X > 0).mean(axis=0)
variance = X.var(axis=0)

# Compute IDF if available
if arts and arts.get('idf_weights') is not None and arts.get('combined_weights') is not None:
    idf_col = np.zeros(n_features)
    cw_col = np.zeros(n_features)
    n_idf = len(arts['idf_weights'])
    idf_col[:n_idf] = arts['idf_weights']
    cw_col[:n_idf] = arts['combined_weights']
else:
    idf_col = np.log(n_films / ((X > 0).sum(axis=0) + 1))
    cw_col = idf_col * np.sqrt(variance)

var_df = pd.DataFrame({
    'Feature': feature_names,
    'Presence %': (presence_pct * 100).round(1),
    'Variance': variance.round(4),
    'IDF Weight': idf_col.round(4),
    'Combined Weight': cw_col.round(4),
}).sort_values('Combined Weight', ascending=False)

var_threshold = st.slider("Show features with variance below:", 0.0, 1.0, 1.0, 0.01)
filtered = var_df[var_df['Variance'] <= var_threshold]
st.dataframe(filtered, use_container_width=True, height=400)

csv = filtered.to_csv(index=False)
st.download_button("Download filtered table as CSV", csv, "feature_variance.csv", "text/csv")
