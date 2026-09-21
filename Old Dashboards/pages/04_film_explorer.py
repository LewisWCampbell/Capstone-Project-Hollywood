"""Page 4 — Film Explorer: Per-film genome profile, soft membership, nearest neighbours."""

import streamlit as st
import numpy as np
import pandas as pd
from utils.data import (
    load_pipeline_artifacts, build_title_lookup, load_genres, load_omdb,
    get_poster_src,
)
from utils.plots import feature_radar, soft_membership_bar

st.set_page_config(page_title="Film Explorer", layout="wide")
st.title("Film Explorer")

arts = load_pipeline_artifacts()
if arts is None:
    st.error("Pipeline artifacts not found. Run the notebook or Pipeline Runner first.")
    st.stop()

# --- Unpack ---
X = arts['X']
X_weighted = arts.get('X_weighted', X)
feature_names = arts['feature_names']
tt_codes = arts['tt_codes']
cluster_labels = arts['cluster_labels']
soft_clusters = arts.get('soft_clusters')
embedding_cluster = arts['embedding_cluster']
cluster_names = arts.get('cluster_names', {})

title_lookup = build_title_lookup()
omdb = load_omdb()
genres = load_genres()
genre_map = dict(zip(genres['movie_id'], genres['genre']))

# --- Film Search ---
film_options = [f"{title_lookup.get(tt, tt)} ({tt})" for tt in tt_codes]
selected = st.selectbox("Search for a film:", film_options, index=0)
tt_selected = selected.split("(")[-1].rstrip(")")
film_idx = tt_codes.index(tt_selected) if tt_selected in tt_codes else 0
tt = tt_codes[film_idx]

# --- Film Header ---
title = title_lookup.get(tt, tt)
omdb_entry = omdb.get(tt, {})
year = omdb_entry.get('Year', 'N/A')
director = omdb_entry.get('Director', 'N/A')
genre_str = genre_map.get(tt, omdb_entry.get('Genre', 'Unknown'))
cl = cluster_labels[film_idx]
cl_name = cluster_names.get(cl, {}).get('name', f'Cluster {cl}') if cl >= 0 else 'Outlier'
_probabilities = arts.get('probabilities', arts['clusterer'].probabilities_ if 'clusterer' in arts else np.zeros(len(tt_codes)))
cl_prob = _probabilities[film_idx]

st.markdown(f"### {title} <span style='color: #888; font-size: 0.7em;'>({year})</span>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    # Show poster (local file first, then remote URL, then placeholder)
    st.image(get_poster_src(tt, omdb_entry), use_container_width=True)

with col2:
    st.markdown(f"**Director:** {director}")
    st.markdown(f"**Genres:** {genre_str}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Primary Rail", cl_name)
    m2.metric("Membership Confidence", f"{cl_prob:.1%}", delta="High" if cl_prob >= 0.7 else "Low", delta_color="normal" if cl_prob >= 0.5 else "off")
    
    outlier_scores = arts.get('outlier_scores')
    score = outlier_scores[film_idx] if outlier_scores is not None else 0
    m3.metric("Outlier Score", f"{score:.2f}", delta="Tier 3" if score > 0.8 else "Safe", delta_color="inverse" if score > 0.5 else "normal")
    
    st.markdown(f"_{omdb_entry.get('Plot', 'No plot available.')}_")

st.markdown("---")

# --- Genome Radar Chart ---
st.subheader("Genome Profile")
col1, col2 = st.columns([3, 1])

with col2:
    show_weighted = st.checkbox("Show IDF-weighted values", value=False)
    show_centroid = st.checkbox("Overlay cluster centroid", value=True)

    # Comparison film
    comp_options = ["None"] + film_options
    comp_selected = st.selectbox("Compare with:", comp_options, index=0)

values = X_weighted[film_idx] if show_weighted else X[film_idx]
overlay_values = None
overlay_name = None

if show_centroid and cl >= 0:
    centroid = arts.get('centroids_original', {}).get(cl)
    if centroid is not None:
        if show_weighted:
            # Approximate weighted centroid
            cluster_mask = cluster_labels == cl
            overlay_values = X_weighted[cluster_mask].mean(axis=0)
        else:
            overlay_values = centroid
        overlay_name = f"Centroid: {cl_name}"

with col1:
    fig = feature_radar(
        values, feature_names,
        title=f"Genome: {title}",
        overlay_values=overlay_values,
        overlay_name=overlay_name,
        top_n=20,
    )
    st.plotly_chart(fig, use_container_width=True)

# Comparison film radar
if comp_selected != "None":
    comp_tt = comp_selected.split("(")[-1].rstrip(")")
    if comp_tt in tt_codes:
        comp_idx = tt_codes.index(comp_tt)
        comp_values = X_weighted[comp_idx] if show_weighted else X[comp_idx]
        fig = feature_radar(
            values, feature_names,
            title=f"{title} vs {title_lookup.get(comp_tt, comp_tt)}",
            overlay_values=comp_values,
            overlay_name=title_lookup.get(comp_tt, comp_tt),
            top_n=20,
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Soft Membership Bar ---
if soft_clusters is not None:
    st.subheader("Soft Cluster Membership")
    fig = soft_membership_bar(soft_clusters[film_idx], cluster_names)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Nearest Neighbours ---
st.subheader("Nearest Neighbours")
n_neighbours = st.slider("Number of neighbours:", 5, 30, 10)

# Compute distances in embedding space
from scipy.spatial.distance import cdist
dists = cdist(embedding_cluster[film_idx:film_idx+1], embedding_cluster, metric='euclidean')[0]
sorted_idx = np.argsort(dists)[1:n_neighbours+1]  # skip self

max_dist = dists[sorted_idx[-1]] if len(sorted_idx) > 0 else 1.0
neighbour_data = []
for idx in sorted_idx:
    ntt = tt_codes[idx]
    similarity = max(0, (1 - dists[idx] / (max_dist * 1.5))) * 100
    ncl = cluster_labels[idx]
    ncl_name = cluster_names.get(ncl, {}).get('name', f'Cluster {ncl}') if ncl >= 0 else 'Outlier'
    neighbour_data.append({
        'Title': title_lookup.get(ntt, ntt),
        'TT Code': ntt,
        'Similarity': f"{similarity:.0f}%",
        'Distance': f"{dists[idx]:.3f}",
        'Rail': ncl_name,
        'Genre': genre_map.get(ntt, 'Unknown'),
    })

st.dataframe(pd.DataFrame(neighbour_data), use_container_width=True, hide_index=True)

st.markdown("---")

# --- Genre Tags ---
st.subheader("Genre Flags")
genre_str_full = genre_map.get(tt, '')
tags_html = ""
for g in genre_str_full.split(','):
    g = g.strip()
    if g:
        tags_html += f"<span style='background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:4px; margin-right:5px; font-size:0.8em;'>🎬 {g}</span>"
if tags_html:
    st.markdown(tags_html, unsafe_allow_html=True)

# --- Outlier Assessment ---
if cl == -1:
    st.markdown("---")
    st.subheader("Outlier Assessment")
    outlier_scores = arts.get('outlier_scores')
    score = outlier_scores[film_idx] if outlier_scores is not None else 0

    tier = "Unknown"
    if arts.get('tier3_mask') is not None and arts['tier3_mask'][film_idx]:
        tier = "Tier 3 (True Outlier)"
    elif arts.get('tier2_mask') is not None and arts['tier2_mask'][film_idx]:
        tier = "Tier 2 (Moderate)"
    elif arts.get('tier1_mask') is not None and arts['tier1_mask'][film_idx]:
        tier = "Tier 1 (Mild)"

    col1, col2 = st.columns(2)
    col1.metric("Outlier Tier", tier)
    col2.metric("Outlier Score", f"{score:.3f}")

    # Nearest centroid
    centroids = arts.get('centroids_umap', {})
    if centroids:
        centroid_dists = {
            cid: np.linalg.norm(embedding_cluster[film_idx] - c)
            for cid, c in centroids.items()
        }
        nearest_cid = min(centroid_dists, key=centroid_dists.get)
        nearest_name = cluster_names.get(nearest_cid, {}).get('name', f'Cluster {nearest_cid}')
        st.info(f"Nearest cluster: **{nearest_name}** (distance: {centroid_dists[nearest_cid]:.3f})")
