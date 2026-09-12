"""Page 4 — Film Explorer: Per-film genome profile, soft membership, nearest neighbours."""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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

# --- Film Search (supports ?tt= query param from Cluster Explorer clicks) ---
film_options = [f"{title_lookup.get(tt, tt)} ({tt})" for tt in tt_codes]

# Check for query parameter from cluster explorer click-through
query_tt = st.query_params.get("tt", None)
default_idx = 0
if query_tt and query_tt in tt_codes:
    default_idx = tt_codes.index(query_tt)

selected = st.selectbox("Search for a film:", film_options, index=default_idx)
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
    # Show poster
    _poster_src = get_poster_src(tt, omdb_entry, size='300x450')
    if _poster_src.startswith('http') or _poster_src.startswith('./'):
        st.markdown(
            f"<img src='{_poster_src}' style='width:100%; border-radius:8px;' alt='poster'>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='height:300px; width:100%; background:#1e232d; display:flex; "
            "align-items:center; justify-content:center; border-radius:8px;'>No Poster</div>",
            unsafe_allow_html=True,
        )

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

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR MOVIES (KNN in feature space)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Movies Like This")

knn_col1, knn_col2, knn_col3 = st.columns([1, 1, 2])
with knn_col1:
    n_neighbours = st.slider("Number of results:", 5, 50, 20)
with knn_col2:
    knn_space = st.selectbox("Search space:", ["Feature (Genome)", "Embedding (UMAP)"],
                              help="Feature space uses the full genome profile; Embedding space uses the UMAP-reduced coordinates")
with knn_col3:
    knn_scope = st.selectbox("Scope:", ["All Movies", "Same Rail Only", "Different Rails Only"])

# Compute distances
if knn_space == "Feature (Genome)":
    _search_matrix = X_weighted
    _metric = 'cosine'
else:
    _search_matrix = embedding_cluster
    _metric = 'euclidean'

dists = cdist(_search_matrix[film_idx:film_idx+1], _search_matrix, metric=_metric)[0]

# Build candidate list (skip self)
candidates = []
for idx in range(len(tt_codes)):
    if idx == film_idx:
        continue
    ncl = cluster_labels[idx]
    if knn_scope == "Same Rail Only" and ncl != cl:
        continue
    if knn_scope == "Different Rails Only" and ncl == cl:
        continue
    candidates.append((idx, dists[idx]))

# Sort by distance and take top N
candidates.sort(key=lambda x: x[1])
neighbours = candidates[:n_neighbours]

# Compute similarity scores (normalised against the max distance in the result set)
max_dist = neighbours[-1][1] if neighbours else 1.0

# Build the visual poster rail
knn_rail_html = """
<style>
.knn-container {
    display: flex;
    overflow-x: auto;
    gap: 15px;
    padding-bottom: 15px;
    padding-top: 10px;
    -ms-overflow-style: none;
    scrollbar-width: none;
}
.knn-container::-webkit-scrollbar { display: none; }
.knn-card {
    flex: 0 0 auto;
    width: 140px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.knn-poster {
    width: 140px;
    height: 210px;
    object-fit: cover;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.knn-poster:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 12px rgba(156, 39, 176, 0.6);
}
.knn-title {
    margin-top: 8px;
    font-size: 0.85em;
    font-weight: 600;
    line-height: 1.2;
    word-wrap: break-word;
    max-width: 100%;
    height: 2.4em;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.knn-meta { font-size: 0.75em; margin-top: 2px; color: #888; }
.knn-sim { font-size: 0.8em; margin-top: 1px; }
</style>
<div class="knn-container">
"""

for rank, (idx, dist) in enumerate(neighbours, 1):
    ntt = tt_codes[idx]
    ntitle = title_lookup.get(ntt, ntt)
    n_omdb = omdb.get(ntt, {})
    n_year = n_omdb.get('Year', '')
    ncl = cluster_labels[idx]
    ncl_name = cluster_names.get(ncl, {}).get('name', f'Cluster {ncl}') if ncl >= 0 else 'Outlier'

    similarity = max(0, (1 - dist / (max_dist * 1.2))) * 100
    if similarity >= 75:
        sim_color = "#a3be8c"
    elif similarity >= 50:
        sim_color = "#ebcb8b"
    else:
        sim_color = "#d08770"

    img_src = get_poster_src(ntt, n_omdb)
    safe_title = ntitle.replace('"', '&quot;').replace("'", "&#39;")
    n_year_label = f" ({n_year})" if n_year else ""
    film_link = f"/film_explorer?tt={ntt}"

    knn_rail_html += f'''
<a href="{film_link}" target="_self" style="text-decoration: none; color: inherit;">
<div class="knn-card" title="{safe_title}{n_year_label} — {ncl_name} — Similarity: {similarity:.0f}%">
    <img class="knn-poster" src="{img_src}" alt="{safe_title} poster" loading="lazy">
    <div class="knn-title">{safe_title}</div>
    <div class="knn-meta">{n_year if n_year else "?"} · {ncl_name}</div>
    <div class="knn-sim" style="color: {sim_color};">{similarity:.0f}% match</div>
</div>
</a>'''

knn_rail_html += '</div>'
st.markdown(knn_rail_html, unsafe_allow_html=True)

# Expandable detailed table
with st.expander("Detailed neighbour data"):
    neighbour_data = []
    for rank, (idx, dist) in enumerate(neighbours, 1):
        ntt = tt_codes[idx]
        similarity = max(0, (1 - dist / (max_dist * 1.2))) * 100
        ncl = cluster_labels[idx]
        ncl_name = cluster_names.get(ncl, {}).get('name', f'Cluster {ncl}') if ncl >= 0 else 'Outlier'
        neighbour_data.append({
            'Rank': rank,
            'Title': title_lookup.get(ntt, ntt),
            'Year': omdb.get(ntt, {}).get('Year', ''),
            'TT Code': ntt,
            'Similarity': f"{similarity:.0f}%",
            'Distance': f"{dist:.4f}",
            'Rail': ncl_name,
            'Genre': genre_map.get(ntt, 'Unknown'),
        })
    st.dataframe(pd.DataFrame(neighbour_data), use_container_width=True, hide_index=True)

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
