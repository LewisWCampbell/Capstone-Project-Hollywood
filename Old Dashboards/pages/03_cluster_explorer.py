"""Page 3 — Cluster Explorer: Full-stack cluster visualization and inspection."""

import streamlit as st
import numpy as np
import pandas as pd
import json as json_mod
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from datetime import datetime
from pathlib import Path
from utils.data import (
    load_pipeline_artifacts, load_cluster_summary, load_sub_cluster_summary,
    load_genres, build_title_lookup, load_omdb, get_poster_src, RESULTS_DIR,
)
from utils.plots import (
    cluster_feature_bars, feature_radar, CLUSTER_COLOURS, OUTLIER_COLOURS,
)
from utils.llm import name_cluster, get_discriminative_features

# ── Run persistence helpers ──────────────────────────────────────────────────
RUNS_DIR = RESULTS_DIR / 'runs'


def _list_saved_runs() -> list[dict]:
    """Return list of saved runs sorted newest first."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted(RUNS_DIR.glob('run_*.json'), reverse=True):
        try:
            with open(f) as fh:
                meta = json_mod.load(fh)
            meta['_path'] = str(f)
            meta['_filename'] = f.stem
            runs.append(meta)
        except Exception:
            pass
    return runs


def _next_run_id() -> int:
    """Return the next available run ID number."""
    existing = list(RUNS_DIR.glob('run_*.json'))
    if not existing:
        return 1
    ids = []
    for f in existing:
        try:
            ids.append(int(f.stem.split('_')[1]))
        except (IndexError, ValueError):
            pass
    return max(ids, default=0) + 1


def _save_run(run_name: str, cluster_labels_arr, working_names: dict,
              cluster_status: dict, config: dict, notes: str = '') -> str:
    """Save current working state to a JSON run file. Returns filename."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = _next_run_id()
    filename = f'run_{run_id:03d}.json'

    payload = {
        'run_id': run_id,
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'notes': notes,
        'n_clusters': int(len(set(cluster_labels_arr)) - (1 if -1 in cluster_labels_arr else 0)),
        'n_films': int(len(cluster_labels_arr)),
        'n_outliers': int((np.array(cluster_labels_arr) == -1).sum()),
        'cluster_labels': [int(x) for x in cluster_labels_arr],
        'working_names': {str(k): v for k, v in working_names.items()},
        'cluster_status': {str(k): v for k, v in cluster_status.items()},
        'config': config,
    }

    with open(RUNS_DIR / filename, 'w') as f:
        json_mod.dump(payload, f, indent=2)

    return filename


def _load_run(filepath: str) -> dict:
    """Load a saved run from disk."""
    with open(filepath) as f:
        return json_mod.load(f)

st.set_page_config(page_title="Cluster Explorer", layout="wide")

# --- Global Custom CSS for a Premium "Hollywood" Aesthetic ---
st.markdown("""
<style>
    /* Main background & glassmorphism elements */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sleek container for metrics and cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 35, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 215, 0, 0.4); /* Subtle gold hover */
    }

    /* Titles and text styling */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #FFD700, #FDB931);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2937, #111827);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #FFD700;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.2);
    }
    
    /* Adjust expander matching */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Interactive Cluster Explorer")

arts = load_pipeline_artifacts()
if arts is None:
    st.error("Pipeline artifacts not found. Run the notebook or Pipeline Runner first.")
    st.stop()

# ── Unpack artifacts ──────────────────────────────────────────────────────────
embedding_3d = arts['embedding_3d']
embedding_2d = arts.get('embedding_2d', embedding_3d[:, :2])
cluster_labels = arts['cluster_labels']
n_clusters = arts['n_clusters']
soft_clusters = arts.get('soft_clusters')
cluster_names = arts.get('cluster_names', {})
sub_clusters = arts.get('sub_clusters', {})
X = arts['X']
feature_names = arts['feature_names']
tt_codes = arts['tt_codes']
tier1_mask = arts.get('tier1_mask', np.zeros(len(tt_codes), dtype=bool))
tier2_mask = arts.get('tier2_mask', np.zeros(len(tt_codes), dtype=bool))
tier3_mask = arts.get('tier3_mask', np.zeros(len(tt_codes), dtype=bool))
outlier_scores = arts.get('outlier_scores', np.zeros(len(tt_codes)))
probabilities = arts.get('probabilities', arts['clusterer'].probabilities_ if 'clusterer' in arts else np.zeros(len(tt_codes)))

title_lookup = build_title_lookup()
omdb = load_omdb()
genres = load_genres()
genre_map = dict(zip(genres['movie_id'], genres['genre']))
summary = load_cluster_summary()

# ── Per-movie language & rating lookups (from OMDB) ──────────────────────────
movie_language = {}
movie_rating = {}
for tt in tt_codes:
    entry = omdb.get(tt, {})
    lang_raw = entry.get('Language', 'Unknown')
    movie_language[tt] = lang_raw.split(',')[0].strip() if lang_raw else 'Unknown'
    movie_rating[tt] = entry.get('Rated', 'Unknown') or 'Unknown'

all_languages = sorted(set(movie_language.values()))
all_ratings = sorted(set(movie_rating.values()))

# ── Pre-compute per-cluster stats ─────────────────────────────────────────────
global_mean = X.mean(axis=0)
cluster_ids_sorted = summary.sort_values('movie_count', ascending=False)['cluster_id'].tolist()

def _colour(cid):
    return CLUSTER_COLOURS[cid % len(CLUSTER_COLOURS)]

# Initialize Session State
if 'run_id' not in st.session_state: st.session_state.run_id = 'default'
if 'action_history' not in st.session_state: st.session_state.action_history = []
if 'unsaved_changes' not in st.session_state: st.session_state.unsaved_changes = 0
if 'selected_cluster' not in st.session_state: st.session_state.selected_cluster = None
if 'selected_film' not in st.session_state: st.session_state.selected_film = None
if 'cluster_status' not in st.session_state:
    # Initialize all as 'draft'
    st.session_state.cluster_status = {cid: 'draft' for cid in cluster_ids_sorted}
if 'working_names' not in st.session_state:
    st.session_state.working_names = {
        cid: cluster_names.get(cid, {}).get('name', f'Cluster {cid}')
        for cid in cluster_ids_sorted if cid >= 0
    }
if 'rename_target' not in st.session_state: st.session_state.rename_target = None
if 'merge_source' not in st.session_state: st.session_state.merge_source = None
if 'delete_confirm' not in st.session_state: st.session_state.delete_confirm = None
if 'autoname_running' not in st.session_state: st.session_state.autoname_running = None
if 'show_save_dialog' not in st.session_state: st.session_state.show_save_dialog = False
if 'loaded_run_id' not in st.session_state: st.session_state.loaded_run_id = None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOOLBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
# Row 1 of Toolbar
tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([1, 1, 1, 1])

with tb_col1:
    colour_by = st.selectbox("Colour by:", ["Cluster", "Genre", "Outlier Tier", "Confidence", "Membership Entropy"], label_visibility="collapsed", placeholder="Colour by ▾")

with tb_col2:
    point_size_method = st.selectbox("Point size:", ["Uniform", "By Outlier Score", "By Confidence"], label_visibility="collapsed")

with tb_col3:
    view_filter = st.selectbox("View Filter:", ["All Films", "Only Outliers", "Only Sub-clusters"], label_visibility="collapsed")

with tb_col4:
    # Build search list: "Title (Year) [tt1234]"
    search_options = [""] + [f"{title_lookup.get(tt, tt)} [{tt}]" for tt in tt_codes]
    search_query = st.selectbox("🔍 Search film", search_options, label_visibility="collapsed", placeholder="🔍 Search film")
    
# Row 2 of Toolbar
tb2_col1, tb2_col2, tb2_col3, tb2_col4 = st.columns([1, 1, 1, 1])

with tb2_col1:
    select_mode = st.selectbox("Select mode:", ["Click", "Lasso", "Box"], label_visibility="collapsed")
    dragmode = select_mode.lower() if select_mode != 'Click' else 'select'

with tb2_col2:
    st.button(f"← Undo {'(' + str(len(st.session_state.action_history)) + ')' if st.session_state.action_history else ''}", disabled=len(st.session_state.action_history) == 0)

with tb2_col3:
    save_label = f"💾 Save ({st.session_state.unsaved_changes} unsaved)" if st.session_state.unsaved_changes > 0 else "💾 Save"
    if st.button(save_label, type="primary", disabled=st.session_state.unsaved_changes == 0):
        st.session_state.show_save_dialog = True
        st.rerun()

with tb2_col4:
    # ── Run ID selector ──────────────────────────────────────────────────────
    saved_runs = _list_saved_runs()
    run_options = ['Current (unsaved)'] + [
        f"Run {r['run_id']:03d}: {r.get('run_name', 'Unnamed')} ({r['timestamp'][:10]})"
        for r in saved_runs
    ]
    selected_run_label = st.selectbox("Load run:", run_options, label_visibility="collapsed",
                                       key="run_selector")

    # If user selected a saved run (not "Current"), load it
    if selected_run_label != 'Current (unsaved)':
        sel_idx = run_options.index(selected_run_label) - 1  # offset for "Current"
        if sel_idx >= 0 and sel_idx < len(saved_runs):
            run_data = saved_runs[sel_idx]
            # Only reload if we haven't already loaded this run
            if st.session_state.get('loaded_run_id') != run_data['run_id']:
                loaded = _load_run(run_data['_path'])
                # Restore cluster labels
                loaded_labels = np.array(loaded['cluster_labels'])
                cluster_labels[:] = loaded_labels
                # Restore working names
                st.session_state.working_names = {
                    int(k): v for k, v in loaded['working_names'].items()
                }
                # Restore cluster status
                st.session_state.cluster_status = {
                    int(k): v for k, v in loaded['cluster_status'].items()
                }
                st.session_state.loaded_run_id = run_data['run_id']
                st.session_state.unsaved_changes = 0
                st.session_state.run_id = f"run_{run_data['run_id']:03d}"
                st.toast(f"Loaded Run {run_data['run_id']:03d}: {run_data.get('run_name', '')}")
                st.rerun()

# Row 3 — Language & Rating Filters
tb3_col1, tb3_col2, _ = st.columns([2, 2, 4])
with tb3_col1:
    selected_languages = st.multiselect(
        "Filter by language:", all_languages, default=all_languages,
        key="lang_filter", placeholder="All languages",
    )
with tb3_col2:
    selected_ratings = st.multiselect(
        "Filter by rating:", all_ratings, default=all_ratings,
        key="rating_filter", placeholder="All ratings",
    )

# ── Save Dialog ──────────────────────────────────────────────────────────────
if st.session_state.get('show_save_dialog', False):
    @st.dialog("Save Run")
    def save_dialog():
        st.markdown("Save the current cluster assignments, names, and statuses as a named run.")
        run_name = st.text_input("Run name:", placeholder="e.g. zscore_15clusters_v2",
                                  key="save_run_name")
        notes = st.text_area("Notes (optional):", placeholder="What's different about this run?",
                              key="save_run_notes", height=80)

        # Gather config snapshot from artifacts
        config_snapshot = arts.get('config', {})

        c1, c2 = st.columns(2)
        if c1.button("Save", type="primary", use_container_width=True, key="save_confirm"):
            if not run_name:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
            filename = _save_run(
                run_name=run_name,
                cluster_labels_arr=cluster_labels,
                working_names=st.session_state.working_names,
                cluster_status=st.session_state.cluster_status,
                config=config_snapshot,
                notes=notes,
            )
            st.session_state.unsaved_changes = 0
            st.session_state.show_save_dialog = False
            st.session_state.loaded_run_id = _next_run_id() - 1
            st.toast(f"Saved as {filename}")
            st.rerun()
        if c2.button("Cancel", use_container_width=True, key="save_cancel"):
            st.session_state.show_save_dialog = False
            st.rerun()

    save_dialog()


# ── Process actions from Toolbar ──
# If a film was searched, update session state
if search_query:
    tt_match = search_query.split('[')[-1].strip(']')
    st.session_state.selected_film = tt_match
elif st.session_state.selected_film and not search_query:
    # If search is cleared but we had a film selected, keep it selected unless they double click map
    pass 

# ── Build colour arrays ──
def build_colours(colour_by):
    if colour_by == "Cluster":
        vals = []
        for cl in cluster_labels:
            if cl >= 0:
                vals.append(cluster_names.get(cl, {}).get('name', f'Cluster {cl}'))
            else:
                vals.append('Outlier')
        return vals, True
    elif colour_by == "Genre":
        return [genre_map.get(tt, 'Unknown').split(',')[0].strip() for tt in tt_codes], True
    elif colour_by == "Outlier Tier":
        vals = []
        for i in range(len(tt_codes)):
            if tier3_mask[i]:
                vals.append('Tier 3 (True)')
            elif tier2_mask[i]:
                vals.append('Tier 2 (Moderate)')
            elif tier1_mask[i]:
                vals.append('Tier 1 (Mild)')
            elif cluster_labels[i] == -1:
                vals.append('Outlier')
            else:
                vals.append('Clustered')
        return vals, True
    elif colour_by == "Confidence":
        return probabilities.tolist(), False
    else:  # Entropy
        if soft_clusters is not None:
            entropy = -np.sum(soft_clusters * np.log(soft_clusters + 1e-10), axis=1)
            return entropy.tolist(), False
        return probabilities.tolist(), False

colour_values, colour_categorical = build_colours(colour_by)

# Filter mask based on View Filter
vis_mask = np.ones(len(tt_codes), dtype=bool)
if view_filter == "Only Outliers":
    vis_mask = vis_mask & (cluster_labels == -1)
elif view_filter == "Only Sub-clusters":
    # Identify items that belong to a sub-cluster
    has_sub = np.zeros(len(tt_codes), dtype=bool)
    for cid in sub_clusters:
        idx = np.where(cluster_labels == cid)[0]
        has_sub[idx] = True
    vis_mask = vis_mask & has_sub

# Apply language & rating filters
if selected_languages and len(selected_languages) < len(all_languages):
    lang_mask = np.array([movie_language.get(tt, 'Unknown') in selected_languages for tt in tt_codes])
    vis_mask = vis_mask & lang_mask
if selected_ratings and len(selected_ratings) < len(all_ratings):
    rating_mask = np.array([movie_rating.get(tt, 'Unknown') in selected_ratings for tt in tt_codes])
    vis_mask = vis_mask & rating_mask

# Point sizes
point_sizes = np.full(len(tt_codes), 5)
if point_size_method == "By Outlier Score":
    point_sizes = 2 + (outlier_scores / outlier_scores.max()) * 10
elif point_size_method == "By Confidence":
    point_sizes = 2 + (probabilities * 8)

# Hover text
hover_text = []
for i, tt in enumerate(tt_codes):
    name = title_lookup.get(tt, tt)
    cl = cluster_labels[i]
    cl_name = cluster_names.get(cl, {}).get('name', f'Cluster {cl}') if cl >= 0 else 'Outlier'
    genre = genre_map.get(tt, 'Unknown')
    conf = f"{probabilities[i]:.0%}"
    hover_text.append(f"{name}<br>{cl_name}<br>{genre}<br>Conf: {conf}")
hover_arr = np.array(hover_text)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 2D UMAP CANVAS & CLUSTER INFO PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

col_map, col_info = st.columns([7, 3]) if st.session_state.selected_cluster is not None else (st.columns([1, 0.01]))

with col_map:
    # ── Main UMAP rendering ──
    fig_2d = go.Figure()

    # Draw Convex Hulls
    for cid in cluster_ids_sorted:
        pts = embedding_2d[cluster_labels == cid]
        if len(pts) < 4: continue
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            
            # Highlight chosen hull if selected
            is_sel = st.session_state.selected_cluster == cid
            c_rgb = "156, 39, 176" if is_sel else "150, 150, 150"
            fill_alpha = 0.25 if is_sel else 0.05
            
            fig_2d.add_trace(go.Scatter(
                x=hull_pts[:, 0], y=hull_pts[:, 1],
                fill='toself', fillcolor=f'rgba({c_rgb},{fill_alpha})',
                line=dict(color=f'rgb({c_rgb})', width=2 if is_sel else 1, dash='dot'),
                hoverinfo='skip', showlegend=False,
                name=f'cluster_{cid}_hull'
            ))
            
            # Centroid label
            cx, cy = pts.mean(axis=0)
            name = cluster_names.get(cid, {}).get('name', f'C{cid}')
            fig_2d.add_annotation(
                x=cx, y=cy,
                text=f"<b>{name}</b>", showarrow=False,
                font=dict(size=11, color='white' if is_sel else '#aaaaaa'),
                bgcolor='rgba(0,0,0,0.6)', borderpad=3,
            )
        except Exception as e:
            pass # Skip if hull fails (e.g. points co-linear)

    # Draw scatter points
    if colour_categorical:
        unique_vals = sorted(set(np.array(colour_values)[vis_mask]))
        for val in unique_vals:
            idx = [i for i in range(len(tt_codes)) if vis_mask[i] and colour_values[i] == val]
            if not idx: continue
            
            cdata = np.array(cluster_labels)[idx].reshape(-1, 1)
            
            # Highlighting logic: dim everything if looking at a specific cluster
            opacity = 0.8
            if st.session_state.selected_cluster is not None:
                # if this trace matches the selected cluster
                cname = cluster_names.get(st.session_state.selected_cluster, {}).get('name', f'Cluster {st.session_state.selected_cluster}')
                if val != cname: opacity = 0.15
                
            fig_2d.add_trace(go.Scattergl(
                x=embedding_2d[idx, 0], y=embedding_2d[idx, 1],
                mode='markers', name=str(val),
                marker=dict(size=point_sizes[idx], opacity=opacity, line=dict(width=0.5, color='#222')),
                text=hover_arr[idx], customdata=cdata,
                hovertemplate="%{text}<extra></extra>",
            ))
    else:
        idx = np.where(vis_mask)[0]
        cdata = np.array(cluster_labels)[idx].reshape(-1, 1)
        fig_2d.add_trace(go.Scattergl(
            x=embedding_2d[idx, 0], y=embedding_2d[idx, 1],
            mode='markers',
            marker=dict(
                size=point_sizes[idx], opacity=0.7,
                color=np.array(colour_values)[idx], colorscale='Plasma', colorbar=dict(title=colour_by),
            ),
            text=hover_arr[idx], customdata=cdata,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Highlight searched/selected film
    if st.session_state.selected_film:
        if st.session_state.selected_film in tt_codes.tolist():
            f_idx = np.where(tt_codes == st.session_state.selected_film)[0][0]
            if vis_mask[f_idx]:
                fig_2d.add_trace(go.Scattergl(
                    x=[embedding_2d[f_idx, 0]], y=[embedding_2d[f_idx, 1]],
                    mode='markers',
                    marker=dict(size=14, color='white', line=dict(width=2, color='#9c27b0')),
                    hoverinfo='skip', showlegend=False
                ))

    fig_2d.update_layout(
        height=600 if st.session_state.selected_cluster is None else 700, 
        template="plotly_dark",
        plot_bgcolor="#05000a", paper_bgcolor="#05000a",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        dragmode=dragmode,
        hovermode='closest'
    )

    event = st.plotly_chart(fig_2d, use_container_width=True, on_select="rerun", selection_mode=("points","lasso"))
    
    # Process Map Clicks
    if event and "selection" in event and event["selection"]["points"]:
        # Standard point click logic
        clicked_pt = event["selection"]["points"][0]
        if "customdata" in clicked_pt:
            clicked_cid = int(clicked_pt["customdata"][0])
            if clicked_cid >= 0:
                if st.session_state.selected_cluster != clicked_cid:
                    st.session_state.selected_cluster = clicked_cid
                    st.rerun()

    # Clear Selection button if selected
    if st.session_state.selected_cluster is not None or st.session_state.selected_film is not None:
        if st.button("✕ Reset Selection"):
            st.session_state.selected_cluster = None
            st.session_state.selected_film = None
            st.rerun()


with col_info:
    if st.session_state.selected_cluster is not None:
        cid = st.session_state.selected_cluster
        mask_sel = cluster_labels == cid
        c_name = st.session_state.working_names.get(cid, cluster_names.get(cid, {}).get('name', f'Cluster {cid}'))
        count = int(mask_sel.sum())
        
        # Calculate silhouette approximation (just using probabilities as a proxy for now)
        sil = float(probabilities[mask_sel].mean()) if count > 0 else 0
        status = st.session_state.cluster_status.get(cid, 'draft')
        status_icon = "✓" if status == 'confirmed' else "✎"
        
        st.markdown(f"### {c_name}")
        st.caption(f"**{count} films** | Proxy Sil: {sil:.2f} | {status_icon} {status.title()}")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Features", "Similar Clusters"])
        
        with tab1:
            st.markdown(f"**{count} films** mapped to this cluster.")
            
            # Genre breakdown
            st.markdown("<br>**GENRE DISTRIBUTION**", unsafe_allow_html=True)
            cluster_indices = np.where(mask_sel)[0]
            gen_list = [g.strip() for i in cluster_indices for g in genre_map.get(tt_codes[i], '').split(',') if g.strip()]
            if gen_list:
                gen_counts = pd.Series(gen_list).value_counts(normalize=True).head(5)
                # Create a small horizontal bar chart for genres
                for g, p in gen_counts.items():
                    bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
                    st.markdown(f"<span style='font-family: monospace; font-size: 0.85em; color: #a3be8c;'>{g[:12]:<12} {bar} {p:.0%}</span>", unsafe_allow_html=True)
                    
            st.markdown("<br>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            b1.button("View Rail", key="btn_rail", use_container_width=True)
            b2.button("Re-cluster", disabled=True, use_container_width=True, help="Coming soon")

            b3, b4 = st.columns(2)
            if b3.button("⊕ Merge into...", key="panel_merge", use_container_width=True):
                st.session_state.merge_source = cid
                st.rerun()
            if b4.button("⊗ Delete", key="panel_delete", use_container_width=True):
                st.session_state.delete_confirm = cid
                st.rerun()
            
            if st.button("✓ Confirm", type="primary", use_container_width=True):
                st.session_state.cluster_status[cid] = 'confirmed'
                st.session_state.unsaved_changes += 1
                st.rerun()
                
        with tab2:
            st.markdown("**TOP DEFINING FEATURES**")
            # Compute discriminative features (delta from global mean)
            cluster_mean_feats = X[mask_sel].mean(axis=0)
            delta = cluster_mean_feats - global_mean
            top_pos_idx = np.argsort(delta)[-10:][::-1]
            for i in top_pos_idx:
                feat = feature_names[i]
                val = delta[i]
                if val <= 0:
                    break
                bar_len = min(10, int(abs(val) * 20))
                bar = "█" * bar_len + "░" * (10 - bar_len)
                st.markdown(f"<span style='font-family: monospace; font-size: 0.85em; color: #ebcb8b;'>{feat[:20]:<20} {bar} +{val:.2f}</span>", unsafe_allow_html=True)
                
        with tab3:
            # Nearest Clusters
            st.markdown("**NEAREST CLUSTERS**", unsafe_allow_html=True)
            centroids_umap = arts.get('centroids_umap', {})
            if centroids_umap and cid in centroids_umap:
                c_pos = centroids_umap[cid]
                dists = []
                for other_cid, o_pos in centroids_umap.items():
                    if other_cid != cid and other_cid >= 0:
                        dist = np.linalg.norm(c_pos - o_pos)
                        dists.append((other_cid, dist))
                dists.sort(key=lambda x: x[1])
                for ocid, d in dists[:4]:
                    oname = cluster_names.get(ocid, {}).get('name', f'C{ocid}')[:25]
                    st.markdown(f"<div style='padding: 5px; background: rgba(255,255,255,0.05); border-radius: 5px; margin-bottom: 5px;'>• {oname} <span style='float:right; color:#888; font-size:0.8em;'>dist: {d:.2f}</span></div>", unsafe_allow_html=True)
            else:
                st.caption("Centroid data unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RAIL BROWSER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Rail Browser")

placeholder_url = "https://via.placeholder.com/140x210/1e232d/c9d1d9?text=No+Poster"

for cid in cluster_ids_sorted:
    if cid == -1: continue # Skip root outliers for now
    
    mask_c = cluster_labels == cid
    if mask_c.sum() == 0: continue
    
    count = int(mask_c.sum())
    filtered_count = int((mask_c & vis_mask).sum())
    c_name = st.session_state.working_names.get(cid, cluster_names.get(cid, {}).get('name', f'Cluster {cid}'))
    status = st.session_state.cluster_status.get(cid, 'draft')
    sil = float(probabilities[mask_c].mean())
    hex_color = f"#{int(_colour(cid).split('(')[1].split(',')[0]):02x}{int(_colour(cid).split(',')[1]):02x}{int(_colour(cid).split(',')[2].split(')')[0]):02x}" if _colour(cid).startswith('rgb') else _colour(cid)

    count_label = f"{filtered_count}/{count} films" if filtered_count < count else f"{count} films"

    # ── Row Header ──
    st.markdown(f"<h4 style='margin-bottom: 0;'><span style='color:{hex_color};'>●</span> {c_name} <span style='font-size:0.6em; color:#888; font-weight:normal;'>&nbsp;&nbsp;&nbsp;{count_label} &nbsp;|&nbsp; Sil: {sil:.2f} &nbsp;|&nbsp; {status}</span></h4>", unsafe_allow_html=True)
    
    # Action buttons for this row
    r_col1, r_col2, r_col3, r_col4, r_col5, _ = st.columns([1, 1.2, 1.2, 1, 1, 6])
    if r_col1.button("✎ Rename", key=f"rnm_{cid}"):
        st.session_state.rename_target = cid
        st.rerun()
    if r_col2.button("⚡ Auto-name", key=f"auto_{cid}"):
        st.session_state.autoname_running = cid
        st.rerun()
    if r_col3.button("⊕ Merge into...", key=f"mrg_{cid}"):
        st.session_state.merge_source = cid
        st.rerun()
    if r_col4.button("⊗ Delete", key=f"del_{cid}"):
        st.session_state.delete_confirm = cid
        st.rerun()
    if status != 'confirmed':
        if r_col5.button("✓ Confirm", key=f"cnf_{cid}"):
            st.session_state.cluster_status[cid] = 'confirmed'
            st.rerun()
            
    # ── Film Rail ──
    cluster_indices = np.where(mask_c)[0]
    probs_c = probabilities[cluster_indices]
    sorted_idx = np.argsort(probs_c)[::-1] # All films, sorted by confidence
    
    rail_html = """
<style>
.rail-container {
    display: flex;
    overflow-x: auto;
    gap: 15px;
    padding-bottom: 15px;
    padding-top: 10px;
    -ms-overflow-style: none;
    scrollbar-width: none;
}
.rail-container::-webkit-scrollbar { display: none; }
.film-card {
    flex: 0 0 auto;
    width: 140px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
}
.film-poster {
    width: 140px;
    height: 210px;
    object-fit: cover;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.film-poster:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 12px rgba(156, 39, 176, 0.6);
}
.film-title {
    margin-top: 8px;
    font-size: 0.9em;
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
.film-conf { font-size: 0.8em; color: #88c0d0; margin-top: 2px; }
</style>
<div class="rail-container">
"""
    
    shown = 0
    for local_i in sorted_idx:
        if shown >= 50: break  # Limit to top 50 per rail for performance
        gi = cluster_indices[local_i]
        if not vis_mask[gi]:
            continue
        tt = tt_codes[gi]
        title = title_lookup.get(tt, tt)
        omdb_entry = omdb.get(tt, {})
        conf = probs_c[local_i]
        shown += 1

        # Color code confidence
        if conf >= 0.8: conf_color = "#a3be8c" # green
        elif conf >= 0.5: conf_color = "#ebcb8b" # yellow
        else: conf_color = "#d08770" # orange

        img_src = get_poster_src(tt, omdb_entry)
        safe_title = title.replace('"', '&quot;').replace("'", "&#39;")
        
        card_html = f"""
<div class="film-card" title="{safe_title}">
    <img class="film-poster" src="{img_src}" alt="{safe_title} poster">
    <div class="film-title">{safe_title}</div>
    <div class="film-conf" style="color: {conf_color};">{conf:.0%}</div>
</div>
"""
        rail_html += card_html
        
    rail_html += "</div>"
    st.markdown(rail_html, unsafe_allow_html=True)

    # ── Sub-cluster Rails (nested under main cluster) ──
    if cid in sub_clusters:
        sub_labels_arr = np.array(sub_clusters[cid])
        sub_ids = sorted(set(sub_labels_arr[sub_labels_arr >= 0]))
        if sub_ids:
            cluster_indices_all = np.where(mask_c)[0]
            sub_names_dict = arts.get('sub_cluster_names', {})
            with st.expander(f"▸ {len(sub_ids)} sub-clusters", expanded=False):
                for sid in sub_ids:
                    sub_mask_local = sub_labels_arr == sid
                    sub_global_idx = cluster_indices_all[sub_mask_local]
                    sub_count = len(sub_global_idx)
                    sub_filtered = int(sum(vis_mask[gi] for gi in sub_global_idx))
                    sub_name_info = sub_names_dict.get((cid, sid), sub_names_dict.get(str((cid, sid)), {}))
                    sub_name = sub_name_info.get('name', f'Sub-cluster {sid}') if isinstance(sub_name_info, dict) else f'Sub-cluster {sid}'
                    sub_count_label = f"{sub_filtered}/{sub_count} films" if sub_filtered < sub_count else f"{sub_count} films"

                    st.markdown(
                        f"<h5 style='margin: 8px 0 2px 20px; color: #88c0d0;'>↳ {sub_name} "
                        f"<span style='font-size:0.7em; color:#888; font-weight:normal;'>"
                        f"{sub_count_label}</span></h5>",
                        unsafe_allow_html=True,
                    )

                    # Build sub-cluster rail HTML
                    sub_probs = probabilities[sub_global_idx]
                    sub_sorted = np.argsort(sub_probs)[::-1]

                    sub_rail_html = '<div class="rail-container">'
                    sub_shown = 0
                    for si in sub_sorted:
                        if sub_shown >= 30: break
                        gi = sub_global_idx[si]
                        if not vis_mask[gi]:
                            continue
                        tt = tt_codes[gi]
                        sub_shown += 1
                        title = title_lookup.get(tt, tt)
                        omdb_entry = omdb.get(tt, {})
                        conf = sub_probs[si]
                        if conf >= 0.8: conf_color = "#a3be8c"
                        elif conf >= 0.5: conf_color = "#ebcb8b"
                        else: conf_color = "#d08770"
                        img_src = get_poster_src(tt, omdb_entry)
                        safe_title = title.replace('"', '&quot;').replace("'", "&#39;")
                        sub_rail_html += f'''
<div class="film-card" title="{safe_title}">
    <img class="film-poster" src="{img_src}" alt="{safe_title} poster" style="width:120px; height:180px;">
    <div class="film-title" style="font-size:0.8em;">{safe_title}</div>
    <div class="film-conf" style="color: {conf_color}; font-size:0.75em;">{conf:.0%}</div>
</div>'''
                    sub_rail_html += '</div>'
                    st.markdown(sub_rail_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3b — ACTION DIALOGS (Rename / Auto-name / Merge / Delete)
# ══════════════════════════════════════════════════════════════════════════════

# ── Rename Dialog ──
if st.session_state.rename_target is not None:
    ren_cid = st.session_state.rename_target
    old_name = st.session_state.working_names.get(ren_cid, f'Cluster {ren_cid}')

    @st.dialog(f"Rename: {old_name}")
    def rename_dialog():
        new_name = st.text_input("New name:", value=old_name, key="rename_input")
        c1, c2 = st.columns(2)
        if c1.button("Save", type="primary", use_container_width=True):
            st.session_state.working_names[ren_cid] = new_name
            # Also update the cluster_names dict so hover text etc. update
            if ren_cid in cluster_names:
                cluster_names[ren_cid]['name'] = new_name
            else:
                cluster_names[ren_cid] = {'name': new_name}
            st.session_state.unsaved_changes += 1
            st.session_state.rename_target = None
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            st.session_state.rename_target = None
            st.rerun()

    rename_dialog()

# ── Auto-name Dialog ──
if st.session_state.autoname_running is not None:
    auto_cid = st.session_state.autoname_running

    @st.dialog(f"Auto-naming Cluster {auto_cid}")
    def autoname_dialog():
        st.info("Querying local Ollama model for a cluster name...")
        mask_auto = cluster_labels == auto_cid
        sample_idx = np.where(mask_auto)[0][:10]
        sample_tts = [tt_codes[i] for i in sample_idx]

        with st.spinner("Generating name via Ollama..."):
            result = name_cluster(auto_cid, sample_tts, X, cluster_labels, feature_names, title_lookup)

        suggested_name = result.get('name', f'Cluster {auto_cid}')
        suggested_desc = result.get('description', '')

        st.success(f"**Suggested name:** {suggested_name}")
        if suggested_desc:
            st.caption(suggested_desc)

        final_name = st.text_input("Edit name if desired:", value=suggested_name, key="autoname_edit")
        c1, c2 = st.columns(2)
        if c1.button("Accept", type="primary", use_container_width=True, key="autoname_accept"):
            st.session_state.working_names[auto_cid] = final_name
            if auto_cid in cluster_names:
                cluster_names[auto_cid]['name'] = final_name
                if suggested_desc:
                    cluster_names[auto_cid]['description'] = suggested_desc
            else:
                cluster_names[auto_cid] = {'name': final_name, 'description': suggested_desc}
            st.session_state.unsaved_changes += 1
            st.session_state.autoname_running = None
            st.rerun()
        if c2.button("Cancel", use_container_width=True, key="autoname_cancel"):
            st.session_state.autoname_running = None
            st.rerun()

    autoname_dialog()

# ── Merge Dialog ──
if st.session_state.merge_source is not None:
    mrg_cid = st.session_state.merge_source
    mrg_name = st.session_state.working_names.get(mrg_cid, f'Cluster {mrg_cid}')

    @st.dialog(f"Merge: {mrg_name}")
    def merge_dialog():
        st.warning(f"Merge **{mrg_name}** ({int((cluster_labels == mrg_cid).sum())} films) into another cluster.")
        target_options = {
            cid: st.session_state.working_names.get(cid, f'Cluster {cid}')
            for cid in cluster_ids_sorted if cid != mrg_cid and cid >= 0
        }
        target_choice = st.selectbox(
            "Merge into:",
            options=list(target_options.keys()),
            format_func=lambda x: f"{x}: {target_options[x]}",
            key="merge_target_select",
        )
        c1, c2 = st.columns(2)
        if c1.button("Confirm Merge", type="primary", use_container_width=True, key="merge_confirm"):
            # Reassign all films from source to target
            cluster_labels[cluster_labels == mrg_cid] = target_choice
            if mrg_cid in st.session_state.working_names:
                del st.session_state.working_names[mrg_cid]
            if mrg_cid in st.session_state.cluster_status:
                del st.session_state.cluster_status[mrg_cid]
            st.session_state.unsaved_changes += 1
            st.session_state.merge_source = None
            st.rerun()
        if c2.button("Cancel", use_container_width=True, key="merge_cancel"):
            st.session_state.merge_source = None
            st.rerun()

    merge_dialog()

# ── Delete Confirmation ──
if st.session_state.delete_confirm is not None:
    del_cid = st.session_state.delete_confirm
    del_name = st.session_state.working_names.get(del_cid, f'Cluster {del_cid}')

    @st.dialog(f"Delete: {del_name}")
    def delete_dialog():
        n_films = int((cluster_labels == del_cid).sum())
        st.error(f"Delete **{del_name}**? This will move {n_films} films to Outlier status (-1).")
        c1, c2 = st.columns(2)
        if c1.button("Delete", type="primary", use_container_width=True, key="delete_yes"):
            cluster_labels[cluster_labels == del_cid] = -1
            if del_cid in st.session_state.working_names:
                del st.session_state.working_names[del_cid]
            if del_cid in st.session_state.cluster_status:
                del st.session_state.cluster_status[del_cid]
            st.session_state.unsaved_changes += 1
            st.session_state.delete_confirm = None
            st.rerun()
        if c2.button("Cancel", use_container_width=True, key="delete_no"):
            st.session_state.delete_confirm = None
            st.rerun()

    delete_dialog()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FILM DETAIL MODAL
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("Film Details", width="large")
def film_detail_modal(tt_code):
    if tt_code not in tt_codes.tolist():
        st.error("Film data not found.")
        return
        
    idx = np.where(tt_codes == tt_code)[0][0]
    title = title_lookup.get(tt_code, tt_code)
    omdb_entry = omdb.get(tt_code, {})
    poster_src = get_poster_src(tt_code, omdb_entry)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(poster_src, use_container_width=True)
            
        st.markdown(f"**Genre:** {genre_map.get(tt_code, 'N/A')}")
        st.markdown(f"**Year:** {omdb_entry.get('Year', 'N/A')}")
        st.markdown(f"**Director:** {omdb_entry.get('Director', 'N/A')}")
        
    with col2:
        st.subheader(title)
        st.markdown(f"_{omdb_entry.get('Plot', 'No plot available.')}_")
        st.markdown("---")
        
        cid = cluster_labels[idx]
        c_name = cluster_names.get(cid, {}).get('name', f'Cluster {cid}') if cid >= 0 else 'Outlier'
        conf = probabilities[idx]
        out_score = outlier_scores[idx]
        
        # Highlighting confidence and outlier scores aesthetically
        c1, c2, c3 = st.columns(3)
        c1.metric("Assigned Cluster", c_name)
        
        conf_color = "normal" if conf >= 0.5 else "off"
        c2.metric("Confidence", f"{conf:.1%}", delta="High" if conf >= 0.8 else "Low", delta_color=conf_color)
        
        out_color = "inverse" if out_score > 0.5 else "normal"
        c3.metric("Outlier Score", f"{out_score:.2f}", delta="Tier 3" if out_score > 0.8 else "Safe", delta_color=out_color)
        
        st.markdown("#### Feature Profile")
        film_feats = X[idx]
        if cid >= 0:
            cluster_mean = X[cluster_labels == cid].mean(axis=0)
            fig = feature_radar(film_feats, feature_names, title="Film vs Cluster Average", 
                               overlay_values=cluster_mean, overlay_name=f"{c_name} Avg", top_n=10)
        else:
            fig = feature_radar(film_feats, feature_names, title="Film vs Global Average", 
                               overlay_values=global_mean, overlay_name="Global Avg", top_n=10)
                               
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Close Modal", type="primary"):
            st.session_state.selected_film = None
            st.rerun()

# Trigger Modal
if st.session_state.selected_film:
    film_detail_modal(st.session_state.selected_film)
