"""Page 3 — Cluster Explorer: Full-stack cluster visualization and inspection."""

import streamlit as st
import numpy as np
import pandas as pd
import json as json_mod
from datetime import datetime
from pathlib import Path
from utils.data import (
    load_pipeline_artifacts, load_cluster_summary, load_sub_cluster_summary,
    load_genres, build_title_lookup, load_omdb, load_tmdb_popularity,
    RESULTS_DIR, get_poster_src,
)
from utils.plots import (
    cluster_feature_bars, feature_radar, CLUSTER_COLOURS, OUTLIER_COLOURS,
)
from utils.llm import name_cluster

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

    # Serialize disabled_films: {cid: set(int)} -> {str(cid): list(int)}
    _disabled_films_ser = {}
    if 'disabled_films' in st.session_state:
        _disabled_films_ser = {
            str(k): sorted(v) for k, v in st.session_state.disabled_films.items() if v
        }

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
        'disabled_films': _disabled_films_ser,
        'config': config,
    }

    with open(RUNS_DIR / filename, 'w') as f:
        json_mod.dump(payload, f, indent=2)

    return filename


def _load_run(filepath: str) -> dict:
    """Load a saved run from disk."""
    with open(filepath) as f:
        return json_mod.load(f)


def _save_cluster_status(cluster_status: dict):
    """Auto-save cluster_status + disabled_films to a standalone JSON for the Final Rails page."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / 'cluster_status.json'
    _df = {}
    if 'disabled_films' in st.session_state:
        _df = {str(k): sorted(v) for k, v in st.session_state.disabled_films.items() if v}
    payload = {str(k): v for k, v in cluster_status.items()}
    payload['_disabled_films'] = _df
    with open(path, 'w') as f:
        json_mod.dump(payload, f, indent=2)

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
cluster_labels = arts['cluster_labels']
n_clusters = arts['n_clusters']
soft_clusters = arts.get('soft_clusters')
cluster_names = arts.get('cluster_names', {})
sub_clusters = arts.get('sub_clusters', {})
sub_cluster_silhouettes = arts.get('sub_cluster_silhouettes', {})
sub_embeddings_2d = arts.get('sub_embeddings_2d', {})
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
tmdb_popularity = load_tmdb_popularity()  # {tt_code: popularity_float}
summary = load_cluster_summary()

# ── Load Jordan metadata (language + content rating) ─────────────────────────
_jordan_path = Path(__file__).resolve().parent.parent.parent / 'Data for Pipeline' / 'extra_data' / 'jordan_metadata.csv'
if _jordan_path.exists():
    _jordan_df = pd.read_csv(_jordan_path, dtype={'imdb_id': str})
    _jordan_map = _jordan_df.set_index('imdb_id').to_dict('index')
else:
    _jordan_map = {}
language_map = {tt: (lambda v: v if isinstance(v, str) else 'Unknown')(_jordan_map.get(tt, {}).get('language', 'Unknown')) for tt in tt_codes}
content_rating_map = {tt: (lambda v: v if isinstance(v, str) else 'Unknown')(_jordan_map.get(tt, {}).get('content_category', 'Unknown')) for tt in tt_codes}
all_languages = sorted(set(language_map.values()))
all_ratings = sorted(set(content_rating_map.values()))

# ── Load rating data (IMDB + RT + Jordan year) ──────────────────────────────
_proj_root = Path(__file__).resolve().parent.parent.parent
_data_dir = _proj_root / 'Data for Pipeline'
_imdb_path = _data_dir / 'Manual_imdb_ratings.csv'
_rt_path = _data_dir / 'Manual_rt_ratings.csv'

_imdb_rating_map = {}
if _imdb_path.exists():
    _imdb_df = pd.read_csv(_imdb_path, dtype={'movie_id': str})
    for _, _row in _imdb_df.iterrows():
        _val = pd.to_numeric(_row.get('User Rating'), errors='coerce')
        if pd.notna(_val):
            _imdb_rating_map[_row['movie_id']] = float(_val)

_rt_tomato_map = {}
_rt_popcorn_map = {}
if _rt_path.exists():
    _rt_df = pd.read_csv(_rt_path, dtype={'movie_id': str})
    for _, _row in _rt_df.iterrows():
        _tm = pd.to_numeric(_row.get('Tomatometer'), errors='coerce')
        _pm = pd.to_numeric(_row.get('Popcornmeter'), errors='coerce')
        if pd.notna(_tm):
            _rt_tomato_map[_row['movie_id']] = float(_tm)
        if pd.notna(_pm):
            _rt_popcorn_map[_row['movie_id']] = float(_pm)

# Jordan year map
_jordan_year_map = {}
_jordan_csv_path = _data_dir / "Jordan's Data.csv"
if _jordan_csv_path.exists():
    _jcsv = pd.read_csv(_jordan_csv_path)
    # Jordan's Data has movie_name but not movie_id — need to cross-reference via IMDB data
    # Use the OMDB data which has imdb_id -> Year
    for tt in tt_codes:
        _omdb_entry = omdb.get(tt, {})
        try:
            _yr = int(str(_omdb_entry.get('Year', '0'))[:4])
        except (ValueError, TypeError):
            _yr = 0
        if _yr > 0:
            _jordan_year_map[tt] = _yr

def _composite_score(tt):
    """Compute a 0-10 composite score from IMDB rating, RT scores, and year."""
    imdb_r = _imdb_rating_map.get(tt)  # 0-10 scale
    tomato = _rt_tomato_map.get(tt)    # 0-10 scale (already /10 in data)
    popcorn = _rt_popcorn_map.get(tt)  # 0-10 scale
    year = _jordan_year_map.get(tt, 0)

    # Collect available rating signals (all on 0-10 scale)
    scores = []
    if imdb_r is not None:
        scores.append(imdb_r)
    if tomato is not None:
        scores.append(tomato)
    if popcorn is not None:
        scores.append(popcorn)

    if not scores:
        return 0.0  # no rating data

    avg_rating = sum(scores) / len(scores)

    # Small year bonus: more recent films get a slight boost (max +0.5)
    year_bonus = 0.0
    if year >= 1920:
        year_bonus = min(0.5, (year - 1920) / (2025 - 1920) * 0.5)

    return round(avg_rating + year_bonus, 2)

# Pre-compute composite scores for all films
_composite_scores = {tt: _composite_score(tt) for tt in tt_codes}

# ── Pre-compute per-cluster stats ─────────────────────────────────────────────
global_mean = X.mean(axis=0)
# Sort rails by average confidence (highest first)
_cid_conf = {}
for _cid in summary['cluster_id'].tolist():
    _mask = cluster_labels == _cid
    if _mask.any():
        _cid_conf[_cid] = float(probabilities[_mask].mean())
    else:
        _cid_conf[_cid] = 0.0
cluster_ids_sorted = sorted(_cid_conf.keys(), key=lambda c: _cid_conf[c], reverse=True)

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
if 'loaded_run_id' not in st.session_state: st.session_state.loaded_run_id = None
if 'enabled_films' not in st.session_state:
    # {cid: set(global_indices)} — films the user manually toggled into/out of the rail
    # NOTE: All films are now enabled by default (no 95% confidence gate).
    # The curation layer handles filtering — users can remove films via Edit Films.
    st.session_state.enabled_films = {}
if 'disabled_films' not in st.session_state:
    # {cid: set(global_indices)} — films the user explicitly disabled from the rail
    st.session_state.disabled_films = {}

# ── Initialize sub-cluster working names + status ──
# Keys use string format "sub_{cid}_{sid}" to avoid collisions with int parent IDs
if 'sub_cluster_working_names' not in st.session_state:
    _sub_names_dict = arts.get('sub_cluster_names', {})
    st.session_state.sub_cluster_working_names = {}
    for _key, _info in _sub_names_dict.items():
        if isinstance(_key, tuple):
            _sk = f"sub_{_key[0]}_{_key[1]}"
        else:
            _sk = str(_key)
        _name = _info.get('name', f'Sub-cluster') if isinstance(_info, dict) else f'Sub-cluster'
        st.session_state.sub_cluster_working_names[_sk] = _name
if 'sub_cluster_status' not in st.session_state:
    st.session_state.sub_cluster_status = {}
if 'sub_disabled_films' not in st.session_state:
    st.session_state.sub_disabled_films = {}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOOLBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
# Default sort order (toolbar removed)
rail_sort = "Popularity"

# Row 1 of Toolbar — Run selector only
tb_col1, _ = st.columns([1, 3])
with tb_col1:
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
                # Restore disabled films (if saved) — backwards compat with old enabled_films format
                if 'disabled_films' in loaded:
                    st.session_state.disabled_films = {
                        int(k): set(v) for k, v in loaded['disabled_films'].items()
                    }
                else:
                    st.session_state.disabled_films = {}
                st.session_state.enabled_films = {}
                st.session_state.loaded_run_id = run_data['run_id']
                st.session_state.unsaved_changes = 0
                st.session_state.run_id = f"run_{run_data['run_id']:03d}"
                st.toast(f"Loaded Run {run_data['run_id']:03d}: {run_data.get('run_name', '')}")
                st.rerun()

# Row 2 — Language & Content Rating Filters
tb2_col1, tb2_col2 = st.columns([1, 1])
with tb2_col1:
    selected_languages = st.multiselect("Language filter:", all_languages, default=[], placeholder="All Languages")
with tb2_col2:
    selected_ratings = st.multiselect("Content rating filter:", all_ratings, default=[], placeholder="All Ratings")

# Build a set of tt_codes that pass the language/rating filters
_lang_rating_filter = set()
if selected_languages or selected_ratings:
    for _tt in tt_codes:
        if selected_languages and language_map.get(_tt, 'Unknown') not in selected_languages:
            continue
        if selected_ratings and content_rating_map.get(_tt, 'Unknown') not in selected_ratings:
            continue
        _lang_rating_filter.add(_tt)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RAIL BROWSER
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Rail Browser")

placeholder_url = "https://via.placeholder.com/140x210/1e232d/c9d1d9?text=No+Poster"

for cid in cluster_ids_sorted:
    if cid == -1: continue # Skip root outliers for now
    
    mask_c = cluster_labels == cid
    if mask_c.sum() == 0: continue
    
    count = int(mask_c.sum())
    c_name = st.session_state.working_names.get(cid, cluster_names.get(cid, {}).get('name', f'Cluster {cid}'))
    status = st.session_state.cluster_status.get(cid, 'draft')
    avg_conf = float(probabilities[mask_c].mean())
    hex_color = f"#{int(_colour(cid).split('(')[1].split(',')[0]):02x}{int(_colour(cid).split(',')[1]):02x}{int(_colour(cid).split(',')[2].split(')')[0]):02x}" if _colour(cid).startswith('rgb') else _colour(cid)

    # Film counts — all films enabled by default, minus any user-disabled ones
    _c_indices = np.where(mask_c)[0]
    _c_probs = probabilities[_c_indices]
    _disabled_set = st.session_state.disabled_films.get(cid, set())
    _n_selected = count - len(_disabled_set & set(_c_indices.tolist()))
    _n_enabled = 0  # legacy — no longer needed since all start enabled
    _n_pending = len(_disabled_set & set(_c_indices.tolist()))
    
    # ── Compute top genomes for this cluster ──
    _cl_mean = X[mask_c].mean(axis=0)
    _cl_delta = _cl_mean - global_mean
    _cl_top_idx = np.argsort(_cl_delta)[-8:][::-1]
    _cl_top_feats = [(feature_names[i], float(_cl_delta[i])) for i in _cl_top_idx if _cl_delta[i] > 0]

    # ── Row Header with top genomes ──
    feat_tags = ''.join(
        f"<span style='display:inline-block; background:rgba(235,203,139,0.15); "
        f"border:1px solid rgba(235,203,139,0.3); border-radius:3px; "
        f"padding:1px 6px; margin:0 3px; font-size:0.65em; color:#ebcb8b;'>"
        f"{f}</span>"
        for f, v in _cl_top_feats[:5]
    )
    _status_badge = (
        "<span style='background:#2ea043; color:#fff; padding:2px 8px; border-radius:4px; "
        "font-size:0.55em; vertical-align:middle; margin-left:8px;'>Approved</span>"
        if status == 'confirmed' else
        "<span style='background:#555; color:#ccc; padding:2px 8px; border-radius:4px; "
        "font-size:0.55em; vertical-align:middle; margin-left:8px;'>Draft</span>"
    )
    _enabled_label = f" + {_n_enabled} added" if _n_enabled > 0 else ""
    _pending_label = f" · {_n_pending} pending review" if _n_pending > 0 else ""
    st.markdown(
        f"<h4 style='margin-bottom: 0;'><span style='color:{hex_color};'>●</span> {c_name} "
        f"{_status_badge}"
        f"<span style='font-size:0.6em; color:#888; font-weight:normal;'>"
        f"&nbsp;&nbsp;&nbsp;{_n_selected} selected{_enabled_label}{_pending_label}"
        f" &nbsp;|&nbsp; Avg conf: {avg_conf:.0%}</span></h4>"
        f"<div style='margin: 2px 0 4px 18px;'>{feat_tags}</div>",
        unsafe_allow_html=True,
    )

    # ── Description (if available) ───────────────────────────────────────
    _desc = cluster_names.get(cid, {}).get('description', '')
    if _desc:
        st.markdown(
            f"<p style='margin:0 0 8px 18px; color:#999; font-size:0.9em; "
            f"font-style:italic;'>{_desc}</p>",
            unsafe_allow_html=True,
        )

    # Action buttons for this row
    r_col1, r_col2, r_col3, r_col4, r_col5, r_col6, _ = st.columns([1, 1.2, 1.2, 1, 1.2, 1.2, 4])
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
        if r_col5.button("✓ Approve", key=f"cnf_{cid}"):
            st.session_state.cluster_status[cid] = 'confirmed'
            _save_cluster_status(st.session_state.cluster_status)
            st.rerun()
    else:
        if r_col5.button("↩ Unapprove", key=f"uncnf_{cid}"):
            st.session_state.cluster_status[cid] = 'draft'
            _save_cluster_status(st.session_state.cluster_status)
            st.rerun()
    if r_col6.button("✂ Edit Films", key=f"edit_{cid}"):
        st.session_state[f'editing_rail_{cid}'] = not st.session_state.get(f'editing_rail_{cid}', False)
        st.rerun()
            
    # ── Edit Films Panel (visual poster-grid curation) ──
    if st.session_state.get(f'editing_rail_{cid}', False):
        _edit_indices = np.where(mask_c)[0]
        _edit_probs = probabilities[_edit_indices]
        _edit_order = np.argsort(_edit_probs)[::-1]
        _disabled_set_edit = st.session_state.disabled_films.get(cid, set())

        # Initialise toggle state for this editing session
        _toggle_key = f'edit_toggles_{cid}'
        if _toggle_key not in st.session_state:
            # Pre-populate: all enabled by default, disabled films unchecked
            _init = {}
            for _ei in _edit_order:
                _gi = int(_edit_indices[_ei])
                _init[_gi] = (_gi not in _disabled_set_edit)
            st.session_state[_toggle_key] = _init

        _toggles = st.session_state[_toggle_key]

        # ── Build selected items (checked) and pending items (unchecked) ──
        _sel_items = []
        _pend_items = []
        for _ei in _edit_order:
            _gi = int(_edit_indices[_ei])
            _conf = float(_edit_probs[_ei])
            _tt = tt_codes[_gi]
            _omdb_e = omdb.get(_tt, {})
            _item = {
                'gi': _gi, 'tt': _tt, 'conf': _conf,
                'title': title_lookup.get(_tt, _tt),
                'poster': get_poster_src(_tt, _omdb_e, size='120x180'),
                'genre': genre_map.get(_tt, 'Unknown'),
                'manually_disabled': _gi in _disabled_set_edit,
            }
            if _toggles.get(_gi, False):
                _sel_items.append(_item)
            else:
                _pend_items.append(_item)

        with st.expander(f"Curating films in {c_name}", expanded=True):
            # ── CSS for poster grid editing ──
            st.markdown("""
<style>
.edit-grid { display: flex; flex-wrap: wrap; gap: 10px; padding: 10px 0; }
.edit-card {
    width: 110px; text-align: center; position: relative; cursor: pointer;
    border-radius: 8px; padding: 4px; transition: all 0.2s;
}
.edit-card img {
    width: 110px; height: 165px; object-fit: cover; border-radius: 6px;
    transition: all 0.2s;
}
.edit-card .card-title {
    font-size: 0.7em; font-weight: 600; line-height: 1.2;
    margin-top: 4px; height: 2.4em; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.edit-card .card-conf { font-size: 0.65em; margin-top: 2px; }
.edit-card.selected { border: 2px solid #a3be8c; }
.edit-card.selected img { opacity: 1; }
.edit-card.deselected { border: 2px solid rgba(255,255,255,0.08); opacity: 0.5; }
.edit-card.deselected:hover { opacity: 0.8; }
.edit-card .check-badge {
    position: absolute; top: 6px; right: 6px; width: 22px; height: 22px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: bold;
}
.edit-card.selected .check-badge {
    background: #a3be8c; color: #0d1117;
}
.edit-card.deselected .check-badge {
    background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.4);
    border: 1px solid rgba(255,255,255,0.2);
}
</style>""", unsafe_allow_html=True)

            # ── Selected Films poster grid ──
            st.markdown(
                f"**In Rail** — {len(_sel_items)} films "
                f"<span style='font-size:0.8em; color:#888;'>(uncheck to remove)</span>",
                unsafe_allow_html=True,
            )

            # Render selected films as poster cards with checkboxes
            _n_cols = 8
            _sel_cols = st.columns(_n_cols)
            for _idx, _item in enumerate(_sel_items):
                with _sel_cols[_idx % _n_cols]:
                    _conf_color = "#a3be8c" if _item['conf'] >= 0.95 else (
                        "#ebcb8b" if _item['conf'] >= 0.80 else "#d08770")
                    _star = " ★" if _item['manually_added'] else ""
                    st.markdown(
                        f"<div class='edit-card selected'>"
                        f"<div class='check-badge'>✓</div>"
                        f"<img src='{_item['poster']}' loading='lazy'>"
                        f"<div class='card-title'>{_item['title']}{_star}</div>"
                        f"<div class='card-conf' style='color:{_conf_color};'>{_item['conf']:.0%}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if st.checkbox(
                        "Keep", value=True, key=f"sel_{cid}_{_item['gi']}",
                        label_visibility="collapsed",
                    ):
                        _toggles[_item['gi']] = True
                    else:
                        _toggles[_item['gi']] = False

            st.markdown("---")

            # ── Removed Films poster grid ──
            if _pend_items:
                st.markdown(
                    f"**Removed from Rail** — {len(_pend_items)} films "
                    f"<span style='font-size:0.8em; color:#888;'>(check to re-add)</span>",
                    unsafe_allow_html=True,
                )
                _pend_cols = st.columns(_n_cols)
                for _idx, _item in enumerate(_pend_items):
                    with _pend_cols[_idx % _n_cols]:
                        _conf_color = "#ebcb8b" if _item['conf'] >= 0.80 else "#d08770"
                        st.markdown(
                            f"<div class='edit-card deselected'>"
                            f"<div class='check-badge'>+</div>"
                            f"<img src='{_item['poster']}' loading='lazy'>"
                            f"<div class='card-title'>{_item['title']}</div>"
                            f"<div class='card-conf' style='color:{_conf_color};'>{_item['conf']:.0%}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if st.checkbox(
                            "Add", value=False, key=f"pend_{cid}_{_item['gi']}",
                            label_visibility="collapsed",
                        ):
                            _toggles[_item['gi']] = True
                        else:
                            _toggles[_item['gi']] = False
            else:
                st.info("All films in this cluster are currently enabled.")

            # ── Compute changes ──
            _removals = []
            _additions = []
            for _ei in _edit_order:
                _gi = int(_edit_indices[_ei])
                _was_selected = (_gi not in _disabled_set_edit)
                _now_selected = _toggles.get(_gi, _was_selected)
                _tt = tt_codes[_gi]
                if _was_selected and not _now_selected:
                    _removals.append({
                        '_idx': _gi,
                        'TT Code': _tt,
                        'Title': title_lookup.get(_tt, _tt),
                    })
                elif not _was_selected and _now_selected:
                    _additions.append({'_idx': _gi, 'TT Code': _tt})

            if _removals:
                st.warning(f"{len(_removals)} film(s) will be removed: "
                           + ", ".join(r['Title'] for r in _removals[:5])
                           + ("..." if len(_removals) > 5 else ""))
            if _additions:
                st.success(f"{len(_additions)} film(s) will be added to this rail.")

            # ── Apply button ──
            if _removals or _additions:
                if st.button("Apply Changes", key=f"apply_edit_{cid}", type="primary"):
                    for _r in _removals:
                        _rm_idx = _r['_idx']
                        _disabled_set_edit.add(_rm_idx)
                        st.session_state.action_history.append({
                            'action': 'disable_film', 'film_idx': _rm_idx,
                            'from_cluster': cid, 'tt_code': _r['TT Code'],
                        })
                        st.session_state.unsaved_changes += 1
                    for _a in _additions:
                        _disabled_set_edit.discard(_a['_idx'])
                    st.session_state.disabled_films[cid] = _disabled_set_edit
                    # Clear toggle state so it re-initialises on next edit
                    del st.session_state[_toggle_key]
                    st.session_state[f'editing_rail_{cid}'] = False
                    st.rerun()

            # Cancel button
            if st.button("Cancel", key=f"cancel_edit_{cid}"):
                if _toggle_key in st.session_state:
                    del st.session_state[_toggle_key]
                st.session_state[f'editing_rail_{cid}'] = False
                st.rerun()

    # ── Film Rail (all enabled by default) ──
    cluster_indices = np.where(mask_c)[0]
    probs_c = probabilities[cluster_indices]
    disabled_set = st.session_state.disabled_films.get(cid, set())

    # Build display list: all films except user-disabled ones, filtered by language/rating
    _display_items = []  # list of (local_index, global_index, confidence)
    for _li, _gi in enumerate(cluster_indices):
        _conf = probs_c[_li]
        if _gi not in disabled_set:
            # Apply language/rating filter if active
            if _lang_rating_filter and tt_codes[_gi] not in _lang_rating_filter:
                continue
            _display_items.append((_li, _gi, float(_conf)))
    # Sort by selected film sort order
    if rail_sort == "Popularity":
        _display_items.sort(key=lambda x: tmdb_popularity.get(tt_codes[x[1]], 0.0), reverse=True)
    elif rail_sort == "Year (Newest)":
        _display_items.sort(key=lambda x: _jordan_year_map.get(tt_codes[x[1]], 0), reverse=True)
    elif rail_sort == "Year (Oldest)":
        _display_items.sort(key=lambda x: _jordan_year_map.get(tt_codes[x[1]], 9999))
    else:
        _display_items.sort(key=lambda x: x[2], reverse=True)

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
    position: relative;
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
.film-score { font-size: 0.8em; margin-top: 2px; }
/* ── Plot tooltip on hover ── */
.film-card .plot-tooltip {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    width: 260px;
    background: #1e232d;
    color: #c9d1d9;
    border: 1px solid rgba(156, 39, 176, 0.4);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.75em;
    line-height: 1.4;
    text-align: left;
    z-index: 1000;
    pointer-events: none;
    transition: opacity 0.2s, visibility 0.2s;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    margin-bottom: 6px;
}
.film-card .plot-tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 6px;
    border-style: solid;
    border-color: #1e232d transparent transparent transparent;
}
.film-card .plot-tooltip .tt-title {
    font-weight: 700;
    font-size: 1.05em;
    color: #e6edf3;
    margin-bottom: 4px;
}
.film-card .plot-tooltip .tt-meta {
    font-size: 0.9em;
    color: #8b949e;
    margin-bottom: 6px;
}
.film-card:hover .plot-tooltip {
    visibility: visible;
    opacity: 1;
}
</style>
<div class="rail-container">
"""

    for _rank, (_li, _gi, _conf) in enumerate(_display_items[:100], 1):
        tt = tt_codes[_gi]
        title = title_lookup.get(tt, tt)
        omdb_entry = omdb.get(tt, {})
        year = _jordan_year_map.get(tt, '')
        # Color code confidence
        if _conf >= 0.95: conf_color = "#a3be8c"    # green — high confidence
        elif _conf >= 0.80: conf_color = "#ebcb8b"   # yellow — moderate
        else: conf_color = "#d08770"                  # orange — low

        conf_label = f"{_conf:.0%}"
        year_label = f" ({year})" if year else ""
        _border_style = ""
        _added_tag = ""

        img_src = get_poster_src(tt, omdb_entry)
        safe_title = title.replace('"', '&quot;').replace("'", "&#39;")

        _year_str = f" ({year})" if year else ""
        if rail_sort == "Popularity":
            _pop = tmdb_popularity.get(tt, 0.0)
            _score_line = f"Pop: {_pop:.1f} · {conf_label}"
        elif rail_sort == "Confidence":
            _score_line = f"#{_rank} · {conf_label}"
        else:
            _score_line = f"{year if year else '?'} · {conf_label}"
        _film_link = f"/film_explorer?tt={tt}"

        # Build plot tooltip HTML
        _plot_raw = omdb_entry.get('Plot', '') if omdb_entry else ''
        if _plot_raw and _plot_raw != 'N/A':
            _safe_plot = _plot_raw.replace('"', '&quot;').replace("'", "&#39;").replace('<', '&lt;').replace('>', '&gt;')
            _rated = omdb_entry.get('Rated', '')
            _runtime = omdb_entry.get('Runtime', '')
            _meta_parts = [p for p in [str(year) if year else '', _rated, _runtime] if p and p != 'N/A']
            _meta_line = ' · '.join(_meta_parts)
            _tooltip_html = (
                f'<div class="plot-tooltip">'
                f'<div class="tt-title">{safe_title}</div>'
                f'<div class="tt-meta">{_meta_line}</div>'
                f'{_safe_plot}'
                f'</div>'
            )
        else:
            _tooltip_html = ''

        _lang = language_map.get(tt, '')
        _rating = content_rating_map.get(tt, '')
        _meta_tag = ' · '.join(p for p in [_lang, _rating] if p and p != 'Unknown')

        card_html = f"""
<a href="{_film_link}" target="_self" style="text-decoration: none; color: inherit;">
<div class="film-card" style="{_border_style} border-radius: 8px; padding: 4px;">
    {_tooltip_html}
    <img class="film-poster" src="{img_src}" alt="{safe_title} poster" loading="lazy">
    <div class="film-title">{safe_title}{_added_tag}</div>
    <div class="film-meta" style="font-size:0.65em; color:#8b949e; margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{_meta_tag}</div>
</div>
</a>
"""
        rail_html += card_html

    rail_html += "</div>"
    st.markdown(rail_html, unsafe_allow_html=True)

    # ── Sub-cluster Rails (inline, editable — same pattern as parent rails) ──
    if cid in sub_clusters:
        sub_labels_arr = np.array(sub_clusters[cid])
        sub_ids = sorted(set(sub_labels_arr[sub_labels_arr >= 0]))
        if sub_ids:
            cluster_indices_all = np.where(mask_c)[0]
            _sub_names_dict = arts.get('sub_cluster_names', {})

            for sid in sub_ids:
                _sk = f"sub_{cid}_{sid}"
                sub_mask_local = sub_labels_arr == sid
                sub_global_idx = cluster_indices_all[sub_mask_local]
                sub_count = len(sub_global_idx)

                # Get sub-cluster name (with "Sub Cluster" prefix for display)
                _raw_name = st.session_state.sub_cluster_working_names.get(
                    _sk,
                    (_sub_names_dict.get((cid, sid), _sub_names_dict.get(str((cid, sid)), {})) or {}).get('name', f'Sub-cluster {sid}')
                )
                # Ensure working_names has this entry
                if _sk not in st.session_state.sub_cluster_working_names:
                    st.session_state.sub_cluster_working_names[_sk] = _raw_name
                _display_name = f"Sub Cluster: {_raw_name}"

                _sub_status = st.session_state.sub_cluster_status.get(_sk, 'draft')
                _sub_avg_conf = float(probabilities[sub_global_idx].mean()) if sub_count > 0 else 0.0

                # Top features for this sub-cluster
                _sub_name_info = _sub_names_dict.get((cid, sid), _sub_names_dict.get(str((cid, sid)), {}))
                _naming_feats = []
                if isinstance(_sub_name_info, dict):
                    _naming_feats = _sub_name_info.get('top_features', [])
                if not _naming_feats:
                    _sub_mean = X[sub_global_idx].mean(axis=0)
                    _parent_mean = X[cluster_indices_all].mean(axis=0)
                    _sub_delta = _sub_mean - _parent_mean
                    _sub_top_idx = np.argsort(_sub_delta)[-8:][::-1]
                    _naming_feats = [f'{feature_names[i]} (+{_sub_delta[i]:.2f})'
                                     for i in _sub_top_idx if _sub_delta[i] > 0]

                _sub_feat_tags = ''.join(
                    f"<span style='display:inline-block; background:rgba(136,192,208,0.15); "
                    f"border:1px solid rgba(136,192,208,0.3); border-radius:3px; "
                    f"padding:1px 5px; margin:0 2px; font-size:0.65em; color:#88c0d0;'>"
                    f"{f}</span>"
                    for f in _naming_feats[:5]
                )

                # Disabled films for this sub-cluster
                _sub_disabled = st.session_state.sub_disabled_films.get(_sk, set())
                _sub_n_selected = sub_count - len(_sub_disabled & set(sub_global_idx.tolist()))
                _sub_n_pending = len(_sub_disabled & set(sub_global_idx.tolist()))

                _sub_status_badge = (
                    "<span style='background:#2ea043; color:#fff; padding:2px 8px; border-radius:4px; "
                    "font-size:0.55em; vertical-align:middle; margin-left:8px;'>Approved</span>"
                    if _sub_status == 'confirmed' else
                    "<span style='background:#555; color:#ccc; padding:2px 8px; border-radius:4px; "
                    "font-size:0.55em; vertical-align:middle; margin-left:8px;'>Draft</span>"
                )
                _sub_pending_label = f" · {_sub_n_pending} pending review" if _sub_n_pending > 0 else ""

                # ── Sub-cluster Rail Header ──
                st.markdown(
                    f"<h4 style='margin-bottom: 0; margin-left: 20px;'>"
                    f"<span style='color:#88c0d0;'>↳</span> {_display_name} "
                    f"{_sub_status_badge}"
                    f"<span style='font-size:0.6em; color:#888; font-weight:normal;'>"
                    f"&nbsp;&nbsp;&nbsp;{_sub_n_selected} selected{_sub_pending_label}"
                    f" &nbsp;|&nbsp; Avg conf: {_sub_avg_conf:.0%}</span></h4>"
                    f"<div style='margin: 2px 0 4px 38px;'>{_sub_feat_tags}</div>",
                    unsafe_allow_html=True,
                )

                # ── Sub-cluster Description (if available) ───────────────
                _sub_desc = _sub_names_dict.get((cid, sid), {}).get('description', '')
                if not _sub_desc:
                    _sub_desc = _sub_names_dict.get(_sk, {}).get('description', '')
                if _sub_desc:
                    st.markdown(
                        f"<p style='margin:0 0 8px 38px; color:#999; font-size:0.85em; "
                        f"font-style:italic;'>{_sub_desc}</p>",
                        unsafe_allow_html=True,
                    )

                # ── Sub-cluster Action Buttons ──
                _sr1, _sr2, _sr3, _sr4, _sr5, _ = st.columns([1, 1.2, 1, 1.2, 1.2, 5])
                if _sr1.button("✎ Rename", key=f"rnm_{_sk}"):
                    st.session_state.rename_target = _sk
                    st.rerun()
                if _sr2.button("⚡ Auto-name", key=f"auto_{_sk}"):
                    st.session_state.autoname_running = _sk
                    st.rerun()
                if _sub_status != 'confirmed':
                    if _sr3.button("✓ Approve", key=f"cnf_{_sk}"):
                        st.session_state.sub_cluster_status[_sk] = 'confirmed'
                        _save_cluster_status(st.session_state.cluster_status)
                        st.rerun()
                else:
                    if _sr3.button("↩ Unapprove", key=f"uncnf_{_sk}"):
                        st.session_state.sub_cluster_status[_sk] = 'draft'
                        _save_cluster_status(st.session_state.cluster_status)
                        st.rerun()
                if _sr4.button("✂ Edit Films", key=f"edit_{_sk}"):
                    st.session_state[f'editing_rail_{_sk}'] = not st.session_state.get(f'editing_rail_{_sk}', False)
                    st.rerun()

                # ── Sub-cluster Edit Films Panel ──
                if st.session_state.get(f'editing_rail_{_sk}', False):
                    _sub_probs = probabilities[sub_global_idx]
                    _sub_order = np.argsort(_sub_probs)[::-1]
                    _sub_dis_edit = st.session_state.sub_disabled_films.get(_sk, set())
                    _sub_toggle_key = f'edit_toggles_{_sk}'
                    if _sub_toggle_key not in st.session_state:
                        _init = {}
                        for _ei in _sub_order:
                            _gi = int(sub_global_idx[_ei])
                            _init[_gi] = (_gi not in _sub_dis_edit)
                        st.session_state[_sub_toggle_key] = _init
                    _sub_toggles = st.session_state[_sub_toggle_key]

                    _sub_sel = []
                    _sub_pend = []
                    for _ei in _sub_order:
                        _gi = int(sub_global_idx[_ei])
                        _item = {
                            'gi': _gi, 'tt': tt_codes[_gi], 'conf': float(_sub_probs[_ei]),
                            'title': title_lookup.get(tt_codes[_gi], tt_codes[_gi]),
                            'poster': get_poster_src(tt_codes[_gi], omdb.get(tt_codes[_gi], {}), size='120x180'),
                            'manually_added': False,
                        }
                        if _sub_toggles.get(_gi, False):
                            _sub_sel.append(_item)
                        else:
                            _sub_pend.append(_item)

                    with st.expander(f"Curating films in {_display_name}", expanded=True):
                        st.markdown(
                            f"**In Rail** — {len(_sub_sel)} films "
                            f"<span style='font-size:0.8em; color:#888;'>(uncheck to remove)</span>",
                            unsafe_allow_html=True,
                        )
                        _n_cols = 8
                        _sel_cols = st.columns(_n_cols)
                        for _idx, _item in enumerate(_sub_sel):
                            with _sel_cols[_idx % _n_cols]:
                                _conf_color = "#a3be8c" if _item['conf'] >= 0.95 else (
                                    "#ebcb8b" if _item['conf'] >= 0.80 else "#d08770")
                                st.markdown(
                                    f"<div class='edit-card selected'><div class='check-badge'>✓</div>"
                                    f"<img src='{_item['poster']}' loading='lazy'>"
                                    f"<div class='card-title'>{_item['title']}</div>"
                                    f"<div class='card-conf' style='color:{_conf_color};'>{_item['conf']:.0%}</div></div>",
                                    unsafe_allow_html=True,
                                )
                                if st.checkbox("Keep", value=True, key=f"sel_{_sk}_{_item['gi']}", label_visibility="collapsed"):
                                    _sub_toggles[_item['gi']] = True
                                else:
                                    _sub_toggles[_item['gi']] = False

                        if _sub_pend:
                            st.markdown("---")
                            st.markdown(
                                f"**Removed from Rail** — {len(_sub_pend)} films "
                                f"<span style='font-size:0.8em; color:#888;'>(check to re-add)</span>",
                                unsafe_allow_html=True,
                            )
                            _pend_cols = st.columns(_n_cols)
                            for _idx, _item in enumerate(_sub_pend):
                                with _pend_cols[_idx % _n_cols]:
                                    st.markdown(
                                        f"<div class='edit-card deselected'><div class='check-badge'>+</div>"
                                        f"<img src='{_item['poster']}' loading='lazy'>"
                                        f"<div class='card-title'>{_item['title']}</div>"
                                        f"<div class='card-conf' style='color:#d08770;'>{_item['conf']:.0%}</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                    if st.checkbox("Add", value=False, key=f"pend_{_sk}_{_item['gi']}", label_visibility="collapsed"):
                                        _sub_toggles[_item['gi']] = True
                                    else:
                                        _sub_toggles[_item['gi']] = False

                        _sc1, _sc2, _ = st.columns([1, 1, 6])
                        if _sc1.button("Apply Changes", type="primary", key=f"apply_{_sk}"):
                            _new_disabled = set()
                            for _gi, _on in _sub_toggles.items():
                                if not _on:
                                    _new_disabled.add(_gi)
                            st.session_state.sub_disabled_films[_sk] = _new_disabled
                            st.session_state.unsaved_changes += 1
                            if _sub_toggle_key in st.session_state:
                                del st.session_state[_sub_toggle_key]
                            st.session_state[f'editing_rail_{_sk}'] = False
                            st.rerun()
                        if _sc2.button("Cancel", key=f"cancel_{_sk}"):
                            if _sub_toggle_key in st.session_state:
                                del st.session_state[_sub_toggle_key]
                            st.session_state[f'editing_rail_{_sk}'] = False
                            st.rerun()

                # ── Sub-cluster Film Rail (same card layout as parent) ──
                _sub_probs = probabilities[sub_global_idx]
                _sub_display = []
                _sub_disabled = st.session_state.sub_disabled_films.get(_sk, set())
                for _li, _gi in enumerate(sub_global_idx):
                    if _gi not in _sub_disabled:
                        if _lang_rating_filter and tt_codes[_gi] not in _lang_rating_filter:
                            continue
                        _sub_display.append((_li, _gi, float(_sub_probs[_li])))

                if rail_sort == "Popularity":
                    _sub_display.sort(key=lambda x: tmdb_popularity.get(tt_codes[x[1]], 0.0), reverse=True)
                elif rail_sort == "Year (Newest)":
                    _sub_display.sort(key=lambda x: _jordan_year_map.get(tt_codes[x[1]], 0), reverse=True)
                elif rail_sort == "Year (Oldest)":
                    _sub_display.sort(key=lambda x: _jordan_year_map.get(tt_codes[x[1]], 9999))
                else:
                    _sub_display.sort(key=lambda x: x[2], reverse=True)

                sub_rail_html = '<div class="rail-container">'
                for _rank, (_li, _gi, _conf) in enumerate(_sub_display[:100], 1):
                    tt = tt_codes[_gi]
                    title = title_lookup.get(tt, tt)
                    omdb_entry = omdb.get(tt, {})
                    _sub_year = _jordan_year_map.get(tt, '')
                    if _conf >= 0.95: conf_color = "#a3be8c"
                    elif _conf >= 0.80: conf_color = "#ebcb8b"
                    else: conf_color = "#d08770"
                    img_src = get_poster_src(tt, omdb_entry)
                    safe_title = title.replace('"', '&quot;').replace("'", "&#39;")
                    if rail_sort == "Popularity":
                        _sub_pop = tmdb_popularity.get(tt, 0.0)
                        _sub_score = f"Pop: {_sub_pop:.1f} · {_conf:.0%}"
                    elif rail_sort == "Confidence":
                        _sub_score = f"#{_rank} · {_conf:.0%}"
                    else:
                        _sub_score = f"{_sub_year if _sub_year else '?'} · {_conf:.0%}"
                    _sub_link = f"/film_explorer?tt={tt}"

                    _sub_plot_raw = omdb_entry.get('Plot', '') if omdb_entry else ''
                    if _sub_plot_raw and _sub_plot_raw != 'N/A':
                        _sub_safe_plot = _sub_plot_raw.replace('"', '&quot;').replace("'", "&#39;").replace('<', '&lt;').replace('>', '&gt;')
                        _sub_rated = omdb_entry.get('Rated', '')
                        _sub_runtime = omdb_entry.get('Runtime', '')
                        _sub_meta_parts = [p for p in [str(_sub_year) if _sub_year else '', _sub_rated, _sub_runtime] if p and p != 'N/A']
                        _sub_meta_line = ' · '.join(_sub_meta_parts)
                        _sub_tooltip = (
                            f'<div class="plot-tooltip"><div class="tt-title">{safe_title}</div>'
                            f'<div class="tt-meta">{_sub_meta_line}</div>{_sub_safe_plot}</div>'
                        )
                    else:
                        _sub_tooltip = ''

                    _sub_lang = language_map.get(tt, '')
                    _sub_rating = content_rating_map.get(tt, '')
                    _sub_meta_tag = ' · '.join(p for p in [_sub_lang, _sub_rating] if p and p != 'Unknown')

                    sub_rail_html += f"""
<a href="{_sub_link}" target="_self" style="text-decoration: none; color: inherit;">
<div class="film-card" style="border-radius: 8px; padding: 4px;">
    {_sub_tooltip}
    <img class="film-poster" src="{img_src}" alt="{safe_title} poster" loading="lazy">
    <div class="film-title">{safe_title}</div>
    <div class="film-meta" style="font-size:0.65em; color:#8b949e; margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{_sub_meta_tag}</div>
</div>
</a>"""
                sub_rail_html += '</div>'
                st.markdown(sub_rail_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3b — ACTION DIALOGS (Rename / Auto-name / Merge / Delete)
# ══════════════════════════════════════════════════════════════════════════════

# ── Rename Dialog ──
if st.session_state.rename_target is not None:
    ren_cid = st.session_state.rename_target
    _is_sub = isinstance(ren_cid, str) and ren_cid.startswith('sub_')
    if _is_sub:
        old_name = st.session_state.sub_cluster_working_names.get(ren_cid, ren_cid)
    else:
        old_name = st.session_state.working_names.get(ren_cid, f'Cluster {ren_cid}')

    @st.dialog(f"Rename: {old_name}")
    def rename_dialog():
        new_name = st.text_input("New name:", value=old_name, key="rename_input")
        c1, c2 = st.columns(2)
        if c1.button("Save", type="primary", use_container_width=True):
            if _is_sub:
                st.session_state.sub_cluster_working_names[ren_cid] = new_name
            else:
                st.session_state.working_names[ren_cid] = new_name
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
    _auto_is_sub = isinstance(auto_cid, str) and auto_cid.startswith('sub_')

    @st.dialog(f"Auto-naming {'Sub-cluster' if _auto_is_sub else 'Cluster'} {auto_cid}")
    def autoname_dialog():
        st.info("Querying local Ollama model for a cluster name...")
        if _auto_is_sub:
            # Parse sub_CID_SID and get the film indices
            _parts = auto_cid.split('_')
            _parent_cid, _sub_sid = int(_parts[1]), int(_parts[2])
            _parent_mask = cluster_labels == _parent_cid
            _parent_idx = np.where(_parent_mask)[0]
            _sub_labels = np.array(sub_clusters.get(_parent_cid, []))
            _sub_mask = _sub_labels == _sub_sid
            sample_idx = _parent_idx[_sub_mask][:10]
        else:
            mask_auto = cluster_labels == auto_cid
            sample_idx = np.where(mask_auto)[0][:10]
        sample_tts = [tt_codes[i] for i in sample_idx]

        from utils import llm as _llm
        with st.spinner(f"Generating name via {_llm.llm_backend()}..."):
            if _auto_is_sub:
                result = name_cluster(_parent_cid, sample_tts, X, cluster_labels, feature_names, title_lookup)
            else:
                result = name_cluster(auto_cid, sample_tts, X, cluster_labels, feature_names, title_lookup)

        suggested_name = result.get('name', f'Cluster {auto_cid}')
        suggested_desc = result.get('description', '')

        if result.get('method') == 'fallback' and _llm.LAST_LLM_ERROR:
            st.warning(f"LLM naming failed, showing heuristic name instead. Backend: "
                       f"{_llm.llm_backend()}. Error: {_llm.LAST_LLM_ERROR}")
        st.success(f"**Suggested name:** {suggested_name}")
        if suggested_desc:
            st.caption(suggested_desc)

        top_feats = result.get('top_features', [])
        unique_diff = result.get('unique_differentiators', [])
        if top_feats:
            with st.expander("Discriminative features", expanded=True):
                st.markdown(f"**Top features:** {', '.join(top_feats)}")
                if unique_diff:
                    st.markdown(f"**Unique differentiators:** {', '.join(unique_diff)}")

        final_name = st.text_input("Edit name if desired:", value=suggested_name, key="autoname_edit")
        c1, c2 = st.columns(2)
        if c1.button("Accept", type="primary", use_container_width=True, key="autoname_accept"):
            if _auto_is_sub:
                st.session_state.sub_cluster_working_names[auto_cid] = final_name
            else:
                st.session_state.working_names[auto_cid] = final_name
                updated = {
                    'name': final_name,
                    'description': suggested_desc,
                    'top_features': top_feats,
                    'unique_differentiators': unique_diff,
                }
                if auto_cid in cluster_names:
                    cluster_names[auto_cid].update(updated)
                else:
                    cluster_names[auto_cid] = updated
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
    poster_url = omdb_entry.get('Poster', '')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if poster_url and poster_url != 'N/A':
            st.image(poster_url, use_container_width=True)
        else:
            st.write("No Poster")
            
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
