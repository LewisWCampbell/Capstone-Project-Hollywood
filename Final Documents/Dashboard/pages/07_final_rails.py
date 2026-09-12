"""Page 6 — Final Approved Rails: Read-only summary of approved content rails."""

import streamlit as st
import numpy as np
import pandas as pd
import json as json_mod
from pathlib import Path
from utils.data import (
    load_pipeline_artifacts, build_title_lookup, load_genres, load_omdb, RESULTS_DIR,
)

st.set_page_config(page_title="Final Approved Rails", layout="wide")
st.title("Final Approved Rails")
st.markdown("Only rails that have been **approved** in the Cluster Explorer appear here.")

arts = load_pipeline_artifacts()
if arts is None:
    st.error("Pipeline artifacts not found. Run the notebook or Pipeline Runner first.")
    st.stop()

# --- Unpack ---
cluster_labels = arts['cluster_labels']
n_clusters = arts['n_clusters']
cluster_names = arts.get('cluster_names', {})
tt_codes = arts['tt_codes']
feature_names = arts['feature_names']
X = arts['X']
probabilities = arts.get('probabilities', np.zeros(len(tt_codes)))
outlier_scores = arts.get('outlier_scores', np.zeros(len(tt_codes)))

title_lookup = build_title_lookup()
omdb = load_omdb()
genres = load_genres()
genre_map = dict(zip(genres['movie_id'], genres['genre']))

# --- Jordan metadata for language/rating ---
_jordan_path = Path(__file__).resolve().parent.parent.parent / 'Data for Pipeline' / 'extra_data' / 'jordan_metadata.csv'
if _jordan_path.exists():
    _jordan_df = pd.read_csv(_jordan_path, dtype={'imdb_id': str})
    _jordan_map = _jordan_df.set_index('imdb_id').to_dict('index')
else:
    _jordan_map = {}

# --- Load cluster_status ---
# Priority: session state (shared across pages) > saved file > empty
cluster_status = {}
if 'cluster_status' in st.session_state:
    cluster_status = st.session_state.cluster_status
else:
    status_path = RESULTS_DIR / 'cluster_status.json'
    if status_path.exists():
        with open(status_path) as f:
            raw = json_mod.load(f)
        cluster_status = {int(k): v for k, v in raw.items()}

# --- Determine approved rails ---
approved_cids = [cid for cid in range(n_clusters) if cluster_status.get(cid) == 'confirmed']

if not approved_cids:
    st.warning("No rails have been approved yet. Go to the **Cluster Explorer** page and click "
               "**Approve** on the rails you want to finalize.")
    st.stop()

# --- Top-level summary metrics ---
approved_mask = np.isin(cluster_labels, approved_cids)
n_approved_films = int(approved_mask.sum())
avg_confidence = float(probabilities[approved_mask].mean()) if n_approved_films > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Films in Dataset", f"{len(tt_codes):,}")
col2.metric("Approved Rails", f"{len(approved_cids)}")
col3.metric("Films in Approved Rails", f"{n_approved_films:,}")
col4.metric("Avg Confidence", f"{avg_confidence:.0%}")

st.markdown("---")

# --- Rail Summary Table ---
st.subheader("Approved Rail Summary")

rail_rows = []
for cid in approved_cids:
    mask = cluster_labels == cid
    count = int(mask.sum())
    info = cluster_names.get(cid, {})
    name = info.get('name', f'Cluster {cid}')
    desc = info.get('description', '')
    avg_conf = float(probabilities[mask].mean()) if count > 0 else 0

    # Top genres
    rail_genres = []
    for tt in np.array(tt_codes)[mask]:
        for g in genre_map.get(tt, '').split(','):
            g = g.strip()
            if g:
                rail_genres.append(g)
    top_genres = pd.Series(rail_genres).value_counts().head(5).index.tolist() if rail_genres else []

    # Top discriminative features
    top_feats = info.get('top_features', [])

    # Language breakdown
    rail_langs = []
    for tt in np.array(tt_codes)[mask]:
        lang = _jordan_map.get(tt, {}).get('language', None)
        if isinstance(lang, str):
            rail_langs.append(lang)
    top_langs = pd.Series(rail_langs).value_counts().head(3).index.tolist() if rail_langs else []

    rail_rows.append({
        'Rail ID': cid,
        'Name': name,
        'Films': count,
        'Avg Confidence': f"{avg_conf:.0%}",
        'Description': desc,
        'Top Genres': ', '.join(top_genres),
        'Top Languages': ', '.join(top_langs),
        'Discriminative Features': ', '.join(top_feats[:3]) if top_feats else '',
    })

rail_df = pd.DataFrame(rail_rows)
st.dataframe(rail_df, use_container_width=True, hide_index=True, height=min(800, 40 + 35 * len(rail_df)))

st.markdown("---")

# --- Export (approved rails + approved sub-clusters) ---
st.subheader("Export Approved Rail Mapping")

# Load sub-cluster data for export
_sub_clusters = arts.get('sub_clusters', {})
_sub_cluster_names = arts.get('sub_cluster_names', {})
_sub_cluster_status = st.session_state.get('sub_cluster_status', {})
_sub_cluster_working_names = st.session_state.get('sub_cluster_working_names', {})
_sub_disabled_films = st.session_state.get('sub_disabled_films', {})

# Build sub-cluster lookup: global_index → (sub_key, sub_name) for approved sub-clusters
_sub_lookup = {}  # {global_idx: (sub_key, sub_rail_name)}
for _parent_cid, _sub_labels_list in _sub_clusters.items():
    _sub_labels_arr = np.array(_sub_labels_list)
    _parent_idx = np.where(cluster_labels == _parent_cid)[0]
    for _sid in sorted(set(_sub_labels_arr[_sub_labels_arr >= 0])):
        _sk = f"sub_{_parent_cid}_{_sid}"
        if _sub_cluster_status.get(_sk) == 'confirmed':
            _sub_mask = _sub_labels_arr == _sid
            _sub_gi = _parent_idx[_sub_mask]
            _sub_disabled = _sub_disabled_films.get(_sk, set())
            # Get the raw name (no "Sub Cluster" prefix for CSV)
            _raw_name = _sub_cluster_working_names.get(
                _sk,
                (_sub_cluster_names.get((_parent_cid, _sid),
                 _sub_cluster_names.get(str((_parent_cid, _sid)), {})) or {}).get('name', f'Sub-cluster {_sid}')
            )
            for _gi in _sub_gi:
                if _gi not in _sub_disabled:
                    _sub_lookup[int(_gi)] = (_sk, _raw_name)

export_rows = []
for i, tt in enumerate(tt_codes):
    cl = cluster_labels[i]
    if cl not in approved_cids:
        continue
    cl_name = cluster_names.get(cl, {}).get('name', f'Cluster {cl}')
    prob = float(probabilities[i])
    score = float(outlier_scores[i]) if outlier_scores is not None else 0
    jordan = _jordan_map.get(tt, {})

    # Check if this film belongs to an approved sub-cluster
    _sub_info = _sub_lookup.get(i)
    _sub_rail_name = _sub_info[1] if _sub_info else ''

    export_rows.append({
        'tt_code': tt,
        'title': title_lookup.get(tt, tt),
        'rail_id': int(cl),
        'rail_name': cl_name,
        'sub_rail_name': _sub_rail_name,
        'confidence': round(prob, 4),
        'outlier_score': round(score, 4),
        'genre': genre_map.get(tt, ''),
        'language': (lambda v: v if isinstance(v, str) else 'Unknown')(jordan.get('language', 'Unknown')),
        'content_rating': (lambda v: v if isinstance(v, str) else 'Unknown')(jordan.get('content_category', 'Unknown')),
    })

export_df = pd.DataFrame(export_rows)
csv = export_df.to_csv(index=False)
st.download_button("Download Approved Rail Mapping (CSV)", csv, "approved_rail_mapping.csv", "text/csv", type="primary")
_n_with_sub = int((export_df['sub_rail_name'] != '').sum()) if 'sub_rail_name' in export_df.columns else 0
st.caption(f"{len(export_df):,} films across {len(approved_cids)} approved rails"
           + (f" ({_n_with_sub} with sub-rail assignments)" if _n_with_sub > 0 else ""))
