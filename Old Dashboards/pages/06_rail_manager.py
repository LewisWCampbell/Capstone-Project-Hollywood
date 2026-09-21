"""Page 6 — Rail Manager: Rename, merge, split, export content rails."""

import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime
from utils.data import (
    load_pipeline_artifacts, load_cluster_summary, load_sub_cluster_summary,
    build_title_lookup, load_genres, RESULTS_DIR,
)

st.set_page_config(page_title="Rail Manager", layout="wide")
st.title("Rail Manager")
st.markdown("Manage content rails — rename, merge, split, and export.")

arts = load_pipeline_artifacts()
if arts is None:
    st.error("Pipeline artifacts not found. Run the notebook or Pipeline Runner first.")
    st.stop()

cluster_labels = arts['cluster_labels']
n_clusters = arts['n_clusters']
cluster_names = arts.get('cluster_names', {})
tt_codes = arts['tt_codes']
title_lookup = build_title_lookup()
genres = load_genres()
genre_map = dict(zip(genres['movie_id'], genres['genre']))

# --- Initialize session state ---
if 'working_labels' not in st.session_state:
    st.session_state.working_labels = cluster_labels.copy()
if 'working_names' not in st.session_state:
    st.session_state.working_names = {
        cid: cluster_names.get(cid, {}).get('name', f'Cluster {cid}')
        for cid in range(n_clusters)
    }

working_labels = st.session_state.working_labels
working_names = st.session_state.working_names

# --- Rail Overview Table ---
st.subheader("Rail Overview")

rail_data = []
for cid in sorted(working_names.keys()):
    count = int((working_labels == cid).sum())
    desc = cluster_names.get(cid, {}).get('description', '')
    top_genres = []
    for tt in np.array(tt_codes)[working_labels == cid][:100]:
        for g in genre_map.get(tt, '').split(','):
            top_genres.append(g.strip())
    if top_genres:
        top_5 = pd.Series(top_genres).value_counts().head(5).index.tolist()
    else:
        top_5 = []
    rail_data.append({
        'Rail ID': cid,
        'Name': working_names.get(cid, f'Cluster {cid}'),
        'Films': count,
        'Description': desc,
        'Top Genres': ', '.join(top_5),
        'Status': 'Active',
    })

rail_df = pd.DataFrame(rail_data)
edited_df = st.data_editor(
    rail_df, use_container_width=True, hide_index=True,
    column_config={
        'Rail ID': st.column_config.NumberColumn(disabled=True),
        'Films': st.column_config.NumberColumn(disabled=True),
        'Top Genres': st.column_config.TextColumn(disabled=True),
        'Name': st.column_config.TextColumn(),
        'Status': st.column_config.SelectboxColumn(options=['Active', 'Draft', 'Under Review', 'Archived']),
    },
)

# Persist name edits
for _, row in edited_df.iterrows():
    cid = int(row['Rail ID'])
    if cid in working_names and row['Name'] != working_names[cid]:
        working_names[cid] = row['Name']
        st.session_state.working_names = working_names

st.markdown("---")

# --- Merge Tool ---
st.subheader("Merge Clusters")
merge_options = [f"{cid}: {working_names.get(cid, f'Cluster {cid}')}" for cid in sorted(working_names.keys())]
merge_selected = st.multiselect("Select 2+ clusters to merge:", merge_options)

if len(merge_selected) >= 2:
    merge_ids = [int(s.split(":")[0]) for s in merge_selected]
    total_films = sum((working_labels == cid).sum() for cid in merge_ids)
    st.info(f"Merging {len(merge_ids)} clusters ({total_films} films total) into Cluster {min(merge_ids)}.")

    if st.button("Confirm Merge", type="primary"):
        target = min(merge_ids)
        for cid in merge_ids:
            if cid != target:
                working_labels[working_labels == cid] = target
                if cid in working_names:
                    del working_names[cid]
        st.session_state.working_labels = working_labels
        st.session_state.working_names = working_names

        # Log
        RESULTS_DIR.mkdir(exist_ok=True)
        log = {'action': 'merge', 'clusters': merge_ids, 'target': target,
               'timestamp': datetime.now().isoformat()}
        with open(RESULTS_DIR / 'rail_edits.jsonl', 'a') as f:
            f.write(json.dumps(log) + '\n')
        st.success(f"Merged into Cluster {target}.")
        st.rerun()

st.markdown("---")

# --- Split Tool ---
st.subheader("Split Cluster")
split_options = [f"{cid}: {working_names.get(cid, f'Cluster {cid}')}" for cid in sorted(working_names.keys())]
split_selected = st.selectbox("Select cluster to split:", split_options, key="split_select")

if split_selected:
    split_id = int(split_selected.split(":")[0])
    sub_clusters = arts.get('sub_clusters', {})
    if split_id in sub_clusters:
        sub_labels = sub_clusters[split_id]
        n_subs = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        st.info(f"Cluster {split_id} has {n_subs} sub-clusters from recursive clustering.")

        sub_counts = pd.Series(sub_labels).value_counts().sort_index()
        st.bar_chart(sub_counts)

        if st.button("Confirm Split", type="primary"):
            # Promote sub-clusters to new top-level clusters
            new_base = max(working_names.keys()) + 1
            cluster_mask = working_labels == split_id
            indices = np.where(cluster_mask)[0]
            for i, idx in enumerate(indices):
                sl = sub_labels[i]
                if sl >= 0:
                    working_labels[idx] = new_base + sl
                # else leave as outlier (-1)

            # Create names for new clusters
            sub_names = arts.get('sub_cluster_names', {})
            for sl in set(sub_labels):
                if sl >= 0:
                    name = sub_names.get((split_id, sl), {}).get('name', f'Sub {sl}')
                    working_names[new_base + sl] = f"{working_names.get(split_id, '')} > {name}"
            del working_names[split_id]

            st.session_state.working_labels = working_labels
            st.session_state.working_names = working_names

            log = {'action': 'split', 'cluster': split_id, 'n_new': n_subs,
                   'timestamp': datetime.now().isoformat()}
            RESULTS_DIR.mkdir(exist_ok=True)
            with open(RESULTS_DIR / 'rail_edits.jsonl', 'a') as f:
                f.write(json.dumps(log) + '\n')
            st.success(f"Split into {n_subs} new clusters.")
            st.rerun()
    else:
        st.caption("No sub-cluster data available for this cluster.")

st.markdown("---")

# --- Manual Assignment ---
st.subheader("Manual Film Assignment")
film_options = [f"{title_lookup.get(tt, tt)} ({tt})" for tt in tt_codes]
film_selected = st.selectbox("Search film:", film_options, key="manual_film")
film_tt = film_selected.split("(")[-1].rstrip(")")
film_idx = tt_codes.index(film_tt) if film_tt in tt_codes else 0

current_cl = working_labels[film_idx]
current_name = working_names.get(current_cl, 'Outlier' if current_cl == -1 else f'Cluster {current_cl}')
st.info(f"Current assignment: **{current_name}** (ID: {current_cl})")

new_cluster = st.selectbox("Assign to:", [-1] + sorted(working_names.keys()),
                            format_func=lambda x: f"Outlier" if x == -1 else f"{x}: {working_names.get(x, '')}",
                            key="new_cluster")
reason = st.text_input("Reason for override:")

if st.button("Confirm Assignment") and new_cluster != current_cl:
    working_labels[film_idx] = new_cluster
    st.session_state.working_labels = working_labels
    log = {'action': 'manual_assign', 'tt_code': film_tt, 'from': int(current_cl),
           'to': int(new_cluster), 'reason': reason, 'timestamp': datetime.now().isoformat()}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / 'rail_edits.jsonl', 'a') as f:
        f.write(json.dumps(log) + '\n')
    st.success(f"Reassigned {title_lookup.get(film_tt, film_tt)} to {working_names.get(new_cluster, 'Outlier')}.")

st.markdown("---")

# --- Export ---
st.subheader("Rail Export")

export_data = []
_probs = arts.get('probabilities', arts['clusterer'].probabilities_ if 'clusterer' in arts else np.zeros(len(tt_codes)))
for i, tt in enumerate(tt_codes):
    cl = working_labels[i]
    prob = _probs[i]
    sc = arts.get('soft_clusters')
    sec_cl = -1
    sec_prob = 0
    if sc is not None and cl >= 0:
        sorted_idx = np.argsort(sc[i])[::-1]
        if len(sorted_idx) > 1:
            sec_cl = int(sorted_idx[1])
            sec_prob = float(sc[i][sorted_idx[1]])

    tier = 'Clustered'
    if arts.get('tier1_mask') is not None and arts['tier1_mask'][i]:
        tier = 'Tier 1'
    elif arts.get('tier2_mask') is not None and arts['tier2_mask'][i]:
        tier = 'Tier 2'
    elif arts.get('tier3_mask') is not None and arts['tier3_mask'][i]:
        tier = 'Tier 3'
    elif cl == -1:
        tier = 'Outlier'

    export_data.append({
        'tt_code': tt,
        'title': title_lookup.get(tt, tt),
        'rail_id': int(cl),
        'rail_name': working_names.get(cl, 'Outlier' if cl == -1 else f'Cluster {cl}'),
        'confidence': float(prob),
        'secondary_rail': sec_cl,
        'secondary_prob': sec_prob,
        'outlier_tier': tier,
    })

export_df = pd.DataFrame(export_data)
csv = export_df.to_csv(index=False)
st.download_button("Download rail mapping (CSV)", csv, "rail_mapping.csv", "text/csv")

st.caption(f"{len(export_df)} films | {len(working_names)} active rails | "
           f"{(working_labels == -1).sum()} outliers")

# --- Reset ---
if st.button("Reset to Pipeline Results"):
    st.session_state.working_labels = arts['cluster_labels'].copy()
    st.session_state.working_names = {
        cid: cluster_names.get(cid, {}).get('name', f'Cluster {cid}')
        for cid in range(n_clusters)
    }
    st.rerun()
