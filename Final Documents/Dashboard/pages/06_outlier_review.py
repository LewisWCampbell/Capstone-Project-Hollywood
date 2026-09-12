"""Page 7 — Outlier Review: Approve or reject XGBoost recovery suggestions for outlier films."""

import streamlit as st
import numpy as np
import json as json_mod
from datetime import datetime
from pathlib import Path
from utils.data import (
    load_pipeline_artifacts, load_omdb, build_title_lookup, get_poster_src, RESULTS_DIR,
)

st.set_page_config(page_title="Outlier Review", layout="wide")

# ── Persistence helpers ──────────────────────────────────────────────────────
RUNS_DIR = RESULTS_DIR / 'runs'


def _next_outlier_run_id() -> int:
    """Return the next available outlier run ID."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(RUNS_DIR.glob('outlier_run_*.json'))
    if not existing:
        return 1
    ids = []
    for f in existing:
        try:
            ids.append(int(f.stem.split('_')[2]))
        except (IndexError, ValueError):
            pass
    return max(ids, default=0) + 1


def _conf_color(conf: float) -> str:
    """Return hex colour for a confidence value."""
    if conf >= 0.95:
        return '#a3be8c'
    if conf >= 0.80:
        return '#ebcb8b'
    if conf >= 0.50:
        return '#d08770'
    return '#bf616a'


# ── Load data ────────────────────────────────────────────────────────────────
arts = load_pipeline_artifacts()
if arts is None:
    st.error('Pipeline artifacts not found. Run the notebook first.')
    st.stop()

recovery_suggestions = arts.get('recovery_suggestions', None)
if not recovery_suggestions:
    st.warning(
        'No recovery suggestions found in pipeline artifacts. '
        'This page requires the **Experimental** pipeline (XGBoost outlier recovery). '
        'Run the experimental notebook through the Outlier Recovery cell, then re-export artifacts.'
    )
    st.stop()

cluster_labels = arts['cluster_labels']
cluster_names = arts.get('cluster_names', {})
probabilities = arts.get('probabilities', np.zeros(len(cluster_labels)))
tt_codes = arts['tt_codes']
title_lookup = arts.get('title_lookup', {}) or build_title_lookup()
omdb = load_omdb()

# ── Initialise session state ─────────────────────────────────────────────────
if 'outlier_decisions' not in st.session_state:
    st.session_state.outlier_decisions = {}  # {global_idx_str: "approved"|"rejected"}
if 'outlier_unsaved' not in st.session_state:
    st.session_state.outlier_unsaved = 0
if 'outlier_page' not in st.session_state:
    st.session_state.outlier_page = 0

decisions = st.session_state.outlier_decisions

# ── Build outlier list ───────────────────────────────────────────────────────
outliers = []
for gi_str, info in recovery_suggestions.items():
    gi = int(gi_str)
    tt = info.get('tt_code', tt_codes[gi] if gi < len(tt_codes) else '')
    omdb_entry = omdb.get(tt, {})

    # Extract year
    year_raw = omdb_entry.get('Year', '')
    try:
        year = int(str(year_raw)[:4])
    except (ValueError, TypeError):
        year = 0

    suggested_cid = info.get('suggested_cluster', -1)
    cname_entry = cluster_names.get(suggested_cid, {})
    cname = cname_entry.get('name', f'Cluster {suggested_cid}') if isinstance(cname_entry, dict) else str(cname_entry)

    outliers.append({
        'gi': gi,
        'tt': tt,
        'title': info.get('title', title_lookup.get(tt, tt)),
        'year': year,
        'poster': get_poster_src(tt, omdb_entry, size='140x210'),
        'suggested_cluster': suggested_cid,
        'cluster_name': cname,
        'confidence': info.get('confidence', 0.0),
    })

# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title('Outlier Review')
st.caption(
    f'{len(outliers)} outlier films with XGBoost recovery suggestions — '
    'approve to assign to a rail, reject to keep as outlier.'
)

# ── Summary metrics ──────────────────────────────────────────────────────────
n_approved = sum(1 for o in outliers if decisions.get(str(o['gi'])) == 'approved')
n_rejected = sum(1 for o in outliers if decisions.get(str(o['gi'])) == 'rejected')
n_pending = len(outliers) - n_approved - n_rejected

m1, m2, m3, m4 = st.columns(4)
m1.metric('Total Outliers', len(outliers))
m2.metric('Approved', n_approved, delta=f'+{n_approved}' if n_approved else None,
          delta_color='normal')
m3.metric('Rejected', n_rejected, delta=f'-{n_rejected}' if n_rejected else None,
          delta_color='inverse')
m4.metric('Pending', n_pending)

# ══════════════════════════════════════════════════════════════════════════════
# TOOLBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
tb1, tb2, tb3, tb4 = st.columns([2, 2, 2, 2])

with tb1:
    sort_by = st.selectbox(
        'Sort by',
        ['Confidence (High → Low)', 'Confidence (Low → High)',
         'Year (Newest)', 'Year (Oldest)', 'Title (A-Z)'],
        label_visibility='collapsed',
        key='outlier_sort',
    )

with tb2:
    # Build cluster filter options
    _rail_options = ['All Rails'] + sorted(
        set(o['cluster_name'] for o in outliers),
        key=lambda x: x.lower(),
    )
    rail_filter = st.selectbox(
        'Filter by rail', _rail_options,
        label_visibility='collapsed',
        key='outlier_rail_filter',
    )

with tb3:
    status_filter = st.selectbox(
        'Status', ['All', 'Pending', 'Approved', 'Rejected'],
        label_visibility='collapsed',
        key='outlier_status_filter',
    )

with tb4:
    conf_min = st.slider(
        'Min confidence', 0, 100, 0, 5,
        format='%d%%',
        key='outlier_conf_slider',
    )

# ── Batch actions ────────────────────────────────────────────────────────────
ba1, ba2, ba3, ba4, _ = st.columns([1.5, 1.5, 1, 1, 3])

with ba1:
    batch_threshold = st.number_input(
        'Batch threshold (%)', min_value=50, max_value=100, value=80, step=5,
        format='%d', label_visibility='collapsed',
    )
with ba2:
    if st.button(f'✓ Approve all ≥ {batch_threshold}%', use_container_width=True):
        _count = 0
        for o in outliers:
            if o['confidence'] >= batch_threshold / 100.0 and decisions.get(str(o['gi'])) != 'approved':
                decisions[str(o['gi'])] = 'approved'
                _count += 1
        if _count:
            st.session_state.outlier_unsaved += _count
            st.rerun()

with ba3:
    if st.button('↩ Reset All', use_container_width=True):
        st.session_state.outlier_decisions = {}
        st.session_state.outlier_unsaved = 0
        st.session_state.outlier_page = 0
        st.rerun()

with ba4:
    _save_label = f'💾 Save ({st.session_state.outlier_unsaved})' if st.session_state.outlier_unsaved else '💾 Save'
    _save_disabled = st.session_state.outlier_unsaved == 0
    if st.button(_save_label, use_container_width=True, disabled=_save_disabled, type='primary'):
        st.session_state.outlier_show_save = True
        st.rerun()

# ── Save dialog ──────────────────────────────────────────────────────────────
if st.session_state.get('outlier_show_save', False):
    @st.dialog('Save Outlier Decisions')
    def _save_dialog():
        run_name = st.text_input('Run name', value=f'outlier_recovery_{datetime.now().strftime("%Y%m%d_%H%M")}')
        notes = st.text_area('Notes (optional)', placeholder='e.g. Approved all high-confidence recoveries')

        c1, c2 = st.columns(2)
        if c1.button('Cancel', use_container_width=True):
            st.session_state.outlier_show_save = False
            st.rerun()

        if c2.button('Save', type='primary', use_container_width=True):
            # Build updated cluster labels
            new_labels = np.array(cluster_labels).copy()
            for o in outliers:
                d = decisions.get(str(o['gi']))
                if d == 'approved':
                    new_labels[o['gi']] = o['suggested_cluster']

            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            run_id = _next_outlier_run_id()
            payload = {
                'run_type': 'outlier_recovery',
                'run_id': run_id,
                'run_name': run_name,
                'timestamp': datetime.now().isoformat(),
                'notes': notes,
                'decisions': dict(decisions),
                'updated_cluster_labels': [int(x) for x in new_labels],
                'stats': {
                    'total_outliers': len(outliers),
                    'approved': n_approved,
                    'rejected': n_rejected,
                    'pending': n_pending,
                },
            }
            out_path = RUNS_DIR / f'outlier_run_{run_id:03d}.json'
            with open(out_path, 'w') as f:
                json_mod.dump(payload, f, indent=2)

            st.session_state.outlier_unsaved = 0
            st.session_state.outlier_show_save = False
            st.toast(f'Saved as outlier_run_{run_id:03d}.json')
            st.rerun()

    _save_dialog()

# ══════════════════════════════════════════════════════════════════════════════
# FILTER & SORT
# ══════════════════════════════════════════════════════════════════════════════
filtered = list(outliers)

# Rail filter
if rail_filter != 'All Rails':
    filtered = [o for o in filtered if o['cluster_name'] == rail_filter]

# Status filter
if status_filter == 'Pending':
    filtered = [o for o in filtered if str(o['gi']) not in decisions]
elif status_filter == 'Approved':
    filtered = [o for o in filtered if decisions.get(str(o['gi'])) == 'approved']
elif status_filter == 'Rejected':
    filtered = [o for o in filtered if decisions.get(str(o['gi'])) == 'rejected']

# Confidence filter (conf_min is 0-100 integer, confidence is 0.0-1.0 float)
filtered = [o for o in filtered if o['confidence'] >= conf_min / 100.0]

# Sort
if sort_by == 'Confidence (High → Low)':
    filtered.sort(key=lambda x: x['confidence'], reverse=True)
elif sort_by == 'Confidence (Low → High)':
    filtered.sort(key=lambda x: x['confidence'])
elif sort_by == 'Year (Newest)':
    filtered.sort(key=lambda x: x['year'], reverse=True)
elif sort_by == 'Year (Oldest)':
    filtered.sort(key=lambda x: x['year'])
elif sort_by == 'Title (A-Z)':
    filtered.sort(key=lambda x: x['title'].lower())

# ══════════════════════════════════════════════════════════════════════════════
# CARD GRID CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.outlier-card {
    width: 140px; text-align: center; position: relative;
    border-radius: 8px; padding: 4px; transition: all 0.2s;
}
.outlier-card img {
    width: 140px; height: 210px; object-fit: cover; border-radius: 6px;
}
.outlier-card .card-title {
    font-size: 0.7em; font-weight: 600; line-height: 1.2;
    margin-top: 4px; height: 2.4em; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.outlier-card .card-meta { font-size: 0.6em; color: #888; margin-top: 2px; }
.outlier-card .card-rail {
    font-size: 0.6em; margin-top: 2px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    transition: font-size 0.2s ease, color 0.2s ease, font-weight 0.2s ease;
}
.outlier-card:hover .card-rail {
    font-size: 0.95em;
    font-weight: 700;
    color: #ebcb8b;
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    margin-top: 4px;
}
.outlier-card .status-badge {
    position: absolute; top: 6px; right: 6px; width: 24px; height: 24px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: bold;
}
.outlier-card.approved {
    border: 2px solid #a3be8c;
}
.outlier-card.approved .status-badge { background: #a3be8c; color: #0d1117; }
.outlier-card.rejected {
    border: 2px solid #bf616a; opacity: 0.5;
}
.outlier-card.rejected .status-badge { background: #bf616a; color: #0d1117; }
.outlier-card.pending {
    border: 2px solid rgba(255,255,255,0.08);
}
.outlier-card.pending .status-badge {
    background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.4);
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RENDER OUTLIER CARDS
# ══════════════════════════════════════════════════════════════════════════════
PER_PAGE = 40
total_pages = max(1, (len(filtered) + PER_PAGE - 1) // PER_PAGE)
page = min(st.session_state.outlier_page, total_pages - 1)
page_start = page * PER_PAGE
page_end = min(page_start + PER_PAGE, len(filtered))
page_items = filtered[page_start:page_end]

if not filtered:
    st.info('No outliers match the current filters.')
    st.stop()

st.caption(f'Showing {page_start + 1}–{page_end} of {len(filtered)} outliers')

N_COLS = 8
cols = st.columns(N_COLS)

for idx, o in enumerate(page_items):
    gi_str = str(o['gi'])
    decision = decisions.get(gi_str)
    css_class = 'approved' if decision == 'approved' else ('rejected' if decision == 'rejected' else 'pending')
    badge_icon = '✓' if decision == 'approved' else ('✗' if decision == 'rejected' else '?')
    cc = _conf_color(o['confidence'])
    film_link = f'/film_explorer?tt={o["tt"]}'

    with cols[idx % N_COLS]:
        st.markdown(
            f"<a href='{film_link}' target='_self' style='text-decoration:none; color:inherit;'>"
            f"<div class='outlier-card {css_class}'>"
            f"<div class='status-badge'>{badge_icon}</div>"
            f"<img src='{o['poster']}' loading='lazy'>"
            f"<div class='card-title'>{o['title']}</div>"
            f"<div class='card-meta'>{o['year'] if o['year'] else '?'} · "
            f"<span style='color:{cc}; font-weight:600;'>{o['confidence']:.0%}</span></div>"
            f"<div class='card-rail'>→ {o['cluster_name']}</div>"
            f"</div></a>",
            unsafe_allow_html=True,
        )

        # Action buttons
        _b1, _b2 = st.columns(2)
        with _b1:
            _ap_label = '✓' if decision != 'approved' else '↩'
            if st.button(_ap_label, key=f'ap_{o["gi"]}', use_container_width=True):
                if decision == 'approved':
                    decisions.pop(gi_str, None)
                else:
                    decisions[gi_str] = 'approved'
                st.session_state.outlier_unsaved += 1
                st.rerun()
        with _b2:
            _rj_label = '✗' if decision != 'rejected' else '↩'
            if st.button(_rj_label, key=f'rj_{o["gi"]}', use_container_width=True):
                if decision == 'rejected':
                    decisions.pop(gi_str, None)
                else:
                    decisions[gi_str] = 'rejected'
                st.session_state.outlier_unsaved += 1
                st.rerun()

# ── Pagination ───────────────────────────────────────────────────────────────
if total_pages > 1:
    st.markdown('---')
    p1, p2, p3 = st.columns([1, 3, 1])
    with p1:
        if st.button('← Previous', disabled=page == 0, use_container_width=True):
            st.session_state.outlier_page = max(0, page - 1)
            st.rerun()
    with p2:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px; color:#888;'>"
            f"Page {page + 1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with p3:
        if st.button('Next →', disabled=page >= total_pages - 1, use_container_width=True):
            st.session_state.outlier_page = min(total_pages - 1, page + 1)
            st.rerun()
