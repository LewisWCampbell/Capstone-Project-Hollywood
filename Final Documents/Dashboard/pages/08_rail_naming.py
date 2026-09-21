"""Page 8 - Rail Naming: regenerate every rail and sub-rail name with the LLM backend.

Runs inside the app so the hosted deployment can use its own HF_TOKEN secret.
Results are written to results/artifacts/cluster_names.json and
sub_cluster_names.json (picked up immediately by every other page) and offered
as downloads so they can be committed back to the repository.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data import load_pipeline_artifacts, RESULTS_DIR  # noqa: E402
from utils import llm as _llm  # noqa: E402
from utils.llm import get_genre_decade_summary, query_ollama  # noqa: E402

st.title("Rail Naming")
st.markdown(
    "Regenerate the name and one-line description of **every rail and sub-rail** using the "
    "configured language-model backend. Each rail is described to the model by its twelve "
    "highest-confidence films, its genre mix and its dominant decade. The prompt forbids naming a "
    "rail after a single film or franchise."
)

ART_DIR = RESULTS_DIR / 'artifacts'


# ── Access control: this spends the app owner's LLM credits ──────────────────
def _admin_key() -> str:
    key = os.getenv('ADMIN_KEY', '')
    if not key:
        try:
            key = str(st.secrets.get('ADMIN_KEY', '') or '')
        except Exception:
            key = ''
    return key


_expected = _admin_key()
if _expected:
    _given = st.text_input("Admin key", type="password",
                           help="Set ADMIN_KEY in secrets to protect this page.")
    if _given != _expected:
        st.info("Enter the admin key to run batch naming. Every other page is unaffected.")
        st.stop()
else:
    st.warning("No ADMIN_KEY secret is set, so anyone who can open this page can trigger the batch job.")

backend = _llm.llm_backend()
st.markdown(f"**LLM backend:** `{backend}`" + (f" · model `{_llm.HF_MODEL}`" if backend == 'huggingface' else ''))
if backend == 'none':
    st.error("No LLM backend available: start Ollama locally or set HF_TOKEN in secrets.")
    st.stop()

arts = load_pipeline_artifacts()
if arts is None:
    st.error("Pipeline artifacts not found.")
    st.stop()

X = arts['X']
feature_names = arts['feature_names']
tt_codes = arts['tt_codes']
labels = np.asarray(arts['cluster_labels'])
probs = np.asarray(arts['probabilities'])
title_lookup = arts['title_lookup']
cluster_names = dict(arts['cluster_names'])
sub_cluster_names = dict(arts['sub_cluster_names'])
sub_clusters = arts['sub_clusters']
n_clusters = int(arts['n_clusters'])


# ── Prompt construction ──────────────────────────────────────────────────────
def _rep_titles(idx, n=12):
    order = idx[np.argsort(probs[idx])[::-1]]
    return [title_lookup.get(str(tt_codes[i]), str(tt_codes[i])) for i in order[:n]]


def _name_prompt(titles, genres, decade, n, parent_name=None):
    scope = ''
    if parent_name:
        scope = (f'This is a sub-rail nested inside the larger rail "{parent_name}"; its name must be '
                 f'more specific than the parent\'s and highlight what sets this group apart.\n')
    return (f'These films are grouped together as one row (a "rail") on a streaming service:\n'
            f'{", ".join(titles)}\n\nGenre mix: {genres}\nDominant decade: {decade}s\n'
            f'Number of films in the rail: {n}\n{scope}\n'
            f'Give this rail a catchy 2-5 word name that describes what unites these films as a whole, '
            f'in the style of streaming-service categories. Rules: do not name it after a single film, '
            f'character or franchise; do not use the word "cluster"; no quotation marks; title case.\n'
            f'Respond with ONLY the name.')


def _desc_prompt(name, titles, genres, decade):
    return (f'The streaming rail "{name}" contains these representative films:\n{", ".join(titles)}\n\n'
            f'Genre mix: {genres}\nDominant decade: {decade}s\n\n'
            f'Write ONE sentence (max 25 words) describing what unites the films in this rail, as it '
            f'would appear under the rail title on a streaming homepage. Respond with ONLY the sentence.')


def _clean(text):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S).strip()
    text = text.strip().strip('"').strip("'").strip()
    return text.splitlines()[0].strip().rstrip('.') if text else ''


# ── Build the job list ───────────────────────────────────────────────────────
jobs = []
for cid in range(n_clusters):
    idx = np.where(labels == cid)[0]
    if idx.size == 0:
        continue
    jobs.append({'kind': 'parent', 'cid': cid, 'sid': None, 'idx': idx,
                 'old': cluster_names.get(cid, {}).get('name', f'Cluster {cid}')})
for pcid, arr in sub_clusters.items():
    pidx = np.where(labels == int(pcid))[0]
    arr = np.asarray(arr)
    if arr.shape[0] != pidx.shape[0]:
        continue
    for sid in sorted(set(int(s) for s in arr if s >= 0)):
        idx = pidx[arr == sid]
        old = (sub_cluster_names.get((int(pcid), sid), {}) or {}).get('name', f'Sub-cluster {sid}')
        jobs.append({'kind': 'sub', 'cid': int(pcid), 'sid': sid, 'idx': idx, 'old': old})

n_parent = sum(1 for j in jobs if j['kind'] == 'parent')
st.markdown(f"**{n_parent} rails** and **{len(jobs) - n_parent} sub-rails** will be renamed "
            f"({2 * len(jobs)} model calls).")

scope = st.radio("Scope", ["Rails and sub-rails", "Rails only", "Sub-rails only"], horizontal=True)
if scope == "Rails only":
    jobs = [j for j in jobs if j['kind'] == 'parent']
elif scope == "Sub-rails only":
    jobs = [j for j in jobs if j['kind'] == 'sub']

# ── Run ──────────────────────────────────────────────────────────────────────
if st.button("Regenerate names", type="primary"):
    results = []
    bar = st.progress(0.0)
    status = st.empty()
    failures = 0
    for i, j in enumerate(jobs):
        titles = _rep_titles(j['idx'])
        genres, decade = get_genre_decade_summary(j['idx'], X, feature_names)
        parent_name = None
        if j['kind'] == 'sub':
            parent_name = cluster_names.get(j['cid'], {}).get('name', f"Cluster {j['cid']}")
        label = f"Rail {j['cid']}" if j['kind'] == 'parent' else f"Rail {j['cid']} / sub {j['sid']}"
        status.markdown(f"Naming **{label}** ({i + 1}/{len(jobs)})...")

        name = _clean(query_ollama(_name_prompt(titles, genres, decade, len(j['idx']), parent_name)))
        desc = _clean(query_ollama(_desc_prompt(name, titles, genres, decade))) if name else ''
        if not name:
            failures += 1
            results.append({**j, 'name': '', 'description': '', 'error': _llm.LAST_LLM_ERROR})
        else:
            results.append({**j, 'name': name, 'description': desc, 'genres': genres})
            if j['kind'] == 'parent':
                entry = dict(cluster_names.get(j['cid'], {}))
                entry.update({'name': name, 'description': desc, 'top_features': genres,
                              'method': 'movies_first_batch'})
                cluster_names[j['cid']] = entry
            else:
                key = (j['cid'], j['sid'])
                entry = dict(sub_cluster_names.get(key, {}) or {})
                entry.update({'name': name, 'description': desc})
                sub_cluster_names[key] = entry
        bar.progress((i + 1) / len(jobs))

    status.empty()
    # ── Persist (clean keys) so every other page picks the names up ──────────
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ART_DIR / 'cluster_names.json', 'w') as f:
        json.dump({str(k): v for k, v in cluster_names.items()}, f, indent=1)
    with open(ART_DIR / 'sub_cluster_names.json', 'w') as f:
        json.dump({f'({k[0]}, {k[1]})': v for k, v in sub_cluster_names.items()
                   if isinstance(k, tuple)}, f, indent=1)
    st.session_state['batch_naming_results'] = results
    if failures:
        st.warning(f"{failures} of {len(jobs)} names failed. Last error: {_llm.LAST_LLM_ERROR}")
    else:
        st.success(f"Renamed {len(jobs)} rails. Other pages now show the new names.")

# ── Review + download ────────────────────────────────────────────────────────
results = st.session_state.get('batch_naming_results')
if results:
    st.subheader("Old vs new")
    rows = [{'Rail': (f"{r['cid']}" if r['kind'] == 'parent' else f"{r['cid']} / {r['sid']}"),
             'Type': r['kind'], 'Films': int(len(r['idx'])), 'Old name': r['old'],
             'New name': r['name'] or f"FAILED: {r.get('error', '')}",
             'New description': r['description']} for r in results]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.download_button("Download cluster_names.json",
                       data=(ART_DIR / 'cluster_names.json').read_text(),
                       file_name='cluster_names.json', mime='application/json')
    c2.download_button("Download sub_cluster_names.json",
                       data=(ART_DIR / 'sub_cluster_names.json').read_text(),
                       file_name='sub_cluster_names.json', mime='application/json')
    st.caption(
        "Hosted deployments have an ephemeral filesystem: the new names last until the next "
        "redeploy unless the two files above are committed to "
        "`Final Documents/Dashboard/results/artifacts/`."
    )
