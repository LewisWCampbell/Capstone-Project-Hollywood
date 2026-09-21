"""Automatic rail naming.

On the first visit after a deployment, `ensure_rail_names()` starts a background
thread that names every rail and sub-rail with the configured LLM backend
(local Ollama, or Hugging Face via HF_TOKEN). Names are written to
results/artifacts/cluster_names.json and sub_cluster_names.json as they arrive,
so every page picks them up on its next render. Progress is recorded in
results/artifacts/naming_status.json for a small status banner.

The run is skipped when the committed names were already produced by this
batch process (method == 'movies_first_batch'), so a repository that ships
generated names never spends LLM credits on startup.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data import RESULTS_DIR
from utils import llm as _llm
from utils.llm import get_genre_decade_summary, query_ollama

ART_DIR = RESULTS_DIR / 'artifacts'
STATUS_PATH = ART_DIR / 'naming_status.json'
BATCH_METHOD = 'movies_first_batch'

_lock = threading.Lock()
_started = False


# ── Prompts ──────────────────────────────────────────────────────────────────
def name_prompt(titles, genres, decade, n, parent_name=None):
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


def desc_prompt(name, titles, genres, decade):
    return (f'The streaming rail "{name}" contains these representative films:\n{", ".join(titles)}\n\n'
            f'Genre mix: {genres}\nDominant decade: {decade}s\n\n'
            f'Write ONE sentence (max 25 words) describing what unites the films in this rail, as it '
            f'would appear under the rail title on a streaming homepage. Respond with ONLY the sentence.')


def clean(text):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S).strip()
    text = text.strip().strip('"').strip("'").strip()
    return text.splitlines()[0].strip().rstrip('.') if text else ''


# ── Status file ──────────────────────────────────────────────────────────────
def read_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:
        return {}


def _write_status(**kw):
    s = read_status()
    s.update(kw)
    s['updated'] = time.time()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(s))


# ── Data access (no Streamlit calls: safe inside a background thread) ────────
def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def _parse_sub_key(k):
    nums = re.findall(r'-?\d+', str(k))
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


def load_inputs():
    X_df = pd.read_parquet(ART_DIR / 'X_raw.parquet', engine='pyarrow')
    labels_df = pd.read_parquet(ART_DIR / 'cluster_labels.parquet', engine='pyarrow')
    feature_cols = _read_json(ART_DIR / 'feature_columns.json')
    return {
        'X': X_df.values,
        'feature_names': feature_cols.get('feature_names', list(X_df.columns)),
        'tt_codes': feature_cols.get('tt_codes', list(X_df.index)),
        'labels': labels_df['cluster_label'].values,
        'probs': labels_df['probability'].values,
        'title_lookup': _read_json(ART_DIR / 'title_lookup.json'),
        'cluster_names': {int(k): v for k, v in _read_json(ART_DIR / 'cluster_names.json').items()},
        'sub_cluster_names': {_parse_sub_key(k): v for k, v in _read_json(ART_DIR / 'sub_cluster_names.json').items()
                              if _parse_sub_key(k)},
        'sub_clusters': {int(k): np.asarray(v) for k, v in _read_json(ART_DIR / 'sub_clusters.json').items()},
        'n_clusters': int(_read_json(ART_DIR / 'validation_metrics.json').get('n_clusters', 0)),
    }


def build_jobs(d):
    labels, probs = d['labels'], d['probs']
    n_clusters = d['n_clusters'] or (int(labels.max()) + 1)
    jobs = []
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        if idx.size:
            jobs.append({'kind': 'parent', 'cid': cid, 'sid': None, 'idx': idx,
                         'old': d['cluster_names'].get(cid, {}).get('name', f'Cluster {cid}')})
    for pcid, arr in d['sub_clusters'].items():
        pidx = np.where(labels == pcid)[0]
        if arr.shape[0] != pidx.shape[0]:
            continue
        for sid in sorted(set(int(s) for s in arr if s >= 0)):
            idx = pidx[arr == sid]
            old = (d['sub_cluster_names'].get((pcid, sid), {}) or {}).get('name', f'Sub-cluster {sid}')
            jobs.append({'kind': 'sub', 'cid': pcid, 'sid': sid, 'idx': idx, 'old': old})
    return jobs


def names_need_generation(d=None) -> bool:
    """True unless every rail already carries a batch-generated name."""
    d = d or {'cluster_names': {int(k): v for k, v in _read_json(ART_DIR / 'cluster_names.json').items()}}
    names = d['cluster_names']
    if not names:
        return True
    return any(v.get('method') != BATCH_METHOD for v in names.values())


def _save(d):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ART_DIR / 'cluster_names.json', 'w') as f:
        json.dump({str(k): v for k, v in d['cluster_names'].items()}, f, indent=1)
    with open(ART_DIR / 'sub_cluster_names.json', 'w') as f:
        json.dump({f'({k[0]}, {k[1]})': v for k, v in d['sub_cluster_names'].items()}, f, indent=1)


# ── The batch run ────────────────────────────────────────────────────────────
def run_batch(scope='all', progress=None, d=None) -> list:
    """Name every rail/sub-rail. Returns a list of result dicts. Safe to call from a thread."""
    d = d or load_inputs()
    jobs = build_jobs(d)
    if scope == 'parent':
        jobs = [j for j in jobs if j['kind'] == 'parent']
    elif scope == 'sub':
        jobs = [j for j in jobs if j['kind'] == 'sub']

    results, failures = [], 0
    _write_status(state='running', done=0, total=len(jobs), failures=0, backend=_llm.llm_backend())
    for i, j in enumerate(jobs):
        idx = j['idx']
        order = idx[np.argsort(d['probs'][idx])[::-1]]
        titles = [d['title_lookup'].get(str(d['tt_codes'][k]), str(d['tt_codes'][k])) for k in order[:12]]
        genres, decade = get_genre_decade_summary(idx, d['X'], d['feature_names'])
        parent_name = None
        if j['kind'] == 'sub':
            parent_name = d['cluster_names'].get(j['cid'], {}).get('name', f"Cluster {j['cid']}")

        name = clean(query_ollama(name_prompt(titles, genres, decade, len(idx), parent_name)))
        desc = clean(query_ollama(desc_prompt(name, titles, genres, decade))) if name else ''
        if not name:
            failures += 1
            results.append({**j, 'name': '', 'description': '', 'error': _llm.LAST_LLM_ERROR})
        else:
            results.append({**j, 'name': name, 'description': desc, 'genres': genres})
            if j['kind'] == 'parent':
                entry = dict(d['cluster_names'].get(j['cid'], {}))
                entry.update({'name': name, 'description': desc, 'top_features': genres, 'method': BATCH_METHOD})
                d['cluster_names'][j['cid']] = entry
            else:
                key = (j['cid'], j['sid'])
                entry = dict(d['sub_cluster_names'].get(key, {}) or {})
                entry.update({'name': name, 'description': desc, 'method': BATCH_METHOD})
                d['sub_cluster_names'][key] = entry
            _save(d)  # incremental: pages see names as they land
        _write_status(done=i + 1, failures=failures, last_error=_llm.LAST_LLM_ERROR if not name else '')
        if progress:
            progress(i + 1, len(jobs), j, name)
        time.sleep(0.3)

    _write_status(state='done', done=len(jobs), total=len(jobs), failures=failures)
    return results


def ensure_rail_names(secrets_getter=None):
    """Start the background naming run once per server process, if needed.

    Call from the main script (has Streamlit context). `secrets_getter(name)`
    lets us copy HF_TOKEN / HF_MODEL into the environment for the worker thread,
    which has no Streamlit context of its own.
    """
    global _started
    with _lock:
        if _started:
            return read_status()
        _started = True

    if secrets_getter:
        for key in ('HF_TOKEN', 'HF_MODEL', 'OLLAMA_BASE_URL'):
            val = secrets_getter(key)
            if val and not os.getenv(key):
                os.environ[key] = str(val)
                if key == 'HF_MODEL':
                    _llm.HF_MODEL = str(val)

    if not (ART_DIR / 'X_raw.parquet').exists() or not names_need_generation():
        _write_status(state='skipped', reason='names already generated')
        return read_status()
    if _llm.llm_backend() == 'none':
        _write_status(state='skipped', reason='no LLM backend (set HF_TOKEN)')
        return read_status()

    def _worker():
        try:
            run_batch('all')
        except Exception as e:  # never take the app down
            _write_status(state='error', error=str(e))

    threading.Thread(target=_worker, name='rail-naming', daemon=True).start()
    return read_status()


def status_banner(st):
    """Render a compact status line for the current naming run, if any."""
    s = read_status()
    state = s.get('state')
    if state == 'running':
        st.info(f"Naming rails with the language model in the background: "
                f"{s.get('done', 0)}/{s.get('total', '?')} done. Names appear as they are generated; "
                f"refresh to see the latest.")
    elif state == 'error':
        st.warning(f"Automatic rail naming stopped: {s.get('error', '')}")
    elif state == 'done' and s.get('failures'):
        st.warning(f"Automatic rail naming finished with {s['failures']} failures "
                   f"(last error: {s.get('last_error', '')}).")
