"""LLM integration for AI-powered cluster naming (movies-first approach).

Backends, tried in order:
  1. Local Ollama (http://localhost:11434) - used when running on your own machine.
  2. Hugging Face Inference Providers (OpenAI-compatible router) - used when an
     HF_TOKEN is available, e.g. on Streamlit Community Cloud. Free tier.

Set HF_TOKEN (and optionally HF_MODEL) in SECRETS / env / Streamlit secrets.
Nothing in the prompts references the content-genome feature names: naming is
driven by film titles, genre mix and decades only.
"""

import os
import re
import requests
import json
import numpy as np

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
HF_ROUTER_URL = 'https://router.huggingface.co/v1/chat/completions'
# Default is an ungated model so a fresh HF token works with no license clicks.
HF_MODEL = os.getenv('HF_MODEL', 'Qwen/Qwen3-8B:fastest')

_ollama_available: bool | None = None


def _get_secret(name: str) -> str:
    """Env var first, then Streamlit secrets (if running inside Streamlit)."""
    val = os.getenv(name, '')
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(name, '') or '')
    except Exception:
        return ''


def _ollama_reachable() -> bool:
    global _ollama_available
    if _ollama_available is None:
        try:
            requests.get(f'{OLLAMA_URL}/api/tags', timeout=1.5)
            _ollama_available = True
        except Exception:
            _ollama_available = False
    return _ollama_available


def _query_ollama_local(prompt, model=OLLAMA_MODEL):
    response = requests.post(
        f'{OLLAMA_URL}/api/generate',
        json={'model': model, 'prompt': prompt, 'stream': False,
              'options': {'temperature': 0.3}},
        timeout=180,
    )
    data = response.json()
    if 'error' in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    return data['response'].strip()


def _query_huggingface(prompt, token, model=HF_MODEL):
    response = requests.post(
        HF_ROUTER_URL,
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        json={'model': model,
              'messages': [{'role': 'user', 'content': prompt}],
              'temperature': 0.3, 'max_tokens': 400},
        timeout=60,
    )
    if response.status_code != 200:
        # The router returns HTML (not JSON) for 401/403, so report the status first.
        snippet = response.text[:200].replace('\n', ' ')
        raise RuntimeError(f"Hugging Face HTTP {response.status_code} for {model}: {snippet}")
    data = response.json()
    if 'error' in data:
        raise RuntimeError(f"Hugging Face error: {data['error']}")
    text = data['choices'][0]['message']['content'].strip()
    # Reasoning models (Qwen3 etc.) may prepend a <think>...</think> block.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    return text


LAST_LLM_ERROR = ''


def llm_backend() -> str:
    """Which backend a call would use right now: 'ollama', 'huggingface' or 'none'."""
    if _ollama_reachable():
        return 'ollama'
    if _get_secret('HF_TOKEN'):
        return 'huggingface'
    return 'none'


def query_ollama(prompt, model=OLLAMA_MODEL):
    """Submit a naming prompt to the best available LLM backend.

    Kept under its historical name so existing callers work unchanged.
    Returns '' on any failure so callers fall back to heuristic names.
    """
    try:
        if _ollama_reachable():
            return _query_ollama_local(prompt, model)
        token = _get_secret('HF_TOKEN')
        if token:
            return _query_huggingface(prompt, token)
        print("  No LLM backend available (no local Ollama, no HF_TOKEN).")
        return ''
    except Exception as e:
        global LAST_LLM_ERROR
        LAST_LLM_ERROR = str(e)
        print(f"  LLM call failed: {e}")
        return ''


def get_genre_decade_summary(global_indices, X, feature_names):
    """Compute genre distribution (%) and dominant decade for a set of movies.

    Automatically detects genre_ and decade_ prefixed columns from feature_names.

    Parameters
    ----------
    global_indices : array-like
        Row indices into X for the cluster members.
    X : ndarray
        The full weighted feature matrix.
    feature_names : list
        Column names matching X columns.

    Returns
    -------
    genre_str : str
        e.g. "45% Action, 30% Thriller, 15% Drama"
    decade_str : str
        e.g. "2010s"
    """
    _sub = X[global_indices]

    # Auto-detect genre and decade columns
    _genre_idx = [i for i, f in enumerate(feature_names) if f.startswith('genre_')]
    _decade_idx = [i for i, f in enumerate(feature_names) if f.startswith('decade_')]

    # ── Genre distribution ──
    if _genre_idx:
        _genre_means = _sub[:, _genre_idx].mean(axis=0)
        _total = _genre_means.sum()
        if _total > 0:
            _genre_pcts = (_genre_means / _total * 100)
            _order = np.argsort(_genre_pcts)[::-1]
            _parts = []
            for idx in _order:
                pct = _genre_pcts[idx]
                if pct < 5:
                    break
                name = feature_names[_genre_idx[idx]].replace('genre_', '')
                _parts.append(f'{pct:.0f}% {name}')
            genre_str = ', '.join(_parts) if _parts else 'Mixed genres'
        else:
            genre_str = 'Mixed genres'
    else:
        genre_str = 'No genre data'

    # ── Dominant decade ──
    if _decade_idx:
        _decade_means = _sub[:, _decade_idx].mean(axis=0)
        _best = np.argmax(_decade_means)
        decade_str = feature_names[_decade_idx[_best]].replace('decade_', '')
    else:
        decade_str = 'Mixed decades'

    return genre_str, decade_str


def name_cluster(cluster_id, sample_tt_codes, X, cluster_labels, feature_names,
                 title_lookup, embedding=None, all_cluster_tops=None):
    """Generate a name and description for a cluster using movies-first approach.

    Parameters
    ----------
    cluster_id : int
        The cluster to name.
    sample_tt_codes : list
        TT codes for centroid-closest movies (fallback if embedding is None).
    X : ndarray
        Full weighted feature matrix.
    cluster_labels : ndarray
        Cluster assignment for each row.
    feature_names : list
        Column names matching X columns.
    title_lookup : dict
        Maps tt_code → movie title.
    embedding : ndarray, optional
        UMAP embedding for centroid-closest selection. If None, uses sample_tt_codes directly.
    all_cluster_tops : dict, optional
        Unused (kept for backward compatibility).

    Returns
    -------
    dict with keys: name, description, top_features, method
    """
    mask = cluster_labels == cluster_id
    _mask_idx = np.where(mask)[0]

    if _mask_idx.size == 0:
        return {'name': f'Cluster {cluster_id}', 'description': 'Empty cluster',
                'top_features': '', 'method': 'fallback'}

    # ── 1. Get centroid-closest movies ──
    if embedding is not None:
        _centroid = embedding[_mask_idx].mean(axis=0)
        _dists = np.linalg.norm(embedding[_mask_idx] - _centroid, axis=1)
        _closest_order = np.argsort(_dists)[:10]
        _sample_tts = [sample_tt_codes[j] if j < len(sample_tt_codes)
                       else sample_tt_codes[0] for j in range(10)]
        # Rebuild from mask indices
        from itertools import compress
        _all_tts = sample_tt_codes  # fallback
    else:
        _sample_tts = sample_tt_codes[:10]

    _titles = [title_lookup.get(str(tt), str(tt)) for tt in _sample_tts]

    # ── 2. Genre + decade summary ──
    genre_str, decade_str = get_genre_decade_summary(_mask_idx, X, feature_names)

    # ── 3. LLM Call 1: Name ──
    _name_prompt = (
        f'These movies are grouped together in a streaming service cluster:\n'
        f'{", ".join(_titles)}\n\n'
        f'Genre mix: {genre_str}\n'
        f'Dominant decade: {decade_str}\n\n'
        f'Give this cluster a 2-5 word name suitable for a streaming service rail/category. '
        f'It should be catchy and descriptive of what unifies these films. '
        f'Respond ONLY with the name, nothing else.'
    )

    _raw_name = query_ollama(_name_prompt)
    _name = _raw_name.strip().strip('"').strip("'").strip() if _raw_name else f'Cluster {cluster_id}'

    # ── 4. LLM Call 2: Description ──
    _desc_prompt = (
        f'The streaming rail "{_name}" contains these representative movies:\n'
        f'{", ".join(_titles)}\n\n'
        f'Genre mix: {genre_str}\n'
        f'Dominant decade: {decade_str}\n\n'
        f'Write ONE sentence describing what unifies the movies in this rail. '
        f'Respond ONLY with the sentence, nothing else.'
    )

    _raw_desc = query_ollama(_desc_prompt)
    _desc = _raw_desc.strip().strip('"').strip("'").strip() if _raw_desc else f'Cluster of {mask.sum()} films'

    return {
        'name': _name,
        'description': _desc,
        'top_features': genre_str,
        'method': 'movies_first' if _raw_name else 'fallback',
    }
