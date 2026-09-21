"""Ollama API integration for AI-powered cluster naming."""

import requests
import json
import numpy as np

OLLAMA_MODEL = 'llama3.2:3b'  # Can be overridden via environment variables if needed later

def get_discriminative_features(cluster_id, X, cluster_labels, feature_names, top_n=20):
    """Compute discriminative features for a given cluster vs the global mean."""
    mask = cluster_labels == cluster_id
    if mask.sum() == 0:
        return {}, {}
        
    cluster_centroid = X[mask].mean(axis=0)
    global_centroid = X.mean(axis=0)
    
    delta = cluster_centroid - global_centroid
    
    # Sort indices
    sorted_idx = np.argsort(delta)
    top_pos_idx = sorted_idx[-top_n:][::-1]
    top_neg_idx = sorted_idx[:top_n]
    
    top_pos = {feature_names[i]: float(delta[i]) for i in top_pos_idx if delta[i] > 0}
    top_neg = {feature_names[i]: float(delta[i]) for i in top_neg_idx if delta[i] < 0}
    
    return top_pos, top_neg

def query_ollama(prompt, model=OLLAMA_MODEL):
    """Submit a prompt to the local Ollama instance and return the response."""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.3}
            },
            timeout=180
        )
        data = response.json()
        if 'error' in data:
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data['response'].strip()
    except Exception as e:
        print(f"  ⚠ Ollama call failed: {e}")
        return json.dumps({"name": "Unnamed", "description": f"LLM unavailable: {e}"})

def name_cluster(cluster_id, sample_tt_codes, X, cluster_labels, feature_names, title_lookup):
    """Generate a name and description for a cluster by profiling its distinctive features."""
    
    top_pos, top_neg = get_discriminative_features(cluster_id, X, cluster_labels, feature_names)

    pos_str = '\n'.join([f'  - {f}: +{v:.3f} above average' for f, v in top_pos.items()])
    neg_str = '\n'.join([f'  - {f}: {v:.3f} below average'  for f, v in top_neg.items()])

    sample_str = ''
    if sample_tt_codes and title_lookup:
        titles = [title_lookup.get(tt, tt) for tt in sample_tt_codes]
        sample_str = f'\nExample movies: {", ".join(titles)}'

    prompt = f"""You are a film taxonomy expert. A cluster of movies has been grouped by
the similarity of their content profile relative to their own averages — meaning these
movies share the same pattern of which elements are prominent and which are understated.

FEATURES ELEVATED ABOVE AVERAGE IN THIS CLUSTER:
{pos_str}

FEATURES SUPPRESSED BELOW AVERAGE IN THIS CLUSTER:
{neg_str}
{sample_str}

Based on this profile, give this cluster a precise, evocative name (3-6 words)
and a one-sentence description of what unifies these films.

Respond in JSON only:
{{"name": "cluster name", "description": "one sentence"}}"""

    raw = query_ollama(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if the LLM output was poorly formatted
        return {'name': f'Cluster {cluster_id}', 'description': raw}
