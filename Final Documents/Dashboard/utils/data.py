"""Data loading and caching layer for the Movie Genome Dashboard."""

import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent  # Final Documents/
PIPELINE_DATA_DIR = PROJECT_DIR / 'Data for Pipeline'       # raw inputs for notebook
OUTPUT_DATA_DIR = PROJECT_DIR / 'Outputs for Dashboard'     # pipeline outputs
EXTRA_DATA_DIR = PIPELINE_DATA_DIR / 'extra_data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'      # Dashboard/results/
# Legacy alias — some helpers still reference BASE_DIR for imdb_posters etc.
BASE_DIR = PROJECT_DIR
LOCAL_POSTER_DIR = PROJECT_DIR.parent / 'posters'   # repo-root posters/ (Git LFS)


def _mtime_ns(path: Path) -> int:
    """Return file mtime in ns (or -1 when file is missing)."""
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


def check_data_status() -> dict:
    """Quick check of what data files are available."""
    output_csvs = ['movie_clusters_final.csv', 'cluster_summary.csv', 'sub_cluster_summary.csv']
    input_csvs = ['feature_data_longform.csv', 'feature_taxonomy.csv', 'genre_data.csv']
    found = (
        sum(1 for c in output_csvs if (OUTPUT_DATA_DIR / c).exists())
        + sum(1 for c in input_csvs if (PIPELINE_DATA_DIR / c).exists())
    )
    return {
        'csv_count': found,
        'total_csvs': len(output_csvs) + len(input_csvs),
        'artifacts_exist': (RESULTS_DIR / 'artifacts').is_dir(),
        'omdb_exists': (EXTRA_DATA_DIR / 'omdb_movies.csv').exists(),
    }


@st.cache_data
def _load_csv_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    # mtime_ns is included to invalidate cache when file contents change.
    return pd.read_csv(path_str)


def load_clusters_final() -> pd.DataFrame:
    path = OUTPUT_DATA_DIR / 'movie_clusters_final.csv'
    return _load_csv_cached(str(path), _mtime_ns(path))


def load_cluster_summary() -> pd.DataFrame:
    path = OUTPUT_DATA_DIR / 'cluster_summary.csv'
    return _load_csv_cached(str(path), _mtime_ns(path))


def load_sub_cluster_summary() -> pd.DataFrame:
    path = OUTPUT_DATA_DIR / 'sub_cluster_summary.csv'
    return _load_csv_cached(str(path), _mtime_ns(path))


@st.cache_data
def load_feature_longform() -> pd.DataFrame:
    return pd.read_csv(PIPELINE_DATA_DIR / 'feature_data_longform.csv')


@st.cache_data
def load_taxonomy() -> pd.DataFrame:
    return pd.read_csv(PIPELINE_DATA_DIR / 'feature_taxonomy.csv')


@st.cache_data
def load_genres() -> pd.DataFrame:
    return pd.read_csv(PIPELINE_DATA_DIR / 'genre_data.csv')


@st.cache_data
def load_omdb() -> dict:
    """Load OMDB data from CSV, return dict keyed by imdb_id."""
    path = EXTRA_DATA_DIR / 'omdb_movies.csv'
    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna('')
        return {row['imdb_id']: row.to_dict() for _, row in df.iterrows()}
    return {}


def _read_parquet_array(path: Path) -> np.ndarray:
    """Read a Parquet file and return the values as a NumPy array (dropping the index)."""
    df = pd.read_parquet(path, engine='pyarrow')
    return df.values


def _read_json(path: Path):
    """Read a JSON file, returning None if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def _load_artifacts_dir_cached(dir_str: str, _mtime_ns: int) -> dict:
    """Load pipeline artifacts from Parquet + JSON directory.

    Reconstructs the same dict interface the dashboard expects,
    so all downstream code works unchanged.
    """
    d = Path(dir_str)

    # ── Parquet: feature matrices ────────────────────────────────────────
    X_weighted_df = pd.read_parquet(d / 'X_weighted.parquet', engine='pyarrow')
    X_raw_df = pd.read_parquet(d / 'X_raw.parquet', engine='pyarrow')

    # ── Parquet: embeddings ──────────────────────────────────────────────
    embedding_cluster = _read_parquet_array(d / 'embedding_cluster.parquet')
    embedding_3d = _read_parquet_array(d / 'embedding_3d.parquet')
    embedding_2d = _read_parquet_array(d / 'embedding_2d.parquet')

    # ── Parquet: cluster labels + scores ─────────────────────────────────
    labels_df = pd.read_parquet(d / 'cluster_labels.parquet', engine='pyarrow')

    # ── Parquet: IDF weights ─────────────────────────────────────────────
    idf_df = pd.read_parquet(d / 'idf_weights.parquet', engine='pyarrow')

    # ── Parquet: Jordan metadata ─────────────────────────────────────────
    jordan_df = pd.read_parquet(d / 'jordan_metadata.parquet', engine='pyarrow')

    # ── JSON: metadata + nested objects ──────────────────────────────────
    feature_cols = _read_json(d / 'feature_columns.json') or {}
    cluster_names_raw = _read_json(d / 'cluster_names.json') or {}
    shap_raw = _read_json(d / 'cluster_shap_features.json') or {}
    recovery_raw = _read_json(d / 'recovery_suggestions.json') or {}
    title_lookup = _read_json(d / 'title_lookup.json') or {}
    validation = _read_json(d / 'validation_metrics.json') or {}
    config = _read_json(d / 'config.json') or {}
    sub_clusters_raw = _read_json(d / 'sub_clusters.json') or {}
    sub_sil_raw = _read_json(d / 'sub_cluster_silhouettes.json') or {}
    sub_names_raw = _read_json(d / 'sub_cluster_names.json') or {}
    sub_shap_raw = _read_json(d / 'sub_cluster_shap_features.json') or {}

    # ── Convert JSON string keys back to ints where needed ───────────────
    cluster_names = {int(k): v for k, v in cluster_names_raw.items()}
    cluster_shap_features = {int(k): v for k, v in shap_raw.items()}
    recovery_suggestions = {int(k): v for k, v in recovery_raw.items()}
    sub_clusters = {int(k): v for k, v in sub_clusters_raw.items()}
    sub_cluster_silhouettes = {int(k): v for k, v in sub_sil_raw.items()}

    # Sub-cluster names/SHAP: keys are stringified tuples like "(0, 1)"
    import ast
    def _parse_tuple_key(k):
        try:
            t = ast.literal_eval(k)
            return (int(t[0]), int(t[1]))
        except Exception:
            return k
    sub_cluster_names = {_parse_tuple_key(k): v for k, v in sub_names_raw.items()}
    sub_cluster_shap_features = {_parse_tuple_key(k): v for k, v in sub_shap_raw.items()}

    # ── Reconstruct the dict interface ───────────────────────────────────
    return {
        'embedding_cluster': embedding_cluster,
        'embedding_3d': embedding_3d,
        'embedding_2d': embedding_2d,
        'cluster_labels': labels_df['cluster_label'].values,
        'probabilities': labels_df['probability'].values,
        'outlier_scores': labels_df['outlier_score'].values,
        'n_clusters': int(validation.get('n_clusters', len(cluster_names))),
        'X': X_raw_df.values,
        'X_weighted': X_weighted_df.values,
        'feature_names': feature_cols.get('feature_names', list(X_weighted_df.columns)),
        'tt_codes': feature_cols.get('tt_codes', list(X_weighted_df.index)),
        'idf_weights': idf_df['idf_weight'].values,
        'genome_cols': feature_cols.get('genome_cols', []),
        'genre_cols': feature_cols.get('genre_cols', []),
        'decade_cols': feature_cols.get('decade_cols', []),
        'cluster_names': cluster_names,
        'cluster_shap_features': cluster_shap_features,
        'recovery_suggestions': recovery_suggestions,
        'xgb_cv_accuracy': validation.get('xgb_cv_accuracy', 0.0),
        'xgb_cv_std': validation.get('xgb_cv_std', 0.0),
        'title_lookup': title_lookup,
        'jordan_df': jordan_df,
        'config': config,
        'sub_clusters': sub_clusters,
        'sub_cluster_silhouettes': sub_cluster_silhouettes,
        'sub_cluster_names': sub_cluster_names,
        'sub_cluster_shap_features': sub_cluster_shap_features,
    }


@st.cache_data
def load_tmdb_popularity() -> dict:
    """Load TMDB popularity scores from the raw cache. Returns {tt_code: popularity_float}."""
    path = EXTRA_DATA_DIR / 'tmdb_raw_cache.json'
    if path.exists():
        with open(path) as f:
            cache = json.load(f)
        return {
            k: float(v.get('popularity', 0.0))
            for k, v in cache.items()
            if v is not None and 'popularity' in v
        }
    return {}


def _secret(name: str) -> str:
    val = os.getenv(name, '')
    if val:
        return val
    try:
        return str(st.secrets.get(name, '') or '')
    except Exception:
        return ''


def _kick_off_auto_naming():
    """Start the one-time background rail-naming run (no-op after the first call)."""
    try:
        from utils import auto_naming
        auto_naming.ensure_rail_names(_secret)
    except Exception as e:  # naming is a nicety; never block the dashboard
        print(f"auto naming not started: {e}")


def load_pipeline_artifacts() -> dict:
    """Load pipeline artifacts from the Parquet + JSON artifacts directory.

    Falls back to legacy .pkl file if the artifacts directory doesn't exist.
    """
    artifacts_dir = RESULTS_DIR / 'artifacts'
    if artifacts_dir.is_dir() and any(artifacts_dir.glob('*.parquet')):
        _kick_off_auto_naming()
        # Use newest file mtime to invalidate cache (names update as they are generated)
        newest = max(f.stat().st_mtime_ns for f in artifacts_dir.iterdir()
                     if f.is_file() and f.name != 'naming_status.json')
        return _load_artifacts_dir_cached(str(artifacts_dir), newest)

    # Legacy fallback: try .pkl file
    pkl_path = RESULTS_DIR / 'pipeline_artifacts.pkl'
    if pkl_path.exists():
        import joblib
        return joblib.load(pkl_path)

    return None


@st.cache_data
def build_feature_matrix(include_genres: bool = True) -> tuple:
    """Pivot longform data into (movie x feature) matrix. Returns (DataFrame, tt_codes, feature_names)."""
    fl = load_feature_longform()
    tax = load_taxonomy()
    fl = fl.merge(tax[['feature_id', 'feature']], on='feature_id', how='left')
    matrix = fl.pivot_table(
        index='imdb_id', columns='feature', values='trigger',
        fill_value=0, aggfunc='first'
    )

    if include_genres:
        genres = load_genres()
        genre_expanded = (
            genres.assign(genre_list=genres['genre'].str.split(', '))
            .explode('genre_list')
            .reset_index(drop=True)
        )
        genre_onehot = pd.crosstab(genre_expanded['movie_id'], genre_expanded['genre_list'])
        genre_onehot.columns = ['genre_' + col for col in genre_onehot.columns]
        genre_onehot.index.name = 'imdb_id'

        # Merge genre features into main matrix
        matrix = matrix.join(genre_onehot, how='left').fillna(0)

    return matrix, list(matrix.index), list(matrix.columns)


import os as _os
OMDB_API_KEY = _os.getenv('OMDB_API_KEY', '')  # set in SECRETS/env; free key at omdbapi.com

# ── IMDB scraped poster URL mapping (populated by imdb_poster_scraper.py) ────
_IMDB_POSTER_URLS: dict | None = None


def _load_imdb_poster_urls() -> dict:
    """Lazy-load the IMDB poster URL mapping JSON."""
    global _IMDB_POSTER_URLS
    if _IMDB_POSTER_URLS is None:
        imdb_json = BASE_DIR / 'imdb_posters' / 'imdb_poster_urls.json'
        if imdb_json.exists():
            with open(imdb_json) as f:
                _IMDB_POSTER_URLS = json.load(f)
        else:
            _IMDB_POSTER_URLS = {}
    return _IMDB_POSTER_URLS


def get_poster_src(tt_code: str, omdb_entry: dict | None = None, size: str = '140x210') -> str:
    """Return the best available poster source for a film.

    Priority: IMDB scraped URL > OMDB JSON Poster URL (Amazon CDN) > OMDB poster API > placeholder.
    """
    # 1. Check IMDB scraped poster URLs (highest quality, scraped from IMDB pages)
    imdb_urls = _load_imdb_poster_urls()
    if tt_code in imdb_urls:
        return imdb_urls[tt_code]

    # 2. Use the Amazon CDN URL stored in the OMDB JSON
    if omdb_entry:
        url = omdb_entry.get('Poster', '')
        if url and url != 'N/A':
            return url

    # 3. Local poster shipped in the repo (posters/<tt>.jpg, served via static/posters symlink)
    if (LOCAL_POSTER_DIR / f'{tt_code}.jpg').exists():
        return f'app/static/posters/{tt_code}.jpg'

    # 4. Fall back to OMDB poster API using the tt_code directly
    if OMDB_API_KEY and tt_code.startswith('tt'):
        return f'http://img.omdbapi.com/?apikey={OMDB_API_KEY}&i={tt_code}'

    # 5. Placeholder
    w, h = size.split('x')
    return f'https://via.placeholder.com/{w}x{h}/1e232d/c9d1d9?text=No+Poster'


def has_poster(tt_code: str, omdb_entry: dict | None = None) -> bool:
    """True when a poster is known to exist (IMDb/TMDB URL map, OMDb URL, or local file)."""
    if tt_code in _load_imdb_poster_urls():
        return True
    if omdb_entry:
        url = omdb_entry.get('Poster', '')
        if url and url != 'N/A':
            return True
    return (LOCAL_POSTER_DIR / f'{tt_code}.jpg').exists()


def get_title(tt_code: str) -> str:
    """Get movie title from OMDB or genre_data fallback."""
    omdb = load_omdb()
    if tt_code in omdb:
        t = omdb[tt_code].get('Title')
        if t:
            return t
    genres = load_genres()
    row = genres[genres['movie_id'] == tt_code]
    if len(row) > 0:
        return row.iloc[0]['movie_name']
    return tt_code


@st.cache_data
def build_title_lookup() -> dict:
    """Return dict: tt_code -> title for all movies."""
    omdb = load_omdb()
    genres = load_genres()
    genre_map = dict(zip(genres['movie_id'], genres['movie_name']))
    lookup = {}
    try:
        clusters = load_clusters_final()
        all_tts = clusters['tt_code'].tolist()
    except Exception:
        all_tts = list(genre_map.keys())
    for tt in all_tts:
        if tt in omdb and omdb[tt].get('Title'):
            lookup[tt] = omdb[tt]['Title']
        elif tt in genre_map:
            lookup[tt] = genre_map[tt]
        else:
            lookup[tt] = tt
    return lookup


@st.cache_data
def get_feature_matrix_from_artifacts():
    """Get feature matrix preferring pkl artifacts over CSV pivot."""
    arts = load_pipeline_artifacts()
    if arts is not None:
        continuous_cols = arts.get('continuous_cols', arts.get('genome_cols', []))
        binary_cols = arts.get('binary_cols',
                               arts.get('genre_cols', []) + arts.get('decade_cols', []))
        # Prefer X_weighted (IDF output) → X (raw)
        X_mat = arts.get('X_weighted', arts['X'])
        return (
            X_mat,
            arts['feature_names'],
            arts['tt_codes'],
            continuous_cols,
            binary_cols,
        )
    matrix, tt_codes, feature_names = build_feature_matrix()
    return matrix.values, feature_names, tt_codes, [], []
