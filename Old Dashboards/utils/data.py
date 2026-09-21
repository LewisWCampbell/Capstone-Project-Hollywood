"""Data loading and caching layer for the Movie Genome Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'


def _mtime_ns(path: Path) -> int:
    """Return file mtime in ns (or -1 when file is missing)."""
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


def check_data_status() -> dict:
    """Quick check of what data files are available."""
    csvs = [
        'movie_clusters_final.csv', 'cluster_summary.csv',
        'sub_cluster_summary.csv', 'feature_data_longform.csv',
        'feature_taxonomy.csv', 'genre_data.csv',
    ]
    return {
        'csv_count': sum(1 for c in csvs if (BASE_DIR / c).exists()),
        'total_csvs': len(csvs),
        'pkl_exists': (RESULTS_DIR / 'pipeline_artifacts.pkl').exists(),
        'omdb_exists': (BASE_DIR / 'omdb_data' / 'omdb_movies.json').exists(),
    }


@st.cache_data
def _load_csv_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    # mtime_ns is included to invalidate cache when file contents change.
    return pd.read_csv(path_str)


def load_clusters_final() -> pd.DataFrame:
    path = BASE_DIR / 'movie_clusters_final.csv'
    return _load_csv_cached(str(path), _mtime_ns(path))


def load_cluster_summary() -> pd.DataFrame:
    path = BASE_DIR / 'cluster_summary.csv'
    return _load_csv_cached(str(path), _mtime_ns(path))


def load_sub_cluster_summary() -> pd.DataFrame:
    path = BASE_DIR / 'sub_cluster_summary.csv'
    return _load_csv_cached(str(path), _mtime_ns(path))


@st.cache_data
def load_feature_longform() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / 'feature_data_longform.csv')


@st.cache_data
def load_taxonomy() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / 'feature_taxonomy.csv')


@st.cache_data
def load_taxonomy_categorized() -> pd.DataFrame:
    path = BASE_DIR / 'feature_taxonomy_categorized.csv'
    if path.exists():
        return pd.read_csv(path, encoding='utf-8-sig')
    return pd.DataFrame()


@st.cache_data
def load_genres() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / 'genre_data.csv')


@st.cache_data
def load_omdb() -> dict:
    path = BASE_DIR / 'omdb_data' / 'omdb_movies.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def _load_pipeline_artifacts_cached(path_str: str, mtime_ns: int):
    # mtime_ns is included to invalidate cache when artifact file is replaced.
    path = Path(path_str)
    if path.exists():
        return joblib.load(path)
    return None


def load_pipeline_artifacts() -> dict:
    path = RESULTS_DIR / 'pipeline_artifacts.pkl'
    return _load_pipeline_artifacts_cached(str(path), _mtime_ns(path))


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


POSTER_DIR = BASE_DIR / 'omdb_data' / 'posters'


def get_poster_src(tt_code: str, omdb_entry: dict | None = None) -> str:
    """Return the best poster source for a movie: local file first, then remote URL.
    Returns a placeholder URL if no poster is available."""
    # Check local file first
    local_path = POSTER_DIR / f'{tt_code}.jpg'
    if local_path.exists():
        return str(local_path)
    # Fall back to remote URL from OMDB data
    if omdb_entry is None:
        omdb_entry = load_omdb().get(tt_code, {})
    poster_url = omdb_entry.get('Poster', '')
    if poster_url and poster_url != 'N/A':
        return poster_url
    return 'https://via.placeholder.com/140x210/1e232d/c9d1d9?text=No+Poster'


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
        return (
            arts['X'],
            arts['feature_names'],
            arts['tt_codes'],
            arts.get('continuous_cols', []),
            arts.get('binary_cols', []),
        )
    matrix, tt_codes, feature_names = build_feature_matrix()
    return matrix.values, feature_names, tt_codes, [], []
