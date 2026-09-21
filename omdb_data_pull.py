#!/usr/bin/env python3
"""
OMDB Data Pull Script for Project HOLLYWOOD
Pulls movie metadata, ratings, plot, cast/crew from OMDB API
Designed for daily scheduled runs with incremental updates
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from pathlib import Path
import logging

# Import secrets loader
try:
    from secrets_loader import get_secret
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False
    print("⚠️  secrets_loader not found, falling back to environment variables")

# ============================================
# CONFIGURATION
# ============================================

# OMDB API Configuration
if SECRETS_AVAILABLE:
    OMDB_API_KEY = get_secret('OMDB_API_KEY')
else:
    OMDB_API_KEY = os.getenv('OMDB_API_KEY')

OMDB_BASE_URL = 'http://www.omdbapi.com/'

# File paths
SCRIPT_DIR = Path(__file__).parent
INPUT_GENRE_FILE = SCRIPT_DIR / 'genre_data.csv'  # Your existing movie data
OUTPUT_DIR = SCRIPT_DIR / 'omdb_data'
OUTPUT_FILE = OUTPUT_DIR / 'omdb_enriched_data.csv'
CACHE_FILE = OUTPUT_DIR / 'omdb_cache.json'
LOG_FILE = OUTPUT_DIR / 'omdb_pull.log'

# Rate limiting (OMDB free tier: 1,000 requests/day)
REQUESTS_PER_DAY = 1000
DELAY_BETWEEN_REQUESTS = 1.0  # seconds
MAX_MOVIES_PER_RUN = 1000  # Use full daily limit for faster sync

# Poster download settings
DOWNLOAD_POSTERS = True  # Set to False to skip poster downloads
POSTER_DIR = OUTPUT_DIR / 'posters'
POSTER_DIR.mkdir(exist_ok=True)

# Incremental update settings
DAYS_BEFORE_REFRESH = 30  # Re-fetch movie data after 30 days

# ============================================
# SETUP
# ============================================

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# OMDB API FUNCTIONS
# ============================================

def fetch_omdb_data(title=None, imdb_id=None, year=None):
    """
    Fetch movie data from OMDB API.

    Args:
        title: Movie title
        imdb_id: IMDB ID (preferred - more accurate)
        year: Release year (helps with disambiguation)

    Returns:
        dict with movie data or None if error
    """
    if not OMDB_API_KEY:
        logger.error("OMDB API key not set!")
        logger.error("Add OMDB_API_KEY to your SECRETS file or set as environment variable")
        logger.error("Get a free key at http://www.omdbapi.com/apikey.aspx")
        return None

    params = {
        'apikey': OMDB_API_KEY,
        'plot': 'full',  # Get full plot
        'type': 'movie'
    }

    if imdb_id:
        params['i'] = imdb_id
    elif title:
        params['t'] = title
        if year:
            params['y'] = year
    else:
        logger.error("Must provide either title or imdb_id")
        return None

    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('Response') == 'True':
            return data
        else:
            logger.warning(f"OMDB error for {title or imdb_id}: {data.get('Error')}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {title or imdb_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {title or imdb_id}: {e}")
        return None

def parse_omdb_response(data):
    """
    Parse OMDB response into structured format.

    Returns:
        dict with cleaned/structured data
    """
    if not data:
        return None

    # Extract ratings
    ratings = {}
    for rating in data.get('Ratings', []):
        source = rating.get('Source', '').replace(' ', '_').lower()
        value = rating.get('Value', 'N/A')
        ratings[f'rating_{source}'] = value

    # Parse runtime to minutes
    runtime_str = data.get('Runtime', 'N/A')
    runtime_minutes = None
    if runtime_str != 'N/A' and 'min' in runtime_str:
        try:
            runtime_minutes = int(runtime_str.split()[0])
        except (ValueError, IndexError):
            pass

    return {
        # Basic metadata
        'imdb_id': data.get('imdbID'),
        'title': data.get('Title'),
        'year': data.get('Year'),
        'rated': data.get('Rated'),
        'released': data.get('Released'),
        'runtime': runtime_str,
        'runtime_minutes': runtime_minutes,
        'genre': data.get('Genre'),
        'director': data.get('Director'),
        'writer': data.get('Writer'),
        'actors': data.get('Actors'),
        'plot': data.get('Plot'),
        'language': data.get('Language'),
        'country': data.get('Country'),
        'awards': data.get('Awards'),
        'poster_url': data.get('Poster'),
        'metascore': data.get('Metascore'),
        'imdb_rating': data.get('imdbRating'),
        'imdb_votes': data.get('imdbVotes'),
        'box_office': data.get('BoxOffice'),
        'production': data.get('Production'),
        'website': data.get('Website'),

        # Individual ratings
        **ratings,

        # Metadata
        'omdb_fetch_date': datetime.now().isoformat(),
        'omdb_response': 'True'
    }

def download_poster(poster_url, movie_id, imdb_id=None):
    """
    Download movie poster image.

    Args:
        poster_url: URL to poster image
        movie_id: Internal movie ID
        imdb_id: IMDB ID (if available)

    Returns:
        str: Local path to downloaded poster or None
    """
    if not DOWNLOAD_POSTERS:
        return None

    if not poster_url or poster_url == 'N/A':
        return None

    try:
        # Use IMDB ID if available, otherwise movie_id
        filename = f"{imdb_id}.jpg" if imdb_id else f"movie_{movie_id}.jpg"
        poster_path = POSTER_DIR / filename

        # Skip if already downloaded
        if poster_path.exists():
            return str(poster_path)

        # Download poster
        response = requests.get(poster_url, timeout=10)
        response.raise_for_status()

        # Save to file
        with open(poster_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"  📸 Downloaded poster: {filename}")
        return str(poster_path)

    except Exception as e:
        logger.warning(f"  Failed to download poster: {e}")
        return None

# ============================================
# CACHE MANAGEMENT
# ============================================

def load_cache():
    """Load cached OMDB data."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Cache file corrupted, starting fresh")
            return {}
    return {}

def save_cache(cache):
    """Save OMDB data to cache."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def should_refresh(cache_entry):
    """Check if cached entry should be refreshed."""
    if not cache_entry:
        return True

    fetch_date = cache_entry.get('omdb_fetch_date')
    if not fetch_date:
        return True

    try:
        fetch_datetime = datetime.fromisoformat(fetch_date)
        days_old = (datetime.now() - fetch_datetime).days
        return days_old >= DAYS_BEFORE_REFRESH
    except (ValueError, TypeError):
        return True

# ============================================
# MAIN PULL LOGIC
# ============================================

def run_omdb_pull():
    """
    Main function to pull OMDB data for movies in dataset.
    """
    logger.info("=" * 80)
    logger.info("OMDB DATA PULL STARTED")
    logger.info("=" * 80)

    # Check API key
    if not OMDB_API_KEY:
        logger.error("⚠️  OMDB_API_KEY not set!")
        logger.error("Get a free API key at: http://www.omdbapi.com/apikey.aspx")
        logger.error("")
        logger.error("Then add it to your SECRETS file:")
        logger.error("  1. Copy SECRETS.template to SECRETS")
        logger.error("  2. Edit SECRETS and add: OMDB_API_KEY=your_key_here")
        logger.error("")
        logger.error("Or set as environment variable: export OMDB_API_KEY='your_key'")
        return

    # Load existing movie data
    if not INPUT_GENRE_FILE.exists():
        logger.error(f"Input file not found: {INPUT_GENRE_FILE}")
        logger.error("Please ensure genre_data.csv exists in the same directory")
        return

    logger.info(f"Loading movie data from {INPUT_GENRE_FILE}")
    try:
        genre_data = pd.read_csv(INPUT_GENRE_FILE)
    except Exception as e:
        logger.error(f"Error loading genre_data.csv: {e}")
        return

    logger.info(f"Loaded {len(genre_data)} movies from dataset")

    # Load cache
    cache = load_cache()
    logger.info(f"Loaded cache with {len(cache)} entries")

    # Determine which movies to fetch
    movies_to_fetch = []
    for idx, row in genre_data.iterrows():
        movie_id = row.get('movie_id')
        title = row.get('title', f'Movie_{movie_id}')
        imdb_id = row.get('imdb_id')  # If you have IMDB IDs in your data

        cache_key = imdb_id if imdb_id else str(movie_id)

        if should_refresh(cache.get(cache_key)):
            movies_to_fetch.append({
                'movie_id': movie_id,
                'title': title,
                'imdb_id': imdb_id,
                'cache_key': cache_key
            })

    logger.info(f"Movies needing refresh: {len(movies_to_fetch)}")

    # Limit to MAX_MOVIES_PER_RUN
    if len(movies_to_fetch) > MAX_MOVIES_PER_RUN:
        logger.info(f"Limiting to {MAX_MOVIES_PER_RUN} movies per run")
        movies_to_fetch = movies_to_fetch[:MAX_MOVIES_PER_RUN]

    # Fetch data
    successful_fetches = 0
    failed_fetches = 0

    for i, movie_info in enumerate(movies_to_fetch, 1):
        title = movie_info['title']
        imdb_id = movie_info['imdb_id']
        cache_key = movie_info['cache_key']

        logger.info(f"[{i}/{len(movies_to_fetch)}] Fetching: {title}")

        # Fetch from OMDB
        omdb_data = fetch_omdb_data(title=title, imdb_id=imdb_id)

        if omdb_data:
            # Parse and cache
            parsed_data = parse_omdb_response(omdb_data)
            if parsed_data:
                parsed_data['movie_id'] = movie_info['movie_id']

                # Download poster
                poster_url = parsed_data.get('poster_url')
                poster_path = download_poster(
                    poster_url,
                    movie_info['movie_id'],
                    imdb_id
                )
                if poster_path:
                    parsed_data['poster_local_path'] = poster_path

                cache[cache_key] = parsed_data
                successful_fetches += 1
                logger.info(f"  ✓ Success: {parsed_data.get('title')} ({parsed_data.get('year')})")
            else:
                failed_fetches += 1
                logger.warning(f"  ✗ Failed to parse response")
        else:
            failed_fetches += 1
            logger.warning(f"  ✗ Failed to fetch")

        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Save cache periodically (every 50 movies)
        if i % 50 == 0:
            save_cache(cache)
            logger.info(f"  💾 Cache saved (checkpoint)")

    # Final cache save
    save_cache(cache)
    logger.info("💾 Final cache saved")

    # Convert cache to DataFrame
    if cache:
        enriched_df = pd.DataFrame.from_dict(cache, orient='index')
        enriched_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"📊 Enriched data saved to {OUTPUT_FILE}")
        logger.info(f"   Total movies: {len(enriched_df)}")

    # Summary
    logger.info("=" * 80)
    logger.info("OMDB DATA PULL COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Successful fetches: {successful_fetches}")
    logger.info(f"Failed fetches: {failed_fetches}")
    logger.info(f"Total cached movies: {len(cache)}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 80)

# ============================================
# CLI ENTRY POINT
# ============================================

if __name__ == '__main__':
    run_omdb_pull()
