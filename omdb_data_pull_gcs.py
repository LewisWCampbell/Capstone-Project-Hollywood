#!/usr/bin/env python3
"""
OMDB Data Pull Script for Project HOLLYWOOD - GCS Version
Streams movie metadata directly to Google Cloud Storage
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from pathlib import Path
import logging
from io import StringIO, BytesIO

# Import secrets loader
try:
    from secrets_loader import get_secret
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False
    print("⚠️  secrets_loader not found, falling back to environment variables")

# Google Cloud Storage imports
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("⚠️  google-cloud-storage not installed")
    print("   Install with: pip install google-cloud-storage")

# ============================================
# CONFIGURATION
# ============================================

# OMDB API Configuration
if SECRETS_AVAILABLE:
    OMDB_API_KEY = get_secret('OMDB_API_KEY')
else:
    OMDB_API_KEY = os.getenv('OMDB_API_KEY')

OMDB_BASE_URL = 'http://www.omdbapi.com/'

# GCS Configuration
if SECRETS_AVAILABLE:
    GCS_BUCKET_NAME = get_secret('GCS_BUCKET_NAME')
    GCS_PROJECT_ID = get_secret('GCS_PROJECT_ID')
    GCS_CREDENTIALS_PATH = get_secret('GCS_CREDENTIALS_PATH')
else:
    GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
    GCS_PROJECT_ID = os.getenv('GCS_PROJECT_ID')
    GCS_CREDENTIALS_PATH = os.getenv('GCS_CREDENTIALS_PATH')

# GCS paths (within bucket)
GCS_OUTPUT_PATH = 'omdb_data/omdb_enriched_data.csv'
GCS_CACHE_PATH = 'omdb_data/omdb_cache.json'
GCS_POSTERS_PATH = 'omdb_data/posters/'  # Folder for posters in GCS
GCS_LOG_PATH = f'omdb_data/logs/omdb_pull_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Local temporary directory
SCRIPT_DIR = Path(__file__).parent
TEMP_DIR = SCRIPT_DIR / 'temp'
TEMP_DIR.mkdir(exist_ok=True)

# Rate limiting
REQUESTS_PER_DAY = 1000
DELAY_BETWEEN_REQUESTS = 1.0
MAX_MOVIES_PER_RUN = 1000  # Use full daily limit for faster sync
DAYS_BEFORE_REFRESH = 30

# Poster download settings
DOWNLOAD_POSTERS = True  # Download and upload posters to GCS

# ============================================
# SETUP LOGGING
# ============================================

# Setup dual logging (local + will upload to GCS)
LOG_FILE = TEMP_DIR / 'omdb_pull.log'

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
# GCS FUNCTIONS
# ============================================

def get_gcs_client():
    """
    Initialize GCS client with credentials.

    Returns:
        storage.Client or None
    """
    if not GCS_AVAILABLE:
        logger.error("google-cloud-storage not installed")
        return None

    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME not set in SECRETS")
        return None

    try:
        # Try service account credentials first
        if GCS_CREDENTIALS_PATH and Path(GCS_CREDENTIALS_PATH).exists():
            credentials = service_account.Credentials.from_service_account_file(
                GCS_CREDENTIALS_PATH
            )
            client = storage.Client(
                project=GCS_PROJECT_ID,
                credentials=credentials
            )
            logger.info(f"✓ GCS client initialized with service account")
        else:
            # Fall back to Application Default Credentials
            client = storage.Client(project=GCS_PROJECT_ID)
            logger.info(f"✓ GCS client initialized with default credentials")

        return client

    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        return None

def upload_to_gcs(client, local_path, gcs_path, content_type='text/csv'):
    """
    Upload a file to GCS.

    Args:
        client: GCS client
        local_path: Path to local file
        gcs_path: Path within GCS bucket
        content_type: MIME type

    Returns:
        bool: Success
    """
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        blob.upload_from_filename(str(local_path), content_type=content_type)
        logger.info(f"✓ Uploaded to gs://{GCS_BUCKET_NAME}/{gcs_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        return False

def upload_string_to_gcs(client, content, gcs_path, content_type='text/csv'):
    """
    Upload string content directly to GCS.

    Args:
        client: GCS client
        content: String content
        gcs_path: Path within GCS bucket
        content_type: MIME type

    Returns:
        bool: Success
    """
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        blob.upload_from_string(content, content_type=content_type)
        logger.info(f"✓ Uploaded to gs://{GCS_BUCKET_NAME}/{gcs_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        return False

def download_from_gcs(client, gcs_path, local_path=None):
    """
    Download a file from GCS.

    Args:
        client: GCS client
        gcs_path: Path within GCS bucket
        local_path: Optional local path to save to

    Returns:
        Content as string or None
    """
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            logger.warning(f"File not found in GCS: {gcs_path}")
            return None

        if local_path:
            blob.download_to_filename(str(local_path))
            logger.info(f"✓ Downloaded from gs://{GCS_BUCKET_NAME}/{gcs_path}")
            with open(local_path, 'r') as f:
                return f.read()
        else:
            content = blob.download_as_text()
            return content

    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")
        return None

def load_cache_from_gcs(client):
    """Load cache from GCS."""
    content = download_from_gcs(client, GCS_CACHE_PATH)

    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Cache file corrupted, starting fresh")
            return {}

    return {}

def save_cache_to_gcs(client, cache):
    """Save cache to GCS."""
    content = json.dumps(cache, indent=2)
    return upload_string_to_gcs(client, content, GCS_CACHE_PATH, content_type='application/json')

def load_existing_data_from_gcs(client):
    """Load existing CSV from GCS."""
    content = download_from_gcs(client, GCS_OUTPUT_PATH)

    if content:
        try:
            # Read CSV from string
            df = pd.read_csv(StringIO(content))
            logger.info(f"✓ Loaded {len(df)} existing records from GCS")
            return df
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
            return pd.DataFrame()

    return pd.DataFrame()

def save_data_to_gcs(client, df):
    """Save DataFrame to GCS as CSV."""
    # Convert to CSV string
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    return upload_string_to_gcs(client, csv_content, GCS_OUTPUT_PATH, content_type='text/csv')

# ============================================
# OMDB API FUNCTIONS (same as before)
# ============================================

def fetch_omdb_data(title=None, imdb_id=None, year=None):
    """Fetch movie data from OMDB API."""
    if not OMDB_API_KEY:
        logger.error("OMDB API key not set!")
        return None

    params = {
        'apikey': OMDB_API_KEY,
        'plot': 'full',
        'type': 'movie'
    }

    if imdb_id:
        params['i'] = imdb_id
    elif title:
        params['t'] = title
        if year:
            params['y'] = year
    else:
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

    except Exception as e:
        logger.error(f"Request error: {e}")
        return None

def parse_omdb_response(data):
    """Parse OMDB response into structured format."""
    if not data:
        return None

    ratings = {}
    for rating in data.get('Ratings', []):
        source = rating.get('Source', '').replace(' ', '_').lower()
        value = rating.get('Value', 'N/A')
        ratings[f'rating_{source}'] = value

    runtime_str = data.get('Runtime', 'N/A')
    runtime_minutes = None
    if runtime_str != 'N/A' and 'min' in runtime_str:
        try:
            runtime_minutes = int(runtime_str.split()[0])
        except (ValueError, IndexError):
            pass

    return {
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
        **ratings,
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
        bytes: Poster image data or None
    """
    if not DOWNLOAD_POSTERS:
        return None

    if not poster_url or poster_url == 'N/A':
        return None

    try:
        response = requests.get(poster_url, timeout=10)
        response.raise_for_status()
        logger.info(f"  📸 Downloaded poster")
        return response.content
    except Exception as e:
        logger.warning(f"  Failed to download poster: {e}")
        return None

def upload_poster_to_gcs(client, poster_data, movie_id, imdb_id=None):
    """
    Upload poster to GCS.

    Args:
        client: GCS client
        poster_data: Image bytes
        movie_id: Internal movie ID
        imdb_id: IMDB ID (if available)

    Returns:
        str: GCS path or None
    """
    if not poster_data:
        return None

    try:
        filename = f"{imdb_id}.jpg" if imdb_id else f"movie_{movie_id}.jpg"
        gcs_path = f"{GCS_POSTERS_PATH}{filename}"

        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        # Check if already exists
        if blob.exists():
            return f"gs://{GCS_BUCKET_NAME}/{gcs_path}"

        # Upload
        blob.upload_from_string(poster_data, content_type='image/jpeg')
        logger.info(f"  ☁️  Uploaded poster to GCS")
        return f"gs://{GCS_BUCKET_NAME}/{gcs_path}"

    except Exception as e:
        logger.warning(f"  Failed to upload poster to GCS: {e}")
        return None

def should_refresh(cache_entry):
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

def run_omdb_pull_gcs():
    """
    Main function to pull OMDB data and stream to GCS.
    """
    logger.info("=" * 80)
    logger.info("OMDB DATA PULL STARTED (GCS MODE)")
    logger.info("=" * 80)

    # Check API key
    if not OMDB_API_KEY:
        logger.error("⚠️  OMDB_API_KEY not set in SECRETS file")
        return

    # Initialize GCS client
    gcs_client = get_gcs_client()
    if not gcs_client:
        logger.error("⚠️  Failed to initialize GCS client")
        logger.error("Check GCS configuration in SECRETS file:")
        logger.error("  - GCS_BUCKET_NAME")
        logger.error("  - GCS_PROJECT_ID")
        logger.error("  - GCS_CREDENTIALS_PATH")
        return

    logger.info(f"✓ Connected to GCS bucket: {GCS_BUCKET_NAME}")

    # Load cache from GCS
    cache = load_cache_from_gcs(gcs_client)
    logger.info(f"Loaded cache with {len(cache)} entries from GCS")

    # For this version, we'll need a local genre_data.csv to know which movies to fetch
    # You could also store this in GCS
    genre_file = SCRIPT_DIR / 'genre_data.csv'
    if not genre_file.exists():
        logger.error(f"Input file not found: {genre_file}")
        logger.error("Upload genre_data.csv or modify script to read from GCS")
        return

    genre_data = pd.read_csv(genre_file)
    logger.info(f"Loaded {len(genre_data)} movies from dataset")

    # Determine which movies to fetch
    movies_to_fetch = []
    for idx, row in genre_data.iterrows():
        movie_id = row.get('movie_id')
        title = row.get('title', f'Movie_{movie_id}')
        imdb_id = row.get('imdb_id')
        cache_key = imdb_id if imdb_id else str(movie_id)

        if should_refresh(cache.get(cache_key)):
            movies_to_fetch.append({
                'movie_id': movie_id,
                'title': title,
                'imdb_id': imdb_id,
                'cache_key': cache_key
            })

    logger.info(f"Movies needing refresh: {len(movies_to_fetch)}")

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

        omdb_data = fetch_omdb_data(title=title, imdb_id=imdb_id)

        if omdb_data:
            parsed_data = parse_omdb_response(omdb_data)
            if parsed_data:
                parsed_data['movie_id'] = movie_info['movie_id']

                # Download and upload poster
                poster_url = parsed_data.get('poster_url')
                poster_data = download_poster(
                    poster_url,
                    movie_info['movie_id'],
                    imdb_id
                )
                if poster_data:
                    poster_gcs_path = upload_poster_to_gcs(
                        gcs_client,
                        poster_data,
                        movie_info['movie_id'],
                        imdb_id
                    )
                    if poster_gcs_path:
                        parsed_data['poster_gcs_path'] = poster_gcs_path

                cache[cache_key] = parsed_data
                successful_fetches += 1
                logger.info(f"  ✓ Success: {parsed_data.get('title')}")
            else:
                failed_fetches += 1
        else:
            failed_fetches += 1

        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Save cache to GCS periodically
        if i % 50 == 0:
            save_cache_to_gcs(gcs_client, cache)
            logger.info(f"  💾 Cache checkpoint saved to GCS")

    # Final save
    save_cache_to_gcs(gcs_client, cache)
    logger.info("💾 Final cache saved to GCS")

    # Convert cache to DataFrame and save to GCS
    if cache:
        enriched_df = pd.DataFrame.from_dict(cache, orient='index')
        save_data_to_gcs(gcs_client, enriched_df)
        logger.info(f"📊 Enriched data saved to GCS")
        logger.info(f"   Location: gs://{GCS_BUCKET_NAME}/{GCS_OUTPUT_PATH}")
        logger.info(f"   Total movies: {len(enriched_df)}")

    # Upload log file to GCS
    upload_to_gcs(gcs_client, LOG_FILE, GCS_LOG_PATH, content_type='text/plain')
    logger.info(f"📝 Log uploaded to gs://{GCS_BUCKET_NAME}/{GCS_LOG_PATH}")

    # Summary
    logger.info("=" * 80)
    logger.info("OMDB DATA PULL COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Successful fetches: {successful_fetches}")
    logger.info(f"Failed fetches: {failed_fetches}")
    logger.info(f"Total cached movies: {len(cache)}")
    logger.info(f"GCS bucket: {GCS_BUCKET_NAME}")
    logger.info(f"Data file: {GCS_OUTPUT_PATH}")
    logger.info("=" * 80)

if __name__ == '__main__':
    run_omdb_pull_gcs()
