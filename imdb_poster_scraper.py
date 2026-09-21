"""
Movie Poster Fetcher (TMDB API)
───────────────────────────────
Fetches high-quality movie poster URLs from The Movie Database (TMDB) API
using IMDB tt codes. TMDB is free, rate-limit friendly, and supports
direct IMDB ID lookups — no scraping needed.

Outputs imdb_posters/imdb_poster_urls.json which the Dashboard picks up
automatically via get_poster_src().

Setup:
    1. Get a free TMDB API key at: https://www.themoviedb.org/settings/api
    2. Add it to your SECRETS file:   TMDB_API_KEY=your_key_here
    3. Run:  python imdb_poster_scraper.py

Usage:
    python imdb_poster_scraper.py                 # Fetch all missing poster URLs
    python imdb_poster_scraper.py --force          # Re-fetch all poster URLs
    python imdb_poster_scraper.py --limit 100      # Fetch first 100 missing
    python imdb_poster_scraper.py --download       # Also download poster images
"""

import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
OMDB_JSON = PROJECT_DIR / 'omdb_data' / 'omdb_movies.json'
CLUSTERS_CSV = PROJECT_DIR / 'movie_clusters_final.csv'
OUTPUT_DIR = PROJECT_DIR / 'imdb_posters'
URL_MAPPING_FILE = OUTPUT_DIR / 'imdb_poster_urls.json'
LOG_FILE = PROJECT_DIR / 'imdb_poster_scraper.log'
SECRETS_FILE = PROJECT_DIR / 'SECRETS'

# TMDB API config
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p'
POSTER_SIZE = 'w500'          # w92, w154, w185, w342, w500, w780, original

# Rate limiting (TMDB allows ~40 req/sec, we'll be conservative)
REQUEST_DELAY = 0.05           # 50ms between requests = ~20 req/sec
MAX_WORKERS = 8                # concurrent threads
TIMEOUT = 10                   # request timeout in seconds
MAX_RETRIES = 2                # retry failed requests
BATCH_SAVE_INTERVAL = 100      # save progress every N movies

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Secrets loader ───────────────────────────────────────────────────────────
def load_tmdb_api_key() -> str | None:
    """Load TMDB API key from SECRETS file or environment."""
    # 1. Check environment variable
    key = os.environ.get('TMDB_API_KEY')
    if key:
        return key.strip()

    # 2. Check SECRETS file
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == 'TMDB_API_KEY':
                    val = v.strip()
                    if val:
                        return val

    return None


# ── TMDB API functions ───────────────────────────────────────────────────────
def find_tmdb_movie(tt_code: str, api_key: str, session: requests.Session) -> dict | None:
    """Look up a movie on TMDB using its IMDB tt code.

    Uses TMDB's /find endpoint which directly supports IMDB IDs.
    Returns the movie dict with poster_path, or None if not found.
    """
    url = f'{TMDB_BASE_URL}/find/{tt_code}'
    params = {
        'api_key': api_key,
        'external_source': 'imdb_id',
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT)

            # Handle rate limiting (429)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get('Retry-After', 2))
                logger.warning(f'{tt_code}: Rate limited, waiting {retry_after}s...')
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()

            # TMDB /find returns results grouped by type
            movie_results = data.get('movie_results', [])
            if movie_results:
                return movie_results[0]

            # Sometimes films are listed as TV movies
            tv_results = data.get('tv_results', [])
            if tv_results:
                return tv_results[0]

            return None

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(1 * (attempt + 1))
                continue
            logger.error(f'{tt_code}: TMDB lookup failed: {e}')
            return None

    return None


def get_poster_url(tmdb_movie: dict, size: str = POSTER_SIZE) -> str | None:
    """Build the full poster URL from a TMDB movie result."""
    poster_path = tmdb_movie.get('poster_path')
    if poster_path:
        return f'{TMDB_IMAGE_BASE}/{size}{poster_path}'
    return None


def fetch_poster_url(tt_code: str, api_key: str, session: requests.Session) -> tuple[str, str | None]:
    """Fetch the poster URL for a single tt_code. Returns (tt_code, url_or_None)."""
    movie = find_tmdb_movie(tt_code, api_key, session)
    if movie:
        poster_url = get_poster_url(movie)
        if poster_url:
            return tt_code, poster_url

    return tt_code, None


def download_poster(tt_code: str, img_url: str, session: requests.Session) -> bool:
    """Download a poster image and save it as a JPEG."""
    output_path = OUTPUT_DIR / f'{tt_code}.jpg'

    try:
        resp = session.get(img_url, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size = output_path.stat().st_size
        if size < 1000:
            output_path.unlink()
            logger.warning(f'{tt_code}: Downloaded file too small ({size} bytes), removed')
            return False

        return True

    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        logger.error(f'{tt_code}: Image download failed: {e}')
        return False


# ── Persistence ──────────────────────────────────────────────────────────────
def load_existing_urls() -> dict:
    """Load previously saved URL mapping."""
    if URL_MAPPING_FILE.exists():
        with open(URL_MAPPING_FILE) as f:
            return json.load(f)
    return {}


def save_url_mapping(url_map: dict):
    """Save the tt_code -> poster_url mapping as JSON for the dashboard."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(URL_MAPPING_FILE, 'w') as f:
        json.dump(url_map, f, indent=2)
    logger.info(f'Saved URL mapping ({len(url_map)} entries) to {URL_MAPPING_FILE}')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Fetch movie poster URLs from TMDB using IMDB tt codes'
    )
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch all poster URLs, even existing ones')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max number of posters to fetch (0 = all)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Concurrent threads (default: {MAX_WORKERS})')
    parser.add_argument('--download', action='store_true',
                        help='Also download poster images to imdb_posters/')
    parser.add_argument('--size', default=POSTER_SIZE,
                        choices=['w92', 'w154', 'w185', 'w342', 'w500', 'w780', 'original'],
                        help=f'Poster image size (default: {POSTER_SIZE})')
    args = parser.parse_args()

    # ── Load API key ──
    api_key = load_tmdb_api_key()
    if not api_key:
        logger.error('TMDB API key not found!')
        logger.error('Get a free key at: https://www.themoviedb.org/settings/api')
        logger.error('Then add to your SECRETS file:  TMDB_API_KEY=your_key_here')
        logger.error('Or set environment variable:    export TMDB_API_KEY=your_key_here')
        sys.exit(1)

    # ── Validate API key ──
    logger.info('Validating TMDB API key...')
    try:
        test_resp = requests.get(
            f'{TMDB_BASE_URL}/configuration',
            params={'api_key': api_key},
            timeout=TIMEOUT,
        )
        if test_resp.status_code == 401:
            logger.error('Invalid TMDB API key! Check your key and try again.')
            sys.exit(1)
        test_resp.raise_for_status()
        logger.info('API key valid ✓')
    except requests.exceptions.RequestException as e:
        logger.error(f'Could not validate API key: {e}')
        sys.exit(1)

    # ── Load tt codes from ALL sources ──
    all_tt_set = set()

    # Source 1: OMDB JSON
    if OMDB_JSON.exists():
        with open(OMDB_JSON) as f:
            omdb_data = json.load(f)
        all_tt_set.update(omdb_data.keys())
        logger.info(f'OMDB JSON: {len(omdb_data)} movies')
    else:
        logger.warning(f'OMDB JSON not found: {OMDB_JSON}')

    # Source 2: movie_clusters_final.csv (catches any movies not in OMDB)
    if CLUSTERS_CSV.exists():
        clusters_df = pd.read_csv(CLUSTERS_CSV, usecols=['tt_code'], dtype={'tt_code': str})
        csv_tts = set(clusters_df['tt_code'].dropna().tolist())
        new_from_csv = csv_tts - all_tt_set
        if new_from_csv:
            logger.info(f'Clusters CSV: +{len(new_from_csv)} movies not in OMDB JSON')
        all_tt_set.update(csv_tts)
    else:
        logger.warning(f'Clusters CSV not found: {CLUSTERS_CSV}')

    if not all_tt_set:
        logger.error('No tt codes found in any source!')
        sys.exit(1)

    all_tt_codes = sorted(all_tt_set)
    logger.info(f'Total unique movies: {len(all_tt_codes)}')

    # ── Determine which need fetching ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_urls = load_existing_urls()

    if args.force:
        tt_codes_to_fetch = all_tt_codes
        logger.info('Force mode: re-fetching all poster URLs')
    else:
        tt_codes_to_fetch = [tt for tt in all_tt_codes if tt not in existing_urls]
        logger.info(f'Already have {len(existing_urls)} URLs, {len(tt_codes_to_fetch)} remaining')

    if args.limit > 0:
        tt_codes_to_fetch = tt_codes_to_fetch[:args.limit]

    if not tt_codes_to_fetch:
        logger.info('Nothing to fetch — all poster URLs already collected!')
        if args.download:
            logger.info('Checking for missing image downloads...')
            _download_missing(existing_urls, args.workers)
        return

    logger.info(f'Fetching {len(tt_codes_to_fetch)} poster URLs with {args.workers} workers...')

    # ── Fetch poster URLs ──
    success_count = 0
    fail_count = 0
    new_urls = {}

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_poster_url, tt, api_key, session): tt
                for tt in tt_codes_to_fetch
            }

            for i, future in enumerate(as_completed(futures), 1):
                tt, poster_url = future.result()

                if poster_url:
                    # If user requested a different size, adjust the URL
                    if args.size != POSTER_SIZE:
                        poster_url = poster_url.replace(f'/{POSTER_SIZE}/', f'/{args.size}/')
                    new_urls[tt] = poster_url
                    success_count += 1
                else:
                    fail_count += 1

                # Progress logging
                if i % 50 == 0 or i == len(tt_codes_to_fetch):
                    pct = i / len(tt_codes_to_fetch) * 100
                    logger.info(
                        f'Progress: {i}/{len(tt_codes_to_fetch)} ({pct:.0f}%) '
                        f'— ✓ {success_count}  ✗ {fail_count}'
                    )

                # Batch save progress
                if i % BATCH_SAVE_INTERVAL == 0:
                    merged = {**existing_urls, **new_urls}
                    save_url_mapping(merged)

                time.sleep(REQUEST_DELAY)

    # ── Save final results ──
    merged = {**existing_urls, **new_urls}
    save_url_mapping(merged)

    logger.info(
        f'\nDone! Fetched {success_count} poster URLs, '
        f'{fail_count} not found, out of {len(tt_codes_to_fetch)} attempted.'
    )
    logger.info(f'Total URLs in mapping: {len(merged)}')

    # ── Optional: download images ──
    if args.download:
        logger.info('\nDownloading poster images...')
        _download_missing(merged, args.workers)


def _download_missing(url_map: dict, workers: int):
    """Download any poster images we have URLs for but haven't downloaded yet."""
    existing_images = {f.stem for f in OUTPUT_DIR.iterdir()
                       if f.suffix == '.jpg' and f.stat().st_size > 1000}
    to_download = {tt: url for tt, url in url_map.items() if tt not in existing_images}

    if not to_download:
        logger.info('All poster images already downloaded!')
        return

    logger.info(f'Downloading {len(to_download)} poster images...')
    dl_success = 0
    dl_fail = 0

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_poster, tt, url, session): tt
                for tt, url in to_download.items()
            }

            for i, future in enumerate(as_completed(futures), 1):
                if future.result():
                    dl_success += 1
                else:
                    dl_fail += 1

                if i % 100 == 0 or i == len(to_download):
                    logger.info(f'Download progress: {i}/{len(to_download)} '
                                f'(✓ {dl_success} | ✗ {dl_fail})')

    logger.info(f'Downloaded {dl_success} images, {dl_fail} failed.')


if __name__ == '__main__':
    main()
