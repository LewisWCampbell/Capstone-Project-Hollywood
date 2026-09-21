"""Page 7 — Audio Coverage: Phase 2 gate for YouTube trailer audio features."""

import streamlit as st
from pathlib import Path
from utils.data import BASE_DIR

st.set_page_config(page_title="Audio Coverage", layout="wide")
st.title("Audio Coverage — Phase 2 Gate")
st.markdown("Monitor YouTube trailer data pipeline coverage before enabling Phase 2 audio features.")

YOUTUBE_COVERAGE_THRESHOLD = 0.80
AUDIO_EXTRACTION_THRESHOLD = 0.80

# --- Check for audio feature data ---
audio_dir = BASE_DIR / 'audio_features'
audio_csv = audio_dir / 'audio_features.csv' if audio_dir.exists() else None
has_audio = audio_csv is not None and audio_csv.exists()

if has_audio:
    import pandas as pd
    audio_df = pd.read_csv(audio_csv)
    youtube_coverage = audio_df['trailer_id'].notna().mean() if 'trailer_id' in audio_df.columns else 0.0
    audio_coverage = audio_df.drop(columns=['trailer_id', 'imdb_id'], errors='ignore').notna().all(axis=1).mean()

    gate_cleared = (
        youtube_coverage >= YOUTUBE_COVERAGE_THRESHOLD and
        audio_coverage >= AUDIO_EXTRACTION_THRESHOLD
    )

    if gate_cleared:
        st.success("Gate cleared — Phase 2 audio features are ENABLED")
    else:
        st.warning("Gate not cleared — Phase 2 pipeline integration is DISABLED")

    col1, col2 = st.columns(2)
    col1.metric(
        "YouTube Trailer ID Coverage",
        f"{youtube_coverage:.1%}",
        f"{youtube_coverage - YOUTUBE_COVERAGE_THRESHOLD:+.1%} vs 80% target",
    )
    col2.metric(
        "Audio Extraction Success",
        f"{audio_coverage:.1%}",
        f"{audio_coverage - AUDIO_EXTRACTION_THRESHOLD:+.1%} vs 80% target",
    )

    # Progress bars
    st.progress(min(youtube_coverage / YOUTUBE_COVERAGE_THRESHOLD, 1.0), "YouTube Coverage")
    st.progress(min(audio_coverage / AUDIO_EXTRACTION_THRESHOLD, 1.0), "Audio Extraction")

else:
    st.error("Phase 2 Audio Integration Status: **DISABLED**")
    st.markdown("""
    No audio feature data found. The audio features directory (`audio_features/`) does not exist.

    To enable Phase 2:
    1. Set up the YouTube trailer extraction pipeline
    2. Extract audio features for >= 80% of the film corpus
    3. Save results to `audio_features/audio_features.csv`
    4. Re-visit this page to check coverage
    """)

    col1, col2 = st.columns(2)
    col1.metric("YouTube Trailer ID Coverage", "0.0%", "-80.0% vs 80% target")
    col2.metric("Audio Extraction Success", "0.0%", "-80.0% vs 80% target")

st.markdown("---")

# --- Placeholder Sections ---
st.subheader("Failure Diagnostics Panel")
st.info("This section will show films missing YouTube trailer IDs, failed extractions by error type, "
        "and coverage gaps by decade/genre once audio data is available.")

st.subheader("Coverage Trend Chart")
st.info("This section will show coverage % over time as the audio pipeline runs, "
        "tracking progress toward the 80% gate.")

st.subheader("Manual Trailer Override")
st.info("This section will allow manual YouTube URL entry for high-priority films "
        "with non-standard trailer sources.")
