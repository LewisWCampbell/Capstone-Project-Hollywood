"""Movie Genome Clustering Dashboard — Entry Point.

Launch with: streamlit run app.py
"""

import streamlit as st
from utils.data import check_data_status

st.set_page_config(
    page_title="Movie Genome Dashboard",
    page_icon=":film_projector:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global Custom CSS for a Premium "Hollywood" Aesthetic ---
st.markdown("""
<style>
    /* Main background & glassmorphism elements */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sleek container for metrics and cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 35, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 215, 0, 0.4); /* Subtle gold hover */
    }

    /* Titles and text styling */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #FFD700, #FDB931);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2937, #111827);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #FFD700;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.2);
    }
    
    /* Adjust expander matching */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Movie Genome Clustering Dashboard")
st.markdown("""
Interactive dashboard for the **UMAP + HDBSCAN genome clustering pipeline** across ~8,000 films with 269 hand-annotated features.

Use the sidebar to navigate between pages.
""")

# --- Status Cards ---
status = check_data_status()

col1, col2, col3 = st.columns(3)
col1.metric("CSV Data Files", f"{status['csv_count']}/{status['total_csvs']}")
col2.metric("Pipeline Artifacts", "Available" if status['artifacts_exist'] else "Not Found")
col3.metric("OMDB Data", "Available" if status['omdb_exists'] else "Not Found")

if not status['artifacts_exist']:
    st.warning(
        "Pipeline artifacts not found. Run the notebook (`Project_HOLLYWOOD.ipynb`) through "
        "the 'Save Pipeline Artifacts' cell, or use the **Pipeline Runner** page to generate them."
    )

# --- Page Directory ---
st.markdown("---")
st.subheader("Pages")

pages = [
    ("1. Data Health", "Validate genome data before running the pipeline. Inspect sparsity, feature distributions, IDF weights, and annotation density."),
    ("2. Pipeline Runner", "Run or re-run the full clustering pipeline with adjustable parameters. See live progress and metric summaries."),
    ("3. Cluster Explorer", "Explore the UMAP embedding interactively. Inspect cluster contents, discriminative features, and outlier tiers."),
    ("4. Film Explorer", "Deep-dive into any individual film — genome profile, soft membership, and nearest neighbours."),
    ("5. Experiment Tracker", "Compare pipeline runs across parameter configurations. Track metric history and parameter sensitivity."),
    ("6. Final Approved Rails", "Read-only summary of all approved content rails — names, descriptions, film counts, and downloadable mapping export."),
    ("7. Outlier Review", "Review XGBoost recovery suggestions for outlier films. Approve or reject cluster assignments one-by-one or in batch."),
]

for title, desc in pages:
    st.markdown(f"**{title}** — {desc}")
