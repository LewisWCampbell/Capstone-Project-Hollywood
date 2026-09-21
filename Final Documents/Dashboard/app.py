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

REPO_URL = "https://github.com/LewisWCampbell/Capstone-Project-Hollywood"
NB_FINAL = f"{REPO_URL}/blob/main/Final%20Documents/PROJECT_HOLLYWOOD_FINAL.ipynb"
NB_EXPERIMENTAL = f"{REPO_URL}/blob/main/Final%20Documents/PROJECT_HOLLYWOOD_EXPERIMENTAL.ipynb"
CASE_STUDY_URL = "https://lewiswcampbell.com/blog/movie-rails-capstone.html"

st.title("Project Hollywood")
st.markdown(
    "#### Can a machine build the rows on a streaming homepage, using nothing but "
    "what the films *are*?"
)

st.markdown(f"""
This dashboard is the delivery layer of an MS Business Analytics capstone (Chapman University).
The pipeline behind it takes **8,000 films**, described by a 248-feature content genome plus
genre and decade flags, and organizes them into **67 coherent, human-readable "rails"** with no
viewing data at all. Every rail is then validated, explained, and named automatically, and this
app is where a curator reviews the result.

**Use the sidebar to explore.** Start with **Final Rails** to see the shippable output, then open
**Cluster Explorer** to click into any rail, see the features that define it, and browse its films.
""")

# --- Headline numbers ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Films organized", "8,000")
m2.metric("Rails discovered", "67")
m3.metric("Silhouette", "0.503")
m4.metric("XGBoost label CV accuracy", "> 90 %")

# --- How it was built: point people at the notebooks ---
st.markdown("---")
st.subheader("How it was built")
st.markdown(f"""
The dashboard shows *results*. The reasoning lives in two Jupyter notebooks in the repository,
and they are the best way to understand each decision:

- **[PROJECT_HOLLYWOOD_FINAL.ipynb]({NB_FINAL})** is the pipeline start to finish, with a markdown
  cell above every code cell explaining what it does and why. Read this one first.
- **[PROJECT_HOLLYWOOD_EXPERIMENTAL.ipynb]({NB_EXPERIMENTAL})** holds the experiments behind the
  final choices: the parameter sweeps, ablations, and the approaches that lost.
""")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1. Weight and embed**")
    st.caption(
        "Each genome feature is IDF-weighted so rare, discriminating attributes carry more signal, "
        "then z-scored. UMAP reduces the 284-dimension space to 20 components; the settings came "
        "from a 105-combination sweep judged on neighbourhood preservation and trustworthiness."
    )
with c2:
    st.markdown("**2. Cluster and validate**")
    st.caption(
        "HDBSCAN finds the rails and is honest about films that do not belong anywhere; "
        "660 parameter combinations were compared. An XGBoost model then re-predicts the labels "
        "from the raw features, proving the rails are real structure, and recovers confident "
        "outliers (about 29 % of the noise)."
    )
with c3:
    st.markdown("**3. Explain and name**")
    st.caption(
        "SHAP surfaces the handful of features that make each rail distinct. A language model "
        "turns each rail's representative films, genre mix and era into a 2 to 5 word name and a "
        "one-line description that a curator can accept, edit, or reject here."
    )

st.markdown(f"""
[Browse the repository]({REPO_URL}) &nbsp;·&nbsp; [Read the case study]({CASE_STUDY_URL})
""")

# --- Data note ---
with st.expander("A note on the data"):
    st.markdown("""
The content genome was provided for the capstone by an industry partner and is proprietary.
To make the project public, the feature, category and sub-category **names** are anonymized to
ID-keyed placeholders (`Feature 0440`, `Category 13`). Feature IDs, relevance scores and every
stage of the pipeline are unchanged, so the results you see here reproduce exactly. Film titles,
genres, and poster metadata are public data from IMDb, OMDb and TMDB.
""")

# --- Status ---
st.markdown("---")
st.subheader("Pages")

pages = [
    ("Final Rails", "The shippable output: every approved rail with its name, description, film count, and an exportable film-to-rail mapping."),
    ("Cluster Explorer", "Interactive UMAP views. Click into a rail to see its defining features, its films with posters, and outlier tiers."),
    ("Film Explorer", "Search any title and see its rail, assignment confidence, genome profile and nearest neighbours."),
    ("Outlier Review", "Triage films HDBSCAN could not place, using the XGBoost recovery suggestions, one at a time or in batch."),
    ("Data Health", "Sparsity, feature distributions, IDF weights and annotation density of the input data."),
    ("Pipeline Runner", "Re-run the clustering pipeline with different parameters and watch the metrics change."),
    ("Experiment Tracker", "Compare runs across parameter configurations and track metric history."),
]
for title, desc in pages:
    st.markdown(f"**{title}**: {desc}")

status = check_data_status()
with st.expander("Environment status"):
    s1, s2, s3 = st.columns(3)
    s1.metric("Data files", f"{status['csv_count']}/{status['total_csvs']}")
    s2.metric("Pipeline artifacts", "Available" if status['artifacts_exist'] else "Not found")
    s3.metric("OMDb metadata", "Available" if status['omdb_exists'] else "Not found")
    if not status['artifacts_exist']:
        st.warning(
            "Pipeline artifacts not found. Run the FINAL notebook through the "
            "'Save Pipeline Artifacts' cell, or use the Pipeline Runner page to generate them."
        )

    from utils import llm as _llm
    st.markdown(f"**LLM naming backend:** `{_llm.llm_backend()}`")
    if st.button("Test LLM naming connection"):
        with st.spinner("Calling the naming model..."):
            reply = _llm.query_ollama(
                "Reply with exactly two words: connection works"
            )
        if reply:
            st.success(f"Backend `{_llm.llm_backend()}` replied: {reply}")
        else:
            st.error(
                f"No reply from backend `{_llm.llm_backend()}`. "
                f"Last error: {_llm.LAST_LLM_ERROR or 'none recorded'}"
            )
