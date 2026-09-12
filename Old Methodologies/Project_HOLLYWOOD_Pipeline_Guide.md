# Project HOLLYWOOD — Pipeline Guide

A comprehensive walkthrough of the movie clustering pipeline that transforms ~8,000 movies × 269+ genome features into a two-tier taxonomy of named, browsable content rails.

---

## Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROJECT HOLLYWOOD PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────────────────┐ │
│  │ CSV Files │   │   OMDB    │   │  Secrets  │   │   Pipeline Config    │ │
│  │ (3 files) │   │   API     │   │  (.env)   │   │   (Cell 6 — 50+     │ │
│  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   │    parameters)      │ │
│        │               │               │          └──────────┬──────────┘ │
│        ▼               ▼               ▼                     │            │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │              STEP 1: DATA INGESTION                         │          │
│  │  feature_data_longform.csv ──┐                              │          │
│  │  feature_taxonomy.csv ───────┤──→ Feature Matrix            │          │
│  │  genre_data.csv ─────────────┘    (8,000 × 247 base)       │          │
│  │                                                              │          │
│  │  OMDB API ──→ Posters + Metadata (jordan_df)                │          │
│  └──────────────────────────┬──────────────────────────────────┘          │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              STEP 2: FEATURE ENGINEERING                     │         │
│  │                                                              │         │
│  │  Base Features (247 genome tags, ordinal 0–3)                │         │
│  │       + Genre one-hot (22 cols)        [toggle]              │         │
│  │       + Jordan metadata (21 cols)      [toggle]              │         │
│  │       ─────────────────────────────                          │         │
│  │       = ~290 features per movie                              │         │
│  │                                                              │         │
│  │  MinMaxScale → IDF weighting → Feature matrix X              │         │
│  └──────────────────────────┬──────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              STEP 3: DIMENSIONALITY REDUCTION (UMAP)         │         │
│  │                                                              │         │
│  │  X (8,000 × ~290)  ──UMAP──→  embedding_cluster (8,000×60)  │         │
│  │                     ──UMAP──→  embedding_3d (8,000×3)        │         │
│  │                     ──UMAP──→  embedding_2d (8,000×2)        │         │
│  └──────────────────────────┬──────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              STEP 4: PRIMARY CLUSTERING                      │         │
│  │                                                              │         │
│  │  embedding_cluster ──HDBSCAN──→ cluster_labels               │         │
│  │                                  probabilities               │         │
│  │                                  soft_clusters               │         │
│  │                                  outlier_scores              │         │
│  └──────────────────────────┬──────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              STEP 5: SUB-CLUSTERING                           │         │
│  │                                                              │         │
│  │  For each cluster:                                           │         │
│  │    raw features (slice) ──UMAP──→ sub_embedding              │         │
│  │                          ──HDBSCAN──→ sub_labels             │         │
│  │                                       silhouette scores      │         │
│  └──────────────────────────┬──────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              STEP 6: NAMING & DELIVERABLES                   │         │
│  │                                                              │         │
│  │  Ollama LLM ──→ cluster names + descriptions                 │         │
│  │              ──→ sub-cluster names + descriptions             │         │
│  │              ──→ 5 deliverables (lists, descriptions, etc.)  │         │
│  └──────────────────────────┬──────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              STEP 7: EXPORT & DASHBOARD                      │         │
│  │                                                              │         │
│  │  pipeline_artifacts.pkl ──→ Streamlit Dashboard              │         │
│  │     (all embeddings, labels, names, config)                  │         │
│  └──────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 0: Setup (Cells 0–4)

**What it does:** Loads libraries, defines LLM helper functions for Ollama, and tests the Ollama connection.

```
┌──────────────────────────────┐
│  Ollama (local LLM server)   │
│  ┌────────────────────────┐  │
│  │  llama3.2:3b model     │  │
│  └────────────────────────┘  │
│         ▲                    │
│         │  HTTP API          │
│         │  localhost:11434   │
│  ┌──────┴─────────────────┐  │
│  │  call_ollama()         │  │
│  │  extract_tone_of_voice │  │
│  │  test_ollama()         │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

The LLM is used later for naming clusters. The `call_ollama()` function wraps the Ollama REST API with configurable temperature and token limits. `extract_tone_of_voice()` is defined here but currently unused — it's scaffolding for future tone-based feature extraction.

---

### Step 1: Pipeline Configuration (Cell 6)

**What it does:** A single master cell that defines every tunable parameter in the pipeline. All downstream cells read from these variables. There are 50+ parameters organized into groups.

```
┌─────────────────── CONFIGURATION GROUPS ───────────────────┐
│                                                             │
│  Feature Weighting          Clustering Method               │
│  ├─ WEIGHTING_METHOD        ├─ CLUSTERING_METHOD            │
│  │  (idf / zscore / none)   │  (hdbscan / kmeans / agglo)  │
│  ├─ IDF_TRANSFORM           ├─ HDBSCAN_MIN_CLUSTER_SIZE    │
│  │  (raw / sublinear / bm25)│  HDBSCAN_MIN_SAMPLES         │
│  └─ APPLY_VARIANCE_WEIGHT   │  HDBSCAN_EPSILON             │
│                              └─ HDBSCAN_SELECTION_METHOD    │
│                                                             │
│  UMAP Settings              Sub-clustering                  │
│  ├─ UMAP_N_COMPONENTS (60)  ├─ SUB_USE_RAW_FEATURES        │
│  ├─ UMAP_N_NEIGHBORS (15)   ├─ SUB_CLUSTERING_METHOD       │
│  ├─ UMAP_MIN_DIST (0.0)     ├─ SUB_UMAP_N_COMPONENTS (10) │
│  └─ UMAP_METRIC (auto)      └─ SUB_HDBSCAN_MIN_SIZE (5)   │
│                                                             │
│  Data Toggles               Sweeps (on/off)                 │
│  ├─ INCLUDE_GENRE_FEATURES  ├─ RUN_UMAP_SWEEP              │
│  ├─ INCLUDE_JORDAN_FEATURES ├─ RUN_HDBSCAN_SWEEP           │
│  └─ USE_ENSEMBLE_DISTANCE   ├─ RUN_KMEANS_SWEEP            │
│                              └─ RUN_AGGLO_SWEEP             │
│                                                             │
│  Outlier Tiers              LLM                             │
│  ├─ OUTLIER_TIER1_PERCENTILE├─ OLLAMA_MODEL                │
│  └─ OUTLIER_TIER3_PERCENTILE└─ OLLAMA_BASE_URL             │
└─────────────────────────────────────────────────────────────┘
```

**Key design decision — `UMAP_METRIC = 'auto'`:** When IDF weighting is active, the metric auto-resolves to `correlation`, which measures whether two movies have the *same relative profile shape* regardless of magnitude. For z-score or unweighted features, it falls back to `euclidean`.

---

### Step 1A: Secrets & OMDB Data Pull (Cells 7–17)

**What it does:** Loads API keys, fetches movie metadata from the OMDB API for all ~8,000 movies, downloads posters, and extracts Jordan's metadata features (content rating, language, decade).

```
┌──────────────────────────────────────────────────────────────────┐
│                    OMDB DATA PIPELINE                            │
│                                                                  │
│  genre_data.csv ──→ 8,000 IMDb IDs                               │
│                          │                                       │
│                          ▼                                       │
│                  ┌───────────────┐                                │
│                  │  Progress     │  (omdb_data/progress.json)    │
│                  │  Tracker      │  Tracks which IDs are done    │
│                  └───────┬───────┘                                │
│                          │                                       │
│            Already fetched? ──Yes──→ Skip                        │
│                  │ No                                            │
│                  ▼                                               │
│          ┌───────────────┐     0.05s delay                       │
│          │  OMDB API     │◄──────────────►  Rate limiting        │
│          │  ?i={imdb_id} │                                       │
│          └───────┬───────┘                                       │
│                  │                                               │
│          ┌───────┴───────┐                                       │
│          ▼               ▼                                       │
│   omdb_movies.json    posters/{id}.jpg                           │
│   (full metadata)     (movie posters)                            │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────────────────────────┐                       │
│   │  Jordan's Metadata Extraction         │                      │
│   │                                       │                      │
│   │  Year → Decade (1950s, 1960s, ...)    │                      │
│   │  Rated → Content Category             │                      │
│   │         (Family/Teen/Mature/Unknown)  │                      │
│   │  Language → Primary language           │                      │
│   │                                       │                      │
│   │  Output: jordan_df (in memory)        │                      │
│   └──────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

The progress tracker makes this **idempotent** — rerunning the cell only fetches movies that haven't been pulled yet. All 8,000 movies are fetched in a single run (no daily cap).

---

### Step 2: Data Loading & Feature Matrix Construction (Cells 19–31)

**What it does:** Loads three CSV files, pivots the long-form feature data into a movie × feature matrix, optionally adds genre columns and Jordan's metadata, then scales everything.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    FEATURE MATRIX ASSEMBLY                                │
│                                                                          │
│  feature_data_longform.csv                                               │
│  ┌──────────────────────────────────┐                                    │
│  │ imdb_id │ feature_id │ trigger   │   ~2M rows                        │
│  │ tt00001 │ 234        │ 2         │   (movie × feature pairs)         │
│  └────────────────┬─────────────────┘                                    │
│                   │  + feature_taxonomy.csv (feature_id → name)          │
│                   │                                                      │
│                   ▼  pivot_table(index=imdb_id, columns=feature)         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐                │
│  │              BASE FEATURE MATRIX                      │               │
│  │              8,000 movies × 247 genome features       │               │
│  │              Values: 0, 1, 2, 3 (ordinal intensity)   │               │
│  │              ~65% sparse (zeros)                       │               │
│  └────────────────────────┬─────────────────────────────┘                │
│                           │                                              │
│      ┌────────────────────┼────────────────────┐                         │
│      ▼                    ▼                    ▼                         │
│  ┌────────┐        ┌────────────┐       ┌──────────────┐                │
│  │ Genre  │        │  Jordan's  │       │   Base       │                │
│  │ One-Hot│        │  Metadata  │       │   Features   │                │
│  │ (22)   │        │  (21 cols) │       │   (247)      │                │
│  │        │        │            │       │              │                │
│  │ toggle │        │  toggle    │       │  always on   │                │
│  └───┬────┘        └──────┬─────┘       └──────┬───────┘                │
│      │                    │                    │                         │
│      └────────────────────┼────────────────────┘                         │
│                           ▼                                              │
│              ┌────────────────────────────┐                               │
│              │  COMBINED FEATURE MATRIX    │                              │
│              │  8,000 × ~290 features     │                              │
│              └─────────────┬──────────────┘                               │
│                            │                                             │
│                            ▼                                             │
│              ┌────────────────────────────┐                               │
│              │  MinMaxScaler [0, 1]       │                              │
│              │  Drop constant columns     │                              │
│              │  Impute missing → median   │                              │
│              └─────────────┬──────────────┘                               │
│                            │                                             │
│                            ▼                                             │
│                    X (scaled matrix)                                      │
│                    tt_codes (movie IDs)                                   │
│                    feature_names (column labels)                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Genre features (22 columns):** One-hot encoded from multi-genre strings like `"Action, Comedy"` → `genre_Action=1, genre_Comedy=1`. Prefixed with `genre_`.

**Jordan's metadata (21 columns):**

| Feature Group | Columns | Encoding |
|---|---|---|
| Content rating | 4 cols (`meta_rating_Family`, `_Teen`, `_Mature`, `_Unknown`) | One-hot |
| Language | 16 cols (`meta_lang_English`, `_French`, ... `_Other`) | One-hot (top 15 + Other) |
| Decade | 1 col (`meta_decade`) | Normalized 0–1 |

---

### Step 3: Feature Weighting (Cell 33)

**What it does:** Transforms the raw scaled features to emphasize rare, distinctive features and dampen ubiquitous ones. This is the most impactful preprocessing step.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     FEATURE WEIGHTING                                    │
│                                                                          │
│  X (MinMaxScaled)                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────┐                 │
│  │  WEIGHTING_METHOD = 'idf'                           │                 │
│  │                                                     │                 │
│  │  1. Count feature presence:                         │                 │
│  │     presence(f) = count(movies where f > 0)         │                 │
│  │                                                     │                 │
│  │  2. Compute IDF weights:                            │                 │
│  │     idf(f) = log( N / (presence(f) + 1) )           │                 │
│  │                                                     │                 │
│  │     Common feature (7000/8000 movies): low IDF      │                 │
│  │     Rare feature   (500/8000 movies):  high IDF     │                 │
│  │                                                     │                 │
│  │  3. Apply IDF_TRANSFORM = 'sublinear':              │                 │
│  │     weighted(f) = (1 + log(score)) × idf(f)         │                 │
│  │     (for score > 0; else 0)                         │                 │
│  │                                                     │                 │
│  │     This dampens the difference between             │                 │
│  │     intensity 2 and 3, while preserving             │                 │
│  │     the ordering: 3 > 2 > 1 > 0                    │                 │
│  └──────────────────────┬──────────────────────────────┘                 │
│                         │                                                │
│                         ▼                                                │
│                  X_weighted (final feature matrix)                        │
│                                                                          │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                    │
│  Alternative methods (configurable):                                     │
│    'zscore'  — StandardScaler (zero mean, unit variance)                │
│    'none'    — pass through MinMaxScaled values as-is                   │
│    'bm25'    — saturation curve: idf × score×(k+1)/(score+k)           │
│  Optional: APPLY_VARIANCE_WEIGHT multiplies by √variance               │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why sublinear IDF?** The raw genome features are ordinal (0, 1, 2, 3). Without weighting, a feature present in 7,000 of 8,000 movies contributes the same weight as one present in 500. IDF fixes this by upweighting distinctive features. Sublinear dampening prevents score=3 from dominating over score=2 — the *presence* matters more than the *intensity gap*.

---

### Step 3B: Ensemble Distance (Cell 35) — Optional

**What it does:** When `USE_ENSEMBLE_DISTANCE = True`, computes separate distance matrices for continuous and binary features using appropriate metrics, then blends them.

```
┌──────────────────────────────────────────────────────────────┐
│  ENSEMBLE DISTANCE (optional — currently disabled)           │
│                                                              │
│  Continuous features ──correlation distance──→ D_cont        │
│  Binary features     ──Jaccard distance──────→ D_binary      │
│  Audio features      ──Bray-Curtis──────────→ D_audio        │
│                                                              │
│  D_ensemble = 0.90 × D_cont + 0.10 × D_binary               │
│              (+ 0.20 × D_audio if Phase 2 enabled)           │
│                                                              │
│  Result: precomputed distance matrix → UMAP                  │
└──────────────────────────────────────────────────────────────┘
```

Currently disabled (`USE_ENSEMBLE_DISTANCE = False`). When active, UMAP receives a precomputed distance matrix instead of raw features.

---

### Step 4: UMAP Dimensionality Reduction (Cells 39–43)

**What it does:** Reduces the ~290-dimensional weighted feature space down to 60 dimensions for clustering, 3D for visualization, and 2D for flat plots. Optionally runs a parameter sweep first.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    UMAP DIMENSIONALITY REDUCTION                         │
│                                                                          │
│  X_weighted (8,000 × ~290)                                               │
│       │                                                                  │
│       ├──── [Optional: UMAP SWEEP] ──────────────────────────────┐       │
│       │     Tests combinations of:                                │       │
│       │       n_components: 10, 15, 20, 25, 30, 40, 50, 60, 80  │       │
│       │       n_neighbors:  15, 30, 50, 100, 150                 │       │
│       │     Metrics: trustworthiness, silhouette, Davies-Bouldin │       │
│       │     On 800-movie sample for speed                        │       │
│       │     Outputs: 6-panel diagnostic plot                     │       │
│       └──────────────────────────────────────────────────────────┘       │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────┐             │
│  │  THREE PARALLEL UMAP EMBEDDINGS                         │             │
│  │                                                         │             │
│  │  ┌─────────────────────────────────────────────┐        │             │
│  │  │  embedding_cluster  (8,000 × 60)            │        │             │
│  │  │  metric: correlation, n_neighbors: 15       │        │             │
│  │  │  PURPOSE: clustering input                  │        │             │
│  │  └─────────────────────────────────────────────┘        │             │
│  │                                                         │             │
│  │  ┌─────────────────────────────────────────────┐        │             │
│  │  │  embedding_3d  (8,000 × 3)                  │        │             │
│  │  │  PURPOSE: 3D interactive scatter plot        │        │             │
│  │  └─────────────────────────────────────────────┘        │             │
│  │                                                         │             │
│  │  ┌─────────────────────────────────────────────┐        │             │
│  │  │  embedding_2d  (8,000 × 2)                  │        │             │
│  │  │  PURPOSE: 2D flat visualisation              │        │             │
│  │  └─────────────────────────────────────────────┘        │             │
│  └─────────────────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why 60 dimensions for clustering?** Higher dimensionality preserves more structure from the original 290D space. HDBSCAN works well in 10–100D (much better than 290D). The sweep helps find the sweet spot where trustworthiness (local neighborhood preservation) is high while cluster quality metrics remain strong.

**Why separate vis embeddings?** The 3D/2D embeddings are optimized for visual separation (`min_dist=0.1`) rather than tight clustering (`min_dist=0.0`). They're only used for plots, never for the actual clustering.

---

### Step 5: Primary Clustering (Cells 45–48)

**What it does:** Groups movies into rails using the 60D clustering embedding. Supports three methods, with HDBSCAN as the recommended default.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    PRIMARY CLUSTERING                                     │
│                                                                          │
│  embedding_cluster (8,000 × 60)                                          │
│       │                                                                  │
│       ├──── [Optional: PARAMETER SWEEP] ─────────────────────────┐       │
│       │     HDBSCAN sweep: min_cluster_size × min_samples × ε    │       │
│       │     K-Means sweep: K = 2..30 (elbow + silhouette)        │       │
│       │     Agglomerative sweep: K = 2..30                       │       │
│       └──────────────────────────────────────────────────────────┘       │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────┐             │
│  │  HDBSCAN (default)                                      │             │
│  │                                                         │             │
│  │  Parameters:                                            │             │
│  │    min_cluster_size = 15  (smallest viable rail)        │             │
│  │    min_samples = 15       (density threshold)           │             │
│  │    epsilon = 0.2          (micro-cluster merging)       │             │
│  │    selection = 'eom'      (excess of mass)              │             │
│  │                                                         │             │
│  │  Outputs:                                               │             │
│  │    cluster_labels  — array of cluster IDs (-1 = outlier)│             │
│  │    probabilities   — per-movie confidence (0.0 – 1.0)   │             │
│  │    outlier_scores  — how "outlier-like" each point is   │             │
│  └─────────────────────────────────────────────────────────┘             │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────┐             │
│  │  SOFT CLUSTER MEMBERSHIPS                               │             │
│  │                                                         │             │
│  │  soft_clusters: (8,000 × n_clusters) matrix             │             │
│  │  Each movie gets a probability for EVERY cluster        │             │
│  │                                                         │             │
│  │  Primary rail   = argmax(soft_clusters)                 │             │
│  │  Secondary rail = 2nd highest if > 0.15 threshold       │             │
│  │                                                         │             │
│  │  Example:                                               │             │
│  │    Movie X: [0.82, 0.12, 0.04, 0.02, ...]              │             │
│  │    → Primary: Rail 0 (82%), Secondary: none             │             │
│  │                                                         │             │
│  │    Movie Y: [0.45, 0.35, 0.15, 0.05, ...]              │             │
│  │    → Primary: Rail 0 (45%), Secondary: Rail 1 (35%)     │             │
│  └─────────────────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
```

**Alternative methods (configurable):**

| Method | How it works | Outlier handling |
|---|---|---|
| **HDBSCAN** | Finds density peaks in the embedding; natural cluster count | Native outliers (label = -1) |
| **K-Means** | Assigns to nearest of K centroids; softmax on distances for probabilities | Synthetic: top 5% most distant flagged |
| **Agglomerative** | Hierarchical merging (Ward linkage); softmax on centroid distances | Synthetic: same as K-Means |

---

### Step 5B: Sub-Clustering (Cell 52)

**What it does:** For each primary cluster above a minimum size, runs a second round of UMAP + HDBSCAN to find sub-rails within the cluster. Computes per-sub-cluster silhouette scores.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    SUB-CLUSTERING (per cluster)                           │
│                                                                          │
│  For each cluster (size ≥ 30):                                           │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │  Source selection:                                           │        │
│  │                                                              │        │
│  │  SUB_USE_RAW_FEATURES = True (recommended):                  │        │
│  │    X[cluster_mask]  (raw weighted features, ~290D)           │        │
│  │    → avoids compounding UMAP distortion                     │        │
│  │                                                              │        │
│  │  SUB_USE_RAW_FEATURES = False:                               │        │
│  │    embedding_cluster[cluster_mask]  (already UMAP'd, 60D)   │        │
│  │    → UMAP-on-UMAP (faster but lossier)                      │        │
│  └──────────────────────┬───────────────────────────────────────┘        │
│                         │                                                │
│                         ▼                                                │
│            UMAP (n_components=10, n_neighbors=15)                        │
│                         │                                                │
│                         ▼                                                │
│            HDBSCAN (min_size=5, min_samples=3)                           │
│              or Agglomerative (n_clusters=3)                             │
│                         │                                                │
│                         ▼                                                │
│            sub_labels + silhouette scores                                │
│                                                                          │
│  Result: sub_clusters = {                                                │
│      0: [0, 1, 1, -1, 2, 0, ...],  ← sub-labels for cluster 0         │
│      1: [0, 0, 1, 2, -1, ...],     ← sub-labels for cluster 1         │
│      ...                                                                 │
│  }                                                                       │
│                                                                          │
│  sub_cluster_silhouettes = {                                             │
│      0: {0: 0.342, 1: 0.281, 2: 0.198, '_overall': 0.274},             │
│      ...                                                                 │
│  }                                                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why go back to raw features?** When `SUB_USE_RAW_FEATURES = True`, the sub-clustering UMAP starts from the full ~290D feature space (filtered to just that cluster's movies). This avoids the "UMAP-on-UMAP" problem where compounding nonlinear transforms can distort local structure. The tradeoff is slightly slower computation, but more faithful sub-clusters.

---

### Step 6: Outlier Classification (Cell 54)

**What it does:** Classifies outlier movies (those not confidently placed in any cluster) into three tiers based on their outlier scores.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    THREE-TIER OUTLIER SYSTEM                              │
│                                                                          │
│  outlier_scores (from HDBSCAN / distance-based)                          │
│       │                                                                  │
│       ▼                                                                  │
│  ┌────────────────────────────────────────────────────────┐              │
│  │                                                        │              │
│  │  Tier 1 (Mild): score < 60th percentile                │              │
│  │    "Near-cluster" — could plausibly belong to a rail    │              │
│  │    Displayed in dashboard, considered for sub-rails     │              │
│  │                                                        │              │
│  │  Tier 2 (Moderate): 60th–85th percentile               │              │
│  │    "Between clusters" — genuinely ambiguous films       │              │
│  │    Shown but flagged as uncertain                       │              │
│  │                                                        │              │
│  │  Tier 3 (True): score ≥ 85th percentile                │              │
│  │    "True outliers" — don't fit any cluster well         │              │
│  │    May be unique films, data quality issues, etc.       │              │
│  │                                                        │              │
│  └────────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Step 7: Cluster Naming via LLM (Cells 60–61)

**What it does:** Uses a local Ollama LLM to generate human-readable names and descriptions for each cluster and sub-cluster, based on their distinguishing features.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LLM NAMING PIPELINE                                    │
│                                                                          │
│  1. PRE-COMPUTE top features for ALL clusters                            │
│     all_cluster_tops = { 0: [feat1, feat2, ...], 1: [...], ... }        │
│                                                                          │
│  2. FOR EACH CLUSTER:                                                    │
│     ┌─────────────────────────────────────────────────────────┐          │
│     │  get_discriminative_features(cluster_id)                 │          │
│     │  → top 20 elevated features vs global mean               │          │
│     │                                                          │          │
│     │  get_differentiating_features(cluster_id, all_tops)      │          │
│     │  → check overlap with other clusters' top features       │          │
│     │  → find features UNIQUE to this cluster                  │          │
│     │                                                          │          │
│     │  Build LLM prompt:                                       │          │
│     │    "You are a film taxonomy expert..."                   │          │
│     │    ELEVATED: feature1 (+0.42), feature2 (+0.31), ...     │          │
│     │    NOTE: Shared with Rail 3: [British, crime]            │          │
│     │    UNIQUE to this rail: [noir, cynical]                  │          │
│     │    Example movies: Movie A, Movie B, ...                 │          │
│     │                                                          │          │
│     │  Ollama → {"name": "...", "description": "..."}          │          │
│     └─────────────────────────────────────────────────────────┘          │
│                                                                          │
│  3. FOR EACH SUB-CLUSTER:                                                │
│     Same process but features are computed vs PARENT cluster mean        │
│     Dedup checks against SIBLING sub-clusters (not all clusters)         │
│                                                                          │
│  Output:                                                                 │
│    cluster_names = {                                                     │
│        0: { 'name': 'Dark British Crime',                                │
│             'description': 'Gritty UK crime dramas...',                  │
│             'top_features': ['British (+0.42)', ...],                     │
│             'unique_differentiators': ['noir (+0.18)', ...] },           │
│        ...                                                               │
│    }                                                                     │
│    sub_cluster_names = {                                                 │
│        (0, 0): { 'name': 'London Heist', ... },                         │
│        (0, 1): { 'name': 'Rural Mystery', ... },                        │
│        ...                                                               │
│    }                                                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

**Deduplication logic:** If two clusters share the same top features (e.g., both have "British" and "crime" as their top-2), the system finds features that are unique to each cluster and adds them to the LLM prompt as differentiators. This pushes the LLM toward more specific names rather than naming both rails "British Crime."

---

### Step 8: Deliverables (Cells 63–73)

**What it does:** Produces five structured deliverables that summarize the pipeline results.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DELIVERABLES                                     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐                │
│  │  Deliverable 1: RAIL LIST                            │                │
│  │  Two-tier hierarchy with sizes and top features      │                │
│  │    Rail 0: Dark British Crime (450 films)            │                │
│  │      → Sub 0: London Heist (180 films)               │                │
│  │      → Sub 1: Rural Mystery (140 films)              │                │
│  │      → Sub 2: Political Thriller (130 films)         │                │
│  └──────────────────────────────────────────────────────┘                │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐                │
│  │  Deliverable 2: DESCRIPTIONS                         │                │
│  │  One-paragraph LLM-generated description per rail    │                │
│  └──────────────────────────────────────────────────────┘                │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐                │
│  │  Deliverable 3: COHESION EXPLANATIONS                │                │
│  │  Why each cluster is internally consistent:          │                │
│  │  top elevated features + suppressed features         │                │
│  └──────────────────────────────────────────────────────┘                │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐                │
│  │  Deliverable 4: FLAWS & CAVEATS                      │                │
│  │  Pipeline limitations, outlier rates, edge cases     │                │
│  └──────────────────────────────────────────────────────┘                │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐                │
│  │  Deliverable 5: GENRE BASELINE COMPARISON            │                │
│  │  K-Means on genre-only features vs full pipeline     │                │
│  │  Metrics: silhouette, Davies-Bouldin, Calinski-H.    │                │
│  └──────────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Step 9: Export & Dashboard (Cells 75–77)

**What it does:** Saves all pipeline artifacts to a compressed pickle file and launches a Streamlit dashboard for interactive exploration.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    EXPORT & DASHBOARD                                     │
│                                                                          │
│  pipeline_artifacts.pkl (compressed, ~50-100MB)                          │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  Embeddings:  embedding_cluster, _3d, _2d          │                  │
│  │  Labels:      cluster_labels, probabilities         │                  │
│  │  Soft:        soft_clusters, outlier_scores          │                  │
│  │  Sub:         sub_clusters, sub_cluster_silhouettes │                  │
│  │  Features:    X, X_weighted, feature_names           │                  │
│  │  Names:       cluster_names, sub_cluster_names       │                  │
│  │  Tiers:       tier1_mask, tier2_mask, tier3_mask     │                  │
│  │  Config:      full pipeline configuration dict       │                  │
│  └────────────────────────────────────────────────────┘                  │
│                         │                                                │
│                         ▼                                                │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  STREAMLIT DASHBOARD (app.py)                      │                  │
│  │                                                    │                  │
│  │  Page 1: Overview & 3D Scatter                     │                  │
│  │  Page 2: Data Health & Sparsity                    │                  │
│  │  Page 3: Cluster Explorer                          │                  │
│  │    → Rail Browser with poster carousels            │                  │
│  │    → Genome tags per cluster/sub-cluster           │                  │
│  │    → Silhouette scores                             │                  │
│  │    → Rename / Auto-name / Merge / Delete actions   │                  │
│  │    → Run persistence (save/load configurations)    │                  │
│  └────────────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
  CSV files (3)          OMDB API
       │                    │
       ▼                    ▼
  Feature Matrix ──+── Jordan Metadata ──+── Genre One-Hot
       │                                       │
       └──────────────── MERGE ────────────────┘
                           │
                    8,000 × ~290
                           │
                     MinMaxScale
                           │
                    IDF + Sublinear
                           │
                      X_weighted
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          UMAP 60D     UMAP 3D     UMAP 2D
              │
          HDBSCAN
              │
       cluster_labels
       probabilities
              │
       ┌──────┴──────┐
       ▼             ▼
   Sub-UMAP      Soft membership
   Sub-HDBSCAN   Secondary rails
       │
  sub_clusters
  silhouette scores
       │
  Ollama LLM naming
       │
  Deliverables + Dashboard
```

---

## Key Configuration Recommendations

The pipeline has been tested extensively. The configuration that produces the most coherent-feeling clusters:

| Parameter | Recommended Value | Why |
|---|---|---|
| `WEIGHTING_METHOD` | `'idf'` | Upweights distinctive features in sparse data |
| `IDF_TRANSFORM` | `'sublinear'` | Dampens intensity differences, emphasizes presence |
| `UMAP_METRIC` | `'correlation'` (via auto) | Compares profile shapes, not magnitudes |
| `CLUSTERING_METHOD` | `'hdbscan'` | Natural cluster count, native outliers, soft membership |
| `SUB_USE_RAW_FEATURES` | `True` | Avoids UMAP-on-UMAP distortion |
| `INCLUDE_GENRE_FEATURES` | `True` | Adds 22 genre signals |
| `INCLUDE_JORDAN_FEATURES` | `True` | Adds rating/language/decade context |

---

## Cell Index Reference

| Cells | Section | Purpose |
|---|---|---|
| 0–4 | Setup | Libraries, Ollama helpers, connection test |
| 5–6 | Config | Master configuration cell (50+ parameters) |
| 7–8 | Secrets | Load API keys from SECRETS file |
| 9–10 | OMDB Check | Verify existing OMDB data, Phase 2 audio gate |
| 11–17 | OMDB Pull | Fetch metadata, download posters, extract Jordan's features |
| 18–20 | Data Load | Import CSVs, create feature matrix |
| 21–24 | EDA | Peek at datasets, inspect distributions |
| 25 | Pivot | Build movie × feature matrix from long-form data |
| 26–27 | Genre | One-hot encode genre features (toggleable) |
| 28–29 | Jordan | Encode OMDB metadata features (toggleable) |
| 30–31 | Scale | MinMaxScale, drop constants, impute |
| 32–33 | Weight | IDF / z-score / none feature weighting |
| 34–35 | Ensemble | Optional ensemble distance matrix |
| 36–37 | Phase 2 | Optional audio feature integration |
| 38–41 | UMAP Sweep | Parameter sweep with diagnostic plots |
| 42–43 | UMAP Final | Fit 60D, 3D, and 2D embeddings |
| 44–46 | K-Means | Sweep + final K-Means (if selected) |
| 47–48 | HDBSCAN | Sweep + final HDBSCAN (if selected) |
| 49–50 | Soft | Soft membership, secondary rail assignment |
| 51–52 | Sub-cluster | Recursive sub-clustering with silhouette scores |
| 53–54 | Outliers | Three-tier outlier classification + centroids |
| 55–56 | 3D Plot | Interactive Plotly 3D scatter |
| 57–58 | Review | Single-cluster deep-dive inspection |
| 59–61 | Naming | LLM naming with deduplication logic |
| 62–63 | Mapping | Map names onto results_df |
| 64–73 | Deliverables | 5 structured output reports |
| 74–75 | Export | Save artifacts to pkl |
| 76–77 | Dashboard | Launch Streamlit app |
