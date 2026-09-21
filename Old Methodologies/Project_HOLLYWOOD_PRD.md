# PROJECT HOLLYWOOD
## Product Requirements Document
### IDF Pipeline + Streamlit Dashboard

**Version 1.0 | March 2026 | Lewis Campbell**

---

## 1. Executive Summary

Project HOLLYWOOD is a movie clustering system that groups approximately 8,000 films into thematic clusters based on 247 genome features, genre metadata, and decade information. The system uses an IDF-weighted feature pipeline feeding into UMAP dimensionality reduction and HDBSCAN density-based clustering, with results served through a multi-page Streamlit dashboard for interactive exploration, cluster management, and production export.

The pipeline processes three distinct feature types with per-type preprocessing, applies Inverse Document Frequency weighting to continuous genome features, reduces dimensionality while preserving neighbourhood structure, and discovers natural density-based clusters without requiring a predefined cluster count.

---

## 2. System Overview

### 2.1 Architecture

The system consists of two primary components: a Jupyter notebook pipeline that processes raw feature data into clustered outputs, and a Streamlit dashboard that provides interactive visualization and cluster management. The pipeline runs as a batch process producing artifact files consumed by the dashboard.

### 2.2 Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Pipeline Runtime | Jupyter Notebook (Python 3) | Project_HOLLYWOOD_IDF.ipynb |
| Data Manipulation | pandas, numpy | DataFrames, array operations |
| IDF Weighting | sklearn TfidfTransformer | norm=None, smooth_idf=True; BM25 uses custom saturation |
| Dimensionality Reduction | umap-learn | UMAP with correlation distance |
| Clustering | hdbscan | Density-based, no predefined k |
| Imputation | sklearn SimpleImputer | Median strategy for genome features |
| One-Hot Encoding | pandas get_dummies | Decade feature encoding |
| Quality Metrics | sklearn.metrics, sklearn.manifold | silhouette_score, trustworthiness |
| DBCV | hdbscan (native) | relative_validity_ attribute |
| Nearest Neighbours | sklearn NearestNeighbors | Embedding quality measurement |
| Cluster Naming | Ollama (llama3.2:3b) | Local LLM for descriptive names |
| Artifact Storage | joblib | Serialised pipeline outputs |
| Dashboard | Streamlit | Multi-page application |
| Visualisation | Plotly | Interactive charts, 3D scatter |

### 2.3 Data Sources

| Source | Format | Contents |
|--------|--------|----------|
| feature_data_longform.csv | CSV (long) | ~8,000 movies × 247 genome scores (0–1) |
| feature_taxonomy.csv | CSV | Genome tag ID to tag name mapping |
| omdb_data/omdb_movies.json | JSON | OMDB metadata: genres, year, rating, plot |
| omdb_data/jordan_metadata.csv | CSV | Extracted decade, IMDB codes, titles |

---

## 3. Pipeline Specification

The pipeline consists of nine sequential steps. Each step is isolated to its own responsibility, and parameter sweeps only measure metrics within their own scope.

### 3.1 Step 1 — Load Genome Data

Pivots longform genome scores into a wide feature matrix. Each row is a movie (indexed by IMDB ID), each column is a genome tag, and values are continuous relevance scores between 0 and 1.

- **Output:** ~8,000 × 247 DataFrame
- **Library:** pandas (pivot_table)

### 3.2 Step 2 — Load OMDB Metadata

Loads OMDB movie data from JSON and extracts structured metadata including year, rating, genre strings, and decade. Builds title lookup tables and IMDB tt-code mappings for downstream dashboard use.

- **Output:** jordan_df, title_lookup, tt_codes
- **Library:** pandas, json (standard library)

### 3.3 Step 3 — Genre + Decade One-Hot Encoding

**Genre features** are extracted from OMDB genre strings (comma-separated), pivoted into binary one-hot columns using a numeric dummy value approach that avoids a known pandas bug with string-valued pivot tables. This produces approximately 23 binary columns (genre_Action, genre_Comedy, genre_Drama, etc.).

**Decade features** are encoded as independent binary columns using `pandas.get_dummies()`, producing approximately 10 columns (decade_1920s through decade_2020s). This treats each decade as an independent categorical signal with no ordinal assumption — a 1980s film is not assumed to be "closer to" a 1990s film than a 1960s film.

- **Output:** Feature matrix expanded to ~280 columns (247 genome + ~23 genre + ~10 decade)
- **Library:** pandas (pivot_table, get_dummies)

### 3.4 Step 4 — Per-Type Preprocessing

Each feature type is handled according to its statistical nature rather than applying a blanket transformation. Genome features are bounded continuous scores, while genre and decade features are binary indicators that would be distorted by scaling.

| Feature Type | Count | Nature | Preprocessing |
|-------------|-------|--------|---------------|
| Genome | ~247 | Continuous 0–1 | Impute missing with median (sklearn SimpleImputer) |
| Genre | ~23 | Binary 0/1 | Fill NaN with 0 (absent = not in genre) |
| Decade | ~10 | Binary 0/1 | Fill NaN with 0 (absent = no decade data) |

Constant features (zero variance across all movies) are dropped before proceeding to prevent degenerate behaviour in downstream distance calculations.

- **Library:** sklearn.impute.SimpleImputer (median), pandas (fillna)

### 3.5 Step 5 — IDF Weighting

Inverse Document Frequency weighting is applied exclusively to genome features. Genre and decade features are left unweighted because they are binary indicators where rarity does not imply greater discriminative value — a rare genre is not more useful for clustering, just less common.

#### 3.5.1 Implementation

Raw and sublinear IDF transforms use sklearn's `TfidfTransformer` with `norm=None` (no row normalisation) and `smooth_idf=True`. The sublinear option uses `sublinear_tf=True`, which applies `1+log(tf)` before multiplying by IDF. BM25 uses sklearn for IDF weight computation but applies the Okapi BM25 saturation formula manually, as sklearn has no BM25 equivalent for feature weighting.

#### 3.5.2 Why sklearn Works Here

sklearn's IDF formula (`log((1+N)/(1+df)) + 1`) differs from classic IDF (`log(N/(df+1))`) by a constant shift. Since the pipeline uses correlation distance downstream (which is shift-invariant), this constant has zero effect on clustering results. The relative ordering of rare vs common features is identical between both formulas.

#### 3.5.3 Transform Options

| Transform | Formula | Behaviour |
|-----------|---------|-----------|
| raw | score × IDF | Direct weighting. Preserves full score magnitude. Default. |
| sublinear | (1 + log(score)) × IDF | Dampens high scores while preserving ordering. |
| bm25 | IDF × score×(k1+1) / (score+k1) | Saturation curve. Low k1 = fast saturation. |

BM25 has no standard sklearn equivalent. The formula is a well-known information retrieval standard (Okapi BM25) and is kept as a direct implementation for transparency.

#### 3.5.4 IDF Sweep

Tests 7 configurations: raw, sublinear, and BM25 with k1 values of 0.5, 1.0, 1.2, 1.5, and 2.0. The sweep measures only feature-level metrics (not clustering quality): mean per-feature variance, top-10 variance concentration ratio, effective dimensionality (features for 90% of total variance), and sparsity.

### 3.6 Step 6 — UMAP Dimensionality Reduction

Uniform Manifold Approximation and Projection reduces the ~280-dimensional weighted feature space to a lower-dimensional embedding suitable for clustering. Two separate UMAP runs are performed: one for clustering (30D) and one for visualisation (3D).

| Parameter | Clustering UMAP | Visualisation UMAP |
|-----------|----------------|-------------------|
| n_components | 30 | 3 |
| n_neighbors | 15 | 15 |
| min_dist | 0.0 | 0.1 |
| metric | correlation | correlation |

Correlation distance was chosen because it is scale-invariant: IDF-weighted genome features and binary genre/decade features are compared on the basis of their pattern shapes rather than raw magnitudes.

- **Library:** umap-learn (umap.UMAP)

#### 3.6.1 UMAP Sweep

Sweeps 216 combinations: 6 n_components × 4 n_neighbors × 3 metrics × 3 min_dist. Measures only embedding-level metrics: trustworthiness (sklearn.manifold.trustworthiness), reconstruction error (normalised pairwise distance distortion via scipy.spatial.distance.pdist), and neighbourhood overlap (fraction of k=15 nearest neighbours preserved, via sklearn NearestNeighbors).

### 3.7 Step 7 — HDBSCAN Clustering

Hierarchical Density-Based Spatial Clustering discovers clusters as dense regions in the 30D UMAP embedding. Unlike k-means, HDBSCAN does not require specifying the number of clusters and naturally identifies outliers as points in low-density regions.

| Parameter | Value | Description |
|-----------|-------|-------------|
| min_cluster_size | 60 | Minimum films to form a valid cluster |
| min_samples | 15 | Core distance smoothing (higher = more conservative) |
| cluster_selection_epsilon | 0.1 | Merge clusters closer than this distance |
| selection_method | eom | Excess of Mass (preserves hierarchy depth) |
| gen_min_span_tree | True | Required for DBCV computation |

- **Library:** hdbscan (hdbscan.HDBSCAN)

#### 3.7.1 HDBSCAN Sweep

Sweeps 350 combinations: 7 min_cluster_size × 5 min_samples × 5 epsilon × 2 selection_method (eom, leaf). This is the only sweep that measures cluster-level quality metrics: silhouette score (sklearn), DBCV (hdbscan native), cluster persistence via Adjusted Rand Index (sklearn), and average HDBSCAN membership probability.

### 3.8 Step 8 — LLM Cluster Naming

Each cluster is named by a local Ollama LLM (llama3.2:3b) based on its most discriminative features. The system computes cluster centroids, identifies features that deviate most from the global average, and prompts the LLM to generate a concise descriptive name.

- **Library:** requests (HTTP to Ollama API)

### 3.9 Step 9 — Export + Dashboard Launch

All pipeline artifacts are serialised via joblib and saved to the results directory. The Streamlit dashboard is launched from a notebook cell, loading these artifacts at startup.

- **Library:** joblib (serialisation), streamlit (dashboard)

---

## 4. Sweep Design Philosophy

Each parameter sweep is scoped to measure only what that step controls. This prevents confounding: if the IDF sweep ran full clustering, a bad UMAP configuration could mask a good IDF choice.

| Sweep | Scope | Metrics Measured | Does NOT Measure |
|-------|-------|-----------------|-----------------|
| IDF Sweep | Feature weighting | Mean variance, top-10 ratio, effective dim, sparsity | Embedding or cluster quality |
| UMAP Sweep | Embedding quality | Trustworthiness, reconstruction error, NN overlap | Cluster quality |
| HDBSCAN Sweep | Cluster quality | Silhouette, DBCV, persistence (ARI), avg probability | Embedding quality |

Results cascade naturally: pick the IDF transform from the IDF sweep, lock it in, pick UMAP params from the UMAP sweep, lock those in, then pick HDBSCAN params from the HDBSCAN sweep.

---

## 5. Library Usage Audit

The pipeline maximises library usage wherever standard implementations exist. Only the BM25 saturation formula is implemented manually (one line), with documented justification.

| Operation | Implementation | Library Alternative | Decision |
|-----------|---------------|-------------------|----------|
| Genome imputation | sklearn SimpleImputer | N/A | Already using library |
| Decade one-hot | pandas get_dummies | N/A | Already using library |
| Genre one-hot | pandas pivot_table | N/A | Already using library |
| Constant removal | pandas nunique | sklearn VarianceThreshold | pandas is simpler here |
| IDF computation | sklearn TfidfTransformer | N/A | Now using library (norm=None, smooth_idf=True) |
| BM25 saturation | Custom: Okapi BM25 | rank_bm25 (pip) | IDF from sklearn; saturation is 1 line, no library needed |
| UMAP | umap-learn | N/A | Already using library |
| HDBSCAN | hdbscan | N/A | Already using library |
| Silhouette | sklearn silhouette_score | N/A | Already using library |
| DBCV | hdbscan relative_validity_ | N/A | Already using library (native) |
| Trustworthiness | sklearn trustworthiness | N/A | Already using library |
| Nearest Neighbours | sklearn NearestNeighbors | N/A | Already using library |
| ARI (persistence) | sklearn adjusted_rand_score | N/A | Already using library |
| Pairwise distances | scipy pdist | N/A | Already using library |
| Cluster naming | Ollama API (requests) | N/A | Already using library |
| Serialisation | joblib | N/A | Already using library |

---

## 6. Dashboard Specification

The Streamlit dashboard is a multi-page application providing interactive exploration, cluster management, and production export capabilities. It consumes artifacts produced by the pipeline notebook.

### 6.1 App Shell (app.py)

Entry point and navigation hub. Implements Hollywood-themed glassmorphic styling with dark background (#0d1117) and gold accents (#FFD700). Displays data availability status cards and provides a directory of all subpages.

### 6.2 Page 1: Data Health

Validates genome data quality before pipeline execution. Displays corpus summary metrics (film count, feature counts, sparsity), a sparsity heatmap across films and features, per-feature distribution histograms with variance warnings, IDF weight previews, per-film annotation density analysis, and a sortable variance table with CSV export.

### 6.3 Page 2: Pipeline Runner

Executes the full UMAP + HDBSCAN pipeline with real-time progress feedback. Provides sidebar controls for all preprocessing and hyperparameter options, displays run results with delta comparison to previous runs, and maintains a JSONL-based run history.

### 6.4 Page 3: Cluster Explorer

The primary interactive workspace (1,130 lines). Features a 2D UMAP scatter plot with convex hull overlays, cluster selection with detailed info panels showing defining features and genre distribution. Full cluster management: merge, delete, rename, and confirm workflows. All edits logged to an audit trail. Supports multiple colour-by modes (cluster, genre, outlier tier, confidence, entropy) and save/load of named run snapshots.

### 6.5 Page 4: Film Explorer

Deep dive into individual films. Shows genome radar charts with optional cluster centroid overlay and comparison mode, soft cluster membership probabilities, nearest-neighbour search in embedding space (configurable k, using sklearn NearestNeighbors), OMDB metadata and poster images, and outlier assessment with tier classification.

### 6.6 Page 5: Experiment Tracker

Compares multiple pipeline runs with metric history charts, parameter sensitivity scatter plots with LOWESS trendlines, side-by-side run comparison with deltas, and CSV/JSON export of run history and best configurations.

### 6.7 Page 6: Rail Manager

Operationalises cluster assignments as content rails for production use. Provides an editable rail overview table with status workflow (Draft, Active, Under Review, Archived), merge and split tools with sub-cluster promotion, manual film assignment with audit trail, and a production CSV export mapping each film to its rail ID, confidence score, and outlier tier.

### 6.8 Page 7: Audio Coverage (Phase 2 Gate)

Monitors YouTube trailer audio feature pipeline readiness. Requires 80% coverage on both trailer ID and audio extraction metrics before enabling Phase 2 integration. Currently a gate check with placeholder sections for failure diagnostics and manual overrides.

---

## 7. Configuration Reference

All pipeline parameters are defined in Cell 2 of the notebook and can be modified before execution.

| Parameter | Default | Description |
|-----------|---------|-------------|
| RUN_IDF_SWEEP | False | Run IDF transform sweep (7 configs) |
| RUN_UMAP_SWEEP | False | Run UMAP parameter sweep (216 combos) |
| RUN_HDBSCAN_SWEEP | False | Run HDBSCAN parameter sweep (350 combos) |
| INCLUDE_GENRE_FEATURES | True | Append ~23 one-hot genre columns |
| INCLUDE_DECADE_FEATURE | True | Append ~10 one-hot decade columns |
| IDF_TRANSFORM | raw | IDF mode: raw, sublinear, or bm25 |
| BM25_K1 | 1.2 | BM25 saturation parameter (only if bm25) |
| UMAP_N_COMPONENTS | 30 | Clustering embedding dimensions |
| UMAP_N_NEIGHBORS | 15 | UMAP local neighbourhood size |
| UMAP_MIN_DIST | 0.0 | Minimum distance in embedding |
| UMAP_METRIC | correlation | Distance metric for UMAP |
| HDBSCAN_MIN_CLUSTER_SIZE | 60 | Minimum cluster membership |
| HDBSCAN_MIN_SAMPLES | 15 | Core distance parameter |
| HDBSCAN_EPSILON | 0.1 | Cluster merge distance threshold |
| HDBSCAN_SELECTION_METHOD | eom | Hierarchy extraction: eom or leaf |
| OLLAMA_MODEL | llama3.2:3b | LLM model for cluster naming |

---

## 8. File Manifest

| File | Type | Purpose |
|------|------|---------|
| Project_HOLLYWOOD_IDF.ipynb | Notebook | Primary IDF pipeline |
| app.py | Python | Streamlit dashboard entry point |
| pages/01_data_health.py | Python | Data quality validation page |
| pages/02_pipeline_runner.py | Python | Interactive pipeline execution |
| pages/03_cluster_explorer.py | Python | UMAP viz + cluster management |
| pages/04_film_explorer.py | Python | Per-film deep dive page |
| pages/05_experiment_tracker.py | Python | Run comparison + sensitivity |
| pages/06_rail_manager.py | Python | Production rail management + export |
| pages/07_audio_coverage.py | Python | Phase 2 audio feature gate |
| feature_data_longform.csv | Data | Source genome scores (long format) |
| feature_taxonomy.csv | Data | Genome tag ID-to-name mapping |
| omdb_data/omdb_movies.json | Data | OMDB movie metadata |
| omdb_data/jordan_metadata.csv | Data | Decade + title metadata |
| results/ | Directory | Pipeline artifacts (joblib pkl) |
| results/runs/ | Directory | Saved cluster explorer snapshots |
| pipeline_diagram.html | HTML | Visual pipeline flow diagram |
