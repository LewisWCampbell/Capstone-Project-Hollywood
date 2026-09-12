# Current Active Pipeline Configuration

*Generated from Cell 6 values as of this session.*

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   feature_data_longform.csv                                                      ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  ~8,000 movies × 247 genome features          │                              ║
║   │  Ordinal values: 0, 1, 2, 3                   │                              ║
║   │  ~65% sparse (zeros)                           │                              ║
║   └──────────────────────┬───────────────────────┘                               ║
║                          │                                                       ║
║          Genre features: OFF       Jordan metadata: OFF                          ║
║          Ensemble distance: OFF    Phase 2 audio: OFF                            ║
║          Language filter: OFF      Rating filter: OFF                            ║
║                          │                                                       ║
║                          │  (247 features only, nothing appended)                ║
║                          │                                                       ║
║                          ▼                                                       ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  MinMaxScale [0, 1]                           │                              ║
║   │                                               │                              ║
║   │  0 → 0.00                                     │                              ║
║   │  1 → 0.33                                     │                              ║
║   │  2 → 0.67                                     │                              ║
║   │  3 → 1.00                                     │                              ║
║   │                                               │                              ║
║   │  + drop constant columns                      │                              ║
║   │  + impute missing → median                    │                              ║
║   └──────────────────────┬───────────────────────┘                               ║
║                          │                                                       ║
║                          ▼                                                       ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  IDF + Sublinear Transform                    │                              ║
║   │                                               │                              ║
║   │  idf(f) = log(8000 / (presence(f) + 1))       │                              ║
║   │                                               │                              ║
║   │  For each movie-feature value:                │                              ║
║   │    score = 0  →  0                            │                              ║
║   │    score > 0  →  (1 + log(score)) × idf(f)    │                              ║
║   │                                               │                              ║
║   │  Variance weight: OFF                         │                              ║
║   │                                               │                              ║
║   │  Output: X_weighted (8,000 × 247)             │                              ║
║   └──────────────────────┬───────────────────────┘                               ║
║                          │                                                       ║
║                          ▼                                                       ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  UMAP #1 — Dimensionality Reduction           │                              ║
║   │                                               │                              ║
║   │  247D ──────────────→ 15D                     │                              ║
║   │                                               │                              ║
║   │  n_components:  15                            │                              ║
║   │  n_neighbors:   15                            │                              ║
║   │  min_dist:      0.0                           │                              ║
║   │  metric:        correlation (auto from IDF)   │                              ║
║   │                                               │                              ║
║   │  Also fits separately:                        │                              ║
║   │    3D embedding (n_neighbors=15, dist=0.1)    │                              ║
║   │    2D embedding (for flat plots)              │                              ║
║   │                                               │                              ║
║   │  Output: embedding_cluster (8,000 × 15)       │                              ║
║   └──────────────────────┬───────────────────────┘                               ║
║                          │                                                       ║
║                          ▼                                                       ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  HDBSCAN #1 — Primary Clustering              │                              ║
║   │                                               │                              ║
║   │  Input: 8,000 × 15                            │                              ║
║   │                                               │                              ║
║   │  min_cluster_size:    60                      │                              ║
║   │  min_samples:         15                      │                              ║
║   │  cluster_selection_ε: 0.1                     │                              ║
║   │  selection_method:    eom                     │                              ║
║   │                                               │                              ║
║   │  Output:                                      │                              ║
║   │    cluster_labels   (rail IDs, -1 = outlier)  │                              ║
║   │    probabilities    (0.0 – 1.0 per movie)     │                              ║
║   │    soft_clusters    (membership matrix)        │                              ║
║   │    outlier_scores   → 3-tier classification    │                              ║
║   │      Tier 1 (mild):     < 60th percentile     │                              ║
║   │      Tier 2 (moderate): 60th – 85th           │                              ║
║   │      Tier 3 (true):     ≥ 85th percentile     │                              ║
║   │                                               │                              ║
║   │  Secondary rail: 2nd highest if > 0.15        │                              ║
║   └──────────────────────┬───────────────────────┘                               ║
║                          │                                                       ║
║                          │  For each cluster with ≥ 30 movies:                   ║
║                          ▼                                                       ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  SUB-CLUSTERING — ENABLED                     │                              ║
║   │                                               │                              ║
║   │  Source: X_weighted[cluster_slice]             │                              ║
║   │    (back to raw 247D, not the 15D embedding)  │                              ║
║   │                                               │                              ║
║   │         │                                     │                              ║
║   │         ▼                                     │                              ║
║   │  UMAP #2:  247D → 20D                        │                              ║
║   │    n_components: 20                           │                              ║
║   │    n_neighbors:  15                           │                              ║
║   │    min_dist:     0.0                          │                              ║
║   │    metric:       euclidean                    │                              ║
║   │         │                                     │                              ║
║   │         ▼                                     │                              ║
║   │  HDBSCAN #2:                                  │                              ║
║   │    min_cluster_size: 15                       │                              ║
║   │    min_samples:      10                       │                              ║
║   │         │                                     │                              ║
║   │         ▼                                     │                              ║
║   │  sub_labels + silhouette scores               │                              ║
║   └──────────────────────┬───────────────────────┘                               ║
║                          │                                                       ║
║                          ▼                                                       ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  LLM Naming — Ollama llama3.2:3b              │                              ║
║   │                                               │                              ║
║   │  Per cluster:  top elevated features           │                              ║
║   │                + dedup vs other clusters        │                              ║
║   │                → name + description             │                              ║
║   │                                               │                              ║
║   │  Per sub-cluster: features vs parent           │                              ║
║   │                   + dedup vs siblings           │                              ║
║   │                   → name + description          │                              ║
║   └──────────────────────┬───────────────────────┘                               ║
║                          │                                                       ║
║                          ▼                                                       ║
║   ┌──────────────────────────────────────────────┐                               ║
║   │  Export → pipeline_artifacts.pkl               │                              ║
║   │        → Streamlit Dashboard                   │                              ║
║   └──────────────────────────────────────────────┘                               ║
║                                                                                  ║
║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ║
║                                                                                  ║
║   SWEEPS:   UMAP ✓    HDBSCAN ✓    K-Means ✗    Agglomerative ✗                ║
║                                                                                  ║
║   DISABLED: Genre features, Jordan metadata, Ensemble distance,                  ║
║             Phase 2 audio, Variance weight, Language/Rating filters              ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

## One-Line Summary

```
247 genomes → MinMaxScale → IDF sublinear → UMAP (→15D, correlation) → HDBSCAN (min_size=60) → per-cluster: UMAP (→20D, euclidean) → HDBSCAN (min_size=15) → LLM naming
```
