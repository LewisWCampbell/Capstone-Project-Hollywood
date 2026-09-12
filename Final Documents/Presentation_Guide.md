# Project HOLLYWOOD — Presentation Guide

*30-minute live presentation + Q&A — pitched to a streaming product team (product managers, a data scientist, and an executive)*

---

## 1. Problem Framing

### What is a "content rail"?

A content rail is a horizontal, scrollable row of movies on a streaming platform's homepage — each rail is built around a coherent theme that helps users discover content they wouldn't have found through simple genre browsing. Think of the rows on Netflix or Hulu: "Mind-Bending Sci-Fi," "Gritty Crime Dramas," "Feel-Good Family Adventures." Each rail is an editorial promise — click anything here and you'll get *this kind of experience*.

### What do our groups represent?

Our clusters are groups of movies that *feel* alike across hundreds of dimensions — not just genre, but narrative tone, visual style, thematic complexity, pacing, target audience, and era. Two movies can both be "Action" but land in completely different rails because one is a gritty, dialogue-heavy political thriller and the other is a colorful superhero spectacle. The genome data captures these nuances at a resolution that genre labels alone cannot.

### What are we giving the customer?

Discovery beyond the obvious. A user who just watched a quiet, emotionally complex family drama gets a rail of *more films that feel like that* — not just "Drama" (which could include courtroom thrillers, war epics, or teen romances). The rails are fine-grained enough to match mood and taste, not just category.

### Conceptual model (before methods)

Our mental model is that every movie lives in a high-dimensional "taste space" defined by ~283 measurable traits. Movies that cluster together in this space share a recognizable vibe — the kind of thing a human curator would group into a shelf at a video store. Our job is to find those natural groupings algorithmically, name them meaningfully, and validate that they'd actually make sense to a user scrolling a homepage.

---

## 2. Data & Preprocessing (~5 min)

### Input data

- **~8,000 movies** from the MovieLens Genome dataset
- **248 genome features** (the "spine"): continuous relevance scores on a 0–3 scale, covering traits like "visually striking," "dark comedy," "twist ending," "based on a book," etc.
- **~23 genre features** (binary "bumps"): one-hot encoded from OMDB genre tags (Action, Comedy, Horror, etc.)
- **~11 decade features** (binary "bumps"): one-hot encoded from release year decade (1920s–2020s)
- **Total: ~283 features per movie**

### The "Spine and Bumps" model

The 248 genome tags are the *spine* of our feature space — they carry the most information and exist on a continuous 0–3 scale. The genre and decade features are *bumps* — binary 0/1 nudge signals that help catch patterns the genome might miss (e.g., era-specific aesthetics, genre conventions). They're intentionally lightweight so they influence clustering without overpowering the genome's nuance.

### Filtering decisions

- Movies with incomplete genome data were excluded during the longform pivot
- No manual movie exclusions — we let the clustering algorithm identify natural outliers

### IDF Weighting (preprocessing)

**Inverse Document Frequency (IDF)** weighting is applied to the 248 genome features using scikit-learn's `TfidfTransformer`. IDF is borrowed from information retrieval — it upweights features that are rare across the film catalog and downweights features that are common. The formula is:

> idf(t) = log((1 + n) / (1 + df(t))) + 1

A feature like "superhero" that appears at high relevance in many films gets downweighted because it doesn't distinguish between films. A feature like "claymation" that appears in only ~45 films gets upweighted because it's highly discriminative.

**Key detail:** Only the 248 continuous genome features are IDF-weighted. The binary genre and decade columns pass through as raw 0/1 — applying IDF to binary features would just multiply each by a constant, which doesn't change relative distances.

**Why IDF over StandardScaler + TruncatedSVD?** We tested SVD (150 components after StandardScaler) as an alternative. A full 216-combination UMAP parameter sweep showed IDF was clearly superior: trustworthiness **0.987 vs 0.918**, neighborhood overlap **0.569 vs 0.387**. SVD optimizes for global variance, but UMAP needs local neighborhood structure. IDF preserves the full 283-dimensional feature space and reweights by informativeness — giving UMAP the richest possible input.

---

## 3. Methodology (~7–10 min)

### Pipeline overview

```
Raw Data (283 features)
    → IDF Weighting (TfidfTransformer on 248 genome features; genre/decade pass through raw)
    → UMAP (reduce to 20D for clustering, 3D for visualization)
    → HDBSCAN (density-based clustering)
    → XGBoost (cluster validation + outlier recovery)
    → SHAP (feature importance per cluster)
    → LLM Naming (Ollama llama3.2 — SHAP-informed rail names)
```

### UMAP — Manifold Learning

- **What it does:** Projects high-dimensional data onto a lower-dimensional manifold while preserving local neighborhood structure
- **Key hyperparameters:**
  - `n_components = 20` — Trust saturated across 17–25; 20D is the clean round number
  - `n_neighbors = 60` — Balances local vs global structure; refined from 216-combination sweep
  - `min_dist = 0.1` — Best neighborhood overlap; 0.0 gave marginal trust gain but worse overlap
  - `metric = correlation` — Lowest reconstruction error; trust identical across metrics
- **Why UMAP over PCA/t-SNE:** PCA is linear (misses non-linear genre blends), t-SNE doesn't preserve global structure well and can't transform new data. UMAP handles both.
- **Validation:** 216-combo parameter sweep measuring trustworthiness, reconstruction error, and neighborhood overlap

### HDBSCAN — Density-Based Clustering

- **What it does:** Finds clusters of varying density without requiring a pre-specified k
- **Key hyperparameters:**
  - `min_cluster_size = 48` — Sweet spot: DBCV=0.511 (best), silhouette=0.598, mean probability=0.908
  - `min_samples = 10` — Quality inflection point (DBCV +0.09 vs ms=3)
  - `cluster_selection_epsilon = 0.2` — Prevents over-merging of nearby clusters
  - `cluster_selection_method = eom` — Excess of Mass outperformed Leaf across all metrics
- **Why HDBSCAN over K-Means:** K-Means forces spherical clusters and requires choosing k upfront. HDBSCAN discovers natural cluster shapes, handles outliers gracefully (labels them -1 instead of forcing assignment), and adapts to varying cluster densities — critical when some rails are naturally larger than others.
- **Validation:** 350-combo parameter sweep + targeted refinement around the optimal region

### XGBoost — Cluster Validation & Outlier Recovery

- **Cluster validation:** 5-fold cross-validated accuracy measures how "learnable" the clusters are. High accuracy (~90%+) means clusters have clear, consistent boundaries in feature space.
- **Outlier recovery:** After HDBSCAN labels ~7% of movies as outliers, XGBoost predicts which cluster each outlier *should* belong to, with a confidence score. High-confidence outliers can be recovered into rails; low-confidence ones are true edge cases.
- **SHAP feature importance:** For each cluster, SHAP values identify the top genome tags and genre features that define membership — this drives both human interpretation and LLM naming.

### Tradeoffs

- **HDBSCAN vs K-Means:** We sacrifice control over exact cluster count (HDBSCAN finds ~41 clusters organically) but gain natural outlier detection and non-spherical cluster shapes
- **IDF vs SVD:** IDF preserves the full 283-feature space (keeping all information for UMAP), while SVD would compress to ~150 latent components. We empirically validated that IDF produces better UMAP embeddings — trustworthiness 0.987 vs 0.918, neighborhood overlap 0.569 vs 0.387
- **Correlation metric:** Captures relative feature profiles (shape of the feature vector) rather than absolute magnitudes — better for "this movie *feels like* that movie" comparisons, but less intuitive than Euclidean distance

---

## 4. The Rails Themselves (~10 min)

*For each rail, cover:*

### Template per rail

1. **Rail name** — The LLM-generated name from SHAP features (e.g., "Gritty Urban Crime Dramas")
2. **Conceptual definition** — What ties these movies together in plain language
3. **Top driving variables** — The SHAP feature importance values: which genome tags and genres most strongly predict membership in this cluster (with exact SHAP magnitudes)
4. **Example movies** — 3–5 recognizable titles with their confidence scores. For less recognizable films, include the OMDB/IMDB plot synopsis as evidence
5. **Why this rail deserves homepage space** — The audience it serves, the discovery value it provides, how it differs from adjacent rails

### Presentation tips

- Lead with the most distinctive, easiest-to-explain rails first
- Show poster grids from the Dashboard's Cluster Explorer for visual impact
- Have the Film Explorer ready to pull up individual movies as evidence
- Prepare 2–3 rails that are *surprising* — groups you wouldn't have predicted but make intuitive sense once you see the members

---

## 5. Business Evaluation (~5 min)

### Rail overlap

- HDBSCAN produces *hard* cluster assignments — each movie belongs to exactly one rail (no overlap)
- This is a deliberate design choice for a streaming homepage: you don't want the same movie appearing in 5 different rows, which creates a "padded" feel
- The tradeoff: some movies genuinely fit multiple rails. Our approach assigns them to their *best* fit based on density, and the confidence score tells you how borderline they are

### Rail size distribution

- Show the Named Rails Overview visualization from the notebook (horizontal bar chart by count, colored by confidence)
- Discuss whether the distribution is healthy: are there a few giant rails and many tiny ones, or is it relatively balanced?
- Very small rails (<20 movies) may not provide enough content for a homepage row
- Very large rails (>500 movies) may be too broad to feel curated

### New movie integration *(critical — do not skip)*

- When a new movie enters the library, it comes with genome scores from MovieLens and genre/decade metadata
- **IDF weights** are already fitted — the saved `TfidfTransformer` can transform the new movie's genome features using the learned weights, then concatenate the raw genre/decade columns
- **XGBoost** can predict which rail the movie belongs to *and* provide a confidence score — no need to re-run UMAP or HDBSCAN
- High-confidence predictions get auto-assigned; low-confidence ones get flagged for human review (the Outlier Review dashboard page supports this workflow)
- **Periodic re-clustering** should happen as the library grows significantly (e.g., quarterly), since new movies may reveal rail structures that didn't exist in the original 8,000

### Strengths

- Fine-grained rails that go beyond genre (captures tone, era, narrative style)
- Outlier detection prevents forcing poor-fit movies into rails
- XGBoost provides a production-ready classifier for new movie assignment
- SHAP makes every rail interpretable — you can explain *why* each movie is there
- Dashboard provides full visual exploration and curation tools

### Weaknesses / Sacrifices

- **No overlap:** Some movies genuinely span multiple rails; hard assignment loses this nuance
- **Genome data dependency:** Rail quality is bounded by genome tag quality — if a trait isn't in the 248 tags, we can't cluster on it
- **Outlier count:** ~7% of movies don't fit any rail confidently. This is honest but means some content has no home
- **Static rails:** The rail structure is fixed until re-clustering. If audience tastes shift, the rails may feel stale
- **No user personalization:** These are editorial rails, not personalized recommendations. Every user sees the same rails (though which rails appear on *their* homepage could be personalized separately)

---

## Q&A Preparation

### Have ready to pull up

- [ ] IDF weight distribution histogram (which features got upweighted/downweighted)
- [ ] UMAP parameter sweep results (trustworthiness, nn_overlap, reconstruction error)
- [ ] HDBSCAN parameter sweep results (DBCV, silhouette, cluster count vs min_cluster_size)
- [ ] XGBoost 5-fold CV accuracy
- [ ] SHAP heatmap (top 30 features × clusters)
- [ ] Cluster Explorer dashboard — poster grids for any rail
- [ ] Film Explorer — individual movie detail + KNN visual rail
- [ ] Outlier Review page — recovery suggestions and confidence scores
- [ ] IDF vs SVD comparison metrics (trustworthiness, neighborhood overlap)

### Likely data scientist questions

- "Why IDF instead of StandardScaler?" → IDF selectively reweights by informativeness (rare features get amplified), while StandardScaler uniformly normalizes everything to zero mean / unit variance. IDF preserves the natural scale differences between genome features and binary bumps, and empirically produced better UMAP embeddings.
- "Why not TruncatedSVD before UMAP?" → We tested it. SVD (150 components after StandardScaler) dropped trustworthiness from 0.987 to 0.918 and neighborhood overlap from 0.569 to 0.387 across a 216-combination UMAP sweep. SVD optimizes for global variance, but UMAP needs local neighborhood signals — IDF preserves the full feature space.
- "Why apply IDF only to genome features?" → The binary genre/decade features are already 0/1. IDF on a binary column just multiplies by a constant, which doesn't change relative distances. Only the continuous genome features benefit from informativeness-based reweighting.
- "Why correlation distance for UMAP?" → We care about the *shape* of feature profiles, not absolute magnitudes. Two movies with identical trait *patterns* at different scales should be neighbors
- "How stable are these clusters?" → XGBoost CV accuracy measures learnability (>90% = very stable boundaries). HDBSCAN's probability scores give per-movie membership confidence

### Likely product/executive questions

- "Can we rename the rails?" → Yes, the Dashboard supports manual renaming per cluster
- "What if we want fewer/more rails?" → Adjust `min_cluster_size` — higher = fewer, larger rails; lower = more, finer-grained rails. We can show the parameter sweep tradeoff curve
- "How do we handle new releases?" → XGBoost classifier + confidence threshold + human review dashboard (already built)
- "Is this better than just using genre labels?" → Show examples of movies in the same genre that land in different rails — the genome captures nuance genre can't
- "What does this cost to maintain?" → Re-clustering is a one-time compute job (minutes on a laptop). Day-to-day, XGBoost prediction for new movies is near-instant
