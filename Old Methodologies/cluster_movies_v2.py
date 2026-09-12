import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
BASE_DIR = '/Users/lewiswilliamcampbell/Desktop/Project 1 ANTI'
FEATURE_FILE = os.path.join(BASE_DIR, 'feature_data_longform.csv')
TAXONOMY_FILE = os.path.join(BASE_DIR, 'feature_taxonomy.csv')
GENRE_FILE = os.path.join(BASE_DIR, 'genre_data.csv')
OUTPUT_CLUSTERS_FILE = os.path.join(BASE_DIR, 'movie_clusters_v2.csv')
OUTPUT_PLOT_FILE = os.path.join(BASE_DIR, 'pca_clusters_v2.png')

def main():
    try:
        print("Loading data...")
        features_df = pd.read_csv(FEATURE_FILE)
        taxonomy_df = pd.read_csv(TAXONOMY_FILE)
        genre_df = pd.read_csv(GENRE_FILE)
        
        # Check for duplicates in genre_data
        if genre_df['movie_id'].duplicated().any():
            print("Warning: Duplicate movie_ids found in genre_data. Dropping duplicates.")
            genre_df = genre_df.drop_duplicates(subset=['movie_id'])

        print("Pivoting data...")
        pivoted_df = features_df.pivot_table(index='imdb_id', columns='feature_id', values='trigger', fill_value=0, aggfunc='max')
        
        feature_map = dict(zip(taxonomy_df['feature_id'], taxonomy_df['feature']))
        pivoted_df.columns = [feature_map.get(col, f'feature_{col}') for col in pivoted_df.columns]
        
        print("Standardizing data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivoted_df)
        
        print("Running PCA...")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        print(f"Explained Variance: {pca.explained_variance_ratio_}")

        print("Running K-Means...")
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pca_data)
        
        print("Generating visualization...")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis', s=50, alpha=0.7)
        plt.title('Movie Clusters Visualization (PCA)')
        plt.savefig(OUTPUT_PLOT_FILE)
        print(f"Plot saved to {OUTPUT_PLOT_FILE}")
        
        print("Saving results...")
        results_df = pd.DataFrame(index=pivoted_df.index)
        results_df['cluster'] = clusters
        results_df = results_df.merge(genre_df[['movie_id', 'movie_name', 'genre']], left_index=True, right_on='movie_id', how='left')
        results_df.to_csv(OUTPUT_CLUSTERS_FILE, index=False)
        print(f"Results saved to {OUTPUT_CLUSTERS_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
