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
OUTPUT_CLUSTERS_FILE = os.path.join(BASE_DIR, 'movie_clusters.csv')
OUTPUT_PLOT_FILE = os.path.join(BASE_DIR, 'pca_clusters.png')

def load_data():
    print("Loading data...")
    features_df = pd.read_csv(FEATURE_FILE)
    taxonomy_df = pd.read_csv(TAXONOMY_FILE)
    genre_df = pd.read_csv(GENRE_FILE)
    
    # Check for duplicates in genre_data to avoid exploding the merge
    if genre_df['movie_id'].duplicated().any():
        print("Warning: Duplicate movie_ids found in genre_data. Dropping duplicates.")
        genre_df = genre_df.drop_duplicates(subset=['movie_id'])
        
    return features_df, taxonomy_df, genre_df

def pivot_data(features_df, taxonomy_df):
    print("Pivoting data...")
    # Pivot so that index is imdb_id, columns are feature_id, values are trigger
    # Using 'max' to handle cases where there might be duplicate entries (though there shouldn't be based on inspection)
    pivoted_df = features_df.pivot_table(index='imdb_id', columns='feature_id', values='trigger', fill_value=0, aggfunc='max')
    
    # Map feature_ids to feature names for better column readability
    feature_map = dict(zip(taxonomy_df['feature_id'], taxonomy_df['feature']))
    pivoted_df.columns = [feature_map.get(col, f'feature_{col}') for col in pivoted_df.columns]
    
    return pivoted_df

def preprocess_data(pivoted_df):
    print("Standardizing data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivoted_df)
    return scaled_data, scaler

def run_pca(scaled_data, n_components=2):
    print(f"Running PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.2f}")
    
    return pca_data, pca

def run_kmeans(pca_data, n_clusters=5):
    print(f"Running K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_data)
    return clusters, kmeans

def visualize_clusters(pca_data, clusters, output_path):
    print("Generating visualization...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis', s=50, alpha=0.7)
    plt.title('Movie Clusters Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def save_results(pivoted_df, clusters, genre_df, output_path):
    print("Saving results...")
    results_df = pd.DataFrame(index=pivoted_df.index)
    results_df['cluster'] = clusters
    
    # Merge with genre data to get movie names
    results_df = results_df.merge(genre_df[['movie_id', 'movie_name', 'genre']], left_index=True, right_on='movie_id', how='left')
    
    # Reorder columns
    cols = ['movie_id', 'movie_name', 'cluster', 'genre']
    results_df = results_df[cols]
    
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    try:
        features_df, taxonomy_df, genre_df = load_data()
        
        # Checking pivot size estimation
        n_movies = features_df['imdb_id'].nunique()
        n_features = features_df['feature_id'].nunique()
        print(f"Estimated matrix size: {n_movies} movies x {n_features} features")
        
        pivoted_df = pivot_data(features_df, taxonomy_df)
        print(f"Actual pivoted shape: {pivoted_df.shape}")
        
        scaled_data, _ = preprocess_data(pivoted_df)
        
        pca_data, _ = run_pca(scaled_data, n_components=2)
        
        clusters, _ = run_kmeans(pca_data, n_clusters=5)
        
        visualize_clusters(pca_data, clusters, OUTPUT_PLOT_FILE)
        
        save_results(pivoted_df, clusters, genre_df, OUTPUT_CLUSTERS_FILE)
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
