import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sys import argv
from pathlib import Path

def run_analysis(base_dir_str, eps=3.0, min_samples=2):
    base_dir = Path(base_dir_str)
    input_file = base_dir / "processed" / "bags_of_nodes.csv"
    output_dir = base_dir / "plots"
    output_file = output_dir / "dbscan_result.png"

    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        return

    # 1. Load data
    df = pd.read_csv(input_file)
    features = df.select_dtypes(include=[np.number])
    
    # Remove columns with zero variance (they provide no information)
    features = features.loc[:, features.std() > 0]
    
    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # 3. PCA for Visualization
    # Compresses high-dimensional node counts into a 2D map
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Fit DBSCAN with a higher epsilon
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    
    # 5. Visualization
    plt.figure(figsize=(12, 8))
    unique_labels = set(labels)
    cmap = plt.get_cmap('Spectral')
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1: 
            col = [0, 0, 0, 1]  # Black for noise
        
        mask = (labels == k)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    c=[col], label=f'Cluster {k}', edgecolors='k', s=100)

    # Annotate points with their category name (e.g., 'recursive')
    for i, file_id in enumerate(df['file_id']):
        # Extract category name from path
        category = Path(file_id).parent.name if '/' in file_id else file_id
        plt.annotate(category, (X_pca[i, 0], X_pca[i, 1]), 
                     fontsize=9, alpha=0.7, xytext=(5, 5), 
                     textcoords='offset points')

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f'DBSCAN Clustering (eps={eps}) - Structural Similarity Map')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close()
    print(f"Analysis complete. Found {len(unique_labels) - (1 if -1 in labels else 0)} clusters.")
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python dbscan.py <base_directory>")
    else:
        run_analysis(argv[1])