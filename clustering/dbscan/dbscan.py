import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sys import argv
from pathlib import Path

def run_analysis(base_dir_str, eps=0.5, min_samples=2):
    # Use Path for robust directory handling
    base_dir = Path(base_dir_str)
    input_file = base_dir / "processed" / "bags_of_nodes.csv"
    
    # We will save the plot in a 'plots' folder within your output directory
    output_dir = base_dir / "plots"
    output_file = output_dir / "dbscan_result.png"

    # 1. Load the standardized dataframe
    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        return

    df = pd.read_csv(input_file)
    
    # Drop non-numeric columns (like filenames/IDs) for clustering
    features = df.select_dtypes(include=[np.number])
    
    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # 3. Fit DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    
    # Fix: Use plt.get_cmap instead of the deprecated plt.cm.get_cmap
    cmap = plt.get_cmap('Spectral')
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1: 
            col = [0, 0, 0, 1]  # Black for noise
        
        class_member_mask = (labels == k)
        # Using first two features for a basic 2D plot
        plt.scatter(X_scaled[class_member_mask, 0], 
                    X_scaled[class_member_mask, 1], 
                    c=[col], label=f'Cluster {k}', edgecolors='k')

    plt.title(f'DBSCAN Results: {input_file.name}')
    plt.legend()
    
    # Ensure the output directory exists before saving
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file)
    plt.close()
    print(f"Analysis complete. Plot saved to {output_file}")

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python dbscan.py <base_directory>")
    else:
        run_analysis(argv[1])