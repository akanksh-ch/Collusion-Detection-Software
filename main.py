#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Supervised Geometric Code Collusion Detector
Integrates a Graph Autoencoder (GAE) training layer to eliminate boilerplate chaining leakage.
"""

import os
import sys
import shutil
import glob
import subprocess
import argparse
import time

# Force non-interactive visualization backend to prevent terminal hanging
import matplotlib
matplotlib.use('Agg')

import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch_geometric.utils as pyg_utils

# Import native optimized geometric clustering utilities
from torch_cluster import radius_graph, knn_graph

# =====================================================================
# 1. STRUCTURAL GNN ENCODER DEFINITION
# =====================================================================
class StructuralNodeEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x

# =====================================================================
# 2. DATA UTILITIES & COMPILATION FRONTEND
# =====================================================================
def generate_cpg_graphml(source_code_path, output_graph_path, tmp_dir="tmp_joern"):
    os.makedirs(tmp_dir, exist_ok=True)
    cpg_bin = os.path.join(tmp_dir, "cpg.bin")
    export_dir = os.path.join(tmp_dir, "export_out")
    
    if os.path.exists(cpg_bin): os.remove(cpg_bin)
    if os.path.exists(export_dir): shutil.rmtree(export_dir)
    
    binary_cmd = "javasrc2cpg" if shutil.which("javasrc2cpg") else "/opt/joern/joern-cli/javasrc2cpg"
    export_cmd = "joern-export" if shutil.which("joern-export") else "/opt/joern/joern-cli/joern-export"

    try:
        subprocess.run([binary_cmd, "-J-Xmx3254m", source_code_path, "-o", cpg_bin], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([export_cmd, "--repr=all", "--format=graphml", f"--output={export_dir}", cpg_bin], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        generated_graphs = glob.glob(os.path.join(export_dir, "*.graphml")) + glob.glob(os.path.join(export_dir, "*.xml"))
        if not generated_graphs:
            return False
            
        os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)
        shutil.copy(generated_graphs[0], output_graph_path)
        return True
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

def map_graphml_to_pyg(graphml_path, vocab):
    if not os.path.exists(graphml_path): return None, []
    G = nx.read_graphml(graphml_path)
    if G.number_of_nodes() == 0: return None, []
    
    node_list = list(G.nodes())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    num_features = len(vocab) + 1
    feature_vectors, node_metadata = [], []
    
    for node_id in node_list:
        node_data = G.nodes[node_id]
        node_type = node_data.get('label', node_data.get('_label', 'UNKNOWN'))
        feat = [0.0] * num_features
        if node_type in vocab: feat[vocab[node_type]] = 1.0
        else: feat[-1] = 1.0
        feature_vectors.append(feat)
        node_metadata.append({'node_id': node_id, 'type': node_type, 'student_source': ''})
        
    x = torch.tensor(feature_vectors, dtype=torch.float)
    edge_list = [[node_to_idx[src], node_to_idx[dst]] for src, dst in G.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
    
    pyg_graph = Data(x=x, edge_index=edge_index)
    updated_edges, _ = pyg_utils.add_self_loops(pyg_graph.edge_index, num_nodes=G.number_of_nodes())
    pyg_graph.edge_index = updated_edges
    return pyg_graph, node_metadata

# =====================================================================
# 3. SELF-SUPERVISED GAE RECONSTRUCTION LOSS FUNCTION
# =====================================================================
def compute_gae_loss(z, edge_index):
    """Computes link reconstruction probabilities over positive and sampled negative edges."""
    pos_edge_index = edge_index
    neg_edge_index = pyg_utils.negative_sampling(edge_index, num_nodes=z.size(0))
    
    pos_loss = -torch.log(torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)) + 1e-15).mean()
    neg_loss = -torch.log(1.0 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)) + 1e-15).mean()
    return pos_loss + neg_loss

# =====================================================================
# 4. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Run self-supervised pipeline to isolate collusion rings.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to extracted dataset directory.")
    parser.add_argument('--out_dir', type=str, default="./processed_graphs", help="Directory to save GraphML outputs.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = {'METHOD': 0, 'CALL': 1, 'CONTROL_STRUCTURE': 2, 'IDENTIFIER': 3, 'LOCAL': 4, 'BLOCK': 5, 'LITERAL': 6, 'RETURN': 7}
    
    source_files = glob.glob(os.path.join(args.dataset_dir, "**", "*.java"), recursive=True)
    if not source_files:
        print(f"[-] Error: No source files found in '{args.dataset_dir}'")
        sys.exit(1)

    print(f"[+] Target Device Configured: {device}")
    print(f"[+] Discovered {len(source_files)} source assets. Initializing ingestion phase...")
    
    loaded_graphs, global_ground_truth, global_metadata = [], [], []
    processed_count = 0
    
    for index, file_path in enumerate(source_files):
        normalized_path = os.path.normpath(file_path)
        path_segments = normalized_path.split(os.sep)
        student_id = f"student_{index:03d}_{os.path.basename(file_path)}"
        
        if 'non-plagiarized' in path_segments: ground_truth_label = 0
        elif 'plagiarized' in path_segments or 'original' in path_segments: ground_truth_label = 1
        else: continue

        graph_out = os.path.join(args.out_dir, f"{student_id}.graphml")
        
        # Ingestion Caching Engine
        if os.path.exists(graph_out) and os.path.getsize(graph_out) > 0:
            success = True
        else:
            print(f"    [~] Cache Missing -> Ingesting: {student_id}")
            success = generate_cpg_graphml(file_path, graph_out)
            
        if not success: continue
        
        pyg_graph, meta = map_graphml_to_pyg(graph_out, vocab)
        if pyg_graph is None: continue
        
        for m in meta:
            m['student_source'] = student_id
            global_metadata.append(m)
            
        global_ground_truth.extend([ground_truth_label] * len(meta))
        loaded_graphs.append(pyg_graph)
        processed_count += 1

    if processed_count == 0:
        print("[-] Error: Ingestion sequence failed to map records.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # PHASE 2: SELF-SUPERVISED OPTIMIZATION LOOP (GAE Layer)
    # -----------------------------------------------------------------
    print(f"\n" + "="*60)
    print(f"[X] ACTIVATING SELF-SUPERVISED STRUCTURAL LEARNING PHASE")
    print(f"="*60)
    
    encoder = StructuralNodeEncoder(in_channels=len(vocab)+1, hidden_channels=32, out_channels=16).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01, weight_decay=1e-4)
    
    encoder.train()
    epochs = 30
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for graph in loaded_graphs:
            graph = graph.to(device)
            optimizer.zero_grad()
            
            z = encoder(graph.x, graph.edge_index)
            loss = compute_gae_loss(z, graph.edge_index)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"    -> GAE Optimization Epoch {epoch:02d}/{epochs} | Structural Reconstruction Loss: {total_loss / len(loaded_graphs):.4f}")

    # Extract learned optimized representations
    encoder.eval()
    optimized_embeddings = []
    with torch.no_grad():
        for graph in loaded_graphs:
            graph = graph.to(device)
            optimized_embeddings.append(encoder(graph.x, graph.edge_index).cpu())
            
    X_embeddings = torch.cat(optimized_embeddings, dim=0).to(device)
    y_ground_truth = np.array(global_ground_truth)

    # -----------------------------------------------------------------
    # PHASE 3: GEOMETRIC COMPONENT ISOLATION
    # -----------------------------------------------------------------
    print(f"\n[X] RUNNING GEOMETRIC CLUSTERING OVER OPTIMIZED GRAPH SPACE")
    X_normalized = F.normalize(X_embeddings, p=2, dim=-1)

    with torch.no_grad():
        knn_edges = knn_graph(X_normalized, k=4, loop=False)
        distances = torch.norm(X_normalized[knn_edges[0]] - X_normalized[knn_edges[1]], dim=-1)
        dynamic_r = max(0.01, distances.mean().item() - (1.5 * distances.std().item()))
        print(f"    -> Dynamic Radius Bound Calculated: {dynamic_r:.4f}")

    sim_edge_index = radius_graph(X_normalized, r=dynamic_r, loop=False, max_num_neighbors=100)
    
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp

    adj_matrix = to_scipy_sparse_matrix(sim_edge_index.cpu(), num_nodes=X_embeddings.size(0))
    num_rings, ring_assignments = sp.csgraph.connected_components(adj_matrix, connection='weak')
    
    # Prune elements that do not meet strict cluster density bounds
    unique_ids, counts = np.unique(ring_assignments, return_counts=True)
    noise_clusters = unique_ids[counts < 15] # Raised cutoff to ensure tight node communities
    
    final_cluster_labels = np.array([-1 if c in noise_clusters else c for c in ring_assignments])
    active_rings = len(set(final_cluster_labels)) - (1 if -1 in final_cluster_labels else 0)
    print(f"[+] Isolation Engine Complete. Discovered {active_rings} tight collusion structures.")

    # Cross-reference suspects using density filtering thresholds
    for cluster_id in list(set(final_cluster_labels)):
        if cluster_id == -1: continue
        indices = np.where(final_cluster_labels == cluster_id)[0]
        
        # Track node frequency to identify true shared logical structures
        raw_suspects = [global_metadata[idx]['student_source'] for idx in indices]
        unique_suspects, suspect_counts = np.unique(raw_suspects, return_counts=True)
        
        # A student is flagged only if they share a significant structural block (>= 5 nodes)
        verified_suspects = unique_suspects[suspect_counts >= 5]
        
        if len(verified_suspects) > 1:
            print(f"    [!] True Collusion Ring Verified in Component #{cluster_id}!")
            print(f"        -> Target Suspect Array: {list(verified_suspects)}")

    # -----------------------------------------------------------------
    # PHASE 4: GLOBAL UNIFORM PLOT EXPORT
    # -----------------------------------------------------------------
    print("\n[+] Compiling t-SNE coordinate projections...")
    total_nodes = X_embeddings.shape[0]
    max_visual_nodes = min(2500, total_nodes)
    
    np.random.seed(42)
    sampled_indices = np.random.choice(total_nodes, max_visual_nodes, replace=False)
    
    X_to_project = X_embeddings[sampled_indices].cpu().numpy()
    y_plot_ground_truth = y_ground_truth[sampled_indices]
    y_plot_clusters = final_cluster_labels[sampled_indices]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_to_project)
    X_jittered = X_2d + np.random.normal(0, 0.1, X_2d.shape)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
    binary_cmap = ListedColormap(['#3498db', '#e74c3c'])
    
    ax1.scatter(X_jittered[:, 0], X_jittered[:, 1], c=y_plot_ground_truth, cmap=binary_cmap, s=20, alpha=0.6)
    ax1.set_title("1. Ground Truth Folders (Clean vs Colluded)", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2.scatter(X_jittered[:, 0], X_jittered[:, 1], c=y_plot_clusters, cmap='tab20', s=20, alpha=0.6)
    ax2.set_title("2. Discovered Collusion Components (GAE Optimized)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle("Dissertation Verification Canvas: Case-07 Self-Supervised Space Map", fontsize=14, fontweight='bold', y=0.96)
    
    out_img = "collusion_space_mapping.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"[X] Process Terminated. Graph visualization successfully dumped to: '{out_img}'")

if __name__ == '__main__':
    main()