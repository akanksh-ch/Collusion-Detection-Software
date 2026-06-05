import logging
import torch
import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer

class StructuralEncoder:
    """Processes structural graph metrics using a Weisfeiler-Lehman (WL) neighborhood 
    isomorphism fingerprinting engine combined with character-level textual style profiles."""
    def __init__(self, node_vocab_size, edge_vocab_size, wl_buckets=128, wl_iterations=2):
        self.node_vocab_size = node_vocab_size
        self.edge_vocab_size = edge_vocab_size
        self.wl_buckets = wl_buckets
        self.wl_iterations = wl_iterations
        self.tfidf_disabled = False
        
        # Switched to character-level word-bound n-grams to capture identifier substring patterns,
        # brackets, spacing configurations, and syntax style fingerprints accurately.
        self.tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=256
        )

    def fit_cohort_text(self, corpus_list):
        """Fits the global TF-IDF spatial vocabulary context across the cohort text strings."""
        clean_corpus = [c for c in corpus_list if c.strip()]
        if not clean_corpus:
            self.tfidf_disabled = True
            return
        try:
            self.tfidf.fit(clean_corpus)
        except ValueError:
            logging.warning("[TF-IDF ALERT] Slices are textually empty. Falling back to topology layers only.")
            self.tfidf_disabled = True

    def compute_wl_topological_fingerprint(self, pyg_data):
        """Implements a color-refinement graph isomorphism mapping trick to isolate sub-tree patterns."""
        num_nodes = pyg_data.x.size(0) if pyg_data.x is not None else 0
        if num_nodes == 0:
            return np.zeros(self.wl_buckets)

        node_colors = pyg_data.x.view(-1).long().numpy().astype(str)
        edge_index = pyg_data.edge_index.numpy()
        
        edge_attrs = (pyg_data.edge_attr.numpy() if hasattr(pyg_data, 'edge_attr') and 
                      pyg_data.edge_attr is not None else np.zeros(edge_index.shape[1], dtype=int))
        
        wl_histogram = np.zeros(self.wl_buckets)

        for c in node_colors:
            bucket = int(hashlib.md5(c.encode('utf-8')).hexdigest(), 16) % self.wl_buckets
            wl_histogram[bucket] += 1

        for _ in range(self.wl_iterations):
            adjacency_map = {i: [] for i in range(num_nodes)}
            for idx in range(edge_index.shape[1]):
                source, target = edge_index[0, idx], edge_index[1, idx]
                if source < num_nodes and target < num_nodes:
                    adjacency_map[target].append((node_colors[source], str(edge_attrs[idx])))

            new_colors = []
            for node_idx in range(num_nodes):
                current_color = node_colors[node_idx]
                sorted_neighbor_tokens = sorted([f"{nc}_{et}" for nc, et in adjacency_map[node_idx]])
                neighbor_signature = ",".join(sorted_neighbor_tokens)
                combined_signature = f"{current_color}|{neighbor_signature}"
                
                new_color = hashlib.md5(combined_signature.encode('utf-8')).hexdigest()
                new_colors.append(new_color)
                
                bucket = int(new_color, 16) % self.wl_buckets
                wl_histogram[bucket] += 1
                
            node_colors = np.array(new_colors)

        total_elements = np.sum(wl_histogram)
        if total_elements > 0:
            wl_histogram = wl_histogram / total_elements

        return wl_histogram

    def encode_submission(self, pyg_data):
        """Transforms relational configurations, micro-scale absolute properties, and code sub-tokens 
        into an isolated, highly distinctive unit-normalized 1D vector fingerprint."""
        # 1. Compute WL Graph Kernel Structural Histogram
        wl_topological_vector = self.compute_wl_topological_fingerprint(pyg_data)

        # 2. Extract Macro Scale Metrics to mathematically split small vs large assignments
        num_nodes = pyg_data.x.size(0) if pyg_data.x is not None else 0
        num_edges = pyg_data.edge_index.size(1) if pyg_data.edge_index is not None else 0
        
        scale_invariants = np.array([
            np.log1p(num_nodes),
            np.log1p(num_edges),
            (num_edges / num_nodes) if num_nodes > 0 else 0.0
        ])

        structural_features = np.concatenate([wl_topological_vector, scale_invariants])
        struct_norm = np.linalg.norm(structural_features)
        if struct_norm > 0:
            structural_features = structural_features / struct_norm

        # 3. Extract Character-level Lexical Style Arrays
        text_document = getattr(pyg_data, 'text_document', '')
        if text_document and not self.tfidf_disabled and hasattr(self.tfidf, 'vocabulary_'):
            try:
                text_features = self.tfidf.transform([text_document]).toarray().flatten()
            except ValueError:
                text_features = np.zeros(self.tfidf.max_features)
        else:
            text_features = np.zeros(self.tfidf.max_features)

        text_norm = np.linalg.norm(text_features)
        if text_norm > 0:
            text_features = text_features / text_norm

        # Balanced fingerprint vector (50% structural topology, 50% character style tokens)
        return np.concatenate([structural_features, text_features])
