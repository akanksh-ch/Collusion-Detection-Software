import logging
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class StructuralEncoder:
    """Processes multi-relational graph metrics and binds node textual distributions into unique vectors."""
    def __init__(self, node_vocab_size, edge_vocab_size):
        self.node_vocab_size = node_vocab_size
        self.edge_vocab_size = edge_vocab_size
        # Track if semantic text needs to fail gracefully due to sparse inputs
        self.tfidf_disabled = False
        
        # Filter out layout labels from the token matrix to preserve semantic logic text
        schema_noise = ['method', 'call', 'identifier', 'local', 'literal', 
                        'block', 'control_structure', 'return', 'param', 'expression', 'unknown']
        
        # FIX 1: Change token_pattern from default (\w\w+) to (\w+) to capture single-character variables (i, j, x)
        self.tfidf = TfidfVectorizer(
            analyzer='word', 
            stop_words=schema_noise, 
            ngram_range=(1, 2), 
            token_pattern=r"(?u)\b\w+\b",
            max_features=128
        )

    def fit_cohort_text(self, corpus_list):
        """Fits the global TF-IDF spatial vocabulary context across the entire student text corpus."""
        if not corpus_list:
            self.tfidf_disabled = True
            return

        try:
            self.tfidf.fit(corpus_list)
        except ValueError as e:
            # FIX 2: Intercept empty vocabulary errors if submissions contain zero unique semantic tokens
            if "empty vocabulary" in str(e):
                logging.warning("[TF-IDF ALERT] Code corpus contains only boilerplate schema labels. Defensively falling back to structural topology layers only.")
                self.tfidf_disabled = True
            else:
                raise e

    def encode_submission(self, pyg_data):
        """Transforms relational properties, data-flow configurations, and code semantics into a distinct 1D vector."""
        # 1. Node Type Distribution Histogram
        if hasattr(pyg_data, 'x') and pyg_data.x is not None:
            node_types = pyg_data.x.view(-1).long()
            node_counts = torch.bincount(node_types, minlength=self.node_vocab_size).float()
            total_nodes = node_types.size(0)
            node_density = node_counts / total_nodes if total_nodes > 0 else node_counts
        else:
            total_nodes = 0
            node_density = torch.zeros(self.node_vocab_size)

        # 2. Multi-Relational Edge Distribution Profile (AST vs CFG vs REACHING_DEF)
        edge_density = torch.zeros(self.edge_vocab_size)
        if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
            edge_types = pyg_data.edge_attr.view(-1).long()
            if edge_types.numel() > 0:
                edge_counts = torch.bincount(edge_types, minlength=self.edge_vocab_size).float()
                edge_density = edge_counts / edge_types.size(0)

        # 3. Structural Macro Densities
        num_edges = pyg_data.edge_index.size(1) if pyg_data.edge_index is not None else 0
        connectivity_ratio = torch.tensor([num_edges / total_nodes if total_nodes > 0 else 0.0])

        # Combine structural components into single baseline array
        structural_features = torch.cat([node_density, edge_density, connectivity_ratio]).numpy()

        # 4. Extract Text Vector Allocations
        text_document = getattr(pyg_data, 'text_document', '')
        
        # FIX 3: Safe-guard transform executions if text modeling was disabled during calibration
        if text_document and hasattr(self.tfidf, 'vocabulary_') and not self.tfidf_disabled:
            try:
                text_features = self.tfidf.transform([text_document]).toarray().flatten()
            except ValueError:
                text_features = np.zeros(self.tfidf.max_features)
        else:
            text_features = np.zeros(self.tfidf.max_features)

        # 5. Monolithic Fingerprint Matrix Row Output
        return np.concatenate([structural_features, text_features])
