import os
import logging
import networkx as nx
import torch
from torch_geometric.data import Data

class CPGDataLoader:
    """Handles loading flattened GraphML exports, dynamically discovering graph schemas, and translating them to PyG Data primitives."""
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.node_vocab = {}
        self.edge_vocab = {}

    def discover_vocabularies(self):
        """First Pass: Scans all GraphML files to dynamically discover every unique node and edge label present in the dataset."""
        node_labels = set()
        edge_labels = set()
        
        if not os.path.exists(self.input_dir) or len(os.listdir(self.input_dir)) == 0:
            return

        for item in os.listdir(self.input_dir):
            if item.endswith(".graphml"):
                file_path = os.path.join(self.input_dir, item)
                try:
                    G = nx.read_graphml(file_path)
                    for _, attrs in G.nodes(data=True):
                        if 'label' in attrs:
                            node_labels.add(attrs['label'].upper())
                    for _, _, attrs in G.edges(data=True):
                        if 'label' in attrs:
                            edge_labels.add(attrs['label'].upper())
                except Exception as e:
                    logging.warning(f"Vocab discovery pass skipped file {item}: {e}")

        # Build dynamic, continuous integer mapping indexes (leaving 0 for unexpected fallbacks)
        self.node_vocab = {label: idx + 1 for idx, label in enumerate(sorted(node_labels))}
        self.node_vocab['UNKNOWN'] = 0
        
        self.edge_vocab = {label: idx + 1 for idx, label in enumerate(sorted(edge_labels))}
        self.edge_vocab['UNKNOWN'] = 0
        
        logging.info(f"[SCHEMA DISCOVERY] Dynamically mapped {len(self.node_vocab)} node classes and {len(self.edge_vocab)} multi-relational edge types across the cohort.")

    def graphml_to_pyg(self, file_path):
        """Parses a multi-relational Joern GraphML file, extracting all discovered relational and textual attributes."""
        try:
            G = nx.read_graphml(file_path)
            
            # 1. Map Node Attributes against Discovered Vocab
            node_features = []
            node_text_corpus = []
            
            for node, attrs in G.nodes(data=True):
                label = attrs.get('label', 'UNKNOWN').upper()
                node_features.append([self.node_vocab.get(label, 0)])
                
                # Semantic Text Aggregation
                node_name = attrs.get('name', '')
                node_code = attrs.get('code', '')
                combined_text = f"{label} {node_name} {node_code}".strip()
                node_text_corpus.append(combined_text)
            
            # 2. Map Edge Attributes against Discovered Vocab
            edge_features = []
            edges = []
            
            for u, v, attrs in G.edges(data=True):
                edges.append((u, v))
                edge_label = attrs.get('label', 'UNKNOWN').upper()
                edge_features.append(self.edge_vocab.get(edge_label, 0))

            G_int = nx.convert_node_labels_to_integers(G)
            int_edges = list(G_int.edges())
            
            if len(int_edges) == 0:
                return None
                
            edge_index = torch.tensor(int_edges, dtype=torch.long).t().contiguous()
            x_tensor = torch.tensor(node_features, dtype=torch.float)
            edge_attr_tensor = torch.tensor(edge_features, dtype=torch.long)
            
            pyg_data = Data(x=x_tensor, edge_index=edge_index)
            pyg_data.edge_attr = edge_attr_tensor
            pyg_data.text_document = " ".join(node_text_corpus)
            
            return pyg_data
            
        except Exception as e:
            logging.error(f"Failed to parse deep graph structures for {file_path}: {str(e)}")
            return None

    def load_cohort(self, executed_successfully=True):
        """Scans the Joern output folder for single GraphML files to build the PyG cohort dataset."""
        cohort_data = {}
        if not executed_successfully or not os.path.exists(self.input_dir) or len(os.listdir(self.input_dir)) == 0:
            logging.warning("No real GraphML assets found. Pipeline defaulting to mock routines.")
            return self._generate_dummy_cohort()

        for item in os.listdir(self.input_dir):
            if item.endswith(".graphml"):
                student_id = item.replace(".graphml", "")
                file_path = os.path.join(self.input_dir, item)
                pyg_graph = self.graphml_to_pyg(file_path)
                if pyg_graph:
                    cohort_data[student_id] = pyg_graph
                    
        return cohort_data

    def _generate_dummy_cohort(self, num_students=5):
        cohort_data = {}
        for i in range(1, num_students + 1):
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            mock_data = Data(x=torch.ones((3, 1)), edge_index=edge_index)
            mock_data.edge_attr = torch.tensor([1, 1, 1], dtype=torch.long)
            mock_data.text_document = "METHOD func main() CALL println System.out.println"
            cohort_data[f"Student_{i:02d}_Mock"] = mock_data
        return cohort_data
