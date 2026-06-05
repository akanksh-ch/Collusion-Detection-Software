import os
import logging
import networkx as nx
import torch
from torch_geometric.data import Data

class CPGDataLoader:
    """Handles loading GraphML exports, executing contextual backward-slicing passes, 
    and translating multi-relational subgraphs into PyG primitives."""
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.node_vocab = {}
        self.edge_vocab = {}

    def discover_vocabularies(self):
        """First Pass: Scans all GraphML files to dynamically discover every unique node and edge label."""
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

        self.node_vocab = {label: idx + 1 for idx, label in enumerate(sorted(node_labels))}
        self.node_vocab['UNKNOWN'] = 0
        
        self.edge_vocab = {label: idx + 1 for idx, label in enumerate(sorted(edge_labels))}
        self.edge_vocab['UNKNOWN'] = 0
        
        logging.info(f"[SCHEMA DISCOVERY] Dynamically mapped {len(self.node_vocab)} node classes and {len(self.edge_vocab)} multi-relational edge types.")

    def execute_backward_slice(self, G):
        """Topological Pass: Traces data/control dependencies backward from terminal outputs 
        and captures immediate 1-hop AST neighbors to preserve architectural context."""
        sinks = [n for n, attrs in G.nodes(data=True) if attrs.get('label', '').upper() in ['RETURN', 'METHOD_RETURN']]
        
        if not sinks:
            sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
            
        if not sinks:
            return G
            
        sliced_nodes = set(sinks)
        queue = list(sinks)
        
        while queue:
            current = queue.pop(0)
            for pred in G.predecessors(current):
                edge_data = G.get_edge_data(pred, current)
                is_valid_dependency = False
                
                for key in edge_data:
                    edge_label = edge_data[key].get('label', '').upper()
                    if edge_label in ['REACHING_DEF', 'CDG', 'CFG', 'DDG']:
                        is_valid_dependency = True
                        break
                        
                if is_valid_dependency and pred not in sliced_nodes:
                    sliced_nodes.add(pred)
                    queue.append(pred)
                    
        if len(sliced_nodes) < 3:
            return G
            
        # Context Enrichment: Include 1-hop structural AST context nodes to prevent graph skeleton collapse
        enriched_nodes = set(sliced_nodes)
        for node in sliced_nodes:
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                for key in edge_data:
                    if edge_data[key].get('label', '').upper() == 'AST':
                        enriched_nodes.add(neighbor)
                        
        sliced_G = G.subgraph(enriched_nodes).copy()
        
        if sliced_G.number_of_edges() == 0:
            return G
            
        return sliced_G

    def graphml_to_pyg(self, file_path):
        """Parses a Joern GraphML file, runs the context-aware backward-slicing pass, and encodes attributes."""
        try:
            G = nx.read_graphml(file_path)
            
            # Execute contextual backward slice
            G = self.execute_backward_slice(G)
            
            node_features = []
            node_text_corpus = []
            
            # Track explicit node key ordering mapping context
            node_to_idx = {}
            for idx, (node, attrs) in enumerate(G.nodes(data=True)):
                node_to_idx[node] = idx
                label = attrs.get('label', 'UNKNOWN').upper()
                node_features.append([self.node_vocab.get(label, 0)])
                
                node_name = attrs.get('name', '')
                node_code = attrs.get('code', '')
                node_text_corpus.append(f"{node_name} {node_code}".strip())
            
            edge_features = []
            edges = []
            for u, v, attrs in G.edges(data=True):
                if u in node_to_idx and v in node_to_idx:
                    edges.append((node_to_idx[u], node_to_idx[v]))
                    edge_label = attrs.get('label', 'UNKNOWN').upper()
                    edge_features.append(self.edge_vocab.get(edge_label, 0))

            if len(edges) == 0:
                return None
                
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            x_tensor = torch.tensor(node_features, dtype=torch.float)
            edge_attr_tensor = torch.tensor(edge_features, dtype=torch.long)
            
            pyg_data = Data(x=x_tensor, edge_index=edge_index)
            pyg_data.edge_attr = edge_attr_tensor
            pyg_data.text_document = " ".join([t for t in node_text_corpus if t])
            
            return pyg_data
            
        except Exception as e:
            logging.error(f"Failed to parse deep graph structures for {file_path}: {str(e)}")
            return None

    def load_cohort(self, executed_successfully=True):
        """Scans folder for single GraphML files to build the PyG cohort dataset."""
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
            mock_data = Data(x=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float), edge_index=edge_index)
            mock_data.edge_attr = torch.tensor([1, 1, 1], dtype=torch.long)
            mock_data.text_document = "public static void main String args System.out.println"
            cohort_data[f"Student_{i:02d}_Mock"] = mock_data
        return cohort_data
