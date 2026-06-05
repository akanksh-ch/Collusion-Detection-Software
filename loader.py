import os
import logging
import networkx as nx
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed

def _load_and_slice_single_file(args):
    """Worker process task: handles parsing, isolates core code logic lineages, 

    and serializes the results into lightweight dictionaries."""
    file_path, node_vocab, edge_vocab, bypass_slicing = args
    student_id = os.path.basename(file_path).replace(".graphml", "")
    
    try:
        G = nx.read_graphml(file_path)
        
        # Isolate code lineages using a backward dependency graph walk from exit points
        if not bypass_slicing:
            sinks = [n for n, attrs in G.nodes(data=True) if attrs.get('label', '').upper() in ['RETURN', 'METHOD_RETURN']]
            if not sinks:
                sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
                
            if sinks:
                sliced_nodes = set(sinks)
                queue = list(sinks)
                while queue:
                    current = queue.pop(0)
                    for pred in G.predecessors(current):
                        edge_data = G.get_edge_data(pred, current)
                        # Verify edge labels map to actual control or data flow transitions
                        is_valid = any(G.get_edge_data(pred, current)[k].get('label', '').upper() in ['REACHING_DEF', 'CDG', 'CFG', 'DDG'] for k in edge_data)
                        if is_valid and pred not in sliced_nodes:
                            sliced_nodes.add(pred)
                            queue.append(pred)
                            
                # Fall back to original graph layout if the slice strips too much context
                if len(sliced_nodes) >= 3:
                    enriched_nodes = set(sliced_nodes)
                    for node in sliced_nodes:
                        for neighbor in G.neighbors(node):
                            if any(G.get_edge_data(node, neighbor)[k].get('label', '').upper() == 'AST' for k in G.get_edge_data(node, neighbor)):
                                enriched_nodes.add(neighbor)
                    sliced_G = G.subgraph(enriched_nodes).copy()
                    if sliced_G.number_of_edges() > 0:
                        G = sliced_G

        # Map graph properties into vocab vectors and aggregate text tokens
        node_features = []
        node_text_corpus = []
        node_to_idx = {}
        
        for idx, (node, attrs) in enumerate(G.nodes(data=True)):
            node_to_idx[node] = idx
            label = attrs.get('label', 'UNKNOWN').upper()
            node_features.append([node_vocab.get(label, 0)])
            node_text_corpus.append(f"{attrs.get('name', '')} {attrs.get('code', '')}".strip())
        
        edges = []
        edge_features = []
        for u, v, attrs in G.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edges.append((node_to_idx[u], node_to_idx[v]))
                edge_label = attrs.get('label', 'UNKNOWN').upper()
                edge_features.append(edge_vocab.get(edge_label, 0))

        if len(edges) == 0:
            return student_id, None

        # Return a primitive dictionary payload to keep process memory usage low
        return student_id, {
            "x": node_features,
            "edge_index": edges,
            "edge_attr": edge_features,
            "text_document": " ".join([t for t in node_text_corpus if t])
        }
    except Exception as e:
        logging.error(f"Failed parsing worker pipeline for asset {student_id}: {e}")
        return student_id, None


class CPGDataLoader:
    """Orchestrates parallel background file streaming and maps schema tokens."""
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.node_vocab = {}
        self.edge_vocab = {}

    def discover_vocabularies(self):
        """Scans all cohort target files to build continuous integer lookups for schemas."""
        node_labels = set()
        edge_labels = set()
        
        if not os.path.exists(self.input_dir):
            return

        for item in os.listdir(self.input_dir):
            if item.endswith(".graphml"):
                try:
                    G = nx.read_graphml(os.path.join(self.input_dir, item))
                    for _, attrs in G.nodes(data=True):
                        if 'label' in attrs: node_labels.add(attrs['label'].upper())
                    for _, _, attrs in G.edges(data=True):
                        if 'label' in attrs: edge_labels.add(attrs['label'].upper())
                except Exception:
                    continue

        self.node_vocab = {label: idx + 1 for idx, label in enumerate(sorted(node_labels))}
        self.node_vocab['UNKNOWN'] = 0
        self.edge_vocab = {label: idx + 1 for idx, label in enumerate(sorted(edge_labels))}
        self.edge_vocab['UNKNOWN'] = 0

    def load_cohort(self, executed_successfully=True, bypass_slicing=False):
        """Asynchronously parses and formats raw GraphML files using background workers."""
        cohort_data = {}
        if not executed_successfully or not os.path.exists(self.input_dir):
            return self._generate_dummy_cohort()

        files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith(".graphml")]
        if not files:
            return self._generate_dummy_cohort()

        max_workers = max(1, os.cpu_count() - 1)
        tasks = [(f, self.node_vocab, self.edge_vocab, bypass_slicing) for f in files]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_and_slice_single_file, task): task for task in tasks}
            for future in as_completed(futures):
                student_id, payload = future.result()
                if payload:
                    # Construct clean PyTorch Geometric primitives on the main thread
                    pyg_data = Data(
                        x=torch.tensor(payload["x"], dtype=torch.float),
                        edge_index=torch.tensor(payload["edge_index"], dtype=torch.long).t().contiguous()
                    )
                    pyg_data.edge_attr = torch.tensor(payload["edge_attr"], dtype=torch.long)
                    pyg_data.text_document = payload["text_document"]
                    cohort_data[student_id] = pyg_data

        return cohort_data

    def _generate_dummy_cohort(self):
        cohort_data = {}
        for i in range(1, 4):
            mock_data = Data(x=torch.tensor([[1.0], [2.0]], dtype=torch.float), edge_index=torch.tensor([[0], [1]], dtype=torch.long))
            mock_data.edge_attr = torch.tensor([1], dtype=torch.long)
            mock_data.text_document = "public static void main String args"
            cohort_data[f"Student_{i:02d}_Mock"] = mock_data
        return cohort_data
