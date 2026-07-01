import os
import logging
import networkx as nx
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed

def _load_and_slice_single_file(args):
    """
    Worker process task that handles a single GraphML asset. It isolates core 
    code logic lineages using backward dependency slicing and maps structural properties.
    """
    file_path, node_vocab, edge_vocab, bypass_slicing, active_label_key = args
    student_id = os.path.basename(file_path).replace(".graphml", "")
    
    try:
        G = nx.read_graphml(file_path)
        
        # Isolate code lineages using a backward dependency graph walk from exit points
        if not bypass_slicing:
            sinks = [n for n, attrs in G.nodes(data=True) if attrs.get(active_label_key, '').upper() in ['RETURN', 'METHOD_RETURN']]
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

        node_features = []
        node_text_corpus = []
        node_to_idx = {}
        
        for idx, (node, attrs) in enumerate(G.nodes(data=True)):
            node_to_idx[node] = idx
            
            raw_label = attrs.get(active_label_key, 'UNKNOWN')
            label = str(raw_label).upper()
            
            # Sub-type enrichment to capture low-level primitive identities and logic structures
            node_name = str(attrs.get('NAME', attrs.get('name', ''))).upper()
            if label == 'CALL' and node_name.startswith('<OPERATOR>.'):
                label = f"CALL_{node_name}"
            elif label == 'CONTROL_STRUCTURE':
                control_type = str(attrs.get('CONTROL_STRUCTURE_TYPE', attrs.get('controlStructureType', ''))).upper()
                if control_type:
                    label = f"CONTROL_{control_type}"
            
            node_features.append([node_vocab.get(label, node_vocab.get('UNKNOWN', 0))])
            
            node_code = attrs.get('CODE', attrs.get('code', ''))
            node_text_corpus.append(f"{node_name} {node_code}".strip())
        
        edges = []
        edge_features = []
        for u, v, attrs in G.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edges.append((node_to_idx[u], node_to_idx[v]))
                edge_label = attrs.get('label', 'UNKNOWN').upper()
                edge_features.append(edge_vocab.get(edge_label, 0))

        if len(edges) == 0:
            return student_id, None

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
    """
    Manages multi-threaded file serialization workflows and dynamic Joern
    schema alignment optimizations across the student cohort dataset.
    """
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.node_vocab = {}
        self.edge_vocab = {}
        self.active_label_key = 'label'

    def discover_vocabularies(self):
        """
        Scans all cohort target files to build continuous integer lookups for schemas.
        Dynamically discovers the operational Joern node attribute key to enforce system resilience.
        """
        node_labels = set()
        edge_labels = set()
        
        if not os.path.exists(self.input_dir):
            return

        candidate_keys = ['labelV', 'label', '_label', 'type', 'nodeType']

        # Determine which attribute layout configuration is currently active in the workspace exports
        for item in os.listdir(self.input_dir):
            if item.endswith(".graphml"):
                try:
                    G = nx.read_graphml(os.path.join(self.input_dir, item))
                    for _, attrs in G.nodes(data=True):
                        for key in candidate_keys:
                            val = str(attrs.get(key, '')).upper()
                            if val in ['METHOD', 'CALL', 'BLOCK', 'IDENTIFIER', 'CONTROL_STRUCTURE']:
                                self.active_label_key = key
                                logging.info(f"[SCHEMA ALIGNMENT] Active Joern node dictionary key set to: '{self.active_label_key}'")
                                break
                        if self.active_label_key != 'label': break
                    if self.active_label_key != 'label': break
                except Exception:
                    continue

        # Extract structural classes using sub-type vocabulary transformations
        for item in os.listdir(self.input_dir):
            if item.endswith(".graphml"):
                try:
                    G = nx.read_graphml(os.path.join(self.input_dir, item))
                    for _, attrs in G.nodes(data=True):
                        raw_label = attrs.get(self.active_label_key)
                        if raw_label: 
                            label = str(raw_label).upper()
                            
                            node_name = str(attrs.get('NAME', attrs.get('name', ''))).upper()
                            if label == 'CALL' and node_name.startswith('<OPERATOR>.'):
                                label = f"CALL_{node_name}"
                            elif label == 'CONTROL_STRUCTURE':
                                control_type = str(attrs.get('CONTROL_STRUCTURE_TYPE', attrs.get('controlStructureType', ''))).upper()
                                if control_type:
                                    label = f"CONTROL_{control_type}"
                            
                            node_labels.add(label)
                            
                    for _, _, attrs in G.edges(data=True):
                        if 'label' in attrs: 
                            edge_labels.add(attrs['label'].upper())
                except Exception:
                    continue

        self.node_vocab = {label: idx + 1 for idx, label in enumerate(sorted(node_labels))}
        self.node_vocab['UNKNOWN'] = 0
        self.edge_vocab = {label: idx + 1 for idx, label in enumerate(sorted(edge_labels))}
        self.edge_vocab['UNKNOWN'] = 0
        
        logging.info(f"[VOCABULARY] Map compilation finalized. Unique functional entities cataloged: {len(self.node_vocab)}")

    def load_cohort(self, executed_successfully=True, bypass_slicing=False):
        """Asynchronously parses and formats raw GraphML files using background workers."""
        cohort_data = {}
        if not executed_successfully or not os.path.exists(self.input_dir):
            return self._generate_dummy_cohort()

        files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith(".graphml")]
        if not files:
            return self._generate_dummy_cohort()

        max_workers = max(1, os.cpu_count() - 1)
        tasks = [(f, self.node_vocab, self.edge_vocab, bypass_slicing, self.active_label_key) for f in files]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_and_slice_single_file, task): task for task in tasks}
            for future in as_completed(futures):
                student_id, payload = future.result()
                if payload:
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
