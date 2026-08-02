import os
import re
import hashlib
import logging
from dataclasses import dataclass

import networkx as nx
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed


_SNAKE   = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_CAMEL   = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")
_PASCAL  = re.compile(r"^[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")
_UPPER   = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
_SINGLE  = re.compile(r"^[a-zA-Z_]$")

# Loop-family control structures that should be treated as one canonical
# node type (CONTROL_LOOP) rather than distinct FOR/WHILE/DO types.
# This gives renaming-invariance-style protection against L6 "change loop
# type" obfuscation attacks (e.g. Karnalim's taxonomy, Nocte's for->while
# normalization), at the cost of losing the specific loop kind as a feature.
_LOOP_TYPES = frozenset({"FOR", "WHILE", "DO", "DOWHILE", "DO_WHILE"})


@dataclass
class CPGGraph:
    """Plain container for a parsed CPG — replaces torch_geometric.data.Data.

    graph2vec (karateclub) consumes networkx graphs built from these
    fields directly; nothing downstream needs PyG's batching/loader
    machinery, so this drops that dependency entirely.
    """
    x: torch.Tensor          # [N, 2] int64 — (node_type_idx, style_idx)
    edge_index: torch.Tensor  # [2, E] int64
    edge_attr: torch.Tensor   # [E] int64


def classify_casing(name: str) -> int:
    if not name or _SINGLE.match(name):
        return 4
    if _SNAKE.match(name):
        return 0
    if _CAMEL.match(name):
        return 1
    if _PASCAL.match(name):
        return 2
    if _UPPER.match(name):
        return 3
    if len(name) <= 3:
        return 4
    if name[0].isupper():
        return 2
    return 1


def _node_structural_hash(G, node, label_key):
    attrs = G.nodes[node]
    node_type = str(attrs.get(label_key, "UNKNOWN")).upper()
    neighbour_types = sorted(
        str(G.nodes[nb].get(label_key, "UNKNOWN")).upper()
        for nb in set(G.predecessors(node)) | set(G.successors(node))
    )
    sig = f"{node_type}|{'|'.join(neighbour_types)}"
    return hashlib.sha256(sig.encode()).hexdigest()


def _resolve_control_label(attrs):
    """Resolve the vocabulary label for a CONTROL_STRUCTURE node.

    Loop-family constructs (FOR/WHILE/DO) are canonicalized to a single
    CONTROL_LOOP label so that changing loop type (an L6 obfuscation
    attack) does not change the node's structural embedding. Other
    control structures (IF, SWITCH, etc.) keep their specific label.
    """
    control_type = str(
        attrs.get("CONTROL_STRUCTURE_TYPE", attrs.get("controlStructureType", ""))
    ).upper()

    if control_type in _LOOP_TYPES:
        return "CONTROL_LOOP"
    if control_type:
        return f"CONTROL_{control_type}"
    return "CONTROL_STRUCTURE"


def _load_and_slice_single_file(args):
    """Worker process: load nested GraphML datasets and profile properties."""
    (file_path, node_vocab, edge_vocab, bypass_slicing,
     active_label_key, skeleton_hashes, student_id) = args

    _IO_SINKS = frozenset({"WRITE", "PRINT", "PRINTLN", "PRINTF", "SYSTEM.OUT"})
    _SLICE_EDGE_TYPES = frozenset({"DDG", "REACHING_DEF"})
    _IDENTIFIER_LABELS = frozenset({"IDENTIFIER", "LOCAL", "MEMBER", "FIELD_IDENTIFIER", "METHOD_PARAMETER_IN"})

    try:
        G = nx.read_graphml(file_path)

        if skeleton_hashes:
            nodes_to_remove = []
            for node in G.nodes():
                h = _node_structural_hash(G, node, active_label_key)
                if h in skeleton_hashes:
                    nodes_to_remove.append(node)
            G.remove_nodes_from(nodes_to_remove)

        if not bypass_slicing:
            sinks = []
            for n, attrs in G.nodes(data=True):
                label = str(attrs.get(active_label_key, "")).upper()
                if label in ("RETURN", "METHOD_RETURN"):
                    sinks.append(n)
                name = str(attrs.get("NAME", attrs.get("name", ""))).upper()
                code = str(attrs.get("CODE", attrs.get("code", ""))).upper()
                if any(io in name or io in code for io in _IO_SINKS):
                    sinks.append(n)

            if not sinks:
                sinks = [n for n in G.nodes() if G.out_degree(n) == 0]

            if sinks:
                sliced_nodes = set(sinks)
                queue = list(sinks)
                while queue:
                    current = queue.pop(0)
                    for pred in G.predecessors(current):
                        if pred in sliced_nodes:
                            continue
                        edge_data = G.get_edge_data(pred, current)
                        if edge_data is None:
                            continue
                        is_data_flow = any(
                            str(edge_data[k].get("label", "")).upper() in _SLICE_EDGE_TYPES
                            for k in edge_data
                        )
                        if is_data_flow:
                            sliced_nodes.add(pred)
                            queue.append(pred)

                if sliced_nodes:
                    sliced_G = G.subgraph(sliced_nodes).copy()
                    if sliced_G.number_of_edges() > 0:
                        G = sliced_G

        node_features = []
        node_to_idx = {}

        for idx, (node, attrs) in enumerate(G.nodes(data=True)):
            node_to_idx[node] = idx
            raw_label = attrs.get(active_label_key, "UNKNOWN")
            label = str(raw_label).upper()

            node_name = str(attrs.get("NAME", attrs.get("name", ""))).upper()
            if label == "CALL" and node_name.startswith("<OPERATOR>."):
                label = f"CALL_{node_name}"
            elif label == "CONTROL_STRUCTURE":
                label = _resolve_control_label(attrs)

            x_type = node_vocab.get(label, node_vocab.get("UNKNOWN", 0))
            raw_name = str(attrs.get("NAME", attrs.get("name", "")))
            if label in _IDENTIFIER_LABELS or label == "IDENTIFIER":
                x_style = classify_casing(raw_name)
            else:
                x_style = 4

            node_features.append([x_type, x_style])

        edges = []
        edge_features = []
        for u, v, attrs in G.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edges.append((node_to_idx[u], node_to_idx[v]))
                edge_label = str(attrs.get("label", "UNKNOWN")).upper()
                edge_features.append(edge_vocab.get(edge_label, 0))

        if len(edges) == 0:
            return student_id, None

        return student_id, {
            "x": node_features,
            "edge_index": edges,
            "edge_attr": edge_features,
        }
    except Exception as e:
        logging.error(f"Failed parsing worker pipeline for asset {student_id}: {e}")
        return student_id, None


class CPGDataLoader:
    """Manages multithreaded file loading with deep recursive schema lookups."""

    def __init__(self, input_dir, skeleton_path=None):
        self.input_dir = input_dir
        self.skeleton_path = skeleton_path
        self.node_vocab = {}
        self.edge_vocab = {}
        self.active_label_key = "label"
        self._skeleton_hashes = set()

    def discover_vocabularies(self):
        node_labels = set()
        edge_labels = set()

        if not os.path.exists(self.input_dir):
            return

        candidate_keys = ["labelV", "label", "_label", "type", "nodeType"]
        graphml_files = []

        for root, _, files in os.walk(self.input_dir):
            if any(x in root for x in ("_stripped", "_comments", "skeleton")):
                continue
            for f in files:
                if f.endswith(".graphml"):
                    graphml_files.append(os.path.join(root, f))

        if not graphml_files:
            return

        try:
            G = nx.read_graphml(graphml_files[0])
            for _, attrs in G.nodes(data=True):
                for key in candidate_keys:
                    val = str(attrs.get(key, "")).upper()
                    if val in ("METHOD", "CALL", "BLOCK", "IDENTIFIER", "CONTROL_STRUCTURE"):
                        self.active_label_key = key
                        logging.info(f"[SCHEMA ALIGNMENT] Active key discovered: '{self.active_label_key}'")
                        break
                if self.active_label_key != "label":
                    break
        except Exception:
            pass

        for file_path in graphml_files:
            try:
                G = nx.read_graphml(file_path)
                for _, attrs in G.nodes(data=True):
                    raw_label = attrs.get(self.active_label_key)
                    if raw_label:
                        label = str(raw_label).upper()
                        node_name = str(attrs.get("NAME", attrs.get("name", ""))).upper()
                        if label == "CALL" and node_name.startswith("<OPERATOR>."):
                            label = f"CALL_{node_name}"
                        elif label == "CONTROL_STRUCTURE":
                            label = _resolve_control_label(attrs)
                        node_labels.add(label)

                for _, _, attrs in G.edges(data=True):
                    if "label" in attrs:
                        edge_labels.add(attrs["label"].upper())
            except Exception:
                continue

        self.node_vocab = {label: idx + 1 for idx, label in enumerate(sorted(node_labels))}
        self.node_vocab["UNKNOWN"] = 0
        self.edge_vocab = {label: idx + 1 for idx, label in enumerate(sorted(edge_labels))}
        self.edge_vocab["UNKNOWN"] = 0

        if self.skeleton_path and os.path.exists(self.skeleton_path):
            try:
                skel_G = nx.read_graphml(self.skeleton_path)
                for node in skel_G.nodes():
                    self._skeleton_hashes.add(_node_structural_hash(skel_G, node, self.active_label_key))
            except Exception:
                pass

    def load_cohort(self, executed_successfully=True, bypass_slicing=False):
        cohort_data = {}
        if not executed_successfully or not os.path.exists(self.input_dir):
            return self._generate_dummy_cohort()

        files = []
        base_abs = os.path.abspath(self.input_dir)
        for root, _, filenames in os.walk(base_abs):
            if any(x in root for x in ("_stripped", "_comments", "skeleton")):
                continue
            for f in filenames:
                if f.endswith(".graphml"):
                    files.append(os.path.join(root, f))

        if not files:
            return self._generate_dummy_cohort()

        max_workers = max(1, os.cpu_count() - 1)
        tasks = []
        for f in files:
            rel_path = os.path.relpath(f, base_abs)
            student_id = os.path.splitext(rel_path)[0].replace(os.sep, "/")
            tasks.append((f, self.node_vocab, self.edge_vocab, bypass_slicing,
                          self.active_label_key, self._skeleton_hashes, student_id))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_and_slice_single_file, task): task for task in tasks}
            for future in as_completed(futures):
                student_id, payload = future.result()
                if payload:
                    x_tensor = torch.tensor(payload["x"], dtype=torch.long)
                    edge_index = torch.tensor(payload["edge_index"], dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(payload["edge_attr"], dtype=torch.long)

                    cohort_data[student_id] = CPGGraph(
                        x=x_tensor, edge_index=edge_index, edge_attr=edge_attr,
                    )

        return cohort_data

    @staticmethod
    def _generate_dummy_cohort():
        cohort_data = {}
        for i in range(1, 4):
            mock_data = CPGGraph(
                x=torch.tensor([[1, 0], [2, 1]], dtype=torch.long),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.tensor([1], dtype=torch.long),
            )
            cohort_data[f"mock/Student_{i:02d}"] = mock_data
        return cohort_data
