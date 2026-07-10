"""Structural Masking, Semantic Slicing, and Tensorisation pipeline.

This module implements Module 2 of the production specification:
  1. Graph-level template masking (boilerplate subtraction)
  2. Backward dependency slicing anchored at exits + I/O sinks
  3. Identifier casing profiling (style enum)
  4. Two-column categorical tensor conversion for PyG

No literal text enters the graph tensor.  Stripped source paths and
comment sidecar paths are returned alongside each Data object so the
text branch can consume them independently.
"""

import os
import re
import hashlib
import logging
import networkx as nx
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────────────────────────────────────────────────────────────
# Casing classifier
# ──────────────────────────────────────────────────────────────────────

_SNAKE   = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_CAMEL   = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")
_PASCAL  = re.compile(r"^[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")
_UPPER   = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
_SINGLE  = re.compile(r"^[a-zA-Z_]$")


def classify_casing(name: str) -> int:
    """Classify an identifier into a casing style enum.

    Returns:
        0 = snake_case
        1 = camelCase
        2 = PascalCase
        3 = UPPER_CASE
        4 = single character / obfuscated
    """
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
    # Short names (2-3 chars) that match nothing above → obfuscated
    if len(name) <= 3:
        return 4
    # Fallback heuristic — inspect first character
    if name[0].isupper():
        return 2
    return 1


# ──────────────────────────────────────────────────────────────────────
# Template masking helpers
# ──────────────────────────────────────────────────────────────────────

def _node_structural_hash(G, node, label_key):
    """Compute a structural signature hash for a node based on its type and
    the sorted list of its direct neighbours' types (1-hop topology)."""
    attrs = G.nodes[node]
    node_type = str(attrs.get(label_key, "UNKNOWN")).upper()

    neighbour_types = sorted(
        str(G.nodes[nb].get(label_key, "UNKNOWN")).upper()
        for nb in set(G.predecessors(node)) | set(G.successors(node))
    )

    sig = f"{node_type}|{'|'.join(neighbour_types)}"
    return hashlib.sha256(sig.encode()).hexdigest()


def mask_boilerplate(student_G, skeleton_G, label_key):
    """Delete boilerplate nodes (and their edges) from *student_G* in-place.

    For each node in the instructor skeleton graph, compute a structural hash
    (type + sorted neighbour types).  Then remove every student node whose hash
    appears in the skeleton set.

    Returns the number of nodes removed.
    """
    skeleton_hashes = set()
    for node in skeleton_G.nodes():
        skeleton_hashes.add(_node_structural_hash(skeleton_G, node, label_key))

    nodes_to_remove = []
    for node in student_G.nodes():
        h = _node_structural_hash(student_G, node, label_key)
        if h in skeleton_hashes:
            nodes_to_remove.append(node)

    student_G.remove_nodes_from(nodes_to_remove)
    return len(nodes_to_remove)


# ──────────────────────────────────────────────────────────────────────
# Per-submission worker
# ──────────────────────────────────────────────────────────────────────

# I/O sink labels that serve as backward-slice anchors in addition to returns
_IO_SINKS = frozenset({"WRITE", "PRINT", "PRINTLN", "PRINTF", "SYSTEM.OUT"})
# Edge types traversed during backward slicing (spec: DDG + REACHING_DEF only)
_SLICE_EDGE_TYPES = frozenset({"DDG", "REACHING_DEF"})
# Node labels considered identifiers for casing profiling
_IDENTIFIER_LABELS = frozenset({"IDENTIFIER", "LOCAL", "MEMBER", "FIELD_IDENTIFIER",
                                 "METHOD_PARAMETER_IN"})


def _load_and_slice_single_file(args):
    """Worker process: load a student GraphML, apply masking + slicing +
    casing profiling, and return a categorical tensor payload."""
    (file_path, node_vocab, edge_vocab, bypass_slicing,
     active_label_key, skeleton_hashes) = args

    student_id = os.path.basename(file_path).replace(".graphml", "")

    try:
        G = nx.read_graphml(file_path)

        # ── 1. Graph-level template masking ──────────────────────────
        if skeleton_hashes:
            nodes_to_remove = []
            for node in G.nodes():
                h = _node_structural_hash(G, node, active_label_key)
                if h in skeleton_hashes:
                    nodes_to_remove.append(node)
            G.remove_nodes_from(nodes_to_remove)

        # ── 2. Backward dependency slicing ───────────────────────────
        if not bypass_slicing:
            # Collect sink anchors: RETURN / METHOD_RETURN + I/O sinks
            sinks = []
            for n, attrs in G.nodes(data=True):
                label = str(attrs.get(active_label_key, "")).upper()
                if label in ("RETURN", "METHOD_RETURN"):
                    sinks.append(n)
                # Also anchor at I/O calls
                name = str(attrs.get("NAME", attrs.get("name", ""))).upper()
                code = str(attrs.get("CODE", attrs.get("code", ""))).upper()
                if any(io in name or io in code for io in _IO_SINKS):
                    sinks.append(n)

            if not sinks:
                # Fallback: terminal nodes with no successors
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
                        # Check all edge keys for DDG / REACHING_DEF
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

        # ── 3. Build node features: x_type + x_style ────────────────
        node_features = []
        node_to_idx = {}

        for idx, (node, attrs) in enumerate(G.nodes(data=True)):
            node_to_idx[node] = idx

            raw_label = attrs.get(active_label_key, "UNKNOWN")
            label = str(raw_label).upper()

            # Sub-type enrichment
            node_name = str(attrs.get("NAME", attrs.get("name", ""))).upper()
            if label == "CALL" and node_name.startswith("<OPERATOR>."):
                label = f"CALL_{node_name}"
            elif label == "CONTROL_STRUCTURE":
                control_type = str(attrs.get("CONTROL_STRUCTURE_TYPE",
                                             attrs.get("controlStructureType", ""))).upper()
                if control_type:
                    label = f"CONTROL_{control_type}"

            x_type = node_vocab.get(label, node_vocab.get("UNKNOWN", 0))

            # Casing profiling (only meaningful for identifier-like nodes)
            raw_name = str(attrs.get("NAME", attrs.get("name", "")))
            if label in _IDENTIFIER_LABELS or label == "IDENTIFIER":
                x_style = classify_casing(raw_name)
            else:
                x_style = 4  # non-identifier → obfuscated/default bucket

            node_features.append([x_type, x_style])

        # ── 4. Build edge index + edge attributes ────────────────────
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
            "x": node_features,          # list of [x_type, x_style]
            "edge_index": edges,
            "edge_attr": edge_features,
        }
    except Exception as e:
        logging.error(f"Failed parsing worker pipeline for asset {student_id}: {e}")
        return student_id, None


# ──────────────────────────────────────────────────────────────────────
# Public loader class
# ──────────────────────────────────────────────────────────────────────

class CPGDataLoader:
    """Manages multi-threaded file serialization workflows and dynamic Joern
    schema alignment optimizations across the student cohort dataset.

    Major changes from the previous revision:
    * Graph-level boilerplate masking via structural hash subtraction
    * Strict backward slicing (DDG + REACHING_DEF only)
    * Two-column categorical tensor [x_type, x_style] — no text in tensor
    * Returns stripped-source and comment paths for the text branch
    """
    def __init__(self, input_dir, skeleton_path=None):
        self.input_dir = input_dir
        self.skeleton_path = skeleton_path
        self.node_vocab = {}
        self.edge_vocab = {}
        self.active_label_key = "label"
        self._skeleton_hashes = set()

    # ------------------------------------------------------------------
    # Vocabulary discovery
    # ------------------------------------------------------------------
    def discover_vocabularies(self):
        """Scans all cohort target files to build continuous integer lookups.
        Dynamically discovers the operational Joern node attribute key."""
        node_labels = set()
        edge_labels = set()

        if not os.path.exists(self.input_dir):
            return

        candidate_keys = ["labelV", "label", "_label", "type", "nodeType"]

        # Determine active attribute key
        for item in os.listdir(self.input_dir):
            if item.endswith(".graphml"):
                try:
                    G = nx.read_graphml(os.path.join(self.input_dir, item))
                    for _, attrs in G.nodes(data=True):
                        for key in candidate_keys:
                            val = str(attrs.get(key, "")).upper()
                            if val in ("METHOD", "CALL", "BLOCK", "IDENTIFIER",
                                       "CONTROL_STRUCTURE"):
                                self.active_label_key = key
                                logging.info(
                                    f"[SCHEMA ALIGNMENT] Active Joern node "
                                    f"dictionary key set to: '{self.active_label_key}'"
                                )
                                break
                        if self.active_label_key != "label":
                            break
                    if self.active_label_key != "label":
                        break
                except Exception:
                    continue

        # Extract structural classes (with sub-type enrichment)
        for item in os.listdir(self.input_dir):
            if item.endswith(".graphml"):
                try:
                    G = nx.read_graphml(os.path.join(self.input_dir, item))
                    for _, attrs in G.nodes(data=True):
                        raw_label = attrs.get(self.active_label_key)
                        if raw_label:
                            label = str(raw_label).upper()

                            node_name = str(attrs.get("NAME", attrs.get("name", ""))).upper()
                            if label == "CALL" and node_name.startswith("<OPERATOR>."):
                                label = f"CALL_{node_name}"
                            elif label == "CONTROL_STRUCTURE":
                                control_type = str(attrs.get(
                                    "CONTROL_STRUCTURE_TYPE",
                                    attrs.get("controlStructureType", "")
                                )).upper()
                                if control_type:
                                    label = f"CONTROL_{control_type}"

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

        logging.info(
            f"[VOCABULARY] Map compilation finalized. "
            f"Unique functional entities cataloged: {len(self.node_vocab)}"
        )

        # Pre-compute skeleton hashes if a boilerplate file was supplied
        if self.skeleton_path and os.path.exists(self.skeleton_path):
            try:
                skel_G = nx.read_graphml(self.skeleton_path)
                for node in skel_G.nodes():
                    self._skeleton_hashes.add(
                        _node_structural_hash(skel_G, node, self.active_label_key)
                    )
                logging.info(
                    f"[MASKING] Skeleton loaded — {len(self._skeleton_hashes)} "
                    f"boilerplate structural hashes registered."
                )
            except Exception as e:
                logging.warning(f"[MASKING] Failed to load skeleton: {e}")

    # ------------------------------------------------------------------
    # Cohort loading
    # ------------------------------------------------------------------
    def load_cohort(self, executed_successfully=True, bypass_slicing=False):
        """Asynchronously parses and formats raw GraphML files.

        Returns:
            dict mapping student_id → PyG ``Data`` objects.
            Each ``Data`` has:
              - ``x``: shape ``[N, 2]`` categorical ints ``[x_type, x_style]``
              - ``edge_index``: ``[2, E]`` long
              - ``edge_attr``: ``[E]`` long
        """
        cohort_data = {}
        if not executed_successfully or not os.path.exists(self.input_dir):
            return self._generate_dummy_cohort()

        files = [
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.endswith(".graphml")
        ]
        if not files:
            return self._generate_dummy_cohort()

        max_workers = max(1, os.cpu_count() - 1)
        tasks = [
            (f, self.node_vocab, self.edge_vocab, bypass_slicing,
             self.active_label_key, self._skeleton_hashes)
            for f in files
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_and_slice_single_file, task): task
                       for task in tasks}
            for future in as_completed(futures):
                student_id, payload = future.result()
                if payload:
                    x_tensor = torch.tensor(payload["x"], dtype=torch.long)
                    edge_index = torch.tensor(
                        payload["edge_index"], dtype=torch.long
                    ).t().contiguous()
                    edge_attr = torch.tensor(payload["edge_attr"], dtype=torch.long)

                    pyg_data = Data(
                        x=x_tensor,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                    )
                    cohort_data[student_id] = pyg_data

        return cohort_data

    # ------------------------------------------------------------------
    # Dummy fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_dummy_cohort():
        cohort_data = {}
        for i in range(1, 4):
            mock_data = Data(
                x=torch.tensor([[1, 0], [2, 1]], dtype=torch.long),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.tensor([1], dtype=torch.long),
            )
            cohort_data[f"Student_{i:02d}_Mock"] = mock_data
        return cohort_data
