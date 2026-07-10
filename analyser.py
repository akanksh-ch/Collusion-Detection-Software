"""HNSW Indexing, Forensic Disaggregation, GST Evidence, & Leiden Clustering.

Implements Module 4 of the production specification:
  1. FAISS HNSW all-pairs similarity search on V_program
  2. Forensic disaggregation: independent cos(V_topo), cos(V_text) per pair
  3. Greedy String Tiling line-level evidence generation
  4. Leiden community detection for collusion-network partitioning
"""

from __future__ import annotations

import os
import logging
import numpy as np
import networkx as nx
import faiss

from gnn import MultiModalEncoder
from gst import tokenize_source, greedy_string_tiling, coverage_score, tiles_to_json
from parse import JoernAutomationParser


class CohortAnalyzer:
    """Orchestrates spatial profiling, disaggregated forensic scoring,
    GST evidence generation, and Leiden community partitioning."""

    def __init__(
        self,
        cohort_data: dict,
        node_vocab_size: int,
        edge_vocab_size: int,
        source_dir: str = "./submissions",
        workspace_dir: str = "./joern_workspace_exports",
        dim: int = 128,
    ):
        self.cohort_data = cohort_data
        self.student_ids = list(cohort_data.keys())
        self.source_dir = source_dir
        self.workspace_dir = workspace_dir

        self.encoder = MultiModalEncoder(
            node_type_vocab_size=node_vocab_size,
            edge_vocab_size=edge_vocab_size,
            dim=dim,
        )

        # Per-student decoupled vectors
        self.vectors_topo: dict[str, np.ndarray] = {}
        self.vectors_text: dict[str, np.ndarray] = {}
        self.vectors_program: dict[str, np.ndarray] = {}
        self.embeddings: np.ndarray | None = None

        self.knn_graph: nx.Graph | None = None
        self.node_to_community: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Step 1: Generate embeddings (all three vectors per student)
    # ------------------------------------------------------------------
    def generate_all_embeddings(self):
        """Generate V_topo, V_text, V_program for every student.

        After encoding, applies isotropic mean-centering to all three
        vector spaces to remove the common mode that makes frozen random
        network outputs cluster in a narrow similarity band.
        """
        if not self.student_ids:
            raise ValueError("Cannot generate embeddings: Cohort dataset is empty.")

        # Collect stripped source texts and comment streams
        stripped_texts = []
        comment_texts = []
        for sid in self.student_ids:
            stripped_texts.append(self._read_stripped_source(sid))
            comment_texts.append(self._read_comments(sid))

        # Fit TF-IDF + SVD on the full cohort
        self.encoder.fit_text_cohort(stripped_texts, comment_texts)

        # Encode each submission
        for idx, sid in enumerate(self.student_ids):
            pyg_data = self.cohort_data[sid]
            vecs = self.encoder.encode_submission(
                pyg_data, stripped_texts[idx], comment_texts[idx]
            )
            self.vectors_topo[sid] = vecs["v_topo"]
            self.vectors_text[sid] = vecs["v_text"]
            self.vectors_program[sid] = vecs["v_program"]

        # Mean-center all three vector spaces (isotropic calibration)
        from gnn import MultiModalEncoder
        self.vectors_topo = MultiModalEncoder.mean_center_vectors(self.vectors_topo)
        self.vectors_text = MultiModalEncoder.mean_center_vectors(self.vectors_text)
        self.vectors_program = MultiModalEncoder.mean_center_vectors(self.vectors_program)

        # Rebuild the program embedding matrix from centered vectors
        self.embeddings = np.array(
            [self.vectors_program[sid] for sid in self.student_ids],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Step 2: HNSW all-pairs search + disaggregation + GST
    # ------------------------------------------------------------------
    def extract_solution_families(
        self,
        sim_floor: float = 0.85,
        leiden_resolution: float = 1.4,
        gst_min_match: int = 8,
        **kwargs,
    ):
        """Build HNSW index, flag candidate pairs, disaggregate scores,
        generate GST evidence, and cluster via Leiden."""
        import igraph as ig
        import leidenalg as la

        num_students = len(self.student_ids)
        self.knn_graph = nx.Graph()
        self.node_to_community = {}

        if num_students < 2:
            return {f"Solution_Family_{i + 1}": [sid]
                    for i, sid in enumerate(self.student_ids)}

        for sid in self.student_ids:
            self.knn_graph.add_node(sid)

        # ── FAISS HNSW index ─────────────────────────────────────────
        embeddings_f32 = np.ascontiguousarray(self.embeddings)
        dimension = embeddings_f32.shape[1]

        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings_f32)

        # All-pairs: query every vector, retrieve all neighbours
        k_search = min(num_students, num_students)
        _, indices = index.search(embeddings_f32, k_search)

        # ── Build similarity edges ───────────────────────────────────
        for idx, student_id in enumerate(self.student_ids):
            for nb_idx in indices[idx]:
                if nb_idx == -1:
                    continue
                neighbor_id = self.student_ids[nb_idx]
                if student_id == neighbor_id:
                    continue

                # Global cosine similarity
                sim_global = float(np.dot(
                    self.vectors_program[student_id],
                    self.vectors_program[neighbor_id],
                ))
                sim_global = max(0.0, min(1.0, sim_global))

                if sim_global < sim_floor:
                    continue

                pair = tuple(sorted([student_id, neighbor_id]))
                if self.knn_graph.has_edge(pair[0], pair[1]):
                    continue

                # ── Forensic disaggregation ──────────────────────────
                sim_structural = float(np.dot(
                    self.vectors_topo[student_id],
                    self.vectors_topo[neighbor_id],
                ))
                sim_lexical = float(np.dot(
                    self.vectors_text[student_id],
                    self.vectors_text[neighbor_id],
                ))
                sim_structural = max(0.0, min(1.0, sim_structural))
                sim_lexical = max(0.0, min(1.0, sim_lexical))

                # ── GST line-level evidence ──────────────────────────
                src_a = self._read_stripped_source(student_id)
                src_b = self._read_stripped_source(neighbor_id)

                gst_tiles = []
                gst_coverage = 0.0
                if src_a.strip() and src_b.strip():
                    tokens_a = tokenize_source(src_a)
                    tokens_b = tokenize_source(src_b)
                    tiles = greedy_string_tiling(tokens_a, tokens_b, gst_min_match)
                    gst_coverage = coverage_score(tiles, len(tokens_a), len(tokens_b))
                    gst_tiles = tiles_to_json(tiles)

                self.knn_graph.add_edge(
                    pair[0], pair[1],
                    weight=sim_global,
                    sim_structural=sim_structural,
                    sim_lexical=sim_lexical,
                    gst_coverage=gst_coverage,
                    gst_tiles=gst_tiles,
                )

        # ── Leiden community detection ───────────────────────────────
        ig_graph = ig.Graph(n=num_students, directed=False)
        ig_graph.vs["name"] = self.student_ids
        name_to_vidx = {name: i for i, name in enumerate(self.student_ids)}

        edges_to_add = []
        weights_to_add = []
        for u, v, d in self.knn_graph.edges(data=True):
            edges_to_add.append((name_to_vidx[u], name_to_vidx[v]))
            weights_to_add.append(d["weight"])

        if edges_to_add:
            ig_graph.add_edges(edges_to_add)
            ig_graph.es["weight"] = weights_to_add

        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            weights="weight" if edges_to_add else None,
            resolution_parameter=leiden_resolution,
        )

        families = {}
        for comm_idx, community_nodes in enumerate(partition):
            family_name = f"Solution_Family_{comm_idx + 1}"
            members = [ig_graph.vs[n_idx]["name"] for n_idx in community_nodes]
            families[family_name] = sorted(members)

        for family_name, members in families.items():
            for node in members:
                self.node_to_community[node] = family_name

        return families

    # ------------------------------------------------------------------
    # Step 3: Build final report payload
    # ------------------------------------------------------------------
    def compute_suspicion_scores(self, families=None):
        """Build a ranked list of flagged pairs with full disaggregated scores
        and GST evidence payloads."""
        report_data = []
        seen_pairs = set()

        if self.knn_graph is None or len(self.knn_graph.edges) == 0:
            return report_data

        # Compute intra-community densities
        community_densities = {}
        if families:
            for family_name, members in families.items():
                internal_weights = []
                for i, node_a in enumerate(members):
                    for j in range(i + 1, len(members)):
                        node_b = members[j]
                        if self.knn_graph.has_edge(node_a, node_b):
                            internal_weights.append(
                                self.knn_graph[node_a][node_b]["weight"]
                            )
                community_densities[family_name] = (
                    float(np.mean(internal_weights)) if internal_weights else 0.0
                )

        for u, v, data in self.knn_graph.edges(data=True):
            if u == v:
                continue

            pair = tuple(sorted([u, v]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            similarity = data["weight"]
            sim_structural = data.get("sim_structural", similarity)
            sim_lexical = data.get("sim_lexical", similarity)
            gst_coverage = data.get("gst_coverage", 0.0)
            gst_tiles = data.get("gst_tiles", [])

            comm_u = self.node_to_community.get(u, "Isolated_Node")
            comm_v = self.node_to_community.get(v, "Isolated_Node")

            if comm_u == comm_v:
                family_label = comm_u
                family_density = community_densities.get(family_label, similarity)
            else:
                family_label = "Cross_Template_Overlap"
                family_density = similarity

            if similarity >= 0.99:
                risk_level = "CRITICAL"
            elif similarity >= 0.94:
                risk_level = "HIGH"
            else:
                risk_level = "SUSPICIOUS"

            report_data.append({
                "family": family_label,
                "student_a": pair[0],
                "student_b": pair[1],
                "similarity": similarity,
                "sim_structural": sim_structural,
                "sim_lexical": sim_lexical,
                "gst_coverage": gst_coverage,
                "gst_tiles": gst_tiles,
                "family_density": family_density,
                "risk_level": risk_level,
                "community_a": comm_u,
                "community_b": comm_v,
            })

        report_data.sort(key=lambda x: x["similarity"], reverse=True)
        return report_data

    # ------------------------------------------------------------------
    # Private helpers: source / comment file reading
    # ------------------------------------------------------------------
    def _read_stripped_source(self, student_id: str) -> str:
        """Try to read the comment-stripped source file.  Falls back to the
        original submission if the stripped version doesn't exist."""
        stripped = JoernAutomationParser.get_stripped_source_path(
            self.workspace_dir, student_id
        )
        if os.path.isfile(stripped):
            with open(stripped, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        # Fallback: try common extensions in the original submissions dir
        for ext in (".java", ".c", ".cpp", ".py"):
            orig = os.path.join(self.source_dir, f"{student_id}{ext}")
            if os.path.isfile(orig):
                with open(orig, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()

        return ""

    def _read_comments(self, student_id: str) -> str:
        """Read the isolated comment stream sidecar file."""
        path = JoernAutomationParser.get_comments_path(
            self.workspace_dir, student_id
        )
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        return ""
