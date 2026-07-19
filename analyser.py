"""HNSW Indexing, Forensic Disaggregation, GST Evidence, & Leiden Clustering.

Implements Module 4 of the production specification:
  1. FAISS Exact all-pairs similarity search on V_program
  2. Forensic disaggregation: independent cos(V_topo), cos(V_text) per pair
  3. Greedy String Tiling line-level evidence generation
  4. Leiden community detection for collusion-network partitioning
  5. Compressed persistence of forensics payloads
"""

from __future__ import annotations

import os
import logging
import gzip
import json
import numpy as np
import networkx as nx
import faiss

from gnn import MultiModalEncoder
from gst import tokenize_source, greedy_string_tiling, merge_neighboring_tiles, coverage_score, tiles_to_json
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

        Calls ``encoder.fit_cohort()`` to fit the text encoder AND
        freeze All-But-The-Top calibration parameters (cohort mean and
        top-K principal components, with K determined dynamically from
        the singular value spectrum).

        Subsequent ``encode_submission()`` calls apply those frozen
        parameters automatically — no per-batch refitting occurs.

        The 0.85 similarity threshold used downstream is *not* derived
        from calibration; it was tuned empirically via precision-recall
        analysis against the labelled ground-truth corpus.
        """
        if not self.student_ids:
            raise ValueError("Cannot generate embeddings: Cohort dataset is empty.")

        # Collect stripped source texts and comment streams
        stripped_texts = []
        comment_texts = []
        for sid in self.student_ids:
            src_dict = self._read_stripped_source(sid)
            concat_src = "\n\n".join([f"// --- {f} ---\n{content}" for f, content in src_dict.items()])
            stripped_texts.append(concat_src)
            comment_texts.append(self._read_comments(sid))

        # Fit text encoder (TF-IDF + SVD) AND freeze isotropic
        # calibration parameters (All-But-The-Top: mean + top-K PCs)
        self.encoder.fit_cohort(self.cohort_data, stripped_texts, comment_texts)

        # Re-encode every submission with calibration now active
        for idx, sid in enumerate(self.student_ids):
            pyg_data = self.cohort_data[sid]
            vecs = self.encoder.encode_submission(
                pyg_data, stripped_texts[idx], comment_texts[idx]
            )
            self.vectors_topo[sid] = vecs["v_topo"]
            self.vectors_text[sid] = vecs["v_text"]
            self.vectors_program[sid] = vecs["v_program"]

        # Build the program embedding matrix from calibrated vectors
        self.embeddings = np.array(
            [self.vectors_program[sid] for sid in self.student_ids],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Step 2: Exact all-pairs search + disaggregation + GST
    # ------------------------------------------------------------------
    def extract_solution_families(
        self,
        sim_floor: float = 0.85,
        leiden_resolution: float = 1.4,
        gst_min_match: int = 9,
        match_merging: bool = True,
        gap_size: int = 6,
        neighbor_length: int = 2,
        cluster_skip: bool = False,
        **kwargs,
    ):
        """Build Exact index, flag candidate pairs, disaggregate scores,
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

        # ── FAISS Exact Index ─────────────────────────────────────────
        embeddings_f32 = np.ascontiguousarray(self.embeddings)
        dimension = embeddings_f32.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_f32)

        # All-pairs: query every vector against the entire cohort
        k_search = num_students
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
                # Use pair[0]/pair[1] (sorted) so a_file/b_file align
                # with student_a/student_b in the report output.
                src_a = self._read_stripped_source(pair[0])
                src_b = self._read_stripped_source(pair[1])

                gst_tiles = []
                gst_coverage = 0.0
                if src_a and src_b:
                    tokens_a = tokenize_source(src_a)
                    tokens_b = tokenize_source(src_b)
                    tiles = greedy_string_tiling(tokens_a, tokens_b, gst_min_match)
                    if match_merging:
                        tiles = merge_neighboring_tiles(
                            tiles, tokens_a, tokens_b,
                            max_gap=gap_size,
                            min_neighbor_length=neighbor_length,
                        )
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
        if cluster_skip:
            # Assign each student to their own family
            families = {}
            for i, sid in enumerate(self.student_ids):
                families[f"Solution_Family_{i + 1}"] = [sid]
            for family_name, members in families.items():
                for node in members:
                    self.node_to_community[node] = family_name
            return families

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

            if family_label == "Cross_Template_Overlap":
                risk_level = "SUSPICIOUS"
            elif similarity >= 0.99:
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
    # Step 4: Standalone Forensic Archive Exporter
    # ------------------------------------------------------------------
    def save_results_archive(self, output_path: str = "results.json.gz", **kwargs):
        """Orchestrates pipeline execution, structures results, embeds raw source
        strings, and exports an optimized compressed JSON file to disk."""
        logging.info("Extracting solution families via Leiden...")
        families = self.extract_solution_families(**kwargs)
        
        logging.info("Computing metrics and extracting line-level matching tiles...")
        report_data = self.compute_suspicion_scores(families=families)
        
        logging.info("Mapping and serializing source files for offline viewer compatibility...")
        source_texts = {sid: self._read_original_source(sid) for sid in self.student_ids}

        archive_payload = {
            "metadata": {
                "total_submissions": len(self.student_ids),
                "flagged_pairs_count": len(report_data),
                "isolated_clusters_count": len(families)
            },
            "families": families,
            "report_data": report_data,
            "source_texts": source_texts
        }

        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(archive_payload, f, indent=2)
            
        logging.info(f"Successfully flushed pipeline metrics archive to: {output_path}")

    # ------------------------------------------------------------------
    # Private helpers: source / comment file reading
    # ------------------------------------------------------------------
    def _read_stripped_source(self, student_id: str) -> dict[str, str]:
        """Try to read the comment-stripped source file. Falls back to the
        original submission if the stripped version doesn't exist."""
        stripped_base = os.path.join(self.workspace_dir, "_stripped", student_id)
        if os.path.isdir(stripped_base):
            return self._read_dir_files(stripped_base, stripped_base)
            
        stripped = JoernAutomationParser.get_stripped_source_path(
            self.workspace_dir, student_id
        )
        if os.path.isfile(stripped):
            with open(stripped, "r", encoding="utf-8", errors="replace") as f:
                return {os.path.basename(stripped): f.read()}

        # Fallback: try original submissions dir
        return self._read_original_source(student_id)

    def _read_original_source(self, student_id: str) -> dict[str, str]:
        source_dirs = self.source_dir if isinstance(self.source_dir, list) else [self.source_dir]
        
        for src_dir in source_dirs:
            # 1. Check direct match (for single root runs)
            orig_dir = os.path.join(src_dir, student_id)
            if os.path.isdir(orig_dir):
                return self._read_dir_files(orig_dir, orig_dir)
            
            # 2. Check prefixed match (for multi-root runs e.g. "orig_o1-wqfqn")
            root_name = os.path.basename(os.path.normpath(src_dir))
            prefix = f"{root_name}_"
            
            if student_id.startswith(prefix):
                real_id = student_id[len(prefix):]
                orig_dir_2 = os.path.join(src_dir, real_id)
                if os.path.isdir(orig_dir_2):
                    return self._read_dir_files(orig_dir_2, orig_dir_2)

        # 3. Fallback check for single files
        for src_dir in source_dirs:
            for ext in (".java", ".c", ".cpp", ".py"):
                orig = os.path.join(src_dir, f"{student_id}{ext}")
                if os.path.isfile(orig):
                    with open(orig, "r", encoding="utf-8", errors="replace") as f:
                        return {os.path.basename(orig): f.read()}
                        
                # Check unpacked file name
                root_name = os.path.basename(os.path.normpath(src_dir))
                prefix = f"{root_name}_"
                if student_id.startswith(prefix):
                    real_id = student_id[len(prefix):]
                    orig2 = os.path.join(src_dir, f"{real_id}{ext}")
                    if os.path.isfile(orig2):
                        with open(orig2, "r", encoding="utf-8", errors="replace") as f:
                            return {os.path.basename(orig2): f.read()}

        return {}

    def _read_dir_files(self, directory: str, base_dir: str) -> dict[str, str]:
        IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml',
                              '.html', '.gitignore', '.class')
        files_dict = {}
        # Keep track of where a basename originally came from to fix collisions
        basename_to_relpath = {}
        
        for root, _, files in os.walk(directory):
            for f in sorted(files):
                if f.startswith('.') or f.lower().endswith(IGNORED_EXTENSIONS):
                    continue
                path = os.path.join(root, f)
                rel_path = os.path.relpath(path, base_dir)
                rel_path = rel_path.replace(os.sep, "/")
                
                # JPlag viewer prefers flat file names (just the basename)
                basename = os.path.basename(rel_path)
                
                with open(path, "r", encoding="utf-8", errors="replace") as file:
                    content = file.read()
                    
                # Handle name collisions
                if basename in files_dict:
                    # A collision occurred! Revert the previous one to its full path
                    old_rel_path = basename_to_relpath[basename]
                    if old_rel_path != basename: # don't overwrite if it was already flat
                        files_dict[old_rel_path] = files_dict.pop(basename)
                    
                    # Store the new one with its full path
                    files_dict[rel_path] = content
                else:
                    files_dict[basename] = content
                    basename_to_relpath[basename] = rel_path
                    
        return files_dict

    def _read_comments(self, student_id: str) -> str:
        """Read the isolated comment stream sidecar file."""
        path = JoernAutomationParser.get_comments_path(
            self.workspace_dir, student_id
        )
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        return ""
