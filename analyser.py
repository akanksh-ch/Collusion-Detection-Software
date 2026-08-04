"""HNSW Indexing, Forensic Disaggregation, GST Evidence, & Clustering.

Implements Module 4 of the production specification:
  1. FAISS Exact all-pairs similarity search on V_program
  2. Forensic disaggregation: independent cos(V_topo), cos(V_text) per pair
  3. Greedy String Tiling line-level evidence generation (always runs —
     this feeds the pairwise report/suspicion scores regardless of which
     clustering method is chosen below, and is NOT skipped by
     cluster_method="hdbscan")
  4. Family/community assignment via either Leiden (on the fused
     GST/embedding graph) or HDBSCAN (on graph2vec + TF-IDF stylometric
     embeddings) — selectable via cluster_method
  5. Compressed persistence of forensics payloads
"""

from __future__ import annotations

import os
import time
import logging
import gzip
import json
import numpy as np
import networkx as nx
import faiss
from concurrent.futures import ProcessPoolExecutor, as_completed

from gnn import MultiModalEncoder
from gst import tokenize_source, greedy_string_tiling, merge_neighboring_tiles, coverage_score, tiles_to_json
from parse import JoernAutomationParser


def _gst_pair_worker(args):
    """Run GST (+ optional match-merging) for a single candidate pair.

    Executed in a worker process. Pulled to module level (rather than a
    method) so it can be pickled and shipped to a ProcessPoolExecutor —
    mirrors the existing worker pattern in loader.py/parse.py.

    Each pair's tiling is fully independent given the pre-tokenized
    inputs, so this was previously the dominant *serial* cost on larger
    corpora (pairs scale as O(n^2), e.g. ~9x more pairs on a
    134-submission corpus than a 45-submission one) despite being
    embarrassingly parallel.
    """
    (a_id, b_id, tokens_a, tokens_b, gst_min_match, match_merging,
     gap_size, neighbor_length, store_gst_tiles) = args

    pair_start = time.monotonic()
    tiles = greedy_string_tiling(
        tokens_a, tokens_b, gst_min_match,
        pair_label=f"{a_id} vs {b_id}",
    )
    pair_elapsed = time.monotonic() - pair_start
    if pair_elapsed > 1.0:
        logging.warning(
            f"[GST] slow pair {a_id} vs {b_id} "
            f"(|A|={len(tokens_a)}, |B|={len(tokens_b)}) took {pair_elapsed:.1f}s"
        )

    if match_merging:
        tiles = merge_neighboring_tiles(
            tiles, tokens_a, tokens_b,
            max_gap=gap_size, min_neighbor_length=neighbor_length,
        )

    gst_coverage = coverage_score(tiles, len(tokens_a), len(tokens_b))
    gst_tiles = tiles_to_json(tiles) if store_gst_tiles else []

    return a_id, b_id, gst_coverage, gst_tiles


class CohortAnalyzer:
    """Orchestrates spatial profiling, disaggregated forensic scoring,
    GST evidence generation, and family/community assignment (Leiden or
    HDBSCAN)."""

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

        # graph2vec / TF-IDF stylometric vectors — populated by
        # generate_structural_stylometric_embeddings(), used only for
        # cluster_method="hdbscan". Independent of the GNN encoder
        # above; never touches self.embeddings or the GST gate.
        self.vectors_g2v: dict[str, np.ndarray] = {}
        self.vectors_stylo: dict[str, np.ndarray] = {}

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
    # Step 1b: graph2vec (structural) + TF-IDF/SVD (stylometric)
    # ------------------------------------------------------------------
    def generate_structural_stylometric_embeddings(self):
        """graph2vec (structural) + TF-IDF/SVD (stylometric) embeddings.

        Independent of generate_all_embeddings() / the GNN encoder —
        does not touch self.embeddings. Populates self.vectors_g2v /
        self.vectors_stylo, consumed by extract_solution_families()
        when cluster_method="hdbscan", and also exported to .npz by
        main.py for offline analysis if requested.

        Must be called before extract_solution_families(cluster_method=
        "hdbscan", ...) — that method will call this automatically if
        the vectors aren't populated yet, but calling it explicitly
        lets you also export it via --export-embeddings without paying
        for it twice.
        """
        import networkx as nx
        from karateclub import Graph2Vec
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        # ---- structural: CPGGraph -> networkx --------------------------
        nx_graphs = []
        valid_ids = []
        skipped = []
        for sid in self.student_ids:
            cpg = self.cohort_data.get(sid)
            if cpg is None or cpg.edge_index.numel() == 0:
                skipped.append(sid)
                continue
            n_nodes = cpg.x.shape[0]
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            for i in range(n_nodes):
                G.nodes[i]["feature"] = int(cpg.x[i, 0])  # node_type_idx
            edges = cpg.edge_index.t().tolist()
            G.add_edges_from((u, v) for u, v in edges if u != v)
            nx_graphs.append(G)
            valid_ids.append(sid)

        if skipped:
            logging.warning(
                f"[graph2vec] {len(skipped)} submissions had empty/missing "
                f"CPGs and were excluded: {skipped[:5]}{'...' if len(skipped) > 5 else ''}"
            )

        if len(nx_graphs) < 2:
            logging.warning("[graph2vec] fewer than 2 valid graphs — skipping structural embeddings")
            self.vectors_g2v = {}
        else:
            g2v = Graph2Vec(
                dimensions=128,
                wl_iterations=2,
                attributed=True,  # use the "feature" node attribute above
                workers=max(1, (os.cpu_count() or 2) - 1),
            )
            g2v.fit(nx_graphs)
            g2v_vecs = g2v.get_embedding()
            self.vectors_g2v = {sid: g2v_vecs[i].astype(np.float32) for i, sid in enumerate(valid_ids)}
            logging.info(f"[graph2vec] embedded {len(valid_ids)} submissions, dim=128")

        # ---- stylometric: TF-IDF over stripped source, SVD to fixed dim -
        stripped_texts = []
        text_ids = []
        for sid in self.student_ids:
            src_dict = self._read_stripped_source(sid)
            text = "\n".join(src_dict.values())
            if text.strip():
                stripped_texts.append(text)
                text_ids.append(sid)

        if len(stripped_texts) < 2:
            logging.warning("[stylo] fewer than 2 non-empty sources — skipping stylometric embeddings")
            self.vectors_stylo = {}
        else:
            tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(stripped_texts)
            n_components = min(128, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
            svd = TruncatedSVD(n_components=n_components)
            stylo_vecs = svd.fit_transform(tfidf_matrix).astype(np.float32)
            self.vectors_stylo = {sid: stylo_vecs[i] for i, sid in enumerate(text_ids)}
            logging.info(f"[stylo] embedded {len(text_ids)} submissions, dim={n_components}")

    # ------------------------------------------------------------------
    # Family assignment via HDBSCAN on graph2vec + stylometric vectors
    # ------------------------------------------------------------------
    def _cluster_families_hdbscan(
        self,
        structural_weight: float = 0.7,
        stylometric_weight: float = 0.3,
        min_cluster_size: int = 2,
        min_samples: int | None = None,
    ) -> dict[str, list[str]]:
        """Build families by running HDBSCAN over a weighted combination
        of graph2vec + TF-IDF stylometric embeddings, instead of Leiden
        over the fused GST/embedding graph.

        This does NOT touch self.knn_graph or GST evidence — that graph
        is still built by extract_solution_families() from GST/embedding
        pair scores and is still what compute_suspicion_scores() reads
        for the pairwise report. Only family/community *assignment*
        changes.

        Students missing from both embedding spaces (e.g. empty CPG AND
        empty source — should be rare) are assigned their own singleton
        family rather than silently dropped, so every student_id still
        ends up in exactly one family.
        """
        from sklearn.cluster import HDBSCAN
        from sklearn.metrics.pairwise import cosine_distances
        from sklearn.preprocessing import normalize

        if not self.vectors_g2v and not self.vectors_stylo:
            logging.info(
                "[HDBSCAN] no structural/stylometric vectors present — generating now"
            )
            self.generate_structural_stylometric_embeddings()

        covered_ids = [
            sid for sid in self.student_ids
            if sid in self.vectors_g2v or sid in self.vectors_stylo
        ]
        uncovered_ids = [sid for sid in self.student_ids if sid not in covered_ids]

        if len(covered_ids) < 2:
            logging.warning(
                "[HDBSCAN] fewer than 2 submissions have structural/stylometric "
                "vectors — falling back to all-singleton families"
            )
            families = {f"Solution_Family_{i + 1}": [sid] for i, sid in enumerate(self.student_ids)}
            for family_name, members in families.items():
                for node in members:
                    self.node_to_community[node] = family_name
            return families

        blocks = []
        total_weight = 0.0
        if self.vectors_g2v:
            g2v_mat = np.stack([
                self.vectors_g2v.get(sid, np.zeros(next(iter(self.vectors_g2v.values())).shape))
                for sid in covered_ids
            ])
            blocks.append(normalize(g2v_mat) * structural_weight)
            total_weight += structural_weight
        if self.vectors_stylo:
            stylo_mat = np.stack([
                self.vectors_stylo.get(sid, np.zeros(next(iter(self.vectors_stylo.values())).shape))
                for sid in covered_ids
            ])
            blocks.append(normalize(stylo_mat) * stylometric_weight)
            total_weight += stylometric_weight

        combined = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        dist = cosine_distances(combined)
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, None)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(dist)

        families: dict[str, list[str]] = {}
        noise_count = 0
        next_singleton_idx = 1
        for sid, label in zip(covered_ids, labels):
            if label == -1:
                family_name = f"Solution_Family_singleton_{next_singleton_idx}"
                next_singleton_idx += 1
                noise_count += 1
            else:
                family_name = f"Solution_Family_{int(label) + 1}"
            families.setdefault(family_name, []).append(sid)

        for sid in uncovered_ids:
            family_name = f"Solution_Family_singleton_{next_singleton_idx}"
            next_singleton_idx += 1
            families[family_name] = [sid]

        logging.info(
            f"[HDBSCAN] {len([f for f, m in families.items() if len(m) > 1])} multi-member "
            f"families formed from {len(covered_ids)} embedded submissions "
            f"({noise_count} labeled noise -> singletons), "
            f"{len(uncovered_ids)} uncovered submissions added as singletons"
        )

        for family_name, members in families.items():
            for node in sorted(members):
                self.node_to_community[node] = family_name
        families = {k: sorted(v) for k, v in families.items()}
        return families

    # ------------------------------------------------------------------
    # Step 2: Exact all-pairs search + disaggregation + GST + families
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
        gst_gate_threshold: float = 0.35,
        cluster_method: str = "leiden",
        hdbscan_structural_weight: float = 0.7,
        hdbscan_stylometric_weight: float = 0.3,
        hdbscan_min_cluster_size: int = 2,
        **kwargs,
    ):
        """Build Exact index, flag candidate pairs, disaggregate scores,
        generate GST evidence, and assign families via Leiden or HDBSCAN.

        cluster_method
        --------------
        "leiden" (default): community detection on the fused GST/
          embedding graph (self.knn_graph), as before.
        "hdbscan": families come from HDBSCAN over graph2vec + TF-IDF
          stylometric embeddings instead (see _cluster_families_hdbscan).
          GST computation below is completely unaffected either way —
          it always runs, because compute_suspicion_scores() and the
          JPlag-format report both depend on self.knn_graph carrying
          gst_coverage / gst_tiles per pair regardless of which
          clustering method assigns families. Only the family-assignment
          step at the bottom of this method branches.

        Edge weight fusion (adaptive per-pair gate) — always computed,
        used both to build self.knn_graph (read by
        compute_suspicion_scores for every risk_level / gst_tiles /
        similarity in the report) and, when cluster_method="leiden", as
        the input to Leiden itself:
          - gst_coverage >= gst_gate_threshold: trust GST outright, use
            gst_coverage as the edge weight.
          - gst_coverage <  gst_gate_threshold: fall back to the
            embedding cosine similarity (sim_global), since a heavily
            refactored Type-3/4 clone can have near-zero token overlap
            but still be caught structurally.
        """
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

        # sim_floor == 0.0 is exclusively the all-pairs AUC/embedding-export
        # run — it only needs the scalar gst_coverage for
        # evaluate_pipeline.py, never the full forensic tile payload.
        # The sim_floor>0 clustering/report run still gets full tiles,
        # since that's what the forensic diff sidebar actually reads.
        store_gst_tiles = sim_floor > 0.0
        if not store_gst_tiles:
            logging.info(
                "[GST] sim_floor=0.0 (all-pairs eval run) — skipping full "
                "tile JSON serialization, keeping only scalar gst_coverage, "
                "to avoid unbounded memory growth over many pairs."
            )

        # ── Pre-tokenize every submission exactly once ────────────────
        tokenize_start = time.monotonic()
        token_cache: dict[str, list] = {}
        for sid in self.student_ids:
            src = self._read_stripped_source(sid)
            if src:
                token_cache[sid] = tokenize_source(src)
        logging.info(
            f"[GST] pre-tokenized {len(token_cache)}/{num_students} "
            f"submissions in {time.monotonic() - tokenize_start:.1f}s "
            f"(reused across all candidate pairs below)"
        )

        # ── Phase A: dedupe candidate pairs + cheap embedding sims ─────
        pair_info: dict[tuple[str, str], dict] = {}
        for idx, student_id in enumerate(self.student_ids):
            for nb_idx in indices[idx]:
                if nb_idx == -1:
                    continue
                neighbor_id = self.student_ids[nb_idx]
                if student_id == neighbor_id:
                    continue

                pair = tuple(sorted([student_id, neighbor_id]))
                if pair in pair_info:
                    continue

                sim_global = float(np.dot(
                    self.vectors_program[student_id],
                    self.vectors_program[neighbor_id],
                ))
                sim_structural = float(np.dot(
                    self.vectors_topo[student_id],
                    self.vectors_topo[neighbor_id],
                ))
                sim_lexical = float(np.dot(
                    self.vectors_text[student_id],
                    self.vectors_text[neighbor_id],
                ))

                pair_info[pair] = {
                    "sim_global": max(0.0, min(1.0, sim_global)),
                    "sim_structural": max(0.0, min(1.0, sim_structural)),
                    "sim_lexical": max(0.0, min(1.0, sim_lexical)),
                    "gst_coverage": 0.0,
                    "gst_tiles": [],
                }

        total_candidate_pairs = len(pair_info)
        logging.info(f"[GST] {total_candidate_pairs} unique candidate pairs to score")

        # ── Phase B: GST across a process pool (always runs — see
        # cluster_method docstring above; this is unaffected by which
        # family-assignment method is selected below) ──────────────────
        gst_tasks = []
        for pair in pair_info:
            tokens_a = token_cache.get(pair[0])
            tokens_b = token_cache.get(pair[1])
            if tokens_a and tokens_b:
                gst_tasks.append((
                    pair[0], pair[1], tokens_a, tokens_b, gst_min_match,
                    match_merging, gap_size, neighbor_length, store_gst_tiles,
                ))

        max_workers = max(1, (os.cpu_count() or 2) - 1)
        LOG_EVERY = max(1, len(gst_tasks) // 20)  # ~20 log lines total
        pairs_seen = 0
        loop_start = time.monotonic()

        if gst_tasks:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_gst_pair_worker, task): task for task in gst_tasks}
                for future in as_completed(futures):
                    a_id, b_id, gst_coverage, gst_tiles = future.result()
                    entry = pair_info[(a_id, b_id)]
                    entry["gst_coverage"] = gst_coverage
                    entry["gst_tiles"] = gst_tiles

                    pairs_seen += 1
                    if pairs_seen % LOG_EVERY == 0 or pairs_seen == len(gst_tasks):
                        elapsed = time.monotonic() - loop_start
                        rate = pairs_seen / elapsed if elapsed > 0 else 0.0
                        remaining = len(gst_tasks) - pairs_seen
                        eta = remaining / rate if rate > 0 else float("inf")
                        logging.info(
                            f"[GST/fusion progress] {pairs_seen}/{len(gst_tasks)} "
                            f"pairs ({100 * pairs_seen / len(gst_tasks):.1f}%) — "
                            f"{elapsed:.1f}s elapsed, {rate:.1f} pairs/s, "
                            f"ETA {eta:.0f}s ({max_workers} worker processes)"
                        )

        # ── Phase C: adaptive confidence gate + sim_floor + build graph ─
        for pair, data in pair_info.items():
            gst_coverage = data["gst_coverage"]
            sim_global = data["sim_global"]

            if gst_coverage >= gst_gate_threshold:
                fused_weight = gst_coverage
                gate_source = "gst"
            else:
                fused_weight = sim_global
                gate_source = "embedding"

            if fused_weight < sim_floor:
                continue

            self.knn_graph.add_edge(
                pair[0], pair[1],
                weight=fused_weight,
                sim_global=sim_global,
                sim_structural=data["sim_structural"],
                sim_lexical=data["sim_lexical"],
                gst_coverage=gst_coverage,
                gst_tiles=data["gst_tiles"],
                gate_source=gate_source,
            )

        # ── Family assignment ────────────────────────────────────────
        if cluster_skip:
            families = {}
            for i, sid in enumerate(self.student_ids):
                families[f"Solution_Family_{i + 1}"] = [sid]
            for family_name, members in families.items():
                for node in members:
                    self.node_to_community[node] = family_name
            return families

        if cluster_method == "hdbscan":
            logging.info(
                "[Families] cluster_method='hdbscan' — assigning families via "
                "graph2vec + TF-IDF stylometric embeddings (GST/JPlag report "
                "data above is unaffected)"
            )
            return self._cluster_families_hdbscan(
                structural_weight=hdbscan_structural_weight,
                stylometric_weight=hdbscan_stylometric_weight,
                min_cluster_size=hdbscan_min_cluster_size,
            )

        # cluster_method == "leiden" (default) — original behavior
        import igraph as ig
        import leidenalg as la

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
        and GST evidence payloads. Unaffected by cluster_method — reads
        self.knn_graph (GST/embedding-gated edges, always built the same
        way) and self.node_to_community (which family/community
        assignment produced, whichever method was used)."""
        report_data = []
        seen_pairs = set()

        if self.knn_graph is None or len(self.knn_graph.edges) == 0:
            return report_data

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
            sim_global = data.get("sim_global", similarity)
            sim_structural = data.get("sim_structural", similarity)
            sim_lexical = data.get("sim_lexical", similarity)
            gst_coverage = data.get("gst_coverage", 0.0)
            gst_tiles = data.get("gst_tiles", [])
            gate_source = data.get("gate_source", "embedding")

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
                "sim_global": sim_global,
                "gate_source": gate_source,
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
        logging.info("Extracting solution families...")
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

        return self._read_original_source(student_id)

    def _read_original_source(self, student_id: str) -> dict[str, str]:
        source_dirs = self.source_dir if isinstance(self.source_dir, list) else [self.source_dir]

        for src_dir in source_dirs:
            orig_dir = os.path.join(src_dir, student_id)
            if os.path.isdir(orig_dir):
                return self._read_dir_files(orig_dir, orig_dir)

            root_name = os.path.basename(os.path.normpath(src_dir))
            prefix = f"{root_name}_"

            if student_id.startswith(prefix):
                real_id = student_id[len(prefix):]
                orig_dir_2 = os.path.join(src_dir, real_id)
                if os.path.isdir(orig_dir_2):
                    return self._read_dir_files(orig_dir_2, orig_dir_2)

        for src_dir in source_dirs:
            for ext in (".java", ".c", ".cpp", ".py"):
                orig = os.path.join(src_dir, f"{student_id}{ext}")
                if os.path.isfile(orig):
                    with open(orig, "r", encoding="utf-8", errors="replace") as f:
                        return {os.path.basename(orig): f.read()}

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
        basename_to_relpath = {}

        for root, _, files in os.walk(directory):
            for f in sorted(files):
                if f.startswith('.') or f.lower().endswith(IGNORED_EXTENSIONS):
                    continue
                path = os.path.join(root, f)
                rel_path = os.path.relpath(path, base_dir)
                rel_path = rel_path.replace(os.sep, "/")

                basename = os.path.basename(rel_path)

                with open(path, "r", encoding="utf-8", errors="replace") as file:
                    content = file.read()

                if basename in files_dict:
                    old_rel_path = basename_to_relpath[basename]
                    if old_rel_path != basename:
                        files_dict[old_rel_path] = files_dict.pop(basename)

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
