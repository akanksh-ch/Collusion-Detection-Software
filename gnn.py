"""Dual-track submission encoder: graph2vec (topological) + TF-IDF/SVD (lexical),
fused via L2-normalise + concatenate.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from karateclub import Graph2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ══════════════════════════════════════════════════════════════════════
# 1. Topological Track — graph2vec (karateclub)
# ══════════════════════════════════════════════════════════════════════

class GraphEncoder:
    """Wraps karateclub's Graph2Vec so it plugs into the existing
    encode_submission / MultiModalEncoder call sites.

    Graph2Vec needs the full cohort up front (fit_corpus), then returns
    per-graph embeddings either from the fitted model (in-sample) or
    via .infer() for graphs seen after fitting (out-of-sample).
    """

    def __init__(
        self,
        node_type_vocab_size: int = 0,
        style_vocab_size: int = 5,
        edge_vocab_size: int = 10,
        output_dim: int = 128,
        wl_iterations: int = 3,
        epochs: int = 100,
        min_count: int = 1,
        seed: int = 90,
    ):
        # node_type_vocab_size / style_vocab_size / edge_vocab_size kept
        # for call-site compatibility, unused (Graph2Vec hashes node
        # features directly, no fixed vocab table needed).
        self.output_dim = output_dim
        self._model = Graph2Vec(
            wl_iterations=wl_iterations,
            attributed=True,
            dimensions=output_dim,
            epochs=epochs,
            min_count=min_count,
            seed=seed,
        )
        self._fitted = False
        # graph identity -> index into the fitted corpus, so re-encoding
        # a cohort graph returns its trained vector instead of re-inferring.
        self._index_by_key: dict[int, int] = {}
        self._trained_vectors: np.ndarray | None = None

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    @staticmethod
    def _to_networkx(x: np.ndarray, edge_index: np.ndarray) -> nx.Graph:
        """Build an undirected nx.Graph with a string "feature" attribute
        per node (Graph2Vec's attributed mode expects this key)."""
        g = nx.Graph()
        for i, (node_type, style) in enumerate(x):
            g.add_node(i, feature=f"{int(node_type)}_{int(style)}")
        if edge_index.size:
            for s, d in zip(edge_index[0], edge_index[1]):
                g.add_edge(int(s), int(d))
        return g

    @staticmethod
    def _graph_key(x: np.ndarray, edge_index: np.ndarray) -> int:
        return hash((x.tobytes(), edge_index.tobytes()))

    def fit_corpus(self, graphs: list[dict]) -> None:
        """Fit Graph2Vec once over the whole cohort.

        Args:
            graphs: list of {"x": ndarray[N,2], "edge_index": ndarray[2,E]}
        """
        nx_graphs = []
        keys = []
        for g in graphs:
            x_np = _to_numpy(g["x"]).astype(np.int64)
            edge_index_np = _to_numpy(g["edge_index"]).astype(np.int64)
            nx_graphs.append(self._to_networkx(x_np, edge_index_np))
            keys.append(self._graph_key(x_np, edge_index_np))

        logging.info(f"[graph2vec] fitting Graph2Vec over {len(nx_graphs)} graphs")
        self._model.fit(nx_graphs)
        self._trained_vectors = self._model.get_embedding()
        self._index_by_key = {k: i for i, k in enumerate(keys)}
        self._fitted = True

    def __call__(self, x, edge_index, edge_attr=None, batch=None, debug: bool = False):
        x_np = _to_numpy(x).astype(np.int64)
        edge_index_np = _to_numpy(edge_index).astype(np.int64)

        if not self._fitted:
            raise RuntimeError("GraphEncoder.fit_corpus(...) must be called before encoding.")

        key = self._graph_key(x_np, edge_index_np)
        idx = self._index_by_key.get(key)
        if idx is not None:
            vec = self._trained_vectors[idx]
        else:
            g = self._to_networkx(x_np, edge_index_np)
            vec = self._model.infer([g])[0]

        if debug:
            print(f"[graph2vec] nodes={x_np.shape[0]} edges={edge_index_np.shape[1]} "
                  f"in_sample={idx is not None} norm={np.linalg.norm(vec):.4f}")

        return torch.from_numpy(np.asarray(vec, dtype=np.float32)).unsqueeze(0)


def _to_numpy(t) -> np.ndarray:
    if t is None:
        return np.empty(0)
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


# ══════════════════════════════════════════════════════════════════════
# 2. Lexical Track — char n-gram TF-IDF + layout metrics, SVD-projected
# ══════════════════════════════════════════════════════════════════════

class TextEncoder:
    def __init__(self, output_dim: int = 128, max_tfidf_features: int = 256):
        self.output_dim = output_dim
        self.max_tfidf_features = max_tfidf_features
        self._tfidf_disabled = False
        self._svd_fitted = False

        self.char_tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=max_tfidf_features,
        )

        self._layout_dim = 3
        self._raw_dim = max_tfidf_features + self._layout_dim
        self._svd_dim = min(output_dim, self._raw_dim - 1)
        self._svd = TruncatedSVD(n_components=self._svd_dim, random_state=42)

    def fit_cohort(self, stripped_texts: list[str], comment_texts: list[str]):
        clean = [t for t in stripped_texts if t.strip()]
        if not clean:
            self._tfidf_disabled = True
            return

        try:
            self.char_tfidf.fit(clean)
        except ValueError:
            logging.warning("[TF-IDF] empty corpus, disabling")
            self._tfidf_disabled = True
            return

        raw_matrix = np.array(
            [self._extract_raw_features(s, c) for s, c in zip(stripped_texts, comment_texts)],
            dtype=np.float64,
        )

        n_samples = raw_matrix.shape[0]
        effective_dim = min(self._svd_dim, n_samples - 1, raw_matrix.shape[1] - 1)
        if effective_dim < 1:
            logging.warning("[SVD] not enough samples, using raw features")
            return

        self._svd = TruncatedSVD(n_components=effective_dim, random_state=42)
        self._svd.fit(raw_matrix)
        self._svd_fitted = True

    def encode(self, stripped_source: str, comment_text: str = "") -> np.ndarray:
        raw_vec = self._extract_raw_features(stripped_source, comment_text)

        if self._svd_fitted:
            projected = self._svd.transform(raw_vec.reshape(1, -1)).flatten()
        else:
            projected = raw_vec[: self.output_dim]

        if len(projected) < self.output_dim:
            projected = np.pad(projected, (0, self.output_dim - len(projected)))

        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        return projected.astype(np.float32)

    def _extract_raw_features(self, stripped_source: str, comment_text: str) -> np.ndarray:
        if stripped_source.strip() and not self._tfidf_disabled and hasattr(self.char_tfidf, "vocabulary_"):
            try:
                tfidf_vec = self.char_tfidf.transform([stripped_source]).toarray().flatten()
            except ValueError:
                tfidf_vec = np.zeros(self.max_tfidf_features)
        else:
            tfidf_vec = np.zeros(self.max_tfidf_features)

        if len(tfidf_vec) < self.max_tfidf_features:
            tfidf_vec = np.pad(tfidf_vec, (0, self.max_tfidf_features - len(tfidf_vec)))

        lines = stripped_source.splitlines()
        n_lines = max(len(lines), 1)
        avg_line_len = np.mean([len(l) for l in lines]) if lines else 0.0
        comment_ratio = len(comment_text) / max(len(stripped_source), 1)
        blank_ratio = sum(1 for l in lines if not l.strip()) / n_lines

        return np.concatenate([tfidf_vec, np.array([avg_line_len, comment_ratio, blank_ratio])])


# ══════════════════════════════════════════════════════════════════════
# 3. Fusion — per-channel L2-normalise, concatenate, re-normalise
# ══════════════════════════════════════════════════════════════════════

class ConcatFusion(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, v_topo: torch.Tensor, v_text: torch.Tensor) -> torch.Tensor:
        v_topo_hat = torch.nn.functional.normalize(v_topo, p=2, dim=-1)
        v_text_hat = torch.nn.functional.normalize(v_text, p=2, dim=-1)
        v_program = torch.cat([v_topo_hat, v_text_hat], dim=-1)
        return torch.nn.functional.normalize(v_program, p=2, dim=-1)


# ══════════════════════════════════════════════════════════════════════
# 4. Orchestrator
# ══════════════════════════════════════════════════════════════════════

class MultiModalEncoder:
    def __init__(
        self,
        node_type_vocab_size: int,
        edge_vocab_size: int,
        dim: int = 128,
        style_vocab_size: int = 5,
        seed: int = 90,
        graph2vec_epochs: int = 100,
        graph2vec_wl_iterations: int = 3,
    ):
        self.dim = dim
        self.seed = seed

        self.graph_encoder = GraphEncoder(
            node_type_vocab_size=node_type_vocab_size,
            style_vocab_size=style_vocab_size,
            edge_vocab_size=edge_vocab_size,
            output_dim=dim,
            wl_iterations=graph2vec_wl_iterations,
            epochs=graph2vec_epochs,
            seed=seed,
        )
        self.graph_encoder.eval()

        self.text_encoder = TextEncoder(output_dim=dim)

        self.fusion = ConcatFusion(dim=dim)
        self.fusion.eval()

        self._calibrated = False
        self._cohort_mean: dict[str, np.ndarray] = {}
        self._cohort_top_pcs: dict[str, np.ndarray] = {}

    def fit_text_cohort(self, stripped_texts: list[str], comment_texts: list[str] | None = None):
        if comment_texts is None:
            comment_texts = [""] * len(stripped_texts)
        self.text_encoder.fit_cohort(stripped_texts, comment_texts)

    def fit_cohort(self, cohort_pyg: dict, stripped_texts: list[str], comment_texts: list[str]):
        student_ids = list(cohort_pyg.keys())

        # Fit graph2vec over the whole cohort first — everything else
        # (text SVD, ABTT calibration) depends on encode_submission,
        # which needs a fitted graph encoder.
        graphs_for_fit = [
            {"x": cohort_pyg[sid].x, "edge_index": cohort_pyg[sid].edge_index}
            for sid in student_ids
        ]
        self.graph_encoder.fit_corpus(graphs_for_fit)

        self.fit_text_cohort(stripped_texts, comment_texts)

        raw_vecs = {
            sid: self.encode_submission(cohort_pyg[sid], stripped_texts[i], comment_texts[i])
            for i, sid in enumerate(student_ids)
        }

        for space in ("v_topo", "v_text", "v_program"):
            mat = np.stack([raw_vecs[sid][space] for sid in student_ids])
            mean = mat.mean(axis=0)
            self._cohort_mean[space] = mean

            centred = mat - mean[np.newaxis, :]
            n, d = centred.shape
            if min(n, d) < 2:
                self._cohort_top_pcs[space] = np.empty((0, d), dtype=np.float32)
                continue

            _, S, Vt = np.linalg.svd(centred, full_matrices=False)
            explained = (S ** 2) / np.sum(S ** 2)

            iso_baseline = max(3.0 * (1.0 / d), 0.05)
            K = 0
            for k_idx in range(min(3, len(explained))):
                if explained[k_idx] > iso_baseline:
                    K = k_idx + 1
                else:
                    break

            self._cohort_top_pcs[space] = Vt[:K].astype(np.float32)
            logging.info(f"[CALIBRATION] {space}: dropping top-{K} PC(s)")

        self._calibrated = True

    def encode_submission(
        self, pyg_data, stripped_source: str = "", comment_text: str = "", debug: bool = False,
    ) -> dict:
        num_nodes = pyg_data.x.size(0) if pyg_data.x is not None else 0
        if num_nodes == 0:
            v_topo = np.zeros(self.dim, dtype=np.float32)
        else:
            with torch.no_grad():
                v_topo = (
                    self.graph_encoder(pyg_data.x.long(), pyg_data.edge_index, debug=debug)
                    .cpu().numpy().flatten()
                )

        norm = np.linalg.norm(v_topo)
        if norm > 0:
            v_topo = v_topo / norm

        v_text = self.text_encoder.encode(stripped_source, comment_text)

        with torch.no_grad():
            t_topo = torch.from_numpy(v_topo).unsqueeze(0).float()
            t_text = torch.from_numpy(v_text).unsqueeze(0).float()
            v_program = self.fusion(t_topo, t_text).squeeze(0).numpy()

        norm = np.linalg.norm(v_program)
        if norm > 0:
            v_program = v_program / norm

        result = {
            "v_topo": v_topo.astype(np.float32),
            "v_text": v_text.astype(np.float32),
            "v_program": v_program.astype(np.float32),
        }

        if self._calibrated:
            for space in ("v_topo", "v_text", "v_program"):
                result[space] = self._apply_calibration(result[space], space)

        return result

    def _apply_calibration(self, vec: np.ndarray, space: str) -> np.ndarray:
        vec = vec - self._cohort_mean[space]
        top_pcs = self._cohort_top_pcs[space]
        if top_pcs.shape[0] > 0:
            vec = vec - top_pcs.T @ (top_pcs @ vec)

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec.astype(np.float32)

    @staticmethod
    def mean_center_vectors(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not vectors:
            return vectors
        ids = list(vectors.keys())
        mat = np.stack([vectors[sid] for sid in ids])
        centered = mat - mat.mean(axis=0, keepdims=True)
        norms = np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
        centered = centered / norms
        return {sid: centered[i].astype(np.float32) for i, sid in enumerate(ids)}
