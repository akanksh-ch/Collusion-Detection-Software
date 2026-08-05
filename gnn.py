"""Dual-track submission encoder: VGAE (topological) + TF-IDF/SVD (lexical),
fused via L2-normalise + concatenate.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ══════════════════════════════════════════════════════════════════════
# 1. Topological Track — Graph VAE, trained fresh per cohort
# ══════════════════════════════════════════════════════════════════════

class _GAEEncoderModule(nn.Module):
    """Trainable 2-layer residual GINEConv encoder with VGAE mean/logvar
    heads. Same node feature scheme the earlier frozen GraphEncoder used
    (twin type/style embeddings + log1p degree features) — the difference
    is every weight here gets gradient updates via edge-reconstruction loss
    instead of sitting at a frozen random init.

    Note: this call site (MultiModalEncoder.encode_submission) doesn't
    currently thread edge_attr through, so edges are treated as untyped
    (a constant zero edge feature) — effectively plain GIN rather than
    GINE. If real edge types become available here later, wire edge_attr
    through encode_submission's pyg_data and this'll pick them up for free
    via the existing edge_embedding table.
    """

    def __init__(
        self,
        style_vocab_size: int = 5,
        edge_vocab_size: int = 10,
        node_type_vocab_size: int = 256,
        type_embed_dim: int = 48,
        style_embed_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()
        cat_dim = type_embed_dim + style_embed_dim
        node_feat_dim = cat_dim + 2  # + log1p(in-deg), log1p(out-deg)

        self.type_embedding = nn.Embedding(node_type_vocab_size + 2, type_embed_dim)
        self.style_embedding = nn.Embedding(style_vocab_size + 2, style_embed_dim)
        self.edge_embedding = nn.Embedding(edge_vocab_size + 2, cat_dim)

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        gin_mlp1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        gin_mlp2 = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.conv1 = GINEConv(gin_mlp1, edge_dim=cat_dim)
        self.conv2 = GINEConv(gin_mlp2, edge_dim=cat_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.residual_proj = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        self.mu_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)

    @staticmethod
    def _degree_features(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
        device = edge_index.device
        out_deg = torch.zeros(num_nodes, device=device)
        in_deg = torch.zeros(num_nodes, device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            out_deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
            in_deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        return torch.stack([torch.log1p(in_deg), torch.log1p(out_deg)], dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None):
        x_type = x[:, 0].long().clamp(max=self.type_embedding.num_embeddings - 1)
        x_style = x[:, 1].long().clamp(max=self.style_embedding.num_embeddings - 1)
        h_type = self.type_embedding(x_type)
        h_style = self.style_embedding(x_style)
        degree_feat = self._degree_features(x.size(0), edge_index)
        h_cat = torch.cat([h_type, h_style, degree_feat], dim=-1)

        if edge_attr is not None and edge_index.size(1) > 0:
            edge_feat = self.edge_embedding(edge_attr.long())
        else:
            edge_feat = torch.zeros(
                (edge_index.size(1), self.edge_embedding.embedding_dim), device=h_cat.device
            )

        h = self.relu(self.input_proj(h_cat))
        h_res = h
        h = self.relu(self.ln1(self.conv1(h, edge_index, edge_attr=edge_feat) + h_res))
        h_res = self.residual_proj(h)
        h = self.relu(self.ln2(self.conv2(h, edge_index, edge_attr=edge_feat) + h_res))

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    @staticmethod
    def pool(node_vecs: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(node_vecs.size(0), dtype=torch.long, device=node_vecs.device)
        return global_mean_pool(node_vecs, batch)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _recon_loss(z: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, generator: torch.Generator) -> torch.Tensor:
    """Inner-product decoder edge reconstruction loss (Kipf & Welling, 2016)."""
    src, dst = edge_index[0], edge_index[1]
    pos_logits = (z[src] * z[dst]).sum(dim=-1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))

    num_neg = max(edge_index.size(1), 1)
    neg_src = torch.randint(0, num_nodes, (num_neg,), generator=generator)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), generator=generator)
    neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=-1)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

    return pos_loss + neg_loss


def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class GraphEncoder:
    """VGAE-based topological encoder — same call-site shape the earlier
    Graph2Vec wrapper had (fit_corpus(...) then __call__(x, edge_index, ...)
    -> [1, dim] tensor), so MultiModalEncoder doesn't need to change.

    Trains from scratch on the cohort's own graphs every time fit_corpus()
    is called (unsupervised, edge-reconstruction + KL loss) — no checkpoint
    is ever saved or loaded, so there's no staleness concern, matching
    Graph2Vec's "retrain from this corpus every run" model. Unlike
    Graph2Vec, the encoder is now actually trained rather than a fixed
    hash, which is what gives it a real "structurally similar graph ->
    nearby embedding" property.

    In-sample graphs (seen during fit_corpus) return the cached embedding
    computed in that training pass; out-of-sample graphs get a fresh
    forward pass through the now-trained (eval-mode) encoder.
    """

    def __init__(
        self,
        node_type_vocab_size: int = 256,
        style_vocab_size: int = 5,
        edge_vocab_size: int = 10,
        output_dim: int = 128,
        epochs: int = 200,
        lr: float = 1e-3,
        kl_weight: float = 1e-3,
        seed: int = 90,
    ):
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._gen = torch.Generator().manual_seed(seed)

        self._model = _GAEEncoderModule(
            node_type_vocab_size=node_type_vocab_size,
            style_vocab_size=style_vocab_size,
            edge_vocab_size=edge_vocab_size,
            output_dim=output_dim,
        )

        self._fitted = False
        self._index_by_key: dict[int, int] = {}
        self._trained_vectors: np.ndarray | None = None
        self.loss_history: list[dict[str, float]] = []

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    @staticmethod
    def _graph_key(x: np.ndarray, edge_index: np.ndarray) -> int:
        return hash((x.tobytes(), edge_index.tobytes()))

    def fit_corpus(self, graphs: list[dict]) -> None:
        """Train the VGAE from scratch over the whole cohort.

        Args:
            graphs: list of {"x": ndarray[N,2], "edge_index": ndarray[2,E]}
        """
        keys = []
        data_list = []
        for g in graphs:
            x_np = _to_numpy(g["x"]).astype(np.int64)
            edge_index_np = _to_numpy(g["edge_index"]).astype(np.int64)
            keys.append(self._graph_key(x_np, edge_index_np))
            edge_index_t = torch.from_numpy(edge_index_np).long()
            if edge_index_t.numel() == 0:
                edge_index_t = edge_index_t.reshape(2, 0)
            data_list.append(Data(x=torch.from_numpy(x_np).long(), edge_index=edge_index_t))

        logging.info(f"[VGAE] fitting over {len(data_list)} graphs")
        batch = Batch.from_data_list(data_list)

        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            mu, logvar = self._model(batch.x, batch.edge_index)
            z = _reparameterize(mu, logvar)

            r_loss = _recon_loss(z, batch.edge_index, z.size(0), self._gen)
            k_loss = _kl_loss(mu, logvar)
            loss = r_loss + self.kl_weight * k_loss

            loss.backward()
            optimizer.step()

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                self.loss_history.append(
                    {"epoch": epoch, "recon": r_loss.item(), "kl": k_loss.item(), "total": loss.item()}
                )
                logging.info(
                    f"[VGAE] epoch {epoch}/{self.epochs} "
                    f"recon={r_loss.item():.4f} kl={k_loss.item():.4f} total={loss.item():.4f}"
                )

        self._model.eval()
        with torch.no_grad():
            mu, _ = self._model(batch.x, batch.edge_index)
            pooled = self._model.pool(mu, batch.batch)
            pooled = F.normalize(pooled, p=2, dim=-1).cpu().numpy()

        self._trained_vectors = pooled
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
            # Out-of-sample graph: forward pass through the now-trained,
            # eval-mode encoder (deterministic — mu, not a sampled z).
            edge_index_t = torch.from_numpy(edge_index_np).long()
            if edge_index_t.numel() == 0:
                edge_index_t = edge_index_t.reshape(2, 0)
            with torch.no_grad():
                mu, _ = self._model(torch.from_numpy(x_np).long(), edge_index_t)
                pooled = self._model.pool(mu)
                vec = F.normalize(pooled, p=2, dim=-1).squeeze(0).cpu().numpy()

        if debug:
            print(f"[VGAE] nodes={x_np.shape[0]} edges={edge_index_np.shape[1]} "
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
        vgae_epochs: int = 200,
        vgae_lr: float = 1e-3,
        vgae_kl_weight: float = 1e-3,
    ):
        self.dim = dim
        self.seed = seed

        self.graph_encoder = GraphEncoder(
            node_type_vocab_size=node_type_vocab_size,
            style_vocab_size=style_vocab_size,
            edge_vocab_size=edge_vocab_size,
            output_dim=dim,
            epochs=vgae_epochs,
            lr=vgae_lr,
            kl_weight=vgae_kl_weight,
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

        # Train the VGAE over the whole cohort first — everything else
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
