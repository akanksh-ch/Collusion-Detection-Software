"""Multi-modal feature encoding: dual-track encoder + Gated Multimodal Fusion.

Implements Module 3 of the production specification:

  Topological Track  — Twin nn.Embedding → frozen GATConv with residual
                       skip connections → Global Attention Pooling
                       → V_topo ∈ R^D

  Lexical Track      — char n-gram TF-IDF + layout metrics
                       → TruncatedSVD projection → V_text ∈ R^D

  Gated Multimodal   — V̄_topo = tanh(W_topo V_topo + b_topo)
  Fusion (GMF)         V̄_text = tanh(W_text V_text + b_text)
                       g       = σ(W_g [V̄_topo ∥ V̄_text] + b_g)
                       V_prog  = g ⊙ V̄_topo + (1-g) ⊙ V̄_text

Notes:
  - GAT uses residual skip connections to prevent over-smoothing with
    frozen random weights (diagnosed: 3-layer GAT collapsed cosine
    similarities to 0.997–1.000 across all pairs).
  - Text encoder uses TruncatedSVD (fitted on cohort) instead of a
    frozen random nn.Linear, which compressed all variance into a
    narrow cone (cosine sims 0.82–0.99).
  - Fusion uses orthogonal weight init for better distance preservation.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sklearn_normalize


# ══════════════════════════════════════════════════════════════════════
# 1. Topological Track — Graph Encoder (with residual skip connections)
# ══════════════════════════════════════════════════════════════════════

class GraphEncoder(nn.Module):
    """Untrained Deep GAT with residual connections acting as a continuous
    topological projection hash.

    Residual skip connections prevent the over-smoothing problem that
    causes all vectors to collapse when using frozen random weights
    through multiple message-passing layers.
    """

    def __init__(
        self,
        node_type_vocab_size: int,
        style_vocab_size: int = 5,
        edge_vocab_size: int = 10,
        type_embed_dim: int = 48,
        style_embed_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 128,
        heads: int = 4,
    ):
        super().__init__()
        self.type_embed_dim = type_embed_dim
        self.style_embed_dim = style_embed_dim
        cat_dim = type_embed_dim + style_embed_dim

        # Twin embedding tables
        self.type_embedding = nn.Embedding(node_type_vocab_size + 2, type_embed_dim)
        self.style_embedding = nn.Embedding(style_vocab_size + 2, style_embed_dim)
        self.edge_embedding = nn.Embedding(edge_vocab_size + 2, cat_dim)

        # Input projection to hidden_dim for residual compatibility
        self.input_proj = nn.Linear(cat_dim, hidden_dim)

        # 2-layer GAT with residual connections (reduced from 3 to avoid
        # over-smoothing with frozen weights)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=False, edge_dim=cat_dim)
        self.conv2 = GATConv(hidden_dim, output_dim, heads=heads,
                             concat=False, edge_dim=cat_dim)

        # Layer norms for stable residuals
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(output_dim)

        self.relu = nn.ReLU()

        # Output projection for residual dimension alignment
        self.residual_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x: [N, 2]  →  col 0 = x_type, col 1 = x_style
        x_type = x[:, 0].long()
        x_style = x[:, 1].long()

        h_type = self.type_embedding(x_type)
        h_style = self.style_embedding(x_style)
        h_cat = torch.cat([h_type, h_style], dim=-1)  # [N, cat_dim]

        # Edge features
        if edge_attr is not None and edge_index.size(1) > 0:
            edge_feat = self.edge_embedding(edge_attr.long())
        else:
            edge_feat = torch.zeros(
                (edge_index.size(1), h_cat.size(-1)), device=h_cat.device
            )

        # Project to hidden_dim
        h = self.relu(self.input_proj(h_cat))

        # Layer 1 with residual skip
        h_res = h
        h = self.relu(self.ln1(self.conv1(h, edge_index, edge_attr=edge_feat) + h_res))

        # Layer 2 with residual skip (project residual to output_dim)
        h_res = self.residual_proj(h)
        h = self.relu(self.ln2(self.conv2(h, edge_index, edge_attr=edge_feat) + h_res))

        # Global mean pooling — every node contributes equally, giving a
        # stable, size-normalised topological hash. (Uses global_mean_pool
        # rather than an AttentionalAggregation gate_nn, since that gate
        # is frozen/untrained and therefore assigns arbitrary, size-sensitive
        # attention weights rather than a meaningful summary of the subgraph.)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return global_mean_pool(h, batch)  # [1, output_dim]


# ══════════════════════════════════════════════════════════════════════
# 2. Lexical Track — Text Encoder (TruncatedSVD-based)
# ══════════════════════════════════════════════════════════════════════

class TextEncoder:
    """Combines character n-gram TF-IDF with continuous layout metrics.

    Uses TruncatedSVD fitted on the cohort corpus (instead of a frozen
    random linear projection) to reduce the raw feature vector to R^D.
    SVD preserves the principal variance directions, giving much better
    pairwise distance discrimination.
    """

    def __init__(self, output_dim: int = 128, max_tfidf_features: int = 256):
        self.output_dim = output_dim
        self.max_tfidf_features = max_tfidf_features
        self._tfidf_disabled = False
        self._svd_fitted = False

        # Character n-gram TF-IDF
        self.char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_tfidf_features,
        )

        # Layout metric dimensionality
        self._layout_dim = 3
        self._raw_dim = max_tfidf_features + self._layout_dim

        # TruncatedSVD for dimensionality reduction (fitted on cohort)
        self._svd_dim = min(output_dim, self._raw_dim - 1)
        self._svd = TruncatedSVD(n_components=self._svd_dim, random_state=42)

    def fit_cohort(self, stripped_texts: list[str], comment_texts: list[str]):
        """Fit TF-IDF and SVD on the full cohort corpus."""
        clean = [t for t in stripped_texts if t.strip()]
        if not clean:
            self._tfidf_disabled = True
            return

        try:
            self.char_tfidf.fit(clean)
        except ValueError:
            logging.warning("[TF-IDF] Empty corpus — fallback mode active.")
            self._tfidf_disabled = True
            return

        # Build raw feature matrix for the entire cohort
        raw_matrix = []
        for src, comments in zip(stripped_texts, comment_texts):
            raw_vec = self._extract_raw_features(src, comments)
            raw_matrix.append(raw_vec)

        raw_matrix = np.array(raw_matrix, dtype=np.float64)

        # Fit SVD on the cohort's raw feature space
        n_samples = raw_matrix.shape[0]
        effective_dim = min(self._svd_dim, n_samples - 1, raw_matrix.shape[1] - 1)
        if effective_dim < 1:
            logging.warning("[SVD] Not enough samples for SVD — using raw features.")
            return

        self._svd = TruncatedSVD(n_components=effective_dim, random_state=42)
        self._svd.fit(raw_matrix)
        self._svd_fitted = True
        logging.info(
            f"[TEXT ENCODER] SVD fitted: {raw_matrix.shape[1]}D → {effective_dim}D "
            f"(explained variance: {self._svd.explained_variance_ratio_.sum():.2%})"
        )

    def encode(self, stripped_source: str, comment_text: str = "") -> np.ndarray:
        """Produce a D-dimensional text style vector for a single submission."""
        raw_vec = self._extract_raw_features(stripped_source, comment_text)

        if self._svd_fitted:
            projected = self._svd.transform(raw_vec.reshape(1, -1)).flatten()
        else:
            # Fallback: truncate/pad to output_dim
            projected = raw_vec[:self.output_dim]
            if len(projected) < self.output_dim:
                projected = np.pad(projected, (0, self.output_dim - len(projected)))

        # Pad to output_dim if SVD produced fewer components
        if len(projected) < self.output_dim:
            projected = np.pad(projected, (0, self.output_dim - len(projected)))

        # L2 normalise
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        return projected.astype(np.float32)

    def _extract_raw_features(self, stripped_source: str, comment_text: str) -> np.ndarray:
        """Extract the raw (un-projected) feature vector."""
        # TF-IDF features
        if (stripped_source.strip()
                and not self._tfidf_disabled
                and hasattr(self.char_tfidf, "vocabulary_")):
            try:
                tfidf_vec = self.char_tfidf.transform(
                    [stripped_source]
                ).toarray().flatten()
            except ValueError:
                tfidf_vec = np.zeros(self.max_tfidf_features)
        else:
            tfidf_vec = np.zeros(self.max_tfidf_features)

        # Layout metrics
        lines = stripped_source.splitlines()
        total_lines = max(len(lines), 1)

        leading_counts = []
        for ln in lines:
            stripped = ln.lstrip()
            indent = ln[: len(ln) - len(stripped)]
            leading_counts.append(indent.count("\t") - indent.count(" "))
        tabs_spaces_var = float(np.var(leading_counts)) if leading_counts else 0.0

        blank_count = sum(1 for ln in lines if not ln.strip())
        blank_ratio = blank_count / total_lines

        code_chars = max(
            len(stripped_source.replace(" ", "").replace("\n", "")), 1
        )
        comment_chars = len(comment_text.replace(" ", "").replace("\n", ""))
        comment_density = comment_chars / (comment_chars + code_chars)

        layout = np.array(
            [tabs_spaces_var, blank_ratio, comment_density], dtype=np.float64
        )

        return np.concatenate([tfidf_vec, layout])


# ══════════════════════════════════════════════════════════════════════
# 3. Gated Multimodal Fusion
# ══════════════════════════════════════════════════════════════════════

class GatedMultimodalFusion(nn.Module):
    """Implements the exact GMF equations from the specification.

    Uses orthogonal weight initialisation for better distance preservation
    through the gating network.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.proj_topo = nn.Linear(dim, dim)
        self.proj_text = nn.Linear(dim, dim)
        self.gate = nn.Linear(2 * dim, dim)

        # Orthogonal init for distance-preserving projections
        nn.init.orthogonal_(self.proj_topo.weight)
        nn.init.orthogonal_(self.proj_text.weight)
        nn.init.orthogonal_(self.gate.weight)

    def forward(self, v_topo: torch.Tensor, v_text: torch.Tensor) -> torch.Tensor:
        v_topo_bar = torch.tanh(self.proj_topo(v_topo))
        v_text_bar = torch.tanh(self.proj_text(v_text))

        g = torch.sigmoid(
            self.gate(torch.cat([v_topo_bar, v_text_bar], dim=-1))
        )
        v_program = g * v_topo_bar + (1 - g) * v_text_bar
        return v_program


# ══════════════════════════════════════════════════════════════════════
# 4. Orchestrator: MultiModalEncoder
# ══════════════════════════════════════════════════════════════════════

class MultiModalEncoder:
    """Top-level encoder wrapping both tracks and the fusion module.

    Returns all three vectors per submission (V_topo, V_text, V_program)
    so that the analyser can perform forensic disaggregation on the
    decoupled components.

    Calibration strategy (All-But-The-Top):
      During ``fit_cohort()`` we compute and permanently freeze:
        1. The cohort mean vector for each embedding space.
        2. The top-K dominant principal components (K determined
           dynamically from the singular value spectrum).
      During ``encode_submission()`` we apply those frozen parameters
      to incoming vectors: subtract the mean, project out the top-K
      PCs, and re-normalise.  No per-batch refitting occurs.

      This guarantees that isotropic calibration only stabilises the
      cosine distance metric; the similarity threshold (e.g. 0.85) is
      derived empirically via precision-recall tuning against the
      labelled ground-truth corpus.
    """

    def __init__(
        self,
        node_type_vocab_size: int,
        edge_vocab_size: int,
        dim: int = 128,
        style_vocab_size: int = 5,
    ):
        self.dim = dim

        # Topological track
        self.graph_encoder = GraphEncoder(
            node_type_vocab_size=node_type_vocab_size,
            style_vocab_size=style_vocab_size,
            edge_vocab_size=edge_vocab_size,
            output_dim=dim,
        )
        self.graph_encoder.eval()
        for param in self.graph_encoder.parameters():
            param.requires_grad = False

        # Lexical track
        self.text_encoder = TextEncoder(output_dim=dim)

        # Gated Multimodal Fusion
        self.fusion = GatedMultimodalFusion(dim=dim)
        self.fusion.eval()
        for param in self.fusion.parameters():
            param.requires_grad = False

        # ── Frozen calibration state (set by fit_cohort) ─────────────
        self._calibrated: bool = False
        self._cohort_mean: dict[str, np.ndarray] = {}    # space → mean vec
        self._cohort_top_pcs: dict[str, np.ndarray] = {} # space → (K, D) PC matrix

    def fit_text_cohort(
        self,
        stripped_texts: list[str],
        comment_texts: list[str] | None = None,
    ):
        """Fit the text encoder's TF-IDF + SVD on the cohort corpus.

        .. deprecated:: Use ``fit_cohort`` instead, which also computes
           and freezes isotropic calibration parameters.
        """
        if comment_texts is None:
            comment_texts = [""] * len(stripped_texts)
        self.text_encoder.fit_cohort(stripped_texts, comment_texts)

    # ------------------------------------------------------------------
    # Stateful cohort calibration (All-But-The-Top)
    # ------------------------------------------------------------------
    def fit_cohort(
        self,
        cohort_pyg: dict,
        stripped_texts: list[str],
        comment_texts: list[str],
    ):
        """Fit all learnable parameters AND freeze calibration state.

        This method:
          1. Fits the lexical encoder's TF-IDF + SVD.
          2. Encodes every submission (uncalibrated).
          3. For each vector space (v_topo, v_text, v_program):
             a. Computes and stores the cohort mean.
             b. Inspects the singular value spectrum of the centred
                matrix to dynamically determine K — the number of
                dominant principal components to remove.
             c. Stores the top-K PCs.
          4. Sets ``_calibrated = True``.

        After this call, ``encode_submission`` will automatically apply
        the frozen calibration parameters to every new vector.
        """
        student_ids = list(cohort_pyg.keys())

        # Step 1 — fit the text encoder's TF-IDF + SVD
        self.fit_text_cohort(stripped_texts, comment_texts)

        # Step 2 — encode the entire cohort (uncalibrated)
        raw_vecs: dict[str, dict[str, np.ndarray]] = {}
        for idx, sid in enumerate(student_ids):
            raw_vecs[sid] = self.encode_submission(
                cohort_pyg[sid], stripped_texts[idx], comment_texts[idx],
            )

        # Step 3 — compute calibration parameters per vector space
        for space in ("v_topo", "v_text", "v_program"):
            mat = np.stack([raw_vecs[sid][space] for sid in student_ids])
            mean = mat.mean(axis=0)
            self._cohort_mean[space] = mean

            centred = mat - mean[np.newaxis, :]

            # SVD on the centred matrix to inspect the spectrum
            n, d = centred.shape
            max_components = min(n, d)
            if max_components < 2:
                self._cohort_top_pcs[space] = np.empty((0, d), dtype=np.float32)
                logging.info(
                    f"[CALIBRATION] {space}: too few samples for spectrum "
                    f"analysis — skipping PC removal."
                )
                continue

            U, S, Vt = np.linalg.svd(centred, full_matrices=False)
            explained = (S ** 2) / np.sum(S ** 2)

            # Dynamically decide K: drop components whose explained
            # variance ratio exceeds the isotropy expectation by ≥3×.
            # For a perfectly isotropic D-dimensional space each PC
            # would explain 1/D of the variance.
            iso_baseline = max(3.0 * (1.0 / d), 0.05)
            K = 0
            for k_idx in range(min(3, len(explained))):
                if explained[k_idx] > iso_baseline:
                    K = k_idx + 1
                else:
                    break

            self._cohort_top_pcs[space] = Vt[:K].astype(np.float32)

            # Logging for dissertation evidence
            top_vals = ", ".join(f"{v:.4f}" for v in explained[:5])
            logging.info(
                f"[CALIBRATION] {space}: singular value spectrum "
                f"(top-5 explained variance): [{top_vals}]  →  "
                f"dropping K={K} dominant PC(s)  "
                f"(isotropy baseline: {iso_baseline:.4f})"
            )

        self._calibrated = True
        logging.info("[CALIBRATION] Frozen calibration parameters stored.")

    def encode_submission(
        self,
        pyg_data,
        stripped_source: str = "",
        comment_text: str = "",
    ) -> dict:
        """Encode a single submission → decoupled + fused vectors.

        If ``fit_cohort`` has been called, the frozen calibration
        parameters (mean subtraction + top-K PC removal) are applied
        automatically.  No per-batch refitting occurs.

        Returns dict with ``v_topo``, ``v_text``, ``v_program`` —
        each an L2-normalised ``np.ndarray`` of shape ``(D,)``.
        """
        # ── Topological vector ───────────────────────────────────────
        num_nodes = pyg_data.x.size(0) if pyg_data.x is not None else 0
        if num_nodes == 0:
            v_topo = np.zeros(self.dim, dtype=np.float32)
        else:
            x = pyg_data.x.long()
            edge_attr = (
                pyg_data.edge_attr if hasattr(pyg_data, "edge_attr") else None
            )
            if edge_attr is not None and edge_attr.dim() > 1:
                edge_attr = edge_attr.squeeze(-1)

            with torch.no_grad():
                v_topo = (
                    self.graph_encoder(x, pyg_data.edge_index, edge_attr)
                    .cpu()
                    .numpy()
                    .flatten()
                )

        norm = np.linalg.norm(v_topo)
        if norm > 0:
            v_topo = v_topo / norm

        # ── Lexical vector ───────────────────────────────────────────
        v_text = self.text_encoder.encode(stripped_source, comment_text)

        # ── Gated Multimodal Fusion ──────────────────────────────────
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

        # Apply frozen calibration if available
        if self._calibrated:
            for space in ("v_topo", "v_text", "v_program"):
                result[space] = self._apply_calibration(
                    result[space], space
                )

        return result

    def _apply_calibration(
        self, vec: np.ndarray, space: str,
    ) -> np.ndarray:
        """Apply frozen All-But-The-Top calibration to a single vector.

        Steps:
          1. Subtract the cohort mean for this space.
          2. Remove the projections onto the top-K principal components.
          3. L2-normalise.

        This uses only the parameters frozen during ``fit_cohort``;
        no per-batch refitting occurs.
        """
        vec = vec - self._cohort_mean[space]

        top_pcs = self._cohort_top_pcs[space]  # (K, D) or (0, D)
        if top_pcs.shape[0] > 0:
            # Project out dominant directions:
            #   v' = v - Σ_k (v · pc_k) pc_k
            projections = top_pcs @ vec           # (K,)
            vec = vec - top_pcs.T @ projections   # (D,)

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec.astype(np.float32)

    @staticmethod
    def mean_center_vectors(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Remove the common mode from a cohort of vectors and re-normalise.

        .. deprecated:: Superseded by the stateful ``fit_cohort`` +
           ``_apply_calibration`` pipeline which also removes dominant
           principal components (All-But-The-Top).  Retained for
           backward compatibility.

        Args:
            vectors: mapping ``student_id → vector (D,)``

        Returns:
            New mapping with mean-centered, L2-normalised vectors.
        """
        if not vectors:
            return vectors

        ids = list(vectors.keys())
        mat = np.stack([vectors[sid] for sid in ids])
        mean = mat.mean(axis=0, keepdims=True)
        centered = mat - mean

        # Re-normalise to unit sphere
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        centered = centered / norms

        return {sid: centered[i].astype(np.float32) for i, sid in enumerate(ids)}
