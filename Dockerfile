# ==========================================
# STAGE 1: Extract Joern Binaries
# ==========================================
FROM ghcr.io/joernio/joern:nightly AS joern-source

# ==========================================
# STAGE 2: Build Final Production Environment
# ==========================================
# 3.12, not 3.14 — every package in this stack (numpy, scipy, faiss-cpu,
# gensim, karateclub) ships mature prebuilt wheels for 3.12. On 3.14 several
# of these have no wheels yet, forcing source builds where numpy ABI
# mismatches (your original error) become far more likely.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir "setuptools<=81.0.0"

COPY requirements.txt .

# Nothing in this stack (gensim/karateclub/sklearn/faiss-cpu/networkx) uses
# a GPU — the frozen GAT/GNN this pipeline used to depend on is gone. Pin
# torch's CPU-only wheel explicitly instead of auto-detecting CUDA/ROCm,
# which just downloads gigabytes of unused GPU runtime. If torch has been
# fully replaced by numpy in loader.py/gnn.py, drop this index url and the
# torch line from requirements.txt entirely.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache -r requirements.txt

# ==========================================
# STAGE 3: Inject Joern & Configure PATH
# ==========================================
COPY --from=joern-source /opt/joern /opt/joern
ENV PATH="/opt/joern/joern-cli:${PATH}"

COPY . .

CMD ["python", "main.py"]
