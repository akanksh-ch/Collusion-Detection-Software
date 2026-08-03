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

# Python randomizes str hash() per process (PEP 456) unless pinned. gensim's
# Doc2Vec (which Graph2Vec wraps) defaults its `hashfxn` param to Python's
# built-in hash() for vocabulary bucketing, so without this the embeddings
# genuinely differ between container runs even with a fixed Graph2Vec seed
# and workers=1 — this isn't a threading issue, it's a per-process interpreter
# seed issue that workers=1 alone does not fix.
ENV PYTHONHASHSEED=0

# ==========================================
# STAGE 3: Inject Joern & Configure PATH
# ==========================================
COPY --from=joern-source /opt/joern /opt/joern
ENV PATH="/opt/joern/joern-cli:${PATH}"

# ==========================================
# STAGE 4: JDK 25 for JPlag 6.3.0 (side-by-side, not on PATH)
# ==========================================
# JPlag >= 6.x requires Java SE 25 (per its own README), but Joern's
# runtime and everything installed above assumes the Debian default JRE
# (JDK 21 via default-jre-headless) stays `java` on PATH. Rather than
# upgrading the system JDK and risking breaking Joern, install Temurin 25
# into its own directory and reference it by full path ONLY when invoking
# JPlag — nothing else in this image changes behavior.
RUN curl -fsSL -o /tmp/temurin25.tar.gz \
        "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" \
    && mkdir -p /opt/temurin25 \
    && tar -xzf /tmp/temurin25.tar.gz -C /opt/temurin25 --strip-components=1 \
    && rm /tmp/temurin25.tar.gz

# JPlag jar — place your v6.3.0 jar at ./jplag.jar in the build context
# (project root, alongside this Dockerfile) before building. Renamed on
# copy so run_pipeline.sh doesn't need to know the version-specific
# filename.
COPY jplag.jar /opt/jplag/jplag.jar

COPY *.py .

CMD ["python", "main.py"]
