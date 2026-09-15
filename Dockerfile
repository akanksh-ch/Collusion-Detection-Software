# syntax=docker/dockerfile:1

# Grab Joern binaries

FROM ghcr.io/joernio/joern:master AS joern-source

# Main image running the project

FROM python:3.12-slim

# Install curl and general dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
	default-jre-headless \
	build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy neccessary binaries
COPY --from=joern-source /opt/joern /opt/joern

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

# Snippet from docs: https://docs.astral.sh/uv/guides/integration/docker/#installing-a-package
ENV UV_SYSTEM_PYTHON=1

# Install dependencies
COPY requirements.txt .

# Karateclub (Graph2Vec) has older pandas and numpy versions, we're overriding them
RUN uv pip install --system -r requirements.txt --override requirements.txt

# karateclub is abandoned and pins numpy<1.23/networkx<2.7/pandas<=1.3.5, which conflict with
# the versions above. Its code works fine on the newer versions, so install it with no deps
# and pull in its other runtime deps (missing from requirements.txt) separately.
RUN uv pip install --system --no-deps karateclub==1.3.3 \
    && uv pip install --system python-louvain pygsp python-Levenshtein decorator

# Python changes seed per process, setting env variable to prevent this. Refer (PEP 456)
ENV PYTHONHASHSEED=0

# Add to path so it's accessible through CLI.
ENV PATH="/opt/joern/joern-cli:${PATH}"

COPY . .

CMD ["python", "main.py"]
