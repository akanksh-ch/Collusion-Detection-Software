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

# Python changes seed per process, setting env variable to prevent this. Refer (PEP 456)
ENV PYTHONHASHSEED=0

# Add to path so it's accessible through CLI.
ENV PATH="/opt/joern/joern-cli:${PATH}"

COPY . .

CMD ["python", "main.py"]
