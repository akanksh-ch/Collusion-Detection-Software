# syntax=docker/dockerfile:1

# Grab Joern binaries

FROM ghcr.io/joernio/joern:nightly AS joern-source

# Add to path so it's accessible through CLI.
ENV PATH="/opt/joern/joern-cli:${PATH}"

RUN <<EOF
if command -v joern-parse &> /dev/null; then
    echo "Available"
else
    echo "Not available"
fi
EOF

# Main image running the project

FROM python:3.12-slim

# Copy neccessary binaries
COPY --from=joern-source /opt/joern /opt/joern

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

# Snippet from docs: https://docs.astral.sh/uv/guides/integration/docker/#installing-a-package

RUN uv venv /opt/venv
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/venv
# Place entry points in the environment at the front of the path
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .

# Karateclub (Graph2Vec) has older pandas and numpy versions, we're overriding them
RUN uv pip install --system -r requirements.txt --override requirements.txt

# Python changes seed per process, setting env variable to prevent this. Refer (PEP 456)
ENV PYTHONHASHSEED=0

COPY *.py .

CMD ["python", "main.py"]
