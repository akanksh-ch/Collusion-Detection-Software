# syntax=docker/dockerfile:1

# Grab Joern binaries

FROM ghcr.io/joernio/joern:nightly AS joern-source

# Copy neccessary binaries
COPY --from=joern-source /opt/joern /opt/joern

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

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

# Install dependencies
COPY requirements.txt .

RUN uv pip install -r requirements.txt

# Python changes seed per process, setting env variable to prevent this. Refer (PEP 456)
ENV PYTHONHASHSEED=0

COPY *.py .

CMD ["python", "main.py"]
