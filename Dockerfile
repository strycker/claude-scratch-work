# ─────────────────────────────────────────────────────────────────────────────
# Trading-Crab — Multi-stage Dockerfile
#
# Two stages:
#   base     — core library deps (pandas, sklearn, scipy, …); no network/viz deps
#   pipeline — full deps (ingestion + plotting + boosting + CLI) — default target
#
# Build:
#   docker build -t trading-crab .
#   docker build --target base -t trading-crab:base .
#
# Run (one-shot pipeline pass):
#   docker run --rm \
#     -e FRED_API_KEY=your_key \
#     -v $(pwd)/config:/app/config:ro \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/outputs:/app/outputs \
#     trading-crab --refresh --recompute --steps 1,2,3,4,5,6,7
#
# Run (weekly report with email):
#   docker run --rm \
#     -e FRED_API_KEY=your_key \
#     -e TC_SMTP_HOST=smtp.gmail.com \
#     -e TC_SMTP_PORT=587 \
#     -e TC_SMTP_USER=you@gmail.com \
#     -e TC_SMTP_PASSWORD=app_password \
#     -e TC_EMAIL_FROM=you@gmail.com \
#     -e TC_EMAIL_TO=recipient@example.com \
#     -v $(pwd)/config:/app/config:ro \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/outputs:/app/outputs \
#     trading-crab --refresh --recompute --steps 1,2,3,4,5,6,7 --weekly-report --send-email
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: base ─────────────────────────────────────────────────────────────
# Installs core library deps only (no ingestion, no plotting).
# Useful as a lightweight base for custom pipeline images.

FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="trading-crab-lib (base)"
LABEL org.opencontainers.image.description="Market regime classification library — core deps only"
LABEL org.opencontainers.image.licenses="MIT"

# System packages needed to compile scipy / lxml native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the files needed to build the library wheel
COPY src/trading_crab_lib/pyproject.toml src/trading_crab_lib/
COPY src/trading_crab_lib/ src/trading_crab_lib/

# Install core library deps (no extras)
RUN pip install --no-cache-dir -e "src/trading_crab_lib/"


# ── Stage 2: pipeline ─────────────────────────────────────────────────────────
# Full install: ingestion + plotting + boosting + CLI entry points.
# This is the production image for running the weekly pipeline.

FROM base AS pipeline

LABEL org.opencontainers.image.title="trading-crab (pipeline)"
LABEL org.opencontainers.image.description="Market regime classification pipeline — full install"

# Copy app package and repo artefacts needed at runtime
COPY pyproject.toml .
COPY src/trading_crab/ src/trading_crab/
COPY pipelines/ pipelines/
COPY scripts/ scripts/
COPY run_pipeline.py .

# Install the library with all optional extras, then the app package
RUN pip install --no-cache-dir -e "src/trading_crab_lib/[ingestion,plotting,boosting]"
RUN pip install --no-cache-dir -e "."

# Optional but recommended: balanced-size clustering
RUN pip install --no-cache-dir k-means-constrained || true

# Runtime directories — will be overridden by volume mounts in practice.
# Pre-creating them ensures the pipeline can write even without a host mount.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/regimes \
             /app/data/checkpoints /app/outputs/models \
             /app/outputs/plots /app/outputs/reports \
             /app/config

# ── Environment ──────────────────────────────────────────────────────────────
# FRED_API_KEY must be passed at runtime via -e or docker-compose env_file.
# TC_* overrides are optional — they redirect the library's path resolution.
ENV FRED_API_KEY=""
ENV TC_CONFIG_DIR="/app/config"
ENV TC_DATA_DIR="/app/data"
ENV TC_OUTPUT_DIR="/app/outputs"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Volumes ────────────────────────────────────────────────────────────────────
# Declare expected mount points so docker-compose and users know what to bind.
VOLUME ["/app/config", "/app/data", "/app/outputs"]

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Default: run the full pipeline via the tradingcrab CLI.
# Override CMD at runtime for different step subsets:
#   docker run trading-crab --steps 3,4,5
ENTRYPOINT ["tradingcrab"]
CMD ["--help"]
