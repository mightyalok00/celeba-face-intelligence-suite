# ==============================================================================
# Multi-Stage Production Container for CelebA Face Intelligence Suite
# Supports Python ML Pipelines, PyTorch Model Serving, and TypeScript Web Frontend
# ==============================================================================

FROM node:22-bookworm-slim AS node-base

FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=3000

# Install Node.js runtime from node-base stage and essential system tools
COPY --from=node-base /usr/local/bin/node /usr/local/bin/
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user for container security
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Install Python ML and Computer Vision dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Install Node.js Express, React & Vite dependencies and build production bundle
COPY package.json package-lock.json tsconfig.json vite.config.ts index.html ./
COPY client ./client
RUN npm ci && npm run build && npm cache clean --force

# Copy application source, models, metadata, and frontend
COPY . .

# Set proper permissions for non-root runtime
RUN chown -R appuser:appuser /app

USER appuser

ENV NODE_ENV=production
EXPOSE 3000

# Container health probe checking the pre-warming /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command launches CelebA AI — Face Intelligence Studio
CMD ["node", "server.ts"]
