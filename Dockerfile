# Pin the base image by digest for reproducible builds (tag: python:3.12-slim).
FROM python:3.12-slim@sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code (the repo .dockerignore keeps .env*/secrets/tests out of the build context)
COPY . .

# Install package and all dependencies
RUN pip install --no-cache-dir .

# Run as an unprivileged user (least privilege; shrinks RCE/SSRF blast radius).
# uvicorn binds 8080 (>1024), so no root is required at runtime.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Liveness probe on the existing health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/v1/health || exit 1

CMD ["uvicorn", "api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
