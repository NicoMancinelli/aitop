# Multi-stage Dockerfile for aitop (gateway / telemetry / TUI server)
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv for fast, reliable builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification files
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/

# Build wheel
RUN uv build --wheel

# Final minimal runtime image
FROM python:3.13-slim

WORKDIR /app

# Install runtime utilities (curl, procps for process monitoring)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder and install
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Expose aitop serve default port
EXPOSE 9090

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    AITOP_NO_UPDATE_CHECK=1

# Run aitop serve by default
ENTRYPOINT ["aitop"]
CMD ["serve", "--host", "0.0.0.0", "--port", "9090"]
