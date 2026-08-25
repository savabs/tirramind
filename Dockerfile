# TirraMind Intelligence Engine — container image
# Serves the Intelligence Brief over HTTP and runs the delivery engine.
FROM python:3.12-slim

WORKDIR /app

# System deps (minimal; torch/quant wheels install from PyPI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata + source
COPY pyproject.toml ./
COPY agent/ agent/
COPY scripts/ scripts/

# Install the package (quant extras; skip heavy ml to keep image small)
RUN pip install --no-cache-dir -e ".[dev,quant]" -q

# Runtime state volumes
VOLUME ["/app/.tirra_delivery", "/app/.tirra_opportunities", "/app/.tirra_pipeline"]

# Expose the brief server
EXPOSE 8787

# Default: serve the delivered brief over HTTP
CMD ["tirra-serve", "--host", "0.0.0.0", "--port", "8787"]
