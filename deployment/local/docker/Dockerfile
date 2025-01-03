# Use multi-stage build for smaller final image
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first to leverage Docker cache
COPY ../../requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV MLFLOW_TRACKING_URI="http://localhost:5001"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application files
# COPY . .
COPY ../../src /app/src
COPY ../../utils /app/utils
COPY ../../screen /app/screen
COPY ../../requirements.txt /app/requirements.txt
COPY ../config /app/config

# Expose ports
EXPOSE 5001 8080

# Run MLflow server and the application
CMD mlflow server --host 0.0.0.0 --port 5001 & \
    sleep 5 && \
    python -m src --config=config/ml_config.yaml && \
    python -m screen --config=config/ml_config.yaml && \
    tail -f /dev/null
