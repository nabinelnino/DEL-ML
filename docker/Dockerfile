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
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV MLFLOW_TRACKING_URI=''
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV USE_EXTERNAL_MLFLOW=${USE_EXTERNAL_MLFLOW}

COPY . .

# Create entrypoint script before switching user
COPY entrypoint.sh /app/entrypoint.sh


# Set permissions before switching user
RUN chmod +x /app/entrypoint.sh && \
    # Create non-root user and set ownership
    useradd -m -r appuser && \
    chown -R appuser:appuser /app


EXPOSE 8080   
# Switch to non-root user
USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]
# CMD ["python", "-m", "src", "/app/config/app-config.json"]



# docker build -t ml-app .   
# docker run -p 127.0.0.1:8080:8080 ml-app
# docker rmi $(docker images -q) -f
# docker-compose down --rmi all



# Stop all running containers
# docker stop $(docker ps -q)

# # Remove all containers
# docker rm $(docker ps -aq)

# # Remove all images
# docker rmi $(docker images -q)

# docker build -t api-module ./API
