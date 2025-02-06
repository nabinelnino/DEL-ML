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

WORKDIR /code

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./app /code/app
COPY ./service_account.json /service_account.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# fastapi dev main.py

# docker build -t apiserver .
# docker run apiserver
# docker run -d --name apicontainer -p 80:80 apiserver

# docker build -t fastapi-app .
# docker run -d -p 8000:8000 fastapi-app
# http://localhost:8000

# docker run --add-host=host.docker.internal:host-gateway -p 8000:8000 apiserver

# docker build -t gcr.io/your-project-id/fastapi-app:latest .
