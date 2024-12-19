#!/bin/bash
if [ -f "/app/.env" ]; then
    set -o allexport
    source /app/.env
    set +o allexport
    echo "Environment variables loaded from .env file"
else
    echo "No .env file found in /app"
fi
echo "MLFLOW_TRACKING_URI is set to: $MLFLOW_TRACKING_URI"



# MLFLOW_TRACKING_URI="http://34.130.56.87:5000/"
# USE_EXTERNAL_MLFLOW=true
echo "MLFLOW_TRACKING_URI is set to: $MLFLOW_TRACKING_URI"
# Function to setup service account if present
setup_service_account() {
    if [ -f "/app/service_account.json" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="/app/service_account.json"
        echo "Service account credentials configured"
    fi
}

# If USE_EXTERNAL_MLFLOW is true and service account exists, set it up
if [ "$USE_EXTERNAL_MLFLOW" = "true" ]; then
    setup_service_account
    if [ -z "$MLFLOW_TRACKING_URI" ]; then
        echo "Warning: External MLflow server requested but no MLFLOW_TRACKING_URI provided"
        echo "Falling back to local MLflow server"
        mlflow server --host 0.0.0.0 --port 5000 &
        export MLFLOW_TRACKING_URI="http://localhost:5000"
    fi
else
    echo "Using local MLflow server"
    mlflow server --host 0.0.0.0 --port 5000 &
    export MLFLOW_TRACKING_URI="http://localhost:5000"
fi

sleep 5

# python scripts/run.py
# python -m src
# CMD ["python", "-m", "src", "/app/config/app-config.json"]
# python -m src --config=${1:-/app/config/ml_config.yaml}
python -m src --config=${1:-config/ml_config.yaml}
exit_code=$?
if [ $exit_code -eq 0 ]; then
    # Run screen module after src completes
    python -m screen --config=${1:-config/ml_config.yaml}
else
    echo "Error: src module failed. Stopping execution."
    exit 1
fi

# tail -f /dev/null

if [ "${USE_EXTERNAL_MLFLOW,,}" = "false" ]; then
    echo "USE_EXTERNAL_MLFLOW is set to false, keeping the container running..."
    tail -f /dev/null
else
    echo "USE_EXTERNAL_MLFLOW is not false, container will exit after script execution."
fi



# docker build -t test-image:latest -f docker/Dockerfile .
# docker run -it test-image:latest


# REGION=northamerica-northeast2
# PROJECT_ID=aircheck-398319
# ARTIFACT_REGISTRY=ml-training
# IMAGE_NAME=ml-model
# TAG=0.0.1


# docker build -t northamerica-northeast2-docker.pkg.dev/aircheck-398319/ml-training/ml-model:0.0.1  --platform=linux/amd64 -f docker/Dockerfile .

# docker push northamerica-northeast2-docker.pkg.dev/aircheck-398319/ml-training/ml-model:0.0.1

