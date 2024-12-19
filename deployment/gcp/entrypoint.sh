#!/bin/bash
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
python -m src --config=${1:-/app/config/ml_config.yaml}

tail -f /dev/null

