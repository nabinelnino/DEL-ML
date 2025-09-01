#!/bin/bash
set -e

# Load environment variables
[ -f "/app/.env" ] && source /app/.env && echo "Environment variables loaded"

# Setup GCP service account
[ -f "/app/service_account.json" ] && export GOOGLE_APPLICATION_CREDENTIALS="/app/service_account.json"

# Start MLflow server conditionally
if [ "${USE_EXTERNAL_MLFLOW,,}" != "true" ]; then
    echo "Starting local MLflow server"
    mlflow server --host 0.0.0.0 --port 5000 &
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    sleep 5
fi

echo "MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo "Starting training with config: ${1:-/app/config/ml_config.yaml}"

# Run training
echo "Starting training with config: ${1:-/app/config/ml_config.yaml}"

python -m src --config="${1:-/app/config/ml_config.yaml}"


# python -m src --config="${1:-/app/config/ml_config.yaml}" && {
#     echo "Training completed successfully, starting screen..."
#     python -m screen --config="${1:-/app/config/ml_config.yaml}"
# }

# Keep container running if using local MLflow
[ "${USE_EXTERNAL_MLFLOW,,}" = "false" ] && {
    echo "Keeping container running for local MLflow access"
    tail -f /dev/null
}



