####

<!-- Create venv-
conda -v <nameofvenv> python=3.12
conda activate <name> -->

# README

## Overview

This codebase is designed for building, training, and deploying machine learning models for DEL compound screening. It leverages the following technologies:

- **Docker** for containerization.
- **MLflow** for model and experiment tracking.
- GCP services (GCS bucket, Vertex AI) (if you have access to it, otherwise you can try on local machine using docker)

## Prerequisites

- Docker
- Python
- FastAPI
- MLflow (local or hosted)
- GCP SDK (optional, for GCP-based workflows)

The system is capable of:

1. Using local data on your local machine.
2. Using Google Cloud Platform (GCP) for reading, storing data, and model training.

### Features

The codebase comprises three main modules:

#### 1. Train

- Responsible for training the machine learning model.
- Saves the trained model inside the container image, which will be used during the screening process.

#### 2. Screen

- Screens compounds after the model has been successfully trained.

#### 3. API

- Provides a web API, implemented using FastAPI, to:
  - Predict compounds individually.
  - Predict compounds from a CSV file.

## Configuration Files

### `.env` File

The `.env` file stores environment variables used by the application. Key settings include:

- `MLFLOW_TRACKING_URI`: URL for the MLflow server. If you have a hosted MLflow server, replace the server URL in the `.env` file. Otherwise, MLflow will spin up a local server accessible at `localhost:5001`. If this port is already in use, you can modify it in the `.env` file.

### `ml_config` File

This file contains configuration parameters for data processing, model training, and result storage. Below is an example configuration:

```yaml
ml_config:
  input_data_path: "gs://test-aircheck/mlops-test/subset_WDR91"
  columns_of_interest:
    - ECFP4
  target_col: Label
  is_binarized_data: True
  cluster_generation: True
  model_name: WDR91_ECFP4-test-vertex
  processed_file_location:
  isdry_run: True
  smile_location: gs://enamine-vs/dataset/
  result_output: gs://test-aircheck/mlops-test/enamine_WDR91_test_mp.smi
  model_save_directory: "/app/models"
```

### Key Parameters

- **`input_data_path`**: Path to the input data (supports GCP paths or local paths).
- **`columns_of_interest`**: Columns used for training the model.
- **`target_col`**: The target column in the dataset.
- **`isdry_run`**: Boolean flag to indicate a dry run.
- **`model_save_directory`**: Directory inside the container where the trained model is saved.
- **`result_output`**: Path to save the screening results.

## How to Use

### 1. Training

- Run the training module to train the model.
- The trained model is saved within the container image.

### 2. Screening

- Use the screening module to evaluate compounds using the trained model.

### 3. API

- Start the API server using FastAPI.
- Predict individual compounds or upload a CSV file for bulk predictions.

### MLflow Integration

- If you have your own hosted MLflow server, specify its URL in the `.env` file under `MLFLOW_TRACKING_URI`.
- If no MLflow server is specified, the program will automatically spin up a local MLflow server.

## Port Configuration (if you opt to use local mlflow server)

- The default port for the local MLflow server is `5001`.
- If this port is occupied, modify the port number in the `.env` file.

## How to Use the Training Module

### 1. For Local Machine

- Navigate to the `deployment/local` folder:
  ```
  cd deployment/local
  ```
- Modify the contents of the `ml_config` file of the config folder to point to the appropriate paths for `input_data_path`, `smile_location`, and `result_output` and change the `.env` file if needed.
- In the `.env` file:
  - If you have a hosted MLflow server, specify its URL under `MLFLOW_TRACKING_URI`.
  - Otherwise, leave it as is to use the local MLflow server.

### 2. For GCP

- Update the `ml_config` file of the config folder to point to the appropriate GCP paths for `input_data_path`, `smile_location`, and `result_output`.
- Ensure GCP credentials are set up and accessible to the application.

### 3. Running the Training Process

- Execute the training process using the provided scripts or Docker commands.
- The model will be trained and saved inside the container image for further use in screening and API predictions.

## Notes

- Ensure the necessary GCP credentials are available if using GCP for data processing.
- Modify the `ml_config` file to match your data and model requirements.
