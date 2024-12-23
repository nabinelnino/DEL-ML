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

## Port Configuration

- The default port for the local MLflow server is `5001`.
- If this port is occupied, modify the port number in the `.env` file.

## How to Use the Training Module

### 1. For Local Machine

- Navigate to the `deployment/local` folder:
  ```
  cd deployment/local
  ```
- Modify the contents of the `config` file and `.env` file if needed.
- download WDR91 target and place the datafile inside data/raw folder
- In the `.env` file:

  - If you have a hosted MLflow server, specify its URL under `MLFLOW_TRACKING_URI`.
  - Otherwise, leave it as is to use the local MLflow server.

  **Running and Managing the Program**:

  - To start the program, run: `make up-local`.
  - To rebuild the Docker image, use: `make rebuild`.
  - To clean up and remove all resources, execute: `make prune`.

### 2. For GCP

- Update the `ml_config` file of the config folder to point to the appropriate GCP paths for `input_data_path`, `smile_location`, and `result_output`.
- Ensure GCP credentials are set up and accessible to the application.
- rename env_example file to .env

#### 1. **Run Locally with GCP Integration**

- **Purpose**: Use GCP Storage for managing data and results while running the model locally in Docker.
- **Steps**:

  1. Configure the system to:
     - Read the target data from a GCP bucket.
     - Track trained model names and metadata in a GCP Storage bucket.
     - Store the final screened compound back to the GCP bucket.
  2. Navigate to the `DEL-ML` repository
     `cd DEL-ML`

  3. Prepare the configuration:
     - Update the configuration file and `.env` file with the necessary GCP settings. (Rename `env_example` to `.env` if not already done.)
  4. Run the program:
     `make up-local`
  5. To rebuild the Docker image:
     `make rebuild`
  6. To clean up and start over:
     `make prune`

#### 2. **Run on GCP Vertex AI**

- **Purpose**: Build and deploy the model using GCP's Vertex AI for scalable training and screen.
- **Steps**:
  1. Build the Docker image:
     `make build`
  2. Push the image to GCP Container Registry:
     `make push`
  3. Use the pushed image in Vertex AI to run a custom training job.
     - Define a Vertex AI custom job configuration, including the Docker image URI and any required arguments or environment variables.
     - Submit the job to Vertex AI.

### 3. Running the Prediction API

This process includes a prediction API built using FastAPI, which utilizes a trained model from the local Docker image to predict compounds. To use the API, follow these steps:

To start the API, follow these steps:

1. Navigate to the `API` folder:
   `cd API`
2. Start the API using Docker Compose:

   `docker-compose up --build`

3. Access the FastAPI documentation:

   - Open your browser and go to `http://0.0.0.0:8000/docs`.
   - Follow steps 1 to 4 below to use the API.

   ### **Steps to Use the Prediction API**

4. **Access the Local MLflow Server**:

   - Navigate to the local MLflow server at `http://localhost:5001`.
   - Click on the **Experiments** tab.
     ![[image.png]]

5. **Retrieve the Model's `run_id`**:

   - Select the name of the experiment from the list.
   - Click on the model name.
   - Copy the `run_id` for the model.
     ![[img2.png]]
     ![[img3.png]]

6. **Use the Prediction API**:

   - Go to the FastAPI server interface.
   - Paste the copied `run_id` into the designated field.
   - Provide input data in one of the following formats:
     - A single valid compound.
     - A list of compounds separated by commas.
     - A CSV file containing compounds.

7. **Run the Prediction**:

   - Upload the data and click the **Process** button.
   - The predictions for each compound will be displayed.

Ensure the inputs are valid to get accurate predictions.

## Notes

- Ensure the necessary GCP credentials are available if using GCP for data processing.
- Modify the `ml_config` file to match your data and model requirements.
