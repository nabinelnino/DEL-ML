
from typing import Union, List
from fastapi import FastAPI,  UploadFile, File, HTTPException, Body
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
import pandas as pd
import io
import os
import requests
from app.fps_conversion import screen_smiles

app = FastAPI()

service_account_path = '../service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    print("Service account credentials set.")
else:
    print("Service account file not found. Skipping credential setup.")


class InputModel(BaseModel):
    text_input: Union[str, List[str], None] = None


@app.post("/get-metrics/")
async def get_metrics(mlflow_server_url: str, run_id: str):
    """
    Retrieve metrics for a specific MLflow run.

    This function queries the MLflow Tracking Server's REST API to fetch 
    all recorded metrics for a given `run_id`. If the run exists and has metrics, 
    it returns them; otherwise, it raises an HTTPException.

    Args:
        mlflow_server_url (str):
            The base URL of the MLflow Tracking Server.
            Example: "http://localhost:5000" or "http://mlflow-server:5000".

        run_id (str):
            The unique identifier of the MLflow run whose metrics you want to retrieve.
            Example: "123acbfer12324".

    Returns:
        dict:
            A JSON-serializable dictionary containing:
            - "status" (str): Status of the request ("success").
            - "message" (str): Human-readable success message.
            - "result" (dict): Dictionary of metrics and their latest values.

    Raises:
        HTTPException:
            - If the request to the MLflow server fails (non-200 status code).
            - If the given run ID exists but no metrics are found.

    Notes:
        - Uses the `/api/2.0/mlflow/runs/get` MLflow REST API endpoint.
        - Metrics returned are the latest recorded values for the given run.
    """

    url = f"{mlflow_server_url}/api/2.0/mlflow/runs/get"

    # Query parameters
    params = {
        'run_id': run_id
    }

    # Send GET request to MLflow server
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to fetch data from MLflow server: {response.text}"
        )

    # Parse the JSON response
    run_data = response.json()
    print("run data_", run_data)

    # Extract metrics
    metrics = run_data['run']['data']['metrics']

    # Print metrics
    print("Metrics:", metrics)
    if metrics:
        return {
            "status": "success",
            "message": "Metrics retrieved successfully",
            "result": metrics,
        }
    raise HTTPException(
        status_code=400,
        detail="Metrics not found for the given run ID."
    )


@app.post("/process-input/")
async def process_input(
        model_id: str,
        mlflow_url: str,
        text_input: Union[str, List[str], None] = Body(
            default=None),
        file: Union[UploadFile, None] = File(None)):
    """
    Process an input file or text for inference using an MLflow-registered model.

    This function loads a model from an MLflow Tracking Server using the provided 
    `model_id` and `mlflow_url`, then processes the given input (either a string, 
    list of strings, or an uploaded file) to produce predictions.

    It attempts to load the model using multiple MLflow flavors (e.g., sklearn, 
    lightgbm, xgboost, tensorflow, etc.) until one is successful.

    Args:
        model_id (str):
            The registered MLflow model id 
            Example: "m-c8473a40daaf42288b729df4d471043a".

        mlflow_url (str):
            The MLflow Tracking Server URL. This is used to set the tracking URI 
            before loading the model.
            Example: "http://localhost:5000" or "http://mlflow-server:5000".

        text_input (Union[str, List[str], None], optional):
            The textual input for prediction. Can be a single string or a list 
            of strings. Defaults to None. If provided, `file` must be None.
            Example: "CCO" (SMILES notation for ethanol).

        file (Union[UploadFile, None], optional):
            An uploaded file containing input data for prediction. Defaults to None. 
            If provided, `text_input` must be None.

    Raises:
        HTTPException:
            If both `text_input` and `file` are provided at the same time, 
            a 400 Bad Request error is returned.

    Returns:
        Any:
            The result of the prediction.

    Notes:
        - The function tries different MLflow model flavors in the following order:
          sklearn, lgbm, xgboost, lightgbm, tensorflow, keras, pytorch, catboost, 
          statsmodels, pyfunc (fallback).
        - If no flavor successfully loads the model, an error is logged but no 
          explicit exception is raised at load time.
    """

    mlflow_tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI')

    mlflow_tracking_uri = mlflow_url
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = f"models:/{model_id}"
    if text_input and file:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one input type: string, list, or file"
        )
    flavors_to_try = [
        ('sklearn', mlflow.sklearn),
        ('lgbm', mlflow.lightgbm),
        ('xgboost', mlflow.xgboost),
        ('lightgbm', mlflow.lightgbm),
        ('tensorflow', mlflow.tensorflow),
        ('keras', mlflow.keras),
        ('pytorch', mlflow.pytorch),
        ('catboost', mlflow.catboost),
        ('statsmodels', mlflow.statsmodels),
        ('pyfunc', mlflow.pyfunc)  # This should always work as fallback
    ]

    logged_model = None
    successful_flavor = None

    for flavor_name, flavor_module in flavors_to_try:
        try:
            print(f"Trying to load with {flavor_name}...")
            logged_model = flavor_module.load_model(model_uri)
            successful_flavor = flavor_name
            print(f"✓ Successfully loaded with {flavor_name}!")
            print(f"Model type: {type(logged_model)}")
            break
        except Exception as e:
            print(f"✗ Failed with {flavor_name}: {str(e)[:100]}...")
            continue

    if logged_model is None:
        print("Failed to load model with any flavor!")
    else:
        print(
            f"\n🎉 Model loaded successfully using {successful_flavor} flavor!")

    # text_input = "h20"

    # Process string or list input
    if text_input is not None:
        if isinstance(text_input, str):
            compound = list(text_input)
            # res = model.screen(text_input)
            res = screen_smiles(logged_model, compound, [
                                "HitGenBinaryECFP4"])
            return res

        elif isinstance(text_input, list):
            split_list = [element.split(',') for element in text_input]

            # Flatten the list of lists into a single list if needed
            result_list = [item for sublist in split_list for item in sublist]

            res = screen_smiles(logged_model, result_list, [
                                "HitGenBinaryECFP4"])
            # return res
            return {
                "message": "File processed successfully",
                "smiles_name": result_list,
                "result": res,
            }

    if file:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, detail="Only CSV files are accepted")
        try:
            # Read the file content
            contents = await file.read()

            # Create a StringIO object
            csv_file = io.StringIO(contents.decode('utf-8'))

            # Read with pandas
            df = pd.read_csv(csv_file)

            if 'SMILE' not in df.columns and 'SMILES' not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail="No SMILE/SMILES column found in the CSV file"
                )

             # Get the correct column name
            smile_column = 'SMILE' if 'SMILE' in df.columns else 'SMILES'

            # Reset file pointer for potential future reads

            original_length = len(df)
            if original_length > 10:
                df = df.head(10)
                truncated = True
            else:
                truncated = False
            smiles_list = df[smile_column].dropna().tolist()
            res = screen_smiles(logged_model, smiles_list, [
                                "HitGenBinaryECFP4"])

            return {
                "message": "File processed successfully" + (" (truncated to first 10 rows)" if truncated else ""),
                "smiles_name": smiles_list,
                "result": res,
            }

        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=400, detail="The CSV file is empty")
        except pd.errors.ParserError:
            raise HTTPException(
                status_code=400, detail="Error parsing the CSV file")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing file: {str(e)}")

    raise HTTPException(
        status_code=400,
        detail="Invalid input type"
    )


# fastapi dev main.py
