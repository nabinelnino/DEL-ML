
from http import client
import mlflow
from mlflow.tracking import MlflowClient
import os
import requests
import pickle
# import 
from fps_conversion import screen_smiles


service_account_path = '../service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    print("Service account credentials set.")
else:
    print("Service account file not found. Skipping credential setup.")





def load_model_from_mlflow_server(run_id):
    """
    Try loading from different MLflow URI formats
    """
    # Different MLflow URI patterns
    mlflow_uris = [
        f"runs:/{run_id}/model",           # Standard format
        f"models:/{run_id}/artifacts/model", # Alternative format 1
        f"{run_id}/artifacts/model",       # Direct path format
        f"artifacts/{run_id}/model",       # Alternative format 2
        f"models/{run_id}/model",          # Alternative format 3
    ]
    
    for uri in mlflow_uris:
        try:
            print(f"Trying MLflow URI: {uri}")
            loaded_model = mlflow.lightgbm.load_model(uri)
            print(f"✓ Successfully loaded from: {uri}")
            return loaded_model, uri
        except Exception as e:
            print(f"✗ Failed to load from {uri}: {e}")
            continue
    
    return None, None








def process_input():

    # if not mlflow_tracking_uri:
    #     return {"message": "please provide ml server host url in the .env file"}
    text_input = "N#CC1CCN(C(=O)C2=CC=C(OC3CCC3)C=C2)CC1"
    # mlflow_url = "http://34.130.80.214:5000/"
    # run_id ="337c17172bf9472f9befba4bdaaa8171"
    mlflow_url = "http://34.130.56.87:5000/"
    run_id ="8878c4ad95b74adebc8dfb3b2a3b1eff"
    compound = [text_input]
    mlflow_tracking_uri = mlflow_url
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    print("mlflow server-----", mlflow_tracking_uri)
    mlflow_uris = [
        f"runs:/{run_id}/model",
        f"runs:/{run_id}/artifacts/model",  # If using model registry
    ]

    for uri in mlflow_uris:
        try:
            print(f"Trying: {uri}")
            loaded_model = mlflow.lightgbm.load_model(uri)
            print(f"✓ Loaded from: {uri}")
            res = screen_smiles(loaded_model, compound, ["HitGenBinaryECFP4"])
            print('RESSSSSSS', res)
            break
        except Exception as e:
            print(f"✗ Failed {uri}: {e}")
            continue






    # try:
    #     # Try framework-specific loader first
    #     logged_model = f"runs:/{run_id}/model"
    #     loaded_model = mlflow.lightgbm.load_model(logged_model)
    #     res = screen_smiles(loaded_model, compound, [
    #                             "HitGenBinaryECFP4"])
    #     print('RESSSSSSS000000000', res)
    # except Exception as e:
    #     print(f"LightGBM loader failed: {e}")
    #     try:
    #         # Fallback to pyfunc loader
    #         logged_model = f"{run_id}/artifacts/model/model"
    #         loaded_model = mlflow.lightgbm.load_model(logged_model)
    #         # loaded_model = mlflow.pyfunc.load_model(logged_model)
    #         res = screen_smiles(loaded_model, compound, [
    #                             "HitGenBinaryECFP4"])
    #         print('RESSSSSSS', res)
    #     except Exception as e2:
    #         print(f"Pyfunc loader also failed: {e2}")

   

    # Process file input

   
    # This should never be reached due to the earlier check



def get_metrics(mlflow_server_url: str, run_id: str):

    url = f"{mlflow_server_url}/api/2.0/mlflow/runs/get"

    # Query parameters
    params = {
        'run_id': run_id
    }

    # Send GET request to MLflow server
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Errrrrrr")

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




if __name__ == "__main__":
    process_input()
    # Usage
    

# fastapi dev main.py
