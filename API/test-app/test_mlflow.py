
from fps_conversion import screen_smiles
import mlflow
import os
from dotenv import load_dotenv
from google.cloud import storage
import requests

load_dotenv()


service_account_path = '../service_account.json'


# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    print("Service account credentials set.")
else:
    print("Service account file not found. Skipping credential setup.")


def process_input():

    mlflow_tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI')
    print("fsdkgsgsdgsdgsdgsd", mlflow_tracking_uri)
    mlflow_tracking_uri = "http://localhost:5001/"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print("mlflow server-----", mlflow_tracking_uri)
    run_id = '6483103c26dd4220960eb16f37e437d0'
    # run_id = '47b300294e244d358935503cf84c6bfc'
    # model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    # print("type of model---", type(model))
    # model_id = 'm-ebe794cd279a49eab0793ec4a8146ef0'
    model_id = 'm-d9ca64d647b44eb2aab4687da9977836'
    model_uri = f"models:/{model_id}"

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

    # model = mlflow.sklearn.load_model(f"models:/{model_id}")
    # logged_model = mlflow.lightgbm.load_model(f"models:/{model_id}")
    # print("Loaded model type:", type(logged_model))

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

    # exit()

    # model = mlflow.sklearn.load_model(f"models:/{model_id}")
    # print("Loaded model type:", type(model))

    # exit()
    # logged_model = f"{run.info.artifact_uri}/model.pkl"

    print("modelllll", logged_model)
    # res = screen_smiles(loaded_model, [compound], ["HitGenBinaryECFP4"])

    text_input = "h20"

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
            print("RESSSSSSSSSS", res)
            return {
                "message": "File processed successfully",
                "smiles_name": result_list,
                "result": res,
            }

    # Process file input


def get_metrics():
    mlflow_tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI')
    print("fsdkgsgsdgsdgsdgsd", mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print("mlflow server-----", mlflow_tracking_uri)
    run_id = "a864763e557445cab9915f0316b08f27"
    url = f"{mlflow_tracking_uri}/api/2.0/mlflow/runs/get"

    # Query parameters
    params = {
        'run_id': run_id
    }

    # Send GET request to MLflow server
    response = requests.get(url, params=params)

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
    # get_metrics()
