
from email import message
from typing import Union, List

from fastapi import FastAPI,  UploadFile, File, HTTPException, Body
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
import pandas as pd
import io
import os
import numpy as np
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


@app.post("/process-input/")
async def process_input(
    run_id: str,
    text_input: Union[str, List[str], None] = Body(
        default=None),
    file: Union[UploadFile, None] = File(None)
):
    mlflow_tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri:
        return {"message": "please provide ml server host url in the .env file"}

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.lightgbm.load_model(logged_model)
    # res = screen_smiles(loaded_model, [compound], ["HitGenBinaryECFP4"])

    if text_input and file:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one input type: string, list, or file"
        )

    # Process string or list input
    if text_input is not None:
        if isinstance(text_input, str):
            compound = list(text_input)
            # res = model.screen(text_input)
            res = screen_smiles(loaded_model, compound, [
                                "HitGenBinaryECFP4"])
            return res

        elif isinstance(text_input, list):
            split_list = [element.split(',') for element in text_input]

            # Flatten the list of lists into a single list if needed
            result_list = [item for sublist in split_list for item in sublist]

            res = screen_smiles(loaded_model, result_list, [
                                "HitGenBinaryECFP4"])
            return res

    # Process file input

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
            res = screen_smiles(loaded_model, smiles_list, [
                                "HitGenBinaryECFP4"])

            return {
                "message": "File processed successfully" + (" (truncated to first 10 rows)" if truncated else ""),
                "filename": file.filename,
                "content_type": file.content_type,
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "first_few_rows": smiles_list,
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

    # This should never be reached due to the earlier check
    raise HTTPException(
        status_code=400,
        detail="Invalid input type"
    )


# fastapi dev main.py
