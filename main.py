from os import truncate
from turtle import mode
from typing import Union, List

from fastapi import FastAPI,  UploadFile, File, HTTPException, Body
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd
import io
import csv
from predict import Model

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/model/{run_id}")
def get_model(
        tracking_url: str,
        run_id: str, run: Union[str, None] = None):
    mlflow.set_tracking_uri("http://localhost:8080")
    logged_model = 'runs:/e9c5b7d11f9d45a6a8db741b6ff08ade/model'
    loaded_model = mlflow.lightgbm.load_model(logged_model)
    print("model------", loaded_model)
    return {"model used": str(loaded_model)}


class InputModel(BaseModel):
    text_input: Union[str, List[str], None] = None


@app.post("/process-input/")
async def process_input(
    text_input: Union[str, List[str], None] = Body(
        default=None),
    file: Union[UploadFile, None] = File(None)
):
    # For debugging
    print(f"Received text_input: {type(text_input)}")
    print(f"Received text_input: {(text_input)}")
    print(f"Received file: {file}")
    model = Model()

    if text_input and file:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one input type: string, list, or file"
        )

    # Process string or list input
    if text_input is not None:
        if isinstance(text_input, str):
            res = model.screen(text_input)
            return {
                "message": "Processed string input",
                "data": res
            }
        elif isinstance(text_input, list):
            split_list = [element.split(',') for element in text_input]

            # Flatten the list of lists into a single list if needed
            result_list = [item for sublist in split_list for item in sublist]

            print("SMILES List", result_list)
            res = model.screen(result_list)

            return {
                "message": "Processed list input",
                "data": res
            }

    # Process file input

    if file:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, detail="Only CSV files are accepted")
        try:
            # Read the file content
            contents = await file.read()
            print("contents---", contents)

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
            if len(df) > 10:
                df = df.head(10)
                truncated = True
            else:
                truncated = False
            smiles_list = df[smile_column].dropna().tolist()
            print("SMILES List", smiles_list)
            res = model.screen(smiles_list)

            return {
                "message": "File processed successfully" + (" (truncated to first 20 rows)" if truncated else ""),
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


class InputModel(BaseModel):
    data: Union[str, List[str]]


@app.post("/process-data/")
async def process_string(input_data: InputModel):
    if isinstance(input_data.data, str):
        result = f"Received a string: {input_data.data}"
    elif isinstance(input_data.data, list):

        result = f"Received a list of strings: {', '.join(input_data.data)}"
    else:
        raise HTTPException(status_code=400, detail="Invalid input type")

    return {"message": result}


# fastapi dev main.py
