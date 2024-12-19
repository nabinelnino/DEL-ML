from tqdm import tqdm

import io
from typing import Optional
from tqdm import tqdm
from google.cloud import storage
import gzip


def read_smile(filepath):
    with open(filepath, "r") as f:
        first_line = f.readline().strip().split("\t")

        # Check if first line looks like a header or data
        try:
            # Attempt to find column indices
            smiles_idx = first_line.index("smiles")
            names_idx = first_line.index("names")

            # If successful, this is a header row
            header = first_line
            print("Header found:", header)
        except ValueError:
            # If column names not found, treat first line as data
            smiles_idx = 0  # assume first column is smiles
            names_idx = 1  # assume second column is names (if exists)
            header = None

            # Reset file pointer and use first line as first data row
            f.seek(0)

        names = []
        smiles = []
        count = 0

        for line in tqdm(f):
            splits = line.strip().split("\t")

            # Ensure we have enough columns
            if len(splits) > smiles_idx:
                smiles.append(splits[smiles_idx].strip())

            # Check if names column exists
            if names_idx is not None and names_idx < len(splits):
                names.append(splits[names_idx].strip())
            else:
                names.append(None)

            count += 1
            if count == 10:
                break

    return smiles, names


def screen_from_bucket():
    # file_path = "gs://test-aircheck/mlops-test/sample_smiles.smi.gz"
    # output_file = "gs://test-aircheck/mlops-test/result.smi"

    # bucket_name, file_name = "test-aircheck", "mlops-test/mcule_purchasable_full_241101.smi.gz"
    bucket_name, file_name = "test-aircheck", "mlops-test/sample_smiles.smi.gz"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    # Download blob as a stream
    download_stream = io.BytesIO()
    blob.download_to_file(download_stream)
    download_stream.seek(0)
    count = 0

    with gzip.GzipFile(fileobj=download_stream, mode='rb') as gz_file:
        names = []
        smiles = []
        lines = io.TextIOWrapper(gz_file, encoding='utf-8')
        header = next(lines).strip().split('\t')
        print("herererer", header)
        # Replace with actual SMILES column name
        try:
            smiles_idx = header.index("smiles")
            names_idx = header.index("names")
        except ValueError:
            # If column names not found, treat first line as data
            smiles_idx = 0  # assume first column is smiles
            names_idx = 1

        count = 0
        for line in lines:
            splits = line.strip().split('\t')
            if count == 10:
                break
            count += 1

            # Append values based on column indices
            smiles.append(splits[smiles_idx].strip())
            if names_idx < len(splits):  # Avoid IndexError for missing columns
                names.append(splits[names_idx].strip())
            else:
                names.append(None)  # Append None if name column is missing
        return smiles, names


if __name__ == "__main__":
    smiles, names = screen_from_bucket()
    print(smiles)
    exit()
    # smiles, names = read_smile("../data/mcule_purchasable_full_241101.smi")
    smiles, names = read_smile("../data/sample_smiles.smi")
    print(smiles)
