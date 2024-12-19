import csv
from google.cloud import storage
from multiprocessing import Pool
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import gzip
import io


def list_files_in_gcs_folder(bucket_name, folder_path):
    """
    Lists all files in a specific folder within a Google Cloud Storage bucket.

    :param bucket_name: Name of the GCS bucket.
    :param folder_path: Path to the folder within the bucket.
                        Example: 'path/to/folder/' (should end with a slash).
    :return: A list of file names (paths) in the specified folder.
    """
    file_list = []
    try:
        # Initialize GCS client
        client = storage.Client()

        # Get the bucket
        bucket = client.bucket(bucket_name)

        # List blobs in the specified folder
        blobs = client.list_blobs(bucket, prefix=folder_path)

        # Collect the file names
        for blob in blobs:
            # Ensure we exclude the folder itself
            if not blob.name.endswith('/'):
                file_list.append(blob.name)
    except Exception as e:
        print(f"An error occurred: {e}")

    return file_list


def read_csv_to_list(file_path):
    """
    Reads a CSV file and converts a specific column into a list.

    :param file_path: Path to the CSV file.
    :param column_index: Index of the column to convert to a list (0-based).
    :return: A list of values from the specified column.
    """
    result_list = []
    smiles = []
    names = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                # if count == 10:
                #     break
                # count += 1
                # Ensure the column index exists in the row
                if len(row) > 2:
                    smiles.append(row[1])
                    names.append(row[2])

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(smiles[:5])
    print(names[:5])
    print(f"Count of smiles is {len(smiles)} and total names are {len(names)}")

    return result_list


def read_csv_from_gcs(args):
    """
    Reads a CSV file from Google Cloud Storage and converts a specific column into a list.

    :param bucket_name: Name of the GCS bucket.
    :param file_path: Path to the CSV file within the bucket.
    :param column_index: Index of the column to convert to a list (0-based).
    :return: A list of values from the specified column.
    """
    bucket_name, file_path = args
    smiles = []
    names = []
    smiles_idx = 1  # assume first column is smiles
    names_idx = 2
    print(f"bucket name and file {bucket_name}, {file_path}")
    try:
        # Initialize GCS client
        client = storage.Client()

        # Get the bucket and the blob (file)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        # Download the CSV file content as a string
        csv_content = blob.download_as_text()

        # Parse the CSV content
        reader = csv.reader(csv_content.splitlines())
        count = 0
        for row in reader:
            count += 1
            # Ensure the column index exists in the row
            if len(row) > 2:
                smiles.append(row[smiles_idx])
                if names_idx < len(row):  # Avoid IndexError for missing columns
                    names.append(row[names_idx].strip())
                else:
                    # Append None if name column is missing
                    names.append(None)

                # result_list.append(row[column_index])
    except Exception as e:
        print(f"An error occurred: {e}")
    print("smiles---", smiles[:10])
    print("Total count is", count)

    return len(smiles)


# Example usage
# Replace with your file path
# file_path = './data/smiles/dataset_xaa_enamine_dataset.csv'
# column_index = 2  # Replace with the desired column index
# # column_data = read_csv_to_list(file_path)
# read_csv_from_gcs("enamine-vs", "dataset/xaa_enamine_dataset.csv")

# print(column_data)


def read_tsv_from_gcs(
    bucket_name: str,
    file_name: str,
    target_column: str,
):
    """
    Reads a .tsv.gz file from a GCP bucket and extracts specified columns.

    Args:
        bucket_name (str): Name of the GCP bucket.
        file_name (str): Path to the .tsv.gz file within the bucket.
        columns_of_interest (list): List of column names to extract.
        target_column (str): The column containing labels.
        binarize (bool): Whether to binarize the data columns.

    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray]: Extracted features and labels.
    """
    # Initialize GCP storage client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download and decompress the file on-the-fly
    with io.BytesIO() as file_buffer:
        blob.download_to_file(file_buffer)
        file_buffer.seek(0)  # Reset buffer pointer
        with gzip.GzipFile(fileobj=file_buffer, mode='rb') as gz_file:
            # Use pandas to read the decompressed .tsv file
            df = pd.read_csv(gz_file, sep='\t')

    # Separate the data into positive and negative samples
    positive_samples = df[df[target_column] == 1]
    negative_samples = df[df[target_column] == 0]

    # Sample 30% from each class
    positive_sampled = positive_samples.sample(frac=0.3, random_state=42)
    negative_sampled = negative_samples.sample(frac=0.3, random_state=42)

    # Combine sampled data and shuffle
    sampled_df = pd.concat([positive_sampled, negative_sampled]).sample(
        frac=1, random_state=42)
    sampled_df.to_csv("./data/subset_WDR91.tsv.gz", sep='\t',
                      index=False, compression='gzip')


if __name__ == '__main__':
    read_tsv_from_gcs("aircheck-ml-ready", "S202309/WDR91.tsv.gz", "Label")
    exit()
    bucket_name = 'enamine-vs'  # Replace with your GCS bucket name
    # Replace with the folder path (include trailing slash)
    folder_path = 'dataset/'
    file_path = './data/smiles/dataset_xaa_enamine_dataset.csv'
    # read_csv_to_list(file_path)
    # exit()

    files = list_files_in_gcs_folder(bucket_name, folder_path)
    print("Files in folder:", len(files))
    print(files[0])
    files = files[:6]
    num_cores = cpu_count() - 1  # Use all available cores except one
    print(num_cores)
    total_count = 0
    file_args = [(bucket_name, file) for file in files]

    with Pool(num_cores) as p:
        results = p.map(read_csv_from_gcs, file_args)
        total_count = sum(results)
    print("Total count is--", total_count)

    # for result in results:
    #     print(result)
