import yaml
import os
from google.cloud import storage
from io import BytesIO
import io
import gzip
from utils.logger import logging


class MLConfigParser:
    def __init__(self, yaml_file_path: str, dict_key: str):
        """
        Initialize the MLConfigParser with the path to a YAML file.

        :param yaml_file_path: Path to the YAML configuration file
        """
        self.yaml_file_path = yaml_file_path
        self.config = {}
        self.dict_key = dict_key

    def read_yaml(self):
        """
        Reads the YAML file and converts it into a dictionary.

        :raises FileNotFoundError: If the file does not exist
        :raises yaml.YAMLError: If the file content is not valid YAML
        """
        try:
            with open(self.yaml_file_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The file {self.yaml_file_path} does not exist.")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error: Failed to parse YAML file. Details: {str(e)}")

    def validate_config(self):
        """
        Validates the YAML configuration dictionary.

        :raises ValueError: If required keys are missing or invalid
        """

        ml_config = self.config.get(self.dict_key)
        if not isinstance(ml_config, dict):
            raise ValueError("Error: 'ml_config' should be a dictionary.")

    def get_config(self):
        """
        Reads and validates the YAML file and returns the configuration.

        :return: Parsed and validated configuration dictionary
        """
        self.read_yaml()
        self.validate_config()
        return self.config[self.dict_key]


class ManageModelDataset:

    def ramove_dataseta_and_model(self, config_path, model_name, source):
        if not config_path.startswith('gs://'):
            self.remove_model_dataset_from_file(
                config_path, model_name, source)
        else:
            self.delete_model_and_file_fromgcs(config_path, model_name, source)

    @staticmethod
    def manage_model_dataset_yaml(file_path, model_name, dataset_url, training_col) -> bool:
        """
        Manage a YAML file to store model names and dataset URLs.

        Args:
            file_path (str): Path to the YAML file.
            model_name (str): Name of the model.
            dataset_url (str): URL of the dataset.

        Returns:
            bool: True if the model and dataset combination is added, False if it already exists.
        """
        # Ensure the directory for the file exists
        logging.info(f"file path {file_path} model name \
              {model_name} and dataset location {dataset_url}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Initialize the data structure
        data = {}

        # Check if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    data = {}

        # Check if the model and dataset model already exists
        if model_name in data and dataset_url in data[model_name] and training_col in data[model_name]:
            return True  # Combination already exists

        # Update the data
        if model_name not in data:
            data[model_name] = []
        data[model_name].append(dataset_url)
        data[model_name].append(training_col)

        # Write the updated data back to the file
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f)

        return False

    def remove_model_dataset_from_file(self, file_path, model_name, dataset_url) -> bool:
        """
        Manage a YAML file to store model names and dataset URLs.

        Args:
            file_path (str): Path to the YAML file.
            model_name (str): Name of the model.
            dataset_url (str): URL of the dataset.

        Returns:
            bool: True if the model and dataset combination is added, False if it already exists.
        """
        # Ensure the directory for the file exists
        logging.info((f"file path {file_path} model name \
              {model_name} and dataset location {dataset_url}"))

        # Initialize the data structure
        data = {}

        # Check if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    data = {}

        # Check if the model and dataset model already exists
        if model_name in data and dataset_url in data[model_name]:
            data[model_name].remove(dataset_url)

            # If the list becomes empty after removal, delete the model_name key
            if not data[model_name]:
                del data[model_name]

        # Write the updated data back to the file
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f)

    @staticmethod
    def manage_model_dataset_gcs(bucket_name: str, file_path: str, model_name: str, dataset_url: str, training_col: list):
        """
        Manage a YAML file in a GCP bucket to store model names and dataset URLs.

        Args:
            bucket_name (str): Name of the GCP bucket.
            file_path (str): Path to the YAML file in the bucket.
            model_name (str): Name of the model.
            dataset_url (str): URL of the dataset.

        Returns:
            bool: True if the model and dataset combination is added, False if it already exists.
        """
        # Initialize the GCP storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        # Initialize the data structure
        data = {}

        # Check if the file exists in the bucket
        if blob.exists():
            # Read the existing file from the bucket
            yaml_content = blob.download_as_bytes()
            try:
                data = yaml.safe_load(yaml_content) or {}
            except yaml.YAMLError:
                data = {}

        # Check if the model and dataset combination already exists
        if model_name in data and dataset_url in data[model_name] and training_col in data[model_name]:
            return True  # Combination already exists

        # Update the data
        if model_name not in data:
            data[model_name] = []
        data[model_name].append(dataset_url)
        data[model_name].append(training_col)

        # Write the updated data back to the bucket
        updated_yaml_content = yaml.safe_dump(data)
        blob.upload_from_string(updated_yaml_content,
                                content_type="application/x-yaml")

        return False

    def delete_model_and_file_fromgcs(self, config_url, model_name, dataset_url):
        # Initialize the GCS client and get the blob
        bucket_name, blob_name = self._parse_gcp_path(config_url)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download the existing YAML file
        yaml_content = blob.download_as_string()
        data = yaml.safe_load(yaml_content) or {}

        # Check if the model_name exists in the YAML data
        if model_name in data and dataset_url in data[model_name]:
            # Remove the last occurrence of dataset_url in the list
            data[model_name].remove(dataset_url)

            # If the list becomes empty after removal, delete the model_name key
            if not data[model_name]:
                del data[model_name]

        # Write the updated data back to the bucket
        updated_yaml_content = yaml.safe_dump(data)
        blob.upload_from_string(updated_yaml_content,
                                content_type="application/x-yaml")

    def _parse_gcp_path(self, gcp_path):
        """
        Parse GCP bucket path into bucket name and blob name.

        :param gcp_path: GCP bucket path
        :return: Tuple of (bucket_name, blob_name)
        """
        if not gcp_path.startswith('gs://'):
            raise ValueError("Invalid GCP path. Must start with 'gs://'")

        # Remove 'gs://' and split into bucket and blob
        path_parts = gcp_path[5:].split('/', 1)
        if len(path_parts) != 2:
            raise ValueError("Invalid GCP path format")

        return path_parts[0], path_parts[1]

    @staticmethod
    def write_tsv_to_gcs(self, file_name, names, smiles, preds, confs):
        # Create an in-memory file object

        bucket_name, file_name = self._parse_gcp_path(file_name)
        tsv_buffer = io.StringIO()

        # Write the TSV content to the buffer
        for n, s, p, c in zip(names, smiles, preds, confs):
            tsv_buffer.write(f"{n}\t{s}\t{round(float(p), 4)}\t \
                             {round(float(c), 4)}\n")

        # Reset the buffer position to the beginning
        tsv_buffer.seek(0)

        # Initialize GCS client
        client = storage.Client()

        # Get the bucket
        bucket = client.bucket(bucket_name)

        # Create a blob (object in GCS)
        blob = bucket.blob(file_name)

        # Upload the TSV data from the buffer
        blob.upload_from_string(tsv_buffer.getvalue(),
                                content_type="text/tab-separated-values")

        logging.info(f"File successfully written to GCS bucket \
              {bucket_name} as {file_name}")

    @staticmethod
    def stream_gz_from_bucket(self, file_name):
        # Initialize storage client
        bucket_name, file_name = self._parse_gcp_path(file_name)
        storage_client = storage.Client()

        # Get bucket and blob (file)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Download blob as a stream
        download_stream = io.BytesIO()
        blob.download_to_file(download_stream)
        download_stream.seek(0)

        # Decompress stream
        with gzip.GzipFile(fileobj=download_stream, mode='rb') as gz_file:
            names = []
            smiles = []
            count = 0

            for line in io.TextIOWrapper(gz_file, encoding='utf-8'):
                count += 1
                splits = line.strip().split('\t')
                smiles.append(splits[0].strip())
                if len(splits) > 1:
                    names.append(splits[1].strip())
                if count % 10000 == 0:
                    print("Length of simes", len(smiles), smiles[-1])

            print("total count", count)

        return smiles, names

    @staticmethod
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
