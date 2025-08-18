import pyarrow.parquet as pq
from typing import Dict, List, Tuple
import io
import logging
import os
import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from rdkit import DataStructs
from tqdm import tqdm
import sys
import tempfile

from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers
from pyarrow.parquet import ParquetFile
import pyarrow as pa


class DataReader:
    def __init__(self):
        """
        Initialize the DataReader with fingerprint column mapping.

        :param fps_map_binary: Mapping of fingerprint columns for binary conversion
        """
        # self.HITGEN_FPS_COLS_MAP_BINARY = fps_map_binary

        self.HITGEN_FPS_COLS_MAP_BINARY = {
            'ECFP4': "HitGenBinaryECFP4",
            'ECFP6': "HitGenBinaryECFP6",
            'FCFP4': "HitGenBinaryFCFP4",
            'FCFP6': "HitGenBinaryFCFP6",
            'AVALON': "HitGenBinaryAvalon",
            'ATOMPAIR': "HitGenBinaryAtomPair",
            'TOPTOR': "HitGenBinaryTopTor",
            'RDK': "HitGenBinaryRDK",
            'MACCS': "HitGenBinaryMACCS",
        }

    def list_gcs_files(self, bucket_name: str, folder_path: str) -> List[str]:
        """
        Lists all files in a specific folder (prefix) in a GCS bucket.

        Args:
            bucket_name (str): The name of the GCS bucket.
            folder_path (str): The folder path within the bucket (prefix).

        Returns:
            List[str]: A list of full file paths within the folder.
        """
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Ensure folder_path ends with "/" to match prefix behavior
        if not folder_path.endswith('/'):
            folder_path += '/'

        blobs = bucket.list_blobs(prefix=folder_path)
        file_list = [
            f"gs://{bucket_name}/{blob.name}" for blob in blobs if not blob.name.endswith('/')]

        return file_list

    def create_model_name(self, dataset_location: str, partner_name: str) -> str:
        """
        Create a model name based on the dataset and partner name.

        :param dataset_location: Path to the dataset
        :param model_name: Name of the model
        :return: Formatted model name
        """
        target_name = dataset_location.split("/")[-1]
        target_name = target_name.split(".")[0]
        model_name = f"{target_name}_{partner_name}"
        return model_name

    def _read_data(self, file_path: str, fps: list | str, label: str, model_name: str, config_file_path: str, binarize: bool = True, dry_run: str = True, list_of_bucket_files=None):
        """
        Generic method to read data from local or GCP storage.

        :param file_path: Path to the file (local path or GCP bucket path)
        :param fps: Fingerprint columns to extract
        :param binarize: Whether to binarize the data
        :param is_gcp: Flag to indicate if reading from GCP bucket
        :return: Tuple of (X, y) data
        """
        logging.info(f"Starting to read file: {file_path}")
        if isinstance(fps, str):
            fps = [fps]

        invalid_fps = [
            fp for fp in fps if fp not in self.HITGEN_FPS_COLS_MAP_BINARY.keys()]
        if invalid_fps:
            possible_columns = ', '.join(
                self.HITGEN_FPS_COLS_MAP_BINARY.keys())
            raise ValueError(f"Invalid fingerprint(s): {', '.join(invalid_fps)}. "
                             f"Possible column names are: {possible_columns}, Please use possible column in config file.")

        # Determine file opening method based on storage type
        if not file_path.startswith('gs://'):

            X, y = self.read_from_local_parquet_file(
                file_path, fps, label, binarize, dry_run)
            return X, y
        else:
            for file in list_of_bucket_files:

                config_bucket, config_file = self._parse_gcp_path(
                    config_file_path)

                bucket_name, blob_name = self._parse_gcp_path(file_path)

                X, y = self.read_parquet_from_gcs(
                    bucket_name, blob_name, fps, label, binarize, dry_run)
                return X, y

    def safe_convert_to_int(self, value):
        """
        Safely convert a string value to integer, handling non-numeric strings.
        Returns 0 for non-numeric or empty values.
        """
        try:
            # Strip any whitespace and handle empty strings
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return 0

            return int(float(value))
        except (ValueError, TypeError):
            return 0

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

    # def read_from_loca_file(self, file_path, fps, label, binarize, dry_run):
    #     logging.info(f"Starting to read file: {file_path}")
    #     logging.info(f"File size: {os.path.getsize(file_path)} bytes")
    #     X = {}
    #     y = []

    #     for fp in tqdm(fps):
    #         if binarize:
    #             fp_key = self.HITGEN_FPS_COLS_MAP_BINARY.get(fp, None)
    #         if fp_key is None:
    #             raise ValueError(f"cannot make {fp} binary for HitGen")
    #         else:
    #             fp_key = self.HITGEN_FPS_COLS_MAP_BINARY.get(fp)
    #         y = []
    #         _x = []
    #         count = 0

    #         with gzip.open(file_path, 'rt', newline='', encoding='utf-8') as f:
    #             header = f.readline().strip().split("\t")

    #             label_index = header.index(label)
    #             fp_idx = header.index(fp)
    #             for line in tqdm(f):

    #                 if dry_run and count == 10000:
    #                     break
    #                 count += 1

    #                 if line.strip() == "":
    #                     continue
    #                 splits = line.strip().split("\t")

    #                 y.append(int(splits[label_index]))
    #                 if binarize:
    #                     _x.append(
    #                         [1 if int(_) > 0 else 0 for _ in splits[fp_idx].split(",")])
    #                 else:
    #                     _x.append([int(_)
    #                                for _ in splits[fp_idx].split(",")])

    #             _x = np.array(_x)

    #             if _x.ndim == 1:
    #                 _x = _x.reshape(-1, 1)

    #             X[fp_key] = _x

    #     logging.info(f"Final shapes - X: {[(k, v.shape) for k, v in X.items()]} \
    #                 , Y: {np.array(y).shape}")

    #     return X, np.array(y)

    def read_parquet_in_batches(file_path, columns, batch_size=10000):
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            yield batch.to_pandas()

    def read_from_local_parquet_file(self, file_path, fps, label, binarize=False, dry_run=False):
        logging.info(f"Reading Parquet file: {file_path}")
        logging.info(f"File size: {os.path.getsize(file_path)} bytes")

        # Resolve column keys
        fp_keys = []
        for fp in fps:
            key = self.HITGEN_FPS_COLS_MAP_BINARY.get(fp) if binarize else fp
            if key is None:
                raise ValueError(
                    f"No binary mapping found for fingerprint: {fp}")
            fp_keys.append(key)

        columns_to_read = [label] + fps

        pf = ParquetFile(file_path)
        total_rows = pf.metadata.num_rows
        print(f"Total rows: {total_rows}")
        dry_run = True
        if dry_run:

            rows_to_load = next(pf.iter_batches(
                columns=columns_to_read, batch_size=10000))
            df = pa.Table.from_batches([rows_to_load]).to_pandas()
        else:
            rows_to_load = next(pf.iter_batches(
                columns=columns_to_read, batch_size=total_rows))
            df = pa.Table.from_batches([rows_to_load]).to_pandas()

        y = df[label].astype(int).to_numpy()
        X = {}

        for fp, key in zip(fps, fp_keys):
            # Convert series of lists to 2D NumPy array efficiently
            arr = np.array(df[fp].tolist())

            if binarize:
                arr = (arr > 0).astype(int)

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            X[key] = arr

        logging.info(
            f"Shapes - X: {[(k, v.shape) for k, v in X.items()]}, y: {y.shape}")
        return X, y

    def read_parquet_from_gcs(self, bucket_name, file_name, fps, label, binarize=False, dry_run=False):
        logging.info(
            f"Reading Parquet file from GCS: gs://{bucket_name}/{file_name}")

        # Download blob to a temporary local file
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
            blob.download_to_filename(tmp_file.name)
            file_path = tmp_file.name
            logging.info(f"Downloaded to temp file: {file_path}")
            logging.info(f"File size: {os.path.getsize(file_path)} bytes")

            # Resolve fingerprint keys
            fp_keys = []
            for fp in fps:
                key = self.HITGEN_FPS_COLS_MAP_BINARY.get(
                    fp) if binarize else fp
                if key is None:
                    raise ValueError(
                        f"No binary mapping found for fingerprint: {fp}")
                fp_keys.append(key)

            pf = ParquetFile(file_path)
            total_rows = pf.metadata.num_rows
            available_columns = pf.schema.names
            print("Available columns", available_columns)
            final_label = "Label" if "Label" in available_columns else label
            print("Using target column:", final_label)
            print(f"Total rows: {total_rows}")
            columns_to_read = [final_label] + fps

            if dry_run:
                rows_to_load = next(pf.iter_batches(
                    columns=columns_to_read, batch_size=10000))
            else:
                rows_to_load = next(pf.iter_batches(
                    columns=columns_to_read, batch_size=total_rows))

            df = pa.Table.from_batches([rows_to_load]).to_pandas()

            y = df[final_label].astype(int).to_numpy()
            X = {}

            for fp, key in zip(fps, fp_keys):
                arr = np.array(df[fp].tolist())

                if binarize:
                    arr = (arr > 0).astype(int)

                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)

                X[key] = arr

            logging.info(
                f"Shapes - X: {[(k, v.shape) for k, v in X.items()]}, y: {y.shape}"
            )

            return X, y

    def cluster_leader_from_array(self, X, thresh: float = 0.65, use_tqdm: bool = False):
        """
        Generate a cluster id map for already featurized array such that each cluster centroid has a tanimoto similarity
        below the passed threshold. Each chemical that is not a centroid is a member of the cluster that it shares the
        highest similarity to.

        This means that not every cluster will have a total separation of 0.35 tanimoto distance.

        Notes
        -----
        passed smiles can be Mol objects for just the raw text SMILES

        Parameters
        ----------
        smis: list[Mol or str]
            chemicals to generate cluster index for
        thresh: float, default 0.65
            the tanimoto distance (1-similarity) that you want centroid to have
        use_tqdm: bool, default False
            track clustering progress with a tqdm progress bar

        Returns
        -------
        cluster_ids: np.ndarray[int]
            an array of cluster ids, index mapped to the passed smis

        """
        _fps = [DataStructs.CreateFromBitString(
            "".join(["1" if __ > 0 else "0" for __ in _])) for _ in X]
        lp = rdSimDivPickers.LeaderPicker()

        _centroids = lp.LazyBitVectorPick(_fps, len(_fps), thresh)
        _centroid_fps = [_fps[i] for i in _centroids]

        _cluster_ids = []
        for _fp in tqdm(_fps, disable=not use_tqdm, desc="assigning SMILES to clusters"):
            sims = BulkTanimotoSimilarity(_fp, _centroid_fps)
            _cluster_ids.append(np.argmax(sims))
        return np.array(_cluster_ids)
