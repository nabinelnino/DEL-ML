from typing import Dict, List, Union, Tuple
import csv
import time
import io
import gzip
import logging
import os
import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from rdkit import DataStructs
from tqdm import tqdm
import sys

from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers


from utils.config_parser import ManageModelDataset


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

    def _read_data(self, file_path: str, fps: list | str, label: str, model_name: str, config_file_path: str, binarize: bool = True, dry_run: str = True):
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
            isAlreadyExist = ManageModelDataset.manage_model_dataset_yaml(
                file_path=config_file_path, model_name=model_name, dataset_url=file_path, training_col=fps)
            if isAlreadyExist:
                print("File and model already exist")
                sys.exit()
            X, y = self.read_from_loca_file(
                file_path, fps, label, binarize, dry_run)
            return X, y
        else:
            config_bucket, config_file = self._parse_gcp_path(config_file_path)
            isAlreadyExist = ManageModelDataset.manage_model_dataset_gcs(bucket_name=config_bucket,
                                                                         file_path=config_file, model_name=model_name, dataset_url=file_path, training_col=fps)
            if isAlreadyExist:
                sys.exit()
            bucket_name, blob_name = self._parse_gcp_path(file_path)
            X, y = self.read_tsv_from_gcs(
                bucket_name, blob_name, fps, label, binarize)
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

    def read_from_loca_file(self, file_path, fps, label, binarize, dry_run):
        logging.info(f"Starting to read file: {file_path}")
        logging.info(f"File size: {os.path.getsize(file_path)} bytes")
        X = {}
        y = []

        for fp in tqdm(fps):
            if binarize:
                fp_key = self.HITGEN_FPS_COLS_MAP_BINARY.get(fp, None)
            if fp_key is None:
                raise ValueError(f"cannot make {fp} binary for HitGen")
            else:
                fp_key = self.HITGEN_FPS_COLS_MAP_BINARY.get(fp)
            y = []
            _x = []
            count = 0

            with gzip.open(file_path, 'rt', newline='', encoding='utf-8') as f:
                header = f.readline().strip().split("\t")

                label_index = header.index(label)
                fp_idx = header.index(fp)
                for line in tqdm(f):

                    if dry_run and count == 10000:
                        break
                    count += 1

                    if line.strip() == "":
                        continue
                    splits = line.strip().split("\t")

                    y.append(int(splits[label_index]))
                    if binarize:
                        _x.append(
                            [1 if int(_) > 0 else 0 for _ in splits[fp_idx].split(",")])
                    else:
                        _x.append([int(_)
                                   for _ in splits[fp_idx].split(",")])

                _x = np.array(_x)

                if _x.ndim == 1:
                    _x = _x.reshape(-1, 1)

                X[fp_key] = _x

        logging.info(f"Final shapes - X: {[(k, v.shape) for k, v in X.items()]} \
                    , Y: {np.array(y).shape}")

        return X, np.array(y)

    def read_tsv_from_gcs(self, bucket_name, file_name, columns_of_interest, target_column, binarize: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Reads a .tsv.gz file from a GCP bucket and extracts specified columns.

        Args:
            bucket_name (str): Name of the GCP bucket.
            file_name (str): Path to the .tsv.gz file within the bucket.
            columns_of_interest (list): List of column names to extract.

        Returns:
            pd.DataFrame: DataFrame containing only the specified columns.
        """
        # Initialize GCP storage client
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        columns_of_interest.append(target_column)

        # Download and decompress the file on-the-fly
        with io.BytesIO() as file_buffer:
            blob.download_to_file(file_buffer)
            file_buffer.seek(0)  # Reset buffer pointer
            with gzip.GzipFile(fileobj=file_buffer, mode='rb') as gz_file:
                # Use pandas to read the decompressed .tsv file
                df = pd.read_csv(gz_file, sep='\t',
                                 usecols=columns_of_interest)

        X = {}

        y = df[target_column].values
        df = df.drop(target_column, axis=1)

        for column in df.columns:
            fp_key = self.HITGEN_FPS_COLS_MAP_BINARY.get(column)
            values = df[column].apply(lambda x: [int(val)
                                      for val in str(x).split(',')]).tolist()
            values = np.array(values)

            # Binarize if required
            if binarize:
                values = np.where(values > 0, 1, 0)

            X[fp_key] = values

            # values = df[column].values
            # values = df[column].apply(safe_convert_to_int).values
            # if binarize:
            #     values = np.where(values > 0, 1, 0)

            # if values.ndim == 1:
            #     values = values.reshape(-1, 1)
            #     X[fp_key] = values

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
