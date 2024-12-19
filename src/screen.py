import shutil
import yaml
import argparse
import sys
import time
from train import Model, Screen
from .utils.data_reader import DataReader
from .utils.config_parser import MLConfigParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

import logging
import multiprocessing
import warnings
from rdkit import RDLogger
import abc
import inspect
import os
from copy import deepcopy
from functools import partial
from time import time
from typing import Dict, Union, Optional
import pickle
import gzip

import io
from typing import Optional
from tqdm import tqdm
from google.cloud import storage


import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier

# rdkit imports
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, MolFromSmiles, rdFingerprintGenerator
from rdkit.Chem import RDKFingerprint

from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score, \
    average_precision_score, RocCurveDisplay
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from tqdm import tqdm
import mlflow
import csv
import mlflow.sklearn


from rdkit import DataStructs
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# Some quick helper func to make things easier


from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (precision_score, recall_score, roc_auc_score,
                             balanced_accuracy_score, average_precision_score)
import matplotlib.pyplot as plt
from typing import Dict, Union, Optional
import numpy.typing as npt
from lightgbm import LGBMClassifier
from lightgbm import LGBMClassifier, plot_importance
from sklearn.metrics import roc_curve, RocCurveDisplay
from multiprocessing import Pool
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor


from utils.config_parser import ManageModelDataset
from dotenv import load_dotenv

from datetime import date
import rdkit
RDLogger.DisableLog('rdApp.*')
today = date.today()
load_dotenv()


warnings.filterwarnings(
    "ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

# mlflow.set_tracking_uri("http://34.130.56.87:5000/")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

service_account_path = '../service_account.json'
# service_account_path = '../app/service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    print("Service account credentials set.")
else:
    print("Service account file not found. Skipping credential setup.")


def configure_mlflow_tracking():
    # Read MLFLOW_TRACKING_URI from environment variables
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()
    print("mlflow uri is----", mlflow_uri)

    # If MLFLOW_TRACKING_URI is empty, use the default localhost URI
    if not mlflow_uri:
        mlflow_uri = 'http://0.0.0.0:5001/'

    # Set the tracking URI
    mlflow.set_tracking_uri(mlflow_uri)

    print(f"MLflow Tracking URI set to: {mlflow_uri}")

    return mlflow_uri


tracking_uri = configure_mlflow_tracking()


def to_1d_array(arr):
    return np.atleast_1d(arr)


def to_array(arr):
    return np.array(arr)


def to_list(obj):
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return [obj]
    elif not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)


# some custom metrics on early enrichment
# (from https://chemrxiv.org/engage/chemrxiv/article-details/6585ddc19138d23161476eb1)

def plate_ppv(y, y_pred, top_n: int = 128):
    y_pred = np.atleast_1d(y_pred)
    y = np.atleast_1d(y)
    _tmp = np.vstack((y, y_pred)).T[y_pred.argsort()[::-1]][:top_n, :]
    _tmp = _tmp[np.where(_tmp[:, 1] > 0.5)[0]].copy()
    return np.sum(_tmp[:, 0]) / len(_tmp)


def diverse_plate_ppv(y, y_pred, clusters: list, top_n_per_group: int = 15):
    df = pd.DataFrame({"pred": y_pred, "real": y, "CLUSTER_ID": clusters})
    df_groups = df.groupby("CLUSTER_ID")

    _vals = []
    for group, idx in df_groups.groups.items():
        _tmp = df.iloc[idx].copy()
        if sum(df.iloc[idx]["pred"] > 0.5) == 0:
            continue
        _tmp = _tmp[_tmp["pred"] > 0.5].copy()
        _tmp = np.vstack((_tmp["real"].to_numpy(), _tmp["pred"].to_numpy(
        ))).T[_tmp["pred"].to_numpy().argsort()[::-1]][:top_n_per_group, :]
        _val = np.sum(_tmp[:, 0]) / len(_tmp)
        _vals.append(_val)

    return np.mean(_vals)


# some functions for generating the chemical fingerprints
class Basefpfunc(abc.ABC):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._func = None

    def __call__(self, chemicals, *args, **kwargs):
        return to_array([list(self._func(MolFromSmiles(c))) for c in to_1d_array(chemicals)])

    def __eq__(self, other):
        if isinstance(other, Basefpfunc):
            if inspect.signature(self._func).parameters == inspect.signature(other).parameters:
                return True
        return False

    def to_dict(self):
        _signature = inspect.signature(self._func)
        args = {
            k: v.default
            for k, v in _signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        args['name'] = self._func.func.__name__
        return args

    @property
    def __name__(self):
        return self._func.func.__name__


class ECFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False})
        self._func = partial(
            AllChem.GetHashedMorganFingerprint, **self._kwargs)


class ECFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
        self._func = partial(
            AllChem.GetHashedMorganFingerprint, **self._kwargs)

    # def __init__(self):
    #     super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
    #     self._func = partial(
    #         rdMolDescriptors.GetMorganFingerprintAsBitVect,
    #         radius=self._kwargs['radius'],
    #         nBits=self._kwargs['nBits']
    #     )


class FCFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})
        self._func = partial(
            AllChem.GetHashedMorganFingerprint, **self._kwargs)


class FCFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})
        self._func = partial(
            AllChem.GetHashedMorganFingerprint, **self._kwargs)


class BinaryECFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False})
        self._func = partial(
            AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class BinaryECFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
        self._func = partial(
            AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class BinaryFCFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})
        self._func = partial(
            AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class BinaryFCFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})
        self._func = partial(
            AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class MACCS(Basefpfunc):
    def __init__(self):
        super().__init__()
        self._func = partial(
            rdMolDescriptors.GetMACCSKeysFingerprint, **self._kwargs)


class RDK(Basefpfunc):
    def __init__(self):
        super().__init__(**{"fpSize": 2048})
        self._func = partial(RDKFingerprint, **self._kwargs)


class Avalon(Basefpfunc):
    def __init__(self):
        super().__init__(**{"nBits": 2048})
        self._func = partial(pyAvalonTools.GetAvalonCountFP, **self._kwargs)


class AtomPair(Basefpfunc):
    def __init__(self):
        super().__init__(**{"nBits": 2048})
        self._func = partial(
            rdMolDescriptors.GetHashedAtomPairFingerprint, **self._kwargs)


class TopTor(Basefpfunc):
    def __init__(self):
        super().__init__(**{"nBits": 2048})
        self._func = partial(
            AllChem.GetHashedTopologicalTorsionFingerprint, **self._kwargs)


FPS_FUNCS = {'HitGenBinaryECFP4': ECFP4(),
             'HitGenBinaryECFP6': ECFP6(),
             'HitGenBinaryFCFP4': FCFP4(),
             'HitGenBinaryFCFP6': FCFP6(),
             '2048-bECFP4': BinaryECFP4(),
             '2048-bECFP6': BinaryECFP6(),
             '2048-bFCFP4': BinaryFCFP4(),
             '2048-bFCFP6': BinaryFCFP6(),
             'HitGenBinaryMACCS': MACCS(),
             'HitGenBinaryRDK': RDK(),
             'HitGenBinaryAvalon': Avalon(),
             'HitGenBinaryAtomPair': AtomPair(),
             'HitGenBinaryTopTor': TopTor()}


def cluster_leader_from_array(X, thresh: float = 0.65, use_tqdm: bool = False):
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


class Screen:
    def __init__(self, run_id):
        self._models = [[]]
        self._train_preds = []
        self._bayes = None
        self._fit = False
        self._ensemble = 0
        self._fp_func = []
        self.output_path = ""
        self.overall_metrics = {}
        self.model_name = ""
        # self.run_id = run_id
        # print("RUN ID", self.run_id)

    def screen(self, file_path, output_path):
        if file_path.startswith("gs://"):
            self.output_path = output_path
            bucket_name, file_path = self._parse_gcp_path(file_path)
            files = ManageModelDataset.list_files_in_gcs_folder(
                bucket_name, file_path)
            files = files[:6]
            print("All files---", files)

            # Determine optimal number of cores
            # Cap at 32 or available cores
            num_cores = min(multiprocessing.cpu_count()-1, 32)
            logging.info(f"Using {num_cores} cores for processing")
            for file in files:
                self.read_enamine_from_gcs(bucket_name, file)

            # Prepare file arguments
            # file_args = [(bucket_name, file) for file in files]
            # with Pool(processes=num_cores) as pool:
            #     results = pool.map(self.read_enamine_from_gcs, file_args)
            # print(results)

            # Use context manager for more robust multiprocessing
            try:
                with Pool(processes=num_cores) as pool:
                    # Use starmap for multiple arguments
                    file_args = [(bucket_name, file, self.run_id,
                                  self.output_path) for file in files]
                    results = pool.starmap(self.process_file_safely, file_args)

                print(results)
            except Exception as e:
                logging.error(f"Multiprocessing error: {e}")

        else:
            # self.screen_from_local_dir(file_path, output_path)
            print(f"reading from {file_path}")
            print(f"reading from {file_path}")
            print(f"reading from {file_path}")
            print("reading from local file path")
            self.read_enamine_frm_local_dir(
                file_path=file_path, output_path=output_path)

    def read_enamine_from_gcs(self, bucket_name, file_path):
        """
        Reads a CSV file from Google Cloud Storage and converts a specific column into a list.

        :param bucket_name: Name of the GCS bucket.
        :param file_path: Path to the CSV file within the bucket.
        :param column_index: Index of the column to convert to a list (0-based).
        :return: A list of values from the specified column.
        """
        # bucket_name, file_path = args
        print(f"bucket name is \
                   {bucket_name} and file path is {file_path} and run id is")

        smiles = []
        names = []
        smiles_idx = 1  # second column is smiles
        names_idx = 2  # third column is smiles
        try:
            # Initialize GCS client
            client = storage.Client()

            # Get the bucket and the blob (file)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            print(f"bucket is {bucket} and blob is {blob}")

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
                if (count % 50000) == 0:
                    print(f"Processing file {file_path}: \
                                 {count} compounds processed")

                    preds, confs = self.screen_smiles(smiles)
                    print(
                        f"total number of compound processed====={count}")
                    # preds, confs = self.screen_smiles_parallel(smiles)

                    if self.output_path.startswith("gs://"):
                        try:
                            self.write_tsv_to_gcs(
                                self.output_path, names, smiles, preds, confs)
                        except Exception as write_error:
                            print(f"Error writing to GCS for \
                                         {file_path}: {write_error}")
                    else:
                        print("Output path is not a GCS path")

                    # Reset lists for the next batch
                    names = []
                    smiles = []

                    # result_list.append(row[column_index])
            if len(smiles) != 0:
                preds, confs = self.screen_smiles(smiles)

                self.write_tsv_to_gcs(
                    self.output_path, names, smiles, preds, confs)
            return count
        except Exception as e:
            print(f"An error occurred: {e}")

    def read_enamine_frm_local_dir(self, file_path, output_path):
        """
        Reads a CSV file and converts a specific column into a list.

        :param file_path: Path to the CSV file.
        :param column_index: Index of the column to convert to a list (0-based).
        :return: A list of values from the specified column.
        """
        smiles = []
        names = []
        smiles_idx = 1  # second column is smiles
        names_idx = 2  # third column is smiles

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path):
            with open(output_path, 'w') as outfile:
                outfile.write("ID\tSMILES\tPRED\tCONF\n")
            print(f"Created file: {output_path}")
            print(f"Created file: {output_path}")
        else:
            print(f"File exists: {output_path}")
            print(f"File exists: {output_path}")
        print("file path----", file_path)

        if os.path.exists(file_path):
            with open(file_path, mode='r',  newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                print(reader)
                # Process the CSV file
        else:
            print(f"Error: File '{file_path}' not found.")

        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                count = 0

                for row in reader:
                    # Ensure the column index exists in the row
                    count += 1
                    if len(row) > 2:
                        smiles.append(row[smiles_idx])
                        if names_idx < len(row):  # Avoid IndexError for missing columns
                            names.append(row[names_idx].strip())
                        else:
                            # Append None if name column is missing
                            names.append(None)

                    if count % 50000 == 0:

                        preds, confs = self.screen_smiles(smiles)
                        print(f"Total compound processed----{count}", )
                        print(f"Total compound processed----{count}", )

                        with open(output_path, "a") as f2:
                            for n, s, p, c in zip(names, smiles, preds, confs):
                                f2.write(f"{n}\t{s}\t{round(float(p), 4)}\t \
                                        {round(float(c), 4)}\n")
                            names = []
                            smiles = []

                # catch the last batch
                if len(smiles) != 0:
                    preds, confs = self.screen_smiles(smiles)
                    with open(output_path, "a") as f2:
                        for n, s, p, c in zip(names, smiles, preds, confs):
                            f2.write(f"{n}\t{s}\t{round(float(p), 4)}\t \
                                    {round(float(c), 4)}\n")

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _parse_gcp_path(self, path: str):
        """Parses a GCS path into bucket name and file name."""
        if not path.startswith("gs://"):
            raise ValueError("Invalid GCS path. Must start with 'gs://'")

        path_parts = path[5:].split("/", 1)
        bucket_name = path_parts[0]
        file_name = path_parts[1] if len(path_parts) > 1 else ''
        return bucket_name, file_name

    def write_tsv_to_gcs(self, file_name, names, smiles, preds, confs):
        """Append TSV content to a GCS bucket file, creating headers if file doesn't exist."""
        bucket_name, gcs_file_name = self._parse_gcp_path(file_name)

        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_file_name)

        # Check if file exists and download existing content
        existing_content = ""
        try:
            existing_content = blob.download_as_text()
        except Exception as e:
            # If file doesn't exist, create with headers
            header = "Name\tSMILES\tPrediction\tConfidence\n"
            existing_content = header

        # Prepare new content to append
        new_content = existing_content
        for n, s, p, c in zip(names, smiles, preds, confs):
            new_content += f"{n}\t{s}\t{round(float(p), 4)}\t \
                                        {round(float(c), 4)}\n"

        # Upload the entire updated content
        blob.upload_from_string(
            new_content, content_type="text/tab-separated-values")

        print(f"File successfully {'created' if not existing_content else 'appended to'} GCS bucket \
             {bucket_name} as {gcs_file_name}")

    def screen_smiles(self, smis: list[str]):
        """
        Screens a list of smiles and returns predictions and confidences
        :param smis:
        :return:
        """
        self._fp_func = ["HitGenBinaryECFP4"]
        print("hererererer")
        invalid_smiles = [smi for smi in smis if MolFromSmiles(smi) is None]
        if invalid_smiles:
            print(f"Invalid SMILES strings:{invalid_smiles}")
            smis = [smi for smi in smis if MolFromSmiles(smi) is not None]
        fps = []

        model = "/app/models/ensemble_models.pkl"
        with open(model, 'rb') as file:
            clf = pickle.load(file)
        # mlflow.set_tracking_uri("http://34.130.56.87:5000/")
        # run_id = "027e487a88284e119183729510cd1ce0"
        run_id = self.run_id
        # run_id = "e9873abc52d84941ae9c0a55a85930e1"
        # print("RUN_id is----", run_id)
        # logged_model = f"runs:/{run_id}/model"
        # loaded_model = mlflow.lightgbm.load_model(logged_model)
        # clf = loaded_model
        # print("Model----", clf, self._fp_func)
        for _fp in self._fp_func:
            fps.append(list(FPS_FUNCS[_fp](smis)))

        test_preds = []

        for fp in fps:
            test_preds.append(clf.predict_proba(fp)[:, 1])
        # for i_model in range(self._ensemble):
        #     for clf, fp in zip(models[i_model], fps):
        #         test_preds.append(clf.predict_proba(fp)[:, 1])

        # for i_model in range(self._ensemble):
        #     for clf, fp in zip(self._models[i_model], fps):
        #         test_preds.append(clf.predict_proba(fp)[:, 1])
        test_preds = np.array(test_preds).T
        preds = test_preds.mean(axis=1)
        confs = test_preds.std(axis=1)

        # Filter predictions and confidences
        valid_indices = preds > 0.3
        filtered_preds = preds[valid_indices]
        filtered_confs = confs[valid_indices]

        return filtered_preds, filtered_confs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    print("arrrrrr", args.config)
    t1 = time.time()
    config = MLConfigParser(config_file, "ml_config")
    config_dict = config.get_config()
    training_cols = config_dict.get("columns_of_interest")
    label_col = config_dict.get("target_col")
    is_binary = config_dict.get("is_binarized_data")
    is_groupped_cluster = config_dict.get("cluster_generation")
    model_name = config_dict.get("model_name")
    dataset_location = config_dict.get("input_data_path")
    config_location = config_dict.get("processed_file_location")
    result_output = config_dict.get("result_output")
    smile_location = config_dict.get("smile_location")
    isdry_run = config_dict.get("isdry_run", True)
    screen = Screen()
    screen.screen(smile_location, result_output)
    t2 = time.time()
    print("total processing time---", t2-t1)
