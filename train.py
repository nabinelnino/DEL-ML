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
from rdkit.Chem import AllChem, rdMolDescriptors, MolFromSmiles
from rdkit.Chem import RDKFingerprint
from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score, \
    average_precision_score, RocCurveDisplay
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from tqdm import tqdm
import mlflow


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
from utils.config_parser import ManageModelDataset


from datetime import date

today = date.today()

# service_account_path = '../service_account.json'
service_account_path = '../app/service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    print("Service account credentials set.")
else:
    print("Service account file not found. Skipping credential setup.")

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../service_account.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './app/service_account.json'


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


class Model:
    def __init__(self):
        self._models = [[]]
        self._train_preds = []
        self._bayes = None
        self._fit = False
        self._ensemble = 0
        self._fp_func = []

    def fit(
            self,
            train_data: Dict[str, Union[npt.NDArray, str]],
            binary_labels: Union[npt.NDArray, str],
            source: str,
            model_name: str,
            config_path=str,
            clusters: Optional[Union[npt.NDArray, str]] = None,
            ensemble: int = 1
    ):
        """
        Fit the model
        :param train_data:
            should be a a dictionary where the key is the fingerprint type
            (see the `FPS_FUNCS` dict for names) and the val the path to a pickle
            or the loaded numpy array of the fingerprints

            Will make a separate model for each fingerprint type and mate. So if you set ensemble to 5
            use 4 different FPs, you will have 5 x 4 = 20 models
        :param binary_labels:
            the path to a pickle or the loaded numpy array of the binary labels
        :param clusters:
            the path to a pickle or the loaded numpy array of the cluster IDs
            not used if ensemble is <= 1
        :param ensemble:
            number of ensembles mates to use. Default is 1 (no ensemble)
        :return:
        """
        try:
            mlflow.set_tracking_uri("http://34.130.56.87:5000/")

            mlflow.set_experiment(f"{model_name}_{today}_full")

            for key, val in train_data.items():
                if isinstance(val, str):
                    train_data[key] = pickle.load(open(val, "rb"))

            if isinstance(binary_labels, str):
                try:
                    y = pickle.load(open(binary_labels, "rb"))
                except (FileNotFoundError, pickle.PickleError) as e:
                    raise ValueError(f"Error loading binary labels: {e}")
            else:
                y = binary_labels

            self._fp_func = list(train_data.keys())

            with mlflow.start_run(run_name="lgbm_"+str("test_run")) as run:
                try:
                    mlflow.log_param("featurizer", self._fp_func)
                    mlflow.set_tag("model_type", "lgbm")
                    mlflow.log_param("ensemble", ensemble)
                    mlflow.lightgbm.autolog(log_input_examples=False, log_model_signatures=True, log_models=True, log_datasets=True, disable=False,
                                            exclusive=False, disable_for_unsupported_versions=False, silent=False, registered_model_name=None, extra_tags=None)

                    if ensemble > 1:
                        mates = []
                        print(f"model for ensamble{ensemble_count}")
                        ensemble_count += 1

                        # load in cluster data
                        if isinstance(clusters, str):
                            clusters = pickle.load(open(clusters, "rb"))

                        print(f"Shape of train_data[{self._fp_func[0]}]: \
                            {train_data[self._fp_func[0]].shape}")
                        if not isinstance(train_data[self._fp_func[0]], np.ndarray):
                            raise TypeError(f"Expected numpy array for train_data[ \
                                            {self._fp_func[0]}], but got {type(train_data[self._fp_func[0]])}")

                        s = StratifiedGroupKFold(
                            n_splits=ensemble, shuffle=True)
                        for i, (train_idx, test_idx) in tqdm(enumerate(s.split(train_data[self._fp_func[0]], y, clusters)), desc="Doing Folds"):
                            with mlflow.start_run(run_name=f"fold_{i}", nested=True) as run:
                                print(f"(\
                                    Fold {i} - Train indices shape: {train_idx.shape}, Test indices shape: {test_idx.shape}")
                                y_train = y[train_idx]
                                models = []
                                train_preds = []
                                for _, x_train in train_data.items():
                                    x_train_fold = x_train[train_idx]
                                    print(f"Shape of x_train after indexing: \
                                        {x_train_fold.shape}")
                                    clf = LGBMClassifier(
                                        n_estimators=150, n_jobs=-1)
                                    clf.fit(x_train_fold, y_train)
                                    # clf.fit(x_train, y_train)
                                    models.append(deepcopy(clf))
                                mates.append(models)

                        self._models = deepcopy(mates)
                    else:
                        for _, x_train in train_data.items():

                            clf = LGBMClassifier(n_estimators=150, n_jobs=-1)
                            clf.fit(x_train, y)
                            with open('model_pkl', 'wb') as files:
                                pickle.dump(clf, files)
                            self._models[0].append(deepcopy(clf))

                            # mates.append(deepcopy(clf))
                except Exception as e:
                    data = ManageModelDataset()
                    data.ramove_dataseta_and_model(
                        config_path, model_name, source)
                    raise RuntimeError(f"Eror during MLflow run : {e}")
            self._fit = True
            self._ensemble = ensemble

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def screen(self, file_path, output_path):
        if file_path.startswith("gs://"):
            self.screen_from_bucket(
                file_path=file_path, output_file=output_path)
        else:
            self.screen_from_local_dir(file_path, output_path)

    def screen_from_local_dir(self, filepath, outpath: Optional[str] = None):
        """
        Screening a file of SMILES and returns predictions
        pred value will the be probability the model thinks something is active
        conf value is the confidence the model has in its predicted probability

        assumes you are using a smi file, so its tab delimited and first column is SMILES second is Name/ID

        :param filepath: path the .smi file to screen
        :param outpath: name of output file
        """

        if outpath is None:
            outpath = os.path.abspath(filepath).split('.')[0] + ".PREDS"

        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if not os.path.exists(outpath):
            with open(outpath, 'w') as outfile:
                outfile.write("ID\tSMILES\tPRED\tCONF\n")
            print(f"Created file: {outpath}")
        else:
            print(f"File exists: {outpath}")

        with open(filepath, "r") as f:
            header = f.readline().strip().split("\t")
            # Replace "SMILES" with the actual column name for smiles
            try:
                smiles_idx = header.index("smiles")
                names_idx = header.index("names")
            except ValueError:
                # If column names not found, treat first line as data
                smiles_idx = 0  # assume first column is smiles
                names_idx = 1
                header = None
                # Reset file pointer and use first line as first data row
                f.seek(0)
            names = []
            smiles = []
            next(f)

            for i, line in tqdm(enumerate(f)):
                splits = line.split("\t")

                # smiles.append(splits[0].strip())
                smiles.append(splits[smiles_idx].strip())

                # Check to avoid IndexError for missing columns
                if names_idx < len(splits):
                    names.append(splits[names_idx].strip())
                else:
                    names.append(None)
                # if len(splits) > 1:
                #     names.append(splits[1].strip())
                # else:
                #     names.append(i)

                if ((i+1) % 10000) == 0:

                    preds, confs = self.screen_smiles(smiles)
                    print("preds---", preds, type(preds))

                    with open(outpath, "a") as f2:
                        for n, s, p, c in zip(names, smiles, preds, confs):
                            f2.write(f"{n}\t{s}\t{round(float(p), 4)}\t \
                                     {round(float(c), 4)}\n")
                        names = []
                        smiles = []

            # catch the last batch
            if len(smiles) != 0:
                preds, confs = self.screen_smiles(smiles)
                with open(outpath, "a") as f2:
                    for n, s, p, c in zip(names, smiles, preds, confs):
                        f2.write(f"{n}\t{s}\t{round(float(p), 4)}\t \
                                {round(float(c), 4)}\n")

    def screen_from_bucket(self, file_path, output_file):
        # file_path = "gs://test-aircheck/mlops-test/sample_smiles.smi.gz"
        # output_file = "gs://test-aircheck/mlops-test/result.smi"

        bucket_name, file_name = self._parse_gcp_path(file_path)
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

                # Append values based on column indices
                smiles.append(splits[smiles_idx].strip())
                if names_idx < len(splits):  # Avoid IndexError for missing columns
                    names.append(splits[names_idx].strip())
                else:
                    names.append(None)  # Append None if name column is missing

                count += 1
                if (count % 10000) == 0:
                    preds, confs = self.screen_smiles(smiles)

                    if output_file.startswith("gs://"):
                        self.write_tsv_to_gcs(
                            output_file, names, smiles, preds, confs)

                    # Reset lists for the next batch
                    names = []
                    smiles = []

            # Process the remaining batch
        if len(smiles) != 0:
            preds, confs = self.screen_smiles(smiles)
            self.write_tsv_to_gcs(output_file, names, smiles, preds, confs)

    def _parse_gcp_path(self, path: str):
        """Parses a GCS path into bucket name and file name."""
        if not path.startswith("gs://"):
            raise ValueError("Invalid GCS path. Must start with 'gs://'")

        path_parts = path[5:].split("/", 1)
        bucket_name = path_parts[0]
        file_name = path_parts[1] if len(path_parts) > 1 else ''
        return bucket_name, file_name

    def write_tsv_to_gcs(self, file_name, names, smiles, preds, confs):
        """Write TSV content to a GCS bucket."""
        bucket_name, gcs_file_name = self._parse_gcp_path(file_name)
        print("hereree", bucket_name, gcs_file_name)
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
        blob = bucket.blob(gcs_file_name)

        # Upload the TSV data from the buffer
        blob.upload_from_string(tsv_buffer.getvalue(),
                                content_type="text/tab-separated-values")

        print(f"File successfully written to GCS bucket \
              {bucket_name} as {gcs_file_name}")

    def screen_smiles(self, smis: list[str]):
        """
        Screens a list of smiles and returns predictions and confidences
        :param smis:
        :return:
        """
        # self._fp_func = ["HitGenBinaryECFP6"]
        invalid_smiles = [smi for smi in smis if MolFromSmiles(smi) is None]
        if invalid_smiles:
            print("Invalid SMILES strings:", invalid_smiles)

            # Optionally, remove invalid SMILES
            smis = [smi for smi in smis if MolFromSmiles(smi) is not None]
        fps = []
        # model = "./model_pkl"
        # with open(model, 'rb') as file:
        #     clf = pickle.load(file)
        # print("Model----", clf, self._fp_func)
        for _fp in self._fp_func:
            fps.append(list(FPS_FUNCS[_fp](smis)))
        test_preds = []
        # for fp in fps:
        #     test_preds.append(clf.predict_proba(fp)[:, 1])

        for i_model in range(self._ensemble):
            for clf, fp in zip(self._models[i_model], fps):
                test_preds.append(clf.predict_proba(fp)[:, 1])
        test_preds = np.array(test_preds).T
        preds = test_preds.mean(axis=1)
        confs = test_preds.std(axis=1)

        # Filter predictions and confidences
        valid_indices = preds > 0.3
        filtered_preds = preds[valid_indices]
        filtered_confs = confs[valid_indices]

        return filtered_preds, filtered_confs

    def cv(
        self,
        train_data: Dict[str, Union[npt.NDArray, str]],
        binary_labels: Union[npt.NDArray, str],
        source: str,
        model_name: str,
        config_path: str,
        clusters: Union[npt.NDArray, str],
        # clusters: Optional[Union[npt.NDArray, str]] = None,
        ensemble: int = 1,
    ):
        """
        Fit the model
        :param train_data:
            should be a a dictionary where the key is the fingerprint type
            (see the `FPS_FUNCS` dict for names) and the val the path to a pickle
            or the loaded numpy array of the fingerprints

            Will make a separate model for each fingerprint type and mate. So if you set ensemble to 5
            use 4 different FPs, you will have 5 x 4 = 20 models
        :param binary_labels:
            the path to a pickle or the loaded numpy array of the binary labels
        :param clusters:
            the path to a pickle or the loaded numpy array of the cluster IDs
            not used if ensemble is <= 1
        :param ensemble:
            number of ensembles mates to use. Default is 1 (no ensemble)
        :return:
        """
        # load in pickles in needed

        try:
            mlflow.set_tracking_uri("http://34.130.56.87:5000/")
            mlflow.set_experiment(f"{model_name}_cross_val_{today}_full")
            for key, val in train_data.items():

                if isinstance(val, str):
                    train_data[key] = pickle.load(open(val, "rb"))

            if isinstance(binary_labels, str):
                y = np.array(pickle.load(open(binary_labels, "rb")))
            else:
                y = np.array(binary_labels)

            # load in cluster data
            if isinstance(clusters, str):
                clusters = pickle.load(open(clusters, "rb"))

            overall_res_ensemble = {
                "fit_time": [],
                "pred_time": [],
                "precision": [],
                "recall": [],
                "balanced_accuracy": [],
                "AUC_PR": [],
                "AUC_ROC": [],
                "PlatePPV": [],
                "DivPlatePPV": []
            }
            with mlflow.start_run(run_name="lgbm_"+str("test_run")) as run:
                try:
                    # mlflow_dataset = mlflow.data.from_numpy(
                    #     train_data, y, source=source)
                    # mlflow.log_input(mlflow_dataset, context="training")
                    mlflow.log_param("featurizer", self._fp_func)
                    mlflow.set_tag("model_type", "lgbm")
                    mlflow.log_param("ensemble", ensemble)
                    # mlflow.lightgbm.autolog(log_input_examples=False, log_model_signatures=True, log_models=True, log_datasets=True, disable=False,
                    #                         exclusive=False, disable_for_unsupported_versions=False, silent=False, registered_model_name=None, extra_tags=None)

                    mlflow.lightgbm.autolog(
                        log_input_examples=False,
                        log_model_signatures=True,
                        log_models=True,
                        log_datasets=True
                    )

                    s = StratifiedShuffleSplit(test_size=0.2)

                    # for i, (train_idx, test_idx) in tqdm(enumerate(s.split(list(train_data.values())[0], y, clusters)), desc="Doing Folds"):
                    for i, (train_idx, test_idx) in tqdm(enumerate(s.split(list(train_data.values())[0], y, clusters)), desc="Doing Folds"):
                        y_train = y[train_idx]
                        y_test = y[test_idx]

                        train_clusters = clusters[train_idx]

                        mates = []
                        all_train_preds = []

                        t0 = time()
                        for _, x_train_ in train_data.items():
                            x_train = x_train_[train_idx]
                            if ensemble > 1:
                                # this is the ensemble builder
                                # should have done this so I could have reused the fit func but too late lol
                                s2 = StratifiedGroupKFold(
                                    n_splits=ensemble, shuffle=True)
                                models = []
                                train_preds = []

                                for ii, (train_idx2, test_idx2) in tqdm(enumerate(s2.split(x_train, y_train, train_clusters)), desc="Doing ensemble"):
                                    clf = LGBMClassifier(
                                        n_estimators=150, n_jobs=-1)
                                    x_train2 = x_train[train_idx2]
                                    y_train2 = y_train[train_idx2]
                                    clf.fit(x_train2, y_train2)
                                    models.append(deepcopy(clf))
                                    train_preds.append(
                                        clf.predict_proba(x_train)[:, 1])
                                mates.append(models)
                                all_train_preds.append(train_preds)

                            else:
                                clf = LGBMClassifier(
                                    n_estimators=150, n_jobs=-1)
                                clf.fit(x_train, y_train)
                                mates.append([deepcopy(clf)])
                                all_train_preds.append(
                                    [clf.predict_proba(x_train)[:, 1]])
                        fit_time = time() - t0

                        t0 = time()
                        test_preds = []
                        for clf_group, (_, x_test) in zip(mates, train_data.items()):
                            x_test = x_test[test_idx]
                            for clf in clf_group:
                                clf.predict_proba(x_test)
                                test_preds.append(
                                    clf.predict_proba(x_test)[:, 1])
                        test_preds = np.array(test_preds).T
                        pred_time = time() - t0

                        preds = test_preds.mean(axis=1)
                        discrete_preds = (preds > 0.3).astype(int)

                        ppv = precision_score(y_test, discrete_preds)
                        recall = recall_score(y_test, discrete_preds)
                        auc_roc = roc_auc_score(y_test, preds)
                        ba = balanced_accuracy_score(y_test, discrete_preds)
                        auc_pr = average_precision_score(y_test, preds)
                        p_ppv = plate_ppv(y_test, preds, top_n=128)
                        dp_ppv = diverse_plate_ppv(
                            y_test, preds, clusters=clusters[test_idx].tolist())

                        overall_res_ensemble["fit_time"].append(fit_time)
                        overall_res_ensemble["pred_time"].append(pred_time)
                        overall_res_ensemble["precision"].append(ppv)
                        overall_res_ensemble["recall"].append(recall)
                        overall_res_ensemble["balanced_accuracy"].append(ba)
                        overall_res_ensemble["AUC_ROC"].append(auc_roc)
                        overall_res_ensemble["AUC_PR"].append(auc_pr)
                        overall_res_ensemble["PlatePPV"].append(p_ppv)
                        overall_res_ensemble["DivPlatePPV"].append(dp_ppv)

                    mlflow.log_metric("mean_precision", np.mean(
                        overall_res_ensemble["precision"]))
                    mlflow.log_metric("mean_recall", np.mean(
                        overall_res_ensemble["recall"]))
                    mlflow.log_metric("mean_balanced_accuracy", np.mean(
                        overall_res_ensemble["balanced_accuracy"]))
                    mlflow.log_metric("mean_AUC_ROC", np.mean(
                        overall_res_ensemble["AUC_ROC"]))
                    mlflow.log_metric("mean_AUC_PR", np.mean(
                        overall_res_ensemble["AUC_PR"]))
                    mlflow.log_metric("mean_PlatePPV", np.mean(
                        overall_res_ensemble["PlatePPV"]))
                    mlflow.log_metric("mean_DivPlatePPV", np.mean(
                        overall_res_ensemble["DivPlatePPV"]))

                    # Standard deviation metrics can also be logged
                    mlflow.log_metric("std_precision", np.std(
                        overall_res_ensemble["precision"]))
                    mlflow.log_metric("std_recall", np.std(
                        overall_res_ensemble["recall"]))

                except Exception as e:
                    data = ManageModelDataset()
                    data.ramove_dataseta_and_model(
                        config_path, model_name, source)
                    raise RuntimeError(f"Error during MLflow run: {e}")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise
