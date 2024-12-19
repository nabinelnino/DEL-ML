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


class Model:
    def __init__(self):
        self._models = [[]]
        self._train_preds = []
        self._bayes = None
        self._fit = False
        self._ensemble = 0
        self._fp_func = []
        self.output_path = ""
        self.overall_metrics = {}
        self.model_name = ""
        self.run_id = ""

    def save_models(self, models, save_path):
        """
        Save models to a single file.

        :param models: List of models to save.
        :param save_path: Path to save the models.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(models, f)
        print(f"Models saved to {save_path}")

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
            # mlflow.set_tracking_uri("http://34.130.56.87:5000/")
            self.model_name = model_name

            mlflow.set_experiment(f"{model_name}_{today}")

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
            model_save_dir = "models"
            os.makedirs(model_save_dir, exist_ok=True)

            with mlflow.start_run(run_name="lgbm_"+str("test_run")) as run:
                try:
                    mlflow.log_param("featurizer", self._fp_func)
                    mlflow.set_tag("model_type", "lgbm")
                    mlflow.log_param("ensemble", ensemble)
                    mlflow.lightgbm.autolog(log_input_examples=False, log_model_signatures=True, log_models=True, log_datasets=True, disable=False,

                                            exclusive=False, disable_for_unsupported_versions=False, silent=False, registered_model_name=model_name, extra_tags=None)

                    if ensemble > 1:
                        mates = []
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

                                    clf = LGBMClassifier(
                                        n_estimators=150, n_jobs=-1)
                                    clf.fit(x_train_fold, y_train)
                                    # clf.fit(x_train, y_train)
                                    models.append(deepcopy(clf))

                                mates.append(models)

                        self._models = deepcopy(mates)
                        # save_path = "./models/ensemble_models.pkl"
                        # self.save_models(mates, save_path)
                    else:
                        for _, x_train in train_data.items():
                            single_models = []

                            clf = LGBMClassifier(n_estimators=150, n_jobs=-1)
                            clf.fit(x_train, y)
                            # with open('model_pkl', 'wb') as files:
                            #     pickle.dump(clf, files)
                            # Save the trained model
                        #     single_models.append(deepcopy(clf))
                        # save_path = "./models/single_model.pkl"
                        # self.save_models(single_models, save_path)

                    # mlflow.log_metric("mean_precision", np.mean(
                    #     self.overall_metrics["precision"]))
                    # mlflow.log_metric("mean_recall", np.mean(
                    #     self.overall_metrics["recall"]))
                    # mlflow.log_metric("mean_balanced_accuracy", np.mean(
                    #     self.overall_metrics["balanced_accuracy"]))
                    # mlflow.log_metric("mean_AUC_ROC", np.mean(
                    #     self.overall_metrics["AUC_ROC"]))
                    # mlflow.log_metric("mean_AUC_PR", np.mean(
                    #     self.overall_metrics["AUC_PR"]))
                    # mlflow.log_metric("mean_PlatePPV", np.mean(
                    #     self.overall_metrics["PlatePPV"]))
                    # mlflow.log_metric("mean_DivPlatePPV", np.mean(
                    #     self.overall_metrics["DivPlatePPV"]))

                    # # Standard deviation metrics can also be logged
                    # mlflow.log_metric("std_precision", np.std(
                    #     self.overall_metrics["precision"]))
                    # mlflow.log_metric("std_recall", np.std(
                    #     self.overall_metrics["recall"]))

                except Exception as e:
                    data = ManageModelDataset()
                    data.ramove_dataseta_and_model(
                        config_path, model_name, source)
                    raise RuntimeError(f"Eror during MLflow run : {e}")

                run = mlflow.active_run()
                print(f"Active run_id: {run.info.run_id}")
                self.run_id = run.info.run_id
            self._fit = True
            self._ensemble = ensemble

        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        return self.run_id

    # def screen(self, file_path, output_path):
    #     self.output_path = output_path
    #     if file_path.startswith("gs://"):
    #         bucket_name, file_path = self._parse_gcp_path(file_path)
    #         print(f"Bucket name is \
    #                      {bucket_name} and file_path is {file_path}")
    #         files = ManageModelDataset.list_files_in_gcs_folder(
    #             bucket_name, file_path)
    #         print(f"Total numnber of files----{len(files)}")
    #         num_cores = cpu_count() - 1  # Use all available cores except one
    #         print(f"counting number of cores---{num_cores}")
    #         print("number of cores----", num_cores)
    #         total_count = 0
    #         files = files[:5]
    #         print("filessssss", files)
    #         file_args = [(bucket_name, file) for file in files]
    #         with Pool(num_cores) as p:
    #             print(f"Processing {len(file_args)} files")
    #             results = p.map(self.read_enamine_from_gcs, file_args)
    #             print(f"Completed processing. Results: {results}")
    #             sum(filter(None, results))
    #         print(f"total processed count is{total_count}")

    #         # self.screen_from_bucket(
    #         #     file_path=file_path, output_file=output_path)
    #     else:
    #         # self.screen_from_local_dir(file_path, output_path)
    #         print(f"reading from {file_path}")
    #         print(f"reading from {file_path}")
    #         print(f"reading from {file_path}")
    #         print("reading from local file path")
    #         self.read_enamine_frm_local_dir(
    #             file_path=file_path, output_path=output_path)

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
            # mlflow.set_tracking_uri("http://34.130.56.87:5000/")
            mlflow.set_experiment(f"{model_name}_cross_val_{today}")
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
                                    n_estimators=150, n_jobs=-1, force_col_wise=True)
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

                    self.overall_metrics = overall_res_ensemble

                except Exception as e:
                    data = ManageModelDataset()
                    data.ramove_dataseta_and_model(
                        config_path, model_name, source)
                    raise RuntimeError(f"Error during MLflow run: {e}")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise


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
        self.run_id = run_id
        print("RUN ID", self.run_id)

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

            # Prepare file arguments
            file_args = [(bucket_name, file) for file in files]
            with Pool(processes=num_cores) as pool:
                results = pool.map(self.read_enamine_from_gcs, file_args)
            print(results)

            # Use ProcessPoolExecutor for better control and error handling
            # with ProcessPoolExecutor(max_workers=num_cores) as executor:
            #     # Submit all tasks
            #     future_to_file = {
            #         executor.submit(self.read_enamine_from_gcs, args): args
            #         for args in file_args
            #     }

            #     total_processed = 0
            #     for future in as_completed(future_to_file):
            #         file_args = future_to_file[future]
            #         try:
            #             result = future.result()
            #             total_processed += result
            #             logging.info(f"Processed {file_args}: \
            #                          {result} compounds")
            #         except Exception as e:
            #             logging.error(f"Error processing {file_args}: {e}")

            #     logging.info(f"Total compounds processed: {total_processed}")
        else:
            # self.screen_from_local_dir(file_path, output_path)
            print(f"reading from {file_path}")
            print(f"reading from {file_path}")
            print(f"reading from {file_path}")
            print("reading from local file path")
            self.read_enamine_frm_local_dir(
                file_path=file_path, output_path=output_path, run_id=self.run_id)

    def read_enamine_from_gcs(self, args):
        """
        Reads a CSV file from Google Cloud Storage and converts a specific column into a list.

        :param bucket_name: Name of the GCS bucket.
        :param file_path: Path to the CSV file within the bucket.
        :param column_index: Index of the column to convert to a list (0-based).
        :return: A list of values from the specified column.
        """
        bucket_name, file_path = args
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
                if (count % 10000) == 0:
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

                    if count % 10000 == 0:

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

        # model = "./model_pkl"
        # with open(model, 'rb') as file:
        #     clf = pickle.load(file)
        # mlflow.set_tracking_uri("http://34.130.56.87:5000/")
        # run_id = "027e487a88284e119183729510cd1ce0"
        run_id = self.run_id
        # run_id = "e9873abc52d84941ae9c0a55a85930e1"
        print("RUN_id is----", run_id)
        logged_model = f"runs:/{run_id}/model"
        loaded_model = mlflow.lightgbm.load_model(logged_model)
        clf = loaded_model
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
