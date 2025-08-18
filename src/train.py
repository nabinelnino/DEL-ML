
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

from typing import Optional
from tqdm import tqdm


import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier

# rdkit imports
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, MolFromSmiles
from rdkit.Chem import RDKFingerprint

from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score, \
    average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from tqdm import tqdm
import mlflow

from rdkit import DataStructs
from tqdm import tqdm

from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers


from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (precision_score, recall_score, roc_auc_score,
                             balanced_accuracy_score, average_precision_score)
from typing import Dict, Union, Optional
import numpy.typing as npt
from lightgbm import LGBMClassifier
from lightgbm import LGBMClassifier


from dotenv import load_dotenv
import logging
from datetime import date
RDLogger.DisableLog('rdApp.*')
today = date.today()
load_dotenv()

warnings.filterwarnings(
    "ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# service_account_path = '../service_account.json'
service_account_path = '../app/service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    logging.info("Service account credentials set.")
else:
    logging.info("Service account file not found. Skipping credential setup.")


def configure_mlflow_tracking():
    # Read MLFLOW_TRACKING_URI from environment variables
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()
    logging.info(f"mlflow uri is----{mlflow_uri}")

    # If MLFLOW_TRACKING_URI is empty, use the default localhost URI
    if not mlflow_uri:
        mlflow_uri = 'http://0.0.0.0:5001/'

    # Set the tracking URI
    mlflow.set_tracking_uri(mlflow_uri)

    logging.info(f"MLflow Tracking URI set to: {mlflow_uri}")

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
    def __init__(self, model_file_path: str):
        self._models = [[]]
        self._train_preds = []
        self._bayes = None
        self._fit = False
        self._ensemble = 0
        self.output_path = ""
        self.overall_metrics = {}
        self.model_name = ""
        self._fp_func = []
        self.model_file_path = model_file_path

    def save_models(self, models, save_path):
        """
        Save models to a single file.

        :param models: List of models to save.
        :param save_path: Path to save the models.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(models, f)
        logging.info(f"Models saved to {save_path}")

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
                    mlflow.lightgbm.autolog(log_input_examples=False, log_model_signatures=True, log_models=False, log_datasets=True, disable=False,

                                            exclusive=False, disable_for_unsupported_versions=False, silent=False, registered_model_name=model_name, extra_tags=None)
                    if ensemble > 1:
                        mates = []

                        # load in cluster data
                        if isinstance(clusters, str):
                            clusters = pickle.load(open(clusters, "rb"))

                        logging.info(f"Shape of train_data[{self._fp_func[0]}]: \
                            {train_data[self._fp_func[0]].shape}")
                        if not isinstance(train_data[self._fp_func[0]], np.ndarray):
                            raise TypeError(f"Expected numpy array for train_data[ \
                                            {self._fp_func[0]}], but got {type(train_data[self._fp_func[0]])}")

                        s = StratifiedGroupKFold(
                            n_splits=ensemble, shuffle=True)
                        for i, (train_idx, test_idx) in tqdm(enumerate(s.split(train_data[self._fp_func[0]], y, clusters)), desc="Doing Folds"):
                            with mlflow.start_run(run_name=f"fold_{i}", nested=True) as run:
                                logging.info(f"(\
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
                        # Combine all feature sets
                        combined_features = np.concatenate(
                            list(train_data.values()), axis=1)

                        clf = LGBMClassifier(n_estimators=150, n_jobs=-1)
                        clf.fit(combined_features, y)

                        with open(self.model_file_path, 'wb') as files:
                            pickle.dump(clf, files)

                        mlflow.lightgbm.log_model(clf, "model")

                    mlflow.log_metric("mean_precision", np.mean(
                        self.overall_metrics["precision"]))
                    mlflow.log_metric("mean_recall", np.mean(
                        self.overall_metrics["recall"]))
                    mlflow.log_metric("mean_balanced_accuracy", np.mean(
                        self.overall_metrics["balanced_accuracy"]))
                    mlflow.log_metric("mean_AUC_ROC", np.mean(
                        self.overall_metrics["AUC_ROC"]))
                    mlflow.log_metric("mean_AUC_PR", np.mean(
                        self.overall_metrics["AUC_PR"]))
                    mlflow.log_metric("mean_PlatePPV", np.mean(
                        self.overall_metrics["PlatePPV"]))
                    mlflow.log_metric("mean_DivPlatePPV", np.mean(
                        self.overall_metrics["DivPlatePPV"]))

                    # Standard deviation metrics can also be logged
                    mlflow.log_metric("std_precision", np.std(
                        self.overall_metrics["precision"]))
                    mlflow.log_metric("std_recall", np.std(
                        self.overall_metrics["recall"]))

                except Exception as e:

                    raise RuntimeError(f"Eror during MLflow run : {e}")

                run = mlflow.active_run()
                logging.info(f"Active run_id: {run.info.run_id}")
                self.run_id = run.info.run_id
            self._fit = True
            self._ensemble = ensemble

        except Exception as e:
            logging.info(f"An error occurred: {e}")
            raise
        return self.run_id

    def cv(
        self,
        train_data: Dict[str, Union[npt.NDArray, str]],
        binary_labels: Union[npt.NDArray, str],
        source: str,
        model_name: str,
        config_path: str,
        clusters: Union[npt.NDArray, str],
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
        try:
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

            try:
                self._fp_func = list(train_data.keys())
                s = StratifiedShuffleSplit(test_size=0.2)

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
                    # or however you track fold numbers
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

                self.overall_metrics = overall_res_ensemble

            except Exception as e:

                raise RuntimeError(f"Error during MLflow run: {e}")

        except Exception as e:

            logging.info(f"An error occurred: {e}")
            raise
