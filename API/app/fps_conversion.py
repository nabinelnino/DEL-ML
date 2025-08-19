import abc
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, MolFromSmiles, rdFingerprintGenerator
from rdkit.Chem import RDKFingerprint

from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score, \
    average_precision_score, RocCurveDisplay
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from tqdm import tqdm
import inspect
import mlflow.sklearn

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from functools import partial
from rdkit import RDLogger
import logging
RDLogger.DisableLog('rdApp.*')


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


def screen_smiles(model, smis: list[str], fingerprint_col: list[str]):
    """
    Screens a list of smiles and returns predictions and confidences
    :param smis:
    :return:
    """
    _fp_func = fingerprint_col
    invalid_smiles = [smi for smi in smis if MolFromSmiles(smi) is None]
    if invalid_smiles:
        logging.warning("Invalid SMILES strings:", invalid_smiles)
        smis = [smi for smi in smis if MolFromSmiles(smi) is not None]
    fps = []
    if not smis:
        return {
            "message": "Not a valid compound",
            "data": smis
        }

    for _fp in _fp_func:
        fps.append(list(FPS_FUNCS[_fp](smis)))

    test_preds = []

    for fp in fps:
        test_preds.append(model.predict_proba(fp)[:, 1])

    # for clf, fp in zip(model, fps):
    #     test_preds.append(clf.predict_proba(fp)[:, 1])
    test_preds = np.array(test_preds).T

    preds = test_preds.mean(axis=1).tolist()  # Convert NumPy array to list
    confs = test_preds.std(axis=1).tolist()
    return {
        "message": "Processed list input",
        "data": {
            "predictions": preds,
            "confidences": confs
        }
    }
