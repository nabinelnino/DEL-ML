import sys

import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.Descriptors as Descriptors


michael_acceptor_smarts = [
    'C=!@CC=[O,S]',
    '[$([CH]),$(CC)]#CC(=O)[C,c]',
    '[$([CH]),$(CC)]#CS(=O)(=O)[C,c]',
    'C=C(C=O)C=O',
    '[$([CH]),$(CC)]#CC(=O)O[C,c]'
]
michael_acceptor_smarts = [Chem.MolFromSmarts(
    s) for s in michael_acceptor_smarts]


def assignPointsToClusters(picks, fps):
    clusters = defaultdict(list)
    for i, idx in enumerate(picks):
        clusters[i].append(idx)
    sims = np.zeros((len(picks), len(fps)))
    for i in tqdm(range(len(picks))):
        pick = picks[i]
        sims[i, :] = DataStructs.BulkTanimotoSimilarity(fps[pick], fps)
        sims[i, i] = 0
    best = np.argmax(sims, axis=0)
    for i, idx in enumerate(best):
        if i not in picks:
            clusters[idx].append(i)
    return clusters


class DrugDesignFilters(object):
    def __init__(self):
        pass

    @staticmethod
    def fetch_attributes(molecule):
        molecular_weight = Descriptors.ExactMolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        h_bond_donor = Descriptors.NumHDonors(molecule)
        h_bond_acceptors = Descriptors.NumHAcceptors(molecule)
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(molecule)
        molar_refractivity = Chem.Crippen.MolMR(molecule)
        topological_surface_area_mapping = Chem.QED.properties(molecule).PSA
        formal_charge = Chem.rdmolops.GetFormalCharge(molecule)
        heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(molecule)
        num_of_rings = Chem.rdMolDescriptors.CalcNumRings(molecule)

        return (
            molecular_weight,
            logp,
            h_bond_donor,
            h_bond_acceptors,
            rotatable_bonds,
            number_of_atoms,
            molar_refractivity,
            topological_surface_area_mapping,
            formal_charge,
            heavy_atoms,
            num_of_rings
        )

    def filter(self, smiles):
        results = {
            'michael_accept': [],
            'lipinski': [],
            'ghose': [],
            'veber': [],
            'rule_of_3': [],
            'reos': [],
            'drug_like': [],
            'pass_all_filters': [],
            "medchem_score": [],

        }
        molecules = [Chem.MolFromSmiles(i) for i in smiles]

        for i, mol in tqdm(enumerate(molecules), total=len(smiles), desc="doing med chem filters"):
            (
                molecular_weight,
                logp,
                h_bond_donor,
                h_bond_acceptors,
                rotatable_bonds,
                number_of_atoms,
                molar_refractivity,
                topological_surface_area_mapping,
                formal_charge,
                heavy_atoms,
                num_of_rings
            ) = self.fetch_attributes(mol)

            score = 0

            # michael acceptor
            match = [mol.HasSubstructMatch(s) for s in michael_acceptor_smarts]
            if not any(match):
                results['michael_accept'].append(True)
                score += 1
            else:
                results['michael_accept'].append(False)

            # lipinkski
            if molecular_weight <= 500 and logp <= 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and rotatable_bonds <= 5:
                results["lipinski"].append(True)
                score += 1
            else:
                results["lipinski"].append(False)

            # ghose
            if molecular_weight >= 160 and molecular_weight <= 480 and logp >= -0.4 and logp <= 5.6 and number_of_atoms >= 20 and number_of_atoms <= 70 and molar_refractivity >= 40 and molar_refractivity <= 130:
                results['ghose'].append(True)
                score += 1

            else:
                results['ghose'].append(False)

            # veber
            if rotatable_bonds <= 10 and topological_surface_area_mapping <= 140:
                results['veber'].append(True)
                score += 1

            else:
                results['veber'].append(False)

            # rule of three
            if molecular_weight <= 300 and logp <= 3 and h_bond_donor <= 3 and h_bond_acceptors <= 3 and rotatable_bonds <= 3:
                results['rule_of_3'].append(True)
                score += 1

            else:
                results['rule_of_3'].append(False)

            # reos
            if molecular_weight >= 200 and molecular_weight <= 500 and logp >= int(
                    0 - 5) and logp <= 5 and h_bond_donor >= 0 and h_bond_donor <= 5 and h_bond_acceptors >= 0 and h_bond_acceptors <= 10 and formal_charge >= int(
                    0 - 2) and formal_charge <= 2 and rotatable_bonds >= 0 and rotatable_bonds <= 8 and heavy_atoms >= 15 and heavy_atoms <= 50:
                results['reos'].append(True)
                score += 1

            else:
                results['reos'].append(False)

            # drug likeness
            if molecular_weight < 400 and num_of_rings > 0 and rotatable_bonds < 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and logp < 5:
                results['drug_like'].append(True)
                score += 1

            else:
                results['drug_like'].append(False)

            if all([results[key][i] for key in ['lipinski', 'ghose', 'veber', 'rule_of_3', 'reos', 'drug_like']]):
                results['pass_all_filters'].append(True)
            else:
                results['pass_all_filters'].append(False)

            results['medchem_score'].append(score)
        return results


def main(file_path):

    # preds = pd.read_csv(sys.argv[1], delimiter='\t')
    preds = pd.read_csv(file_path, delimiter='\t')
    preds.set_index("ID", inplace=True)
    preds["SCORE"] = preds["PRED"] - preds["CONF"]

    hits = preds[preds["SCORE"] >= 0.3]

    hits.sort_values(by='SCORE', inplace=True, ascending=False)

    filter = DrugDesignFilters()
    filter_res = pd.DataFrame(filter.filter(
        hits["SMILES"].tolist()), index=hits.index)
    hits_filtered = pd.merge(
        hits, filter_res, left_index=True, right_index=True)

    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(
        smi), radius=3, nBits=2048) for smi in tqdm(hits_filtered["SMILES"])]
    lp = rdSimDivPickers.LeaderPicker()
    thresh = 0.65  # <- minimum distance between cluster centroids
    picks = lp.LazyBitVectorPick(fps, len(fps), thresh)
    clusters = assignPointsToClusters(picks, fps)

    cluster_ids = np.zeros(len(fps))
    for key, val in clusters.items():
        cluster_ids[val] = key

    hits_filtered['cluster_id'] = cluster_ids
    hits_filtered.sort_values(["medchem_score", "SCORE"],
                              ascending=False, inplace=True)

    picks = []

    # loop through the clusters, picking CLUSTER_SIZE/20 noms for each cluster
    hit_groups = hits_filtered.groupby('cluster_id')
    for _id, group in hit_groups.groups.items():
        picks.append(hits_filtered.loc[group[0]])
        _count = 0
        _pick = 0
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=3, nBits=2048) for smi in
               hits_filtered.loc[group]["SMILES"]]
        _running_sim = np.zeros(len(group))
        while (_count + 20) < len(group):
            sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[_pick], fps))
            _running_sim = _running_sim + sim
            _pick = np.argmin(_running_sim)
            picks.append(hits_filtered.loc[group[_pick]])
            _count += 20

    # pick the top 50
    picks = pd.DataFrame(picks)
    print(picks)
    top_50_picks = hits.sort_values(
        by='SCORE', ascending=False).iloc[:50]
    print(top_50_picks)
    top_50_picks.to_csv("../data/processed/compounds.csv")


if __name__ == "__main__":
    main("../data/processed/final_result.smi")
