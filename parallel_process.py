import os
import numpy as np
import multiprocessing as mp
from typing import Optional, List
from tqdm import tqdm
from rdkit.Chem import MolFromSmiles
import pickle


class ParallelScreener:
    def __init__(self, models, fp_funcs, ensemble):
        self._models = models
        self._fp_func = fp_funcs
        self._ensemble = ensemble

    def chunk_data(self, filepath: str, chunk_size: int = 10000):
        """
        Generator to read and yield chunks of data from a large file
        """
        with open(filepath, "r") as f:
            # Skip header
            header = f.readline().strip().split("\t")
            try:
                smiles_idx = header.index("smiles")
                names_idx = header.index("names")
            except ValueError:
                smiles_idx = 0
                names_idx = 1
                f.seek(0)  # Reset file pointer

            chunk_smiles = []
            chunk_names = []

            for line in f:
                splits = line.strip().split("\t")
                chunk_smiles.append(splits[smiles_idx].strip())

                # Handle potential missing names
                chunk_names.append(splits[names_idx].strip(
                ) if names_idx < len(splits) else None)

                if len(chunk_smiles) == chunk_size:
                    yield chunk_smiles, chunk_names
                    chunk_smiles = []
                    chunk_names = []

            # Yield the last chunk if not empty
            if chunk_smiles:
                yield chunk_smiles, chunk_names

    def process_chunk(self, chunk_data):
        """
        Process a single chunk of data
        """
        smiles, names = chunk_data

        # Screen SMILES
        preds, confs = self.screen_smiles(smiles)

        # Prepare results
        results = []
        for n, s, p, c in zip(names, smiles, preds, confs):
            results.append((n, s, round(float(p), 4), round(float(c), 4)))

        return results

    def parallel_screen_from_local_dir(self, filepath: str, outpath: Optional[str] = None,
                                       num_processes: Optional[int] = None):
        """
        Parallel screening of large datasets

        :param filepath: Input file path
        :param outpath: Output file path
        :param num_processes: Number of processes to use (defaults to all available cores)
        """
        # Set default output path if not provided
        if outpath is None:
            outpath = os.path.abspath(filepath).split('.')[0] + ".PREDS"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        # Create output file with header if it doesn't exist
        if not os.path.exists(outpath):
            with open(outpath, 'w') as outfile:
                outfile.write("ID\tSMILES\tPRED\tCONF\n")
            print(f"Created file: {outpath}")

        # Determine number of processes
        if num_processes is None:
            num_processes = mp.cpu_count()

        # Use a Pool of workers
        with mp.Pool(processes=num_processes) as pool:
            # Create a tqdm progress bar
            with tqdm(total=sum(1 for _ in self.chunk_data(filepath))) as pbar:
                # Prepare to write results
                with open(outpath, "a") as f2:
                    # Process chunks in parallel
                    for results in pool.imap(self.process_chunk, self.chunk_data(filepath)):
                        # Write results to file
                        for result in results:
                            f2.write(f"{result[0]}\t{result[1]}\t{
                                     result[2]}\t{result[3]}\n")
                        # Update progress bar
                        pbar.update(1)

        print(f"Screening complete. Results written to {outpath}")

    def screen_smiles(self, smis: list[str]):
        """
        Screens a list of smiles and returns predictions and confidences
        """
        # Remove invalid SMILES
        invalid_smiles = [smi for smi in smis if MolFromSmiles(smi) is None]
        if invalid_smiles:
            print("Invalid SMILES strings:", invalid_smiles)
            smis = [smi for smi in smis if MolFromSmiles(smi) is not None]

        # Generate fingerprints
        fps = []
        for _fp in self._fp_func:
            fps.append(list(FPS_FUNCS[_fp](smis)))

        # Predict using ensemble models
        test_preds = []
        for i_model in range(self._ensemble):
            for clf, fp in zip(self._models[i_model], fps):
                test_preds.append(clf.predict_proba(fp)[:, 1])

        test_preds = np.array(test_preds).T
        preds = test_preds.mean(axis=1)
        confs = test_preds.std(axis=1)

        # Filter predictions
        valid_indices = preds > 0.3
        filtered_preds = preds[valid_indices]
        filtered_confs = confs[valid_indices]

        return filtered_preds, filtered_confs
