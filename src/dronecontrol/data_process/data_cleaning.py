import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Any, Tuple
from pathlib import Path

from abc  import ABC, abstractmethod

LOGGER = logging.getLogger(__name__)


class DataCleaner(ABC):
	"""Base class for loading and cleaning data from csvs."""
	
	def __init__(self, input_fp: str, output_fp: str):
		self.input_df = pd.read_csv(input_fp, header=None)
		self.output_df = pd.read_csv(output_fp, header=None)
	
	@abstractmethod
	def get_clean_data(self) -> Tuple[np.ndarray, np.ndarray]:
		pass

class AVDataCleaner(DataCleaner):
	def __init__(self, input_fp: str, output_fp: str):
		self.input_df = pd.read_csv(input_fp, header=None)
		self.output_df = pd.read_csv(output_fp, header=None)

	def calculate_energy(self, row : pd.Series) -> float:
		return np.sum(np.square(row))

	def get_clean_data(self) -> Tuple[np.ndarray, np.ndarray]:

		eps = 1e-6  # Small constant to avoid division by zero

		input_energy = (self.input_df.to_numpy()**2).sum(axis=1)
		output_energy = (self.output_df.to_numpy()**2).sum(axis=1)
		ratio = output_energy / (input_energy + eps)

		mean_ratio = np.nanmean(ratio)
		std_ratio = np.nanstd(ratio)

		sigma = 1.98 # Corresponds to 95% confidence interval

		mask = abs(ratio - mean_ratio) <=  sigma * std_ratio

		# Application du masque sur les deux DataFrames
		self.input_df = self.input_df[mask].reset_index(drop=True)
		self.output_df = self.output_df[mask].reset_index(drop=True)

		print(f"✓ Filtered {len(mask) - np.sum(mask)} samples based on energy ratio")

		return self.input_df.to_numpy(), self.output_df.to_numpy()

def csv_to_tensor(csv_path: str | Path) -> np.ndarray:
    """
    Convertit un CSV concaténé avec colonne 'run_id' en tenseur numpy [run, time, dim].
    Hypothèse: tous les runs ont le même nombre d'échantillons.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "run_id" not in df.columns:
        raise ValueError(f"{csv_path} doit contenir une colonne 'run_id'.")

    # Trie par run_id pour reshaper proprement
    df = df.sort_values(["run_id"], kind="stable").reset_index(drop=True)

    runs = df["run_id"].unique()
    runs_sorted = np.sort(runs)
    n_runs = len(runs_sorted)

    # nombre d'échantillons par run (on prend le premier)
    n_time = df[df["run_id"] == runs_sorted[0]].shape[0]
    data_cols = [c for c in df.columns if c != "run_id"]
    n_dim = len(data_cols)

    # Vérif que toutes les longueurs sont identiques
    if not all(df[df["run_id"] == r].shape[0] == n_time for r in runs_sorted):
        raise ValueError("Tous les runs n'ont pas la même longueur.")

    data_flat = df.loc[:, data_cols].to_numpy(dtype=float)
    tensor = data_flat.reshape(n_runs, n_time, n_dim)
    return tensor



class FullMotorDataCleaner(DataCleaner):
    """
    Lit les 3 CSV concaténés (avec 'run_id'): _inputs.csv, _states.csv, _derivatives.csv
    et donne accès aux tenseurs numpy:
        inputs  -> (n_runs, n_time, 4)
        states  -> (n_runs, n_time+1, 12)  # attention: +1 si tu as sauvegardé N+1 états !
        derivs  -> (n_runs, n_time, 12)
    """
    def __init__(self, any_fp_under_data_dir: str, _unused_output_fp: str | None = None):
        data_root = Path(any_fp_under_data_dir).parent  # ex: .../control_4_motors/
        inputs_fp = data_root / "_inputs.csv"
        states_fp = data_root / "_states.csv"
        derivs_fp = data_root / "_derivatives.csv"

        # Lis avec header=0 (par défaut), car tes CSV ont un header ("run_id, ...")
        self.inputs_df = pd.read_csv(inputs_fp)
        self.states_df = pd.read_csv(states_fp)
        self.derivs_df = pd.read_csv(derivs_fp)

        # Tenseurs pré-calculés
        self.inputs_tensor = csv_to_tensor(inputs_fp)   # (n_runs, n_time, 4)
        # _states.csv contient N+1 lignes par run -> on reconstruit pareil
        self.states_tensor = csv_to_tensor(states_fp)   # (n_runs, n_time_states, 12)
        self.derivs_tensor = csv_to_tensor(derivs_fp)   # (n_runs, n_time, 12)

    def get_clean_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.inputs_tensor, self.derivs_tensor