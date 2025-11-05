import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Any, Tuple

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