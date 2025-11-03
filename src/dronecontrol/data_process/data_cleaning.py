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

class AvDataCleaner(DataCleaner):
	def __init__(self, input_fp: str, output_fp: str):
		self.input_df = pd.read_csv(input_fp, header=None)
		self.output_df = pd.read_csv(output_fp, header=None)

	def calculate_energy(self, row : pd.Series) -> float:
		return np.sum(np.square(row))

	def filter_by_energy_ratio(self):
		energy_ratios = []
		energies_in = []
		energies_out = []

		for i in tqdm(range(len(self.input_df)), desc="Calculating energy ratios"):

			input_energy = self.calculate_energy(self.input_df.iloc[i])
			output_energy = self.calculate_energy(self.output_df.iloc[i])
			ratio = output_energy / input_energy if input_energy != 0 else np.nan
			energy_ratios.append(ratio)
			energies_in.append(input_energy)
			energies_out.append(output_energy)

		mean_ratio = np.nanmean(energy_ratios)
		std_ratio = np.nanstd(energy_ratios)

		indices_to_keep = [i for i, r in enumerate(energy_ratios) if r <= mean_ratio + std_ratio]
		filtered_input_df = self.input_df.iloc[indices_to_keep].reset_index(drop=True)
		filtered_output_df = self.output_df.iloc[indices_to_keep].reset_index(drop=True)
		LOGGER.info("Retiré %d relevés avec ratio d'énergie aberrant", len(self.input_df) - len(filtered_input_df))
		return filtered_input_df.to_numpy(), filtered_output_df.to_numpy()

	def get_clean_data(self) -> Any:
		return self.filter_by_energy_ratio()