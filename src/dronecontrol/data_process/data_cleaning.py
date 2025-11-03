import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

class AvDataCleaner:

	def __init__(
			self, 
			input_df: str, 
			output_df: str
			):
		
		self.input_df = pd.read_csv(input_df, header=None)
		self.output_df = pd.read_csv(output_df, header=None)
	

	def filter_by_energy_ratio(self):

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
