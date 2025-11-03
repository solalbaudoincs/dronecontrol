from torch.utils.data import Dataset
import numpy as np
import torch

class AVDataset(Dataset):
    
    def __init__(
            self,
            input: np.ndarray,
            target: np.ndarray
            ) -> None:
        
        super().__init__()

        self.input = input
        self.target = target

    def __len__(self) -> int:
        return len(self.input)
    
    def __getitem__(self, idx: int):
        return {
            "input": torch.tensor(self.input[idx]),
            "target": torch.tensor(self.target[idx]),
        }