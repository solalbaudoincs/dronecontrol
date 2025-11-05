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
        if torch.tensor(self.input[idx], dtype=torch.float32).ndim == 1:
            input_tensor = torch.tensor(self.input[idx], dtype=torch.float32).unsqueeze(-1)
            target_tensor = torch.tensor(self.target[idx], dtype=torch.float32).unsqueeze(-1)
        else:
            input_tensor = torch.tensor(self.input[idx], dtype=torch.float32)
            target_tensor = torch.tensor(self.target[idx], dtype=torch.float32)

        return input_tensor, target_tensor