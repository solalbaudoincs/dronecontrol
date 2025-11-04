"""Base Lightning module definitions."""

from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path


class BaseModel(pl.LightningModule):

    def __init__(self, input_dim: int, output_dim: int, lr: float, scheduler_type: Optional[str], scheduler_kwargs: Optional[dict]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.save_hyperparameters()
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.mse_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        
        # Generate and save plots
        fig = self.plot_predictions(y[0, ...].squeeze(-1), y_hat[0, ...].squeeze(-1))
        log_dir = Path("predictions_plots")
        log_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(log_dir / f"test_predictions_batch_{batch_idx}.png", dpi=100) # type: ignore[attr-defined]
        plt.close(fig)
        
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler_type:
            if self.scheduler_type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
        return optimizer
    
    def plot_predictions(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """Generate plots comparing predictions with ground truth."""
        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy()
        
        # Calculate error
        error = y_pred_np - y_true_np
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Error of model output vs true y
        axes[0].plot(error, label='Prediction Error', linewidth=1.5, alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].fill_between(range(len(error)), error, alpha=0.3)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Error (Predicted - True)')
        axes[0].set_title('Model Output Error vs Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Predictions overlaid on true values
        axes[1].plot(y_true_np, label='Ground Truth', linewidth=2, alpha=0.8)
        axes[1].plot(y_pred_np, label='Predictions', linewidth=2, alpha=0.8, linestyle='--')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Predictions vs Ground Truth')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
