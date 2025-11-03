"""Model training orchestration using PyTorch Lightning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from dronecontrol.data_process.data_loader import load_npz
from .models import MODEL_REGISTRY

LOGGER = logging.getLogger(__name__)

class UAVDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[TensorDataset] = None
        self.val_ds: Optional[TensorDataset] = None
        self.test_ds: Optional[TensorDataset] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz(self.data_path)
        self.input_dim = X_train.shape[-1]
        self.output_dim = Y_train.shape[-1]
        self.train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
        self.val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
        self.test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.train_ds is not None
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.val_ds is not None
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.test_ds is not None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def _determine_accelerator(device: str) -> str:
    if device == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if device in {"cpu", "gpu"}:
        return device
    raise ValueError(f"Unsupported device option: {device}")


def train_models_for_scenario(
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    general_cfg: Dict[str, Any],
    processed_path: str,
) -> List[Tuple[str, Path]]:
    training_cfg = dict(scenario_cfg.get("training", {}) or {})
    models = training_cfg.get("models", [])
    if not models:
        LOGGER.warning("No models specified for scenario %s", scenario_name)
        return []

    checkpoint_dir = Path(general_cfg.get("checkpoint_dir", "models")) / scenario_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(general_cfg.get("log_dir", "logs")) / scenario_name
    log_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(training_cfg.get("batch_size", 32))
    epochs = int(training_cfg.get("epochs", 50))
    lr = float(training_cfg.get("lr", 1e-3))
    num_workers = int(general_cfg.get("num_workers", 0))
    resume = bool(general_cfg.get("resume_from_checkpoint", False))
    model_params: Dict[str, Any] = dict(training_cfg.get("model_params", {}) or {})

    datamodule = UAVDataModule(processed_path, batch_size=batch_size, num_workers=num_workers)
    datamodule.setup()

    checkpoints: List[Tuple[str, Path]] = []
    accelerator = _determine_accelerator(str(general_cfg.get("device", "auto")))

    for model_name in models:
        model_name = model_name.lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            LOGGER.error("Unknown model type %s", model_name)
            continue
        assert datamodule.input_dim is not None and datamodule.output_dim is not None
        extra_kwargs = dict(model_params.get(model_name, {}))
        model = model_cls(datamodule.input_dim, datamodule.output_dim, lr=lr, **extra_kwargs)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(
                dirpath=checkpoint_dir / model_name,
                filename="{epoch:03d}-{val_loss:.4f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
        ]
        csv_logger = CSVLogger(save_dir=log_dir, name=model_name)

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            callbacks=callbacks,
            logger=csv_logger,
            deterministic=bool(general_cfg.get("deterministic", True)),
            enable_checkpointing=True,
            log_every_n_steps=10,
        )

        ckpt_path = None
        if resume:
            last_ckpt = _find_last_checkpoint(checkpoint_dir / model_name)
            if last_ckpt:
                ckpt_path = str(last_ckpt)
                LOGGER.info("Resuming %s from %s", model_name, ckpt_path)

        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.validate(model, datamodule=datamodule)

        best_ckpt = callbacks[1].best_model_path  # ModelCheckpoint
        if best_ckpt:
            checkpoints.append((model_name, Path(best_ckpt)))
            LOGGER.info("Best checkpoint for %s stored at %s", model_name, best_ckpt)
        else:
            LOGGER.warning("Trainer did not produce a checkpoint for %s", model_name)

    return checkpoints


def _find_last_checkpoint(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    checkpoints = sorted(directory.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None
