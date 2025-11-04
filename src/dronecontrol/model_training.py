"""Scenario training orchestration built on PyTorch Lightning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from .models import MODEL_REGISTRY

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {}


def _resolve_accelerator(device_pref: str) -> Tuple[str, str | int]:
    device_pref = device_pref.lower()
    if device_pref == "cpu":
        return "cpu", 1
    if device_pref == "gpu":
        return "gpu", "auto"
    return "auto", "auto"


def _collect_model_params(model_name: str, training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = training_cfg.get("model_params", {}).get(model_name, {})
    overrides = training_cfg.get("model_params", {}).get(model_name, {})
    params.update(overrides)
    return params


def train_models_for_scenario(
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    general_cfg: Dict[str, Any],
    data_module: pl.LightningDataModule,
    input_dim: int,
    output_dim: int,
) -> List[Tuple[str, Path]]:
    training_cfg = dict(scenario_cfg.get("training", {}) or {})
    models: Iterable[str] = training_cfg.get("models", ["gru"])
    models = [model.lower() for model in models]

    if not models:
        LOGGER.warning("No models configured for scenario %s", scenario_name)
        return []

    lr = float(training_cfg["lr"])
    epochs = int(training_cfg["epochs"])
    seed = int(training_cfg["seed"])
    patience = int(training_cfg["early_stopping_patience"])

    pl.seed_everything(seed, workers=True)

    checkpoint_root = Path(general_cfg.get("checkpoint_dir", "models")) / scenario_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    log_root = Path(general_cfg.get("log_dir", "logs")) / scenario_name
    log_root.mkdir(parents=True, exist_ok=True)

    accelerator, devices = _resolve_accelerator(str(general_cfg.get("device", "auto")))
    deterministic = bool(general_cfg.get("deterministic", True))

    trained_models: List[Tuple[str, Path]] = []

    for model_name in models:
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            LOGGER.error("Unknown model '%s' requested for scenario %s", model_name, scenario_name)
            continue

        model_params = _collect_model_params(model_name, training_cfg)
        scheduler_cfg = training_cfg.get("scheduler", {})
        scheduler_type = scheduler_cfg.get("type")
        scheduler_kwargs = scheduler_cfg.get("params", {})
        model = model_cls(
            input_dim,
            output_dim,
            lr=lr,
            scheduler_type=scheduler_type,
            scheduler_kwargs=scheduler_kwargs,
            **model_params,
        )

        checkpoint_dir = checkpoint_root / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            verbose=False,
        )
        csv_logger = CSVLogger(save_dir=log_root, name=model_name)

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=[checkpoint_callback, early_stopping],
            logger=csv_logger,
            deterministic=deterministic,
            log_every_n_steps =10,
            enable_progress_bar=True,
        )

        LOGGER.info(
            "Training %s model for scenario %s (epochs=%d, lr=%.3g)",
            model_name,
            scenario_name,
            epochs,
            lr,
        )

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
        trainer.validate(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt:
            ckpt_path = Path(best_ckpt)
            trained_models.append((model_name, ckpt_path))
            LOGGER.info("Best %s checkpoint stored at %s", model_name, ckpt_path)
        else:
            LOGGER.warning("No checkpoint produced for model %s in scenario %s", model_name, scenario_name)

    return trained_models
