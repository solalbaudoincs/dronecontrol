from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytorch_lightning as pl

from .data_cleaning import AVDataCleaner, FullMotorDataCleaner
from .data_loader import AVDataLoader
from .data_augmentation import DataAugmenter, NoOpAugmenter


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "accel_vs_voltage": {
        "data_cleaner": AVDataCleaner,
        "data_loader": AVDataLoader,
    },
    "control_4_motors" : {
        "data_cleaner": FullMotorDataCleaner,
        "data_loader": AVDataLoader,
    }
}


def prepare_scenario_data(
    scenario_conf: Dict[str, Any],
    project_root: Path,
) -> Tuple[pl.LightningDataModule, Dict[str, Any]]:
    scenario_cfg = SCENARIOS.get(scenario_conf["name"])
    if scenario_cfg is None:
        raise ValueError(f"Unsupported scenario type: {scenario_conf['name']}")

    cleaner_cls = scenario_cfg["data_cleaner"]
    loader_cls = scenario_cfg["data_loader"]
    augmenter_cls: type[DataAugmenter] = scenario_cfg.get("data_augmenter", NoOpAugmenter)

    data_cfg = scenario_conf.get("data", {})
    training_cfg = scenario_conf.get("training", {})

    input_path = project_root / data_cfg["input_file"]
    output_path = project_root / data_cfg["output_file"]

    cleaner = cleaner_cls(str(input_path), str(output_path))
    raw_input, raw_target = cleaner.get_clean_data()

    augmenter = augmenter_cls()
    augmented_input, augmented_target = augmenter.augment(raw_input, raw_target)

    

    batch_size = int(training_cfg.get("batch_size", 32))
    val_split = float(training_cfg.get("val_split", 0.2))
    test_split = float(training_cfg.get("test_split", 0.1))
    seed = int(training_cfg.get("seed", 42))

    data_module = loader_cls(
        augmented_input,
        augmented_target,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    metadata = {
        "input_dim": data_cfg["input_dim"],
        "output_dim": data_cfg["output_dim"],
        "num_samples": int(augmented_input.shape[0]),
    }

    return data_module, metadata

