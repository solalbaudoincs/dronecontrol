from .data_cleaning import AvDataCleaner
from .data_loader import AVDataLoader
from .data_augmentation import DataAugmenter, NoOpAugmenter

import pytorch_lightning as pl


SCENARIOS = {
    "accel_vs_voltage" : {
        "data_cleaner": AvDataCleaner,
        "data_loader": AVDataLoader,
    }
}


def prepare_scenario_data(scenario_conf: dict[str, any]) -> pl.LightningDataModule:
    scenario_cfg = SCENARIOS.get(scenario_conf["name"])
    if scenario_cfg is None:
        raise ValueError(f"Unsupported scenario type: {scenario_conf['name']}")
    
    cleaner_cls = scenario_cfg["data_cleaner"]
    loader_cls = scenario_cfg["data_loader"]
    augmenter_cls = scenario_cfg.get("data_augmenter", NoOpAugmenter)

    input, target = cleaner_cls(scenario_conf["data"]["input_file"], scenario_conf["data"]["output_file"])

    augmented_input, augmented_target = augmenter_cls().augment(input, target)

    data_module = loader_cls(
        augmented_input,
        augmented_target,
        batch_size=scenario_conf["training"]["batch_size"],
    )
    
    return data_module

    