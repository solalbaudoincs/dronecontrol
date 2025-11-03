from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src import control_validation, data_acquisition, data_augmentation, data_cleaning, data_loader, model_training, report_generation
from src.utils import configure_logging, ensure_dir, resolve_path, set_global_seed

LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping")
    config.setdefault("scenarios", [])
    config.setdefault("general", {})
    return config  # type: ignore[return-value]


def run_pipeline(config_path: Path, selected_scenarios: List[str] | None = None) -> None:
    config = load_config(config_path)
    general_cfg = dict(config.get("general", {}))
    scenarios = config.get("scenarios", [])
    project_root = config_path.parent

    log_dir = resolve_path(str(project_root), general_cfg.get("log_dir", "logs"))
    ensure_dir(log_dir)
    general_cfg["log_dir"] = log_dir

    path_defaults = {
        "checkpoint_dir": "models",
        "results_dir": "results",
        "reports_dir": "reports",
        "default_processed_dir": "data/processed",
    }
    for key, default in path_defaults.items():
        value = general_cfg.get(key, default)
        general_cfg[key] = resolve_path(str(project_root), value)

    for scenario in scenarios:
        scenario_name = scenario.get("name")
        if not scenario_name:
            LOGGER.warning("Skipping unnamed scenario entry")
            continue
        if selected_scenarios and scenario_name not in selected_scenarios:
            continue

        configure_logging(log_dir, scenario_name)
        LOGGER.info("=== Running scenario: %s ===", scenario_name)

        seed = int(scenario.get("training", {}).get("seed", general_cfg.get("seed", 42)))
        set_global_seed(seed)

        processed_dir = Path(general_cfg.get("default_processed_dir", "data/processed"))
        data_bundle = data_acquisition.assemble_dataset(
            scenario_name,
            scenario,
            project_root,
            processed_dir if processed_dir.is_absolute() else project_root / processed_dir,
        )

        inputs_df = data_bundle["inputs"].copy()
        outputs_df = data_bundle["outputs"].copy()
        numeric_inputs = inputs_df.select_dtypes(include="number").columns
        numeric_outputs = outputs_df.select_dtypes(include="number").columns

        pre_cfg = dict(scenario.get("preprocessing", {}))
        combined_df = pd.concat([inputs_df, outputs_df], axis=1)
        combined_df = data_cleaning.remove_outliers(combined_df, method=pre_cfg.get("outlier_method", "iqr"))
        combined_df = data_cleaning.interpolate_missing(combined_df, method=pre_cfg.get("fill_method", "linear"))
        combined_df, norm_params = data_cleaning.normalize(
            combined_df,
            scope=pre_cfg.get("normalization_scope", "per_feature"),
        )

        inputs_clean = combined_df[numeric_inputs]
        outputs_clean = combined_df[numeric_outputs]

        X_array, Y_array = data_loader.dataframe_to_arrays(inputs_clean, outputs_clean)
        X_array, Y_array = data_augmentation.append_simulink_samples(
            X_array,
            Y_array,
            data_bundle.get("simulink_data"),
        )

        aug_cfg = dict(scenario.get("augmentation", {}))
        noise_std = float(aug_cfg.get("noise_std", 0.0))
        X_aug, Y_aug = data_augmentation.add_gaussian_noise(X_array, Y_array, noise_std=noise_std, seed=seed)

        window_size = int(aug_cfg.get("window_size", 1))
        window_stride = int(aug_cfg.get("window_stride", 1))
        windowed = data_augmentation.slice_windows(
            X_aug,
            Y_aug,
            window_size=window_size,
            stride=window_stride,
        )

        train_cfg = dict(scenario.get("training", {}))
        split_arrays = data_loader.train_val_test_split(
            windowed["X"],
            windowed["Y"],
            val_ratio=float(train_cfg.get("val_split", 0.2)),
            test_ratio=float(train_cfg.get("test_split", 0.1)),
            seed=seed,
        )

        processed_path = data_bundle["processed_path"]
        data_loader.save_npz(processed_path, split_arrays)

        checkpoints = model_training.train_models_for_scenario(
            scenario_name,
            scenario,
            general_cfg,
            processed_path,
        )

        metrics_path = control_validation.validate_control(
            scenario_name,
            scenario,
            general_cfg,
            processed_path,
            checkpoints,
        )

        report_generation.generate_report(
            scenario_name,
            scenario,
            general_cfg,
            processed_path,
            metrics_path,
            checkpoints,
        )

        _update_metadata(processed_path, {
            "seed": seed,
            "normalization": norm_params,
            "window_size": window_size,
            "window_stride": window_stride,
            "noise_std": noise_std,
        })

        LOGGER.info("=== Completed scenario: %s ===", scenario_name)


def _update_metadata(processed_path: str, updates: Dict[str, Any]) -> None:
    meta_path = Path(processed_path).with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UAV data-driven control pipeline")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration file")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Specific scenario name(s) to execute. Can be used multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.scenario)


if __name__ == "__main__":
    main()
