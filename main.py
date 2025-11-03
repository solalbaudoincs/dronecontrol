from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from dronecontrol.data_process.preparation import prepare_scenario_data
from dronecontrol.model_training import train_models_for_scenario
from dronecontrol.utils import configure_logging, ensure_dir, resolve_path, set_global_seed

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

        data_module, data_meta = prepare_scenario_data(scenario, project_root)

        checkpoints = train_models_for_scenario(
            scenario_name,
            scenario,
            general_cfg,
            data_module,
            data_meta["input_dim"],
            data_meta["output_dim"],
        )

        LOGGER.info("=== Completed scenario: %s ===", scenario_name)


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
