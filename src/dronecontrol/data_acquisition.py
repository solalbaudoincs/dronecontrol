"""Data acquisition utilities for UAV scenarios."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir, resolve_path

LOGGER = logging.getLogger(__name__)


def load_csv_pair(input_path: str, output_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)
    _validate_columns(input_df, output_df)
    LOGGER.info("Loaded input CSV %s and output CSV %s", input_path, output_path)
    return input_df, output_df


def _validate_columns(input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
    if "time" in input_df.columns and "time" in output_df.columns:
        if not np.allclose(input_df["time"], output_df["time"]):
            raise ValueError("Input and output CSVs have mismatched time columns")


def maybe_run_simulink(model_path: str, samples: int, cache_dir: Path) -> Optional[pd.DataFrame]:
    if samples <= 0:
        return None
    try:
        import matlab.engine  # type: ignore
    except ImportError:
        LOGGER.warning("MATLAB Engine not available; skipping Simulink data generation")
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Launching MATLAB Engine to simulate %s for %d samples", model_path, samples)
    eng = matlab.engine.start_matlab()
    try:
        sim_out = eng.sim(model_path, nargout=1)
        data = np.array(sim_out["yout"], dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        data = data[:samples]
        columns = [f"sim_col_{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns)
        cache_path = cache_dir / "simulink_cache.json"
        cache_path.write_text(df.to_json(orient="records"))
        LOGGER.info("Simulink data cached at %s", cache_path)
        return df
    finally:
        eng.quit()


def assemble_dataset(
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    project_root: Path,
    default_processed_dir: Path,
) -> Dict[str, Any]:
    data_cfg: Dict[str, Any] = dict(scenario_cfg.get("data", {}) or {})
    input_raw = data_cfg.get("input_file")
    output_raw = data_cfg.get("output_file")
    if not input_raw or not output_raw:
        raise ValueError("Scenario data configuration must include input_file and output_file")
    input_file = resolve_path(str(project_root), str(input_raw))
    output_file = resolve_path(str(project_root), str(output_raw))
    processed_file = data_cfg.get("processed_file")
    if processed_file:
        processed_path = resolve_path(str(project_root), str(processed_file))
    else:
        processed_path = str(default_processed_dir / f"{scenario_name}.npz")

    ensure_dir(str(Path(processed_path).parent))

    inputs, outputs = load_csv_pair(input_file, output_file)
    sim_data: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = {
        "scenario": scenario_name,
        "input_path": input_file,
        "output_path": output_file,
        "processed_path": processed_path,
    }

    use_sim = bool(data_cfg.get("use_simulink", False))
    if use_sim:
        samples = int(data_cfg.get("extra_simulink_samples", 0))
        model_path = resolve_path(str(project_root), str(data_cfg.get("simulink_model", "UAV_model.slx")))
        sim_data = maybe_run_simulink(model_path, samples, project_root / "data" / "simulink")
        if sim_data is not None:
            meta["simulink_samples"] = len(sim_data)
            meta["simulink_model"] = model_path
            LOGGER.info("Generated %d samples from Simulink", len(sim_data))

    meta_path = Path(processed_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Metadata stored at %s", meta_path)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "processed_path": processed_path,
        "simulink_data": sim_data,
        "metadata": meta,
    }
