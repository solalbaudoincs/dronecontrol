"""Control validation routines for trained models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from dronecontrol.data_process.data_loader import load_npz
from .models import MODEL_REGISTRY

LOGGER = logging.getLogger(__name__)


def validate_control(
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    general_cfg: Dict[str, Any],
    processed_path: str,
    checkpoints: Iterable[Tuple[str, Path]],
) -> Path:
    control_cfg = dict(scenario_cfg.get("control", {}) or {})
    methods = [m.lower() for m in control_cfg.get("methods", [])]
    horizon = int(control_cfg.get("horizon", 10))
    weights = {
        "tracking": float(control_cfg.get("mpc_weight_tracking", 1.0)),
        "smoothness": float(control_cfg.get("mpc_weight_smoothness", 0.1)),
        "effort": float(control_cfg.get("mpc_weight_effort", 0.01)),
    }

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz(processed_path)
    dataset = {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test),
    }

    metrics: Dict[str, Any] = {}
    for model_name, ckpt_path in checkpoints:
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            LOGGER.warning("Skipping unknown model %s", model_name)
            continue
        model = model_cls.load_from_checkpoint(str(ckpt_path))
        model.eval()
        model.freeze()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        eval_results = _evaluate_model(model, dataset)
        metrics[model_name] = {"open_loop": eval_results}

        if "mpc" in methods and model_name == "linear":
            metrics[model_name]["mpc"] = _evaluate_mpc(model, dataset["test"], horizon, weights)
        if any(m in {"ppo", "ddpg", "rl"} for m in methods) and model_name != "linear":
            metrics[model_name]["rl"] = _evaluate_rl_stub(control_cfg)

    results_dir = Path(general_cfg.get("results_dir", "results")) / scenario_name
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = control_cfg.get("metrics_file")
    if metrics_path:
        metrics_file = Path(metrics_path)
        if not metrics_file.is_absolute():
            metrics_file = Path(general_cfg.get("results_dir", "results")) / metrics_path
    else:
        metrics_file = results_dir / "metrics.json"

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Stored validation metrics at %s", metrics_file)
    return metrics_file


def _evaluate_model(model: torch.nn.Module, dataset: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    X_test, Y_test = dataset["test"]
    X_tensor = torch.from_numpy(X_test).float().to(model.device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    tracking_error = float(np.mean((preds - Y_test) ** 2))
    smoothness = float(np.mean(np.linalg.norm(np.diff(preds, axis=0), axis=-1))) if len(preds) > 1 else 0.0
    effort = float(np.mean(np.linalg.norm(X_test.reshape(X_test.shape[0], -1), axis=1)))
    return {
        "tracking_error": tracking_error,
        "control_smoothness": smoothness,
        "input_effort": effort,
    }


def _evaluate_mpc(
    model: torch.nn.Module,
    test_data: Tuple[np.ndarray, np.ndarray],
    horizon: int,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    X_test, Y_test = test_data
    Y_pred = []
    effort_terms = []
    model_device = next(model.parameters()).device
    for start in range(0, len(X_test) - horizon):
        window = X_test[start : start + horizon]
        window_tensor = torch.from_numpy(window).float().to(model_device)
        with torch.no_grad():
            pred = model(window_tensor).cpu().numpy()
        Y_pred.append(pred[-1])
        effort_terms.append(np.linalg.norm(window.reshape(window.shape[0], -1), axis=1).mean())
    if not Y_pred:
        return {"status": "insufficient_data"}
    Y_pred_arr = np.vstack(Y_pred)
    target = Y_test[-len(Y_pred_arr) :]
    tracking = float(np.mean((Y_pred_arr - target) ** 2))
    smoothness = float(np.mean(np.linalg.norm(np.diff(Y_pred_arr, axis=0), axis=-1))) if len(Y_pred_arr) > 1 else 0.0
    effort = float(np.mean(effort_terms)) if effort_terms else 0.0
    cost = (
        weights["tracking"] * tracking
        + weights["smoothness"] * smoothness
        + weights["effort"] * effort
    )
    return {
        "tracking_error": tracking,
        "control_smoothness": smoothness,
        "input_effort": effort,
        "mpc_cost": cost,
    }


def _evaluate_rl_stub(control_cfg: Dict[str, Any]) -> Dict[str, str]:
    algo = control_cfg.get("rl_algorithm", "ppo")
    try:
        import stable_baselines3  # type: ignore  # noqa: F401
    except ImportError:
        return {
            "status": "skipped",
            "reason": "stable-baselines3 not installed",
            "algorithm": algo,
        }
    return {
        "status": "not_implemented",
        "reason": "RL policy evaluation placeholder",
        "algorithm": algo,
    }
