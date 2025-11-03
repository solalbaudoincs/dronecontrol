"""Reporting utilities for UAV control pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from dronecontrol.data_process.data_loader import load_npz
from .models import MODEL_REGISTRY

LOGGER = logging.getLogger(__name__)


def generate_report(
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    general_cfg: Dict[str, Any],
    processed_path: str,
    metrics_path: Path,
    checkpoints: Iterable[Tuple[str, Path]],
) -> Path:
    report_cfg = dict(scenario_cfg.get("reporting", {}) or {})
    if not report_cfg.get("generate_pdf", True):
        LOGGER.info("PDF reporting disabled for %s", scenario_name)
        return Path()

    reports_dir = Path(general_cfg.get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / f"report_{scenario_name}.pdf"

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz(processed_path)
    figures = []
    for model_name, ckpt_path in checkpoints:
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            continue
        model = model_cls.load_from_checkpoint(str(ckpt_path))
        model.eval()
        model.freeze()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        preds = _predict(model, X_test)[: min(len(X_test), 200)]
        truth = Y_test[: len(preds)]
        fig_path = reports_dir / f"{scenario_name}_{model_name}_timeseries.png"
        _plot_timeseries(preds, truth, fig_path, model_name)
        figures.append((model_name, fig_path))

    _render_pdf(pdf_path, scenario_name, metrics, figures)
    LOGGER.info("Generated PDF report at %s", pdf_path)
    return pdf_path


def _predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    X_tensor = torch.from_numpy(X).float().to(model.device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    return preds


def _plot_timeseries(preds: np.ndarray, truth: np.ndarray, path: Path, model_name: str) -> None:
    plt.figure(figsize=(8, 3))
    plt.plot(truth, label="True")
    plt.plot(preds, label="Predicted", linestyle="--")
    plt.title(f"{model_name} Predictions vs Truth")
    plt.xlabel("Samples")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _render_pdf(pdf_path: Path, scenario: str, metrics: Dict[str, Any], figures: Iterable[Tuple[str, Path]]) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin = 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, f"UAV Scenario Report: {scenario}")
    c.setFont("Helvetica", 12)
    y = height - margin - 30

    for model_name, model_metrics in metrics.items():
        c.drawString(margin, y, f"Model: {model_name}")
        y -= 20
        if isinstance(model_metrics, dict):
            for section, section_metrics in model_metrics.items():
                c.drawString(margin + 20, y, f"{section.upper()}")
                y -= 18
                if isinstance(section_metrics, dict):
                    for key, value in section_metrics.items():
                        c.drawString(margin + 40, y, f"{key}: {value}")
                        y -= 16
                y -= 6
        y -= 10
        if y < margin + 200:
            c.showPage()
            y = height - margin

    for model_name, fig_path in figures:
        if not fig_path.exists():
            continue
        if y < margin + 200:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, f"Figure: {model_name}")
        y -= 20
        img = ImageReader(str(fig_path))
        img_width, img_height = img.getSize()
        scale = min((width - 2 * margin) / img_width, 200 / img_height)
        c.drawImage(img, margin, y - img_height * scale, width=img_width * scale, height=img_height * scale)
        y -= img_height * scale + 30

    c.save()
