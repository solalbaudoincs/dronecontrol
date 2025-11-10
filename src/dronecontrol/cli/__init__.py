"""CLI helpers for dronecontrol.

Expose public functions for training and report generation.
"""

from .train import run_training
from .report import run_report

__all__ = ["run_training", "run_report"]
