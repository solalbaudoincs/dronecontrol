"""Demo runner that exercises ControlRunner with a simple AccelModel.

Usage: run with the project's venv Python.
"""
from pathlib import Path
import sys
# ensure `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
import torch.nn as nn
from dronecontrol.commande.control_runner import ControlRunner


class AccelModel(nn.Module):
    """Simple accel model: returns u - g so optimizer must pick u ~= g to hover."""

    def __init__(self, g: float = 9.81):
        super().__init__()
        self.g = g

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # assume u has shape (horizon, 1)
        return u - self.g


def main():
    device = "cpu"
    g = 9.81
    dt = 0.1
    horizon = 20
    x0 = 1.0
    v0 = 0.0

    x_ref = torch.full((horizon, 1), x0, dtype=torch.float32)

    # simple Q and R
    Q = torch.eye(horizon)
    R = torch.eye(horizon)

    runner = ControlRunner(
        ckpt_path=None,
        model_class=AccelModel,
        model_kwargs={"g": g},
        device=device,
    )

    u_opt = runner.run(
        x_ref=x_ref,
        x0=x0,
        v0=v0,
        horizon=horizon,
        dt=dt,
        Q=Q,
        R=R,
        lr=1.0,
        max_iter=40,
        history_size=10,
        max_epochs=80,
    )

    u_np = u_opt.detach().cpu().numpy().flatten()
    print(f"u_opt mean={u_np.mean():.4f}, std={u_np.std():.4f}")
    print("u_opt (first 10):", u_np[:10])

    # save result
    out_path = Path("predictions_plots")
    out_path.mkdir(exist_ok=True)
    torch.save(u_opt, out_path / "u_opt.pt")


if __name__ == "__main__":
    main()
