"""Generate an optimization report with detailed plots for the GRU-based MPC."""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import Optional
import torch
import matplotlib.pyplot as plt
import pandas as pd

from dronecontrol.models.gru_module import GRU
from dronecontrol.commande.MPC_torch import MPCTorch


@dataclass
class ScenarioConfig:
    dt: float = 0.05
    horizon: int = 15
    hidden_dim: int = 8
    num_layers: int = 1
    dropout: float = 0.0
    use_ekf: bool = True
    use_simulink: bool = True
    control_weight: float = 1.0
    tracking_weight: float = 10.0
    lr: float = 0.1
    max_epochs: int = 100
    checkpoint_path: Path = Path("models/accel_vs_voltage/gru/epoch=103-val_loss=0.0025.ckpt")
    report_dir: Path = Path("predictions_plots")
    tau: float = 0.3
    smoothing: bool = True


def load_model(cfg: ScenarioConfig, device: torch.device) -> GRU:
    model = GRU.load_from_checkpoint(cfg.checkpoint_path)
    model.to(device)
    return model


def build_mpc(model: GRU, cfg: ScenarioConfig) -> MPCTorch:
    # Create decaying weights: current control is more important than future
    decay_weights = np.exp(-np.arange(cfg.horizon, dtype=np.float32) * 0.1)
    Q = np.diag(decay_weights * cfg.control_weight)
    R = np.diag(decay_weights * cfg.tracking_weight)
    S = np.eye(cfg.horizon, dtype=np.float32) * (cfg.tracking_weight * 0.1)
    
    return MPCTorch(
        accel_model=model,
        dt=cfg.dt,
        horizon=cfg.horizon,
        Q=Q,
        R=R,
        S=S,
        tau=cfg.tau,
        lr=cfg.lr,
        max_epochs=cfg.max_epochs,
        use_ekf=cfg.use_ekf,
        use_simulink=cfg.use_simulink,
        smoothing=cfg.smoothing,
    )



def integrate_acceleration(
    acceleration: torch.Tensor,
    dt: float,
    x0: float,
    v0: float,
) -> tuple[np.ndarray, np.ndarray]:
    if acceleration.numel() == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    accel_cpu = acceleration.detach().cpu()
    dt_tensor = torch.tensor(dt, dtype=accel_cpu.dtype)
    v0_tensor = torch.tensor(v0, dtype=accel_cpu.dtype)
    x0_tensor = torch.tensor(x0, dtype=accel_cpu.dtype)

    vel = v0_tensor + torch.cumsum(accel_cpu * dt_tensor, dim=0)
    pos = x0_tensor + torch.cumsum(vel * dt_tensor, dim=0)

    return pos.numpy(), vel.numpy()


def plot_results(
    time_grid: np.ndarray,
    x_ref: np.ndarray,
    optimized_trajectory: np.ndarray,
    x_sim: np.ndarray,
    x_nn: np.ndarray,
    v_sim: np.ndarray,
    v_nn: np.ndarray,
    a_sim: np.ndarray,
    a_nn: np.ndarray,
    u_history: np.ndarray,
    figure_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    # Create 5 stacked plots: position, tracking error, velocity, acceleration, control
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # Position tracking: compare reference, simulink and NN predictions
    axes[0].plot(time_grid, optimized_trajectory, label="Filtered reference", color="tab:purple", linewidth=2, alpha=0.7)
    axes[0].plot(time_grid, x_ref, label="Reference", color="tab:blue", linewidth=2, linestyle=":")
    axes[0].plot(time_grid, x_sim, label="Position (Simulink)", color="tab:orange", linewidth=2)
    axes[0].plot(time_grid, x_nn, label="Position (NN)", color="tab:green", linestyle="--")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title("Position tracking: Simulink vs NN")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.6)

    
    # Control effort
    axes[1].plot(time_grid, u_history, color="tab:blue")
    axes[1].set_ylabel("Control input")
    axes[1].set_title("Control sequence")
    axes[1].set_xlabel("Time [s]")

    # Tracking error for both Simulink and NN
    tracking_error_sim = x_sim - x_ref
    tracking_error_nn = x_nn - x_ref
    axes[2].plot(time_grid, tracking_error_sim, label="Error (Sim)", color="tab:red")
    axes[2].plot(time_grid, tracking_error_nn, label="Error (NN)", color="tab:pink", linestyle="--")
    axes[2].fill_between(time_grid, tracking_error_sim, 0, color="tab:red", alpha=0.15)
    axes[2].set_title("Tracking error")
    axes[2].set_ylabel("Error [m]")
    axes[2].legend(loc="upper left", fontsize=8, framealpha=0.6)

    # Acceleration: Simulink measured vs NN predicted
    axes[3].plot(time_grid, a_sim, label="Accel (Simulink)", color="tab:brown")
    axes[3].plot(time_grid, a_nn, label="Accel (NN)", color="tab:cyan", linestyle="--")
    axes[3].set_ylabel("Acceleration [m/s^2]")
    axes[3].set_title("Acceleration: Simulink vs NN")
    axes[3].legend(loc="upper left", fontsize=8, framealpha=0.6)
    
    # Velocity comparison
    axes[4].plot(time_grid, v_sim, label="Velocity (Simulink)", color="tab:purple", linewidth=2)
    axes[4].plot(time_grid, v_nn, label="Velocity (NN)", color="tab:olive", linestyle="--")
    axes[4].set_ylabel("Velocity [m/s]")
    axes[4].set_title("Velocity estimates")
    axes[4].legend(loc="upper left", fontsize=8, framealpha=0.6)


    for axis in axes:
        axis.grid(True, linestyle=":", linewidth=0.5)

    fig.suptitle("GRU-based MPC report", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = ScenarioConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DRONE CONTROL OPTIMIZATION REPORT")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Horizon: {cfg.horizon}")

    model = load_model(cfg, device)
    controller = build_mpc(model, cfg)

    x0 = 0.0
    v0 = 0.0

    x_ref = torch.tensor([0.5, -0.25, 0], dtype=torch.float32, device=device)

    print("Running MPC optimization...")
    u_hist, histories = controller.solve(
        x_ref=x_ref,
        x0=x0,
        v0=v0,
        verbose=True,
    )

    # Extract histories for simulink and NN
    x_sim = histories["simulink"]["x"]
    v_sim = histories["simulink"]["v"]
    a_sim = histories["simulink"]["a"]

    x_nn = histories["nn"]["x"]
    v_nn = histories["nn"]["v"]
    a_nn = histories["nn"]["a"]

    x_ref = histories["references"]["steps"]
    optimized_trajectory = histories["references"]["filtered"]

    time_grid = np.arange(optimized_trajectory.shape[0], dtype=np.float32) * cfg.dt

    x_sim_np = x_sim.detach().cpu().numpy()
    v_sim_np = v_sim.detach().cpu().numpy()
    a_sim_np = a_sim.detach().cpu().numpy()

    x_nn_np = x_nn.detach().cpu().numpy()
    v_nn_np = v_nn.detach().cpu().numpy()
    a_nn_np = a_nn.detach().cpu().numpy()

    u_hist_np = u_hist.detach().cpu().numpy()
    x_ref_np = x_ref.detach().cpu().numpy()
    optimized_trajectory_np = optimized_trajectory.detach().cpu().numpy()

    # --- Save plotted data as CSVs and a combined DataFrame ---
    report_dir = cfg.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data dictionary (ensure same length by trimming to min length)
    data_dict = {
        "time": time_grid,
        "optimized_trajectory": optimized_trajectory_np,
        "reference": x_ref_np,
        "x_sim": x_sim_np,
        "x_nn": x_nn_np,
        "v_sim": v_sim_np,
        "v_nn": v_nn_np,
        "a_sim": a_sim_np,
        "a_nn": a_nn_np,
        "u_history": u_hist_np,
    }

    # Trim all arrays to the shortest length to avoid shape mismatch
    min_len = min(len(v) for v in data_dict.values())
    trimmed = {k: np.asarray(v)[:min_len] for k, v in data_dict.items()}

    df = pd.DataFrame(trimmed)
    combined_path = report_dir / "best_model" / "gru_mpc_report_0.002_rand_nosmooth.csv"
    df.to_csv(combined_path, index=False)

    # Only save the combined DataFrame as a single CSV (contains all series)
    figure_path = report_dir / "best_model" / "gru_mpc_report_0.002_rand_nosmooth.png"

    plot_results(
        time_grid=time_grid,
        x_ref=x_ref_np,
        optimized_trajectory=optimized_trajectory_np,
        x_sim=x_sim_np,
        x_nn=x_nn_np,
        v_sim=v_sim_np,
        v_nn=v_nn_np,
        a_sim=a_sim_np,
        a_nn=a_nn_np,
        u_history=u_hist_np,
        figure_path=figure_path,
    )

    

    print("Optimization completed.")
    print(f"Figure saved to {figure_path}")


if __name__ == "__main__":
    main()
