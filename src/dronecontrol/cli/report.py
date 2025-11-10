"""Optimization report generation wrapper for CLI."""
from pathlib import Path
import torch
import numpy as np

from dronecontrol.globals import BASEDIR
from dronecontrol.models.base_module import BaseModel
from dronecontrol.models import MODEL_REGISTRY
from dronecontrol.commande.MPC_torch import MPCTorch


# Predefined trajectory presets
PREDEFINED_TRAJECTORIES = {
    "default": torch.tensor([0.5, -0.25, 0], dtype=torch.float32),
    "5-step": torch.rand(5, dtype=torch.float32)*10,
    "10-step": torch.tensor(10, dtype=torch.float32)*10,
}


def _load_model(model_name: str, checkpoint_path: Path, device: torch.device) -> BaseModel:
    """Load GRU model from checkpoint."""

    model = (MODEL_REGISTRY[model_name]).load_from_checkpoint(checkpoint_path)
    model.to(device)
    return model


def _build_mpc(model: BaseModel, dt: float, horizon: int, lr: float, max_epochs: int,
               use_ekf: bool, use_simulink: bool, optimize_trajectory: bool,
               max_accel: float, control_weight: float, tracking_weight: float) -> MPCTorch:
    """Build MPC controller with specified configuration."""
    # Create decaying weights
    Q = np.eye(horizon, dtype=np.float32) * control_weight
    R = np.eye(horizon, dtype=np.float32) * tracking_weight
    S = np.eye(horizon, dtype=np.float32) * 0.0
    
    return MPCTorch(
        accel_model=model,
        dt=dt,
        horizon=horizon,
        Q=Q,
        R=R,
        S=S,
        tau=0.3,
        lr=lr,
        max_epochs=max_epochs,
        use_ekf=use_ekf,
        use_simulink=use_simulink,
        optimize_trajectory=optimize_trajectory,
        max_accel=max_accel,
    )


def _plot_results(
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
    """Create comprehensive optimization report plot."""
    import matplotlib.pyplot as plt
    
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # Position tracking
    axes[0].plot(time_grid, optimized_trajectory, label="Optimized Trajectory", 
                 color="tab:purple", linewidth=2, alpha=0.7)
    axes[0].plot(time_grid, x_ref, label="Reference", 
                 color="tab:blue", linewidth=2, linestyle=":")
    axes[0].plot(time_grid, x_sim, label="Position (Simulink)", 
                 color="tab:orange", linewidth=2)
    axes[0].plot(time_grid, x_nn, label="Position (NN)", 
                 color="tab:green", linestyle="--")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title("Position tracking: Simulink vs NN")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.6)

    # Tracking error
    tracking_error_sim = x_sim - x_ref
    tracking_error_nn = x_nn - x_ref
    axes[1].plot(time_grid, tracking_error_sim, label="Error (Sim)", color="tab:red")
    axes[1].plot(time_grid, tracking_error_nn, label="Error (NN)", 
                 color="tab:pink", linestyle="--")
    axes[1].fill_between(time_grid, tracking_error_sim, 0, color="tab:red", alpha=0.15)
    axes[1].set_title("Tracking error")
    axes[1].set_ylabel("Error [m]")
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.6)

    # Velocity
    axes[2].plot(time_grid, v_sim, label="Velocity (Simulink)", 
                 color="tab:purple", linewidth=2)
    axes[2].plot(time_grid, v_nn, label="Velocity (NN)", 
                 color="tab:olive", linestyle="--")
    axes[2].set_ylabel("Velocity [m/s]")
    axes[2].set_title("Velocity estimates")
    axes[2].legend(loc="upper left", fontsize=8, framealpha=0.6)

    # Acceleration
    axes[3].plot(time_grid, a_sim, label="Accel (Simulink)", color="tab:brown")
    axes[3].plot(time_grid, a_nn, label="Accel (NN)", 
                 color="tab:cyan", linestyle="--")
    axes[3].set_ylabel("Acceleration [m/s²]")
    axes[3].set_title("Acceleration: Simulink vs NN")
    axes[3].legend(loc="upper left", fontsize=8, framealpha=0.6)
    
    # Control
    axes[4].plot(time_grid, u_history, color="tab:blue")
    axes[4].set_ylabel("Control input")
    axes[4].set_title("Control sequence")
    axes[4].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True, linestyle=":", linewidth=0.5)

    fig.suptitle("GRU-based MPC Report", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_report(
    model_name: str,
    model_ckpt: str | None = None,
    use_ekf: bool = True,
    use_simulink: bool = True,
    optimize_trajectory: bool = True,
    max_epochs: int = 100,
    trajectory: str = "default",
    dt: float = 0.05,
    horizon: int = 15,
    lr: float = 0.1,
    max_accel: float = 9.81,
    control_weight: float = 1.0,
    tracking_weight: float = 10.0,
):
    """Run optimization report and save plot to repository root.

    Parameters
    ----------
    model_name : str
        Model identifier for default checkpoint path
    model_ckpt : str, optional
        Checkpoint path. Default: {BASEDIR}/{model_name}_best.ckpt
    use_ekf : bool, default=True
        Use Extended Kalman Filter
    use_simulink : bool, default=True
        Use Simulink for simulation
    optimize_trajectory : bool, default=True
        Enable trajectory optimization
    max_epochs : int, default=100
        Maximum optimization epochs
    trajectory : str, default="default"
        Trajectory preset name (choices: default, step, multi, smooth)
    dt : float, default=0.05
        Time step
    horizon : int, default=30
        MPC horizon
    lr : float, default=0.1
        Optimizer learning rate
    max_speed : float, default=5.0
        Maximum speed for trajectory optimization
    control_weight : float, default=1.0
        Control cost weight
    tracking_weight : float, default=10.0
        Tracking error weight

    Returns
    -------
    Path
        Path to saved figure
    """
    # Resolve checkpoint path
    if model_ckpt is None:
        checkpoint_path = Path(BASEDIR) / f"{model_name}-best.ckpt"
    else:
        checkpoint_path = Path(model_ckpt)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Get trajectory
    if trajectory not in PREDEFINED_TRAJECTORIES:
        raise ValueError(
            f"Unknown trajectory: {trajectory}. "
            f"Choose from: {list(PREDEFINED_TRAJECTORIES.keys())}"
        )
    x_ref = PREDEFINED_TRAJECTORIES[trajectory]

    print("=" * 60)
    print("DRONE CONTROL OPTIMIZATION REPORT")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trajectory: {trajectory}")
    print(f"Use EKF: {use_ekf}, Use Simulink: {use_simulink}")
    print(f"Optimize trajectory: {optimize_trajectory}")
    print(f"Max epochs: {max_epochs}, Horizon: {horizon}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = _load_model(model_name, checkpoint_path, device)

    # Build MPC controller
    print("Building MPC controller...")
    controller = _build_mpc(
        model=model,
        dt=dt,
        horizon=horizon,
        lr=lr,
        max_epochs=max_epochs,
        use_ekf=use_ekf,
        use_simulink=use_simulink,
        optimize_trajectory=optimize_trajectory,
        max_accel=max_accel,
        control_weight=control_weight,
        tracking_weight=tracking_weight,
    )

    # Run optimization
    print("Running MPC optimization...")
    x0 = 0.0
    v0 = 0.0

    u_hist, histories = controller.solve(
        x_ref=x_ref,
        x0=x0,
        v0=v0,
        verbose=True,
    )

    # Extract histories
    optimized_trajectory = histories["references"]["filtered"].detach().cpu().numpy()
    x_ref_np = histories["references"]["steps"].detach().cpu().numpy()

    x_sim_np = histories["simulink"]["x"].detach().cpu().numpy()
    v_sim_np = histories["simulink"]["v"].detach().cpu().numpy()
    a_sim_np = histories["simulink"]["a"].detach().cpu().numpy()

    x_nn_np = histories["nn"]["x"].detach().cpu().numpy()
    v_nn_np = histories["nn"]["v"].detach().cpu().numpy()
    a_nn_np = histories["nn"]["a"].detach().cpu().numpy()

    u_hist_np = u_hist.detach().cpu().numpy()

    time_grid = np.arange(optimized_trajectory.shape[0], dtype=np.float32) * dt

    # Save figure to project root
    figure_path = Path(BASEDIR) / f"{model_name}_report_{trajectory}.png"

    print("Generating plot...")
    _plot_results(
        time_grid=time_grid,
        x_ref=x_ref_np,
        optimized_trajectory=optimized_trajectory,
        x_sim=x_sim_np,
        x_nn=x_nn_np,
        v_sim=v_sim_np,
        v_nn=v_nn_np,
        a_sim=a_sim_np,
        a_nn=a_nn_np,
        u_history=u_hist_np,
        figure_path=figure_path,
    )

    print("\n" + "=" * 60)
    print("REPORT GENERATED")
    print("=" * 60)
    print(f"  ✓ Figure saved: {figure_path}")

    return figure_path
