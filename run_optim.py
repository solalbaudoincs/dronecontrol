"""Run the optimizer with a GRU model for drone control."""

import torch
import pytorch_lightning as pl
from pathlib import Path
import numpy as np


from dronecontrol.models.gru_module import GRU
from dronecontrol.commande.MPC_torch import MPCTorch
from dronecontrol.commande.MPC_cvxpy import MPCCVXPY


def main():
    """Run optimization with GRU model."""
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = 9.81
    dt = 0.1
    horizon = 30
    nb_steps = 300  # Number of optimization steps
    hidden_dim = 8
    num_layers = 1
    dropout = 0.0
    
    # Load trained GRU model (assuming checkpoint exists)
    # For now, create untrained model - in practice, load from checkpoint
    # gru_model = GRU(
    #     input_dim=1,
    #     output_dim=1,
    #     hidden_dim=hidden_dim,
    #     num_layers=num_layers,
    #     dropout=dropout,
    #     lr=1e-2
    # )
    
    # Note: In practice, load checkpoint like:
    # checkpoint_path = "path/to/gru-checkpoint.ckpt"
    gru_model = GRU.load_from_checkpoint(r"models/accel_vs_voltage/gru/epoch=120-val_loss=0.0313.ckpt")
    gru_model.to(device)
    
    # Set model to evaluation mode
    # Initialize optimizer

    optimizer = MPCTorch(
        accel_model=gru_model,
        dt=dt,
        Q=np.eye(horizon) * 10,  # Control effort weight
        R=np.eye(horizon) * 1.0,  # Tracking error weight
        lr=0.5,
        max_epochs=100,
        horizon=horizon,
        nb_steps=nb_steps,
        use_ekf=True,  # Set to True to use EKF
        use_simulink=True,
        optimizer_type="adam",
    )

    
    # Initial conditions
    x0 = 1.0  # Initial position
    v0 = 0.0  # Initial velocity
    
    # Reference trajectory (hover at initial position)
    x_ref = torch.linspace(0, 5, nb_steps//2).tolist() + torch.linspace(5, 5, nb_steps//2).tolist()
    x_ref = torch.tensor(x_ref, dtype=torch.float32)


    print("=" * 60)
    print("DRONE CONTROL OPTIMIZATION WITH GRU MODEL")
    print("=" * 60)
    print(f"Initial position: {x0}")
    print(f"Initial velocity: {v0}")
    print(f"Horizon: {horizon}")
    print(f"Number of steps: {nb_steps}")
    print(f"Using EKF: {optimizer.use_ekf}")
    
    # Run optimization
    print("\nStarting optimization...")
    u_opt, histories = optimizer.solve(
        x_ref=x_ref,
        x0=x0,
        v0=v0,
        verbose=True
    )
    
    print("\nOptimization completed!")
    print(f"Optimized control trajectory shape: {u_opt.shape}")
    print(f"Mean control: {u_opt.mean().item():.4f}")
    print(f"Std control: {u_opt.std().item():.4f}")
    
    # Save results
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)
    torch.save(u_opt, results_dir / "u_opt_gru.pt")
    
    print(f"\nResults saved to: {results_dir / 'u_opt_gru.pt'}")


if __name__ == "__main__":
    main()
