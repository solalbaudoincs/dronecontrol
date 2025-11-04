"""Run the optimizer with a GRU model for drone control."""

import torch
import pytorch_lightning as pl
from pathlib import Path


from dronecontrol.models.gru_module import GRU
from dronecontrol.commande.optimizer import Optimizer
from dronecontrol.simulink.simulator import DroneSimulator


def main():
    """Run optimization with GRU model."""
    
    # Configuration
    device = "cpu"
    g = 9.81
    dt = 0.1
    horizon = 20
    nb_steps = 10  # Number of optimization steps
    hidden_dim = 8
    num_layers = 1
    dropout = 0.0
    
    # Load trained GRU model (assuming checkpoint exists)
    # For now, create untrained model - in practice, load from checkpoint
    gru_model = GRU(
        input_dim=1,
        output_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=1e-2
    )
    
    # Note: In practice, load checkpoint like:
    # checkpoint_path = "path/to/gru-checkpoint.ckpt"
    # gru_model = GRU.load_from_checkpoint(checkpoint_path)
    
    # Set model to evaluation mode
    
    # Initialize optimizer
    optimizer = Optimizer(
        lr=1.0,
        accel_model=gru_model,
        dt=dt,
        max_iter=40,
        horizon=horizon,
        nb_steps=nb_steps,
        Q_tensor=torch.eye(horizon),
        R_tensor=torch.eye(horizon),
        max_epochs=80,
        use_ekf=False  # Set to True to use EKF
    )
    
    # Initial conditions
    x0 = torch.tensor([1.0])  # Initial position
    v0 = torch.tensor([0.0])  # Initial velocity
    
    # Reference trajectory (hover at initial position)
    x_ref = torch.tensor([5.0])
    
    print("=" * 60)
    print("DRONE CONTROL OPTIMIZATION WITH GRU MODEL")
    print("=" * 60)
    print(f"Initial position: {x0.item()}")
    print(f"Initial velocity: {v0.item()}")
    print(f"Horizon: {horizon}")
    print(f"Number of steps: {nb_steps}")
    print(f"Using EKF: {optimizer.use_ekf}")
    
    # Run optimization
    print("\nStarting optimization...")
    u_opt = optimizer.optimize(
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
