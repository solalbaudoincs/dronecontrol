"""Top-level CLI script for dronecontrol training and optimization.

Usage examples:
  
  Train a model:
    python scripts/cli.py train --model-name gru
    python scripts/cli.py train --model-name gru --epochs 30 --batch-size 64 --lr 0.005
  
  Generate optimization report:
    python scripts/cli.py run --model-name gru --use-ekf true --use-simulink true --optimize-trajectory true --max-epochs 100 --trajectory default
"""
import argparse
import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dronecontrol.cli import run_training, run_report
from dronecontrol.globals import BASEDIR


def str2bool(s: str) -> bool:
    """Convert string to boolean."""
    return str(s).lower() in ("1", "true", "yes", "y")


def main():
    parser = argparse.ArgumentParser(
        prog="dronecontrol-cli",
        description="Train models and generate optimization reports for drone control"
    )
    sub = parser.add_subparsers(dest="cmd", required=True, help="Command to execute")

    # ==================== TRAIN COMMAND ====================
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument(
        "--model-name",
        required=True,
        choices=["gru", "lstm", "rnn"],
        help="Model type to train"
    )
    p_train.add_argument(
        "--best-model-out",
        type=str,
        help=f"Path to save best model (default: {{BASEDIR}}/{{model_name}}_best.ckpt)"
    )
    
    # Training hyperparameters (optional - use model defaults if not specified)
    p_train.add_argument("--epochs", type=int, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, help="Batch size")
    p_train.add_argument("--lr", type=float, help="Learning rate")
    p_train.add_argument("--hidden-dim", type=int, help="Hidden dimension size")
    p_train.add_argument("--num-layers", type=int, help="Number of RNN layers")
    p_train.add_argument("--dropout", type=float, help="Dropout rate")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")

    # ==================== RUN COMMAND ====================
    p_run = sub.add_parser("run", help="Generate optimization report")
    p_run.add_argument(
        "--model-name",
        required=True,
        help="Model name (used for default checkpoint path)"
    )
    p_run.add_argument(
        "--model-ckpt",
        type=str,
        help=f"Checkpoint path (default: {{BASEDIR}}/{{model_name}}_best.ckpt)"
    )
    
    # Required optimization parameters
    p_run.add_argument(
        "--use-ekf",
        required=False,
        default="true",
        type=str,
        choices=["true", "false", "yes", "no", "1", "0"],
        help="Use Extended Kalman Filter"
    )
    p_run.add_argument(
        "--use-simulink",
        required=False,
        default="true",
        type=str,
        choices=["true", "false", "yes", "no", "1", "0"],
        help="Use Simulink for simulation"
    )
    p_run.add_argument(
        "--optimize-trajectory",
        required=False,
        default="true",
        type=str,
        choices=["true", "false", "yes", "no", "1", "0"],
        help="Enable trajectory optimization"
    )
    p_run.add_argument(
        "--max-epochs",
        required=False,
        default=100,
        type=int,
        help="Maximum optimization epochs"
    )
    p_run.add_argument(
        "--trajectory",
        required=True,
        choices=["default", "step", "multi", "smooth"],
        help="Predefined trajectory to use"
    )
    
    # Optional MPC parameters
    p_run.add_argument("--dt", type=float, default=0.05, help="Time step")
    p_run.add_argument("--horizon", type=int, default=30, help="MPC horizon")
    p_run.add_argument("--lr", type=float, default=0.1, help="Optimizer learning rate")
    p_run.add_argument("--max-speed", type=float, default=9.81, help="Maximum speed")

    args = parser.parse_args()

    try:
        if args.cmd == "train":
            print("\n🚁 Starting training pipeline...\n")
            
            result = run_training(
                model_name=args.model_name,
                best_model_out=args.best_model_out,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            
            print(f"\n✅ Training complete! Best model: {result}\n")

        elif args.cmd == "run":
            print("\n🚁 Starting optimization report generation...\n")
            
            # Convert string bools
            use_ekf = str2bool(args.use_ekf)
            use_simulink = str2bool(args.use_simulink)
            optimize_trajectory = str2bool(args.optimize_trajectory)

            result = run_report(
                model_name=args.model_name,
                model_ckpt=args.model_ckpt,
                use_ekf=use_ekf,
                use_simulink=use_simulink,
                optimize_trajectory=optimize_trajectory,
                max_epochs=args.max_epochs,
                trajectory=args.trajectory,
                dt=args.dt,
                horizon=args.horizon,
                lr=args.lr,
                max_accel=args.max_speed,
            )
            
            print(f"\n✅ Report complete! Figure: {result}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
