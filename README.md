# 🚁 Drone Control

Drone control project using recurrent neural networks (GRU, LSTM, RNN) with trajectory optimization and Extended Kalman Filtering.

## 📋 Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) - Ultra-fast Python package manager
- MATLAB (optional, for Simulink usage)

## 🚀 Installation

### 1. Install uv

```bash
python -m pip install uv
```

### 2. Clone the project

```bash
git clone https://github.com/solalbaudoincs/dronecontrol.git
cd dronecontrol
```

### 3. Create virtual environment and install dependencies

```bash
# uv automatically creates a venv and installs dependencies from pyproject.toml
uv sync
```


## 🎯 CLI Usage

The project provides a command-line interface (CLI) for model training and optimization report generation.

### ⚠️ All default parameters correspond to the parameters presented in our report.

### Available Commands

#### 1. Train a model

```bash
# Basic training with default parameters
uv run scripts/cli.py train --model-name gru

# Training with custom hyperparameters
uv run scripts/cli.py train \
    --model-name gru \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.005 \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.1
```

**Available options:**
- `--model-name`: Model type (`gru`, `lstm`, `rnn`) - **required**
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate
- `--hidden-dim`: Hidden layer dimension
- `--num-layers`: Number of RNN layers
- `--dropout`: Dropout rate
- `--seed`: Random seed (default: 42)
- `--best-model-out`: Path to save the best model

#### 2. Generate optimization report

```bash
# Report generation with default trajectory
uv run scripts/cli.py run \
    --model-name gru \
    --trajectory default

# Full generation with all parameters
uv run scripts/cli.py run \
    --model-name gru \
    --trajectory smooth \
    --use-ekf true \
    --use-simulink true \
    --optimize-trajectory true \
    --max-epochs 100 \
    --horizon 30 \
    --lr 0.1
```

**Available options:**
- `--model-name`: Model name - **required**
- `--trajectory`: Trajectory type (`default`, `step`, `multi`, `smooth`) - **required**
- `--model-ckpt`: Checkpoint path (default: `{model_name}_best.ckpt`)
- `--use-ekf`: Use Extended Kalman Filter (`true`/`false`, default: `true`)
- `--use-simulink`: Use Simulink for simulation (`true`/`false`, default: `true`)
- `--optimize-trajectory`: Enable trajectory optimization (`true`/`false`, default: `true`)
- `--max-epochs`: Maximum number of optimization epochs (default: 100)
- `--dt`: Time step (default: 0.05)
- `--horizon`: MPC horizon (default: 30)
- `--lr`: Optimizer learning rate (default: 0.1)
- `--max-speed`: Maximum speed (default: 9.81)

## 📊 Usage Examples

### Complete Workflow: Training + Optimization

```bash
# 1. Train a GRU model
uv run scripts/cli.py train --model-name gru --epochs 50

# 2. Generate optimization report with smooth trajectory
uv run scripts/cli.py run \
    --model-name gru \
    --trajectory smooth \
    --max-epochs 150
```

### Test different trajectories

```bash
# Default trajectory
uv run scripts/cli.py run --model-name gru --trajectory default

# Step trajectory
uv run scripts/cli.py run --model-name gru --trajectory step

# Multi trajectory
uv run scripts/cli.py run --model-name gru --trajectory multi

# Smooth trajectory
uv run scripts/cli.py run --model-name gru --trajectory smooth
```

### Compare different models

```bash
# Train all models
uv run scripts/cli.py train --model-name gru
uv run scripts/cli.py train --model-name lstm
uv run scripts/cli.py train --model-name rnn

# Generate reports
uv run scripts/cli.py run --model-name gru --trajectory default
uv run scripts/cli.py run --model-name lstm --trajectory default
uv run scripts/cli.py run --model-name rnn --trajectory default
```

## 📁 Project Structure

```
dronecontrol/
├── scripts/
│   └── cli.py              # Command-line interface
├── src/
│   └── dronecontrol/       # Main source code
├── data/                   # Training data
├── logs/                   # Training logs
├── models/                 # Trained models
├── optimization_results/   # Optimization results
├── predictions_plots/      # Generated plots
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 🛠️ Development

### Install development dependencies

```bash
uv pip install pytest
```

### Run tests

```bash
pytest tests/
```

## 📝 Notes

- Trained models are saved in the `models/` folder
- Intermediate checkpoints are in `models_checkpoints/`
- Reports and plots are generated in `predictions_plots/`
- Training logs are in `logs/`

## 🤝 Contributing

To contribute to the project:

1. Fork the project
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

