"""Training pipeline wrapper for CLI."""
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dronecontrol.globals import BASEDIR, INPUT_DATA, OUTPUT_DATA
from dronecontrol.data_process.data_cleaning import AVDataCleaner
from dronecontrol.data_process.data_loader import AVDataLoader


DEFAULTS = {
    "gru": dict(epochs=20, batch_size=32, lr=1e-2, hidden_dim=32, num_layers=2, dropout=0.0),
    "lstm": dict(epochs=20, batch_size=32, lr=1e-2, hidden_dim=32, num_layers=1, dropout=0.0),
    "rnn": dict(epochs=20, batch_size=32, lr=1e-2, hidden_dim=32, num_layers=2, dropout=0.0),
}


def _get_model_class(model_name: str):
    """Import and return model class based on model_name."""
    model_name = model_name.lower()
    if model_name == "gru":
        from dronecontrol.models.gru_module import GRU as ModelClass
    elif model_name == "lstm":
        from dronecontrol.models.lstm import LSTM as ModelClass
    elif model_name == "rnn":
        from dronecontrol.models.rnn import model_nn as ModelClass
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from: gru, lstm, rnn")
    return ModelClass


def run_training(
    model_name: str,
    best_model_out: str | None = None,
    seed: int = 42,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
):
    """Run training pipeline programmatically.

    Parameters
    ----------
    model_name : str
        Model type: 'gru', 'lstm', or 'rnn'
    best_model_out : str, optional
        Path to save best checkpoint. Default: {BASEDIR}/{model_name}_best.ckpt
    seed : int, default=42
        Random seed for reproducibility
    epochs : int, optional
        Training epochs (uses model-specific default if None)
    batch_size : int, optional
        Batch size (uses model-specific default if None)
    lr : float, optional
        Learning rate (uses model-specific default if None)
    hidden_dim : int, optional
        Hidden dimension (uses model-specific default if None)
    num_layers : int, optional
        Number of RNN layers (uses model-specific default if None)
    dropout : float, optional
        Dropout rate (uses model-specific default if None)

    Returns
    -------
    str
        Path to saved checkpoint directory
    """
    model_name = model_name.lower()
    defaults = DEFAULTS.get(model_name, DEFAULTS["gru"]).copy()
    
    # Apply defaults for missing parameters
    if epochs is None:
        epochs = defaults["epochs"]
    if batch_size is None:
        batch_size = defaults["batch_size"]
    if lr is None:
        lr = defaults["lr"]
    if hidden_dim is None:
        hidden_dim = defaults["hidden_dim"]
    if num_layers is None:
        num_layers = defaults["num_layers"]
    if dropout is None:
        dropout = defaults["dropout"]

    if best_model_out is None:
        best_model_out = str(Path(BASEDIR) / f"{model_name}_best.ckpt")

    print("=" * 60)
    print(f"TRAINING PIPELINE: {model_name.upper()}")
    print("=" * 60)
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}, Dropout: {dropout}")
    print(f"Output: {best_model_out}")
    print()

    # Set seed
    pl.seed_everything(seed)

    # Load and clean data
    print("[1/4] Loading data...")
    cleaner = AVDataCleaner(str(INPUT_DATA), str(OUTPUT_DATA))
    input_data, output_data = cleaner.get_clean_data()
    print(f"  ✓ Data cleaned: {input_data.shape[0]} samples")

    # Setup data module
    print("[2/4] Setting up DataModule...")
    data_module = AVDataLoader(
        input=input_data,
        target=output_data,
        batch_size=batch_size,
        val_split=0.2,
        test_split=0.1,
    )

    # Initialize model
    print("[3/4] Initializing model...")
    ModelClass = _get_model_class(model_name)
    model = ModelClass(
        input_dim=1,
        output_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )
    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup logger and callbacks
    logger = CSVLogger(save_dir=Path(BASEDIR) / "logs", name=model_name)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(BASEDIR),
        filename=f"{model_name}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    # Train
    print("[4/4] Training...")
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"  ✓ Best model: {checkpoint_callback.best_model_path}")
    print(f"  ✓ Logs: {logger.log_dir}")
    
    return checkpoint_callback.best_model_path
