"""Complete training pipeline with Lightning."""

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dronecontrol.data_process.data_cleaning import AVDataCleaner
from dronecontrol.data_process.data_loader import AVDataLoader
from dronecontrol.models.gru_module import GRU
from dronecontrol.models.lstm import LSTM
from dronecontrol.models.rnn import model_nn
from dronecontrol.globals import INPUT_DATA, OUTPUT_DATA, BASEDIR


def main():
    """Main training pipeline."""
    
    # Configuration
    seed = 42
    epochs = 20
    batch_size = 32
    lr = 1e-2
    val_split = 0.2
    test_split = 0.1
    hidden_dim = 8
    num_layers = 1
    dropout = 0.0
    log_dir = "logs"
    checkpoint_dir = "models_checkpoints"
    # Set seed for reproducibility
    pl.seed_everything(seed)
    
    print("=" * 60)
    print("DRONE CONTROL MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # ==================== DATA LOADING ====================
    print("\n[1/4] Loading data...")
    cleaner = AVDataCleaner(str(INPUT_DATA), str(OUTPUT_DATA))
    
    # ==================== DATA CLEANING ====================
    print("[2/4] Cleaning data...")
    input_data, output_data = cleaner.get_clean_data()
    print(f"  ✓ Data cleaned: {input_data.shape[0]} samples remaining")
    print(f"    Input shape: {input_data.shape}")
    print(f"    Output shape: {output_data.shape}")
    
    # ==================== DATA MODULE ====================
    print("[3/4] Setting up Lightning DataModule...")
    data_module = AVDataLoader(
        input=input_data,
        target=output_data,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split
    )
    
    # Get data dimensions
    
    print(f"  ✓ DataModule configured")
    print(f"    Batch size: {batch_size}")
    # ==================== MODEL ====================
    print("[4/4] Initializing GRU model...")
    
    model = GRU(
        input_dim=1,
        output_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )
    
    print(f"  ✓ GRU model initialized")
    print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== LOGGER & CALLBACKS ====================
    logger = CSVLogger(
        save_dir=BASEDIR / log_dir,
        name="accel_vs_voltage"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=BASEDIR / checkpoint_dir / "accel_vs_voltage",
        filename="gru-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )
    
    # ==================== TRAINER ====================
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"  ✓ Best model saved to: {checkpoint_callback.dirpath}")
    print(f"  ✓ Logs saved to: {logger.log_dir}")


if __name__ == "__main__":
    main()