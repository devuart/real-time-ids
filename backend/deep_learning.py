import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import sys
import warnings
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import onnx
import onnxruntime as ort
import optuna
from typing import Optional, Dict, Tuple, Union
import logging
import pandas as pd
import optuna.visualization as vis
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deep_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MIN_FEATURES = 5  # Minimum number of features for model

class EnhancedAutoencoder(nn.Module):
    """Improved autoencoder with batch normalization and skip connections."""
    def __init__(self, input_dim: int, encoding_dim: int = 10):
        super().__init__()
        if input_dim < MIN_FEATURES:
            raise ValueError(f"Input dimension must be at least {MIN_FEATURES}")
            
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.Tanh()  # Constrain encoding space
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, input_dim)
        )
        
        # Skip connection
        self.skip = nn.Linear(input_dim, input_dim) if input_dim > 10 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.skip is not None:
            decoded = decoded + self.skip(x)
        return decoded

def check_hardware() -> Dict[str, Union[str, bool, int]]:
    """Check and report available hardware resources with validation."""
    try:
        return {
            "pytorch_version": torch.__version__,
            "gpu_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_threads": torch.get_num_threads()
        }
    except Exception as e:
        logger.error(f"Hardware check failed: {str(e)}")
        return {
            "pytorch_version": "unknown",
            "gpu_available": False,
            "device": "cpu",
            "gpu_count": 0,
            "cpu_threads": 1
        }

def load_and_validate_data() -> Dict[str, np.ndarray]:
    """Load and validate preprocessed data with comprehensive checks."""
    try:
        df = pd.read_csv("models/preprocessed_dataset.csv")
        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
        
        # Validate data
        if "Label" not in df.columns:
            raise ValueError("Dataset missing 'Label' column")
            
        feature_names = artifacts.get("feature_names", [])
        if not feature_names:
            raise ValueError("No feature names found in artifacts")
            
        if len(feature_names) < MIN_FEATURES:
            raise ValueError(f"Too few features ({len(feature_names)}), need at least {MIN_FEATURES}")
            
        # Apply scaling if available
        scaler = artifacts.get("scaler")
        X = df[feature_names].values.astype(np.float32)
        if scaler:
            X = scaler.transform(X)
        
        # Split data
        normal_mask = df["Label"] == 0
        X_normal = X[normal_mask]
        X_attack = X[~normal_mask]
        
        # Balance dataset
        min_samples = min(len(X_normal), len(X_attack))
        if min_samples == 0:
            raise ValueError("One class has zero samples")
            
        return {
            "X_train": X_normal[:min_samples],
            "X_val": X_normal[min_samples:min_samples + min_samples//4],  # 25% for validation
            "X_test": X_attack[:min_samples],
            "feature_names": feature_names
        }
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def generate_synthetic_data(
    normal_samples: int = 8000,
    attack_samples: int = 2000,
    features: int = 20,
    anomaly_factor: float = 1.5,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """Generate realistic synthetic data with proper anomaly separation."""
    np.random.seed(random_state)
    
    # Generate normal data (clustered around 0.5)
    X_normal = np.random.normal(0.5, 0.1, (normal_samples, features))
    X_normal = np.clip(X_normal, 0.1, 0.9)
    
    # Generate anomalies (higher variance and shifted means)
    X_attack = np.random.normal(0.5, 0.3, (attack_samples//2, features)) * anomaly_factor
    X_attack = np.clip(X_attack, 0.1, 0.9)
    
    # Add some extreme anomalies
    X_extreme = np.random.uniform(0, 1, (attack_samples//2, features))
    X_extreme[:, ::2] *= anomaly_factor  # Make some features more extreme
    
    return {
        "X_train": X_normal,
        "X_val": X_normal[:normal_samples//5],  # 20% for validation
        "X_test": np.vstack([X_attack, X_extreme]),
        "feature_names": [f"feature_{i}" for i in range(features)]
    }

def create_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int = 64,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create properly configured dataloaders with pinned memory."""
    train_data = TensorDataset(torch.tensor(data["X_train"], dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(data["X_val"], dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(data["X_test"], dtype=torch.float32))
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1)
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 2,  # Larger batches for validation
        shuffle=False,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    """Train model for one epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        inputs = batch[0].to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray]:
    """Validate model and return MSE values for threshold calculation."""
    model.eval()
    total_loss = 0.0
    all_mse = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
            
            # Calculate per-sample MSE
            mse = torch.mean((inputs - outputs)**2, dim=1).cpu().numpy()
            all_mse.extend(mse)
    
    return total_loss / len(loader), np.array(all_mse)

def calculate_threshold(
    model: nn.Module,
    loader: DataLoader,
    percentile: int,
    device: torch.device
) -> float:
    """Calculate anomaly threshold based on reconstruction error."""
    model.eval()
    mse_values = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            mse = torch.mean((inputs - outputs)**2, dim=1).cpu().numpy()
            mse_values.extend(mse)
    
    if not mse_values:
        raise ValueError("No MSE values calculated for threshold")
    
    return np.percentile(mse_values, percentile)

def export_to_onnx(
    model: nn.Module,
    input_dim: int,
    model_dir: Path,
    device: torch.device
) -> None:
    """Export model to ONNX format with proper configuration."""
    model.eval()
    dummy_input = torch.randn(1, input_dim).to(device)
    onnx_path = model_dir / "autoencoder_ids.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        do_constant_folding=True,
        export_params=True
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX model exported and validated: {onnx_path}")

def train_model(args) -> Dict[str, float]:
    """Main training pipeline with comprehensive features."""
    # Setup logging and hardware
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_DIR / f"train_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    hw = check_hardware()
    logger.info("\nHardware Configuration:")
    for k, v in hw.items():
        logger.info(f"{k}: {v}")
    
    device = torch.device(hw["device"])
    
    try:
        # Data preparation
        logger.info("\nPreparing data...")
        if args.use_real_data:
            try:
                data = load_and_validate_data()
                logger.info("Using real preprocessed data")
            except Exception as e:
                logger.warning(f"Failed to load real data: {str(e)}")
                if args.non_interactive or prompt_user("Use synthetic data instead?", True):
                    data = generate_synthetic_data(
                        normal_samples=args.normal_samples,
                        attack_samples=args.attack_samples,
                        features=args.features
                    )
                    logger.info("Using synthetic data as fallback")
                else:
                    raise
        else:
            data = generate_synthetic_data(
                normal_samples=args.normal_samples,
                attack_samples=args.attack_samples,
                features=args.features
            )
            logger.info("Using synthetic data by request")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data,
            batch_size=args.batch_size
        )
        
        # Model initialization
        model = EnhancedAutoencoder(
            input_dim=args.features,
            encoding_dim=args.encoding_dim
        ).to(device)
        
        # Optimizer and loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=2,
            factor=0.5,
            verbose=True
        )
        criterion = nn.MSELoss()
        
        # Training loop
        logger.info(f"\nTraining for {args.epochs} epochs...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Training
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                grad_clip=1.0
            )
            
            # Validation
            val_loss, val_mse = validate(
                model,
                val_loader,
                criterion,
                device
            )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Logging
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Early stopping and model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), args.model_dir / "best_model.pth")
                logger.debug("New best model saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(args.model_dir / "best_model.pth"))
        
        # Calculate threshold on validation data
        threshold = calculate_threshold(
            model,
            val_loader,
            args.percentile,
            device
        )
        logger.info(f"\nAnomaly threshold (P{args.percentile}): {threshold:.6f}")
        
        # Test set evaluation
        test_loss, test_mse = validate(model, test_loader, criterion, device)
        anomaly_rate = (test_mse > threshold).mean()
        logger.info(f"Test Loss: {test_loss:.4f} | Anomaly Detection Rate: {anomaly_rate:.2%}")
        
        # Save artifacts
        args.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_dir / "autoencoder_ids.pth")
        joblib.dump(threshold, args.model_dir / "anomaly_threshold.pkl")
        
        # Export to ONNX if requested
        if args.export_onnx:
            export_to_onnx(model, args.features, args.model_dir, device)
        
        # Save training metadata
        training_meta = {
            "input_dim": args.features,
            "encoding_dim": args.encoding_dim,
            "threshold": float(threshold),
            "percentile": args.percentile,
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_loss),
            "anomaly_detection_rate": float(anomaly_rate),
            "epochs_trained": epoch + 1,
            "hardware": hw,
            "timestamp": timestamp
        }
        
        with open(args.model_dir / "training_metadata.json", "w") as f:
            json.dump(training_meta, f, indent=2)
        
        logger.info("\nTraining complete!")
        logger.info(f"Model saved to: {args.model_dir / 'autoencoder_ids.pth'}")
        logger.info(f"Threshold saved to: {args.model_dir / 'anomaly_threshold.pkl'}")
        
        return training_meta
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        writer.close()

def hyperparameter_search(trial: optuna.Trial) -> float:
    """Optuna hyperparameter optimization objective with enhanced search space."""
    params = {
        "encoding_dim": trial.suggest_int("encoding_dim", 5, min(50, args.features//2)),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "percentile": trial.suggest_int("percentile", 90, 99),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5)
    }
    
    # Create temporary args with suggested parameters
    search_args = argparse.Namespace(
        **vars(args),
        **params,
        model_dir=DEFAULT_MODEL_DIR / f"hpo_trial_{trial.number}",
        epochs=30,  # Slightly longer trials for better evaluation
        patience=3,
        non_interactive=True,
        export_onnx=False  # Don't export during HPO
    )
    
    try:
        # Run training with these parameters
        train_model(search_args)
        
        # Load the validation metrics
        with open(search_args.model_dir / "training_metadata.json") as f:
            config = json.load(f)
            
        # Clean up trial directory
        if not args.debug:
            for f in search_args.model_dir.glob("*"):
                f.unlink()
            search_args.model_dir.rmdir()
            
        return config["best_val_loss"]
        
    except Exception as e:
        logger.error(f"Trial failed: {str(e)}")
        return float('inf')  # Return worst possible score

def setup_hyperparameter_optimization(args) -> None:
    """Configure and run hyperparameter optimization."""
    logger.info(f"\n=== Starting Hyperparameter Optimization ({args.hpo_trials} trials) ===")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup=5)
    )
    
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    study.optimize(
        hyperparameter_search,
        n_trials=args.hpo_trials,
        timeout=args.hpo_timeout if args.hpo_timeout > 0 else None,
        gc_after_trial=True
    )
    
    logger.info("\nBest trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value:.5f}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Visualize optimization
    try:
        fig = vis.plot_optimization_history(study)
        fig.show()
    except ImportError:
        logger.warning("Could not import optuna.visualization - skipping plots")
    
    return study.best_params

def prompt_user(prompt: str, default: bool = True) -> bool:
    """Interactive user prompt with default handling."""
    while True:
        response = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not response:
            return default
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please answer yes/y or no/n")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Anomaly Detection Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=20,
        help="Number of input features"
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=10,
        help="Encoder hidden dimension"
    )
    parser.add_argument(
        "--normal-samples",
        type=int,
        default=8000,
        help="Normal training samples (synthetic)"
    )
    parser.add_argument(
        "--attack-samples",
        type=int,
        default=2000,
        help="Anomalous test samples (synthetic)"
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=95,
        help="Percentile for anomaly threshold"
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use preprocessed data instead of synthetic"
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export model to ONNX format"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable all interactive prompts"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    # Add HPO-specific arguments
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Enable hyperparameter optimization"
    )
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=0,
        help="Number of hyperparameter optimization trials"
    )
    parser.add_argument(
        "--hpo-timeout",
        type=int,
        default=3600,
        help="Timeout for hyperparameter optimization in seconds (0 for no timeout)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("torch").setLevel(logging.DEBUG)
    
    # Run hyperparameter optimization if requested
    if args.hpo_trials > 0:
        best_params = setup_hyperparameter_optimization(args)
        
        # Update args with best parameters
        for key, value in best_params.items():
            setattr(args, key, value)
            
        logger.info("\nTraining final model with best parameters...")
    
    # Run training
    try:
        train_model(args)
    except Exception as e:
        logger.error(f"Fatal error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    LOG_DIR.mkdir(exist_ok=True)
    main()