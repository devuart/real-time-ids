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
from typing import Optional, Dict, Tuple

# Constants
DEFAULT_MODEL_DIR = Path("models")
LOG_DIR = Path("logs")

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

def check_hardware() -> dict:
    """Check and report available hardware resources."""
    return {
        "pytorch_version": torch.__version__,
        "gpu_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_threads": torch.get_num_threads()
    }

def load_preprocessed_data() -> Optional[Dict[str, np.ndarray]]:
    """Load preprocessed data from preprocessing.py outputs."""
    try:
        df = pd.read_csv("models/preprocessed_dataset.csv")
        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
        
        X = df.drop(columns=["Label"]).values.astype(np.float32)
        y = df["Label"].values
        
        # Split into normal (0) and attack (1)
        X_normal = X[y == 0]
        X_attack = X[y == 1]
        
        return {
            "X_train": X_normal,
            "X_test": np.vstack([X_normal[:len(X_attack)], X_attack])
        }
    except Exception as e:
        print(f"[WARNING] Could not load preprocessed data: {str(e)}")
        return None

def generate_synthetic_data(normal_samples: int = 8000, 
                          attack_samples: int = 2000,
                          features: int = 20,
                          anomaly_factor: float = 1.5) -> dict:
    """Generate synthetic training and test data."""
    np.random.seed(42)
    return {
        "X_train": np.random.rand(normal_samples, features).astype(np.float32),
        "X_test": np.vstack([
            np.random.rand(attack_samples//2, features),
            np.random.rand(attack_samples//2, features) * anomaly_factor
        ]).astype(np.float32)
    }

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def train_model(args):
    """Main training pipeline with all enhancements."""
    print("\n=== Anomaly Detection Model Training ===")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_DIR / f"train_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Hardware setup
    hw = check_hardware()
    print(f"\n[INFO] Hardware Configuration:")
    print(f"1. PyTorch: {hw['pytorch_version']}")
    print(f"2. Device: {hw['device'].upper()}")
    print(f"3. CPU Threads: {hw['cpu_threads']}")
    
    # Data preparation
    print("\n[INFO] Preparing data...")
    if args.use_real_data:
        data = load_preprocessed_data()
        if data is None and not args.non_interactive:
            if prompt_user("Use synthetic data instead?", default=True):
                data = generate_synthetic_data(
                    normal_samples=args.normal_samples,
                    attack_samples=args.attack_samples,
                    features=args.features
                )
            else:
                sys.exit(1)
        elif data is None:
            print("[INFO] Falling back to synthetic data")
            data = generate_synthetic_data(
                normal_samples=args.normal_samples,
                attack_samples=args.attack_samples,
                features=args.features
            )
    else:
        data = generate_synthetic_data(
            normal_samples=args.normal_samples,
            attack_samples=args.attack_samples,
            features=args.features
        )
    
    # Convert to tensors and loaders
    train_data = TensorDataset(torch.tensor(data["X_train"]))
    val_data = TensorDataset(torch.tensor(data["X_train"][:len(data["X_train"])//5]))  # 20% for validation
    test_data = TensorDataset(torch.tensor(data["X_test"]))
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    # Model setup
    device = torch.device(hw["device"])
    model = Autoencoder(
        input_dim=args.features,
        encoding_dim=args.encoding_dim
    ).to(device)
    
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop with early stopping
    print(f"\n[INFO] Training for maximum {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, inputs).item()
        
        # Logging
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(args.model_dir / "best_model.pth"))
    
    # Threshold calculation
    print("\n[INFO] Computing anomaly threshold...")
    model.eval()
    with torch.no_grad():
        sample = train_data.tensors[0][:1000].to(device)
        reconstructions = model(sample)
        mse = torch.mean((sample - reconstructions)**2, dim=1).cpu().numpy()
        threshold = np.percentile(mse, args.percentile)
    
    # Save artifacts
    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_dir / "autoencoder_ids.pth")
    joblib.dump(threshold, args.model_dir / "anomaly_threshold.pkl")
    
    # Export to ONNX if requested
    if args.export_onnx:
        print("\n[INFO] Exporting to ONNX format...")
        dummy_input = torch.randn(1, args.features).to(device)
        onnx_path = args.model_dir / "autoencoder_ids.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"[SUCCESS] ONNX model saved to: {onnx_path}")
    
    # Save configuration
    config = {
        "input_dim": args.features,
        "encoding_dim": args.encoding_dim,
        "threshold_percentile": args.percentile,
        "threshold_value": float(threshold),
        "normal_samples": len(data["X_train"]),
        "attack_samples": len(data["X_test"]),
        "best_val_loss": float(best_val_loss),
        "final_epoch": epoch + 1,
        "used_real_data": args.use_real_data and data is not None
    }
    with open(args.model_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n[SUCCESS] Training complete!")
    print(f"1. Model saved to: {args.model_dir / 'autoencoder_ids.pth'}")
    print(f"2. Best validation loss: {best_val_loss:.6f}")
    print(f"3. Anomaly threshold (P{args.percentile}): {threshold:.6f}")
    print(f"4. TensorBoard logs: {log_dir}")

def hyperparameter_search(trial: optuna.Trial) -> float:
    """Optuna hyperparameter optimization objective."""
    params = {
        "encoding_dim": trial.suggest_int("encoding_dim", 5, 20),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "percentile": trial.suggest_int("percentile", 90, 99)
    }
    
    # Create temporary args with suggested parameters
    search_args = argparse.Namespace(
        **vars(args),
        **params,
        model_dir=DEFAULT_MODEL_DIR / "hpo_trial",
        epochs=20,  # Shorter trials for HPO
        patience=5,
        non_interactive=True
    )
    
    train_model(search_args)
    
    # Return validation loss to minimize
    with open(search_args.model_dir / "training_config.json") as f:
        config = json.load(f)
    return config["best_val_loss"]

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Model Training",
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
        default=50,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        "--hpo-trials",
        type=int,
        default=0,
        help="Number of hyperparameter optimization trials (0 to disable)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable all interactive prompts"
    )
    
    args = parser.parse_args()
    
    # Configure environment
    warnings.filterwarnings("once")
    LOG_DIR.mkdir(exist_ok=True)
    
    # Run hyperparameter optimization if requested
    if args.hpo_trials > 0:
        print(f"\n=== Starting Hyperparameter Optimization ({args.hpo_trials} trials) ===")
        study = optuna.create_study(direction="minimize")
        study.optimize(hyperparameter_search, n_trials=args.hpo_trials)
        
        print("\n[INFO] Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
        
        # Train final model with best params
        args.encoding_dim = study.best_params["encoding_dim"]
        args.lr = study.best_params["lr"]
        args.batch_size = study.best_params["batch_size"]
        args.percentile = study.best_params["percentile"]
    
    # Run training
    train_model(args)