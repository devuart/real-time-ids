import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import subprocess
import sys
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

def check_preprocessing_outputs() -> bool:
    """Verify all required preprocessing outputs exist."""
    required_files = [
        "models/preprocessed_dataset.csv",
        "models/preprocessing_artifacts.pkl"
    ]
    return all(Path(f).exists() for f in required_files)

def run_preprocessing() -> bool:
    """Execute preprocessing.py and verify outputs."""
    print("\n[INFO] Running preprocessing pipeline...")
    try:
        result = subprocess.run(
            [sys.executable, "preprocessing.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return check_preprocessing_outputs()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Preprocessing failed: {e.stderr}")
        return False

def load_artifacts() -> dict:
    if not check_preprocessing_outputs():
        print("\n[WARNING] Required preprocessing outputs not found!")
        print("The following files are required:")
        print("- models/preprocessed_dataset.csv")
        print("- models/preprocessing_artifacts.pkl")
        
        if prompt_user("Run preprocessing now?", default=True):
            if not run_preprocessing():
                print("[ERROR] Preprocessing failed, cannot continue")
                sys.exit(1)
        else:
            print("\n[WARNING] Using minimal synthetic data as fallback")
            return create_synthetic_data()
    
    try:
        # Load actual preprocessed data
        df = pd.read_csv("models/preprocessed_dataset.csv")
        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
        
        # Data sanity checks
        print("\n[INFO] Running data sanity checks...")
        print(f"Total samples: {len(df)}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print(f"NaN values: {df.isna().sum().sum()}")
        print("Class distribution:")
        print(df['Label'].value_counts())
        
        return df, artifacts
    except Exception as e:
        print(f"[ERROR] Data loading failed: {str(e)}")
        sys.exit(1)

def create_synthetic_data():
    """Fallback synthetic data generator."""
    print("[INFO] Generating minimal synthetic dataset (100 samples)")
    X = torch.randn(100, 10)  # 10 features
    y = torch.randint(0, 2, (100,))  # Binary classification
    return (
        pd.DataFrame(X.numpy(), columns=[f"feature_{i}" for i in range(10)]).assign(Label=y.numpy()),
        {"feature_names": [f"feature_{i}" for i in range(10)], "scaler": None}
    )

class IDSModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def train_model(use_mock=False):
    """Main training pipeline."""
    log_dir = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    if use_mock:
        train_dataset, input_size, num_classes = create_synthetic_data()
        test_dataset = train_dataset
    else:
        df, artifacts = load_artifacts()
        if df is None or artifacts is None:
            print("[ERROR] Failed to load dataset or artifacts, cannot continue")
            sys.exit(1)

        chunk_size = artifacts.get("chunk_size", 100000)
        input_features = artifacts.get("feature_names")

        # Chunked stratified split accumulation
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        print(f"\n[INFO] Processing dataset in chunks (size: {chunk_size:,})")
        df_iterator = pd.read_csv("models/preprocessed_dataset.csv", chunksize=chunk_size)

        for i, chunk in enumerate(df_iterator, 1):
            print(f"[INFO] Processing chunk {i}...")

            # Drop rows with missing features or labels
            chunk = chunk.dropna(subset=input_features + ['Label'])
            if chunk.empty:
                print("[WARNING] Skipping empty chunk after NaN drop.")
                continue

            X_chunk = chunk[input_features].values
            y_chunk = chunk['Label'].astype(int).values

            # Skip chunk if it has only one class
            if len(np.unique(y_chunk)) < 2:
                print(f"[WARNING] Skipping chunk {i} due to lack of class diversity.")
                continue

            # Stratified split
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            try:
                train_idx, test_idx = next(splitter.split(X_chunk, y_chunk))
            except ValueError as e:
                print(f"[WARNING] Stratified split failed on chunk {i}: {e}")
                continue

            X_train_list.append(torch.tensor(X_chunk[train_idx], dtype=torch.float32))
            y_train_list.append(torch.tensor(y_chunk[train_idx], dtype=torch.long))
            X_test_list.append(torch.tensor(X_chunk[test_idx], dtype=torch.float32))
            y_test_list.append(torch.tensor(y_chunk[test_idx], dtype=torch.long))

        # Final tensors
        X_train_tensor = torch.cat(X_train_list)
        y_train_tensor = torch.cat(y_train_list)
        X_test_tensor = torch.cat(X_test_list)
        y_test_tensor = torch.cat(y_test_list)

        print(f"[INFO] Total training samples: {len(X_train_tensor):,}")
        print(f"[INFO] Total testing samples: {len(X_test_tensor):,}")

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        input_size = X_train_tensor.shape[1]
        num_classes = len(torch.unique(y_train_tensor))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDSModel(input_size=input_size, output_size=num_classes).to(device)

    # Handle class imbalance
    class_counts = torch.bincount(y_train_tensor)
    class_weights = 1. / class_counts.float()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    print("\n[INFO] Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        val_loss /= len(test_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"[INFO] New best model saved with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()

    print("\n[INFO] Final Model Evaluation:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    torch.save(model.state_dict(), "models/ids_model.pth")
    print("\n[SUCCESS] Training complete. Best model saved to models/ids_model.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS Model Trainer")
    parser.add_argument("--use-mock", action="store_true", help="Use synthetic data for training")
    args = parser.parse_args()

    print("=== Enhanced Network Intrusion Detection Model Trainer ===")
    train_model(use_mock=args.use_mock)
