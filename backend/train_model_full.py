# train_model_full.py
import os, sys, argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from models.deep_learning import IDSModel
from utils import load_artifacts, create_synthetic_data

def train_model(use_mock=False):
    log_dir = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    if use_mock:
        train_dataset, input_size, num_classes = create_synthetic_data()
        test_dataset = train_dataset
    else:
        df, artifacts = load_artifacts()
        if df is None or artifacts is None:
            print("[ERROR] Failed to load dataset or artifacts.")
            sys.exit(1)

        input_features = artifacts.get("feature_names", [])
        print(f"[INFO] Loading full dataset: models/preprocessed_dataset.csv")

        df = pd.read_csv("models/preprocessed_dataset.csv")

        # Drop columns with too many NaNs (e.g., >90%)
        high_nan_cols = df.columns[df.isna().mean() > 0.9].tolist()
        if high_nan_cols:
            print(f"[INFO] Dropping columns with >90% NaNs: {high_nan_cols}")
            df.drop(columns=high_nan_cols, inplace=True)

        required_cols = list(set(input_features + ['Label']) & set(df.columns))
        df = df.dropna(subset=required_cols)
        if df.empty:
            print("[ERROR] All rows dropped after NaN filtering.")
            sys.exit(1)

        df = df.fillna(0)

        X = df[input_features].values
        y = df['Label'].astype(int).values

        if len(np.unique(y)) < 2:
            print("[ERROR] Dataset lacks class diversity. Both classes are required.")
            sys.exit(1)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        print(f"[INFO] Total training samples: {len(X_train):,}")
        print(f"[INFO] Total testing samples: {len(X_test):,}")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        input_size = X_train_tensor.shape[1]
        num_classes = len(torch.unique(y_train_tensor))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDSModel(input_size=input_size, output_size=num_classes).to(device)

    # Class imbalance handling
    class_counts = torch.bincount(torch.tensor(y_train_tensor))
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
    parser = argparse.ArgumentParser(description="IDS Model Trainer (Full Dataset)")
    parser.add_argument("--use-mock", action="store_true", help="Use synthetic data for training")
    args = parser.parse_args()

    print("=== IDS Full Dataset Training Mode ===")
    train_model(use_mock=args.use_mock)
