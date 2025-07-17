# train_model_chunked.py
import os, sys, argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
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
            print("[ERROR] Failed to load dataset or artifacts, cannot continue")
            sys.exit(1)

        chunk_size = artifacts.get("chunk_size", 100_000)
        input_features = artifacts.get("feature_names")

        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        chunk_stats = {"processed": 0, "used": 0, "skipped_empty": 0, "skipped_imbalance": 0}

        print(f"\n[INFO] Processing dataset in chunks (size: {chunk_size:,})")
        df_iterator = pd.read_csv("models/preprocessed_dataset.csv", chunksize=chunk_size)

        for i, chunk in enumerate(df_iterator, 1):
            chunk_stats["processed"] += 1
            print(f"[INFO] Processing chunk {i}...")

            high_nan_cols = chunk.columns[chunk.isna().mean() > 0.9].tolist()
            if high_nan_cols:
                print(f"[INFO] Dropping {len(high_nan_cols)} columns with >90% NaNs")
                chunk = chunk.drop(columns=high_nan_cols)

            required_cols = list(set(input_features + ['Label']) & set(chunk.columns))
            chunk = chunk.dropna(subset=required_cols)

            if chunk.empty:
                print("[WARNING] Skipping empty chunk after NaN drop.")
                chunk_stats["skipped_empty"] += 1
                continue

            chunk = chunk.fillna(0)
            X_chunk = chunk[input_features].values
            y_chunk = chunk['Label'].astype(int).values

            if len(np.unique(y_chunk)) < 2:
                print(f"[WARNING] Skipping chunk {i} due to lack of class diversity.")
                chunk_stats["skipped_imbalance"] += 1
                continue

            try:
                train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X_chunk, y_chunk))
            except ValueError as e:
                print(f"[WARNING] Stratified split failed on chunk {i}: {e}")
                chunk_stats["skipped_imbalance"] += 1
                continue

            X_train_list.append(torch.tensor(X_chunk[train_idx], dtype=torch.float32))
            y_train_list.append(torch.tensor(y_chunk[train_idx], dtype=torch.long))
            X_test_list.append(torch.tensor(X_chunk[test_idx], dtype=torch.float32))
            y_test_list.append(torch.tensor(y_chunk[test_idx], dtype=torch.long))
            chunk_stats["used"] += 1

        if not X_train_list:
            print("[ERROR] No valid chunks found. Cannot proceed with training.")
            print(chunk_stats)
            sys.exit(1)

        X_train_tensor = torch.cat(X_train_list)
        y_train_tensor = torch.cat(y_train_list)
        X_test_tensor = torch.cat(X_test_list)
        y_test_tensor = torch.cat(y_test_list)

        print(f"[INFO] Total training samples: {len(X_train_tensor):,}")
        print(f"[INFO] Total testing samples: {len(X_test_tensor):,}")
        print(f"[INFO] Chunk Summary: {chunk_stats}")

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        input_size = X_train_tensor.shape[1]
        num_classes = len(torch.unique(y_train_tensor))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDSModel(input_size=input_size, output_size=num_classes).to(device)

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
    parser = argparse.ArgumentParser(description="IDS Model Trainer (Chunked)")
    parser.add_argument("--use-mock", action="store_true", help="Use synthetic data for training")
    args = parser.parse_args()

    print("=== IDS Chunked Training Mode ===")
    train_model(use_mock=args.use_mock)
