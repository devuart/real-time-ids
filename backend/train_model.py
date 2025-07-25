# Standard library imports
import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from colorama import Fore, Style, init
import re
import traceback

# Third-party imports
import numpy as np
import pandas as pd
import pkg_resources

# Machine learning and deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.serialization

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning

# Imbalanced learning imports
import imblearn
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import (
    CondensedNearestNeighbour,
    NearMiss,
    RandomUnderSampler,
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Serialization
import joblib
import pickle
import shutil

# Initialize colorama
init(autoreset=True)

# Initialize logger at module level
logger = logging.getLogger(__name__)

# System and environment configuration
def configure_system() -> None:
    """Configure system settings for optimal performance."""
    # Disable verbose logging for libraries
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow
    os.environ['KMP_WARNINGS'] = '0'  # Intel MKL
    os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP
    
    # Set NumPy print options
    np.set_printoptions(precision=4, suppress=True, threshold=10, linewidth=120)
    
    # Configure Python warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.simplefilter('ignore', category=RuntimeWarning)
    warnings.simplefilter('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Visualization configuration
def configure_visualization() -> None:
    """Configure visualization settings."""
    try:
        plt.style.use('seaborn-v0_8')
        # Set appropriate backend
        if os.environ.get('DISPLAY', '') == '':
            matplotlib.use('Agg')  # For headless environments
    except:
        plt.style.use('ggplot')  # Fallback style
    
    # Set global plot parameters
    rcParams['figure.figsize'] = (12, 8)
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 16
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.transparent'] = False
    rcParams['font.family'] = 'DejaVu Sans'  # Better font compatibility

# Reproducibility configuration
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False  # Set to True if input sizes don't vary
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA reproducibility

class UnicodeStreamHandler(logging.StreamHandler):
    """A stream handler that properly handles Unicode characters on Windows."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Fallback to ASCII-only output if Unicode fails
            try:
                msg = record.getMessage().encode('ascii', 'replace').decode('ascii')
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure comprehensive logging system with clean console output."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler (verbose, with timestamps)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler (clean output for info, normal for warnings/errors)
    console_handler = UnicodeStreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # Show info and above
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_directories(logger: logging.Logger) -> Dict[str, Path]:
    """Create and return essential directories with versioned subdirectories."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path().absolute()
    
    directories = {
        'models': base_dir / "models" / timestamp,
        'logs': base_dir / "logs" / timestamp,
        'data': base_dir / "data",
        'figures': base_dir / "figures" / timestamp,
        'tensorboard': base_dir / "runs" / timestamp,
        'checkpoints': base_dir / "checkpoints" / timestamp
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Handle latest pointer
    latest_dir = base_dir / "latest"
    latest_file = base_dir / "latest.txt"
    
    try:
        # Clean up existing latest pointers
        if latest_dir.exists():
            if latest_dir.is_symlink():
                latest_dir.unlink()
            elif latest_dir.is_dir():
                shutil.rmtree(latest_dir)
        if latest_file.exists():
            latest_file.unlink()
            
        # Try creating symlink first
        if platform.system() != 'Windows':
            latest_dir.symlink_to(timestamp, target_is_directory=True)
        else:
            # Windows: Create junction instead of symlink
            subprocess.run(
                ['cmd', '/c', 'mklink', '/J', str(latest_dir), str(directories['logs'])], 
                check=True,
                capture_output=True
            )
    except (OSError, subprocess.CalledProcessError) as e:
        logger.debug(f"Could not create directory junction: {str(e)}")
        try:
            # Fallback: Copy directory structure
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            shutil.copytree(directories['logs'], latest_dir)
            
            # Also maintain a text pointer
            with open(latest_file, 'w') as f:
                f.write(timestamp)
        except Exception as e:
            logger.debug(f"Fallback directory creation failed: {str(e)}")
            # Final fallback: Just write timestamp
            try:
                with open(latest_file, 'w') as f:
                    f.write(timestamp)
            except Exception as e:
                logger.warning(f"All directory pointer methods failed: {str(e)}")
    
    return directories

def setup_gpu(logger: logging.Logger) -> torch.device:
    """Configure GPU settings and return appropriate device with detailed info."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"Using GPU: {gpu_props.name}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {gpu_props.total_memory/1e9:.2f} GB")
        logger.info(f"GPU Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        logger.info(f"GPU Multiprocessors: {gpu_props.multi_processor_count}")
        
        torch.cuda.set_per_process_memory_fraction(0.9)
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
        torch.set_num_threads(os.cpu_count() or 1)
        logger.info(f"Using {torch.get_num_threads()} CPU threads")
    
    return device

def check_versions(logger: logging.Logger) -> bool:
    """Verify package versions with clean console output and full logging."""
    requirements = {
        'torch': '1.10.0',
        'torchvision': '0.11.0',
        'scikit-learn': '1.0.0',
        'imbalanced-learn': '0.9.0',
        'pandas': '1.3.0',
        'numpy': '1.21.0',
        'matplotlib': '3.5.0',
        'seaborn': '0.11.0'
    }

    # Collect version data
    version_data = []
    for pkg, min_ver in requirements.items():
        try:
            current_ver = pkg_resources.get_distribution(pkg).version
            is_ok = (pkg_resources.parse_version(current_ver) >= 
                    pkg_resources.parse_version(min_ver))
            version_data.append((pkg, current_ver, min_ver, is_ok, None))
            if not is_ok:
                logger.warning(f"Version mismatch: {pkg} {current_ver} (needs >= {min_ver})")
        except Exception as e:
            version_data.append((pkg, None, min_ver, False, str(e)))
            logger.warning(f"Package not found: {pkg} - {str(e)}")

    # Calculate column widths
    max_pkg_len = max(len(item[0]) for item in version_data)
    max_ver_len = max(len(item[1]) if item[1] else 0 for item in version_data)

    # Build clean output
    output_lines = ["=== Package Versions ==="]
    for pkg, current_ver, min_ver, is_ok, _ in version_data:
        if current_ver:
            status = "[OK]" if is_ok else f"[FAIL] (needs >= {min_ver})"
            line = f"{pkg:{max_pkg_len}} {current_ver:{max_ver_len}} (>= {min_ver}) {status}"
        else:
            line = f"{pkg:{max_pkg_len}} {'N/A':{max_ver_len}} (>= {min_ver}) [MISSING]"
        output_lines.append(line)

    # Print clean output to console
    print("\n".join(output_lines))
    
    # Log the complete version info at DEBUG level
    logger.debug("\n".join(output_lines))

    return all(item[3] for item in version_data)

# Training constants
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = True

# Model architecture constants
HIDDEN_LAYER_SIZES = [512, 256, 128, 64]
DROPOUT_RATES = [0.5, 0.4, 0.3, 0.2]
ACTIVATION = 'leaky_relu'
ACTIVATION_PARAM = 0.1
USE_BATCH_NORM = True
USE_LAYER_NORM = False

# Main configuration and setup function
configure_system()
configure_visualization()
set_seed(42)

# Setup logging and directories
LOG_DIR = Path("logs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logging(LOG_DIR)

# Setup directories for models, logs, data, etc.
try:
    directories = setup_directories(logger)
    MODEL_DIR = directories['models']
    LOG_DIR = directories['logs']
    DATA_DIR = directories['data']
    FIGURE_DIR = directories['figures']
    TB_DIR = directories['tensorboard']
    CHECKPOINT_DIR = directories['checkpoints']
except Exception as e:
    logger.error(f"Failed to set up directories: {str(e)}")
    sys.exit(1)

# Check package versions
if not check_versions(logger):
    logger.error("Some package requirements not met!")

# Setup GPU or CPU
device = setup_gpu(logger)

# Model architecture
class IDSModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """Enhanced IDS model with flexible architecture and normalization options."""
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for i, (size, dropout) in enumerate(zip(HIDDEN_LAYER_SIZES, DROPOUT_RATES)):
            layers.append(nn.Linear(prev_size, size))
            
            # Add normalization
            if USE_BATCH_NORM:
                layers.append(nn.BatchNorm1d(size))
            elif USE_LAYER_NORM:
                layers.append(nn.LayerNorm(size))
            
            # Add activation
            if ACTIVATION == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=ACTIVATION_PARAM))
            elif ACTIVATION == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            
            # Add dropout
            layers.append(nn.Dropout(dropout))
            prev_size = size
        
        # Add final layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Kaiming normal and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Data preprocessing and validation
def check_preprocessing_outputs() -> bool:
    """Verify all required preprocessing outputs exist."""
    required_files = [
        "models/preprocessed_dataset.csv",
        "models/preprocessing_artifacts.pkl"
    ]
    return all(Path(f).exists() for f in required_files)

def run_preprocessing() -> bool:
    """Execute preprocessing.py with proper error handling."""
    logger.info("Running preprocessing pipeline...")
    try:
        result = subprocess.run(
            [sys.executable, "preprocessing.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        if not check_preprocessing_outputs():
            logger.error("Preprocessing ran but outputs not created")
            return False
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Preprocessing failed: {e.stderr}")
        return False

def load_and_validate_data() -> Tuple[pd.DataFrame, Dict]:
    """Load and validate training data with comprehensive checks."""
    try:
        logger.info("Starting data loading and validation...")
        
        # Load artifacts with version mismatch handling
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                artifacts = joblib.load("models/preprocessing_artifacts.pkl")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {str(e)}")
            raise RuntimeError("Artifacts loading failed") from e

        # Validate artifacts structure
        feature_names = artifacts.get("feature_names", [])
        if not feature_names:
            raise ValueError("No feature names found in artifacts")
            
        # Load and clean data in chunks
        chunksize = 100000
        df_chunks = []
        stats = {
            'original': 0,
            'duplicates': 0,
            'nan_rows': 0,
            'feature_nans': 0,
            'label_nans': 0
        }
        
        try:
            for chunk in pd.read_csv("models/preprocessed_dataset.csv", chunksize=chunksize):
                stats['original'] += len(chunk)
                stats['duplicates'] += chunk.duplicated().sum()
                
                # Track NaN types
                stats['nan_rows'] += chunk.isna().any(axis=1).sum()
                stats['feature_nans'] += chunk[feature_names].isna().sum().sum()
                stats['label_nans'] += chunk['Label'].isna().sum()
                
                # Clean chunk
                chunk = chunk.drop_duplicates().dropna(subset=feature_names + ["Label"])
                df_chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Data reading failed: {str(e)}")
            raise RuntimeError("CSV reading failed") from e

        # Combine cleaned chunks
        if not df_chunks:
            raise ValueError("No valid data remaining after cleaning")
        df = pd.concat(df_chunks, ignore_index=True)
        
        # Data validation report
        logger.info("\n=== Data Validation Report ===")
        logger.info(f"Original samples: {stats['original']:,}")
        logger.info(f"Removed duplicates: {stats['duplicates']:,}")
        logger.info(f"Removed NaN rows: {stats['nan_rows']:,} (Features: {stats['feature_nans']:,}, Labels: {stats['label_nans']:,})")
        logger.info(f"Clean samples remaining: {len(df):,} ({len(df)/stats['original']:.1%})")
        
        # Feature validation
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features[:5]}...")
        logger.info(f"Validated {len(feature_names)} features")
        
        # Label validation
        if "Label" not in df.columns:
            raise ValueError("'Label' column missing")
            
        df["Label"] = df["Label"].astype(int)
        class_counts = df["Label"].value_counts()
        
        logger.info("\n=== Class Distribution ===")
        logger.info(class_counts.to_string())
        
        if len(class_counts) < 2:
            raise ValueError("Dataset must contain both classes (normal and attack)")
        
        # Version-safe scaler handling
        if "scaler" in artifacts:
            try:
                # Align feature names if needed
                if hasattr(artifacts['scaler'], 'feature_names_in_'):
                    feature_map = dict(zip(feature_names, artifacts['scaler'].feature_names_in_))
                    df = df.rename(columns=feature_map)
                    feature_names = list(artifacts['scaler'].feature_names_in_)
                
                # Test and apply scaler
                test_sample = df[feature_names].iloc[:1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    artifacts["scaler"].transform(test_sample)
                logger.info("Scaler validated successfully")
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[feature_names] = artifacts["scaler"].transform(df[feature_names])
                logger.info("Applied feature scaling")
                
            except Exception as e:
                logger.warning(f"Scaler issue ({str(e)}), recreating...")
                new_scaler = MinMaxScaler()
                new_scaler.fit(df[feature_names])
                artifacts["scaler"] = new_scaler
                logger.info("New scaler created and fitted")
        
        # Final quality checks
        if df.isna().any().any():
            remaining_nans = df.isna().sum().sum()
            logger.warning(f"Remaining NaNs: {remaining_nans}, filling with 0")
            df = df.fillna(0)
        
        logger.info("Data validation completed successfully")
        return df, artifacts
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        logger.error("Troubleshooting steps:")
        logger.error("1. Verify preprocessing artifacts exist")
        logger.error("2. Check CSV file integrity")
        logger.error("3. Ensure feature consistency")
        raise RuntimeError("Data loading failed") from e

def create_synthetic_data() -> Tuple[pd.DataFrame, Dict]:
    """Generate balanced synthetic dataset with realistic characteristics."""
    logger.warning("Generating synthetic dataset as fallback")
    num_samples = 10000
    num_features = 20
    
    # Create separable classes
    np.random.seed(42)
    X_normal = np.random.normal(0.2, 0.1, (num_samples//2, num_features))
    X_attack = np.random.normal(0.8, 0.1, (num_samples//2, num_features))
    X = np.vstack([X_normal, X_attack])
    y = np.array([0]*(num_samples//2) + [1]*(num_samples//2))
    
    # Clip to [0,1] range
    X = np.clip(X, 0, 1)
    
    return (
        pd.DataFrame(X, columns=[f"feature_{i}" for i in range(num_features)])
        .assign(Label=y),
        {
            "feature_names": [f"feature_{i}" for i in range(num_features)],
            "scaler": None,
            "chunk_size": 1000
        }
    )

def prepare_dataloaders(
    df: pd.DataFrame,
    artifacts: Dict,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Prepare optimized dataloaders with proper stratification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == 'cuda'
    
    # Prepare features and labels
    X = df[artifacts["feature_names"]].values
    y = df['Label'].values
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(X, y))
    
    # Handle extreme class imbalance
    class_counts = torch.bincount(y_tensor[train_idx])
    logger.info(f"Raw class distribution in training set: {class_counts.tolist()}")
    
    if torch.min(class_counts) < 1000:  # Threshold for extreme imbalance
        logger.warning("Extreme class imbalance detected, applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(
            X[train_idx], 
            y[train_idx]
        )
        X_tensor = torch.tensor(np.vstack([X_res, X[val_idx]]), dtype=torch.float32)
        y_tensor = torch.tensor(np.concatenate([y_res, y[val_idx]]), dtype=torch.long)
        
        # Recalculate indices after SMOTE
        train_size = len(X_res)
        train_idx = np.arange(train_size)
        val_idx = np.arange(train_size, len(X_tensor))
        
        class_counts = torch.bincount(y_tensor[train_idx])
        logger.info(f"Class distribution after SMOTE: {class_counts.tolist()}")
    
    # Create weighted sampler
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_tensor[train_idx]]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create datasets
    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    
    # Configure dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size*2,
        sampler=sampler,
        pin_memory=pin_memory,
        worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() + worker_id),
        num_workers=2 if pin_memory else 0,
        persistent_workers=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        pin_memory=pin_memory
    )
    
    # Calculate batch counts
    train_batches = len(train_loader.dataset) // batch_size
    if len(train_loader.dataset) % batch_size != 0:
        train_batches += 1
    val_batches = len(val_loader.dataset) // (batch_size*2)
    if len(val_loader.dataset) % (batch_size*2) != 0:
        val_batches += 1
        
    logger.info(f"Training batches: {train_batches}, Validation batches: {val_batches}")
    
    return train_loader, val_loader, X.shape[1], len(class_counts)

# Training and validation functions
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> Tuple[float, float]:
    """Train model for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(loader), correct / total

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Validate model performance with comprehensive metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    val_loss = total_loss / len(loader)
    val_acc = (all_preds == all_labels).mean()
    val_auc = roc_auc_score(all_labels, all_probs)
    val_ap = average_precision_score(all_labels, all_probs)
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_auc': val_auc,
        'val_ap': val_ap,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }

def visualize_data_distribution(df: pd.DataFrame, log_dir: Path) -> None:
    """Visualize data distribution using PCA."""
    try:
        logger.info("Creating PCA visualization of data distribution...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df.iloc[:, :-1])
        
        plt.figure(figsize=(10,6))
        plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Label'], alpha=0.1, cmap='viridis')
        plt.title("PCA of Dataset")
        plt.colorbar()
        plt.savefig(log_dir / "data_pca.png")
        plt.close()
        
        logger.info("Saved PCA visualization of data distribution")
    except Exception as e:
        logger.warning(f"Could not create PCA visualization: {str(e)}")

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metrics: Dict[str, Any],
    filename: Path,
    training_meta: Dict[str, Any]
) -> None:
    """Save model checkpoint with comprehensive metadata and checksum."""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in best_metrics.items()},
            'training_meta': training_meta,
            'environment': {
                'pytorch_version': torch.__version__,
                'numpy_version': np.__version__,
                'python_version': platform.python_version(),
                'device': str(device)
            }
        }
        
        torch.save(checkpoint, filename)
        
        # Calculate and save checksum
        with open(filename, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        with open(f"{filename}.md5", 'w') as f:
            f.write(checksum)
            
        logger.info(f"Saved checkpoint to {filename} with checksum {checksum}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")

def load_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Dict, Dict]:
    """Safely load checkpoint with multiple fallback mechanisms."""
    # Define safe numpy globals
    numpy_globals = [
        np._core.multiarray._reconstruct,
        np._core.multiarray.scalar,
        np.dtype,
        np.ndarray,
        np.number,
        np.float64,
        np.int64,
        np._core.multiarray.array,
        np.dtypes.Float64DType
    ]
    
    default_metrics = {
        'epoch': -1,
        'val_loss': float('inf'),
        'val_acc': 0.0,
        'val_auc': 0.0,
        'preds': [],
        'labels': [],
        'probs': []
    }
    
    # Verify checksum if available
    if Path(f"{filename}.md5").exists():
        with open(f"{filename}.md5", 'r') as f:
            expected_checksum = f.read().strip()
        with open(filename, 'rb') as f:
            actual_checksum = hashlib.md5(f.read()).hexdigest()
        if expected_checksum != actual_checksum:
            logger.warning(f"Checksum mismatch for {filename}")

    try:
        # First try with weights_only=True and safe_globals
        with torch.serialization.safe_globals(numpy_globals):
            checkpoint = torch.load(filename, weights_only=True)
            if model:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Convert lists back to numpy arrays
            metrics = checkpoint.get('best_metrics', {})
            for k, v in metrics.items():
                if isinstance(v, list):
                    metrics[k] = np.array(v)
            
            return (
                checkpoint.get('model_state_dict'),
                checkpoint.get('optimizer_state_dict'),
                checkpoint.get('scheduler_state_dict'),
                metrics,
                checkpoint.get('training_meta', {})
            )
    except Exception as e:
        logger.warning(f"Safe load failed ({str(e)}), trying fallback methods...")
        
        # Fallback to weights_only=False
        try:
            checkpoint = torch.load(filename, weights_only=False)
            if model:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            metrics = checkpoint.get('best_metrics', {})
            for k, v in metrics.items():
                if isinstance(v, list):
                    metrics[k] = np.array(v)
            
            return (
                checkpoint.get('model_state_dict'),
                checkpoint.get('optimizer_state_dict'),
                checkpoint.get('scheduler_state_dict'),
                metrics,
                checkpoint.get('training_meta', {})
            )
        except Exception as e:
            logger.error(f"All loading methods failed: {str(e)}")
            return None, None, None, default_metrics, {}

def save_training_artifacts(
    model: nn.Module,
    best_metrics: Dict[str, Any],
    input_size: int,
    num_classes: int,
    class_weights: torch.Tensor,
    timestamp: str
) -> None:
    """Save all training artifacts including model and metadata."""
    try:
        # Save final model
        torch.save(model.state_dict(), MODEL_DIR / "ids_model.pth")
        
        # Save comprehensive training metadata
        training_meta = {
            'input_size': input_size,
            'num_classes': num_classes,
            'best_epoch': best_metrics['epoch'],
            'val_loss': best_metrics['val_loss'],
            'val_acc': best_metrics['val_acc'],
            'val_auc': best_metrics['val_auc'],
            'val_ap': best_metrics.get('ap', best_metrics['val_auc']),
            'train_loss': best_metrics.get('train_loss', float('inf')),
            'train_acc': best_metrics.get('train_acc', 0.0),
            'class_distribution': dict(zip(*np.unique(best_metrics.get('labels', [])), 
                                      return_counts=True)) if len(best_metrics.get('labels', [])) > 0 else {},
            'class_weights': class_weights.cpu().numpy().tolist(),
            'timestamp': timestamp,
            'environment': {
                'pytorch_version': torch.__version__,
                'numpy_version': np.__version__,
                'python_version': platform.python_version(),
                'device': str(device)
            }
        }
        
        joblib.dump(training_meta, MODEL_DIR / "training_metadata.pkl")
        
        # Save human-readable JSON version
        with open(MODEL_DIR / "training_metadata.json", 'w') as f:
            json.dump({k: str(v) if isinstance(v, (np.ndarray, pd.Timestamp)) else v 
                     for k, v in training_meta.items()}, f, indent=2)
        
        logger.info("\n=== Training Summary ===")
        logger.info(f"Best model saved to {MODEL_DIR / 'ids_model.pth'}")
        logger.info(f"Best validation metrics - Loss: {best_metrics['val_loss']:.4f}, "
                   f"Accuracy: {best_metrics['val_acc']:.2%}, AUC: {best_metrics['val_auc']:.4f}")
    except Exception as e:
        logger.error(f"Failed to save training artifacts: {str(e)}")

def banner() -> None:
    """Print banner"""
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print("      IDS | MODEL TRAINING SUITE".center(60))
    print("=" * 60 + Style.RESET_ALL)

def print_menu() -> None:
    """Print menu options"""
    print(Fore.YELLOW + "\nAvailable Options:")
    print("1. Configure System Settings")
    print("2. Setup Directories")
    print("3. Check Package Versions")
    print("4. Setup GPU/CPU")
    print("5. Run Training Pipeline")
    print("6. Run Training with Synthetic Data")
    print("7. Show Current Configuration")
    print("8. Exit")

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent command injection"""
    return re.sub(r'[;&|$]', '', input_str).strip()

def get_current_config() -> Dict[str, Any]:
    """Get current configuration"""
    return {
        'batch_size': DEFAULT_BATCH_SIZE,
        'epochs': DEFAULT_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'gradient_clip': GRADIENT_CLIP,
        'mixed_precision': MIXED_PRECISION,
        'early_stopping': EARLY_STOPPING_PATIENCE,
        'model_architecture': {
            'hidden_layers': HIDDEN_LAYER_SIZES,
            'dropout_rates': DROPOUT_RATES,
            'activation': ACTIVATION,
            'use_batch_norm': USE_BATCH_NORM
        }
    }

def show_config() -> None:
    """Show current configuration"""
    config = get_current_config()
    print(Fore.CYAN + "\nCurrent Configuration:")
    print(Fore.YELLOW + "\nTraining Parameters:")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Weight Decay: {config['weight_decay']}")
    print(f"  Gradient Clip: {config['gradient_clip']}")
    print(f"  Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
    print(f"  Early Stopping Patience: {config['early_stopping']}")
    
    print(Fore.YELLOW + "\nModel Architecture:")
    print(f"  Hidden Layers: {config['model_architecture']['hidden_layers']}")
    print(f"  Dropout Rates: {config['model_architecture']['dropout_rates']}")
    print(f"  Activation: {config['model_architecture']['activation']}")
    print(f"  Batch Normalization: {'Enabled' if config['model_architecture']['use_batch_norm'] else 'Disabled'}")

def configure_training() -> None:
    """Configure training parameters interactively"""
    global DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    global GRADIENT_CLIP, MIXED_PRECISION, EARLY_STOPPING_PATIENCE
    
    print(Fore.CYAN + "\nTraining Configuration")
    
    try:
        DEFAULT_BATCH_SIZE = int(input(f"Batch size [{DEFAULT_BATCH_SIZE}]: ") or DEFAULT_BATCH_SIZE)
        DEFAULT_EPOCHS = int(input(f"Epochs [{DEFAULT_EPOCHS}]: ") or DEFAULT_EPOCHS)
        LEARNING_RATE = float(input(f"Learning rate [{LEARNING_RATE}]: ") or LEARNING_RATE)
        WEIGHT_DECAY = float(input(f"Weight decay [{WEIGHT_DECAY}]: ") or WEIGHT_DECAY)
        GRADIENT_CLIP = float(input(f"Gradient clip [{GRADIENT_CLIP}]: ") or GRADIENT_CLIP)
        MIXED_PRECISION = input(f"Use mixed precision? (y/n) [{'y' if MIXED_PRECISION else 'n'}]: ").lower() == 'y'
        EARLY_STOPPING_PATIENCE = int(input(f"Early stopping patience [{EARLY_STOPPING_PATIENCE}]: ") or EARLY_STOPPING_PATIENCE)
        
        print(Fore.GREEN + "Training configuration updated successfully")
    except ValueError as e:
        print(Fore.RED + f"Invalid input: {str(e)}")

def configure_model() -> None:
    """Configure model architecture interactively"""
    global HIDDEN_LAYER_SIZES, DROPOUT_RATES, ACTIVATION, USE_BATCH_NORM
    
    print(Fore.CYAN + "\nModel Architecture Configuration")
    
    try:
        # Get hidden layer sizes
        layers_input = input(f"Hidden layer sizes (comma separated) [{', '.join(map(str, HIDDEN_LAYER_SIZES))}]: ")
        if layers_input:
            HIDDEN_LAYER_SIZES = [int(x.strip()) for x in layers_input.split(',')]
        
        # Get dropout rates
        dropout_input = input(f"Dropout rates (comma separated) [{', '.join(map(str, DROPOUT_RATES))}]: ")
        if dropout_input:
            DROPOUT_RATES = [float(x.strip()) for x in dropout_input.split(',')]
        
        # Validate layer sizes and dropout rates match
        if len(HIDDEN_LAYER_SIZES) != len(DROPOUT_RATES):
            print(Fore.RED + "Error: Number of hidden layers must match number of dropout rates")
            return
        
        # Get activation function
        ACTIVATION = input(f"Activation function (relu/leaky_relu/gelu) [{ACTIVATION}]: ") or ACTIVATION
        if ACTIVATION == 'leaky_relu':
            ACTIVATION_PARAM = float(input(f"Leaky ReLU negative slope [{ACTIVATION_PARAM}]: ") or ACTIVATION_PARAM)
        
        USE_BATCH_NORM = input(f"Use batch normalization? (y/n) [{'y' if USE_BATCH_NORM else 'n'}]: ").lower() == 'y'
        
        print(Fore.GREEN + "Model configuration updated successfully")
    except ValueError as e:
        print(Fore.RED + f"Invalid input: {str(e)}")

def interactive_main() -> None:
    """Interactive main function"""
    banner()
    
    # Initial setup
    configure_system()
    configure_visualization()
    set_seed(42)
    
    # Setup logging and directories
    LOG_DIR = Path("logs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_DIR)
    
    try:
        directories = setup_directories(logger)
        MODEL_DIR = directories['models']
        LOG_DIR = directories['logs']
        DATA_DIR = directories['data']
        FIGURE_DIR = directories['figures']
        TB_DIR = directories['tensorboard']
        CHECKPOINT_DIR = directories['checkpoints']
    except Exception as e:
        logger.error(f"Failed to set up directories: {str(e)}")
        sys.exit(1)
    
    while True:
        print_menu()
        choice = input(Fore.WHITE + "\nSelect an option (1-8): ").strip()
        
        if choice == "1":
            configure_system()
            print(Fore.GREEN + "System configuration applied")
        elif choice == "2":
            try:
                directories = setup_directories(logger)
                print(Fore.GREEN + "Directories set up successfully")
            except Exception as e:
                print(Fore.RED + f"Directory setup failed: {str(e)}")
        elif choice == "3":
            if check_versions(logger):
                print(Fore.GREEN + "All package versions are compatible")
            else:
                print(Fore.RED + "Some package versions are incompatible")
        elif choice == "4":
            device = setup_gpu(logger)
            print(Fore.GREEN + f"Using device: {device}")
        elif choice == "5":
            print(Fore.YELLOW + "\nStarting training pipeline...")
            train_model(use_mock=False)
        elif choice == "6":
            print(Fore.YELLOW + "\nStarting training with synthetic data...")
            train_model(use_mock=True)
        elif choice == "7":
            show_config()
        elif choice == "8":
            print(Fore.CYAN + "\nExiting... Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid selection. Choose 1-8.")

def train_model(use_mock: bool = False) -> None:
    """Complete training pipeline with all enhancements."""
    # Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_DIR / f"train_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Data preparation
    try:
        if use_mock:
            logger.info("Using synthetic data by request")
            df, artifacts = create_synthetic_data()
        else:
            if not check_preprocessing_outputs():
                logger.warning("Preprocessing outputs not found")
                if not run_preprocessing():
                    raise RuntimeError("Preprocessing failed")
                logger.info("Preprocessing completed successfully")
            
            df, artifacts = load_and_validate_data()
            logger.info(f"Loaded {len(df)} validated samples")
            
            # Handle extreme imbalance
            if len(df['Label'].value_counts()) < 2 or df['Label'].value_counts().min() < 1000:
                logger.warning("Extreme class imbalance detected, applying SMOTE...")
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(
                    df[artifacts['feature_names']], 
                    df['Label']
                )
                df = pd.DataFrame(X_res, columns=artifacts['feature_names'])
                df['Label'] = y_res
                logger.info(f"After SMOTE: {len(df)} samples")
        
        # Visualize data distribution
        visualize_data_distribution(df, log_dir)
        
        # Prepare dataloaders
        train_loader, val_loader, input_size, num_classes = prepare_dataloaders(df, artifacts)
        logger.info(f"Data prepared - Input size: {input_size}, Classes: {num_classes}")
        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
        sys.exit(1)

    # Model configuration
    model = IDSModel(input_size, num_classes).to(device)
    
    # Enhanced class weighting
    class_counts = torch.tensor(df['Label'].value_counts().sort_index().values, dtype=torch.float32)
    class_weights = (1. / class_counts) * (class_counts.sum() / num_classes)
    class_weights = class_weights / class_weights.sum()
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=3, 
        factor=0.5
    )

    # Initialize mixed precision training
    scaler = GradScaler(enabled=False)  # Disable for CPU-only
    
    # Training loop
    logger.info("\nStarting training...")
    best_metrics = {
        'epoch': -1,
        'val_loss': float('inf'),
        'val_acc': 0.0,
        'val_auc': 0.0,
        'preds': np.array([]),
        'labels': np.array([]),
        'probs': np.array([]),
        'train_loss': float('inf'),
        'train_acc': 0.0,
        'ap': 0.0
    }
    best_score = float('inf')
    patience_counter = 0
    
    for epoch in range(DEFAULT_EPOCHS):
        # Training phase with mixed precision
        model.train()
        train_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            with autocast(enabled=False):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch) / GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            # Clear memory
            del X_batch, y_batch, outputs, predicted
            torch.cuda.empty_cache()

        # Validation phase
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # Learning rate adjustment
        scheduler.step(val_metrics['val_loss'])
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_metrics['val_acc'], epoch)
        writer.add_scalar('AUC/val', val_metrics['val_auc'], epoch)
        writer.add_scalar('AP/val', val_metrics['val_ap'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        logger.info(
            f"Epoch {epoch+1:03d}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Acc: {val_metrics['val_acc']:.2%} | "
            f"AUC: {val_metrics['val_auc']:.4f} | "
            f"AP: {val_metrics['val_ap']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Early stopping and checkpointing
        if val_metrics['val_loss'] < best_score:
            best_score = val_metrics['val_loss']
            patience_counter = 0
            best_metrics = {
                'epoch': epoch,
                'val_loss': val_metrics['val_loss'],
                'val_acc': val_metrics['val_acc'],
                'val_auc': val_metrics['val_auc'],
                'val_ap': val_metrics['val_ap'],
                'preds': val_metrics['preds'],
                'labels': val_metrics['labels'],
                'probs': val_metrics['probs'],
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            
            # Save the best model with checksum
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metrics=best_metrics,
                filename=MODEL_DIR / "best_model.pth",
                training_meta={
                    'input_size': input_size,
                    'num_classes': num_classes,
                    'class_weights': class_weights.cpu().numpy().tolist(),
                    'timestamp': timestamp
                }
            )
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    logger.info("\nTraining complete. Loading best model...")
    try:
        # Try loading with enhanced safety
        _, _, _, loaded_metrics, _ = load_checkpoint(
            MODEL_DIR / "best_model.pth",
            model=model
        )
        
        if loaded_metrics['epoch'] != -1:  # Only update if loading succeeded
            best_metrics = loaded_metrics
    except Exception as e:
        logger.error(f"Failed to load best model: {str(e)}")

    # Generate final reports
    logger.info("\n=== Classification Report ===")
    if len(best_metrics['labels']) > 0 and len(best_metrics['preds']) > 0:
        logger.info(classification_report(
            y_true=best_metrics['labels'],
            y_pred=best_metrics['preds'],
            target_names=['Normal', 'Attack'],
            digits=4
        ))
        
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(best_metrics['labels'], best_metrics['preds']))
    else:
        logger.warning("No validation metrics available")

    # Save final artifacts
    save_training_artifacts(
        model=model,
        best_metrics=best_metrics,
        input_size=input_size,
        num_classes=num_classes,
        class_weights=class_weights,
        timestamp=timestamp
    )
    
    writer.close()

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced IDS Model Trainer")
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use synthetic data for training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive mode"
    )
    args = parser.parse_args()

    # Initial setup
    configure_system()
    configure_visualization()
    set_seed(42)
    
    # Setup logging and directories
    LOG_DIR = Path("logs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_DIR)
    
    try:
        directories = setup_directories(logger)
        MODEL_DIR = directories['models']
        LOG_DIR = directories['logs']
        DATA_DIR = directories['data']
        FIGURE_DIR = directories['figures']
        TB_DIR = directories['tensorboard']
        CHECKPOINT_DIR = directories['checkpoints']
    except Exception as e:
        logger.error(f"Failed to set up directories: {str(e)}")
        sys.exit(1)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    if args.interactive:
        # Interactive mode
        try:
            interactive_main()
        except KeyboardInterrupt:
            print(Fore.RED + "\n\nInterrupted by user. Exiting...")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            traceback.print_exc()
    else:
        # Command-line mode
        logger.info("\n=== Enhanced Network Intrusion Detection Model Trainer ===")
        logger.info(f"Command line arguments: {vars(args)}")
        
        # Update config from command line args
        if args.batch_size != DEFAULT_BATCH_SIZE:
            DEFAULT_BATCH_SIZE = args.batch_size
            logger.info(f"Using batch size: {DEFAULT_BATCH_SIZE}")
        
        if args.epochs != DEFAULT_EPOCHS:
            DEFAULT_EPOCHS = args.epochs
            logger.info(f"Using epochs: {DEFAULT_EPOCHS}")
        
        try:
            train_model(use_mock=args.use_mock)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            sys.exit(1)