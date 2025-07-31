# Standard library imports
import argparse
import datetime
import time
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
from collections import defaultdict
from colorama import Fore, Style, init
import re
import traceback
import tarfile
import zipfile
import shutil
import contextlib
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel

# Third-party imports
import numpy as np
import pandas as pd
import pkg_resources

# Machine learning and deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch._logging
from torch.serialization import add_safe_globals
import torch.utils.data
import torch.utils.data.distributed

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
from tqdm import tqdm

# Serialization
import joblib
import pickle

# Initialize colorama
init(autoreset=True)

# Initialize rich console
console = Console()

# Declare the class names that will be defined later
class IDSModel:
    pass  # Forward declaration

class UnicodeStreamHandler:
    pass  # Forward declaration

# Whitelist TorchVersion for safe loading
add_safe_globals([
    # PyTorch essentials
    torch.Tensor,
    torch.nn.Module,
    torch.nn.parameter.Parameter,
    torch.optim.Optimizer,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    torch.utils.data.Dataset,
    torch.utils.data.DataLoader,
    torch.utils.data.distributed.DistributedSampler,
    
    # Version handling
    torch.torch_version.TorchVersion,
    
    # Numpy types
    np.ndarray,
    np.float32,
    np.int64,
    np._core.multiarray._reconstruct,
    np._core.multiarray.scalar,
    np.dtype,
    np.number,
    np.float64,
    np._core.multiarray.array,
    np.dtypes.Float64DType,
    
    # Pandas types
    pd.DataFrame,
    pd.Series,
    
    # Custom classes (forward declared above)
    IDSModel,
    UnicodeStreamHandler,
    
    # Other necessary classes
    TensorDataset,
    WeightedRandomSampler,
    MinMaxScaler,
    StandardScaler
])

# Disable PyTorch's duplicate logging
torch._logging.set_logs(all=logging.ERROR)

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
    """Configure logging with handler deduplication"""
    logger = logging.getLogger(__name__)
    
    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Rest of your logging setup...
    logger.setLevel(logging.DEBUG)
    
    # Add handlers ONLY if they don't exist
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"train_{timestamp}.log", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    if not any(isinstance(h, UnicodeStreamHandler) for h in logger.handlers):
        console_handler = UnicodeStreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    
    return logger

def setup_directories(logger: logging.Logger) -> Dict[str, Path]:
    """Create and return essential directories with versioned subdirectories."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path().absolute()
    
    directories = {
        'models': base_dir / "models",
        'logs': base_dir / "logs",
        'data': base_dir / "data",
        'figures': base_dir / "figures",
        'tensorboard': base_dir / "tensorboard",
        'checkpoints': base_dir / "checkpoints",
        'config': base_dir / "config",
        'results': base_dir / "results",
        'metrics': base_dir / "metrics",
        'reports': base_dir / "reports",
        'latest': base_dir / "latest",
        'info': base_dir / "info",
        'artifacts': base_dir / "artifacts"
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
    print(Fore.GREEN + "\n".join(output_lines) + Style.RESET_ALL)
    
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
LOG_DIR = Path("logs")
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
    CONFIG_DIR = directories['config']
    RESULTS_DIR = directories['results']
    METRICS_DIR = directories['metrics']
    REPORTS_DIR = directories['reports']
    LATEST_DIR = directories['latest']
    INFO_DIR = directories['info']
    ARTIFACTS_DIR = directories['artifacts']
    DOCS_DIR = Path("docs")
except Exception as e:
    logger.error(Fore.RED + f"Failed to set up directories: {str(e)}" + Style.RESET_ALL)
    sys.exit(1)

# Check package versions
if not check_versions(logger):
    logger.error(Fore.RED + "Some package requirements not met!" + Style.RESET_ALL)

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
def check_preprocessing_outputs(
    strict: bool = False,
    use_color: bool = True,
    min_csv_size: int = 1024,
    min_pkl_size: int = 128,
    validate_csv: bool = True,
    validate_pickle: bool = True
) -> bool:
    """Verify preprocessing outputs with optional validation.
    
    Args:
        strict: Enable content validation (default: False)
        use_color: Enable colored output (default: True)
        min_csv_size: Minimum CSV file size in bytes
        min_pkl_size: Minimum pickle file size in bytes
        validate_csv: Perform CSV content checks
        validate_pickle: Perform pickle structure checks
        
    Returns:
        bool: True if all files exist (and are valid in strict mode)
        
    Raises:
        RuntimeWarning: For suspicious but accepted files
    """
    # Color setup
    red = Fore.RED if use_color else ""
    yellow = Fore.YELLOW if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    
    required_files = {
        "models/preprocessed_dataset.csv": {
            "min_size": min_csv_size,
            "checks": ["header", "delimiter"] if validate_csv else []
        },
        "models/preprocessing_artifacts.pkl": {
            "min_size": min_pkl_size,
            "required_keys": ["feature_names", "scaler"] if validate_pickle else []
        }
    }
    
    for filepath, requirements in required_files.items():
        path = Path(filepath)
        
        # File existence check (always performed)
        if not path.exists():
            logger.error(f"{red}Missing required file: {filepath}{reset}")
            return False
            
        # Skip validation in non-strict mode
        if not strict:
            continue
            
        # File size validation
        file_size = path.stat().st_size
        if file_size < requirements["min_size"]:
            logger.warning(f"{yellow}File appears small ({file_size} bytes): {filepath}{reset}")
            
        # CSV validation
        if filepath.endswith('.csv') and validate_csv:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    header = f.readline()
                    if not header.strip():
                        logger.error(f"{red}Empty CSV file: {filepath}{reset}")
                        return False
                        
                    if "delimiter" in requirements["checks"]:
                        if len(header.split(',')) < 2:
                            logger.error(f"{red}Invalid CSV format in: {filepath}{reset}")
                            return False
            except UnicodeDecodeError:
                logger.error(f"{red}Invalid CSV encoding: {filepath}{reset}")
                return False
            except Exception as e:
                logger.error(f"{red}CSV validation failed: {filepath} - {str(e)}{reset}")
                return False
                
        # Pickle validation
        elif filepath.endswith('.pkl') and validate_pickle:
            try:
                with open(path, 'rb') as f:
                    data = joblib.load(f)
                    for key in requirements["required_keys"]:
                        if key not in data:
                            logger.error(f"{red}Missing key '{key}' in: {filepath}{reset}")
                            return False
            except Exception as e:
                logger.error(f"{red}Pickle load failed: {filepath} - {str(e)}{reset}")
                return False
    
    return True

def run_preprocessing(
    timeout_minutes: float = 30.0,
    cleanup: bool = True,
    use_color: bool = True,
    strict_output_check: bool = True,
    reproducible: bool = True,
    debug: bool = False
) -> bool:
    """Execute preprocessing with enhanced controls.
    
    Args:
        timeout_minutes: Maximum runtime in minutes
        cleanup: Remove existing output files
        use_color: Enable colored output
        strict_output_check: Use strict validation
        reproducible: Set PYTHONHASHSEED for reproducibility
        debug: Enable verbose debugging output
        
    Returns:
        bool: True if preprocessing succeeded
        
    Raises:
        RuntimeError: For unrecoverable failures
        FileNotFoundError: If script is missing
    """
    # Configure output styling
    red = Fore.RED if use_color else ""
    yellow = Fore.YELLOW if use_color else ""
    green = Fore.GREEN if use_color else ""
    reset = Style.RESET_ALL if use_color else ""

    logger.info(f"{yellow}=== Preprocessing Pipeline ==={reset}")
    
    # Validate script existence
    if not Path("preprocessing.py").exists():
        logger.error(f"{red}Preprocessing script not found{reset}")
        raise FileNotFoundError("preprocessing.py not found")

    try:
        # Cleanup previous outputs
        output_files = [
            "models/preprocessed_dataset.csv",
            "models/preprocessing_artifacts.pkl"
        ]
        
        if cleanup:
            for fpath in output_files:
                if Path(fpath).exists():
                    logger.info(f"{yellow}Cleaning up: {fpath}{reset}")
                    Path(fpath).unlink(missing_ok=True)

        # Prepare environment
        env = os.environ.copy()
        if reproducible:
            env["PYTHONHASHSEED"] = "42"

        # Execute with timeout
        timeout_seconds = int(timeout_minutes * 60)
        result = subprocess.run(
            [sys.executable, "preprocessing.py"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env
        )

        # Stream output
        for line in result.stdout.splitlines():
            logger.info(f"{green}{line[:500]}{reset}")
            if debug:
                logger.debug(f"STDOUT: {line[:200]}")

        # Validate outputs
        if not check_preprocessing_outputs(strict=strict_output_check):
            logger.error(f"{red}Output validation failed{reset}")
            log_troubleshooting("validation")
            return False

        logger.info(f"{green}Preprocessing completed successfully{reset}")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"{red}Timeout after {timeout_minutes} minutes{reset}")
        log_troubleshooting("timeout")
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"{red}Process failed (code {e.returncode}){reset}")
        log_error_output(e.stderr, use_color)
        log_troubleshooting("execution")
        return False

    except Exception as e:
        logger.error(f"{red}Unexpected error: {type(e).__name__}{reset}")
        if debug:
            logger.error(f"{red}{traceback.format_exc()}{reset}")
        log_troubleshooting("unexpected")
        raise RuntimeError("Preprocessing failed") from e

def log_troubleshooting(error_type: str):
    """Centralized troubleshooting guides."""
    guides = {
        "validation": [
            "1. Verify preprocessing script generates correct outputs",
            "2. Check file permissions in models/ directory",
            "3. Validate disk space is available"
        ],
        "timeout": [
            "1. Optimize preprocessing steps",
            "2. Increase timeout_minutes parameter",
            "3. Check for infinite loops"
        ],
        "execution": [
            "1. Run preprocessing.py manually to debug",
            "2. Check dependency versions",
            "3. Validate input data quality"
        ],
        "unexpected": [
            "1. Check system resource limits",
            "2. Verify Python environment consistency",
            "3. Enable debug mode for details"
        ]
    }
    logger.warning("Troubleshooting steps:")
    for step in guides.get(error_type, guides["unexpected"]):
        logger.warning(f"  {step}")

def log_error_output(stderr: str, use_color: bool):
    """Log last 20 lines of error output."""
    red = Fore.RED if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    logger.error(f"{red}Last error lines:{reset}")
    for line in stderr.splitlines()[-20:]:
        logger.error(f"{red}{line[:200]}{reset}")

def display_data_loading_header(filepath: str) -> None:
    """Display data loading header with rich formatting."""
    console.print(Panel.fit(
        f"[bold green]Data Loading Started[/bold green]\n"
        f"[bold]Source:[/bold] [bold cyan]{filepath}",
        title="[bold yellow]Data Processing Pipeline[/bold yellow]",
        border_style="blue"
    ))

def display_chunk_progress(stats: Dict[str, Any], history: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Display chunk progress in a single updating table with:
    - Dynamic updates (clears previous output)
    - Full history of chunks
    - Threshold-based styling (clean samples <50% = yellow, <30% = red)
    """
    # Initialize history if first run
    if history is None:
        history = []
    
    # Add current stats to history
    history.append(stats.copy())
    
    # Create table
    # table = Table(
    #     title="[bold]Chunk Processing Progress[/bold]",
    #     box=box.ROUNDED,
    #     header_style="bold blue",
    #     title_style="bold yellow",
    #     title_justify="left",
    #     show_lines=True
    # )

    # Define columns
    # table.add_column("Chunk #", justify="center", style="cyan", width=8)
    # table.add_column("Processed", justify="right", style="magenta", width=12)
    # table.add_column("Clean Samples", justify="right", width=16)
    # table.add_column("Clean %", justify="right", width=10)
    # table.add_column("Dtype Conv", justify="right", width=10)
    # table.add_column("Failed", justify="right", width=10)

    # Add all historical rows
    for chunk_stats in history:
        clean_pct = chunk_stats['cleaned'] / chunk_stats['original']
        
        # Conditional styling
        if clean_pct < 0.3:
            pct_style = "bold red"
        elif clean_pct < 0.5:
            pct_style = "bold yellow"
        else:
            pct_style = "bold green"

        # Create table
        table = Table(
            title="[bold]Chunk Processing Progress[/bold]",
            box=box.ROUNDED,
            header_style="bold blue",
            title_style="bold yellow",
            title_justify="left",
            show_lines=True
        )

        # Define columns
        table.add_column("Chunk #", justify="center", style="cyan", width=8)
        table.add_column("Processed", justify="right", style="magenta", width=12)
        table.add_column("Clean Samples", justify="right", width=16)
        table.add_column("Clean %", justify="right", width=10)
        table.add_column("Dtype Conv", justify="right", width=10)
        table.add_column("Failed", justify="right", width=10)
        
        table.add_row(
            str(chunk_stats['total_chunks']),
            f"{chunk_stats['original']:,}",
            f"{chunk_stats['cleaned']:,}",
            f"[{pct_style}]{clean_pct:.1%}",
            f"{chunk_stats['dtype_conversions']:,}",
            f"[red]{chunk_stats['failed_conversions']:,}" if chunk_stats['failed_conversions'] > 0 
              else f"{chunk_stats['failed_conversions']:,}"
        )
    
    # Clear console before printing new table
    console.clear()
    console.print(table)
    return history

def display_data_validation_summary(stats: Dict[str, Any]) -> None:
    """Display data validation summary in a rich table."""
    # Main table
    table = Table(
        title="[bold]Data Validation Report[/bold]",
        box=box.ROUNDED,
        header_style="bold blue",
        title_style="bold yellow",
        show_lines=True
    )
    
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Impact", style="green", justify="right")
    
    # Helper function for consistent row styling
    def add_row(metric: str, value: Any, impact: str = "", style: str = ""):
        table.add_row(
            metric,
            str(value) if not isinstance(value, (int, float)) else f"{value:,}",
            impact,
            style=style
        )
    
    # Add rows with conditional styling
    add_row("Original Samples", stats['original'])
    add_row("Removed Duplicates", stats['duplicates'], 
           f"{-stats['duplicates']/stats['original']:.1%}", "red")
    add_row("Removed NaN Rows", stats['nan_rows'],
           f"{-stats['nan_rows']/stats['original']:.1%}", "red")
    add_row("Feature NaN Values Filled", stats['feature_nans'])
    add_row("Label NaN Rows Removed", stats['label_nans'])
    add_row("Extreme Values Removed", stats['invalid_values'])
    add_row("Dtype Conversions", stats['dtype_conversions'])
    add_row("Failed Conversions", stats['failed_conversions'])
    add_row("Bad Lines Skipped", stats['bad_lines'])
    
    # Final summary row
    clean_percent = stats['clean_samples'] / stats['original']
    final_style = "bold green" if clean_percent > 0.5 else "bold yellow"
    table.add_row(
        "[bold]Clean Samples Remaining",
        f"[{final_style}]{stats['clean_samples']:,}",
        f"[{final_style}]{clean_percent:.1%}"
    )
    
    console.print(table)

def display_class_distribution(class_counts: pd.Series) -> None:
    """Display class distribution in a rich table."""
    table = Table(
        title="[bold]Class Distribution Analysis[/bold]",
        box=box.ROUNDED,
        header_style="bold blue",
        title_style="bold yellow",
        show_lines=True
    )
    
    table.add_column("Class", style="cyan", width=15)
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Percentage", style="green", justify="right")
    
    total_samples = class_counts.sum()
    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        table.add_row(
            str(class_label),
            f"{count:,}",
            f"{percentage:.2f}%"
        )
    
    console.print(table)

def display_imbalance_analysis(imbalance_ratio: float, threshold: float) -> None:
    """Display imbalance analysis with visual indicators."""
    status_style = "bold red" if imbalance_ratio > threshold else "bold green"
    status_text = "[Warning] Above Threshold" if imbalance_ratio > threshold else "[Success] Within Threshold"
    
    ratio_table = Table(
        box=box.SIMPLE,
        show_header=False,
        show_lines=False,
        padding=(0, 2)
    )
    ratio_table.add_column("Metric", style="bold")
    ratio_table.add_column("Value", style=status_style)
    
    ratio_table.add_row("Imbalance Ratio", f"{imbalance_ratio:.1f}:1")
    ratio_table.add_row("Threshold", f"{threshold}:1")
    ratio_table.add_row("Status", status_text)
    
    console.print(Panel.fit(
        ratio_table,
        title="[bold]Class Imbalance Analysis[/bold]",
        border_style="blue"
    ))

def display_smote_results(original_counts: pd.Series, new_counts: pd.Series) -> None:
    """Display SMOTE resampling results in rich tables."""
    # Main results table
    table = Table(
        title="[bold]SMOTE Resampling Results[/bold]",
        box=box.ROUNDED,
        header_style="bold blue",
        title_style="bold yellow",
        show_lines=True
    )
    
    table.add_column("Class", style="cyan", width=15)
    table.add_column("Original", style="magenta", justify="right")
    table.add_column("New Count", style="green", justify="right")
    table.add_column("Change", justify="right")
    
    for class_label in original_counts.index:
        orig = original_counts[class_label]
        new = new_counts[class_label]
        change = new - orig
        change_pct = (change / orig) * 100 if orig else 0
        
        style = "bold green" if change > 0 else ""
        table.add_row(
            str(class_label),
            f"{orig:,}",
            f"{new:,}",
            f"[{style}]{change:+,} ({change_pct:+.1f}%)"
        )
    
    # Summary table
    summary_table = Table(
        box=box.SIMPLE,
        show_header=False,
        show_lines=False,
        padding=(0, 2)
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Original", style="magenta", justify="right")
    summary_table.add_column("New", style="green", justify="right")
    summary_table.add_column("Change", justify="right")
    
    orig_total = original_counts.sum()
    new_total = new_counts.sum()
    change_total = new_total - orig_total
    change_pct_total = (change_total / orig_total) * 100
    
    summary_table.add_row(
        "[bold]Total Samples",
        f"{orig_total:,}",
        f"{new_total:,}",
        f"[bold]{change_total:+,} ({change_pct_total:+.1f}%)"
    )
    
    console.print(table)
    console.print(Panel.fit(
        summary_table,
        title="[bold]Resampling Summary[/bold]",
        border_style="blue"
    ))

def load_preprocessing_artifacts(
    filepath: str = "models/preprocessing_artifacts.pkl",
    strict: bool = True,
    use_color: bool = True,
    required_keys: List[str] = None,
    validate_scaler: bool = True,
    debug: bool = False
) -> Dict:
    """Load preprocessing artifacts with enhanced validation.
    
    Args:
        filepath: Path to artifacts file
        strict: Enable comprehensive validation
        use_color: Enable colored output
        required_keys: List of required keys (default: ['feature_names', 'scaler'])
        validate_scaler: Verify scaler type
        debug: Enable verbose debugging output
        
    Returns:
        Dict: Validated preprocessing artifacts
        
    Raises:
        RuntimeError: For invalid artifacts (when strict=True)
        FileNotFoundError: If file is missing
    """
    # Setup styling
    red = Fore.RED if use_color else ""
    yellow = Fore.YELLOW if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    
    # Default required keys
    if required_keys is None:
        required_keys = ["feature_names", "scaler"]
    
    try:
        # Load with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if debug:
                console.print(f"[dim]Debug: Loading artifacts from {filepath}[/dim]")
            artifacts = joblib.load(filepath)
        
        # Validation table
        validation_table = Table(
            title="[bold]Artifact Validation[/bold]",
            box=box.ROUNDED,
            header_style="bold blue",
            title_style="bold yellow",
            title_justify="left",
            show_lines=True,
            padding=(0, 2)
        )
        validation_table.add_column("Check", style="bold cyan", width=30)
        validation_table.add_column("Status", style="bold green")
        
        # Basic validation
        if not isinstance(artifacts, dict):
            raise ValueError("Artifacts must be a dictionary")
        validation_table.add_row("[bold]Artifact Type[/bold]", "[ ✓ ] Valid Dictionary")
            
        # Key validation
        missing_keys = [k for k in required_keys if k not in artifacts]
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        validation_table.add_row("[bold]Required Keys[/bold]", "[ ✓ ] All Present")
        
        # Scaler validation
        if validate_scaler and "scaler" in artifacts:
            scaler = artifacts["scaler"]
            valid_scalers = (MinMaxScaler, StandardScaler)
            if not isinstance(scaler, valid_scalers):
                raise TypeError(f"Invalid scaler type: {type(scaler).__name__}")
            validation_table.add_row("[bold]Scaler Type[/bold]", "[ ✓ ] Valid Scaler")
        
        # Version-aware feature names
        feature_names = artifacts["feature_names"]
        if hasattr(artifacts.get("scaler"), 'feature_names_in_'):
            feature_names = artifacts["scaler"].feature_names_in_.tolist()
            if debug:
                console.print("[dim]Debug: Using scaler-derived feature names[/dim]")
            validation_table.add_row("[bold]Feature Names[/bold]", "[ ✓ ] From Scaler")
        else:
            validation_table.add_row("[bold]Feature Names[/bold]", "[ ✓ ] From Artifacts")
        
        console.print(validation_table)
        
        # Prepare return dict
        result = {
            "feature_names": feature_names,
            "scaler": artifacts.get("scaler"),
            "chunk_size": artifacts.get("chunk_size", 100000)
        }
        
        if debug:
            console.print(f"[dim]Debug: Artifacts loaded successfully: {list(result.keys())}[/dim]")
        
        return result
        
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: Artifacts file not found: {filepath}[/bold red]")
        if debug:
            console.print_exception()
        raise FileNotFoundError(f"Artifacts file not found: {filepath}") from e
        
    except Exception as e:
        error_type = type(e).__name__
        console.print(f"[bold red]Error: Artifact loading failed ({error_type}): {str(e)}[/bold red]")
        
        if strict:
            troubleshooting = Table(
                title="[bold]Troubleshooting Steps[/bold]",
                box=box.SIMPLE,
                show_header=False,
                show_lines=False,
                padding=(0, 2)
            )
            troubleshooting.add_column("Step", style="cyan")
            troubleshooting.add_column("Action", style="white")
            
            troubleshooting.add_row("1", "Verify preprocessing script completed successfully")
            troubleshooting.add_row("2", "Check artifact file integrity")
            troubleshooting.add_row("3", f"Validate required keys: {required_keys}")
            troubleshooting.add_row("4", "Check sklearn version compatibility")
            
            console.print(troubleshooting)
            
            if debug:
                console.print_exception()
            
            raise RuntimeError(f"Failed to load artifacts: {str(e)}") from e
        else:
            console.print("[bold yellow]Warning: Using partial artifacts with validation disabled[/bold yellow]")
            return {
                "feature_names": [],
                "scaler": None,
                "chunk_size": 100000
            }

def load_and_clean_data(
    filepath: str,
    feature_names: List[str],
    *,
    chunk_size: int = 100000,
    max_value: float = 1e6,
    min_value: float = -1e6,
    keep_extreme_values: bool = False,
    label_col: str = "Label",
    label_dtype: str = "float32",
    use_color: bool = True,
    debug: bool = False,
    safe_dtype_conversion: bool = True,
    sample_size: int = 10000,
    on_bad_lines: str = 'warn',
    float_precision: str = 'high'
) -> pd.DataFrame:
    """Enhanced data loader with comprehensive validation and robust dtype handling.
    
    Args:
        filepath: Path to CSV file
        feature_names: List of feature columns to keep
        chunk_size: Rows per chunk (default: 100000)
        max_value: Maximum valid feature value (default: 1e6)
        min_value: Minimum valid feature value (default: -1e6)
        keep_extreme_values: Whether to keep out-of-range values
        label_col: Name of label column (default: "Label")
        label_dtype: Data type for label (default: "int32")
        use_color: Enable colored output
        debug: Enable verbose debugging
        safe_dtype_conversion: Automatically handle dtype mismatches (default: True)
        sample_size: Number of rows to sample for dtype inference (default: 10000)
        on_bad_lines: How to handle bad CSV lines ('warn', 'skip', or 'error')
        float_precision: Float precision for CSV parsing ('high', 'round_trip')
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        RuntimeError: For file/parsing issues
        ValueError: For data validation failures
    """
    # Setup styling
    red = Fore.RED if use_color else ""
    yellow = Fore.YELLOW if use_color else ""
    green = Fore.GREEN if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    
    stats = {
        'original': 0,
        'duplicates': 0,
        'nan_rows': 0,
        'feature_nans': 0,
        'label_nans': 0,
        'invalid_values': 0,
        'cleaned': 0,
        'total_chunks': 0,
        'dtype_conversions': 0,
        'skipped_rows': 0,
        'bad_lines': 0,
        'failed_conversions': 0,
        'clean_samples': 0
    }
    
    chunks = []
    required_cols = feature_names + [label_col]
    
    try:
        display_data_loading_header(filepath)
        
        # Validate CSV structure first
        try:
            with open(filepath, 'r') as f:
                header = f.readline().strip().split(',')
                missing_cols = set(required_cols) - set(header)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
        except Exception as e:
            console.print(f"[bold red]CSV validation failed: {str(e)}[/bold red]")
            raise RuntimeError(f"CSV validation failed: {str(e)}") from e
        
        # Improved dtype inference with float32 as default for features
        dtype_table = Table(
            title="[bold]Data Type Analysis[/bold]",
            box=box.ROUNDED,
            header_style="bold blue",
            title_style="bold yellow",
            title_justify="left",
            show_lines=True,
            padding=(0, 2)
        )
        
        dtype_table.add_column("Column", style="cyan")
        dtype_table.add_column("Sampled Type", style="magenta")
        dtype_table.add_column("Using Type", style="green")
        
        try:
            sample_df = pd.read_csv(
                filepath, 
                nrows=sample_size,
                usecols=required_cols,
                engine='c',
                on_bad_lines='warn',
                float_precision=float_precision
            )
            
            dtypes_map = {}
            for col in sample_df.columns:
                if col == label_col:
                    dtypes_map[col] = label_dtype
                elif col in feature_names:
                    dtypes_map[col] = 'float32'
                    actual_dtype = str(sample_df[col].dtype)
                    dtype_table.add_row(col, actual_dtype, 'float32')
            
            console.print(dtype_table)
        except Exception as e:
            console.print(f"[bold yellow]Warning: Dtype inference failed, using safe defaults: {str(e)}[/bold yellow]")
            dtypes_map = {col: 'float32' for col in feature_names}
            dtypes_map[label_col] = label_dtype
        
        # Main loading loop with robust dtype handling
        for chunk in pd.read_csv(
            filepath,
            dtype=dtypes_map,
            usecols=required_cols,
            chunksize=chunk_size,
            engine='c',
            na_values=['nan', 'NaN', 'null', 'NULL', '', 'inf', '-inf'],
            keep_default_na=True,
            on_bad_lines=on_bad_lines,
            float_precision=float_precision
        ):
            if len(chunk) == 0:
                stats['bad_lines'] += chunk_size
                continue
                
            stats['original'] += len(chunk)
            stats['total_chunks'] += 1
            
            # Robust dtype conversion handling
            for col in chunk.columns:
                try:
                    # First try the specified dtype
                    chunk[col] = chunk[col].astype(dtypes_map.get(col, 'float32'))
                except (ValueError, TypeError) as e:
                    if safe_dtype_conversion:
                        try:
                            # Try converting to numeric first
                            converted = pd.to_numeric(chunk[col], errors='coerce')
                            if converted.isna().any():
                                stats['failed_conversions'] += converted.isna().sum()
                                if debug:
                                    logger.debug(f"{yellow}Partial conversion failure in {col}: {converted.isna().sum()} NA values introduced{reset}")
                            
                            # Then convert to target dtype
                            chunk[col] = converted.astype(dtypes_map.get(col, 'float32'))
                            stats['dtype_conversions'] += 1
                            
                            if debug:
                                logger.debug(f"{yellow}Converted {col} via safe method{reset}")
                        except Exception as inner_e:
                            stats['failed_conversions'] += len(chunk)
                            logger.warning(f"{yellow}Failed to convert {col}: {str(inner_e)}{reset}")
                            raise ValueError(f"Critical dtype conversion failed for {col}") from inner_e
                    else:
                        raise ValueError(f"Failed to convert {col} to {dtypes_map.get(col)}") from e
            
            # Duplicate removal
            dup_count = chunk.duplicated().sum()
            stats['duplicates'] += dup_count
            chunk = chunk.drop_duplicates()
            
            # NaN handling
            nan_rows = chunk.isna().any(axis=1).sum()
            stats['nan_rows'] += nan_rows
            
            for col in feature_names:
                col_nans = chunk[col].isna().sum()
                if col_nans > 0:
                    stats['feature_nans'] += col_nans
                    chunk[col] = chunk[col].fillna(0)
            
            label_nans = chunk[label_col].isna().sum()
            stats['label_nans'] += label_nans
            chunk = chunk.dropna(subset=required_cols)
            
            # Value range validation
            if not keep_extreme_values:
                invalid_mask = pd.DataFrame(False, index=chunk.index, columns=feature_names)
                for col in feature_names:
                    invalid_mask[col] = (chunk[col] > max_value) | (chunk[col] < min_value)
                
                invalid_count = invalid_mask.any(axis=1).sum()
                if invalid_count > 0:
                    stats['invalid_values'] += invalid_count
                    chunk = chunk[~invalid_mask.any(axis=1)]
            
            stats['cleaned'] += len(chunk)
            
            if len(chunk) > 0:
                chunks.append(chunk)
            elif debug:
                logger.debug(f"{yellow}Empty chunk after cleaning{reset}")
            
            # Progress reporting
            if stats['total_chunks'] % 10 == 0 or debug:
                display_chunk_progress(stats)
    
    except FileNotFoundError:
        console.print(f"[bold red]Error: Data file not found: {filepath}[/bold red]")
        raise RuntimeError(f"Data file not found: {filepath}") from None
    except pd.errors.EmptyDataError:
        console.print("[bold red]Error: CSV file is empty[/bold red]")
        raise RuntimeError("CSV file is empty") from None
    except Exception as e:
        console.print(f"[bold red]Error: Data loading failed: {str(e)}[/bold red]")
        if debug:
            console.print_exception()
        raise RuntimeError("Data loading failed") from e
    
    # Final validation
    if not chunks:
        console.print("[bold red]Error: No valid data remaining after cleaning[/bold red]")
        raise ValueError("No valid data remaining after cleaning")
    
    df = pd.concat(chunks, ignore_index=True)
    stats['clean_samples'] = len(df)
    
    display_data_validation_summary(stats)
    return df

def handle_class_imbalance(
    df: pd.DataFrame,
    artifacts: Dict,
    *,
    apply_smote: bool = True,
    imbalance_threshold: float = 10.0,
    label_col: str = "Label",
    sampling_strategy: str = "auto",
    random_state: int = 42,
    use_color: bool = True,
    debug: bool = False
) -> pd.DataFrame:
    """Enhanced class imbalance handler with flexible controls.
    
    Args:
        df: Input DataFrame
        artifacts: Preprocessing artifacts dict
        apply_smote: Whether to apply SMOTE (default: True)
        imbalance_threshold: Ratio to consider imbalance (default: 10.0)
        label_col: Name of label column (default: "Label")
        sampling_strategy: SMOTE sampling strategy
        random_state: Random seed for reproducibility
        use_color: Enable colored output
        debug: Enable verbose debugging
        
    Returns:
        Balanced DataFrame if SMOTE applied, else original
        
    Raises:
        ValueError: For invalid inputs or single-class data
        RuntimeError: For SMOTE application failures
    """
    # Setup styling
    red = Fore.RED if use_color else ""
    yellow = Fore.YELLOW if use_color else ""
    green = Fore.GREEN if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    
    console.print(Panel.fit(
        "[bold green]Class Imbalance Analysis[/bold green]",
        border_style="blue"
    ))
    
    # Validate inputs
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
        
    if not isinstance(artifacts, dict) or 'feature_names' not in artifacts:
        raise ValueError("Invalid artifacts - must contain feature_names")
    
    # Get class distribution
    class_counts = df[label_col].value_counts()
    n_classes = len(class_counts)
    
    if n_classes < 2:
        raise ValueError("Dataset must contain at least 2 classes")
    
    # Display class distribution
    display_class_distribution(class_counts)
    
    # Calculate imbalance
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    imbalance_ratio = max_samples / min_samples
    
    display_imbalance_analysis(imbalance_ratio, imbalance_threshold)
    
    # Handle imbalance if exceeds threshold
    if imbalance_ratio > imbalance_threshold:
        logger.warning(
            f"{yellow}Significant imbalance detected ({imbalance_ratio:.1f}:1) "
            f"(threshold: {imbalance_threshold}:1){reset}"
        )
        
        if not apply_smote:
            console.print("[bold yellow]SMOTE not applied (apply_smote=False)[/bold yellow]")
            return df
            
        try:
            # Validate features
            missing_features = [
                f for f in artifacts['feature_names'] 
                if f not in df.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Missing features for SMOTE: {missing_features[:3]}..."
                )
            
            # Safe SMOTE configuration
            k_neighbors = min(5, min_samples - 1)
            if k_neighbors < 1:
                raise ValueError(
                    f"Cannot apply SMOTE - minority class has only {min_samples} samples"
                )
            
            console.print(f"[bold green]Applying SMOTE (k_neighbors={k_neighbors})...[/bold green]")
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state
            )
            
            X_res, y_res = smote.fit_resample(
                df[artifacts['feature_names']],
                df[label_col]
            )
            
            # Create balanced DataFrame
            balanced_df = pd.DataFrame(X_res, columns=artifacts['feature_names'])
            balanced_df[label_col] = y_res
            
            # Report results
            new_counts = balanced_df[label_col].value_counts()
            display_smote_results(class_counts, new_counts)
            
            return balanced_df
            
        except Exception as e:
            console.print(f"[bold red]SMOTE failed: {str(e)}[/bold red]")
            if debug:
                console.print_exception()
            raise RuntimeError("Class balancing failed") from e
    
    else:
        console.print("[bold green]Class distribution within acceptable limits[/bold green]")
        return df

def load_and_validate_data(
    enhanced: bool = True,
    use_color: bool = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """Load and validate training data with configurable enhancements.
    
    Args:
        enhanced: Use improved validation pipeline (default: True)
        use_color: Enable colored output (None=auto-detect)
        **kwargs: Forwarded to helper functions
        
    Returns:
        Tuple of (cleaned DataFrame, preprocessing artifacts)
        
    Raises:
        RuntimeError: If loading fails, with troubleshooting info
    """
    # Auto-detect color support if not specified
    if use_color is None:
        use_color = sys.stdout.isatty()
    
    # Setup styling
    reset = Style.RESET_ALL if use_color else ""
    color_kwargs = {'use_color': use_color, **kwargs}
    
    try:
        if not enhanced:
            # Legacy mode - use original simplified implementation
            logger.info("Starting data loading (legacy mode)...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                artifacts = joblib.load("models/preprocessing_artifacts.pkl")
            
            feature_names = artifacts.get("feature_names", [])
            if not feature_names:
                raise ValueError("No feature names found in artifacts")
                
            # Original chunked loading
            chunksize = kwargs.get('chunk_size', 100000)
            df_chunks = []
            for chunk in pd.read_csv("models/preprocessed_dataset.csv", chunksize=chunksize):
                chunk = chunk.drop_duplicates().dropna(subset=feature_names + ["Label"])
                df_chunks.append(chunk)
                
            df = pd.concat(df_chunks, ignore_index=True)
            return df, artifacts
        
        # Enhanced mode
        console.print(Panel.fit(
            "[bold green]Starting Enhanced Data Loading[/bold green]",
            border_style="blue"
        ))
        
        artifacts = load_preprocessing_artifacts(**kwargs)
        df = load_and_clean_data(
            "models/preprocessed_dataset.csv", 
            artifacts["feature_names"],
            **kwargs
        )
        df = handle_class_imbalance(df, artifacts, **kwargs)
        
        # Version-safe scaler handling (preserve original logic)
        if "scaler" in artifacts:
            feature_names = artifacts["feature_names"]
            try:
                if hasattr(artifacts['scaler'], 'feature_names_in_'):
                    feature_map = dict(zip(feature_names, artifacts['scaler'].feature_names_in_))
                    df = df.rename(columns=feature_map)
                
                test_sample = df[feature_names].iloc[:1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    artifacts["scaler"].transform(test_sample)
                    df[feature_names] = artifacts["scaler"].transform(df[feature_names])
            except Exception as e:
                console.print(f"[bold yellow]Warning: Scaler issue, recreating: {str(e)}[/bold yellow]")
                new_scaler = MinMaxScaler().fit(df[feature_names])
                artifacts["scaler"] = new_scaler
        
        console.print(Panel.fit(
            "[bold green]Data validation completed successfully[/bold green]",
            border_style="green"
        ))
        
        return df, artifacts
        
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]Data loading failed: {str(e)}[/bold red]",
            border_style="red"
        ))
        
        # Enhanced troubleshooting
        troubleshooting = Table(
            title="[bold]Troubleshooting Steps[/bold]",
            box=box.SIMPLE,
            show_header=False,
            show_lines=False,
            padding=(0, 2)
        )
        troubleshooting.add_column("Step", style="cyan")
        troubleshooting.add_column("Action", style="white")
        
        troubleshooting.add_row("1", "Verify preprocessing outputs exist:")
        troubleshooting.add_row("", "   - models/preprocessed_dataset.csv")
        troubleshooting.add_row("", "   - models/preprocessing_artifacts.pkl")
        troubleshooting.add_row("2", "Check file permissions and disk space")
        troubleshooting.add_row("3", "Test with enhanced=False for legacy loader")
        
        console.print(troubleshooting)
        
        raise RuntimeError("Data loading failed") from e

def create_synthetic_data() -> Tuple[pd.DataFrame, Dict]:
    """Generate realistic synthetic data as fallback with logging."""
    logger.warning("Generating synthetic dataset as fallback")
    try:
        num_samples = 10000
        num_features = 20
        
        # Create separable classes with realistic distributions
        np.random.seed(42)
        X_normal = np.random.normal(0.2, 0.1, (num_samples//2, num_features))
        X_attack = np.random.normal(0.8, 0.1, (num_samples//2, num_features))
        X = np.vstack([X_normal, X_attack])
        y = np.array([0]*(num_samples//2) + [1]*(num_samples//2))
        
        # Add realistic noise and artifacts
        X += np.random.normal(0, 0.05, X.shape)  # Add noise
        X = np.clip(X, 0, 1)  # Clip to [0,1] range
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(num_features)]
        
        logger.info(f"Generated synthetic dataset with {num_samples} samples")
        return (
            pd.DataFrame(X, columns=feature_names).assign(Label=y),
            {
                "feature_names": feature_names,
                "scaler": None,
                "chunksize": 100000,  # Default chunk size for compatibility
                "synthetic": True  # Flag indicating synthetic data
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {str(e)}")
        raise RuntimeError("Synthetic data generation failed") from e

def prepare_dataloaders(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Prepare optimized dataloaders with proper stratification and imbalance handling.
    
    Args:
        df: DataFrame containing features and labels
        artifacts: Dictionary with preprocessing artifacts
        batch_size: Base batch size (will be doubled for validation)
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, input_size, num_classes)
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If data preparation fails
    """
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input DataFrame is empty or invalid")
            
        if 'feature_names' not in artifacts:
            raise ValueError("Artifacts must contain feature_names")
            
        feature_names = artifacts['feature_names']
        if not all(col in df.columns for col in feature_names + ['Label']):
            missing = [col for col in feature_names + ['Label'] if col not in df.columns]
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin_memory = device.type == 'cuda'
        
        # Prepare features and labels
        X = df[feature_names].values
        y = df['Label'].values
        
        # Convert data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Stratified split
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            random_state=random_state
        )
        train_idx, val_idx = next(sss.split(X, y))
        
        # Handle class imbalance
        class_counts = torch.bincount(y_tensor[train_idx])
        logger.info(f"Initial class distribution: {class_counts.tolist()}")
        
        if torch.min(class_counts) < 1000:  # Threshold for extreme imbalance
            logger.warning("Extreme class imbalance detected, applying SMOTE...")
            try:
                smote = SMOTE(
                    random_state=random_state,
                    k_neighbors=min(5, torch.min(class_counts).item() - 1)
                )
                X_res, y_res = smote.fit_resample(X[train_idx], y[train_idx])
                
                # Rebuild tensors with augmented data
                X_tensor = torch.tensor(
                    np.vstack([X_res, X[val_idx]]), 
                    dtype=torch.float32
                )
                y_tensor = torch.tensor(
                    np.concatenate([y_res, y[val_idx]]), 
                    dtype=torch.long
                )
                
                # Update indices
                train_size = len(X_res)
                train_idx = np.arange(train_size)
                val_idx = np.arange(train_size, len(X_tensor))
                
                class_counts = torch.bincount(y_tensor[train_idx])
                logger.info(f"Class distribution after SMOTE: {class_counts.tolist()}")
            except Exception as e:
                logger.error(f"SMOTE failed: {str(e)}")
                raise RuntimeError("Failed to balance classes") from e
        
        # Create weighted sampler
        if torch.min(class_counts) < 1000:
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[y_tensor[train_idx]]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_idx),
                replacement=True
            )
        else:
            sampler = RandomSampler(train_idx)
        
        # Create datasets
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        
        # Configure dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size * 2,  # Larger batches for training
            sampler=sampler,
            pin_memory=pin_memory,
            worker_init_fn=lambda worker_id: np.random.seed(random_state + worker_id),
            num_workers=min(4, os.cpu_count() or 1) if pin_memory else 0,
            persistent_workers=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            pin_memory=pin_memory
        )
        
        logger.info(f"Prepared dataloaders with {len(train_dataset)} training and {len(val_dataset)} validation samples")
        return train_loader, val_loader, X.shape[1], len(class_counts)
        
    except Exception as e:
        logger.error(f"Failed to prepare dataloaders: {str(e)}")
        raise RuntimeError("DataLoader preparation failed") from e

# Training and validation functions
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 4,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, float]:
    """
    Train model for one epoch with gradient handling and optional mixed precision.
    
    Args:
        model: Model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Target device (cuda/cpu)
        grad_clip: Maximum gradient norm
        grad_accum_steps: Gradient accumulation steps
        scaler: Gradient scaler for mixed precision
        
    Returns:
        Tuple of (average loss, accuracy)
        
    Raises:
        RuntimeError: If training fails
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    try:
        optimizer.zero_grad()
        
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Mixed precision context
            # with autocast(enabled=scaler is not None):
            #     outputs = model(X_batch)
            #     loss = criterion(outputs, y_batch) / grad_accum_steps
            
            # Mixed precision context
            with autocast(enabled=False):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch) / grad_accum_steps
            
            # Backpropagation
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * grad_accum_steps
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            # Clean up
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy
        
    except Exception as e:
        logger.error(f"Training failed at batch {batch_idx}: {str(e)}")
        raise RuntimeError("Training epoch failed") from e

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate model performance with comprehensive metrics.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Target device (cuda/cpu)
        class_names: Optional list of class names for reporting
        
    Returns:
        Dictionary containing:
        - val_loss: Average loss
        - val_acc: Accuracy
        - val_auc: ROC AUC score
        - val_ap: Average precision
        - preds: Array of predictions
        - labels: Array of true labels
        - probs: Array of predicted probabilities
        - report: Classification report (if class_names provided)
        
    Raises:
        RuntimeError: If validation fails
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    try:
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
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        val_loss = total_loss / len(loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        # Handle binary and multiclass cases
        if len(np.unique(all_labels)) == 2:  # Binary classification
            val_auc = roc_auc_score(all_labels, all_probs[:, 1])
            val_ap = average_precision_score(all_labels, all_probs[:, 1])
        else:  # Multiclass
            val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            val_ap = average_precision_score(all_labels, all_probs)
        
        results = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_ap': val_ap,
            'preds': all_preds,
            'labels': all_labels,
            'probs': all_probs
        }
        
        # Add classification report if class names provided
        if class_names:
            results['report'] = classification_report(
                all_labels, all_preds,
                target_names=class_names,
                digits=4,
                output_dict=True
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise RuntimeError("Validation failed") from e

def visualize_data_distribution(
    df: pd.DataFrame,
    log_dir: Path,
    max_samples: int = 10000,
    random_state: int = 42
) -> Optional[Path]:
    """
    Visualize data distribution using PCA and save plot.
    
    Args:
        df: DataFrame containing features and labels
        log_dir: Directory to save visualization
        max_samples: Maximum samples to plot (for large datasets)
        random_state: Random seed for sampling
        
    Returns:
        Path to saved visualization or None if failed
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        logger.info(Fore.YELLOW + "=== Creating PCA visualization of data distribution ===")
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(Fore.RED + "Input DataFrame is empty or invalid")
            
        if 'Label' not in df.columns:
            raise ValueError(Fore.RED + "DataFrame must contain 'Label' column")
            
        # Sample data if too large
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=random_state)
        
        # Prepare data
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=y, alpha=0.5, cmap='viridis',
            edgecolors='w', linewidths=0.5
        )
        
        plt.title("Data Distribution (PCA)", fontsize=14)
        plt.colorbar(scatter, label='Class')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(alpha=0.3)
        
        # Save plot
        log_dir = Path(log_dir)
        if log_dir.suffix == '.log':
            log_dir = log_dir.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = log_dir / f"data_pca_distribution_{timestamp}.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(Fore.GREEN + f"Saved PCA visualization of data distribution {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.warning(Fore.RED + f"Could not create PCA visualization: {str(e)}")
        return None

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, Any],
    filename: Path,
    config: Dict[str, Any],
    safe_mode: bool = True
) -> bool:
    """
    Save training checkpoint with verification.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics
        filename: Path to save checkpoint (relative to CHECKPOINT_DIR)
        config: Training configuration
        safe_mode: Use safe serialization
        
    Returns:
        True if successful, False otherwise
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in metrics.items()},
            'config': config,
            'environment': {
                'numpy_version': np.__version__,
                'pytorch_version': torch.__version__,
                'python_version': platform.python_version(),
                'device': str(device),
                'timestamp': datetime.datetime.now().isoformat()
            }
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Create full path in CHECKPOINT_DIR
        full_path = Path(filename).absolute()
        
        # Create parent directory if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if safe_mode:
            # Safe serialization
            torch.save(
                checkpoint,
                full_path,
                _use_new_zipfile_serialization=True,
                pickle_protocol=pickle.HIGHEST_PROTOCOL
            )
        else:
            torch.save(checkpoint, full_path)
        
        # Calculate and verify checksum
        checksum = hashlib.md5(full_path.read_bytes()).hexdigest()
        with open(f"{full_path}.md5", 'w') as f:
            f.write(checksum)
        
        logger.info(Fore.GREEN + f"Checkpoint saved successfully to {full_path} (checksum: {checksum})")
        return True
        
    except Exception as e:
        logger.error(Fore.RED + f"Failed to save checkpoint: {str(e)}")
        return False

def load_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    device: torch.device = torch.device('cpu'),
    verify: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load training checkpoint with verification.
    
    Args:
        filename: Path to checkpoint file (relative to CHECKPOINT_DIR)
        model: Optional model to load state into
        device: Target device for model
        verify: Verify checksum if available
        
    Returns:
        Tuple of (checkpoint_data, error_message) where error_message is None if successful
        
    Raises:
        ValueError: If checkpoint is invalid
    """
    try:
        # Create full path in CHECKPOINT_DIR
        full_path = Path(filename).absolute()
        
        # Verify checksum
        if verify and full_path.with_suffix('.md5').exists():
            with open(full_path.with_suffix('.md5'), 'r') as f:
                expected_checksum = f.read().strip()
            
            actual_checksum = hashlib.md5(full_path.read_bytes()).hexdigest()
            if expected_checksum != actual_checksum:
                logger.warning(Fore.YELLOW + f"Checksum mismatch for {full_path}")
                return None, Fore.RED + "Checksum verification failed"
        
        # Load with safe mode first
        try:
            checkpoint = torch.load(full_path, map_location=device, weights_only=True)
        except:
            # Fallback to unsafe load if needed
            checkpoint = torch.load(full_path, map_location=device)
        
        # Validate structure
        required_keys = {'epoch', 'model_state_dict', 'metrics'}
        if not required_keys.issubset(checkpoint.keys()):
            missing = required_keys - checkpoint.keys()
            raise ValueError(Fore.RED + f"Checkpoint missing required keys: {missing}")
        
        # Load model state if provided
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
        
        # Convert metrics back to numpy arrays
        for k, v in checkpoint['metrics'].items():
            if isinstance(v, list):
                checkpoint['metrics'][k] = np.array(v)
        
        logger.info(Fore.GREEN + f"Loaded checkpoint from {full_path} (epoch {checkpoint['epoch']})")
        return checkpoint, None
        
    except Exception as e:
        logger.error(Fore.RED + f"Failed to load checkpoint from {full_path}: {str(e)}")
        return None, str(e)

def save_training_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None
) -> bool:
    """
    Save all training artifacts including model, metrics, and configuration.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        config: Training configuration
        class_names: Optional list of class names
        feature_names: Optional list of feature names
        
    Returns:
        True if successful, False otherwise
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save model (in MODELS_DIR)
        model_path = MODEL_DIR / f"ids_model_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        
        # 2. Save metrics (in METRICS_DIR)
        metrics_path = METRICS_DIR / f"ids_model_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }, f, indent=2)
        
        # 3. Save configuration (in CONFIG_DIR)
        config_path = CONFIG_DIR / f"ids_model_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 4. Save additional info (in INFO_DIR)
        info = {
            'timestamp': timestamp,
            'class_names': class_names,
            'feature_names': feature_names,
            'environment': {
                'pytorch_version': torch.__version__,
                'python_version': platform.python_version(),
                'host': platform.node()
            }
        }
        info_path = INFO_DIR / f"ids_model_info_{timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        # 5. Create archive of all files in ARTIFACTS_DIR
        archive_file = f"ids_model_artifacts_{timestamp}.tar.gz"
        archive_path = ARTIFACTS_DIR / archive_file
        with tarfile.open(archive_path, "w:gz") as tar:
            for file in [model_path, metrics_path, config_path, info_path]:
                tar.add(file, arcname=file.name)
        
        saved_artifacts = {
            'model': model_path,
            'metrics': metrics_path,
            'config': config_path,
            'info': info_path,
            'archive': archive_path
        }
        # Log saved artifacts
        logger.info(Fore.GREEN + f"Saved training artifacts: {saved_artifacts}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save training artifacts: {str(e)}")
        return False

def banner() -> None:
    """Print banner"""
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "      IDS | MODEL TRAINING SUITE".center(60))
    print(Fore.CYAN + Style.BRIGHT + "=" * 60 + Style.RESET_ALL)

def print_menu() -> None:
    """Print menu options"""
    print(Fore.YELLOW + Style.BRIGHT + "\nAvailable Options:")
    print(Fore.WHITE + Style.BRIGHT + "1. Configure System Settings")
    print(Fore.WHITE + Style.BRIGHT + "2. Setup Directories")
    print(Fore.WHITE + Style.BRIGHT + "3. Check Package Versions")
    print(Fore.WHITE + Style.BRIGHT + "4. Setup GPU/CPU")
    print(Fore.WHITE + Style.BRIGHT + "5. Run Training Pipeline")
    print(Fore.WHITE + Style.BRIGHT + "6. Run Training with Synthetic Data")
    print(Fore.WHITE + Style.BRIGHT + "7. Show Current Configuration")
    print(Fore.RED + Style.BRIGHT + "8. Exit")

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
    LOG_DIR = Path("logs")
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
        CONFIG_DIR = directories['config']
        RESULTS_DIR = directories['results']
        METRICS_DIR = directories['metrics']
        REPORTS_DIR = directories['reports']
        LATEST_DIR = directories['latest']
        INFO_DIR = directories['info']
        ARTIFACTS_DIR = directories['artifacts']
        DOCS_DIR = Path("docs")
        
    except Exception as e:
        logger.error(f"Failed to set up directories: {str(e)}")
        sys.exit(1)
    
    while True:
        print_menu()
        choice = input(Fore.WHITE + Style.BRIGHT + "\nSelect an option (1-8): ").strip()
        
        if choice == "1":
            configure_system()
            print(Fore.GREEN + Style.BRIGHT + "System configuration applied")
        elif choice == "2":
            try:
                directories = setup_directories(logger)
                print(Fore.GREEN + Style.BRIGHT + "Directories set up successfully")
            except Exception as e:
                print(Fore.RED + Style.BRIGHT + f"Directory setup failed: {str(e)}")
        elif choice == "3":
            if check_versions(logger):
                print(Fore.GREEN + Style.BRIGHT + "All package versions are compatible")
            else:
                print(Fore.RED + Style.BRIGHT + "Some package versions are incompatible")
        elif choice == "4":
            device = setup_gpu(logger)
            print(Fore.GREEN + Style.BRIGHT + f"Using device: {device}")
        elif choice == "5":
            print(Fore.YELLOW + Style.BRIGHT + "\nStarting training pipeline...")
            # Skip re-initializing logging if already set up
            if not logger.handlers:
                logger = setup_logging(LOG_DIR)
            train_model(use_mock=False)
        elif choice == "6":
            print(Fore.YELLOW + Style.BRIGHT + "\nStarting training with synthetic data...")
            # Skip re-initializing logging if already set up
            if not logger.handlers:
                logger = setup_logging(LOG_DIR)
            train_model(use_mock=True)
        elif choice == "7":
            show_config()
        elif choice == "8":
            print(Fore.CYAN + Style.BRIGHT + "\nExiting... Goodbye!")
            break
        else:
            print(Fore.RED + Style.BRIGHT + "Invalid selection. Choose 1-8.")

class TrainingError(Exception):
    """Base class for training-related exceptions"""
    pass

class DataPreparationError(TrainingError):
    """Exception raised for errors in data preparation phase"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.phase = "data_preparation"

class ModelConfigurationError(TrainingError):
    """Exception raised for errors in model setup phase"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.phase = "model_configuration"

class TrainingExecutionError(TrainingError):
    """Exception raised for errors during training execution"""
    def __init__(self, message: str, epoch: Optional[int] = None, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.epoch = epoch
        self.original_exception = original_exception
        self.phase = "training_execution"

class ModelSavingError(TrainingError):
    """Exception raised for errors in model saving phase"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.phase = "model_saving"

def train_model(
    use_mock: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete training pipeline with enhanced error handling and monitoring.
    
    Args:
        use_mock: Whether to use synthetic data (default: False)
        config: Optional configuration dictionary to override defaults
        
    Returns:
        Dictionary containing training results and metrics
        
    Raises:
        DataPreparationError: If data loading/preprocessing fails
        TrainingError: If training process fails
        ModelSavingError: If model artifacts cannot be saved
    """
    # Initialize training
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    
    # Create run-specific directories
    run_log_dir = LOG_DIR / run_id
    run_figure_dir = FIGURE_DIR / run_id
    run_checkpoint_dir = CHECKPOINT_DIR / run_id
    run_tb_dir = TB_DIR / run_id
    run_artifact_dir = ARTIFACTS_DIR / run_id
    
    # Ensure directories exist
    for dir_path in [run_log_dir, run_figure_dir, run_checkpoint_dir, run_tb_dir, run_artifact_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    training_meta = {
        'start_time': timestamp,
        'run_id': run_id,
        'config': config or {},
        'environment': {
            'pytorch_version': torch.__version__,
            'python_version': platform.python_version(),
            'device': str(device)
        },
        'directories': {
            'logs': str(run_log_dir),
            'figures': str(run_figure_dir),
            'checkpoints': str(run_checkpoint_dir),
            'tensorboard': str(run_tb_dir),
            'artifacts': str(run_artifact_dir)
        }
    }

    try:
        # Setup logging
        log_file = run_log_dir / f"training_{timestamp}.log"
        logger = setup_logging(run_log_dir)
        writer = SummaryWriter(log_dir=run_tb_dir)

        # Data preparation
        try:
            if use_mock:
                logger.info("Using synthetic data by request")
                df, artifacts = create_synthetic_data()
                training_meta['data_source'] = 'synthetic'
            else:
                if not check_preprocessing_outputs():
                    logger.warning("Preprocessing outputs not found")
                    if not run_preprocessing():
                        raise DataPreparationError("Preprocessing failed")
                    logger.info("Preprocessing completed successfully")
                
                df, artifacts = load_and_validate_data()
                logger.info(f"Loaded {len(df)} validated samples")
                training_meta['data_source'] = 'real'
                training_meta['original_samples'] = len(df)

            # Handle class imbalance
            df = handle_class_imbalance(df, artifacts, apply_smote=True)
            training_meta['final_samples'] = len(df)
            
            # Visualize data
            viz_path = visualize_data_distribution(df, run_figure_dir)
            if viz_path:
                training_meta['visualization'] = str(viz_path)

            # Prepare dataloaders
            train_loader, val_loader, input_size, num_classes = prepare_dataloaders(
                df, 
                artifacts,
                batch_size=config.get('batch_size', DEFAULT_BATCH_SIZE) if config else DEFAULT_BATCH_SIZE
            )
            training_meta.update({
                'input_size': input_size,
                'num_classes': num_classes,
                'train_batches': len(train_loader),
                'val_batches': len(val_loader)
            })
            logger.info(f"Data prepared - Input size: {input_size}, Classes: {num_classes}")

        except Exception as e:
            raise DataPreparationError(f"Data preparation failed: {str(e)}") from e

        # Model configuration
        try:
            model = IDSModel(input_size, num_classes).to(device)
            
            # Class weighting
            class_counts = torch.tensor(df['Label'].value_counts().sort_index().values, dtype=torch.float32)
            class_weights = (1. / class_counts) * (class_counts.sum() / num_classes)
            class_weights = class_weights / class_weights.sum()
            logger.info(f"Class weights: {class_weights.tolist()}")
            training_meta['class_weights'] = class_weights.cpu().numpy().tolist()
            
            # Training components
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.get('learning_rate', LEARNING_RATE) if config else LEARNING_RATE,
                weight_decay=config.get('weight_decay', WEIGHT_DECAY) if config else WEIGHT_DECAY
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=config.get('lr_patience', 3) if config else 3,
                factor=0.5
            )
            
            scaler = GradScaler(enabled=False)  # Disable for CPU-only

        except Exception as e:
            raise ModelConfigurationError(f"Model setup failed: {str(e)}") from e

        # Training loop
        best_metrics = {
            'epoch': -1,
            'val_loss': float('inf'),
            'val_acc': 0.0,
            'val_auc': 0.0,
            'train_loss': float('inf'),
            'train_acc': 0.0,
            'learning_rate': 0.0
        }
        early_stop_patience = config.get('early_stopping', EARLY_STOPPING_PATIENCE) if config else EARLY_STOPPING_PATIENCE
        patience_counter = 0
        
        logger.info("\n=== Starting Training ===")
        start_time = time.time()
        
        try:
            for epoch in range(config.get('epochs', DEFAULT_EPOCHS) if config else DEFAULT_EPOCHS):
                epoch_start = time.time()
                
                # Train epoch
                train_loss, train_acc = train_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    grad_clip=config.get('gradient_clip', GRADIENT_CLIP) if config else GRADIENT_CLIP,
                    grad_accum_steps=config.get('grad_accum_steps', 1) if config else 1,
                    scaler=scaler
                )
                
                # Validate
                val_metrics = validate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    class_names=['Normal', 'Attack']
                )
                
                # Learning rate adjustment
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_metrics['val_acc'])
                
                # Update best metrics
                if val_metrics['val_loss'] < best_metrics['val_loss']:
                    best_metrics.update({
                        'epoch': epoch,
                        'val_loss': val_metrics['val_loss'],
                        'val_acc': val_metrics['val_acc'],
                        'val_auc': val_metrics['val_auc'],
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'learning_rate': current_lr,
                        'preds': val_metrics['preds'],
                        'labels': val_metrics['labels'],
                        'probs': val_metrics['probs']
                    })
                    patience_counter = 0
                    
                    # Save best model
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics=best_metrics,
                        filename=f"best_model_{timestamp}.pth",
                        config=training_meta,
                        output_dir=run_checkpoint_dir
                    )
                else:
                    patience_counter += 1
                
                # Logging
                epoch_time = time.time() - epoch_start
                writer.add_scalar('Time/epoch', epoch_time, epoch)
                writer.add_scalar('LR', current_lr, epoch)
                
                logger.info(
                    f"Epoch {epoch+1:03d}/{config.get('epochs', DEFAULT_EPOCHS) if config else DEFAULT_EPOCHS} | "
                    f"Time: {epoch_time:.1f}s | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Train Acc: {train_acc:.2%} | Val Acc: {val_metrics['val_acc']:.2%} | "
                    f"Val AUC: {val_metrics['val_auc']:.4f} | LR: {current_lr:.2e} | "
                    f"Patience: {patience_counter}/{early_stop_patience}"
                )
                
                # Early stopping
                if patience_counter >= early_stop_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        except Exception as e:
            raise TrainingExecutionError(
                f"Training execution failed at epoch {epoch}: {str(e)}",
                epoch=epoch,
                original_exception=e
            ) from e

        # Final evaluation and reporting
        training_meta.update({
            'end_time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_time': time.time() - start_time,
            'best_epoch': best_metrics['epoch'],
            'early_stop': patience_counter >= early_stop_patience
        })
        
        # Load best model for final evaluation
        try:
            checkpoint, error = load_checkpoint(
                run_checkpoint_dir / f"best_model_{timestamp}.pth",
                model=model,
                device=device
            )
            if error:
                logger.warning(f"Could not load best model: {error}")
        except Exception as e:
            logger.error(f"Failed to load best model: {str(e)}")

        # Generate reports
        logger.info("\n=== Training Summary ===")
        logger.info(f"Best epoch: {best_metrics['epoch'] + 1}")
        logger.info(f"Best validation loss: {best_metrics['val_loss']:.4f}")
        logger.info(f"Best validation accuracy: {best_metrics['val_acc']:.2%}")
        logger.info(f"Best validation AUC: {best_metrics['val_auc']:.4f}")
        
        if 'preds' in best_metrics and 'labels' in best_metrics:
            logger.info("\n=== Classification Report ===")
            logger.info(classification_report(
                best_metrics['labels'],
                best_metrics['preds'],
                target_names=['Normal', 'Attack'],
                digits=4
            ))
            
            # Save confusion matrix
            cm = confusion_matrix(best_metrics['labels'], best_metrics['preds'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            cm_path = run_figure_dir / f"confusion_matrix_{timestamp}.png"
            plt.savefig(cm_path, bbox_inches='tight')
            plt.close()
            training_meta['confusion_matrix'] = str(cm_path)

        # Save final artifacts
        try:
            artifacts_saved = save_training_artifacts(
                model=model,
                metrics=best_metrics,
                config=training_meta,
                output_dir=run_artifact_dir,
                class_names=['Normal', 'Attack'],
                feature_names=artifacts.get('feature_names')
            )
            
            if not artifacts_saved:
                raise ModelSavingError("Failed to save some training artifacts")
                
        except Exception as e:
            raise ModelSavingError(f"Failed to save training artifacts: {str(e)}") from e

        writer.close()
        return {
            'metrics': best_metrics,
            'meta': training_meta,
            'artifacts_dir': str(run_artifact_dir)
        }

    except DataPreparationError as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise
    except TrainingError as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    except ModelSavingError as e:
        logger.error(f"Model saving failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise TrainingError(f"Unexpected training error: {str(e)}") from e

# Main entry point
if __name__ == "__main__":
    # Initialize colorama for colored console output
    init(autoreset=True)
    
    # Configure argument parser with enhanced help
    parser = argparse.ArgumentParser(
        description="Enhanced IDS Model Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use synthetic data for training (useful when real data is unavailable)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (more verbose output)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Maximum number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Initial learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Patience for early stopping (default: {EARLY_STOPPING_PATIENCE})"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive configuration mode"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base directory for output files"
    )
    args = parser.parse_args()

    try:
        # Initial system configuration
        configure_system()
        configure_visualization()
        set_seed(42)  # For reproducibility
        
        # Setup logging and directories with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(args.output_dir)
        
        try:
            directories = {
                'base': base_dir,
                'logs': base_dir / "logs",
                'models': base_dir / "models",
                'data': base_dir / "data",
                'figures': base_dir / "figures",
                'tensorboard': base_dir / "tensorboard",
                'checkpoints': base_dir / "checkpoints",
                'config': base_dir / "config",
                'results': base_dir / "results",
                'metrics': base_dir / "metrics",
                'reports': base_dir / "reports",
                'latest': base_dir / "latest",
                'info': base_dir / "info",
                'artifacts': base_dir / "artifacts"
            }
            
            # Create all directories
            for dir_path in directories.values():
                dir_path.mkdir(parents=True, exist_ok=True)
                
            # Set global directory variables
            MODEL_DIR = directories['models']
            LOG_DIR = directories['logs']
            DATA_DIR = directories['data']
            FIGURE_DIR = directories['figures']
            TB_DIR = directories['tensorboard']
            CHECKPOINT_DIR = directories['checkpoints']
            CONFIG_DIR = directories['config']
            RESULTS_DIR = directories['results']
            METRICS_DIR = directories['metrics']
            REPORTS_DIR = directories['reports']
            LATEST_DIR = directories['latest']
            INFO_DIR = directories['info']
            ARTIFACTS_DIR = directories['artifacts']
            
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Directory setup failed: {str(e)}")
            raise SystemExit(1) from e

        # Configure logging
        logger = setup_logging(LOG_DIR)
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
            torch._logging.set_logs(all=logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if args.interactive:
            # Interactive mode
            try:
                print(Fore.MAGENTA + Style.BRIGHT + "\n                    === Interactive Mode ===".center(60))
                interactive_main()
            except KeyboardInterrupt:
                print(Fore.YELLOW + Style.BRIGHT + "\n\nOperation cancelled by user")
                sys.exit(0)
            except Exception as e:
                logger.critical(
                    Fore.RED + Style.BRIGHT + f"Interactive session failed: {str(e)}", 
                    exc_info=args.debug
                )
                sys.exit(1)
        else:
            # Command-line mode execution
            logger.info(Fore.CYAN + Style.BRIGHT + "\n=== Enhanced Network Intrusion Detection Model Trainer ===")
            logger.info(Fore.GREEN + Style.BRIGHT + f"Starting training at {timestamp}")
            logger.info(Fore.MAGENTA + Style.BRIGHT + "Configuration:")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Batch size: {args.batch_size}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Epochs: {args.epochs}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Learning rate: {args.learning_rate}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Early stopping patience: {args.early_stopping}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Using {'synthetic' if args.use_mock else 'real'} data")
            
            # Prepare training configuration
            training_config = {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'early_stopping': args.early_stopping,
                'gradient_clip': GRADIENT_CLIP,
                'mixed_precision': MIXED_PRECISION
            }
            
            try:
                # Execute training
                results = train_model(
                    use_mock=args.use_mock,
                    config=training_config
                )
                
                # Final report
                logger.info(Fore.GREEN + Style.BRIGHT + "\n=== Training Completed Successfully ===")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Best validation accuracy: {results['metrics']['val_acc']:.2%}")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Best validation AUC: {results['metrics']['val_auc']:.4f}")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Artifacts saved to: {results['artifacts_dir']}")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Training time: {results['meta']['training_time']:.2f} seconds")
                
            except DataPreparationError as e:
                logger.error(Fore.RED + Style.BRIGHT + "\nData Preparation Failed:")
                logger.error(Fore.RED + Style.BRIGHT + f"Error: {str(e)}")
                if e.original_exception:
                    logger.debug(f"Original exception: {str(e.original_exception)}")
                sys.exit(1)
                
            except ModelConfigurationError as e:
                logger.error(Fore.RED + Style.BRIGHT + "\nModel Configuration Failed:")
                logger.error(Fore.RED + Style.BRIGHT + f"Error: {str(e)}")
                sys.exit(1)
                
            except TrainingExecutionError as e:
                logger.error(Fore.RED + Style.BRIGHT + "\nTraining Execution Failed:")
                logger.error(Fore.RED + Style.BRIGHT + f"Error at epoch {e.epoch if e.epoch else 'N/A'}: {str(e)}")
                sys.exit(1)
                
            except ModelSavingError as e:
                logger.error(Fore.RED + Style.BRIGHT + "\nModel Saving Failed:")
                logger.error(Fore.RED + Style.BRIGHT + f"Error: {str(e)}")
                sys.exit(1)
                
            except Exception as e:
                logger.critical(
                    Fore.RED + Style.BRIGHT + "\nUnexpected Error:",
                    exc_info=args.debug
                )
                sys.exit(1)

    except KeyboardInterrupt:
        print(Fore.YELLOW + Style.BRIGHT + "\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(
            Fore.RED + Style.BRIGHT + f"Fatal initialization error: {str(e)}",
            exc_info=args.debug
        )
        sys.exit(1)