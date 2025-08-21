# Standard library imports
import os
import json
import logging
import argparse
import platform
import shutil
import traceback
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Union, Any, Callable
from enum import Enum, auto
from copy import deepcopy
from collections import defaultdict, OrderedDict
from functools import wraps
import threading
import subprocess
import hashlib
import pickle
import gc
import re
import uuid
import tempfile

# Scientific computing and data manipulation
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# PyTorch ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms

# Hyperparameter optimization
import optuna
import optuna.visualization as vis
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
from optuna.storages import RDBStorage
import joblib

# Visualization and plotting
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Rich UI components for enhanced terminal interface
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, SpinnerColumn, track, ProgressColumn, TextColumn
from rich import box
from rich.text import Text
from rich.columns import Columns
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.syntax import Syntax
from rich.tree import Tree
from rich.layout import Layout
from rich.live import Live
from rich.spinner import Spinner
from rich.rule import Rule
from rich.align import Align
from rich.padding import Padding
from rich.markup import escape
from rich.status import Status, Spinner

# Terminal styling and colors
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

# Initialize rich console
console = Console()

# Configuration and serialization
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import toml
import configparser
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# Networking and API (for future remote capabilities)
import socket
import urllib.request
import urllib.parse
import urllib.error

# Parallel processing and concurrency
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
#import asyncio

# File handling and compression
import zipfile
import tarfile
import gzip
import lzma

# Additional utility libraries
import itertools
import random
import string
from contextlib import contextmanager, suppress
import weakref
from types import SimpleNamespace
import pathlib
from pynput.keyboard import Key, Listener
#import pkg_resources
import packaging.version
from packaging import version as pkg_version
from sklearn.exceptions import ConvergenceWarning
from matplotlib import MatplotlibDeprecationWarning

# Development and debugging tools
import inspect
import cProfile
import pstats

# Version checking utilities (safe, future-proof)
import sys
import warnings

# Use importlib.metadata (stdlib) or backport
try:
    from importlib.metadata import version as _get_version, PackageNotFoundError
    IMPORTLIB_METADATA_AVAILABLE = True
except ImportError:  # Python <3.8
    try:
        from importlib_metadata import version as _get_version, PackageNotFoundError  # type: ignore
        IMPORTLIB_METADATA_AVAILABLE = True
    except ImportError:
        IMPORTLIB_METADATA_AVAILABLE = False
        PackageNotFoundError = Exception  # type: ignore

# Packaging for version parsing/comparison
try:
    from packaging import version as pkg_version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False
    # Create a dummy version class for fallback
    class DummyVersion:
        def __init__(self, version_str):
            self.version_str = str(version_str)
        
        def __str__(self):
            return self.version_str
        
        def __lt__(self, other):
            return self.version_str < str(other)
        
        def __le__(self, other):
            return self.version_str <= str(other)
        
        def __eq__(self, other):
            return self.version_str == str(other)
        
        def __ge__(self, other):
            return self.version_str >= str(other)
        
        def __gt__(self, other):
            return self.version_str > str(other)
    
    def dummy_parse(version_str):
        return DummyVersion(version_str)
    
    # Fallback Dummy version module to mimic packaging.version
    pkg_version = type('DummyVersionModule', (), {'parse': dummy_parse})()

def safe_version(package_name: str) -> str:
    """
    Safely get the version of a package.
    Uses importlib.metadata, falls back to getattr(__version__),
    and returns 'N/A' if not found.
    """
    try:
        # Special cases where direct import is safer or required
        if package_name == "sklearn":
            import sklearn
            return sklearn.__version__
        if package_name == "torch":
            import torch
            return torch.__version__
        if package_name == "numpy":
            import numpy as np
            return np.__version__
        if package_name == "pandas":
            import pandas as pd
            return pd.__version__
        if package_name == "optuna":
            import optuna
            return optuna.__version__
        if package_name == "plotly":
            import plotly
            return plotly.__version__

        # Try importlib.metadata (preferred)
        if IMPORTLIB_METADATA_AVAILABLE:
            return _get_version(package_name)
        
        # Fallback: import the module and check __version__
        module = __import__(package_name)
        return getattr(module, "__version__", "unknown")
        
    except (PackageNotFoundError, ImportError):
        return "N/A"
    except Exception:
        return "unknown"

# Alias for backward compatibility with your existing code
get_package_version = safe_version

# Model export and optimization
import onnx

# Global flag and dummy module
ONNXRUNTIME_AVAILABLE = False
ort: Any = None

def initialize_onnx_runtime() -> bool:
    """
    Safely initialize ONNX Runtime with proper error handling.
    Returns True if ONNX Runtime is available and functional.
    """
    global ONNXRUNTIME_AVAILABLE, ort
    
    try:
        import onnxruntime as _ort
        # Test basic functionality
        try:
            # Simple test to verify DLL loading works
            _ort.get_device()
            ort = _ort
            ONNXRUNTIME_AVAILABLE = True
            return True
        except Exception as dll_error:
            warnings.warn(
                f"ONNX Runtime DLL load failed: {str(dll_error)}. "
                "ONNX validation will be disabled.",
                RuntimeWarning
            )
            create_dummy_ort()
            return False
    except ImportError:
        create_dummy_ort()
        return False

def create_dummy_ort() -> None:
    """Create a dummy ONNX Runtime module for compatibility."""
    global ort
    
    class DummyORT:
        __version__ = "N/A"
        
        class InferenceSession:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "ONNX Runtime not available. "
                    "Original error: DLL load failed"
                )
        
        @staticmethod
        def get_available_providers() -> list:
            return []
        
        @staticmethod
        def get_device() -> str:
            return "CPU (ONNX Runtime not available)"
    
    ort = DummyORT()

# Initialize at module level
initialize_onnx_runtime()

try:
    from torch.jit import script, trace
    TORCH_JIT_AVAILABLE = True
except ImportError:
    TORCH_JIT_AVAILABLE = False

# System monitoring and profiling
import psutil
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Cryptography and security (for model protection)
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Database connectivity (for advanced storage)
try:
    import sqlite3
    import sqlalchemy
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Additional ML libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_ANOMALY_AVAILABLE = True
except ImportError:
    SKLEARN_ANOMALY_AVAILABLE = False

# Time series analysis (for temporal anomaly detection)
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Advanced numerical computation
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Profiling tools
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Availability flags for optional dependencies
OPTIONAL_DEPENDENCIES = {
    'torch_jit': TORCH_JIT_AVAILABLE,
    'onnxruntime': ONNXRUNTIME_AVAILABLE,
    'nvml': NVML_AVAILABLE,
    'crypto': CRYPTO_AVAILABLE,
    'database': DATABASE_AVAILABLE,
    'sklearn_anomaly': SKLEARN_ANOMALY_AVAILABLE,
    'statsmodels': STATSMODELS_AVAILABLE,
    'numba': NUMBA_AVAILABLE,
    'memory_profiler': MEMORY_PROFILER_AVAILABLE,
    'line_profiler': LINE_PROFILER_AVAILABLE,
    'packaging': PACKAGING_AVAILABLE,
    'importlib_metadata': IMPORTLIB_METADATA_AVAILABLE
}

# Version information - Updated to properly detect Rich
VERSION_INFO = {
    'python': sys.version.split()[0],
    'torch': safe_version('torch'),
    'numpy': safe_version('numpy'),
    'pandas': safe_version('pandas'),
    'optuna': safe_version('optuna'),
    'rich': safe_version('rich'),
    'plotly': safe_version('plotly'),
    'sklearn': safe_version('sklearn'),
    'onnx': safe_version('onnx'),
    'psutil': safe_version('psutil')
}

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = "deep_learning.log"
LOG_FILE = LOG_DIR / LOG_FILE_NAME

# Configure directories
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_DIR.mkdir(exist_ok=True)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE_NAME = "deep_learning_config.json"
CONFIG_FILE = CONFIG_DIR / CONFIG_FILE_NAME

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

TB_DIR = Path("tensorboard")
TB_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize logger at module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configuration Constants
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = True

# Model Architecture Constants
DEFAULT_ENCODING_DIM = 10
HIDDEN_LAYER_SIZES = [128, 64]
DROPOUT_RATES = [0.2, 0.15]
ACTIVATION = 'leaky_relu'
ACTIVATION_PARAM = 0.2
NORMALIZATION = 'batch'
USE_BATCH_NORM = True
USE_LAYER_NORM = False
DIVERSITY_FACTOR = 0.1
MIN_FEATURES = 5
NUM_MODELS = 3
FEATURES = 20
NORMALIZATION_OPTIONS = ['batch', 'layer', None]
NORMAL_SAMPLES = 8000
ATTACK_SAMPLES = 2000
ANOMALY_FACTOR = 1.5
RANDOM_STATE = 42

# Security Constants
DEFAULT_PERCENTILE = 95
DEFAULT_ATTACK_THRESHOLD = 0.3
FALSE_NEGATIVE_COST = 2.0
SECURITY_METRICS = True

# System Constants
NUM_WORKERS = min(4, os.cpu_count() or 1)
MAX_MEMORY_PERCENT = 80
# Cache timeout for model artifacts in seconds
CACHE_TIMEOUT = 3600

# Logging and Directory Setup
# Stream handler for unicode output on windows
class UnicodeStreamHandler(logging.StreamHandler):
    """
    A logging StreamHandler that preserves Unicode output on Windows consoles,
    falling back to ASCII-safe output if encoding errors occur.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            if sys.platform == 'win32':
                try:
                    self.stream.write(msg + self.terminator)
                except UnicodeEncodeError:
                    # Fallback to ASCII-only output
                    msg = msg.encode('ascii', errors='replace').decode('ascii')
                    self.stream.write(msg + self.terminator)
            else:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Setup logging configuration
def setup_logging(log_dir: Path = None) -> logging.Logger:
    """
    Configure the logger with:
    - UTF-8 file handler
    - Unicode-safe console handler (via UnicodeStreamHandler)

    Features:
    1. If log_dir is None, defaults to 'logs' folder next to this script.
    2. Adds handlers only if they don't already exist.
    3. Falls back to basic logging config if setup fails.
    """
    try:
        # Determine log_dir (default: script's directory / logs)
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # File Handler
        log_file = log_dir / "deep_learning.log"
        file_handler_exists = any(
            isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file)
            for h in logger.handlers
        )
        if not file_handler_exists:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Unicode-Safe Console Handler
        console_handler_exists = any(isinstance(h, UnicodeStreamHandler) for h in logger.handlers)
        if not console_handler_exists:
            console_handler = UnicodeStreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    except Exception as e:
        # Fallback basic configuration if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to setup proper logging: {e}")
        return logger

# Setup directories and assign global directory variables
def setup_directories(logger: logging.Logger) -> Dict[str, Path]:
    """Create and return essential directories with versioned subdirectories."""
    base_dir = Path(__file__).resolve().parent
    
    dirs = {
        'logs': base_dir / "logs",
        'models': base_dir / "models",
        'data': base_dir / "data",
        'config': base_dir / "config",
        'reports': base_dir / "reports",
        'tensorboard': base_dir / "tensorboard",
        'cache': base_dir / "cache",
        'exports': base_dir / "exports"
    }
    
    # Create directories
    for name, path in dirs.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            if logger:
                logger.debug(f"Created directory: {path}")
        except PermissionError as e:
            if logger:
                logger.error(f"Permission denied creating directory {path}: {e}")
            raise
        except Exception as e:
            if logger:
                logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    return dirs

def configure_directories(logger: logging.Logger) -> Dict[str, Path]:
    """
    Initialize and assign global directory path variables using
    setup_directories(), ensuring they are accessible across modules.
    """
    try:
        dirs = setup_directories(logger)
        
        # Assign to global variables if needed
        global LOG_DIR, DEFAULT_MODEL_DIR, DATA_DIR, CONFIG_DIR, REPORTS_DIR, TB_DIR, CACHE_DIR
        LOG_DIR = dirs['logs']
        DEFAULT_MODEL_DIR = dirs['models']
        DATA_DIR = dirs['data']
        CONFIG_DIR = dirs['config']
        REPORTS_DIR = dirs['reports']
        TB_DIR = dirs['tensorboard']
        CACHE_DIR = dirs['cache']
        
        if logger:
            logger.info("Directory configuration completed successfully")
        
        return dirs
        
    except Exception as e:
        if logger:
            logger.critical(f"Directory configuration failed: {e}")
        raise

# Forward declarations for classes that will be defined later
class SimpleAutoencoder:
    """Forward declaration for SimpleAutoencoder class."""
    pass

class EnhancedAutoencoder:
    """Forward declaration for EnhancedAutoencoder class."""
    pass

class AutoencoderEnsemble:
    """Forward declaration for AutoencoderEnsemble class."""
    pass

def setup_safe_globals():
    """
    Register a curated list of safe, version-stable classes for torch.load
    to prevent unpickling errors across different PyTorch, NumPy, and Pandas
    versions, including fallbacks for older/newer library structures.
    """
    # Dictionary of safe classes with their full import paths
    safe_classes = {
        # PyTorch core classes
        'torch.Tensor': torch.Tensor,
        'torch.nn.Module': torch.nn.Module,
        'torch.nn.parameter.Parameter': torch.nn.parameter.Parameter,
        'torch.FloatTensor': torch.FloatTensor,
        'torch.LongTensor': torch.LongTensor,
        'torch.IntTensor': torch.IntTensor,
        'torch.DoubleTensor': torch.DoubleTensor,
        
        # PyTorch optimizers and schedulers
        'torch.optim.Optimizer': torch.optim.Optimizer,
        'torch.optim.AdamW': torch.optim.AdamW,
        'torch.optim.SGD': torch.optim.SGD,
        'torch.optim.lr_scheduler.ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        
        # PyTorch data loading
        'torch.utils.data.Dataset': torch.utils.data.Dataset,
        'torch.utils.data.DataLoader': torch.utils.data.DataLoader,
        'torch.utils.data.TensorDataset': torch.utils.data.TensorDataset,
        
        # Path handling classes (critical for the error you're seeing)
        'pathlib.Path': pathlib.Path,
        'pathlib.WindowsPath': pathlib.WindowsPath,
        'pathlib.PosixPath': pathlib.PosixPath,
        
        # Version handling
        'torch.torch_version.TorchVersion': torch.torch_version.TorchVersion,
        
        # NumPy core classes
        'numpy.ndarray': np.ndarray,
        'numpy.float32': np.float32,
        'numpy.float64': np.float64,
        'numpy.int32': np.int32,
        'numpy.int64': np.int64,
        'numpy.dtype': np.dtype,
        'numpy.number': np.number,
        
        # Python built-ins
        'builtins.dict': dict,
        'builtins.list': list,
        'builtins.tuple': tuple,
        'builtins.set': set,
    }

    # Add custom model classes if they exist
    try:
        from models import SimpleAutoencoder, EnhancedAutoencoder, AutoencoderEnsemble
        safe_classes.update({
            'models.SimpleAutoencoder': SimpleAutoencoder,
            'models.EnhancedAutoencoder': EnhancedAutoencoder,
            'models.AutoencoderEnsemble': AutoencoderEnsemble
        })
    except ImportError:
        pass

    # Special handling for PyTorch storage types
    storage_types = [
        'torch.FloatStorage',
        'torch.LongStorage',
        'torch.IntStorage',
        'torch.DoubleStorage',
    ]
    
    for stype in storage_types:
        try:
            if hasattr(torch, stype):
                safe_classes[f'torch.{stype}'] = getattr(torch, stype)
        except Exception:
            pass

    # Register all safe classes using torch.serialization.add_safe_globals
    try:
        # Convert values to list for add_safe_globals
        safe_objects = list(safe_classes.values())
        torch.serialization.add_safe_globals(safe_objects)
        
        # Additional numpy-specific reconstruction functions
        numpy_reconstructors = []
        try:
            # For numpy >= 1.20 (new structure)
            if hasattr(np, '_core') and hasattr(np._core, 'multiarray'):
                numpy_reconstructors.extend([
                    np._core.multiarray._reconstruct,
                    np._core.multiarray.scalar,
                    np._core.multiarray.array,
                ])
            # For numpy < 1.20 (old structure)
            elif hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
                numpy_reconstructors.extend([
                    np.core.multiarray._reconstruct,
                    np.core.multiarray.scalar,
                    np.core.multiarray.array,
                ])
        except AttributeError:
            pass
        
        if numpy_reconstructors:
            torch.serialization.add_safe_globals(numpy_reconstructors)

        # Add PyTorch internal reconstruction functions
        torch_reconstructors = []
        try:
            torch_reconstructors.extend([
                torch._utils._rebuild_tensor_v2,
                torch._utils._rebuild_parameter,
                torch._utils._rebuild_tensor,
            ])
        except AttributeError:
            pass
        
        if torch_reconstructors:
            torch.serialization.add_safe_globals(torch_reconstructors)

    except Exception as e:
        logging.warning(f"Failed to register some safe globals: {str(e)}")

# Setup safe globals at module level
setup_safe_globals()

# Disable PyTorch's duplicate logging
torch._logging.set_logs(all=logging.ERROR)

# Loading Screen and System Check Framework
class CheckLevel(Enum):
    """Enumeration representing the severity of a system check."""
    # Check must pass for the program to continue running
    CRITICAL = auto()
    
    # Check should pass for full functionality but not fatal
    IMPORTANT = auto()
    
    # Non-essential check providing useful system information
    INFORMATIONAL = auto()

class CheckResult:
    """Encapsulates the outcome of a system check with enhanced functionality."""
    
    def __init__(self, 
                 passed: bool, 
                 message: str, 
                 level: CheckLevel = CheckLevel.IMPORTANT,
                 details: Optional[Union[str, Dict[str, Any]]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 exception: Optional[Exception] = None):
        self.passed = passed
        self.message = message
        self.level = level
        self.details = details
        self.metadata = metadata if metadata is not None else {}
        self.exception = exception

    def with_details(self, details: Union[str, Dict[str, Any]]) -> 'CheckResult':
        """Return CheckResult with additional details."""
        self.details = details
        return self
    
    def with_exception(self, exception: Exception) -> 'CheckResult':
        """Return CheckResult with an exception."""
        self.exception = exception
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'CheckResult':
        """Return CheckResult with additional metadata."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(metadata)
        return self
    
    def with_passed(self, passed: bool) -> 'CheckResult':
        """Update the passed status and return self."""
        self.passed = passed
        return self
    
    def with_message(self, message: str) -> 'CheckResult':
        """Update the message and return self."""
        self.message = message
        return self
    
    def with_level(self, level: CheckLevel) -> 'CheckResult':
        """Update the check level and return self."""
        self.level = level
        return self

def loading_screen(
    logger: logging.Logger,
    extended: bool = False,
    include_performance: bool = False
) -> bool:
    """
    Display loading screen with system checks and interactive prompts.
    
    Args:
        logger: Logger for recording system check results
        extended: Whether to run extended initialization-specific checks
        include_performance: Whether to include performance-related checks
        
    Returns:
        bool: True if all critical checks pass and user chooses to continue,
              False if critical checks fail or user chooses to quit
    """
    # Thread safety lock
    _loading_lock = threading.RLock()
    
    with _loading_lock:
        try:
            # Console safety checks
            if not hasattr(console, 'width'):
                console_width = 80  # Safe default
            else:
                console_width = max(60, getattr(console, 'width', 80))  # Minimum width
            
            # Terminal capability detection
            is_tty = sys.stdout.isatty()
            supports_color = is_tty and hasattr(sys.stdout, 'isatty')
            
            # Safe console clear
            try:
                if is_tty:
                    console.clear()
                else:
                    console.print("\n" * 3)  # Fallback for non-TTY
            except Exception:
                console.print("\n" * 3)  # Safe fallback
            
            # Initialize timing with thread-safe approach
            start_time = time.perf_counter()
            status_messages = [
                "Running System Diagnostics...",
                "Initializing system checks...",
                "Validating environment...",
                "Executing system checks..."
            ]
            
            # Non-blocking loading animation with proper status management
            current_status = None
            try:
                # Sequential status updates to avoid context conflicts
                for i, message in enumerate(status_messages):
                    if current_status:
                        current_status.stop()
                    
                    if is_tty:
                        current_status = console.status(
                            f"[bold blue]{message}[/bold blue]" if supports_color else message,
                            spinner="dots" if is_tty else None
                        )
                        current_status.start()
                        time.sleep(0.3 + i * 0.1)  # Progressive timing
                    else:
                        console.print(f"• {message}")
                        time.sleep(0.1)  # Minimal delay for non-TTY
            finally:
                if current_status:
                    current_status.stop()
                    current_status = None
            
            # ASCII art banner with width adaptation
            banner_width = min(console_width - 8, 100)
            ascii_art = """
        ⠀⠀⠀⢠⣾⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⣰⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⢰⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣄⣀⣀⣤⣤⣶⣾⣿⣿⣿⡷
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀
        ⣿⣿⣿⡇⠀⡾⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀
        ⣿⣿⣿⣧⡀⠁⣀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠉⢹⠉⠙⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣀⠀⣀⣼⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠀⠤⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⠿⣿⣿⣿⣿⣿⣿⣿⠿⠋⢃⠈⠢⡁⠒⠄⡀⠈⠁⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⠟⠁⠀⠀⠈⠉⠉⠁⠀⠀⠀⠀⠈⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀
            """
            
            # Safe banner display with width adaptation
            try:
                if banner_width > 80 and supports_color:
                    console.print("\n", Panel.fit(
                        ascii_art,
                        style="bold cyan" if supports_color else "",
                        title="[bold yellow]GreyChamp | IDS[/]" if supports_color else "GreyChamp | IDS",
                        subtitle="[magenta]SYSTEM INITIALIZATION[/]" if supports_color else "SYSTEM INITIALIZATION",
                        border_style="bold blue" if supports_color else "ascii",
                        box=box.DOUBLE if supports_color else box.ASCII,
                        padding=(1, 1),
                        width=min(banner_width, console_width - 4)
                    ))
                else:
                    # Simple fallback for narrow terminals
                    console.print("\n" + "=" * min(60, banner_width))
                    console.print("    GreyChamp | IDS - SYSTEM INITIALIZATION")
                    console.print("=" * min(60, banner_width) + "\n")
            except Exception as banner_error:
                # Ultra-safe fallback
                console.print("\nGreyChamp | IDS - SYSTEM INITIALIZATION\n")
                if logger:
                    logger.debug(f"Banner display failed: {banner_error}")
            
            # Check type information display
            check_type_info = "BASIC CHECKS"
            if extended and include_performance:
                check_type_info = "EXTENDED CHECKS (with Performance)"
            elif extended:
                check_type_info = "EXTENDED CHECKS"
            
            try:
                if supports_color and banner_width > 60:
                    console.print(Panel.fit(
                        f"[bold cyan]Running {check_type_info}[/bold cyan]\n"
                        "[dim]Please wait while we validate your system...[/dim]",
                        border_style="cyan",
                        padding=(0, 2),
                        width=min(banner_width, console_width - 4)
                    ))
                else:
                    console.print(f"\nRunning {check_type_info}")
                    console.print("Please wait while we validate your system...\n")
            except Exception:
                console.print(f"\nRunning {check_type_info}\n")
            
            # Thread-safe system checks execution
            console.print("Executing system checks..." if not supports_color else "[dim]Executing system checks...[/dim]")
            
            # Use thread-safe timing measurement
            checks_start = time.perf_counter()
            results = run_system_checks(logger, extended=extended, include_performance=include_performance)
            elapsed_time = time.perf_counter() - checks_start
            
            # Thread-safe results processing
            if "system_summary" in results and results["system_summary"].details:
                if isinstance(results["system_summary"].details, dict):
                    results["system_summary"].details["execution_time"] = f"{elapsed_time:.2f}s"
            
            # Display results with error handling
            console.print()
            try:
                display_check_results(results, logger, extended=extended, include_performance=include_performance)
            except Exception as display_error:
                console.print(f"[red]Error displaying results: {display_error}[/red]" if supports_color else f"Error displaying results: {display_error}")
                if logger:
                    logger.error(f"Failed to display check results: {display_error}")
            console.print()
            
            # Safe results analysis
            summary = results.get("system_summary")
            system_error = results.get("system_error")
            
            # Determine system status with null safety
            system_status = "UNKNOWN"
            if summary and summary.details and isinstance(summary.details, dict):
                system_status = summary.details.get('system_status', 'UNKNOWN')
            
            # Count failures by level with safety checks
            critical_failed = sum(1 for result in results.values() 
                                if result and hasattr(result, 'level') and hasattr(result, 'passed') and
                                result.level == CheckLevel.CRITICAL and not result.passed 
                                and result != summary)
            
            important_failed = sum(1 for result in results.values() 
                                 if result and hasattr(result, 'level') and hasattr(result, 'passed') and
                                 result.level == CheckLevel.IMPORTANT and not result.passed)
            
            informational_failed = sum(1 for result in results.values() 
                                     if result and hasattr(result, 'level') and hasattr(result, 'passed') and
                                     result.level == CheckLevel.INFORMATIONAL and not result.passed)
            
            # Handle different scenarios with proper cleanup
            return_value = False
            
            try:
                if system_error or system_status == "CRITICAL_FAILURE" or critical_failed > 0:
                    # Critical failure - system cannot continue
                    failed_critical_checks = [
                        name.replace("_", " ").title() 
                        for name, result in results.items() 
                        if (result and hasattr(result, 'level') and hasattr(result, 'passed') and
                            result.level == CheckLevel.CRITICAL and not result.passed and 
                            name != "system_summary")
                    ]
                    
                    error_message = (
                        f"CRITICAL SYSTEM CHECKS FAILED\n\n"
                        f"The system cannot continue due to critical failures.\n"
                        f"Failed checks: {', '.join(failed_critical_checks) if failed_critical_checks else 'System error occurred'}\n\n"
                        f"Please check the logs and resolve these issues before continuing."
                    )
                    
                    try:
                        if supports_color and banner_width > 60:
                            console.print(Panel.fit(
                                f"[bold red]{error_message}[/bold red]",
                                border_style="red",
                                title="Critical Failure",
                                padding=(1, 3),
                                width=min(banner_width, console_width - 4)
                            ))
                        else:
                            console.print(f"\nCRITICAL FAILURE:\n{error_message}")
                    except Exception:
                        console.print(f"\nCRITICAL FAILURE:\n{error_message}")
                    
                    if logger:
                        logger.critical(f"Critical system checks failed - cannot continue. Failed checks: {failed_critical_checks}")
                    
                    return_value = False
                    
                elif system_status in ["DEGRADED", "LIMITED"] or important_failed > 0 or informational_failed > 0:
                    # Non-critical failures - user decision with proper input handling
                    user_choice = _handle_user_decision_safe(
                        results, system_status, important_failed, informational_failed, 
                        elapsed_time, supports_color, banner_width, console_width, logger
                    )
                    
                    if user_choice is False:
                        return_value = False
                        # Cleanup handled in _handle_user_decision_safe
                    else:
                        return_value = True
                        
                else:
                    # All checks passed - success scenario
                    return_value = _handle_success_scenario(
                        summary, elapsed_time, supports_color, banner_width, console_width, logger
                    )
            
            except Exception as scenario_error:
                console.print(f"Error handling system check results: {scenario_error}")
                if logger:
                    logger.error(f"Error in scenario handling: {scenario_error}")
                return_value = False
            
            # Safe console clear before return
            try:
                if return_value and is_tty:
                    console.clear()
            except Exception:
                pass  # Ignore clear failures
            
            return return_value
            
        except KeyboardInterrupt:
            # Thread-safe interrupt handling
            try:
                console.print(Panel.fit(
                    "INITIALIZATION INTERRUPTED\n\n"
                    "System initialization was cancelled by user.",
                    border_style="red" if supports_color else "ascii",
                    title="Interrupted",
                    padding=(1, 3)
                ) if supports_color else "\nINITIALIZATION INTERRUPTED\n\nSystem initialization was cancelled by user.\n")
            except Exception:
                console.print("\nINITIALIZATION INTERRUPTED\n\nSystem initialization was cancelled by user.\n")
            
            if logger:
                logger.warning("System initialization interrupted by user (Ctrl+C)")
            
            sys.exit(0)
            
        except Exception as e:
            # Thread-safe error handling
            error_msg = f"UNEXPECTED ERROR DURING INITIALIZATION\n\nAn unexpected error occurred: {str(e)}\nError type: {type(e).__name__}"
            
            try:
                if supports_color:
                    console.print(Panel.fit(
                        f"[bold red]{error_msg}[/bold red]",
                        border_style="red",
                        title="System Error",
                        padding=(1, 3)
                    ))
                else:
                    console.print(f"\nSYSTEM ERROR:\n{error_msg}\n")
            except Exception:
                console.print(f"\nSYSTEM ERROR:\n{error_msg}\n")
            
            if logger:
                logger.critical(f"Loading screen failed with unexpected error: {str(e)}", exc_info=True)
                logger.error(f"Error occurred during {'extended' if extended else 'basic'} system checks")
            
            return False

def _handle_user_decision_safe(results, system_status, important_failed, informational_failed, 
                              elapsed_time, supports_color, banner_width, console_width, logger):
    """Thread-safe user decision handling with proper resource cleanup."""
    try:
        # Collect failed non-critical checks safely
        failed_checks = []
        for name, result in results.items():
            if (result and hasattr(result, 'passed') and hasattr(result, 'level') and 
                hasattr(result, 'message') and not result.passed and 
                result.level in [CheckLevel.IMPORTANT, CheckLevel.INFORMATIONAL] and 
                name not in ["system_summary", "system_error"]):
                failed_checks.append({
                    'name': name.replace("_", " ").title(),
                    'level': result.level.name,
                    'message': result.message
                })
        
        # Display failed checks summary safely
        if failed_checks:
            try:
                if supports_color and banner_width > 80:
                    fail_table = Table(
                        title="[bold yellow]Failed Non-Critical Checks[/bold yellow]",
                        box=box.SIMPLE,
                        header_style="bold magenta",
                        title_justify="left",
                        show_header=True,
                        show_lines=True,
                        width=min(100, console_width - 4)
                    )
                    fail_table.add_column("Check", style="bold cyan", width=28)
                    fail_table.add_column("Issue", style="bold white", no_wrap=False)
                    fail_table.add_column("Level", justify="center", width=14)
                    
                    for check in failed_checks:
                        level_style = {
                            "IMPORTANT": "bold yellow",
                            "INFORMATIONAL": "bold blue"
                        }.get(check['level'], "white")
                        
                        fail_table.add_row(
                            check['name'],
                            check['message'],
                            Text(check['level'], style=level_style)
                        )
                    
                    console.print(fail_table)
                else:
                    # Simple fallback display
                    console.print("Failed Non-Critical Checks:")
                    for check in failed_checks:
                        console.print(f"  - {check['name']}: {check['message']} ({check['level']})")
                
                console.print()
                
            except Exception as table_error:
                # Ultra-safe fallback
                console.print("Some non-critical checks failed:")
                for check in failed_checks:
                    console.print(f"  {check['name']}: {check['message']}")
                console.print()
                if logger:
                    logger.debug(f"Failed checks table display error: {table_error}")
        
        # Status display with safe formatting
        status_color = "yellow" if system_status == "DEGRADED" else "blue" if system_status == "LIMITED" else "yellow"
        status_message = {
            "DEGRADED": "SYSTEM DEGRADED",
            "LIMITED": "LIMITED FUNCTIONALITY", 
        }.get(system_status, "SOME CHECKS FAILED")
        
        prompt_text = (
            f"{status_message}\n\n"
            f"System Status Details:\n"
            f"- Important failures: {important_failed}\n"
            f"- Informational failures: {informational_failed}\n"
            f"- Total execution time: {elapsed_time:.2f}s\n\n"
            f"The system can continue with reduced functionality.\n"
            f"Press Enter to continue anyway or Esc to quit and resolve issues"
        )
        
        try:
            if supports_color and banner_width > 60:
                console.print(Panel.fit(
                    f"[bold {status_color}]{prompt_text}[/bold {status_color}]" if supports_color else prompt_text,
                    border_style=status_color if supports_color else "ascii",
                    title="User Decision Required",
                    padding=(1, 3),
                    width=min(banner_width, console_width - 4)
                ))
            else:
                console.print(f"\n{status_message}\n")
                console.print(prompt_text)
        except Exception:
            console.print(f"\n{status_message}\n")
            console.print(prompt_text)
        
        # Thread-safe keyboard input with timeout and cleanup
        user_choice = None
        listener = None
        listener_thread = None
        
        def safe_key_handler(key):
            nonlocal user_choice
            try:
                if key == Key.enter:
                    user_choice = True
                    return False  # Stop listener
                elif key == Key.esc:
                    user_choice = False
                    return False  # Stop listener
                elif hasattr(key, 'char') and key.char:
                    char = key.char.lower()
                    if char in ['q', 'n']:  # Quit/No
                        user_choice = False
                        return False
                    elif char in ['c', 'y']:  # Continue/Yes
                        user_choice = True
                        return False
            except Exception:
                pass  # Ignore key handling errors
            return True  # Continue listening
        
        # Safe input handling with timeout
        console.print("[dim]Waiting for user input... (Enter to continue, Esc to quit)[/dim]" if supports_color else "Waiting for user input... (Enter to continue, Esc to quit)")
        
        try:
            # Use a timeout to prevent infinite waiting
            listener = Listener(on_press=safe_key_handler)
            listener.start()
            
            # Wait with timeout
            start_wait = time.perf_counter()
            timeout_seconds = 300  # 5 minute timeout
            
            while user_choice is None and (time.perf_counter() - start_wait) < timeout_seconds:
                time.sleep(0.1)
                if not listener.running:
                    break
            
            # Timeout handling
            if user_choice is None:
                user_choice = True  # Default to continue on timeout
                console.print("[yellow]Input timeout - continuing with warnings[/yellow]" if supports_color else "Input timeout - continuing with warnings")
                
        except Exception as input_error:
            user_choice = True  # Default to continue on input error
            console.print(f"Input error - continuing with warnings: {input_error}")
            if logger:
                logger.warning(f"User input error: {input_error}")
        finally:
            # Cleanup listener safely
            try:
                if listener:
                    listener.stop()
            except Exception:
                pass
        
        # Handle user choice with safe output
        if user_choice is False:
            try:
                cancel_message = (
                    "USER CANCELLED INITIALIZATION\n\n"
                    "You chose to quit and resolve the issues.\n"
                    "Please check the logs and fix the failed checks."
                )
                
                if supports_color and banner_width > 60:
                    console.print(Panel.fit(
                        f"[bold red]{cancel_message}[/bold red]",
                        border_style="red",
                        title="Cancelled",
                        padding=(1, 3),
                        width=min(banner_width, console_width - 4)
                    ))
                else:
                    console.print(f"\nCANCELLED:\n{cancel_message}")
            except Exception:
                console.print(f"\nCANCELLED:\n{cancel_message}")
            
            if logger:
                logger.warning("User chose to quit after seeing failed checks")
                logger.info(f"Failed checks summary: {len(failed_checks)} non-critical failures")
            
            sys.exit(0)
        
        # User chose to continue
        try:
            continue_message = (
                "CONTINUING WITH WARNINGS\n\n"
                "You chose to continue despite the warnings.\n"
                "Some functionality may be limited."
            )
            
            if supports_color and banner_width > 60:
                console.print(Panel.fit(
                    f"[bold green]{continue_message}[/bold green]",
                    border_style="green",
                    title="Continuing",
                    padding=(1, 2),
                    width=min(banner_width, console_width - 4)
                ))
            else:
                console.print(f"\nCONTINUING:\n{continue_message}")
        except Exception:
            console.print(f"\nCONTINUING:\n{continue_message}")
        
        if logger:
            logger.info("User chose to continue despite failed checks")
            logger.info(f"System status: {system_status} with {len(failed_checks)} failed checks")
        
        # Brief pause before clearing
        time.sleep(1)
        return True
        
    except Exception as decision_error:
        if logger:
            logger.error(f"Error in user decision handling: {decision_error}")
        console.print(f"Error in user input - continuing with warnings: {decision_error}")
        return True  # Default to continue on error

def _handle_success_scenario(summary, elapsed_time, supports_color, banner_width, console_width, logger):
    """Handle successful system checks scenario with safe input."""
    try:
        success_details = ""
        if summary and summary.details and isinstance(summary.details, dict):
            total_checks = summary.details.get('total_checks', 0)
            success_details = f"\nCompleted {total_checks} checks successfully"
        
        success_message = (
            f"ALL SYSTEM CHECKS PASSED\n"
            f"System is fully operational and ready!\n"
            f"Completed in {elapsed_time:.2f} seconds"
            f"{success_details}\n\n"
            f"Press Enter to continue"
        )
        
        try:
            if supports_color and banner_width > 60:
                console.print(Panel.fit(
                    f"[bold green]{success_message}[/bold green]",
                    border_style="green",
                    title="Success",
                    padding=(1, 3),
                    width=min(banner_width, console_width - 4)
                ))
            else:
                console.print(f"\nSUCCESS:\n{success_message}")
        except Exception:
            console.print(f"\nSUCCESS:\n{success_message}")
        
        # Safe success input handling
        console.print("[dim green]Ready to proceed...[/dim green]" if supports_color else "Ready to proceed...")
        
        listener = None
        try:
            def success_key_handler(key):
                if key == Key.enter:
                    return False  # Stop listener
                return True  # Continue listening
            
            listener = Listener(on_press=success_key_handler)
            listener.start()
            
            # Wait with timeout
            start_wait = time.perf_counter()
            while listener.running and (time.perf_counter() - start_wait) < 30:  # 30 second timeout
                time.sleep(0.1)
                
        except Exception as success_input_error:
            if logger:
                logger.debug(f"Success input error: {success_input_error}")
        finally:
            try:
                if listener:
                    listener.stop()
            except Exception:
                pass
        
        if logger:
            logger.info(f"All system checks passed successfully in {elapsed_time:.2f}s")
            if summary and summary.details:
                system_status = summary.details.get('system_status', 'OPTIMAL')
                logger.info(f"System status: {system_status}")
        
        return True
        
    except Exception as success_error:
        if logger:
            logger.error(f"Error in success scenario: {success_error}")
        console.print("All checks passed - continuing...")
        return True

def run_system_checks(
    logger: logging.Logger, 
    extended: bool = False,
    include_performance: bool = False
) -> Dict[str, CheckResult]:
    """
    Run comprehensive system checks with optional extended validations.
    
    Args:
        logger: Configured logger for recording check results
        extended: Whether to include initialization-specific checks
        include_performance: Whether to include performance-related checks
        
    Returns:
        Dictionary mapping check names to their CheckResult objects
    """
    checks: Dict[str, CheckResult] = {}
    
    try:
        # Core system checks (always run)
        raw_checks = {
            # Critical checks (essential for operation)
            'python_version': check_python_version(),
            'torch_available': check_torch(),
            
            # Important checks (affects functionality but not critical)
            'package_versions': check_package_versions_wrapper(),
            'directory_access': check_directory_access_wrapper(),
            'disk_space': check_disk_space(),
            'cuda_available': check_cuda(),
            
            # Hardware resource checks
            'hardware': check_hardware(),
            
            # Informational checks (diagnostic purposes)
            'cpu_cores': check_cpu_cores(),
            'system_ram': check_system_ram(),
            'system_arch': check_system_architecture(),
            'logging_setup': check_logging_setup(),
            'seed_config': check_seed_config()
        }

        # Extended system checks (when requested)
        if extended:
            raw_checks.update({
                'exception_handler': check_global_exception_handler(),
                'configuration_system': check_configuration_system(),
                'model_variants': check_model_variants()
            })
            
            # Performance-related checks (only when explicitly requested)
            if include_performance:
                raw_checks.update({
                    'performance_monitoring': check_performance_monitoring(),
                    'memory_management': check_memory_management(),
                    'performance_baseline': check_performance_baseline()
                })
        
        # Convert all results to CheckResult objects if they aren't already
        for name, result in raw_checks.items():
            if isinstance(result, CheckResult):
                checks[name] = result
            elif isinstance(result, dict):
                # Convert dictionary result to CheckResult
                checks[name] = CheckResult(
                    passed=result.get('passed', False),
                    message=result.get('message', f"Check {name} completed"),
                    # Default level, should be overridden by specific checks
                    level=CheckLevel.INFORMATIONAL,
                    details=result.get('details', result)
                )
            else:
                # Handle unexpected result types
                checks[name] = CheckResult(
                    passed=False,
                    message=f"Invalid result type for {name}: {type(result)}",
                    level=CheckLevel.CRITICAL,
                    details={'raw_result': str(result), 'result_type': str(type(result))}
                )
        
        # Calculate overall system status
        critical_checks = [
            result for result in checks.values() 
            if result.level in {CheckLevel.CRITICAL, CheckLevel.IMPORTANT}
        ]
        overall_passed = all(result.passed for result in critical_checks)
        
        # Determine system status based on failures
        critical_failures = sum(1 for r in checks.values() if not r.passed and r.level == CheckLevel.CRITICAL)
        important_failures = sum(1 for r in checks.values() if not r.passed and r.level == CheckLevel.IMPORTANT)
        
        if critical_failures > 0:
            system_status = "CRITICAL_FAILURE"
        elif important_failures > 0:
            system_status = "DEGRADED"
        elif any(not r.passed for r in checks.values()):
            system_status = "LIMITED"
        else:
            system_status = "OPTIMAL"
        
        # Create comprehensive summary
        summary_details = {
            'total_checks': len(checks),
            'passed_checks': sum(1 for r in checks.values() if r.passed),
            'failed_checks': sum(1 for r in checks.values() if not r.passed),
            'critical_failures': critical_failures,
            'important_failures': important_failures,
            'system_status': system_status,
            'check_results': {
                name: {
                    'passed': result.passed,
                    'message': result.message,
                    'level': result.level.name,
                    'details': result.details if isinstance(result.details, (str, dict)) else str(result.details)
                }
                for name, result in checks.items()
            }
        }
        
        summary_message = (
            f"{system_status}: Extended system check summary" if extended 
            else f"{system_status}: Basic system check summary"
        )
        
        checks['system_summary'] = CheckResult(
            passed=overall_passed,
            message=summary_message,
            level=CheckLevel.CRITICAL if not overall_passed else CheckLevel.INFORMATIONAL,
            details=summary_details
        )
        
        # Log critical failures immediately
        if logger:
            for name, result in checks.items():
                if not result.passed and result.level == CheckLevel.CRITICAL:
                    logger.error(
                        f"Critical check failed: {name} - {result.message}",
                        extra={'check_details': result.details}
                    )
            
            if not overall_passed:
                logger.warning(
                    f"System checks completed with failures - Status: {system_status}",
                    extra={'summary': summary_details}
                )
            else:
                logger.info(
                    f"All system checks passed - Status: {system_status}",
                    extra={'summary': {
                        'total_checks': summary_details['total_checks'],
                        'passed_checks': summary_details['passed_checks']
                    }}
                )
        
        return checks
    
    except Exception as e:
        error_result = CheckResult(
            passed=False,
            message="System checks failed to complete",
            level=CheckLevel.CRITICAL,
            details={
                'error': str(e),
                'completed_checks': list(checks.keys()),
                'traceback': traceback.format_exc()
            }
        ).with_exception(e)
        
        checks['system_error'] = error_result
        
        if logger:
            logger.critical(
                "Fatal error during system checks",
                exc_info=True,
                extra={
                    'completed_checks': list(checks.keys()),
                    'error': str(e)
                }
            )
        
        return checks

def display_check_results(
    results: Dict[str, CheckResult], 
    logger: logging.Logger,
    extended: bool = False,
    include_performance: bool = False
) -> None:
    """
    Display check results in a styled table with improved formatting that matches
    the structure of run_system_checks output.
    
    Args:
        results: Dictionary of check results from run_system_checks()
        logger: Configured logger for recording the output
        extended: Whether extended checks were included (affects display)
        include_performance: Whether performance checks were included (affects display)
    """
    try:
        # Create the report table with dynamic title
        report_type = "Extended" if extended else "Basic"
        if include_performance:
            report_type += " (Performance)"
            
        table = Table(
            title=f"\n[bold]SYSTEM DIAGNOSTICS REPORT - {report_type}[/bold]",
            box=box.ROUNDED,
            header_style="bold bright_white",
            border_style="bright_white",
            title_style="bold yellow",
            title_justify="left",
            show_lines=True,
            expand=True,
            width=min(120, console.width - 4)
        )
        
        # Configure columns
        table.add_column("Check", style="bold cyan", width=25)
        table.add_column("Status", width=12, justify="center")
        table.add_column("Level", width=12, justify="center")
        table.add_column("Details", style="dim", min_width=50, max_width=80)
        
        # Group by check level in priority order
        for level in [CheckLevel.CRITICAL, CheckLevel.IMPORTANT, CheckLevel.INFORMATIONAL]:
            # Filter checks for this level, excluding summary/error entries
            level_rows = [
                (name, result) for name, result in results.items() 
                if result.level == level 
                and name not in ["system_summary", "system_error"]
            ]
            
            if not level_rows:
                continue
                
            # Add section header with colored background
            level_style = {
                CheckLevel.CRITICAL: "bold white on red",
                CheckLevel.IMPORTANT: "bold black on yellow",
                CheckLevel.INFORMATIONAL: "bold white on blue"
            }[level]
            
            table.add_row(
                Text(level.name, style=level_style),
                "",
                "",
                "",
                style=level_style
            )
            
            # Add checks for this level
            for name, result in level_rows:
                # Determine status styling
                if result.passed:
                    status_style = "bold green"
                    status_text = "PASS"
                else:
                    status_style = "bold red" if level == CheckLevel.CRITICAL else "bold yellow"
                    status_text = "FAIL" if level == CheckLevel.CRITICAL else "WARN"
                
                # Format details with special handling for configuration and model checks
                details_lines = []
                
                # Main message
                details_lines.append(f"[bright_white]{result.message}[/bright_white]")
                
                # Enhanced detail formatting
                if isinstance(result.details, dict):
                    # Special handling for configuration system
                    if name == 'configuration_system':
                        if 'sections_loaded' in result.details:
                            details_lines.append(f"[dim]Sections: {result.details['sections_loaded']}, Parameters: {result.details['total_parameters']}[/dim]")
                        if 'active_preset' in result.details:
                            details_lines.append(f"[dim]Active preset: {result.details['active_preset'] or 'default'}[/dim]")
                        if 'model_type' in result.details:
                            details_lines.append(f"[dim]Model type: {result.details['model_type']}[/dim]")
                    
                    # Special handling for model variants
                    elif name == 'model_variants':
                        if 'variant_names' in result.details:
                            details_lines.append(f"[dim]Available: {', '.join(result.details['variant_names'])}[/dim]")
                        if 'initialization_summary' in result.details:
                            summary = result.details['initialization_summary']
                            details_lines.append(f"[dim]Success rate: {summary['successful']}/{summary['attempted']}[/dim]")
                    
                    # Special handling for version info
                    elif 'version_info' in result.details:
                        versions = result.details['version_info']
                        details_lines.append("[dim]Dependencies:")
                        for pkg, info in versions.items():
                            status = "[green][OK]" if info['compatible'] else "[red][FAIL]"
                            details_lines.append(
                                f"  {status} {pkg}: {info['version']} "
                                f"(requires {info['required_version'] or 'any'})"
                            )
                    
                    # General dict handling for other checks
                    else:
                        for key, value in result.details.items():
                            if key not in ['error', 'exception', 'traceback', 'variant_status']:
                                if isinstance(value, (int, float, str, bool)):
                                    details_lines.append(f"[dim]{key}: {value}[/dim]")
                
                elif result.details and isinstance(result.details, str):
                    details_lines.append(f"[dim]{result.details}[/dim]")
                
                # Add exception if present
                if result.exception:
                    details_lines.append(f"[bold red]Error: {str(result.exception)}[/bold red]")
                
                details_text = "\n".join(details_lines)
                
                # Add row to table
                table.add_row(
                    Text(name.replace("_", " ").title()), 
                    Text(status_text, style=status_style),
                    Text(level.name, style="dim"),
                    details_text
                )
        
        # Add summary/error rows if present
        if "system_summary" in results:
            summary = results["system_summary"]
            summary_style = "bold green" if summary.passed else "bold red"
            
            table.add_row(
                Text("SUMMARY", style="bold bright_white on black"),
                Text("PASS" if summary.passed else "FAIL", style=summary_style),
                "",
                Text(
                    f"{summary.details['passed_checks']}/{summary.details['total_checks']} checks passed | "
                    f"{summary.details['critical_failures']} critical failures | "
                    f"Status: {summary.details.get('system_status', 'UNKNOWN')}",
                    style="bright_white"
                )
            )
        
        if "system_error" in results:
            error = results["system_error"]
            table.add_row(
                Text("FATAL ERROR", style="bold white on red"),
                Text("ERROR", style="bold white on red"),
                "",
                Text(
                    f"{error.message}\n"
                    f"[red]{error.details.get('error', 'Unknown error')}\n"
                    f"Completed checks: {', '.join(error.details.get('completed_checks', []))}",
                    style="bright_white"
                )
            )
        
        # Print the table
        console.print(table)
        
        # Enhanced logging - suppress redundant configuration/model messages
        if logger:
            # Log only the summary
            summary = results.get("system_summary")
            if summary:
                logger.info(
                    f"System diagnostics completed: {summary.details['passed_checks']}/"
                    f"{summary.details['total_checks']} checks passed, "
                    f"{summary.details['critical_failures']} critical failures, "
                    f"Status: {summary.details.get('system_status', 'UNKNOWN')}"
                )
            
            # Log configuration and model status briefly
            if 'configuration_system' in results:
                config_result = results['configuration_system']
                if config_result.passed and isinstance(config_result.details, dict):
                    logger.info(f"Configuration system: {config_result.details.get('sections_loaded', 0)} sections loaded")
            
            if 'model_variants' in results:
                model_result = results['model_variants']
                if model_result.passed and isinstance(model_result.details, dict):
                    variants = model_result.details.get('variant_names', [])
                    logger.info(f"Model variants: {len(variants)} available ({', '.join(variants)})")
            
            # Log only critical failures
            critical_failures = [
                (name, result) for name, result in results.items()
                if not result.passed and result.level == CheckLevel.CRITICAL
                and name not in ["system_summary", "system_error"]
            ]
            
            if critical_failures:
                for name, result in critical_failures:
                    logger.critical(f"Critical check failed: {name} - {result.message}")
            
            # Log any system error
            if "system_error" in results:
                error = results["system_error"]
                logger.critical(f"System checks failed: {error.details.get('error', 'Unknown error')}")
    
    except Exception as e:
        error_msg = f"Failed to display check results: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        if logger:
            logger.critical(error_msg, exc_info=True)

# Individual check implementations
def check_python_version(min_version: Tuple[int, int] = (3, 8)) -> CheckResult:
    """Verify that the current Python version meets the minimum requirement."""
    try:
        # Leverage the version info from check_versions
        version_info = check_versions(include_optional=False)
        python_info = version_info.get('Python', {})
        
        if not python_info:
            return CheckResult(
                passed=False,
                message="Python version information not available",
                level=CheckLevel.CRITICAL
            )
        
        # Get version from the comprehensive check
        current_version = tuple(map(int, python_info['version'].split('.')[:2]))
        passed = current_version >= min_version
        
        message = (
            f"Python version {'meets' if passed else 'fails'} minimum requirement "
            f"({'.'.join(map(str, min_version))})"
        )
        
        base_result = CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.CRITICAL if not passed else CheckLevel.INFORMATIONAL,
            details=python_info.get('details')
        )
        
        return base_result.with_details(
            f"Current version: {python_info['version']}\n"
            f"Minimum required: {'.'.join(map(str, min_version))}\n"
            f"Status: {'Compatible' if passed else 'Incompatible'}"
        )
        
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Python version check failed",
                level=CheckLevel.CRITICAL
            )
            .with_details(f"Could not determine Python version: {str(e)}")
            .with_exception(e)
        )

def safe_version_compare(current_version: str, requirement: Optional[str]) -> bool:
    """Safely compare version against requirement."""
    if current_version in ['N/A', 'unknown', 'Available']:
        return current_version == 'Available'
    if not requirement:
        return True
    
    try:
        # Extract version number from requirement (remove >=, ==, etc.)
        req_version = requirement.replace('>=', '').replace('<=', '').replace('==', '').replace('>', '').replace('<', '').strip()
        
        if not PACKAGING_AVAILABLE:
            # Fallback comparison without packaging
            return str(current_version) >= req_version
        
        return pkg_version.parse(current_version) >= pkg_version.parse(req_version)
    except Exception:
        return False

def check_versions(include_optional: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Verify package versions with comprehensive dependency checking.
    Returns a dictionary containing version status for all dependencies.
    
    Args:
        include_optional: Whether to include optional dependencies in the check
        
    Returns:
        Dictionary with package names as keys and version info as values
    """
    version_info = {}
    
    # Core dependencies - get actual versions from VERSION_INFO
    core_deps = {
        'Python': (VERSION_INFO['python'], '>=3.7', True),
        'PyTorch': (VERSION_INFO['torch'], '>=1.8', True),
        'NumPy': (VERSION_INFO['numpy'], '>=1.19', True),
        'Pandas': (VERSION_INFO['pandas'], '>=1.2', True),
        'Scikit-learn': (VERSION_INFO['sklearn'], '>=0.24', True),
        'Optuna': (VERSION_INFO['optuna'], '>=2.8', True),
        'Rich': (VERSION_INFO['rich'], '>=10.0', True),
        'Plotly': (VERSION_INFO['plotly'], '>=5.0', False)
    }
    
    # Optional dependencies - all using safe_version now
    optional_deps = {
        'ONNX Runtime': (safe_version('onnxruntime') if OPTIONAL_DEPENDENCIES.get('onnxruntime', False) else 'N/A', '>=1.8', False),
        'ONNX': (VERSION_INFO['onnx'], '>=1.8', False),
        'NVIDIA ML': (safe_version('nvidia-ml-py') if OPTIONAL_DEPENDENCIES.get('nvml', False) else 'N/A', '>=11.0', False),
        'Torch JIT': ('Available' if OPTIONAL_DEPENDENCIES.get('torch_jit', False) else 'N/A', None, False),
        'Cryptography': (safe_version('cryptography') if OPTIONAL_DEPENDENCIES.get('crypto', False) else 'N/A', None, False),
        'Database': ('Available' if OPTIONAL_DEPENDENCIES.get('database', False) else 'N/A', None, False),
        'Sklearn Anomaly': ('Available' if OPTIONAL_DEPENDENCIES.get('sklearn_anomaly', False) else 'N/A', None, False),
        'Statsmodels': (safe_version('statsmodels') if OPTIONAL_DEPENDENCIES.get('statsmodels', False) else 'N/A', None, False),
        'Numba': (safe_version('numba') if OPTIONAL_DEPENDENCIES.get('numba', False) else 'N/A', None, False),
        'Memory Profiler': (safe_version('memory-profiler') if OPTIONAL_DEPENDENCIES.get('memory_profiler', False) else 'N/A', None, False),
        'Line Profiler': (safe_version('line-profiler') if OPTIONAL_DEPENDENCIES.get('line_profiler', False) else 'N/A', None, False),
        'Packaging': ('Available' if OPTIONAL_DEPENDENCIES.get('packaging', False) else 'N/A', None, False),
        'PSUtil': (VERSION_INFO['psutil'], '>=5.8', False)
    }
    
    # Check all dependencies
    all_deps = {**core_deps}
    if include_optional:
        all_deps.update(optional_deps)
    
    for name, (version, requirement, required) in all_deps.items():
        if not include_optional and not required:
            continue
            
        try:
            if version == 'N/A':
                status = 'MISSING'
                meets_req = False
            elif version == 'Available' or requirement is None:
                status = 'OK'
                meets_req = True
            elif version == 'unknown':
                status = 'UNKNOWN'
                meets_req = False
            else:
                meets_req = safe_version_compare(version, requirement)
                status = 'OK' if meets_req else 'WARNING'
                
        except Exception as e:
            meets_req = False
            status = 'ERROR'
            version = f"Error: {str(e)}"
        
        version_info[name] = {
            'version': version,
            'required_version': requirement,
            'status': status,
            'required': required,
            'description': get_dependency_description(name),
            'compatible': meets_req,
            'available': version not in ['N/A', 'unknown'] and not str(version).startswith('Error:')
        }
    
    return version_info

def check_torch() -> CheckResult:
    """Confirm that PyTorch is installed and operational."""
    try:
        # Use the comprehensive version check
        version_info = check_versions(include_optional=False)
        torch_info = version_info.get('PyTorch', {})
        
        if not torch_info:
            return CheckResult(
                passed=False,
                message="PyTorch version information not available",
                level=CheckLevel.CRITICAL
            )
        
        passed = torch_info.get('compatible', False)
        base_result = CheckResult(
            passed=passed,
            message=f"PyTorch is {'available and compatible' if passed else 'not properly installed or incompatible'}",
            level=CheckLevel.CRITICAL,
            details=torch_info.get('details')
        )
        
        if passed:
            return base_result.with_details(
                f"Version: {torch_info['version']}\n"
                f"Required: {torch_info['required_version'] or 'Not specified'}\n"
                f"Description: {torch_info['description']}"
            )
        return base_result.with_details(
            f"PyTorch check failed\n"
            f"Installed version: {torch_info.get('version', 'unknown')}\n"
            f"Required version: {torch_info.get('required_version', 'unknown')}"
        )
        
    except ImportError as e:
        return (
            CheckResult(
                passed=False,
                message="PyTorch is not installed",
                level=CheckLevel.CRITICAL
            )
            .with_details("PyTorch package not found in Python environment")
            .with_exception(e)
        )
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="PyTorch check failed unexpectedly",
                level=CheckLevel.CRITICAL
            )
            .with_details(str(e))
            .with_exception(e)
        )

def check_cuda() -> CheckResult:
    """Check if CUDA is available and report GPU details if present."""
    try:
        cuda_available = torch.cuda.is_available()
        message = "CUDA is " + ("available" if cuda_available else "not available")
        
        base_result = CheckResult(
            passed=cuda_available,
            message=message,
            level=CheckLevel.IMPORTANT
        )
        
        if not cuda_available:
            return base_result.with_details("CUDA not available - Using CPU")
        
        # Build detailed GPU information
        details = []
        details.append(f"CUDA Version: {torch.version.cuda}")
        details.append(f"PyTorch CUDA Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        details.append(f"Detected {torch.cuda.device_count()} GPU(s):")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            details.append(
                f"  GPU {i}: {props.name} (Compute Capability: {props.major}.{props.minor}, "
                f"Memory: {props.total_memory/1024**3:.1f}GB, "
                f"Multiprocessors: {props.multi_processor_count})"
            )
        
        return base_result.with_details("\n".join(details))
        
    except ImportError as e:
        return (
            CheckResult(
                passed=False,
                message="PyTorch not available for CUDA check",
                level=CheckLevel.IMPORTANT
            )
            .with_details("PyTorch installation not found")
            .with_exception(e)
        )
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="CUDA check failed unexpectedly",
                level=CheckLevel.IMPORTANT
            )
            .with_details(str(e))
            .with_exception(e)
        )

def check_package_versions_wrapper(include_optional: bool = True) -> CheckResult:
    """
    Run comprehensive package version validation with rich output.
    
    Args:
        include_optional: Whether to include optional dependencies
        
    Returns:
        CheckResult object with detailed version information
    """
    try:
        version_info = check_versions(include_optional)
        details = []
        passed = True
        
        # Prepare detailed information
        for name, info in version_info.items():
            status_icon = "[PASS]" if info['status'] == 'OK' else "[WARN]" if info['status'] == 'WARNING' else "[FAIL]"
            req_text = f"(requires {info['required_version']})" if info['required_version'] else ""
            details.append(
                f"{status_icon} {name}: {info['version']} {req_text} - "
                f"{'Required' if info['required'] else 'Optional'}"
            )
            if info['required'] and not info['compatible']:
                passed = False
        
        # Create rich table for display
        table = Table(title="DEPENDENCY CHECK", title_justify="left", title_style="bold yellow", box=box.ROUNDED, show_lines=True)
        table.add_column("Package", style="bold cyan", no_wrap=True)
        table.add_column("Version", style="bold magenta")
        table.add_column("Type", style="bold green")
        table.add_column("Status", justify="center")
        table.add_column("Description", style="bold blue")
        
        for name, info in version_info.items():
            status_style = "bold green" if info['status'] == 'OK' else "bold yellow" if info['status'] == 'WARNING' else "bold red"
            required_style = "bold green" if info['required'] else "bold yellow"
            required_text = f"[{required_style}]{'Required' if info ['required'] else 'Optional'}[/{required_style}]"
            table.add_row(
                name,
                info['version'],
                required_text,
                f"[{status_style}]{info['status']}[/{status_style}]",
                info['description']
            )
        
        console.print(table)
        
        base_result = CheckResult(
            passed=passed,
            message="Package version check completed",
            level=CheckLevel.CRITICAL if not passed else CheckLevel.IMPORTANT
        )
        
        return (
            base_result
            .with_details("\n".join(details))
            .with_metadata({
                'version_info': version_info,
                'table': table,
                'summary': {
                    'total': len(version_info),
                    'passed': sum(1 for info in version_info.values() if info['status'] == 'OK'),
                    'warnings': sum(1 for info in version_info.values() if info['status'] == 'WARNING'),
                    'missing': sum(1 for info in version_info.values() if info['status'] == 'MISSING')
                }
            })
        )
        
    except ImportError as e:
        return (
            CheckResult(
                passed=False,
                message="Package version check failed - missing dependencies",
                level=CheckLevel.CRITICAL
            )
            .with_details(f"Required package not found: {str(e)}")
            .with_exception(e)
        )
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Package version check failed unexpectedly",
                level=CheckLevel.CRITICAL
            )
            .with_details(f"Error during version check: {str(e)}")
            .with_exception(e)
        )

def get_dependency_description(dep_name: str) -> str:
    """Get detailed description for a dependency."""
    descriptions = {
        # Core dependencies
        'Python': 'Python programming language runtime',
        'PyTorch': 'Deep learning framework (core dependency)',
        'NumPy': 'Fundamental package for numerical computing',
        'Pandas': 'Data manipulation and analysis toolkit',
        'Scikit-learn': 'Machine learning algorithms and utilities',
        'Optuna': 'Hyperparameter optimization framework',
        'Rich': 'Rich text and beautiful formatting in terminal',
        'Plotly': 'Interactive visualization library',
        'PSUtil': 'Process and system monitoring utilities',
        
        # Optional dependencies
        'ONNX Runtime': 'Cross-platform inference engine for ONNX models',
        'ONNX': 'Open Neural Network Exchange format',
        'NVIDIA ML': 'NVIDIA Management Library for GPU monitoring',
        'Torch JIT': 'PyTorch Just-In-Time compilation for model optimization',
        'Cryptography': 'Cryptographic primitives for model security',
        'Database': 'Database connectivity for model storage and retrieval',
        'Sklearn Anomaly': 'Scikit-learn anomaly detection algorithms',
        'Statsmodels': 'Statistical modeling and econometrics',
        'Numba': 'JIT compiler for numerical functions',
        'Memory Profiler': 'Memory usage tracking and analysis',
        'Line Profiler': 'Line-by-line performance profiling',
        'Packaging': 'Core utilities for Python packaging'
    }
    return descriptions.get(dep_name, 'Additional functionality')

def check_directory_access_wrapper() -> CheckResult:
    """Verify directory access using the setup_directories function."""
    try:
        dirs = setup_directories(logger)
        access_issues = []
        
        # Check each directory
        for name, path in dirs.items():
            if not path.exists():
                access_issues.append(f"Missing: {name} ({path})")
            elif not os.access(path, os.W_OK):
                access_issues.append(f"No write access: {name} ({path})")
        
        passed = len(access_issues) == 0
        base_result = CheckResult(
            passed=passed,
            message="Directory access verification",
            level=CheckLevel.CRITICAL
        )
        
        if passed:
            details = "\n".join(f"{name}: {path}" for name, path in dirs.items())
            return base_result.with_details(details)
        else:
            details = "\n".join([
                "Directory access issues found:",
                *access_issues,
                "",
                "All directories:",
                *[f"- {name}: {path}" for name, path in dirs.items()]
            ])
            return base_result.with_details(details)
            
    except PermissionError as e:
        return (
            CheckResult(
                passed=False,
                message="Permission denied for directory access",
                level=CheckLevel.CRITICAL
            )
            .with_details(f"Permission error: {str(e)}")
            .with_exception(e)
        )
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Directory access check failed unexpectedly",
                level=CheckLevel.CRITICAL
            )
            .with_details(f"Error: {str(e)}")
            .with_exception(e)
        )

def check_disk_space(min_gb: float = 1.0) -> CheckResult:
    """Ensure the system has at least the specified amount of free disk space."""
    try:
        usage = shutil.disk_usage('.')
        free_gb = usage.free / (1024**3)
        passed = free_gb >= min_gb
        
        base_result = CheckResult(
            passed=passed,
            message=f"Disk space {'meets' if passed else 'below'} minimum requirement ({min_gb}GB)",
            level=CheckLevel.IMPORTANT
        )
        
        details = (
            f"Free space: {free_gb:.1f}GB (minimum required: {min_gb}GB)\n"
            f"Total space: {usage.total/(1024**3):.1f}GB\n"
            f"Used space: {usage.used/(1024**3):.1f}GB"
        )
        
        return base_result.with_details(details)
        
    except PermissionError as e:
        return (
            CheckResult(
                passed=False,
                message="Disk space check failed - permission denied",
                level=CheckLevel.IMPORTANT
            )
            .with_details(f"Cannot access disk usage information: {str(e)}")
            .with_exception(e)
        )
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Disk space check failed unexpectedly",
                level=CheckLevel.IMPORTANT
            )
            .with_details(str(e))
            .with_exception(e)
        )

def check_cpu_cores() -> CheckResult:
    """Report the number of logical and physical CPU cores available."""
    try:
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        base_result = CheckResult(
            passed=True,
            message="CPU core information",
            level=CheckLevel.INFORMATIONAL
        )
        
        details = (
            f"Logical cores: {logical_cores}\n"
            f"Physical cores: {physical_cores}\n"
            f"Hyperthreading: {'Enabled' if logical_cores != physical_cores else 'Disabled'}"
        )
        
        return base_result.with_details(details)
        
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="CPU core check failed",
                level=CheckLevel.INFORMATIONAL
            )
            .with_details(str(e))
            .with_exception(e)
        )

def check_system_ram() -> CheckResult:
    """Report detailed system RAM information."""
    try:
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        base_result = CheckResult(
            passed=True,
            message="System memory information",
            level=CheckLevel.INFORMATIONAL
        )
        
        details = (
            f"RAM Total: {ram.total/(1024**3):.1f}GB\n"
            f"RAM Available: {ram.available/(1024**3):.1f}GB ({ram.available/ram.total:.1%})\n"
            f"RAM Used: {ram.used/(1024**3):.1f}GB ({ram.percent}%)\n"
            f"Swap Total: {swap.total/(1024**3):.1f}GB\n"
            f"Swap Used: {swap.used/(1024**3):.1f}GB ({swap.percent}%)"
        )
        
        return base_result.with_details(details)
        
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Memory check failed",
                level=CheckLevel.INFORMATIONAL
            )
            .with_details(str(e))
            .with_exception(e)
        )

def check_system_architecture() -> CheckResult:
    """Report detailed system architecture information."""
    try:
        base_result = CheckResult(
            passed=True,
            message="System architecture information",
            level=CheckLevel.INFORMATIONAL
        )
        
        details = (
            f"Architecture: {platform.architecture()[0]}\n"
            f"Machine: {platform.machine()}\n"
            f"System: {platform.system()} {platform.release()}\n"
            f"Processor: {platform.processor()}\n"
            f"Python Build: {' '.join(platform.python_build())}\n"
            f"Word Size: {platform.architecture()[0]}"
        )
        
        return base_result.with_details(details)
        
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Architecture check failed",
                level=CheckLevel.INFORMATIONAL
            )
            .with_details(str(e))
            .with_exception(e)
        )

def check_logging_setup() -> CheckResult:
    """
    Verify that logging is configured according to setup_logging().

    Checks:
    - At least one file handler with UTF-8 encoding
    - Log file exists
    - At least one console handler using UnicodeStreamHandler
    - Produces a compliance score (0-100%)
    - Returns human-readable feedback
    - Colorized terminal output (auto-disabled if not TTY)
    """
    try:
        # Initialize base result
        base_result = CheckResult(
            passed=False,
            message="Logging configuration check",
            level=CheckLevel.IMPORTANT
        )

        # Detect if output is a TTY (interactive terminal)
        use_color = sys.stdout.isatty()

        # Get logger and handlers
        logger = logging.getLogger(__name__)
        handlers = logger.handlers

        # Initialize check data
        check_data: Dict[str, any] = {
            'handlers': [],
            'checks': {
                'file_handler_found': False,
                'file_handler_utf8': False,
                'file_handler_exists': False,
                'console_handler_found': False,
                'console_handler_unicode': False
            },
            'feedback': [],
            'compliance_score': 0
        }

        # If no handlers, return immediately
        if not handlers:
            return (
                base_result
                .with_details({
                    'error': 'No handlers configured',
                    'compliance_score': 0,
                    'feedback': ['No logging handlers configured — run setup_logging()']
                })
                .with_message("No logging handlers configured")
            )

        # Analyze each handler
        for handler in handlers:
            handler_info = {
                'type': handler.__class__.__name__,
                'level': logging.getLevelName(handler.level)
            }

            # File handler checks
            if isinstance(handler, logging.FileHandler):
                check_data['checks']['file_handler_found'] = True
                encoding = getattr(handler, 'encoding', '').lower()
                if encoding == 'utf-8':
                    check_data['checks']['file_handler_utf8'] = True

                log_path = Path(getattr(handler, 'baseFilename', ''))
                if log_path.exists():
                    check_data['checks']['file_handler_exists'] = True

                handler_info.update({
                    'encoding': encoding,
                    'file_path': str(log_path),
                    'exists': log_path.exists()
                })

            # Console handler checks (must be UnicodeStreamHandler)
            elif isinstance(handler, UnicodeStreamHandler):
                check_data['checks']['console_handler_found'] = True
                check_data['checks']['console_handler_unicode'] = True
                handler_info['stream'] = getattr(handler.stream, 'name', str(handler.stream))

            check_data['handlers'].append(handler_info)

        # Calculate compliance score
        total_checks = len(check_data['checks'])
        passed_checks = sum(check_data['checks'].values())
        compliance_score = int((passed_checks / total_checks) * 100)
        passed = compliance_score == 100

        # Generate feedback messages
        feedback = []
        if not check_data['checks']['file_handler_found']:
            feedback.append("Missing file handler")
        if not check_data['checks']['file_handler_utf8']:
            feedback.append("File handler not using UTF-8 encoding")
        if not check_data['checks']['file_handler_exists']:
            feedback.append("Log file does not exist")
        if not check_data['checks']['console_handler_found']:
            feedback.append("Missing console handler")
        if not check_data['checks']['console_handler_unicode']:
            feedback.append("Console handler is not Unicode-safe (must use UnicodeStreamHandler)")

        # Prepare final details
        details = {
            **check_data,
            'compliance_score': compliance_score,
            'feedback': feedback,
            'passed_checks': passed_checks,
            'total_checks': total_checks
        }

        return (
            base_result
            .with_passed(passed)
            .with_details(details)
            .with_message(f"Logging configuration {'passed' if passed else 'failed'} ({compliance_score}%)")
        )

    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Logging check failed",
                level=CheckLevel.IMPORTANT
            )
            .with_details({
                'error': str(e),
                'compliance_score': 0,
                'feedback': ['Exception occurred during logging setup check']
            })
            .with_exception(e)
        )

def check_seed_config() -> CheckResult:
    """
    Verify that reproducibility seeds and related configurations are set.

    Checks:
    - PYTHONHASHSEED environment variable set and numeric
    - CUBLAS_WORKSPACE_CONFIG set to ':4096:8'
    - NumPy RNG available (seed not directly verifiable)
    - PyTorch RNG available, CUDA deterministic mode on, benchmark off
    - TensorFlow RNG available (if installed)
    - Produces compliance score (0-100%)
    - Human-readable feedback with colorized PASS/WARN/FAIL
    - Auto-disables colors if not in a TTY
    - Structured details for programmatic use
    - Graceful handling and exception safety
    
    Returns:
        CheckResult with detailed seed configuration status
    """
    try:
        # Initialize base result
        base_result = CheckResult(
            # Passed will update after checks
            passed=False,
            message="Reproducibility configuration check",
            level=CheckLevel.IMPORTANT
        )
        
        # Initialize check data
        check_data: Dict[str, any] = {
            'checks': {
                'PYTHONHASHSEED': {'passed': False, 'value': None},
                'CUBLAS_WORKSPACE_CONFIG': {'passed': False, 'value': None},
                'numpy_rng': {'passed': False, 'value': None},
                'torch_rng': {'passed': False, 'value': None},
                'torch_cuda': {
                    'deterministic': False,
                    'benchmark': None,
                    'passed': False
                },
                'tensorflow_rng': {'passed': False, 'value': None}
            },
            'weights': {
                'PYTHONHASHSEED': 25,
                'CUBLAS_WORKSPACE_CONFIG': 25,
                'numpy_rng': 15,
                'torch_rng': 20 if torch.cuda.is_available() else 10,
                # Check if TensorFlow is installed
                'tensorflow_rng': 10
            },
            'feedback': [],
            'compliance_score': 0
        }
        
        # Check PYTHONHASHSEED
        hash_seed = os.environ.get('PYTHONHASHSEED')
        check_data['checks']['PYTHONHASHSEED']['passed'] = (
            hash_seed is not None and hash_seed.isdigit()
        )
        check_data['checks']['PYTHONHASHSEED']['value'] = hash_seed or "Not set"
        if not check_data['checks']['PYTHONHASHSEED']['passed']:
            check_data['feedback'].append("PYTHONHASHSEED not set or invalid")
        
        # Check CUBLAS_WORKSPACE_CONFIG
        cublas_cfg = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        check_data['checks']['CUBLAS_WORKSPACE_CONFIG']['passed'] = (
            cublas_cfg == ':4096:8'
        )
        check_data['checks']['CUBLAS_WORKSPACE_CONFIG']['value'] = cublas_cfg or "Not set"
        if not check_data['checks']['CUBLAS_WORKSPACE_CONFIG']['passed']:
            check_data['feedback'].append("CUBLAS_WORKSPACE_CONFIG not set to ':4096:8'")
        
        # Check NumPy RNG
        try:
            _ = np.random.rand()
            check_data['checks']['numpy_rng']['passed'] = True
            check_data['checks']['numpy_rng']['value'] = "Available"
        except Exception as e:
            check_data['checks']['numpy_rng']['value'] = f"Error: {str(e)}"
            check_data['feedback'].append("NumPy RNG not available")
        
        # Check PyTorch RNG and CUDA settings
        try:
            _ = torch.rand(1)
            check_data['checks']['torch_rng']['passed'] = True
            check_data['checks']['torch_rng']['value'] = "Available"
            
            if torch.cuda.is_available():
                check_data['checks']['torch_cuda']['deterministic'] = (
                    torch.backends.cudnn.deterministic
                )
                check_data['checks']['torch_cuda']['benchmark'] = (
                    not torch.backends.cudnn.benchmark
                )
                check_data['checks']['torch_cuda']['passed'] = (
                    check_data['checks']['torch_cuda']['deterministic'] and
                    check_data['checks']['torch_cuda']['benchmark']
                )
                
                if not check_data['checks']['torch_cuda']['passed']:
                    check_data['feedback'].append("PyTorch CUDA settings not deterministic")
        except Exception as e:
            check_data['checks']['torch_rng']['value'] = f"Error: {str(e)}"
            check_data['feedback'].append("PyTorch RNG not available")
        
        # Check TensorFlow RNG (optional)
        try:
            import tensorflow as tf
            _ = tf.random.uniform((1,))
            check_data['checks']['tensorflow_rng']['passed'] = True
            check_data['checks']['tensorflow_rng']['value'] = "Available"
        except ImportError:
            check_data['checks']['tensorflow_rng']['value'] = "Not installed"
            # Not required
            check_data['checks']['tensorflow_rng']['passed'] = True
        except Exception as e:
            check_data['checks']['tensorflow_rng']['value'] = f"Error: {str(e)}"
            check_data['feedback'].append("TensorFlow RNG not available")
        
        # Calculate compliance score
        total_points = sum(check_data['weights'].values())
        earned_points = sum(
            check_data['weights'][check] 
            for check, status in check_data['checks'].items()
            if status['passed']
        )
        compliance_score = round((earned_points / total_points) * 100, 2)
        # 90% threshold for passing
        passed = compliance_score >= 90
        
        # Prepare final details
        details = {
            **check_data,
            'compliance_score': compliance_score,
            'passed': passed
        }
        
        return (
            base_result
            .with_passed(passed)
            .with_details(details)
            .with_message(
                f"Reproducibility configuration {'passed' if passed else 'failed'} "
                f"({compliance_score}%)"
            )
        )
        
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Seed configuration check failed",
                level=CheckLevel.IMPORTANT
            )
            .with_details({
                'error': str(e),
                'compliance_score': 0,
                'feedback': ['Exception occurred during seed config check']
            })
            .with_exception(e)
        )

# Additional system initialization checks
def check_global_exception_handler() -> CheckResult:
    """Check if global exception handler is properly configured."""
    try:
        # Test if our custom handler is set
        current_handler = sys.excepthook
        is_custom = current_handler.__name__ == 'enhanced_global_exception_handler'
        
        return CheckResult(
            passed=is_custom,
            message="Global exception handler configured" if is_custom else "Using default exception handler",
            level=CheckLevel.IMPORTANT,
            details=f"Handler: {current_handler.__name__}"
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Failed to check exception handler",
            level=CheckLevel.IMPORTANT
        ).with_exception(e)

def check_performance_monitoring() -> CheckResult:
    """Check if performance monitoring is available."""
    try:
        # Check if enhanced_monitor_performance decorator is available
        has_monitoring = 'enhanced_monitor_performance' in globals()
        
        return CheckResult(
            passed=has_monitoring,
            message="Performance monitoring available" if has_monitoring else "Performance monitoring not initialized",
            level=CheckLevel.INFORMATIONAL,
            details="Enhanced monitoring decorator configured"
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Failed to check performance monitoring",
            level=CheckLevel.INFORMATIONAL
        ).with_exception(e)

def check_memory_management() -> CheckResult:
    """Check if memory management functions are available."""
    try:
        # Check if memory management functions are available
        has_clear_memory = 'enhanced_clear_memory' in globals()
        has_memory_usage = 'get_memory_usage' in globals()
        
        passed = has_clear_memory and has_memory_usage
        
        details = {
            'clear_memory_available': has_clear_memory,
            'get_memory_usage_available': has_memory_usage
        }
        
        return CheckResult(
            passed=passed,
            message="Memory management functions available" if passed else "Some memory management functions missing",
            level=CheckLevel.IMPORTANT,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Failed to check memory management",
            level=CheckLevel.IMPORTANT
        ).with_exception(e)

def check_configuration_system() -> CheckResult:
    """Check if configuration system is properly initialized with improved logging."""
    try:
        # Suppress individual log messages during system checks
        config_logger = logging.getLogger('config')
        original_level = config_logger.level
        config_logger.setLevel(logging.WARNING)  # Only show warnings/errors
        
        try:
            # Initialize configuration silently
            config = initialize_config()
            validate_config(config)
            
            config_details = {
                'config_file_exists': CONFIG_FILE.exists(),
                'active_preset': config.get('presets', {}).get('current_preset', 'none'),
                'model_type': config.get('model', {}).get('model_type', 'unknown'),
                'sections_loaded': len(config),
                'total_parameters': sum(len(section) if isinstance(section, dict) else 1 for section in config.values())
            }
            
            return CheckResult(
                passed=True,
                message="Configuration system operational",
                level=CheckLevel.CRITICAL,
                details=config_details
            )
        finally:
            # Restore original logging level
            config_logger.setLevel(original_level)
            
    except ValueError as e:
        return CheckResult(
            passed=False,
            message=f"Configuration validation failed: {str(e)}",
            level=CheckLevel.CRITICAL
        ).with_exception(e)
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Configuration system initialization failed",
            level=CheckLevel.CRITICAL
        ).with_exception(e)

def check_model_variants() -> CheckResult:
    """Check if model variants are properly initialized with improved logging."""
    try:
        # Suppress individual log messages during system checks
        model_logger = logging.getLogger('models')
        original_level = model_logger.level
        # Only show warnings/errors
        model_logger.setLevel(logging.WARNING)
        
        try:
            # Initialize model variants silently
            initialize_model_variants(silent=True)
            
            if not MODEL_VARIANTS:
                return CheckResult(
                    passed=False,
                    message="No model variants available",
                    level=CheckLevel.CRITICAL,
                    details="Model variants dictionary is empty"
                )
            
            # Validate model variants silently
            variant_status = validate_model_variants(logger, silent=True)
            available_variants = [name for name, status in variant_status.items() if status == 'available']
            
            if not available_variants:
                return CheckResult(
                    passed=False,
                    message="No working model variants available",
                    level=CheckLevel.CRITICAL,
                    details=variant_status
                )
            
            # Prepare structured details
            variant_details = {
                'total_variants': len(MODEL_VARIANTS),
                'available_variants': len(available_variants),
                'variant_names': available_variants,
                'variant_status': variant_status,
                'initialization_summary': {
                    'attempted': len(MODEL_VARIANTS),
                    'successful': len(available_variants),
                    'failed': len(MODEL_VARIANTS) - len(available_variants)
                }
            }
            
            return CheckResult(
                passed=True,
                message=f"Model variants available: {', '.join(available_variants)}",
                level=CheckLevel.CRITICAL,
                details=variant_details
            )
        finally:
            # Restore original logging level
            model_logger.setLevel(original_level)
            
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Model variants initialization failed",
            level=CheckLevel.CRITICAL
        ).with_exception(e)

def check_performance_baseline() -> CheckResult:
    """Check if performance baseline can be established."""
    try:
        performance_metrics = establish_performance_baseline()
        
        return CheckResult(
            passed=True,
            message="Performance baseline established",
            level=CheckLevel.INFORMATIONAL,
            details=performance_metrics
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Failed to establish performance baseline",
            level=CheckLevel.INFORMATIONAL
        ).with_exception(e)

# System and environment configuration
def configure_system() -> Dict[str, Any]:
    """
    Configure system-wide settings for optimal performance and logging.
    Returns a dictionary containing all applied configurations.
    
    Features:
    - Disables verbose logging from common libraries
    - Configures PyTorch for optimal performance
    - Sets NumPy print formatting
    - Suppresses common warning types
    - Configures thread usage
    """
    config = {
        'torch': {},
        'numpy': {},
        'warnings': {},
        'environment': {}
    }

    # Environment variable configurations
    env_vars = {
        # TensorFlow logging
        'TF_CPP_MIN_LOG_LEVEL': '3',
        # Intel MKL warnings
        'KMP_WARNINGS': '0',
        # OpenMP threads
        'OMP_NUM_THREADS': str(min(4, os.cpu_count() or 1)),
        # MKL threads
        'MKL_NUM_THREADS': str(min(4, os.cpu_count() or 1)),
        # For CUDA reproducibility
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        # Disable TensorFlow GPU if present
        'CUDA_VISIBLE_DEVICES': '' if not torch.cuda.is_available() else None
    }
    
    # Apply environment variables
    for key, value in env_vars.items():
        if value is not None:
            os.environ[key] = value
            config['environment'][key] = value

    # PyTorch configuration
    torch_config = {
        'deterministic': True,
        'benchmark': False,
        'float32_matmul_precision': 'high',
        'num_threads': min(4, os.cpu_count() or 1),
        'precision': 4,
        'sci_mode': False
    }
    
    torch.set_num_threads(torch_config['num_threads'])
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = torch_config['deterministic']
        torch.backends.cudnn.benchmark = torch_config['benchmark']
    
    torch.set_printoptions(
        precision=torch_config['precision'],
        sci_mode=torch_config['sci_mode']
    )
    
    config['torch'].update(torch_config)

    # NumPy configuration
    np_config = {
        'precision': 4,
        'suppress': True,
        'threshold': 100,
        'linewidth': 120,
        'float_division_warning': False
    }
    
    np.set_printoptions(
        precision=np_config['precision'],
        suppress=np_config['suppress'],
        threshold=np_config['threshold'],
        linewidth=np_config['linewidth']
    )
    config['numpy'].update(np_config)

    # Warning configurations
    warning_config = {
        'ignored_categories': {
            UserWarning: ['joblib', 'torch', 'numpy'],
            FutureWarning: None,
            DeprecationWarning: None,
            ConvergenceWarning: ['sklearn'],
            RuntimeWarning: None,
            MatplotlibDeprecationWarning: None
        },
        'simplefilter': 'ignore'
    }
    
    # Apply warning filters
    for category, modules in warning_config['ignored_categories'].items():
        if modules:
            for module in modules:
                warnings.filterwarnings('ignore', category=category, module=module)
        else:
            warnings.filterwarnings('ignore', category=category)
    
    warnings.simplefilter(warning_config['simplefilter'])
    config['warnings'].update(warning_config)

    return config

# Reproducibility configuration
def set_seed(seed: int = 42) -> Dict[str, Any]:
    """
    Configure all random seeds for full reproducibility.
    Returns a dictionary containing seed configuration.
    
    Features:
    - Seeds Python, NumPy, and PyTorch RNGs
    - Configures CUDA for deterministic operations
    - Sets hash seed for Python
    - Configures TensorFlow if present
    
    Returns:
        Dictionary containing the seed configuration
    """
    seed_config = {
        'base_seed': seed,
        'python': {
            'random_seed': seed,
            'hash_seed': seed
        },
        'numpy_seed': seed,
        'torch': {
            'cpu_seed': seed,
            'cuda_seeds': None,
            'cuda_deterministic': False,
            'cuda_benchmark': False
        },
        'environment': {
            'PYTHONHASHSEED': str(seed),
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8'
        },
        'tensorflow_seed': None
    }
    
    # Set Python seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_config['torch'].update({
            'cuda_seeds': [seed] * torch.cuda.device_count(),
            'cuda_deterministic': True,
            'cuda_benchmark': False
        })
    
    # Set CUDA workspace config
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        seed_config['tensorflow_seed'] = seed
    except ImportError:
        pass
    
    return seed_config

# Hardware and Package Configuration
def setup_gpu(logger: logging.Logger) -> torch.device:
    """
    Detect and configure the primary compute device (GPU if available, else CPU),
    apply performance tuning (e.g., cuDNN benchmarking), and log device specs.
    """
    device = torch.device('cpu')
    device_info = {
        'type': 'CPU',
        'count': os.cpu_count() or 'Unknown',
        'details': platform.processor()
    }
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_info.update({
            'type': 'CUDA',
            'count': torch.cuda.device_count(),
            'details': []
        })
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info['details'].append({
                'id': i,
                'name': props.name,
                'capability': f"{props.major}.{props.minor}",
                'memory_gb': props.total_memory / (1024**3),
                'multiprocessors': props.multi_processor_count
            })
        
        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
    
    # Log device information
    logger.info(f"Using device: {device}")
    logger.info(f"Device details: {device_info}")
    
    return device

def check_hardware() -> Dict[str, Union[str, bool, int, float]]:
    """Check and report available hardware resources with validation."""
    try:
        info = {
            "pytorch_version": torch.__version__,
            "gpu_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_threads": torch.get_num_threads(),
            "system": platform.system(),
            "cpu_count": os.cpu_count() or 1,
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": props.name,
                "cuda_capability": f"{props.major}.{props.minor}",
                "gpu_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_version": torch.version.cuda
            })
        
        return info
    except Exception as e:
        logger.error(f"Hardware check failed: {str(e)}")
        return {
            "pytorch_version": "unknown",
            "gpu_available": False,
            "device": "cpu",
            "gpu_count": 0,
            "cpu_threads": 1,
            "system": "unknown",
            "cpu_count": 1
        }

def get_memory_usage() -> Dict[str, Any]:
    """
    Collect detailed memory usage information for system RAM, process memory, and GPU memory (if available).
    
    Returns:
        dict: Nested memory usage details, including human-readable sizes and raw byte values.
    """
    try:
        # System-wide RAM usage
        sys_mem = psutil.virtual_memory()
        process_mem = psutil.Process().memory_info()

        memory_info = {
            'system_ram': {
                'total_bytes': sys_mem.total,
                'available_bytes': sys_mem.available,
                'used_bytes': sys_mem.used,
                'percent_used': sys_mem.percent,
                'total_gb': sys_mem.total / (1024 ** 3),
                'available_gb': sys_mem.available / (1024 ** 3),
                'used_gb': sys_mem.used / (1024 ** 3)
            },
            'process_memory': {
                'rss_bytes': process_mem.rss,
                'vms_bytes': process_mem.vms,
                'rss_mb': process_mem.rss / (1024 ** 2),
                'vms_mb': process_mem.vms / (1024 ** 2)
            }
        }

        # Optional GPU memory stats
        if torch.cuda.is_available():
            memory_info['gpu_memory'] = {}
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                memory_info['gpu_memory'][f'device_{i}'] = {
                    'name': torch.cuda.get_device_name(i),
                    'allocated_bytes': torch.cuda.memory_allocated(i),
                    'reserved_bytes': torch.cuda.memory_reserved(i),
                    'max_allocated_bytes': torch.cuda.max_memory_allocated(i),
                    'allocated_mb': torch.cuda.memory_allocated(i) / (1024 ** 2),
                    'reserved_mb': torch.cuda.memory_reserved(i) / (1024 ** 2),
                    'max_allocated_mb': torch.cuda.max_memory_allocated(i) / (1024 ** 2)
                }

        return memory_info

    except Exception as e:
        return {
            'error': str(e),
            'system_ram': None,
            'process_memory': None,
            'gpu_memory': None
        }

def get_system_info() -> Dict[str, Any]:
    """
    Gather comprehensive system and environment information for diagnostics and logging.
    
    Returns:
        dict: System specifications, Python environment details, dependency info, and hardware stats.
    """
    try:
        info = {
            'platform': platform.platform(),
            'os': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version()
            },
            'processor': platform.processor(),
            'architecture': {
                'bits': platform.architecture()[0],
                'linkage': platform.architecture()[1],
                'machine': platform.machine()
            },
            'python': {
                'version': sys.version,
                'version_info': {
                    'major': sys.version_info.major,
                    'minor': sys.version_info.minor,
                    'micro': sys.version_info.micro
                },
                'executable': sys.executable
            },
            'torch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            },
            'cpu': {
                'count_logical': os.cpu_count(),
                'count_physical': psutil.cpu_count(logical=False),
                'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': get_memory_usage(),
            'optional_dependencies': OPTIONAL_DEPENDENCIES,
            'version_info': VERSION_INFO
        }

        # GPU information if CUDA is available
        if torch.cuda.is_available():
            info['gpu'] = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info['gpu'][f'device_{i}'] = {
                    'name': props.name,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'total_memory_gb': props.total_memory / (1024 ** 3),
                    'multi_processor_count': props.multi_processor_count
                }

        return info

    except Exception as e:
        return {
            'error': str(e),
            'platform': None,
            'processor': None,
            'architecture': None,
            'python': None,
            'torch': None,
            'cpu': None,
            'memory': None
        }

def check_core_dependencies() -> CheckResult:
    """Check and report status of core dependencies."""
    try:
        # Use the comprehensive version check
        version_info = check_versions(include_optional=False)
        
        # Filter only core dependencies (required=True)
        core_deps = {k: v for k, v in version_info.items() if v.get('required', False)}
        
        # Determine overall status
        passed = all(info['compatible'] for info in core_deps.values())
        
        # Prepare details
        details = []
        for name, info in core_deps.items():
            status = "[OK]" if info['compatible'] else "[FAIL]"
            details.append(
                f"{status} {name}: {info['version']} "
                f"(requires {info['required_version'] or 'any'})"
            )
        
        base_result = CheckResult(
            passed=passed,
            message="Core dependencies check completed",
            level=CheckLevel.CRITICAL if not passed else CheckLevel.IMPORTANT
        )
        
        return (
            base_result
            .with_details("\n".join(details))
            .with_metadata({
                'core_dependencies': core_deps,
                'summary': {
                    'total': len(core_deps),
                    'compatible': sum(1 for info in core_deps.values() if info['compatible']),
                    'incompatible': sum(1 for info in core_deps.values() if not info['compatible'])
                }
            })
        )
        
    except Exception as e:
        return (
            CheckResult(
                passed=False,
                message="Core dependencies check failed",
                level=CheckLevel.CRITICAL
            )
            .with_details(f"Error checking core dependencies: {str(e)}")
            .with_exception(e)
        )

# Utility function for external use
def get_version_info(package_name: str) -> Dict[str, str]:
    """
    Get comprehensive version information for a package.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        Dictionary with version information and availability status
    """
    version_str = safe_version(package_name)
    
    return {
        'package': package_name,
        'version': version_str,
        'available': version_str not in ['N/A', 'unknown'],
        'importlib_metadata_available': IMPORTLIB_METADATA_AVAILABLE,
        'packaging_available': PACKAGING_AVAILABLE
    }

# Helper functions for system diagnostics and error handling
def enhanced_global_exception_handler(exc_type, exc_value, exc_traceback):
    """Enhanced global exception handler with detailed logging and recovery."""
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info("System interrupted by user")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Log the exception with full context
    logger.critical("CRITICAL: Uncaught exception occurred", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Create error report
    error_report = {
        'timestamp': datetime.now().isoformat(),
        'exception_type': exc_type.__name__,
        'exception_message': str(exc_value),
        'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback),
        'system_info': get_system_info(),
        'memory_usage': get_memory_usage()
    }
    
    # Save error report
    try:
        error_file = LOG_DIR / f"critical_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        logger.info(f"Error report saved to: {error_file}")
    except Exception as save_error:
        logger.error(f"Failed to save error report: {save_error}")
    
    # Display user-friendly error in interactive mode
    if hasattr(sys, 'ps1') or sys.stdin.isatty():
        console.print(f"[red]CRITICAL ERROR: {exc_value}[/red]")
        console.print(f"[dim]Error report saved to logs directory[/dim]")
        console.print(f"[dim]Exception type: {exc_type.__name__}[/dim]")
    
    # Cleanup on critical error
    try:
        enhanced_clear_memory()
    except:
        pass

def performance_monitor_wrapper(func, include_memory, log_level):
    """Wrapper function for performance monitoring."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Pre-execution metrics
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss if include_memory else 0
        start_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            result = func(*args, **kwargs)
            
            # Post-execution metrics
            end_time = time.time()
            duration = end_time - start_time
            
            metrics = {'duration': duration, 'function': func.__name__}
            
            if include_memory:
                end_memory = psutil.Process().memory_info().rss
                memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
                metrics['memory_delta_mb'] = memory_delta
                
                if torch.cuda.is_available():
                    end_gpu_memory = torch.cuda.memory_allocated()
                    gpu_memory_delta = (end_gpu_memory - start_gpu_memory) / 1024 / 1024  # MB
                    metrics['gpu_memory_delta_mb'] = gpu_memory_delta
            
            # Log performance metrics
            log_message = f"{func.__name__} completed in {duration:.3f}s"
            if include_memory:
                log_message += f", memory: {metrics.get('memory_delta_mb', 0):+.1f}MB"
                if torch.cuda.is_available():
                    log_message += f", GPU: {metrics.get('gpu_memory_delta_mb', 0):+.1f}MB"
            
            logger.log(log_level, log_message)
            
            # Store metrics for analysis
            if not hasattr(wrapper, '_performance_metrics'):
                wrapper._performance_metrics = []
            wrapper._performance_metrics.append(metrics)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
            raise
    
    return wrapper

def enhanced_monitor_performance(include_memory=True, log_level=logging.DEBUG):
    """Enhanced performance monitoring decorator with configurable options."""
    def decorator(func):
        return performance_monitor_wrapper(func, include_memory, log_level)
    return decorator

def establish_performance_baseline():
    """Run performance tests to establish system baselines."""
    performance_metrics = {}
    
    # CPU performance test
    start_time = time.time()
    test_array = np.random.rand(1000, 1000)
    np.dot(test_array, test_array.T)
    cpu_time = time.time() - start_time
    performance_metrics['cpu_baseline'] = cpu_time
    
    # Memory allocation test
    start_memory = psutil.Process().memory_info().rss
    test_tensor = torch.randn(1000, 1000)
    end_memory = psutil.Process().memory_info().rss
    memory_overhead = (end_memory - start_memory) / 1024 / 1024  # MB
    performance_metrics['memory_overhead'] = memory_overhead
    
    # GPU performance test if available
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
            torch.mm(gpu_tensor, gpu_tensor.t())
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            performance_metrics['gpu_baseline'] = gpu_time
            del gpu_tensor
        except Exception as e:
            logger.warning(f"GPU performance test failed: {e}")
    
    # Cleanup test objects
    del test_array, test_tensor
    enhanced_clear_memory()
    
    return performance_metrics

def enhanced_clear_memory(aggressive=False):
    """Enhanced memory clearing with aggressive mode."""
    try:
        # Clear Python garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if aggressive:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
        
        # Force garbage collection multiple times in aggressive mode
        if aggressive:
            for _ in range(3):
                gc.collect()
        
        logger.debug("Memory cleared successfully")
        
    except Exception as e:
        logger.warning(f"Memory clearing failed: {e}")

def get_detailed_memory_usage():
    """Get comprehensive memory usage information."""
    memory_info = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent,
            'used': psutil.virtual_memory().used
        },
        'process': {
            'rss': psutil.Process().memory_info().rss,
            'vms': psutil.Process().memory_info().vms,
            'percent': psutil.Process().memory_percent()
        }
    }
    
    if torch.cuda.is_available():
        memory_info['gpu'] = {
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
            'max_reserved': torch.cuda.max_memory_reserved()
        }
        
        # Add per-device information if multiple GPUs
        if torch.cuda.device_count() > 1:
            memory_info['gpu_devices'] = {}
            for i in range(torch.cuda.device_count()):
                memory_info['gpu_devices'][f'device_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i),
                    'cached': torch.cuda.memory_reserved(i)
                }
    
    return memory_info

def enhance_hardware_info(hw_info):
    """Enhance basic hardware information with additional details."""
    enhanced_info = {
        **hw_info,
        'detailed_cpu_info': {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'architecture': platform.architecture(),
            'processor': platform.processor()
        },
        'memory_info': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'usage_percent': psutil.virtual_memory().percent
        }
    }
    
    # GPU information enhancement
    if torch.cuda.is_available():
        gpu_details = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_details[f'gpu_{i}'] = {
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'memory_gb': props.total_memory / (1024**3),
                'multiprocessor_count': props.multi_processor_count
            }
        enhanced_info['gpu_details'] = gpu_details
    
    return enhanced_info

def log_hardware_config(hw_info):
    """Log hardware configuration in a structured way."""
    logger.info("\n[Hardware Configuration]")
    for k, v in hw_info.items():
        if isinstance(v, dict):
            logger.info(f"{k:>20}:")
            for sub_k, sub_v in v.items():
                logger.info(f"{'':>22}{sub_k}: {sub_v}")
        else:
            logger.info(f"{k:>20}: {v}")

def list_custom_presets() -> List[str]:
    """List all available custom presets."""
    custom_dir = CONFIG_DIR / "custom_presets"
    if not custom_dir.exists():
        return []
    
    return [f.stem.replace("preset_", "") for f in custom_dir.glob("preset_*.json")]

# Model Architecture Options
MODEL_VARIANTS = {
    'SimpleAutoencoder': SimpleAutoencoder,
    'EnhancedAutoencoder': EnhancedAutoencoder,
    'AutoencoderEnsemble': AutoencoderEnsemble
}

# Consolidated Preset Configurations
# Initialize empty PRESET_CONFIGS to avoid forward reference errors
PRESET_CONFIGS = {}

def get_available_presets():
    """Dynamically get available preset names."""
    return list(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else ['default', 'stability', 'performance', 'baseline', 'debug', 'lightweight', 'advanced']

def get_preset_descriptions():
    """Dynamically get preset descriptions."""
    return {k: v.get("metadata", {}).get("description", f"{k.title()} preset") 
            for k, v in PRESET_CONFIGS.items() 
            if isinstance(v, dict)} if PRESET_CONFIGS else {
        'default': 'Default balanced configuration for general use',
        'stability': 'High stability configuration for reliable training',
        'performance': 'High-performance configuration for production deployment',
        'baseline': 'Standardized configuration for benchmarking',
        'debug': 'Lightweight configuration for debugging',
        'lightweight': 'Lightweight configuration for edge devices',
        'advanced': 'Advanced configuration for research experiments'
    }

# Preset Configurations for Testing Different Architectures
DEFAULT_PRESET = {
    'metadata': {
        'description': 'Default balanced configuration for general use',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 8,
            'cpu_cores': 4,
            'ram_gb': 8
        },
        'compatibility': ['SimpleAutoencoder', 'EnhancedAutoencoder', 'AutoencoderEnsemble'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'default'
    },
    'training': {
        'batch_size': 64,
        'epochs': 10,
        'learning_rate': 0.001,
        'patience': 100,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'num_workers': min(4, os.cpu_count() or 1),
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau'
    },
    'model': {
        'model_type': 'EnhancedAutoencoder',
        'encoding_dim': 10,
        'hidden_dims': [128, 64],
        'dropout_rates': [0.2, 0.15],
        'activation': 'leaky_relu',
        'activation_param': 0.2,
        'normalization': 'batch',
        'use_batch_norm': True,
        'use_layer_norm': False,
        'diversity_factor': 0.1,
        'min_features': 5,
        'num_models': 3,
        'skip_connection': True,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 95,
        'attack_threshold': 0.3,
        'false_negative_cost': 2.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'percentile',
        'early_warning_threshold': 0.25
    },
    'data': {
        'normal_samples': 8000,
        'attack_samples': 2000,
        'features': 20,
        'normalization': 'standard',
        'anomaly_factor': 1.5,
        'random_state': 42,
        'validation_split': 0.2,
        'test_split': 0.2,
        'synthetic_generation': {
            'cluster_variance': 0.1,
            'anomaly_sparsity': 0.3
        }
    },
    'monitoring': {
        'metrics_frequency': 10,
        'checkpoint_frequency': 5,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 8,
        'minimum_system_requirements': {
            'cpu_cores': 4,
            'ram_gb': 8,
            'disk_space': 10
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'default',
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo',
        'direction': 'minimize',
        'n_trials': 100,
        'timeout': 3600,
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner'
    }
}

STABILITY_PRESET = {
    'metadata': {
        'description': 'High stability configuration for reliable training',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 4,
            'cpu_cores': 2,
            'ram_gb': 4
        },
        'compatibility': ['SimpleAutoencoder', 'EnhancedAutoencoder'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'stability'
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0005,
        'patience': 15,
        'weight_decay': 1e-3,
        'gradient_clip': 0.5,
        'gradient_accumulation_steps': 1,
        'mixed_precision': False,
        'num_workers': 2,
        'optimizer': 'Adam',
        'scheduler': None
    },
    'model': {
        'model_type': 'SimpleAutoencoder',
        'encoding_dim': 8,
        'hidden_dims': [64],
        'dropout_rates': [0.3],
        'activation': 'relu',
        'activation_param': 0.0,
        'normalization': None,
        'use_batch_norm': False,
        'use_layer_norm': False,
        'diversity_factor': 0.0,
        'min_features': 5,
        'num_models': 1,
        'skip_connection': False,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 99,
        'attack_threshold': 0.2,
        'false_negative_cost': 3.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'fixed_percentile',
        'early_warning_threshold': 0.15
    },
    'data': {
        'normal_samples': 5000,
        'attack_samples': 1000,
        'features': 15,
        'normalization': 'minmax',
        'anomaly_factor': 2.0,
        'random_state': 42,
        'validation_split': 0.25,
        'test_split': 0.25,
        'synthetic_generation': {
            'cluster_variance': 0.05,
            'anomaly_sparsity': 0.2
        }
    },
    'monitoring': {
        'metrics_frequency': 5,
        'checkpoint_frequency': 10,
        'tensorboard_logging': False,
        'console_logging_level': 'DEBUG'
    },
    'hardware': {
        'recommended_gpu_memory': 4,
        'minimum_system_requirements': {
            'cpu_cores': 2,
            'ram_gb': 4,
            'disk_space': 10
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'stability',
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo',
        'direction': 'minimize',
        'n_trials': 100,
        'timeout': 3600,
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner'
    }
}

PERFORMANCE_PRESET = {
    'metadata': {
        'description': 'High-performance configuration for production deployment',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 8,
            'cpu_cores': 8,
            'ram_gb': 16
        },
        'compatibility': ['EnhancedAutoencoder', 'AutoencoderEnsemble'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'performance'
    },
    'training': {
        'batch_size': 128,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patience': 20,
        'weight_decay': 1e-5,
        'gradient_clip': 0.1,
        'gradient_accumulation_steps': 8,
        'mixed_precision': True,
        'num_workers': max(4, os.cpu_count() or 4),
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealing'
    },
    'model': {
        'model_type': 'AutoencoderEnsemble',
        'encoding_dim': 16,
        'hidden_dims': [256, 128, 64],
        'dropout_rates': [0.1, 0.1, 0.05],
        'activation': 'gelu',
        'activation_param': 0.0,
        'normalization': 'batch',
        'use_batch_norm': True,
        'use_layer_norm': False,
        'diversity_factor': 0.2,
        'min_features': 10,
        'num_models': 5,
        'skip_connection': True,
        'residual_blocks': True,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 90,
        'attack_threshold': 0.4,
        'false_negative_cost': 1.5,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'dynamic_percentile',
        'early_warning_threshold': 0.35
    },
    'data': {
        'normal_samples': 10000,
        'attack_samples': 3000,
        'features': 30,
        'normalization': 'standard',
        'anomaly_factor': 1.2,
        'random_state': 42,
        'validation_split': 0.15,
        'test_split': 0.15,
        'synthetic_generation': {
            'cluster_variance': 0.15,
            'anomaly_sparsity': 0.4
        }
    },
    'monitoring': {
        'metrics_frequency': 20,
        'checkpoint_frequency': 25,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 8,
        'minimum_system_requirements': {
            'cpu_cores': 8,
            'ram_gb': 16,
            'disk_space': 20
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'performance',
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': True,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_performance',
        'direction': 'minimize',
        'n_trials': 200,
        'timeout': 7200,
        'sampler': 'TPESampler',
        'pruner': 'HyperbandPruner'
    }
}

BASELINE_PRESET = {
    'metadata': {
        'description': 'Standardized configuration for benchmarking',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 6,
            'cpu_cores': 4,
            'ram_gb': 8
        },
        'compatibility': ['EnhancedAutoencoder'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'baseline'
    },
    'training': {
        'batch_size': 64,
        'epochs': 75,
        'learning_rate': 0.001,
        'patience': 10,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 4,
        'mixed_precision': False,
        'num_workers': min(4, os.cpu_count() or 1),
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau'
    },
    'model': {
        'model_type': 'EnhancedAutoencoder',
        'encoding_dim': 12,
        'hidden_dims': [128, 64],
        'dropout_rates': [0.25, 0.2],
        'activation': 'leaky_relu',
        'activation_param': 0.1,
        'normalization': 'batch',
        'use_batch_norm': True,
        'use_layer_norm': False,
        'diversity_factor': 0.05,
        'min_features': 5,
        'num_models': 1,
        'skip_connection': True,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 95,
        'attack_threshold': 0.3,
        'false_negative_cost': 2.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'fixed_percentile',
        'early_warning_threshold': 0.25
    },
    'data': {
        'normal_samples': 8000,
        'attack_samples': 2000,
        'features': 20,
        'normalization': 'standard',
        'anomaly_factor': 1.5,
        'random_state': 42,
        'validation_split': 0.2,
        'test_split': 0.2,
        'synthetic_generation': {
            'cluster_variance': 0.1,
            'anomaly_sparsity': 0.3
        }
    },
    'monitoring': {
        'metrics_frequency': 10,
        'checkpoint_frequency': 5,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 6,
        'minimum_system_requirements': {
            'cpu_cores': 4,
            'ram_gb': 8,
            'disk_space': 10
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'baseline',
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_baseline',
        'direction': 'minimize',
        'n_trials': 100,
        'timeout': 3600,
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner'
    }
}

DEBUG_PRESET = {
    'metadata': {
        'description': 'Lightweight configuration for debugging',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 2,
            'cpu_cores': 1,
            'ram_gb': 2
        },
        'compatibility': ['SimpleAutoencoder'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'debug'
    },
    'training': {
        'batch_size': 16,
        'epochs': 5,
        'learning_rate': 0.01,
        'patience': 3,
        'weight_decay': 0.0, 
        'gradient_clip': 5.0,
        'gradient_accumulation_steps': 1,
        'mixed_precision': False, 
        'num_workers': 1,
        'optimizer': 'SGD',
        'scheduler': None
    },
    'model': {
        'model_type': 'SimpleAutoencoder',
        'encoding_dim': 4,
        'hidden_dims': [32],
        'dropout_rates': [0.1],
        'activation': 'relu',
        'activation_param': 0.0, 
        'normalization': None,
        'use_batch_norm': False,
        'use_layer_norm': False,
        'diversity_factor': 0.0,
        'min_features': 3,
        'num_models': 1,
        'skip_connection': False,
        'residual_blocks': False, 
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 85, 
        'attack_threshold': 0.5,
        'false_negative_cost': 1.0,
        'enable_security_metrics': False,
        'anomaly_threshold_strategy': 'fixed_percentile',
        'early_warning_threshold': 0.45
    },
    'data': {
        'normal_samples': 100,
        'attack_samples': 50,
        'features': 10,
        'normalization': 'minmax',
        'anomaly_factor': 2.0,
        'random_state': 42,
        'validation_split': 0.3,
        'test_split': 0.3,
        'synthetic_generation': {
            'cluster_variance': 0.2,
            'anomaly_sparsity': 0.5
        }
    },
    'monitoring': {
        'metrics_frequency': 1,
        'checkpoint_frequency': 1,
        'tensorboard_logging': False,
        'console_logging_level': 'DEBUG'
    },
    'hardware': {
        'recommended_gpu_memory': 2,
        'minimum_system_requirements': {
            'cpu_cores': 1,
            'ram_gb': 2,
            'disk_space': 5
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'debug',
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_debug',
        'direction': 'minimize',
        'n_trials': 10,
        'timeout': 600,
        'sampler': 'RandomSampler',
        'pruner': 'NopPruner'
    }
}

LIGHTWEIGHT_PRESET = {
    'metadata': {
        'description': 'Lightweight configuration for edge devices',
        'version': '2.1',  # Updated to match config_version in base_config
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 1,
            'cpu_cores': 1,
            'ram_gb': 2
        },
        'compatibility': ['SimpleAutoencoder'],
        'system': {  # Added to match base_config structure
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',  # Added to match base_config
        'preset_used': 'lightweight'  # Added to identify this preset
    },
    'training': {
        'batch_size': 8,  # Very small batches for edge devices
        'epochs': 30,  # Limited training duration
        'learning_rate': 0.005,  # Slightly higher rate for faster convergence
        'patience': 5,  # Short patience
        'weight_decay': 0.0,  # No regularization to reduce complexity
        'gradient_clip': 2.0,  # Moderate clipping
        'gradient_accumulation_steps': 1,  # No accumulation
        'mixed_precision': False,  # Disabled for edge compatibility
        'num_workers': 1,  # Minimal workers
        'optimizer': 'Adam',  # Basic optimizer
        'scheduler': None  # No scheduler for simplicity
    },
    'model': {
        'model_type': 'SimpleAutoencoder',  # Simplest model type
        'encoding_dim': 6,  # Small encoding dimension
        'hidden_dims': [48],  # Minimal hidden layers
        'dropout_rates': [0.15],  # Light dropout
        'activation': 'relu',  # Simple activation
        'activation_param': 0.0,  # No parameters
        'normalization': None,  # No normalization layers
        'use_batch_norm': False,  # Disabled
        'use_layer_norm': False,  # Disabled
        'diversity_factor': 0.0,  # No diversity
        'min_features': 4,  # Very few features
        'num_models': 1,  # Single model
        'skip_connection': False,  # Disabled
        'residual_blocks': False,  # Disabled
        'model_types': list(MODEL_VARIANTS.keys()),  # Added to match base_config
        'available_activations': ["relu", "leaky_relu", "gelu"],  # Added to match base_config
        'available_normalizations': ['batch', 'layer', None]  # Added to match base_config
    },
    'security': {
        'percentile': 92,  # Slightly relaxed threshold
        'attack_threshold': 0.35,  # Moderate threshold
        'false_negative_cost': 1.2,  # Balanced cost
        'enable_security_metrics': True,  # Enabled but lightweight
        'anomaly_threshold_strategy': 'fixed_percentile',  # Simple strategy
        'early_warning_threshold': 0.3  # Moderate warning
    },
    'data': {
        'normal_samples': 2000,  # Small dataset
        'attack_samples': 500,
        'features': 12,  # Few features
        'normalization': 'minmax',  # Simple normalization
        'anomaly_factor': 1.8,  # Clear anomalies
        'random_state': 42,
        'validation_split': 0.25,  # Larger validation set
        'test_split': 0.25,  # Larger test set
        'synthetic_generation': {
            'cluster_variance': 0.08,  # Tight clusters
            'anomaly_sparsity': 0.25  # Fewer anomalies
        }
    },
    'monitoring': {
        'metrics_frequency': 5,  # Reduced monitoring
        'checkpoint_frequency': 5,  # Reduced checkpoints
        'tensorboard_logging': False,  # Disabled for edge
        'console_logging_level': 'INFO'  # Standard logging
    },
    'hardware': {  # Added to match base_config
        'recommended_gpu_memory': 1,
        'minimum_system_requirements': {
            'cpu_cores': 1,
            'ram_gb': 2,
            'disk_space': 5  # Minimal disk space
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'lightweight',
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {  # Added to match base_config
        'enabled': False,  # Disabled for edge
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_lightweight',
        'direction': 'minimize',
        'n_trials': 50,  # Few trials if enabled
        'timeout': 1800,  # Short timeout (30 minutes)
        'sampler': 'RandomSampler',  # Simple sampler
        'pruner': 'NopPruner'  # No pruning
    }
}

ADVANCED_PRESET = {
    'metadata': {
        'description': 'Advanced configuration for research experiments',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 16,
            'cpu_cores': 16,
            'ram_gb': 32
        },
        'compatibility': ['EnhancedAutoencoder', 'AutoencoderEnsemble'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'advanced'
    },
    'training': {
        'batch_size': 256,
        'epochs': 300,
        'learning_rate': 0.00005,
        'patience': 30,
        'weight_decay': 1e-6,
        'gradient_clip': 0.05,
        'gradient_accumulation_steps': 16,
        'mixed_precision': True,
        'num_workers': max(8, os.cpu_count() or 8),
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingWarmRestarts'
    },
    'model': {
        'model_type': 'AutoencoderEnsemble',
        'encoding_dim': 24,
        'hidden_dims': [512, 256, 128, 64],
        'dropout_rates': [0.05, 0.05, 0.03, 0.02],
        'activation': 'gelu',
        'activation_param': 0.0,
        'normalization': 'layer',
        'use_batch_norm': False,
        'use_layer_norm': True,
        'diversity_factor': 0.3,
        'min_features': 15,
        'num_models': 7,
        'skip_connection': True,
        'residual_blocks': True,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 88,
        'attack_threshold': 0.45,
        'false_negative_cost': 1.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'dynamic_percentile',
        'early_warning_threshold': 0.4
    },
    'data': {
        'normal_samples': 20000,
        'attack_samples': 5000,
        'features': 50,
        'normalization': 'standard',
        'anomaly_factor': 1.1,
        'random_state': 42,
        'validation_split': 0.1,
        'test_split': 0.1,
        'synthetic_generation': {
            'cluster_variance': 0.2,
            'anomaly_sparsity': 0.5
        }
    },
    'monitoring': {
        'metrics_frequency': 50,
        'checkpoint_frequency': 50,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 16,
        'minimum_system_requirements': {
            'cpu_cores': 16,
            'ram_gb': 32,
            'disk_space': 50
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'advanced',
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': True,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_advanced',
        'direction': 'minimize',
        'n_trials': 500,
        'timeout': 14400,
        'sampler': 'TPESampler',
        'pruner': 'HyperbandPruner'
    }
}

# Configuration for Testing Different Architectures
DEFAULT_CONFIG = {
    'metadata': {
        'description': 'Default balanced configuration for general use',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 8,
            'cpu_cores': 4,
            'ram_gb': 8
        },
        'compatibility': ['SimpleAutoencoder', 'EnhancedAutoencoder', 'AutoencoderEnsemble'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'default'
    },
    'training': {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 10,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'num_workers': min(4, os.cpu_count() or 1),
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau'
    },
    'model': {
        'model_type': 'EnhancedAutoencoder',
        'encoding_dim': 12,
        'hidden_dims': [128, 64],
        'dropout_rates': [0.2, 0.15],
        'activation': 'leaky_relu',
        'activation_param': 0.1,
        'normalization': 'batch',
        'use_batch_norm': True,
        'use_layer_norm': False,
        'diversity_factor': 0.1,
        'min_features': 5,
        'num_models': 1,
        'skip_connection': True,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 95,
        'attack_threshold': 0.3,
        'false_negative_cost': 2.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'percentile',
        'early_warning_threshold': 0.25
    },
    'data': {
        'normal_samples': 8000,
        'attack_samples': 2000,
        'features': 20,
        'normalization': 'standard',
        'anomaly_factor': 1.5,
        'random_state': 42,
        'validation_split': 0.2,
        'test_split': 0.2,
        'synthetic_generation': {
            'cluster_variance': 0.1,
            'anomaly_sparsity': 0.3
        }
    },
    'monitoring': {
        'metrics_frequency': 10,
        'checkpoint_frequency': 5,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 8,
        'minimum_system_requirements': {
            'cpu_cores': 4,
            'ram_gb': 8,
            'disk_space': 10
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'default',
        'current_override': None,
        'override_rules': {
            'security': False,
            'monitoring': True,
            'hardware': False
        },
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo',
        'direction': 'minimize',
        'n_trials': 100,
        'timeout': 3600,
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner'
    }
}

STABILITY_CONFIG = {
    'metadata': {
        'description': 'Stability-focused configuration for architecture testing',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'compatibility': list(MODEL_VARIANTS.keys()),
        'base_preset': 'stability',
        'config_type': 'architecture_test',
        'recommended_hardware': {
            'gpu_memory_gb': 4,
            'cpu_cores': 2,
            'ram_gb': 4
        },
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        }
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 1e-3,
        'patience': 15,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 1,
        'mixed_precision': False,
        'num_workers': 2,
        'optimizer': 'Adam',
        'scheduler': None
    },
    'model': {
        'model_type': 'SimpleAutoencoder',
        'encoding_dim': 8,
        'hidden_dims': [64, 32],
        'dropout_rates': [0.3, 0.25],
        'activation': 'relu',
        'activation_param': 0.0,
        'normalization': None,
        'use_batch_norm': True,
        'use_layer_norm': False,
        'diversity_factor': 0.0,
        'min_features': 5,
        'num_models': 1,
        'skip_connection': False,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 99,
        'attack_threshold': 0.2,
        'false_negative_cost': 2.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'fixed_percentile',
        'early_warning_threshold': 0.15
    },
    'data': STABILITY_PRESET['data'],
    'monitoring': STABILITY_PRESET['monitoring'],
    'hardware': STABILITY_PRESET['hardware'],
    'testing': {
        'num_architecture_variants': 5,
        'stability_threshold': 0.95,
        'convergence_tolerance': 1e-4,
        'max_variance': 0.1,
        'test_cycles': 3,
        'stability_metrics': ['loss_variance', 'gradient_norm', 'parameter_updates']
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'stability',
        'current_override': None,
        'override_rules': {
            'security': False,
            'monitoring': True,
            'hardware': False
        },
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'stability_hpo',
        'direction': 'minimize',
        'n_trials': 50,
        'timeout': 1800,
        'sampler': 'RandomSampler',
        'pruner': 'NopPruner'
    }
}

PERFORMANCE_CONFIG = {
    'metadata': {
        'description': 'Performance-optimized configuration for architecture testing',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'compatibility': ['EnhancedAutoencoder', 'AutoencoderEnsemble'],
        'base_preset': 'performance',
        'config_type': 'performance_test',
        'benchmark_reference': ['PERFORMANCE_PRESET', 'ADVANCED_PRESET'],
        'testing_focus': ['throughput', 'latency', 'memory_efficiency'],
        'recommended_hardware': {
            'gpu_memory_gb': 8,
            'cpu_cores': 8,
            'ram_gb': 16
        }
    },
    'training': {
        'batch_size': 128,
        'epochs': 200,
        'learning_rate': 5e-4,
        'patience': 20,
        'weight_decay': 1e-5,
        'gradient_clip': 0.5,
        'gradient_accumulation_steps': 8,
        'mixed_precision': True,
        'num_workers': max(4, os.cpu_count() or 4),
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealing'
    },
    'model': {
        'model_type': 'AutoencoderEnsemble',
        'encoding_dim': 16,
        'hidden_dims': [256, 128, 64],
        'dropout_rates': [0.2, 0.15],
        'activation': 'gelu',
        'activation_param': 0.0,
        'normalization': 'batch',
        'use_batch_norm': True,
        'use_layer_norm': False,
        'diversity_factor': 0.2,
        'min_features': 10,
        'num_models': 3,
        'skip_connection': True,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 92,
        'attack_threshold': 0.4,
        'false_negative_cost': 1.5,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'dynamic_percentile',
        'early_warning_threshold': 0.35
    },
    'data': {
        **PERFORMANCE_PRESET['data'],
        'features': 25
    },
    'monitoring': {
        'metrics_frequency': 15,
        'checkpoint_frequency': 20,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': PERFORMANCE_PRESET['hardware'],
    'performance_metrics': {
        'target_throughput': 1000,
        'max_latency': 50,
        'memory_threshold': 0.8,
        'warmup_cycles': 3,
        'measurement_cycles': 5,
        'stability_requirement': 0.9
    },
    'architecture_tests': {
        'variants_to_test': [
            {'name': 'baseline', 'config': 'PERFORMANCE_PRESET'},
            {'name': 'reduced_ensemble', 'num_models': 3},
            {'name': 'increased_dropout', 'dropout_rates': [0.2, 0.15]},
            {'name': 'batch_norm_only', 'use_batch_norm': True, 'use_layer_norm': False}
        ],
        'comparison_metrics': [
            'throughput',
            'latency',
            'memory_usage',
            'reconstruction_error',
            'training_stability'
        ]
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'performance',
        'current_override': None,
        'override_rules': {
            'security': False,
            'monitoring': True,
            'hardware': False
        },
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': True,
        'strategy': 'optuna',
        'study_name': 'performance_hpo',
        'direction': 'minimize',
        'n_trials': 200,
        'timeout': 7200,
        'sampler': 'TPESampler',
        'pruner': 'HyperbandPruner'
    }
}

BASELINE_CONFIG = {
    'metadata': {
        'description': 'Standardized configuration for benchmarking',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 6,
            'cpu_cores': 4,
            'ram_gb': 8
        },
        'compatibility': ['EnhancedAutoencoder'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'baseline'
    },
    'training': {
        'batch_size': 64,
        'epochs': 75,
        'learning_rate': 0.001,
        'patience': 10,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 4,
        'mixed_precision': False,
        'num_workers': min(4, os.cpu_count() or 1),
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau'
    },
    'model': {
        'model_type': 'EnhancedAutoencoder',
        'encoding_dim': 12,
        'hidden_dims': [128, 64],
        'dropout_rates': [0.25, 0.2],
        'activation': 'leaky_relu',
        'activation_param': 0.1,
        'normalization': 'batch',
        'use_batch_norm': True,
        'use_layer_norm': False,
        'diversity_factor': 0.05,
        'min_features': 5,
        'num_models': 1,
        'skip_connection': True,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 95,
        'attack_threshold': 0.3,
        'false_negative_cost': 2.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'fixed_percentile',
        'early_warning_threshold': 0.25
    },
    'data': {
        'normal_samples': 8000,
        'attack_samples': 2000,
        'features': 20,
        'normalization': 'standard',
        'anomaly_factor': 1.5,
        'random_state': 42,
        'validation_split': 0.2,
        'test_split': 0.2,
        'synthetic_generation': {
            'cluster_variance': 0.1,
            'anomaly_sparsity': 0.3
        }
    },
    'monitoring': {
        'metrics_frequency': 10,
        'checkpoint_frequency': 5,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 6,
        'minimum_system_requirements': {
            'cpu_cores': 4,
            'ram_gb': 8,
            'disk_space': 10
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'baseline',
        'current_override': None,
        'override_rules': {
            'security': False,
            'monitoring': True,
            'hardware': False
        },
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_baseline',
        'direction': 'minimize',
        'n_trials': 100,
        'timeout': 3600,
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner'
    }
}

DEBUG_CONFIG = {
    'metadata': {
        'description': 'Lightweight configuration for debugging',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 2,
            'cpu_cores': 1,
            'ram_gb': 2
        },
        'compatibility': ['SimpleAutoencoder'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'debug'
    },
    'training': {
        'batch_size': 16,
        'epochs': 5,
        'learning_rate': 0.01,
        'patience': 3,
        'weight_decay': 0.0,
        'gradient_clip': 5.0,
        'gradient_accumulation_steps': 1,
        'mixed_precision': False,
        'num_workers': 1,
        'optimizer': 'SGD',
        'scheduler': None
    },
    'model': {
        'model_type': 'SimpleAutoencoder',
        'encoding_dim': 4,
        'hidden_dims': [32],
        'dropout_rates': [0.1],
        'activation': 'relu',
        'activation_param': 0.0,
        'normalization': None,
        'use_batch_norm': False,
        'use_layer_norm': False,
        'diversity_factor': 0.0,
        'min_features': 3,
        'num_models': 1,
        'skip_connection': False,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 85,
        'attack_threshold': 0.5,
        'false_negative_cost': 1.0,
        'enable_security_metrics': False,
        'anomaly_threshold_strategy': 'fixed_percentile',
        'early_warning_threshold': 0.45
    },
    'data': {
        'normal_samples': 100,
        'attack_samples': 50,
        'features': 10,
        'normalization': 'minmax',
        'anomaly_factor': 2.0,
        'random_state': 42,
        'validation_split': 0.3,
        'test_split': 0.3,
        'synthetic_generation': {
            'cluster_variance': 0.2,
            'anomaly_sparsity': 0.5
        }
    },
    'monitoring': {
        'metrics_frequency': 1,
        'checkpoint_frequency': 1,
        'tensorboard_logging': False,
        'console_logging_level': 'DEBUG'
    },
    'hardware': {
        'recommended_gpu_memory': 2,
        'minimum_system_requirements': {
            'cpu_cores': 1,
            'ram_gb': 2,
            'disk_space': 5
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'debug',
        'current_override': None,
        'override_rules': {
            'security': False,
            'monitoring': True,
            'hardware': False
        },
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_debug',
        'direction': 'minimize',
        'n_trials': 10,
        'timeout': 600,
        'sampler': 'RandomSampler',
        'pruner': 'NopPruner'
    }
}

LIGHTWEIGHT_CONFIG = {
    'metadata': {
        'description': 'Lightweight configuration for edge devices',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 1,
            'cpu_cores': 1,
            'ram_gb': 2
        },
        'compatibility': ['SimpleAutoencoder'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'lightweight'
    },
    'training': {
        'batch_size': 8,
        'epochs': 30,
        'learning_rate': 0.005,
        'patience': 5,
        'weight_decay': 0.0,
        'gradient_clip': 2.0,
        'gradient_accumulation_steps': 1,
        'mixed_precision': False,
        'num_workers': 1,
        'optimizer': 'Adam',
        'scheduler': None
    },
    'model': {
        'model_type': 'SimpleAutoencoder',
        'encoding_dim': 6,
        'hidden_dims': [48],
        'dropout_rates': [0.15],
        'activation': 'relu',
        'activation_param': 0.0,
        'normalization': None,
        'use_batch_norm': False,
        'use_layer_norm': False,
        'diversity_factor': 0.0,
        'min_features': 4,
        'num_models': 1,
        'skip_connection': False,
        'residual_blocks': False,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 92,
        'attack_threshold': 0.35,
        'false_negative_cost': 1.2,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'fixed_percentile',
        'early_warning_threshold': 0.3
    },
    'data': {
        'normal_samples': 2000,
        'attack_samples': 500,
        'features': 12,
        'normalization': 'minmax',
        'anomaly_factor': 1.8,
        'random_state': 42,
        'validation_split': 0.25,
        'test_split': 0.25,
        'synthetic_generation': {
            'cluster_variance': 0.08,
            'anomaly_sparsity': 0.25
        }
    },
    'monitoring': {
        'metrics_frequency': 5,
        'checkpoint_frequency': 5,
        'tensorboard_logging': False,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 1,
        'minimum_system_requirements': {
            'cpu_cores': 1,
            'ram_gb': 2,
            'disk_space': 5
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'lightweight',
        'current_override': None,
        'override_rules': {
            'security': False,
            'monitoring': True,
            'hardware': False
        },
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': False,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_lightweight',
        'direction': 'minimize',
        'n_trials': 50,
        'timeout': 1800,
        'sampler': 'RandomSampler',
        'pruner': 'NopPruner'
    }
}

ADVANCED_CONFIG = {
    'metadata': {
        'description': 'Advanced configuration for research experiments',
        'version': '2.1',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 16,
            'cpu_cores': 16,
            'ram_gb': 32
        },
        'compatibility': ['EnhancedAutoencoder', 'AutoencoderEnsemble'],
        'system': {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'hostname': platform.node(),
            'os': platform.system()
        },
        'config_type': 'autoencoder',
        'preset_used': 'advanced'
    },
    'training': {
        'batch_size': 256,
        'epochs': 300,
        'learning_rate': 0.00005,
        'patience': 30,
        'weight_decay': 1e-6,
        'gradient_clip': 0.05,
        'gradient_accumulation_steps': 16,
        'mixed_precision': True,
        'num_workers': max(8, os.cpu_count() or 8),
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingWarmRestarts'
    },
    'model': {
        'model_type': 'AutoencoderEnsemble',
        'encoding_dim': 24,
        'hidden_dims': [512, 256, 128, 64],
        'dropout_rates': [0.05, 0.05, 0.03, 0.02],
        'activation': 'gelu',
        'activation_param': 0.0,
        'normalization': 'layer',
        'use_batch_norm': False,
        'use_layer_norm': True,
        'diversity_factor': 0.3,
        'min_features': 15,
        'num_models': 7,
        'skip_connection': True,
        'residual_blocks': True,
        'model_types': list(MODEL_VARIANTS.keys()),
        'available_activations': ["relu", "leaky_relu", "gelu"],
        'available_normalizations': ['batch', 'layer', None]
    },
    'security': {
        'percentile': 88,
        'attack_threshold': 0.45,
        'false_negative_cost': 1.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'dynamic_percentile',
        'early_warning_threshold': 0.4
    },
    'data': {
        'normal_samples': 20000,
        'attack_samples': 5000,
        'features': 50,
        'normalization': 'standard',
        'anomaly_factor': 1.1,
        'random_state': 42,
        'validation_split': 0.1,
        'test_split': 0.1,
        'synthetic_generation': {
            'cluster_variance': 0.2,
            'anomaly_sparsity': 0.5
        }
    },
    'monitoring': {
        'metrics_frequency': 50,
        'checkpoint_frequency': 50,
        'tensorboard_logging': True,
        'console_logging_level': 'INFO'
    },
    'hardware': {
        'recommended_gpu_memory': 16,
        'minimum_system_requirements': {
            'cpu_cores': 16,
            'ram_gb': 32,
            'disk_space': 50
        }
    },
    'presets': {
        'available_presets': get_available_presets(),
        'current_preset': 'advanced',
        'current_override': None,
        'override_rules': {
            'security': False,
            'monitoring': True,
            'hardware': False
        },
        'preset_configs': get_preset_descriptions(),
        'custom_presets_available': list_custom_presets()
    },
    'hyperparameter_optimization': {
        'enabled': True,
        'strategy': 'optuna',
        'study_name': 'autoencoder_hpo_advanced',
        'direction': 'minimize',
        'n_trials': 500,
        'timeout': 14400,
        'sampler': 'TPESampler',
        'pruner': 'HyperbandPruner'
    }
}

# Populate PRESET_CONFIGS after all presets are defined
PRESET_CONFIGS.update({
    'default': DEFAULT_PRESET,
    'stability': STABILITY_PRESET,
    'performance': PERFORMANCE_PRESET,
    'baseline': BASELINE_PRESET,
    'debug': DEBUG_PRESET,
    'lightweight': LIGHTWEIGHT_PRESET,
    'advanced': ADVANCED_PRESET,
    'stability_config': STABILITY_CONFIG,
    'performance_config': PERFORMANCE_CONFIG
})

_cached_config = None
_config_cache_time = None

def get_current_config() -> Dict[str, Any]:
    """Return comprehensive configuration with preset awareness and validation.
    
    Returns:
        Nested dictionary containing all configuration parameters with metadata,
        respecting any active preset configuration.
    """
    global _cached_config, _config_cache_time
    
    # Use cached config if it's fresh (less than 30 seconds old)
    current_time = time.time()
    if (_cached_config is not None and _config_cache_time is not None and 
        current_time - _config_cache_time < 30):
        return _cached_config
    
    try:
        # First check if there's a loaded config with an active preset
        loaded_config = load_config()
        current_preset = loaded_config.get('presets', {}).get('current_preset') if loaded_config else None
    except Exception as e:
        logger.debug(f"Failed to load config: {e}")
        loaded_config = None
        current_preset = None
    
    # If we have an active preset, start with that configuration
    if current_preset and current_preset in PRESET_CONFIGS:
        try:
            base_config = deepcopy(PRESET_CONFIGS[current_preset])
            logger.info(f"Using preset configuration: {current_preset}")
        except Exception as e:
            logger.warning(f"Failed to load preset {current_preset}: {e}")
            base_config = None
    else:
        base_config = None
    
    # If preset loading failed or no preset, use default configuration structure
    if base_config is None:
        base_config = {
            "metadata": {
                "config_version": "2.1",
                "config_type": "autoencoder",
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "system": {
                    "python_version": platform.python_version(),
                    "pytorch_version": torch.__version__ if 'torch' in globals() else "unknown",
                    "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
                    "hostname": platform.node(),
                    "os": platform.system()
                },
                "preset_used": current_preset if current_preset else "none"
            },
            "training": {
                "batch_size": globals().get('DEFAULT_BATCH_SIZE', 64),
                "epochs": globals().get('DEFAULT_EPOCHS', 100),
                "learning_rate": globals().get('LEARNING_RATE', 0.001),
                "patience": globals().get('EARLY_STOPPING_PATIENCE', 10),
                "weight_decay": globals().get('WEIGHT_DECAY', 1e-4),
                "gradient_clip": globals().get('GRADIENT_CLIP', 1.0),
                "gradient_accumulation_steps": globals().get('GRADIENT_ACCUMULATION_STEPS', 4),
                "mixed_precision": globals().get('MIXED_PRECISION', True),
                "num_workers": globals().get('NUM_WORKERS', min(4, os.cpu_count() or 1)),
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau"
            },
            "model": {
                "model_type": "EnhancedAutoencoder",
                "encoding_dim": globals().get('DEFAULT_ENCODING_DIM', 12),
                "hidden_dims": globals().get('HIDDEN_LAYER_SIZES', [128, 64]),
                "dropout_rates": globals().get('DROPOUT_RATES', [0.2, 0.15]),
                "activation": globals().get('ACTIVATION', 'leaky_relu'),
                "activation_param": globals().get('ACTIVATION_PARAM', 0.1),
                "normalization": globals().get('NORMALIZATION', 'batch'),
                "use_batch_norm": globals().get('USE_BATCH_NORM', True),
                "use_layer_norm": globals().get('USE_LAYER_NORM', False),
                "diversity_factor": globals().get('DIVERSITY_FACTOR', 0.1),
                "min_features": globals().get('MIN_FEATURES', 5),
                "num_models": globals().get('NUM_MODELS', 1),
                "skip_connection": True,
                "residual_blocks": False,
                "model_types": list(MODEL_VARIANTS.keys()) if 'MODEL_VARIANTS' in globals() and MODEL_VARIANTS else ['SimpleAutoencoder', 'EnhancedAutoencoder'],
                "available_activations": ["relu", "leaky_relu", "gelu"],
                "available_normalizations": ["batch", "layer", None]
            },
            "security": {
                "percentile": globals().get('DEFAULT_PERCENTILE', 95),
                "attack_threshold": globals().get('DEFAULT_ATTACK_THRESHOLD', 0.3),
                "false_negative_cost": globals().get('FALSE_NEGATIVE_COST', 2.0),
                "enable_security_metrics": globals().get('SECURITY_METRICS', True),
                "anomaly_threshold_strategy": "percentile",
                "early_warning_threshold": 0.25
            },
            "data": {
                "normal_samples": globals().get('NORMAL_SAMPLES', 8000),
                "attack_samples": globals().get('ATTACK_SAMPLES', 2000),
                "features": globals().get('FEATURES', 20),
                "normalization": "standard",
                "anomaly_factor": globals().get('ANOMALY_FACTOR', 1.5),
                "random_state": globals().get('RANDOM_STATE', 42),
                "validation_split": 0.2,
                "test_split": 0.2,
                "synthetic_generation": {
                    "cluster_variance": 0.1,
                    "anomaly_sparsity": 0.3
                }
            },
            "monitoring": {
                "metrics_frequency": 10,
                "checkpoint_frequency": 5,
                "tensorboard_logging": True,
                "console_logging_level": "INFO"
            },
            "hardware": {
                "recommended_gpu_memory": 8,
                "minimum_system_requirements": {
                    "cpu_cores": 4,
                    "ram_gb": 8,
                    "disk_space": 10
                }
            },
            "presets": {
                "available_presets": get_available_presets(),
                "current_preset": current_preset,
                "preset_configs": get_preset_descriptions(),
                "custom_presets_available": get_safe_custom_presets()
            },
            "hyperparameter_optimization": {
                "enabled": False,
                "strategy": "optuna",
                "study_name": "autoencoder_hpo",
                "direction": "minimize",
                "n_trials": 100,
                "timeout": 3600,
                "sampler": "TPESampler",
                "pruner": "MedianPruner"
            }
        }
    
    # Merge with any loaded configuration (but be careful about errors)
    if loaded_config:
        try:
            base_config = deep_update(base_config, loaded_config)
        except Exception as e:
            logger.warning(f"Failed to merge loaded config: {e}")
    
    # Ensure model architecture compatibility if preset was used
    if current_preset:
        try:
            model_type = base_config.get('model', {}).get('model_type')
            if 'MODEL_VARIANTS' in globals() and MODEL_VARIANTS and not validate_model_preset_compatibility(model_type, base_config):
                logger.warning(f"Model type {model_type} may not be fully compatible with preset {current_preset}")
                # Apply fallback to simple model if compatibility issues
                if model_type == 'AutoencoderEnsemble' and base_config.get('model', {}).get('num_models', 1) < 1:
                    base_config.setdefault('model', {})['num_models'] = 1
        except Exception as e:
            logger.debug(f"Model compatibility check failed: {e}")
    
    # Update preset information dynamically in case PRESET_CONFIGS was populated after creation
    try:
        if 'presets' in base_config:
            base_config['presets']['available_presets'] = get_available_presets()
            base_config['presets']['preset_configs'] = get_preset_descriptions()
            base_config['presets']['custom_presets_available'] = get_safe_custom_presets()
    except Exception as e:
        logger.debug(f"Failed to update preset information: {e}")
    
    # Cache the result
    _cached_config = base_config
    _config_cache_time = current_time
    
    return base_config

def invalidate_config_cache():
    """Invalidate the configuration cache to force reload with enhanced logging."""
    global _cached_config, _config_cache_time
    
    # Check if cache was active before invalidation
    was_cached = _cached_config is not None and _config_cache_time is not None
    
    if was_cached:
        cache_age = time.time() - _config_cache_time
        logger.debug(f"Invalidating configuration cache (age: {cache_age:.1f}s)")
    
    _cached_config = None
    _config_cache_time = None
    
    # Log cache invalidation for debugging
    if was_cached:
        logger.info("Configuration cache invalidated - next access will reload from source")
    else:
        logger.debug("Cache invalidation requested but no cache was active")

def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary with enhanced conflict resolution and validation.
    
    Args:
        original: The original dictionary to update
        update: The update dictionary containing new values
        
    Returns:
        Updated dictionary with merged values
        
    Raises:
        ValueError: If incompatible values are detected
    """
    if not isinstance(original, dict) or not isinstance(update, dict):
        raise ValueError("Both original and update must be dictionaries")
    
    # Track changes for logging
    changes_made = []
    
    for key, value in update.items():
        try:
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                # Special handling for critical configuration sections
                if key == 'model':
                    # Enhanced model configuration validation
                    original[key] = deep_update_model_section(original[key], value, changes_made)
                elif key == 'training':
                    # Enhanced training configuration validation
                    original[key] = deep_update_training_section(original[key], value, changes_made)
                elif key == 'presets':
                    # Special handling for presets to maintain consistency
                    original[key] = deep_update_presets_section(original[key], value, changes_made)
                elif key == 'metadata':
                    # Special handling for metadata to preserve system info
                    original[key] = deep_update_metadata_section(original[key], value, changes_made)
                else:
                    # Standard recursive update for other sections
                    original[key] = deep_update(original[key], value)
                    
            else:
                # Handle non-dict values with validation
                if value is not None:
                    # Validate critical parameters
                    if key in ['model_type'] and 'MODEL_VARIANTS' in globals() and MODEL_VARIANTS:
                        if value not in MODEL_VARIANTS:
                            logger.warning(f"Ignoring invalid model_type: {value}")
                            continue
                    
                    # Track the change
                    if key in original and original[key] != value:
                        changes_made.append({
                            'key': key,
                            'old_value': original[key],
                            'new_value': value,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    original[key] = value
                else:
                    # Skip None values to avoid overwriting with null
                    logger.debug(f"Skipping None value for key: {key}")
                    
        except Exception as e:
            logger.warning(f"Error updating key '{key}': {e}")
            continue
    
    # Log significant changes
    if changes_made:
        logger.debug(f"deep_update made {len(changes_made)} changes")
        # Log first 10 changes to avoid spam
        for change in changes_made[:10]:
            logger.debug(f"  {change['key']}: {change['old_value']} -> {change['new_value']}")
    
    return original

def deep_update_model_section(original_model: Dict[str, Any], update_model: Dict[str, Any], 
                             changes_made: List[Dict]) -> Dict[str, Any]:
    """Enhanced model section update with validation."""
    result_model = original_model.copy()
    
    # Validate model type first
    if 'model_type' in update_model:
        model_type = update_model['model_type']
        if MODEL_VARIANTS and model_type not in MODEL_VARIANTS:
            logger.warning(f"Ignoring invalid model_type: {model_type}")
            # Don't update model_type, but continue with other updates
        else:
            result_model['model_type'] = model_type
            changes_made.append({
                'key': 'model.model_type',
                'old_value': original_model.get('model_type'),
                'new_value': model_type,
                'timestamp': datetime.now().isoformat()
            })
    
    # Handle hidden_dims and dropout_rates with length validation
    hidden_dims_updated = False
    dropout_rates_updated = False
    
    if 'hidden_dims' in update_model:
        hidden_dims = update_model['hidden_dims']
        if isinstance(hidden_dims, list) and all(isinstance(x, int) and x > 0 for x in hidden_dims):
            result_model['hidden_dims'] = hidden_dims
            hidden_dims_updated = True
            changes_made.append({
                'key': 'model.hidden_dims',
                'old_value': original_model.get('hidden_dims'),
                'new_value': hidden_dims,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.warning(f"Invalid hidden_dims format: {hidden_dims}")
    
    if 'dropout_rates' in update_model:
        dropout_rates = update_model['dropout_rates']
        if isinstance(dropout_rates, list) and all(isinstance(x, (int, float)) and 0 <= x < 1 for x in dropout_rates):
            result_model['dropout_rates'] = dropout_rates
            dropout_rates_updated = True
            changes_made.append({
                'key': 'model.dropout_rates',
                'old_value': original_model.get('dropout_rates'),
                'new_value': dropout_rates,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.warning(f"Invalid dropout_rates format: {dropout_rates}")
    
    # Ensure hidden_dims and dropout_rates have matching lengths
    if hidden_dims_updated or dropout_rates_updated:
        hidden_dims = result_model.get('hidden_dims', [])
        dropout_rates = result_model.get('dropout_rates', [])
        
        if len(hidden_dims) != len(dropout_rates):
            min_length = min(len(hidden_dims), len(dropout_rates))
            if min_length > 0:
                result_model['hidden_dims'] = hidden_dims[:min_length]
                result_model['dropout_rates'] = dropout_rates[:min_length]
                logger.warning(f"Adjusted hidden_dims and dropout_rates to matching length: {min_length}")
                changes_made.append({
                    'key': 'model.length_adjustment',
                    'old_value': {'hidden': len(hidden_dims), 'dropout': len(dropout_rates)},
                    'new_value': {'hidden': min_length, 'dropout': min_length},
                    'timestamp': datetime.now().isoformat()
                })
    
    # Handle other model parameters with validation
    for param in ['encoding_dim', 'activation', 'normalization', 'num_models', 'diversity_factor']:
        if param in update_model:
            value = update_model[param]
            
            # Validate specific parameters
            is_valid = True
            if param == 'encoding_dim' and (not isinstance(value, int) or value < 1):
                logger.warning(f"Invalid encoding_dim: {value}")
                is_valid = False
            elif param == 'num_models' and (not isinstance(value, int) or value < 1):
                logger.warning(f"Invalid num_models: {value}")
                is_valid = False
            elif param == 'diversity_factor' and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                logger.warning(f"Invalid diversity_factor: {value}")
                is_valid = False
            elif param == 'activation' and value not in ['relu', 'leaky_relu', 'gelu']:
                logger.warning(f"Invalid activation: {value}")
                is_valid = False
            elif param == 'normalization' and value not in ['batch', 'layer', None]:
                logger.warning(f"Invalid normalization: {value}")
                is_valid = False
            
            if is_valid:
                result_model[param] = value
                changes_made.append({
                    'key': f'model.{param}',
                    'old_value': original_model.get(param),
                    'new_value': value,
                    'timestamp': datetime.now().isoformat()
                })
    
    # Recursively update remaining keys
    for key, value in update_model.items():
        if key not in ['model_type', 'hidden_dims', 'dropout_rates', 'encoding_dim', 
                      'activation', 'normalization', 'num_models', 'diversity_factor']:
            if isinstance(value, dict) and key in result_model and isinstance(result_model[key], dict):
                result_model[key] = deep_update(result_model[key], value)
            elif value is not None:
                result_model[key] = value
    
    return result_model

def deep_update_training_section(original_training: Dict[str, Any], update_training: Dict[str, Any], 
                                changes_made: List[Dict]) -> Dict[str, Any]:
    """Enhanced training section update with validation."""
    result_training = original_training.copy()
    
    # Validate training parameters
    training_validators = {
        'batch_size': lambda x: isinstance(x, int) and x > 0,
        'epochs': lambda x: isinstance(x, int) and x > 0,
        'learning_rate': lambda x: isinstance(x, (int, float)) and x > 0,
        'patience': lambda x: isinstance(x, int) and x >= 0,
        'weight_decay': lambda x: isinstance(x, (int, float)) and x >= 0,
        'gradient_clip': lambda x: isinstance(x, (int, float)) and x >= 0,
        'num_workers': lambda x: isinstance(x, int) and 0 <= x <= (os.cpu_count() or 1),
        'mixed_precision': lambda x: isinstance(x, bool)
    }
    
    for param, value in update_training.items():
        if param in training_validators:
            if training_validators[param](value):
                result_training[param] = value
                changes_made.append({
                    'key': f'training.{param}',
                    'old_value': original_training.get(param),
                    'new_value': value,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.warning(f"Invalid training parameter {param}: {value}")
        elif value is not None:
            result_training[param] = value
    
    return result_training

def deep_update_presets_section(original_presets: Dict[str, Any], update_presets: Dict[str, Any], 
                               changes_made: List[Dict]) -> Dict[str, Any]:
    """Enhanced presets section update with consistency checks."""
    result_presets = original_presets.copy()
    
    # Handle current_preset with validation
    if 'current_preset' in update_presets:
        preset_name = update_presets['current_preset']
        if preset_name is None or preset_name in get_available_presets():
            result_presets['current_preset'] = preset_name
            changes_made.append({
                'key': 'presets.current_preset',
                'old_value': original_presets.get('current_preset'),
                'new_value': preset_name,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.warning(f"Invalid preset name: {preset_name}")
    
    # Always update dynamic preset information
    try:
        result_presets['available_presets'] = get_available_presets()
        result_presets['preset_configs'] = get_preset_descriptions()
        result_presets['custom_presets_available'] = get_safe_custom_presets()
    except Exception as e:
        logger.warning(f"Failed to update preset information: {e}")
    
    # Update other preset parameters
    for key, value in update_presets.items():
        if key not in ['current_preset', 'available_presets', 'preset_configs', 'custom_presets_available']:
            if value is not None:
                result_presets[key] = value
    
    return result_presets

def deep_update_metadata_section(original_metadata: Dict[str, Any], update_metadata: Dict[str, Any], 
                                changes_made: List[Dict]) -> Dict[str, Any]:
    """Enhanced metadata section update preserving system information."""
    result_metadata = original_metadata.copy()
    
    # Always update modification time
    result_metadata['modified'] = datetime.now().isoformat()
    
    # Preserve system information but allow updates
    if 'system' in update_metadata:
        if 'system' not in result_metadata:
            result_metadata['system'] = {}
        
        # Update system info while preserving existing values
        system_update = update_metadata['system']
        for key, value in system_update.items():
            if value is not None:
                result_metadata['system'][key] = value
    
    # Update other metadata fields
    for key, value in update_metadata.items():
        if key != 'system' and value is not None:
            result_metadata[key] = value
            changes_made.append({
                'key': f'metadata.{key}',
                'old_value': original_metadata.get(key),
                'new_value': value,
                'timestamp': datetime.now().isoformat()
            })
    
    return result_metadata

def save_config(config: Dict, config_path: Path = CONFIG_FILE) -> None:
    """Save config with enhanced metadata, backup handling, and validation.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
        
    Raises:
        RuntimeError: If configuration save fails
        ValueError: If configuration is invalid
    """
    try:
        # Validate configuration before saving
        logger.debug("Validating configuration before save")
        validate_config(config)
        
        # Prepare comprehensive metadata
        save_metadata = {
            "created": config.get('metadata', {}).get('created', datetime.now().isoformat()),
            "modified": datetime.now().isoformat(),
            "version": "2.1",
            "system": {
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__ if 'torch' in globals() else "unknown",
                "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
                "hostname": platform.node(),
                "os": platform.system(),
                "platform_release": platform.release(),
                "architecture": platform.machine()
            },
            "config": {
                "preset_used": config.get('presets', {}).get('current_preset', 'none'),
                "model_type": config.get('model', {}).get('model_type', 'unknown'),
                "sections": list(config.keys()),
                "total_parameters": sum(len(v) if isinstance(v, dict) else 1 for v in config.values()),
                "checksum": generate_config_checksum(config)
            },
            "save_info": {
                # This could be enhanced to track reasons for saves
                "save_reason": "manual_save",
                "backup_created": False,
                "atomic_write": True
            }
        }
        
        # Create comprehensive configuration structure
        full_config = {
            "metadata": save_metadata,
            "config": config
        }
        
        # Enhanced backup handling with versioning and cleanup
        backup_created = False
        if config_path.exists():
            backup_created = create_config_backup(config_path, full_config['metadata'])
            save_metadata["save_info"]["backup_created"] = backup_created
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write operation with enhanced error handling
        temp_path = config_path.with_suffix(f".tmp_{int(time.time())}")
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=4, ensure_ascii=False, sort_keys=False)
            
            # Verify the written file can be read back
            with open(temp_path, 'r', encoding='utf-8') as f:
                verification_data = json.load(f)
                if not verification_data.get('config'):
                    raise ValueError("Verification failed: saved config is empty or invalid")
            
            # Atomic replacement
            # Windows requires unlinking existing file before replacement
            if os.name == 'nt':
                if config_path.exists():
                    # Remove existing file on Windows
                    config_path.unlink()
            temp_path.replace(config_path)
            
            logger.info(f"Configuration successfully saved to {config_path}")
            if backup_created:
                logger.info("Previous configuration backed up")
                
        except Exception as e:
            # Clean up temp file if write failed
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise RuntimeError(f"Failed to write configuration file: {e}") from e
        
        # Invalidate cache after successful save
        invalidate_config_cache()
        
        # Handle preset-specific save operations
        current_preset = config.get('presets', {}).get('current_preset')
        if current_preset and current_preset not in PRESET_CONFIGS:
            try:
                save_custom_preset(current_preset, config)
                logger.info(f"Custom preset '{current_preset}' saved")
            except Exception as e:
                logger.warning(f"Failed to save custom preset '{current_preset}': {e}")
        
        # Log save statistics
        config_size = config_path.stat().st_size if config_path.exists() else 0
        logger.debug(f"Configuration saved: {config_size} bytes, {len(config)} sections")
        
    except ValueError as e:
        logger.error(f"Configuration validation failed during save: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}", exc_info=True)
        raise RuntimeError(f"Configuration save failed: {str(e)}") from e

def create_config_backup(config_path: Path, save_metadata: Dict) -> bool:
    """Create a backup of the existing configuration with enhanced versioning."""
    try:
        backup_dir = config_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Enhanced backup naming with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = save_metadata.get('version', '2.1')
        preset_used = save_metadata.get('config', {}).get('preset_used', 'unknown')
        
        backup_name = f"{config_path.stem}_v{version}_{preset_used}_{timestamp}{config_path.suffix}"
        backup_path = backup_dir / backup_name
        
        # Copy with metadata preservation
        shutil.copy2(config_path, backup_path)
        
        # Cleanup old backups (keep last 10)
        cleanup_old_backups(backup_dir, config_path.stem, keep_count=10)
        
        logger.info(f"Configuration backup created: {backup_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to create backup: {e}")
        return False

def cleanup_old_backups(backup_dir: Path, config_stem: str, keep_count: int = 10):
    """Clean up old backup files keeping only the most recent ones."""
    try:
        backup_pattern = f"{config_stem}_v*"
        backup_files = list(backup_dir.glob(backup_pattern))
        
        if len(backup_files) > keep_count:
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old backups
            for old_backup in backup_files[keep_count:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
                
    except Exception as e:
        logger.debug(f"Failed to cleanup old backups: {e}")

def generate_config_checksum(config: Dict[str, Any]) -> str:
    """Generate a checksum for configuration integrity verification.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        String checksum (SHA-256 hash of serialized config)
    """
    try:
        import hashlib
        
        # Create a normalized string representation
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        checksum = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
        
        # Return first 16 characters for brevity
        return checksum[:16]
        
    except Exception as e:
        logger.debug(f"Failed to generate config checksum: {e}")
        return "checksum_unavailable"

def load_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load config file with enhanced validation, error recovery, and migration support.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the loaded configuration
        
    Raises:
        ValueError: If configuration format is invalid
    """
    try:
        if not config_path.exists():
            logger.info(f"No configuration file found at {config_path}")
            return {}
        
        # Check file size and basic validity
        file_size = config_path.stat().st_size
        if file_size == 0:
            logger.warning(f"Configuration file {config_path} is empty")
            return {}
        
        # 10MB limit
        if file_size > 10 * 1024 * 1024:
            logger.warning(f"Configuration file {config_path} is unusually large ({file_size} bytes)")
        
        # Load with enhanced error handling
        logger.debug(f"Loading configuration from {config_path} ({file_size} bytes)")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError as e:
                # Try to recover from common JSON errors
                logger.error(f"JSON decode error in {config_path}: {e}")
                
                # Attempt basic recovery
                f.seek(0)
                content = f.read()
                
                # Try to fix common issues
                recovered_config = attempt_json_recovery(content, config_path)
                if recovered_config:
                    config_data = recovered_config
                    logger.warning("Configuration recovered from JSON errors")
                else:
                    raise ValueError(f"Cannot parse configuration file: {e}")
        
        # Validate basic structure
        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a JSON object")
        
        # Handle different configuration formats
        if 'config' in config_data and 'metadata' in config_data:
            # New format with metadata
            loaded_config = config_data['config']
            metadata = config_data['metadata']
            
            logger.debug(f"Loaded configuration with metadata: version={metadata.get('version', 'unknown')}")
            
            # Check version compatibility
            file_version = metadata.get('version', '1.0')
            if file_version != '2.1':
                logger.info(f"Configuration version {file_version} detected, may need migration")
                # The migration will be handled by the caller if needed
            
            # Verify checksum if present
            expected_checksum = metadata.get('config', {}).get('checksum')
            if expected_checksum:
                actual_checksum = generate_config_checksum(loaded_config)
                if actual_checksum != expected_checksum:
                    logger.warning("Configuration checksum mismatch - file may have been modified externally")
            
        else:
            # Legacy format - assume it's the configuration directly
            loaded_config = config_data
            logger.info("Loaded legacy configuration format")
        
        # Validate loaded configuration structure
        if not isinstance(loaded_config, dict):
            raise ValueError("Invalid configuration structure")
        
        # Basic sanity checks
        if not loaded_config:
            logger.warning("Configuration is empty")
            return {}
        
        # Log loading statistics
        section_count = len([k for k, v in loaded_config.items() if isinstance(v, dict)])
        total_params = sum(len(v) if isinstance(v, dict) else 1 for v in loaded_config.values())
        
        logger.info(f"Successfully loaded configuration: {section_count} sections, {total_params} parameters")
        logger.debug(f"Configuration sections: {list(loaded_config.keys())}")
        
        return loaded_config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {str(e)}")
        raise ValueError(f"Configuration file contains invalid JSON: {e}")
    except FileNotFoundError:
        logger.info(f"Configuration file not found: {config_path}")
        return {}
    except PermissionError as e:
        logger.error(f"Permission denied reading config file {config_path}: {e}")
        raise ValueError(f"Cannot read configuration file: permission denied")
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}", exc_info=True)
        
        # Try to load from backup if available
        backup_config = try_load_from_backup(config_path)
        if backup_config:
            logger.warning("Loaded configuration from backup due to primary file error")
            return backup_config
        
        raise ValueError(f"Failed to load configuration: {str(e)}")

def attempt_json_recovery(content: str, config_path: Path) -> Optional[Dict]:
    """Attempt to recover from common JSON formatting errors."""
    try:
        # Try to fix common trailing comma issues
        import re
        
        # Remove trailing commas before closing brackets/braces
        fixed_content = re.sub(r',\s*([}\]])', r'\1', content)
        
        # Try parsing the fixed content
        recovered_data = json.loads(fixed_content)
        logger.info(f"Successfully recovered JSON from {config_path}")
        return recovered_data
        
    except Exception:
        logger.debug("JSON recovery attempt failed")
        return None

def try_load_from_backup(config_path: Path) -> Optional[Dict]:
    """Try to load configuration from the most recent backup."""
    try:
        backup_dir = config_path.parent / "backups"
        if not backup_dir.exists():
            return None
        
        # Find the most recent backup
        backup_pattern = f"{config_path.stem}_v*{config_path.suffix}"
        backup_files = list(backup_dir.glob(backup_pattern))
        
        if not backup_files:
            return None
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        most_recent_backup = backup_files[0]
        
        logger.info(f"Attempting to load from backup: {most_recent_backup}")
        return load_config(most_recent_backup)
        
    except Exception as e:
        logger.debug(f"Failed to load from backup: {e}")
        return None

def initialize_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Initialize or load configuration with enhanced preset awareness and validation.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the initialized configuration
        
    Raises:
        ValueError: If configuration cannot be initialized
        RuntimeError: If configuration initialization fails
    """
    try:
        logger.info(f"Initializing configuration from {config_path}")
        
        # Load existing configuration if available
        loaded_config = load_config(config_path)
        
        # Get the current default configuration template
        try:
            default_config = get_current_config()
        except Exception as e:
            logger.warning(f"Failed to get current config template: {e}")
            # Fallback to DEFAULT_PRESET if available
            if 'DEFAULT_PRESET' in globals():
                default_config = deepcopy(DEFAULT_PRESET)
                logger.info("Using DEFAULT_PRESET as fallback template")
            else:
                raise RuntimeError("No configuration template available") from e
        
        if loaded_config:
            logger.info("Existing configuration found, processing merge and migration")
            
            # Check version compatibility and migration needs
            loaded_version = loaded_config.get('metadata', {}).get('version', '1.0')
            default_version = default_config.get('metadata', {}).get('version', '2.1')
            
            if loaded_version != default_version:
                logger.info(f"Configuration migration needed: {loaded_version} -> {default_version}")
                try:
                    # Use the enhanced migration function
                    migrated_config = migrate_config(loaded_config, default_config)
                    logger.info("Configuration migration completed successfully")
                    loaded_config = migrated_config
                except Exception as e:
                    logger.error(f"Configuration migration failed: {e}")
                    if prompt_user_for_migration_fallback():
                        loaded_config = default_config
                        logger.warning("Using default configuration due to migration failure")
                    else:
                        raise RuntimeError(f"Configuration migration failed: {e}") from e
            
            # Merge configurations with preference for loaded config
            try:
                merged_config = deep_update(deepcopy(default_config), loaded_config)
                logger.info("Configuration merge completed")
            except Exception as e:
                logger.error(f"Configuration merge failed: {e}")
                merged_config = loaded_config
                logger.warning("Using loaded configuration without merge")
            
            # Validate the merged configuration
            try:
                validate_config(merged_config)
                logger.info("Merged configuration passed validation")
            except ValueError as e:
                logger.error(f"Merged configuration validation failed: {str(e)}")
                
                # Offer recovery options
                if handle_validation_failure(e, merged_config, default_config):
                    merged_config = default_config
                    logger.warning("Using default configuration due to validation failure")
                else:
                    raise ValueError(f"Configuration validation failed: {e}") from e
            
        else:
            logger.info("No existing configuration found, using defaults")
            merged_config = default_config
        
        # Ensure preset consistency
        try:
            merged_config = ensure_preset_consistency(merged_config)
        except Exception as e:
            logger.warning(f"Preset consistency check failed: {e}")
        
        # Save the finalized configuration
        try:
            save_config(merged_config, config_path)
            logger.info("Initialized configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save initialized configuration: {e}")
            # Continue with in-memory config even if save fails
        
        # Log initialization summary
        preset_used = merged_config.get('presets', {}).get('current_preset', 'none')
        model_type = merged_config.get('model', {}).get('model_type', 'unknown')
        logger.info(f"Configuration initialized: preset={preset_used}, model={model_type}")
        
        return merged_config
        
    except Exception as e:
        logger.error(f"Configuration initialization failed: {str(e)}", exc_info=True)
        
        # Final fallback - try to return a basic working configuration
        try:
            fallback_config = create_fallback_config()
            logger.warning("Using minimal fallback configuration")
            return fallback_config
        except Exception as fallback_error:
            logger.critical(f"Even fallback configuration failed: {fallback_error}")
            raise RuntimeError(f"Complete configuration initialization failure: {str(e)}") from e

def prompt_user_for_migration_fallback() -> bool:
    """Prompt user for migration failure handling."""
    if sys.stdin.isatty():
        try:
            response = input("Configuration migration failed. Use default configuration? [Y/n]: ").strip().lower()
            return response in ['', 'y', 'yes']
        except:
            # Default to yes if input fails
            return True
    # Non-interactive mode defaults to yes
    return True

def handle_validation_failure(error: ValueError, failed_config: Dict, default_config: Dict) -> bool:
    """Handle configuration validation failure with user interaction."""
    logger.error(f"Configuration validation error: {error}")
    
    if sys.stdin.isatty():
        try:
            print(f"\nConfiguration validation failed: {error}")
            print("Options:")
            print("1. Use default configuration (recommended)")
            print("2. Attempt to fix and retry")
            print("3. Exit with error")
            
            choice = input("Choose option [1-3]: ").strip()
            
            if choice == '1':
                return True
            elif choice == '2':
                # Could implement basic fix attempts here
                logger.info("Automatic fixes not yet implemented")
                return True
            else:
                return False
                
        except:
            # Default to using default config if input fails
            return True
    
    # Non-interactive mode uses default config
    return True

def ensure_preset_consistency(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure preset configuration is consistent and up-to-date."""
    try:
        if 'presets' not in config:
            config['presets'] = {}
        
        # Update dynamic preset information
        config['presets']['available_presets'] = get_available_presets()
        config['presets']['preset_configs'] = get_preset_descriptions()
        config['presets']['custom_presets_available'] = get_safe_custom_presets()
        
        # Validate current preset
        current_preset = config['presets'].get('current_preset')
        if current_preset and current_preset not in get_available_presets():
            logger.warning(f"Current preset '{current_preset}' is not available, clearing")
            config['presets']['current_preset'] = None
        
        return config
        
    except Exception as e:
        logger.warning(f"Preset consistency check failed: {e}")
        return config

def create_fallback_config() -> Dict[str, Any]:
    """Create a minimal fallback configuration that should always work with enhanced robustness.
    
    This function creates a safe, minimal configuration that can be used when all other
    configuration loading methods fail. It ensures system compatibility and basic functionality.
    
    Returns:
        Dictionary containing a minimal but complete configuration
    """
    try:
        # Determine safe system defaults
        # Very conservative for memory safety
        safe_batch_size = 16
        # Quick training for testing
        safe_epochs = 5
        # Conservative worker count
        safe_workers = min(2, os.cpu_count() or 1)
        
        # Create timestamp for tracking
        current_time = datetime.now().isoformat()
        
        fallback_config = {
            'metadata': {
                'version': '2.1',
                'config_type': 'autoencoder',
                'created': current_time,
                'modified': current_time,
                'description': 'Minimal fallback configuration - auto-generated for system recovery',
                'preset_used': 'fallback',
                'fallback_reason': 'Configuration system failure recovery',
                'system': {
                    'python_version': platform.python_version(),
                    'pytorch_version': getattr(torch, '__version__', 'unknown') if 'torch' in globals() else 'unknown',
                    'cuda_available': torch.cuda.is_available() if 'torch' in globals() and hasattr(torch, 'cuda') else False,
                    'hostname': platform.node(),
                    'os': platform.system(),
                    'cpu_count': os.cpu_count() or 1,
                    # Could be enhanced with psutil if available
                    'memory_available': 'unknown'
                },
                # Most basic model only
                'compatibility': ['SimpleAutoencoder'],
                'warnings': [
                    'This is a fallback configuration',
                    'Limited functionality available',
                    'Recommend fixing configuration system'
                ]
            },
            'training': {
                'batch_size': safe_batch_size,
                'epochs': safe_epochs,
                # Safe default learning rate
                'learning_rate': 0.001,
                # Quick early stopping
                'patience': 3,
                # No regularization to reduce complexity
                'weight_decay': 0.0,
                # Basic gradient clipping
                'gradient_clip': 1.0,
                # No accumulation
                'gradient_accumulation_steps': 1,
                # Disabled for compatibility
                'mixed_precision': False,
                'num_workers': safe_workers,
                # Most compatible optimizer
                'optimizer': 'Adam',
                # No scheduler for simplicity
                'scheduler': None,
                # Larger validation for stability
                'validation_split': 0.3,
                'early_stopping': True
            },
            'model': {
                # Most basic model
                'model_type': 'SimpleAutoencoder',
                # Very small encoding
                'encoding_dim': 4,
                # Single small hidden layer
                'hidden_dims': [32],
                # Minimal dropout
                'dropout_rates': [0.1],
                # Most stable activation
                'activation': 'relu',
                # No parameters
                'activation_param': 0.0,
                # No normalization for simplicity
                'normalization': None,
                # Disabled
                'use_batch_norm': False,
                # Disabled
                'use_layer_norm': False,
                # No diversity
                'diversity_factor': 0.0,
                # Absolute minimum
                'min_features': 2,
                # Single model only
                'num_models': 1,
                # Disabled
                'skip_connection': False,
                # Disabled
                'residual_blocks': False,
                # Include available options for reference
                'model_types': ['SimpleAutoencoder'] if 'MODEL_VARIANTS' not in globals() or not MODEL_VARIANTS else list(MODEL_VARIANTS.keys()),
                'available_activations': ['relu', 'leaky_relu', 'gelu'],
                'available_normalizations': ['batch', 'layer', None]
            },
            'security': {
                # Slightly relaxed
                'percentile': 90,
                # More permissive threshold
                'attack_threshold': 0.4,
                # Balanced cost
                'false_negative_cost': 1.0,
                # Disabled for simplicity
                'enable_security_metrics': False,
                # Simple strategy
                'anomaly_threshold_strategy': 'fixed_percentile',
                'early_warning_threshold': 0.35
            },
            'data': {
                # Minimal dataset
                'normal_samples': 200,
                # Very small attack set
                'attack_samples': 50,
                # Minimal features
                'features': 8,
                # Simple normalization
                'normalization': 'minmax',
                # Clear separation
                'anomaly_factor': 2.0,
                # Reproducible
                'random_state': 42,
                # Large validation set
                'validation_split': 0.3,
                # Large test set
                'test_split': 0.3,
                'synthetic_generation': {
                    # Tight clusters
                    'cluster_variance': 0.05,
                    # Few anomalies
                    'anomaly_sparsity': 0.2
                },
                # Always use synthetic for safety
                'use_real_data': False
            },
            'monitoring': {
                # Monitor every epoch
                'metrics_frequency': 1,
                # Minimal checkpointing
                'checkpoint_frequency': 5,
                # Disabled
                'tensorboard_logging': False,
                # Reduced logging
                'console_logging_level': 'WARNING',
                # Don't save by default
                'save_model': False
            },
            'hardware': {
                # Minimal requirements
                'recommended_gpu_memory': 1,
                'minimum_system_requirements': {
                    'cpu_cores': 1,
                    'ram_gb': 1,
                    'disk_space': 1
                },
                # Force CPU for compatibility
                'device': 'cpu',
                # Use minimal memory
                'memory_limit': 0.5
            },
            'presets': {
                'available_presets': get_available_presets() if callable(get_available_presets) else [],
                'current_preset': 'fallback',
                # Empty for safety
                'preset_configs': {},
                'custom_presets_available': []
            },
            'hyperparameter_optimization': {
                # Completely disabled
                'enabled': False,
                # Simple strategy if enabled
                'strategy': 'random',
                'study_name': 'fallback_hpo',
                'direction': 'minimize',
                # Very few trials
                'n_trials': 5,
                # 5 minutes max
                'timeout': 300,
                'sampler': 'RandomSampler',
                'pruner': 'NopPruner'
            },
            'fallback_info': {
                'is_fallback': True,
                'creation_time': current_time,
                'reason': 'Configuration system failure - using minimal safe defaults',
                'recommendations': [
                    'Check configuration file format',
                    'Verify preset definitions',
                    'Check file permissions',
                    'Review system requirements'
                ],
                'limitations': [
                    'Limited model architectures',
                    'Reduced functionality',
                    'Basic monitoring only',
                    'CPU-only execution'
                ]
            }
        }
        
        logger.warning("Created fallback configuration due to system failure")
        logger.info(f"Fallback config features: {fallback_config['data']['features']} features, "
                   f"{fallback_config['training']['batch_size']} batch size, "
                   f"{fallback_config['model']['model_type']} model")
        
        return fallback_config
        
    except Exception as e:
        # Ultimate fallback if even this fails
        logger.critical(f"Failed to create fallback configuration: {e}")
        return {
            'metadata': {
                'version': '2.1',
                'created': datetime.now().isoformat() if datetime else 'unknown',
                'description': 'Emergency minimal configuration',
                'preset_used': 'emergency'
            },
            'training': {'batch_size': 8, 'epochs': 2, 'learning_rate': 0.01},
            'model': {'model_type': 'SimpleAutoencoder', 'encoding_dim': 2, 'hidden_dims': [16]},
            'data': {'normal_samples': 50, 'attack_samples': 10, 'features': 4},
            'security': {'percentile': 90, 'attack_threshold': 0.5},
            'presets': {'current_preset': 'emergency', 'available_presets': []},
            'fallback_info': {'is_fallback': True, 'level': 'emergency'}
        }

def update_global_config(config: Dict[str, Any]) -> None:
    """Update module-level constants from config with enhanced validation, logging, and preset support.
    
    This function synchronizes global configuration variables with the provided configuration
    dictionary, ensuring type safety, value validation, and comprehensive change tracking.
    
    Args:
        config: Configuration dictionary to update from
        
    Raises:
        ValueError: If any configuration values are invalid
        TypeError: If any configuration values are of incorrect type
        KeyError: If required configuration sections are missing
    """
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
    
    # Validate config structure with better error messages
    required_sections = ['training', 'model', 'security', 'data']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise KeyError(f"Missing required configuration sections: {missing_sections}")
    
    # Initialize comprehensive change tracking
    changes = {
        'metadata': {
            'config_version': '2.1',
            'update_time': datetime.now().isoformat(),
            'source': config.get('metadata', {}).get('preset_used', 'manual'),
            'total_changes': 0
        },
        'training': {},
        'model': {},
        'security': {},
        'data': {},
        'system': {}
    }
    
    try:
        # Training configuration with enhanced validation
        training = config.get("training", {})
        
        # Define training parameter mappings with validators
        training_mappings = [
            ('batch_size', 'DEFAULT_BATCH_SIZE', 
             lambda x: isinstance(x, int) and 1 <= x <= 1024, 
             "integer between 1 and 1024"),
            ('epochs', 'DEFAULT_EPOCHS', 
             lambda x: isinstance(x, int) and 1 <= x <= 10000, 
             "integer between 1 and 10000"),
            ('learning_rate', 'LEARNING_RATE', 
             lambda x: isinstance(x, (int, float)) and 1e-6 <= x <= 1.0, 
             "number between 1e-6 and 1.0"),
            ('patience', 'EARLY_STOPPING_PATIENCE', 
             lambda x: isinstance(x, int) and 0 <= x <= 1000, 
             "integer between 0 and 1000"),
            ('weight_decay', 'WEIGHT_DECAY', 
             lambda x: isinstance(x, (int, float)) and 0 <= x <= 1.0, 
             "number between 0 and 1.0"),
            ('gradient_clip', 'GRADIENT_CLIP', 
             lambda x: isinstance(x, (int, float)) and x >= 0, 
             "non-negative number"),
            ('gradient_accumulation_steps', 'GRADIENT_ACCUMULATION_STEPS', 
             lambda x: isinstance(x, int) and 1 <= x <= 64, 
             "integer between 1 and 64"),
            ('mixed_precision', 'MIXED_PRECISION', 
             lambda x: isinstance(x, bool), 
             "boolean"),
            ('num_workers', 'NUM_WORKERS', 
             lambda x: isinstance(x, int) and 0 <= x <= (os.cpu_count() or 1), 
             f"integer between 0 and {os.cpu_count() or 1}")
        ]
        
        for param, global_var, validator, desc in training_mappings:
            if param in training:
                new_value = training[param]
                if not validator(new_value):
                    raise ValueError(f"training.{param} must be a {desc}, got {new_value}")
                
                current_value = globals().get(global_var)
                if current_value != new_value:
                    changes['training'][param] = {
                        'old': current_value,
                        'new': new_value,
                        'type': type(new_value).__name__,
                        'time': datetime.now().isoformat()
                    }
                    globals()[global_var] = new_value
                    logger.debug(f"Updated {global_var}: {current_value} -> {new_value}")
        
    except Exception as e:
        logger.error("Failed to update training configuration", exc_info=True)
        raise ValueError(f"Training configuration error: {str(e)}") from e
    
    try:
        # Model architecture configuration with cross-validation
        model = config.get("model", {})
        
        # Model parameter mappings with enhanced validation
        model_mappings = [
            ('encoding_dim', 'DEFAULT_ENCODING_DIM', 
             lambda x: isinstance(x, int) and 1 <= x <= 1000, 
             "integer between 1 and 1000"),
            ('hidden_dims', 'HIDDEN_LAYER_SIZES', 
             lambda x: (isinstance(x, list) and len(x) >= 1 and len(x) <= 10 and 
                       all(isinstance(i, int) and 1 <= i <= 2048 for i in x)), 
             "list of 1-10 integers between 1 and 2048"),
            ('dropout_rates', 'DROPOUT_RATES',
             lambda x: (isinstance(x, list) and len(x) >= 1 and len(x) <= 10 and
                       all(isinstance(i, (int, float)) and 0 <= i < 1 for i in x)),
             "list of 1-10 numbers between 0 and 1"),
            ('activation', 'ACTIVATION',
             lambda x: x in ['relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid'],
             "one of: 'relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid'"),
            ('activation_param', 'ACTIVATION_PARAM',
             lambda x: isinstance(x, (int, float)) and -1 <= x <= 1,
             "number between -1 and 1"),
            ('normalization', 'NORMALIZATION',
             lambda x: x in ['batch', 'layer', 'instance', None],
             "one of: 'batch', 'layer', 'instance', None"),
            ('use_batch_norm', 'USE_BATCH_NORM',
             lambda x: isinstance(x, bool),
             "boolean"),
            ('use_layer_norm', 'USE_LAYER_NORM',
             lambda x: isinstance(x, bool),
             "boolean"),
            ('diversity_factor', 'DIVERSITY_FACTOR',
             lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
             "number between 0 and 1"),
            ('min_features', 'MIN_FEATURES',
             lambda x: isinstance(x, int) and 1 <= x <= 100,
             "integer between 1 and 100"),
            ('num_models', 'NUM_MODELS',
             lambda x: isinstance(x, int) and 1 <= x <= 20,
             "integer between 1 and 20")
        ]
        
        for param, global_var, validator, desc in model_mappings:
            if param in model:
                new_value = model[param]
                if not validator(new_value):
                    raise ValueError(f"model.{param} must be a {desc}, got {new_value}")
                
                current_value = globals().get(global_var)
                if current_value != new_value:
                    changes['model'][param] = {
                        'old': current_value,
                        'new': new_value,
                        'type': type(new_value).__name__,
                        'time': datetime.now().isoformat()
                    }
                    globals()[global_var] = new_value
                    logger.debug(f"Updated {global_var}: {current_value} -> {new_value}")
        
        # Cross-validation: ensure hidden_dims and dropout_rates have matching lengths
        if 'hidden_dims' in changes['model'] or 'dropout_rates' in changes['model']:
            current_hidden = globals().get('HIDDEN_LAYER_SIZES', [])
            current_dropout = globals().get('DROPOUT_RATES', [])
            
            if len(current_hidden) != len(current_dropout):
                min_length = min(len(current_hidden), len(current_dropout))
                if min_length > 0:
                    globals()['HIDDEN_LAYER_SIZES'] = current_hidden[:min_length]
                    globals()['DROPOUT_RATES'] = current_dropout[:min_length]
                    logger.warning(f"Auto-adjusted layer dimensions to matching length: {min_length}")
                    changes['model']['auto_adjustment'] = {
                        'hidden_dims': current_hidden[:min_length],
                        'dropout_rates': current_dropout[:min_length],
                        'reason': 'length_mismatch',
                        'time': datetime.now().isoformat()
                    }
                else:
                    raise ValueError("Cannot have zero-length hidden layers")
        
        # Validate model type compatibility
        model_type = model.get('model_type')
        if model_type and 'MODEL_VARIANTS' in globals() and MODEL_VARIANTS:
            if model_type not in MODEL_VARIANTS:
                logger.warning(f"Model type '{model_type}' not in MODEL_VARIANTS, may cause issues")
                changes['model']['compatibility_warning'] = {
                    'model_type': model_type,
                    'available_types': list(MODEL_VARIANTS.keys()),
                    'time': datetime.now().isoformat()
                }
    
    except Exception as e:
        logger.error("Failed to update model configuration", exc_info=True)
        raise ValueError(f"Model configuration error: {str(e)}") from e
    
    try:
        # Security configuration
        security = config.get("security", {})
        
        security_mappings = [
            ('percentile', 'DEFAULT_PERCENTILE',
             lambda x: isinstance(x, (int, float)) and 50 <= x <= 100,
             "number between 50 and 100"),
            ('attack_threshold', 'DEFAULT_ATTACK_THRESHOLD',
             lambda x: isinstance(x, (int, float)) and 0 <= x <= 10,
             "number between 0 and 10"),
            ('false_negative_cost', 'FALSE_NEGATIVE_COST',
             lambda x: isinstance(x, (int, float)) and 0 <= x <= 100,
             "number between 0 and 100"),
            ('enable_security_metrics', 'SECURITY_METRICS',
             lambda x: isinstance(x, bool),
             "boolean")
        ]
        
        for param, global_var, validator, desc in security_mappings:
            if param in security:
                new_value = security[param]
                if not validator(new_value):
                    raise ValueError(f"security.{param} must be a {desc}, got {new_value}")
                
                current_value = globals().get(global_var)
                if current_value != new_value:
                    changes['security'][param] = {
                        'old': current_value,
                        'new': new_value,
                        'type': type(new_value).__name__,
                        'time': datetime.now().isoformat()
                    }
                    globals()[global_var] = new_value
                    logger.debug(f"Updated {global_var}: {current_value} -> {new_value}")
    
    except Exception as e:
        logger.error("Failed to update security configuration", exc_info=True)
        raise ValueError(f"Security configuration error: {str(e)}") from e
    
    try:
        # Data configuration with dependency validation
        data = config.get("data", {})
        
        data_mappings = [
            ('normal_samples', 'NORMAL_SAMPLES',
             lambda x: isinstance(x, int) and 10 <= x <= 1000000,
             "integer between 10 and 1000000"),
            ('attack_samples', 'ATTACK_SAMPLES',
             lambda x: isinstance(x, int) and 0 <= x <= 1000000,
             "integer between 0 and 1000000"),
            ('features', 'FEATURES',
             lambda x: isinstance(x, int) and 1 <= x <= 10000,
             "integer between 1 and 10000"),
            ('anomaly_factor', 'ANOMALY_FACTOR',
             lambda x: isinstance(x, (int, float)) and 0.1 <= x <= 10.0,
             "number between 0.1 and 10.0"),
            ('random_state', 'RANDOM_STATE',
             lambda x: isinstance(x, (int, type(None))) and (x is None or 0 <= x <= 2**31-1),
             "integer between 0 and 2^31-1 or None")
        ]
        
        for param, global_var, validator, desc in data_mappings:
            if param in data:
                new_value = data[param]
                if not validator(new_value):
                    raise ValueError(f"data.{param} must be a {desc}, got {new_value}")
                
                current_value = globals().get(global_var)
                if current_value != new_value:
                    changes['data'][param] = {
                        'old': current_value,
                        'new': new_value,
                        'type': type(new_value).__name__,
                        'time': datetime.now().isoformat()
                    }
                    globals()[global_var] = new_value
                    logger.debug(f"Updated {global_var}: {current_value} -> {new_value}")
        
        # Validate data dependencies
        if 'features' in changes['data']:
            min_features = globals().get('MIN_FEATURES', 1)
            if globals().get('FEATURES', 0) < min_features:
                raise ValueError(f"Features count must be >= MIN_FEATURES ({min_features})")
    
    except Exception as e:
        logger.error("Failed to update data configuration", exc_info=True)
        raise ValueError(f"Data configuration error: {str(e)}") from e
    
    # Handle preset application with comprehensive validation
    try:
        presets = config.get("presets", {})
        current_preset = presets.get("current_preset")
        
        if current_preset and current_preset != 'fallback':
            if current_preset in PRESET_CONFIGS:
                logger.info(f"Validating preset configuration: {current_preset}")
                preset_config = PRESET_CONFIGS[current_preset]
                
                # Validate model-preset compatibility
                model_type = config.get('model', {}).get('model_type')
                if model_type:
                    try:
                        if not validate_model_preset_compatibility(model_type, preset_config):
                            logger.warning(f"Model type {model_type} may have limited compatibility with preset {current_preset}")
                            changes['system']['compatibility_warning'] = {
                                'model_type': model_type,
                                'preset': current_preset,
                                'time': datetime.now().isoformat()
                            }
                    except Exception as e:
                        logger.debug(f"Compatibility check failed: {e}")
                
                # Track preset application
                changes['system']['preset_applied'] = {
                    'name': current_preset,
                    'version': preset_config.get('metadata', {}).get('version', 'unknown'),
                    'time': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Unknown preset '{current_preset}' specified")
                changes['system']['preset_warning'] = {
                    'requested': current_preset,
                    'available': list(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else [],
                    'time': datetime.now().isoformat()
                }
    
    except Exception as e:
        logger.warning(f"Failed to process preset configuration: {e}")
    
    # Calculate total changes and log summary
    total_changes = sum(len(section) for section in changes.values() 
                       if isinstance(section, dict) and section != changes['metadata'])
    changes['metadata']['total_changes'] = total_changes
    
    if total_changes > 0:
        logger.info(f"Applied {total_changes} configuration changes from {changes['metadata']['source']}")
        
        # Log detailed changes by section
        for section, section_changes in changes.items():
            if section != 'metadata' and section_changes:
                logger.info(f"  [{section.upper()}] {len(section_changes)} changes:")
                for param, change in section_changes.items():
                    if isinstance(change, dict) and 'old' in change and 'new' in change:
                        logger.info(f"    {param}: {change['old']} -> {change['new']}")
        
        # Save change log for audit trail
        try:
            save_change_log(changes)
            logger.debug("Configuration change log saved")
        except Exception as e:
            logger.warning(f"Failed to save change log: {e}")
        
        # Invalidate configuration cache
        invalidate_config_cache()
        
        # Reinitialize model variants if architecture changed
        if 'model' in changes and changes['model']:
            try:
                if 'initialize_model_variants' in globals():
                    initialize_model_variants(silent=True)
                    logger.info("Model variants reinitialized due to architecture changes")
            except Exception as e:
                logger.warning(f"Failed to reinitialize model variants: {e}")
    else:
        logger.debug("No configuration changes detected")
    
    # Final validation of global state
    try:
        validate_global_config_state()
    except Exception as e:
        logger.error(f"Global configuration state validation failed: {e}")
        raise ValueError(f"Configuration state is invalid after updates: {e}") from e

def get_default_config() -> Dict[str, Any]:
    """Get comprehensive default system configuration with enhanced metadata and validation.
    
    This function returns the complete default configuration structure that serves as the
    foundation for all other configurations. It includes safe defaults, system information,
    and comprehensive metadata.
    
    Returns:
        Dictionary containing the complete default configuration
    """
    try:
        current_time = datetime.now().isoformat()
        
        # Gather system information safely
        system_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'architecture': platform.machine(),
            'processor': platform.processor() or 'unknown',
            'hostname': platform.node(),
            'os': platform.system(),
            'os_release': platform.release(),
            'cpu_count': os.cpu_count() or 1,
            'pytorch_version': getattr(torch, '__version__', 'unknown') if 'torch' in globals() else 'not_available',
            'cuda_available': torch.cuda.is_available() if 'torch' in globals() and hasattr(torch, 'cuda') else False,
            'cuda_version': torch.version.cuda if 'torch' in globals() and hasattr(torch.version, 'cuda') else 'unknown'
        }
        
        # Add CUDA device information if available
        if system_info['cuda_available']:
            try:
                system_info['cuda_devices'] = torch.cuda.device_count()
                system_info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'unknown'
            except Exception:
                system_info['cuda_devices'] = 0
                system_info['cuda_device_name'] = 'unknown'
        
        # Create comprehensive default configuration
        default_config = {
            "metadata": {
                "config_version": "2.1",
                "config_type": "autoencoder",
                "created": current_time,
                "modified": current_time,
                "description": "Default system configuration with optimized parameters",
                "preset_used": "default",
                "compatibility": ["SimpleAutoencoder", "EnhancedAutoencoder", "AutoencoderEnsemble"],
                "system": system_info,
                "validation": {
                    "schema_version": "2.1",
                    "required_sections": ["training", "model", "security", "data"],
                    "optional_sections": ["monitoring", "hardware", "presets", "hyperparameter_optimization"]
                }
            },
            
            "training": {
                "epochs": globals().get('DEFAULT_EPOCHS', 100),
                "batch_size": globals().get('DEFAULT_BATCH_SIZE', 64),
                "learning_rate": globals().get('LEARNING_RATE', 0.001),
                "weight_decay": globals().get('WEIGHT_DECAY', 1e-4),
                "patience": globals().get('EARLY_STOPPING_PATIENCE', 10),
                "gradient_clip": globals().get('GRADIENT_CLIP', 1.0),
                "gradient_accumulation_steps": globals().get('GRADIENT_ACCUMULATION_STEPS', 4),
                "mixed_precision": globals().get('MIXED_PRECISION', True),
                "num_workers": globals().get('NUM_WORKERS', min(4, os.cpu_count() or 1)),
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
                "scheduler_params": {
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 1e-6
                },
                "early_stopping": True,
                "validation_split": 0.2,
                "shuffle": True,
                "pin_memory": system_info['cuda_available'],
                "persistent_workers": False
            },
            
            "model": {
                "model_type": "EnhancedAutoencoder",
                "encoding_dim": globals().get('DEFAULT_ENCODING_DIM', 12),
                "hidden_dims": globals().get('HIDDEN_LAYER_SIZES', [128, 64]).copy(),
                "dropout_rates": globals().get('DROPOUT_RATES', [0.2, 0.15]).copy(),
                "activation": globals().get('ACTIVATION', 'leaky_relu'),
                "activation_param": globals().get('ACTIVATION_PARAM', 0.1),
                "normalization": globals().get('NORMALIZATION', 'batch'),
                "use_batch_norm": globals().get('USE_BATCH_NORM', True),
                "use_layer_norm": globals().get('USE_LAYER_NORM', False),
                "diversity_factor": globals().get('DIVERSITY_FACTOR', 0.1),
                "min_features": globals().get('MIN_FEATURES', 5),
                "num_models": globals().get('NUM_MODELS', 1),
                "skip_connection": True,
                "residual_blocks": False,
                "bias": True,
                "weight_init": "xavier_uniform",
                "model_types": list(MODEL_VARIANTS.keys()) if 'MODEL_VARIANTS' in globals() and MODEL_VARIANTS else ['SimpleAutoencoder', 'EnhancedAutoencoder'],
                "available_activations": ["relu", "leaky_relu", "gelu", "tanh", "sigmoid"],
                "available_normalizations": ["batch", "layer", "instance", None],
                "available_initializers": ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]
            },
            
            "security": {
                "percentile": globals().get('DEFAULT_PERCENTILE', 95),
                "attack_threshold": globals().get('DEFAULT_ATTACK_THRESHOLD', 0.3),
                "false_negative_cost": globals().get('FALSE_NEGATIVE_COST', 2.0),
                "enable_security_metrics": globals().get('SECURITY_METRICS', True),
                "anomaly_threshold_strategy": "percentile",
                "early_warning_threshold": 0.25,
                "adaptive_threshold": True,
                "confidence_interval": 0.95,
                "detection_methods": ["reconstruction_error", "statistical_analysis"],
                "alert_levels": ["low", "medium", "high", "critical"]
            },
            
            "data": {
                "features": globals().get('FEATURES', 20),
                "normal_samples": globals().get('NORMAL_SAMPLES', 8000),
                "attack_samples": globals().get('ATTACK_SAMPLES', 2000),
                "use_real_data": False,
                "normalization": "standard",
                "anomaly_factor": globals().get('ANOMALY_FACTOR', 1.5),
                "random_state": globals().get('RANDOM_STATE', 42),
                "validation_split": 0.2,
                "test_split": 0.2,
                "stratified_split": True,
                "synthetic_generation": {
                    "cluster_variance": 0.1,
                    "anomaly_sparsity": 0.3,
                    "noise_factor": 0.05,
                    "correlation_strength": 0.3
                },
                "preprocessing": {
                    "remove_outliers": True,
                    "outlier_threshold": 3.0,
                    "impute_missing": True,
                    "imputation_strategy": "mean"
                }
            },
            
            "monitoring": {
                "metrics_frequency": 10,
                "checkpoint_frequency": 5,
                "tensorboard_logging": True,
                "console_logging_level": "INFO",
                "save_best_model": True,
                "save_model_history": True,
                "metrics_to_track": [
                    "loss", "reconstruction_error", "validation_loss", 
                    "learning_rate", "epoch_time", "memory_usage"
                ],
                "early_stopping_metric": "validation_loss",
                "checkpoint_format": "pytorch",
                "log_model_summary": True
            },
            
            "hardware": {
                # Will be determined at runtime
                "device": "auto",
                "recommended_gpu_memory": 8,
                "minimum_system_requirements": {
                    "cpu_cores": 2,
                    "ram_gb": 4,
                    "disk_space": 5
                },
                "optimal_system_requirements": {
                    "cpu_cores": 4,
                    "ram_gb": 8,
                    "disk_space": 10,
                    "gpu_memory": 8
                },
                "memory_management": {
                    "max_memory_fraction": 0.8,
                    "allow_memory_growth": True,
                    "memory_limit": None
                },
                "performance_optimization": {
                    "use_cuda": system_info['cuda_available'],
                    "use_amp": system_info['cuda_available'],
                    "benchmark_mode": True,
                    "deterministic": False
                }
            },
            
            "system": {
                "model_dir": str(globals().get('DEFAULT_MODEL_DIR', Path('./models'))),
                "log_dir": str(globals().get('LOG_DIR', Path('./logs'))),
                "config_dir": str(globals().get('CONFIG_DIR', Path('./config'))),
                "data_dir": str(Path('./data')),
                "checkpoint_dir": str(Path('./checkpoints')),
                "debug": False,
                "verbose": True,
                "random_seed": globals().get('RANDOM_STATE', 42),
                "reproducible": True,
                "parallel_processing": True,
                "max_workers": min(4, os.cpu_count() or 1)
            },
            
            "presets": {
                "available_presets": get_available_presets() if callable(get_available_presets) else [
                    'default', 'stability', 'performance', 'baseline', 'debug', 'lightweight', 'advanced'
                ],
                "current_preset": "default",
                "current_override": None,
                "override_rules": {
                    "security": False,
                    "monitoring": True,
                    "hardware": False
                },
                "preset_configs": get_preset_descriptions() if callable(get_preset_descriptions) else {},
                "custom_presets_available": get_safe_custom_presets() if callable(get_safe_custom_presets) else [],
                "auto_apply": False,
                "validate_compatibility": True
            },
            
            "hyperparameter_optimization": {
                "enabled": False,
                "strategy": "optuna",
                "study_name": "autoencoder_hpo",
                "direction": "minimize",
                "n_trials": 100,
                # 1 hour timeout
                "timeout": 3600,
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "objective_metric": "validation_loss",
                "optimization_space": {
                    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
                    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                    "encoding_dim": {"type": "int", "low": 4, "high": 32},
                    "hidden_dims": {"type": "suggest", "options": [[64], [128, 64], [256, 128, 64]]},
                    "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5}
                },
                "early_stopping": {
                    "enabled": True,
                    "patience": 10,
                    "min_improvement": 1e-4
                }
            },
            
            "validation": {
                "cross_validation": {
                    "enabled": False,
                    "folds": 5,
                    "stratified": True,
                    "random_state": 42
                },
                "metrics": [
                    "mse", "mae", "r2_score", "explained_variance",
                    "precision", "recall", "f1_score", "auc_roc"
                ],
                # Validate every epoch
                "validation_frequency": 1,
                "save_validation_results": True,
                "detailed_metrics": False
            },
            
            "experimental": {
                "features": {
                    "advanced_logging": False,
                    "model_interpretability": False,
                    "federated_learning": False,
                    "active_learning": False
                },
                "settings": {
                    "experimental_mode": False,
                    "beta_features": False,
                    "research_mode": False
                }
            }
        }
        
        # Add runtime configuration
        default_config["runtime"] = {
            "config_loaded_at": current_time,
            "config_source": "get_default_config",
            "runtime_id": hashlib.md5(current_time.encode()).hexdigest()[:8] if hashlib else "unknown",
            "process_id": os.getpid(),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable
        }
        
        logger.debug(f"Generated default configuration with {len(default_config)} sections")
        return default_config
        
    except Exception as e:
        logger.error(f"Failed to generate default configuration: {e}", exc_info=True)
        
        # Ultra-minimal fallback
        return {
            "metadata": {
                "config_version": "2.1",
                "created": datetime.now().isoformat() if datetime else "unknown",
                "description": "Emergency default configuration"
            },
            "training": {
                "epochs": 10, "batch_size": 32, "learning_rate": 0.001,
                "weight_decay": 1e-4, "patience": 5
            },
            "model": {
                "model_type": "SimpleAutoencoder", "encoding_dim": 8,
                "hidden_dims": [64], "dropout_rates": [0.2], "activation": "relu"
            },
            "security": {
                "percentile": 95, "attack_threshold": 0.3,
                "enable_security_metrics": True
            },
            "data": {
                "features": 10, "normal_samples": 1000, "attack_samples": 200,
                "use_real_data": False, "validation_split": 0.2
            },
            "system": {
                "model_dir": "./models", "log_dir": "./logs",
                "config_dir": "./config", "debug": False
            },
            "presets": {
                "available_presets": ["default"], "current_preset": "default"
            }
        }

# Helper functions for the updated configuration system
def save_change_log(changes: Dict[str, Any]) -> None:
    """Save configuration change log for audit trail."""
    try:
        log_dir = Path(globals().get('LOG_DIR', './logs'))
        log_dir.mkdir(exist_ok=True)
        change_log_dir = log_dir / "deep_learning_config_changes"
        change_log_dir.mkdir(exist_ok=True)
        
        log_file = change_log_dir / f"change_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Load existing log or create new
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                logger.debug(f"Loaded existing change log from {log_file}")
        else:
            log_data = {'changes': []}
            logger.debug(f"Created new change log at {log_file}")
        
        # Add new changes
        log_data['changes'].append(changes)
        logger.debug(f"Added {len(changes)} changes to the log")
        
        # Keep only last 100 changes
        if len(log_data['changes']) > 100:
            logger.debug(f"Trimming change log to last 100 entries")
            log_data['changes'] = log_data['changes'][-100:]
        
        # Save updated log
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved change log to {log_file}")
            
    except Exception as e:
        logger.debug(f"Failed to save change log: {e}")

def validate_global_config_state() -> None:
    """Validate the current global configuration state for consistency."""
    try:
        # Check critical variables exist and have valid values
        required_globals = {
            'DEFAULT_BATCH_SIZE': (int, lambda x: x > 0),
            'DEFAULT_EPOCHS': (int, lambda x: x > 0),
            'LEARNING_RATE': ((int, float), lambda x: x > 0),
            'FEATURES': (int, lambda x: x > 0),
            'NORMAL_SAMPLES': (int, lambda x: x > 0)
        }
        
        for var_name, (expected_type, validator) in required_globals.items():
            if var_name not in globals():
                raise ValueError(f"Required global variable {var_name} is missing")
            
            value = globals()[var_name]
            if not isinstance(value, expected_type):
                raise TypeError(f"{var_name} must be of type {expected_type}, got {type(value)}")
            
            if not validator(value):
                raise ValueError(f"{var_name} has invalid value: {value}")
        
        # Check list length compatibility
        hidden_dims = globals().get('HIDDEN_LAYER_SIZES', [])
        dropout_rates = globals().get('DROPOUT_RATES', [])
        
        if len(hidden_dims) != len(dropout_rates):
            raise ValueError(f"HIDDEN_LAYER_SIZES and DROPOUT_RATES must have same length: "
                           f"{len(hidden_dims)} != {len(dropout_rates)}")
        
    except Exception as e:
        logger.error(f"Global configuration state validation failed: {e}")
        raise

# Helper functions for preset and configuration management
def get_safe_custom_presets() -> List[str]:
    """Safely get list of custom presets without raising exceptions."""
    try:
        if 'list_custom_presets' in globals() and callable(globals()['list_custom_presets']):
            return globals()['list_custom_presets']()
        return []
    except Exception:
        return []

def validate_model_preset_compatibility(model_type: str, config: Dict[str, Any]) -> bool:
    """Check if a model type is compatible with a preset configuration.
    
    Args:
        model_type: The model type to validate (e.g., 'SimpleAutoencoder')
        config: Configuration dictionary (can be preset or full config)
        
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        # Basic validation - check if model_type is provided
        if not model_type or not isinstance(model_type, str):
            logger.debug("Invalid model_type provided for compatibility check")
            return False
        
        # Check if MODEL_VARIANTS is initialized and contains the model type
        if not MODEL_VARIANTS:
            logger.debug("MODEL_VARIANTS not initialized, attempting initialization")
            try:
                initialize_model_variants(silent=True)
            except Exception as e:
                logger.warning(f"Failed to initialize model variants: {e}")
                # Fallback to basic string validation
                valid_types = ['SimpleAutoencoder', 'EnhancedAutoencoder', 'AutoencoderEnsemble']
                return model_type in valid_types
        
        if model_type not in MODEL_VARIANTS:
            logger.debug(f"Model type '{model_type}' not found in MODEL_VARIANTS")
            return False
        
        # Extract configuration sections safely
        if 'metadata' in config and 'model' in config:
            # This is a full configuration structure
            metadata = config.get('metadata', {})
            model_config = config.get('model', {})
            preset_name = config.get('presets', {}).get('current_preset')
        else:
            # This might be a preset structure or partial config
            metadata = config.get('metadata', {})
            model_config = config.get('model', config)  # Fallback to root level
            preset_name = metadata.get('preset_used') or config.get('preset_used')
        
        # 1. Check explicit compatibility list in metadata
        compatible_models = metadata.get('compatibility', [])
        if compatible_models and isinstance(compatible_models, list):
            if model_type not in compatible_models:
                logger.debug(f"Model type '{model_type}' not in compatibility list: {compatible_models}")
                return False
        
        # 2. Validate model-specific requirements
        if model_type == 'SimpleAutoencoder':
            # Simple autoencoder just needs encoding_dim > 0
            encoding_dim = model_config.get('encoding_dim', 1)
            if not isinstance(encoding_dim, (int, float)) or encoding_dim <= 0:
                logger.debug(f"Invalid encoding_dim for SimpleAutoencoder: {encoding_dim}")
                return False
                
        elif model_type == 'EnhancedAutoencoder':
            # Enhanced autoencoder needs valid hidden_dims and dropout_rates
            encoding_dim = model_config.get('encoding_dim', 1)
            hidden_dims = model_config.get('hidden_dims', [])
            dropout_rates = model_config.get('dropout_rates', [])
            
            if not isinstance(encoding_dim, (int, float)) or encoding_dim <= 0:
                logger.debug(f"Invalid encoding_dim for EnhancedAutoencoder: {encoding_dim}")
                return False
            
            if not isinstance(hidden_dims, list) or not hidden_dims:
                logger.debug(f"Invalid hidden_dims for EnhancedAutoencoder: {hidden_dims}")
                return False
            
            if any(dim <= 0 for dim in hidden_dims if isinstance(dim, (int, float))):
                logger.debug(f"Invalid dimension values in hidden_dims: {hidden_dims}")
                return False
            
            if not isinstance(dropout_rates, list) or not dropout_rates:
                logger.debug(f"Invalid dropout_rates for EnhancedAutoencoder: {dropout_rates}")
                return False
            
            if any(rate < 0 or rate >= 1 for rate in dropout_rates if isinstance(rate, (int, float))):
                logger.debug(f"Invalid dropout rate values: {dropout_rates}")
                return False
            
            # Check if dimensions match (allowing for auto-correction)
            if len(hidden_dims) != len(dropout_rates):
                logger.debug(f"Length mismatch - hidden_dims: {len(hidden_dims)}, dropout_rates: {len(dropout_rates)}")
                # This is correctable, so we don't fail validation
                
        elif model_type == 'AutoencoderEnsemble':
            # Ensemble needs valid num_models and other parameters
            num_models = model_config.get('num_models', 1)
            encoding_dim = model_config.get('encoding_dim', 1)
            diversity_factor = model_config.get('diversity_factor', 0.1)
            
            if not isinstance(num_models, int) or num_models < 1:
                logger.debug(f"Invalid num_models for AutoencoderEnsemble: {num_models}")
                return False
            
            if not isinstance(encoding_dim, (int, float)) or encoding_dim <= 0:
                logger.debug(f"Invalid encoding_dim for AutoencoderEnsemble: {encoding_dim}")
                return False
            
            if not isinstance(diversity_factor, (int, float)) or diversity_factor < 0:
                logger.debug(f"Invalid diversity_factor for AutoencoderEnsemble: {diversity_factor}")
                return False
        
        # 3. Check preset-specific compatibility if preset is known
        if preset_name and preset_name in PRESET_CONFIGS:
            try:
                preset_config = PRESET_CONFIGS[preset_name]
                preset_compatible_models = preset_config.get('metadata', {}).get('compatibility', [])
                
                if preset_compatible_models and model_type not in preset_compatible_models:
                    logger.debug(f"Model type '{model_type}' not compatible with preset '{preset_name}'")
                    return False
                    
                # Check if preset's model configuration is compatible with requested model type
                preset_model_config = preset_config.get('model', {})
                preset_model_type = preset_model_config.get('model_type')
                
                if preset_model_type and preset_model_type != model_type:
                    logger.debug(f"Preset '{preset_name}' configured for '{preset_model_type}', requested '{model_type}'")
                    # This might be intentional override, so we don't fail validation
                    
            except Exception as e:
                logger.debug(f"Error validating preset compatibility: {e}")
        
        # 4. Check hardware requirements if specified
        try:
            hardware_config = config.get('hardware', {})
            if hardware_config:
                min_gpu_memory = hardware_config.get('minimum_system_requirements', {}).get('gpu_memory_gb', 0)
                
                # Ensemble models typically need more memory
                if model_type == 'AutoencoderEnsemble':
                    num_models = model_config.get('num_models', 1)
                    if num_models > 3 and min_gpu_memory < 4:
                        logger.debug(f"Large ensemble ({num_models} models) may need more GPU memory than specified ({min_gpu_memory}GB)")
                        # Warning but don't fail validation
        except Exception as e:
            logger.debug(f"Error checking hardware requirements: {e}")
        
        # 5. Validate activation function compatibility
        try:
            activation = model_config.get('activation', 'relu')
            available_activations = model_config.get('available_activations', ['relu', 'leaky_relu', 'gelu'])
            
            if activation and activation not in available_activations:
                logger.debug(f"Activation '{activation}' not in available list: {available_activations}")
                return False
        except Exception as e:
            logger.debug(f"Error validating activation compatibility: {e}")
        
        # 6. Validate normalization compatibility
        try:
            normalization = model_config.get('normalization')
            available_normalizations = model_config.get('available_normalizations', ['batch', 'layer', None])
            
            if normalization is not None and normalization not in available_normalizations:
                logger.debug(f"Normalization '{normalization}' not in available list: {available_normalizations}")
                return False
        except Exception as e:
            logger.debug(f"Error validating normalization compatibility: {e}")
        
        # 7. Validate training configuration compatibility
        try:
            training_config = config.get('training', {})
            if training_config:
                batch_size = training_config.get('batch_size', 32)
                
                # Very small batch sizes might cause issues with batch normalization
                if model_config.get('use_batch_norm', False) and batch_size < 2:
                    logger.debug(f"Batch size {batch_size} too small for batch normalization")
                    return False
                    
                # Ensemble models need reasonable batch sizes
                if model_type == 'AutoencoderEnsemble':
                    num_models = model_config.get('num_models', 1)
                    if batch_size < num_models:
                        logger.debug(f"Batch size {batch_size} smaller than ensemble size {num_models}")
                        # This might work but is not optimal
        except Exception as e:
            logger.debug(f"Error validating training compatibility: {e}")
        
        # If we've made it this far, the configuration is compatible
        logger.debug(f"Model type '{model_type}' is compatible with the provided configuration")
        return True
        
    except Exception as e:
        logger.warning(f"Error during model-preset compatibility validation: {e}")
        # Default to compatible if validation fails to avoid blocking functionality
        return True

def validate_config(config: Dict[str, Any]) -> None:
    """Validate entire configuration structure with enhanced checks and automatic fixes.
    
    Args:
        config: Full configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid and cannot be auto-fixed
    """
    try:
        # 1. Validate basic structure
        required_sections = ['training', 'model', 'security', 'data']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        # 2. Validate and auto-fix training parameters
        training = config.get('training', {})
        
        # Batch size validation
        batch_size = training.get('batch_size', 32)
        if not isinstance(batch_size, int) or batch_size < 1:
            logger.warning(f"Invalid batch_size {batch_size}, setting to 32")
            training['batch_size'] = 32
        elif batch_size > 1024:
            logger.warning(f"Very large batch_size {batch_size}, consider reducing for memory efficiency")
        
        # Epochs validation
        epochs = training.get('epochs', 100)
        if not isinstance(epochs, int) or epochs < 1:
            logger.warning(f"Invalid epochs {epochs}, setting to 100")
            training['epochs'] = 100
        
        # Learning rate validation
        learning_rate = training.get('learning_rate', 0.001)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            logger.warning(f"Invalid learning_rate {learning_rate}, setting to 0.001")
            training['learning_rate'] = 0.001
        elif learning_rate > 1.0:
            logger.warning(f"Very high learning_rate {learning_rate}, consider reducing")
        
        # Patience validation
        patience = training.get('patience', 10)
        if not isinstance(patience, int) or patience < 0:
            logger.warning(f"Invalid patience {patience}, setting to 10")
            training['patience'] = 10
        
        # Weight decay validation
        weight_decay = training.get('weight_decay', 1e-4)
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            logger.warning(f"Invalid weight_decay {weight_decay}, setting to 1e-4")
            training['weight_decay'] = 1e-4
        
        # Gradient clipping validation
        gradient_clip = training.get('gradient_clip', 1.0)
        if not isinstance(gradient_clip, (int, float)) or gradient_clip < 0:
            logger.warning(f"Invalid gradient_clip {gradient_clip}, setting to 1.0")
            training['gradient_clip'] = 1.0
        
        # Mixed precision validation
        mixed_precision = training.get('mixed_precision', True)
        if not isinstance(mixed_precision, bool):
            logger.warning(f"Invalid mixed_precision {mixed_precision}, setting to True")
            training['mixed_precision'] = True
        
        # Num workers validation
        num_workers = training.get('num_workers', min(4, os.cpu_count() or 1))
        max_workers = os.cpu_count() or 1
        if not isinstance(num_workers, int) or num_workers < 0:
            logger.warning(f"Invalid num_workers {num_workers}, setting to {min(4, max_workers)}")
            training['num_workers'] = min(4, max_workers)
        elif num_workers > max_workers:
            logger.warning(f"num_workers {num_workers} exceeds CPU count {max_workers}, reducing")
            training['num_workers'] = max_workers
        
        # 3. Validate and auto-fix model configuration
        model = config.get('model', {})
        
        # Model type validation
        model_type = model.get('model_type', 'EnhancedAutoencoder')
        if not MODEL_VARIANTS:
            try:
                initialize_model_variants(silent=True)
            except Exception as e:
                logger.warning(f"Could not initialize model variants: {e}")
        
        if MODEL_VARIANTS and model_type not in MODEL_VARIANTS:
            logger.warning(f"Invalid model_type '{model_type}', defaulting to 'SimpleAutoencoder'")
            model['model_type'] = 'SimpleAutoencoder'
            model_type = 'SimpleAutoencoder'
        
        # Encoding dimension validation
        encoding_dim = model.get('encoding_dim', 10)
        if not isinstance(encoding_dim, (int, float)) or encoding_dim < 1:
            logger.warning(f"Invalid encoding_dim {encoding_dim}, setting to 10")
            model['encoding_dim'] = 10
        elif encoding_dim > 100:
            logger.warning(f"Very large encoding_dim {encoding_dim}, may cause memory issues")
        
        # Hidden dimensions validation and auto-fix
        hidden_dims = model.get('hidden_dims', [128, 64])
        if not isinstance(hidden_dims, list):
            if isinstance(hidden_dims, (int, float)) and hidden_dims > 0:
                hidden_dims = [int(hidden_dims)]
                logger.warning(f"Converted hidden_dims to list: {hidden_dims}")
                model['hidden_dims'] = hidden_dims
            else:
                logger.warning("Invalid hidden_dims, setting to [128, 64]")
                model['hidden_dims'] = [128, 64]
                hidden_dims = [128, 64]
        else:
            # Validate all dimensions are positive
            valid_dims = [dim for dim in hidden_dims if isinstance(dim, (int, float)) and dim > 0]
            if len(valid_dims) != len(hidden_dims):
                logger.warning(f"Removed invalid dimensions from hidden_dims: {hidden_dims} -> {valid_dims}")
                model['hidden_dims'] = valid_dims
                hidden_dims = valid_dims
            
            if not hidden_dims:
                logger.warning("Empty hidden_dims, setting to [64]")
                model['hidden_dims'] = [64]
                hidden_dims = [64]
        
        # Dropout rates validation and auto-fix
        dropout_rates = model.get('dropout_rates', [0.2, 0.15])
        if not isinstance(dropout_rates, list):
            if isinstance(dropout_rates, (int, float)) and 0 <= dropout_rates < 1:
                dropout_rates = [float(dropout_rates)]
                logger.warning(f"Converted dropout_rates to list: {dropout_rates}")
                model['dropout_rates'] = dropout_rates
            else:
                logger.warning("Invalid dropout_rates, setting to [0.2]")
                model['dropout_rates'] = [0.2]
                dropout_rates = [0.2]
        else:
            # Validate all rates are between 0 and 1
            valid_rates = [rate for rate in dropout_rates if isinstance(rate, (int, float)) and 0 <= rate < 1]
            if len(valid_rates) != len(dropout_rates):
                logger.warning(f"Fixed invalid dropout rates: {dropout_rates} -> {valid_rates}")
                model['dropout_rates'] = valid_rates
                dropout_rates = valid_rates
            
            if not dropout_rates:
                logger.warning("Empty dropout_rates, setting to [0.2]")
                model['dropout_rates'] = [0.2]
                dropout_rates = [0.2]
        
        # Ensure hidden_dims and dropout_rates have matching lengths
        if len(hidden_dims) != len(dropout_rates):
            min_length = min(len(hidden_dims), len(dropout_rates))
            max_length = max(len(hidden_dims), len(dropout_rates))
            
            if len(hidden_dims) < len(dropout_rates):
                # Extend hidden_dims by repeating the last dimension
                last_dim = hidden_dims[-1] if hidden_dims else 64
                while len(hidden_dims) < len(dropout_rates):
                    hidden_dims.append(max(32, int(last_dim * 0.8)))
                    last_dim = hidden_dims[-1]
                logger.warning(f"Extended hidden_dims to match dropout_rates length: {hidden_dims}")
                model['hidden_dims'] = hidden_dims
            else:
                # Extend dropout_rates by using the last rate
                last_rate = dropout_rates[-1] if dropout_rates else 0.2
                while len(dropout_rates) < len(hidden_dims):
                    dropout_rates.append(max(0.1, last_rate * 0.9))
                    last_rate = dropout_rates[-1]
                logger.warning(f"Extended dropout_rates to match hidden_dims length: {dropout_rates}")
                model['dropout_rates'] = dropout_rates
        
        # Activation function validation
        activation = model.get('activation', 'leaky_relu')
        available_activations = model.get('available_activations', ['relu', 'leaky_relu', 'gelu'])
        if activation not in available_activations:
            logger.warning(f"Invalid activation '{activation}', setting to 'leaky_relu'")
            model['activation'] = 'leaky_relu'
        
        # Activation parameter validation
        activation_param = model.get('activation_param', 0.1)
        if not isinstance(activation_param, (int, float)):
            logger.warning(f"Invalid activation_param {activation_param}, setting to 0.1")
            model['activation_param'] = 0.1
        
        # Normalization validation
        normalization = model.get('normalization', 'batch')
        available_normalizations = model.get('available_normalizations', ['batch', 'layer', None])
        if normalization not in available_normalizations:
            logger.warning(f"Invalid normalization '{normalization}', setting to 'batch'")
            model['normalization'] = 'batch'
        
        # Ensemble-specific validations
        if model_type == 'AutoencoderEnsemble':
            num_models = model.get('num_models', 3)
            if not isinstance(num_models, int) or num_models < 1:
                logger.warning(f"Invalid num_models {num_models} for ensemble, setting to 3")
                model['num_models'] = 3
            elif num_models > 10:
                logger.warning(f"Large ensemble size {num_models} may cause memory issues")
            
            diversity_factor = model.get('diversity_factor', 0.1)
            if not isinstance(diversity_factor, (int, float)) or diversity_factor < 0:
                logger.warning(f"Invalid diversity_factor {diversity_factor}, setting to 0.1")
                model['diversity_factor'] = 0.1
        
        # 4. Validate security configuration
        security = config.get('security', {})
        
        # Percentile validation
        percentile = security.get('percentile', 95)
        if not isinstance(percentile, (int, float)) or not (0 <= percentile <= 100):
            logger.warning(f"Invalid percentile {percentile}, setting to 95")
            security['percentile'] = 95
        
        # Attack threshold validation
        attack_threshold = security.get('attack_threshold', 0.3)
        if not isinstance(attack_threshold, (int, float)) or attack_threshold < 0:
            logger.warning(f"Invalid attack_threshold {attack_threshold}, setting to 0.3")
            security['attack_threshold'] = 0.3
        
        # False negative cost validation
        false_negative_cost = security.get('false_negative_cost', 2.0)
        if not isinstance(false_negative_cost, (int, float)) or false_negative_cost < 0:
            logger.warning(f"Invalid false_negative_cost {false_negative_cost}, setting to 2.0")
            security['false_negative_cost'] = 2.0
        
        # 5. Validate data configuration
        data = config.get('data', {})
        
        # Normal samples validation
        normal_samples = data.get('normal_samples', 8000)
        if not isinstance(normal_samples, int) or normal_samples < 1:
            logger.warning(f"Invalid normal_samples {normal_samples}, setting to 8000")
            data['normal_samples'] = 8000
        
        # Attack samples validation
        attack_samples = data.get('attack_samples', 2000)
        if not isinstance(attack_samples, int) or attack_samples < 1:
            logger.warning(f"Invalid attack_samples {attack_samples}, setting to 2000")
            data['attack_samples'] = 2000
        
        # Features validation
        features = data.get('features', 20)
        min_features = model.get('min_features', 5)
        if not isinstance(features, int) or features < min_features:
            logger.warning(f"Invalid features {features} (min: {min_features}), setting to {max(20, min_features)}")
            data['features'] = max(20, min_features)
        
        # Validation splits
        validation_split = data.get('validation_split', 0.2)
        if not isinstance(validation_split, (int, float)) or not (0 < validation_split < 1):
            logger.warning(f"Invalid validation_split {validation_split}, setting to 0.2")
            data['validation_split'] = 0.2
        
        test_split = data.get('test_split', 0.2)
        if not isinstance(test_split, (int, float)) or not (0 < test_split < 1):
            logger.warning(f"Invalid test_split {test_split}, setting to 0.2")
            data['test_split'] = 0.2
        
        # Ensure splits don't exceed 1.0
        total_split = validation_split + test_split
        if total_split >= 1.0:
            logger.warning(f"Combined splits {total_split} >= 1.0, adjusting proportionally")
            factor = 0.8 / total_split
            data['validation_split'] = validation_split * factor
            data['test_split'] = test_split * factor
            logger.warning(f"Adjusted splits: validation={data['validation_split']:.2f}, test={data['test_split']:.2f}")
        
        # 6. Cross-validation checks
        # Check batch size compatibility with batch normalization
        if model.get('use_batch_norm', False) and training.get('batch_size', 32) < 2:
            logger.warning("Batch normalization requires batch_size >= 2, disabling batch_norm")
            model['use_batch_norm'] = False
        
        # Check model-preset compatibility if preset is specified
        preset_name = config.get('presets', {}).get('current_preset')
        if preset_name and not validate_model_preset_compatibility(model_type, config):
            logger.warning(f"Model type '{model_type}' may not be compatible with preset '{preset_name}'")
        
        # 7. Update preset information to ensure it's current
        try:
            if 'presets' not in config:
                config['presets'] = {}
            config['presets']['available_presets'] = get_available_presets()
            config['presets']['preset_configs'] = get_preset_descriptions()
            config['presets']['custom_presets_available'] = get_safe_custom_presets()
        except Exception as e:
            logger.debug(f"Failed to update preset information: {e}")
        
        logger.debug("Configuration validation completed successfully")
        
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Unexpected error during config validation: {e}")
        raise ValueError(f"Configuration validation failed: {str(e)}")

def migrate_config(legacy_config: Dict, new_template: Dict = None) -> Dict:
    """Migrate an older configuration to the current version using enhanced preset matching.
    
    Args:
        legacy_config: The old configuration dictionary to migrate from
        new_template: Optional template to use as base (defaults to best matching preset)
        
    Returns:
        New configuration dictionary with migrated values
        
    Raises:
        ValueError: If legacy_config is invalid
    """
    if not isinstance(legacy_config, dict):
        raise ValueError("legacy_config must be a dictionary")
    
    logger.info("Migrating configuration to current format")
    
    # Determine the best template to use
    if new_template is None:
        # Try to find the best matching preset
        try:
            if PRESET_CONFIGS:
                # Use convert_legacy_config's scoring system to find best match
                from collections import defaultdict
                
                def score_preset_simple(preset_cfg: Dict[str, Any]) -> float:
                    """Simplified scoring for migration."""
                    score = 0.0
                    total_checks = 0
                    
                    # Check training parameters
                    for param in ['batch_size', 'learning_rate', 'epochs']:
                        if param in legacy_config and param in preset_cfg.get('training', {}):
                            legacy_val = legacy_config[param]
                            preset_val = preset_cfg['training'][param]
                            if isinstance(legacy_val, (int, float)) and isinstance(preset_val, (int, float)):
                                similarity = 1 - min(abs(legacy_val - preset_val) / max(abs(legacy_val), abs(preset_val), 1), 1)
                                score += similarity
                            elif legacy_val == preset_val:
                                score += 1
                            total_checks += 1
                    
                    # Check model parameters
                    for param in ['model_type', 'encoding_dim', 'activation']:
                        if param in legacy_config and param in preset_cfg.get('model', {}):
                            if legacy_config[param] == preset_cfg['model'][param]:
                                score += 1
                            total_checks += 1
                    
                    return score / max(total_checks, 1)
                
                best_preset = None
                best_score = 0
                
                for preset_name, preset_cfg in PRESET_CONFIGS.items():
                    score = score_preset_simple(preset_cfg)
                    if score > best_score:
                        best_score = score
                        best_preset = preset_name
                
                if best_preset and best_score > 0.3:
                    new_template = PRESET_CONFIGS[best_preset]
                    logger.info(f"Using {best_preset} preset as migration template (score: {best_score:.2f})")
                else:
                    new_template = DEFAULT_PRESET
                    logger.info("Using DEFAULT_PRESET as migration template")
            else:
                new_template = DEFAULT_PRESET
                logger.info("PRESET_CONFIGS not available, using DEFAULT_PRESET")
        except Exception as e:
            logger.warning(f"Error selecting migration template: {e}")
            new_template = DEFAULT_PRESET
    
    # Create base configuration from template
    migrated_config = deepcopy(new_template)
    
    # Enhanced key mapping with better fallback handling
    key_mapping = {
        # Direct mappings for training parameters
        'batch_size': ('training', 'batch_size'),
        'epochs': ('training', 'epochs'),
        'learning_rate': ('training', 'learning_rate'),
        'patience': ('training', 'patience'),
        'weight_decay': ('training', 'weight_decay'),
        'gradient_clip': ('training', 'gradient_clip'),
        'gradient_accumulation_steps': ('training', 'gradient_accumulation_steps'),
        'mixed_precision': ('training', 'mixed_precision'),
        'num_workers': ('training', 'num_workers'),
        'optimizer': ('training', 'optimizer'),
        'scheduler': ('training', 'scheduler'),
        
        # Direct mappings for model parameters
        'model_type': ('model', 'model_type'),
        'encoding_dim': ('model', 'encoding_dim'),
        'hidden_dims': ('model', 'hidden_dims'),
        'dropout_rates': ('model', 'dropout_rates'),
        'activation': ('model', 'activation'),
        'activation_param': ('model', 'activation_param'),
        'normalization': ('model', 'normalization'),
        'use_batch_norm': ('model', 'use_batch_norm'),
        'use_layer_norm': ('model', 'use_layer_norm'),
        'diversity_factor': ('model', 'diversity_factor'),
        'min_features': ('model', 'min_features'),
        'skip_connection': ('model', 'skip_connection'),
        'residual_blocks': ('model', 'residual_blocks'),
        'num_models': ('model', 'num_models'),
        
        # Direct mappings for security parameters
        'percentile': ('security', 'percentile'),
        'attack_threshold': ('security', 'attack_threshold'),
        'false_negative_cost': ('security', 'false_negative_cost'),
        'enable_security_metrics': ('security', 'enable_security_metrics'),
        'anomaly_threshold_strategy': ('security', 'anomaly_threshold_strategy'),
        'early_warning_threshold': ('security', 'early_warning_threshold'),
        
        # Direct mappings for data parameters
        'normal_samples': ('data', 'normal_samples'),
        'attack_samples': ('data', 'attack_samples'),
        'features': ('data', 'features'),
        'data_normalization': ('data', 'normalization'),  # Avoid conflict with model normalization
        'anomaly_factor': ('data', 'anomaly_factor'),
        'random_state': ('data', 'random_state'),
        'validation_split': ('data', 'validation_split'),
        'test_split': ('data', 'test_split'),
        
        # Nested parameters
        'synthetic_generation.cluster_variance': ('data', 'synthetic_generation', 'cluster_variance'),
        'synthetic_generation.anomaly_sparsity': ('data', 'synthetic_generation', 'anomaly_sparsity'),
        
        # Direct mappings for monitoring
        'metrics_frequency': ('monitoring', 'metrics_frequency'),
        'checkpoint_frequency': ('monitoring', 'checkpoint_frequency'),
        'tensorboard_logging': ('monitoring', 'tensorboard_logging'),
        'console_logging_level': ('monitoring', 'console_logging_level'),
        
        # Legacy parameter mappings with transformations
        'hidden_layer_sizes': ('model', 'hidden_dims'),  # Common legacy name
        'dropout_rate': ('model', 'dropout_rates'),  # Single rate to list
        'n_epochs': ('training', 'epochs'),  # Alternative name
        'lr': ('training', 'learning_rate'),  # Short form
        'early_stopping': ('training', 'patience'),  # Boolean to patience value
    }
    
    # Track migration statistics
    migration_stats = {
        'mapped_keys': 0,
        'skipped_keys': 0,
        'transformed_keys': 0,
        'invalid_values': 0,
        'auto_fixed': 0
    }
    
    # Apply mapped values with enhanced handling
    def set_nested_value(config_dict: Dict, path: tuple, value: Any) -> bool:
        """Set a nested value in the configuration."""
        try:
            target = config_dict
            for key in path[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            target[path[-1]] = value
            return True
        except Exception as e:
            logger.warning(f"Failed to set nested value at {path}: {e}")
            return False
    
    def get_nested_value(config_dict: Dict, keys: List[str]) -> Any:
        """Get a nested value from legacy config."""
        try:
            value = config_dict
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    # Process all legacy configuration keys
    for legacy_key, new_path in key_mapping.items():
        # Handle dot notation for nested legacy keys
        legacy_keys = legacy_key.split('.')
        legacy_value = get_nested_value(legacy_config, legacy_keys)
        
        if legacy_value is None:
            migration_stats['skipped_keys'] += 1
            continue
        
        try:
            # Handle special transformations
            if legacy_key == 'dropout_rate' and not isinstance(legacy_value, list):
                # Convert single dropout rate to list
                legacy_value = [float(legacy_value)]
                migration_stats['transformed_keys'] += 1
                logger.info(f"Converted single dropout_rate to list: {legacy_value}")
            
            elif legacy_key == 'early_stopping' and isinstance(legacy_value, bool):
                # Convert boolean early stopping to patience value
                legacy_value = 10 if legacy_value else 0
                migration_stats['transformed_keys'] += 1
                logger.info(f"Converted early_stopping boolean to patience: {legacy_value}")
            
            elif legacy_key == 'hidden_layer_sizes' and not isinstance(legacy_value, list):
                # Convert single size to list
                legacy_value = [int(legacy_value)]
                migration_stats['transformed_keys'] += 1
                logger.info(f"Converted single hidden_layer_size to list: {legacy_value}")
            
            # Validate the value before setting
            if isinstance(new_path, tuple):
                # Check if the value is reasonable for the parameter
                param_name = new_path[-1]
                
                if param_name in ['batch_size', 'epochs'] and (not isinstance(legacy_value, int) or legacy_value < 1):
                    logger.warning(f"Invalid {param_name} value {legacy_value}, using template default")
                    migration_stats['invalid_values'] += 1
                    continue
                
                elif param_name == 'learning_rate' and (not isinstance(legacy_value, (int, float)) or legacy_value <= 0):
                    logger.warning(f"Invalid learning_rate value {legacy_value}, using template default")
                    migration_stats['invalid_values'] += 1
                    continue
                
                elif param_name in ['hidden_dims', 'dropout_rates'] and not isinstance(legacy_value, list):
                    logger.warning(f"Invalid {param_name} value {legacy_value}, using template default")
                    migration_stats['invalid_values'] += 1
                    continue
                
                # Set the value
                if set_nested_value(migrated_config, new_path, legacy_value):
                    migration_stats['mapped_keys'] += 1
                else:
                    migration_stats['invalid_values'] += 1
            
        except Exception as e:
            logger.warning(f"Could not migrate {legacy_key}: {str(e)}")
            migration_stats['invalid_values'] += 1
            continue
    
    # Handle any remaining unmapped keys
    unmapped_keys = set(legacy_config.keys()) - set(k.split('.')[0] for k in key_mapping.keys())
    if unmapped_keys:
        logger.info(f"Unmapped legacy keys found: {list(unmapped_keys)}")
        
        # Try to map some common patterns
        for key in unmapped_keys:
            value = legacy_config[key]
            
            # Try common training parameter patterns
            if 'batch' in key.lower():
                if isinstance(value, int) and value > 0:
                    migrated_config.setdefault('training', {})['batch_size'] = value
                    migration_stats['auto_fixed'] += 1
                    logger.info(f"Auto-mapped {key} -> training.batch_size")
            
            elif 'epoch' in key.lower():
                if isinstance(value, int) and value > 0:
                    migrated_config.setdefault('training', {})['epochs'] = value
                    migration_stats['auto_fixed'] += 1
                    logger.info(f"Auto-mapped {key} -> training.epochs")
            
            elif 'learn' in key.lower() or 'lr' in key.lower():
                if isinstance(value, (int, float)) and value > 0:
                    migrated_config.setdefault('training', {})['learning_rate'] = value
                    migration_stats['auto_fixed'] += 1
                    logger.info(f"Auto-mapped {key} -> training.learning_rate")
    
    # Add comprehensive migration metadata
    migrated_config['metadata']['migration'] = {
        'source_version': legacy_config.get('version', '1.x'),
        'target_version': migrated_config.get('metadata', {}).get('version', '2.1'),
        'timestamp': datetime.now().isoformat(),
        'stats': migration_stats,
        'template_used': new_template.get('metadata', {}).get('description', 'Unknown'),
        'compatibility_checked': True,
        'legacy_keys_count': len(legacy_config),
        'success_rate': migration_stats['mapped_keys'] / max(len(legacy_config), 1)
    }
    
    # Validate and auto-fix the migrated configuration
    try:
        validate_config(migrated_config)
        logger.info("Migrated configuration passed validation")
    except ValueError as e:
        logger.warning(f"Migrated config validation issues: {e}")
        # The validation function should have auto-fixed issues
        migrated_config['metadata']['migration']['validation_fixes_applied'] = True
    
    # Log migration summary
    total_keys = len(legacy_config)
    success_rate = (migration_stats['mapped_keys'] + migration_stats['auto_fixed']) / max(total_keys, 1) * 100
    
    logger.info(f"Migration completed:")
    logger.info(f"  - Total legacy keys: {total_keys}")
    logger.info(f"  - Successfully mapped: {migration_stats['mapped_keys']}")
    logger.info(f"  - Auto-fixed: {migration_stats['auto_fixed']}")
    logger.info(f"  - Transformed: {migration_stats['transformed_keys']}")
    logger.info(f"  - Skipped: {migration_stats['skipped_keys']}")
    logger.info(f"  - Invalid: {migration_stats['invalid_values']}")
    logger.info(f"  - Success rate: {success_rate:.1f}%")
    
    return migrated_config

def convert_legacy_config(
    legacy_config: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    preset_similarity_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Convert legacy configuration to current format using intelligent preset matching.
    
    Args:
        legacy_config: The old configuration dictionary to convert
        config: Current configuration dictionary (for migration settings)
        preset_similarity_threshold: Override for similarity threshold (0-1)
        
    Returns:
        New configuration dictionary in current format with metadata
        
    Raises:
        ValueError: If legacy_config is invalid or conversion fails
    """
    # --- Initial Validation ---
    if not isinstance(legacy_config, dict):
        raise ValueError("legacy_config must be a dictionary")
    
    if not legacy_config:
        raise ValueError("legacy_config cannot be empty")
    
    logger.info("Initiating legacy configuration conversion with preset matching")
    
    # --- Step 1: Threshold Determination with Enhanced Logic ---
    def determine_threshold() -> float:
        """Determine the appropriate similarity threshold with enhanced fallbacks."""
        # Argument precedence
        if preset_similarity_threshold is not None:
            if 0 < preset_similarity_threshold <= 1:
                logger.info(f"Using provided threshold: {preset_similarity_threshold:.3f}")
                return preset_similarity_threshold
            logger.warning(f"Invalid threshold {preset_similarity_threshold}, using fallbacks")
        
        # Config file precedence
        if config and config.get("migration", {}).get("preset_similarity_threshold"):
            try:
                threshold = float(config["migration"]["preset_similarity_threshold"])
                if 0 < threshold <= 1:
                    logger.info(f"Using config threshold: {threshold:.3f}")
                    return threshold
                logger.warning(f"Invalid config threshold {threshold}, using fallbacks")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing config threshold: {e}")
        
        # Adaptive threshold based on legacy config complexity
        complexity_score = 0
        complexity_score += len(legacy_config) * 0.1  # Number of keys
        complexity_score += sum(1 for v in legacy_config.values() if isinstance(v, dict)) * 0.2  # Nested dicts
        complexity_score += sum(1 for v in legacy_config.values() if isinstance(v, list)) * 0.1  # Lists
        
        # Adjust threshold based on complexity
        if complexity_score > 10:
            adaptive_threshold = 0.15  # More lenient for complex configs
        elif complexity_score > 5:
            adaptive_threshold = 0.10  # Moderate
        else:
            adaptive_threshold = 0.05  # Strict for simple configs
        
        logger.info(f"Using adaptive threshold based on complexity ({complexity_score:.1f}): {adaptive_threshold:.3f}")
        return adaptive_threshold
    
    similarity_threshold = determine_threshold()
    
    # --- Step 2: Enhanced Preset Scoring System ---
    class AdvancedPresetScorer:
        """Enhanced scoring engine with machine learning-inspired features."""
        
        def __init__(self, legacy_config: Dict[str, Any]):
            self.legacy = legacy_config
            self.weights = {
                'training': 0.35,
                'model': 0.40,
                'security': 0.15,
                'data': 0.10
            }
            
            # Feature importance weights based on common usage patterns
            self.feature_weights = {
                'model_type': 0.30,      # Most important
                'encoding_dim': 0.20,
                'batch_size': 0.15,
                'learning_rate': 0.15,
                'hidden_dims': 0.10,
                'activation': 0.10,
                'percentile': 0.08,
                'features': 0.05,
                'normalization': 0.05,
                'use_batch_norm': 0.05,
                'num_models': 0.05,
                'diversity_factor': 0.03,
                'optimizer': 0.03,
                'scheduler': 0.02
            }
            
            # Value ranges for normalization
            self.value_ranges = {
                'batch_size': (8, 256),
                'learning_rate': (1e-5, 1e-1),
                'encoding_dim': (4, 24),
                'percentile': (85, 99),
                'features': (10, 50),
                'normal_samples': (100, 20000),
                'attack_samples': (50, 5000),
                'epochs': (5, 300),
                'patience': (3, 30),
                'weight_decay': (0, 1e-2),
                'gradient_clip': (0.1, 5.0),
                'diversity_factor': (0, 1),
                'anomaly_factor': (1, 3)
            }
            
            # Pattern matching for string values
            self.string_patterns = {
                'activation': {
                    'relu': ['relu', 'ReLU'],
                    'leaky_relu': ['leaky_relu', 'leakyrelu', 'LeakyReLU'],
                    'gelu': ['gelu', 'GELU']
                },
                'optimizer': {
                    'Adam': ['adam', 'Adam'],
                    'AdamW': ['adamw', 'AdamW'],
                    'SGD': ['sgd', 'SGD']
                },
                'normalization': {
                    'batch': ['batch', 'batch_norm', 'BatchNorm'],
                    'layer': ['layer', 'layer_norm', 'LayerNorm'],
                    None: ['none', 'None', None]
                }
            }
        
        def normalize_value(self, key: str, value: Any) -> float:
            """Enhanced value normalization with outlier handling."""
            if key not in self.value_ranges:
                return 0.5  # Neutral value for unknown parameters
            
            min_val, max_val = self.value_ranges[key]
            
            if isinstance(value, (int, float)):
                # Handle outliers by capping
                capped_value = max(min_val, min(value, max_val))
                normalized = (capped_value - min_val) / (max_val - min_val)
                
                # Apply log scaling for learning rate and weight decay
                if key in ['learning_rate', 'weight_decay'] and value > 0:
                    log_normalized = (np.log10(value) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val))
                    normalized = max(0, min(1, log_normalized))
                
                return normalized
            
            return 0.5
        
        def compare_numeric_enhanced(self, key: str, preset_val: Any) -> float:
            """Enhanced numeric comparison with weighted similarity."""
            if key not in self.legacy:
                return 0.3  # Penalty for missing values
            
            legacy_val = self.legacy[key]
            if not isinstance(legacy_val, (int, float)) or not isinstance(preset_val, (int, float)):
                return 0
            
            # Normalize both values
            norm_legacy = self.normalize_value(key, legacy_val)
            norm_preset = self.normalize_value(key, preset_val)
            
            # Calculate similarity with sigmoid-like function for smoother scoring
            diff = abs(norm_preset - norm_legacy)
            similarity = 1 / (1 + diff * 2)  # Sigmoid-like curve
            
            # Apply importance weighting
            weight = self.feature_weights.get(key, 0.1)
            return similarity * weight
        
        def compare_string_enhanced(self, key: str, preset_val: Any) -> float:
            """Enhanced string comparison with pattern matching."""
            if key not in self.legacy:
                return 0.3
            
            legacy_val = self.legacy[key]
            
            # Direct match
            if legacy_val == preset_val:
                return self.feature_weights.get(key, 0.1)
            
            # Pattern matching
            if key in self.string_patterns:
                for standard_val, patterns in self.string_patterns[key].items():
                    if legacy_val in patterns and preset_val == standard_val:
                        return self.feature_weights.get(key, 0.1) * 0.9  # Slight penalty for pattern match
                    elif preset_val in patterns and legacy_val == standard_val:
                        return self.feature_weights.get(key, 0.1) * 0.9
            
            return 0
        
        def compare_list_enhanced(self, key: str, preset_val: Any) -> float:
            """Enhanced list comparison with element-wise similarity."""
            if key not in self.legacy:
                return 0.3
            
            legacy_list = self.legacy[key] if isinstance(self.legacy[key], list) else []
            preset_list = preset_val if isinstance(preset_val, list) else []
            
            if not legacy_list and not preset_list:
                return self.feature_weights.get(key, 0.1)
            
            if not legacy_list or not preset_list:
                return 0.1  # Some penalty for missing data
            
            # Length similarity
            max_len = max(len(legacy_list), len(preset_list))
            min_len = min(len(legacy_list), len(preset_list))
            length_similarity = min_len / max_len
            
            # Element-wise similarity
            element_scores = []
            for i in range(min_len):
                if isinstance(legacy_list[i], (int, float)) and isinstance(preset_list[i], (int, float)):
                    # Numeric similarity
                    diff = abs(legacy_list[i] - preset_list[i]) / max(abs(legacy_list[i]), abs(preset_list[i]), 1e-6)
                    element_scores.append(max(0, 1 - diff))
                else:
                    # Exact match
                    element_scores.append(1.0 if legacy_list[i] == preset_list[i] else 0.0)
            
            avg_element_similarity = sum(element_scores) / len(element_scores) if element_scores else 0
            
            # Combine similarities
            total_similarity = (length_similarity * 0.3 + avg_element_similarity * 0.7)
            return total_similarity * self.feature_weights.get(key, 0.1)
        
        def score_preset_comprehensive(self, preset_name: str, preset_cfg: Dict[str, Any]) -> Dict[str, Any]:
            """Comprehensive preset scoring with detailed breakdown."""
            section_scores = defaultdict(float)
            parameter_scores = {}
            
            # Training parameters
            training_params = [
                ('batch_size', self.compare_numeric_enhanced),
                ('learning_rate', self.compare_numeric_enhanced),
                ('epochs', self.compare_numeric_enhanced),
                ('patience', self.compare_numeric_enhanced),
                ('weight_decay', self.compare_numeric_enhanced),
                ('gradient_clip', self.compare_numeric_enhanced),
                ('mixed_precision', self.compare_string_enhanced),
                ('optimizer', self.compare_string_enhanced),
                ('scheduler', self.compare_string_enhanced),
                ('num_workers', self.compare_numeric_enhanced)
            ]
            
            for param, compare_fn in training_params:
                if param in preset_cfg.get('training', {}):
                    score = compare_fn(param, preset_cfg['training'][param])
                    section_scores['training'] += score
                    parameter_scores[f'training.{param}'] = score
            
            # Model parameters
            model_params = [
                ('model_type', self.compare_string_enhanced),
                ('encoding_dim', self.compare_numeric_enhanced),
                ('hidden_dims', self.compare_list_enhanced),
                ('dropout_rates', self.compare_list_enhanced),
                ('activation', self.compare_string_enhanced),
                ('activation_param', self.compare_numeric_enhanced),
                ('normalization', self.compare_string_enhanced),
                ('use_batch_norm', self.compare_string_enhanced),
                ('use_layer_norm', self.compare_string_enhanced),
                ('skip_connection', self.compare_string_enhanced),
                ('residual_blocks', self.compare_string_enhanced),
                ('num_models', self.compare_numeric_enhanced),
                ('diversity_factor', self.compare_numeric_enhanced)
            ]
            
            for param, compare_fn in model_params:
                if param in preset_cfg.get('model', {}):
                    score = compare_fn(param, preset_cfg['model'][param])
                    section_scores['model'] += score
                    parameter_scores[f'model.{param}'] = score
            
            # Security parameters
            security_params = [
                ('percentile', self.compare_numeric_enhanced),
                ('attack_threshold', self.compare_numeric_enhanced),
                ('false_negative_cost', self.compare_numeric_enhanced),
                ('enable_security_metrics', self.compare_string_enhanced),
                ('anomaly_threshold_strategy', self.compare_string_enhanced)
            ]
            
            for param, compare_fn in security_params:
                if param in preset_cfg.get('security', {}):
                    score = compare_fn(param, preset_cfg['security'][param])
                    section_scores['security'] += score
                    parameter_scores[f'security.{param}'] = score
            
            # Data parameters
            data_params = [
                ('features', self.compare_numeric_enhanced),
                ('normalization', self.compare_string_enhanced),
                ('anomaly_factor', self.compare_numeric_enhanced),
                ('normal_samples', self.compare_numeric_enhanced),
                ('attack_samples', self.compare_numeric_enhanced),
                ('validation_split', self.compare_numeric_enhanced),
                ('test_split', self.compare_numeric_enhanced)
            ]
            
            for param, compare_fn in data_params:
                if param in preset_cfg.get('data', {}):
                    score = compare_fn(param, preset_cfg['data'][param])
                    section_scores['data'] += score
                    parameter_scores[f'data.{param}'] = score
            
            # Apply section weights
            weighted_sections = {}
            for section in section_scores:
                weighted_sections[section] = section_scores[section] * self.weights.get(section, 0.1)
            
            total_score = sum(weighted_sections.values())
            
            return {
                'name': preset_name,
                'total_score': total_score,
                'section_scores': dict(section_scores),
                'weighted_section_scores': weighted_sections,
                'parameter_scores': parameter_scores,
                'config': preset_cfg,
                'compatibility': preset_cfg.get('metadata', {}).get('compatibility', [])
            }

    # --- Step 3: Score All Available Presets ---
    try:
        if not PRESET_CONFIGS:
            logger.warning("PRESET_CONFIGS not available, using basic migration")
            return migrate_config(legacy_config)
        
        scorer = AdvancedPresetScorer(legacy_config)
        preset_scores = []
        
        for name, cfg in PRESET_CONFIGS.items():
            try:
                score_result = scorer.score_preset_comprehensive(name, cfg)
                preset_scores.append(score_result)
            except Exception as e:
                logger.warning(f"Error scoring preset {name}: {e}")
                continue
        
        if not preset_scores:
            logger.warning("No presets could be scored, using basic migration")
            return migrate_config(legacy_config)
        
        preset_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
    except Exception as e:
        logger.error(f"Error during preset scoring: {e}")
        return migrate_config(legacy_config)
    
    # --- Step 4: Enhanced Results Analysis ---
    def analyze_results_enhanced(scores: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """Enhanced analysis with confidence scoring."""
        if not scores:
            return [], {}, {}
        
        best_score = scores[0]['total_score']
        close_presets = []
        analysis = {
            'best_score': best_score,
            'score_distribution': [s['total_score'] for s in scores[:5]],
            'confidence': 'high' if best_score > 0.7 else 'medium' if best_score > 0.4 else 'low'
        }
        
        # Dynamic threshold adjustment based on score distribution
        effective_threshold = similarity_threshold
        
        if len(scores) > 1:
            second_best = scores[1]['total_score']
            score_gap = best_score - second_best
            
            if score_gap < 0.1 and best_score > 0.3:
                effective_threshold = max(similarity_threshold, 0.15)
                logger.info(f"Close scores detected, adjusting threshold to {effective_threshold:.3f}")
            elif best_score < 0.3:
                effective_threshold = min(similarity_threshold * 2, 0.2)
                logger.info(f"Low best score, relaxing threshold to {effective_threshold:.3f}")
        
        # Find close matches
        for score in scores:
            if (best_score - score['total_score']) <= effective_threshold:
                close_presets.append(score['name'])
            else:
                break
        
        return close_presets, scores[0], analysis
    
    close_presets, best_preset, analysis = analyze_results_enhanced(preset_scores)
    
    # --- Step 5: Enhanced Reporting ---
    def generate_enhanced_report(scores: List[Dict[str, Any]], close_presets: List[str], analysis: Dict[str, Any]) -> None:
        """Generate comprehensive conversion report."""
        logger.info("\n" + "="*80)
        logger.info("LEGACY CONFIGURATION CONVERSION REPORT")
        logger.info("="*80)
        
        logger.info(f"Analysis Confidence: {analysis['confidence'].upper()}")
        logger.info(f"Best Score: {analysis['best_score']:.3f}")
        logger.info(f"Threshold Used: {similarity_threshold:.3f}")
        
        logger.info("\nTop 10 Preset Matches:")
        logger.info(f"{'Rank':<5} {'Preset':<20} {'Total':<8} {'Training':<9} {'Model':<8} {'Security':<9} {'Data':<8}")
        logger.info("-" * 80)
        
        for i, score in enumerate(scores[:10], 1):
            logger.info(
                f"{i:<5} "
                f"{score['name']:<20} "
                f"{score['total_score']:.3f}    "
                f"{score['section_scores'].get('training', 0):.3f}     "
                f"{score['section_scores'].get('model', 0):.3f}    "
                f"{score['section_scores'].get('security', 0):.3f}     "
                f"{score['section_scores'].get('data', 0):.3f}"
            )
        
        if close_presets:
            logger.info(f"\nClose Matches (within threshold {similarity_threshold:.3f}):")
            for i, preset in enumerate(close_presets, 1):
                preset_score = next(s for s in scores if s['name'] == preset)
                logger.info(f"  {i}. {preset:<18} (score={preset_score['total_score']:.3f})")
            
            logger.info(f"\nRecommended: {close_presets[0]}")
        else:
            logger.info("\nNo close matches found - will use best available or default")
        
        # Show parameter-level analysis for best match
        if scores and 'parameter_scores' in scores[0]:
            logger.info(f"\nParameter Analysis for '{scores[0]['name']}':")
            param_scores = scores[0]['parameter_scores']
            sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
            
            for param, score in sorted_params[:15]:  # Top 15 parameters
                if score > 0.1:  # Only show meaningful scores
                    logger.info(f"  {param:<30} {score:.3f}")

    generate_enhanced_report(preset_scores, close_presets, analysis)
    
    # --- Step 6: Smart Selection Logic ---
    def smart_select_preset(close_presets: List[str], scores: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Smart preset selection with fallback logic."""
        if not close_presets:
            # No close matches - use best available if reasonable, otherwise default
            if scores and scores[0]['total_score'] > 0.2:
                logger.info(f"Using best available preset: {scores[0]['name']} (score: {scores[0]['total_score']:.3f})")
                return scores[0]['name']
            else:
                logger.info("No reasonable matches found, using default preset")
                return "default"
        
        if len(close_presets) == 1:
            return close_presets[0]
        
        # Multiple close matches - use additional criteria
        best_candidate = close_presets[0]
        
        # Prefer presets with model type compatibility
        legacy_model_type = legacy_config.get('model_type')
        if legacy_model_type:
            for preset_name in close_presets:
                preset_score = next(s for s in scores if s['name'] == preset_name)
                compatibility = preset_score.get('compatibility', [])
                if legacy_model_type in compatibility:
                    logger.info(f"Selected {preset_name} for model type compatibility with {legacy_model_type}")
                    return preset_name
        
        # Interactive selection for terminal environments
        if sys.stdin.isatty():
            return interactive_select_enhanced(close_presets, scores)
        
        return best_candidate
    
    def interactive_select_enhanced(close_presets: List[str], scores: List[Dict[str, Any]]) -> str:
        """Enhanced interactive selection with detailed comparisons."""
        print(f"\nFound {len(close_presets)} similar presets:")
        
        for i, name in enumerate(close_presets, 1):
            score = next(s for s in scores if s['name'] == name)
            print(f"  {i}. {name:<18} (score={score['total_score']:.3f})")
        
        print("  a. Show detailed analysis")
        print("  c. Compare presets side-by-side")
        print("  d. Use default preset")
        
        while True:
            try:
                choice = input(f"\nSelect [1-{len(close_presets)}], or (a/c/d): ").strip().lower()
                
                if choice == 'a':
                    # Show detailed analysis
                    preset_name = input("Enter preset name for analysis: ").strip()
                    preset_data = next((s for s in scores if s['name'] == preset_name), None)
                    if preset_data:
                        print(f"\nDetailed Analysis for '{preset_name}':")
                        print(f"Total Score: {preset_data['total_score']:.3f}")
                        print(f"Section Scores:")
                        for section, score in preset_data['section_scores'].items():
                            print(f"  {section}: {score:.3f}")
                        
                        if 'parameter_scores' in preset_data:
                            print(f"\nTop Parameter Matches:")
                            sorted_params = sorted(preset_data['parameter_scores'].items(), 
                                                 key=lambda x: x[1], reverse=True)
                            for param, score in sorted_params[:10]:
                                if score > 0.1:
                                    print(f"  {param}: {score:.3f}")
                    else:
                        print("Preset not found")
                    continue
                
                elif choice == 'c':
                    # Compare presets
                    if len(close_presets) >= 2:
                        print(f"\nComparison of top {min(3, len(close_presets))} presets:")
                        print(f"{'Parameter':<25}", end="")
                        for name in close_presets[:3]:
                            print(f"{name[:15]:<16}", end="")
                        print()
                        print("-" * (25 + 16 * min(3, len(close_presets))))
                        
                        # Compare key parameters
                        key_params = ['model.model_type', 'model.encoding_dim', 'training.batch_size', 
                                    'training.learning_rate', 'security.percentile']
                        
                        for param in key_params:
                            print(f"{param:<25}", end="")
                            for name in close_presets[:3]:
                                preset_data = next(s for s in scores if s['name'] == name)
                                section, key = param.split('.')
                                value = preset_data['config'].get(section, {}).get(key, 'N/A')
                                print(f"{str(value)[:15]:<16}", end="")
                            print()
                    continue
                
                elif choice == 'd':
                    return "default"
                
                # Numeric selection
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(close_presets):
                    return close_presets[choice_idx]
                print(f"Please enter 1-{len(close_presets)}, or a/c/d")
                
            except ValueError:
                print(f"Please enter 1-{len(close_presets)}, or a/c/d")
    
    selected_preset = smart_select_preset(close_presets, preset_scores, analysis)
    
    # --- Step 7: Create Final Configuration ---
    try:
        if selected_preset not in PRESET_CONFIGS:
            logger.warning(f"Selected preset '{selected_preset}' not found, using default")
            selected_preset = "default"
        
        new_config = migrate_config(legacy_config, PRESET_CONFIGS[selected_preset])
        
        # Add comprehensive conversion metadata
        new_config['metadata']['conversion'] = {
            'method': 'advanced_preset_matching',
            'selected_preset': selected_preset,
            'similar_presets': [p for p in close_presets if p != selected_preset],
            'similarity_threshold': similarity_threshold,
            'best_score': best_preset['total_score'],
            'confidence': analysis['confidence'],
            'timestamp': datetime.now().isoformat(),
            'legacy_keys_analyzed': len(legacy_config),
            'presets_evaluated': len(preset_scores),
            'selection_method': 'interactive' if sys.stdin.isatty() and len(close_presets) > 1 else 'automatic'
        }
        
        # Validate the final configuration
        try:
            validate_config(new_config)
            logger.info("Converted configuration passed validation")
        except ValueError as e:
            logger.warning(f"Validation issues with converted config: {e}")
            new_config['metadata']['conversion']['validation_warnings'] = str(e)
        
        logger.info(f"Legacy configuration successfully converted using preset: {selected_preset}")
        logger.info(f"Conversion confidence: {analysis['confidence']}")
        
        return new_config
        
    except Exception as e:
        logger.error(f"Error creating final configuration: {e}")
        # Fallback to basic migration
        logger.info("Falling back to basic migration")
        return migrate_config(legacy_config)

def save_custom_preset(name: str, config: Dict) -> Path:
    """Save a custom preset configuration with enhanced validation and metadata.
    
    Args:
        name: Name for the custom preset
        config: Configuration dictionary to save as preset
        
    Returns:
        Path: Path to the saved preset file
        
    Raises:
        ValueError: If name or config is invalid
        RuntimeError: If save operation fails
    """
    try:
        # Input validation
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Preset name must be a non-empty string")
        
        if not isinstance(config, dict) or not config:
            raise ValueError("Configuration must be a non-empty dictionary")
        
        # Sanitize the name
        safe_name = "".join(c for c in name.strip() if c.isalnum() or c in (' ', '_', '-')).strip()
        safe_name = safe_name.replace(' ', '_').lower()
        
        if not safe_name:
            raise ValueError(f"Invalid preset name '{name}' - must contain alphanumeric characters")
        
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
            logger.warning(f"Preset name truncated to: {safe_name}")
        
        # Setup custom presets directory
        custom_dir = CONFIG_DIR / "deep_learning_custom_presets"
        custom_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"preset_{safe_name}.json"
        filepath = custom_dir / filename
        
        # Check for name conflicts
        if filepath.exists():
            # Create backup of existing preset
            backup_path = filepath.with_suffix(f".backup_{int(time.time())}.json")
            shutil.copy2(filepath, backup_path)
            logger.info(f"Existing preset backed up to: {backup_path}")
        
        # Validate configuration structure
        try:
            validate_config(config)
            logger.info("Custom preset configuration passed validation")
        except ValueError as e:
            logger.warning(f"Custom preset validation issues (will save anyway): {e}")
        
        # Extract metadata from config
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        security_config = config.get('security', {})
        data_config = config.get('data', {})
        
        # Determine model compatibility
        model_type = model_config.get('model_type', 'unknown')
        compatibility = []
        
        if MODEL_VARIANTS:
            if model_type in MODEL_VARIANTS:
                compatibility.append(model_type)
            # Add other compatible types based on configuration
            if model_type == 'SimpleAutoencoder':
                compatibility.extend(['EnhancedAutoencoder', 'AutoencoderEnsemble'])
            elif model_type == 'EnhancedAutoencoder':
                compatibility.extend(['SimpleAutoencoder', 'AutoencoderEnsemble'])
            elif model_type == 'AutoencoderEnsemble':
                compatibility.append('EnhancedAutoencoder')
        else:
            compatibility = ['SimpleAutoencoder', 'EnhancedAutoencoder', 'AutoencoderEnsemble']
        
        # Generate comprehensive preset metadata
        preset_metadata = {
            "name": name,
            "safe_name": safe_name,
            "description": f"Custom preset '{name}' - created from current configuration",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "version": "2.1",
            "preset_type": "custom",
            "model_type": model_type,
            "compatibility": list(set(compatibility)),
            "system": {
                "python_version": platform.python_version(),
                "pytorch_version": getattr(torch, '__version__', 'unknown') if 'torch' in globals() else 'unknown',
                "cuda_available": torch.cuda.is_available() if 'torch' in globals() and hasattr(torch, 'cuda') else False,
                "hostname": platform.node(),
                "os": platform.system(),
                "created_by": "save_custom_preset"
            },
            "configuration_summary": {
                "batch_size": training_config.get('batch_size'),
                "learning_rate": training_config.get('learning_rate'),
                "encoding_dim": model_config.get('encoding_dim'),
                "hidden_layers": len(model_config.get('hidden_dims', [])),
                "features": data_config.get('features'),
                "security_percentile": security_config.get('percentile'),
                "total_sections": len(config),
                "estimated_complexity": estimate_config_complexity(config)
            },
            "usage_guidelines": {
                "recommended_for": determine_preset_recommendations(config),
                "memory_requirements": estimate_memory_requirements(config),
                "training_time_estimate": estimate_training_time(config),
                "resource_level": determine_resource_level(config)
            },
            "validation": {
                "config_validated": True,
                "validation_timestamp": datetime.now().isoformat(),
                "warnings": [],
                "auto_fixes_applied": []
            },
            "checksum": generate_config_checksum(config)
        }
        
        # Create complete preset structure
        preset_data = {
            "metadata": preset_metadata,
            "config": deepcopy(config)
        }
        
        # Add preset-specific enhancements
        preset_data["config"]["metadata"] = preset_data["config"].get("metadata", {})
        preset_data["config"]["metadata"]["preset_used"] = safe_name
        preset_data["config"]["metadata"]["is_custom_preset"] = True
        
        # Atomic write operation
        temp_path = filepath.with_suffix(f".tmp_{int(time.time())}")
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=4, ensure_ascii=False, sort_keys=False)
            
            # Verify the written file
            with open(temp_path, 'r', encoding='utf-8') as f:
                verification_data = json.load(f)
                if not verification_data.get('config') or not verification_data.get('metadata'):
                    raise ValueError("Verification failed: preset data is incomplete")
            
            # Atomic replacement
            if os.name == 'nt' and filepath.exists():
                filepath.unlink()
            temp_path.replace(filepath)
            
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise RuntimeError(f"Failed to write preset file: {e}") from e
        
        # Update global preset registry if available
        try:
            if 'PRESET_CONFIGS' in globals() and PRESET_CONFIGS is not None:
                PRESET_CONFIGS[safe_name] = preset_data["config"]
                logger.info(f"Added custom preset '{safe_name}' to global registry")
            
            # Refresh available presets
            invalidate_config_cache()
            
        except Exception as e:
            logger.warning(f"Could not update global preset registry: {e}")
        
        # Log success with statistics
        file_size = filepath.stat().st_size
        logger.info(f"Custom preset '{name}' saved successfully:")
        logger.info(f"  - File: {filepath}")
        logger.info(f"  - Size: {file_size} bytes")
        logger.info(f"  - Model type: {model_type}")
        logger.info(f"  - Compatible with: {', '.join(compatibility)}")
        logger.info(f"  - Resource level: {preset_metadata['usage_guidelines']['resource_level']}")
        
        return filepath
        
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        logger.error(f"Failed to save custom preset '{name}': {str(e)}", exc_info=True)
        raise RuntimeError(f"Custom preset save failed: {str(e)}") from e

# Helper functions for the enhanced implementations
def estimate_config_complexity(config: Dict[str, Any]) -> str:
    """Estimate configuration complexity level with comprehensive analysis.
    
    Args:
        config: Configuration dictionary to analyze
        
    Returns:
        String indicating complexity level: 'low', 'medium', 'high', or 'very_high'
    """
    try:
        complexity_score = 0.0
        analysis_factors = []
        
        # Model architecture complexity
        model_config = config.get('model', {})
        model_type = model_config.get('model_type', 'SimpleAutoencoder')
        
        # Base complexity by model type
        if model_type == 'SimpleAutoencoder':
            complexity_score += 1.0
            analysis_factors.append('simple_model')
        elif model_type == 'EnhancedAutoencoder':
            complexity_score += 3.0
            analysis_factors.append('enhanced_model')
        elif model_type == 'AutoencoderEnsemble':
            complexity_score += 5.0
            analysis_factors.append('ensemble_model')
            
            # Ensemble-specific complexity
            num_models = model_config.get('num_models', 3)
            if num_models > 5:
                complexity_score += 2.0
                analysis_factors.append('large_ensemble')
            elif num_models > 3:
                complexity_score += 1.0
                analysis_factors.append('medium_ensemble')
        
        # Hidden layer complexity
        hidden_dims = model_config.get('hidden_dims', [])
        if isinstance(hidden_dims, list):
            # Number of layers
            layer_count = len(hidden_dims)
            complexity_score += layer_count * 0.5
            if layer_count > 4:
                analysis_factors.append('deep_architecture')
            
            # Layer sizes
            large_layers = sum(1 for dim in hidden_dims if isinstance(dim, (int, float)) and dim > 256)
            medium_layers = sum(1 for dim in hidden_dims if isinstance(dim, (int, float)) and 128 <= dim <= 256)
            
            complexity_score += large_layers * 1.0
            complexity_score += medium_layers * 0.5
            
            if large_layers > 0:
                analysis_factors.append('large_hidden_layers')
            
            # Non-standard architectures
            if any(dim > 512 for dim in hidden_dims if isinstance(dim, (int, float))):
                complexity_score += 1.5
                analysis_factors.append('very_large_layers')
        
        # Encoding dimension complexity
        encoding_dim = model_config.get('encoding_dim', 12)
        if isinstance(encoding_dim, (int, float)):
            if encoding_dim > 50:
                complexity_score += 1.0
                analysis_factors.append('large_encoding_dim')
            elif encoding_dim > 24:
                complexity_score += 0.5
                analysis_factors.append('medium_encoding_dim')
        
        # Activation and normalization complexity
        activation = model_config.get('activation', 'relu')
        if activation in ['gelu', 'swish', 'mish']:
            complexity_score += 0.5
            analysis_factors.append('advanced_activation')
        elif activation in ['leaky_relu', 'elu']:
            complexity_score += 0.2
            analysis_factors.append('parameterized_activation')
        
        normalization = model_config.get('normalization')
        if normalization == 'batch':
            complexity_score += 0.3
            analysis_factors.append('batch_normalization')
        elif normalization == 'layer':
            complexity_score += 0.4
            analysis_factors.append('layer_normalization')
        elif normalization == 'group':
            complexity_score += 0.6
            analysis_factors.append('group_normalization')
        
        # Advanced features
        if model_config.get('use_batch_norm', False):
            complexity_score += 0.3
            analysis_factors.append('batch_norm_layers')
        
        if model_config.get('use_layer_norm', False):
            complexity_score += 0.4
            analysis_factors.append('layer_norm_layers')
        
        if model_config.get('skip_connection', False):
            complexity_score += 0.5
            analysis_factors.append('skip_connections')
        
        if model_config.get('residual_blocks', False):
            complexity_score += 1.0
            analysis_factors.append('residual_architecture')
        
        # Training complexity
        training_config = config.get('training', {})
        
        # Mixed precision and advanced training features
        if training_config.get('mixed_precision', False):
            complexity_score += 0.5
            analysis_factors.append('mixed_precision')
        
        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        if gradient_accumulation_steps > 1:
            complexity_score += 0.3 * min(gradient_accumulation_steps / 2, 2)
            analysis_factors.append('gradient_accumulation')
        
        # Advanced optimizers and schedulers
        optimizer = training_config.get('optimizer', 'Adam')
        if optimizer in ['AdamW', 'RMSprop', 'Adagrad']:
            complexity_score += 0.2
            analysis_factors.append('advanced_optimizer')
        elif optimizer == 'LBFGS':
            complexity_score += 0.5
            analysis_factors.append('second_order_optimizer')
        
        scheduler = training_config.get('scheduler')
        if scheduler in ['CosineAnnealingLR', 'ReduceLROnPlateau']:
            complexity_score += 0.2
            analysis_factors.append('adaptive_scheduler')
        elif scheduler in ['CyclicLR', 'OneCycleLR']:
            complexity_score += 0.4
            analysis_factors.append('cyclic_scheduler')
        
        # Regularization complexity
        dropout_rates = model_config.get('dropout_rates', [])
        if isinstance(dropout_rates, list) and len(dropout_rates) > 2:
            complexity_score += 0.3
            analysis_factors.append('complex_dropout')
        
        weight_decay = training_config.get('weight_decay', 0)
        if isinstance(weight_decay, (int, float)) and weight_decay > 1e-3:
            complexity_score += 0.1
            analysis_factors.append('strong_regularization')
        
        # Data complexity factors
        data_config = config.get('data', {})
        features = data_config.get('features', 20)
        if isinstance(features, int):
            if features > 100:
                complexity_score += 1.0
                analysis_factors.append('high_dimensional_data')
            elif features > 50:
                complexity_score += 0.5
                analysis_factors.append('medium_dimensional_data')
        
        # Advanced data preprocessing
        if data_config.get('synthetic_generation', {}).get('cluster_variance', 1.0) != 1.0:
            complexity_score += 0.2
            analysis_factors.append('custom_data_generation')
        
        # Security and monitoring complexity
        security_config = config.get('security', {})
        if security_config.get('enable_security_metrics', False):
            complexity_score += 0.3
            analysis_factors.append('security_monitoring')
        
        monitoring_config = config.get('monitoring', {})
        if monitoring_config.get('tensorboard_logging', False):
            complexity_score += 0.2
            analysis_factors.append('tensorboard_logging')
        
        if monitoring_config.get('wandb_logging', False):
            complexity_score += 0.3
            analysis_factors.append('wandb_integration')
        
        # Hardware-specific complexity
        hardware_config = config.get('hardware', {})
        if hardware_config.get('distributed_training', False):
            complexity_score += 2.0
            analysis_factors.append('distributed_training')
        
        if hardware_config.get('multi_gpu', False):
            complexity_score += 1.0
            analysis_factors.append('multi_gpu_training')
        
        # Determine complexity level with more granular categories
        if complexity_score < 2.0:
            level = 'low'
        elif complexity_score < 5.0:
            level = 'medium'
        elif complexity_score < 10.0:
            level = 'high'
        else:
            level = 'very_high'
        
        # Log analysis for debugging
        logger.debug(f"Complexity analysis: score={complexity_score:.2f}, level={level}, factors={analysis_factors}")
        
        return level
        
    except Exception as e:
        logger.warning(f"Error estimating config complexity: {e}")
        return 'unknown'

def determine_preset_recommendations(config: Dict[str, Any]) -> List[str]:
    """Determine comprehensive recommendations for what this preset is suitable for.
    
    Args:
        config: Configuration dictionary to analyze
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    try:
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        security_config = config.get('security', {})
        
        # Model type based recommendations
        model_type = model_config.get('model_type', '')
        if model_type == 'SimpleAutoencoder':
            recommendations.extend([
                'debugging and development',
                'prototyping new features',
                'resource-constrained environments',
                'educational purposes',
                'baseline comparisons',
                'rapid experimentation'
            ])
        elif model_type == 'EnhancedAutoencoder':
            recommendations.extend([
                'production deployment',
                'balanced performance needs',
                'configurable complexity scenarios',
                'standard anomaly detection tasks',
                'research and development',
                'performance optimization studies'
            ])
        elif model_type == 'AutoencoderEnsemble':
            recommendations.extend([
                'high accuracy requirements',
                'critical applications',
                'robust anomaly detection',
                'production systems with high stakes',
                'research requiring state-of-the-art performance',
                'applications where false negatives are costly'
            ])
        
        # Complexity-based recommendations
        complexity = estimate_config_complexity(config)
        if complexity == 'low':
            recommendations.extend([
                'beginner-friendly setups',
                'quick validation experiments',
                'resource-limited testing',
                'CI/CD pipeline integration'
            ])
        elif complexity == 'medium':
            recommendations.extend([
                'balanced complexity needs',
                'typical production workloads',
                'standard research applications'
            ])
        elif complexity == 'high':
            recommendations.extend([
                'advanced users',
                'complex anomaly patterns',
                'high-performance computing environments',
                'research pushing boundaries'
            ])
        elif complexity == 'very_high':
            recommendations.extend([
                'expert users only',
                'cutting-edge research',
                'specialized high-performance applications',
                'dedicated infrastructure requirements'
            ])
        
        # Training configuration recommendations
        batch_size = training_config.get('batch_size', 32)
        if batch_size <= 8:
            recommendations.extend([
                'severely memory-constrained environments',
                'edge computing applications',
                'single-sample inference needs'
            ])
        elif batch_size <= 32:
            recommendations.extend([
                'memory-constrained environments',
                'standard development setups',
                'typical research configurations'
            ])
        elif batch_size <= 128:
            recommendations.extend([
                'high-throughput scenarios',
                'batch processing applications',
                'GPU-optimized training'
            ])
        else:
            recommendations.extend([
                'very high-throughput scenarios',
                'large-scale data processing',
                'distributed computing environments'
            ])
        
        # Learning rate recommendations
        learning_rate = training_config.get('learning_rate', 0.001)
        if isinstance(learning_rate, (int, float)):
            if learning_rate >= 0.01:
                recommendations.append('fast convergence requirements')
            elif learning_rate <= 0.0001:
                recommendations.append('stable, fine-tuned training')
        
        # Mixed precision recommendations
        if training_config.get('mixed_precision', False):
            recommendations.extend([
                'GPU-accelerated training',
                'memory efficiency requirements',
                'modern hardware utilization'
            ])
        
        # Advanced training features
        if training_config.get('gradient_accumulation_steps', 1) > 1:
            recommendations.append('limited memory with large effective batch size needs')
        
        # Hardware-specific recommendations
        encoding_dim = model_config.get('encoding_dim', 12)
        hidden_dims = model_config.get('hidden_dims', [])
        
        total_params_estimate = 0
        if isinstance(hidden_dims, list) and hidden_dims:
            features = data_config.get('features', 20)
            total_params_estimate = features * hidden_dims[0]
            for i in range(len(hidden_dims) - 1):
                total_params_estimate += hidden_dims[i] * hidden_dims[i + 1]
            total_params_estimate += hidden_dims[-1] * encoding_dim
        
        if model_type == 'AutoencoderEnsemble':
            num_models = model_config.get('num_models', 3)
            total_params_estimate *= num_models
        
        if total_params_estimate < 10000:
            recommendations.extend([
                'CPU-only environments',
                'minimal resource scenarios',
                'embedded systems (with modifications)'
            ])
        elif total_params_estimate < 100000:
            recommendations.extend([
                'standard desktop environments',
                'entry-level GPU systems',
                'typical cloud instances'
            ])
        elif total_params_estimate < 1000000:
            recommendations.extend([
                'mid-range GPU systems',
                'professional workstations',
                'dedicated training servers'
            ])
        else:
            recommendations.extend([
                'high-end GPU systems',
                'specialized ML infrastructure',
                'enterprise-grade hardware'
            ])
        
        # Data characteristics recommendations
        features = data_config.get('features', 20)
        if isinstance(features, int):
            if features <= 10:
                recommendations.append('low-dimensional data analysis')
            elif features <= 50:
                recommendations.append('medium-dimensional data analysis')
            else:
                recommendations.append('high-dimensional data analysis')
        
        normal_samples = data_config.get('normal_samples', 8000)
        attack_samples = data_config.get('attack_samples', 2000)
        if isinstance(normal_samples, int) and isinstance(attack_samples, int):
            total_samples = normal_samples + attack_samples
            if total_samples < 1000:
                recommendations.append('small dataset scenarios')
            elif total_samples < 10000:
                recommendations.append('medium dataset scenarios')
            else:
                recommendations.append('large dataset scenarios')
        
        # Security-specific recommendations
        percentile = security_config.get('percentile', 95)
        if isinstance(percentile, (int, float)):
            if percentile >= 99:
                recommendations.append('high-security applications')
            elif percentile >= 95:
                recommendations.append('standard security requirements')
            else:
                recommendations.append('relaxed security thresholds')
        
        if security_config.get('enable_security_metrics', False):
            recommendations.append('security-focused deployments')
        
        # Use case pattern matching
        if 'high accuracy' in ' '.join(recommendations) and 'GPU' in ' '.join(recommendations):
            recommendations.append('mission-critical anomaly detection systems')
        
        if 'memory-constrained' in ' '.join(recommendations) and 'CPU' in ' '.join(recommendations):
            recommendations.append('IoT and edge computing deployments')
        
        if 'research' in ' '.join(recommendations) and complexity in ['high', 'very_high']:
            recommendations.append('academic and industrial research projects')
        
        # Remove duplicates and sort
        recommendations = list(set(recommendations))
        recommendations.sort()
        
        # Ensure we have at least one recommendation
        if not recommendations:
            recommendations = ['general purpose anomaly detection']
        
        return recommendations
        
    except Exception as e:
        logger.warning(f"Error determining preset recommendations: {e}")
        return ['general purpose']

def estimate_memory_requirements(config: Dict[str, Any]) -> str:
    """Estimate comprehensive memory requirements for the configuration.
    
    Args:
        config: Configuration dictionary to analyze
        
    Returns:
        Formatted string describing memory requirements
    """
    try:
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        hardware_config = config.get('hardware', {})
        
        # Extract key parameters
        model_type = model_config.get('model_type', 'SimpleAutoencoder')
        encoding_dim = model_config.get('encoding_dim', 12)
        hidden_dims = model_config.get('hidden_dims', [128, 64])
        features = data_config.get('features', 20)
        batch_size = training_config.get('batch_size', 64)
        mixed_precision = training_config.get('mixed_precision', False)
        
        # Normalize hidden_dims
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else [64]
        
        # Parameter count estimation with more accuracy
        total_params = 0
        
        if model_type == 'SimpleAutoencoder':
            # Encoder: input -> encoding
            # weights + bias
            total_params += features * encoding_dim + encoding_dim
            # Decoder: encoding -> output
            # weights + bias
            total_params += encoding_dim * features + features
            
        elif model_type == 'EnhancedAutoencoder':
            current_dim = features
            
            # Encoder layers
            for hidden_dim in hidden_dims:
                # weights + bias
                total_params += current_dim * hidden_dim + hidden_dim
                current_dim = hidden_dim
            
            # Bottleneck layer
            total_params += current_dim * encoding_dim + encoding_dim
            
            # Decoder layers (reverse)
            current_dim = encoding_dim
            for hidden_dim in reversed(hidden_dims):
                total_params += current_dim * hidden_dim + hidden_dim
                current_dim = hidden_dim
            
            # Output layer
            total_params += current_dim * features + features
            
            # Normalization parameters
            normalization = model_config.get('normalization')
            if normalization in ['batch', 'layer']:
                # Each layer has scale and shift parameters
                norm_params = sum(hidden_dims) * 2 + encoding_dim * 2
                total_params += norm_params
            
        elif model_type == 'AutoencoderEnsemble':
            num_models = model_config.get('num_models', 3)
            
            # Estimate single model parameters (simplified as Enhanced)
            single_model_params = 0
            if hidden_dims:
                current_dim = features
                # Simplified estimation
                for hidden_dim in hidden_dims[:min(2, len(hidden_dims))]:
                    single_model_params += current_dim * hidden_dim + hidden_dim
                    current_dim = hidden_dim
                single_model_params += current_dim * encoding_dim + encoding_dim
                single_model_params += encoding_dim * current_dim + current_dim
                single_model_params += current_dim * features + features
            else:
                single_model_params += features * encoding_dim + encoding_dim
                single_model_params += encoding_dim * features + features
            
            total_params = single_model_params * num_models
        
        # Memory calculations (in bytes)
        # float32 vs float16
        bytes_per_param = 4 if not mixed_precision else 2
        
        # Model parameters memory
        param_memory = total_params * bytes_per_param
        
        # Gradient memory (same size as parameters during training)
        gradient_memory = param_memory
        
        # Optimizer state memory (Adam uses 2x parameters for momentum and variance)
        optimizer = training_config.get('optimizer', 'Adam')
        if optimizer in ['Adam', 'AdamW']:
            optimizer_memory = param_memory * 2
        elif optimizer in ['SGD']:
            momentum = training_config.get('momentum', 0.9)
            optimizer_memory = param_memory if momentum > 0 else 0
        else:
            optimizer_memory = param_memory  # Conservative estimate
        
        # Activation memory (depends on batch size and architecture)
        activation_memory = 0
        
        # Input/output activations
        activation_memory += batch_size * features * bytes_per_param * 2
        
        # Hidden layer activations
        if model_type == 'EnhancedAutoencoder':
            for hidden_dim in hidden_dims:
                activation_memory += batch_size * hidden_dim * bytes_per_param
        
        # Encoding layer activation
        activation_memory += batch_size * encoding_dim * bytes_per_param
        
        # Ensemble multiplier
        if model_type == 'AutoencoderEnsemble':
            num_models = model_config.get('num_models', 3)
            activation_memory *= num_models
        
        # Additional memory for data loading and preprocessing
        data_memory = 0
        
        # Training data in memory
        normal_samples = data_config.get('normal_samples', 8000)
        attack_samples = data_config.get('attack_samples', 2000)
        total_samples = normal_samples + attack_samples
        
        # Assume data is kept in memory during training
        data_memory += total_samples * features * bytes_per_param
        
        # Validation and test sets
        validation_split = data_config.get('validation_split', 0.2)
        test_split = data_config.get('test_split', 0.2)
        data_memory += total_samples * (validation_split + test_split) * features * bytes_per_param
        
        # Buffer for data loading and augmentation
        data_memory *= 1.5
        
        # GPU-specific considerations
        gpu_overhead = 0
        if hardware_config.get('device', 'auto') != 'cpu':
            # GPU memory fragmentation and CUDA overhead
            # At least 100MB overhead
            gpu_overhead = max(param_memory * 0.1, 100 * 1024 * 1024)
        
        # Total memory calculation
        training_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
        total_memory = training_memory + data_memory + gpu_overhead
        
        # Convert to human-readable format
        def format_memory(bytes_val):
            if bytes_val < 1024 ** 2:
                return f"{bytes_val / 1024:.1f} KB"
            elif bytes_val < 1024 ** 3:
                return f"{bytes_val / (1024 ** 2):.1f} MB"
            else:
                return f"{bytes_val / (1024 ** 3):.2f} GB"
        
        # Categorize memory requirements
        total_memory_mb = total_memory / (1024 ** 2)
        
        if total_memory_mb < 50:
            category = "Very Low"
            recommendation = "Suitable for any system"
        elif total_memory_mb < 200:
            category = "Low"
            recommendation = "Standard desktop/laptop"
        elif total_memory_mb < 1000:
            category = "Medium"
            recommendation = "8GB+ RAM, entry GPU"
        elif total_memory_mb < 4000:
            category = "High"
            recommendation = "16GB+ RAM, mid-range GPU"
        elif total_memory_mb < 16000:
            category = "Very High"
            recommendation = "32GB+ RAM, high-end GPU"
        else:
            category = "Extreme"
            recommendation = "Specialized hardware required"
        
        # Create detailed breakdown
        breakdown = {
            'model_parameters': format_memory(param_memory),
            'gradients': format_memory(gradient_memory),
            'optimizer_state': format_memory(optimizer_memory),
            'activations': format_memory(activation_memory),
            'data_storage': format_memory(data_memory),
            'gpu_overhead': format_memory(gpu_overhead) if gpu_overhead > 0 else "N/A",
            'total_training': format_memory(training_memory),
            'total_with_data': format_memory(total_memory)
        }
        
        # Format final result
        result = f"{category} ({format_memory(total_memory)}) - {recommendation}"
        
        # Add breakdown in debug mode
        logger.debug(f"Memory estimation breakdown: {breakdown}")
        
        return result
        
    except Exception as e:
        logger.warning(f"Error estimating memory requirements: {e}")
        return "Unknown - estimation failed"

def estimate_training_time(config: Dict[str, Any]) -> str:
    """Estimate training time based on configuration complexity and data size.
    
    Args:
        config: Configuration dictionary to analyze
        
    Returns:
        Formatted string describing estimated training time
    """
    try:
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        hardware_config = config.get('hardware', {})
        
        # Extract key parameters
        model_type = model_config.get('model_type', 'SimpleAutoencoder')
        encoding_dim = model_config.get('encoding_dim', 12)
        hidden_dims = model_config.get('hidden_dims', [128, 64])
        features = data_config.get('features', 20)
        batch_size = training_config.get('batch_size', 64)
        epochs = training_config.get('epochs', 100)
        mixed_precision = training_config.get('mixed_precision', False)
        
        # Data size
        normal_samples = data_config.get('normal_samples', 8000)
        attack_samples = data_config.get('attack_samples', 2000)
        total_samples = normal_samples + attack_samples
        
        # Calculate basic metrics
        steps_per_epoch = max(1, total_samples // batch_size)
        total_steps = steps_per_epoch * epochs
        
        # Base time per step estimation (in seconds)
        # 1ms baseline for very simple operations
        base_time_per_step = 0.001
        
        # Model complexity multiplier
        complexity_multiplier = 1.0
        
        if model_type == 'SimpleAutoencoder':
            complexity_multiplier = 1.0
        elif model_type == 'EnhancedAutoencoder':
            complexity_multiplier = 2.0
            
            # Hidden layer complexity
            if isinstance(hidden_dims, list):
                layer_complexity = len(hidden_dims) * 0.5
                size_complexity = sum(dim for dim in hidden_dims if isinstance(dim, (int, float))) / 1000
                complexity_multiplier += layer_complexity + size_complexity
            
            # Normalization overhead
            normalization = model_config.get('normalization')
            if normalization == 'batch':
                complexity_multiplier *= 1.2
            elif normalization == 'layer':
                complexity_multiplier *= 1.3
                
        elif model_type == 'AutoencoderEnsemble':
            num_models = model_config.get('num_models', 3)
            # Base ensemble overhead + linear scaling
            complexity_multiplier = 1.5 * num_models
        
        # Feature dimension impact
        if isinstance(features, int):
            # Normalize to 20 features baseline
            feature_multiplier = max(1.0, features / 20)
            complexity_multiplier *= feature_multiplier
        
        # Batch size impact (smaller batches = more overhead)
        if batch_size < 32:
            complexity_multiplier *= 1.5
        elif batch_size > 128:
            # Better GPU utilization
            complexity_multiplier *= 0.8
        
        # Activation function impact
        activation = model_config.get('activation', 'relu')
        if activation in ['gelu', 'swish']:
            complexity_multiplier *= 1.1
        elif activation in ['tanh', 'sigmoid']:
            complexity_multiplier *= 1.05
        
        # Hardware considerations
        device = hardware_config.get('device', 'auto')
        hardware_multiplier = 1.0
        
        # Try to detect actual hardware or use config
        try:
            if torch.cuda.is_available() and device != 'cpu':
                # GPU training - much faster
                hardware_multiplier = 0.1
                
                # GPU-specific optimizations
                if mixed_precision:
                    # Mixed precision speedup
                    hardware_multiplier *= 0.7
                
                # Multi-GPU
                if hardware_config.get('multi_gpu', False):
                    gpu_count = torch.cuda.device_count()
                    # Diminishing returns after 4 GPUs
                    hardware_multiplier /= min(gpu_count, 4)
                    
            else:
                # CPU training
                hardware_multiplier = 1.0
                
                # CPU-specific considerations
                num_workers = training_config.get('num_workers', 1)
                if num_workers > 1:
                    hardware_multiplier *= max(0.5, 1.0 / min(num_workers, 8))
                    
        except Exception:
            # Fallback assumptions
            if device == 'cpu':
                hardware_multiplier = 1.0
            else:
                # Assume some GPU acceleration
                hardware_multiplier = 0.2
        
        # Training-specific factors
        optimizer = training_config.get('optimizer', 'Adam')
        if optimizer in ['LBFGS']:
            # Second-order methods are much slower
            complexity_multiplier *= 3.0
        elif optimizer in ['AdamW', 'RMSprop']:
            complexity_multiplier *= 1.1
        
        # Gradient accumulation
        grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        if grad_accum_steps > 1:
            # Some overhead for accumulation
            complexity_multiplier *= 1.2
        
        # Early stopping consideration
        patience = training_config.get('patience', 0)
        early_stop_factor = 1.0
        if patience > 0:
            # Assume early stopping might reduce training by 20-50%
            early_stop_factor = 0.7
        
        # Calculate total time
        time_per_step = base_time_per_step * complexity_multiplier * hardware_multiplier
        total_time_seconds = total_steps * time_per_step * early_stop_factor
        
        # Add overhead for data loading, checkpointing, etc.
        overhead_factor = 1.3
        total_time_seconds *= overhead_factor
        
        # Convert to human-readable format
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f} seconds"
            elif seconds < 3600:
                return f"{seconds / 60:.1f} minutes"
            elif seconds < 86400:
                return f"{seconds / 3600:.1f} hours"
            else:
                return f"{seconds / 86400:.1f} days"
        
        # Determine category
        if total_time_seconds < 60:
            category = "Very Fast"
        # 10 minutes
        elif total_time_seconds < 600:
            category = "Fast"
        # 1 hour
        elif total_time_seconds < 3600:
            category = "Moderate"
        # 4 hours
        elif total_time_seconds < 14400:
            category = "Slow"
        # 1 day
        elif total_time_seconds < 86400:
            category = "Very Slow"
        else:
            category = "Extremely Slow"
        
        # Create estimate ranges (±50% uncertainty)
        min_time = total_time_seconds * 0.5
        max_time = total_time_seconds * 1.5
        
        # Format result
        if min_time < 60 and max_time > 60:
            result = f"{category} ({format_time(min_time)} - {format_time(max_time)})"
        else:
            result = f"{category} (~{format_time(total_time_seconds)})"
        
        # Add context information
        context_info = []
        if hardware_multiplier <= 0.2:
            context_info.append("GPU-accelerated")
        else:
            context_info.append("CPU-based")
            
        if mixed_precision and hardware_multiplier <= 0.2:
            context_info.append("mixed precision")
            
        if early_stop_factor < 1.0:
            context_info.append("with early stopping")
            
        if context_info:
            result += f" ({', '.join(context_info)})"
        
        # Debug information
        logger.debug(f"Training time estimation: steps={total_steps}, complexity_mult={complexity_multiplier:.2f}, "
                    f"hardware_mult={hardware_multiplier:.2f}, time_per_step={time_per_step:.6f}s")
        
        return result
        
    except Exception as e:
        logger.warning(f"Error estimating training time: {e}")
        return "Unknown - estimation failed"

def determine_resource_level(config: Dict[str, Any]) -> str:
    """Determine overall resource level required for the configuration.
    
    Args:
        config: Configuration dictionary to analyze
        
    Returns:
        String indicating resource level: 'minimal', 'low', 'medium', 'high', or 'extreme'
    """
    try:
        # Get individual assessments
        complexity = estimate_config_complexity(config)
        memory_req = estimate_memory_requirements(config)
        training_time = estimate_training_time(config)
        
        # Extract key indicators from other assessments
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        hardware_config = config.get('hardware', {})
        
        # Initialize scoring system
        resource_score = 0.0
        factors = []
        
        # Complexity contribution (30% weight)
        complexity_scores = {
            'low': 1.0,
            'medium': 2.5,
            'high': 4.0,
            'very_high': 6.0,
            'unknown': 2.0
        }
        resource_score += complexity_scores.get(complexity, 2.0) * 0.3
        factors.append(f"complexity_{complexity}")
        
        # Memory contribution (25% weight)
        if 'Very Low' in memory_req or 'KB' in memory_req:
            memory_score = 0.5
        elif 'Low' in memory_req and 'MB' in memory_req:
            memory_score = 1.0
        elif 'Medium' in memory_req:
            memory_score = 2.0
        elif 'High' in memory_req and 'GB' not in memory_req:
            memory_score = 3.5
        elif 'Very High' in memory_req or 'GB' in memory_req:
            memory_score = 5.0
        elif 'Extreme' in memory_req:
            memory_score = 6.0
        else:
            memory_score = 2.0
        
        resource_score += memory_score * 0.25
        factors.append(f"memory_score_{memory_score}")
        
        # Training time contribution (20% weight)
        if 'Very Fast' in training_time or 'Fast' in training_time:
            time_score = 1.0
        elif 'Moderate' in training_time:
            time_score = 2.0
        elif 'Slow' in training_time:
            time_score = 3.0
        elif 'Very Slow' in training_time:
            time_score = 4.0
        elif 'Extremely Slow' in training_time:
            time_score = 5.0
        else:
            time_score = 2.0
        
        resource_score += time_score * 0.20
        factors.append(f"time_score_{time_score}")
        
        # Model-specific factors (15% weight)
        model_type = model_config.get('model_type', 'SimpleAutoencoder')
        if model_type == 'SimpleAutoencoder':
            model_score = 1.0
        elif model_type == 'EnhancedAutoencoder':
            model_score = 2.0
            
            # Enhanced model complexity factors
            hidden_dims = model_config.get('hidden_dims', [])
            if isinstance(hidden_dims, list):
                if len(hidden_dims) > 3:
                    model_score += 0.5
                if any(dim > 256 for dim in hidden_dims if isinstance(dim, (int, float))):
                    model_score += 0.5
                    
        elif model_type == 'AutoencoderEnsemble':
            num_models = model_config.get('num_models', 3)
            model_score = 2.0 + (num_models - 1) * 0.5
        else:
            model_score = 2.0
        
        resource_score += model_score * 0.15
        factors.append(f"model_score_{model_score}")
        
        # Hardware requirements (10% weight)
        hardware_score = 1.0
        
        # GPU requirements
        if hardware_config.get('multi_gpu', False):
            hardware_score += 2.0
            factors.append("multi_gpu")
        elif hardware_config.get('device', 'auto') != 'cpu':
            hardware_score += 1.0
            factors.append("gpu_required")
        
        # Distributed training
        if hardware_config.get('distributed_training', False):
            hardware_score += 2.0
            factors.append("distributed")
        
        # Mixed precision (actually reduces requirements)
        if training_config.get('mixed_precision', False):
            hardware_score -= 0.2
            factors.append("mixed_precision_benefit")
        
        resource_score += hardware_score * 0.10
        factors.append(f"hardware_score_{hardware_score}")
        
        # Data size and complexity factors (bonus/penalty)
        features = data_config.get('features', 20)
        normal_samples = data_config.get('normal_samples', 8000)
        attack_samples = data_config.get('attack_samples', 2000)
        
        if isinstance(features, int) and features > 100:
            resource_score += 0.5
            factors.append("high_dimensional")
        
        total_samples = (normal_samples if isinstance(normal_samples, int) else 8000) + \
                       (attack_samples if isinstance(attack_samples, int) else 2000)
        
        if total_samples > 50000:
            resource_score += 0.5
            factors.append("large_dataset")
        elif total_samples < 1000:
            resource_score -= 0.2
            factors.append("small_dataset_benefit")
        
        # Batch size considerations
        batch_size = training_config.get('batch_size', 64)
        if isinstance(batch_size, int):
            if batch_size > 256:
                resource_score += 0.3
                factors.append("large_batch")
            elif batch_size < 8:
                resource_score -= 0.1
                factors.append("small_batch_benefit")
        
        # Determine final resource level
        if resource_score < 1.5:
            level = 'minimal'
            description = "Basic CPU, <4GB RAM"
        elif resource_score < 2.5:
            level = 'low'
            description = "Standard desktop, 4-8GB RAM"
        elif resource_score < 4.0:
            level = 'medium'
            description = "Mid-range GPU, 8-16GB RAM"
        elif resource_score < 5.5:
            level = 'high'
            description = "High-end GPU, 16-32GB RAM"
        else:
            level = 'extreme'
            description = "Specialized hardware, >32GB RAM"
        
        # Create detailed result
        result = f"{level} ({description})"
        
        # Debug information
        logger.debug(f"Resource level determination: score={resource_score:.2f}, level={level}, factors={factors}")
        
        return result
        
    except Exception as e:
        logger.warning(f"Error determining resource level: {e}")
        return 'unknown - assessment failed'

# Model variants validation
def validate_model_variants(logger: logging.Logger, silent: bool = False) -> Dict[str, str]:
    """Validate all registered model variants with comprehensive testing.
    
    Args:
        logger: Logger instance for reporting
        silent: If True, suppress detailed logging messages during system checks
        
    Returns:
        Dictionary mapping model names to their status
    """
    variant_status = {}
    validation_details = {}
    
    if not MODEL_VARIANTS:
        if not silent:
            logger.warning("MODEL_VARIANTS is empty, attempting initialization")
        try:
            initialize_model_variants(silent=silent)
        except Exception as e:
            if not silent:
                logger.error(f"Failed to initialize model variants for validation: {e}")
            return {'error': f'initialization_failed: {str(e)}'}
    
    if not silent:
        logger.info(f"Validating {len(MODEL_VARIANTS)} model variants: {list(MODEL_VARIANTS.keys())}")
    
    # Get current configuration for realistic testing
    try:
        current_config = get_current_config()
        test_config = current_config.get('model', {}) if isinstance(current_config, dict) else {}
        data_config = current_config.get('data', {}) if isinstance(current_config, dict) else {}
        
        test_input_dim = data_config.get('features', 20)
        test_encoding_dim = test_config.get('encoding_dim', DEFAULT_ENCODING_DIM)
        test_hidden_dims = test_config.get('hidden_dims', HIDDEN_LAYER_SIZES.copy())
        test_dropout_rates = test_config.get('dropout_rates', DROPOUT_RATES.copy())
        
    except Exception as e:
        if not silent:
            logger.warning(f"Could not load config for validation, using defaults: {e}")
        test_input_dim = 20
        test_encoding_dim = DEFAULT_ENCODING_DIM
        test_hidden_dims = HIDDEN_LAYER_SIZES.copy()
        test_dropout_rates = DROPOUT_RATES.copy()
    
    # Extract and validate test parameters with comprehensive fallbacks
    def get_safe_param(config: Dict, key: str, global_name: str, default: Any, validator=None):
        """Safely get parameter with validation and fallbacks."""
        try:
            # Try config first
            value = config.get(key)
            if value is not None and (validator is None or validator(value)):
                return value
            
            # Try global variable
            if global_name in globals():
                global_value = globals()[global_name]
                if validator is None or validator(global_value):
                    return global_value
            
            # Use default
            return default
        except Exception:
            return default
    
    # Get validated test parameters
    test_input_dim = get_safe_param(
        data_config, 'features', 'FEATURES', 20,
        lambda x: isinstance(x, int) and x > 0
    )
    
    test_encoding_dim = get_safe_param(
        test_config, 'encoding_dim', 'DEFAULT_ENCODING_DIM', 8,
        lambda x: isinstance(x, int) and x > 0
    )
    
    test_hidden_dims = get_safe_param(
        test_config, 'hidden_dims', 'HIDDEN_LAYER_SIZES', [64, 32],
        lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(d, int) and d > 0 for d in x)
    )
    
    test_dropout_rates = get_safe_param(
        test_config, 'dropout_rates', 'DROPOUT_RATES', [0.2, 0.15],
        lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(r, (int, float)) and 0 <= r < 1 for r in x)
    )
    
    # Ensure list compatibility and matching lengths
    if not isinstance(test_hidden_dims, list):
        test_hidden_dims = [test_hidden_dims] if isinstance(test_hidden_dims, int) else [128, 64]
    
    if not isinstance(test_dropout_rates, list):
        test_dropout_rates = [test_dropout_rates] if isinstance(test_dropout_rates, (int, float)) else [0.2, 0.15]
    
    # Fix length mismatch
    min_length = min(len(test_hidden_dims), len(test_dropout_rates))
    if min_length > 0:
        test_hidden_dims = test_hidden_dims[:min_length]
        test_dropout_rates = test_dropout_rates[:min_length]
    else:
        test_hidden_dims = [64]
        test_dropout_rates = [0.2]
    
    if not silent:
        logger.debug(f"Using test parameters: input_dim={test_input_dim}, encoding_dim={test_encoding_dim}, "
                    f"hidden_dims={test_hidden_dims}, dropout_rates={test_dropout_rates}")
    
    validation_tests = [
        {'batch_size': 2, 'description': 'batch_norm_compatible'},
        {'batch_size': 1, 'description': 'single_sample'},
        {'batch_size': 4, 'description': 'small_batch'},
    ]
    
    for name, variant_class in MODEL_VARIANTS.items():
        validation_start_time = time.time()
        status_details = []
        overall_status = 'available'
        variant_details = {
            'class_name': variant_class.__name__ if variant_class else 'None',
            'tests_performed': [],
            'errors': [],
            'warnings': [],
            'performance': {}
        }
        
        try:
            if not silent:
                logger.debug(f"Validating model variant: {name}")
            
            # Test class availability
            if variant_class is None:
                variant_status[name] = 'class_not_found'
                variant_details['errors'].append('Model class is None')
                continue
            
            if not callable(variant_class):
                variant_status[name] = 'class_not_callable'
                variant_details['errors'].append('Model class is not callable')
                continue
            
            variant_details['tests_performed'].append('class_availability')
            
            # Create test parameters based on model type
            if name == 'SimpleAutoencoder':
                test_params = {
                    'input_dim': test_input_dim,
                    'encoding_dim': test_encoding_dim,
                    # Disable for testing stability
                    'mixed_precision': False
                }
            elif name == 'EnhancedAutoencoder':
                test_params = {
                    'input_dim': test_input_dim,
                    'encoding_dim': test_encoding_dim,
                    'hidden_dims': test_hidden_dims.copy(),
                    'dropout_rates': test_dropout_rates.copy(),
                    'activation': ACTIVATION,
                    'activation_param': ACTIVATION_PARAM,
                    'normalization': NORMALIZATION,
                    'skip_connection': True,
                    'mixed_precision': False,
                    'legacy_mode': False
                }
            elif name == 'AutoencoderEnsemble':
                test_params = {
                    'input_dim': test_input_dim,
                    'num_models': max(1, NUM_MODELS),
                    'encoding_dim': test_encoding_dim,
                    'diversity_factor': DIVERSITY_FACTOR,
                    'mixed_precision': False
                }
            else:
                # Generic test parameters
                test_params = {
                    'input_dim': test_input_dim,
                    'encoding_dim': test_encoding_dim
                }
            
            # Test instantiation
            try:
                test_instance = variant_class(**test_params)
                status_details.append('instantiation_ok')
                variant_details['tests_performed'].append('instantiation')
                if not silent:
                    logger.debug(f"[OK] {name}: Successfully instantiated")
            except Exception as e:
                error_msg = f"Instantiation failed: {str(e)}"
                variant_status[name] = 'instantiation_failed'
                variant_details['errors'].append(error_msg)
                if not silent:
                    logger.error(f"[FAIL] {name}: {error_msg}")
                continue
            
            # Test model structure validation
            try:
                total_params = sum(p.numel() for p in test_instance.parameters())
                trainable_params = sum(p.numel() for p in test_instance.parameters() if p.requires_grad)
                
                if total_params == 0:
                    variant_details['warnings'].append('Model has no parameters')
                    
                variant_details['performance']['total_parameters'] = total_params
                variant_details['performance']['trainable_parameters'] = trainable_params
                variant_details['tests_performed'].append('structure_validation')
                
            except Exception as e:
                variant_details['warnings'].append(f'Parameter count failed: {str(e)}')
            
            # Test different modes and batch sizes
            forward_pass_success = False
            for test in validation_tests:
                try:
                    test_instance.eval()  # Set to eval mode
                    test_input = torch.randn(test['batch_size'], test_input_dim)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        output = test_instance(test_input)
                    inference_time = time.time() - start_time
                    
                    # Validate output shape
                    expected_output_shape = (test['batch_size'], test_input_dim)
                    if output.shape != expected_output_shape:
                        error_msg = f"{test['description']}_shape_mismatch"
                        status_details.append(error_msg)
                        variant_details['errors'].append(
                            f'Output shape mismatch for {test["description"]}: '
                            f'expected {expected_output_shape}, got {output.shape}'
                        )
                        overall_status = 'warning'
                    else:
                        status_details.append(f"{test['description']}_ok")
                        variant_details['tests_performed'].append(f'forward_pass_{test["description"]}')
                        forward_pass_success = True
                    
                    # Check for NaN values
                    if torch.isnan(output).any():
                        warning_msg = f"{test['description']}_nan_detected"
                        status_details.append(warning_msg)
                        variant_details['warnings'].append(
                            f'NaN values in output for {test["description"]}'
                        )
                        overall_status = 'warning'
                    
                    # Check output range is reasonable
                    if torch.abs(output).max() > 1000:
                        warning_msg = f"{test['description']}_extreme_values"
                        status_details.append(warning_msg)
                        variant_details['warnings'].append(
                            f'Extreme values in output for {test["description"]}'
                        )
                        overall_status = 'warning'
                    
                    variant_details['performance'][f'inference_time_{test["description"]}'] = inference_time
                    
                except Exception as test_error:
                    error_msg = f"{test['description']}_failed: {str(test_error)}"
                    status_details.append(error_msg)
                    variant_details['errors'].append(f'Forward pass failed for {test["description"]}: {str(test_error)}')
                    if test['description'] == 'batch_norm_compatible':
                        overall_status = 'error'
            
            if not forward_pass_success:
                variant_status[name] = 'forward_pass_failed'
                continue
            
            # Test training mode if applicable
            try:
                test_instance.train()
                # Use batch_size > 1 for batch norm
                test_input = torch.randn(4, test_input_dim)
                with torch.no_grad():
                    _ = test_instance(test_input)
                status_details.append('training_mode_ok')
                variant_details['tests_performed'].append('training_mode_compatibility')
            except Exception as train_error:
                error_msg = f'training_mode_failed: {str(train_error)}'
                status_details.append(error_msg)
                variant_details['warnings'].append(f'Training mode issues: {str(train_error)}')
                overall_status = 'warning'
            
            # Test configuration methods
            try:
                if hasattr(test_instance, 'get_config'):
                    config_dict = test_instance.get_config()
                    if not isinstance(config_dict, dict):
                        variant_details['warnings'].append('get_config() does not return a dictionary')
                    else:
                        # Check for required config fields
                        required_fields = ['model_type', 'input_dim', 'encoding_dim']
                        missing_fields = [field for field in required_fields if field not in config_dict]
                        if missing_fields:
                            variant_details['warnings'].append(f'Missing config fields: {missing_fields}')
                    
                    variant_details['tests_performed'].append('configuration_methods')
            except Exception as e:
                variant_details['warnings'].append(f'Configuration method test failed: {str(e)}')
            
            # Test device compatibility (if CUDA available)
            if torch.cuda.is_available():
                try:
                    device = torch.device('cuda')
                    test_instance_cuda = test_instance.to(device)
                    test_input_cuda = torch.randn(2, test_input_dim, device=device)
                    
                    test_instance_cuda.eval()
                    with torch.no_grad():
                        cuda_output = test_instance_cuda(test_input_cuda)
                    
                    # Move back to CPU for cleanup
                    test_instance_cuda = test_instance_cuda.cpu()
                    
                    variant_details['tests_performed'].append('cuda_compatibility')
                    variant_details['performance']['cuda_compatible'] = True
                    
                except Exception as e:
                    variant_details['warnings'].append(f'CUDA compatibility issues: {str(e)}')
                    variant_details['performance']['cuda_compatible'] = False
            else:
                variant_details['performance']['cuda_compatible'] = False
            
            # Test memory efficiency
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss
                
                # Create larger batch for memory test
                large_input = torch.randn(16, test_input_dim)
                test_instance.eval()
                with torch.no_grad():
                    _ = test_instance(large_input)
                
                memory_after = process.memory_info().rss
                memory_used_mb = (memory_after - memory_before) / (1024 * 1024)
                variant_details['performance']['memory_usage_mb'] = memory_used_mb
                variant_details['tests_performed'].append('memory_efficiency')
                
            except ImportError:
                variant_details['warnings'].append('psutil not available for memory testing')
            except Exception as e:
                variant_details['warnings'].append(f'Memory test failed: {str(e)}')
            
            # Determine final status based on test results
            error_count = len(variant_details['errors'])
            warning_count = len(variant_details['warnings'])
            
            if overall_status == 'available':
                variant_status[name] = 'available'
            else:
                variant_status[name] = f'{overall_status}: {"; ".join(status_details)}'
            
            if not silent:
                logger.debug(f"Model variant {name} validation: {variant_status[name]}")
            
        except Exception as e:
            error_msg = str(e)
            variant_status[name] = f'error: {error_msg}'
            variant_details['errors'].append(f'Validation error: {error_msg}')
            if not silent:
                logger.warning(f"Model variant {name} failed validation: {error_msg}")
        
        finally:
            # Record validation timing
            validation_time = time.time() - validation_start_time
            variant_details['performance']['validation_time_seconds'] = validation_time
            validation_details[name] = variant_details
            
            # Memory cleanup
            try:
                if 'test_instance' in locals():
                    del test_instance
                if 'test_instance_cuda' in locals():
                    del test_instance_cuda
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except:
                pass
    
    # Log summary
    available_count = sum(1 for status in variant_status.values() if status == 'available')
    warning_count = sum(1 for status in variant_status.values() if status.startswith('warning'))
    error_count = sum(1 for status in variant_status.values() if status.startswith('error'))
    
    if not silent:
        logger.info(f"Model variants validation summary: {available_count} available, {warning_count} warnings, {error_count} errors")
        
        # Log detailed results for debugging
        logger.debug(f"Detailed validation results: {json.dumps(validation_details, indent=2, default=str)}")
    
    # Force memory cleanup
    try:
        enhanced_clear_memory()
    except:
        pass
    
    return variant_status

def initialize_model_variants(silent: bool = False) -> None:
    """Initialize MODEL_VARIANTS dictionary with enhanced validation and error recovery.
    
    Args:
        silent: If True, suppress detailed logging messages during system checks
    """
    global MODEL_VARIANTS
    
    if not silent:
        logger.info("Initializing model variants with enhanced configuration support")
    
    # Clear existing variants
    MODEL_VARIANTS = {}
    
    # Get current configuration with comprehensive fallbacks
    try:
        current_config = get_current_config()
        model_config = current_config.get('model', {}) if isinstance(current_config, dict) else {}
        data_config = current_config.get('data', {}) if isinstance(current_config, dict) else {}
        if not silent:
            logger.debug("Loaded current configuration for model variant initialization")
    except Exception as e:
        if not silent:
            logger.warning(f"Could not load current config, using defaults: {e}")
        model_config = {}
        data_config = {}
    
    # Helper function to safely extract parameters
    def get_config_param(config: Dict, key: str, global_var: str, default: Any, validator=None):
        """Safely extract configuration parameter with validation and fallbacks."""
        try:
            # Try configuration first
            if key in config:
                value = config[key]
                if validator is None or validator(value):
                    return value
                else:
                    if not silent:
                        logger.warning(f"Invalid config value for {key}: {value}")
            
            # Try global variable
            if global_var in globals():
                global_value = globals()[global_var]
                if validator is None or validator(global_value):
                    return global_value
                else:
                    if not silent:
                        logger.warning(f"Invalid global value for {key} ({global_var}): {global_value}")
            
            # Use default
            if not silent:
                logger.debug(f"Using default value for {key}: {default}")
            return default
            
        except Exception as e:
            if not silent:
                logger.warning(f"Error extracting parameter {key}: {e}, using default: {default}")
            return default
    
    # Extract configuration values with intelligent defaults and validation
    test_input_dim = get_config_param(
        data_config, 'features', 'FEATURES', 20,
        lambda x: isinstance(x, int) and x > 0
    )
    
    base_encoding_dim = get_config_param(
        model_config, 'encoding_dim', 'DEFAULT_ENCODING_DIM', 8,
        lambda x: isinstance(x, int) and x > 0
    )
    
    base_hidden_dims = get_config_param(
        model_config, 'hidden_dims', 'HIDDEN_LAYER_SIZES', [64, 32],
        lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(d, int) and d > 0 for d in x)
    )
    
    base_dropout_rates = get_config_param(
        model_config, 'dropout_rates', 'DROPOUT_RATES', [0.2, 0.15],
        lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(r, (int, float)) and 0 <= r < 1 for r in x)
    )
    
    activation = get_config_param(
        model_config, 'activation', 'ACTIVATION', 'relu',
        lambda x: x in ['relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid']
    )
    
    activation_param = get_config_param(
        model_config, 'activation_param', 'ACTIVATION_PARAM', 0.2,
        lambda x: isinstance(x, (int, float)) and 0 <= x <= 1
    )
    
    normalization = get_config_param(
        model_config, 'normalization', 'NORMALIZATION', None,
        lambda x: x in ['batch', 'layer', 'instance', None]
    )
    
    num_models = get_config_param(
        model_config, 'num_models', 'NUM_MODELS', 3,
        lambda x: isinstance(x, int) and 1 <= x <= 5
    )
    
    diversity_factor = get_config_param(
        model_config, 'diversity_factor', 'DIVERSITY_FACTOR', 0.1,
        lambda x: isinstance(x, (int, float)) and 0 <= x <= 1
    )
    
    use_batch_norm = get_config_param(
        model_config, 'use_batch_norm', 'USE_BATCH_NORM', False,
        lambda x: isinstance(x, bool)
    )
    
    use_layer_norm = get_config_param(
        model_config, 'use_layer_norm', 'USE_LAYER_NORM', False,
        lambda x: isinstance(x, bool)
    )
    
    # Validate and fix configuration parameters
    def validate_and_fix_lists():
        nonlocal base_hidden_dims, base_dropout_rates
        
        # Ensure hidden_dims is a valid list
        if not isinstance(base_hidden_dims, list):
            if isinstance(base_hidden_dims, (int, float)) and base_hidden_dims > 0:
                base_hidden_dims = [int(base_hidden_dims)]
                if not silent:
                    logger.info(f"Converted hidden_dims to list: {base_hidden_dims}")
            else:
                base_hidden_dims = [128, 64]
                if not silent:
                    logger.warning(f"Invalid hidden_dims, using default: {base_hidden_dims}")
        
        # Remove invalid dimensions
        base_hidden_dims = [dim for dim in base_hidden_dims 
                           if isinstance(dim, (int, float)) and dim > 0]
        if not base_hidden_dims:
            base_hidden_dims = [64]
            if not silent:
                logger.warning("No valid hidden dimensions found, using [64]")
        
        # Ensure dropout_rates is a valid list
        if not isinstance(base_dropout_rates, list):
            if isinstance(base_dropout_rates, (int, float)) and 0 <= base_dropout_rates < 1:
                base_dropout_rates = [float(base_dropout_rates)]
                if not silent:
                    logger.info(f"Converted dropout_rates to list: {base_dropout_rates}")
            else:
                base_dropout_rates = [0.2, 0.15]
                if not silent:
                    logger.warning(f"Invalid dropout_rates, using default: {base_dropout_rates}")
        
        # Remove invalid rates
        base_dropout_rates = [rate for rate in base_dropout_rates 
                             if isinstance(rate, (int, float)) and 0 <= rate < 1]
        if not base_dropout_rates:
            base_dropout_rates = [0.2]
            if not silent:
                logger.warning("No valid dropout rates found, using [0.2]")
        
        # Ensure matching lengths
        if len(base_hidden_dims) != len(base_dropout_rates):
            target_length = min(len(base_hidden_dims), len(base_dropout_rates))
            if target_length == 0:
                target_length = 1
                base_hidden_dims = [64]
                base_dropout_rates = [0.2]
            else:
                if len(base_hidden_dims) > target_length:
                    base_hidden_dims = base_hidden_dims[:target_length]
                if len(base_dropout_rates) > target_length:
                    base_dropout_rates = base_dropout_rates[:target_length]
                
                # Extend shorter list
                while len(base_hidden_dims) < target_length:
                    base_hidden_dims.append(max(32, int(base_hidden_dims[-1] * 0.8)))
                while len(base_dropout_rates) < target_length:
                    base_dropout_rates.append(max(0.1, base_dropout_rates[-1] * 0.9))
            
            if not silent:
                logger.info(f"Adjusted dimensions for compatibility: hidden_dims={base_hidden_dims}, dropout_rates={base_dropout_rates}")
    
    validate_and_fix_lists()
    
    # Enhanced model definitions with fallback configurations
    model_definitions = {
        'SimpleAutoencoder': {
            'class_check': lambda: SimpleAutoencoder is not None and callable(SimpleAutoencoder),
            'class_getter': lambda: SimpleAutoencoder,
            'primary_params': {
                'input_dim': test_input_dim,
                'encoding_dim': base_encoding_dim,
                'mixed_precision': False  # Disable for testing stability
            },
            'fallback_params': {
                'input_dim': 20,
                'encoding_dim': max(1, int(base_encoding_dim / 2)) if base_encoding_dim > 2 else 4,
                'mixed_precision': False
            },
            'minimal_params': {
                'input_dim': 10,
                'encoding_dim': 4,
                'mixed_precision': False
            },
            'required_config': ['encoding_dim'],
            'description': 'Basic autoencoder with encoder-decoder architecture'
        },
        'EnhancedAutoencoder': {
            'class_check': lambda: EnhancedAutoencoder is not None and callable(EnhancedAutoencoder),
            'class_getter': lambda: EnhancedAutoencoder,
            'primary_params': {
                'input_dim': test_input_dim,
                'encoding_dim': base_encoding_dim,
                'hidden_dims': base_hidden_dims.copy(),
                'dropout_rates': base_dropout_rates.copy(),
                'activation': activation,
                'activation_param': activation_param,
                'normalization': normalization,
                'use_batch_norm': use_batch_norm,
                'use_layer_norm': use_layer_norm,
                'skip_connection': True,
                'mixed_precision': False,
                'legacy_mode': False
            },
            'fallback_params': {
                'input_dim': 20,
                'encoding_dim': max(4, int(base_encoding_dim / 2)) if base_encoding_dim > 8 else 6,
                'hidden_dims': [64],
                'dropout_rates': [0.2],
                'activation': 'relu',
                'activation_param': 0.0,
                'normalization': None,
                'use_batch_norm': False,
                'use_layer_norm': False,
                'skip_connection': False,
                'mixed_precision': False,
                'legacy_mode': True
            },
            'minimal_params': {
                'input_dim': 20,
                'encoding_dim': 4,
                'hidden_dims': [32],
                'dropout_rates': [0.1],
                'activation': 'relu',
                'normalization': None,
                'use_batch_norm': False,
                'use_layer_norm': False,
                'skip_connection': False,
                'mixed_precision': False,
                'legacy_mode': True
            },
            'required_config': ['encoding_dim', 'hidden_dims', 'dropout_rates'],
            'description': 'Advanced autoencoder with configurable layers and normalization'
        },
        'AutoencoderEnsemble': {
            'class_check': lambda: AutoencoderEnsemble is not None and callable(AutoencoderEnsemble),
            'class_getter': lambda: AutoencoderEnsemble,
            'primary_params': {
                'input_dim': test_input_dim,
                'num_models': max(1, num_models),
                'encoding_dim': base_encoding_dim,
                'diversity_factor': diversity_factor,
                'mixed_precision': False
            },
            'fallback_params': {
                'input_dim': 20,
                'num_models': 2,
                'encoding_dim': max(4, int(base_encoding_dim / 2)) if base_encoding_dim > 8 else 6,
                'diversity_factor': 0.1,
                'mixed_precision': False
            },
            'minimal_params': {
                'input_dim': 20,
                'num_models': 2,
                'encoding_dim': 4,
                'diversity_factor': 0.05,
                'mixed_precision': False
            },
            'required_config': ['num_models', 'encoding_dim', 'diversity_factor'],
            'description': 'Ensemble of autoencoders for improved robustness'
        }
    }
    
    # Track initialization statistics
    initialization_stats = {
        'attempted': 0,
        'successful': [],
        'failed': [],
        'skipped': [],
        'fallback_used': [],
        'minimal_used': [],
        'validation_passed': 0,
        'errors': []
    }
    
    # Initialize each model variant with comprehensive error handling
    for name, definition in model_definitions.items():
        initialization_stats['attempted'] += 1
        
        try:
            if not silent:
                logger.debug(f"Attempting to initialize {name}")
            
            # Check if model class exists and is callable
            if not definition['class_check']():
                error_msg = f"Class not available or not callable"
                if not silent:
                    logger.warning(f"{name}: {error_msg}")
                initialization_stats['errors'].append(f"{name}: {error_msg}")
                initialization_stats['skipped'].append(name)
                continue
            
            # Get the model class
            model_class = definition['class_getter']()
            
            # Validate required configuration parameters
            missing_config = []
            for req_param in definition.get('required_config', []):
                if req_param not in model_config and req_param not in globals():
                    missing_config.append(req_param)
            
            if missing_config and not silent:
                logger.info(f"Missing configuration for {name}: {missing_config} - will use defaults")
            
            # Primary initialization attempt
            test_model = None
            params_used = None
            initialization_method = None
            
            try:
                params_used = definition['primary_params'].copy()
                test_model = model_class(**params_used)
                initialization_method = 'primary_params'
                if not silent:
                    logger.debug(f"[OK] {name}: Initialized with primary parameters")
                
            except Exception as primary_error:
                if not silent:
                    logger.debug(f"{name}: Primary initialization failed: {primary_error}")
                
                # Fallback initialization attempt
                try:
                    params_used = definition['fallback_params'].copy()
                    test_model = model_class(**params_used)
                    initialization_method = 'fallback_params'
                    initialization_stats['fallback_used'].append(name)
                    if not silent:
                        logger.info(f"[OK] {name}: Initialized with fallback parameters")
                    
                except Exception as fallback_error:
                    if not silent:
                        logger.debug(f"{name}: Fallback initialization failed: {fallback_error}")
                    
                    # Minimal initialization attempt (especially for EnhancedAutoencoder)
                    if 'minimal_params' in definition:
                        try:
                            params_used = definition['minimal_params'].copy()
                            test_model = model_class(**params_used)
                            initialization_method = 'minimal_params'
                            initialization_stats['minimal_used'].append(name)
                            initialization_stats['fallback_used'].append(name)
                            if not silent:
                                logger.info(f"[OK] {name}: Initialized with minimal parameters")
                            
                        except Exception as minimal_error:
                            error_msg = f"All initialization attempts failed. Primary: {primary_error}. Fallback: {fallback_error}. Minimal: {minimal_error}"
                            if not silent:
                                logger.error(f"[FAIL] {name}: {error_msg}")
                            initialization_stats['errors'].append(f"{name}: {error_msg}")
                            initialization_stats['failed'].append(name)
                            continue
                    else:
                        error_msg = f"Both primary and fallback initialization failed. Primary: {primary_error}. Fallback: {fallback_error}"
                        if not silent:
                            logger.error(f"[FAIL] {name}: {error_msg}")
                        initialization_stats['errors'].append(f"{name}: {error_msg}")
                        initialization_stats['failed'].append(name)
                        continue
            
            # Comprehensive functionality test
            if test_model is not None:
                try:
                    test_model.eval()
                    
                    # Test forward pass with appropriate batch size
                    batch_size = 4 if (normalization == 'batch' or use_batch_norm) else 1
                    test_input = torch.randn(batch_size, params_used['input_dim'])
                    
                    with torch.no_grad():
                        output = test_model(test_input)
                    
                    # Validate output
                    expected_shape = (batch_size, params_used['input_dim'])
                    if output.shape != expected_shape:
                        raise ValueError(f"Invalid output shape: {output.shape}, expected {expected_shape}")
                    
                    if torch.isnan(output).any():
                        raise ValueError("Output contains NaN values")
                    
                    # Training mode test
                    test_model.train()
                    _ = test_model(test_input)
                    
                    # Test configuration methods if available
                    if hasattr(test_model, 'get_config'):
                        config_dict = test_model.get_config()
                        if not isinstance(config_dict, dict):
                            raise ValueError("get_config() does not return a dictionary")
                    
                    MODEL_VARIANTS[name] = model_class
                    initialization_stats['successful'].append(name)
                    initialization_stats['validation_passed'] += 1
                    
                    if not silent:
                        logger.info(f"[OK] {name}: Successfully initialized and validated ({initialization_method})")
                    
                except Exception as validation_error:
                    error_msg = f"Validation failed: {str(validation_error)}"
                    if not silent:
                        logger.error(f"[FAIL] {name}: {error_msg}")
                    initialization_stats['errors'].append(f"{name}: {error_msg}")
                    initialization_stats['failed'].append(name)
            
            # Cleanup test model
            if test_model is not None:
                del test_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            if not silent:
                logger.error(f"Failed to initialize model variant {name}: {e}")
            initialization_stats['errors'].append(f"{name}: Unexpected error: {str(e)}")
            initialization_stats['failed'].append(name)
    
    # Log comprehensive initialization summary (always log summary for important info)
    total_attempted = initialization_stats['attempted']
    successful_count = len(initialization_stats['successful'])
    failed_count = len(initialization_stats['failed'])
    fallback_count = len(initialization_stats['fallback_used'])
    minimal_count = len(initialization_stats['minimal_used'])
    
    if not silent:
        logger.info("Model variants initialization completed:")
        logger.info(f"  - Total attempted: {total_attempted}")
        logger.info(f"  - Successful: {successful_count}")
        logger.info(f"  - Failed: {failed_count}")
        logger.info(f"  - Using fallback: {fallback_count}")
        logger.info(f"  - Using minimal: {minimal_count}")
        logger.info(f"  - Passed validation: {initialization_stats['validation_passed']}")
        
        if initialization_stats['successful']:
            logger.info(f"  - Available models: {', '.join(initialization_stats['successful'])}")
        
        if initialization_stats['failed']:
            logger.warning(f"  - Failed models: {', '.join(initialization_stats['failed'])}")
        
        if initialization_stats['fallback_used']:
            logger.info(f"  - Fallback used for: {', '.join(initialization_stats['fallback_used'])}")
        
        if initialization_stats['minimal_used']:
            logger.info(f"  - Minimal used for: {', '.join(initialization_stats['minimal_used'])}")
        
        # Log errors for debugging
        if initialization_stats['errors']:
            logger.warning("Initialization errors encountered:")
            for error in initialization_stats['errors'][:5]:
                logger.warning(f"  - {error}")
            if len(initialization_stats['errors']) > 5:
                logger.warning(f"  ... and {len(initialization_stats['errors']) - 5} more errors")
    
    # Ensure at least one model variant is available
    if not MODEL_VARIANTS:
        error_msg = "No model variants could be initialized"
        if not silent:
            logger.error(error_msg)
            logger.error("This indicates a serious configuration or dependency issue")
        
        # Last resort - try to create a minimal working variant
        try:
            if SimpleAutoencoder is not None:
                test_simple = SimpleAutoencoder(input_dim=10, encoding_dim=4, mixed_precision=False)
                test_simple.eval()
                with torch.no_grad():
                    _ = test_simple(torch.randn(1, 10))
                MODEL_VARIANTS['SimpleAutoencoder'] = SimpleAutoencoder
                if not silent:
                    logger.warning("Emergency fallback: Only SimpleAutoencoder available")
            else:
                raise RuntimeError(error_msg)
        except Exception as e:
            if not silent:
                logger.critical(f"Emergency fallback failed: {e}")
            raise RuntimeError(f"{error_msg}: {str(e)}")
    
    # Run post-initialization validation
    try:
        if not silent:
            logger.info("Running post-initialization validation")
        variant_validation_results = validate_model_variants(logger, silent=silent)
        
        available_variants = [
            name for name, status in variant_validation_results.items() 
            if status == 'available'
        ]
        
        if available_variants:
            if not silent:
                logger.info(f"[SUCCESS] Fully validated model variants: {', '.join(available_variants)}")
        else:
            if not silent:
                logger.warning("[WARN] No model variants passed comprehensive validation")
                
    except Exception as validation_error:
        if not silent:
            logger.error(f"Post-initialization validation failed: {validation_error}")
    
    # Force memory cleanup
    try:
        enhanced_clear_memory()
    except:
        pass
    
    if not silent:
        logger.info(f"Model variants initialization completed successfully with {len(MODEL_VARIANTS)} available variants")

# Model architecture comparison
def compare_model_architectures(input_dim: int = None) -> Dict[str, Any]:
    """Compare parameter counts and complexity of different model architectures with enhanced analysis.
    
    Args:
        input_dim: Input dimension for comparison (uses config/default if None)
        
    Returns:
        Dictionary containing detailed comparison results and recommendations
    """
    try:
        logger.debug("Starting comprehensive model architecture comparison")
        
        # Initialize results structure
        results = {
            '_metadata': {
                'comparison_timestamp': datetime.now().isoformat(),
                'comparison_version': '2.1',
                'input_dimension': None,
                'config_source': 'unknown',
                'available_variants': 0,
                'successful_comparisons': 0,
                'hardware_context': {}
            },
            '_summary': {
                'recommendations': [],
                'warnings': [],
                'optimal_choices': {}
            }
        }
        
        # Get current configuration for realistic comparison
        try:
            current_config = get_current_config()
            model_config = current_config.get('model', {})
            data_config = current_config.get('data', {})
            training_config = current_config.get('training', {})
            hardware_config = current_config.get('hardware', {})
            
            results['_metadata']['config_source'] = 'current_config'
            logger.debug("Using current configuration for comparison")
            
        except Exception as e:
            logger.warning(f"Could not load current config, using defaults: {e}")
            model_config = {}
            data_config = {}
            training_config = {}
            hardware_config = {}
            results['_metadata']['config_source'] = 'defaults'
        
        # Determine input dimension with validation
        if input_dim is None:
            input_dim = data_config.get('features', globals().get('FEATURES', 20))
        
        if not isinstance(input_dim, int) or input_dim < 1:
            logger.warning(f"Invalid input_dim {input_dim}, using default 20")
            input_dim = 20
        
        results['_metadata']['input_dimension'] = input_dim
        
        # Ensure MODEL_VARIANTS is initialized
        if not MODEL_VARIANTS:
            logger.info("MODEL_VARIANTS empty, attempting initialization")
            try:
                initialize_model_variants(silent=True)
            except Exception as e:
                error_msg = f"Model initialization failed: {str(e)}"
                logger.error(error_msg)
                results['initialization_error'] = error_msg
                return results
        
        results['_metadata']['available_variants'] = len(MODEL_VARIANTS)
        
        # Extract configuration parameters with comprehensive defaults
        def extract_config_param(config_dict: Dict, key: str, default_val: Any, global_var: str = None):
            """Extract parameter with fallbacks."""
            value = config_dict.get(key)
            if value is not None:
                return value
            
            if global_var and global_var in globals():
                return globals()[global_var]
            
            return default_val
        
        # Model configuration parameters
        encoding_dim = extract_config_param(model_config, 'encoding_dim', 12, 'DEFAULT_ENCODING_DIM')
        hidden_dims = extract_config_param(model_config, 'hidden_dims', [128, 64], 'HIDDEN_LAYER_SIZES')
        dropout_rates = extract_config_param(model_config, 'dropout_rates', [0.2, 0.15], 'DROPOUT_RATES')
        activation = extract_config_param(model_config, 'activation', 'leaky_relu', 'ACTIVATION')
        activation_param = extract_config_param(model_config, 'activation_param', 0.1, 'ACTIVATION_PARAM')
        normalization = extract_config_param(model_config, 'normalization', 'batch', 'NORMALIZATION')
        num_models = extract_config_param(model_config, 'num_models', 3, 'NUM_MODELS')
        diversity_factor = extract_config_param(model_config, 'diversity_factor', 0.1, 'DIVERSITY_FACTOR')
        
        # Training configuration
        batch_size = extract_config_param(training_config, 'batch_size', 64, 'DEFAULT_BATCH_SIZE')
        mixed_precision = extract_config_param(training_config, 'mixed_precision', True, 'MIXED_PRECISION')
        
        # Validate and fix list parameters
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else [128, 64]
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] if isinstance(dropout_rates, (int, float)) else [0.2, 0.15]
        
        # Ensure matching lengths
        if len(hidden_dims) != len(dropout_rates):
            min_length = min(len(hidden_dims), len(dropout_rates))
            if min_length == 0:
                hidden_dims, dropout_rates = [64], [0.2]
            else:
                hidden_dims = hidden_dims[:min_length]
                dropout_rates = dropout_rates[:min_length]
            logger.debug(f"Adjusted list lengths: hidden_dims={hidden_dims}, dropout_rates={dropout_rates}")
        
        # Get hardware context
        try:
            hardware_context = check_hardware()
            results['_metadata']['hardware_context'] = hardware_context
        except Exception:
            hardware_context = {'gpu_available': False, 'cpu_count': os.cpu_count() or 1}
            results['_metadata']['hardware_context'] = hardware_context
        
        # Define test configurations for each model
        test_configurations = {
            'SimpleAutoencoder': {
                'params': {
                    'input_dim': input_dim,
                    'encoding_dim': encoding_dim
                },
                'description': 'Minimal autoencoder for basic reconstruction',
                'use_cases': ['debugging', 'prototyping', 'resource-constrained environments'],
                'complexity_level': 'low'
            },
            
            'EnhancedAutoencoder': {
                'params': {
                    'input_dim': input_dim,
                    'encoding_dim': encoding_dim,
                    'hidden_dims': hidden_dims,
                    'dropout_rates': dropout_rates,
                    'activation': activation,
                    'activation_param': activation_param,
                    'normalization': normalization
                },
                'description': 'Configurable autoencoder with advanced features',
                'use_cases': ['production deployment', 'balanced performance', 'general purpose'],
                'complexity_level': 'medium'
            },
            
            'AutoencoderEnsemble': {
                'params': {
                    'input_dim': input_dim,
                    'num_models': max(1, num_models),
                    'encoding_dim': encoding_dim,
                    'diversity_factor': diversity_factor
                },
                'description': 'Ensemble of multiple autoencoders for maximum accuracy',
                'use_cases': ['critical applications', 'maximum accuracy', 'research'],
                'complexity_level': 'high'
            }
        }
        
        # Perform comprehensive analysis for each model
        for model_name, model_class in MODEL_VARIANTS.items():
            if model_name not in test_configurations:
                logger.warning(f"No test configuration for {model_name}")
                continue
            
            test_config = test_configurations[model_name]
            analysis_start_time = time.time()
            
            try:
                # Initialize model
                model = model_class(**test_config['params'])
                model.eval()
                
                # Basic metrics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
                
                # Performance testing
                performance_metrics = {}
                
                # Forward pass timing
                test_input = torch.randn(batch_size, input_dim)
                model.eval()
                
                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(test_input)
                
                # Actual timing
                times = []
                for _ in range(20):
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(test_input)
                    times.append(time.time() - start_time)
                
                avg_inference_time = sum(times) / len(times)
                inference_fps = batch_size / avg_inference_time
                
                performance_metrics.update({
                    'avg_inference_time_ms': avg_inference_time * 1000,
                    'inference_fps': inference_fps,
                    'throughput_samples_per_second': inference_fps
                })
                
                # Memory analysis
                try:
                    if hardware_context.get('gpu_available') and torch.cuda.is_available():
                        # GPU memory analysis
                        torch.cuda.empty_cache()
                        memory_before = torch.cuda.memory_allocated()
                        
                        model_gpu = model.cuda()
                        test_input_gpu = test_input.cuda()
                        
                        with torch.no_grad():
                            _ = model_gpu(test_input_gpu)
                        
                        memory_after = torch.cuda.memory_allocated()
                        gpu_memory_mb = (memory_after - memory_before) / (1024 * 1024)
                        
                        performance_metrics['gpu_memory_mb'] = gpu_memory_mb
                        
                        # Clean up GPU memory
                        del model_gpu, test_input_gpu
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.debug(f"GPU memory analysis failed for {model_name}: {e}")
                
                # Computational complexity estimation
                flops_estimate = estimate_flops(model, input_dim, batch_size)
                
                # Scaling analysis
                scaling_metrics = analyze_scaling_behavior(
                    model_class, test_config['params'], input_dim
                )
                
                # Training resource estimation
                training_resources = estimate_training_resources(
                    total_params, batch_size, input_dim, hardware_context
                )
                
                # Compile comprehensive results
                results[model_name] = {
                    'architecture': {
                        'total_params': total_params,
                        'trainable_params': trainable_params,
                        'model_size_mb': model_size_mb,
                        'complexity_level': test_config['complexity_level'],
                        'description': test_config['description'],
                        # Subtract 1 for root module
                        'layer_count': len(list(model.modules())) - 1
                    },
                    
                    'performance': performance_metrics,
                    
                    'computational_complexity': {
                        'estimated_flops': flops_estimate,
                        'flops_per_param': flops_estimate / max(total_params, 1),
                        'complexity_class': classify_computational_complexity(flops_estimate)
                    },
                    
                    'scaling': scaling_metrics,
                    
                    'resource_requirements': training_resources,
                    
                    'use_cases': test_config['use_cases'],
                    
                    'configuration_used': test_config['params'],
                    
                    'analysis_metadata': {
                        'analysis_time_seconds': time.time() - analysis_start_time,
                        'test_batch_size': batch_size,
                        'measurement_samples': len(times)
                    },
                    
                    'recommendations': generate_model_recommendations(
                        model_name, total_params, performance_metrics, hardware_context
                    )
                }
                
                results['_metadata']['successful_comparisons'] += 1
                logger.debug(f"Successfully analyzed {model_name}")
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to analyze {model_name}: {error_msg}")
                
                results[model_name] = {
                    'error': error_msg,
                    'configuration_attempted': test_config['params'],
                    'analysis_metadata': {
                        'analysis_time_seconds': time.time() - analysis_start_time,
                        'error_type': type(e).__name__
                    }
                }
            
            finally:
                # Cleanup
                try:
                    if 'model' in locals():
                        del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except:
                    pass
        
        # Generate comparative analysis and recommendations
        results['_summary'] = generate_comparative_summary(results, hardware_context)
        
        logger.info(f"Model architecture comparison completed: {results['_metadata']['successful_comparisons']}/{results['_metadata']['available_variants']} models analyzed")
        
        return results
        
    except Exception as e:
        logger.error(f"Model architecture comparison failed: {str(e)}", exc_info=True)
        return {
            'error': f'Comparison failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def estimate_flops(model: torch.nn.Module, input_dim: int, batch_size: int) -> int:
    """Estimate floating-point operations (FLOPs) for a forward pass.
    
    Args:
        model: PyTorch model to analyze
        input_dim: Input dimension size
        batch_size: Batch size for calculation
        
    Returns:
        Estimated FLOPs for one forward pass
    """
    try:
        total_flops = 0
        layer_details = []
        
        # Get model's state dict to understand architecture
        model.eval()
        
        # Analyze each layer in the model
        for name, module in model.named_modules():
            # Skip root module
            if name == '':
                continue
            
            layer_flops = 0
            layer_info = {'name': name, 'type': type(module).__name__, 'flops': 0}
            
            if isinstance(module, torch.nn.Linear):
                # Linear layer: input_features * output_features * batch_size
                # Plus bias if present: output_features * batch_size
                in_features = module.in_features
                out_features = module.out_features
                
                # Multiply-accumulate operations
                layer_flops = in_features * out_features * batch_size
                
                # Add bias operations
                if module.bias is not None:
                    layer_flops += out_features * batch_size
                
                layer_info.update({
                    'in_features': in_features,
                    'out_features': out_features,
                    'has_bias': module.bias is not None
                })
                
            elif isinstance(module, torch.nn.Conv1d):
                # 1D Convolution
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                
                # Estimate output length (simplified)
                # Simplified assumption
                output_length = input_dim
                
                layer_flops = (kernel_size * in_channels * out_channels * output_length * batch_size)
                
                if module.bias is not None:
                    layer_flops += out_channels * output_length * batch_size
                
            elif isinstance(module, torch.nn.Conv2d):
                # 2D Convolution (if used)
                kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                in_channels = module.in_channels
                out_channels = module.out_channels
                
                # Simplified output size estimation
                # Assume square input
                output_h = output_w = int(input_dim ** 0.5)
                
                layer_flops = (kernel_h * kernel_w * in_channels * out_channels * output_h * output_w * batch_size)
                
                if module.bias is not None:
                    layer_flops += out_channels * output_h * output_w * batch_size
                
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                # Normalization layers
                # Normalization: mean, variance calculation + normalization
                if hasattr(module, 'num_features'):
                    num_features = module.num_features
                else:
                    num_features = input_dim  # Fallback
                
                # Mean and variance calculation: 2 * num_features * batch_size
                # Normalization: 4 * num_features * batch_size (subtract mean, divide by std, scale, shift)
                layer_flops = 6 * num_features * batch_size
                
            elif isinstance(module, torch.nn.Dropout):
                # Dropout: minimal computational cost during inference (0 during eval mode)
                layer_flops = 0
                
            elif isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.GELU)):
                # Activation functions - estimate based on typical hidden layer size
                # This is a rough estimate as we don't know exact tensor sizes
                # Reasonable estimate for neurons in hidden layers
                estimated_neurons = min(512, max(64, input_dim))
                layer_flops = estimated_neurons * batch_size
                
                if isinstance(module, torch.nn.LeakyReLU):
                    # Additional computation for negative slope
                    layer_flops *= 2
                elif isinstance(module, torch.nn.ELU):
                    # Exponential computation
                    layer_flops *= 3
                elif isinstance(module, torch.nn.GELU):
                    # More complex activation
                    layer_flops *= 4
                
            elif isinstance(module, torch.nn.Sigmoid):
                estimated_neurons = min(512, max(64, input_dim))
                # Exponential + division
                layer_flops = estimated_neurons * batch_size * 3
                
            elif isinstance(module, torch.nn.Tanh):
                estimated_neurons = min(512, max(64, input_dim))
                # Hyperbolic function
                layer_flops = estimated_neurons * batch_size * 4
                
            # Add to total
            layer_info['flops'] = layer_flops
            total_flops += layer_flops
            layer_details.append(layer_info)
        
        # If we couldn't estimate from layers, use parameter-based estimation
        if total_flops == 0:
            total_params = sum(p.numel() for p in model.parameters())
            # Rough estimate: 2 FLOPs per parameter (forward pass)
            total_flops = total_params * 2 * batch_size
            
            logger.debug(f"Used parameter-based FLOP estimation: {total_params} params -> {total_flops} FLOPs")
        
        logger.debug(f"FLOP estimation details: {layer_details}")
        logger.debug(f"Total estimated FLOPs: {total_flops}")
        
        return max(total_flops, 1)  # Ensure at least 1 FLOP
        
    except Exception as e:
        logger.warning(f"FLOP estimation failed: {e}")
        # Fallback: use parameter count as rough estimate
        try:
            total_params = sum(p.numel() for p in model.parameters())
            fallback_flops = total_params * batch_size
            logger.debug(f"Fallback FLOP estimation: {fallback_flops}")
            return max(fallback_flops, 1000)
        except:
            # Minimal fallback
            return 10000

def analyze_scaling_behavior(model_class: type, base_params: Dict[str, Any], input_dim: int) -> Dict[str, Any]:
    """Analyze how the model scales with different input dimensions and configurations.
    
    Args:
        model_class: Model class to analyze
        base_params: Base parameters for model instantiation
        input_dim: Base input dimension
        
    Returns:
        Dictionary containing scaling analysis results
    """
    try:
        scaling_results = {
            'input_dimension_scaling': {},
            'parameter_scaling': {},
            'complexity_growth': 'unknown',
            'scaling_efficiency': 'unknown',
            'recommended_limits': {}
        }
        
        # Test different input dimensions
        test_input_dims = [
            max(1, input_dim // 4),
            max(1, input_dim // 2),
            input_dim,
            input_dim * 2,
            input_dim * 4
        ]
        
        input_scaling_data = []
        
        for test_dim in test_input_dims:
            try:
                # Create test parameters
                test_params = base_params.copy()
                test_params['input_dim'] = test_dim
                
                # Handle model-specific parameters
                if 'encoding_dim' in test_params:
                    # Keep encoding dimension proportional but reasonable
                    original_encoding = base_params.get('encoding_dim', 12)
                    test_params['encoding_dim'] = max(2, min(original_encoding, test_dim // 2))
                
                # Create and analyze model
                test_model = model_class(**test_params)
                param_count = sum(p.numel() for p in test_model.parameters())
                
                # Quick performance test
                test_model.eval()
                test_input = torch.randn(1, test_dim)
                
                start_time = time.time()
                with torch.no_grad():
                    _ = test_model(test_input)
                inference_time = time.time() - start_time
                
                input_scaling_data.append({
                    'input_dim': test_dim,
                    'param_count': param_count,
                    'inference_time_ms': inference_time * 1000,
                    'params_per_input_dim': param_count / test_dim
                })
                
                # Cleanup
                del test_model, test_input
                
            except Exception as e:
                logger.debug(f"Scaling test failed for input_dim {test_dim}: {e}")
                continue
        
        scaling_results['input_dimension_scaling'] = input_scaling_data
        
        # Analyze parameter scaling patterns
        if len(input_scaling_data) >= 3:
            # Calculate scaling coefficients
            dims = [d['input_dim'] for d in input_scaling_data]
            params = [d['param_count'] for d in input_scaling_data]
            times = [d['inference_time_ms'] for d in input_scaling_data]
            
            # Determine scaling complexity
            if len(dims) >= 2:
                # Calculate growth rates
                param_growth_rates = []
                time_growth_rates = []
                
                for i in range(1, len(dims)):
                    if dims[i-1] > 0 and params[i-1] > 0:
                        dim_ratio = dims[i] / dims[i-1]
                        param_ratio = params[i] / params[i-1]
                        time_ratio = times[i] / max(times[i-1], 0.001)
                        
                        param_growth_rates.append(param_ratio / dim_ratio)
                        time_growth_rates.append(time_ratio / dim_ratio)
                
                if param_growth_rates:
                    avg_param_growth = sum(param_growth_rates) / len(param_growth_rates)
                    avg_time_growth = sum(time_growth_rates) / len(time_growth_rates)
                    
                    # Classify complexity
                    if avg_param_growth < 1.2:
                        complexity = 'linear'
                    elif avg_param_growth < 2.0:
                        complexity = 'polynomial'
                    else:
                        complexity = 'quadratic'
                    
                    scaling_results['complexity_growth'] = complexity
                    scaling_results['parameter_scaling'] = {
                        'average_growth_rate': avg_param_growth,
                        'time_growth_rate': avg_time_growth,
                        'scaling_type': complexity
                    }
                    
                    # Efficiency assessment
                    if avg_time_growth < 1.5:
                        efficiency = 'excellent'
                    elif avg_time_growth < 2.5:
                        efficiency = 'good'
                    elif avg_time_growth < 4.0:
                        efficiency = 'moderate'
                    else:
                        efficiency = 'poor'
                    
                    scaling_results['scaling_efficiency'] = efficiency
        
        # Generate recommended limits based on scaling behavior
        recommended_limits = {}
        
        if input_scaling_data:
            # Find the point where parameter count becomes excessive
            for data in input_scaling_data:
                param_count = data['param_count']
                input_dim = data['input_dim']
                
                # 1M parameters
                if param_count > 1000000:
                    recommended_limits['max_input_dim_1M_params'] = max(input_dim // 2, base_params.get('input_dim', input_dim))
                    break
            
            # Memory-based recommendations
            # 100K parameters for reasonable memory usage
            max_reasonable_params = 100000
            for data in input_scaling_data:
                if data['param_count'] > max_reasonable_params:
                    recommended_limits['max_input_dim_reasonable'] = max(data['input_dim'] // 2, base_params.get('input_dim', input_dim))
                    break
            
            # Performance-based recommendations
            # 10ms inference time limit
            max_inference_time_ms = 10.0
            for data in input_scaling_data:
                if data['inference_time_ms'] > max_inference_time_ms:
                    recommended_limits['max_input_dim_fast_inference'] = max(data['input_dim'] // 2, base_params.get('input_dim', input_dim))
                    break
        
        scaling_results['recommended_limits'] = recommended_limits
        
        logger.debug(f"Scaling analysis completed: {scaling_results}")
        
        return scaling_results
        
    except Exception as e:
        logger.warning(f"Scaling analysis failed: {e}")
        return {
            'input_dimension_scaling': [],
            'parameter_scaling': {'error': str(e)},
            'complexity_growth': 'unknown',
            'scaling_efficiency': 'unknown',
            'recommended_limits': {}
        }

def estimate_training_resources(param_count: int, batch_size: int, input_dim: int, hardware_context: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate comprehensive training resource requirements.
    
    Args:
        param_count: Number of model parameters
        batch_size: Training batch size
        input_dim: Input dimension size
        hardware_context: Hardware information from check_hardware()
        
    Returns:
        Dictionary containing resource requirement estimates
    """
    try:
        resources = {
            'memory': {},
            'compute': {},
            'time_estimates': {},
            'hardware_recommendations': {},
            'optimization_suggestions': []
        }
        
        # Memory calculations (in bytes)
        bytes_per_float = 4  # float32
        
        # Model parameters memory
        param_memory = param_count * bytes_per_float
        
        # Gradient memory (same size as parameters)
        gradient_memory = param_memory
        
        # Optimizer state (Adam uses 2x parameters for momentum and variance)
        optimizer_memory = param_memory * 2  # Assuming Adam optimizer
        
        # Activation memory (estimated)
        # This is a rough estimate based on typical autoencoder architectures
        estimated_activation_size = input_dim * batch_size * 4  # 4 for different layers
        activation_memory = estimated_activation_size * bytes_per_float
        
        # Data batch memory
        # input + target
        data_memory = input_dim * batch_size * bytes_per_float * 2
        
        # Total training memory
        total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory + data_memory
        
        # Add overhead (PyTorch overhead, fragmentation, etc.)
        overhead_factor = 1.5
        total_memory_with_overhead = total_memory * overhead_factor
        
        resources['memory'] = {
            'parameters_mb': param_memory / (1024 ** 2),
            'gradients_mb': gradient_memory / (1024 ** 2),
            'optimizer_state_mb': optimizer_memory / (1024 ** 2),
            'activations_mb': activation_memory / (1024 ** 2),
            'data_batch_mb': data_memory / (1024 ** 2),
            'total_training_mb': total_memory / (1024 ** 2),
            'total_with_overhead_mb': total_memory_with_overhead / (1024 ** 2),
            # 2x for system overhead
            'recommended_system_memory_gb': (total_memory_with_overhead * 2) / (1024 ** 3)
        }
        
        # Compute requirements
        # Estimate FLOPs per training step (forward + backward)
        # Rough estimate: 2 ops per parameter
        flops_per_forward = param_count * 2
        # Backward pass is typically 2x forward
        flops_per_backward = param_count * 4
        total_flops_per_step = flops_per_forward + flops_per_backward
        
        resources['compute'] = {
            'flops_per_forward': flops_per_forward,
            'flops_per_backward': flops_per_backward,
            'flops_per_training_step': total_flops_per_step,
            # Assume 10k samples per epoch for estimation
            'flops_per_epoch_estimate': total_flops_per_step * (10000 // batch_size),
            'computational_intensity': classify_computational_intensity(total_flops_per_step)
        }
        
        # Time estimates based on hardware
        gpu_available = hardware_context.get('gpu_available', False)
        gpu_memory_gb = hardware_context.get('gpu_memory_gb', 0)
        cpu_count = hardware_context.get('cpu_count', 1)
        
        # Estimate training time per epoch
        if gpu_available and gpu_memory_gb > 0:
            # GPU estimates
            # Assume modern GPU can handle ~1e12 FLOPS/second
            # 1 TFLOPS (conservative estimate)
            gpu_flops_per_second = 1e12
            
            # Memory constraint check
            required_memory_gb = resources['memory']['total_with_overhead_mb'] / 1024
            
            if required_memory_gb <= gpu_memory_gb:
                time_per_epoch_seconds = resources['compute']['flops_per_epoch_estimate'] / gpu_flops_per_second
                hardware_bottleneck = 'compute'
            else:
                # Memory constrained - need to reduce batch size or use CPU
                time_per_epoch_seconds = resources['compute']['flops_per_epoch_estimate'] / (gpu_flops_per_second * 0.5)
                hardware_bottleneck = 'memory'
                resources['optimization_suggestions'].append('reduce_batch_size_for_gpu')
        else:
            # CPU estimates
            # Assume modern CPU can handle ~1e10 FLOPS/second per core
            # Diminishing returns after 8 cores
            cpu_flops_per_second = 1e10 * min(cpu_count, 8)
            time_per_epoch_seconds = resources['compute']['flops_per_epoch_estimate'] / cpu_flops_per_second
            hardware_bottleneck = 'compute'
        
        resources['time_estimates'] = {
            'seconds_per_epoch': time_per_epoch_seconds,
            'minutes_per_epoch': time_per_epoch_seconds / 60,
            'hardware_bottleneck': hardware_bottleneck,
            'estimated_epochs_per_hour': 3600 / max(time_per_epoch_seconds, 1)
        }
        
        # Hardware recommendations
        required_memory_gb = resources['memory']['recommended_system_memory_gb']
        
        if gpu_available:
            if required_memory_gb <= 2:
                gpu_rec = "Entry-level GPU (4GB+ VRAM)"
            elif required_memory_gb <= 8:
                gpu_rec = "Mid-range GPU (8GB+ VRAM)"
            elif required_memory_gb <= 16:
                gpu_rec = "High-end GPU (16GB+ VRAM)"
            else:
                gpu_rec = "Professional GPU or distributed training"
        else:
            gpu_rec = "GPU recommended for reasonable training times"
        
        cpu_rec = f"CPU: {max(4, cpu_count)} cores, {max(8, int(required_memory_gb))}GB RAM"
        
        resources['hardware_recommendations'] = {
            'gpu': gpu_rec,
            'cpu_memory': cpu_rec,
            'minimum_system_memory_gb': max(8, int(required_memory_gb)),
            'recommended_system_memory_gb': max(16, int(required_memory_gb * 2))
        }
        
        # Optimization suggestions
        if param_count > 500000:
            resources['optimization_suggestions'].append('consider_mixed_precision_training')
        
        if batch_size > 256:
            resources['optimization_suggestions'].append('gradient_accumulation_might_help')
        
        # 10 minutes threshold for training time
        if time_per_epoch_seconds > 600:
            resources['optimization_suggestions'].extend([
                'consider_model_pruning',
                'early_stopping_recommended'
            ])
        
        if required_memory_gb > 16:
            resources['optimization_suggestions'].extend([
                'consider_gradient_checkpointing',
                'distributed_training_beneficial'
            ])
        
        logger.debug(f"Training resource estimation completed: {resources}")
        
        return resources
        
    except Exception as e:
        logger.warning(f"Training resource estimation failed: {e}")
        return {
            'memory': {'error': str(e)},
            'compute': {'error': str(e)},
            'time_estimates': {'error': str(e)},
            'hardware_recommendations': {'error': str(e)},
            'optimization_suggestions': ['check_configuration']
        }

def classify_computational_complexity(flops: int) -> str:
    """Classify computational complexity based on FLOP count.
    
    Args:
        flops: Number of floating-point operations
        
    Returns:
        String classification of computational complexity
    """
    try:
        # 1M FLOPs
        if flops < 1e6:
            return 'very_low'
        # 10M FLOPs
        elif flops < 1e7:
            return 'low'
        # 100M FLOPs
        elif flops < 1e8:
            return 'medium'
        # 1B FLOPs
        elif flops < 1e9:
            return 'high'
        # 10B FLOPs
        elif flops < 1e10:
            return 'very_high'
        else:
            return 'extreme'
    except:
        return 'unknown'

def classify_computational_intensity(flops_per_step: int) -> str:
    """Classify computational intensity for training steps.
    
    Args:
        flops_per_step: FLOPs per training step
        
    Returns:
        String classification of computational intensity
    """
    try:
        # 10M FLOPs
        if flops_per_step < 1e7:
            return 'light'
        # 100M FLOPs
        elif flops_per_step < 1e8:
            return 'moderate'
        # 1B FLOPs
        elif flops_per_step < 1e9:
            return 'intensive'
        # 10B FLOPs
        elif flops_per_step < 1e10:
            return 'very_intensive'
        else:
            return 'extreme'
    except:
        return 'unknown'

def generate_model_recommendations(model_name: str, param_count: int, performance_metrics: Dict[str, Any], hardware_context: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations for a model based on its characteristics.
    
    Args:
        model_name: Name of the model
        param_count: Number of parameters
        performance_metrics: Performance measurement results
        hardware_context: Available hardware information
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    try:
        # Parameter-based recommendations
        if param_count < 10000:
            recommendations.extend([
                "Excellent for prototyping and development",
                "Minimal resource requirements",
                "Fast training and inference"
            ])
        elif param_count < 100000:
            recommendations.extend([
                "Good balance of complexity and efficiency",
                "Suitable for most production environments",
                "Reasonable training times"
            ])
        elif param_count < 1000000:
            recommendations.extend([
                "High capacity model for complex tasks",
                "Requires adequate hardware resources",
                "Consider mixed precision training"
            ])
        else:
            recommendations.extend([
                "Large model requiring significant resources",
                "Best suited for specialized hardware",
                "Distributed training may be beneficial"
            ])
        
        # Performance-based recommendations
        inference_fps = performance_metrics.get('inference_fps', 0)
        if inference_fps > 1000:
            recommendations.append("Excellent for real-time applications")
        elif inference_fps > 100:
            recommendations.append("Suitable for interactive applications")
        elif inference_fps < 10:
            recommendations.append("May not be suitable for real-time use")
        
        # Memory-based recommendations
        gpu_memory_mb = performance_metrics.get('gpu_memory_mb', 0)
        if gpu_memory_mb > 0:
            if gpu_memory_mb < 100:
                recommendations.append("Low GPU memory footprint")
            elif gpu_memory_mb < 1000:
                recommendations.append("Moderate GPU memory requirements")
            else:
                recommendations.append("High GPU memory requirements")
        
        # Hardware-specific recommendations
        gpu_available = hardware_context.get('gpu_available', False)
        gpu_memory_gb = hardware_context.get('gpu_memory_gb', 0)
        
        if not gpu_available:
            if param_count < 50000:
                recommendations.append("Suitable for CPU-only environments")
            else:
                recommendations.append("GPU strongly recommended for reasonable performance")
        elif gpu_memory_gb > 0:
            # Using >80% of GPU memory
            if gpu_memory_mb / 1024 > gpu_memory_gb * 0.8:
                recommendations.append("Consider reducing batch size or model complexity")
            # Using <30% of GPU memory
            elif gpu_memory_mb / 1024 < gpu_memory_gb * 0.3:
                recommendations.append("Could increase batch size for better GPU utilization")
        
        # Model-specific recommendations
        if model_name == 'SimpleAutoencoder':
            recommendations.extend([
                "Ideal for baseline comparisons",
                "Good starting point for hyperparameter tuning",
                "Limited capacity for complex patterns"
            ])
        elif model_name == 'EnhancedAutoencoder':
            recommendations.extend([
                "Configurable complexity for different requirements",
                "Production-ready with good performance",
                "Supports advanced training techniques"
            ])
        elif model_name == 'AutoencoderEnsemble':
            recommendations.extend([
                "Maximum accuracy for critical applications",
                "Requires more resources but provides robustness",
                "Consider ensemble size vs. resource trade-offs"
            ])
        
        # Training time recommendations
        avg_inference_time = performance_metrics.get('avg_inference_time_ms', 0)
        # >100ms inference time is considered long
        if avg_inference_time > 100:
            recommendations.append("Long inference time - consider model optimization")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        # Limit to top 10 recommendations
        return unique_recommendations[:10]
        
    except Exception as e:
        logger.debug(f"Error generating recommendations for {model_name}: {e}")
        return [f"Analysis completed for {model_name}", "Review performance metrics for details"]

def generate_comparative_summary(results: Dict[str, Any], hardware_context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive comparative summary and recommendations.
    
    Args:
        results: Complete results dictionary from model comparison
        hardware_context: Hardware context information
        
    Returns:
        Dictionary containing comparative analysis and recommendations
    """
    try:
        summary = {
            'recommendations': [],
            'warnings': [],
            'optimal_choices': {},
            'performance_ranking': {},
            'resource_efficiency': {},
            'use_case_recommendations': {}
        }
        
        # Extract successful model results
        model_results = {}
        for key, value in results.items():
            if not key.startswith('_') and isinstance(value, dict) and 'error' not in value:
                model_results[key] = value
        
        if not model_results:
            summary['warnings'].append("No models were successfully analyzed")
            return summary
        
        # Performance ranking
        performance_metrics = {}
        
        for model_name, data in model_results.items():
            arch = data.get('architecture', {})
            perf = data.get('performance', {})
            resources = data.get('resource_requirements', {})
            
            # Collect metrics for ranking
            performance_metrics[model_name] = {
                'param_count': arch.get('total_params', 0),
                'inference_fps': perf.get('inference_fps', 0),
                'model_size_mb': arch.get('model_size_mb', 0),
                'memory_requirement_mb': resources.get('memory', {}).get('total_with_overhead_mb', 0)
            }
        
        # Rank by different criteria
        rankings = {}
        
        # Speed ranking (highest FPS first)
        speed_ranking = sorted(
            performance_metrics.items(),
            key=lambda x: x[1]['inference_fps'],
            reverse=True
        )
        rankings['speed'] = [name for name, _ in speed_ranking]
        
        # Efficiency ranking (highest FPS per parameter)
        efficiency_ranking = sorted(
            performance_metrics.items(),
            key=lambda x: x[1]['inference_fps'] / max(x[1]['param_count'], 1),
            reverse=True
        )
        rankings['efficiency'] = [name for name, _ in efficiency_ranking]
        
        # Memory efficiency ranking (lowest memory usage)
        memory_ranking = sorted(
            performance_metrics.items(),
            key=lambda x: x[1]['memory_requirement_mb']
        )
        rankings['memory_efficiency'] = [name for name, _ in memory_ranking]
        
        # Size ranking (smallest model first)
        size_ranking = sorted(
            performance_metrics.items(),
            key=lambda x: x[1]['param_count']
        )
        rankings['size'] = [name for name, _ in size_ranking]
        
        summary['performance_ranking'] = rankings
        
        # Optimal choices for different scenarios
        optimal_choices = {}
        
        if speed_ranking:
            optimal_choices['fastest_inference'] = speed_ranking[0][0]
        
        if efficiency_ranking:
            optimal_choices['most_efficient'] = efficiency_ranking[0][0]
        
        if memory_ranking:
            optimal_choices['lowest_memory'] = memory_ranking[0][0]
        
        if size_ranking:
            optimal_choices['smallest_model'] = size_ranking[0][0]
        
        # Balanced recommendation (consider multiple factors)
        balanced_scores = {}
        for model_name, metrics in performance_metrics.items():
            # Normalize metrics (0-1 scale)
            max_fps = max(m['inference_fps'] for m in performance_metrics.values())
            max_params = max(m['param_count'] for m in performance_metrics.values())
            max_memory = max(m['memory_requirement_mb'] for m in performance_metrics.values())
            
            if max_fps > 0 and max_params > 0 and max_memory > 0:
                speed_score = metrics['inference_fps'] / max_fps
                # Smaller is better
                size_score = 1.0 - (metrics['param_count'] / max_params)
                # Less is better
                memory_score = 1.0 - (metrics['memory_requirement_mb'] / max_memory)
                
                # Weighted combination
                balanced_scores[model_name] = (speed_score * 0.4 + size_score * 0.3 + memory_score * 0.3)
        
        if balanced_scores:
            best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
            optimal_choices['best_balanced'] = best_balanced[0]
        
        summary['optimal_choices'] = optimal_choices
        
        # Use case recommendations
        use_case_recs = {
            'prototyping_development': [],
            'production_deployment': [],
            'resource_constrained': [],
            'high_performance': [],
            'research_experimentation': []
        }
        
        for model_name, data in model_results.items():
            arch = data.get('architecture', {})
            param_count = arch.get('total_params', 0)
            complexity = arch.get('complexity_level', 'unknown')
            use_cases = data.get('use_cases', [])
            
            # Categorize by parameter count and complexity
            if param_count < 50000 or complexity == 'low':
                use_case_recs['prototyping_development'].append(model_name)
                use_case_recs['resource_constrained'].append(model_name)
            
            if 10000 < param_count < 500000 or complexity == 'medium':
                use_case_recs['production_deployment'].append(model_name)
            
            if param_count > 100000 or complexity in ['high', 'very_high']:
                use_case_recs['high_performance'].append(model_name)
                use_case_recs['research_experimentation'].append(model_name)
        
        summary['use_case_recommendations'] = use_case_recs
        
        # Generate overall recommendations
        recommendations = []
        
        # Hardware-based recommendations
        gpu_available = hardware_context.get('gpu_available', False)
        gpu_memory_gb = hardware_context.get('gpu_memory_gb', 0)
        
        if not gpu_available:
            recommendations.append("GPU acceleration not available - consider smallest models for reasonable performance")
            if use_case_recs['resource_constrained']:
                recommendations.append(f"For CPU-only: Recommended models: {', '.join(use_case_recs['resource_constrained'][:2])}")
        elif gpu_memory_gb < 4:
            recommendations.append("Limited GPU memory - avoid largest models or reduce batch sizes")
        elif gpu_memory_gb >= 8:
            recommendations.append("Adequate GPU memory - all models should run efficiently")
        
        # Performance-based recommendations
        if speed_ranking:
            fastest_model = speed_ranking[0][0]
            fastest_fps = speed_ranking[0][1]['inference_fps']
            recommendations.append(f"Fastest inference: {fastest_model} ({fastest_fps:.1f} samples/sec)")
        
        if optimal_choices.get('best_balanced'):
            recommendations.append(f"Best overall balance: {optimal_choices['best_balanced']}")
        
        # Resource efficiency recommendations
        if efficiency_ranking:
            most_efficient = efficiency_ranking[0][0]
            recommendations.append(f"Most parameter-efficient: {most_efficient}")
        
        summary['recommendations'] = recommendations
        
        # Generate warnings
        warnings = []
        
        # Check for potential issues
        for model_name, data in model_results.items():
            resources = data.get('resource_requirements', {})
            memory_req = resources.get('memory', {}).get('recommended_system_memory_gb', 0)
            
            if memory_req > 32:
                warnings.append(f"{model_name} requires >32GB RAM - ensure adequate system memory")
            
            time_estimates = resources.get('time_estimates', {})
            minutes_per_epoch = time_estimates.get('minutes_per_epoch', 0)
            
            # >1 hour per epoch
            if minutes_per_epoch > 60:
                warnings.append(f"{model_name} estimated >1 hour per training epoch")
        
        # Hardware warnings
        if not gpu_available and any(metrics['param_count'] > 100000 for metrics in performance_metrics.values()):
            warnings.append("Large models detected but no GPU available - training will be very slow")
        
        summary['warnings'] = warnings
        
        # Resource efficiency summary
        efficiency_summary = {}
        for model_name, metrics in performance_metrics.items():
            if metrics['param_count'] > 0:
                fps_per_param = metrics['inference_fps'] / metrics['param_count']
                fps_per_mb = metrics['inference_fps'] / max(metrics['memory_requirement_mb'], 1)
                
                efficiency_summary[model_name] = {
                    'fps_per_parameter': fps_per_param,
                    'fps_per_mb_memory': fps_per_mb,
                    'efficiency_class': classify_efficiency(fps_per_param, fps_per_mb)
                }
        
        summary['resource_efficiency'] = efficiency_summary
        
        logger.debug(f"Comparative summary generated: {len(recommendations)} recommendations, {len(warnings)} warnings")
        
        return summary
        
    except Exception as e:
        logger.warning(f"Error generating comparative summary: {e}")
        return {
            'recommendations': ["Comparison completed with errors - review individual model results"],
            'warnings': [f"Summary generation failed: {str(e)}"],
            'optimal_choices': {},
            'performance_ranking': {},
            'resource_efficiency': {},
            'use_case_recommendations': {}
        }

def classify_efficiency(fps_per_param: float, fps_per_mb: float) -> str:
    """Classify model efficiency based on FPS per parameter and FPS per MB.
    
    Args:
        fps_per_param: Frames per second per model parameter
        fps_per_mb: Frames per second per MB of memory
        
    Returns:
        String classification of efficiency
    """
    try:
        # Normalize and combine metrics
        # Scale and cap at 10
        param_score = min(fps_per_param * 1000, 10)
        # Scale and cap at 10
        memory_score = min(fps_per_mb / 10, 10)
        
        combined_score = (param_score + memory_score) / 2
        
        if combined_score >= 7:
            return 'excellent'
        elif combined_score >= 5:
            return 'good'
        elif combined_score >= 3:
            return 'moderate'
        elif combined_score >= 1:
            return 'poor'
        else:
            return 'very_poor'
    except:
        return 'unknown'

# Display model comparison
def display_model_comparison() -> None:
    """Display model architecture comparison in a comprehensive formatted report with enhanced error handling and rich formatting."""
    try:
        logger.info("Generating comprehensive model architecture comparison display")
        
        # Get comparison results with enhanced error handling
        try:
            results = compare_model_architectures()
        except Exception as e:
            logger.error(f"Failed to generate model comparison: {e}")
            console.print(f"[bold red]COMPARISON FAILED: {str(e)}[/bold red]")
            console.print("\n[yellow]Troubleshooting Steps:[/yellow]")
            console.print("1. Run 'initialize_model_variants()' to refresh model registry")
            console.print("2. Check configuration with 'get_current_config()'")
            console.print("3. Validate models with 'validate_model_variants(logger)'")
            return
        
        # Handle critical errors
        if isinstance(results, dict) and 'error' in results:
            error_msg = results['error']
            console.print(f"[bold red]ANALYSIS ERROR: {error_msg}[/bold red]")
            
            # Provide specific guidance based on error type
            if 'initialization_failed' in error_msg.lower():
                console.print("\n[yellow]Model Initialization Issues Detected:[/yellow]")
                console.print("1. Check if model classes are properly imported")
                console.print("2. Verify PyTorch installation")
                console.print("3. Try: initialize_model_variants()")
            elif 'comparison failed' in error_msg.lower():
                console.print("\n[yellow]Comparison Engine Issues:[/yellow]")
                console.print("1. Check system resources (memory, CPU)")
                console.print("2. Verify configuration parameters")
                console.print("3. Try with default configuration")
            
            return
        
        if 'initialization_error' in results:
            console.print(f"[bold red]INITIALIZATION ERROR: {results['initialization_error']}[/bold red]")
            console.print("\n[yellow]This usually indicates:[/yellow]")
            console.print("1. Model classes are not properly defined")
            console.print("2. Configuration parameters are invalid")
            console.print("3. System dependencies are missing")
            return
        
        # Extract metadata and summary with enhanced validation
        metadata = results.get('_metadata', {})
        summary = results.get('_summary', {})
        
        # Validate metadata
        if not metadata:
            logger.warning("No metadata found in comparison results")
            metadata = {
                'comparison_timestamp': datetime.now().isoformat(),
                'available_variants': 0,
                'successful_comparisons': 0,
                'hardware_context': {},
                'input_dimension': 'unknown',
                'config_source': 'unknown'
            }
        
        # Filter model results (exclude metadata)
        model_results = {}
        for key, value in results.items():
            if not key.startswith('_') and isinstance(value, dict):
                model_results[key] = value
        
        if not model_results:
            console.print("[bold yellow]NO MODEL RESULTS AVAILABLE[/bold yellow]")
            console.print("\nThis could indicate:")
            console.print("1. No model variants are initialized")
            console.print("2. All model analyses failed")
            console.print("3. Configuration issues prevent analysis")
            console.print("\nTry running: [cyan]initialize_model_variants()[/cyan]")
            return
        
        # === MAIN HEADER SECTION ===
        header_panel = Panel.fit(
            f"[bold bright_white]MODEL ARCHITECTURE COMPARISON REPORT[/bold bright_white]\n"
            f"Generated: {metadata.get('comparison_timestamp', 'Unknown')[:19]}\n"
            f"Input Dimension: {metadata.get('input_dimension', 'Unknown')} | "
            f"Config Source: {metadata.get('config_source', 'Unknown')}\n"
            f"Models Available: {metadata.get('available_variants', 0)} | "
            f"Successfully Analyzed: {metadata.get('successful_comparisons', 0)}",
            title="[bold yellow]ANALYSIS OVERVIEW[/bold yellow]",
            border_style="bright_yellow",
            title_align="left",
            padding=(1, 2)
        )
        console.print(header_panel)
        
        # === MAIN COMPARISON TABLE ===
        main_table = Table(
            title="\n[bold bright_yellow]PERFORMANCE & RESOURCE COMPARISON[/bold bright_yellow]",
            box=box.ROUNDED,
            header_style="bold bright_cyan",
            border_style="bright_white",
            title_style="bold green",
            title_justify="left",
            show_lines=True,
            expand=True,
            width=min(140, console.width - 2)
        )
        
        # Configure main table columns with optimal widths
        main_table.add_column("Model", style="bold cyan", width=18, no_wrap=True)
        main_table.add_column("Parameters", width=12, justify="left")
        main_table.add_column("Size (MB)", width=10, justify="left")
        main_table.add_column("Complexity", width=12, justify="left")
        main_table.add_column("Inference", width=12, justify="left")
        main_table.add_column("Throughput", width=12, justify="left")
        main_table.add_column("Memory", width=12, justify="left")
        main_table.add_column("Status", width=12, justify="center")
        
        # Prepare and sort model data
        model_data_list = []
        successful_models = []
        failed_models = []
        
        for model_name, model_data in model_results.items():
            if 'error' in model_data:
                failed_models.append((model_name, model_data))
            else:
                param_count = model_data.get('architecture', {}).get('total_params', 0)
                model_data_list.append((param_count, model_name, model_data))
                successful_models.append((model_name, model_data))
        
        # Sort by parameter count for logical ordering
        model_data_list.sort(key=lambda x: x[0])
        
        # Add successful models to main table
        for param_count, model_name, model_data in model_data_list:
            arch = model_data.get('architecture', {})
            perf = model_data.get('performance', {})
            resources = model_data.get('resource_requirements', {})
            
            # Determine status and styling based on performance
            avg_inference_ms = perf.get('avg_inference_time_ms', float('inf'))
            inference_fps = perf.get('inference_fps', 0)
            
            if avg_inference_ms < 50 and inference_fps > 100:
                status_text = "EXCELLENT"
                status_style = "bold bright_green"
            elif avg_inference_ms < 100 and inference_fps > 50:
                status_text = "GOOD"
                status_style = "bold green"
            elif avg_inference_ms < 500:
                status_text = "ACCEPTABLE"
                status_style = "bold yellow"
            else:
                status_text = "SLOW"
                status_style = "bold red"
            
            # Format memory requirement
            memory_mb = resources.get('memory', {}).get('total_with_overhead_mb', 0)
            if memory_mb < 100:
                memory_text = f"{memory_mb:.0f}MB"
                memory_style = "green"
            elif memory_mb < 1000:
                memory_text = f"{memory_mb:.0f}MB"
                memory_style = "yellow"
            else:
                memory_text = f"{memory_mb/1024:.1f}GB"
                memory_style = "red"
            
            main_table.add_row(
                Text(model_name, style="bold"),
                f"{param_count:,}",
                f"{arch.get('model_size_mb', 0):.1f}",
                arch.get('complexity_level', 'Unknown').title(),
                f"{avg_inference_ms:.1f}ms",
                f"{inference_fps:.0f}/s",
                Text(memory_text, style=memory_style),
                Text(status_text, style=status_style)
            )
        
        # Add failed models to table
        for model_name, model_data in failed_models:
            error_msg = model_data.get('error', 'Unknown error')
            error_display = error_msg[:25] + "..." if len(error_msg) > 25 else error_msg
            
            main_table.add_row(
                Text(model_name, style="dim"),
                Text("--", style="dim"),
                Text("--", style="dim"),
                Text("ERROR", style="bold red"),
                Text("--", style="dim"),
                Text("--", style="dim"),
                Text("--", style="dim"),
                Text("FAILED", style="bold red")
            )
        
        console.print(main_table)
        
        # === DETAILED ARCHITECTURE ANALYSIS ===
        if successful_models:
            detail_table = Table(
                title="[bold bright_yellow]DETAILED ARCHITECTURE BREAKDOWN[/bold bright_yellow]",
                box=box.ROUNDED,
                header_style="bold bright_cyan",
                border_style="cyan",
                title_style="bold magenta",
                title_justify="left",
                show_lines=True,
                expand=True,
                width=min(150, console.width - 2)
            )
            
            detail_table.add_column("Model", style="bold cyan", width=15)
            detail_table.add_column("Description", width=30, justify="left", no_wrap=True)
            detail_table.add_column("Layers", width=7, justify="left")
            detail_table.add_column("Trainable", width=12, justify="left")
            detail_table.add_column("FLOPs", width=6, justify="left")
            detail_table.add_column("Complexity", width=8, justify="left")
            detail_table.add_column("Use Cases", width=45, justify="left", no_wrap=True)
            
            for model_name, model_data in successful_models:
                arch = model_data.get('architecture', {})
                comp = model_data.get('computational_complexity', {})
                use_cases = model_data.get('use_cases', [])
                
                # Format use cases
                use_cases_text = ', '.join(use_cases[:2]) + ('...' if len(use_cases) > 2 else '')
                
                # Format FLOPs
                flops = comp.get('estimated_flops', 0)
                if flops > 1e9:
                    flops_text = f"{flops/1e9:.1f}G"
                elif flops > 1e6:
                    flops_text = f"{flops/1e6:.1f}M"
                elif flops > 1e3:
                    flops_text = f"{flops/1e3:.1f}K"
                else:
                    flops_text = str(flops)
                
                detail_table.add_row(
                    model_name,
                    arch.get('description', 'No description')[:35],
                    str(arch.get('layer_count', 0)),
                    f"{arch.get('trainable_params', 0):,}",
                    flops_text,
                    comp.get('complexity_class', 'Unknown'),
                    use_cases_text
                )
            
            console.print(detail_table)
        
        # === HARDWARE CONTEXT & SYSTEM INFO ===
        hardware_ctx = metadata.get('hardware_context', {})
        gpu_available = hardware_ctx.get('gpu_available', False)
        gpu_memory_gb = hardware_ctx.get('gpu_memory_gb', 0)
        cpu_count = hardware_ctx.get('cpu_count', os.cpu_count() or 1)
        
        # Create hardware status text with color coding
        if gpu_available:
            gpu_status = f"[bold green][OK] GPU Available[/bold green] ({gpu_memory_gb}GB VRAM)" if gpu_memory_gb > 0 else "[bold green][OK] GPU Available[/bold green]"
        else:
            gpu_status = "[bold yellow][WARN] CPU Only[/bold yellow]"
        
        if cpu_count >= 8:
            cpu_status = f"[bold green][OK] {cpu_count} CPU Cores[/bold green] (Excellent)"
        elif cpu_count >= 4:
            cpu_status = f"[bold yellow][WARN] {cpu_count} CPU Cores[/bold yellow] (Good)"
        else:
            cpu_status = f"[bold red][WARN] {cpu_count} CPU Cores[/bold red] (Limited)"
        
        hardware_text = f"{gpu_status}\n{cpu_status}"
        
        hardware_panel = Panel.fit(
            hardware_text,
            title="[bold]SYSTEM CAPABILITIES[/bold]",
            border_style="bright_blue",
            title_align="left"
        )
        console.print(hardware_panel)
        
        # === RECOMMENDATIONS & OPTIMAL CHOICES ===
        if summary:
            # Overall recommendations
            overall_recs = summary.get('recommendations', [])
            if overall_recs:
                # Limit to top 8
                rec_text = "\n".join([f"[bright_green][+][/bright_green] {rec}" for rec in overall_recs[:8]])
                
                rec_panel = Panel.fit(
                    rec_text,
                    title="[bold bright_yellow]RECOMMENDATIONS[/bold bright_yellow]",
                    border_style="bright_green",
                    title_align="left",
                    padding=(1, 2)
                )
                console.print(rec_panel)
            
            # Optimal choices table
            optimal = summary.get('optimal_choices', {})
            if optimal:
                opt_table = Table(
                    title="[bold bright_yellow]OPTIMAL MODEL SELECTION[/bold bright_yellow]",
                    box=box.ROUNDED,
                    header_style="bold bright_white",
                    title_style="bold bright_green",
                    title_justify="left",
                    border_style="green",
                    show_lines=True
                )
                
                opt_table.add_column("Scenario", style="bright_cyan", width=25)
                opt_table.add_column("Recommended Model", style="bright_green", width=20)
                opt_table.add_column("Rationale", width=40)
                
                # Enhanced rationale for each choice
                choice_rationales = {
                    'fastest_inference': 'Minimizes latency for real-time applications',
                    'most_efficient': 'Best performance per parameter ratio',
                    'lowest_memory': 'Suitable for memory-constrained environments',
                    'smallest_model': 'Minimal resource footprint',
                    'best_balanced': 'Optimal trade-off across all metrics'
                }
                
                for scenario, choice in optimal.items():
                    scenario_display = scenario.replace('_', ' ').title()
                    rationale = choice_rationales.get(scenario, 'Best for this specific use case')
                    
                    opt_table.add_row(
                        scenario_display,
                        Text(choice, style="bold"),
                        rationale
                    )
                
                console.print(opt_table)
            
            # Performance rankings
            rankings = summary.get('performance_ranking', {})
            if rankings:
                rank_table = Table(
                    title="[bold bright_yellow]PERFORMANCE RANKINGS[/bold bright_yellow]",
                    box=box.ROUNDED,
                    header_style="bold bright_white",
                    title_style="bold bright_blue",
                    title_justify="left",
                    border_style="blue",
                    show_lines=True
                )
                
                rank_table.add_column("Metric", style="bright_cyan", width=20)
                rank_table.add_column("1st Place", style="bold green", width=18)
                rank_table.add_column("2nd Place", style="yellow", width=18)
                rank_table.add_column("3rd Place", style="dim white", width=18)
                
                ranking_labels = {
                    'speed': 'Fastest Inference',
                    'efficiency': 'Most Efficient',
                    'memory_efficiency': 'Memory Efficient',
                    'size': 'Smallest Size'
                }
                
                for metric, models in rankings.items():
                    if models and metric in ranking_labels:
                        first = models[0] if len(models) > 0 else "N/A"
                        second = models[1] if len(models) > 1 else "N/A"
                        third = models[2] if len(models) > 2 else "N/A"
                        
                        rank_table.add_row(
                            ranking_labels[metric],
                            first,
                            second,
                            third
                        )
                
                console.print(rank_table)
        
        # === WARNINGS & ISSUES ===
        warnings = summary.get('warnings', []) if summary else []
        if warnings or failed_models:
            warn_items = []
            
            # Add summary warnings
            for warning in warnings[:5]:  # Limit warnings
                warn_items.append(f"[yellow][WARN][/yellow] {warning}")
            
            # Add failed model warnings
            for model_name, model_data in failed_models:
                error_msg = model_data.get('error', 'Unknown error')
                warn_items.append(f"[red][FAIL][/red] {model_name}: {error_msg[:50]}{'...' if len(error_msg) > 50 else ''}")
            
            if warn_items:
                warn_text = "\n".join(warn_items)
                warn_panel = Panel.fit(
                    warn_text,
                    title="[bold yellow]WARNINGS & ISSUES[/bold yellow]",
                    border_style="yellow",
                    title_align="left",
                    padding=(1, 2)
                )
                console.print(warn_panel)
        
        # === USE CASE RECOMMENDATIONS ===
        use_case_recs = summary.get('use_case_recommendations', {}) if summary else {}
        if use_case_recs:
            use_case_table = Table(
                title="[bold bright_yellow]USE CASE RECOMMENDATIONS[/bold bright_yellow]",
                box=box.ROUNDED,
                header_style="bold bright_white",
                title_style="bold bright_magenta",
                title_justify="left",
                border_style="magenta",
                show_lines=True
            )
            
            use_case_table.add_column("Use Case", style="bright_magenta", width=25)
            use_case_table.add_column("Recommended Models", width=50)
            
            use_case_labels = {
                'prototyping_development': 'Prototyping & Development',
                'production_deployment': 'Production Deployment',
                'resource_constrained': 'Resource-Constrained',
                'high_performance': 'High Performance',
                'research_experimentation': 'Research & Experimentation'
            }
            
            for use_case, models in use_case_recs.items():
                if models and use_case in use_case_labels:
                    # Limit to top 3
                    models_text = ", ".join(models[:3])
                    if len(models) > 3:
                        models_text += f" (+{len(models) - 3} more)"
                    
                    use_case_table.add_row(
                        use_case_labels[use_case],
                        models_text
                    )
            
            console.print(use_case_table)
        
        # === CONFIGURATION GUIDANCE ===
        config_guidance = []
        
        # Hardware-specific guidance
        if gpu_available:
            if gpu_memory_gb >= 8:
                config_guidance.append("[green][OK][/green] GPU with adequate memory - All models supported")
                config_guidance.append("[green][OK][/green] Enable mixed precision training for better performance")
                config_guidance.append("[green][OK][/green] Consider AutoencoderEnsemble for maximum accuracy")
            elif gpu_memory_gb >= 4:
                config_guidance.append("[yellow][WARN][/yellow] Limited GPU memory - Avoid largest batch sizes")
                config_guidance.append("[yellow][WARN][/yellow] EnhancedAutoencoder recommended")
            else:
                config_guidance.append("[yellow][WARN][/yellow] Very limited GPU memory - Use small models")
        else:
            config_guidance.append("[yellow][WARN][/yellow] CPU-only training - Use SimpleAutoencoder")
            config_guidance.append("[yellow][WARN][/yellow] Reduce batch_size to 16-32 for CPU efficiency")
            config_guidance.append("[yellow][WARN][/yellow] Set num_workers to CPU core count")
        
        # General configuration tips
        config_guidance.extend([
            "[cyan][i][/cyan] Use presets: 'lightweight', 'performance', 'accuracy'",
            "[cyan][i][/cyan] Monitor training with tensorboard logging",
            "[cyan][i][/cyan] Enable early stopping to prevent overfitting"
        ])
        
        config_text = "\n".join(config_guidance)
        config_panel = Panel.fit(
            config_text,
            title="[bold bright_yellow]CONFIGURATION GUIDANCE[/bold bright_yellow]",
            border_style="bright_cyan",
            title_align="left",
            padding=(1, 2)
        )
        console.print(config_panel)
        
        # === TROUBLESHOOTING SECTION ===
        troubleshoot_text = (
            "[bold]Common Commands:[/bold]\n"
            "[cyan]initialize_model_variants()[/cyan] - Refresh model registry\n"
            "[cyan]validate_model_variants(logger)[/cyan] - Test model functionality\n"
            "[cyan]get_current_config()[/cyan] - View current configuration\n"
            "[cyan]load_preset('performance')[/cyan] - Load optimized preset\n\n"
            "[bold]Performance Optimization:[/bold]\n"
            "- Adjust batch_size based on available memory\n"
            "- Use mixed precision for GPU acceleration\n"
            "- Enable gradient clipping for training stability"
        )
        
        troubleshoot_panel = Panel.fit(
            troubleshoot_text,
            title="[bold bright_yellow]TROUBLESHOOTING & OPTIMIZATION[/bold bright_yellow]",
            border_style="bright_cyan",
            title_align="left",
            padding=(1, 2)
        )
        console.print(troubleshoot_panel)
        
        # === FOOTER WITH STATISTICS ===
        total_models = len(model_results)
        success_count = len(successful_models)
        fail_count = len(failed_models)
        success_rate = (success_count / total_models * 100) if total_models > 0 else 0
        
        footer_text = (
            f"Analysis completed successfully\n"
            f"Models analyzed: {total_models} | Successful: {success_count} | Failed: {fail_count} | Success rate: {success_rate:.1f}% | Hardware: {'GPU available' if gpu_available else 'CPU only'}\n"
            f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        footer_panel = Panel.fit(
            footer_text,
            title="[bold yellow]ANALYSIS SUMMARY[/bold yellow]",
            border_style="dim green",
            title_align="left",
            padding=(0, 2)
        )
        console.print(footer_panel)
        
        # Enhanced logging with detailed metrics
        logger.debug(f"Model comparison display completed successfully:")
        logger.debug(f"  - Total models: {total_models}")
        logger.debug(f"  - Successful analyses: {success_count}")
        logger.debug(f"  - Failed analyses: {fail_count}")
        logger.debug(f"  - Success rate: {success_rate:.1f}%")
        logger.debug(f"  - Hardware: {'GPU available' if gpu_available else 'CPU only'}")
        
        # Log individual model performance for debugging
        for model_name, model_data in successful_models:
            perf = model_data.get('performance', {})
            arch = model_data.get('architecture', {})
            
            logger.debug(
                f"Model {model_name}: "
                f"{arch.get('total_params', 0):,} params, "
                f"{perf.get('avg_inference_time_ms', 0):.2f}ms inference, "
                f"{perf.get('inference_fps', 0):.1f} FPS"
            )
        
        # Log errors for failed models
        for model_name, model_data in failed_models:
            error_msg = model_data.get('error', 'Unknown error')
            logger.error(f"Model {model_name} analysis failed: {error_msg}")
    
    except Exception as e:
        error_msg = f"Critical failure in display_model_comparison: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        
        # User-friendly error display
        console.print(f"[bold red]DISPLAY ERROR: {error_msg}[/bold red]")
        console.print("\n[bold yellow]Recovery Actions:[/bold yellow]")
        console.print("1. [cyan]initialize_model_variants()[/cyan] - Reinitialize models")
        console.print("2. [cyan]check_hardware()[/cyan] - Verify system capabilities")
        console.print("3. [cyan]get_current_config()[/cyan] - Check configuration")
        console.print("4. Contact support if issue persists")
        
        # Attempt to provide basic system information
        try:
            console.print(f"\n[dim]System Info: Python {platform.python_version()}, "
                         f"PyTorch {torch.__version__ if 'torch' in globals() else 'Not Available'}, "
                         f"CPU cores: {os.cpu_count()}[/dim]")
        except:
            pass

# System initialization validation and setup
def save_initialization_report(system_status: Dict[str, Any], report_dir: Path) -> None:
    """
    Save a comprehensive initialization report to disk with multiple formats.
    
    This function creates both machine-readable (JSON) and human-readable (TXT)
    reports that match the comprehensive system_status structure from initialize_system.
    
    Args:
        system_status: Complete system status dictionary from initialize_system
        report_dir: Directory to save the report (should be a Path object)
        
    Raises:
        Exception: If report saving fails (logged but not re-raised to avoid
                  interrupting initialization)
    """
    try:
        # Determine report_dir (default: script's directory / reports)
        if report_dir is None:
            report_dir = Path(__file__).resolve().parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure report_dir is a Path object
        if not isinstance(report_dir, Path):
            report_dir = Path(__file__).resolve().parent / "reports"
        
        # Create timestamp for report files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON Report
        json_report_path = report_dir / f"deep_init_report_{timestamp}.json"
        
        # Create a serializable version of the report
        serializable_status = {}
        for key, value in system_status.items():
            try:
                # Test if the value is JSON serializable
                json.dumps(value, default=str)
                serializable_status[key] = value
            except (TypeError, ValueError):
                # Convert problematic values to strings
                serializable_status[key] = str(value)
        
        # Add metadata to the JSON report
        serializable_status['_metadata'] = {
            'report_version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'report_type': 'system_initialization',
            'format': 'json',
            'generator': 'initialize_system'
        }
        
        # Save the JSON report
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_status, f, indent=2, default=str, ensure_ascii=False)
        
        # Save Human-Readable Summary
        summary_path = report_dir / f"deep_init_summary_{timestamp}.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("SYSTEM INITIALIZATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Report Version: 2.0\n")
            f.write("=" * 80 + "\n\n")
            
            # Initialization Status
            init_info = system_status.get('initialization', {})
            f.write("INITIALIZATION STATUS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Status: {init_info.get('status', 'unknown').upper()}\n")
            f.write(f"Duration: {init_info.get('duration_seconds', 0):.2f} seconds\n")
            f.write(f"Method: {init_info.get('method', 'unknown')}\n")
            f.write(f"Start Time: {init_info.get('start_time', 'unknown')}\n")
            f.write(f"End Time: {init_info.get('end_time', 'unknown')}\n")
            
            if 'error' in init_info:
                f.write(f"Error: {init_info['error']}\n")
                f.write(f"Error Type: {init_info.get('error_type', 'unknown')}\n")
            f.write("\n")
            
            # System Information
            sys_info = system_status.get('system', {})
            f.write("SYSTEM ENVIRONMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Platform: {sys_info.get('platform', 'unknown')}\n")
            f.write(f"Python Version: {sys_info.get('python_version', 'unknown')}\n")
            f.write(f"PyTorch Version: {sys_info.get('pytorch_version', 'unknown')}\n")
            f.write(f"CUDA Available: {sys_info.get('cuda_available', False)}\n")
            f.write(f"CUDA Devices: {sys_info.get('cuda_device_count', 0)}\n")
            f.write(f"Working Directory: {sys_info.get('working_directory', 'unknown')}\n")
            f.write(f"Log Directory: {sys_info.get('log_directory', 'unknown')}\n")
            f.write(f"Model Directory: {sys_info.get('model_directory', 'unknown')}\n")
            f.write(f"Config Directory: {sys_info.get('config_directory', 'unknown')}\n")
            f.write("\n")
            
            # Configuration Information
            config_info = system_status.get('config', {})
            f.write("CONFIGURATION\n")
            f.write("-" * 15 + "\n")
            f.write(f"Preset Name: {config_info.get('preset_name', 'custom')}\n")
            f.write(f"Validation Status: {config_info.get('validation_status', 'unknown')}\n")
            f.write(f"Config File: {config_info.get('config_file', 'unknown')}\n")
            available_presets = config_info.get('available_presets', [])
            f.write(f"Available Presets: {', '.join(available_presets) if available_presets else 'none'}\n")
            
            # Write key configuration parameters if available
            active_config = config_info.get('active_config', {})
            if isinstance(active_config, dict) and active_config:
                f.write("Key Configuration Parameters:\n")
                for key, value in active_config.items():
                    # Skip private keys
                    if not key.startswith('_'):
                        if isinstance(value, dict):
                            f.write(f"  {key}: {len(value)} items\n")
                        elif isinstance(value, (list, tuple)):
                            f.write(f"  {key}: [{len(value)} items]\n")
                        else:
                            # Truncate very long values
                            str_value = str(value)
                            if len(str_value) > 50:
                                str_value = str_value[:47] + "..."
                            f.write(f"  {key}: {str_value}\n")
            f.write("\n")
            
            # Hardware Information
            hw_info = system_status.get('hardware', {})
            f.write("HARDWARE RESOURCES\n")
            f.write("-" * 20 + "\n")
            f.write(f"CPU Cores: {hw_info.get('cpu_count', 'unknown')}\n")
            f.write(f"System Memory: {hw_info.get('memory_gb', 0):.1f} GB\n")
            f.write(f"Available Disk Space: {hw_info.get('disk_space_gb', 0):.1f} GB\n")
            f.write(f"CUDA Available: {hw_info.get('cuda_available', False)}\n")
            
            cuda_devices = hw_info.get('cuda_devices', [])
            if cuda_devices:
                f.write(f"CUDA Devices ({len(cuda_devices)}):\n")
                for device in cuda_devices:
                    f.write(f"  Device {device.get('id', '?')}: {device.get('name', 'unknown')}\n")
                    f.write(f"    Memory: {device.get('memory_gb', 0):.1f} GB\n")
            else:
                f.write("CUDA Devices: None\n")
            f.write("\n")
            
            # Model Information
            model_info = system_status.get('models', {})
            f.write("MODEL VARIANTS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Available Variants: {model_info.get('variants_available', 0)}\n")
            
            variant_names = model_info.get('variant_names', [])
            if variant_names:
                f.write(f"Variant Names: {', '.join(variant_names)}\n")
            else:
                f.write("Variant Names: None\n")
            
            # Detailed variant status
            variant_status = model_info.get('variant_status', {})
            if variant_status:
                f.write("Variant Status Details:\n")
                for name, status in variant_status.items():
                    status_indicator = "OK" if status == 'available' else "MISSING"
                    f.write(f"  {status_indicator} {name}: {status}\n")
            f.write("\n")
            
            # Performance Metrics
            performance = system_status.get('performance', {})
            f.write("PERFORMANCE BASELINE\n")
            f.write("-" * 22 + "\n")
            
            if performance:
                if 'baseline_failed' in performance:
                    f.write(f"Baseline establishment failed: {performance['baseline_failed']}\n")
                else:
                    f.write("Performance Metrics:\n")
                    for metric, value in performance.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")
                        else:
                            f.write(f"  {metric}: {value}\n")
            else:
                f.write("No performance metrics available\n")
            f.write("\n")
            
            # Dependencies Information
            deps_info = system_status.get('dependencies', {})
            f.write("DEPENDENCIES\n")
            f.write("-" * 14 + "\n")
            f.write(f"PyTorch Version: {deps_info.get('torch_version', 'unknown')}\n")
            f.write(f"Python Version: {deps_info.get('python_version', 'unknown')}\n")
            f.write(f"Platform: {deps_info.get('platform', 'unknown')}\n")
            
            optional_deps = deps_info.get('optional_available', {})
            if optional_deps:
                f.write("Optional Dependencies:\n")
                for name, available in optional_deps.items():
                    status_indicator = "OK" if available else "MISSING"
                    f.write(f"  {status_indicator} {name}: {'Available' if available else 'Not Available'}\n")
            f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
        
        # Save Compact Status File (for quick status checks)
        status_path = report_dir / f"deep_init_status_{timestamp}.json"
        compact_status = {
            'status': system_status.get('initialization', {}).get('status', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': system_status.get('initialization', {}).get('duration_seconds', 0),
            'cuda_available': system_status.get('system', {}).get('cuda_available', False),
            'model_variants': system_status.get('models', {}).get('variants_available', 0),
            'config_preset': system_status.get('config', {}).get('preset_name', 'unknown'),
            'errors': system_status.get('initialization', {}).get('error') is not None
        }
        
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(compact_status, f, indent=2, default=str)
        
        # Log successful report generation
        logger.info(f"Initialization reports saved successfully:")
        logger.info(f"  - Full report: {json_report_path}")
        logger.info(f"  - Summary: {summary_path}")
        logger.info(f"  - Status: {status_path}")
        
        # Create latest symlinks (if supported)
        try:
            # Create "latest" symlinks for easy access
            latest_json = report_dir / "deep_latest_init_report.json"
            latest_summary = report_dir / "deep_latest_init_summary.txt"
            latest_status = report_dir / "deep_latest_init_status.json"
            
            # Remove existing symlinks if they exist
            for latest_file in [latest_json, latest_summary, latest_status]:
                if latest_file.exists() or latest_file.is_symlink():
                    latest_file.unlink()
            
            # Create new symlinks
            latest_json.symlink_to(json_report_path.name)
            latest_summary.symlink_to(summary_path.name)
            latest_status.symlink_to(status_path.name)
            
            logger.debug("Latest report symlinks created successfully")
            
        except (OSError, NotImplementedError):
            # Symlinks not supported on this system (e.g., Windows without admin)
            logger.debug("Symlinks not supported, skipping latest report links")
        
        # Create report index
        try:
            index_path = report_dir / "deep_initialization_reports.txt"
            report_entry = f"{timestamp}: {compact_status['status']} ({compact_status['duration_seconds']:.2f}s)\n"
            
            # Append to index file
            with open(index_path, 'a', encoding='utf-8') as f:
                f.write(report_entry)
            
            logger.debug(f"Report entry added to index: {index_path}")
            
        except Exception as index_error:
            logger.warning(f"Failed to update report index: {index_error}")
        
    except Exception as e:
        # Log the error but don't raise it - we don't want report saving
        # to interrupt the initialization process
        error_msg = f"Failed to save initialization report: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Report save error details:", exc_info=True)
        
        # Try to save a minimal error report
        try:
            error_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            error_path = report_dir / f"deep_report_error_{error_timestamp}.txt"
            
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"INITIALIZATION REPORT SAVE FAILED\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Error Type: {type(e).__name__}\n")
                f.write(f"Original system_status keys: {list(system_status.keys()) if isinstance(system_status, dict) else 'Not a dict'}\n")
            
            logger.info(f"Error report saved to: {error_path}")
            
        except Exception as fallback_error:
            logger.critical(f"Failed to save even minimal error report: {fallback_error}")

def display_configuration_changes(changes: List[Dict], console: Console = None, logger: logging.Logger = None):
    """
    Display configuration changes in a rich table format.
    
    Args:
        changes: List of configuration change dictionaries
        console: Rich console instance (creates new if None)
        logger: Logger for summary only
    """
    if not changes:
        return
    
    if console is None:
        console = Console()
    
    # Create configuration changes table
    config_table = Table(
        title=f"\n[bold]CONFIGURATION CHANGES APPLIED[/bold]",
        box=box.ROUNDED,
        header_style="bold bright_white",
        border_style="bright_blue",
        title_style="bold blue",
        title_justify="left",
        show_lines=True,
        expand=True,
        width=min(100, console.width - 4)
    )
    
    config_table.add_column("Section", style="bold cyan", width=12)
    config_table.add_column("Parameter", style="bold yellow", width=20)
    config_table.add_column("From", style="dim red", width=15, justify="center")
    config_table.add_column("To", style="dim green", width=15, justify="center")
    config_table.add_column("Type", style="dim", width=8, justify="center")
    
    # Group changes by section
    sections = {}
    for change in changes:
        section = change.get('section', 'UNKNOWN')
        if section not in sections:
            sections[section] = []
        sections[section].append(change)
    
    # Add rows grouped by section
    for section_name, section_changes in sections.items():
        # Add section header
        config_table.add_row(
            Text(section_name, style="bold white on blue"),
            "",
            "",
            "",
            "",
            style="bold white on blue"
        )
        
        # Add changes for this section
        for change in section_changes:
            # Format values for display
            old_val = str(change.get('old_value', 'N/A'))
            new_val = str(change.get('new_value', 'N/A'))
            
            # Truncate long values
            if len(old_val) > 12:
                old_val = old_val[:9] + "..."
            if len(new_val) > 12:
                new_val = new_val[:9] + "..."
            
            config_table.add_row(
                "",
                change.get('parameter', 'unknown'),
                old_val,
                new_val,
                change.get('source', 'auto')
            )
    
    # Add summary row
    config_table.add_row(
        Text("TOTAL CHANGES", style="bold bright_white on black"),
        Text(f"{len(changes)} parameters", style="bold white"),
        "",
        "",
        "",
        style="bold bright_white on black"
    )
    
    console.print(config_table)
    
    # Log only summary
    if logger:
        section_counts = {}
        for change in changes:
            section = change.get('section', 'UNKNOWN')
            section_counts[section] = section_counts.get(section, 0) + 1
        
        summary = ", ".join([f"{section}: {count}" for section, count in section_counts.items()])
        logger.info(f"Applied {len(changes)} configuration changes ({summary})")

def initialize_system() -> Dict[str, Any]:
    """
    Initialize the complete system with comprehensive setup and validation.
    
    This function performs a complete system initialization by leveraging the
    existing System Check Framework and loading_screen functionality.
    
    Returns:
        Dict containing comprehensive system status and configuration
    
    Raises:
        RuntimeError: If critical system components fail to initialize
        SystemExit: If user chooses to quit during initialization
    """
    initialization_start = time.time()
    console = Console()
    
    # Create initialization status table
    init_table = Table(
        title=f"\n[bold]SYSTEM INITIALIZATION REPORT[/bold]",
        box=box.ROUNDED,
        header_style="bold bright_white",
        border_style="bright_white",
        title_style="bold green",
        title_justify="left",
        show_lines=True,
        expand=True,
        width=min(120, console.width - 4)
    )
    
    init_table.add_column("Step", style="bold cyan", width=25)
    init_table.add_column("Status", width=12, justify="center")
    init_table.add_column("Time", width=10, justify="center")
    init_table.add_column("Details", style="dim", min_width=50)
    
    initialization_steps = []
    
    def add_step(step_name: str, status: str, duration: float, details: str):
        """Add a step to the initialization tracking."""
        status_style = "bold green" if status == "SUCCESS" else "bold red" if status == "FAILED" else "bold yellow"
        initialization_steps.append((step_name, status, duration, details, status_style))
    
    try:
        # Step 1: Early setup - Basic configuration and logging
        step_start = time.time()
        configure_system()
        set_seed(42)
        
        # Setup logging first
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(log_dir)
        
        # Setup reports directory
        report_dir = Path(__file__).resolve().parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        step_duration = time.time() - step_start
        add_step(
            "Early Setup", 
            "SUCCESS", 
            step_duration,
            f"Configured system, set seed (42), created directories\nLog dir: {log_dir}\nReports dir: {report_dir}"
        )
        
        # Minimal startup logging
        logger.info("SYSTEM INITIALIZATION STARTING")
        
        # Step 2: Run comprehensive system checks with interactive loading screen
        step_start = time.time()
        sys.excepthook = enhanced_global_exception_handler
        global get_memory_usage, enhanced_clear_memory, enhanced_monitor_performance, get_detailed_memory_usage, establish_performance_baseline
        
        get_memory_usage = get_detailed_memory_usage
        enhanced_clear_memory = enhanced_clear_memory
        enhanced_monitor_performance = enhanced_monitor_performance
        get_detailed_memory_usage = get_detailed_memory_usage
        establish_performance_baseline = establish_performance_baseline
        
        # Use loading_screen with extended checks to perform all system validation
        system_ready = loading_screen(
            logger=logger,
            extended=True,
            include_performance=True
        )
        
        step_duration = time.time() - step_start
        if not system_ready:
            add_step(
                "System Validation", 
                "FAILED", 
                step_duration,
                "System checks failed or user cancelled initialization"
            )
            raise RuntimeError("System checks failed or user cancelled initialization")
        
        add_step(
            "System Validation", 
            "SUCCESS", 
            step_duration,
            "Extended system checks completed successfully\nAll critical components validated"
        )
        
        # Step 3: Configuration system initialization with improved display
        step_start = time.time()
        config_details = []
        changes_applied = []
        
        try:
            # Temporarily suppress verbose configuration logging
            original_level = logger.level
            logger.setLevel(logging.WARNING)
            
            # Capture configuration changes during initialization
            config = initialize_config()
            config_details.append(f"Configuration loaded from: {CONFIG_FILE.name}")
            
            # Validate configuration
            try:
                validate_config(config)
                config_details.append("Configuration validation: PASSED")
            except ValueError as e:
                config_details.append(f"Configuration validation: FAILED - {str(e)}")
                
                # Handle configuration validation failure
                if sys.stdin.isatty():
                    console.print(Panel.fit(
                        f"[bold yellow]Configuration validation failed:[/bold yellow]\n"
                        f"[white]{str(e)}[/white]\n\n"
                        f"[dim]Would you like to use default configuration instead?[/dim]",
                        border_style="yellow",
                        title="Configuration Issue",
                        padding=(1, 2)
                    ))
                    
                    if prompt_user("Use default configuration?", default=True):
                        config = get_default_config()
                        save_config(config)
                        config_details.append("Applied default configuration")
                    else:
                        raise ValueError("Configuration validation failed and user declined defaults")
                else:
                    # Non-interactive mode - use defaults
                    config = get_default_config()
                    save_config(config)
                    config_details.append("Applied default configuration (non-interactive)")
            
            # Apply configuration globally and capture changes
            previous_config = get_current_config() if hasattr(sys.modules[__name__], 'CURRENT_CONFIG') else {}
            update_global_config(config)
            
            # Restore logging level
            logger.setLevel(original_level)
            
            # Get configuration changes (if your config system tracks them)
            if hasattr(config, '_changes_applied'):
                changes_applied = config._changes_applied
            
            preset_name = config.get('_preset_name', 'custom')
            config_details.append(f"Active preset: {preset_name}")
            
            # Display configuration changes in table format if any were made
            if changes_applied:
                display_configuration_changes(changes_applied, console, logger)
            
        except Exception as e:
            # Restore logging level in case of error
            if 'original_level' in locals():
                logger.setLevel(original_level)
            
            config = get_default_config()
            update_global_config(config)
            config_details.append(f"Fallback to default config due to error: {str(e)}")
        
        step_duration = time.time() - step_start
        add_step(
            "Configuration System", 
            "SUCCESS", 
            step_duration,
            "\n".join(config_details)
        )
        
        # Step 4: Model variants initialization (with suppressed verbose logging)
        step_start = time.time()
        
        try:
            # Temporarily suppress INFO level logging for model initialization
            original_level = logger.level
            logger.setLevel(logging.WARNING)
            
            initialize_model_variants(silent=True)
            
            if not MODEL_VARIANTS:
                raise RuntimeError("No model variants could be initialized")
            
            variant_status = validate_model_variants(logger, silent=True)
            available_variants = [name for name, status in variant_status.items() if status == 'available']
            
            # Restore logging level
            logger.setLevel(original_level)
            
            if not available_variants:
                raise RuntimeError("No working model variants available")
            
            step_duration = time.time() - step_start
            add_step(
                "Model Variants", 
                "SUCCESS", 
                step_duration,
                f"Initialized {len(available_variants)}/{len(MODEL_VARIANTS)} variants\n" +
                f"Available: {', '.join(available_variants)}"
            )
            
        except Exception as e:
            # Restore logging level in case of error
            if 'original_level' in locals():
                logger.setLevel(original_level)
            
            step_duration = time.time() - step_start
            add_step(
                "Model Variants", 
                "FAILED", 
                step_duration,
                f"Model initialization failed: {str(e)}"
            )
            raise RuntimeError(f"Model initialization failed: {e}")
        
        # Step 5: Performance baseline establishment (with suppressed verbose logging)
        step_start = time.time()
        performance_metrics = {}
        
        try:
            # Temporarily suppress INFO level logging for baseline establishment
            original_level = logger.level
            logger.setLevel(logging.WARNING)
            
            performance_metrics = establish_performance_baseline()
            
            # Restore logging level
            logger.setLevel(original_level)
            
            baseline_details = []
            for metric, value in performance_metrics.items():
                baseline_details.append(f"{metric}: {value:.4f}")
            
            step_duration = time.time() - step_start
            add_step(
                "Performance Baseline", 
                "SUCCESS", 
                step_duration,
                "Performance baseline established\n" + "\n".join(baseline_details)
            )
            
        except Exception as e:
            # Restore logging level in case of error
            if 'original_level' in locals():
                logger.setLevel(original_level)
            
            step_duration = time.time() - step_start
            performance_metrics = {'baseline_failed': str(e)}
            add_step(
                "Performance Baseline", 
                "WARNING", 
                step_duration,
                f"Baseline establishment failed: {str(e)}\nContinuing with defaults"
            )
        
        # Display initialization table
        for step_name, status, duration, details, status_style in initialization_steps:
            init_table.add_row(
                Text(step_name, style="bold cyan"),
                Text(status, style=status_style),
                Text(f"{duration:.2f}s", style="dim"),
                Text(details, style="dim")
            )
        
        # Step 6: Finalize initialization and create system status
        initialization_time = time.time() - initialization_start
        
        # Add summary row
        init_table.add_row(
            Text("TOTAL INITIALIZATION", style="bold bright_white on black"),
            Text("SUCCESS", style="bold green"),
            Text(f"{initialization_time:.2f}s", style="bold white"),
            Text(f"All systems operational and ready", style="bright_white")
        )
        
        # Print the initialization table
        console.print(init_table)
        
        # Create comprehensive system status report (unchanged)
        system_status = {
            'initialization': {
                'start_time': datetime.now() - timedelta(seconds=initialization_time),
                'end_time': datetime.now(),
                'duration_seconds': initialization_time,
                'status': 'success',
                'method': 'system_check_framework'
            },
            'system': {
                'platform': platform.platform(),
                'python_version': sys.version.split()[0],
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'working_directory': str(Path.cwd()),
                'log_directory': str(log_dir),
                'model_directory': str(DEFAULT_MODEL_DIR),
                'config_directory': str(CONFIG_DIR),
                'report_directory': str(report_dir)
            },
            'config': {
                'active_config': config,
                'config_file': str(CONFIG_FILE),
                'preset_name': config.get('_preset_name', 'custom'),
                'available_presets': list(PRESET_CONFIGS.keys()),
                'validation_status': 'passed'
            },
            'hardware': {
                'cpu_count': os.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'disk_space_gb': shutil.disk_usage('.').free / (1024**3),
                'cuda_available': torch.cuda.is_available(),
                'cuda_devices': [
                    {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    } for i in range(torch.cuda.device_count())
                ] if torch.cuda.is_available() else []
            },
            'models': {
                'variants_available': len(MODEL_VARIANTS),
                'variant_names': list(MODEL_VARIANTS.keys()),
                'variant_status': variant_status
            },
            'performance': performance_metrics,
            'dependencies': {
                'torch_version': torch.__version__,
                'python_version': sys.version_info[:3],
                'platform': platform.system(),
                'optional_available': {
                    name: available for name, available in OPTIONAL_DEPENDENCIES.items()
                }
            }
        }
        
        # Save initialization report
        try:
            save_initialization_report(system_status, report_dir)
            logger.info(f"Initialization report saved to {report_dir}")
        except Exception as e:
            logger.warning(f"Failed to save initialization report: {e}")
        
        # Minimal logging summary
        logger.info("SYSTEM INITIALIZATION COMPLETED SUCCESSFULLY")
        logger.info(f"Time: {initialization_time:.2f}s | Config: {config.get('_preset_name', 'custom')} | Models: {len(MODEL_VARIANTS)} | CUDA: {torch.cuda.is_available()}")
        
        return system_status, config, logger
        
    except KeyboardInterrupt:
        # Add failed initialization step to table before displaying
        for step_name, status, duration, details, status_style in initialization_steps:
            init_table.add_row(
                Text(step_name, style="bold cyan"),
                Text(status, style=status_style),
                Text(f"{duration:.2f}s", style="dim"),
                Text(details, style="dim")
            )
        
        init_table.add_row(
            Text("INITIALIZATION", style="bold white on red"),
            Text("INTERRUPTED", style="bold white on red"),
            Text(f"{time.time() - initialization_start:.2f}s", style="bold white"),
            Text("System initialization was cancelled by user (Ctrl+C)", style="bright_white")
        )
        
        console.print(init_table)
        logger.warning("System initialization interrupted by user (Ctrl+C)")
        sys.exit(0)
        
    except SystemExit:
        # User chose to quit during loading_screen
        logger.info("System initialization cancelled by user choice")
        raise
        
    except Exception as e:
        initialization_time = time.time() - initialization_start
        
        # Add failed steps to table
        for step_name, status, duration, details, status_style in initialization_steps:
            init_table.add_row(
                Text(step_name, style="bold cyan"),
                Text(status, style=status_style),
                Text(f"{duration:.2f}s", style="dim"),
                Text(details, style="dim")
            )
        
        init_table.add_row(
            Text("INITIALIZATION", style="bold white on red"),
            Text("FAILED", style="bold white on red"),
            Text(f"{initialization_time:.2f}s", style="bold white"),
            Text(f"Error: {str(e)}\nType: {type(e).__name__}", style="bright_white")
        )
        
        console.print(init_table)
        
        # Create error status report (unchanged)
        error_status = {
            'initialization': {
                'start_time': datetime.now() - timedelta(seconds=initialization_time),
                'end_time': datetime.now(),
                'duration_seconds': initialization_time,
                'status': 'critical_failure',
                'error': str(e),
                'error_type': type(e).__name__
            },
            'system': {
                'platform': platform.platform(),
                'python_version': sys.version.split()[0]
            }
        }
        
        logger.critical("SYSTEM INITIALIZATION FAILED")
        logger.critical(f"Error: {str(e)} | Type: {type(e).__name__} | Time: {initialization_time:.2f}s")
        logger.exception("Detailed error information:")
        
        # Save error report
        try:
            error_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            error_report_path = report_dir / f"deep_init_failure_{error_timestamp}.json"
            with open(error_report_path, 'w') as f:
                json.dump(error_status, f, indent=2, default=str)
            logger.error(f"Error report saved to {error_report_path}")
        except Exception as save_error:
            logger.error(f"Failed to save error report: {save_error}")
        
        raise RuntimeError(f"System initialization failed: {e}") from e

class SimpleAutoencoder(nn.Module):
    """Simple autoencoder with enhanced initialization, mixed precision support, and CPU/GPU awareness.
    
    This class provides comprehensive parameter validation, configuration loading, and robust
    error handling for basic autoencoder functionality.
    
    Args:
        input_dim: Dimension of input features (required)
        encoding_dim: Size of latent representation (default: from config or DEFAULT_ENCODING_DIM)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU)
        min_features: Minimum allowed input dimension (default: from config or MIN_FEATURES)
        config: Optional configuration dictionary to override defaults
        **kwargs: Additional parameters for compatibility (ignored with warning)
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: Optional[int] = None,
        mixed_precision: Optional[bool] = None,
        min_features: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs  # Accept additional parameters for compatibility
    ):
        super().__init__()
        
        # Warn about unused parameters for debugging
        if kwargs:
            logger.debug(f"SimpleAutoencoder ignoring unused parameters: {list(kwargs.keys())}")
        
        # Load configuration with comprehensive fallbacks
        if config is None:
            try:
                current_config = get_current_config()
                config = current_config.get('model', {}) if isinstance(current_config, dict) else {}
            except Exception as e:
                logger.debug(f"Could not load configuration, using defaults: {e}")
                config = {}
        
        # Apply configuration with parameter precedence and validation
        try:
            self.encoding_dim = encoding_dim if encoding_dim is not None else config.get('encoding_dim', globals().get('DEFAULT_ENCODING_DIM', 32))
            self.min_features = min_features if min_features is not None else config.get('min_features', globals().get('MIN_FEATURES', 10))
        except Exception as e:
            logger.warning(f"Error loading config parameters, using fallbacks: {e}")
            self.encoding_dim = encoding_dim if encoding_dim is not None else 32
            self.min_features = min_features if min_features is not None else 10
        
        # Input validation with detailed error messages
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim} (type: {type(input_dim)})")
        
        if input_dim < self.min_features:
            raise ValueError(f"Input dimension ({input_dim}) must be at least {self.min_features}")
        
        if not isinstance(self.encoding_dim, int) or self.encoding_dim <= 0:
            raise ValueError(f"encoding_dim must be a positive integer, got {self.encoding_dim}")
        
        if self.encoding_dim >= input_dim:
            logger.warning(f"encoding_dim ({self.encoding_dim}) >= input_dim ({input_dim}), "
                          f"this may not provide meaningful compression")
        
        # Mixed precision handling with configuration awareness
        try:
            mixed_precision_config = config.get('mixed_precision', globals().get('MIXED_PRECISION', True))
            self._mixed_precision_requested = mixed_precision if mixed_precision is not None else mixed_precision_config
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        except Exception as e:
            logger.debug(f"Error configuring mixed precision, using defaults: {e}")
            self._mixed_precision_requested = mixed_precision if mixed_precision is not None else True
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        
        # Store input dimension for config export and validation
        self.input_dim = input_dim
        
        # Architecture definition with error handling
        try:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, self.encoding_dim),
                nn.ReLU(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.encoding_dim, input_dim),
                nn.Sigmoid()
            )
        except Exception as e:
            logger.error(f"Failed to create network layers: {e}")
            raise RuntimeError(f"Failed to initialize SimpleAutoencoder architecture: {e}")
        
        # Initialize weights
        try:
            self._initialize_weights()
        except Exception as e:
            logger.warning(f"Weight initialization failed: {e}")
        
        # Log successful initialization
        logger.debug(f"SimpleAutoencoder initialized successfully: "
                    f"input_dim={input_dim}, encoding_dim={self.encoding_dim}, "
                    f"mixed_precision={self.mixed_precision} (requested={self._mixed_precision_requested})")
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization with proper scaling."""
        try:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        except Exception as e:
            logger.error(f"Weight initialization error: {e}")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic mixed precision support and input validation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of same shape as input
            
        Raises:
            ValueError: If input tensor has wrong shape or type
            RuntimeError: If forward pass fails
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor input, got {type(x)}")
        
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (batch_size, input_dim), got shape {x.shape}")
        
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Input feature dimension {x.size(-1)} doesn't match expected {self.input_dim}")
        
        try:
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(f"SimpleAutoencoder forward pass error: {e}")
    
    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the comprehensive configuration of the autoencoder."""
        return {
            "model_type": "SimpleAutoencoder",
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "mixed_precision": self.mixed_precision,
            "mixed_precision_requested": self._mixed_precision_requested,
            "min_features": self.min_features,
            "architecture": "simple",
            "initialized_with_cuda": torch.cuda.is_available(),
            "config_version": "2.1",
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update model settings from configuration (limited to non-architectural changes)."""
        try:
            model_config = config.get('model', {})
            
            # Only update non-architectural parameters
            if 'mixed_precision' in model_config:
                self._mixed_precision_requested = model_config['mixed_precision']
                self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
                logger.info(f"Updated mixed_precision to {self.mixed_precision}")
        except Exception as e:
            logger.error(f"Config update failed: {e}")

class EnhancedAutoencoder(nn.Module):
    """Enhanced autoencoder with configurable architecture, mixed precision support, and advanced features.
    
    This class provides comprehensive configuration options, robust error handling, and
    advanced features like skip connections, normalization, and flexible activations.
    
    Args:
        input_dim: Dimension of input features (required)
        encoding_dim: Size of latent representation (default: from config)
        hidden_dims: List of hidden layer dimensions (default: from config)
        dropout_rates: Dropout rates for each layer (default: from config)
        activation: Activation function ('relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid')
        activation_param: Parameter for activation (e.g., slope for LeakyReLU)
        normalization: Normalization type ('batch', 'layer', None)
        legacy_mode: Use simple architecture if True
        skip_connection: Enable skip connection if True
        min_features: Minimum allowed input dimension
        mixed_precision: Enable mixed precision training (auto-disabled for CPU)
        config: Optional configuration dictionary to override defaults
        **kwargs: Additional parameters for compatibility
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        activation: Optional[str] = None,
        activation_param: Optional[float] = None,
        normalization: Optional[str] = None,
        legacy_mode: bool = False,
        skip_connection: Optional[bool] = None,
        min_features: Optional[int] = None,
        mixed_precision: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs  # Accept additional parameters for compatibility
    ):
        super().__init__()
        
        # Warn about unused parameters for debugging
        if kwargs:
            logger.debug(f"EnhancedAutoencoder ignoring unused parameters: {list(kwargs.keys())}")
        
        # Load configuration with comprehensive fallbacks
        if config is None:
            try:
                current_config = get_current_config()
                config = current_config.get('model', {}) if isinstance(current_config, dict) else {}
            except Exception as e:
                logger.debug(f"Could not load configuration, using defaults: {e}")
                config = {}
        
        # Apply configuration with parameter precedence and robust fallbacks
        try:
            self.encoding_dim = encoding_dim if encoding_dim is not None else config.get('encoding_dim', globals().get('DEFAULT_ENCODING_DIM', 32))
            self.hidden_dims = hidden_dims if hidden_dims is not None else config.get('hidden_dims', globals().get('HIDDEN_LAYER_SIZES', [128, 64]).copy())
            self.dropout_rates = dropout_rates if dropout_rates is not None else config.get('dropout_rates', globals().get('DROPOUT_RATES', [0.2, 0.15]).copy())
            self.activation = activation if activation is not None else config.get('activation', globals().get('ACTIVATION', 'relu'))
            self.activation_param = activation_param if activation_param is not None else config.get('activation_param', globals().get('ACTIVATION_PARAM', 0.2))
            self.normalization = normalization if normalization is not None else config.get('normalization', globals().get('NORMALIZATION', None))
            self.skip_connection = skip_connection if skip_connection is not None else config.get('skip_connection', True)
            self.min_features = min_features if min_features is not None else config.get('min_features', globals().get('MIN_FEATURES', 10))
        except Exception as e:
            logger.warning(f"Error loading config parameters, using hardcoded defaults: {e}")
            self.encoding_dim = encoding_dim if encoding_dim is not None else 32
            self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
            self.dropout_rates = dropout_rates if dropout_rates is not None else [0.2, 0.15]
            self.activation = activation if activation is not None else 'relu'
            self.activation_param = activation_param if activation_param is not None else 0.2
            self.normalization = normalization if normalization is not None else None
            self.skip_connection = skip_connection if skip_connection is not None else True
            self.min_features = min_features if min_features is not None else 10
        
        self.legacy_mode = legacy_mode
        self.input_dim = input_dim
        
        # Input validation with detailed error messages
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim} (type: {type(input_dim)})")
        
        if input_dim < self.min_features:
            raise ValueError(f"Input dimension ({input_dim}) must be at least {self.min_features}")
        
        # Validate and fix configuration parameters
        self._validate_and_fix_config()
        
        # Mixed precision handling with configuration awareness
        try:
            mixed_precision_config = config.get('mixed_precision', globals().get('MIXED_PRECISION', True))
            self._mixed_precision_requested = mixed_precision if mixed_precision is not None else mixed_precision_config
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        except Exception as e:
            logger.debug(f"Error configuring mixed precision, using defaults: {e}")
            self._mixed_precision_requested = mixed_precision if mixed_precision is not None else True
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        
        # Log configuration before building architecture
        logger.debug(f"EnhancedAutoencoder configuration: input_dim={input_dim}, "
                    f"encoding_dim={self.encoding_dim}, hidden_dims={self.hidden_dims}, "
                    f"dropout_rates={self.dropout_rates}, activation={self.activation}, "
                    f"normalization={self.normalization}, skip_connection={self.skip_connection}, "
                    f"mixed_precision={self.mixed_precision}, legacy_mode={legacy_mode}")

        # Build architecture with error handling
        try:
            self._build_architecture()
        except Exception as e:
            logger.error(f"Failed to build EnhancedAutoencoder architecture: {e}")
            raise RuntimeError(f"Architecture initialization failed: {e}")
        
        # Initialize weights
        try:
            self._initialize_weights()
        except Exception as e:
            logger.warning(f"Weight initialization failed: {e}")
        
        logger.debug("EnhancedAutoencoder initialized successfully")
    
    def _validate_and_fix_config(self) -> None:
        """Validate and fix configuration parameters with comprehensive error handling."""
        # Validate activation
        valid_activations = ['relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid']
        if self.activation not in valid_activations:
            logger.warning(f"Unknown activation '{self.activation}', defaulting to 'relu'. "
                          f"Valid options: {valid_activations}")
            self.activation = 'relu'
        
        # Validate normalization
        valid_normalizations = ['batch', 'layer', None]
        if self.normalization not in valid_normalizations:
            logger.warning(f"Unknown normalization '{self.normalization}', defaulting to None. "
                          f"Valid options: {valid_normalizations}")
            self.normalization = None
        
        # Validate and fix hidden dimensions
        if not isinstance(self.hidden_dims, list) or not self.hidden_dims:
            logger.warning(f"Invalid hidden_dims '{self.hidden_dims}', using default [128, 64]")
            self.hidden_dims = [128, 64]
        
        # Ensure all hidden dimensions are positive integers
        self.hidden_dims = [max(1, int(dim)) for dim in self.hidden_dims if isinstance(dim, (int, float)) and dim > 0]
        if not self.hidden_dims:
            logger.warning("No valid hidden dimensions found, using default [128, 64]")
            self.hidden_dims = [128, 64]
        
        # Validate and fix dropout rates
        if not isinstance(self.dropout_rates, list):
            logger.warning(f"Invalid dropout_rates type '{type(self.dropout_rates)}', using default [0.2, 0.15]")
            self.dropout_rates = [0.2, 0.15]
        
        # Ensure all dropout rates are valid floats between 0 and 1
        fixed_dropout_rates = []
        for rate in self.dropout_rates:
            if isinstance(rate, (int, float)) and 0 <= rate <= 1:
                fixed_dropout_rates.append(float(rate))
            else:
                logger.warning(f"Invalid dropout rate {rate}, using 0.2")
                fixed_dropout_rates.append(0.2)
        
        self.dropout_rates = fixed_dropout_rates if fixed_dropout_rates else [0.2, 0.15]
        
        # Fix length mismatch between hidden_dims and dropout_rates
        if len(self.hidden_dims) != len(self.dropout_rates):
            logger.warning(f"Length mismatch: hidden_dims({len(self.hidden_dims)}) vs "
                          f"dropout_rates({len(self.dropout_rates)})")
            
            if len(self.dropout_rates) < len(self.hidden_dims):
                # Extend dropout_rates with decreasing values
                last_dropout = self.dropout_rates[-1] if self.dropout_rates else 0.2
                while len(self.dropout_rates) < len(self.hidden_dims):
                    new_dropout = max(0.1, last_dropout * 0.8)
                    self.dropout_rates.append(new_dropout)
                    last_dropout = new_dropout
                logger.info(f"Extended dropout_rates to: {self.dropout_rates}")
            else:
                # Truncate dropout_rates
                self.dropout_rates = self.dropout_rates[:len(self.hidden_dims)]
                logger.info(f"Truncated dropout_rates to: {self.dropout_rates}")
        
        # Validate activation parameter
        if not isinstance(self.activation_param, (int, float)) or self.activation_param < 0:
            logger.warning(f"Invalid activation_param {self.activation_param}, using 0.2")
            self.activation_param = 0.2
        
        # Validate encoding dimension
        if not isinstance(self.encoding_dim, int) or self.encoding_dim <= 0:
            logger.warning(f"Invalid encoding_dim {self.encoding_dim}, using 32")
            self.encoding_dim = 32

    def _build_architecture(self) -> None:
        """Build the encoder and decoder networks with comprehensive error handling."""
        if self.legacy_mode:
            # Simple architecture for backward compatibility
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.encoding_dim, self.input_dim),
                nn.Sigmoid()
            )
            self.skip = None
        else:
            # Build advanced encoder and decoder networks
            try:
                # Encoder architecture: input -> hidden_layers -> encoding
                encoder_dims = self.hidden_dims + [self.encoding_dim]
                encoder_dropouts = self.dropout_rates + [min(0.1, max(self.dropout_rates))]
                
                self.encoder = self._build_network(
                    input_dim=self.input_dim,
                    layer_dims=encoder_dims,
                    dropout_rates=encoder_dropouts,
                    activation=self.activation,
                    activation_param=self.activation_param,
                    normalization=self.normalization,
                    final_activation="tanh"
                )
                
                # Decoder architecture: encoding -> hidden_layers_reversed -> output
                decoder_dims = list(reversed(self.hidden_dims)) + [self.input_dim]
                decoder_dropouts = list(reversed(self.dropout_rates)) + [0.0]  # No dropout on final layer
                
                self.decoder = self._build_network(
                    input_dim=self.encoding_dim,
                    layer_dims=decoder_dims,
                    dropout_rates=decoder_dropouts,
                    activation=self.activation,
                    activation_param=self.activation_param,
                    normalization=self.normalization,
                    final_activation="sigmoid"
                )
                
                # Skip connection (when architecturally reasonable)
                self.skip = (
                    nn.Linear(self.input_dim, self.input_dim)
                    if self.skip_connection and self.input_dim <= self.encoding_dim * 2
                    else None
                )
                
                if self.skip_connection and self.skip is None:
                    logger.info(f"Skip connection disabled due to dimension mismatch: "
                               f"input_dim({self.input_dim}) > 2*encoding_dim({self.encoding_dim})")
                
            except Exception as e:
                logger.error(f"Error building advanced architecture: {e}")
                # Fallback to simple architecture
                logger.warning("Falling back to simple architecture")
                self.encoder = nn.Sequential(
                    nn.Linear(self.input_dim, self.encoding_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(self.encoding_dim, self.input_dim),
                    nn.Sigmoid()
                )
                self.skip = None

    def _build_network(
        self,
        input_dim: int,
        layer_dims: List[int],
        dropout_rates: List[float],
        activation: str,
        activation_param: float,
        normalization: Optional[str],
        final_activation: Optional[str] = None
    ) -> nn.Sequential:
        """Build encoder/decoder networks with robust error handling and validation."""
        layers = []
        prev_dim = input_dim
        
        # Ensure parameters are properly aligned
        if len(dropout_rates) != len(layer_dims):
            logger.debug(f"Aligning dropout_rates({len(dropout_rates)}) with layer_dims({len(layer_dims)})")
            if len(dropout_rates) < len(layer_dims):
                # Extend with last value or default
                last_dropout = dropout_rates[-1] if dropout_rates else 0.2
                dropout_rates = dropout_rates + [last_dropout] * (len(layer_dims) - len(dropout_rates))
            else:
                # Truncate
                dropout_rates = dropout_rates[:len(layer_dims)]
        
        for i, (h_dim, dropout) in enumerate(zip(layer_dims, dropout_rates)):
            is_final_layer = (i == len(layer_dims) - 1)
            
            try:
                # Linear layer
                layers.append(nn.Linear(prev_dim, h_dim))
                
                # Normalization (skip for final layer to avoid output conflicts)
                if not is_final_layer and normalization:
                    if normalization == "batch" and h_dim > 1:
                        layers.append(nn.BatchNorm1d(h_dim))
                    elif normalization == "layer":
                        layers.append(nn.LayerNorm(h_dim))
                
                # Activation (skip for final layer if final_activation is specified)
                if not is_final_layer or final_activation is None:
                    if activation == "leaky_relu":
                        layers.append(nn.LeakyReLU(negative_slope=activation_param, inplace=True))
                    elif activation == "gelu":
                        layers.append(nn.GELU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    elif activation == "sigmoid":
                        layers.append(nn.Sigmoid())
                    else:  # Default to ReLU
                        layers.append(nn.ReLU(inplace=True))
                
                # Dropout (skip for final layer to avoid corrupting output)
                if dropout > 0 and not is_final_layer:
                    layers.append(nn.Dropout(dropout))
                
                prev_dim = h_dim
                
            except Exception as e:
                logger.error(f"Error building layer {i}: {e}")
                raise RuntimeError(f"Failed to build network layer {i}: {e}")
        
        # Final activation if specified
        if final_activation:
            try:
                if final_activation == "tanh":
                    layers.append(nn.Tanh())
                elif final_activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif final_activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
            except Exception as e:
                logger.error(f"Error adding final activation '{final_activation}': {e}")
        
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize weights using appropriate methods based on activation function."""
        try:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if self.activation == "leaky_relu":
                        nn.init.kaiming_normal_(
                            m.weight, 
                            mode='fan_in', 
                            nonlinearity='leaky_relu',
                            a=self.activation_param
                        )
                    elif self.activation in ["gelu", "tanh", "sigmoid"]:
                        nn.init.xavier_normal_(m.weight, gain=1.0)
                    else:  # ReLU and others
                        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
                elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        except Exception as e:
            logger.error(f"Weight initialization error: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision and optional skip connection."""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor input, got {type(x)}")
        
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (batch_size, input_dim), got shape {x.shape}")
        
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Input feature dimension {x.size(-1)} doesn't match expected {self.input_dim}")
        
        try:
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                
                # Apply skip connection if available and not in legacy mode
                if self.skip is not None and not self.legacy_mode:
                    skip_connection = self.skip(x)
                    decoded = decoded + skip_connection
                
                return decoded
        except Exception as e:
            logger.error(f"EnhancedAutoencoder forward pass failed: {e}")
            raise RuntimeError(f"Forward pass error: {e}")

    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested

    def get_config(self) -> Dict[str, Any]:
        """Returns the comprehensive configuration of the autoencoder."""
        return {
            "model_type": "EnhancedAutoencoder",
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rates": self.dropout_rates,
            "activation": self.activation,
            "activation_param": self.activation_param,
            "normalization": self.normalization,
            "skip_connection": self.skip is not None,
            "mixed_precision": self.mixed_precision,
            "mixed_precision_requested": self._mixed_precision_requested,
            "legacy_mode": self.legacy_mode,
            "min_features": self.min_features,
            "architecture": "enhanced",
            "config_version": "2.1",
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "layer_info": {
                "encoder_layers": len(self.encoder),
                "decoder_layers": len(self.decoder),
                "has_skip_connection": self.skip is not None
            }
        }
    
    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update model settings from configuration (limited to non-architectural changes)."""
        try:
            model_config = config.get('model', {})
            
            # Only update non-architectural parameters
            if 'mixed_precision' in model_config:
                self._mixed_precision_requested = model_config['mixed_precision']
                self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
                logger.info(f"Updated mixed_precision to {self.mixed_precision}")
        except Exception as e:
            logger.error(f"Config update failed: {e}")

class AutoencoderEnsemble(nn.Module):
    """Ensemble of autoencoders with configurable diversity and comprehensive error handling.
    
    This class creates an ensemble of different autoencoder architectures to improve
    robustness and performance through model diversity.
    
    Args:
        input_dim: Dimension of input features (required)
        num_models: Number of autoencoders in ensemble (default: from config)
        encoding_dim: Base size of latent representation (default: from config)
        diversity_factor: Scale factor for varying architectures (default: from config)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU)
        min_features: Minimum allowed input dimension (default: from config)
        config: Optional configuration dictionary to override defaults
        **kwargs: Additional parameters for compatibility
    """
    
    def __init__(
        self,
        input_dim: int,
        num_models: Optional[int] = None,
        encoding_dim: Optional[int] = None,
        diversity_factor: Optional[float] = None,
        mixed_precision: Optional[bool] = None,
        min_features: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs  # Accept additional parameters for compatibility
    ):
        super().__init__()
        
        # Warn about unused parameters for debugging
        if kwargs:
            logger.debug(f"AutoencoderEnsemble ignoring unused parameters: {list(kwargs.keys())}")
        
        # Load configuration with comprehensive fallbacks
        if config is None:
            try:
                current_config = get_current_config()
                config = current_config.get('model', {}) if isinstance(current_config, dict) else {}
            except Exception as e:
                logger.debug(f"Could not load configuration, using defaults: {e}")
                config = {}
        
        # Apply configuration with parameter precedence and robust fallbacks
        try:
            self.num_models = num_models if num_models is not None else config.get('num_models', globals().get('NUM_MODELS', 3))
            self.encoding_dim = encoding_dim if encoding_dim is not None else config.get('encoding_dim', globals().get('DEFAULT_ENCODING_DIM', 32))
            self.diversity_factor = diversity_factor if diversity_factor is not None else config.get('diversity_factor', globals().get('DIVERSITY_FACTOR', 0.2))
            self.min_features = min_features if min_features is not None else config.get('min_features', globals().get('MIN_FEATURES', 10))
        except Exception as e:
            logger.warning(f"Error loading config parameters, using hardcoded defaults: {e}")
            self.num_models = num_models if num_models is not None else 3
            self.encoding_dim = encoding_dim if encoding_dim is not None else 32
            self.diversity_factor = diversity_factor if diversity_factor is not None else 0.2
            self.min_features = min_features if min_features is not None else 10
        
        self.input_dim = input_dim
        
        # Input validation with detailed error messages
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim} (type: {type(input_dim)})")
        
        if input_dim < self.min_features:
            raise ValueError(f"Input dimension ({input_dim}) must be at least {self.min_features}")
        
        if not isinstance(self.num_models, int) or self.num_models < 1:
            raise ValueError(f"num_models must be a positive integer, got {self.num_models}")
        
        if not isinstance(self.diversity_factor, (int, float)) or not 0 <= self.diversity_factor <= 1:
            raise ValueError(f"diversity_factor must be between 0 and 1, got {self.diversity_factor}")
        
        if not isinstance(self.encoding_dim, int) or self.encoding_dim <= 0:
            raise ValueError(f"encoding_dim must be a positive integer, got {self.encoding_dim}")

        # Mixed precision handling with configuration awareness
        try:
            mixed_precision_config = config.get('mixed_precision', globals().get('MIXED_PRECISION', True))
            self._mixed_precision_requested = mixed_precision if mixed_precision is not None else mixed_precision_config
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        except Exception as e:
            logger.debug(f"Error configuring mixed precision, using defaults: {e}")
            self._mixed_precision_requested = mixed_precision if mixed_precision is not None else True
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        
        # Get base configuration for ensemble members with fallbacks
        try:
            base_activation = config.get('activation', globals().get('ACTIVATION', 'relu'))
            base_activation_param = config.get('activation_param', globals().get('ACTIVATION_PARAM', 0.2))
            base_normalization = config.get('normalization', globals().get('NORMALIZATION', None))
        except Exception as e:
            logger.debug(f"Error loading base config parameters: {e}")
            base_activation = 'relu'
            base_activation_param = 0.2
            base_normalization = None
        
        # Initialize ensemble models with architectural diversity and error handling
        self.models = nn.ModuleList()
        successful_models = 0
        
        for i in range(self.num_models):
            try:
                # Calculate diversity parameters
                diversity_offset = (i - self.num_models // 2) * self.diversity_factor
                encoding_dim_variant = max(4, int(self.encoding_dim * (1 + diversity_offset)))
                
                # Create diverse hidden layer configurations
                base_hidden_1 = max(32, int(128 * (1 + diversity_offset * 0.5)))
                base_hidden_2 = max(16, int(64 * (1 + diversity_offset * 0.5)))
                
                # Different architectures for diversity
                if i % 3 == 0:
                    # Standard architecture
                    hidden_dims = [base_hidden_1, base_hidden_2]
                    dropout_rates = [0.2 + i * 0.02, 0.15 + i * 0.02]
                elif i % 3 == 1:
                    # Wider architecture
                    hidden_dims = [base_hidden_1 + 32, base_hidden_2 + 16, base_hidden_2]
                    dropout_rates = [0.25 + i * 0.02, 0.2 + i * 0.02, 0.15 + i * 0.02]
                else:
                    # Simpler architecture for the third variant
                    hidden_dims = [base_hidden_1]
                    dropout_rates = [0.3 + i * 0.02]
                
                # Create ensemble member with error handling
                if i == 0 and self.num_models == 1:
                    # Single model - use simple architecture
                    model = SimpleAutoencoder(
                        input_dim=input_dim,
                        encoding_dim=encoding_dim_variant,
                        mixed_precision=self.mixed_precision,
                        min_features=self.min_features,
                        config=config
                    )
                else:
                    # Multiple models - use enhanced architecture with diversity
                    model = EnhancedAutoencoder(
                        input_dim=input_dim,
                        encoding_dim=encoding_dim_variant,
                        hidden_dims=hidden_dims,
                        dropout_rates=dropout_rates,
                        activation=base_activation,
                        activation_param=base_activation_param if i % 2 == 0 else max(0.1, base_activation_param * 0.5),
                        normalization=base_normalization if i % 2 == 0 else None,
                        skip_connection=(i % 2 == 0),
                        mixed_precision=self.mixed_precision,
                        min_features=self.min_features,
                        config=config
                    )
                
                self.models.append(model)
                successful_models += 1
                
                logger.debug(f"Ensemble model {i} created successfully: "
                            f"encoding_dim={encoding_dim_variant}, hidden_dims={hidden_dims}")
                
            except Exception as e:
                logger.error(f"Failed to create ensemble model {i}: {e}")
                
                # Try to create a simple fallback model
                try:
                    fallback_model = SimpleAutoencoder(
                        input_dim=input_dim,
                        encoding_dim=max(4, self.encoding_dim // 2),
                        mixed_precision=self.mixed_precision,
                        min_features=self.min_features,
                        config=config
                    )
                    self.models.append(fallback_model)
                    successful_models += 1
                    logger.warning(f"Created fallback model for ensemble position {i}")
                    
                except Exception as fallback_error:
                    logger.error(f"Failed to create fallback model for position {i}: {fallback_error}")
        
        # Validate that at least one model was created successfully
        if successful_models == 0:
            raise RuntimeError("Failed to create any ensemble models")
        
        if successful_models < self.num_models:
            logger.warning(f"Only {successful_models} out of {self.num_models} ensemble models created successfully")
            self.num_models = successful_models  # Update to reflect actual count
        
        # Log successful initialization
        logger.debug(f"AutoencoderEnsemble initialized successfully: "
                    f"num_models={self.num_models}, encoding_dim={self.encoding_dim}, "
                    f"mixed_precision={self.mixed_precision} (requested={self._mixed_precision_requested}), "
                    f"diversity_factor={self.diversity_factor}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision support and robust error handling.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Averaged reconstruction from all ensemble members
            
        Raises:
            ValueError: If input tensor has wrong shape or type
            RuntimeError: If forward pass fails
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor input, got {type(x)}")
        
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (batch_size, input_dim), got shape {x.shape}")
        
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Input feature dimension {x.size(-1)} doesn't match expected {self.input_dim}")
        
        try:
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                reconstructions = []
                successful_outputs = 0
                
                for i, model in enumerate(self.models):
                    try:
                        output = model(x)
                        reconstructions.append(output)
                        successful_outputs += 1
                    except Exception as e:
                        logger.warning(f"Ensemble model {i} failed during forward pass: {e}")
                        continue
                
                if successful_outputs == 0:
                    raise RuntimeError("All ensemble models failed during forward pass")
                
                if successful_outputs < len(self.models):
                    logger.debug(f"Only {successful_outputs} out of {len(self.models)} ensemble models "
                                f"produced outputs")
                
                # Average the successful reconstructions
                ensemble_output = torch.stack(reconstructions).mean(dim=0)
                return ensemble_output
                
        except Exception as e:
            logger.error(f"AutoencoderEnsemble forward pass failed: {e}")
            raise RuntimeError(f"Ensemble forward pass error: {e}")

    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested

    def get_config(self) -> Dict[str, Any]:
        """Returns the comprehensive configuration of the ensemble."""
        try:
            ensemble_configs = []
            for i, model in enumerate(self.models):
                try:
                    ensemble_configs.append(model.get_config())
                except Exception as e:
                    logger.warning(f"Failed to get config for ensemble model {i}: {e}")
                    ensemble_configs.append({"error": str(e), "model_index": i})
            
            return {
                "model_type": "AutoencoderEnsemble",
                "input_dim": self.input_dim,
                "num_models": self.num_models,
                "actual_models": len(self.models),
                "encoding_dim": self.encoding_dim,
                "diversity_factor": self.diversity_factor,
                "mixed_precision": self.mixed_precision,
                "mixed_precision_requested": self._mixed_precision_requested,
                "min_features": self.min_features,
                "architecture": "ensemble",
                "model_types": [type(m).__name__ for m in self.models],
                "ensemble_configs": ensemble_configs,
                "config_version": "2.1",
                "total_parameters": sum(sum(p.numel() for p in model.parameters()) for model in self.models),
                "trainable_parameters": sum(sum(p.numel() for p in model.parameters() if p.requires_grad) for model in self.models)
            }
        except Exception as e:
            logger.error(f"Failed to generate ensemble config: {e}")
            return {
                "model_type": "AutoencoderEnsemble",
                "error": str(e),
                "input_dim": self.input_dim,
                "num_models": getattr(self, 'num_models', 0),
                "config_version": "2.1"
            }
    
    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update model settings from configuration (limited to non-architectural changes)."""
        try:
            model_config = config.get('model', {})
            
            # Update mixed precision for ensemble and all sub-models
            if 'mixed_precision' in model_config:
                self._mixed_precision_requested = model_config['mixed_precision']
                self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
                
                # Update all sub-models
                for i, model in enumerate(self.models):
                    try:
                        model.update_from_config(config)
                    except Exception as e:
                        logger.warning(f"Failed to update config for ensemble model {i}: {e}")
                
                logger.info(f"Updated ensemble mixed_precision to {self.mixed_precision}")
        except Exception as e:
            logger.error(f"Ensemble config update failed: {e}")

def load_autoencoder_model(
    model_path: Path,
    input_dim: Optional[int] = None,
    encoding_dim: int = None,
    config: Optional[Dict] = None
) -> Union[SimpleAutoencoder, EnhancedAutoencoder, AutoencoderEnsemble]:
    """Load autoencoder with automatic architecture detection and config handling.
    
    Args:
        model_path: Path to the saved model file
        input_dim: Expected input dimension (optional, will be inferred if None)
        encoding_dim: Default encoding dimension if not found in state_dict
        config: Optional configuration dictionary for model parameters
        
    Returns:
        Loaded model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
        ValueError: If architecture parameters are invalid
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load configuration with fallbacks
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    model_config = config.get('model', {})
    
    # Load the state dict to inspect its structure
    try:
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict from {model_path}: {str(e)}")
    
    # Architecture detection
    is_ensemble = any(k.startswith('models.') for k in state_dict.keys())
    legacy_keys = ['encoder.0.weight', 'encoder.0.bias', 'decoder.0.weight', 'decoder.0.bias']
    is_legacy = all(key in state_dict for key in legacy_keys)
    
    # Infer input dimension from state dict if not provided
    if input_dim is None:
        try:
            if is_legacy:
                input_dim = state_dict['encoder.0.weight'].shape[1]
            else:
                # Search for encoder input layer
                for k in state_dict:
                    if any(pattern in k for pattern in ['encoder.0.weight', 'encoder.net.0.weight', 'models.0.encoder']):
                        input_dim = state_dict[k].shape[1]
                        break
                if input_dim is None:
                    raise ValueError("Could not infer input_dim from state_dict")
        except Exception as e:
            raise ValueError(f"Failed to infer input dimension: {str(e)}")
    
    # Set encoding dimension with configuration precedence
    if encoding_dim is None:
        encoding_dim = model_config.get('encoding_dim', DEFAULT_ENCODING_DIM)
    
    # Initialize model based on detected architecture
    try:
        if is_ensemble:
            logger.info("Loading AutoencoderEnsemble model")
            # Try to extract ensemble configuration
            num_models = model_config.get('num_models', NUM_MODELS)
            diversity_factor = model_config.get('diversity_factor', DIVERSITY_FACTOR)
            
            model = AutoencoderEnsemble(
                input_dim=input_dim,
                num_models=num_models,
                encoding_dim=encoding_dim,
                diversity_factor=diversity_factor,
                config=config
            )
        elif is_legacy:
            # Extract dimensions from saved model
            encoding_dim = state_dict['encoder.0.weight'].shape[0]
            logger.info(f"Loading legacy autoencoder: input_dim={input_dim}, encoding_dim={encoding_dim}")
            
            model = EnhancedAutoencoder(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
                legacy_mode=True,
                config=config
            )
        else:
            # Enhanced autoencoder
            logger.info(f"Loading enhanced autoencoder: input_dim={input_dim}, encoding_dim={encoding_dim}")
            
            model = EnhancedAutoencoder(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
                hidden_dims=model_config.get('hidden_dims', HIDDEN_LAYER_SIZES),
                dropout_rates=model_config.get('dropout_rates', DROPOUT_RATES),
                activation=model_config.get('activation', ACTIVATION),
                activation_param=model_config.get('activation_param', ACTIVATION_PARAM),
                normalization=model_config.get('normalization', NORMALIZATION),
                skip_connection=model_config.get('skip_connection', True),
                min_features=model_config.get('min_features', MIN_FEATURES),
                legacy_mode=False,
                config=config
            )
        
        # Load state dict with robust error handling
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Log loading details
        missing_keys = [k for k in model_state_dict if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in model_state_dict]
        
        if missing_keys:
            logger.warning(f"Missing keys in state_dict: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {unexpected_keys[:5]}...")
        
        logger.info(f"Successfully loaded {type(model).__name__} model")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def load_and_validate_data(
    data_path: Path = None,
    artifacts_path: Path = None,
    config: Optional[Dict] = None
) -> Dict[str, np.ndarray]:
    """Load and validate preprocessed data with comprehensive checks."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    data_config = config.get('data', {})
    
    # Set default paths if not provided
    if data_path is None:
        data_path = DEFAULT_MODEL_DIR / "preprocessed_dataset.csv"
    if artifacts_path is None:
        artifacts_path = DEFAULT_MODEL_DIR / "preprocessing_artifacts.pkl"
    
    try:
        # Validate file existence
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
            
        logger.info(f"Loading data from {data_path} and artifacts from {artifacts_path}")
        
        # Load data and artifacts
        df = pd.read_csv(data_path)
        artifacts = joblib.load(artifacts_path)
        
        # Validate data structure
        if "Label" not in df.columns:
            raise ValueError("Dataset missing 'Label' column")
            
        feature_names = artifacts.get("feature_names", [])
        if not feature_names:
            raise ValueError("No feature names found in artifacts")
        
        min_features = data_config.get('min_features', MIN_FEATURES)
        if len(feature_names) < min_features:
            raise ValueError(f"Too few features ({len(feature_names)}), need at least {min_features}")
            
        # Apply scaling if available
        scaler = artifacts.get("scaler")
        X = df[feature_names].values.astype(np.float32)
        if scaler:
            logger.info("Applying feature scaling from artifacts")
            X = scaler.transform(X)
        
        # Split data according to configuration
        normal_mask = df["Label"] == 0
        X_normal = X[normal_mask]
        X_attack = X[~normal_mask]
        
        # Validate class balance
        if len(X_normal) == 0 or len(X_attack) == 0:
            raise ValueError("One class has zero samples")
        
        # Use configuration for validation split
        validation_split = data_config.get('validation_split', 0.2)
        test_split = data_config.get('test_split', 0.2)
        
        # Calculate splits
        normal_val_size = int(len(X_normal) * validation_split)
        attack_test_size = int(len(X_attack) * test_split)
        
        logger.info(f"Loaded dataset: {len(X_normal)} normal, {len(X_attack)} attack samples")
        logger.info(f"Using validation split: {validation_split:.1%}, test split: {test_split:.1%}")
            
        return {
            "X_train": X_normal[normal_val_size:],
            "X_val": X_normal[:normal_val_size],
            "X_test": X_attack[:attack_test_size] if attack_test_size > 0 else X_attack,
            "feature_names": feature_names,
            "metadata": {
                "total_normal": len(X_normal),
                "total_attack": len(X_attack),
                "scaler_applied": scaler is not None,
                "validation_split": validation_split,
                "test_split": test_split
            }
        }
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def generate_synthetic_data(
    normal_samples: int = None,
    attack_samples: int = None,
    features: int = None,
    anomaly_factor: float = None,
    random_state: int = None,
    config: Optional[Dict] = None
) -> Dict[str, np.ndarray]:
    """Generate realistic synthetic data with configuration integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    data_config = config.get('data', {})
    synthetic_config = data_config.get('synthetic_generation', {})
    
    # Apply configuration with parameter precedence
    normal_samples = normal_samples if normal_samples is not None else data_config.get('normal_samples', NORMAL_SAMPLES)
    attack_samples = attack_samples if attack_samples is not None else data_config.get('attack_samples', ATTACK_SAMPLES)
    features = features if features is not None else data_config.get('features', FEATURES)
    anomaly_factor = anomaly_factor if anomaly_factor is not None else data_config.get('anomaly_factor', ANOMALY_FACTOR)
    random_state = random_state if random_state is not None else data_config.get('random_state', RANDOM_STATE)
    
    # Additional synthetic generation parameters
    cluster_variance = synthetic_config.get('cluster_variance', 0.1)
    anomaly_sparsity = synthetic_config.get('anomaly_sparsity', 0.3)
    validation_split = data_config.get('validation_split', 0.2)
    
    # Validate parameters
    if normal_samples <= 0 or attack_samples <= 0 or features <= 0:
        raise ValueError("Sample counts and features must be positive")
    if not 0 < validation_split < 1:
        raise ValueError("Validation split must be between 0 and 1")
    
    np.random.seed(random_state)
    logger.info(f"Generating synthetic data: {normal_samples} normal, {attack_samples} attack samples")
    logger.info(f"Features: {features}, anomaly_factor: {anomaly_factor}, random_state: {random_state}")
    
    # Generate normal data with clustering
    X_normal = np.random.normal(0.5, cluster_variance, (normal_samples, features))
    X_normal = np.clip(X_normal, 0.1, 0.9)
    
    # Generate diverse anomalies
    anomaly_types = ['high_variance', 'shifted_mean', 'sparse_extreme', 'clustered_outliers']
    samples_per_type = attack_samples // len(anomaly_types)
    remainder = attack_samples % len(anomaly_types)
    
    X_attack_parts = []
    
    # High variance anomalies
    n_samples = samples_per_type + (1 if remainder > 0 else 0)
    X_high_var = np.random.normal(0.5, 0.3 * anomaly_factor, (n_samples, features))
    X_attack_parts.append(np.clip(X_high_var, 0.0, 1.0))
    remainder -= 1
    
    # Shifted mean anomalies
    n_samples = samples_per_type + (1 if remainder > 0 else 0)
    X_shifted = np.random.normal(0.5 + 0.3 * anomaly_factor, cluster_variance, (n_samples, features))
    X_attack_parts.append(np.clip(X_shifted, 0.0, 1.0))
    remainder -= 1
    
    # Sparse extreme anomalies
    n_samples = samples_per_type + (1 if remainder > 0 else 0)
    X_sparse = np.random.normal(0.5, cluster_variance, (n_samples, features))
    # Make some features extreme
    extreme_mask = np.random.random((n_samples, features)) < anomaly_sparsity
    X_sparse[extreme_mask] = np.random.choice([0.1, 0.9], size=np.sum(extreme_mask))
    X_attack_parts.append(X_sparse)
    remainder -= 1
    
    # Clustered outliers
    n_samples = samples_per_type + (1 if remainder > 0 else 0)
    X_clustered = np.random.normal(0.2, cluster_variance, (n_samples, features))
    X_attack_parts.append(np.clip(X_clustered, 0.0, 1.0))
    
    X_attack = np.vstack(X_attack_parts)
    
    # Create validation split
    val_size = int(normal_samples * validation_split)
    
    return {
        "X_train": X_normal[val_size:],
        "X_val": X_normal[:val_size],
        "X_test": X_attack,
        "feature_names": [f"feature_{i}" for i in range(features)],
        "metadata": {
            "normal_samples": normal_samples,
            "attack_samples": attack_samples,
            "features": features,
            "anomaly_factor": anomaly_factor,
            "random_state": random_state,
            "generation_config": synthetic_config,
            "validation_split": validation_split
        }
    }

def create_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int = None,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create properly configured dataloaders with configuration integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    training_config = config.get('training', {})
    
    # Apply configuration with parameter precedence
    batch_size = batch_size if batch_size is not None else training_config.get('batch_size', DEFAULT_BATCH_SIZE)
    num_workers = num_workers if num_workers is not None else training_config.get('num_workers', NUM_WORKERS)
    
    # Validate parameters
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if num_workers < 0:
        raise ValueError("Number of workers must be non-negative")
    
    # Limit num_workers based on system capabilities
    max_workers = os.cpu_count() or 1
    num_workers = min(num_workers, max_workers)
    
    # Create tensor datasets
    train_data = TensorDataset(torch.tensor(data["X_train"], dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(data["X_val"], dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(data["X_test"], dtype=torch.float32))
    
    # Configure dataloaders with optimized settings
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        drop_last=True if len(train_data) > batch_size else False
    )
    
    # Larger batches for validation/test (no gradient computation)
    eval_batch_size = min(batch_size * 2, len(val_data), 1024)
    
    val_loader = DataLoader(
        val_data,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    logger.info(f"Created dataloaders: train={batch_size}, val/test={eval_batch_size}, workers={num_workers}")
    return train_loader, val_loader, test_loader

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Optional[Dict] = None
) -> Tuple[float, Dict[str, float]]:
    """Train model for one epoch with configuration integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    training_config = config.get('training', {})
    
    # Extract training parameters from config
    grad_clip = training_config.get('gradient_clip', GRADIENT_CLIP)
    accumulation_steps = training_config.get('gradient_accumulation_steps', GRADIENT_ACCUMULATION_STEPS)
    mixed_precision = training_config.get('mixed_precision', MIXED_PRECISION) and torch.cuda.is_available()
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = GradScaler(enabled=mixed_precision)
    
    try:
        optimizer.zero_grad()
        
        for i, batch in enumerate(loader):
            inputs = batch[0].to(device)
            
            with autocast(enabled=mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, inputs) / accumulation_steps
            
            # Backpropagation with gradient accumulation
            if mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient step with accumulation
            if (i + 1) % accumulation_steps == 0:
                if mixed_precision:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
        
        # Handle any remaining gradients
        if (len(loader) % accumulation_steps) != 0:
            if mixed_precision:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Additional metrics
        metrics = {
            'loss': avg_loss,
            'batches_processed': num_batches,
            'mixed_precision_used': mixed_precision,
            'gradient_clipping_used': grad_clip > 0
        }
        
        return avg_loss, metrics
        
    except Exception as e:
        logger.error(f"Training epoch failed: {str(e)}")
        raise RuntimeError("Training epoch failed") from e
    finally:
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Optional[Dict] = None
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Validate model with enhanced metrics and configuration integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    training_config = config.get('training', {})
    mixed_precision = training_config.get('mixed_precision', MIXED_PRECISION) and torch.cuda.is_available()
    
    model.eval()
    total_loss = 0.0
    all_mse = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            
            with autocast(enabled=mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate per-sample MSE
            mse = torch.mean((inputs - outputs)**2, dim=1).cpu().numpy()
            all_mse.extend(mse)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    mse_array = np.array(all_mse)
    
    # Additional validation metrics
    metrics = {
        'loss': avg_loss,
        'mean_mse': np.mean(mse_array) if len(mse_array) > 0 else 0.0,
        'std_mse': np.std(mse_array) if len(mse_array) > 0 else 0.0,
        'min_mse': np.min(mse_array) if len(mse_array) > 0 else 0.0,
        'max_mse': np.max(mse_array) if len(mse_array) > 0 else 0.0,
        'samples_validated': len(mse_array)
    }
    
    return avg_loss, mse_array, metrics

def calculate_threshold(
    model: nn.Module,
    loader: DataLoader,
    percentile: int = None,
    device: torch.device = None,
    config: Optional[Dict] = None
) -> Tuple[float, Dict[str, float]]:
    """Calculate anomaly threshold with configuration integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    security_config = config.get('security', {})
    
    # Apply configuration with parameter precedence
    percentile = percentile if percentile is not None else security_config.get('percentile', DEFAULT_PERCENTILE)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    mse_array = np.array(mse_values)
    threshold = np.percentile(mse_array, percentile)
    
    # Calculate additional threshold statistics
    threshold_stats = {
        'threshold': threshold,
        'percentile_used': percentile,
        'mean_mse': np.mean(mse_array),
        'std_mse': np.std(mse_array),
        'samples_used': len(mse_array),
        'threshold_strategy': security_config.get('anomaly_threshold_strategy', 'percentile')
    }
    
    logger.info(f"Calculated anomaly threshold (P{percentile}): {threshold:.6f}")
    logger.debug(f"Threshold statistics: {threshold_stats}")
    
    return threshold, threshold_stats

def train_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Main training pipeline with comprehensive configuration integration."""
    # Initialize configuration system
    config = initialize_config()
    
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        if getattr(args, 'non_interactive', False):
            config = get_current_config()
        else:
            if prompt_user("Continue with default config?", default=True):
                config = get_current_config()
            else:
                raise
    
    update_global_config(config)
    
    # Apply configuration with command line args taking precedence
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    security_config = config.get('security', {})
    monitoring_config = config.get('monitoring', {})
    system_config = config.get('system', {})
    
    # Resolve all parameters with proper precedence (args > config > defaults)
    resolved_params = {
        # Training parameters
        'batch_size': getattr(args, 'batch_size', None) or training_config.get('batch_size', DEFAULT_BATCH_SIZE),
        'epochs': getattr(args, 'epochs', None) or training_config.get('epochs', DEFAULT_EPOCHS),
        'lr': getattr(args, 'lr', None) or training_config.get('learning_rate', LEARNING_RATE),
        'patience': getattr(args, 'patience', None) or training_config.get('patience', EARLY_STOPPING_PATIENCE),
        'weight_decay': getattr(args, 'weight_decay', None) or training_config.get('weight_decay', WEIGHT_DECAY),
        'grad_clip': getattr(args, 'grad_clip', None) or training_config.get('gradient_clip', GRADIENT_CLIP),
        'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', None) or training_config.get('gradient_accumulation_steps', GRADIENT_ACCUMULATION_STEPS),
        
        # Model parameters
        'encoding_dim': getattr(args, 'encoding_dim', None) or model_config.get('encoding_dim', DEFAULT_ENCODING_DIM),
        'model_type': getattr(args, 'model_type', None) or model_config.get('model_type', 'EnhancedAutoencoder'),
        'hidden_dims': model_config.get('hidden_dims', HIDDEN_LAYER_SIZES),
        'dropout_rates': model_config.get('dropout_rates', DROPOUT_RATES),
        'activation': model_config.get('activation', ACTIVATION),
        'activation_param': model_config.get('activation_param', ACTIVATION_PARAM),
        'normalization': model_config.get('normalization', NORMALIZATION),
        'legacy_mode': model_config.get('legacy_mode', False),
        'min_features': model_config.get('min_features', MIN_FEATURES),
        'mixed_precision': getattr(args, 'mixed_precision', None) or model_config.get('mixed_precision', MIXED_PRECISION),
        'num_models': getattr(args, 'num_models', None) or model_config.get('num_models', NUM_MODELS),
        'diversity_factor': getattr(args, 'diversity_factor', None) or model_config.get('diversity_factor', DIVERSITY_FACTOR),
        'skip_connection': getattr(args, 'skip_connection', None) or model_config.get('skip_connection', True),
        
        # Data parameters
        'features': getattr(args, 'features', None) or data_config.get('features', FEATURES),
        'normal_samples': getattr(args, 'normal_samples', None) or data_config.get('normal_samples', NORMAL_SAMPLES),
        'attack_samples': getattr(args, 'attack_samples', None) or data_config.get('attack_samples', ATTACK_SAMPLES),
        'use_real_data': getattr(args, 'use_real_data', False) or data_config.get('use_real_data', False),
        'validation_split': data_config.get('validation_split', 0.2),
        'anomaly_factor': getattr(args, 'anomaly_factor', None) or data_config.get('anomaly_factor', ANOMALY_FACTOR),
        'random_state': getattr(args, 'random_state', None) or data_config.get('random_state', RANDOM_STATE),
        
        # Security parameters
        'percentile': getattr(args, 'percentile', None) or security_config.get('percentile', DEFAULT_PERCENTILE),
        'attack_threshold': getattr(args, 'attack_threshold', None) or security_config.get('attack_threshold', DEFAULT_ATTACK_THRESHOLD),
        'false_negative_cost': getattr(args, 'false_negative_cost', None) or security_config.get('false_negative_cost', FALSE_NEGATIVE_COST),
        'enable_security_metrics': getattr(args, 'enable_security_metrics', None) or security_config.get('enable_security_metrics', SECURITY_METRICS),
        
        # System parameters
        'model_dir': Path(getattr(args, 'model_dir', DEFAULT_MODEL_DIR)),
        'tb_dir': Path(getattr(args, 'tb_dir', TB_DIR)),
        'log_dir': Path(getattr(args, 'log_dir', LOG_DIR)),
        'config_dir': Path(getattr(args, 'config_dir', CONFIG_DIR)),
        
        # Flags
        'export_onnx': getattr(args, 'export_onnx', False),
        'debug': getattr(args, 'debug', False),
        'non_interactive': getattr(args, 'non_interactive', False),
        'preset': getattr(args, 'preset', None),
    }
    
    # Update args object with resolved parameters
    for key, value in resolved_params.items():
        setattr(args, key, value)
    
    # Validate resolved parameters
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.features <= 0:
        raise ValueError("Number of features must be positive")
    
    # Log resolved configuration
    logger.info("═" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("═" * 80)
    logger.info(f"Training Parameters:")
    logger.info(f"  [+] Batch size: {args.batch_size}")
    logger.info(f"  [+] Epochs: {args.epochs}")
    logger.info(f"  [+] Learning rate: {args.lr:.2e}")
    logger.info(f"  [+] Weight decay: {args.weight_decay:.2e}")
    logger.info(f"  [+] Patience: {args.patience}")
    logger.info(f"Model Parameters:")
    logger.info(f"  [+] Type: {args.model_type}")
    logger.info(f"  [+] Features: {args.features}")
    logger.info(f"  [+] Encoding dim: {args.encoding_dim}")
    logger.info(f"Data Parameters:")
    logger.info(f"  [+] Normal samples: {args.normal_samples}")
    logger.info(f"  [+] Attack samples: {args.attack_samples}")
    logger.info(f"  [+] Use real data: {args.use_real_data}")
    logger.info(f"System:")
    logger.info(f"  [+] Model directory: {args.model_dir}")
    logger.info(f"  [+] Preset: {args.preset or 'None'}")
    
    # Setup experiment tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    experiment_dir = args.tb_dir / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=experiment_dir)
    
    # Hardware setup with configuration awareness
    hw = check_hardware()
    device = torch.device(hw["device"])
    
    # Apply hardware-specific optimizations from config
    if torch.cuda.is_available() and system_config.get('cuda_optimizations', True):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    logger.info("─" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("─" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Ensure directories exist
        args.model_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Data preparation with robust error handling
        logger.info("─" * 60)
        logger.info("DATA PREPARATION")
        logger.info("─" * 60)
        
        if args.use_real_data:
            try:
                data = load_and_validate_data(config=config)
                # Update features from actual data
                actual_features = len(data["feature_names"])
                if actual_features != args.features:
                    logger.info(f"Updating features from {args.features} to {actual_features} (from real data)")
                    args.features = actual_features
                logger.info("[INFO] Using real preprocessed data")
            except Exception as e:
                logger.warning(f"Failed to load real data: {str(e)}")
                if args.non_interactive or prompt_user("Use synthetic data instead?", True):
                    data = generate_synthetic_data(
                        normal_samples=args.normal_samples,
                        attack_samples=args.attack_samples,
                        features=args.features,
                        config=config
                    )
                    logger.info("[INFO] Using synthetic data as fallback")
                else:
                    raise RuntimeError("Real data loading failed and synthetic fallback rejected")
        else:
            data = generate_synthetic_data(
                normal_samples=args.normal_samples,
                attack_samples=args.attack_samples,
                features=args.features,
                config=config
            )
            logger.info("[INFO] Using synthetic data")
        
        # Log data statistics
        logger.info(f"Dataset statistics:")
        logger.info(f"  [+] Training samples: {len(data['X_train'])}")
        logger.info(f"  [+] Validation samples: {len(data['X_val'])}")
        logger.info(f"  [+] Test samples: {len(data['X_test'])}")
        logger.info(f"  [+] Features: {len(data['feature_names'])}")
        
        # Create dataloaders with configuration
        train_loader, val_loader, test_loader = create_dataloaders(
            data, 
            batch_size=args.batch_size,
            shuffle=not args.debug,  # Disable shuffling in debug mode
            config=config
        )
        
        logger.info(f"Created dataloaders: train={args.batch_size}, val/test={len(val_loader.dataset)}, workers={training_config.get('num_workers', NUM_WORKERS)}")
        
        # Model initialization
        logger.info("─" * 60)
        logger.info("MODEL INITIALIZATION")
        logger.info("─" * 60)
        
        # Initialize model variants if needed
        if not MODEL_VARIANTS:
            initialize_model_variants(silent=True)
        
        model_class = MODEL_VARIANTS.get(args.model_type, EnhancedAutoencoder)
        logger.info(f"Using model class: {model_class.__name__}")
        
        try:
            # Get model-specific parameters based on model type
            if model_class.__name__ == 'SimpleAutoencoder':
                model_params = {
                    'input_dim': args.features,
                    'encoding_dim': args.encoding_dim,
                    'mixed_precision': args.mixed_precision,
                    'min_features': args.min_features,
                    'config': config
                }
            elif model_class.__name__ == 'EnhancedAutoencoder':
                model_params = {
                    'input_dim': args.features,
                    'encoding_dim': args.encoding_dim,
                    'hidden_dims': args.hidden_dims,
                    'dropout_rates': args.dropout_rates,
                    'activation': args.activation,
                    'activation_param': args.activation_param,
                    'normalization': args.normalization,
                    'legacy_mode': args.legacy_mode,
                    'skip_connection': args.skip_connection,
                    'mixed_precision': args.mixed_precision,
                    'min_features': args.min_features,
                    'config': config
                }
            elif model_class.__name__ == 'AutoencoderEnsemble':
                model_params = {
                    'input_dim': args.features,
                    'num_models': args.num_models,
                    'encoding_dim': args.encoding_dim,
                    'diversity_factor': args.diversity_factor,
                    'mixed_precision': args.mixed_precision,
                    'min_features': args.min_features,
                    'config': config
                }
            else:
                # Fallback for unknown model types - use minimal parameters
                logger.warning(f"Unknown model type: {model_class.__name__}, using minimal parameters")
                model_params = {
                    'input_dim': args.features,
                    'encoding_dim': args.encoding_dim,
                    'config': config
                }
            
            # Create model with type-specific parameters
            model = model_class(**model_params).to(device)
            
        except Exception as e:
            logger.error(f"Failed to instantiate model {model_class.__name__}: {str(e)}")
            logger.error(f"Attempted parameters: {list(model_params.keys()) if 'model_params' in locals() else 'None'}")
            raise RuntimeError(f"Model instantiation failed: {str(e)}") from e
        
        # Log model details
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model: {type(model).__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
        
        # Log model configuration if available
        if hasattr(model, 'get_config'):
            model_config_info = model.get_config()
            logger.debug(f"Model configuration: {model_config_info}")
        
        # Training setup
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=training_config.get('adam_betas', (0.9, 0.999)),
            eps=training_config.get('adam_eps', 1e-8)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=training_config.get('lr_patience', 2),
            factor=training_config.get('lr_factor', 0.5),
            min_lr=training_config.get('min_lr', 1e-7)
        )
        
        criterion = nn.MSELoss()
        
        # Mixed precision setup if enabled
        use_mixed_precision = (training_config.get('mixed_precision', MIXED_PRECISION) 
                              and torch.cuda.is_available() 
                              and hasattr(torch.cuda.amp, 'GradScaler'))
        
        scaler = GradScaler(enabled=use_mixed_precision) if use_mixed_precision else None
        
        logger.info(f"Mixed precision training: {'Enabled' if use_mixed_precision else 'Disabled'}")
        
        # Training loop
        logger.info("─" * 60)
        logger.info("TRAINING")
        logger.info("─" * 60)
        logger.info(f"Starting training for {args.epochs} epochs")
        logger.info(f"Initial learning rate: {args.lr:.2e}")
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': [],
            'gpu_memory': [] if torch.cuda.is_available() else None
        }
        
        # Training metrics
        total_train_time = 0
        start_time = time.time()
        
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            
            # Training phase with configuration-aware training
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, config
            )
            
            # Validation phase
            val_loss, val_mse, val_metrics = validate(
                model, val_loader, criterion, device, config
            )
            
            # Update training history
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['learning_rate'].append(current_lr)
            training_history['epoch_times'].append(epoch_time)
            
            # GPU memory tracking
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                training_history['gpu_memory'].append(gpu_memory)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Logging and visualization
            if monitoring_config.get('tensorboard_logging', True):
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Learning_Rate", current_lr, epoch)
                writer.add_scalar("Epoch_Time", epoch_time, epoch)
                
                # Add additional metrics
                for key, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"Train/{key}", value, epoch)
                
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"Validation/{key}", value, epoch)
                
                if torch.cuda.is_available():
                    writer.add_scalar("System/GPU_Memory_GB", gpu_memory, epoch)
            
            # Console logging with configurable frequency
            log_frequency = monitoring_config.get('log_frequency', 1)
            if epoch % log_frequency == 0 or epoch == args.epochs - 1:
                gpu_info = f" | GPU: {gpu_memory:.1f}GB" if torch.cuda.is_available() else ""
                logger.info(
                    f"Epoch {epoch+1:3d}/{args.epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s{gpu_info}"
                )
            
            # Model checkpointing and early stopping
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'args': vars(args)
                }, args.model_dir / "best_model.pth")
                
                if epoch % monitoring_config.get('checkpoint_frequency', 10) == 0:
                    logger.info(f"[INFO] New best model saved (epoch {epoch+1})")
            else:
                patience_counter += 1
                
            # Early stopping check
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1} (patience: {args.patience})")
                break
            
            # Periodic checkpoint saving
            if monitoring_config.get('save_checkpoints', True) and epoch % monitoring_config.get('checkpoint_frequency', 50) == 0:
                checkpoint_path = args.model_dir / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'training_history': training_history
                }, checkpoint_path)
        
        total_train_time = time.time() - start_time
        
        # Load best model for evaluation
        logger.info("Loading best model for final evaluation...")
        checkpoint_pth = args.model_dir / "best_model.pth"
        checkpoint = torch.load(checkpoint_pth, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Calculate anomaly threshold with configuration
        logger.info("─" * 60)
        logger.info("THRESHOLD CALCULATION")
        logger.info("─" * 60)
        
        threshold, threshold_stats = calculate_threshold(
            model, val_loader, args.percentile, device, config
        )
        
        logger.info(f"Anomaly threshold (P{args.percentile}): {threshold:.6f}")
        logger.info(f"Threshold statistics: {threshold_stats}")
        
        # Final evaluation on test set
        logger.info("─" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("─" * 60)
        
        test_loss, test_mse, test_metrics = validate(
            model, test_loader, criterion, device, config
        )
        
        # Calculate detection metrics
        anomaly_predictions = test_mse > threshold
        anomaly_rate = anomaly_predictions.mean()
        
        # Additional evaluation metrics
        mse_stats = {
            'mean': np.mean(test_mse),
            'std': np.std(test_mse),
            'min': np.min(test_mse),
            'max': np.max(test_mse),
            'median': np.median(test_mse),
            'p95': np.percentile(test_mse, 95),
            'p99': np.percentile(test_mse, 99)
        }
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Anomaly Detection Rate: {anomaly_rate:.2%}")
        logger.info(f"MSE Statistics: {mse_stats}")
        
        # Save all artifacts
        logger.info("─" * 60)
        logger.info("SAVING ARTIFACTS")
        logger.info("─" * 60)
        
        # Save final model
        torch.save(model.state_dict(), args.model_dir / "autoencoder_ids.pth")
        
        # Save threshold with additional metadata
        threshold_data = {
            'threshold': threshold,
            'threshold_stats': threshold_stats,
            'mse_statistics': mse_stats,
            'percentile': args.percentile
        }
        joblib.dump(threshold_data, args.model_dir / "anomaly_threshold.pkl")
        
        # Export to ONNX if requested
        onnx_path = None
        if args.export_onnx:
            try:
                onnx_path = export_to_onnx(model, args.features, device, args.model_dir)
                logger.info(f"[INFO] ONNX model exported: {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
        
        # Prepare comprehensive training metadata
        training_metadata = {
            "experiment": {
                "run_id": run_id,
                "timestamp": timestamp,
                "duration_seconds": total_train_time,
                "total_epochs": epoch + 1,
                "early_stopped": patience_counter >= args.patience
            },
            "config": config,
            "resolved_params": resolved_params,
            "model": {
                "type": type(model).__name__,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / 1024 / 1024,
                "config": model.get_config() if hasattr(model, 'get_config') else {}
            },
            "training": {
                "final_epoch": epoch + 1,
                "best_val_loss": float(best_val_loss),
                "training_history": training_history,
                "optimizer": {
                    "type": type(optimizer).__name__,
                    "final_lr": current_lr,
                    "weight_decay": args.weight_decay
                },
                "mixed_precision": use_mixed_precision
            },
            "evaluation": {
                "test_loss": float(test_loss),
                "test_metrics": test_metrics,
                "threshold_data": threshold_data,
                "anomaly_detection_rate": float(anomaly_rate),
                "mse_statistics": mse_stats
            },
            "data": {
                "source": "real" if args.use_real_data else "synthetic",
                "metadata": data.get("metadata", {}),
                "train_samples": len(data["X_train"]),
                "val_samples": len(data["X_val"]),
                "test_samples": len(data["X_test"]),
                "features": len(data["feature_names"])
            },
            "system": {
                "hardware": hw,
                "software": {
                    "python_version": sys.version,
                    "pytorch_version": torch.__version__,
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
                },
                "device": str(device),
                "cuda_available": torch.cuda.is_available()
            },
            "artifacts": {
                "model_path": str(args.model_dir / "autoencoder_ids.pth"),
                "best_model_path": str(args.model_dir / "best_model.pth"),
                "threshold_path": str(args.model_dir / "anomaly_threshold.pkl"),
                "onnx_path": str(onnx_path) if onnx_path else None,
                "tensorboard_dir": str(experiment_dir)
            }
        }
        
        # Save metadata
        metadata_path = args.model_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(training_metadata, f, indent=2, default=str)
        
        # Save configuration used for this training
        config_path = args.model_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save TensorBoard data in readable format
        if monitoring_config.get('tensorboard_logging', True):
            save_tensorboard_data(writer, experiment_dir, f"_{run_id}")
        
        # Save training summary
        summary = {
            "success": True,
            "run_id": run_id,
            "timestamp": timestamp,
            "duration_minutes": total_train_time / 60,
            "final_metrics": {
                "best_val_loss": float(best_val_loss),
                "test_loss": float(test_loss),
                "anomaly_detection_rate": float(anomaly_rate),
                "threshold": float(threshold)
            },
            "model_info": {
                "type": type(model).__name__,
                "parameters": total_params,
                "size_mb": total_params * 4 / 1024 / 1024
            }
        }
        
        summary_path = args.model_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Log completion
        logger.info("═" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("═" * 80)
        logger.info(f"[+] Total training time: {total_train_time/60:.1f} minutes")
        logger.info(f"[+] Best validation loss: {best_val_loss:.4f}")
        logger.info(f"[+] Test loss: {test_loss:.4f}")
        logger.info(f"[+] Anomaly detection rate: {anomaly_rate:.2%}")
        logger.info(f"[+] Model saved to: {args.model_dir / 'autoencoder_ids.pth'}")
        logger.info(f"[+] Threshold saved to: {args.model_dir / 'anomaly_threshold.pkl'}")
        logger.info(f"[+] Metadata saved to: {metadata_path}")
        logger.info(f"[+] Training summary: {summary_path}")
        if onnx_path:
            logger.info(f"[INFO] ONNX model: {onnx_path}")
        logger.info("═" * 80)
        
        return training_metadata
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        
        # Save error information
        error_info = {
            "success": False,
            "error": str(e),
            "timestamp": timestamp,
            "config": config,
            "resolved_params": resolved_params if 'resolved_params' in locals() else {},
            "traceback": traceback.format_exc()
        }
        
        error_path = args.model_dir / "training_error.json"
        try:
            with open(error_path, "w") as f:
                json.dump(error_info, f, indent=2, default=str)
        except:
            pass
        
        raise
        
    finally:
        # Cleanup
        try:
            writer.close()
        except:
            pass
        
        # Memory cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def export_to_onnx(
    model: nn.Module,
    input_dim: int,
    device: torch.device,
    model_dir: Path = None,
    opset_version: int = None,
    config: Optional[Dict] = None
) -> Optional[Path]:
    """
    Export model to ONNX format with memory protection and comprehensive error handling.
    Features:
    - Memory-efficient export with cleanup safeguards
    - Chunked operations for large models
    - Automatic memory recovery
    - Detailed memory diagnostics
    """
    # Initialize memory tracking
    memory_stats = {
        'initial_ram': psutil.virtual_memory().used / (1024**3),
        'initial_vram': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    }
    
    def log_memory_usage(stage: str):
        """Log current memory usage at different stages"""
        mem = psutil.virtual_memory()
        memory_stats[f'{stage}_ram'] = mem.used / (1024**3)
        if torch.cuda.is_available():
            memory_stats[f'{stage}_vram'] = torch.cuda.memory_allocated() / (1024**3)
        logger.debug(f"Memory at {stage}: RAM {memory_stats[f'{stage}_ram']:.2f}GB | "
                    f"VRAM {memory_stats.get(f'{stage}_vram', 0):.2f}GB")

    try:
        # Load configuration with memory limits
        if config is None:
            try:
                config = get_current_config()
            except Exception:
                config = {}
        
        system_config = config.get('system', {})
        export_config = system_config.get('onnx_export', {})
        
        # Apply memory-aware configuration
        max_ram_usage = export_config.get('max_ram_gb', 8)
        max_vram_usage = export_config.get('max_vram_gb', 2) if torch.cuda.is_available() else 0
        chunk_size = export_config.get('chunk_size', min(128, input_dim))  # Smaller chunks for memory safety
        
        # Verify system resources before starting
        current_ram = psutil.virtual_memory().used / (1024**3)
        current_vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        if current_ram > max_ram_usage * 0.8:  # 80% threshold
            raise MemoryError(f"High RAM usage detected: {current_ram:.1f}/{max_ram_usage}GB")
        if current_vram > max_vram_usage * 0.8:
            raise MemoryError(f"High VRAM usage detected: {current_vram:.1f}/{max_vram_usage}GB")

        # Apply configuration with parameter precedence
        if model_dir is None:
            model_dir = Path(system_config.get('model_dir', DEFAULT_MODEL_DIR))
        if opset_version is None:
            opset_version = export_config.get('opset_version', 14)
        
        model.eval()
        onnx_path = model_dir / "autoencoder_ids.onnx"
        
        logger.info(f"Exporting model to ONNX format at {onnx_path}")
        log_memory_usage('pre_export')

        # Memory protection context
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            # 1. Prepare directory and inputs with memory cleanup
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy input in chunks to reduce memory spikes
                dummy_input = None
                try:
                    chunks = []
                    for i in range(0, input_dim, chunk_size):
                        chunk = torch.randn(1, min(chunk_size, input_dim - i), device='cpu')
                        chunks.append(chunk)
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    dummy_input = torch.cat(chunks, dim=1).to(device)
                    del chunks
                except Exception as e:
                    if dummy_input is not None:
                        del dummy_input
                    raise MemoryError(f"Input creation failed: {str(e)}")

                log_memory_usage('post_input_creation')

                # 2. Export with memory monitoring
                try:
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        opset_version=opset_version,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes=export_config.get('dynamic_axes', {'input': {0: 'batch_size'}}),
                        do_constant_folding=export_config.get('constant_folding', True),
                        export_params=True,
                        verbose=export_config.get('verbose', False),
                        training=torch.onnx.TrainingMode.EVAL,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX
                        #use_external_data_format=input_dim > 2048  # For large models
                    )
                finally:
                    del dummy_input
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                log_memory_usage('post_export')

                # 3. Validation with memory protection
                validation_result = None
                if export_config.get('runtime_validation', True) and ONNXRUNTIME_AVAILABLE:
                    validation_result = validate_onnx_model(
                        model, onnx_path, device, 
                        tolerance=export_config.get('validation_tolerance', 1e-5),
                        strict=export_config.get('strict_validation', False)
                    )
                
                # 4. Generate metadata with memory info
                metadata = create_export_metadata(
                    model, onnx_path, config, validation_result, memory_stats
                )
                
                save_metadata(metadata, model_dir / "onnx_export_metadata.json")
                logger.info(f"Export completed: {onnx_path}")
                return onnx_path

            except Exception as e:
                # Cleanup any partial files
                if onnx_path.exists():
                    try:
                        onnx_path.unlink()
                    except:
                        pass
                raise

    except MemoryError as e:
        logger.error(f"Memory error during export: {str(e)}")
        log_memory_usage('error')
        if export_config.get('fail_silently', False):
            return None
        raise
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        if export_config.get('fail_silently', False):
            return None
        raise RuntimeError(f"ONNX export failed: {str(e)}") from e
    finally:
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def validate_onnx_model(
    model: nn.Module, 
    onnx_path: Path, 
    device: torch.device,
    tolerance: float = 1e-5,
    strict: bool = False
) -> Dict:
    """Memory-safe ONNX model validation"""
    validation_result = {
        'status': 'skipped',
        'max_difference': None,
        'error': None
    }
    
    try:
        # Load in memory-safe way
        with warnings.catch_warnings(), torch.no_grad():
            # 1. Verify ONNX model structure
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # 2. Runtime validation
            ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            
            # Create test input in chunks
            input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
            test_input = torch.randn(1, input_shape, device='cpu').numpy()
            
            # Run comparison
            ort_output = ort_session.run(None, {'input': test_input})[0]
            with torch.no_grad():
                torch_output = model(torch.from_numpy(test_input).to(device).cpu().numpy())
            
            # Validate
            if ort_output.shape != torch_output.shape:
                error_msg = f"Shape mismatch: ONNX {ort_output.shape} vs PyTorch {torch_output.shape}"
                if strict:
                    raise RuntimeError(error_msg)
                validation_result.update({
                    'status': 'failed',
                    'error': error_msg
                })
            else:
                max_diff = np.abs(ort_output - torch_output).max()
                validation_result['max_difference'] = float(max_diff)
                if max_diff > tolerance:
                    error_msg = f"Numerical difference {max_diff:.2e} > tolerance {tolerance:.1e}"
                    if strict:
                        raise RuntimeError(error_msg)
                    validation_result.update({
                        'status': 'warning',
                        'error': error_msg
                    })
                else:
                    validation_result['status'] = 'passed'
    
    except Exception as e:
        validation_result.update({
            'status': 'failed',
            'error': str(e)
        })
        if strict:
            raise
    
    return validation_result

def create_export_metadata(
    model: nn.Module,
    onnx_path: Path,
    config: Dict,
    validation_result: Optional[Dict],
    memory_stats: Dict
) -> Dict:
    """Generate comprehensive export metadata"""
    return {
        "export_timestamp": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "input_dim": getattr(model, 'input_dim', 'unknown'),
        "opset_version": config.get('system', {}).get('onnx_export', {}).get('opset_version', 14),
        "file_size_mb": onnx_path.stat().st_size / (1024**2),
        "memory_usage": memory_stats,
        "validation": validation_result or {'status': 'not_performed'},
        "system": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
            "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        },
        "config": config.get('system', {}).get('onnx_export', {})
    }

def save_metadata(metadata: Dict, path: Path) -> None:
    """Safely save metadata with error handling"""
    try:
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")

def save_tensorboard_data(
    writer: SummaryWriter,
    save_dir: Path = None,
    filename_suffix: str = "",
    config: Optional[Dict] = None
) -> Dict[str, Path]:
    """Save TensorBoard event data in multiple readable formats with configuration integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    monitoring_config = config.get('monitoring', {})
    tensorboard_config = monitoring_config.get('tensorboard', {})
    
    # Apply configuration with parameter precedence
    if save_dir is None:
        save_dir = Path(monitoring_config.get('tensorboard_dir', TB_DIR))
    
    export_formats = tensorboard_config.get('export_formats', ['json', 'csv'])
    include_histograms = tensorboard_config.get('include_histograms', False)
    include_images = tensorboard_config.get('include_images', False)
    
    logger.info(f"Saving TensorBoard data to: {save_dir}")
    logger.info(f"Export formats: {export_formats}")
    
    saved_files = {}
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("Could not import EventAccumulator - skipping TensorBoard data export")
        return saved_files

    try:
        # Ensure save directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the event file path from the SummaryWriter
        event_files = list(Path(writer.log_dir).glob('events.out.tfevents.*'))
        if not event_files:
            logger.warning("No TensorBoard event files found")
            return saved_files
        
        # Use the most recent event file
        event_file = max(event_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Processing event file: {event_file}")
        
        # Load the event data with size guidance
        size_guidance = {
            'scalars': tensorboard_config.get('max_scalars', 1000),
            'histograms': tensorboard_config.get('max_histograms', 100) if include_histograms else 0,
            'images': tensorboard_config.get('max_images', 10) if include_images else 0,
        }
        
        event_acc = EventAccumulator(str(event_file), size_guidance=size_guidance)
        event_acc.Reload()
        
        # Extract scalar data
        tb_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'source_file': str(event_file),
                'tensorflow_version': getattr(event_acc, '_tensorflow_version', 'unknown'),
                'export_config': tensorboard_config
            },
            'scalars': {},
            'tags_metadata': {}
        }
        
        # Process scalar data
        scalar_tags = event_acc.Tags().get('scalars', [])
        logger.info(f"Found {len(scalar_tags)} scalar tags")
        
        for tag in scalar_tags:
            try:
                events = event_acc.Scalars(tag)
                tb_data['scalars'][tag] = {
                    'steps': [e.step for e in events],
                    'values': [e.value for e in events],
                    'wall_times': [e.wall_time for e in events],
                    'count': len(events)
                }
                tb_data['tags_metadata'][tag] = {
                    'type': 'scalar',
                    'first_step': events[0].step if events else 0,
                    'last_step': events[-1].step if events else 0,
                    'min_value': min(e.value for e in events) if events else 0,
                    'max_value': max(e.value for e in events) if events else 0
                }
            except Exception as e:
                logger.warning(f"Failed to process scalar tag '{tag}': {str(e)}")
        
        # Process histograms if requested
        if include_histograms:
            histogram_tags = event_acc.Tags().get('histograms', [])
            if histogram_tags:
                tb_data['histograms'] = {}
                logger.info(f"Processing {len(histogram_tags)} histogram tags")
                
                for tag in histogram_tags:
                    try:
                        events = event_acc.Histograms(tag)
                        tb_data['histograms'][tag] = [
                            {
                                'step': e.step,
                                'wall_time': e.wall_time,
                                'bucket_limits': e.histogram_value.bucket_limit,
                                'bucket_counts': e.histogram_value.bucket,
                                'min': e.histogram_value.min,
                                'max': e.histogram_value.max,
                                'sum': e.histogram_value.sum,
                                'count': e.histogram_value.num
                            }
                            for e in events
                        ]
                    except Exception as e:
                        logger.warning(f"Failed to process histogram tag '{tag}': {str(e)}")
        
        # Save in requested formats
        base_filename = f"tensorboard_data{filename_suffix}"
        
        # JSON format
        if 'json' in export_formats:
            json_path = save_dir / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(tb_data, f, indent=2, default=str)
            saved_files['json'] = json_path
            logger.info(f"[INFO] Saved JSON data: {json_path}")
        
        # CSV format for scalars
        if 'csv' in export_formats and tb_data['scalars']:
            csv_path = save_dir / f"{base_filename}.csv"
            try:
                # Create a flattened DataFrame
                csv_data = []
                for tag, values in tb_data['scalars'].items():
                    for step, value, wall_time in zip(values['steps'], values['values'], values['wall_times']):
                        csv_data.append({
                            'tag': tag,
                            'step': step,
                            'value': value,
                            'wall_time': wall_time,
                            'timestamp': datetime.fromtimestamp(wall_time).isoformat()
                        })
                
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                saved_files['csv'] = csv_path
                logger.info(f"[INFO] Saved CSV data: {csv_path}")
            except Exception as e:
                logger.warning(f"Could not save CSV format: {str(e)}")
        
        # Parquet format for efficient storage
        if 'parquet' in export_formats and tb_data['scalars']:
            try:
                parquet_path = save_dir / f"{base_filename}.parquet"
                # Similar to CSV but save as Parquet
                parquet_data = []
                for tag, values in tb_data['scalars'].items():
                    for step, value, wall_time in zip(values['steps'], values['values'], values['wall_times']):
                        parquet_data.append({
                            'tag': tag,
                            'step': step,
                            'value': value,
                            'wall_time': wall_time
                        })
                
                df = pd.DataFrame(parquet_data)
                df.to_parquet(parquet_path, index=False)
                saved_files['parquet'] = parquet_path
                logger.info(f"[INFO] Saved Parquet data: {parquet_path}")
            except Exception as e:
                logger.warning(f"Could not save Parquet format: {str(e)}")
        
        # Save summary statistics
        if tensorboard_config.get('save_summary', True):
            summary = {
                'export_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'total_scalar_tags': len(tb_data['scalars']),
                    'total_data_points': sum(len(v['values']) for v in tb_data['scalars'].values()),
                    'file_size_mb': event_file.stat().st_size / 1024 / 1024,
                    'time_range': {
                        'first_event': min(
                            min(v['wall_times']) for v in tb_data['scalars'].values() 
                            if v['wall_times']
                        ) if tb_data['scalars'] else 0,
                        'last_event': max(
                            max(v['wall_times']) for v in tb_data['scalars'].values() 
                            if v['wall_times']
                        ) if tb_data['scalars'] else 0
                    },
                    'exported_formats': list(saved_files.keys()),
                    'tags_summary': tb_data['tags_metadata']
                }
            }
            
            summary_path = save_dir / f"{base_filename}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            saved_files['summary'] = summary_path
            logger.info(f"[INFO] Saved summary: {summary_path}")
        
        logger.info(f"[INFO] TensorBoard data export complete ({len(saved_files)} files)")
        return saved_files
        
    except Exception as e:
        logger.error(f"Failed to save TensorBoard data: {str(e)}")
        return saved_files

def hyperparameter_search(
    trial: optuna.Trial, 
    base_args: argparse.Namespace,
    config: Optional[Dict] = None
) -> float:
    """Enhanced Optuna hyperparameter optimization with comprehensive configuration integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    hpo_config = config.get('hyperparameter_optimization', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    # Define search space with configuration-aware bounds
    search_space = hpo_config.get('search_space', {})
    
    # Dynamic parameter bounds based on problem size
    max_encoding_dim = min(
        search_space.get('encoding_dim_max', 64),
        getattr(base_args, 'features', 20) // 2
    )
    
    params = {
        # Core model parameters
        "encoding_dim": trial.suggest_int(
            "encoding_dim", 
            search_space.get('encoding_dim_min', 4), 
            max_encoding_dim
        ),
        "num_hidden_layers": trial.suggest_int(
            "num_hidden_layers", 
            search_space.get('hidden_layers_min', 1), 
            search_space.get('hidden_layers_max', 3)
        ),
        
        # Training parameters
        "lr": trial.suggest_float(
            "lr", 
            search_space.get('lr_min', 1e-5), 
            search_space.get('lr_max', 1e-2), 
            log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", 
            search_space.get('batch_sizes', [32, 64, 128, 256])
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            search_space.get('weight_decay_min', 1e-6),
            search_space.get('weight_decay_max', 1e-2),
            log=True
        ),
        
        # Regularization parameters
        "dropout_base": trial.suggest_float(
            "dropout_base", 
            search_space.get('dropout_min', 0.1), 
            search_space.get('dropout_max', 0.5)
        ),
        
        # Model architecture parameters
        "activation": trial.suggest_categorical(
            "activation",
            search_space.get('activations', ['relu', 'leaky_relu', 'gelu'])
        ),
        "normalization": trial.suggest_categorical(
            "normalization",
            search_space.get('normalizations', [None, 'batch', 'layer'])
        ),
        
        # Security parameters
        "percentile": trial.suggest_int(
            "percentile", 
            search_space.get('percentile_min', 90), 
            search_space.get('percentile_max', 99)
        )
    }
    
    # Generate derived parameters
    # Architecture scaling
    base_hidden_size = search_space.get('base_hidden_size', 128)
    decay_factor = search_space.get('hidden_decay_factor', 0.5)
    
    params["hidden_dims"] = [
        max(16, int(base_hidden_size * (decay_factor ** i)))
        for i in range(params["num_hidden_layers"])
    ]
    
    # Dropout schedule
    dropout_decay = search_space.get('dropout_decay', 0.1)
    params["dropout_rates"] = [
        max(0.05, params["dropout_base"] * (1 - dropout_decay * i))
        for i in range(params["num_hidden_layers"])
    ]
    
    # Activation parameters
    if params["activation"] == "leaky_relu":
        params["activation_param"] = trial.suggest_float(
            "leaky_relu_slope", 
            search_space.get('leaky_relu_min', 0.01),
            search_space.get('leaky_relu_max', 0.3)
        )
    else:
        params["activation_param"] = model_config.get('activation_param', ACTIVATION_PARAM)

    # Create trial-specific configuration
    trial_config = {
        "training": {
            "batch_size": params["batch_size"],
            "learning_rate": params["lr"],
            "weight_decay": params["weight_decay"],
            "epochs": hpo_config.get('trial_epochs', 30),
            "patience": hpo_config.get('trial_patience', 5),
            "gradient_clip": training_config.get('gradient_clip', GRADIENT_CLIP),
            "mixed_precision": training_config.get('mixed_precision', MIXED_PRECISION)
        },
        "model": {
            "encoding_dim": params["encoding_dim"],
            "hidden_dims": params["hidden_dims"],
            "dropout_rates": params["dropout_rates"],
            "activation": params["activation"],
            "activation_param": params["activation_param"],
            "normalization": params["normalization"],
            "skip_connection": model_config.get('skip_connection', True)
        },
        "security": {
            "percentile": params["percentile"]
        },
        "monitoring": {
            "tensorboard_logging": False,  # Disable for HPO trials
            "checkpoint_frequency": 999999,  # Minimal checkpointing
            "log_frequency": 10
        }
    }

    # Create trial args
    search_args = argparse.Namespace(**vars(base_args))
    
    # Update with trial parameters
    for key, value in params.items():
        setattr(search_args, key, value)
    
    # Set trial-specific parameters
    search_args.model_dir = DEFAULT_MODEL_DIR / "hpo_trials" / f"trial_{trial.number}"
    search_args.epochs = trial_config["training"]["epochs"]
    search_args.patience = trial_config["training"]["patience"]
    search_args.non_interactive = True
    search_args.export_onnx = False
    search_args.debug = False

    logger.info(f"Starting trial {trial.number} with params: {params}")

    try:
        # Ensure trial directory exists
        search_args.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial configuration
        trial_config_path = search_args.model_dir / "trial_config.json"
        with open(trial_config_path, "w") as f:
            json.dump(trial_config, f, indent=2, default=str)
        
        # Update global config for this trial
        original_config = get_current_config()
        trial_full_config = deep_update(original_config.copy(), trial_config)
        update_global_config(trial_full_config)
        
        # Run training with these parameters
        trial_result = train_model(search_args)
        
        # Extract metrics
        best_val_loss = trial_result["training"]["best_val_loss"]
        test_loss = trial_result["evaluation"]["test_loss"]
        anomaly_rate = trial_result["evaluation"]["anomaly_detection_rate"]
        total_params = trial_result["model"]["total_parameters"]
        
        # Calculate composite score if configured
        scoring_config = hpo_config.get('scoring', {})
        if scoring_config.get('use_composite_score', False):
            # Multi-objective optimization
            val_weight = scoring_config.get('validation_weight', 0.7)
            test_weight = scoring_config.get('test_weight', 0.2)
            complexity_weight = scoring_config.get('complexity_weight', 0.1)
            
            # Normalize complexity penalty
            max_params = scoring_config.get('max_params_penalty', 100000)
            complexity_penalty = (total_params / max_params) * complexity_weight
            
            composite_score = (
                val_weight * best_val_loss + 
                test_weight * test_loss + 
                complexity_penalty
            )
            objective_value = composite_score
        else:
            objective_value = best_val_loss
        
        # Store trial metadata
        trial.set_user_attr("test_loss", test_loss)
        trial.set_user_attr("anomaly_detection_rate", anomaly_rate)
        trial.set_user_attr("total_parameters", total_params)
        trial.set_user_attr("model_size_mb", trial_result["model"]["model_size_mb"])
        trial.set_user_attr("training_duration", trial_result["experiment"]["duration_seconds"])
        trial.set_user_attr("final_epoch", trial_result["training"]["final_epoch"])
        trial.set_user_attr("config", trial_config)
        
        # Save trial summary
        trial_summary = {
            "trial_number": trial.number,
            "objective_value": objective_value,
            "parameters": params,
            "metrics": {
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
                "anomaly_detection_rate": anomaly_rate,
                "total_parameters": total_params
            },
            "config": trial_config,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = search_args.model_dir / "trial_summary.json"
        with open(summary_path, "w") as f:
            json.dump(trial_summary, f, indent=2, default=str)
        
        logger.info(f"Trial {trial.number} completed - Objective: {objective_value:.5f}")
        
        return objective_value
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("failed", True)
        return float('inf')
        
    finally:
        # Restore original configuration
        try:
            update_global_config(original_config)
        except:
            pass
        
        # Clean up trial artifacts if not in debug mode
        cleanup_trials = hpo_config.get('cleanup_trials', True)
        if cleanup_trials and not getattr(base_args, 'debug', False):
            try:
                import shutil
                if search_args.model_dir.exists():
                    # Keep only essential files
                    keep_files = ['trial_summary.json', 'trial_config.json']
                    for file in search_args.model_dir.glob("*"):
                        if file.name not in keep_files:
                            if file.is_file():
                                file.unlink()
                            elif file.is_dir():
                                shutil.rmtree(file)
            except Exception as e:
                logger.warning(f"Could not clean up trial directory: {str(e)}")

def setup_hyperparameter_optimization(
    args: argparse.Namespace,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Configure and run comprehensive hyperparameter optimization with enhanced integration."""
    # Load configuration
    if config is None:
        try:
            config = get_current_config()
        except Exception:
            config = {}
    
    hpo_config = config.get('hyperparameter_optimization', {})
    
    # Apply configuration with argument precedence
    n_trials = getattr(args, 'hpo_trials', None) or hpo_config.get('n_trials', 100)
    timeout = getattr(args, 'hpo_timeout', None) or hpo_config.get('timeout_seconds', 0)
    study_name = hpo_config.get('study_name', f"autoencoder_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    logger.info("═" * 80)
    logger.info("HYPERPARAMETER OPTIMIZATION SETUP")
    logger.info("═" * 80)
    logger.info(f"Study name: {study_name}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Timeout: {timeout if timeout > 0 else 'None'} seconds")
    
    # Configure sampler
    sampler_config = hpo_config.get('sampler', {})
    sampler_type = sampler_config.get('type', 'TPE')
    
    if sampler_type == 'TPE':
        sampler = optuna.samplers.TPESampler(
            seed=sampler_config.get('seed', 42),
            consider_prior=sampler_config.get('consider_prior', True),
            prior_weight=sampler_config.get('prior_weight', 1.0),
            consider_magic_clip=sampler_config.get('consider_magic_clip', True),
            consider_endpoints=sampler_config.get('consider_endpoints', False),
            n_startup_trials=sampler_config.get('n_startup_trials', 10),
            n_ei_candidates=sampler_config.get('n_ei_candidates', 24),
            multivariate=sampler_config.get('multivariate', False)
        )
    elif sampler_type == 'Random':
        sampler = optuna.samplers.RandomSampler(
            seed=sampler_config.get('seed', 42)
        )
    elif sampler_type == 'CmaEs':
        sampler = optuna.samplers.CmaEsSampler(
            seed=sampler_config.get('seed', 42),
            n_startup_trials=sampler_config.get('n_startup_trials', 1)
        )
    else:
        logger.warning(f"Unknown sampler type '{sampler_type}', using TPE")
        sampler = optuna.samplers.TPESampler(seed=42)
    
    # Configure pruner
    pruner_config = hpo_config.get('pruner', {})
    pruner_type = pruner_config.get('type', 'Median')
    
    if pruner_type == 'Median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.get('n_startup_trials', 5),
            n_warmup_steps=pruner_config.get('n_warmup_steps', 10),
            interval_steps=pruner_config.get('interval_steps', 1)
        )
    elif pruner_type == 'Hyperband':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=pruner_config.get('min_resource', 1),
            max_resource=pruner_config.get('max_resource', 30),
            reduction_factor=pruner_config.get('reduction_factor', 3)
        )
    elif pruner_type == 'None':
        pruner = optuna.pruners.NopPruner()
    else:
        logger.warning(f"Unknown pruner type '{pruner_type}', using Median")
        pruner = optuna.pruners.MedianPruner()
    
    logger.info(f"Sampler: {type(sampler).__name__}")
    logger.info(f"Pruner: {type(pruner).__name__}")
    
    # Setup study storage if configured
    storage_config = hpo_config.get('storage', {})
    storage = None
    if storage_config.get('enabled', False):
        storage_url = storage_config.get('url', f'sqlite:///{DEFAULT_MODEL_DIR}/hpo_studies/study.db')
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=storage_config.get('heartbeat_interval', 60),
            grace_period=storage_config.get('grace_period', 120)
        )
        logger.info(f"Using persistent storage: {storage_url}")
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage_config.get('load_if_exists', False)
    )
    
    # Configure logging
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.setLevel(getattr(logging, hpo_config.get('log_level', 'INFO')))
    
    # Add callbacks
    callbacks = []
    
    # Progress callback
    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(
                f"Trial {trial.number:3d} complete | "
                f"Value: {trial.value:.5f} | "
                f"Best: {study.best_value:.5f} | "
                f"Params: {trial.params}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"Trial {trial.number:3d} pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            logger.warning(f"Trial {trial.number:3d} failed")
    
    callbacks.append(progress_callback)
    
    # Early stopping callback if configured
    early_stopping_config = hpo_config.get('early_stopping', {})
    if early_stopping_config.get('enabled', False):
        def early_stopping_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if len(study.trials) >= early_stopping_config.get('min_trials', 20):
                # Check if no improvement in last N trials
                recent_trials = study.trials[-early_stopping_config.get('patience', 10):]
                recent_values = [t.value for t in recent_trials if t.state == optuna.trial.TrialState.COMPLETE]
                
                if recent_values and min(recent_values) > study.best_value:
                    logger.info("Early stopping triggered - no improvement in recent trials")
                    study.stop()
        
        callbacks.append(early_stopping_callback)
    
    # Setup study directories
    study_dir = DEFAULT_MODEL_DIR / "hpo_studies" / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study configuration
    study_config = {
        "study_name": study_name,
        "configuration": hpo_config,
        "sampler": {
            "type": type(sampler).__name__,
            "config": sampler_config
        },
        "pruner": {
            "type": type(pruner).__name__,
            "config": pruner_config
        },
        "n_trials": n_trials,
        "timeout": timeout,
        "timestamp": datetime.now().isoformat()
    }
    
    config_path = study_dir / "study_config.json"
    with open(config_path, "w") as f:
        json.dump(study_config, f, indent=2, default=str)
    
    logger.info("─" * 60)
    logger.info("STARTING OPTIMIZATION")
    logger.info("─" * 60)
    
    try:
        # Run optimization
        study.optimize(
            lambda trial: hyperparameter_search(trial, args, config),
            n_trials=n_trials,
            timeout=timeout if timeout > 0 else None,
            gc_after_trial=True,
            callbacks=callbacks,
            show_progress_bar=hpo_config.get('show_progress', True)
        )
        
        logger.info("═" * 80)
        logger.info("HYPERPARAMETER OPTIMIZATION COMPLETE")
        logger.info("═" * 80)
        
        # Analyze results
        best_trial = study.best_trial
        logger.info(f"\nBest trial (#{best_trial.number}):")
        logger.info(f"  Objective value: {best_trial.value:.5f}")
        logger.info(f"  Parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key:>20}: {value}")
        
        # Additional metrics from best trial
        if best_trial.user_attrs:
            logger.info(f"  Additional metrics:")
            for key, value in best_trial.user_attrs.items():
                if key not in ["config", "error", "failed"]:
                    logger.info(f"    {key:>20}: {value}")
        
        # Save study results
        study_path = study_dir / "study_results.pkl"
        joblib.dump(study, study_path)
        logger.info(f"\nStudy saved to: {study_path}")
        
        # Generate optimization report
        report = {
            "study_summary": {
                "study_name": study_name,
                "n_trials": len(study.trials),
                "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "n_failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                "best_value": study.best_value,
                "best_trial_number": best_trial.number,
                "optimization_duration": study.trials[-1].datetime_complete - study.trials[0].datetime_start if study.trials else None
            },
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs
            },
            "configuration": study_config,
            "timestamp": datetime.now().isoformat()
        }
        
        report_path = study_dir / "optimization_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations if configured
        if hpo_config.get('generate_plots', True):
            try:
                plot_dir = study_dir / "plots"
                plot_dir.mkdir(exist_ok=True)
                
                # Optimization history
                fig = vis.plot_optimization_history(study)
                fig.write_html(plot_dir / "optimization_history.html")
                
                # Parameter importances
                fig = vis.plot_param_importances(study)
                fig.write_html(plot_dir / "param_importances.html")
                
                # Parallel coordinate plot
                fig = vis.plot_parallel_coordinate(study)
                fig.write_html(plot_dir / "parallel_coordinate.html")
                
                # Slice plot
                fig = vis.plot_slice(study)
                fig.write_html(plot_dir / "slice_plot.html")
                
                logger.info(f"[INFO] Plots saved to: {plot_dir}")
                
            except Exception as e:
                logger.warning(f"Failed to generate plots: {str(e)}")
        
        # Return best configuration for final training
        best_config = best_trial.user_attrs.get("config", {})
        
        logger.info("═" * 80)
        
        return {
            "best_params": best_trial.params,
            "best_value": best_trial.value,
            "best_config": best_config,
            "study": study,
            "study_path": study_path,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        
        # Save error information
        error_info = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "configuration": study_config,
            "completed_trials": len(study.trials) if 'study' in locals() else 0
        }
        
        error_path = study_dir / "optimization_error.json"
        with open(error_path, "w") as f:
            json.dump(error_info, f, indent=2, default=str)
        
        raise
        
    finally:
        # Cleanup
        optuna_logger.handlers.clear()

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

def show_banner() -> None:
    """Display the application banner"""
    # ASCII art banner
    console.print("\n" , Panel.fit(
        """
                            
⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣠⣶⠚⠛⠿⠷⠶⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⣴⠟⠉⠀⠀⢠⡄⠀⠀⠀⠀⠀⠉⠙⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⡴⠛⠁⠀⠀⠀⠀⠘⣷⣴⠏⠀⠀⣠⡄⠀⠀⢨⡇⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠺⣇⠀⠀⠀⠀⠀⠀⠀⠘⣿⠀⠀⠘⣻⣻⡆⠀⠀⠙⠦⣄⣀⠀⠀⠀⠀
⠀⠀⠀⢰⡟⢷⡄⠀⠀⠀⠀⠀⠀⢸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢻⠶⢤⡀
⠀⠀⠀⣾⣇⠀⠻⣄⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣀⣴⣿
⠀⠀⢸⡟⠻⣆⠀⠈⠳⢄⡀⠀⠀⡼⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠶⢤⣬⡿⠁
⠀⢀⣿⠃⠀⠹⣆⠀⠀⠀⠙⠓⠿⢧⡀⠀⢠⡴⣶⣶⣒⣋⣀⣀⣤⣶⣶⠟⠁⠀
⠀⣼⡏⠀⠀⠀⠙⠀⠀⠀⠀⠀⠀⠀⠙⠳⠶⠤⠵⣶⠒⠚⠻⠿⠋⠁⠀⠀⠀⠀
⢰⣿⡇⠀⠀⠀⠀⠀⠀⠀⣆⠀⠀⠀⠀⠀⠀⠀⢠⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠘⣦⡀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⡄⠀⠀⠀⠀⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢷⡀⠀⠀⠀⢸⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⠇⠀⠀⠀⠀⠀⠀⠀

    """,
        style="bold cyan", 
        title="[bold yellow]GreyChamp | IDS[/]", 
        subtitle="[magenta]DEEP LEARNING SUITE[/]",
        border_style="bold blue",
        box=box.DOUBLE,
        padding=(1, 2)
    ))

    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 40)
    print(Fore.GREEN + Style.BRIGHT + "  - Interactive Mode -  ".center(40))
    print(Fore.CYAN + Style.BRIGHT + "=" * 40 + Style.RESET_ALL)

def reset_config() -> None:
    """Reset configuration to default values with comprehensive cleanup."""
    try:
        config = get_current_config()
        logger.info("Resetting configuration to defaults...")
        
        # Create fresh default configuration
        default_config = {
            "training": {
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "patience": 100,
                "gradient_clip": 1.0,
                "mixed_precision": True,
                "num_workers": min(4, os.cpu_count() or 1),
                "adam_betas": (0.9, 0.999),
                "adam_eps": 1e-8,
                "lr_patience": 2,
                "lr_factor": 0.5,
                "min_lr": 1e-7
            },
            "model": {
                "model_type": "EnhancedAutoencoder",
                "encoding_dim": 10,
                "hidden_dims": [128, 64],
                "dropout_rates": [0.2, 0.15],
                "activation": 'leaky_relu',
                "activation_param": 0.2,
                "normalization": 'batch',
                "legacy_mode": False,
                "skip_connection": True,
                "use_attention": False
            },
            "data": {
                "features": 20,
                "normal_samples": 8000,
                "attack_samples": 2000,
                "use_real_data": False,
                "validation_split": 0.2,
                "data_path": str(DEFAULT_MODEL_DIR / "preprocessed_dataset.csv"),
                "artifacts_path": str(DEFAULT_MODEL_DIR / "preprocessing_artifacts.pkl"),
                "shuffle": True,
                "pin_memory": True
            },
            "security": {
                "percentile": 95,
                "anomaly_threshold_strategy": "percentile",
                "threshold_validation": True
            },
            "system": {
                "model_dir": str(DEFAULT_MODEL_DIR),
                "log_dir": str(LOG_DIR),
                "config_dir": str(CONFIG_DIR),
                "export_onnx": False,
                "non_interactive": False,
                "debug": False,
                "cuda_optimizations": True,
                "onnx_export": {
                    "opset_version": 14,
                    "dynamic_axes": True,
                    "constant_folding": True,
                    "optimize_for_mobile": False,
                    "runtime_validation": True,
                    "validation_tolerance": 1e-5,
                    "verbose": False
                }
            },
            "monitoring": {
                "tensorboard_logging": True,
                "tensorboard_dir": str(TB_DIR),
                "log_frequency": 1,
                "checkpoint_frequency": 10,
                "metrics_frequency": 10,
                "save_checkpoints": True,
                "tensorboard": {
                    "export_formats": ["json", "csv"],
                    "include_histograms": False,
                    "include_images": False,
                    "max_scalars": 1000,
                    "max_histograms": 100,
                    "max_images": 10,
                    "save_summary": True
                }
            },
            "hyperparameter_optimization": {
                "enabled": False,
                "n_trials": 50,
                "timeout_seconds": 3600,
                "trial_epochs": 30,
                "trial_patience": 5,
                "cleanup_trials": True,
                "generate_plots": True,
                "search_space": {
                    "encoding_dim_min": 4,
                    "encoding_dim_max": 64,
                    "hidden_layers_min": 1,
                    "hidden_layers_max": 3,
                    "lr_min": 1e-5,
                    "lr_max": 1e-2,
                    "batch_sizes": [32, 64, 128, 256],
                    "weight_decay_min": 1e-6,
                    "weight_decay_max": 1e-2,
                    "dropout_min": 0.1,
                    "dropout_max": 0.5,
                    "activations": ["relu", "leaky_relu", "gelu"],
                    "normalizations": [None, "batch", "layer"],
                    "percentile_min": 90,
                    "percentile_max": 99
                },
                "sampler": {
                    "type": "TPE",
                    "seed": 42,
                    "consider_prior": True,
                    "prior_weight": 1.0,
                    "consider_magic_clip": True,
                    "consider_endpoints": False,
                    "n_startup_trials": 10,
                    "n_ei_candidates": 24,
                    "multivariate": False
                },
                "pruner": {
                    "type": "Median",
                    "n_startup_trials": 5,
                    "n_warmup_steps": 10,
                    "interval_steps": 1
                },
                "scoring": {
                    "use_composite_score": False,
                    "validation_weight": 0.7,
                    "test_weight": 0.2,
                    "complexity_weight": 0.1,
                    "max_params_penalty": 100000
                },
                "early_stopping": {
                    "enabled": False,
                    "min_trials": 20,
                    "patience": 10
                },
                "storage": {
                    "enabled": False,
                    "url": f"sqlite:///{DEFAULT_MODEL_DIR}/hpo_studies/study.db",
                    "load_if_exists": False,
                    "heartbeat_interval": 60,
                    "grace_period": 120
                }
            }
        }
        
        # Update global configuration
        update_global_config(default_config)
        
        # Clear any cached configurations
        save_config(default_config)
        
        logger.info("[INFO] Configuration reset to defaults successfully")
        
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        raise

def print_main_menu():
    """Print the main menu options."""
    # Print menu  options
    print(Fore.YELLOW + Style.BRIGHT + "\nMain Menu:")
    print(Fore.WHITE + Style.BRIGHT + "1. Model Training")
    print(Fore.WHITE + Style.BRIGHT + "2. Hyperparameter Optimization")
    print(Fore.WHITE + Style.BRIGHT + "3. Model Architecture Comparison")
    print(Fore.WHITE + Style.BRIGHT + "4. Configuration Management")
    print(Fore.WHITE + Style.BRIGHT + "5. System Information")
    print(Fore.WHITE + Style.BRIGHT + "6. Performance Benchmark")
    print(Fore.WHITE + Style.BRIGHT + "7. Model Analysis & Visualization")
    print(Fore.WHITE + Style.BRIGHT + "8. Advanced Tools")
    print(Fore.RED + Style.BRIGHT + "9. Exit")

def model_training_menu(config: Optional[Dict[str, Any]] = get_current_config()):
    """Menu for model training options"""
    while True:
        # Get current configuration
        #config = get_current_config()
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        
        print(Fore.YELLOW + "\nModel Training Options:")
        print(Fore.CYAN + "1. Train with Current Configuration" +          f"Epochs: {training_config.get('epochs', DEFAULT_EPOCHS)}")
        print(Fore.CYAN + "2. Train with Synthetic Data" +                 f"Samples: {data_config.get('normal_samples', NORMAL_SAMPLES)}")
        print(Fore.CYAN + "3. Train with Real Data" +                      f"Source: {data_config.get('data_path', 'Default')}")
        print(Fore.CYAN + "4. Quick Training (Fast Test)" +                f"Model: {model_config.get('model_type', 'Enhanced')}")
        print(Fore.CYAN + "5. Custom Training Parameters" +                 "(Interactive Setup)")
        print(Fore.CYAN + "6. Select Preset Configuration" +               f"Available: {len(PRESET_CONFIGS)}")
        print(Fore.CYAN + "7. Stability Test" +                             "(10 Epochs Quick Test)")
        print(Fore.GREEN + "8. Back to main menu")
        
        choice = input(Fore.WHITE + "\nSelect option (1-8): ").strip()
        
        if choice == "1":
            train_model_interactive(use_current_config=True)
        elif choice == "2":
            train_model_interactive(use_real_data=False)
        elif choice == "3":
            train_model_interactive(use_real_data=True)
        elif choice == "4":
            train_model_quick()
        elif choice == "5":
            train_model_custom()
        elif choice == "6":
            select_preset_config()
        elif choice == "7":
            run_stability_test()
        elif choice == "8":
            return
        
        if choice != "8":
            console.input("\n[dim]Press Enter to continue...[/dim]")

def train_model_interactive(use_real_data: bool = None, use_current_config: bool = False):
    """Interactive wrapper for train_model with parameter setup."""
    try:
        # Create args namespace
        args = argparse.Namespace()
        
        if use_current_config:
            # Use current configuration
            config = get_current_config()
            print("\nUsing current configuration...")
            
            # Apply configuration to args
            training_config = config.get('training', {})
            model_config = config.get('model', {})
            data_config = config.get('data', {})
            
            args.epochs = training_config.get('epochs', DEFAULT_EPOCHS)
            args.batch_size = training_config.get('batch_size', DEFAULT_BATCH_SIZE)
            args.lr = training_config.get('learning_rate', LEARNING_RATE)
            args.use_real_data = data_config.get('use_real_data', False)
            args.model_type = model_config.get('model_type', 'EnhancedAutoencoder')
            args.encoding_dim = model_config.get('encoding_dim', DEFAULT_ENCODING_DIM)
            args.non_interactive = False
            
        else:
            # Interactive parameter setup
            print("\nInteractive Training Setup")
            print("Configure training parameters (press Enter for defaults):\n")
            
            # Data configuration
            if use_real_data is None:
                print("Data source options:")
                print("1. Real data")
                print("2. Synthetic data")
                data_choice = input("Select data source (1-2, default=2): ").strip()
                args.use_real_data = data_choice == "1"
            else:
                args.use_real_data = use_real_data
            
            # Training parameters
            args.epochs = int(input(f"Number of epochs (default=50): ") or 50)
            args.batch_size = int(input(f"Batch size (default=64): ") or 64)
            args.lr = float(input(f"Learning rate (default=0.001): ") or 0.001)
            args.patience = int(input(f"Early stopping patience (default=10): ") or 10)
            
            # Model configuration
            model_types = list(MODEL_VARIANTS.keys())
            print("\nModel type options:")
            for i, mtype in enumerate(model_types, 1):
                print(f"{i}. {mtype}")
            model_choice = input(f"Select model type (1-{len(model_types)}, default=1): ").strip()
            args.model_type = model_types[int(model_choice)-1] if model_choice else "EnhancedAutoencoder"
            args.encoding_dim = int(input(f"Encoding dimension (default=16): ") or 16)
            
            # Advanced options
            advanced = input("Configure advanced options? (y/N): ").lower().strip() == 'y'
            if advanced:
                args.weight_decay = float(input(f"Weight decay (default=1e-4): ") or 1e-4)
                args.mixed_precision = input("Use mixed precision? (y/N): ").lower().strip() == 'y'
                args.export_onnx = input("Export to ONNX? (y/N): ").lower().strip() == 'y'
            else:
                args.weight_decay = 1e-4
                args.mixed_precision = False
                args.export_onnx = False
            
            args.non_interactive = False
        
        # Setup directories
        args.model_dir = DEFAULT_MODEL_DIR
        args.log_dir = LOG_DIR
        args.tb_dir = TB_DIR
        
        # Display training configuration
        print("\nTraining Configuration:")
        print(f"Data Source: {'Real Data' if args.use_real_data else 'Synthetic Data'}")
        print(f"Model Type: {args.model_type}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr:.4f}")
        print(f"Encoding Dim: {args.encoding_dim}")
        
        if not input("\nProceed with training? (Y/n): ").lower().strip() in ('', 'y', 'yes'):
            print("Training cancelled")
            return
        
        # Start training
        print("\nStarting training...")
        results = train_model(args)
        
        # Display results
        if results:
            print("\nTraining Results:")
            training_results = results.get('training', {})
            evaluation_results = results.get('evaluation', {})
            model_results = results.get('model', {})
            
            print(f"Best Validation Loss: {training_results.get('best_val_loss', 'N/A'):.6f}")
            print(f"Test Loss: {evaluation_results.get('test_loss', 'N/A'):.6f}")
            print(f"Anomaly Detection Rate: {evaluation_results.get('anomaly_detection_rate', 'N/A'):.2%}")
            print(f"Model Parameters: {model_results.get('total_parameters', 'N/A'):,}")
            print(f"Training Duration: {results.get('experiment', {}).get('duration_seconds', 0)/60:.1f} min")
            print(f"\nModel saved to: {args.model_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nERROR: Training failed: {str(e)}")

def train_model_quick():
    """Quick training with optimized parameters for fast testing."""
    print("\nQuick Training Mode")
    
    args = argparse.Namespace()
    args.epochs = 10
    args.batch_size = 128
    args.lr = 0.01
    args.use_real_data = False
    args.model_type = "SimpleAutoencoder"
    args.encoding_dim = 8
    args.non_interactive = True
    args.export_onnx = False
    args.model_dir = DEFAULT_MODEL_DIR / "quick_test"
    args.normal_samples = 1000
    args.attack_samples = 200
    
    print("Quick test configuration: 10 epochs, synthetic data, simple model")
    
    try:
        print("\nStarting quick training...")
        results = train_model(args)
        print(f"\nQuick training complete! Loss: {results.get('training', {}).get('best_val_loss', 'N/A'):.4f}")
    except Exception as e:
        print(f"\nERROR: Quick training failed: {str(e)}")

def train_model_custom():
    """Custom training with full parameter control."""
    print("\nCustom Training Configuration")
    
    # Get current config as starting point
    config = get_current_config()
    
    # Let user modify each section
    sections = ["training", "model", "data", "security"]
    
    for section in sections:
        if input(f"\nConfigure {section} parameters? (y/N): ").lower().strip() == 'y':
            section_config = config.get(section, {})
            
            if section == "training":
                section_config['epochs'] = int(input(f"Epochs (default={section_config.get('epochs', 50)}): ") or section_config.get('epochs', 50))
                section_config['batch_size'] = int(input(f"Batch size (default={section_config.get('batch_size', 64)}): ") or section_config.get('batch_size', 64))
                section_config['learning_rate'] = float(input(f"Learning rate (default={section_config.get('learning_rate', 0.001)}): ") or section_config.get('learning_rate', 0.001))
                section_config['weight_decay'] = float(input(f"Weight decay (default={section_config.get('weight_decay', 1e-4)}): ") or section_config.get('weight_decay', 1e-4))
            
            elif section == "model":
                model_types = list(MODEL_VARIANTS.keys())
                print("\nModel type options:")
                for i, mtype in enumerate(model_types, 1):
                    print(f"{i}. {mtype}")
                model_choice = input(f"Select model type (1-{len(model_types)}, default=1): ").strip()
                section_config['model_type'] = model_types[int(model_choice)-1] if model_choice else "EnhancedAutoencoder"
                
                section_config['encoding_dim'] = int(input(f"Encoding dimension (default={section_config.get('encoding_dim', 16)}): ") or section_config.get('encoding_dim', 16))
                
                activations = ['relu', 'leaky_relu', 'gelu', 'elu', 'swish']
                print("\nActivation options:")
                for i, act in enumerate(activations, 1):
                    print(f"{i}. {act}")
                act_choice = input(f"Select activation (1-{len(activations)}, default=1): ").strip()
                section_config['activation'] = activations[int(act_choice)-1] if act_choice else 'relu'
            
            elif section == "data":
                section_config['use_real_data'] = input("Use real data? (y/N): ").lower().strip() == 'y'
                if not section_config['use_real_data']:
                    section_config['normal_samples'] = int(input(f"Normal samples (default={section_config.get('normal_samples', 10000)}): ") or section_config.get('normal_samples', 10000))
                    section_config['attack_samples'] = int(input(f"Attack samples (default={section_config.get('attack_samples', 2000)}): ") or section_config.get('attack_samples', 2000))
            
            elif section == "security":
                section_config['percentile'] = int(input(f"Anomaly percentile (default={section_config.get('percentile', 95)}): ") or section_config.get('percentile', 95))
            
            config[section] = section_config
    
    # Update configuration and train
    update_global_config(config)
    train_model_interactive(use_current_config=True)

def select_preset_config():
    """Preset configuration selection."""
    print("\nAvailable Preset Configurations:")
    
    presets = list(PRESET_CONFIGS.items())
    for i, (name, preset) in enumerate(presets, 1):
        print(f"{i}. {name.title()}")
        print(f"   {preset.get('description', 'No description available')}")
        print(f"   Epochs: {preset.get('training', {}).get('epochs', 'N/A')}")
        print(f"   Model: {preset.get('model', {}).get('model_type', 'N/A')}")
        print(f"   Batch: {preset.get('training', {}).get('batch_size', 'N/A')}\n")
    
    print("0. Cancel")
    
    max_choice = len(PRESET_CONFIGS)
    choice = input(f"\nSelect preset (0-{max_choice}): ").strip()
    
    if choice == "0":
        print("Selection cancelled")
        return
    
    if choice.isdigit() and 1 <= int(choice) <= max_choice:
        preset_name = presets[int(choice)-1][0]
        preset_config = PRESET_CONFIGS[preset_name].copy()
        
        print(f"\nSelected preset: {preset_name.title()}")
        print(f"Description: {preset_config.get('description', 'No description')}")
        
        if input("\nApply this configuration? (Y/n): ").lower().strip() in ('', 'y', 'yes'):
            current_config = get_current_config()
            merged_config = deep_update(current_config, preset_config)
            update_global_config(merged_config)
            
            # Mark the preset name for reference
            merged_config['_preset_name'] = preset_name
            
            print(f"Applied preset configuration: {preset_name}")
        else:
            print("Configuration not applied")
    else:
        print("Invalid selection")

def configuration_menu():
    """Menu for configuration management."""
    while True:
        # Configuration status
        config = get_current_config()
        preset_name = config.get('_preset_name', 'Custom')
        
        print("\nConfiguration Management:")
        print(f"Active Configuration: {preset_name}\n")
        print("1. Show Current Configuration")
        print("2. Save Current Configuration")
        print("3. Load Configuration from File")
        print("4. Load Saved Configuration")
        print("5. Reset to Default Configuration")
        print("6. Validate Current Configuration")
        print("7. Edit Configuration Interactively")
        print("8. Compare Configurations")
        print("0. Back to Main Menu")
        
        choice = input("\nSelect option (0-8): ").strip()
        
        if choice == "1":
            show_current_config()
        elif choice == "2":
            save_config_interactive()
        elif choice == "3":
            load_config_from_file()
        elif choice == "4":
            load_saved_config_interactive()
        elif choice == "5":
            reset_config_interactive()
        elif choice == "6":
            validate_config_interactive()
        elif choice == "7":
            edit_config_interactive()
        elif choice == "8":
            compare_configs_interactive()
        elif choice == "0":
            return
        
        if choice != "0":
            input("\nPress Enter to continue...")

def interactive_main():
    """Main interactive interface."""
    while True:
        show_banner()
        print_main_menu()
        
        choice = input(Fore.WHITE + Style.BRIGHT + "\nSelect an option " + Fore.YELLOW + Style.BRIGHT + "(1-9): ").strip()
        
        try:
            if choice == "1":
                model_training_menu()
            elif choice == "2":
                run_hyperparameter_optimization()
            elif choice == "3":
                display_model_comparison()
            elif choice == "4":
                configuration_menu()
            elif choice == "5":
                show_system_info()
            elif choice == "6":
                run_performance_benchmark_interactive()
            elif choice == "7":
                model_analysis_menu()
            elif choice == "8":
                advanced_tools_menu()
            elif choice == "9":
                print(Fore.RED + Style.BRIGHT + "\nExiting...")
                print(Fore.YELLOW + Style.BRIGHT + "Goodbye!")
                break
            else:
                print(Fore.RED + Style.BRIGHT + "Invalid selection. Please try again.")
            
        except KeyboardInterrupt:
            print(Fore.YELLOW + Style.BRIGHT + "\nOperation interrupted")
        except Exception as e:
            print(Fore.RED + Style.BRIGHT + f"\nError: {str(e)}")
        
        if choice not in ("9", "0"):
            input(Style.DIM + "\nPress Enter to continue..." + Style.RESET_ALL)

def show_system_info():
    """System information display."""
    # Get system information
    hw = check_hardware()
    config = get_current_config()
    
    print("\nHardware Information:")
    print(f"Device: {hw['device'].upper()}")
    print(f"CPU: {hw['cpu_count']} cores")
    
    if torch.cuda.is_available():
        print(f"GPU: {hw.get('gpu_name', 'Unknown')}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda or 'Unknown'}")
    else:
        print("GPU: Not available")
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Configuration Information
    print("\nCurrent Configuration:")
    
    # Training config
    training_config = config.get('training', {})
    print("\nTraining:")
    print(f"Epochs: {training_config.get('epochs', 'N/A')}")
    print(f"Batch Size: {training_config.get('batch_size', 'N/A')}")
    print(f"Learning Rate: {training_config.get('learning_rate', 'N/A')}")
    print(f"Mixed Precision: {training_config.get('mixed_precision', 'N/A')}")
    
    # Model config
    model_config = config.get('model', {})
    print("\nModel:")
    print(f"Type: {model_config.get('model_type', 'N/A')}")
    print(f"Encoding Dim: {model_config.get('encoding_dim', 'N/A')}")
    print(f"Hidden Layers: {len(model_config.get('hidden_dims', []))}")
    print(f"Activation: {model_config.get('activation', 'N/A')}")
    
    # Data config
    data_config = config.get('data', {})
    print("\nData:")
    print(f"Features: {data_config.get('features', 'N/A')}")
    print(f"Use Real Data: {data_config.get('use_real_data', 'N/A')}")
    print(f"Normal Samples: {data_config.get('normal_samples', 'N/A')}")
    print(f"Validation Split: {data_config.get('validation_split', 'N/A')}")
    
    # Security config
    security_config = config.get('security', {})
    print("\nSecurity:")
    print(f"Percentile: {security_config.get('percentile', 'N/A')}")
    print(f"Threshold Strategy: {security_config.get('anomaly_threshold_strategy', 'N/A')}")
    
    # System Status
    print("\nSystem Status:")
    
    # Check directories
    dirs_ok = all(d.exists() for d in [DEFAULT_MODEL_DIR, LOG_DIR, TB_DIR, CONFIG_DIR])
    print(f"Directories: {'OK' if dirs_ok else 'MISSING'} - Required directories")
    
    # Check model variants
    variants_ok = bool(MODEL_VARIANTS)
    print(f"Model Variants: {'OK' if variants_ok else 'MISSING'} - {len(MODEL_VARIANTS)} available")
    
    # Check configuration
    try:
        validate_config(config)
        print("Configuration: VALID")
    except:
        print("Configuration: INVALID")

def run_hyperparameter_optimization():
    """Interactive HPO interface."""
    print("\nHyperparameter Optimization Setup")
    
    # Get current HPO configuration
    config = get_current_config()
    hpo_config = config.get('hyperparameter_optimization', {})
    
    # Interactive configuration
    print("\nConfigure optimization parameters:")
    
    trials = int(input(f"Number of trials (default={hpo_config.get('n_trials', 50)}): ") or hpo_config.get('n_trials', 50))
    timeout = int(input(f"Timeout in minutes (0 for no timeout, default={hpo_config.get('timeout_seconds', 0) // 60}): ") or hpo_config.get('timeout_seconds', 0) // 60)
    
    # Advanced options
    if input("\nConfigure advanced options? (y/N): ").lower().strip() == 'y':
        print("\nSampler type options:")
        print("1. TPE")
        print("2. Random")
        print("3. CmaEs")
        sampler_choice = input("Select sampler (1-3, default=1): ").strip()
        sampler = ['TPE', 'Random', 'CmaEs'][int(sampler_choice)-1] if sampler_choice else 'TPE'
        
        print("\nPruner type options:")
        print("1. Median")
        print("2. Hyperband")
        print("3. None")
        pruner_choice = input("Select pruner (1-3, default=1): ").strip()
        pruner = ['Median', 'Hyperband', 'None'][int(pruner_choice)-1] if pruner_choice else 'Median'
        
        # Update HPO config
        hpo_config.update({
            'n_trials': trials,
            'timeout_seconds': timeout * 60 if timeout > 0 else 0,
            'sampler': {'type': sampler},
            'pruner': {'type': pruner}
        })
    else:
        hpo_config.update({
            'n_trials': trials,
            'timeout_seconds': timeout * 60 if timeout > 0 else 0
        })
    
    # Update configuration
    config['hyperparameter_optimization'] = hpo_config
    update_global_config(config)
    
    # Display configuration
    print("\nOptimization Configuration:")
    print(f"Trials: {trials}")
    print(f"Timeout: {timeout} minutes" if timeout > 0 else "Timeout: No timeout")
    print(f"Sampler: {hpo_config.get('sampler', {}).get('type', 'TPE')}")
    print(f"Pruner: {hpo_config.get('pruner', {}).get('type', 'Median')}")
    
    if not input("\nStart optimization? (Y/n): ").lower().strip() in ('', 'y', 'yes'):
        print("Optimization cancelled")
        return
    
    print("\nStarting hyperparameter optimization...")
    
    try:
        # Create args
        args = argparse.Namespace()
        args.hpo_trials = trials
        args.hpo_timeout = timeout * 60 if timeout > 0 else 0
        args.non_interactive = True
        args.model_dir = DEFAULT_MODEL_DIR
        
        # Run optimization
        hpo_results = setup_hyperparameter_optimization(args, config)
        
        if hpo_results:
            # Display results
            print("\nOptimization Results:")
            print(f"Best Objective Value: {hpo_results['best_value']:.6f}")
            print(f"Total Trials: {len(hpo_results['study'].trials)}")
            print(f"Completed Trials: {len([t for t in hpo_results['study'].trials if t.state == optuna.trial.TrialState.COMPLETE])}")
            
            # Best parameters
            print("\nBest Parameters:")
            for param, value in hpo_results['best_params'].items():
                print(f"{param}: {value}")
            
            # Ask about final training
            if input("\nTrain final model with best parameters? (Y/n): ").lower().strip() in ('', 'y', 'yes'):
                print("\nTraining final model...")
                # Update configuration with best parameters
                best_config = hpo_results.get('best_config', {})
                current_config = get_current_config()
                final_config = deep_update(current_config, best_config)
                update_global_config(final_config)
                
                # Train final model
                train_model_interactive(use_current_config=True)
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"\nERROR: HPO failed: {str(e)}")

def run_stability_test():
    """Stability test with reporting."""
    print("\nRunning Stability Test")
    print("This will run a quick 10-epoch training to test system stability.\n")
    
    # Test configuration
    test_config = {
        'epochs': 10,
        'batch_size': 32,
        'lr': 0.01,
        'use_real_data': False,
        'model_type': 'SimpleAutoencoder',
        'encoding_dim': 8,
        'normal_samples': 1000,
        'attack_samples': 200
    }
    
    # Display test configuration
    print("\nTest Configuration:")
    for key, value in test_config.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    if not input("\nRun stability test? (Y/n): ").lower().strip() in ('', 'y', 'yes'):
        print("Test cancelled")
        return
    
    try:
        # Create test args
        args = argparse.Namespace()
        for key, value in test_config.items():
            setattr(args, key, value)
        
        args.non_interactive = True
        args.export_onnx = False
        args.model_dir = DEFAULT_MODEL_DIR / "stability_test"
        args.patience = 5
        
        # Run test
        print("\nRunning stability test...")
        start_time = time.time()
        results = train_model(args)
        duration = time.time() - start_time
        
        # Analyze results
        if results:
            best_loss = results.get('training', {}).get('best_val_loss', float('inf'))
            test_loss = results.get('evaluation', {}).get('test_loss', float('inf'))
            
            # Determine test status
            if best_loss < 0.1:
                status = "PASSED"
                status_desc = "System is stable and functioning correctly"
            elif best_loss < 0.5:
                status = "WARNING"
                status_desc = "System functional but performance suboptimal"
            else:
                status = "FAILED"
                status_desc = "System may have stability issues"
            
            # Display results
            print("\nStability Test Results:")
            print(f"Test Status: {status} - {status_desc}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Best Validation Loss: {best_loss:.6f}")
            print(f"Test Loss: {test_loss:.6f}")
            
            # Additional diagnostics
            if best_loss >= 0.1:
                print("\nDiagnostic Information:")
                print("1. Check system resources (CPU/GPU/Memory)")
                print("2. Verify data pipeline integrity")
                print("3. Consider adjusting learning parameters")
                
        else:
            print("\nERROR: Stability test failed - no results returned")
        
    except Exception as e:
        print(f"\nERROR: Stability test failed: {str(e)}")
        
        # Basic system diagnostics
        print("\nSystem Diagnostics:")
        try:
            hw = check_hardware()
            print(f"1. Device: {hw['device']}")
            print(f"2. PyTorch: {torch.__version__}")
            print(f"3. CUDA available: {torch.cuda.is_available()}")
        except:
            print("[ERROR] Unable to get hardware information")

# Helper functions for configuration management
def show_current_config():
    """Display current configuration."""
    config = get_current_config()
    print("\nCurrent Configuration:")
    print(json.dumps(config, indent=2, default=str))

def save_config_interactive():
    """Interactive configuration saving."""
    config = get_current_config()
    
    name = input("\nConfiguration name: ")
    if name:
        try:
            save_named_config(name, config)
            print(f"Configuration saved as '{name}'")
        except Exception as e:
            print(f"Failed to save configuration: {e}")

def load_config_from_file():
    """Load configuration from file."""
    file_path = input("\nConfiguration file path: ")
    if file_path:
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            update_global_config(config)
            print(f"Configuration loaded from '{file_path}'")
        except Exception as e:
            print(f"Failed to load configuration: {e}")

def load_saved_config_interactive():
    """Interactive loading of saved configurations."""
    saved_configs = list_saved_configs()
    if not saved_configs:
        print("\nNo saved configurations found")
        return
    
    print("\nAvailable saved configurations:")
    for i, name in enumerate(saved_configs, 1):
        print(f"{i}. {name}")
    
    print("0. Cancel")
    choice = input(f"\nSelect configuration (0-{len(saved_configs)}): ").strip()
    
    if choice == "0":
        print("Selection cancelled")
        return
    
    if choice.isdigit() and 1 <= int(choice) <= len(saved_configs):
        name = saved_configs[int(choice)-1]
        try:
            config = load_saved_config(name)
            update_global_config(config)
            print(f"Loaded configuration '{name}'")
        except Exception as e:
            print(f"Failed to load configuration: {e}")

def reset_config_interactive():
    """Interactive configuration reset."""
    if input("\nReset configuration to defaults? This cannot be undone. (y/N): ").lower().strip() == 'y':
        try:
            reset_config()
            print("Configuration reset to defaults")
        except Exception as e:
            print(f"Failed to reset configuration: {e}")

def validate_config_interactive():
    """Interactive configuration validation."""
    try:
        config = get_current_config()
        validate_config(config)
        print("\nConfiguration is valid")
    except ValueError as e:
        print(f"\nConfiguration validation failed: {e}")
        
        if input("\nShow detailed validation report? (y/N): ").lower().strip() == 'y':
            print("\nValidation Details:")
            # Additional validation logic could go here

def edit_config_interactive():
    """Interactive configuration editor."""
    print("\nInteractive configuration editing not yet implemented")
    print("Please use the preset configurations or manual file editing for now.")

def compare_configs_interactive():
    """Interactive configuration comparison."""
    print("\nConfiguration comparison not yet implemented")
    print("This feature will allow side-by-side comparison of different configurations.")

def run_performance_benchmark_interactive():
    """Interactive performance benchmark."""
    print("\nPerformance Benchmark")
    
    if input("\nRun performance benchmark? This will train multiple models. (Y/n): ").lower().strip() in ('', 'y', 'yes'):
        args = argparse.Namespace()
        args.model_dir = DEFAULT_MODEL_DIR / "benchmarks"
        run_performance_benchmark(args)
    else:
        print("Benchmark cancelled")

def model_analysis_menu():
    """Menu for model analysis and visualization."""
    print("\nModel Analysis & Visualization")
    print("This feature will include:")
    print("[+] Training curve analysis")
    print("[+] Model performance visualization")
    print("[+] Anomaly detection results")
    print("[+] Feature importance analysis")
    print("\nComing soon...")

def advanced_tools_menu():
    """Menu for advanced tools and utilities."""
    print("\nAdvanced Tools")
    print("This feature will include:")
    print("[+] Model export utilities")
    print("[+] Custom data preprocessing")
    print("[+] Batch processing tools")
    print("[+] Performance profiling")
    print("\nComing soon...")

def main():
    """Main entry point with comprehensive argument parsing and system orchestration."""
    # Initialize basic logger first (fallback)
    logger = logging.getLogger(__name__)
    
    # Initialize system to set up logging and configuration
    try:
        system_status, config, logger = initialize_system()
        #validate_config(config)
    except Exception as e:
        # Use fallback logger since initialize_system failed
        logger.warning(f"Configuration initialization failed, using defaults: {e}")
        config = get_current_config()
        
        # Setup minimal logging if not already configured
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
    
    # If no arguments provided, launch interactive mode
    if len(sys.argv) == 1:
        interactive_main()
        return

    # Initialize configuration system early
    
    # Extract configuration sections for defaults
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    security_config = config.get('security', {})
    system_config = config.get('system', {})
    hpo_config = config.get('hyperparameter_optimization', {})
    monitoring_config = config.get('monitoring', {})
    
    # Create argument parser with comprehensive configuration integration
    parser = argparse.ArgumentParser(
        description="Enhanced Anomaly Detection Model Training with Configuration Management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  %(prog)s --preset development               # Use development preset
  %(prog)s --epochs 100 --batch-size 64     # Custom training parameters
  %(prog)s --hpo-trials 50                   # Hyperparameter optimization
  %(prog)s --show-config                     # Display current configuration
  %(prog)s --compare-models                  # Compare model architectures
        """
    )
    
    # Training configuration group
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        "--epochs",
        type=int,
        default=training_config.get('epochs'),
        help=f"Maximum number of training epochs (config: {training_config.get('epochs', DEFAULT_EPOCHS)})"
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=training_config.get('batch_size'),
        help=f"Training batch size (config: {training_config.get('batch_size', DEFAULT_BATCH_SIZE)})"
    )
    training_group.add_argument(
        "--lr",
        type=float,
        default=training_config.get('learning_rate'),
        help=f"Learning rate (config: {training_config.get('learning_rate', LEARNING_RATE)})"
    )
    training_group.add_argument(
        "--patience",
        type=int,
        default=training_config.get('patience'),
        help=f"Early stopping patience in epochs (config: {training_config.get('patience', EARLY_STOPPING_PATIENCE)})"
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=training_config.get('weight_decay'),
        help=f"Weight decay for optimizer (config: {training_config.get('weight_decay', WEIGHT_DECAY)})"
    )
    training_group.add_argument(
        "--grad-clip",
        type=float,
        default=training_config.get('gradient_clip'),
        help=f"Gradient clipping value (config: {training_config.get('gradient_clip', GRADIENT_CLIP)})"
    )
    training_group.add_argument(
        "--mixed-precision",
        action="store_true",
        default=training_config.get('mixed_precision', MIXED_PRECISION),
        help="Enable mixed precision training"
    )
    
    # Model configuration group
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument(
        "--model-type",
        choices=['SimpleAutoencoder', 'EnhancedAutoencoder', 'AutoencoderEnsemble'],
        default=model_config.get('model_type', 'EnhancedAutoencoder'),
        help="Type of model architecture to use"
    )
    model_group.add_argument(
        "--features",
        type=int,
        default=data_config.get('features'),
        help=f"Number of input features (config: {data_config.get('features', FEATURES)})"
    )
    model_group.add_argument(
        "--encoding-dim",
        type=int,
        default=model_config.get('encoding_dim'),
        help=f"Encoder hidden dimension (config: {model_config.get('encoding_dim', DEFAULT_ENCODING_DIM)})"
    )
    model_group.add_argument(
        "--num-models",
        type=int,
        default=model_config.get('num_models', NUM_MODELS),
        help=f"Number of models in ensemble (config: {model_config.get('num_models', NUM_MODELS)})"
    )
    model_group.add_argument(
        "--hidden-dims",
        type=int,
        nargs='+',
        default=model_config.get('hidden_dims'),
        help=f"Hidden layer dimensions (config: {model_config.get('hidden_dims', HIDDEN_LAYER_SIZES)})"
    )
    model_group.add_argument(
        "--dropout-rates",
        type=float,
        nargs='+',
        default=model_config.get('dropout_rates'),
        help=f"Dropout rates for each layer (config: {model_config.get('dropout_rates', DROPOUT_RATES)})"
    )
    model_group.add_argument(
        "--activation",
        choices=['relu', 'leaky_relu', 'gelu', 'elu', 'swish'],
        default=model_config.get('activation', ACTIVATION),
        help=f"Activation function (config: {model_config.get('activation', ACTIVATION)})"
    )
    model_group.add_argument(
        "--normalization",
        choices=[None, 'batch', 'layer', 'instance'],
        default=model_config.get('normalization'),
        help=f"Normalization type (config: {model_config.get('normalization', NORMALIZATION)})"
    )
    
    # Data configuration group
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        "--use-real-data",
        action="store_true",
        default=data_config.get('use_real_data', False),
        help="Use preprocessed data instead of synthetic"
    )
    data_group.add_argument(
        "--normal-samples",
        type=int,
        default=data_config.get('normal_samples'),
        help=f"Normal training samples for synthetic data (config: {data_config.get('normal_samples', NORMAL_SAMPLES)})"
    )
    data_group.add_argument(
        "--attack-samples",
        type=int,
        default=data_config.get('attack_samples'),
        help=f"Anomalous test samples for synthetic data (config: {data_config.get('attack_samples', ATTACK_SAMPLES)})"
    )
    data_group.add_argument(
        "--validation-split",
        type=float,
        default=data_config.get('validation_split', 0.2),
        help="Fraction of data to use for validation"
    )
    data_group.add_argument(
        "--data-path",
        type=Path,
        default=data_config.get('data_path', DEFAULT_MODEL_DIR / "preprocessed_dataset.csv"),
        help="Path to preprocessed data file"
    )
    data_group.add_argument(
        "--artifacts-path",
        type=Path,
        default=data_config.get('artifacts_path', DEFAULT_MODEL_DIR / "preprocessing_artifacts.pkl"),
        help="Path to preprocessing artifacts file"
    )
    
    # Security configuration group
    security_group = parser.add_argument_group('Security Parameters')
    security_group.add_argument(
        "--percentile",
        type=int,
        default=security_config.get('percentile'),
        help=f"Percentile for anomaly threshold (config: {security_config.get('percentile', DEFAULT_PERCENTILE)})"
    )
    security_group.add_argument(
        "--anomaly-threshold-strategy",
        choices=['percentile', 'iqr', 'zscore', 'isolation_forest'],
        default=security_config.get('anomaly_threshold_strategy', 'percentile'),
        help="Strategy for calculating anomaly threshold"
    )
    
    # System configuration group
    system_group = parser.add_argument_group('System Parameters')
    system_group.add_argument(
        "--model-dir",
        type=Path,
        default=Path(system_config.get('model_dir', DEFAULT_MODEL_DIR)),
        help="Directory to save model artifacts"
    )
    system_group.add_argument(
        "--tb-dir",
        type=Path,
        default=Path(monitoring_config.get('tensorboard_dir', TB_DIR)),
        help="TensorBoard logging directory"
    )
    system_group.add_argument(
        "--log-dir",
        type=Path,
        default=Path(system_config.get('log_dir', LOG_DIR)),
        help="Logging directory"
    )
    system_group.add_argument(
        "--config-dir",
        type=Path,
        default=Path(system_config.get('config_dir', CONFIG_DIR)),
        help="Configuration directory"
    )
    system_group.add_argument(
        "--num-workers",
        type=int,
        default=training_config.get('num_workers', min(4, os.cpu_count() or 1)),
        help="Number of workers for data loading"
    )
    system_group.add_argument(
        "--export-onnx",
        action="store_true",
        default=system_config.get('export_onnx', False),
        help="Export model to ONNX format"
    )
    system_group.add_argument(
        "--non-interactive",
        action="store_true",
        default=system_config.get('non_interactive', False),
        help="Disable all interactive prompts"
    )
    system_group.add_argument(
        "--debug",
        action="store_true",
        default=system_config.get('debug', False),
        help="Enable debug logging"
    )
    
    # Hyperparameter optimization group
    hpo_group = parser.add_argument_group('Hyperparameter Optimization')
    hpo_group.add_argument(
        "--hpo",
        action="store_true",
        default=hpo_config.get('enabled', False),
        help="Enable hyperparameter optimization"
    )
    hpo_group.add_argument(
        "--hpo-trials",
        type=int,
        default=hpo_config.get('n_trials', 50),
        help=f"Number of hyperparameter optimization trials (config: {hpo_config.get('n_trials', 50)})"
    )
    hpo_group.add_argument(
        "--hpo-timeout",
        type=int,
        default=hpo_config.get('timeout_seconds', 3600),
        help="Timeout for hyperparameter optimization in seconds (0 for no timeout)"
    )
    hpo_group.add_argument(
        "--hpo-sampler",
        choices=['TPE', 'Random', 'CmaEs'],
        default=hpo_config.get('sampler', {}).get('type', 'TPE'),
        help="Sampler type for hyperparameter optimization"
    )
    hpo_group.add_argument(
        "--hpo-pruner",
        choices=['Median', 'Hyperband', 'None'],
        default=hpo_config.get('pruner', {}).get('type', 'Median'),
        help="Pruner type for hyperparameter optimization"
    )
    
    # Configuration management group
    config_group = parser.add_argument_group('Configuration Management')
    config_group.add_argument(
        "--preset",
        choices=list(PRESET_CONFIGS.keys()),
        help=f"Use a preset configuration. Available: {list(PRESET_CONFIGS.keys())}"
    )
    config_group.add_argument(
        "--show-config",
        action="store_true",
        help="Display current configuration and exit"
    )
    config_group.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate current configuration and exit"
    )
    config_group.add_argument(
        "--save-config",
        type=str,
        metavar="NAME",
        help="Save current configuration with given name"
    )
    config_group.add_argument(
        "--load-config",
        type=str,
        metavar="NAME",
        help="Load saved configuration by name"
    )
    config_group.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configurations"
    )
    config_group.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults"
    )
    config_group.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare available model architectures"
    )
    config_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    
    # Monitoring configuration group
    monitoring_group = parser.add_argument_group('Monitoring Parameters')
    monitoring_group.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    monitoring_group.add_argument(
        "--log-frequency",
        type=int,
        default=monitoring_config.get('log_frequency', 1),
        help="Frequency of progress logging (epochs)"
    )
    monitoring_group.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=monitoring_config.get('checkpoint_frequency', 10),
        help="Frequency of model checkpointing (epochs)"
    )
    monitoring_group.add_argument(
        "--metrics-frequency",
        type=int,
        default=monitoring_config.get('metrics_frequency', 10),
        help="Frequency of detailed metrics logging (epochs)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle configuration commands first (these exit after completion)
    if args.show_config:
        current_config = get_current_config()
        #current_config = config  # Use the config we have
        print("Current Configuration:")
        print("=" * 60)
        print(json.dumps(current_config, indent=2, default=str))
        return
    
    if args.validate_config:
        try:
            validate_config(config)
            logger.info("[INFO] Configuration is valid")
        except ValueError as e:
            logger.error(f"[ERROR] Configuration validation failed: {e}")
            sys.exit(1)
        return
    
    if args.list_configs:
        print("Available Configurations:")
        print("=" * 40)
        print("\nPreset Configurations:")
        for name, preset in PRESET_CONFIGS.items():
            print(f"  [+] {name}: {preset.get('description', 'No description')}")
        
        # List saved configurations if any
        saved_configs = list_saved_configs()
        if saved_configs:
            print("\nSaved Configurations:")
            for name in saved_configs:
                print(f"  [+] {name}")
        return
    
    if args.reset_config:
        if args.non_interactive or prompt_user("Reset configuration to defaults?", default=False):
            reset_config()
            logger.info("[INFO] Configuration reset to defaults")
        return
    
    if args.compare_models:
        display_model_comparison()
        return
    
    if args.benchmark:
        run_performance_benchmark(args)
        return
    
    # Handle configuration loading
    if args.load_config:
        try:
            config = load_saved_config(args.load_config)
            update_global_config(config)
            logger.info(f"[INFO] Loaded configuration: {args.load_config}")
        except FileNotFoundError:
            logger.error(f"[ERROR] Configuration '{args.load_config}' not found")
            sys.exit(1)
    
    # Apply preset configuration if specified
    if args.preset:
        if args.preset not in PRESET_CONFIGS:
            logger.error(f"[ERROR] Invalid preset: {args.preset}")
            logger.info(f"Available presets: {list(PRESET_CONFIGS.keys())}")
            sys.exit(1)
        
        preset_config = PRESET_CONFIGS[args.preset].copy()
        current_config = get_current_config()
        merged_config = deep_update(current_config, preset_config)
        update_global_config(merged_config)
        logger.info(f"[INFO] Applied preset configuration: {args.preset}")
    
    # Configure logging level early
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("torch").setLevel(logging.DEBUG)
        logging.getLogger("optuna").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        logging.getLogger("optuna").setLevel(logging.WARNING)
    
    # Update configuration with command line arguments
    current_config = get_current_config()
    
    # Map command line args to configuration structure
    arg_config_mapping = {
        # Training parameters
        'epochs': ('training', 'epochs'),
        'batch_size': ('training', 'batch_size'),
        'lr': ('training', 'learning_rate'),
        'patience': ('training', 'patience'),
        'weight_decay': ('training', 'weight_decay'),
        'grad_clip': ('training', 'gradient_clip'),
        'mixed_precision': ('training', 'mixed_precision'),
        'num_workers': ('training', 'num_workers'),
        
        # Model parameters
        'model_type': ('model', 'model_type'),
        'encoding_dim': ('model', 'encoding_dim'),
        'num_models': ('model', 'num_models'),
        'hidden_dims': ('model', 'hidden_dims'),
        'dropout_rates': ('model', 'dropout_rates'),
        'activation': ('model', 'activation'),
        'normalization': ('model', 'normalization'),
        
        # Data parameters
        'features': ('data', 'features'),
        'normal_samples': ('data', 'normal_samples'),
        'attack_samples': ('data', 'attack_samples'),
        'use_real_data': ('data', 'use_real_data'),
        'validation_split': ('data', 'validation_split'),
        'data_path': ('data', 'data_path'),
        'artifacts_path': ('data', 'artifacts_path'),
        
        # Security parameters
        'percentile': ('security', 'percentile'),
        'anomaly_threshold_strategy': ('security', 'anomaly_threshold_strategy'),
        
        # System parameters
        'model_dir': ('system', 'model_dir'),
        'tb_dir': ('system', 'tensorboard_dir'),
        'log_dir': ('system', 'log_dir'),
        'config_dir': ('system', 'config_dir'),
        'export_onnx': ('system', 'export_onnx'),
        'non_interactive': ('system', 'non_interactive'),
        'debug': ('system', 'debug'),
        
        # HPO parameters
        'hpo_trials': ('hyperparameter_optimization', 'n_trials'),
        'hpo_timeout': ('hyperparameter_optimization', 'timeout_seconds'),
        'hpo_sampler': ('hyperparameter_optimization', 'sampler', 'type'),
        'hpo_pruner': ('hyperparameter_optimization', 'pruner', 'type'),
        
        # Monitoring parameters
        'log_frequency': ('monitoring', 'log_frequency'),
        'checkpoint_frequency': ('monitoring', 'checkpoint_frequency'),
        'metrics_frequency': ('monitoring', 'metrics_frequency'),
    }
    
    # Update configuration with non-None command line arguments
    for arg_name, config_path in arg_config_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            # Navigate to the correct nested dictionary
            target = current_config
            for key in config_path[:-1]:
                target = target.setdefault(key, {})
            target[config_path[-1]] = arg_value
    
    # Handle special flags
    if args.disable_tensorboard:
        current_config.setdefault('monitoring', {})['tensorboard_logging'] = False
    
    # Update global configuration
    update_global_config(current_config)
    
    # Save configuration if requested
    if args.save_config:
        try:
            save_named_config(args.save_config, current_config)
            logger.info(f"[INFO] Configuration saved as: {args.save_config}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save configuration: {e}")
    
    # Validate final configuration
    try:
        validate_config(current_config)
    except ValueError as e:
        logger.error(f"[ERROR] Configuration validation failed: {e}")
        if not args.non_interactive and prompt_user("Continue with invalid configuration?", default=False):
            logger.warning("Proceeding with potentially invalid configuration")
        else:
            sys.exit(1)
    
    # Log system information
    logger.info("=" * 80)
    logger.info("SYSTEM INITIALIZATION")
    logger.info("=" * 80)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Configuration preset: {args.preset or 'custom'}")
    
    # Run hyperparameter optimization if requested
    if args.hpo or args.hpo_trials > 0:
        logger.info("=" * 80)
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 80)
        
        try:
            hpo_results = setup_hyperparameter_optimization(args, current_config)
            
            # Update args with best parameters for final training
            if hpo_results and 'best_config' in hpo_results:
                best_config = hpo_results['best_config']
                merged_config = deep_update(current_config, best_config)
                update_global_config(merged_config)
                
                logger.info("[INFO] Hyperparameter optimization completed")
                logger.info(f"Best objective value: {hpo_results['best_value']:.5f}")
                logger.info("Best parameters:")
                for key, value in hpo_results['best_params'].items():
                    logger.info(f"  {key}: {value}")
                
                # Ask if user wants to train final model
                if not args.non_interactive:
                    if not prompt_user("Train final model with best parameters?", default=True):
                        logger.info("Hyperparameter optimization completed. Exiting.")
                        return
                
                logger.info("Training final model with optimized parameters...")
            else:
                logger.warning("Hyperparameter optimization failed to produce results")
                return
                
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            if args.debug:
                logger.exception("HPO error details:")
            return
    
    # Ensure all required directories exist
    try:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        args.tb_dir.mkdir(parents=True, exist_ok=True)
        args.config_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)
    
    # Log final training configuration
    logger.info("=" * 80)
    logger.info("FINAL CONFIGURATION")
    logger.info("=" * 80)
    
    final_config = get_current_config()
    config_summary = {
        'training': {
            'epochs': final_config.get('training', {}).get('epochs', DEFAULT_EPOCHS),
            'batch_size': final_config.get('training', {}).get('batch_size', DEFAULT_BATCH_SIZE),
            'learning_rate': final_config.get('training', {}).get('learning_rate', LEARNING_RATE),
            'patience': final_config.get('training', {}).get('patience', EARLY_STOPPING_PATIENCE),
            'mixed_precision': final_config.get('training', {}).get('mixed_precision', MIXED_PRECISION)
        },
        'model': {
            'type': final_config.get('model', {}).get('model_type', 'EnhancedAutoencoder'),
            'encoding_dim': final_config.get('model', {}).get('encoding_dim', DEFAULT_ENCODING_DIM),
            'features': final_config.get('data', {}).get('features', FEATURES)
        },
        'data': {
            'use_real_data': final_config.get('data', {}).get('use_real_data', False),
            'normal_samples': final_config.get('data', {}).get('normal_samples', NORMAL_SAMPLES),
            'attack_samples': final_config.get('data', {}).get('attack_samples', ATTACK_SAMPLES)
        },
        'system': {
            'model_dir': str(args.model_dir),
            'export_onnx': final_config.get('system', {}).get('export_onnx', False),
            'debug': final_config.get('system', {}).get('debug', False)
        }
    }
    
    for section, params in config_summary.items():
        logger.info(f"{section.upper()}:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
    
    # Save final configuration to model directory
    final_config_path = args.model_dir / "run_configuration.json"
    try:
        with open(final_config_path, 'w') as f:
            json.dump(final_config, f, indent=2, default=str)
        logger.info(f"[INFO] Configuration saved to: {final_config_path}")
    except Exception as e:
        logger.warning(f"Could not save run configuration: {e}")
    
    # Run training
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    try:
        training_results = train_model(args)
        
        # Log training completion
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        # Display key results
        if training_results:
            metrics = training_results.get('evaluation', {})
            logger.info("Key Results:")
            logger.info(f"  Best validation loss: {training_results.get('training', {}).get('best_val_loss', 'N/A'):.4f}")
            logger.info(f"  Test loss: {metrics.get('test_loss', 'N/A'):.4f}")
            logger.info(f"  Anomaly detection rate: {metrics.get('anomaly_detection_rate', 'N/A'):.2%}")
            logger.info(f"  Model parameters: {training_results.get('model', {}).get('total_parameters', 'N/A'):,}")
            
            # Save training summary
            summary_path = args.model_dir / "training_complete.json"
            try:
                with open(summary_path, 'w') as f:
                    json.dump({
                        'status': 'completed',
                        'timestamp': datetime.now().isoformat(),
                        'configuration': final_config,
                        'results': training_results
                    }, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Could not save training summary: {e}")
        
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save interruption info
        interruption_path = args.model_dir / "training_interrupted.json"
        try:
            with open(interruption_path, 'w') as f:
                json.dump({
                    'status': 'interrupted',
                    'timestamp': datetime.now().isoformat(),
                    'configuration': final_config
                }, f, indent=2, default=str)
        except:
            pass
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if args.debug:
            logger.exception("Training error details:")
        
        # Save error info
        error_path = args.model_dir / "training_failed.json"
        try:
            with open(error_path, 'w') as f:
                json.dump({
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'configuration': final_config,
                    'traceback': traceback.format_exc()
                }, f, indent=2, default=str)
        except:
            pass
        
        sys.exit(1)
        
    finally:
        # Cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

# Helper functions for configuration management
def list_saved_configs() -> List[str]:
    """List all saved configuration names."""
    try:
        config_files = list(CONFIG_DIR.glob("*.json"))
        return [f.stem for f in config_files if f.stem != "current"]
    except:
        return []

def load_saved_config(name: str) -> Dict[str, Any]:
    """Load a saved configuration by name."""
    config_path = CONFIG_DIR / f"{name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration '{name}' not found")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def save_named_config(name: str, config: Dict[str, Any]) -> None:
    """Save configuration with a specific name."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{name}.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def run_performance_benchmark(args: argparse.Namespace) -> None:
    """Run performance benchmark across different configurations."""
    logger.info("Running performance benchmark...")
    
    benchmark_configs = [
        ('small', {'model': {'encoding_dim': 8}, 'training': {'batch_size': 32, 'epochs': 10}}),
        ('medium', {'model': {'encoding_dim': 16}, 'training': {'batch_size': 64, 'epochs': 10}}),
        ('large', {'model': {'encoding_dim': 32}, 'training': {'batch_size': 128, 'epochs': 10}})
    ]
    
    results = {}
    
    for name, config_override in benchmark_configs:
        logger.info(f"Benchmarking {name} configuration...")
        
        # Create benchmark args
        benchmark_args = argparse.Namespace(**vars(args))
        benchmark_args.model_dir = args.model_dir / f"benchmark_{name}"
        benchmark_args.epochs = 10
        benchmark_args.non_interactive = True
        benchmark_args.export_onnx = False
        
        # Apply config override
        current_config = get_current_config()
        benchmark_config = deep_update(current_config, config_override)
        update_global_config(benchmark_config)
        
        start_time = time.time()
        try:
            training_results = train_model(benchmark_args)
            duration = time.time() - start_time
            
            results[name] = {
                'duration': duration,
                'final_loss': training_results.get('evaluation', {}).get('test_loss', float('inf')),
                'parameters': training_results.get('model', {}).get('total_parameters', 0),
                'success': True
            }
            
        except Exception as e:
            duration = time.time() - start_time
            results[name] = {
                'duration': duration,
                'error': str(e),
                'success': False
            }
    
    # Save benchmark results
    benchmark_path = args.model_dir / "benchmark_results.json"
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Display results
    logger.info("Benchmark Results:")
    logger.info("=" * 50)
    for name, result in results.items():
        if result['success']:
            logger.info(f"{name:>10}: {result['duration']:6.1f}s | Loss: {result['final_loss']:.4f} | Params: {result['parameters']:,}")
        else:
            logger.info(f"{name:>10}: {result['duration']:6.1f}s | FAILED: {result['error']}")

if __name__ == "__main__":
    # Configure warnings and logging before anything else
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Ensure required directories exist
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        TB_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create required directories: {e}")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)