# Standard library imports
import os
import sys
import json
import logging
import argparse
import warnings
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
from rich.status import Status

# Terminal styling and colors
from colorama import Fore, Back, Style, init
init(autoreset=True)  # Initialize colorama

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

# Parallel processing and concurrency
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
#import asyncio

# File handling and compression
import zipfile
import tarfile
import gzip
import lzma

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

# Development and debugging tools
import inspect
import cProfile
import pstats
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Version checking utilities
try:
    from packaging import version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False

# Additional utility libraries
import itertools
import random
import string
from contextlib import contextmanager, suppress
import weakref
from types import SimpleNamespace
import pathlib

from sklearn.exceptions import ConvergenceWarning
from matplotlib import MatplotlibDeprecationWarning

# Initialize rich console
console = Console()

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

# Logging and Directory Setup
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

# Setup logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure the root logger with both a UTF-8 file handler and
    a Unicode-safe console handler, removing duplicate handlers
    and standardizing the log format.
    """
    # Create logger
    #logger = logging.getLogger()
    logger = logging.getLogger(__name__)
    #logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    logger.setLevel(logging.DEBUG)
    
    # File handler (UTF-8 encoded)
    log_file = log_dir / "deep_learning.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (Unicode-safe)
    console_handler = UnicodeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global directory variables
#LOG_DIR = DEFAULT_MODEL_DIR = DATA_DIR = CONFIG_DIR = REPORTS_DIR = TB_DIR = CACHE_DIR = None

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
            logger.debug(f"Created directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    return dirs

def configure_directories(logger: logging.Logger) -> Dict[str, Path]:
    """
    Initialize and assign global directory path variables using
    setup_directories(), ensuring they are accessible across modules.
    """
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
    
    return dirs

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
    CRITICAL = auto()    # Check must pass for the program to continue running
    IMPORTANT = auto()   # Check should pass for full functionality but not fatal
    INFORMATIONAL = auto()  # Non-essential check providing useful system information

class CheckResult:
    """Encapsulates the outcome of a system check."""
    def __init__(self, 
                 passed: bool, 
                 message: str, 
                 level: CheckLevel = CheckLevel.IMPORTANT,
                 details: Optional[str] = None,
                 exception: Optional[Exception] = None):
        self.passed = passed
        self.message = message
        self.level = level
        self.details = details
        self.exception = exception

    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"{status}: {self.message} ({self.level.name})"

def loading_screen() -> bool:
    """Display loading screen with system checks."""
    console.print("[bold blue]Running System Checks...[/bold blue]")
    results = run_system_checks()
    display_check_results(results)
    
    # Check if any critical checks failed
    critical_failures = any(
        not result.passed and result.level == CheckLevel.CRITICAL 
        for result in results
    )
    
    if critical_failures:
        console.print("\n[bold red]Critical checks failed - system cannot continue[/bold red]")
        return False
    return True

def run_system_checks() -> List[CheckResult]:
    """Run all system checks using existing functions where possible."""
    checks = [
        check_python_version(),
        check_torch(),
        check_cuda(),
        check_package_versions_wrapper(),
        check_directory_access_wrapper(),
        check_disk_space(),
        check_cpu_cores(),
        check_system_ram(),
        check_system_architecture(),
        check_logging_setup(),
        check_seed_config()
    ]
    return [result for result in checks if result is not None]

def display_check_results(results: List[CheckResult]):
    """Display check results in a styled table with improved formatting."""
    table = Table(title="System Check Results", box=box.SIMPLE)
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="right")
    table.add_column("Level", style="magenta")
    table.add_column("Details", style="dim")
    
    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        level_style = {
            CheckLevel.CRITICAL: "bold red",
            CheckLevel.IMPORTANT: "yellow",
            CheckLevel.INFORMATIONAL: "dim"
        }.get(result.level, "")
        
        level_text = f"[{level_style}]{result.level.name}[/{level_style}]"
        details = result.details or ""
        
        table.add_row(
            result.message,
            status,
            level_text,
            details
        )
    
    console.print(table)

# Individual check implementations
def check_python_version(min_version: Tuple[int, int] = (3, 7)) -> CheckResult:
    """Verify that the current Python version meets the minimum requirement."""
    try:
        version = sys.version_info
        passed = (version.major, version.minor) >= min_version
        message = f"Python >= {'.'.join(map(str, min_version))}"
        details = f"Current: {version.major}.{version.minor}.{version.micro}"
        
        return CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.CRITICAL if not passed else CheckLevel.INFORMATIONAL,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Python version check failed",
            level=CheckLevel.CRITICAL,
            details=str(e),
            exception=e
        )

def check_torch() -> CheckResult:
    """Confirm that PyTorch is installed and operational."""
    try:
        import torch
        passed = hasattr(torch, '__version__')
        message = "PyTorch installation"
        details = f"Version: {torch.__version__}" if passed else "PyTorch not found"
        
        return CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.CRITICAL,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="PyTorch check failed",
            level=CheckLevel.CRITICAL,
            details=str(e),
            exception=e
        )

def check_cuda() -> CheckResult:
    """Check if CUDA is available and report GPU details if present."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        message = "CUDA availability"
        
        if cuda_available:
            details = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                details.append(
                    f"GPU {i}: {props.name} (CC {props.major}.{props.minor}, "
                    f"{props.total_memory/1024**3:.1f}GB)"
                )
            details = "\n".join(details)
        else:
            details = "CUDA not available - Using CPU"
        
        return CheckResult(
            passed=cuda_available,
            message=message,
            level=CheckLevel.IMPORTANT,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="CUDA check failed",
            level=CheckLevel.IMPORTANT,
            details=str(e),
            exception=e
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
        table = Table(title="Dependency Check", box=box.SIMPLE)
        table.add_column("Package", style="cyan", no_wrap=True)
        table.add_column("Version", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Status", justify="right")
        table.add_column("Description", style="dim")
        
        for name, info in version_info.items():
            status_style = "green" if info['status'] == 'OK' else "yellow" if info['status'] == 'WARNING' else "red"
            required_text = "Required" if info['required'] else "Optional"
            table.add_row(
                name,
                info['version'],
                required_text,
                f"[{status_style}]{info['status']}[/{status_style}]",
                info['description']
            )
        
        console.print(table)
        
        return CheckResult(
            passed=passed,
            message="Package version check",
            level=CheckLevel.CRITICAL if not passed else CheckLevel.IMPORTANT,
            details="\n".join(details),
            metadata={
                'version_info': version_info,
                'table': table,
                'summary': {
                    'total': len(version_info),
                    'passed': sum(1 for info in version_info.values() if info['status'] == 'OK'),
                    'warnings': sum(1 for info in version_info.values() if info['status'] == 'WARNING'),
                    'missing': sum(1 for info in version_info.values() if info['status'] == 'MISSING')
                }
            }
        )
        
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Package version check failed",
            level=CheckLevel.CRITICAL,
            details=str(e),
            exception=e
        )

def check_directory_access_wrapper() -> CheckResult:
    """Verify directory access using the setup_directories function."""
    try:
        dirs = setup_directories()
        passed = all(path.exists() and os.access(path, os.W_OK) for path in dirs.values())
        message = "Directory access"
        details = "\n".join(f"{name}: {path}" for name, path in dirs.items())
        
        return CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.CRITICAL,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Directory access check failed",
            level=CheckLevel.CRITICAL,
            details=str(e),
            exception=e
        )

def check_disk_space(min_gb: float = 1.0) -> CheckResult:
    """Ensure the system has at least the specified amount of free disk space."""
    try:
        usage = shutil.disk_usage('.')
        free_gb = usage.free / (1024**3)
        passed = free_gb >= min_gb
        message = f"Disk space (>={min_gb}GB)"
        details = f"Free: {free_gb:.1f}GB"
        
        return CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.IMPORTANT,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Disk space check failed",
            level=CheckLevel.IMPORTANT,
            details=str(e),
            exception=e
        )

def check_cpu_cores() -> CheckResult:
    """Report the number of logical CPU cores available."""
    try:
        cores = psutil.cpu_count(logical=True)
        message = "CPU cores"
        details = f"Logical cores: {cores}"
        
        return CheckResult(
            passed=True,
            message=message,
            level=CheckLevel.INFORMATIONAL,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="CPU core check failed",
            level=CheckLevel.INFORMATIONAL,
            details=str(e),
            exception=e
        )

def check_system_ram() -> CheckResult:
    """Report the total amount of system RAM."""
    try:
        ram = psutil.virtual_memory()
        total_gb = ram.total / (1024**3)
        message = "System RAM"
        details = f"Total: {total_gb:.1f}GB, Available: {ram.available/(1024**3):.1f}GB"
        
        return CheckResult(
            passed=True,
            message=message,
            level=CheckLevel.INFORMATIONAL,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="RAM check failed",
            level=CheckLevel.INFORMATIONAL,
            details=str(e),
            exception=e
        )

def check_system_architecture() -> CheckResult:
    """Report the system's CPU architecture."""
    try:
        arch = platform.architecture()
        machine = platform.machine()
        message = "System architecture"
        details = f"Arch: {arch[0]}, Machine: {machine}"
        
        return CheckResult(
            passed=True,
            message=message,
            level=CheckLevel.INFORMATIONAL,
            details=details
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="Architecture check failed",
            level=CheckLevel.INFORMATIONAL,
            details=str(e),
            exception=e
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
        # Detect if output is a TTY (interactive terminal)
        use_color = sys.stdout.isatty()

        # ANSI colors for terminal output
        COLOR_GREEN = "\033[92m" if use_color else ""
        COLOR_YELLOW = "\033[93m" if use_color else ""
        COLOR_RED = "\033[91m" if use_color else ""
        COLOR_RESET = "\033[0m" if use_color else ""

        logger = logging.getLogger()
        handlers = logger.handlers
        details = {}
        compliance_points = 0
        total_points = 5
        feedback_messages = []

        if not handlers:
            msg = f"{COLOR_RED}Logging setup compliance: 0% — No handlers configured{COLOR_RESET}"
            #print(msg)
            return CheckResult(
                passed=False,
                message="Logging configuration",
                level=CheckLevel.IMPORTANT,
                details={
                    "error": "No handlers configured",
                    "compliance_score": 0,
                    "feedback": ["No logging handlers configured — run setup_logging()"],
                    "summary": msg
                }
            )

        handler_info = []
        file_handler_found = False
        file_handler_utf8 = False
        file_handler_exists = False
        console_handler_found = False
        console_handler_unicode = False

        for h in handlers:
            h_type = h.__class__.__name__

            # File handler checks
            if isinstance(h, logging.FileHandler):
                file_handler_found = True
                if getattr(h, 'encoding', '').lower() == 'utf-8':
                    file_handler_utf8 = True
                log_path = Path(getattr(h, 'baseFilename', ''))
                if log_path.exists():
                    file_handler_exists = True

                handler_info.append({
                    'type': h_type,
                    'level': logging.getLevelName(h.level),
                    'encoding': h.encoding,
                    'file_path': str(log_path)
                })

            # Console handler checks
            elif isinstance(h, UnicodeStreamHandler):
                console_handler_found = True
                console_handler_unicode = True
                handler_info.append({
                    'type': h_type,
                    'level': logging.getLevelName(h.level),
                    'stream': getattr(h.stream, 'name', str(h.stream))
                })

            else:
                handler_info.append({
                    'type': h_type,
                    'level': logging.getLevelName(h.level)
                })

        # Score calculation + feedback
        if file_handler_found:
            compliance_points += 1
        else:
            feedback_messages.append("Missing file handler.")

        if file_handler_utf8:
            compliance_points += 1
        else:
            feedback_messages.append("File handler not set to UTF-8 encoding.")

        if file_handler_exists:
            compliance_points += 1
        else:
            feedback_messages.append("Log file does not exist.")

        if console_handler_found:
            compliance_points += 1
        else:
            feedback_messages.append("Missing console handler.")

        if console_handler_unicode:
            compliance_points += 1
        else:
            feedback_messages.append("Console handler is not UnicodeStreamHandler.")

        compliance_score = int((compliance_points / total_points) * 100)
        passed = compliance_score == 100

        # Choose color based on score
        if passed:
            color = COLOR_GREEN
            status_icon = "PASS"
        elif compliance_score >= 60:
            color = COLOR_YELLOW
            status_icon = "WARN"
        else:
            color = COLOR_RED
            status_icon = "FAIL"

        # Build human-readable summary
        if passed:
            human_feedback = f"{color}{status_icon} Logging setup is fully compliant ({compliance_score}%){COLOR_RESET}"
        else:
            human_feedback = (
                f"{color}{status_icon} Logging setup compliance: {compliance_score}% — Issues: "
                + "; ".join(feedback_messages)
                + COLOR_RESET
            )

        details['handlers'] = handler_info
        details['file_handler_found'] = file_handler_found
        details['file_handler_utf8'] = file_handler_utf8
        details['file_handler_exists'] = file_handler_exists
        details['console_handler_found'] = console_handler_found
        details['console_handler_unicode'] = console_handler_unicode
        details['compliance_score'] = compliance_score
        details['feedback'] = feedback_messages
        details['summary'] = human_feedback

        # Print colorized feedback (or plain text if redirected)
        #print(human_feedback)

        return CheckResult(
            passed=passed,
            message="Logging configuration",
            level=CheckLevel.IMPORTANT,
            details=details
        )

    except Exception as e:
        use_color = sys.stdout.isatty()
        COLOR_RED = "\033[91m" if use_color else ""
        COLOR_RESET = "\033[0m" if use_color else ""
        error_msg = f"{COLOR_RED}Logging check failed: {e}{COLOR_RESET}"
        #print(error_msg)
        return CheckResult(
            passed=False,
            message="Logging check failed",
            level=CheckLevel.IMPORTANT,
            details={
                "error": str(e),
                "compliance_score": 0,
                "feedback": ["Exception occurred during logging setup check."],
                "summary": error_msg
            },
            exception=e
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
    """
    try:
        # Detect if output is a TTY
        use_color = sys.stdout.isatty()

        # ANSI colors
        COLOR_GREEN = "\033[92m" if use_color else ""
        COLOR_YELLOW = "\033[93m" if use_color else ""
        COLOR_RED = "\033[91m" if use_color else ""
        COLOR_RESET = "\033[0m" if use_color else ""

        details = {}
        feedback_messages = []
        compliance_points = 0
        #total_points = 5

        require_pytorch = False
        require_tensorflow=False
        
        # Weighted importance of each check
        weights = {
            "PYTHONHASHSEED": 25,
            "CUBLAS_WORKSPACE_CONFIG": 25,
            "numpy_rng": 15,
            "torch_rng": 20 if require_pytorch else 10,
            "tensorflow_rng": 20 if require_tensorflow else 10
        }
        total_points = sum(weights.values())

        # --- PYTHONHASHSEED ---
        hash_seed = os.environ.get('PYTHONHASHSEED')
        hash_seed_ok = hash_seed is not None and hash_seed.isdigit()
        details['PYTHONHASHSEED'] = hash_seed if hash_seed_ok else "Not set"
        if hash_seed_ok:
            compliance_points += weights['PYTHONHASHSEED']
        else:
            feedback_messages.append("PYTHONHASHSEED not set or invalid.")

        # --- CUBLAS_WORKSPACE_CONFIG ---
        cublas_cfg = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        cublas_ok = cublas_cfg == ':4096:8'
        details['CUBLAS_WORKSPACE_CONFIG'] = cublas_cfg if cublas_cfg else "Not set"
        if cublas_ok:
            compliance_points += weights['CUBLAS_WORKSPACE_CONFIG']
        else:
            feedback_messages.append("CUBLAS_WORKSPACE_CONFIG not set to ':4096:8'.")

        # --- NumPy ---
        numpy_ok = False
        try:
            import numpy as np
            _ = np.random.rand()  # trigger RNG
            details['numpy_rng'] = "Configured (seed not directly verifiable)"
            numpy_ok = True
        except Exception as e:
            details['numpy_rng'] = f"Error: {e}"
            feedback_messages.append("NumPy RNG not available.")
        if numpy_ok:
            compliance_points += weights['numpy_rng']

        # --- PyTorch ---
        torch_ok = False
        try:
            import torch
            torch_ok = True
            if torch.cuda.is_available():
                deterministic_ok = getattr(torch.backends.cudnn, "deterministic", False)
                benchmark_ok = getattr(torch.backends.cudnn, "benchmark", None) is False
                details['torch_cuda_deterministic'] = deterministic_ok
                details['torch_cuda_benchmark'] = benchmark_ok
                if not deterministic_ok or not benchmark_ok:
                    torch_ok = False
                    feedback_messages.append("PyTorch CUDA settings not fully deterministic.")
            details['torch_rng'] = "Configured (seed not directly verifiable)"
        except Exception as e:
            details['torch_rng'] = f"Error: {e}"
            feedback_messages.append("PyTorch RNG not available.")
            torch_ok = False
        if torch_ok:
            compliance_points += weights['torch_rng']

        # --- TensorFlow ---
        tf_ok = False
        try:
            import tensorflow as tf
            _ = tf.random.uniform((1,))
            details['tensorflow_rng'] = "Configured (seed not directly verifiable)"
            tf_ok = True
        except ImportError:
            details['tensorflow_rng'] = "Not installed"
            # Not counted as a fail — optional
            compliance_points += weights['tensorflow_rng']
            tf_ok = True
        except Exception as e:
            details['tensorflow_rng'] = f"Error: {e}"
            feedback_messages.append("TensorFlow RNG not available.")
        if tf_ok and 'tensorflow_rng' in details and details['tensorflow_rng'] != "Not installed":
            compliance_points += 1

        # --- Compliance calculation ---
        compliance_score = round((compliance_points / total_points) * 100, 2)
        # Whether check passed or failed (threshold 90%)
        passed = compliance_score >= 90

        # --- Status coloring ---
        if passed:
            color = COLOR_GREEN
            status_icon = "PASS"
        elif compliance_score >= 60:
            color = COLOR_YELLOW
            status_icon = "WARN"
        else:
            color = COLOR_RED
            status_icon = "FAIL"

        # --- Summary feedback ---
        if passed:
            summary = f"{color}{status_icon} Seed config is fully compliant ({compliance_score}%){COLOR_RESET}"
        else:
            summary = (
                f"{color}{status_icon} Seed config compliance: {compliance_score}% — Issues: "
                + "; ".join(feedback_messages)
                + COLOR_RESET
            )

        details['compliance_score'] = compliance_score
        details['feedback'] = feedback_messages
        details['summary'] = summary

        # Print human-readable result
        #print(summary)

        return CheckResult(
            passed=passed,
            message="Reproducibility seed configuration",
            level=CheckLevel.IMPORTANT,
            details=details
        )

    except Exception as e:
        use_color = sys.stdout.isatty()
        COLOR_RED = "\033[91m" if use_color else ""
        COLOR_RESET = "\033[0m" if use_color else ""
        error_msg = f"{COLOR_RED}Seed config check failed: {e}{COLOR_RESET}"
        #print(error_msg)
        return CheckResult(
            passed=False,
            message="Seed check failed",
            level=CheckLevel.IMPORTANT,
            details={
                "error": str(e),
                "compliance_score": 0,
                "feedback": ["Exception occurred during seed config check."],
                "summary": error_msg
            },
            exception=e
        )

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
    """
    seed_config = {
        'base_seed': seed,
        'python_hash_seed': seed,
        'numpy_seed': seed,
        'torch_seed': seed,
        'cuda_deterministic': False,
        'cuda_benchmark': False
    }

    # Set Python seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    seed_config['torch_seed'] = seed
    
    # Configure CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_config.update({
            'cuda_deterministic': True,
            'cuda_benchmark': False,
            'cuda_seeds': [seed] * torch.cuda.device_count()
        })

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed_config['environment'] = {
        'PYTHONHASHSEED': seed,
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'
    }

    # Configure TensorFlow if installed
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
    'packaging': PACKAGING_AVAILABLE
}

# Version information
VERSION_INFO = {
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'optuna': optuna.__version__,
    'rich': getattr(__import__('rich'), '__version__', 'unknown'),
    'plotly': plotly.__version__,
    'sklearn': getattr(__import__('sklearn'), '__version__', 'unknown')
}

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
        
        # Optional dependencies
        'ONNX Runtime': 'Cross-platform inference engine for ONNX models',
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
    
    # Core dependencies with minimum versions (aligned with VERSION_INFO)
    core_deps = {
        'Python': (VERSION_INFO['python'], '>=3.7', True),
        'PyTorch': (VERSION_INFO['torch'], '>=1.8', True),
        'NumPy': (VERSION_INFO['numpy'], '>=1.19', True),
        'Pandas': (VERSION_INFO['pandas'], '>=1.2', True),
        'Scikit-learn': (VERSION_INFO['sklearn'], '>=0.24', True),
        'Optuna': (VERSION_INFO['optuna'], '>=2.8', True),
        'Rich': (VERSION_INFO['rich'], '>=10.0', True),
        # Plotly is optional
        'Plotly': (VERSION_INFO['plotly'], '>=5.0', False)
    }
    
    # Optional dependencies (aligned with OPTIONAL_DEPENDENCIES)
    optional_deps = {
        'ONNX Runtime': (ort.__version__ if OPTIONAL_DEPENDENCIES['onnxruntime'] else 'N/A', '>=1.8', False),
        'NVIDIA ML': (nvml.__version__ if OPTIONAL_DEPENDENCIES['nvml'] else 'N/A', '>=11.0', False),
        'Torch JIT': ('Available' if OPTIONAL_DEPENDENCIES['torch_jit'] else 'N/A', None, False),
        'Cryptography': ('Available' if OPTIONAL_DEPENDENCIES['crypto'] else 'N/A', None, False),
        'Database': ('Available' if OPTIONAL_DEPENDENCIES['database'] else 'N/A', None, False),
        'Sklearn Anomaly': ('Available' if OPTIONAL_DEPENDENCIES['sklearn_anomaly'] else 'N/A', None, False),
        'Statsmodels': ('Available' if OPTIONAL_DEPENDENCIES['statsmodels'] else 'N/A', None, False),
        'Numba': ('Available' if OPTIONAL_DEPENDENCIES['numba'] else 'N/A', None, False),
        'Memory Profiler': ('Available' if OPTIONAL_DEPENDENCIES['memory_profiler'] else 'N/A', None, False),
        'Line Profiler': ('Available' if OPTIONAL_DEPENDENCIES['line_profiler'] else 'N/A', None, False),
        'Packaging': ('Available' if OPTIONAL_DEPENDENCIES['packaging'] else 'N/A', None, False)
    }
    
    # Check all dependencies
    for name, (version, req, required) in {**core_deps, **optional_deps}.items():
        if not include_optional and not required:
            continue
            
        try:
            if version == 'N/A' or version == 'Available' or req is None:
                status = 'OK' if version == 'Available' else 'MISSING'
                meets_req = version == 'Available'
            else:
                from packaging import version as pkg_version
                meets_req = pkg_version.parse(version) >= pkg_version.parse(req)
                status = 'OK' if meets_req else 'WARNING'
        except Exception:
            meets_req = False
            status = 'UNKNOWN'
        
        version_info[name] = {
            'version': version,
            'required_version': req,
            'status': status,
            'required': required,
            'description': get_dependency_description(name),
            'compatible': meets_req,
            'available': version != 'N/A' and version != 'unknown'
        }
    
    return version_info

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

def check_core_dependencies():
    """Check and report status of core dependencies."""
    dependency_status = {}
    
    # Check core dependencies
    core_deps = {
        'torch': torch.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': getattr(__import__('sklearn'), '__version__', 'unknown')
    }
    
    for dep, version in core_deps.items():
        dependency_status[dep] = {'version': version, 'status': 'required', 'available': True}
    
    # Check optional dependencies
    for dep, available in OPTIONAL_DEPENDENCIES.items():
        dependency_status[dep] = {
            'status': 'optional',
            'available': available,
            'description': get_dependency_description(dep)
        }
    
    return dependency_status

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
        clear_memory()
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
    clear_memory()
    
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

def save_change_log(changes: Dict) -> None:
    """Save configuration changes to a log file."""
    log_dir = LOG_DIR / "deep_learning_config_changes"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"changes_{timestamp}.json"
    
    try:
        with open(log_file, 'w') as f:
            json.dump(changes, f, indent=2)
        logger.debug(f"Saved configuration change log to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to save change log: {str(e)}")

# Configuration for Testing Different Architectures
STABILITY_CONFIG = {
    'model_type': 'SimpleAutoencoder',
    'use_batch_norm': True,
    'dropout_rates': [0.3, 0.25],
    'gradient_clip': 1.0,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'fn_cost': 2.0
}

PERFORMANCE_CONFIG = {
    'model_type': 'AutoencoderEnsemble',
    'num_models': 3,
    'use_batch_norm': True,
    'dropout_rates': [0.2, 0.15],
    'gradient_clip': 0.5,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5
}

# Model Architecture Options
MODEL_VARIANTS = {
    'SimpleAutoencoder': SimpleAutoencoder,
    'EnhancedAutoencoder': EnhancedAutoencoder,
    'AutoencoderEnsemble': AutoencoderEnsemble
}

# Preset Configurations for Testing Different Architectures
DEFAULT_PRESET = {
    'metadata': {
        'description': 'Default balanced configuration for general use',
        'version': '2.0',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 6,
            'cpu_cores': 4,
            'ram_gb': 8
        },
        'compatibility': ['SimpleAutoencoder', 'EnhancedAutoencoder', 'AutoencoderEnsemble']
    },
    'training': {
        'batch_size': 64,
        'epochs': 50,
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
        'skip_connection': True,
        'residual_blocks': False
    },
    'security': {
        'percentile': 95,
        'attack_threshold': 0.3,
        'false_negative_cost': 2.0,
        'enable_security_metrics': True,
        'anomaly_threshold_strategy': 'dynamic_percentile',
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
    }
}

STABILITY_PRESET = {
    'metadata': {
        'description': 'High stability configuration for reliable training',
        'version': '2.0',
        'created': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 4,
            'cpu_cores': 2,
            'ram_gb': 4
        },
        'compatibility': ['SimpleAutoencoder', 'EnhancedAutoencoder']
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
        'skip_connection': False,
        'residual_blocks': False
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
    }
}

PERFORMANCE_PRESET = {
    'metadata': {
        'description': 'High-performance configuration for production deployment',
        'version': '2.0',
        'created': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 8,
            'cpu_cores': 8,
            'ram_gb': 16
        },
        'compatibility': ['EnhancedAutoencoder', 'AutoencoderEnsemble']
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
        'residual_blocks': True
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
    }
}

BASELINE_PRESET = {
    'metadata': {
        'description': 'Standardized configuration for benchmarking',
        'version': '2.0',
        'created': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 6,
            'cpu_cores': 4,
            'ram_gb': 8
        },
        'compatibility': ['EnhancedAutoencoder']
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
        'skip_connection': True,
        'residual_blocks': False
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
    }
}

DEBUG_PRESET = {
    'metadata': {
        'description': 'Lightweight configuration for debugging',
        'version': '2.0',
        'created': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 2,
            'cpu_cores': 1,
            'ram_gb': 2
        },
        'compatibility': ['SimpleAutoencoder']
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
        'skip_connection': False,
        'residual_blocks': False
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
    }
}

# New Additional Presets
LIGHTWEIGHT_PRESET = {
    'metadata': {
        'description': 'Lightweight configuration for edge devices',
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 1,
            'cpu_cores': 1,
            'ram_gb': 2
        },
        'compatibility': ['SimpleAutoencoder']
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
        'skip_connection': False,
        'residual_blocks': False
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
    }
}

ADVANCED_PRESET = {
    'metadata': {
        'description': 'Advanced configuration for research experiments',
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'recommended_hardware': {
            'gpu_memory_gb': 16,
            'cpu_cores': 16,
            'ram_gb': 32
        },
        'compatibility': ['EnhancedAutoencoder', 'AutoencoderEnsemble']
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
        'residual_blocks': True
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
    }
}

# Consolidated Preset Configurations
PRESET_CONFIGS = {
    'default': DEFAULT_PRESET,
    'stability': STABILITY_PRESET,
    'performance': PERFORMANCE_PRESET,
    'baseline': BASELINE_PRESET,
    'debug': DEBUG_PRESET,
    'lightweight': LIGHTWEIGHT_PRESET,
    'advanced': ADVANCED_PRESET
}

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
    
    # First check if there's a loaded config with an active preset
    loaded_config = load_config()
    current_preset = loaded_config.get('presets', {}).get('current_preset')
    
    # If we have an active preset, start with that configuration
    if current_preset and current_preset in PRESET_CONFIGS:
        base_config = deepcopy(PRESET_CONFIGS[current_preset])
        logger.info(f"Using preset configuration: {current_preset}")
    else:
        # Otherwise use the default configuration structure
        base_config = {
            "metadata": {
                "config_version": "2.1",
                "config_type": "autoencoder",
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "system": {
                    "python_version": platform.python_version(),
                    "pytorch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "hostname": platform.node(),
                    "os": platform.system()
                },
                "preset_used": current_preset if current_preset else "none"
            },
            "training": {
                "batch_size": DEFAULT_BATCH_SIZE,
                "epochs": DEFAULT_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "patience": EARLY_STOPPING_PATIENCE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "mixed_precision": MIXED_PRECISION,
                "num_workers": NUM_WORKERS,
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau"
            },
            "model": {
                "encoding_dim": DEFAULT_ENCODING_DIM,
                "hidden_dims": HIDDEN_LAYER_SIZES,
                "dropout_rates": DROPOUT_RATES,
                "activation": ACTIVATION,
                "activation_param": ACTIVATION_PARAM,
                "normalization": NORMALIZATION,
                "use_batch_norm": USE_BATCH_NORM,
                "use_layer_norm": USE_LAYER_NORM,
                "diversity_factor": DIVERSITY_FACTOR,
                "min_features": MIN_FEATURES,
                "num_models": NUM_MODELS,
                "skip_connection": True,
                "residual_blocks": False,
                "model_types": list(MODEL_VARIANTS.keys()),
                "available_activations": ["relu", "leaky_relu", "gelu"],
                "available_normalizations": ["batch", "layer", None]
            },
            "security": {
                "percentile": DEFAULT_PERCENTILE,
                "attack_threshold": DEFAULT_ATTACK_THRESHOLD,
                "false_negative_cost": FALSE_NEGATIVE_COST,
                "enable_security_metrics": SECURITY_METRICS,
                "anomaly_threshold_strategy": "percentile",
                "early_warning_threshold": 0.25
            },
            "data": {
                "normal_samples": NORMAL_SAMPLES,
                "attack_samples": ATTACK_SAMPLES,
                "features": FEATURES,
                "normalization": "standard",
                "anomaly_factor": ANOMALY_FACTOR,
                "random_state": RANDOM_STATE,
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
                "available_presets": list(PRESET_CONFIGS.keys()),
                "current_preset": current_preset,
                "preset_configs": {k: v["metadata"]["description"] 
                                  for k, v in PRESET_CONFIGS.items() 
                                  if "metadata" in v and "description" in v["metadata"]},
                "custom_presets_available": list_custom_presets()
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
    
    # Merge with any loaded configuration (preset values take precedence)
    if loaded_config:
        base_config = deep_update(base_config, loaded_config)
    
    # Ensure model architecture compatibility if preset was used
    if current_preset:
        model_type = base_config['model'].get('model_type')
        if MODEL_VARIANTS and not validate_model_preset_compatibility(model_type, base_config):
            logger.warning(f"Model type {model_type} may not be fully compatible with preset {current_preset}")
            # Apply fallback to simple model if compatibility issues
            if model_type == 'AutoencoderEnsemble' and base_config['model'].get('num_models', 1) < 1:
                base_config['model']['num_models'] = 1
    
    # Cache the result
    _cached_config = base_config
    _config_cache_time = current_time
    
    return base_config

def invalidate_config_cache():
    """Invalidate the configuration cache to force reload."""
    global _cached_config, _config_cache_time
    _cached_config = None
    _config_cache_time = None

def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary with enhanced conflict resolution."""
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            # Special handling for model configuration to prevent invalid combinations
            if key == 'model':
                # Only validate model type if MODEL_VARIANTS is initialized and not empty
                if 'model_type' in value and MODEL_VARIANTS and value['model_type'] not in MODEL_VARIANTS:
                    logger.warning(f"Ignoring invalid model type: {value['model_type']}")
                    del value['model_type']
                
                # Ensure hidden_dims and dropout_rates match in length
                if 'hidden_dims' in value and 'dropout_rates' in value:
                    if len(value['hidden_dims']) != len(value['dropout_rates']):
                        min_length = min(len(value['hidden_dims']), len(value['dropout_rates']))
                        value['hidden_dims'] = value['hidden_dims'][:min_length]
                        value['dropout_rates'] = value['dropout_rates'][:min_length]
                        logger.warning("Adjusted hidden_dims and dropout_rates to matching lengths")
            
            original[key] = deep_update(original[key], value)
        else:
            # Skip None values to avoid overwriting with null
            if value is not None:
                original[key] = value
    return original

def save_config(config: Dict, config_path: Path = CONFIG_FILE) -> None:
    """Save config with enhanced metadata and backup handling."""
    try:
        # Prepare the full configuration with metadata
        full_config = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "version": "2.1",
                "system": {
                    "python_version": platform.python_version(),
                    "pytorch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "hostname": platform.node(),
                    "os": platform.system()
                },
                "preset_used": config.get('presets', {}).get('current_preset', 'none'),
                "model_type": config.get('model', {}).get('model_type', 'unknown')
            },
            "config": config
        }
        
        # Create backup if exists (with versioning)
        if config_path.exists():
            backup_dir = config_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{config_path.stem}_v{full_config['metadata']['version']}_{timestamp}{config_path.suffix}"
            backup_path = backup_dir / backup_name
            shutil.copy(config_path, backup_path)
            logger.info(f"Config backup created at {backup_path}")
        
        # Atomic write operation
        temp_path = config_path.with_suffix(".tmp")
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=4, ensure_ascii=False)
        
        # Replace the original file atomically
        temp_path.replace(config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # Invalidate cache after saving
        invalidate_config_cache()
        
        # If this is a preset, also save it to the custom presets directory
        if config.get('presets', {}).get('current_preset'):
            preset_name = config['presets']['current_preset']
            if preset_name not in PRESET_CONFIGS:
                save_custom_preset(preset_name, config)
                
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}", exc_info=True)
        raise RuntimeError(f"Configuration save failed: {str(e)}") from e

def load_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load config file with enhanced validation and error recovery."""
    try:
        if not config_path.exists():
            logger.info(f"No configuration file found at {config_path}, using defaults")
            return {}
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        loaded_config = config_data.get("config", {})
        
        # Validate basic structure
        if not isinstance(loaded_config, dict):
            raise ValueError("Invalid configuration format")
            
        # Check for deprecated version
        if config_data.get("metadata", {}).get("version", "1.0") == "1.0":
            logger.warning("Loading legacy config format, attempting conversion")
            loaded_config = convert_legacy_config(loaded_config)
        
        logger.info(f"Loaded configuration from {config_path}")
        logger.debug(f"Configuration metadata: {config_data.get('metadata', {})}")
        
        return loaded_config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        return {}

def initialize_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Initialize or load configuration with preset awareness."""
    # Load existing config if available
    loaded_config = load_config(config_path)
    
    # Get the current default configuration
    default_config = get_current_config()
    
    if loaded_config:
        # Check if we need to migrate from older version
        if loaded_config.get('metadata', {}).get('version', '1.0') != default_config.get('metadata', {}).get('version', '2.1'):
            loaded_config = migrate_config(loaded_config, default_config)
        
        # Merge configurations (loaded config overrides defaults)
        merged_config = deep_update(default_config, loaded_config)
        
        # Validate the merged configuration
        try:
            validate_config(merged_config)
        except ValueError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            if sys.stdin.isatty():  # Interactive mode
                if prompt_user("Continue with default config?", default=True):
                    merged_config = default_config
                else:
                    raise
            else:
                merged_config = default_config
        
        # Save the merged configuration
        save_config(merged_config, config_path)
        return merged_config
    else:
        # No config exists, save and return defaults
        save_config(default_config, config_path)
        return default_config

# Model variants validation
def validate_model_variants(logger: logging.Logger):
    """Validate all registered model variants."""
    variant_status = {}
    for name, variant_class in MODEL_VARIANTS.items():
        try:
            # Test instantiation with minimal parameters
            test_instance = variant_class(input_dim=10, encoding_dim=5)
            
            # Set model to eval mode to avoid batch norm issues
            test_instance.eval()
            
            # Test with batch size > 1 for batch norm compatibility
            # Changed from size 1 to size 2
            test_input = torch.randn(2, 10)
            with torch.no_grad():
                _ = test_instance(test_input)
            
            variant_status[name] = 'available'
            del test_instance
        except Exception as e:
            variant_status[name] = f'error: {str(e)}'
            logger.warning(f"Model variant {name} failed validation: {e}")
    return variant_status

# Model variants initialization
def initialize_model_variants() -> None:
    """Initialize MODEL_VARIANTS dictionary with validation and error recovery."""
    global MODEL_VARIANTS
    
    # Clear existing variants
    MODEL_VARIANTS = {}
    
    # Get current configuration to use for initialization
    try:
        current_config = get_current_config()
        model_config = current_config.get('model', {})
    except Exception as e:
        logger.warning(f"Could not load current config, using defaults: {e}")
        model_config = {}
    
    # Use configuration values with fallbacks to global constants
    compatible_hidden_dims = model_config.get('hidden_dims', HIDDEN_LAYER_SIZES.copy())
    compatible_dropout_rates = model_config.get('dropout_rates', DROPOUT_RATES.copy())
    encoding_dim = model_config.get('encoding_dim', DEFAULT_ENCODING_DIM)
    activation = model_config.get('activation', ACTIVATION)
    activation_param = model_config.get('activation_param', ACTIVATION_PARAM)
    normalization = model_config.get('normalization', NORMALIZATION)
    num_models = model_config.get('num_models', NUM_MODELS)
    diversity_factor = model_config.get('diversity_factor', DIVERSITY_FACTOR)
    
    # Ensure lists are valid
    if not isinstance(compatible_hidden_dims, list):
        compatible_hidden_dims = [compatible_hidden_dims] if isinstance(compatible_hidden_dims, int) else [128, 64]
        logger.warning(f"Converted hidden_dims to list: {compatible_hidden_dims}")
    
    if not isinstance(compatible_dropout_rates, list):
        compatible_dropout_rates = [compatible_dropout_rates] if isinstance(compatible_dropout_rates, (int, float)) else [0.2, 0.15]
        logger.warning(f"Converted dropout_rates to list: {compatible_dropout_rates}")
    
    # Fix length mismatch if it exists
    if len(compatible_hidden_dims) != len(compatible_dropout_rates):
        if len(compatible_dropout_rates) < len(compatible_hidden_dims):
            # Extend dropout_rates
            last_dropout = compatible_dropout_rates[-1] if compatible_dropout_rates else 0.2
            while len(compatible_dropout_rates) < len(compatible_hidden_dims):
                compatible_dropout_rates.append(max(0.1, last_dropout * 0.8))
        else:
            # Truncate dropout_rates
            compatible_dropout_rates = compatible_dropout_rates[:len(compatible_hidden_dims)]
        
        logger.info(f"Adjusted dropout_rates for compatibility: {compatible_dropout_rates}")
    
    # Define expected model classes and their initialization parameters
    model_definitions = {
        'SimpleAutoencoder': {
            'class': SimpleAutoencoder,
            'params': {
                # Test input size
                'input_dim': 20,
                'encoding_dim': encoding_dim
            },
            'required_config': ['encoding_dim']
        },
        'EnhancedAutoencoder': {
            'class': EnhancedAutoencoder,
            'params': {
                'input_dim': 20,
                'encoding_dim': encoding_dim,
                'hidden_dims': compatible_hidden_dims,
                'dropout_rates': compatible_dropout_rates,
                'activation': activation,
                'activation_param': activation_param,
                'normalization': normalization
            },
            'required_config': ['encoding_dim', 'hidden_dims', 'dropout_rates', 'activation']
        },
        'AutoencoderEnsemble': {
            'class': AutoencoderEnsemble,
            'params': {
                'input_dim': 20,
                # Ensure at least 1
                'num_models': max(1, num_models),
                'encoding_dim': encoding_dim,
                'diversity_factor': diversity_factor
            },
            'required_config': ['num_models', 'encoding_dim', 'diversity_factor']
        }
    }
    
    # Track initialization status
    initialization_stats = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    # Initialize each model variant with validation
    for name, definition in model_definitions.items():
        try:
            # Check if model class exists
            if definition['class'] is None:
                logger.debug(f"Model class not found: {name}")
                initialization_stats['skipped'].append(name)
                continue
            
            # Validate required configuration parameters
            missing_config = []
            for req_param in definition.get('required_config', []):
                if req_param not in model_config:
                    missing_config.append(req_param)
            
            if missing_config:
                logger.warning(f"Missing configuration for {name}: {missing_config}")
            
            # Test instantiation with error recovery
            try:
                model = definition['class'](**definition['params'])
                MODEL_VARIANTS[name] = definition['class']
                initialization_stats['successful'].append(name)
                logger.debug(f"Successfully initialized model variant: {name}")
                
                # Validate model is functional - use eval mode and batch size > 1
                model.eval()
                test_input = torch.randn(2, definition['params']['input_dim'])
                with torch.no_grad():
                    _ = model(test_input)
                    
            except Exception as init_error:
                logger.warning(f"Model {name} instantiated but failed validation: {init_error}")
                # Try with simplified parameters
                if name == 'EnhancedAutoencoder':
                    simplified_params = {
                        'input_dim': 20,
                        'encoding_dim': encoding_dim,
                        'hidden_dims': [64],
                        'dropout_rates': [0.2],
                        'activation': 'relu',
                        'normalization': None
                    }
                    try:
                        model = definition['class'](**simplified_params)
                        MODEL_VARIANTS[name] = definition['class']
                        initialization_stats['successful'].append(name)
                        logger.info(f"Successfully initialized {name} with simplified parameters")
                    except Exception:
                        initialization_stats['failed'].append(name)
                else:
                    initialization_stats['failed'].append(name)
                    
        except Exception as e:
            logger.warning(f"Failed to initialize model variant {name}: {str(e)}")
            logger.debug(f"Parameters used: {definition['params']}")
            initialization_stats['failed'].append(name)
    
    # Log initialization summary
    logger.info(f"Model variants initialized: {len(initialization_stats['successful'])}/{len(model_definitions)}")
    if initialization_stats['failed']:
        logger.warning(f"Failed to initialize: {', '.join(initialization_stats['failed'])}")
    
    # Ensure at least one model variant is available
    if not MODEL_VARIANTS:
        logger.error("No model variants could be initialized")
        raise RuntimeError("No valid model variants available")

# Model architecture comparison
def compare_model_architectures(input_dim: int = None) -> Dict[str, Dict]:
    """Compare parameter counts and complexity of different model architectures"""
    results = {}
    
    # Get current configuration for accurate comparison
    try:
        current_config = get_current_config()
        model_config = current_config.get('model', {})
        data_config = current_config.get('data', {})
        
        # Use configured input dimension or default
        if input_dim is None:
            input_dim = data_config.get('features', FEATURES)
    except Exception as e:
        logger.warning(f"Could not load config for comparison, using defaults: {e}")
        if input_dim is None:
            input_dim = FEATURES
        model_config = {}
    
    # Initialize model variants if empty
    if not MODEL_VARIANTS:
        try:
            initialize_model_variants()
        except Exception as e:
            logger.error(f"Failed to initialize model variants: {e}")
            return {'error': f"Model initialization failed: {str(e)}"}
    
    # Extract configuration parameters with fallbacks
    encoding_dim = model_config.get('encoding_dim', DEFAULT_ENCODING_DIM)
    hidden_dims = model_config.get('hidden_dims', HIDDEN_LAYER_SIZES)
    dropout_rates = model_config.get('dropout_rates', DROPOUT_RATES)
    activation = model_config.get('activation', ACTIVATION)
    activation_param = model_config.get('activation_param', ACTIVATION_PARAM)
    normalization = model_config.get('normalization', NORMALIZATION)
    num_models = model_config.get('num_models', NUM_MODELS)
    diversity_factor = model_config.get('diversity_factor', DIVERSITY_FACTOR)
    
    for model_name, model_class in MODEL_VARIANTS.items():
        try:
            # Create model with current configuration
            if model_name == 'SimpleAutoencoder':
                model = model_class(
                    input_dim=input_dim,
                    encoding_dim=encoding_dim
                )
            elif model_name == 'EnhancedAutoencoder':
                # Ensure list compatibility
                if not isinstance(hidden_dims, list):
                    hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else [128, 64]
                if not isinstance(dropout_rates, list):
                    dropout_rates = [dropout_rates] if isinstance(dropout_rates, (int, float)) else [0.2, 0.15]
                
                # Ensure matching lengths
                min_length = min(len(hidden_dims), len(dropout_rates))
                model = model_class(
                    input_dim=input_dim,
                    encoding_dim=encoding_dim,
                    hidden_dims=hidden_dims[:min_length],
                    dropout_rates=dropout_rates[:min_length],
                    activation=activation,
                    activation_param=activation_param,
                    normalization=normalization
                )
            elif model_name == 'AutoencoderEnsemble':
                model = model_class(
                    input_dim=input_dim,
                    num_models=max(1, num_models),
                    encoding_dim=encoding_dim,
                    diversity_factor=diversity_factor
                )
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue
                
            # Calculate parameters and complexity metrics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage (rough approximation)
            param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            # Calculate training complexity score
            complexity_score = 1.0
            if model_name == 'AutoencoderEnsemble':
                complexity_score *= num_models
            if normalization in ['batch', 'layer']:
                complexity_score *= 1.2
            if activation in ['gelu', 'leaky_relu']:
                complexity_score *= 1.1
            
            # Determine complexity level with thresholds from config
            if total_params < 10000:
                complexity = "Low"
            elif total_params < 100000:
                complexity = "Medium" 
            elif total_params < 1000000:
                complexity = "High"
            else:
                complexity = "Very High"
            
            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'memory_mb': param_memory_mb,
                'complexity_score': complexity_score,
                'complexity_level': complexity,
                'model_class': model_class.__name__,
                'config_used': {
                    'input_dim': input_dim,
                    'encoding_dim': encoding_dim,
                    'hidden_dims': hidden_dims if model_name == 'EnhancedAutoencoder' else None,
                    'num_models': num_models if model_name == 'AutoencoderEnsemble' else None
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Failed to analyze {model_name}: {error_msg}")
            results[model_name] = {
                'error': error_msg,
                'config_attempted': {
                    'input_dim': input_dim,
                    'encoding_dim': encoding_dim
                }
            }
    
    # Add comparison metadata
    results['_metadata'] = {
        'input_dimension': input_dim,
        'config_source': 'current_config' if model_config else 'defaults',
        'comparison_timestamp': datetime.now().isoformat(),
        'available_variants': len(MODEL_VARIANTS),
        'successful_comparisons': len([r for r in results.values() if isinstance(r, dict) and 'error' not in r])
    }
    
    return results

# Display model comparison
def display_model_comparison():
    """Display model architecture comparison in a formatted table with enhanced information"""
    try:
        results = compare_model_architectures()
        
        # Check for initialization errors
        if 'error' in results:
            print(f"\n[ERROR] Model Comparison Failed: {results['error']}")
            return
        
        metadata = results.pop('_metadata', {})
        
        print(f"\nModel Architecture Comparison (Input Dim: {metadata.get('input_dimension', 'unknown')})")
        print("=" * 100)
        print(f"{'Model':<20} | {'Params':>12} | {'Trainable':>12} | {'Memory (MB)':>12} | {'Complexity':>15} | Status")
        print("=" * 100)
        
        # Sort by complexity for better display
        sorted_results = sorted(
            [(name, stats) for name, stats in results.items()],
            key=lambda x: x[1].get('total_params', 0) if 'error' not in x[1] else 0
        )
        
        successful_models = []
        failed_models = []
        
        for model_name, stats in sorted_results:
            if 'error' in stats:
                error_msg = stats['error'][:30] + "..." if len(stats['error']) > 30 else stats['error']
                print(f"{model_name:<20} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'Error':>15} | {error_msg}")
                failed_models.append(model_name)
            else:
                complexity_display = f"{stats['complexity_level']} ({stats['complexity_score']:.1f})"
                print(f"{model_name:<20} | {stats['total_params']:>12,} | {stats['trainable_params']:>12,} | {stats['memory_mb']:>12.1f} | {complexity_display:>15} | Ready")
                successful_models.append(model_name)
        
        print("=" * 100)
        
        # Display summary and recommendations
        print(f"\nSummary:")
        print(f"  1. Successful models: {len(successful_models)}/{len(results)}")
        print(f"  2. Configuration source: {metadata.get('config_source', 'unknown')}")
        print(f"  3. Analysis time: {metadata.get('comparison_timestamp', 'unknown')}")
        
        if failed_models:
            print(f"  [WARNING] Failed models: {', '.join(failed_models)}")
        
        print(f"\nRecommendations:")
        
        # Give specific recommendations based on available models
        if successful_models:
            for model_name in successful_models:
                stats = dict(sorted_results)[model_name]
                if 'error' not in stats:
                    if model_name == 'SimpleAutoencoder':
                        print(f"  1. {model_name}: Best for debugging, fast prototyping, resource-constrained environments")
                    elif model_name == 'EnhancedAutoencoder':
                        print(f"  2. {model_name}: Balanced performance, good for production, moderate resource usage")
                    elif model_name == 'AutoencoderEnsemble':
                        print(f"  3. {model_name}: Highest accuracy, best for critical applications, requires more resources")
        else:
            print("  [WARNING] No models available - check system configuration and try reinitializing")
        
        # Configuration-specific recommendations
        current_config = get_current_config()
        preset_name = current_config.get('presets', {}).get('current_preset')
        if preset_name:
            print(f"  [+] Current preset '{preset_name}' optimized for: {PRESET_CONFIGS[preset_name]['metadata']['description']}")
        
        # Hardware recommendations
        try:
            hw_info = check_hardware()
            if hw_info.get('gpu_available'):
                print(f"  [+] GPU detected: Consider using larger models for better performance")
            else:
                print(f"  [+] CPU-only: SimpleAutoencoder recommended for optimal performance")
        except:
            pass
            
    except Exception as e:
        logger.error(f"Failed to display model comparison: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to display model comparison: {str(e)}")
        print("Try running 'initialize_model_variants()' first or check your configuration.")

# Helper functions for preset and configuration management
def validate_model_preset_compatibility(model_type: str, preset_config: Dict) -> bool:
    """Check if a model type is compatible with a preset configuration."""
    if model_type not in MODEL_VARIANTS:
        return False
    
    # Get preset's compatible models (if specified)
    compatible_models = preset_config.get('metadata', {}).get('compatibility', list(MODEL_VARIANTS.keys()))
    
    if model_type not in compatible_models:
        return False
    
    # Special checks for ensemble models
    if model_type == 'AutoencoderEnsemble':
        if preset_config['model'].get('num_models', 1) < 1:
            return False
    
    # Check feature dimensions
    if preset_config['model'].get('encoding_dim', 1) < 1:
        return False
    
    return True

def list_custom_presets() -> List[str]:
    """List all available custom presets."""
    custom_dir = CONFIG_DIR / "custom_presets"
    if not custom_dir.exists():
        return []
    
    return [f.stem.replace("preset_", "") for f in custom_dir.glob("preset_*.json")]

def save_custom_preset(name: str, config: Dict) -> Path:
    """Save a custom preset configuration."""
    custom_dir = CONFIG_DIR / "custom_presets"
    custom_dir.mkdir(exist_ok=True)
    
    # Create safe filename
    safe_name = "".join(c for c in name.lower() if c.isalnum() or c in (' ', '_')).rstrip()
    filename = f"preset_{safe_name}.json"
    filepath = custom_dir / filename
    
    preset_data = {
        "metadata": {
            "name": name,
            "description": f"Custom preset '{name}'",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "model_type": config.get('model', {}).get('model_type'),
            "compatibility": config.get('metadata', {}).get('compatibility', list(MODEL_VARIANTS.keys()))
        },
        "config": config
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(preset_data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Saved custom preset '{name}' to {filepath}")
    return filepath

def migrate_config(legacy_config: Dict, new_template: Dict = None) -> Dict:
    """
    Migrate an older configuration to the current version using the provided presets as templates.
    
    Args:
        legacy_config: The old configuration dictionary to migrate from
        new_template: Optional template to use as base (defaults to DEFAULT_PRESET)
        
    Returns:
        New configuration dictionary with migrated values
        
    Raises:
        ValueError: If legacy_config is invalid
    """
    if not isinstance(legacy_config, dict):
        raise ValueError("legacy_config must be a dictionary")
    
    logger.info("Migrating legacy configuration to current format")
    
    # Use DEFAULT_PRESET as template if none provided
    template = deepcopy(new_template) if new_template else deepcopy(DEFAULT_PRESET)
    migrated_config = deepcopy(template)
    
    # Comprehensive key mapping with transformations
    key_mapping = {
        # Training parameters
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
        
        # Model parameters
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
        
        # Security parameters
        'percentile': ('security', 'percentile'),
        'attack_threshold': ('security', 'attack_threshold'),
        'false_negative_cost': ('security', 'false_negative_cost'),
        'enable_security_metrics': ('security', 'enable_security_metrics'),
        'anomaly_threshold_strategy': ('security', 'anomaly_threshold_strategy'),
        'early_warning_threshold': ('security', 'early_warning_threshold'),
        
        # Data parameters
        'normal_samples': ('data', 'normal_samples'),
        'attack_samples': ('data', 'attack_samples'),
        'features': ('data', 'features'),
        'normalization': ('data', 'normalization'),
        'anomaly_factor': ('data', 'anomaly_factor'),
        'random_state': ('data', 'random_state'),
        'validation_split': ('data', 'validation_split'),
        'test_split': ('data', 'test_split'),
        'synthetic_generation.cluster_variance': ('data', 'synthetic_generation', 'cluster_variance'),
        'synthetic_generation.anomaly_sparsity': ('data', 'synthetic_generation', 'anomaly_sparsity'),
        
        # Monitoring parameters
        'metrics_frequency': ('monitoring', 'metrics_frequency'),
        'checkpoint_frequency': ('monitoring', 'checkpoint_frequency'),
        'tensorboard_logging': ('monitoring', 'tensorboard_logging'),
        'console_logging_level': ('monitoring', 'console_logging_level'),
        
        # Special transformations
        'legacy_list_value': lambda v: {'model': {'hidden_dims': [x*2 for x in v]}} if isinstance(v, list) else None,
        'old_anomaly_factor': lambda v: {'data': {'anomaly_factor': v * 1.5}} if isinstance(v, (int, float)) else None
    }
    
    # Track migration statistics
    migration_stats = {
        'mapped_keys': 0,
        'skipped_keys': 0,
        'transformed_keys': 0,
        'invalid_values': 0
    }
    
    # Apply mapped values with enhanced handling
    for old_key, new_path in key_mapping.items():
        # Handle dot notation for nested legacy keys
        legacy_keys = old_key.split('.')
        try:
            legacy_value = legacy_config
            for k in legacy_keys:
                if k not in legacy_value:
                    raise KeyError(f"Key {k} not found")
                legacy_value = legacy_value[k]
        except KeyError:
            migration_stats['skipped_keys'] += 1
            continue
            
        try:
            # Handle transformation functions
            if callable(new_path):
                transformed = new_path(legacy_value)
                if transformed:
                    for section, values in transformed.items():
                        if section in migrated_config:
                            migrated_config[section].update(values)
                        else:
                            migrated_config[section] = values
                    migration_stats['transformed_keys'] += 1
                continue
            
            # Handle nested path assignment
            if isinstance(new_path, (list, tuple)):
                target = migrated_config
                for key in new_path[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                
                # Special handling for different types
                if isinstance(legacy_value, list) and isinstance(target.get(new_path[-1]), list):
                    # Merge lists while preserving template's length
                    template_val = target.get(new_path[-1], [])
                    target[new_path[-1]] = [
                        legacy_value[i] if i < len(legacy_value) else template_val[i]
                        for i in range(max(len(legacy_value), len(template_val)))
                    ]
                else:
                    target[new_path[-1]] = legacy_value
                
                migration_stats['mapped_keys'] += 1
                
        except Exception as e:
            logger.warning(f"Could not migrate {old_key}: {str(e)}")
            migration_stats['invalid_values'] += 1
            continue
    
    # Add migration metadata
    migrated_config['metadata']['migration'] = {
        'source_version': '1.x',
        'target_version': '2.0',
        'timestamp': datetime.now().isoformat(),
        'stats': migration_stats,
        'preset_used': template['metadata']['description'] if new_template else 'DEFAULT_PRESET',
        'compatibility_checked': False
    }
    
    # Validate model compatibility
    model_type = migrated_config['model']['model_type']
    if model_type not in MODEL_VARIANTS:
        logger.warning(f"Invalid model type '{model_type}' in migrated config, defaulting to SimpleAutoencoder")
        migrated_config['model']['model_type'] = 'SimpleAutoencoder'
        migrated_config['metadata']['migration']['compatibility_checked'] = True
        migrated_config['metadata']['migration']['original_model_type'] = model_type
    
    return migrated_config

def convert_legacy_config(
    legacy_config: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    preset_similarity_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convert legacy configuration to current format using intelligent preset matching.
    
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
    
    logger.info("Initiating legacy configuration conversion")
    
    # --- Step 0: Threshold Determination ---
    def determine_threshold() -> float:
        """Determine the appropriate similarity threshold with fallbacks."""
        # Argument precedence
        if preset_similarity_threshold is not None:
            if 0 < preset_similarity_threshold <= 1:
                logger.info(f"Using provided threshold: {preset_similarity_threshold:.2f}")
                return preset_similarity_threshold
            logger.warning("Invalid argument threshold, using fallbacks")
        
        # Config file precedence
        if config and config.get("migration", {}).get("preset_similarity_threshold"):
            try:
                threshold = float(config["migration"]["preset_similarity_threshold"])
                if 0 < threshold <= 1:
                    logger.info(f"Using config threshold: {threshold:.2f}")
                    return threshold
            except (ValueError, TypeError):
                logger.warning("Invalid config threshold, using default")
        
        # Default value
        default_threshold = 0.05
        logger.info(f"Using default threshold: {default_threshold:.2f}")
        return default_threshold
    
    similarity_threshold = determine_threshold()
    
    # --- Step 1: Preset Scoring System ---
    class PresetScorer:
        """Advanced scoring engine with dynamic weights and normalization."""
        
        def __init__(self, legacy_config: Dict[str, Any]):
            self.legacy = legacy_config
            self.weights = {
                'training': 0.35,
                'model': 0.40,
                'security': 0.15,
                'data': 0.10
            }
            self.feature_weights = {
                'model_type': 0.25,
                'encoding_dim': 0.20,
                'batch_size': 0.15,
                'learning_rate': 0.15,
                'percentile': 0.10,
                'features': 0.05,
                'hidden_dims': 0.05,
                'activation': 0.05
            }
            self.value_ranges = {
                'batch_size': (8, 256),
                'learning_rate': (1e-5, 1e-1),
                'encoding_dim': (4, 24),
                'percentile': (85, 99),
                'hidden_dims': ([32], [512, 256, 128, 64]),
                'features': (10, 50),
                'normal_samples': (100, 20000),
                'attack_samples': (50, 5000)
            }
        
        def normalize_value(self, key: str, value: Any) -> float:
            """Normalize values based on expected ranges."""
            if key not in self.value_ranges:
                return 0 if not isinstance(value, (int, float)) else value
            
            min_val, max_val = self.value_ranges[key]
            
            if isinstance(min_val, list):  # Handle list-type parameters
                if not isinstance(value, list):
                    return 0
                len_score = len(value) / max(len(min_val), len(max_val))
                val_score = sum(v/max(1, max(max_val)) for v in value[:5]) / min(5, len(value))
                return (len_score + val_score) / 2
            
            if isinstance(value, (int, float)):
                return (min(max(value, min_val), max_val) - min_val) / (max_val - min_val)
            return 0
        
        def compare_numeric(self, key: str, preset_val: Any) -> float:
            """Compare numeric values with normalized difference."""
            if key not in self.legacy:
                return 0
            
            legacy_val = self.legacy[key]
            if not isinstance(legacy_val, (int, float)) or not isinstance(preset_val, (int, float)):
                return 0
                
            norm_legacy = self.normalize_value(key, legacy_val)
            norm_preset = self.normalize_value(key, preset_val)
            return 1 - abs(norm_preset - norm_legacy)
        
        def compare_exact(self, key: str, preset_val: Any) -> float:
            """Compare exact match values with type checking."""
            if key not in self.legacy:
                return 0
            return 1 if self.legacy[key] == preset_val and type(self.legacy[key]) == type(preset_val) else 0
        
        def compare_list(self, key: str, preset_val: Any) -> float:
            """Compare list-type parameters with length and value similarity."""
            if key not in self.legacy:
                return 0
                
            legacy_list = self.legacy[key] if isinstance(self.legacy[key], list) else []
            preset_list = preset_val if isinstance(preset_val, list) else []
            
            if not legacy_list or not preset_list:
                return 0.5 if not legacy_list and not preset_list else 0
            
            # Compare lengths
            length_score = 1 - (abs(len(legacy_list) - len(preset_list)) / max(len(legacy_list), len(preset_list), 1))
            
            # Compare values
            value_scores = []
            for lv, pv in zip(legacy_list, preset_list):
                if isinstance(lv, (int, float)) and isinstance(pv, (int, float)):
                    value_scores.append(1 - (abs(lv - pv) / max(abs(lv), abs(pv), 1e-6)))
                else:
                    value_scores.append(1 if lv == pv else 0)
            
            avg_value_score = sum(value_scores) / len(value_scores) if value_scores else 0
            return (length_score * 0.3) + (avg_value_score * 0.7)
        
        def score_preset(self, preset_name: str, preset_cfg: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate comprehensive similarity score for a preset."""
            section_scores = defaultdict(float)
            
            # Training parameters
            training_params = [
                ('batch_size', self.compare_numeric),
                ('learning_rate', self.compare_numeric),
                ('epochs', self.compare_numeric),
                ('patience', self.compare_numeric),
                ('mixed_precision', self.compare_exact),
                ('optimizer', self.compare_exact),
                ('scheduler', self.compare_exact),
                ('gradient_accumulation_steps', self.compare_numeric)
            ]
            for param, compare_fn in training_params:
                if param in preset_cfg.get('training', {}):
                    section_scores['training'] += compare_fn(param, preset_cfg['training'][param]) * self.feature_weights.get(param, 0.1)
            
            # Model parameters
            model_params = [
                ('model_type', self.compare_exact),
                ('encoding_dim', self.compare_numeric),
                ('hidden_dims', self.compare_list),
                ('dropout_rates', self.compare_list),
                ('activation', self.compare_exact),
                ('normalization', self.compare_exact),
                ('use_batch_norm', self.compare_exact),
                ('skip_connection', self.compare_exact),
                ('residual_blocks', self.compare_exact),
                ('num_models', self.compare_numeric)
            ]
            for param, compare_fn in model_params:
                if param in preset_cfg.get('model', {}):
                    section_scores['model'] += compare_fn(param, preset_cfg['model'][param]) * self.feature_weights.get(param, 0.1)
            
            # Security parameters
            security_params = [
                ('percentile', self.compare_numeric),
                ('attack_threshold', self.compare_numeric),
                ('false_negative_cost', self.compare_numeric),
                ('enable_security_metrics', self.compare_exact),
                ('anomaly_threshold_strategy', self.compare_exact)
            ]
            for param, compare_fn in security_params:
                if param in preset_cfg.get('security', {}):
                    section_scores['security'] += compare_fn(param, preset_cfg['security'][param]) * self.feature_weights.get(param, 0.1)
            
            # Data parameters
            data_params = [
                ('features', self.compare_numeric),
                ('normalization', self.compare_exact),
                ('anomaly_factor', self.compare_numeric),
                ('normal_samples', self.compare_numeric),
                ('attack_samples', self.compare_numeric)
            ]
            for param, compare_fn in data_params:
                if param in preset_cfg.get('data', {}):
                    section_scores['data'] += compare_fn(param, preset_cfg['data'][param]) * self.feature_weights.get(param, 0.1)
            
            # Apply section weights
            for section in section_scores:
                section_scores[section] *= self.weights[section]
            
            total_score = sum(section_scores.values())
            
            return {
                'name': preset_name,
                'total_score': total_score,
                'section_scores': dict(section_scores),
                'config': preset_cfg
            }

    # Score all presets
    scorer = PresetScorer(legacy_config)
    preset_scores = [scorer.score_preset(name, cfg) for name, cfg in PRESET_CONFIGS.items()]
    preset_scores.sort(key=lambda x: x['total_score'], reverse=True)
    
    # --- Step 2: Results Analysis ---
    def analyze_results(scores: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
        """Analyze scoring results and determine close matches."""
        if not scores:
            return [], {}
            
        best_score = scores[0]['total_score']
        close_presets = []
        
        # Dynamic threshold adjustment
        effective_threshold = similarity_threshold
        if best_score < 0.5:
            effective_threshold = min(similarity_threshold * 2, 0.2)
            logger.info(f"Low best score ({best_score:.2f}), adjusting threshold to {effective_threshold:.2f}")
        
        for score in scores:
            if (best_score - score['total_score']) <= effective_threshold:
                close_presets.append(score['name'])
            else:
                break
                
        return close_presets, scores[0]
    
    close_presets, best_preset = analyze_results(preset_scores)
    
    # --- Step 3: Logging and Reporting ---
    def generate_report(scores: List[Dict[str, Any]], close_presets: List[str]) -> None:
        """Generate detailed conversion report."""
        logger.info("\nPreset Matching Report")
        logger.info("=" * 80)
        logger.info(f"{'Preset':<20} {'Total':<8} {'Training':<8} {'Model':<8} {'Security':<8} {'Data':<8}")
        
        for score in scores[:10]:  # Show top 10
            logger.info(
                f"{score['name']:<20} "
                f"{score['total_score']:.2f}  "
                f"{score['section_scores']['training']:.2f}  "
                f"{score['section_scores']['model']:.2f}  "
                f"{score['section_scores']['security']:.2f}  "
                f"{score['section_scores']['data']:.2f}"
            )
        
        logger.info("\nClose Matches:")
        for preset in close_presets:
            logger.info(f"- {preset}")
        
        if close_presets:
            logger.info(f"\nRecommended: {close_presets[0]}")
        else:
            logger.warning("No close matches found, using default configuration")

    generate_report(preset_scores, close_presets)
    
    # --- Step 4: Interactive Selection ---
    def interactive_select(close_presets: List[str], scores: List[Dict[str, Any]]) -> str:
        """Handle interactive preset selection."""
        if not close_presets:
            return "default"
            
        if not sys.stdin.isatty() or len(close_presets) == 1:
            return close_presets[0]
            
        print("\nMultiple close presets detected:")
        for i, name in enumerate(close_presets, 1):
            score = next(s for s in scores if s['name'] == name)
            print(f"  {i}. {name:<18} (score={score['total_score']:.2f})")
        print("  d. Show detailed comparison")
        print("  v. View configuration details")
        
        while True:
            try:
                choice = input(f"\nSelect preset [1-{len(close_presets)}], or (d/v): ").strip().lower()
                
                if choice == 'd':
                    print("\nDetailed Parameter Comparison:")
                    print(f"{'Parameter':<30} {'Legacy':<20} {'Best Match':<20} {'Score':<6}")
                    
                    # Get best preset config
                    preset_cfg = best_preset['config']
                    scorer = PresetScorer(legacy_config)
                    
                    # Compare all parameters
                    all_params = set(legacy_config.keys())
                    for section in ['training', 'model', 'security', 'data']:
                        if section in preset_cfg:
                            for param in preset_cfg[section].keys():
                                all_params.add(f"{section}.{param}")
                    
                    for param in sorted(all_params):
                        if '.' in param:
                            section, p = param.split('.', 1)
                            preset_val = preset_cfg.get(section, {}).get(p, 'N/A')
                        else:
                            preset_val = 'N/A'
                        
                        legacy_val = legacy_config.get(param, 'N/A')
                        
                        if preset_val != 'N/A' and legacy_val != 'N/A':
                            if isinstance(preset_val, (int, float)) and isinstance(legacy_val, (int, float)):
                                score = f"{1 - abs(preset_val - legacy_val)/max(abs(preset_val), abs(legacy_val), 1):.2f}"
                            else:
                                score = "1.0" if preset_val == legacy_val else "0.0"
                        else:
                            score = "N/A"
                        
                        print(f"{param:<30} {str(legacy_val)[:20]:<20} {str(preset_val)[:20]:<20} {score:<6}")
                    continue
                    
                elif choice == 'v':
                    preset = input("Enter preset name to view: ").strip()
                    details = next((s for s in scores if s['name'] == preset), None)
                    if details:
                        print(f"\nConfiguration for {preset}:")
                        print(f"Total similarity score: {details['total_score']:.2f}")
                        for section in ['training', 'model', 'security', 'data', 'monitoring']:
                            if section in details['config']:
                                print(f"\n{section.capitalize()} Parameters:")
                                for k, v in details['config'][section].items():
                                    print(f"  {k}: {v}")
                    else:
                        print("Invalid preset name")
                    continue
                    
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(close_presets):
                    return close_presets[choice_idx]
                print("Invalid selection")
            except ValueError:
                print("Please enter a number, 'd' for comparison, or 'v' for details")
    
    selected_preset = interactive_select(close_presets, preset_scores)
    
    # --- Step 5: Create New Config ---
    new_config = migrate_config(legacy_config, PRESET_CONFIGS[selected_preset])
    
    # Add conversion metadata
    new_config['metadata']['conversion'] = {
        'selected_preset': selected_preset,
        'similar_presets': [p for p in close_presets if p != selected_preset],
        'similarity_threshold': similarity_threshold,
        'best_score': best_preset['total_score'],
        'method': 'preset_matching',
        'timestamp': datetime.now().isoformat()
    }
    
    # Validate model compatibility
    model_type = new_config['model']['model_type']
    if model_type not in MODEL_VARIANTS:
        logger.warning(f"Invalid model type '{model_type}' in converted config, defaulting to SimpleAutoencoder")
        new_config['model']['model_type'] = 'SimpleAutoencoder'
        new_config['metadata']['conversion']['original_model_type'] = model_type
    
    # --- Step 6: Validation ---
    def validate_config(config: Dict[str, Any]) -> bool:
        """Basic configuration validation."""
        required_sections = ['training', 'model', 'data']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate model type
        if config['model']['model_type'] not in MODEL_VARIANTS:
            raise ValueError(f"Invalid model type: {config['model']['model_type']}")
        
        return True
    
    try:
        validate_config(new_config)
        logger.info("Converted configuration passed validation")
    except ValueError as e:
        logger.error(f"Converted config validation failed: {str(e)}")
        # Apply fallback strategy
        logger.info("Attempting fallback to default configuration")
        new_config = migrate_config(legacy_config, DEFAULT_PRESET)
        validate_config(new_config)  # This should always pass for default
    
    return new_config

def update_global_config(config: Dict[str, Any]) -> None:
    """Update module-level constants from config with enhanced validation, logging, and preset support.
    
    Args:
        config: Configuration dictionary to update from
        
    Raises:
        ValueError: If any configuration values are invalid
        TypeError: If any configuration values are of incorrect type
        KeyError: If required configuration sections are missing
    """
    # Validate config structure
    required_sections = ['training', 'model', 'security', 'data']
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required configuration section: {section}")
    
    # Initialize change tracking with timestamps
    changes = {
        'metadata': {
            'config_version': '2.2',
            'update_time': datetime.now().isoformat()
        },
        'training': {},
        'model': {},
        'security': {},
        'data': {}
    }
    
    # Training configuration
    training = config.get("training", {})
    global DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE
    global WEIGHT_DECAY, GRADIENT_CLIP, GRADIENT_ACCUMULATION_STEPS, MIXED_PRECISION, NUM_WORKERS
    
    try:
        # Enhanced training parameter validation with range checking
        training_params = [
            ('batch_size', DEFAULT_BATCH_SIZE, lambda x: isinstance(x, int) and x > 0, "positive integer"),
            ('epochs', DEFAULT_EPOCHS, lambda x: isinstance(x, int) and x > 0, "positive integer"),
            ('learning_rate', LEARNING_RATE, lambda x: isinstance(x, (int, float)) and x > 0, "positive number"),
            ('patience', EARLY_STOPPING_PATIENCE, lambda x: isinstance(x, int) and x >= 0, "non-negative integer"),
            ('weight_decay', WEIGHT_DECAY, lambda x: isinstance(x, (int, float)) and x >= 0, "non-negative number"),
            ('gradient_clip', GRADIENT_CLIP, lambda x: isinstance(x, (int, float)) and x >= 0, "non-negative number"),
            ('gradient_accumulation_steps', GRADIENT_ACCUMULATION_STEPS, 
             lambda x: isinstance(x, int) and x > 0, "positive integer"),
            ('mixed_precision', MIXED_PRECISION, lambda x: isinstance(x, bool), "boolean"),
            ('num_workers', NUM_WORKERS, 
             lambda x: isinstance(x, int) and 0 <= x <= (os.cpu_count() or 1), 
             f"integer between 0 and {os.cpu_count() or 1}")
        ]
        
        for param, current_val, validator, desc in training_params:
            if param in training:
                if not validator(training[param]):
                    raise ValueError(f"{param} must be a {desc}")
                if current_val != training[param]:
                    changes['training'][param] = {
                        'old': current_val,
                        'new': training[param],
                        'time': datetime.now().isoformat()
                    }
                    globals()[param.upper()] = training[param]
    
    except Exception as e:
        logger.error("Failed to update training configuration", exc_info=True)
        raise ValueError(f"Training configuration error: {str(e)}") from e
    
    # Model architecture configuration
    model = config.get("model", {})
    global DEFAULT_ENCODING_DIM, HIDDEN_LAYER_SIZES, DROPOUT_RATES
    global ACTIVATION, ACTIVATION_PARAM, NORMALIZATION
    global USE_BATCH_NORM, USE_LAYER_NORM, DIVERSITY_FACTOR, MIN_FEATURES, NUM_MODELS
    
    try:
        # Model parameter validation
        model_params = [
            ('encoding_dim', DEFAULT_ENCODING_DIM, 
             lambda x: isinstance(x, int) and x >= 1, "positive integer"),
            ('hidden_dims', HIDDEN_LAYER_SIZES, 
             lambda x: isinstance(x, list) and all(isinstance(i, int) and i > 0 for i in x), 
             "list of positive integers"),
            ('dropout_rates', DROPOUT_RATES,
             lambda x: isinstance(x, list) and all(isinstance(i, (int, float)) and 0 <= i < 1 for i in x),
             "list of numbers between 0 and 1"),
            ('activation', ACTIVATION,
             lambda x: x in ['relu', 'leaky_relu', 'gelu'],
             "one of: 'relu', 'leaky_relu', 'gelu'"),
            ('activation_param', ACTIVATION_PARAM,
             lambda x: isinstance(x, (int, float)),
             "number"),
            ('normalization', NORMALIZATION,
             lambda x: x in ['batch', 'layer', None],
             "one of: 'batch', 'layer', None"),
            ('use_batch_norm', USE_BATCH_NORM,
             lambda x: isinstance(x, bool),
             "boolean"),
            ('use_layer_norm', USE_LAYER_NORM,
             lambda x: isinstance(x, bool),
             "boolean"),
            ('diversity_factor', DIVERSITY_FACTOR,
             lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
             "number between 0 and 1"),
            ('min_features', MIN_FEATURES,
             lambda x: isinstance(x, int) and x >= 1,
             "positive integer"),
            ('num_models', NUM_MODELS,
             lambda x: isinstance(x, int) and x >= 1,
             "positive integer")
        ]
        
        for param, current_val, validator, desc in model_params:
            if param in model:
                if not validator(model[param]):
                    raise ValueError(f"model.{param} must be a {desc}")
                if current_val != model[param]:
                    changes['model'][param] = {
                        'old': current_val,
                        'new': model[param],
                        'time': datetime.now().isoformat()
                    }
                    globals()[param.upper()] = model[param]
        
        # Special handling for hidden_dims and dropout_rates length matching
        if 'hidden_dims' in changes['model'] or 'dropout_rates' in changes['model']:
            if len(HIDDEN_LAYER_SIZES) != len(DROPOUT_RATES):
                min_length = min(len(HIDDEN_LAYER_SIZES), len(DROPOUT_RATES))
                HIDDEN_LAYER_SIZES = HIDDEN_LAYER_SIZES[:min_length]
                DROPOUT_RATES = DROPOUT_RATES[:min_length]
                logger.warning(f"Adjusted hidden_dims and dropout_rates to matching length {min_length}")
                changes['model']['hidden_dims_adjusted'] = HIDDEN_LAYER_SIZES
                changes['model']['dropout_rates_adjusted'] = DROPOUT_RATES
    
    except Exception as e:
        logger.error("Failed to update model configuration", exc_info=True)
        raise ValueError(f"Model configuration error: {str(e)}") from e
    
    # Security configuration
    security = config.get("security", {})
    global DEFAULT_PERCENTILE, DEFAULT_ATTACK_THRESHOLD, FALSE_NEGATIVE_COST, SECURITY_METRICS
    
    try:
        security_params = [
            ('percentile', DEFAULT_PERCENTILE,
             lambda x: isinstance(x, (int, float)) and 0 <= x <= 100,
             "number between 0 and 100"),
            ('attack_threshold', DEFAULT_ATTACK_THRESHOLD,
             lambda x: isinstance(x, (int, float)) and x >= 0,
             "non-negative number"),
            ('false_negative_cost', FALSE_NEGATIVE_COST,
             lambda x: isinstance(x, (int, float)) and x >= 0,
             "non-negative number"),
            ('enable_security_metrics', SECURITY_METRICS,
             lambda x: isinstance(x, bool),
             "boolean")
        ]
        
        for param, current_val, validator, desc in security_params:
            if param in security:
                if not validator(security[param]):
                    raise ValueError(f"security.{param} must be a {desc}")
                if current_val != security[param]:
                    changes['security'][param] = {
                        'old': current_val,
                        'new': security[param],
                        'time': datetime.now().isoformat()
                    }
                    globals()[param.upper()] = security[param]
    
    except Exception as e:
        logger.error("Failed to update security configuration", exc_info=True)
        raise ValueError(f"Security configuration error: {str(e)}") from e
    
    # Data configuration
    data = config.get("data", {})
    global NORMAL_SAMPLES, ATTACK_SAMPLES, FEATURES, ANOMALY_FACTOR, RANDOM_STATE
    
    try:
        data_params = [
            ('normal_samples', NORMAL_SAMPLES,
             lambda x: isinstance(x, int) and x > 0,
             "positive integer"),
            ('attack_samples', ATTACK_SAMPLES,
             lambda x: isinstance(x, int) and x >= 0,
             "non-negative integer"),
            ('features', FEATURES,
             lambda x: isinstance(x, int) and x >= MIN_FEATURES,
             f"integer >= {MIN_FEATURES}"),
            ('anomaly_factor', ANOMALY_FACTOR,
             lambda x: isinstance(x, (int, float)) and x > 0,
             "positive number"),
            ('random_state', RANDOM_STATE,
             lambda x: isinstance(x, int),
             "integer")
        ]
        
        for param, current_val, validator, desc in data_params:
            if param in data:
                if not validator(data[param]):
                    raise ValueError(f"data.{param} must be a {desc}")
                if current_val != data[param]:
                    changes['data'][param] = {
                        'old': current_val,
                        'new': data[param],
                        'time': datetime.now().isoformat()
                    }
                    globals()[param.upper()] = data[param]
    
    except Exception as e:
        logger.error("Failed to update data configuration", exc_info=True)
        raise ValueError(f"Data configuration error: {str(e)}") from e
    
    # Handle preset application with validation
    presets = config.get("presets", {})
    if "current_preset" in presets and presets["current_preset"]:
        preset_name = presets["current_preset"]
        if preset_name in PRESET_CONFIGS:
            logger.info(f"Applying preset configuration: {preset_name}")
            try:
                preset_config = PRESET_CONFIGS[preset_name]
                
                # Validate model-preset compatibility
                model_type = config.get('model', {}).get('model_type')
                if model_type and not validate_model_preset_compatibility(model_type, preset_config):
                    raise ValueError(f"Model type {model_type} is not compatible with preset {preset_name}")
                
                # Apply preset with change tracking
                preset_changes = {}
                for section in ['training', 'model', 'security', 'data']:
                    if section in preset_config:
                        for key, value in preset_config[section].items():
                            current_val = globals().get(key.upper(), None)
                            if current_val is not None and current_val != value:
                                preset_changes.setdefault(section, {})[key] = {
                                    'old': current_val,
                                    'new': value,
                                    'time': datetime.now().isoformat(),
                                    'source': f'preset:{preset_name}'
                                }
                                globals()[key.upper()] = value
                
                if preset_changes:
                    changes['preset'] = {
                        'name': preset_name,
                        'changes': preset_changes,
                        'time': datetime.now().isoformat()
                    }
            
            except Exception as e:
                logger.error(f"Failed to apply preset {preset_name}", exc_info=True)
                raise ValueError(f"Preset configuration error: {str(e)}") from e
    
    # Log configuration changes in detail
    if any(v for k, v in changes.items() if k not in ['metadata']):
        change_count = sum(len(section) for section in changes.values() 
                          if isinstance(section, dict) and section)
        logger.info(f"Applied {change_count} configuration changes:")
        
        for section, section_changes in changes.items():
            if section not in ['metadata'] and section_changes:
                logger.info(f"  [{section.upper()}]")
                for param, change in section_changes.items():
                    if isinstance(change, dict):
                        source = change.get('source', 'manual')
                        logger.info(f"    {param}: {change['old']} -> {change['new']} ({source})")
        
        # Save change log
        save_change_log(changes)
    else:
        logger.debug("No configuration changes detected")
    
    # Reinitialize model variants if architecture changed
    if 'model' in changes:
        initialize_model_variants()

def get_default_config() -> Dict[str, Any]:
    """Get default system configuration."""
    return {
        "training": {
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": EARLY_STOPPING_PATIENCE,
            "gradient_clip": GRADIENT_CLIP,
            "mixed_precision": MIXED_PRECISION,
            "num_workers": NUM_WORKERS
        },
        "model": {
            "model_type": "EnhancedAutoencoder",
            "encoding_dim": DEFAULT_ENCODING_DIM,
            "hidden_dims": HIDDEN_LAYER_SIZES,
            "dropout_rates": DROPOUT_RATES,
            "activation": ACTIVATION,
            "normalization": NORMALIZATION
        },
        "data": {
            "features": FEATURES,
            "normal_samples": NORMAL_SAMPLES,
            "attack_samples": ATTACK_SAMPLES,
            "use_real_data": False,
            "validation_split": 0.2
        },
        "security": {
            "percentile": DEFAULT_PERCENTILE,
            "anomaly_threshold_strategy": "percentile"
        },
        "system": {
            "model_dir": str(DEFAULT_MODEL_DIR),
            "log_dir": str(LOG_DIR),
            "config_dir": str(CONFIG_DIR),
            "debug": False
        }
    }

def validate_config(config: Dict[str, Any]) -> None:
    """Enhanced configuration validation with automatic fixes."""
    required_sections = ['training', 'model', 'security', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate training parameters
    training = config['training']
    if not isinstance(training.get('batch_size', 1), int) or training['batch_size'] < 1:
        raise ValueError("batch_size must be a positive integer")
    
    # Validate and auto-fix model architecture
    model = config['model']
    if model.get('model_type') not in MODEL_VARIANTS:
        raise ValueError(f"Invalid model type: {model.get('model_type')}")
    
    hidden_dims = model.get('hidden_dims', [])
    dropout_rates = model.get('dropout_rates', [])
    
    # Ensure hidden_dims and dropout_rates are lists
    if not isinstance(hidden_dims, list):
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else [64]
        model['hidden_dims'] = hidden_dims
        logger.warning(f"Converted hidden_dims to list: {hidden_dims}")
    
    if not isinstance(dropout_rates, list):
        dropout_rates = [dropout_rates] if isinstance(dropout_rates, (int, float)) else [0.2]
        model['dropout_rates'] = dropout_rates
        logger.warning(f"Converted dropout_rates to list: {dropout_rates}")
    
    # Ensure matching lengths
    if len(hidden_dims) != len(dropout_rates):
        min_length = min(len(hidden_dims), len(dropout_rates))
        hidden_dims = hidden_dims[:min_length]
        dropout_rates = dropout_rates[:min_length]
        model['hidden_dims'] = hidden_dims
        model['dropout_rates'] = dropout_rates
        logger.warning(f"Adjusted hidden_dims and dropout_rates to matching length {min_length}")
    
    # Validate security parameters
    security = config['security']
    if not 0 <= security.get('percentile', 95) <= 100:
        raise ValueError("percentile must be between 0 and 100")
    
    # Validate data parameters
    data = config['data']
    if data.get('features', 10) < model.get('min_features', 1):
        raise ValueError(f"features must be >= {model.get('min_features', 1)}")

def save_initialization_report(system_status, report_dir):
    """Save initialization report to JSON file."""
    try:
        init_report_path = report_dir / f"system_init_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(init_report_path, 'w') as f:
            json.dump(system_status, f, indent=2, default=str)
        logger.info(f"Initialization report saved: {init_report_path}")
    except Exception as e:
        logger.warning(f"Failed to save initialization report: {e}")

# System initialization validation and setup
def initialize_system() -> Dict[str, Any]:
    """
    Initialize the complete system with comprehensive setup and validation.
    
    This function performs a complete system initialization including:
    - Hardware detection and validation
    - Configuration system setup
    - Global error handling setup
    - Performance monitoring initialization
    - Memory management setup
    - System information gathering
    - Startup checks and validation
    - Logging configuration
    
    Returns:
        Dict containing comprehensive system status and configuration
    
    Raises:
        RuntimeError: If critical system components fail to initialize
        ValueError: If configuration validation fails
    """
    
    initialization_start = time.time()
    system_status = {
        'initialization': {
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress',
            'errors': [],
            'warnings': []
        },
        'system': {},
        'config': {},
        'hardware': {},
        'dependencies': {},
        'performance': {},
        'checks': {}
    }
    
    try:
        # Basic configuration
        configure_system()
        set_seed(42)
        
        # Early logging setup
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(log_dir)
        
        # Step 1: Setup global exception handling
        logger.info("Setting up global exception handling...")
        try:
            sys.excepthook = enhanced_global_exception_handler
            logger.info("[INFO] Global exception handler configured")
        except Exception as e:
            system_status['initialization']['errors'].append(f"Exception handler setup failed: {e}")
            logger.error(f"Failed to setup global exception handler: {e}")
        
        # Step 2: Performance monitoring setup
        logger.info("Initializing performance monitoring...")
        try:
            global monitor_performance
            monitor_performance = enhanced_monitor_performance()
            logger.info("[INFO] Performance monitoring initialized")
        except Exception as e:
            system_status['initialization']['errors'].append(f"Performance monitoring setup failed: {e}")
            logger.error(f"Failed to setup performance monitoring: {e}")
        
        # Step 3: Memory management initialization
        logger.info("Setting up memory management...")
        try:
            global clear_memory, get_memory_usage
            clear_memory = enhanced_clear_memory
            get_memory_usage = get_detailed_memory_usage
            clear_memory()
            logger.info("[INFO] Memory management configured")
        except Exception as e:
            system_status['initialization']['errors'].append(f"Memory management setup failed: {e}")
            logger.error(f"Failed to setup memory management: {e}")
        
        # Step 4: Comprehensive startup checks
        logger.info("Performing comprehensive startup checks...")
        startup_checks_result = {}
        
        try:
            checks_to_perform = [
                ("Python Version", lambda: sys.version_info >= (3, 7)),
                ("PyTorch Installation", lambda: hasattr(torch, '__version__')),
                ("CUDA Availability", lambda: torch.cuda.is_available()),
                ("Required Directories", lambda: all(
                    (dir_path.mkdir(parents=True, exist_ok=True) or True) and dir_path.exists()
                    for dir_path in [DEFAULT_MODEL_DIR, LOG_DIR, CONFIG_DIR, TB_DIR, DATA_DIR, CACHE_DIR]
                )),
                ("Disk Space (>1GB)", lambda: shutil.disk_usage('.').free / (1024**3) >= 1.0),
                ("Available Memory (>2GB)", lambda: psutil.virtual_memory().available / (1024**3) >= 2.0),
                ("Write Permissions", lambda: os.access('.', os.W_OK)),
                ("Optional Dependencies", lambda: sum(OPTIONAL_DEPENDENCIES.values()) >= len(OPTIONAL_DEPENDENCIES) // 2)
            ]
            
            all_checks_passed = True
            for check_name, check_func in checks_to_perform:
                try:
                    result = check_func()
                    startup_checks_result[check_name] = {
                        'status': 'PASSED' if result else 'FAILED',
                        'result': result
                    }
                    
                    if not result:
                        all_checks_passed = False
                        logger.warning(f"Startup check failed: {check_name}")
                    else:
                        logger.debug(f"Startup check passed: {check_name}")
                        
                except Exception as e:
                    startup_checks_result[check_name] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    all_checks_passed = False
                    logger.error(f"Startup check error for {check_name}: {e}")
            
            system_status['checks'] = startup_checks_result
            
            if all_checks_passed:
                logger.info("[INFO] All startup checks passed")
            else:
                logger.warning("[WARNING] Some startup checks failed - system may not function optimally")
                system_status['initialization']['warnings'].append("Some startup checks failed")
            
        except Exception as e:
            system_status['initialization']['errors'].append(f"Startup checks failed: {e}")
            logger.error(f"Failed to perform startup checks: {e}")
        
        # Step 5: Hardware detection and validation
        logger.info("Detecting and validating hardware...")
        try:
            hw_info = check_hardware()
            enhanced_hw_info = enhance_hardware_info(hw_info)
            system_status['hardware'] = enhanced_hw_info
            log_hardware_config(enhanced_hw_info)
        except Exception as e:
            system_status['initialization']['errors'].append(f"Hardware detection failed: {e}")
            logger.error(f"Failed to detect hardware: {e}")
        
        # Step 6: Configuration system initialization
        logger.info("Initializing configuration system...")
        try:
            config = initialize_config()
            
            try:
                validate_config(config)
                logger.info("[INFO] Configuration validation passed")
            except ValueError as e:
                logger.error(f"Configuration validation failed: {e}")
                system_status['initialization']['warnings'].append(f"Configuration validation failed: {e}")
                
                if sys.stdin.isatty():
                    if prompt_user("Use default configuration?", default=True):
                        config = get_default_config()
                        save_config(config)
                        logger.info("[INFO] Default configuration applied")
                    else:
                        raise ValueError("Configuration validation failed and user declined defaults")
                else:
                    config = get_default_config()
                    save_config(config)
                    logger.warning("[WARNING] Using default configuration due to validation failure")
            
            update_global_config(config)
            
            system_status['config'] = {
                'active_config': config,
                'config_file': str(CONFIG_FILE),
                'available_presets': list(PRESET_CONFIGS.keys()),
                'validation_status': 'passed'
            }
            
        except Exception as e:
            system_status['initialization']['errors'].append(f"Configuration initialization failed: {e}")
            logger.error(f"Failed to initialize configuration: {e}")
            config = get_default_config()
            update_global_config(config)
        
        # Step 7: Model variants initialization
        logger.info("Initializing model variants...")
        try:
            initialize_model_variants()
            
            if not MODEL_VARIANTS:
                raise RuntimeError("No model variants could be initialized")
            
            variant_status = validate_model_variants(logger)
            system_status['config']['model_variants'] = variant_status
            available_variants = [name for name, status in variant_status.items() if status == 'available']
            
            logger.info(f"[INFO] Model variants initialized: {', '.join(available_variants)}")
            
            if not available_variants:
                raise RuntimeError("No working model variants available")
            
        except Exception as e:
            system_status['initialization']['errors'].append(f"Model initialization failed: {e}")
            logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        
        # Step 8: Dependencies validation
        logger.info("Validating dependencies...")
        try:
            dependency_status = check_core_dependencies()
            system_status['dependencies'] = dependency_status
            
            total_deps = len(dependency_status)
            available_deps = sum(1 for dep in dependency_status.values() if dep['available'])
            logger.info(f"[INFO] Dependencies: {available_deps}/{total_deps} available")
            
        except Exception as e:
            system_status['initialization']['errors'].append(f"Dependency validation failed: {e}")
            logger.error(f"Failed to validate dependencies: {e}")
        
        # Step 9: Performance baseline establishment
        logger.info("Establishing performance baseline...")
        try:
            performance_metrics = establish_performance_baseline()
            system_status['performance'] = performance_metrics
            
            logger.info("[INFO] Performance baseline established")
            for metric, value in performance_metrics.items():
                logger.debug(f"  {metric}: {value:.4f}")
            
        except Exception as e:
            system_status['initialization']['warnings'].append(f"Performance baseline failed: {e}")
            logger.warning(f"Failed to establish performance baseline: {e}")
        
        # Step 10: System validation and finalization
        logger.info("Finalizing system initialization...")
        
        initialization_time = time.time() - initialization_start
        has_critical_errors = any('failed' in error.lower() for error in system_status['initialization']['errors'])
        overall_status = 'failed' if has_critical_errors else 'success'
        
        system_status['initialization'].update({
            'end_time': datetime.now().isoformat(),
            'duration_seconds': initialization_time,
            'status': overall_status
        })
        
        system_status['system'] = {
            'platform': platform.platform(),
            'python_version': sys.version.split()[0],
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'working_directory': str(Path.cwd()),
            'log_directory': str(LOG_DIR),
            'model_directory': str(DEFAULT_MODEL_DIR),
            'config_directory': str(CONFIG_DIR)
        }
        
        save_initialization_report(system_status, LOG_DIR)
        
        logger.info("=" * 80)
        if overall_status == 'success':
            logger.info("SYSTEM INITIALIZATION COMPLETED SUCCESSFULLY")
            logger.info(f"Initialization time: {initialization_time:.2f} seconds")
            logger.info(f"Active configuration: {config.get('_preset_name', 'custom')}")
            logger.info(f"Available model variants: {len(MODEL_VARIANTS)}")
            if system_status['initialization']['warnings']:
                logger.info(f"Warnings: {len(system_status['initialization']['warnings'])}")
        else:
            logger.error("SYSTEM INITIALIZATION COMPLETED WITH ERRORS")
            logger.error(f"Errors encountered: {len(system_status['initialization']['errors'])}")
            for error in system_status['initialization']['errors']:
                logger.error(f"  - {error}")
        
        logger.info("=" * 80)
        
        return system_status, config, logger
        
    except Exception as e:
        system_status['initialization'].update({
            'status': 'critical_failure',
            'end_time': datetime.now().isoformat(),
            'duration_seconds': time.time() - initialization_start,
            'critical_error': str(e)
        })
        
        logger.critical(f"CRITICAL: System initialization failed: {e}")
        logger.exception("Detailed error information:")
        
        try:
            error_report_path = LOG_DIR / f"init_failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_report_path, 'w') as f:
                json.dump(system_status, f, indent=2, default=str)
        except:
            pass
        
        raise RuntimeError(f"System initialization failed: {e}") from e

class SimpleAutoencoder(nn.Module):
    """Simple autoencoder with enhanced initialization, mixed precision support, and CPU/GPU awareness.
    
    Args:
        input_dim: Dimension of input features
        encoding_dim: Size of latent representation (default: DEFAULT_ENCODING_DIM)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU) (default: MIXED_PRECISION)
        min_features: Minimum allowed input dimension (default: MIN_FEATURES)
        config: Optional configuration dictionary to override defaults
    """
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = None,
        mixed_precision: bool = None,
        min_features: int = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Load configuration with fallbacks
        if config is None:
            try:
                config = get_current_config().get('model', {})
            except Exception:
                config = {}
        
        # Apply configuration with parameter precedence
        self.encoding_dim = encoding_dim if encoding_dim is not None else config.get('encoding_dim', DEFAULT_ENCODING_DIM)
        self.min_features = min_features if min_features is not None else config.get('min_features', MIN_FEATURES)
        
        # Input validation
        if input_dim < self.min_features:
            raise ValueError(f"Input dimension must be at least {self.min_features}")
        
        # Mixed precision handling with configuration awareness
        mixed_precision_config = config.get('mixed_precision', MIXED_PRECISION)
        self._mixed_precision_requested = mixed_precision if mixed_precision is not None else mixed_precision_config
        self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        
        # Store input dimension for config export
        self.input_dim = input_dim
        
        # Architecture definition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Logging configuration
        logger.debug(f"SimpleAutoencoder initialized with: "
                    f"input_dim={input_dim}, encoding_dim={self.encoding_dim}, "
                    f"mixed_precision={self.mixed_precision} (requested={self._mixed_precision_requested})")
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic mixed precision support.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of same shape as input
        """
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the autoencoder."""
        return {
            "model_type": "SimpleAutoencoder",
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "mixed_precision": self.mixed_precision,
            "min_features": self.min_features,
            "architecture": "simple",
            "initialized_with_cuda": torch.cuda.is_available(),
            "config_version": "2.0"
        }
    
    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update model settings from configuration (limited to non-architectural changes)."""
        model_config = config.get('model', {})
        
        # Only update non-architectural parameters
        if 'mixed_precision' in model_config:
            self._mixed_precision_requested = model_config['mixed_precision']
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
            logger.info(f"Updated mixed_precision to {self.mixed_precision}")

class EnhancedAutoencoder(nn.Module):
    """Enhanced autoencoder with configurable architecture, mixed precision support, and advanced features.
    
    Args:
        input_dim: Dimension of input features
        encoding_dim: Size of latent representation (default: DEFAULT_ENCODING_DIM)
        hidden_dims: List of hidden layer dimensions (default: HIDDEN_LAYER_SIZES)
        dropout_rates: Dropout rates for each layer (default: DROPOUT_RATES)
        activation: Activation function ('relu', 'leaky_relu', 'gelu') (default: ACTIVATION)
        activation_param: Parameter for activation (e.g., slope for LeakyReLU) (default: ACTIVATION_PARAM)
        normalization: Normalization type ('batch', 'layer', None) (default: NORMALIZATION)
        legacy_mode: Use simple architecture if True (default: False)
        skip_connection: Enable skip connection if True (default: True)
        min_features: Minimum allowed input dimension (default: MIN_FEATURES)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU) (default: MIXED_PRECISION)
        config: Optional configuration dictionary to override defaults
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = None,
        hidden_dims: List[int] = None,
        dropout_rates: Optional[List[float]] = None,
        activation: str = None,
        activation_param: float = None,
        normalization: str = None,
        legacy_mode: bool = False,
        skip_connection: bool = None,
        min_features: int = None,
        mixed_precision: bool = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Load configuration with fallbacks
        if config is None:
            try:
                config = get_current_config().get('model', {})
            except Exception:
                config = {}
        
        # Apply configuration with parameter precedence
        self.encoding_dim = encoding_dim if encoding_dim is not None else config.get('encoding_dim', DEFAULT_ENCODING_DIM)
        self.hidden_dims = hidden_dims if hidden_dims is not None else config.get('hidden_dims', HIDDEN_LAYER_SIZES.copy())
        self.dropout_rates = dropout_rates if dropout_rates is not None else config.get('dropout_rates', DROPOUT_RATES.copy())
        self.activation = activation if activation is not None else config.get('activation', ACTIVATION)
        self.activation_param = activation_param if activation_param is not None else config.get('activation_param', ACTIVATION_PARAM)
        self.normalization = normalization if normalization is not None else config.get('normalization', NORMALIZATION)
        self.skip_connection = skip_connection if skip_connection is not None else config.get('skip_connection', True)
        self.min_features = min_features if min_features is not None else config.get('min_features', MIN_FEATURES)
        self.legacy_mode = legacy_mode
        self.input_dim = input_dim
        
        # Input validation
        if input_dim < self.min_features:
            raise ValueError(f"Input dimension must be at least {self.min_features}")
        
        # Validate configuration compatibility
        self._validate_config()
        
        # Fix length mismatch between hidden_dims and dropout_rates
        if len(self.hidden_dims) != len(self.dropout_rates):
            logger.warning(f"Length mismatch: hidden_dims({len(self.hidden_dims)}) vs dropout_rates({len(self.dropout_rates)})")
            
            if len(self.dropout_rates) < len(self.hidden_dims):
                # Extend dropout_rates
                last_dropout = self.dropout_rates[-1] if self.dropout_rates else 0.2
                while len(self.dropout_rates) < len(self.hidden_dims):
                    self.dropout_rates.append(max(0.1, last_dropout * 0.8))
                logger.info(f"Extended dropout_rates to: {self.dropout_rates}")
            else:
                # Truncate dropout_rates
                self.dropout_rates = self.dropout_rates[:len(self.hidden_dims)]
                logger.info(f"Truncated dropout_rates to: {self.dropout_rates}")
        
        # Mixed precision handling with configuration awareness
        mixed_precision_config = config.get('mixed_precision', MIXED_PRECISION)
        self._mixed_precision_requested = mixed_precision if mixed_precision is not None else mixed_precision_config
        self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        
        # Logging configuration
        logger.debug(f"EnhancedAutoencoder initialized with: input_dim={input_dim}, "
                    f"encoding_dim={self.encoding_dim}, hidden_dims={self.hidden_dims}, "
                    f"dropout_rates={self.dropout_rates}, mixed_precision={self.mixed_precision}")

        if legacy_mode:
            # Simple architecture for backward compatibility
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, self.encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.encoding_dim, input_dim),
                nn.Sigmoid()
            )
            self.skip = None
        else:
            # Build encoder and decoder networks with proper dropout handling
            encoder_dims = self.hidden_dims + [self.encoding_dim]
            # Add dropout for encoding layer
            encoder_dropouts = self.dropout_rates + [0.1]
            
            decoder_dims = self.hidden_dims[::-1] + [input_dim]
            # Add dropout for final layer
            decoder_dropouts = self.dropout_rates[::-1] + [0.1]
            
            self.encoder = self._build_network(
                input_dim=input_dim,
                layer_dims=encoder_dims,
                dropout_rates=encoder_dropouts,
                activation=self.activation,
                activation_param=self.activation_param,
                normalization=self.normalization,
                final_activation="tanh"
            )
            
            self.decoder = self._build_network(
                input_dim=self.encoding_dim,
                layer_dims=decoder_dims,
                dropout_rates=decoder_dropouts,
                activation=self.activation,
                activation_param=self.activation_param,
                normalization=self.normalization,
                final_activation="sigmoid"
            )
            
            # Skip connection (only when dimensions match)
            self.skip = (
                nn.Linear(input_dim, input_dim)
                # More flexible condition
                if self.skip_connection and input_dim <= self.encoding_dim * 2
                else None
            )
        
        self._initialize_weights()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.activation not in ['relu', 'leaky_relu', 'gelu']:
            logger.warning(f"Unknown activation '{self.activation}', defaulting to 'relu'")
            self.activation = 'relu'
        
        if self.normalization not in ['batch', 'layer', None]:
            logger.warning(f"Unknown normalization '{self.normalization}', defaulting to None")
            self.normalization = None
        
        if not isinstance(self.hidden_dims, list) or not self.hidden_dims:
            logger.warning(f"Invalid hidden_dims '{self.hidden_dims}', using default")
            self.hidden_dims = [128, 64]
        
        if not isinstance(self.dropout_rates, list) or not self.dropout_rates:
            logger.warning(f"Invalid dropout_rates '{self.dropout_rates}', using default")
            self.dropout_rates = [0.2, 0.15]

    def _build_network(
        self,
        input_dim: int,
        layer_dims: List[int],
        dropout_rates: List[float],
        activation: str,
        activation_param: float,
        normalization: str,
        final_activation: Optional[str] = None
    ) -> nn.Sequential:
        """Construct encoder/decoder networks with robust dropout handling."""
        layers = []
        prev_dim = input_dim
        
        # Ensure dropout_rates matches layer_dims length
        if len(dropout_rates) != len(layer_dims):
            if len(dropout_rates) < len(layer_dims):
                # Extend with last value
                last_dropout = dropout_rates[-1] if dropout_rates else 0.2
                dropout_rates = dropout_rates + [last_dropout] * (len(layer_dims) - len(dropout_rates))
            else:
                # Truncate
                dropout_rates = dropout_rates[:len(layer_dims)]
        
        for i, (h_dim, dropout) in enumerate(zip(layer_dims, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, h_dim))
            
            # Normalization (skip for final layer to avoid conflicts)
            if i < len(layer_dims) - 1:
                if normalization == "batch" and h_dim > 1:
                    layers.append(nn.BatchNorm1d(h_dim))
                elif normalization == "layer":
                    layers.append(nn.LayerNorm(h_dim))
            
            # Activation (skip for final layer if final_activation is specified)
            if i < len(layer_dims) - 1 or final_activation is None:
                if activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(negative_slope=activation_param))
                elif activation == "gelu":
                    layers.append(nn.GELU())
                else:  # Default to ReLU
                    layers.append(nn.ReLU())
            
            # Dropout (skip for final layer to avoid output corruption)
            if dropout > 0 and i < len(layer_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = h_dim
        
        # Final activation if specified
        if final_activation == "tanh":
            layers.append(nn.Tanh())
        elif final_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize weights using appropriate methods based on activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation == "leaky_relu":
                    nn.init.kaiming_normal_(
                        m.weight, 
                        mode='fan_in', 
                        nonlinearity='leaky_relu',
                        a=self.activation_param
                    )
                else:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision and optional skip connection."""
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            if self.skip is not None and not self.legacy_mode:
                decoded = decoded + self.skip(x)
            return decoded

    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the autoencoder."""
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
            "legacy_mode": self.legacy_mode,
            "min_features": self.min_features,
            "architecture": "enhanced",
            "config_version": "2.0"
        }
    
    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update model settings from configuration (limited to non-architectural changes)."""
        model_config = config.get('model', {})
        
        # Only update non-architectural parameters
        if 'mixed_precision' in model_config:
            self._mixed_precision_requested = model_config['mixed_precision']
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
            logger.info(f"Updated mixed_precision to {self.mixed_precision}")

class AutoencoderEnsemble(nn.Module):
    """Ensemble of autoencoders with configurable diversity and mixed precision support.
    
    Args:
        input_dim: Dimension of input features
        num_models: Number of autoencoders in ensemble (default: NUM_MODELS)
        encoding_dim: Base size of latent representation (default: DEFAULT_ENCODING_DIM)
        diversity_factor: Scale factor for varying architectures (default: DIVERSITY_FACTOR)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU) (default: MIXED_PRECISION)
        min_features: Minimum allowed input dimension (default: MIN_FEATURES)
        config: Optional configuration dictionary to override defaults
    """
    def __init__(
        self,
        input_dim: int,
        num_models: int = None,
        encoding_dim: int = None,
        diversity_factor: float = None,
        mixed_precision: bool = None,
        min_features: int = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Load configuration with fallbacks
        if config is None:
            try:
                config = get_current_config().get('model', {})
            except Exception:
                config = {}
        
        # Apply configuration with parameter precedence
        self.num_models = num_models if num_models is not None else config.get('num_models', NUM_MODELS)
        self.encoding_dim = encoding_dim if encoding_dim is not None else config.get('encoding_dim', DEFAULT_ENCODING_DIM)
        self.diversity_factor = diversity_factor if diversity_factor is not None else config.get('diversity_factor', DIVERSITY_FACTOR)
        self.min_features = min_features if min_features is not None else config.get('min_features', MIN_FEATURES)
        self.input_dim = input_dim
        
        # Input validation
        if input_dim < self.min_features:
            raise ValueError(f"Input dimension must be at least {self.min_features}")
        if self.num_models < 1:
            raise ValueError("Number of models must be at least 1")
        if not 0 <= self.diversity_factor <= 1:
            raise ValueError("Diversity factor must be between 0 and 1")

        # Mixed precision handling with configuration awareness
        mixed_precision_config = config.get('mixed_precision', MIXED_PRECISION)
        self._mixed_precision_requested = mixed_precision if mixed_precision is not None else mixed_precision_config
        self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
        
        # Get base configuration for ensemble members
        base_activation = config.get('activation', ACTIVATION)
        base_activation_param = config.get('activation_param', ACTIVATION_PARAM)
        base_normalization = config.get('normalization', NORMALIZATION)
        
        # Initialize ensemble models with architectural diversity
        self.models = nn.ModuleList([
            EnhancedAutoencoder(
                input_dim=input_dim,
                encoding_dim=max(4, int(self.encoding_dim * (1 + (i - self.num_models//2) * self.diversity_factor))),
                hidden_dims=[
                    max(32, int(128 * (1 + (i - self.num_models//2) * self.diversity_factor * 0.5))),
                    max(16, int(64 * (1 + (i - self.num_models//2) * self.diversity_factor * 0.5)))
                ],
                dropout_rates=[0.2 + i*0.05, 0.15 + i*0.05],
                skip_connection=(i % 2 == 0),
                mixed_precision=self.mixed_precision,
                normalization=base_normalization if i % 2 == 0 else None,
                activation=base_activation,
                activation_param=base_activation_param if i % 2 == 0 else 0.1,
                legacy_mode=(i == 0 and self.num_models == 1),
                min_features=self.min_features,
                config=config  # Pass full config to sub-models
            )
            for i in range(self.num_models)
        ])
        
        # Log initialization details
        logger.debug(f"AutoencoderEnsemble initialized with: "
                    f"num_models={self.num_models}, encoding_dim={self.encoding_dim}, "
                    f"mixed_precision={self.mixed_precision} (requested={self._mixed_precision_requested}), "
                    f"diversity_factor={self.diversity_factor}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision support.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Averaged reconstruction from all ensemble members
        """
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            reconstructions = [model(x) for model in self.models]
            return torch.stack(reconstructions).mean(dim=0)

    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the ensemble."""
        return {
            "model_type": "AutoencoderEnsemble",
            "input_dim": self.input_dim,
            "num_models": self.num_models,
            "encoding_dim": self.encoding_dim,
            "diversity_factor": self.diversity_factor,
            "mixed_precision": self.mixed_precision,
            "min_features": self.min_features,
            "architecture": "ensemble",
            "model_types": [type(m).__name__ for m in self.models],
            "ensemble_configs": [m.get_config() for m in self.models],
            "config_version": "2.0"
        }
    
    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update model settings from configuration (limited to non-architectural changes)."""
        model_config = config.get('model', {})
        
        # Update mixed precision for ensemble and all sub-models
        if 'mixed_precision' in model_config:
            self._mixed_precision_requested = model_config['mixed_precision']
            self.mixed_precision = self._mixed_precision_requested and torch.cuda.is_available()
            
            # Update all sub-models
            for model in self.models:
                model.update_from_config(config)
            
            logger.info(f"Updated ensemble mixed_precision to {self.mixed_precision}")

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
            initialize_model_variants()
        
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
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                        use_external_data_format=input_dim > 2048  # For large models
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
                logger.success(f"Export completed: {onnx_path}")
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
    console = Console()
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
    print(Fore.YELLOW + "\nMain Menu:")
    print(Fore.CYAN + "1. Model Training")
    print(Fore.CYAN + "2. Hyperparameter Optimization")
    print(Fore.CYAN + "3. Model Architecture Comparison")
    print(Fore.CYAN + "4. Configuration Management")
    print(Fore.CYAN + "5. System Information")
    print(Fore.CYAN + "6. Performance Benchmark")
    print(Fore.CYAN + "7. Model Analysis & Visualization")
    print(Fore.CYAN + "8. Advanced Tools")
    print(Fore.RED + "9. Exit")

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
        
        choice = input("\nSelect option (1-9): ").strip()
        
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
                if input("\nAre you sure you want to exit? (y/N): ").lower().strip() == 'y':
                    print("Goodbye!")
                    break
            
        except KeyboardInterrupt:
            print("\nOperation interrupted")
        except Exception as e:
            print(f"\nError: {str(e)}")
        
        if choice != "9":
            input("\nPress Enter to continue...")

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
            print(f"[+] Device: {hw['device']}")
            print(f"[+] PyTorch: {torch.__version__}")
            print(f"[+] CUDA available: {torch.cuda.is_available()}")
        except:
            print("[!] Unable to get hardware information")

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
    # Initialize system first to set up logging and configuration
    try:
        system_status, config, logger = initialize_system()
        validate_config(config)
    except Exception as e:
        logger.warning(f"Configuration initialization failed using defaults: {e}")
        config = get_current_config()
        logger = logging.getLogger(__name__)
    
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