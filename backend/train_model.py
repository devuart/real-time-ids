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
from enum import Enum, auto
import re
import traceback
import tarfile
import zipfile
import shutil
import contextlib
import psutil

# Third-party imports
from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, BarColumn, track
from rich.prompt import Prompt
from copy import deepcopy
import numpy as np
import pandas as pd
import pkg_resources
from pynput.keyboard import Key, Listener
from mpl_toolkits.mplot3d import Axes3D

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
import torch.nn.functional as F

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from scipy.stats import entropy, ttest_ind, ttest_ind_from_stats, ks_2samp
from functools import partial
from multiprocessing import Pool, cpu_count
from statsmodels.stats.multitest import multipletests
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
#from umap import UMAP
import umap

# Imbalanced learning imports
import imblearn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import (
    CondensedNearestNeighbour,
    NearMiss,
    RandomUnderSampler,
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks

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
    # Forward declaration
    pass

class UnicodeStreamHandler:
    # Forward declaration
    pass

class SimpleIDSModel:
    # Forward declaration
    pass

class StabilizedIDSModel:
    # Forward declaration
    pass

class EnsembleIDSModel:
    # Forward declaration
    pass

class WarmupScheduler:
    # Foward declaration
    pass

class SecurityAwareLoss:
    # Foward declaration
    pass

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

# Loading Screen and System Check Framework
class CheckLevel(Enum):
    # Must pass for program to run
    CRITICAL = auto()
    
    # Should pass for full functionality
    IMPORTANT = auto()
    
    # Nice-to-have information
    INFORMATIONAL = auto()

class CheckResult:
    def __init__(self, passed: bool, message: str, level: CheckLevel = CheckLevel.IMPORTANT):
        self.passed = passed
        self.message = message
        self.level = level
        self.details: Optional[str] = None
        self.exception: Optional[Exception] = None
    
    def with_details(self, details: str) -> 'CheckResult':
        self.details = details
        return self
    
    def with_exception(self, exc: Exception) -> 'CheckResult':
        self.exception = exc
        return self

def loading_screen(logger: logging.Logger) -> bool:
    """Display loading screen with system checks. Returns True if all critical checks pass."""
    console.clear()
    
    # ASCII art banner
    console.print("\n" , Panel.fit("""
                            
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

    """,
        style="bold cyan", 
        title="[bold yellow]GreyChamp | IDS[/]", 
        subtitle="[magenta]SYSTEM INITIALIZATION[/]",
        border_style="bold blue",
        box=box.DOUBLE,
        padding=(1, 1)
    ))
    
    # Animated progress bar
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Running system diagnostics...", total=100)
        
        # Simulate progress while actually running checks
        for i in range(10):
            time.sleep(0.05)
            progress.update(task, advance=10)
    
    # Run checks
    checks = run_system_checks(logger)
    
    # Display results table
    display_check_results(checks)
    
    # Determine check status
    critical_passed = all(
        check.passed 
        for name, check in checks.items() 
        if check.level == CheckLevel.CRITICAL
    )
    
    important_failed = any(
        not check.passed 
        for name, check in checks.items() 
        if check.level == CheckLevel.IMPORTANT
    )
    
    informational_failed = any(
        not check.passed 
        for name, check in checks.items() 
        if check.level == CheckLevel.INFORMATIONAL
    )

    # Handle different scenarios
    if not critical_passed:
        console.print(
            Panel.fit(
                "[bold red]CRITICAL SYSTEM CHECKS FAILED[/bold red]\n"
                "The system cannot continue due to critical failures.",
                border_style="red"
            )
        )
        logger.critical("Critical system checks failed - cannot continue")
        return False
    
    elif important_failed or informational_failed:
        # Show failed checks
        fail_table = Table(
            title="\n[bold yellow]FAILED NON-CRITICAL CHECKS[/bold yellow]",
            box=box.SIMPLE,
            header_style="bold magenta",
            title_justify="left",
            show_header=True,
            show_lines=True
        )
        fail_table.add_column("Check", style="bold cyan")
        fail_table.add_column("Message", style="bold yellow")
        
        for name, check in [(n,c) for n,c in checks.items() if not c.passed and c.level != CheckLevel.CRITICAL]:
            fail_table.add_row(
                name.replace("_", " ").title(),
                check.message
            )
        
        console.print(fail_table)
        
        # Custom key prompt with proper key handling
        console.print(
            Panel.fit(
                "[bold yellow]WARNING: Some system checks failed[/bold yellow]\n"
                "[bold white]Press [/bold white][bold green]Enter[/bold green][bold white] to continue or [/bold white]"
                "[bold red]Esc[/bold red][bold white] to quit[/bold white]",
                border_style="yellow"
            )
        )
        
        # Use a flag to track the user's choice
        user_choice = None
        
        def on_press(key):
            nonlocal user_choice
            if key == Key.enter:
                user_choice = True
                # Stop listener
                return False
            elif key == Key.esc:
                user_choice = False
                # Stop listener
                return False
        
        with Listener(on_press=on_press) as listener:
            listener.join()
        
        if user_choice is False:
            logger.warning("User chose to quit after seeing failed checks")
            # Exit the program completely
            sys.exit(0)
        
        console.clear()
        return True
    
    else:
        # All checks passed
        console.print(
            Panel.fit(
                "[bold green]ALL SYSTEM CHECKS PASSED[/bold green]\n"
                "Press [bold green]Enter[/bold green] to continue...",
                border_style="green"
            )
        )
        
        # Wait specifically for Enter key
        def on_press(key):
            if key == Key.enter:
                # Stop listener
                return False
        
        with Listener(on_press=on_press) as listener:
            listener.join()
        
        console.clear()
        return True

def run_system_checks(logger: logging.Logger) -> Dict[str, CheckResult]:
    """Run all system checks using existing functions where possible"""
    checks = {}
    
    # Critical checks
    checks['python_version'] = check_python_version()
    checks['torch_available'] = check_torch()
    
    # Important checks
    checks['package_versions'] = check_package_versions_wrapper(logger)
    checks['directory_access'] = check_directory_access_wrapper(logger)
    checks['disk_space'] = check_disk_space()
    checks['cuda_available'] = check_cuda()
    
    # Informational checks
    checks['cpu_cores'] = check_cpu_cores()
    checks['system_ram'] = check_system_ram()
    checks['system_arch'] = check_system_architecture()
    checks['logging_setup'] = check_logging_setup(logger)
    checks['seed_config'] = check_seed_config()
    
    return checks

def display_check_results(checks: Dict[str, CheckResult]):
    """Display check results in a styled table with improved formatting"""
    result_table = Table(
        title="\n[bold]SYSTEM DIAGNOSTICS REPORT[/bold]",
        box=box.ROUNDED,
        header_style="bold white",
        border_style="white",
        title_style="bold yellow",
        title_justify="left",
        show_lines=True,
        expand=False,
        width=65
    )
    
    result_table.add_column("Check", style="bold cyan", width=20)
    result_table.add_column("Status", width=10)
    result_table.add_column("Details", style="dim", min_width=30)
    
    # Group by check level
    for level in CheckLevel:
        # Add section header
        level_style = {
            CheckLevel.CRITICAL: "bold red",
            CheckLevel.IMPORTANT: "bold yellow",
            CheckLevel.INFORMATIONAL: "bold magenta"
        }.get(level, "bold white")
        
        result_table.add_row(
            Text(level.name, style=level_style),
            "",
            "",
            style="dim"
        )
        
        # Add checks for this level
        for name, check in checks.items():
            if check.level == level:
                status_style = "bold green" if check.passed else "bold red" if level == CheckLevel.CRITICAL else "bold yellow"
                status_text = "PASS" if check.passed else "FAIL" if level == CheckLevel.CRITICAL else "WARN"
                
                details = check.message
                if check.details:
                    details += f"\n[dim]{check.details}[/dim]"
                if check.exception:
                    details += f"\n[bold red]Error: {str(check.exception)}[/bold red]"
                
                result_table.add_row(
                    Text(name.replace("_", " ").title(), style="cyan"),
                    Text(status_text, style=status_style),
                    details
                )
    
    console.print(result_table)

# Individual check implementations
def check_python_version(min_version: Tuple[int, int] = (3, 8)) -> CheckResult:
    try:
        current = tuple(map(int, platform.python_version().split('.')[:2]))
        passed = current >= min_version
        return CheckResult(
            passed=passed,
            message=f"[bold green]Python {platform.python_version()} (requires >= {'.'.join(map(str, min_version))})[/bold green]",
            level=CheckLevel.CRITICAL
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Could not determine Python version[/bold red]",
            level=CheckLevel.CRITICAL
        ).with_exception(e)

def check_torch() -> CheckResult:
    try:
        # Basic functionality test
        torch.zeros(1)
        return CheckResult(
            passed=True,
            message=f"[bold green]PyTorch {torch.__version__} available[/bold green]",
            level=CheckLevel.CRITICAL
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]PyTorch not available[/bold red]",
            level=CheckLevel.CRITICAL
        ).with_exception(e)

def check_cuda() -> CheckResult:
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return CheckResult(
                passed=True,
                message=f"[bold green]CUDA available ({props.name}, {props.total_memory/1e9:.1f}GB)[/bold green]",
                level=CheckLevel.IMPORTANT
            )
        return CheckResult(
            passed=False,
            message="[bold yellow]CUDA not available - Using CPU[/bold yellow]",
            level=CheckLevel.IMPORTANT
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]CUDA check failed[/bold red]",
            level=CheckLevel.IMPORTANT
        ).with_exception(e)

def check_package_versions_wrapper(logger: logging.Logger) -> CheckResult:
    try:
        # Test using check_versions function
        all_ok = check_versions(logger)
        return CheckResult(
            passed=all_ok,
            message="[bold green]Package version check completed[/bold green]",
            level=CheckLevel.IMPORTANT
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Package version check failed[/bold red]",
            level=CheckLevel.IMPORTANT
        ).with_exception(e)

def check_directory_access_wrapper(logger: logging.Logger) -> CheckResult:
    try:
        # Test using setup_directories function
        dirs = setup_directories(logger)
        problematic = []
        
        for name, path in dirs.items():
            try:
                test_file = path / ".permission_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                problematic.append(f"{name}: {str(e)}")
        
        if not problematic:
            return CheckResult(
                passed=True,
                message="[bold green]All directories accessible[/bold green]",
                level=CheckLevel.IMPORTANT
            )
        
        return CheckResult(
            passed=False,
            message=f"[bold yellow]{len(problematic)} directory issues[/bold yellow]",
            level=CheckLevel.IMPORTANT
        ).with_details("\n".join(problematic))
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Directory check failed[/bold red]",
            level=CheckLevel.IMPORTANT
        ).with_exception(e)

def check_disk_space(min_gb: int = 5) -> CheckResult:
    try:
        usage = psutil.disk_usage(".")
        free_gb = usage.free / (1024 ** 3)
        passed = free_gb >= min_gb
        return CheckResult(
            passed=passed,
            message=f"[bold green]{free_gb:.1f}GB free (needs >= {min_gb}GB)[/bold green]",
            level=CheckLevel.IMPORTANT
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Could not check disk space[/bold red]",
            level=CheckLevel.INFORMATIONAL
        ).with_exception(e)

def check_cpu_cores() -> CheckResult:
    cores = os.cpu_count() or 1
    return CheckResult(
        passed=True,
        message=f"[bold green]{cores} logical cores available[/bold green]",
        level=CheckLevel.INFORMATIONAL
    )

def check_system_ram() -> CheckResult:
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return CheckResult(
        passed=True,
        message=f"[bold green]{ram_gb:.1f}GB system RAM[/bold green]",
        level=CheckLevel.INFORMATIONAL
    )

def check_system_architecture() -> CheckResult:
    arch = platform.machine()
    return CheckResult(
        passed=True,
        message=f"[bold green]{arch} architecture[/bold green]",
        level=CheckLevel.INFORMATIONAL
    )

def check_logging_setup(logger: logging.Logger) -> CheckResult:
    try:
        if logger.handlers:
            handler_types = [type(h).__name__ for h in logger.handlers]
            return CheckResult(
                passed=True,
                message=f"[bold green]Logging configured ({', '.join(handler_types)})[/bold green]",
                level=CheckLevel.INFORMATIONAL
            )
        return CheckResult(
            passed=False,
            message="[bold yellow]Logging not configured[/bold yellow]",
            level=CheckLevel.IMPORTANT
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Logging check failed[/bold red]",
            level=CheckLevel.INFORMATIONAL
        ).with_exception(e)

def check_seed_config() -> CheckResult:
    try:
        seed = int(os.environ.get('PYTHONHASHSEED', '0'))
        return CheckResult(
            passed=seed != 0,
            message=f"[bold green]Reproducibility seed {'set' if seed !=0 else 'not set'}[/bold green]",
            level=CheckLevel.INFORMATIONAL
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Seed check failed[/bold red]",
            level=CheckLevel.INFORMATIONAL
        ).with_exception(e)

# System and environment configuration
def configure_system() -> None:
    """Configure system settings for optimal performance."""
    # Disable verbose logging for libraries
    
    # TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Intel MKL
    os.environ['KMP_WARNINGS'] = '0'
    
    # OpenMP
    os.environ['OMP_NUM_THREADS'] = '1'
    
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
VIZ_CONFIG = {
    'interactive': False,
    'max_samples': 5000,
    'projections': ['pca', 'tsne', 'umap'],
    'dpi': 150,
    'style': 'seaborn-v0_8',
    'backend': 'Agg' if os.environ.get('DISPLAY', '') == '' else None
}

def configure_visualization(
    interactive: bool = False,
    max_samples: int = 5000,
    projections: List[str] = None,
    dpi: int = 150,
    style: str = 'seaborn-v0_8'
) -> None:
    """Centralized visualization configuration with matplotlib defaults."""
    global VIZ_CONFIG
    
    # Handle projection methods
    if projections:
        valid_projections = ['pca', 'tsne', 'umap']
        if not all(p in valid_projections for p in projections):
            raise ValueError(f"Invalid projection method. Choose from {valid_projections}")
        VIZ_CONFIG['projections'] = projections
        
    # Update configuration
    VIZ_CONFIG.update({
        'interactive': interactive,
        'max_samples': max_samples,
        'dpi': dpi,
        'style': style
    })
    
    # Configure matplotlib
    try:
        plt.style.use(style)
        if VIZ_CONFIG['backend']:
            matplotlib.use(VIZ_CONFIG['backend'])
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
    rcParams['figure.dpi'] = dpi
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.transparent'] = False
    rcParams['font.family'] = 'DejaVu Sans'

# Reproducibility configuration
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    
    # Set to True if input sizes don't vary
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For CUDA reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Logging and Directory Setup
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

# Global directory variables
MODEL_DIR = LOG_DIR = DATA_DIR = FIGURE_DIR = TB_DIR = CHECKPOINT_DIR = None
CONFIG_DIR = RESULTS_DIR = METRICS_DIR = REPORTS_DIR = LATEST_DIR = None
INFO_DIR = ARTIFACTS_DIR = DOCS_DIR = None

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging with a single log file and proper handler management."""
    logger = logging.getLogger(__name__)
    
    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # Rest of the logging setup
    logger.setLevel(logging.DEBUG)
    
    # Use a fixed log filename (instead of timestamp-based)
    log_file = log_dir / "training_model.log"
    
    # Add handlers ONLY if they don't exist
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        # Append mode
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
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
    base_dir = Path(__file__).resolve().parent
    
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
        'artifacts': base_dir / "artifacts",
        'docs': base_dir / "docs"
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

def configure_directories(logger: logging.Logger) -> Dict[str, Path]:
    """
    Creates and assigns global variables for all essential directories.
    Returns a dictionary with all paths.
    """
    global MODEL_DIR, LOG_DIR, DATA_DIR, FIGURE_DIR, TB_DIR, CHECKPOINT_DIR
    global CONFIG_DIR, RESULTS_DIR, METRICS_DIR, REPORTS_DIR, LATEST_DIR
    global INFO_DIR, ARTIFACTS_DIR, DOCS_DIR

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
        DOCS_DIR = directories['docs']

        return directories

    except Exception as e:
        logger.error(Fore.RED + f"Failed to set up directories: {str(e)}" + Style.RESET_ALL)
        sys.exit(1)

# Hardware and Package Configuration
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
        logger.info(Fore.WHITE + Style.BRIGHT + "Using " + Fore.GREEN + Style.BRIGHT + "CPU " + Fore.WHITE + Style.BRIGHT + "for training")
        torch.set_num_threads(os.cpu_count() or 1)
        logger.info(Fore.WHITE + Style.BRIGHT + "Using " + Fore.GREEN + Style.BRIGHT + f"{torch.get_num_threads()} CPU threads")
    
    return device

def check_versions(logger: logging.Logger) -> bool:
    """Verify package versions with rich table output and full logging."""
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

    # Create rich table
    table = Table(title="Package Versions", box=box.ROUNDED, title_justify="left", title_style="bold yellow")
    table.add_column("Package", style="bold cyan", no_wrap=True)
    table.add_column("Installed", style="bold magenta")
    table.add_column("Required", style="bold green")
    table.add_column("Status", justify="center")

    # Add rows to table
    all_ok = True
    for pkg, current_ver, min_ver, is_ok, error in version_data:
        if current_ver:
            status = Text("[OK]", style="bold green") if is_ok else Text(f"[FAILED] Needs >= {min_ver}", style="bold red")
        else:
            status = Text("[MISSING]", style="bold red")
            current_ver = Text("[N/A]", style="italic")
        
        if not is_ok:
            all_ok = False
        
        table.add_row(pkg, current_ver, f">= {min_ver}", status)

    # Print the table
    console.print()
    console.print(table)
    
    # Add summary panel
    if all_ok:
        console.print(Panel.fit(
            Text("All package versions are compatible", style="bold green"),
            title="Status", style="bold green",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            Text("Some package versions are incompatible", style="bold red"),
            title="Status", style="bold red",
            border_style="red"
        ))

    # Log the complete version info at DEBUG level
    debug_output = ["=== Package Versions ==="]
    for pkg, current_ver, min_ver, is_ok, _ in version_data:
        if current_ver:
            status = "[OK]" if is_ok else f"[FAIL] (needs >= {min_ver})"
            line = f"{pkg} {current_ver} (>= {min_ver}) {status}"
        else:
            line = f"{pkg} [N/A] (>= {min_ver}) [MISSING]"
        debug_output.append(line)
    
    logger.debug("\n".join(debug_output))

    return all_ok

# Configuration Management
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
#DROPOUT_RATES = [0.5, 0.4, 0.3, 0.2]
DROPOUT_RATES = [0.3, 0.25, 0.2, 0.15]
ACTIVATION = 'leaky_relu'
ACTIVATION_PARAM = 0.1
USE_BATCH_NORM = True
USE_LAYER_NORM = False

DEFAULT_ATTACK_THRESHOLD = 0.3
FALSE_NEGATIVE_COST = 2.0
SECURITY_METRICS = True

# Configuration for testing different architectures
STABILITY_CONFIG = {
    # Start with simpler model
    'model_type': 'simple',
    'use_batch_norm': True,
    # Less aggressive
    'dropout_rates': [0.3, 0.25, 0.2, 0.15],
    'gradient_clip': 1.0,
    'warmup_epochs': 5,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    # Security-focused
    'fn_cost': 2.0
}

PERFORMANCE_CONFIG = {
    # For best performance
    'model_type': 'ensemble',
    'num_ensemble_models': 3,
    'use_batch_norm': True,
    'dropout_rates': [0.2, 0.15, 0.1],
    'gradient_clip': 0.5,
    'warmup_epochs': 10,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5
}

# Model architecture options
MODEL_VARIANTS = {
    'standard': IDSModel,
    'simple': SimpleIDSModel,
    'stabilized': StabilizedIDSModel,
    'ensemble': EnsembleIDSModel
}

# Add configuration presets for easy testing
PRESET_CONFIGS = {
    'stability': {
        'model_type': 'simple',
        'use_batch_norm': True,
        'dropout_rates': [0.3, 0.25, 0.2, 0.15],
        'gradient_clip': 1.0,
        'warmup_epochs': 5,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'fn_cost': 2.0,
        'batch_size': 64,
        'epochs': 50,
        'early_stopping': 8
    },
    'performance': {
        'model_type': 'ensemble',
        'num_ensemble_models': 3,
        'use_batch_norm': True,
        'dropout_rates': [0.2, 0.15, 0.1],
        'gradient_clip': 0.5,
        'warmup_epochs': 10,
        'learning_rate': 5e-4,
        'weight_decay': 1e-5,
        'fn_cost': 2.0,
        'batch_size': 128,
        'epochs': 100,
        'early_stopping': 12
    },
    'baseline': {
        'model_type': 'standard',
        'use_batch_norm': True,
        'dropout_rates': [0.3, 0.25, 0.2, 0.15],
        'gradient_clip': 1.0,
        'warmup_epochs': 3,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'fn_cost': 1.5,
        'batch_size': 64,
        'epochs': 75,
        'early_stopping': 10
    },
    'debug': {
        'model_type': 'simple',
        'use_batch_norm': True,
        'dropout_rates': [0.2, 0.1],
        'gradient_clip': 0.5,
        'warmup_epochs': 2,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'fn_cost': 1.0,
        'batch_size': 32,
        'epochs': 10,
        'early_stopping': 5
    }
}

def initialize_model_variants():
    """Initialize MODEL_VARIANTS dictionary after all classes are defined"""
    global MODEL_VARIANTS
    MODEL_VARIANTS = {}
    
    # Check if each model class exists and add it
    try:
        if 'IDSModel' in globals():
            MODEL_VARIANTS['standard'] = IDSModel
    except NameError:
        pass
    
    try:
        if 'SimpleIDSModel' in globals():
            MODEL_VARIANTS['simple'] = SimpleIDSModel
    except NameError:
        pass
    
    try:
        if 'StabilizedIDSModel' in globals():
            MODEL_VARIANTS['stabilized'] = StabilizedIDSModel
    except NameError:
        pass
    
    try:
        if 'EnsembleIDSModel' in globals():
            MODEL_VARIANTS['ensemble'] = EnsembleIDSModel
    except NameError:
        pass

def compare_model_architectures() -> Dict[str, Dict[str, Any]]:
    """Compare parameter counts and complexity of different model architectures"""
    results = {}
    
    # Initialize model variants if empty
    if not MODEL_VARIANTS:
        initialize_model_variants()
    
    # Test input size (typical for network traffic features)
    # Example feature count
    test_input_size = 78
    
    # Binary classification
    test_output_size = 2
    
    console.print(f"[dim]Testing with input size: {test_input_size}, output size: {test_output_size}[/]")
    console.print(f"[dim]Available models: {list(MODEL_VARIANTS.keys())}[/]")
    
    for model_name, model_class in MODEL_VARIANTS.items():
        try:
            console.print(f"[dim]Creating {model_name} model...[/]")
            
            if model_name == 'standard':
                model = model_class(
                    input_size=test_input_size,
                    output_size=test_output_size,
                    use_batch_norm=True,
                    # Ensure we don't exceed available rates
                    dropout_rates=DROPOUT_RATES[:4]
                )
            elif model_name == 'simple':
                model = model_class(
                    input_size=test_input_size,
                    output_size=test_output_size,
                    dropout_rate=0.2
                )
            elif model_name == 'ensemble':
                model = model_class(
                    input_size=test_input_size,
                    output_size=test_output_size,
                    num_models=3
                )
            elif model_name == 'stabilized':
                model = model_class(
                    input_size=test_input_size,
                    num_classes=test_output_size,
                    dropout_rate=0.2
                )
            else:
                continue
                
            # Calculate parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage (rough approximation)
            param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'memory_mb': param_memory_mb,
                'model_class': model_class.__name__
            }
            
            console.print(f"[dim]✓ {model_name}: {total_params:,} parameters[/]")
            
        except Exception as e:
            console.print(f"[dim]✗ {model_name} failed: {str(e)}[/]")
            results[model_name] = {'error': str(e)}
    
    return results

def display_model_comparison():
    """Display model architecture comparison in a rich table"""
    console.print("\n[bold cyan]Analyzing model architectures...[/]")
    
    # Check if models are available
    available_models = []
    model_check_results = []
    
    # Check each model class
    models_to_check = [
        ('standard', 'IDSModel'),
        ('simple', 'SimpleIDSModel'), 
        ('stabilized', 'StabilizedIDSModel'),
        ('ensemble', 'EnsembleIDSModel')
    ]
    
    for model_key, class_name in models_to_check:
        if class_name in globals():
            available_models.append(model_key)
            model_check_results.append(f"✓ {class_name}")
        else:
            model_check_results.append(f"✗ {class_name} (not found)")
    
    # Show what models are available
    console.print("\n[bold]Model Availability Check:[/]")
    for result in model_check_results:
        if "✓" in result:
            console.print(f"[green]{result}[/]")
        else:
            console.print(f"[red]{result}[/]")
    
    if not available_models:
        console.print(
            Panel.fit(
                Text("No model classes found! Please ensure all model classes are defined.", style="bold red"),
                title="[bold red]Error[/]",
                border_style="red"
            )
        )
        return
    
    # Run comparison only on available models
    comparison = compare_model_architectures()
    
    if not comparison:
        console.print(
            Panel.fit(
                Text("Model comparison failed - no results generated", style="bold red"),
                title="[bold red]Error[/]",
                border_style="red"
            )
        )
        return
    
    console.print()
    comp_table = Table(
        title="[bold cyan]Model Architecture Comparison[/]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        show_header=True,
        show_lines=True
    )
    
    comp_table.add_column("Model", style="bold yellow", width=12)
    comp_table.add_column("Parameters", style="bold white", justify="right")
    comp_table.add_column("Memory (MB)", style="bold green", justify="right")
    comp_table.add_column("Complexity", style="bold magenta", justify="center")
    comp_table.add_column("Status", style="bold white", justify="center")
    
    for model_name, stats in comparison.items():
        if 'error' in stats:
            comp_table.add_row(
                model_name,
                "[red]Error[/]",
                "[red]N/A[/]",
                "[red]N/A[/]",
                f"[red]{stats['error'][:30]}...[/]" if len(stats['error']) > 30 else f"[red]{stats['error']}[/]"
            )
        else:
            # Determine complexity level
            params = stats['total_params']
            if params < 10000:
                complexity = "[green]Low[/]"
            elif params < 100000:
                complexity = "[yellow]Medium[/]"
            else:
                complexity = "[red]High[/]"
            
            comp_table.add_row(
                model_name,
                f"{stats['total_params']:,}",
                f"{stats['memory_mb']:.1f}",
                complexity,
                "[green]✓ Working[/]"
            )
    
    console.print(comp_table)
    
    # Show recommendations
    console.print()
    recommendations = Table(
        title="[bold]Recommendations[/]",
        box=box.SIMPLE,
        header_style="bold yellow",
        border_style="yellow"
    )
    recommendations.add_column("Use Case", style="bold cyan")
    recommendations.add_column("Recommended Model", style="bold green")
    recommendations.add_column("Reason", style="bold white")
    
    recommendations.add_row("Quick Testing", "simple", "Fastest training, good for debugging")
    recommendations.add_row("Stable Training", "stabilized", "Balanced performance and stability")
    recommendations.add_row("Best Performance", "ensemble", "Highest accuracy, longer training")
    recommendations.add_row("Production", "standard", "Good balance of all factors")
    
    console.print(recommendations)

def update_config(key_path: str, value: Any, config_path: Path, logger: logging.Logger) -> None:
    """Update a specific configuration value and track changes."""
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Update modification timestamp
        config_data['metadata']['modified'] = datetime.datetime.now().isoformat()
        
        # Update the specific value
        keys = key_path.split('.')
        current_level = config_data['config']
        
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
        
        # Log the change
        old_value = current_level.get(keys[-1], '<not set>')
        logger.info(f"Updating config: {key_path} from {old_value} to {value}")
        
        current_level[keys[-1]] = value
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        logger.info(f"Configuration updated in: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {str(e)}")
        raise

def get_current_config() -> Dict[str, Any]:
    """Get current configuration with all constants."""
    return {
        'training': {
            'batch_size': DEFAULT_BATCH_SIZE,
            'epochs': DEFAULT_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'gradient_clip': GRADIENT_CLIP,
            'mixed_precision': MIXED_PRECISION,
            'early_stopping': EARLY_STOPPING_PATIENCE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS
        },
        'model': {
            'architecture': {
                'hidden_layers': HIDDEN_LAYER_SIZES,
                'dropout_rates': DROPOUT_RATES,
                'activation': ACTIVATION,
                'activation_param': ACTIVATION_PARAM,
                'use_batch_norm': USE_BATCH_NORM,
                'use_layer_norm': USE_LAYER_NORM
            }
        },
        'system': {
            'seed': 42,
            'logging_level': 'INFO'
        },
        'security': {
            'attack_threshold': DEFAULT_ATTACK_THRESHOLD,
            'false_negative_cost': FALSE_NEGATIVE_COST,
            'enable_security_metrics': SECURITY_METRICS
        }
    }

def save_config(config: Dict[str, Any], config_path: Path, logger: logging.Logger) -> None:
    """Save configuration to JSON file with metadata and versioning."""
    try:
        #from datetime import datetime
        # Add metadata
        config_with_meta = {
            "metadata": {
                "created": datetime.datetime.now().isoformat(),
                "modified": datetime.datetime.now().isoformat(),
                "version": "1.0",
                "system": {
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                    "os": platform.system()
                }
            },
            "config": config
        }
        
        # Create backup if file exists
        if config_path.exists():
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.parent / f"{config_path.stem}_backup_{timestamp}{config_path.suffix}"
            shutil.copy(config_path, backup_path)
            logger.info(f"Created backup of config at: {backup_path}")
        
        # Save new config
        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=4)
        logger.info(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}")
        raise

def load_config(config_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Load configuration from JSON file with error handling."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config_data.get('config', {})
    
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return {}

def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary with another dictionary."""
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original

def initialize_config(logger: logging.Logger) -> Dict[str, Any]:
    """Initialize or load configuration with version control."""
    config_path = CONFIG_DIR / "train_model_config.json"
    
    # Try loading existing config
    loaded_config = load_config(config_path, logger)
    
    if loaded_config:
        # Validate loaded config against current defaults
        current_config = get_current_config()
        validated_config = deep_update(current_config, loaded_config)
        save_config(validated_config, config_path, logger)
        return validated_config
    else:
        # Create new config with defaults
        current_config = get_current_config()
        save_config(current_config, config_path, logger)
        return current_config

def update_global_config(config: Dict[str, Any]) -> None:
    """Update global variables from configuration."""
    global DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    global GRADIENT_CLIP, MIXED_PRECISION, EARLY_STOPPING_PATIENCE
    global HIDDEN_LAYER_SIZES, DROPOUT_RATES, ACTIVATION, ACTIVATION_PARAM
    global USE_BATCH_NORM, USE_LAYER_NORM
    
    # Update training parameters
    training = config.get('training', {})
    DEFAULT_BATCH_SIZE = training.get('batch_size', DEFAULT_BATCH_SIZE)
    DEFAULT_EPOCHS = training.get('epochs', DEFAULT_EPOCHS)
    LEARNING_RATE = training.get('learning_rate', LEARNING_RATE)
    WEIGHT_DECAY = training.get('weight_decay', WEIGHT_DECAY)
    GRADIENT_CLIP = training.get('gradient_clip', GRADIENT_CLIP)
    MIXED_PRECISION = training.get('mixed_precision', MIXED_PRECISION)
    EARLY_STOPPING_PATIENCE = training.get('early_stopping', EARLY_STOPPING_PATIENCE)
    
    # Update model architecture
    model_arch = config.get('model', {}).get('architecture', {})
    HIDDEN_LAYER_SIZES = model_arch.get('hidden_layers', HIDDEN_LAYER_SIZES)
    DROPOUT_RATES = model_arch.get('dropout_rates', DROPOUT_RATES)
    ACTIVATION = model_arch.get('activation', ACTIVATION)
    ACTIVATION_PARAM = model_arch.get('activation_param', ACTIVATION_PARAM)
    USE_BATCH_NORM = model_arch.get('use_batch_norm', USE_BATCH_NORM)
    USE_LAYER_NORM = model_arch.get('use_layer_norm', USE_LAYER_NORM)

def show_config() -> None:
    """Show current configuration with rich formatting as separate distinct tables"""
    #from datetime import datetime
    config = get_current_config()
    
    # Display main configuration panel
    console.print()
    console.print(Panel.fit(
        Text("Current Configuration", justify="center", style="bold blue"),
        border_style="blue",
        padding=(0, 1)
    ))
    
    # Training Parameters Table
    train_table = Table(
        title="[bold cyan]Training Parameters[/]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        show_header=True,
        show_lines=False,
        min_width=40
    )
    train_table.add_column("Parameter", style="bold yellow", no_wrap=True, justify="left")
    train_table.add_column("Value", style="bold white", justify="left")
    
    train_table.add_row("Batch Size", str(config['training']['batch_size']))
    train_table.add_row("Epochs", str(config['training']['epochs']))
    train_table.add_row("Learning Rate", f"{config['training']['learning_rate']:.0e}")
    train_table.add_row("Weight Decay", f"{config['training']['weight_decay']:.0e}")
    train_table.add_row("Gradient Clip", str(config['training']['gradient_clip']))
    train_table.add_row(
        "Mixed Precision", 
        Text("[Enabled]", style="bold green") if config['training']['mixed_precision'] 
        else Text("[Disabled]", style="bold red")
    )
    train_table.add_row("Early Stopping", str(config['training']['early_stopping']))
    
    # Model Architecture Table
    model_table = Table(
        title="[bold cyan]Model Architecture[/]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        show_header=True,
        show_lines=False,
        min_width=40
    )
    model_table.add_column("Parameter", style="bold yellow", no_wrap=True, justify="left")
    model_table.add_column("Value", style="bold white", justify="left")
    
    model_table.add_row(
        "Hidden Layers", 
        Text(", ".join(map(str, config['model']['architecture']['hidden_layers'])))
    )
    model_table.add_row(
        "Dropout Rates", 
        Text(", ".join(map(str, config['model']['architecture']['dropout_rates'])))
    )
    model_table.add_row(
        "Activation", 
        Text(config['model']['architecture']['activation'])
    )
    model_table.add_row(
        "Batch Norm", 
        Text("[Enabled]", style="bold green") if config['model']['architecture']['use_batch_norm'] 
        else Text("[Disabled]", style="bold red")
    )
    model_table.add_row(
        "Layer Norm", 
        Text("[Enabled]", style="bold green") if config['model']['architecture']['use_layer_norm'] 
        else Text("[Disabled]", style="bold red")
    )
    
    # Display tables in a grid layout with proper spacing
    console.print(Panel.fit(train_table, border_style="blue"))
    # Add spacing between tables
    console.print()
    console.print(Panel.fit(model_table, border_style="blue"))
    
    # Add config file info
    config_path = CONFIG_DIR / "train_model_config.json"
    if config_path.exists():
        modified_time = datetime.datetime.fromtimestamp(config_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        console.print(
            Panel.fit(
                Text(f"Config last saved: {modified_time}\nLocation: {config_path}", style="bold dim"),
                border_style="dim",
                padding=(0, 1)
            )
        )
    
    # Add some vertical spacing
    console.print()

def configure_training() -> None:
    """Configure training parameters interactively with rich formatting"""
    global DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    global GRADIENT_CLIP, MIXED_PRECISION, EARLY_STOPPING_PATIENCE
    
    # Create input panel
    console.print()
    input_panel = Panel.fit(
        Text("Enter new values or press Enter to keep current", style="italic"),
        title="[bold cyan]Training Configuration[/]",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(input_panel)
    
    try:
        # Get inputs with current values as defaults
        DEFAULT_BATCH_SIZE = int(console.input(f"Batch size [[yellow]{DEFAULT_BATCH_SIZE}[/]]: ") or DEFAULT_BATCH_SIZE)
        DEFAULT_EPOCHS = int(console.input(f"Epochs [[yellow]{DEFAULT_EPOCHS}[/]]: ") or DEFAULT_EPOCHS)
        LEARNING_RATE = float(console.input(f"Learning rate [[yellow]{LEARNING_RATE}[/]]: ") or LEARNING_RATE)
        WEIGHT_DECAY = float(console.input(f"Weight decay [[yellow]{WEIGHT_DECAY}[/]]: ") or WEIGHT_DECAY)
        GRADIENT_CLIP = float(console.input(f"Gradient clip [[yellow]{GRADIENT_CLIP}[/]]: ") or GRADIENT_CLIP)
        mp_input = console.input(f"Use mixed precision? (y/n) [[yellow]{'y' if MIXED_PRECISION else 'n'}[/]]: ").lower()
        MIXED_PRECISION = mp_input == 'y' if mp_input else MIXED_PRECISION
        EARLY_STOPPING_PATIENCE = int(console.input(
            f"Early stopping patience [[yellow]{EARLY_STOPPING_PATIENCE}[/]]: "
        ) or EARLY_STOPPING_PATIENCE)
        
        # Update config file
        config = get_current_config()
        config['training'].update({
            'batch_size': DEFAULT_BATCH_SIZE,
            'epochs': DEFAULT_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'gradient_clip': GRADIENT_CLIP,
            'mixed_precision': MIXED_PRECISION,
            'early_stopping': EARLY_STOPPING_PATIENCE
        })
        save_config(config, CONFIG_DIR / "train_model_config.json", logger)
        
        # Show success message
        console.print(
            Panel.fit(
                Text("Training configuration updated successfully", style="bold green"),
                border_style="green",
                padding=(1, 2)
            )
        )
    except ValueError as e:
        console.print(
            Panel.fit(
                Text(f"Invalid input: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
                padding=(1, 2)
            )
        )

def configure_model() -> None:
    """Configure model architecture interactively with rich formatting"""
    global HIDDEN_LAYER_SIZES, DROPOUT_RATES, ACTIVATION, ACTIVATION_PARAM
    global USE_BATCH_NORM, USE_LAYER_NORM
    
    # Create input panel
    console.print()
    input_panel = Panel.fit(
        Text("Enter new values or press Enter to keep current", style="italic"),
        title="[bold cyan]Model Architecture Configuration[/]",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(input_panel)
    
    try:
        # Get hidden layer sizes
        layers_input = console.input(
            f"Hidden layer sizes (comma separated) [[yellow]{', '.join(map(str, HIDDEN_LAYER_SIZES))}[/]]: "
        )
        if layers_input:
            HIDDEN_LAYER_SIZES = [int(x.strip()) for x in layers_input.split(',')]
        
        # Get dropout rates
        dropout_input = console.input(
            f"Dropout rates (comma separated) [[yellow]{', '.join(map(str, DROPOUT_RATES))}[/]]: "
        )
        if dropout_input:
            DROPOUT_RATES = [float(x.strip()) for x in dropout_input.split(',')]
        
        # Validate layer sizes and dropout rates match
        if len(HIDDEN_LAYER_SIZES) != len(DROPOUT_RATES):
            console.print(
                Panel.fit(
                    Text("Error: Number of hidden layers must match number of dropout rates", style="bold red"),
                    title="Error",
                    border_style="red",
                    padding=(1, 2)
                )
            )
            return
        
        # Get activation function
        act_input = console.input(
            f"Activation function (relu/leaky_relu/gelu) [[yellow]{ACTIVATION}[/]]: "
        )
        ACTIVATION = act_input or ACTIVATION
        
        if ACTIVATION == 'leaky_relu':
            ACTIVATION_PARAM = float(console.input(
                f"Leaky ReLU negative slope [[yellow]{ACTIVATION_PARAM}[/]]: "
            ) or ACTIVATION_PARAM)
        
        # Get batch norm input
        bn_input = console.input(
            f"Use batch normalization? (y/n) [[yellow]{'y' if USE_BATCH_NORM else 'n'}[/]]: "
        ).lower()
        USE_BATCH_NORM = bn_input == 'y' if bn_input else USE_BATCH_NORM
        
        # Get layer norm input
        ln_input = console.input(
            f"Use layer normalization? (y/n) [[yellow]{'y' if USE_LAYER_NORM else 'n'}[/]]: "
        ).lower()
        USE_LAYER_NORM = ln_input == 'y' if ln_input else USE_LAYER_NORM
        
        # Update config file
        config = get_current_config()
        config['model']['architecture'].update({
            'hidden_layers': HIDDEN_LAYER_SIZES,
            'dropout_rates': DROPOUT_RATES,
            'activation': ACTIVATION,
            'activation_param': ACTIVATION_PARAM,
            'use_batch_norm': USE_BATCH_NORM,
            'use_layer_norm': USE_LAYER_NORM
        })
        save_config(config, CONFIG_DIR / "train_model_config.json", logger)
        
        # Show success message
        console.print(
            Panel.fit(
                Text("Model configuration updated successfully", style="bold green"),
                border_style="green",
                padding=(1, 2)
            )
        )
    except ValueError as e:
        console.print(
            Panel.fit(
                Text(f"Invalid input: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
                padding=(1, 2)
            )
        )

# Main configuration and setup function
def initialize_system():
    """Centralized system initialization with single logging setup"""
    # Basic configuration
    configure_system()
    configure_visualization()
    set_seed(42)
    
    # Early logging setup
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir)
    
    # Run loading screen with checks
    if not loading_screen(logger):
        console.print(
            Panel.fit(
                Text("CRITICAL SYSTEM CHECKS FAILED - Cannot continue", style="bold red"),
                border_style="red"
            )
        )
        sys.exit(1)
    
    # Continue with normal setup if checks passed
    if not check_versions(logger):
        logger.error(Fore.RED + "Some package requirements not met!" + Style.RESET_ALL)
    
    device = setup_gpu(logger)
    directories = configure_directories(logger)
    
    config = initialize_config(logger)
    update_global_config(config)
    
    # Initialize model variants after all classes are loaded
    initialize_model_variants()
    
    return logger, device, directories, config

# Model architecture
class IDSModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, use_batch_norm: bool = True, dropout_rates: List[float] = None):
        """Enhanced IDS model with flexible architecture and normalization options."""
        super().__init__()
        
        # Default dropout rates (less aggressive for stability)
        if dropout_rates is None:
            dropout_rates = [0.3, 0.25, 0.2, 0.15]
        
        # Ensure we have enough dropout rates
        while len(dropout_rates) < len(HIDDEN_LAYER_SIZES):
            dropout_rates.append(dropout_rates[-1])
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for i, (size, dropout) in enumerate(zip(HIDDEN_LAYER_SIZES, dropout_rates)):
            layers.append(nn.Linear(prev_size, size))
            
            # Add normalization (configurable)
            if use_batch_norm or USE_BATCH_NORM:
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimpleIDSModel(nn.Module):
    """Simplified IDS model for testing - less parameters to reduce overfitting"""
    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # Smaller architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, output_size)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier normal for smaller network."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EnsembleIDSModel(nn.Module):
    """Ensemble of multiple models for improved performance"""
    def __init__(self, input_size: int, output_size: int, num_models: int = 3):
        super().__init__()
        
        self.models = nn.ModuleList([
            SimpleIDSModel(input_size, output_size, dropout_rate=0.1 + i*0.1)
            for i in range(num_models)
        ])
        
        # Optional: Add a combiner layer
        self.combiner = nn.Linear(output_size * num_models, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get outputs from all models
        outputs = [model(x) for model in self.models]
        
        # Simple averaging
        ensemble_output = torch.stack(outputs).mean(dim=0)
        
        # Optional: Use learnable combination
        # combined = torch.cat(outputs, dim=1)
        # ensemble_output = self.combiner(combined)
        
        return ensemble_output

class StabilizedIDSModel(nn.Module):
    """IDS model with batch normalization and dropout for stability"""
    def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

class WarmupScheduler:
    """Learning rate warmup scheduler"""
    def __init__(self, optimizer, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_epochs:
            lr_scale = self.current_step / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * self.base_lr

# Fix SecurityAwareLoss class if missing
class SecurityAwareLoss(nn.Module):
    """Custom loss function that penalizes false negatives more heavily"""
    def __init__(self, class_weights: torch.Tensor = None, false_negative_cost: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.false_negative_cost = false_negative_cost
        self.base_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_criterion(outputs, targets)
        
        # Additional penalty for false negatives (missed attacks)
        probs = torch.softmax(outputs, dim=1)
        attack_class = 1  # Assuming attack is class 1
        
        # Penalty when model predicts normal (low attack prob) but target is attack
        fn_mask = (targets == attack_class) & (probs[:, attack_class] < 0.5)
        fn_penalty = fn_mask.float().mean() * self.false_negative_cost
        
        return base_loss + fn_penalty

# Data preprocessing and validation
def check_preprocessing_outputs(
    logger: logging.Logger,
    strict: bool = False,
    use_color: bool = True,
    min_csv_size: int = 1024,
    min_pkl_size: int = 128,
    validate_csv: bool = True,
    validate_pickle: bool = True
) -> bool:
    """Verify preprocessing outputs with optional validation.
    
    Args:
        logger: Configured logger instance for logging messages
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
            "checks": ["header", "delimiter"] if validate_csv else [],
            "description": "Preprocessed dataset CSV"
        },
        "models/preprocessing_artifacts.pkl": {
            "min_size": min_pkl_size,
            "required_keys": ["feature_names", "scaler"] if validate_pickle else [],
            "description": "Preprocessing artifacts pickle"
        }
    }
    
    all_valid = True
    
    logger.info("Starting preprocessing outputs validation...")
    logger.debug(f"Validation mode: {'STRICT' if strict else 'BASIC'}")
    
    for filepath, requirements in required_files.items():
        path = Path(filepath)
        logger.debug(f"Checking {requirements['description']} at {path}")
        
        # File existence check (always performed)
        if not path.exists():
            logger.error(f"{red}Missing required file: {filepath}{reset}")
            all_valid = False
            continue
            
        # File size validation
        file_size = path.stat().st_size
        if file_size < requirements["min_size"]:
            msg = f"File appears small ({file_size} bytes): {filepath}"
            if strict:
                logger.error(f"{red}{msg}{reset}")
                all_valid = False
            else:
                logger.warning(f"{yellow}{msg}{reset}")
        
        # Skip content validation in non-strict mode
        if not strict:
            continue
            
        # CSV validation
        if filepath.endswith('.csv') and validate_csv:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    header = f.readline()
                    if not header.strip():
                        logger.error(f"{red}Empty CSV file: {filepath}{reset}")
                        all_valid = False
                        continue
                        
                    if "delimiter" in requirements["checks"]:
                        if len(header.split(',')) < 2:
                            logger.error(f"{red}Invalid CSV format in: {filepath}{reset}")
                            all_valid = False
                            
                # Additional validation - check sample rows
                try:
                    sample_df = pd.read_csv(path, nrows=10)
                    if sample_df.empty:
                        logger.error(f"{red}CSV contains no data: {filepath}{reset}")
                        all_valid = False
                except Exception as e:
                    logger.error(f"{red}CSV sample read failed: {filepath} - {str(e)}{reset}")
                    all_valid = False
                    
            except UnicodeDecodeError:
                logger.error(f"{red}Invalid CSV encoding: {filepath}{reset}")
                all_valid = False
            except Exception as e:
                logger.error(f"{red}CSV validation failed: {filepath} - {str(e)}{reset}")
                all_valid = False
                
        # Pickle validation
        elif filepath.endswith('.pkl') and validate_pickle:
            try:
                with open(path, 'rb') as f:
                    data = joblib.load(f)
                    for key in requirements["required_keys"]:
                        if key not in data:
                            logger.error(f"{red}Missing key '{key}' in: {filepath}{reset}")
                            all_valid = False
                            
                    # Additional validation for specific artifacts
                    if "feature_names" in data:
                        if not isinstance(data["feature_names"], list) or len(data["feature_names"]) == 0:
                            logger.error(f"{red}Invalid feature_names in: {filepath}{reset}")
                            all_valid = False
                    if "scaler" in data and data["scaler"] is not None:
                        if not hasattr(data["scaler"], "transform"):
                            logger.error(f"{red}Invalid scaler object in: {filepath}{reset}")
                            all_valid = False
                            
            except Exception as e:
                logger.error(f"{red}Pickle load failed: {filepath} - {str(e)}{reset}")
                all_valid = False
    
    if all_valid:
        logger.info("All preprocessing outputs validated successfully")
    else:
        logger.error("Preprocessing outputs validation failed")
        
    return all_valid

def run_preprocessing(
    logger: logging.Logger,
    timeout_minutes: float = 30.0,
    cleanup: bool = True,
    use_color: bool = True,
    strict_output_check: bool = True,
    reproducible: bool = True,
    debug: bool = False
) -> bool:
    """Execute preprocessing with enhanced controls.
    
    Args:
        logger: Configured logger instance for logging messages
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

    logger.info(f"{yellow}=== Starting Preprocessing Pipeline ==={reset}")
    logger.debug(f"Parameters: timeout={timeout_minutes}min, cleanup={cleanup}, strict={strict_output_check}, reproducible={reproducible}")

    # Validate script existence
    script_path = Path("preprocessing.py")
    if not script_path.exists():
        logger.error(f"{red}Preprocessing script not found at {script_path.absolute()}{reset}")
        raise FileNotFoundError(f"preprocessing.py not found at {script_path.absolute()}")

    try:
        # Cleanup previous outputs
        output_files = [
            Path("models/preprocessed_dataset.csv"),
            Path("models/preprocessing_artifacts.pkl")
        ]
        
        if cleanup:
            logger.info(f"{yellow}Cleaning up previous outputs...{reset}")
            for fpath in output_files:
                if fpath.exists():
                    logger.debug(f"Removing existing file: {fpath}")
                    fpath.unlink(missing_ok=True)
                    logger.info(f"Removed: {fpath}")

        # Prepare environment
        env = os.environ.copy()
        if reproducible:
            env["PYTHONHASHSEED"] = "42"
            logger.debug("Set PYTHONHASHSEED=42 for reproducibility")

        # Execute with timeout
        timeout_seconds = int(timeout_minutes * 60)
        logger.info(f"Running preprocessing with timeout of {timeout_minutes} minutes...")
        
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "preprocessing.py"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env
        )
        elapsed_time = time.time() - start_time

        # Stream output with proper logging levels
        logger.debug(f"Preprocessing completed in {elapsed_time:.2f} seconds")
        for line in result.stdout.splitlines():
            if line.startswith("ERROR"):
                logger.error(f"{red}{line}{reset}")
            elif line.startswith("WARNING"):
                logger.warning(f"{yellow}{line}{reset}")
            else:
                # Truncate very long lines
                logger.info(f"{green}{line[:500]}{reset}")
            
            if debug and len(line) > 500:
                logger.debug(f"Full output line: {line}")

        # Validate outputs
        logger.info("Validating preprocessing outputs...")
        if not check_preprocessing_outputs(
            logger=logger,
            strict=strict_output_check,
            use_color=use_color
        ):
            logger.error(f"{red}Output validation failed{reset}")
            log_troubleshooting(logger, "validation")
            return False

        logger.info(f"{green}✓ Preprocessing completed successfully{reset}")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"{red}Timeout after {timeout_minutes} minutes{reset}")
        log_troubleshooting(logger, "timeout")
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"{red}Process failed with exit code {e.returncode}{reset}")
        logger.error(f"{red}Error output:{reset}")
        log_error_output(logger, e.stderr, use_color)
        log_troubleshooting(logger, "execution")
        return False

    except Exception as e:
        logger.error(f"{red}Unexpected error: {type(e).__name__}{reset}")
        logger.error(f"{red}Error details: {str(e)}{reset}")
        if debug:
            logger.error(f"{red}Stack trace:{reset}")
            logger.error(traceback.format_exc())
        log_troubleshooting(logger, "unexpected")
        raise RuntimeError("Preprocessing failed") from e

def log_troubleshooting(logger: logging.Logger, error_type: str) -> None:
    """Centralized troubleshooting guides with logging integration."""
    guides = {
        "validation": [
            "1. Verify preprocessing script generates correct outputs",
            "2. Check file permissions in models/ directory",
            "3. Validate disk space is available",
            "4. Review preprocessing requirements documentation"
        ],
        "timeout": [
            "1. Optimize preprocessing steps",
            "2. Increase timeout_minutes parameter",
            "3. Check for infinite loops",
            "4. Profile script performance"
        ],
        "execution": [
            "1. Run preprocessing.py manually to debug",
            "2. Check dependency versions match requirements",
            "3. Validate input data quality",
            "4. Verify sufficient system resources"
        ],
        "unexpected": [
            "1. Check system resource limits",
            "2. Verify Python environment consistency",
            "3. Enable debug mode for details",
            "4. Contact support with full logs"
        ]
    }
    logger.warning("Troubleshooting steps:")
    for step in guides.get(error_type, guides["unexpected"]):
        logger.warning(f"  {step}")

def log_error_output(logger: logging.Logger, stderr: str, use_color: bool) -> None:
    """Log error output with proper formatting and truncation."""
    red = Fore.RED if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    logger.error(f"{red}=== Error Output ==={reset}")
    # Show last 20 lines
    for line in stderr.splitlines()[-20:]:
        # Truncate very long lines
        if len(line) > 200:
            logger.error(f"{red}{line[:200]}...{reset}")
        else:
            logger.error(f"{red}{line}{reset}")

def display_data_loading_header(filepath: str) -> None:
    """Display data loading header with rich formatting."""
    console.print("\n")
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
        table.add_column("Chunk #", justify="left", style="bold cyan", width=8)
        table.add_column("Processed", justify="left", style="bold magenta", width=12)
        table.add_column("Clean Samples", justify="left", style="bold white", width=16)
        table.add_column("Clean %", justify="left", width=10)
        table.add_column("Dtype Conv", justify="left",  style="bold white", width=10)
        table.add_column("Failed", justify="left", style="bold red", width=10)
        
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
        title_justify="left",
        show_lines=True
    )
    
    table.add_column("Metric", style="bold cyan", width=35)
    table.add_column("Count", style="bold magenta", justify="left")
    table.add_column("Impact", style="bold green", justify="left")
    
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
        title_justify="left",
        show_lines=True
    )
    
    table.add_column("Class", style="bold cyan")
    table.add_column("Count", style="bold magenta", justify="left")
    table.add_column("Percentage", style="bold green", justify="left")
    
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
        padding=(0, 1)
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
        title_justify="left",
        show_lines=True
    )
    
    table.add_column("Class", style="bold cyan", width=15)
    table.add_column("Original", style="bold magenta", justify="left")
    table.add_column("New Count", style="bold green", justify="left")
    table.add_column("Change", justify="left")
    
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
        padding=(0, 1)
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Original", style="bold magenta", justify="left")
    summary_table.add_column("New", style="bold green", justify="left")
    summary_table.add_column("Change", justify="left", style="bold")
    
    orig_total = original_counts.sum()
    new_total = new_counts.sum()
    change_total = new_total - orig_total
    change_pct_total = (change_total / orig_total) * 100
    
    summary_table.add_row(
        "[bold yellow]Total Samples",
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
            show_lines=True
            #padding=(0, 2)
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

def auto_select_oversampler(
    results: Dict[str, Dict],
    metric_weights: Optional[Dict[str, float]] = None,
    min_validation_acc: float = 0.7,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Automatically select the best oversampling method based on comprehensive evaluation metrics.
    
    Args:
        results: Dictionary containing evaluation results for each oversampling method
                Format: {method_name: {metric1: value, metric2: value, ...}}
        metric_weights: Optional dictionary to customize metric weighting
                      Default: {'val_acc': 0.5, 'boundary_violation_rate': 0.3, 'silhouette_score': 0.2}
        min_validation_acc: Minimum validation accuracy threshold (methods below this are filtered out)
        verbose: Whether to print scoring details
        
    Returns:
        Tuple of (best_method_name, scoring_details) where scoring_details contains:
        - method_scores: Individual scores for each method
        - metric_weights: Actual weights used
        - filtered_methods: Methods removed due to failing min_validation_acc
        
    Raises:
        ValueError: If no valid methods are available after filtering
        
    Selection Criteria (prioritized):
    1. Must meet minimum validation accuracy threshold
    2. Weighted combination of:
       - Validation accuracy (higher better)
       - Boundary violation rate (lower better) 
       - Silhouette score (higher better)
       - Feature correlation difference (lower better)
    """
    # Default metric weights if not provided
    default_weights = {
        'val_acc': 0.5,
        'boundary_violation_rate': -0.3,  # Negative because lower is better
        'silhouette_score': 0.15,
        'feature_correlation_diff': -0.05  # Negative because lower is better
    }
    weights = metric_weights or default_weights
    
    # Filter methods that meet minimum accuracy threshold
    valid_methods = {
        method: metrics 
        for method, metrics in results.items() 
        if not metrics.get('error') and metrics.get('val_acc', 0) >= min_validation_acc
    }
    
    if not valid_methods:
        raise ValueError(f"No methods met minimum validation accuracy of {min_validation_acc}")
    
    # Normalize metrics and calculate scores
    method_scores = {}
    metric_stats = {}
    
    # First pass to collect stats for normalization
    for metric in weights.keys():
        values = [m.get(metric, np.nan) for m in valid_methods.values()]
        metric_stats[metric] = {
            'min': np.nanmin(values),
            'max': np.nanmax(values),
            'mean': np.nanmean(values)
        }
    
    # Score calculation
    for method, metrics in valid_methods.items():
        score = 0
        details = {}
        
        for metric, weight in weights.items():
            raw_value = metrics.get(metric, metric_stats[metric]['mean'])
            
            # Handle NaN values by using metric average
            if np.isnan(raw_value):
                raw_value = metric_stats[metric]['mean']
                if verbose:
                    logger.warning(f"Using mean value for {metric} in {method} due to NaN")
            
            # Normalize value between 0-1 (except for negative weights)
            if weight > 0:  # Higher is better
                norm_value = ((raw_value - metric_stats[metric]['min']) / 
                             (metric_stats[metric]['max'] - metric_stats[metric]['min'] + 1e-10))
            else:  # Lower is better (negative weight)
                norm_value = 1 - ((raw_value - metric_stats[metric]['min']) / 
                                 (metric_stats[metric]['max'] - metric_stats[metric]['min'] + 1e-10))
            
            # Accumulate weighted score
            contribution = norm_value * abs(weight)
            score += contribution
            details[metric] = {
                'raw': raw_value,
                'normalized': norm_value,
                'weight': abs(weight),
                'contribution': contribution
            }
        
        method_scores[method] = {
            'total_score': score,
            'details': details
        }
    
    # Select method with highest score
    best_method = max(method_scores.items(), key=lambda x: x[1]['total_score'])[0]
    
    if verbose:
        # Print scoring breakdown
        console.print("\n[bold]Oversampler Selection Report[/bold]")
        table = Table(title="Method Scoring", box=box.ROUNDED)
        table.add_column("Method", style="cyan")
        table.add_column("Total Score", style="magenta")
        for metric in weights:
            table.add_column(metric, justify="right")
        
        for method, scores in method_scores.items():
            row = [method, f"{scores['total_score']:.3f}"]
            for metric in weights:
                row.append(f"{scores['details'][metric]['raw']:.3f}")
            table.add_row(*row)
        
        console.print(table)
        console.print(f"[bold green]Selected method: {best_method}[/bold green]")
    
    return best_method, {
        'method_scores': method_scores,
        'metric_weights': weights,
        'filtered_methods': set(results.keys()) - set(valid_methods.keys())
    }

def handle_class_imbalance(
    df: pd.DataFrame,
    artifacts: Dict,
    *,
    oversampler: str = "SMOTE",
    apply_smote: bool = True,
    imbalance_threshold: float = 10.0,
    label_col: str = "Label",
    sampling_strategy: Union[str, dict] = "auto",
    random_state: int = 42,
    n_jobs: int = -1,
    evaluate_quality: bool = True,
    visualize: bool = True,
    sample_metrics: Optional[int] = None,
    debug: bool = False
) -> pd.DataFrame:
    """Enhanced class imbalance handler with flexible controls.
    
    Args:
        df: Input DataFrame
        artifacts: Preprocessing artifacts dict
        oversampler: Type of oversampler (see available_samplers)
        apply_smote: Whether to apply oversampling
        imbalance_threshold: Ratio to consider imbalance
        label_col: Name of label column
        sampling_strategy: Oversampling strategy
        random_state: Random seed (None uses class default)
        evaluate_quality: Whether to calculate quality metrics
        visualize: Whether to generate visualizations
        sample_metrics: Number of samples to use for metrics (None for all)
        debug: Enable verbose debugging
        
    Returns:
        Balanced DataFrame if oversampling applied, else original
        
    Raises:
        ValueError: For invalid inputs or single-class data
        RuntimeError: For SMOTE application failures
    """
    # Validate inputs
    _validate_inputs(df, artifacts, label_col)
    
    # Get class distribution
    class_counts = df[label_col].value_counts()
    min_samples = class_counts.min()
    
    # Display initial analysis
    _display_initial_analysis(class_counts, imbalance_threshold)
    
    # Check if imbalance exceeds threshold
    if (class_counts.max() / min_samples) <= imbalance_threshold:
        console.print("[bold green]Class distribution within acceptable limits[/bold green]")
        return df
        
    if not apply_smote:
        console.print("[bold yellow]Oversampling not applied (apply_smote=False)[/bold yellow]")
        return df
        
    # Apply oversampling
    try:
        start_time = time.time()
        balanced_df, metrics = _apply_oversampling(
            df=df,
            artifacts=artifacts,
            oversampler=oversampler,
            label_col=label_col,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            evaluate_quality=evaluate_quality,
            visualize=visualize,
            sample_metrics=sample_metrics,
            min_samples=min_samples,
            n_jobs=n_jobs
        )
        
        # Report performance
        elapsed = time.time() - start_time
        _report_results(
            original_counts=class_counts,
            new_counts=balanced_df[label_col].value_counts(),
            metrics=metrics,
            elapsed_time=elapsed,
            sampler_name=oversampler
        )
        
        return balanced_df
        
    except Exception as e:
        logger.error(f"{oversampler} failed: {str(e)}")
        if debug:
            logger.exception("Oversampling error details:")
        raise RuntimeError("Class balancing failed") from e

def _validate_inputs(
    df: pd.DataFrame,
    artifacts: Dict,
    label_col: str
):
    """Validate input parameters."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")
        
    if not isinstance(artifacts, dict) or 'feature_names' not in artifacts:
        raise ValueError("Artifacts must be a dict containing 'feature_names'")
        
    missing_features = [f for f in artifacts['feature_names'] if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")

def _display_initial_analysis(
    class_counts: pd.Series,
    threshold: float
):
    """Display initial class distribution analysis."""
    table = Table(title="Class Distribution Analysis")
    table.add_column("Class")
    table.add_column("Count")
    table.add_column("Percentage")
    
    total = class_counts.sum()
    for cls, count in class_counts.items():
        table.add_row(
            str(cls),
            str(count),
            f"{count/total:.1%}"
        )
        
    console.print(table)
    
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > threshold:
        console.print(
            f"[bold yellow]Imbalance detected: {imbalance_ratio:.1f}:1 "
            f"(threshold: {threshold}:1)[/bold yellow]"
        )

def _apply_oversampling(
    df: pd.DataFrame,
    artifacts: Dict,
    oversampler: str,
    label_col: str,
    sampling_strategy: Union[str, dict],
    random_state: int,
    evaluate_quality: bool,
    visualize: bool,
    sample_metrics: Optional[int],
    min_samples: int,
    n_jobs: int = -1
) -> tuple[pd.DataFrame, Optional[Dict]]:
    """Apply oversampling and evaluate results."""
    # Get sampler with safe configuration
    sampler = _get_oversampler(
        X=df[artifacts['feature_names']].values,  # Add X parameter
        y=df[label_col].values,                   # Add y parameter
        method=oversampler,
        min_samples=min_samples,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    # Apply oversampling
    X_res, y_res = sampler.fit_resample(
        df[artifacts['feature_names']],
        df[label_col]
    )
    
    # Create balanced DataFrame
    balanced_df = pd.DataFrame(X_res, columns=artifacts['feature_names'])
    balanced_df[label_col] = y_res
    
    # Evaluate quality if enabled
    metrics = None
    if evaluate_quality:
        metrics = _evaluate_oversamplers(
            original=df[artifacts['feature_names']],
            resampled=X_res,
            labels=y_res,
            sample_size=sample_metrics,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        if visualize:
            _visualize_resampling(
                original=df[artifacts['feature_names']],
                resampled=X_res,
                labels=y_res,
                sampler_name=oversampler,
                show_plot=VIZ_CONFIG['interactive'],
                save_plot=True,
                progress_bar=True,
                max_samples=VIZ_CONFIG['max_samples'],
                dpi=VIZ_CONFIG['dpi'],
                projections=VIZ_CONFIG['projections']
            )
    
    return balanced_df, metrics

def _optimize_k_neighbors(
    X: np.ndarray,
    y: np.ndarray,
    max_k: int = 10,
    min_samples: int = 5,
    n_splits: int = 3,
    metric: str = 'silhouette',
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = False
) -> Tuple[int, Dict[str, Any]]:
    """
    Dynamically optimize the k_neighbors parameter for SMOTE using cross-validated evaluation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        max_k: Maximum k value to test (default: 10)
        min_samples: Minimum samples required in minority class (default: 5)
        n_splits: Number of cross-validation splits (default: 3)
        metric: Optimization metric ('silhouette', 'davies_bouldin', or 'both')
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (optimal_k, results_dict) where results_dict contains:
        - all_scores: Dictionary of scores for each k
        - best_score: Best score achieved
        - best_params: Parameters giving best score
        - cv_results: Full cross-validation results
        
    Raises:
        ValueError: If input data is invalid or metric is unknown
    """
    # Input validation
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    if max_k < 3:
        raise ValueError("max_k must be at least 3")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if metric not in ['silhouette', 'davies_bouldin', 'both']:
        raise ValueError("metric must be 'silhouette', 'davies_bouldin', or 'both'")
    
    # Get class distribution
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    minority_count = counts.min()
    
    # Determine maximum possible k
    actual_max_k = min(max_k, minority_count - 1)
    if actual_max_k < 3:
        if verbose:
            logger.warning(f"Minority class has only {minority_count} samples, using k={actual_max_k}")
        return actual_max_k, {'warning': 'minority class too small for optimization'}
    
    # Initialize results storage
    k_values = range(3, actual_max_k + 1)
    results = {
        'silhouette': {k: [] for k in k_values},
        'davies_bouldin': {k: [] for k in k_values},
        'failed_runs': 0
    }
    
    # Cross-validated evaluation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for k in k_values:
            try:
                # Apply SMOTE only to training fold
                smote = SMOTE(
                    k_neighbors=min(k, len(X_train) - 1),
                    random_state=random_state
                )
                X_res, y_res = smote.fit_resample(X_train, y_train)
                
                # Calculate metrics
                if metric in ['silhouette', 'both']:
                    sil_score = silhouette_score(X_res, y_res)
                    results['silhouette'][k].append(sil_score)
                
                if metric in ['davies_bouldin', 'both']:
                    db_score = davies_bouldin_score(X_res, y_res)
                    results['davies_bouldin'][k].append(db_score)
                
            except Exception as e:
                results['failed_runs'] += 1
                if verbose:
                    logger.warning(f"Failed evaluation for k={k}: {str(e)}")
    
    # Calculate mean scores
    mean_scores = {}
    if metric in ['silhouette', 'both']:
        mean_scores['silhouette'] = {
            k: np.mean(scores) if scores else -1
            for k, scores in results['silhouette'].items()
        }
    
    if metric in ['davies_bouldin', 'both']:
        mean_scores['davies_bouldin'] = {
            k: np.mean(scores) if scores else float('inf')
            for k, scores in results['davies_bouldin'].items()
        }
    
    # Determine best k based on selected metric(s)
    if metric == 'silhouette':
        best_k = max(mean_scores['silhouette'].items(), key=lambda x: x[1])[0]
        best_score = mean_scores['silhouette'][best_k]
    elif metric == 'davies_bouldin':
        best_k = min(mean_scores['davies_bouldin'].items(), key=lambda x: x[1])[0]
        best_score = mean_scores['davies_bouldin'][best_k]
    else:
        # 'both' - combined score
        # Normalize scores to [0,1] range
        sil_scores = np.array(list(mean_scores['silhouette'].values()))
        db_scores = np.array(list(mean_scores['davies_bouldin'].values()))
        
        norm_sil = (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min())
        norm_db = 1 - ((db_scores - db_scores.min()) / (db_scores.max() - db_scores.min()))
        
        combined_scores = {
            # Weighted combination
            k: 0.6 * norm_sil[i] + 0.4 * norm_db[i]
            for i, k in enumerate(k_values)
        }
        
        best_k = max(combined_scores.items(), key=lambda x: x[1])[0]
        best_score = combined_scores[best_k]
    
    # Prepare return dictionary
    result_details = {
        'all_scores': mean_scores,
        'best_score': best_score,
        'best_params': {'k_neighbors': best_k},
        'cv_results': results,
        'metric_used': metric,
        'n_splits': n_splits,
        'minority_class_size': minority_count,
        'actual_max_k_tested': actual_max_k
    }
    
    if verbose:
        logger.info(f"Optimal k_neighbors: {best_k} with score: {best_score:.3f}")
        logger.info(f"Tested k values: {list(k_values)}")
        if metric == 'both':
            logger.info("Used combined silhouette and Davies-Bouldin score")
    
    return best_k, result_details

def _get_oversampler(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    min_samples: int,
    sampling_strategy: str,
    random_state: int,
    auto_optimize: bool = False,
    optimize_params: Optional[Dict[str, Any]] = None
) -> Union[SMOTE, ADASYN, SMOTETomek, BorderlineSMOTE]:
    """
    Factory method for oversamplers with safe configurations and optional k_neighbors optimization.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        method: Oversampling method ('SMOTE', 'ADASYN', 'SMOTE+TOMEK', 'Borderline-SMOTE')
        min_samples: Minimum samples required in minority class
        sampling_strategy: Sampling strategy for oversampling
        random_state: Random seed for reproducibility
        auto_optimize: Whether to automatically optimize k_neighbors (default: False)
        optimize_params: Parameters for k_neighbors optimization (only used if auto_optimize=True)
        
    Returns:
        Configured oversampler instance
        
    Raises:
        ValueError: If invalid parameters or insufficient samples
    """
    # Set default optimization parameters if not provided
    if optimize_params is None:
        optimize_params = {}
    
    # Determine k_neighbors
    if auto_optimize:
        try:
            optimal_k, _ = _optimize_k_neighbors(
                X=X,
                y=y,
                min_samples=min_samples,
                random_state=random_state,
                **optimize_params
            )
            k_neighbors = optimal_k
        except Exception as e:
            if optimize_params.get('verbose', False):
                logger.warning(f"k_neighbors optimization failed, using fallback: {str(e)}")
            k_neighbors = min(3, min_samples - 1)
    else:
        k_neighbors = min(3, min_samples - 1)
    
    # Safety check
    if k_neighbors < 1:
        raise ValueError(
            f"Cannot apply oversampling - minority class has only {min_samples} samples"
        )
    
    # Configure samplers
    samplers = {
        "SMOTE": SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        ),
        "ADASYN": ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=k_neighbors,
            random_state=random_state
        ),
        "SMOTE+TOMEK": SMOTETomek(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state
            ),
            tomek=TomekLinks(),
            random_state=random_state
        ),
        "Borderline-SMOTE": BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            kind='borderline-1'
        )
    }
    
    if method not in samplers:
        raise ValueError(
            f"Unknown oversampler: {method}. "
            f"Choose from {list(samplers.keys())}"
        )
    
    return samplers[method]

def _calculate_feature_correlation_diff(
    original: pd.DataFrame, 
    synthetic: np.ndarray
) -> float:
    """
    Calculate mean absolute difference in feature correlation matrices between original and synthetic data.
    
    Args:
        original: DataFrame containing original features
        synthetic: Numpy array containing synthetic features
        
    Returns:
        Mean absolute difference between correlation matrices (lower is better)
        
    Notes:
        - Handles NaN values gracefully using np.nanmean
        - Transposes input matrices for correct correlation calculation
        - Returns float between 0 (perfect match) and 2 (complete mismatch)
    """
    # Calculate correlation matrices
    orig_corr = np.corrcoef(original.T)
    synth_corr = np.corrcoef(synthetic.T)
    
    # Calculate absolute differences and take mean, ignoring NaNs
    correlation_diff = np.nanmean(np.abs(orig_corr - synth_corr))
    
    return correlation_diff

def _evaluate_oversamplers(
    original: pd.DataFrame,
    resampled: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
    n_jobs: int = -1,
    sample_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive quality metrics for oversampled data.
    
    Args:
        original: Original feature DataFrame
        resampled: Oversampled feature matrix
        labels: Corresponding labels
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        sample_size: Optional size for subsampling large datasets
        
    Returns:
        Dictionary of quality metrics including:
        - neighbor_distance: Average distance to nearest neighbor
        - boundary_violation_rate: Fraction of samples near decision boundary  
        - distribution_divergence: KL divergence between distributions
        - silhouette_score: Cluster quality metric
        - imbalance_ratio: Ratio between majority/minority classes
        - feature_correlation_diff: New metric for correlation preservation
    """
    metrics = {}
    
    # Subsample if requested for large datasets
    if sample_size and len(resampled) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(resampled), sample_size, replace=False)
        resampled = resampled[idx]
        labels = labels[idx]
    
    # 1. Nearest Neighbor Analysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nbrs = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs).fit(resampled)
        distances, _ = nbrs.kneighbors(resampled)
        metrics['avg_neighbor_distance'] = np.mean(distances[:, 1])
        metrics['neighbor_std'] = np.std(distances[:, 1])
    
    # 2. Boundary Analysis
    try:
        clf = SVC(kernel='linear', random_state=random_state).fit(resampled, labels)
        metrics['boundary_violations'] = sum(clf.predict(resampled) != labels)
        metrics['boundary_violation_rate'] = metrics['boundary_violations'] / len(resampled)
    except Exception as e:
        logger.warning(f"Boundary analysis failed: {str(e)}")
        metrics['boundary_violations'] = float('nan')
        metrics['boundary_violation_rate'] = float('nan')
    
    # 3. Distribution Analysis
    bins = np.histogram_bin_edges(original.values.ravel(), bins='auto')
    p = np.histogram(original.values, bins=bins)[0] + 1e-10
    q = np.histogram(resampled, bins=bins)[0] + 1e-10
    metrics['distribution_divergence'] = entropy(p, q)
    
    # 4. Cluster Quality
    try:
        metrics['silhouette_score'] = silhouette_score(resampled, labels)
    except:
        metrics['silhouette_score'] = float('nan')
    
    # 5. Class Balance
    new_counts = np.bincount(labels)
    metrics['new_imbalance_ratio'] = new_counts.max() / new_counts.min()
    
    # 6. Feature Correlation Preservation (New Metric)
    metrics['feature_correlation_diff'] = _calculate_feature_correlation_diff(
        original,
        resampled
    )
    
    return metrics

def _visualize_resampling(
    original: pd.DataFrame,
    resampled: np.ndarray,
    labels: np.ndarray,
    sampler_name: str,
    random_state: int = 42,
    figsize: tuple = (18, 6),
    max_samples: int = 5000,
    dpi: int = 150,
    show_plot: bool = False,
    save_plot: bool = True,
    progress_bar: bool = True,
    projections: List[str] = ['pca', 'tsne'],
    interactive_backend: str = 'matplotlib',
    dimensions: int = 2
) -> Optional[Union[plt.Figure, 'plotly.graph_objs.Figure']]:
    """Generate comparative visualizations of resampling results with enhanced controls.
    
    Args:
        original: Original feature DataFrame
        resampled: Oversampled feature matrix
        labels: Corresponding labels
        sampler_name: Name of oversampling method used
        random_state: Random seed for reproducibility
        figsize: Figure dimensions (width, height) in inches
        max_samples: Maximum samples to use for visualization
        dpi: Image resolution for saved figure
        show_plot: Whether to display the plot interactively
        save_plot: Whether to save the plot to file
        progress_bar: Whether to show progress during sampling
        projections: List of projection methods to use ('pca', 'tsne', 'umap')
        interactive_backend: Visualization library to use ('matplotlib' or 'plotly')
        dimensions: Number of dimensions for visualization (2 or 3)
        
    Returns:
        Figure object if show_plot=True, None otherwise
        
    Raises:
        ValueError: If invalid input parameters are provided
        ImportError: If plotly is requested but not installed
    """
    # Validate dimensions parameter
    if dimensions not in (2, 3):
        raise ValueError("Dimensions must be 2 or 3")
    
    try:
        # Validate inputs
        if not isinstance(original, pd.DataFrame):
            raise ValueError("original must be a pandas DataFrame")
        if not isinstance(resampled, np.ndarray):
            raise ValueError("resampled must be a numpy array")
        if len(original) == 0 or len(resampled) == 0:
            raise ValueError("Input data cannot be empty")
            
        # Handle interactive backend
        if interactive_backend.lower() == 'plotly':
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError:
                logger.warning("Plotly not available, falling back to matplotlib")
                interactive_backend = 'matplotlib'

        # Sampling progress tracking
        if progress_bar:
            with Progress() as progress:
                task = progress.add_task("Preparing visualization...", total=2)
                
                # Sample original data if too large
                if len(original) > max_samples:
                    progress.update(task, description="Sampling original data")
                    rng = np.random.RandomState(random_state)
                    orig_idx = rng.choice(len(original), min(max_samples, len(original)), replace=False)
                    original = original.iloc[orig_idx]
                    progress.update(task, advance=1)
                
                # Sample resampled data if too large
                    progress.update(task, description="Sampling resampled data")
                    rng = np.random.RandomState(random_state)
                    res_idx = rng.choice(len(resampled), min(max_samples, len(resampled)), replace=False)
                    resampled = resampled[res_idx]
                    labels = labels[res_idx]
                    progress.update(task, advance=1)
        else:
            # Without progress bar
            if len(original) > max_samples:
                rng = np.random.RandomState(random_state)
                orig_idx = rng.choice(len(original), min(max_samples, len(original)), replace=False)
                original = original.iloc[orig_idx]
            
            if len(resampled) > max_samples:
                rng = np.random.RandomState(random_state)
                res_idx = rng.choice(len(resampled), min(max_samples, len(resampled)), replace=False)
                resampled = resampled[res_idx]
                labels = labels[res_idx]

        # Create visualizations based on backend
        if interactive_backend.lower() == 'plotly':
            return _plotly_visualization(
                original, resampled, labels, sampler_name,
                projections, show_plot, save_plot, dpi, dimensions
            )
        else:
            return _matplotlib_visualization(
                original, resampled, labels, sampler_name,
                figsize, dpi, show_plot, save_plot, projections,
                random_state, dimensions
            )
            
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise

def _matplotlib_visualization(
    original: pd.DataFrame,
    resampled: np.ndarray,
    labels: np.ndarray,
    sampler_name: str,
    figsize: tuple,
    dpi: int,
    show_plot: bool,
    save_plot: bool,
    projections: List[str],
    random_state: int,
    dimensions: int = 2
) -> Optional[plt.Figure]:
    """Generate matplotlib visualizations with 2D or 3D support."""
    try:
        if dimensions == 3:
            # Create figure with 2 columns (original and resampled)
            fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
            
            # Project data to 3D using PCA
            pca = PCA(n_components=3, random_state=random_state)
            orig_proj = pca.fit_transform(original.values)
            res_proj = pca.transform(resampled)
            
            # Original data plot
            ax1 = fig.add_subplot(121, projection='3d')
            sc1 = ax1.scatter(
                orig_proj[:, 0], orig_proj[:, 1], orig_proj[:, 2],
                alpha=0.5, label='Original'
            )
            ax1.set_title("Original Data (3D PCA)")
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            ax1.set_zlabel("PC3")
            
            # Resampled data plot
            ax2 = fig.add_subplot(122, projection='3d')
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax2.scatter(
                    res_proj[mask, 0], res_proj[mask, 1], res_proj[mask, 2],
                    color=colors(i),
                    label=str(label),
                    alpha=0.5
                )
            
            ax2.set_title(f"Resampled Data (3D PCA)\n{sampler_name}")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_zlabel("PC3")
            ax2.legend()
            
        else:
            # 2D visualization (original implementation)
            n_plots = len(projections) + 1
            fig = plt.figure(figsize=(figsize[0] * n_plots/3, figsize[1]), dpi=dpi)
            
            # Original data plot (always PCA)
            plt.subplot(1, n_plots, 1)
            _plot_projection(
                original.values, 
                labels=None,
                title="Original Data (PCA)",
                method='pca',
                random_state=random_state
            )
            
            # Resampled data plots
            for i, method in enumerate(projections, 2):
                plt.subplot(1, n_plots, i)
                _plot_projection(
                    resampled,
                    labels,
                    title=f"Resampled Data ({method.upper()})\n{sampler_name}",
                    method=method,
                    random_state=random_state
                )
        
        plt.tight_layout()
        
        if save_plot:
            dim_suffix = "3d" if dimensions == 3 else "2d"
            filename = f"oversampling_{sampler_name.lower().replace('+', '_')}_{dim_suffix}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
            logger.info(f"Saved visualization to {filename}")
        
        if show_plot:
            plt.show()
            return fig
        else:
            plt.close(fig)
            return None
            
    except Exception as e:
        logger.error(f"Matplotlib visualization failed: {str(e)}")
        if 'fig' in locals():
            plt.close(fig)
        raise

def _plotly_visualization(
    original: pd.DataFrame,
    resampled: np.ndarray,
    labels: np.ndarray,
    sampler_name: str,
    projections: List[str],
    show_plot: bool,
    save_plot: bool,
    dpi: int,
    dimensions: int = 2
) -> Optional['plotly.graph_objs.Figure']:
    """Generate interactive Plotly visualizations with 2D or 3D support."""
    try:
        if dimensions == 3:
            # 3D visualization with Plotly
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=["Original Data (3D PCA)", f"Resampled Data (3D PCA) - {sampler_name}"]
            )
            
            # Project data to 3D using PCA
            pca = PCA(n_components=3)
            orig_proj = pca.fit_transform(original.values)
            res_proj = pca.transform(resampled)
            
            # Original data trace
            fig.add_trace(
                go.Scatter3d(
                    x=orig_proj[:, 0],
                    y=orig_proj[:, 1],
                    z=orig_proj[:, 2],
                    mode='markers',
                    name='Original',
                    marker=dict(
                        size=3,
                        color='blue',
                        opacity=0.5
                    )
                ),
                row=1, col=1
            )
            
            # Resampled data traces by class
            unique_labels = np.unique(labels)
            colors = px.colors.qualitative.Plotly
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                fig.add_trace(
                    go.Scatter3d(
                        x=res_proj[mask, 0],
                        y=res_proj[mask, 1],
                        z=res_proj[mask, 2],
                        mode='markers',
                        name=str(label),
                        marker=dict(
                            size=3,
                            color=colors[i % len(colors)],
                            opacity=0.5
                        ),
                        showlegend=True
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title_text=f"Oversampling Comparison: {sampler_name}",
                width=1200,
                height=600
            )
            
        else:
            # Original 2D implementation
            n_cols = len(projections) + 1
            fig = make_subplots(
                rows=1, cols=n_cols,
                subplot_titles=["Original Data (PCA)"] + 
                [f"Resampled Data ({m.upper()})" for m in projections]
            )
            
            # Original data (PCA)
            pca = PCA(n_components=2).fit_transform(original.values)
            fig.add_trace(
                go.Scatter(
                    x=pca[:, 0], y=pca[:, 1],
                    mode='markers',
                    name='Original',
                    marker=dict(color='blue', opacity=0.5)
                ),
                row=1, col=1
            )
            
            # Resampled data projections
            for i, method in enumerate(projections, 2):
                if method == 'pca':
                    proj = PCA(n_components=2).fit_transform(resampled)
                elif method == 'tsne':
                    proj = TSNE(n_components=2).fit_transform(resampled)
                elif method == 'umap':
                    proj = UMAP(n_components=2).fit_transform(resampled)
                
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    fig.add_trace(
                        go.Scatter(
                            x=proj[mask, 0], y=proj[mask, 1],
                            mode='markers',
                            name=str(label),
                            marker=dict(opacity=0.5),
                            showlegend=(i == 2)
                        ),
                        row=1, col=i
                    )
            
            fig.update_layout(
                title_text=f"Oversampling Comparison: {sampler_name}",
                width=300 * n_cols,
                height=400
            )
        
        if save_plot:
            dim_suffix = "3d" if dimensions == 3 else "2d"
            filename = f"oversampling_{sampler_name.lower().replace('+', '_')}_{dim_suffix}.html"
            fig.write_html(filename)
            logger.info(f"Saved interactive visualization to {filename}")
        
        if show_plot:
            fig.show()
            return fig
        return None
        
    except Exception as e:
        logger.error(f"Plotly visualization failed: {str(e)}")
        raise

def _plot_projection(
    data: np.ndarray,
    labels: Optional[np.ndarray],
    title: str,
    method: str = 'pca',
    random_state: int = 42,
    alpha: float = 0.5,
    legend: bool = True
) -> None:
    """Project high-dim data to 2D for visualization with enhanced controls.
    
    Args:
        data: Input feature matrix
        labels: Target labels (None for unlabeled data)
        title: Plot title
        method: Projection method ('pca' or 'tsne')
        random_state: Random seed
        alpha: Point transparency
        legend: Whether to show legend
    """
    # Validate method
    if method not in ['pca', 'tsne']:
        raise ValueError(f"Invalid projection method: {method}. Choose 'pca' or 'tsne'")
    
    # Perform projection
    if method == 'pca':
        proj = PCA(n_components=2, random_state=random_state).fit_transform(data)
    else:
        # t-SNE
        proj = TSNE(n_components=2, random_state=random_state).fit_transform(data)
    
    # Create plot
    if labels is None:
        plt.scatter(proj[:, 0], proj[:, 1], alpha=alpha)
    else:
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                proj[mask, 0], proj[mask, 1],
                color=colors[i],
                label=str(label),
                alpha=alpha
            )
        # Only show legend for reasonable numbers
        if legend and len(unique_labels) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")

def _report_results(
    original_counts: pd.Series,
    new_counts: pd.Series,
    metrics: Dict[str, float],
    elapsed_time: float,
    sampler_name: str
):
    """Display comprehensive results report."""
    # Class distribution comparison
    table = Table(title=f"Oversampling Results ({sampler_name})")
    table.add_column("Class")
    table.add_column("Original Count")
    table.add_column("New Count")
    table.add_column("Change")
    
    for cls in original_counts.index:
        orig = original_counts[cls]
        new = new_counts.get(cls, 0)
        change = f"{(new-orig)/orig:+.1%}" if orig != 0 else "N/A"
        table.add_row(str(cls), str(orig), str(new), change)
    
    console.print(table)
    console.print(f"[italic]Processing time: {elapsed_time:.2f}s[/italic]")
    
    # Quality metrics if available
    if metrics:
        metric_table = Table(title="Quality Metrics")
        metric_table.add_column("Metric")
        metric_table.add_column("Value")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                metric_table.add_row(name, f"{value:.4f}")
            else:
                metric_table.add_row(name, str(value))
        
        console.print(metric_table)

def compare_oversamplers(
    df: pd.DataFrame,
    artifacts: Dict,
    methods: list = None,
    n_splits: int = 5,
    label_col: str = "Label",
    sampling_strategy: str = "auto",
    imbalance_threshold: float = 10.0,
    visualize: bool = True,
    random_state: int = 42,
    sample_metrics: int = None
) -> Dict[str, Dict]:
    """Comparative evaluation of multiple oversampling methods.
    
    Args:
        df: Input DataFrame
        artifacts: Preprocessing artifacts
        methods: List of oversamplers to compare (None for all)
        n_splits: Number of cross-validation splits
        label_col: Name of label column
        sampling_strategy: Oversampling strategy
        imbalance_threshold: Ratio to consider imbalance
        visualize: Whether to generate visualizations
        sample_metrics: Sample size for metrics calculation
        
    Returns:
        Dictionary of evaluation results for each method
    """
    available_samplers = ["SMOTE", "ADASYN", "SMOTE+TOMEK", "Borderline-SMOTE"]
    methods = methods or available_samplers
    results = defaultdict(dict)
    
    # Validate data first
    _validate_inputs(df, artifacts, label_col)
    
    # Check if imbalance exceeds threshold
    class_counts = df[label_col].value_counts()
    if (class_counts.max() / class_counts.min()) <= imbalance_threshold:
        console.print("[bold yellow]Data is balanced - skipping oversampling comparison[/bold yellow]")
        return dict(results)
    
    # Evaluate each method
    for method in track(methods, description="Evaluating oversamplers..."):
        try:
            fold_metrics = []
            fold_times = []
            
            # Cross-validation
            for fold in range(n_splits):
                start_time = time.time()
                
                # Apply oversampling with different random state for each fold
                balanced_df, metrics = _apply_oversampling(
                    df=df,
                    artifacts=artifacts,
                    oversampler=method,
                    label_col=label_col,
                    sampling_strategy=sampling_strategy,
                    random_state=random_state + fold,
                    evaluate_quality=True,
                    # Only visualize first fold
                    visualize=visualize and (fold == 0),
                    sample_metrics=sample_metrics,
                    min_samples=class_counts.min()
                )
                
                elapsed = time.time() - start_time
                fold_times.append(elapsed)
                if metrics:
                    fold_metrics.append(metrics)
            
            # Aggregate results
            if fold_metrics:
                avg_metrics = {
                    f"avg_{k}": np.nanmean([m.get(k, np.nan) for m in fold_metrics])
                    for k in fold_metrics[0].keys()
                }
                std_metrics = {
                    f"std_{k}": np.nanstd([m.get(k, np.nan) for m in fold_metrics])
                    for k in fold_metrics[0].keys()
                }
                
                results[method].update(avg_metrics)
                results[method].update(std_metrics)
                results[method]['avg_time'] = np.mean(fold_times)
                results[method]['std_time'] = np.std(fold_times)
            
        except Exception as e:
            logger.error(f"Evaluation failed for {method}: {str(e)}")
            results[method]['error'] = str(e)
    
    # Add statistical significance testing
    _add_statistical_tests(results, n_splits=n_splits)
    
    # Display comprehensive comparison
    _display_comparison(results)
    
    return dict(results)

def _add_statistical_tests(
    results: Dict[str, Dict],
    n_splits: int = 5,
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh'
) -> None:
    """Add statistical significance tests between methods with multiple comparison correction.
    
    Args:
        results: Dictionary containing evaluation results for each method
        n_splits: Number of cross-validation splits used
        alpha: Significance level
        correction_method: Multiple testing correction method (see statsmodels.stats.multitest)
        
    Modifies:
        The input results dictionary by adding:
        - p-values for each comparison
        - adjusted p-values
        - significance indicators
    """
    # Get all metrics that should be tested (average metrics excluding timing)
    metrics_to_test = [
        k for k in next(iter(results.values())).keys() 
        if k.startswith('avg_') and not k.endswith('_time')
    ]
    
    methods = list(results.keys())
    if len(methods) < 2:
        # Need at least 2 methods for comparison
        return
    
    # For each metric, compare all methods against the first one (baseline)
    for metric in metrics_to_test:
        # Collect all valid values for this metric across methods
        valid_methods = []
        values = []
        stds = []
        
        for method in methods:
            val = results[method].get(metric)
            if val is not None and not np.isnan(val):
                valid_methods.append(method)
                values.append(val)
                stds.append(max(results[method].get(f'std_{metric[len("avg_"):]}', 0.1), 0.01))
        
        if len(valid_methods) < 2:
            # Skip if not enough valid values
            continue
            
        baseline_method = valid_methods[0]
        baseline_value = values[0]
        baseline_std = stds[0]
        
        # Store all p-values for correction
        p_values = []
        comparisons = []
        
        # Compare each method to baseline
        for i, (method, value, std) in enumerate(zip(valid_methods, values, stds)):
            if i == 0:
                # Skip baseline
                # Don't compare baseline to itself
                continue
                
            # Perform t-test
            _, p = ttest_ind_from_stats(
                mean1=baseline_value,
                std1=baseline_std,
                nobs1=n_splits,
                mean2=value,
                std2=std,
                nobs2=n_splits
            )
            
            p_values.append(p)
            comparisons.append((method, baseline_method))
        
        # Apply multiple testing correction
        if p_values:
            reject, adj_pvals, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
            
            # Store results
            for (method, base), p, adj_p, is_sig in zip(comparisons, p_values, adj_pvals, reject):
                results[method][f"{metric}_p_value"] = p
                results[method][f"{metric}_adj_p_value"] = adj_p
                results[method][f"{metric}_significant"] = is_sig
                results[method][f"{metric}_baseline"] = base
                results[method]['p_value_correction'] = correction_method

def _display_comparison(
    results: Dict[str, Dict]
):
    """Display comprehensive comparison table."""
    if not results:
        console.print("[bold yellow]No valid results to display[/bold yellow]")
        return
        
    # Get all unique metrics
    all_metrics = set()
    for method_metrics in results.values():
        all_metrics.update(metric for metric in method_metrics if not metric.startswith('std_'))
    
    # Create table
    table = Table(title="Oversampler Performance Comparison")
    table.add_column("Method")
    
    # Add columns for each metric
    metrics_to_show = sorted(
        m for m in all_metrics 
        if not any(x in m for x in ['std_', 'p_value', 'error'])
    )
    
    for metric in metrics_to_show:
        table.add_column(metric.replace('avg_', ''), justify="right")
    
    # Add rows for each method
    for method, metrics in results.items():
        row = [method]
        for metric in metrics_to_show:
            value = metrics.get(metric)
            if value is None:
                row.append("N/A")
            elif isinstance(value, float):
                # Highlight statistically significant differences
                p_key = f"{metric}_p_value"
                p_value = metrics.get(p_key, 1)
                
                if p_value < 0.05:
                    style = "bold green" if "violation" not in metric else "bold red"
                    row.append(f"[{style}]{value:.4f}[/{style}]")
                else:
                    row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        
        table.add_row(*row)
    
    console.print(table)
    
    # Add footnote about statistical significance
    console.print(
        "[italic]Note: Bold values indicate statistically significant differences (p < 0.05) "
        "compared to the first method[/italic]"
    )

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

def create_synthetic_data(logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate realistic synthetic data as fallback with logging.
    
    Args:
        logger: Configured logger instance for logging messages
        
    Returns:
        Tuple containing:
        - DataFrame with synthetic features and labels
        - Dictionary of artifacts including feature names and metadata
        
    Raises:
        RuntimeError: If synthetic data generation fails
    """
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
        # Add noise
        X += np.random.normal(0, 0.05, X.shape)
        
        # Clip to [0,1] range
        X = np.clip(X, 0, 1)
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(num_features)]
        
        # Create DataFrame with proper typing
        df = pd.DataFrame(X, columns=feature_names)
        # Save memory with smaller dtype
        df['Label'] = y.astype('int8')
        
        # Log dataset statistics
        logger.info(f"Generated synthetic dataset with {num_samples} samples")
        logger.debug(f"Class distribution:\n{df['Label'].value_counts().to_string()}")
        logger.debug(f"Feature statistics:\n{df.describe().to_string()}")
        
        return (
            df,
            {
                "feature_names": feature_names,
                "scaler": None,
                # Random importances for compatibility
                "feature_importances": np.random.rand(num_features),
                # Default chunk size for compatibility
                "chunksize": 100000,
                # Flag indicating synthetic data
                "synthetic": True,
                # Default class names
                "class_names": ["Normal", "Attack"],
                # Track number of missing values (0 since we don't generate NaNs)
                "missing_values": 0,
                "data_quality": {
                    'has_nans': False,
                    'min_values': X.min(axis=0).tolist(),
                    'max_values': X.max(axis=0).tolist()
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
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
        logger.info(Fore.GREEN + Style.BRIGHT + "Initial class distribution:" + Fore.MAGENTA + Style.BRIGHT + f"{class_counts.tolist()}")
        
        # Threshold for extreme imbalance
        if torch.min(class_counts) < 1000:
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
                logger.info(Fore.WHITE + Style.BRIGHT + "Class distribution after SMOTE:" + Fore.YELLOW + Style.BRIGHT + f"{class_counts.tolist()}")
            except Exception as e:
                logger.error(Fore.RED + Style.BRIGHT + f"SMOTE failed: {str(e)}")
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
        
        logger.info(Fore.GREEN + Style.BRIGHT + f"Prepared dataloaders with " + Fore.MAGENTA + Style.BRIGHT + f"{len(train_dataset)}" + Fore.GREEN + Style.BRIGHT + " training and " + Fore.MAGENTA + Style.BRIGHT + f"{len(val_dataset)}" + Fore.GREEN + Style.BRIGHT + " validation samples")
        return train_loader, val_loader, X.shape[1], len(class_counts)
        
    except Exception as e:
        logger.error(f"Failed to prepare dataloaders: {str(e)}")
        raise RuntimeError("DataLoader preparation failed") from e

# Training and validation functions
class SecurityAwareLoss(nn.Module):
    """Loss function that penalizes false negatives more heavily for security applications"""
    def __init__(self, class_weights: torch.Tensor, false_negative_cost: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.fn_cost = false_negative_cost
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        
        # Apply extra penalty for false negatives (missed attacks)
        if self.fn_cost != 1.0:
            preds = torch.argmax(inputs, dim=1)
            # Actual attack but predicted normal
            fn_mask = (targets == 1) & (preds == 0)
            ce_loss[fn_mask] *= self.fn_cost
            
        return ce_loss.mean()

class WarmupScheduler:
    """Learning rate warmup scheduler"""
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 4,
    scaler: Optional[GradScaler] = None,
    warmup_scheduler: Optional[WarmupScheduler] = None
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
            
            # Forward pass (mixed precision disabled for stability)
            with autocast(enabled=False):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch) / grad_accum_steps
            
            # Backpropagation
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation and clipping
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    # Enhanced gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Enhanced gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                
                # Apply warmup if provided
                if warmup_scheduler:
                    warmup_scheduler.step()
                    
                optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * grad_accum_steps
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy
        
    except Exception as e:
        logger.error(f"Training failed at batch {batch_idx}: {str(e)}")
        raise RuntimeError("Training epoch failed") from e

def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = 'f2'
) -> Tuple[float, float]:
    """
    Find optimal decision threshold for security-focused metrics.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for positive class
        metric: Metric to optimize ('f2', 'recall', 'precision')
        
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        if metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    attack_threshold: float = 0.5,
    security_metrics: bool = True
) -> Dict[str, Any]:
    """
    Enhanced validation with security-focused metrics and threshold tuning.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Target device (cuda/cpu)
        class_names: Optional list of class names
        attack_threshold: Decision threshold for attack class (0-1)
        security_metrics: Whether to compute additional security metrics
        
    Returns:
        Dictionary containing:
        - Standard metrics (loss, accuracy, AUC)
        - Security metrics (recall, F2-score, confusion stats)
        - Threshold-adjusted predictions
        - Full probability distributions
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

    try:
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item()
                
                # Store all raw outputs
                all_logits.extend(outputs.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Threshold-adjusted predictions
        if attack_threshold != 0.5 and all_probs.shape[1] == 2:  # Binary case
            all_preds = (all_probs[:, 1] >= attack_threshold).astype(int)
        else:
            all_preds = np.argmax(all_probs, axis=1)

        # Base metrics
        val_loss = total_loss / len(loader)
        metrics = {
            'val_loss': val_loss,
            'val_acc': accuracy_score(all_labels, all_preds),
            'preds': all_preds,
            'labels': all_labels,
            'probs': all_probs,
            'logits': np.array(all_logits),
            'attack_threshold': attack_threshold
        }

        # Handle binary vs multiclass
        if len(np.unique(all_labels)) == 2:  # Binary classification
            metrics.update({
                'val_auc': roc_auc_score(all_labels, all_probs[:, 1]),
                'val_ap': average_precision_score(all_labels, all_probs[:, 1])
            })
            
            if security_metrics:
                tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
                metrics.update({
                    'recall': recall_score(all_labels, all_preds),
                    'precision': precision_score(all_labels, all_preds),
                    'f2_score': fbeta_score(all_labels, all_preds, beta=2),
                    'false_negatives': fn,
                    'false_positives': fp,
                    'true_positives': tp,
                    'true_negatives': tn,
                    'attack_detection_rate': tp / (tp + fn),
                    'false_alarm_rate': fp / (fp + tn)
                })
        else:  # Multiclass
            metrics.update({
                'val_auc': roc_auc_score(all_labels, all_probs, multi_class='ovr'),
                'val_ap': average_precision_score(all_labels, all_probs)
            })

        # Classification report
        if class_names:
            metrics['report'] = classification_report(
                all_labels, all_preds,
                target_names=class_names,
                digits=4,
                output_dict=True
            )

        return metrics

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise RuntimeError("Validation failed") from e

def visualize_data_distribution(
    df: pd.DataFrame,
    #log_dir: Path,
    filename: Path,
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
        logger.info(Fore.YELLOW + Style.BRIGHT + "\n=== Creating PCA visualization of data distribution ===")
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
        #plot_dir = Path(filename).absolute()
        plot_dir = Path("figures")
        plot_dir.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        plot_file = f"data_pca_distribution_{run_id}.png"
        plot_path = plot_dir / plot_file
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(Fore.GREEN + Style.BRIGHT + "Saved PCA visualization of data distribution " + Fore.MAGENTA +Style.BRIGHT + f"{plot_path}")
        return plot_path
        
    except Exception as e:
        logger.warning(Fore.RED + Style.BRIGHT + f"Could not create PCA visualization: {str(e)}")
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
    Save training checkpoint with verification and SMOTE metrics.
    
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
            },
            # Add SMOTE metrics
            'smote_metrics': {
                'oversampler': config.get('oversampler', 'SMOTE'),
                'k_neighbors': config.get('k_neighbors', 3),
                'feature_correlation_diff': metrics.get('feature_correlation_diff'),
                'neighbor_ratio': metrics.get('avg_neighbor_distance'),
                'minority_class_size': metrics.get('minority_class_size'),
                'synthetic_samples_generated': metrics.get('synthetic_samples_generated')
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
        
        logger.info(Fore.GREEN + f"Checkpoint saved successfully to \n{full_path} \n(checksum: {checksum})")
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
    # Define default metrics
    default_metrics = {
        'epoch': -1,
        'val_loss': float('inf'),
        'val_acc': 0.0,
        'val_auc': 0.0,
        'preds': np.array([]),
        'labels': np.array([]),
        'probs': np.array([])
    }
    returns = (None, None, None, default_metrics, {})

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
        checkpoint = torch.load(filename, weights_only=False)
        if model:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Convert lists back to numpy arrays
        metrics = checkpoint.get('metrics', {})
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
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Save all training artifacts including model, metrics, configuration, visualizations,
    and SMOTE evaluation metrics.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics (must contain 'labels' and 'preds' for confusion matrix)
        config: Training configuration
        class_names: Optional list of class names
        feature_names: Optional list of feature names
        output_dir: Optional custom output directory
        
    Returns:
        Dictionary of saved artifact paths, or empty dict if failed
    """
    saved_artifacts = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize training
    run_id = f"run_{timestamp}"
    
    # Use custom output directory if provided
    artifact_dir = output_dir if output_dir is not None else ARTIFACTS_DIR
    model_dir = output_dir if output_dir is not None else MODEL_DIR
    metrics_dir = output_dir if output_dir is not None else METRICS_DIR
    config_dir = output_dir if output_dir is not None else CONFIG_DIR
    info_dir = output_dir if output_dir is not None else INFO_DIR
    figure_dir = output_dir if output_dir is not None else FIGURE_DIR
    
    try:
        # 1. Ensure directories exist
        for dir_path in [model_dir, metrics_dir, config_dir, info_dir, artifact_dir, figure_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 2. Save model state dict
        #model_path = model_dir / f"ids_model_{timestamp}.pth"
        #model_path = model_dir / f"ids_model_{run_id}.pth"
        model_path = model_dir / f"ids_model.pth"
        torch.save(model.state_dict(), model_path)
        saved_artifacts['model'] = model_path

        # 3. Save metrics (convert numpy arrays to lists)
        #metrics_path = metrics_dir / f"ids_model_metrics_{timestamp}.json"
        metrics_path = metrics_dir / f"ids_model_metrics_{run_id}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }, f, indent=2)
        saved_artifacts['metrics'] = metrics_path

        # 4. Save SMOTE metrics separately if they exist
        if 'smote_metrics' in metrics or any(k in metrics for k in ['feature_correlation_diff', 'avg_neighbor_distance']):
            smote_metrics = {
                'oversampler': config.get('oversampler', 'SMOTE'),
                'k_neighbors': config.get('k_neighbors', 3),
                'feature_correlation_diff': metrics.get('feature_correlation_diff'),
                'neighbor_ratio': metrics.get('avg_neighbor_distance'),
                'minority_class_size': metrics.get('minority_class_size'),
                'synthetic_samples_generated': metrics.get('synthetic_samples_generated'),
                'timestamp': timestamp
            }
            #smote_metrics_path = metrics_dir / f"smote_evaluation_{timestamp}.json"
            smote_metrics_path = metrics_dir / f"smote_evaluation_{run_id}.json"
            with open(smote_metrics_path, 'w') as f:
                json.dump(smote_metrics, f, indent=2)
            saved_artifacts['smote_metrics'] = smote_metrics_path

        # 5. Save configuration
        #config_path = config_dir / f"ids_model_config_{timestamp}.json"
        config_path = config_dir / f"ids_model_config_{run_id}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        saved_artifacts['config'] = config_path

        # 6. Save additional info
        info = {
            'timestamp': timestamp,
            'class_names': class_names,
            'feature_names': feature_names,
            'environment': {
                'pytorch_version': torch.__version__,
                'python_version': platform.python_version(),
                'host': platform.node()
            },
            'smote_config': {
                'method': config.get('oversampler'),
                'k_neighbors': config.get('k_neighbors'),
                'sampling_strategy': config.get('sampling_strategy')
            }
        }
        #info_path = info_dir / f"ids_model_info_{timestamp}.json"
        info_path = info_dir / f"ids_model_info_{run_id}.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        saved_artifacts['info'] = info_path

        # 7. Save confusion matrix if labels and predictions exist
        if 'labels' in metrics and 'preds' in metrics:
            try:
                cm = confusion_matrix(metrics['labels'], metrics['preds'])
                plt.figure(figsize=(10, 8))
                
                if class_names:
                    tick_labels = class_names
                else:
                    tick_labels = sorted(set(metrics['labels']))
                
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=tick_labels,
                    yticklabels=tick_labels
                )
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                
                #cm_path = figure_dir / f"confusion_matrix_{timestamp}.png"
                cm_path = figure_dir / f"confusion_matrix_{run_id}.png"
                plt.savefig(cm_path, bbox_inches='tight', dpi=300)
                plt.close()
                saved_artifacts['confusion_matrix'] = cm_path
            except Exception as cm_error:
                logger.warning(f"Failed to save confusion matrix: {str(cm_error)}")

        # 8. Create archive of all artifacts
        #archive_path = artifact_dir / f"ids_model_artifacts_{timestamp}.tar.gz"
        archive_path = artifact_dir / f"ids_model_artifacts_{run_id}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for file in saved_artifacts.values():
                if isinstance(file, Path) and file.exists():
                    tar.add(file, arcname=file.name)
        
        saved_artifacts['archive'] = archive_path

        # Log success
        if saved_artifacts:
            artifacts_table = Table(
                title="[bold green]Training Artifacts Saved Successfully[/bold green]",
                box=box.ROUNDED,
                header_style="bold blue",
                title_style="bold green",
                title_justify="left",
                show_lines=True
            )
            
            artifacts_table.add_column("Artifact Type", style="bold cyan", width=20)
            artifacts_table.add_column("File Path", style="bold magenta")
            
            for artifact_type, path in saved_artifacts.items():
                display_path = str(path).replace("\\", "/")
                if "C:/Users" in display_path:
                    display_path = display_path.split("backend/")[-1]
                
                artifacts_table.add_row(
                    artifact_type.replace("_", " ").title(),
                    display_path
                )
            
            console.print()
            console.print(artifacts_table)
            
            success_panel = Panel(
                Text(f"All artifacts archived at: {saved_artifacts['archive']}", style="bold green"),
                border_style="green",
                expand=False
            )
            console.print(success_panel)
        
        return saved_artifacts

    except Exception as e:
        error_table = Table(
            title="[bold red]Artifact Saving Failed[/bold red]",
            box=box.ROUNDED,
            header_style="bold red",
            title_style="bold red",
            show_lines=True
        )
        error_table.add_column("Error Type", style="cyan")
        error_table.add_column("Details", style="magenta")
        error_table.add_row(type(e).__name__, str(e))
        
        console.print()
        console.print(error_table)
        
        for path in saved_artifacts.values():
            try:
                if isinstance(path, Path) and path.exists():
                    path.unlink()
            except:
                pass
        
        return {}

def banner() -> None:
    """Print banner"""

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
        subtitle="[magenta]MODEL TRAINING SUITE[/]",
        border_style="bold blue",
        box=box.DOUBLE,
        padding=(1, 2)
    ))

    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 40)
    print(Fore.GREEN + Style.BRIGHT + "  - Interactive Mode -  ".center(40))
    print(Fore.CYAN + Style.BRIGHT + "=" * 40 + Style.RESET_ALL)

def select_config_preset() -> Optional[Dict[str, Any]]:
    """Interactive preset selection with rich formatting"""
    console.print()
    preset_table = Table(
        title="[bold cyan]Available Configuration Presets[/]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        show_header=True,
        show_lines=True
    )
    
    preset_table.add_column("Option", style="bold yellow", width=8)
    preset_table.add_column("Preset", style="bold white", width=12)
    preset_table.add_column("Model", style="bold green", width=10)
    preset_table.add_column("Description", style="bold dim", width=40)
    
    preset_table.add_row("1", "stability", "simple", "Conservative settings for stable training")
    preset_table.add_row("2", "performance", "ensemble", "Optimized for best performance")
    preset_table.add_row("3", "baseline", "standard", "Standard configuration baseline")
    preset_table.add_row("4", "debug", "simple", "Fast training for debugging")
    preset_table.add_row("5", "custom", "varies", "Use current configuration")
    
    console.print(preset_table)
    
    try:
        choice = console.input("\n[bold cyan]Select preset (1-5): [/]")
        
        if choice == '1':
            return PRESET_CONFIGS['stability']
        elif choice == '2':
            return PRESET_CONFIGS['performance']
        elif choice == '3':
            return PRESET_CONFIGS['baseline']
        elif choice == '4':
            return PRESET_CONFIGS['debug']
        elif choice == '5':
            return None  # Use current config
        else:
            console.print(
                Panel.fit(
                    Text("Invalid selection. Using current configuration.", style="bold yellow"),
                    border_style="yellow"
                )
            )
            return None
            
    except KeyboardInterrupt:
        console.print("\n[bold red]Selection cancelled.[/]")
        return None

def run_stability_test(logger: logging.Logger, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a quick stability test with a simple model"""
    console.print()
    console.print(
        Panel.fit(
            Text("Running Stability Test", justify="center", style="bold cyan"),
            title="[bold yellow]Phase 1: Stability Check[/]",
            border_style="cyan",
            padding=(1, 2)
        )
    )
    
    # Use stability config if no config provided
    if config is None:
        config = PRESET_CONFIGS['stability']
    
    # Override some settings for quick test
    test_config = config.copy()
    test_config.update({
        'epochs': 10,
        'early_stopping': 5,
        'model_type': 'simple'
    })
    
    try:
        # Run training with test config
        results = train_model(logger, use_mock=True, config=test_config)
        
        # Analyze stability
        stability_metrics = {
            'completed': results.get('completed', False),
            'final_epoch': results.get('best_metrics', {}).get('epoch', -1),
            'best_val_loss': results.get('best_metrics', {}).get('val_loss', float('inf')),
            'training_stable': results.get('best_metrics', {}).get('val_loss', float('inf')) < 2.0,
            'config_used': test_config
        }
        
        # Display results
        if stability_metrics['training_stable']:
            console.print(
                Panel.fit(
                    Text("✓ Stability test PASSED", style="bold green"),
                    title="[bold green]Test Result[/]",
                    border_style="green"
                )
            )
        else:
            console.print(
                Panel.fit(
                    Text("✗ Stability test FAILED", style="bold red"),
                    title="[bold red]Test Result[/]",
                    border_style="red"
                )
            )
        
        return stability_metrics
        
    except Exception as e:
        logger.error(f"Stability test failed: {str(e)}")
        return {
            'completed': False,
            'error': str(e),
            'training_stable': False
        }

def progressive_training_pipeline(logger: logging.Logger) -> Dict[str, Any]:
    """Run progressive training: stability → baseline → performance"""
    console.print()
    console.print(
        Panel.fit(
            Text("Progressive Training Pipeline", justify="center", style="bold yellow"),
            subtitle="Phase 1: Stability → Phase 2: Baseline → Phase 3: Performance",
            border_style="yellow",
            padding=(1, 2)
        )
    )
    
    results = {}
    
    # Phase 1: Stability Test
    console.print("\n[bold cyan]Phase 1: Stability Test[/]")
    stability_result = run_stability_test(logger, PRESET_CONFIGS['stability'])
    results['stability'] = stability_result
    
    if not stability_result.get('training_stable', False):
        console.print(
            Panel.fit(
                Text("Stability test failed. Stopping pipeline.", style="bold red"),
                border_style="red"
            )
        )
        return results
    
    # Phase 2: Baseline Training
    console.print("\n[bold cyan]Phase 2: Baseline Training[/]")
    try:
        baseline_result = train_model(logger, use_mock=False, config=PRESET_CONFIGS['baseline'])
        results['baseline'] = baseline_result
        
        baseline_f2 = baseline_result.get('best_metrics', {}).get('val_f2', 0)
        if baseline_f2 < 0.7:  # Threshold for proceeding
            console.print(
                Panel.fit(
                    Text(f"Baseline F2-score too low ({baseline_f2:.3f}). Consider tuning.", style="bold yellow"),
                    border_style="yellow"
                )
            )
    except Exception as e:
        logger.error(f"Baseline training failed: {str(e)}")
        results['baseline'] = {'error': str(e)}
        return results
    
    # Phase 3: Performance Training (if baseline was good)
    if results['baseline'].get('best_metrics', {}).get('val_f2', 0) >= 0.7:
        console.print("\n[bold cyan]Phase 3: Performance Training[/]")
        try:
            performance_result = train_model(logger, use_mock=False, config=PRESET_CONFIGS['performance'])
            results['performance'] = performance_result
        except Exception as e:
            logger.error(f"Performance training failed: {str(e)}")
            results['performance'] = {'error': str(e)}
    
    return results

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent command injection"""
    return re.sub(r'[;&|$]', '', input_str).strip()

def print_menu() -> None:
    """Print enhanced menu options with new configuration and testing features"""
    print(Fore.YELLOW + Style.BRIGHT + "\nAvailable Options:")
    print(Fore.WHITE + Style.BRIGHT + "1. Configure System Settings")
    print(Fore.WHITE + Style.BRIGHT + "2. Setup Directories")
    print(Fore.WHITE + Style.BRIGHT + "3. Check Package Versions")
    print(Fore.WHITE + Style.BRIGHT + "4. Setup GPU/CPU")
    print(Fore.WHITE + Style.BRIGHT + "5. Enhanced Configuration Menu")
    print(Fore.WHITE + Style.BRIGHT + "6. Run Training Pipeline")
    print(Fore.WHITE + Style.BRIGHT + "7. Run Training with Synthetic Data")
    print(Fore.WHITE + Style.BRIGHT + "8. Progressive Training Pipeline")
    print(Fore.WHITE + Style.BRIGHT + "9. Quick Stability Test")
    print(Fore.WHITE + Style.BRIGHT + "10. Show Current Configuration")
    print(Fore.WHITE + Style.BRIGHT + "11. Compare Model Architectures")
    print(Fore.RED + Style.BRIGHT + "12. Exit")

def verify_model_classes():
    """Verify that all model classes are properly defined"""
    required_classes = ['IDSModel', 'SimpleIDSModel', 'StabilizedIDSModel', 'EnsembleIDSModel']
    missing_classes = []
    
    for class_name in required_classes:
        if class_name not in globals():
            missing_classes.append(class_name)
    
    if missing_classes:
        console.print(
            Panel.fit(
                Text(f"Missing model classes: {', '.join(missing_classes)}", style="bold red"),
                title="[bold red]Model Class Error[/]",
                border_style="red"
            )
        )
        return False
    
    return True

def enhanced_config_menu(logger: logging.Logger) -> None:
    """Enhanced configuration menu with presets and testing options"""
    while True:
        console.print()
        menu_table = Table(
            title="[bold cyan]Enhanced Configuration Menu[/]",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="blue"
        )
        
        menu_table.add_column("Option", style="bold yellow", width=8)
        menu_table.add_column("Action", style="bold white", width=25)
        menu_table.add_column("Description", style="bold dim")
        
        menu_table.add_row("1", "Show Current Config", "Display current settings")
        menu_table.add_row("2", "Configure Training", "Set training parameters")
        menu_table.add_row("3", "Configure Model", "Set model architecture")
        menu_table.add_row("4", "Select Preset", "Choose from predefined configs")
        menu_table.add_row("5", "Compare Models", "View model complexity comparison")
        menu_table.add_row("6", "Stability Test", "Quick stability check")
        menu_table.add_row("7", "Progressive Pipeline", "Run full training pipeline")
        menu_table.add_row("8", "Verify Models", "Check model class availability")  # NEW
        menu_table.add_row("0", "Return to Main Menu", "Go back")
        
        console.print(menu_table)
        
        choice = console.input("\n[bold cyan]Select option: [/]")
        
        if choice == '1':
            show_config()
        elif choice == '2':
            configure_training()
        elif choice == '3':
            configure_model()
        elif choice == '4':
            preset = select_config_preset()
            if preset:
                # Update global config with preset
                config = get_current_config()
                config = deep_update(config, {'training': preset, 'model': {'type': preset.get('model_type')}})
                save_config(config, CONFIG_DIR / "train_model_config.json", logger)
                update_global_config(config)
                console.print(
                    Panel.fit(
                        Text("Configuration updated with preset", style="bold green"),
                        border_style="green"
                    )
                )
        elif choice == '5':
            if verify_model_classes():
                display_model_comparison()
            else:
                console.print(
                    Panel.fit(
                        Text("Please ensure all model classes are defined before comparison", style="bold yellow"),
                        border_style="yellow"
                    )
                )
        elif choice == '6':
            run_stability_test(logger)
        elif choice == '7':
            progressive_training_pipeline(logger)
        elif choice == '8':
            verify_model_classes()
            initialize_model_variants()
            console.print(f"[green]Model verification complete. Available models: {list(MODEL_VARIANTS.keys())}[/]")
        elif choice == '0':
            break
        else:
            console.print(
                Panel.fit(
                    Text("Invalid option. Please try again.", style="bold red"),
                    border_style="red"
                )
            )

def interactive_main(
    logger: logging.Logger,
    device: torch.device,
    config: Dict[str, Any],
    directories: Dict[str, Path]
) -> None:
    """Enhanced interactive main function with new options"""
    # Use the initialized objects passed from main
    while True:
        banner()
        print_menu()
        choice = input(Fore.YELLOW + Style.BRIGHT + "\nSelect an option " + Fore.WHITE + Style.BRIGHT + "(1-12): ").strip()
        
        if choice == "1":
            print("\033c", end="")
            configure_system()
            print(Fore.GREEN + Style.BRIGHT + "System configuration applied")
            
        elif choice == "2":
            try:
                print("\033c", end="")
                configure_directories(logger)
                print(Fore.GREEN + Style.BRIGHT + "Directories set up successfully")
            except Exception as e:
                print(Fore.RED + Style.BRIGHT + f"Directory setup failed: {str(e)}")
                
        elif choice == "3":
            print("\033c", end="")
            check_versions(logger)
            
        elif choice == "4":
            print("\033c", end="")
            setup_gpu(logger)
            
        elif choice == "5":
            print("\033c", end="")
            enhanced_config_menu(logger)
            
        elif choice == "6":
            print("\033c", end="")
            
            # Offer preset selection before training
            console.print(
                Panel.fit(
                    Text("Select training configuration", justify="center", style="bold cyan"),
                    title="[bold yellow]Training Pipeline[/]",
                    border_style="cyan"
                )
            )
            
            preset = select_config_preset()
            if preset:
                print(Fore.YELLOW + Style.BRIGHT + "\nStarting training pipeline with selected preset...")
                train_config = preset
            else:
                print(Fore.YELLOW + Style.BRIGHT + "\nStarting training pipeline with current configuration...")
                train_config = None
                
            # Skip re-initializing logging if already set up
            try:
                train_model(logger, use_mock=False, config=train_config)
            except Exception as e:
                console.print(
                    Panel.fit(
                        Text(f"Training failed: {str(e)}", style="bold red"),
                        title="[bold red]Training Error[/]",
                        border_style="red"
                    )
                )
            
        elif choice == "7":
            print("\033c", end="")
            
            # Offer preset selection for synthetic training
            console.print(
                Panel.fit(
                    Text("Select configuration for synthetic data training", justify="center", style="bold cyan"),
                    title="[bold yellow]Synthetic Data Training[/]",
                    border_style="cyan"
                )
            )
            
            preset = select_config_preset()
            if preset:
                print(Fore.YELLOW + Style.BRIGHT + "\nStarting synthetic training with selected preset...")
                train_config = preset
            else:
                print(Fore.YELLOW + Style.BRIGHT + "\nStarting synthetic training with current configuration...")
                train_config = None
                
            # Skip re-initializing logging if already set up
            try:
                train_model(logger, use_mock=True, config=train_config)
            except Exception as e:
                console.print(
                    Panel.fit(
                        Text(f"Synthetic training failed: {str(e)}", style="bold red"),
                        title="[bold red]Training Error[/]",
                        border_style="red"
                    )
                )
            
        elif choice == "8":
            print("\033c", end="")
            console.print(
                Panel.fit(
                    Text("Starting Progressive Training Pipeline", justify="center", style="bold yellow"),
                    subtitle="This will run: Stability Test → Baseline → Performance",
                    border_style="yellow",
                    padding=(1, 2)
                )
            )
            
            # Confirm before starting long process
            proceed = console.input("\n[bold cyan]This may take a while. Continue? (y/n): [/]").lower()
            if proceed == 'y':
                if not logger.handlers:
                    logger = setup_logging(LOG_DIR)
                try:
                    results = progressive_training_pipeline(logger)
                    
                    # Display summary of results
                    console.print()
                    summary_table = Table(
                        title="[bold]Progressive Training Results[/]",
                        box=box.ROUNDED,
                        header_style="bold cyan"
                    )
                    summary_table.add_column("Phase", style="bold yellow")
                    summary_table.add_column("Status", style="bold white")
                    summary_table.add_column("Best F2", style="bold green")
                    
                    for phase, result in results.items():
                        if 'error' in result:
                            summary_table.add_row(phase.title(), "[red]Failed[/]", "[red]N/A[/]")
                        else:
                            f2_score = result.get('best_metrics', {}).get('val_f2', 0)
                            status = "[green]Completed[/]" if result.get('completed', False) else "[yellow]Partial[/]"
                            summary_table.add_row(phase.title(), status, f"{f2_score:.3f}")
                    
                    console.print(summary_table)
                except Exception as e:
                    console.print(
                        Panel.fit(
                            Text(f"Progressive training failed: {str(e)}", style="bold red"),
                            title="[bold red]Pipeline Error[/]",
                            border_style="red"
                        )
                    )
            else:
                console.print(
                    Panel.fit(
                        Text("Progressive training cancelled", style="bold yellow"),
                        border_style="yellow"
                    )
                )
                
        elif choice == "9":
            print("\033c", end="")
            console.print(
                Panel.fit(
                    Text("Running Quick Stability Test", justify="center", style="bold cyan"),
                    subtitle="10 epochs with simple model on synthetic data",
                    border_style="cyan"
                )
            )
            
            try:
                stability_result = run_stability_test(logger)
                
                # Show detailed results
                if stability_result.get('training_stable', False):
                    console.print(
                        Panel.fit(
                            Text(f"✓ Training completed successfully!\nFinal epoch: {stability_result['final_epoch']}\nBest validation loss: {stability_result['best_val_loss']:.4f}", 
                                 style="bold green"),
                            title="[bold green]Stability Test Results[/]",
                            border_style="green"
                        )
                    )
                else:
                    error_msg = stability_result.get('error', 'Training was unstable')
                    console.print(
                        Panel.fit(
                            Text(f"✗ Test failed: {error_msg}", style="bold red"),
                            title="[bold red]Stability Test Results[/]",
                            border_style="red"
                        )
                    )
            except Exception as e:
                console.print(
                    Panel.fit(
                        Text(f"Stability test failed: {str(e)}", style="bold red"),
                        title="[bold red]Test Error[/]",
                        border_style="red"
                    )
                )
                
        elif choice == "10":
            print("\033c", end="")
            try:
                show_config()
            except Exception as e:
                console.print(
                    Panel.fit(
                        Text(f"Failed to show config: {str(e)}", style="bold red"),
                        title="[bold red]Config Error[/]",
                        border_style="red"
                    )
                )
            
        elif choice == "11":
            print("\033c", end="")
            try:
                display_model_comparison()
                
                # Optionally offer to switch to a different model
                console.print()
                switch = console.input("[bold cyan]Switch to a different model architecture? (y/n): [/]").lower()
                if switch == 'y':
                    model_choice = console.input("[bold cyan]Enter model type (simple/standard/ensemble/stabilized): [/]").lower()
                    if model_choice in MODEL_VARIANTS:
                        # Update config
                        config = get_current_config()
                        config['model']['type'] = model_choice
                        save_config(config, CONFIG_DIR / "train_model_config.json", logger)
                        console.print(
                            Panel.fit(
                                Text(f"Model architecture changed to: {model_choice}", style="bold green"),
                                border_style="green"
                            )
                        )
                    else:
                        console.print(
                            Panel.fit(
                                Text("Invalid model type", style="bold red"),
                                border_style="red"
                            )
                        )
            except Exception as e:
                console.print(
                    Panel.fit(
                        Text(f"Model comparison failed: {str(e)}", style="bold red"),
                        title="[bold red]Comparison Error[/]",
                        border_style="red"
                    )
                )
                    
        elif choice == "12":
            print(Fore.RED + Style.BRIGHT + "\nExiting...")
            print(Fore.YELLOW + Style.BRIGHT + "Goodbye!")
            break
            
        else:
            print(Fore.RED + Style.BRIGHT + "Invalid selection. Choose 1-12.")
            
        # Add pause before returning to menu (fixed colorama usage)
        # Don't pause for long operations or exit
        if choice not in ["8", "12"]:
            input(Style.DIM + "\nPress Enter to continue..." + Style.RESET_ALL)

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

def display_training_summary(best_metrics: Dict[str, Any]) -> None:
    """Display training summary in a rich table."""
    summary_table = Table(
        title="[bold]Training Summary[/bold]",
        box=box.ROUNDED,
        header_style="bold blue",
        title_style="bold yellow",
        title_justify="left",
        show_header=True,
        show_lines=False
    )
    
    summary_table.add_column("Metric", style="bold cyan", width=20)
    summary_table.add_column("Value", style="bold magenta", justify="left")
    
    summary_table.add_row("Best Epoch", str(best_metrics['epoch'] + 1))
    summary_table.add_row("Validation Loss", f"{best_metrics['val_loss']:.4f}")
    summary_table.add_row("Validation Accuracy", f"{best_metrics['val_acc']:.2%}")
    summary_table.add_row("Validation AUC", f"{best_metrics['val_auc']:.4f}")
    
    console.print(summary_table)

def display_classification_report(labels: np.ndarray, preds: np.ndarray) -> None:
    """Display classification report in a rich table."""
    report = classification_report(
        labels,
        preds,
        target_names=['Normal', 'Attack'],
        output_dict=True,
        digits=4
    )
    
    # Main report table
    report_table = Table(
        title="[bold]Classification Report[/bold]",
        box=box.ROUNDED,
        header_style="bold blue",
        title_style="bold yellow",
        title_justify="left",
        show_header=True,
        show_lines=True
    )
    
    # Add columns
    report_table.add_column("Class", style="bold cyan", width=12)
    report_table.add_column("Precision", style="bold green", justify="left")
    report_table.add_column("Recall", style="bold green", justify="left")
    report_table.add_column("F1-Score", style="bold green", justify="left")
    report_table.add_column("Support", style="bold magenta", justify="left")
    
    # Add rows for each class
    for class_name in ['Normal', 'Attack']:
        metrics = report[class_name]
        report_table.add_row(
            class_name,
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1-score']:.4f}",
            str(metrics['support'])
        )
    
    # Add accuracy row
    report_table.add_row(
        "[bold]Accuracy[/bold]",
        "",
        "",
        f"{report['accuracy']:.4f}",
        str(report['macro avg']['support']),
        style="bold yellow"
    )
    
    # Add macro avg row
    report_table.add_row(
        "[bold]Macro Avg[/bold]",
        f"{report['macro avg']['precision']:.4f}",
        f"{report['macro avg']['recall']:.4f}",
        f"{report['macro avg']['f1-score']:.4f}",
        str(report['macro avg']['support']),
        style="bold blue"
    )
    
    console.print(report_table)

def log_epoch_progress(
    epoch: int,
    total_epochs: int,
    epoch_time: float,
    train_loss: float,
    train_acc: float,
    val_metrics: Dict[str, float],
    current_lr: float,
    patience_counter: int,
    early_stop_patience: int
) -> None:
    """Display epoch progress in a rich table format."""
    # Create the epoch table
    epoch_table = Table(
        title=f"[bold yellow]Epoch {epoch+1:03d}/{total_epochs:03d}[/bold yellow]",
        box=box.ROUNDED,
        show_header=False,
        show_lines=False,
        padding=(0, 1),
        min_width=50
    )
    
    # Add columns for metrics and values
    epoch_table.add_column("Metric", style="bold cyan", no_wrap=True, width=15)
    epoch_table.add_column("Value", style="bold magenta", justify="left")
    
    # Add rows for each metric with error handling
    try:
        epoch_table.add_row("Time", f"{epoch_time:.1f}s")
        epoch_table.add_row("Train Loss", f"{train_loss:.4f}")
        epoch_table.add_row("Val Loss", f"{val_metrics.get('val_loss', float('nan')):.4f}")
        epoch_table.add_row("Train Acc", f"{train_acc:.2%}")
        epoch_table.add_row("Val Acc", f"{val_metrics.get('val_acc', float('nan')):.2%}")
        epoch_table.add_row("Val AUC", f"{val_metrics.get('val_auc', float('nan')):.4f}")
        epoch_table.add_row("Learning Rate", f"{current_lr:.2e}")
        
        # Color-code patience counter
        patience_style = "red" if patience_counter >= early_stop_patience / 2 else "green"
        epoch_table.add_row(
            "Patience", 
            f"[{patience_style}]{patience_counter}[/{patience_style}]/[green]{early_stop_patience}[/green]"
        )
        
        # Print the table
        console.print(epoch_table)
        
    except Exception as e:
        logger.error(f"Error formatting epoch progress: {str(e)}")

def log_epoch_progress_enhanced(
    epoch: int,
    total_epochs: int,
    epoch_time: float,
    train_loss: float,
    train_acc: float,
    val_metrics: Dict[str, Any],
    current_lr: float,
    patience_counter: int,
    early_stop_patience: int,
    warmup_phase: bool = False
):
    """Enhanced logging with security metrics"""
    phase_indicator = "[WARMUP]" if warmup_phase else "[TRAIN]"
    
    logger.info(
        f"{phase_indicator} Epoch {epoch+1:3d}/{total_epochs} "
        f"({epoch_time:.1f}s) - "
        f"Loss: {train_loss:.4f} -> {val_metrics['val_loss']:.4f} | "
        f"Acc: {train_acc:.3f} -> {val_metrics['val_acc']:.3f} | "
        f"Recall: {val_metrics.get('val_recall', 0):.3f} | "
        f"F2: {val_metrics.get('val_f2', 0):.3f} | "
        f"LR: {current_lr:.2e} | "
        f"Patience: {patience_counter}/{early_stop_patience}"
    )
    
    if val_metrics.get('optimal_threshold'):
        logger.info(f"    Optimal attack threshold: {val_metrics['optimal_threshold']:.3f}")

def train_model(
    logger: logging.Logger,
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
    run_log_dir = LOG_DIR
    run_figure_dir = FIGURE_DIR
    run_checkpoint_dir = CHECKPOINT_DIR
    run_tb_dir = TB_DIR
    run_artifact_dir = ARTIFACTS_DIR
    
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
        #log_file = run_log_dir / f"training_{timestamp}.log"
        #logger = setup_logging(run_log_dir)
        #writer = SummaryWriter(log_dir=run_tb_dir)
        writer = SummaryWriter(log_dir=run_tb_dir, filename_suffix=f"_{run_id}")

        # Data preparation
        try:
            if use_mock:
                logger.info("Using synthetic data by request")
                df, artifacts = create_synthetic_data(logger)
                training_meta['data_source'] = 'synthetic'
            else:
                if not check_preprocessing_outputs(logger):
                    logger.warning("Preprocessing outputs not found")
                    if not run_preprocessing(logger):
                        raise DataPreparationError("Preprocessing failed")
                    logger.info("Preprocessing completed successfully")
                
                df, artifacts = load_and_validate_data()
                logger.info(Fore.GREEN + Style.BRIGHT + "Loaded " + Fore.YELLOW + Style.BRIGHT + f"{len(df)}" + Fore.GREEN + Style.BRIGHT + " validated samples")
                training_meta['data_source'] = 'real'
                training_meta['original_samples'] = len(df)

            # Handle class imbalance
            df = handle_class_imbalance(df, artifacts, apply_smote=True)
            training_meta['final_samples'] = len(df)
            
            # Visualize data
            #viz_path = visualize_data_distribution(df, filename=run_figure_dir / f"data_pca_distribution_{timestamp}.png")
            viz_path = visualize_data_distribution(df, filename=run_figure_dir / f"data_pca_distribution_{run_id}.png")
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
            logger.info(Fore.GREEN + Style.BRIGHT + "Data prepared - Input size: " + Fore.MAGENTA + Style.BRIGHT + f"{input_size}" + Fore.GREEN + Style.BRIGHT + ", Classes: " + Fore.MAGENTA + Style.BRIGHT + f"{num_classes}")

        except Exception as e:
            raise DataPreparationError(f"Data preparation failed: {str(e)}") from e

        # Model configuration
        try:
            model_type = config.get('model_type', 'standard') if config else 'standard'
            
            # Initialize MODEL_VARIANTS if it's empty or missing
            if not MODEL_VARIANTS:
                initialize_model_variants()
            
            model_class = MODEL_VARIANTS.get(model_type, MODEL_VARIANTS.get('simple', SimpleIDSModel))
            
            # Create model with enhanced configuration
            if model_type == 'standard':
                model = model_class(
                    input_size=input_size, 
                    output_size=num_classes,
                    use_batch_norm=config.get('use_batch_norm', True) if config else True,
                    dropout_rates=config.get('dropout_rates', DROPOUT_RATES) if config else DROPOUT_RATES
                ).to(device)
            elif model_type == 'simple':
                model = model_class(
                    input_size=input_size, 
                    output_size=num_classes,
                    dropout_rate=config.get('dropout_rate', 0.2) if config else 0.2
                ).to(device)
            elif model_type == 'ensemble':
                model = model_class(
                    input_size=input_size, 
                    output_size=num_classes,
                    num_models=config.get('num_ensemble_models', 3) if config else 3
                ).to(device)
            else:
                model = model_class(input_size, num_classes).to(device)
            
            logger.info(f"Using {model_type} model architecture")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Class weighting for security applications
            class_counts = torch.tensor(df['Label'].value_counts().sort_index().values, dtype=torch.float32)
            class_weights = (1. / class_counts) * (class_counts.sum() / num_classes)
            class_weights = class_weights / class_weights.sum()
            logger.info(Fore.GREEN + Style.BRIGHT + "Class weights: " + Fore.MAGENTA + Style.BRIGHT + f"{class_weights.tolist()}")
            training_meta['class_weights'] = class_weights.cpu().numpy().tolist()
            
            # Security-aware loss function
            criterion = SecurityAwareLoss(
                class_weights=class_weights.to(device),
                false_negative_cost=config.get('fn_cost', 2.0) if config else 2.0
            )
            
            # Optimizer with improved settings
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.get('learning_rate', LEARNING_RATE) if config else LEARNING_RATE,
                weight_decay=config.get('weight_decay', WEIGHT_DECAY) if config else WEIGHT_DECAY,
                # Better numerical stability
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
            # Learning rate warmup scheduler
            warmup_epochs = config.get('warmup_epochs', 5) if config else 5
            base_lr = config.get('learning_rate', LEARNING_RATE) if config else LEARNING_RATE
            warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, base_lr)
            
            # Main scheduler - Fix for PyTorch version compatibility
            try:
                # Try with verbose parameter first (older PyTorch versions)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    patience=config.get('lr_patience', 3) if config else 3,
                    factor=0.5,
                    verbose=True
                )
            except TypeError:
                # Fallback without verbose parameter (newer PyTorch versions)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    patience=config.get('lr_patience', 3) if config else 3,
                    factor=0.5
                )
                logger.info("Using ReduceLROnPlateau scheduler without verbose parameter (newer PyTorch)")
            
            # Disable mixed precision for CPU or if not available
            use_amp = torch.cuda.is_available() and config.get('mixed_precision', False) if config else False
            scaler = GradScaler(enabled=use_amp)
            
            if use_amp:
                logger.info("Using Automatic Mixed Precision (AMP)")
            else:
                logger.info("AMP disabled (CPU mode or not requested)")

        except Exception as e:
            raise ModelConfigurationError(f"Model setup failed: {str(e)}") from e

        # Training loop
        best_metrics = {
            'epoch': -1,
            'val_loss': float('inf'),
            'val_acc': 0.0,
            'val_auc': 0.0,
            # Security-focused metric
            'val_recall': 0.0,
            # Security-focused metric
            'val_f2': 0.0,
            'train_loss': float('inf'),
            'train_acc': 0.0,
            'learning_rate': 0.0,
            'optimal_threshold': 0.5
        }
        
        early_stop_patience = config.get('early_stopping', EARLY_STOPPING_PATIENCE) if config else EARLY_STOPPING_PATIENCE
        patience_counter = 0
        
        logger.info(Fore.YELLOW + Style.BRIGHT + "\n=== Starting Training with Security Enhancements ===")
        start_time = time.time()
        
        try:
            for epoch in range(config.get('epochs', DEFAULT_EPOCHS) if config else DEFAULT_EPOCHS):
                epoch_start = time.time()
                
                # Learning Rate Warmup Implementation
                if epoch < warmup_epochs:
                    # Gradual LR increase during warmup
                    lr_scale = (epoch + 1) / warmup_epochs
                    current_lr = lr_scale * base_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    
                    logger.info(f"[WARMUP] Epoch {epoch+1}/{warmup_epochs} | LR: {current_lr:.2e} (scale: {lr_scale:.3f})")
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                # Train epoch with enhanced stability
                train_loss, train_acc = train_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    grad_clip=config.get('gradient_clip', 1.0) if config else 1.0,
                    grad_accum_steps=config.get('grad_accum_steps', 1) if config else 1,
                    scaler=scaler,
                    warmup_scheduler=warmup_scheduler if epoch < warmup_epochs else None
                )
                
                # Check for training stability
                if not np.isfinite(train_loss):
                    logger.error(f"Training became unstable at epoch {epoch+1}: loss = {train_loss}")
                    break
                
                # Validate with security metrics
                val_metrics = validate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    class_names=['Normal', 'Attack'],
                    attack_threshold=0.5,
                    security_metrics=True
                )
                
                # Find optimal threshold for attack detection
                if 'probs' in val_metrics and 'labels' in val_metrics:
                    # Convert to numpy if needed
                    labels_np = val_metrics['labels']
                    probs_np = val_metrics['probs']
                    
                    if hasattr(labels_np, 'cpu'):
                        labels_np = labels_np.cpu().numpy()
                    if hasattr(probs_np, 'cpu'):
                        probs_np = probs_np.cpu().numpy()
                    
                    # For binary classification, use attack class probabilities
                    if len(probs_np.shape) > 1 and probs_np.shape[1] == 2:
                        attack_probs = probs_np[:, 1]  # Attack class probabilities
                    else:
                        attack_probs = probs_np
                    
                    optimal_threshold, f2_score = find_optimal_threshold(
                        labels_np, attack_probs, metric='f2'
                    )
                    val_metrics['optimal_threshold'] = optimal_threshold
                    val_metrics['val_f2'] = f2_score
                
                # Learning rate adjustment after warmup
                current_lr = optimizer.param_groups[0]['lr']
                if epoch >= warmup_epochs:
                    # Use F2 score for security-focused applications
                    metric_for_scheduler = val_metrics.get('val_f2', val_metrics['val_acc'])
                    scheduler.step(metric_for_scheduler)
                    
                    # Log LR changes
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != current_lr:
                        logger.info(f"Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
                
                # Update best metrics (prioritize security metrics)
                is_best = False
                if epoch < warmup_epochs:
                    # During warmup, use loss
                    if val_metrics['val_loss'] < best_metrics['val_loss']:
                        is_best = True
                else:
                    # After warmup, prioritize F2 score for security
                    if val_metrics.get('val_f2', 0) > best_metrics.get('val_f2', 0):
                        is_best = True
                
                if is_best:
                    best_metrics.update({
                        'epoch': epoch,
                        'val_loss': val_metrics['val_loss'],
                        'val_acc': val_metrics['val_acc'],
                        'val_auc': val_metrics.get('val_auc', 0.0),
                        'val_recall': val_metrics.get('val_recall', 0.0),
                        'val_f2': val_metrics.get('val_f2', 0.0),
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'learning_rate': current_lr,
                        'optimal_threshold': val_metrics.get('optimal_threshold', 0.5),
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
                        #filename=run_checkpoint_dir / f"best_model_{timestamp}.pth",
                        filename=run_checkpoint_dir / f"best_model_{run_id}.pth",
                        config=training_meta,
                    )
                else:
                    patience_counter += 1
                
                # Enhanced logging with security metrics
                epoch_time = time.time() - epoch_start
                
                log_epoch_progress_enhanced(
                    epoch=epoch,
                    total_epochs=config.get('epochs', DEFAULT_EPOCHS) if config else DEFAULT_EPOCHS,
                    epoch_time=epoch_time,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_metrics=val_metrics,
                    current_lr=current_lr,
                    patience_counter=patience_counter,
                    early_stop_patience=early_stop_patience,
                    warmup_phase=epoch < warmup_epochs
                )
                
                # Early stopping
                if patience_counter >= early_stop_patience:
                    console.print(
                        Panel.fit(
                            Text(f"Early stopping at epoch {epoch+1}", justify="center"),
                            title="[bold red]Training Stopped[/bold red]",
                            border_style="red",
                            padding=(1, 1)
                        )
                    )
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
            checkpoint_result = load_checkpoint(
                #filename=run_checkpoint_dir / f"best_model_{timestamp}.pth",
                filename=run_checkpoint_dir / f"best_model_{run_id}.pth",
                model=model,
                device=device
            )
            
            if checkpoint_result[0] is None:
                logger.warning("Could not load best model state")
            else:
                # Unpack the checkpoint result properly
                model_state, optim_state, scheduler_state, metrics, meta = checkpoint_result
                
                # Load the model state
                model.load_state_dict(model_state)
                
                # Load optimizer state if available and optimizer exists
                if optim_state is not None and optimizer is not None:
                    optimizer.load_state_dict(optim_state)
                    
                # Load scheduler state if available and scheduler exists
                if scheduler_state is not None and scheduler is not None:
                    scheduler.load_state_dict(scheduler_state)
                    
                logger.info("Successfully loaded best model checkpoint")
                
        except Exception as e:
            logger.error(f"Failed to load best model: {str(e)}")
            # Optionally add traceback for debugging
            logger.debug(f"Error details: {traceback.format_exc()}")

        # Generate reports
        console.print("\n")
        display_training_summary(best_metrics)

        if 'preds' in best_metrics and 'labels' in best_metrics:
            console.print("\n")
            display_classification_report(best_metrics['labels'], best_metrics['preds'])
            
            # Save confusion matrix
            cm = confusion_matrix(best_metrics['labels'], best_metrics['preds'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            #cm_path = run_figure_dir / f"confusion_matrix_{timestamp}.png"
            cm_path = run_figure_dir / f"confusion_matrix_{run_id}.png"
            plt.savefig(cm_path, bbox_inches='tight')
            plt.close()
            training_meta['confusion_matrix'] = str(cm_path)

        # Save final artifacts
        try:
            artifacts_saved = save_training_artifacts(
                model=model,
                metrics=best_metrics,
                config=training_meta,
                class_names=['Normal', 'Attack'],
                feature_names=artifacts.get('feature_names')
            )
            
            if not artifacts_saved:
                raise ModelSavingError("Failed to save some training artifacts")
                
        except Exception as e:
            raise ModelSavingError(f"Failed to save training artifacts: {str(e)}") from e

        writer.close()
        return {
            'completed': True,
            'best_metrics': best_metrics,
            'meta': training_meta,
            'artifacts_dir': str(run_artifact_dir)
        }

    except DataPreparationError as e:
        logger.error(f"Data preparation failed: {str(e)}")
        return {'completed': False, 'error': str(e)}
    except TrainingError as e:
        logger.error(f"Training failed: {str(e)}")
        return {'completed': False, 'error': str(e)}
    except ModelSavingError as e:
        logger.error(f"Model saving failed: {str(e)}")
        return {'completed': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        return {'completed': False, 'error': str(e)}

# Main entry point
if __name__ == "__main__":
    # Initialize styling for colored console output
    init(autoreset=True)
    console = Console()
    
    # Configure argument parser with enhanced help
    parser = argparse.ArgumentParser(
        description="Enhanced IDS Model Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Existing arguments
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
    
    # New SMOTE-related arguments
    parser.add_argument(
        "--compare-oversamplers",
        action="store_true",
        help="Run comparative evaluation of all oversampling methods"
    )
    parser.add_argument(
        "--oversampler",
        type=str,
        choices=["SMOTE", "ADASYN", "SMOTE+TOMEK", "Borderline-SMOTE"],
        default="SMOTE",
        help="Specify which oversampling method to use"
    )
    parser.add_argument(
        "--auto-optimize-k",
        action="store_true",
        help="Automatically find optimal k_neighbors value"
    )
    parser.add_argument(
        "--max-k-neighbors",
        type=int,
        default=5,
        help="Maximum k_neighbors value to test when auto-optimizing"
    )
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        help="Generate 3D visualizations of resampling results"
    )

    args = parser.parse_args()

    try:
        # Initial system configuration
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger, device, directories, config = initialize_system()

        # Handle SMOTE comparison mode
        if args.compare_oversamplers:
            try:
                logger.info(Fore.CYAN + Style.BRIGHT + "\n=== Oversampler Comparison Mode ===")
                df, artifacts = load_and_validate_data()
                results = compare_oversamplers(
                    df=df,
                    artifacts=artifacts,
                    methods=["SMOTE", "ADASYN", "SMOTE+TOMEK", "Borderline-SMOTE"],
                    # Disable visualization in debug mode
                    visualize=not args.debug,
                    random_state=config.get('random_state', 42)
                )
                best_method = auto_select_oversampler(results)
                logger.info(Fore.GREEN + Style.BRIGHT + f"\nRecommended oversampler: {best_method}")
                sys.exit(0)
            except Exception as e:
                logger.error(Fore.RED + Style.BRIGHT + f"Oversampler comparison failed: {str(e)}")
                sys.exit(1)

        if args.interactive:
            # Interactive mode
            try:
                interactive_main(logger, device, directories, config)
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
            
            # Enhanced configuration logging
            logger.info(Fore.MAGENTA + Style.BRIGHT + "Configuration:")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Batch size: {args.batch_size}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Epochs: {args.epochs}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Learning rate: {args.learning_rate}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Early stopping patience: {args.early_stopping}")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Oversampler: {args.oversampler}")
            if args.auto_optimize_k:
                logger.info(Fore.WHITE + Style.BRIGHT + 
                          f"  Auto k_neighbors optimization (max: {args.max_k_neighbors})")
            logger.info(Fore.WHITE + Style.BRIGHT + f"  Using {'synthetic' if args.use_mock else 'real'} data")
            
            # Prepare training configuration with SMOTE settings
            training_config = {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'early_stopping': args.early_stopping,
                'gradient_clip': GRADIENT_CLIP,
                'mixed_precision': MIXED_PRECISION,
                'oversampler': args.oversampler,
                'auto_optimize_k': args.auto_optimize_k,
                'max_k_neighbors': args.max_k_neighbors,
                'visualize_3d': args.visualize_3d
            }
            
            try:
                # Execute training
                results = train_model(
                    use_mock=args.use_mock,
                    config=training_config
                )
                
                # Enhanced final report
                logger.info(Fore.GREEN + Style.BRIGHT + "\n=== Training Completed Successfully ===")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Best validation accuracy: {results['metrics']['val_acc']:.2%}")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Best validation AUC: {results['metrics']['val_auc']:.4f}")
                if 'smote_metrics' in results['metrics']:
                    logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + 
                              f"SMOTE quality score: {results['metrics']['smote_metrics'].get('quality_score', 'N/A')}")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Artifacts saved to: {results['artifacts_dir']}")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Training time: {results['meta']['training_time']:.2f} seconds")
                
            except DataPreparationError as e:
                logger.error(Fore.RED + Style.BRIGHT + "\nData Preparation Failed:")
                logger.error(Fore.RED + Style.BRIGHT + f"Error: {str(e)}")
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
