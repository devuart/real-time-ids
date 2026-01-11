# Standard library imports
import argparse
import datetime
import time
import hashlib
import json
import logging
import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import platform
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
from enum import Enum, auto
import re
import traceback
import tarfile
import zipfile
import shutil
import contextlib
from contextlib import nullcontext
import psutil
from alive_progress import alive_bar
import threading
import select
import importlib
import gc

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
import importlib.metadata
from packaging import version as pkg_version
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
    calinski_harabasz_score,
)
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances, cosine_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from scipy.stats import entropy, ttest_ind, ttest_ind_from_stats, ks_2samp, gaussian_kde
from functools import partial
from multiprocessing import Pool, cpu_count
from statsmodels.stats.multitest import multipletests
#import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
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
def setup_safe_globals():
    """Setup safe globals for torch.load with version compatibility"""
    safe_classes = [
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
        
        # Basic numpy types that are version-stable
        np.ndarray,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.dtype,
        np.number,
        
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
    ]
    
    # Add numpy-version-specific classes safely
    try:
        # For newer numpy versions
        if hasattr(np, '_core') and hasattr(np._core, 'multiarray'):
            safe_classes.extend([
                np._core.multiarray._reconstruct,
                np._core.multiarray.scalar,
                np._core.multiarray.array,
            ])
        # For older numpy versions
        elif hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
            safe_classes.extend([
                np.core.multiarray._reconstruct,
                np.core.multiarray.scalar,
                np.core.multiarray.array,
            ])
    except AttributeError:
        # Skip numpy multiarray classes if not available
        pass
    
    # Add dtype classes safely
    try:
        if hasattr(np, 'dtypes'):
            safe_classes.append(np.dtypes.Float64DType)
    except AttributeError:
        pass
    
    # Apply safe globals
    add_safe_globals(safe_classes)

# Setup safe globals
setup_safe_globals()

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
    """Encapsulates the outcome of a system check."""
    
    def __init__(self, passed: bool, message: str, level: CheckLevel = CheckLevel.IMPORTANT, details: Optional[Union[str, Dict[str, Any]]] = None, metadata: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
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

def loading_screen(logger: logging.Logger) -> Tuple[bool, Optional[Dict[str, CheckResult]]]:
    """
    Display loading screen with system checks and interactive prompts.
    
    Args:
        logger: Logger for recording system check results
        
    Returns:
        Tuple[bool, Optional[Dict[str, CheckResult]]]: 
            - bool: True if all critical checks pass and user chooses to continue, False if critical checks fail or user chooses to quit
            - Optional[Dict[str, CheckResult]]: The check results if successful, None if failed
    """
    # Thread safety lock
    _loading_lock = threading.RLock()
    
    with _loading_lock:
        try:
            # Console safety checks
            if not hasattr(console, 'width'):
                # Safe default
                console_width = 80
            else:
                # Minimum width
                console_width = max(60, getattr(console, 'width', 80))
            
            # Terminal capability detection
            is_tty = sys.stdout.isatty()
            supports_color = is_tty and hasattr(console, 'is_terminal') and console.is_terminal
            
            # Safe console clear
            try:
                if is_tty and supports_color:
                    console.clear()
                else:
                    # Fallback for non-TTY
                    console.print("\n" * 3)
            except Exception:
                # Safe fallback
                console.print("\n" * 3)
            
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
                    
                    if is_tty and supports_color:
                        current_status = console.status(
                            f"[bold green]{message}[/bold green]",
                            spinner="dots"
                        )
                        current_status.start()
                        # Progressive timing
                        time.sleep(0.3 + i * 0.1)
                    else:
                        console.print(f"- {message}")
                        # Minimal delay for non-TTY
                        time.sleep(0.1)
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
                        style="bold cyan",
                        title="[bold yellow]GreyChamp | IDS[/bold yellow]",
                        subtitle="[bold magenta]SYSTEM INITIALIZATION[/bold magenta]",
                        border_style="bold cyan",
                        box=box.DOUBLE,
                        padding=(1, 1),
                        width=min(banner_width, console_width - 4)
                    ))
                else:
                    # Simple fallback for narrow terminals
                    console.print("\n" + "=" * min(60, banner_width))
                    console.print("\tGreyChamp | IDS - SYSTEM INITIALIZATION")
                    console.print("=" * min(60, banner_width) + "\n")
            except Exception as banner_error:
                # Ultra-safe fallback
                console.print("\nGreyChamp | IDS - SYSTEM INITIALIZATION\n")
                if logger:
                    logger.debug(f"Banner display failed: {banner_error}")
            
            # Check type information display
            check_type_info = "Basic System Checks"
            
            try:
                if supports_color and banner_width > 60:
                    console.print(Panel.fit(
                        f"Running {check_type_info}\n"
                        "Please wait while we validate your system...",
                        border_style="cyan",
                        style="bold cyan",
                        padding=(0, 2),
                        width=min(banner_width, console_width - 4)
                    ))
                else:
                    console.print(f"\nRunning {check_type_info}")
                    console.print("Please wait while we validate your system...\n")
            except Exception:
                console.print(f"\nRunning {check_type_info}\n")
            
            # Thread-safe system checks execution
            console.print("[bold green]Executing system checks...[/bold green]" if supports_color else "Executing system checks...")
            
            # Use thread-safe timing measurement
            checks_start = time.perf_counter()
            results = run_system_checks(logger)
            elapsed_time = time.perf_counter() - checks_start
            
            # Add execution time to summary if available
            if "system_summary" in results and results["system_summary"].details:
                if isinstance(results["system_summary"].details, dict):
                    results["system_summary"].details["execution_time"] = f"{elapsed_time:.2f}s"
            
            # Display results with error handling
            console.print()
            try:
                display_check_results(checks=results, logger=logger)
            except Exception as display_error:
                error_msg = f"[bold red]Error displaying results: {display_error}[/bold red]" if supports_color else f"Error displaying results: {display_error}"
                console.print(error_msg)
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
            critical_failed = 0
            important_failed = 0
            informational_failed = 0
            
            for name, result in results.items():
                if (result and hasattr(result, 'level') and hasattr(result, 'passed') and 
                    name not in ["system_summary", "system_error"]):
                    if not result.passed:
                        if result.level == CheckLevel.CRITICAL:
                            critical_failed += 1
                        elif result.level == CheckLevel.IMPORTANT:
                            important_failed += 1
                        elif result.level == CheckLevel.INFORMATIONAL:
                            informational_failed += 1
            
            # Handle different scenarios with proper cleanup
            return_value = False
            results_to_return = None
            
            try:
                if system_error or system_status == "CRITICAL_FAILURE" or critical_failed > 0:
                    # Critical failure - system cannot continue
                    failed_critical_checks = []
                    for name, result in results.items():
                        if (result and hasattr(result, 'level') and hasattr(result, 'passed') and
                            result.level == CheckLevel.CRITICAL and not result.passed and 
                            name not in ["system_summary", "system_error"]):
                            failed_critical_checks.append(name.replace("_", " ").title())
                    
                    error_message = (
                        f"CRITICAL SYSTEM CHECKS FAILED\n"
                        f"The system cannot continue due to critical failures.\n"
                        f"Failed checks: {', '.join(failed_critical_checks) if failed_critical_checks else 'System error occurred'}\n"
                        f"Please check the logs and resolve these issues before continuing."
                    )
                    
                    try:
                        if supports_color and banner_width > 60:
                            console.print(Panel.fit(
                                f"{error_message}",
                                border_style="red",
                                style="bold red",
                                padding=(1, 3),
                                width=min(banner_width, console_width - 4)
                            ))
                        else:
                            console.print(f"\n{error_message}")
                    except Exception:
                        console.print(f"\n{error_message}")
                    
                    if logger:
                        logger.critical(f"Critical system checks failed - cannot continue. Failed checks: {failed_critical_checks}")
                    
                    return_value = False
                    results_to_return = None
                    
                elif system_status in ["DEGRADED", "LIMITED"] or important_failed > 0 or informational_failed > 0:
                    # Non-critical failures - user decision with proper input handling
                    user_choice = _handle_user_decision_safe(
                        results, system_status, important_failed, informational_failed, 
                        elapsed_time, supports_color, banner_width, console_width, logger
                    )
                    
                    if user_choice is False:
                        return_value = False
                        results_to_return = None
                    else:
                        return_value = True
                        results_to_return = results
                        
                else:
                    # All checks passed - success scenario
                    return_value = _handle_success_scenario(
                        summary, elapsed_time, supports_color, banner_width, console_width, logger
                    )
                    results_to_return = results if return_value else None
            
            except Exception as scenario_error:
                error_msg = f"[bold red]Error handling system check results: {scenario_error}[/bold red]" if supports_color else f"Error handling system check results: {scenario_error}"
                console.print(error_msg)
                if logger:
                    logger.error(f"Error in scenario handling: {scenario_error}")
                return_value = False
                results_to_return = None
            
            # Safe console clear before return
            try:
                if return_value and is_tty and supports_color:
                    console.clear()
            except Exception:
                # Ignore clear failures
                pass
            
            return return_value, results_to_return
            
        except KeyboardInterrupt:
            # Thread-safe interrupt handling
            try:
                if supports_color:
                    console.print(Panel.fit(
                        "System initialization was interrupted by user (Ctrl+C).",
                        border_style="red",
                        title="INITIALIZATION INTERRUPTED",
                        padding=(1, 3)
                    ))
                else:
                    console.print("\nSystem initialization was interrupted by user (Ctrl+C).")
            except Exception:
                console.print("\nSystem initialization was cancelled by user (Ctrl+C).")
            
            if logger:
                logger.warning("System initialization interrupted by user (KeyboardInterrupt).")
            
            sys.exit(0)
            
        except Exception as e:
            # Thread-safe error handling
            error_msg = (
                f"SYSTEM ERROR\n"
                f"An unexpected error occurred during initialization: {str(e)}\n"
                f"Error details: {type(e).__name__}"
            )
            
            try:
                if supports_color:
                    console.print(Panel.fit(
                        f"{error_msg}",
                        border_style="red",
                        style="bold red",
                        padding=(1, 3)
                    ))
                else:
                    console.print(f"\n{error_msg}")
            except Exception:
                console.print(f"\n{error_msg}")
            
            if logger:
                logger.critical(f"Loading screen failed with unexpected error: {str(e)}", exc_info=True)
                logger.error(f"Error occurred during {check_type_info}")
            
            return False, None

def _handle_user_decision_safe(
    results: Dict[str, CheckResult],
    system_status: str,
    important_failed: int,
    informational_failed: int,
    elapsed_time: float,
    supports_color: bool,
    banner_width: int,
    console_width: int,
    logger: logging.Logger
) -> bool:
    """
    Thread-safe user decision handling with proper resource cleanup.
    
    Args:
        results: Dictionary of check results
        system_status: Current system status
        important_failed: Number of important check failures
        informational_failed: Number of informational check failures
        elapsed_time: Total execution time
        supports_color: Whether terminal supports color
        banner_width: Available banner width
        console_width: Console width
        logger: Logger for recording
        
    Returns:
        bool: True if user chooses to continue, False if user chooses to quit
    """
    
    try:
        # Collect failed non-critical checks safely
        failed_checks = []
        for name, result in results.items():
            if (result and hasattr(result, 'passed') and hasattr(result, 'level') and hasattr(result, 'message') and not result.passed and result.level in [CheckLevel.IMPORTANT, CheckLevel.INFORMATIONAL] and name not in ["system_summary", "system_error"]):
                failed_checks.append({
                    'name': name.replace("_", " ").title(),
                    'level': result.level.name,
                    'message': result.message.replace('[bold red]', '').replace('[/bold red]', '')
                                              .replace('[bold yellow]', '').replace('[/bold yellow]', '')
                                              .replace('[bold green]', '').replace('[/bold green]', '')
                })
        
        # Display failed checks summary safely
        if failed_checks:
            try:
                if supports_color and banner_width > 80:
                    fail_table = Table(
                        title="Failed Non-Critical System Checks",
                        box=box.SIMPLE,
                        header_style="bold magenta",
                        title_justify="left",
                        title_style="bold yellow",
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
                    print(Fore.YELLOW + Style.BRIGHT + "\nFailed Non-Critical Checks:" + Style.RESET_ALL)
                    for check in failed_checks:
                        level_indicator = "IMPORTANT" if check['level'] == "IMPORTANT" else "INFO"
                        console.print(f"  - {check['name']}: {check['message']} {level_indicator}")
            
            except Exception as table_error:
                # Ultra-safe fallback
                print(Fore.YELLOW + Style.BRIGHT + "\nSome non-critical checks failed:" + Style.RESET_ALL)
                for check in failed_checks:
                    console.print(f"  {check['name']}: {check['message']}")
                if logger:
                    logger.debug(f"Failed checks table display error: {table_error}")
        
        # Status display with safe formatting
        status_color = "yellow" if system_status == "DEGRADED" else "cyan" if system_status == "LIMITED" else "yellow"
        status_message = {
            "DEGRADED": "SYSTEM DEGRADED",
            "LIMITED": "LIMITED FUNCTIONALITY", 
        }.get(system_status, "SOME CHECKS FAILED")
        
        prompt_text = (
            f"{status_message}\n"
            f"System Status Details:\n"
            f"- Important failures: {important_failed}\n"
            f"- Informational failures: {informational_failed}\n"
            f"- Total execution time: {elapsed_time:.2f}s\n\n"
            f"The system can continue with reduced functionality."
        )
        try:
            if supports_color and banner_width > 60:
                console.print(Panel.fit(
                    f"[bold {status_color}]{prompt_text}[/bold {status_color}]",
                    border_style=status_color,
                    title="[bold cyan]User Decision Required[/bold cyan]",
                    padding=(1, 1),
                    width=min(banner_width, console_width - 4)
                ))
            else:
                console.print(f"\n{status_message}\n")
                console.print(prompt_text)
        except Exception:
            console.print(f"\n{status_message}\n")
            console.print(prompt_text)
        
        # Use standard input with proper handling
        user_choice = None
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                prompt = Fore.YELLOW + Style.BRIGHT + "\nContinue anyway? (Y/n/q): " + Style.RESET_ALL
                response = input(prompt).strip().lower()
                
                # Default to yes
                if response in ['y', 'yes', '']:
                    user_choice = True
                    break
                elif response in ['n', 'no', 'q', 'quit']:
                    user_choice = False
                    break
                else:
                    if attempt < max_attempts - 1:
                        print(Fore.YELLOW + Style.BRIGHT + "\nPlease enter 'y' for yes or 'n' for no or 'q' for quit." + Style.RESET_ALL)
            
            except (EOFError, KeyboardInterrupt):
                user_choice = False
                break
            except Exception as input_error:
                if logger:
                    logger.debug(f"Input error on attempt {attempt + 1}: {input_error}")
                if attempt < max_attempts - 1:
                    print(Fore.RED + Style.BRIGHT + "\nInput error, please try again." + Style.RESET_ALL)
        
        # Default to continue if no valid choice after max attempts
        if user_choice is None:
            user_choice = True
            print(Fore.CYAN + Style.BRIGHT + "\nUsing default choice: continue" + Style.RESET_ALL)
        
        # Handle user choice with safe output
        if user_choice is False:
            try:
                cancel_message = (
                    "USER CANCELLED INITIALIZATION\n"
                    "User chose to quit and resolve the issues.\n"
                    f"Failed checks summary: {len(failed_checks)} non-critical failures\n"
                    "Please check the logs and fix the failed checks."
                )
                if supports_color and banner_width > 60:
                    console.print(Panel.fit(
                        f"{cancel_message}",
                        border_style="red",
                        style="bold red",
                        width=min(banner_width, console_width - 4)
                    ))
                else:
                    print(Fore.RED + Style.BRIGHT + f"\n{cancel_message}" + Style.RESET_ALL)
            except Exception:
                print(Fore.RED + Style.BRIGHT + f"\n{cancel_message}" + Style.RESET_ALL)
            
            print(Fore.RED + Style.BRIGHT + "\nExiting system initialization..." + Style.RESET_ALL)
            
            # Give user time to read the message
            time.sleep(2)
            sys.exit(0)
            
            return False
        
        # User chose to continue
        try:
            continue_message = (
                "CONTINUING WITH WARNINGS\n"
                f"System status: {system_status} with {len(failed_checks)} failed checks\n"
                "User chose to continue despite the warnings.\n"
                "Some functionality may be limited."
            )
            if supports_color and banner_width > 60:
                console.print(Panel.fit(
                    f"[bold yellow]{continue_message}[/bold yellow]",
                    border_style="green",
                    style="bold green",
                    width=min(banner_width, console_width - 4)
                ))
            else:
                print(Fore.GREEN + Style.BRIGHT + f"\n{continue_message}" + Style.RESET_ALL)
        except Exception:
            print(Fore.GREEN + Style.BRIGHT + f"\n{continue_message}" + Style.RESET_ALL)
        
        # Give user time to read the message
        time.sleep(2)
        
        return True
        
    except Exception as decision_error:
        logger.error(f"Error in user decision handling: {decision_error}")
        print(Fore.RED + Style.BRIGHT + f"\nError in user input - continuing with warnings: {decision_error}" + Style.RESET_ALL)
        
        # Default to continue on error
        return True
    
    finally:
        # Clean up input buffer - Windows-safe version
        try:
            # Small delay for any pending I/O
            time.sleep(0.05)
            
            # Flush streams
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Windows-specific input buffer clearing
            if sys.platform == 'win32':
                try:
                    import msvcrt
                    while msvcrt.kbhit():
                        msvcrt.getch()
                except Exception:
                    pass
            
            # Final flush
            sys.stdin.flush()
            
        except Exception as cleanup_error:
            # Silently ignore cleanup errors - they're not critical
            logger.debug(f"Input buffer cleanup failed (non-critical): {cleanup_error}")

def _handle_success_scenario(
    summary: Optional[CheckResult],
    elapsed_time: float,
    supports_color: bool,
    banner_width: int,
    console_width: int,
    logger: logging.Logger
) -> bool:
    """
    Handle successful system checks scenario with safe input.
    
    Args:
        summary: System summary check result
        elapsed_time: Total execution time
        supports_color: Whether terminal supports color
        banner_width: Available banner width
        console_width: Console width
        logger: Logger for recording
        
    Returns:
        bool: True if user chooses to continue, False if user chooses to quit
    """
    try:
        success_details = ""
        elapsed_time_details = f"{elapsed_time:.2f}"
        if summary and summary.details and isinstance(summary.details, dict):
            total_checks = summary.details.get('total_checks', 0)
            if supports_color:
                success_details = f"[bold yellow]{total_checks}[/bold yellow]"
                elapsed_time_details = f"[bold yellow]{elapsed_time:.2f}[/bold yellow]"
            else:
                success_details = str(total_checks)
        
        success_message = (
            f"SUCCESS! ALL SYSTEM CHECKS PASSED\n"
            f"System is fully operational and ready!\n"
            f"Completed {success_details} checks successfully.\n"
            f"Completed in {elapsed_time_details} seconds.\n\n"
            f"Continue to system? (Y/n)"
        )
        
        try:
            if supports_color and banner_width > 60:
                console.print(Panel.fit(
                    f"{success_message}",
                    border_style="green",
                    style="bold green",
                    padding=(1, 3),
                    width=min(banner_width, console_width - 4)
                ))
            else:
                print(Fore.GREEN + Style.BRIGHT + f"\n{success_message}" + Style.RESET_ALL)
        except Exception:
            print(Fore.GREEN + Style.BRIGHT + f"\n{success_message}" + Style.RESET_ALL)
        
        # Handle user choice with safe input
        user_choice = None
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                response = input(Fore.YELLOW + Style.BRIGHT + "\nYour choice: " + Style.RESET_ALL).strip().lower()
                
                # Default to yes (continue)
                if response in ['y', 'yes', '']:
                    user_choice = True
                    break
                elif response in ['n', 'no', 'q', 'quit']:
                    user_choice = False
                    break
                else:
                    if attempt < max_attempts - 1:
                        print(Fore.YELLOW + Style.BRIGHT + "\nPlease enter 'y' for yes or 'n' for no." + Style.RESET_ALL)
            
            except (EOFError, KeyboardInterrupt):
                user_choice = False
                break
            except Exception as input_error:
                if logger:
                    logger.debug(f"Input error on attempt {attempt + 1}: {input_error}")
                if attempt < max_attempts - 1:
                    print(Fore.RED + Style.BRIGHT + "\nInput error, please try again." + Style.RESET_ALL)
        
        # Default to continue if no valid choice after max attempts
        if user_choice is None:
            user_choice = True
            print(Fore.CYAN + Style.BRIGHT + "\nUsing default choice: continue" + Style.RESET_ALL)
        
        # Handle user choice
        if user_choice is False:
            try:
                quit_message = (
                    "USER CHOSE TO QUIT\n"
                    "You chose to quit despite all checks passing.\n"
                    "System initialization cancelled."
                )
                if supports_color and banner_width > 60:
                    console.print(Panel.fit(
                        f"{quit_message}",
                        border_style="red",
                        style="bold red",
                        padding=(1, 3),
                        width=min(banner_width, console_width - 4)
                    ))
                else:
                    print(Fore.RED + Style.BRIGHT + f"\n{quit_message}" + Style.RESET_ALL)
            except Exception:
                print(Fore.RED + Style.BRIGHT + f"\n{quit_message}" + Style.RESET_ALL)
            
            if logger:
                logger.debug("User chose to quit after successful system checks")
            
            print(Fore.RED + Style.BRIGHT + "\nExiting system initialization..." + Style.RESET_ALL)
            
            time.sleep(2)
            sys.exit(0)
        
        # User chose to continue
        try:
            continue_message = (
                "PROCEEDING TO SYSTEM\n"
                f"All checks passed successfully in {elapsed_time:.2f}s.\n"
                "Proceeding to main system."
            )
            
            if supports_color and banner_width > 60:
                console.print(Panel.fit(
                    f"{continue_message}",
                    border_style="green",
                    style="bold green",
                    padding=(1, 2),
                    width=min(banner_width, console_width - 4)
                ))
            else:
                print(Fore.GREEN + Style.BRIGHT + f"\n{continue_message}" + Style.RESET_ALL)
        except Exception:
            print(Fore.GREEN + Style.BRIGHT + f"\n{continue_message}" + Style.RESET_ALL)
        
        if logger:
            logger.debug(f"All system checks passed successfully in {elapsed_time:.2f}s - user chose to continue")
            if summary and summary.details:
                system_status = summary.details.get('system_status', 'OPTIMAL')
                logger.info(f"System status: {system_status}")
        
        # Give user time to read the message
        time.sleep(2)
        
        return True
    
    except Exception as success_error:
        if logger:
            logger.error(f"Error in success scenario: {success_error}")
        print(Fore.GREEN + Style.BRIGHT + f"\nAll checks passed - continuing..." + Style.RESET_ALL)
        return True
    
    finally:
        # Clean up input buffer - Windows-safe version
        try:
            # Small delay for any pending I/O
            time.sleep(0.05)
            
            # Flush streams
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Windows-specific input buffer clearing
            if sys.platform == 'win32':
                try:
                    import msvcrt
                    while msvcrt.kbhit():
                        msvcrt.getch()
                except Exception:
                    pass
            
            # Final flush
            sys.stdin.flush()
        
        except Exception as cleanup_error:
            # Silently ignore cleanup errors - they're not critical
            logger.debug(f"Input buffer cleanup failed (non-critical): {cleanup_error}")

def run_system_checks(logger: logging.Logger) -> Dict[str, CheckResult]:
    """
    Run all system checks and compile results.
    
    Args:
        logger: Configured logger for recording check results
        
    Returns:
        Dictionary mapping check names to their CheckResult objects
    """
    checks: Dict[str, CheckResult] = {}
    
    try:
        # Run all system checks in order of importance
        raw_checks = {
            # Critical checks (essential for operation)
            'python_version': check_python_version(),
            'torch_available': check_torch(),
            
            # Important checks (affects functionality but not critical)
            'cuda_available': check_cuda(),
            'package_versions': check_package_versions_wrapper(logger),
            'directory_access': check_directory_access_wrapper(logger),
            'disk_space': check_disk_space(),
            
            # Informational checks (diagnostic purposes)
            'cpu_cores': check_cpu_cores(),
            'system_ram': check_system_ram(),
            'system_architecture': check_system_architecture(),
            'logging_setup': check_logging_setup(logger),
            'seed_config': check_seed_config()
        }
        
        # Process all results
        for name, result in raw_checks.items():
            if isinstance(result, CheckResult):
                checks[name] = result
            else:
                # Handle unexpected result types
                checks[name] = CheckResult(
                    passed=False,
                    message=f"Invalid result type for {name}: {type(result)}",
                    level=CheckLevel.CRITICAL,
                    details={
                        'raw_result': str(result), 
                        'result_type': str(type(result)),
                        'error': 'Result was not a CheckResult instance'
                    }
                )
        
        # Calculate overall system status
        critical_checks = [result for result in checks.values() if result.level == CheckLevel.CRITICAL]
        important_checks = [result for result in checks.values() if result.level == CheckLevel.IMPORTANT]
        
        # Check if all critical checks passed
        critical_passed = all(result.passed for result in critical_checks)
        important_passed = all(result.passed for result in important_checks)
        
        # Count failures
        critical_failures = sum(1 for r in checks.values() if not r.passed and r.level == CheckLevel.CRITICAL)
        important_failures = sum(1 for r in checks.values() if not r.passed and r.level == CheckLevel.IMPORTANT)
        informational_failures = sum(1 for r in checks.values() if not r.passed and r.level == CheckLevel.INFORMATIONAL)
        
        # Determine system status
        if not critical_passed:
            system_status = "CRITICAL_FAILURE"
            overall_passed = False
        elif not important_passed:
            system_status = "DEGRADED"
            overall_passed = False  # Important checks failed, so not fully operational
        elif informational_failures > 0:
            system_status = "LIMITED"
            overall_passed = True  # Informational failures don't affect operation
        else:
            system_status = "OPTIMAL"
            overall_passed = True
        
        # Create summary
        check_results_detail = {}
        for name, result in checks.items():
            # Extract relevant information for summary
            result_summary = {
                'passed': result.passed,
                'message': result.message,
                'level': result.level.name,
                'has_details': result.details is not None,
                'has_exception': result.exception is not None
            }
            
            # Include selective details
            if isinstance(result.details, dict):
                # Include only key metrics to avoid bloating summary
                key_metrics = {}
                if name == 'python_version':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['current_version', 'current_major_minor', 'required_major_minor']}
                elif name == 'torch_available':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['version', 'backend']}
                elif name == 'cuda_available':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['cuda_available', 'device_count']}
                elif name == 'package_versions':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['overall_passed', 'summary']}
                elif name == 'directory_access':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['overall_passed', 'summary']}
                elif name == 'disk_space':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['free_gb', 'required_gb']}
                elif name in ['cpu_cores', 'system_ram', 'system_architecture']:
                    key_metrics = result.details  # Include all details for these
                elif name == 'logging_setup':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['logger_name', 'handler_count']}
                elif name == 'seed_config':
                    key_metrics = {k: v for k, v in result.details.items() if k in ['seed_value', 'seed_set']}
                
                if key_metrics:
                    result_summary['key_metrics'] = key_metrics
            
            check_results_detail[name] = result_summary
        
        summary_details = {
            'total_checks': len(checks),
            'passed_checks': sum(1 for r in checks.values() if r.passed),
            'failed_checks': sum(1 for r in checks.values() if not r.passed),
            'critical_failures': critical_failures,
            'important_failures': important_failures,
            'informational_failures': informational_failures,
            'system_status': system_status,
            'overall_passed': overall_passed,
            'check_results': check_results_detail,
            'timestamp': datetime.datetime.now().isoformat(),
            'execution_context': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        # Create appropriate summary message
        if overall_passed:
            if system_status == "OPTIMAL":
                #summary_message = "[bold green]All system checks passed - System is optimal[/bold green]"
                summary_message = Text("All system checks passed - System is optimal", style="bold green")
            else:
                #summary_message = f"[bold yellow]System checks completed with {informational_failures} informational issues - System is operational[/bold yellow]"
                summary_message = Text(f"System checks completed with {informational_failures} informational issues - System is operational", style="bold yellow")
        else:
            if system_status == "CRITICAL_FAILURE":
                #summary_message = f"[bold red]{critical_failures} critical failures - System cannot operate[/bold red]"
                summary_message = Text(f"{critical_failures} critical failures - System cannot operate", style="bold red")
            else:
                #summary_message = f"[bold yellow]{important_failures} important failures - System is degraded[/bold yellow]"
                summary_message = Text(f"{important_failures} important failures - System is degraded", style="bold yellow")
        
        # Add system summary as a check result
        checks['system_summary'] = CheckResult(
            passed=overall_passed,
            message=summary_message,
            level=CheckLevel.CRITICAL if not overall_passed else CheckLevel.INFORMATIONAL,
            details=summary_details,
            metadata={
                'check_type': 'summary',
                'generated_by': 'run_system_checks',
                'checks_included': list(raw_checks.keys())
            }
        )
        
        # Log results based on severity
        if logger:
            # Log critical failures immediately
            for name, result in checks.items():
                if not result.passed and result.level == CheckLevel.CRITICAL:
                    logger.error(
                        f"Critical check failed: {name} - {result.message}",
                        extra={
                            'check_name': name,
                            'check_level': 'CRITICAL',
                            'details': result.details if isinstance(result.details, dict) else str(result.details),
                            'exception': str(result.exception) if result.exception else None
                        }
                    )
            
            # Log important failures
            for name, result in checks.items():
                if not result.passed and result.level == CheckLevel.IMPORTANT:
                    logger.warning(
                        f"Important check failed: {name} - {result.message}",
                        extra={
                            'check_name': name,
                            'check_level': 'IMPORTANT',
                            'details': result.details if isinstance(result.details, dict) else str(result.details)
                        }
                    )
            
            # Log summary
            if not overall_passed:
                logger.warning(
                    f"System checks completed with failures - Status: {system_status}",
                    extra={'summary': summary_details}
                )
            else:
                logger.debug(
                    f"All system checks passed - Status: {system_status}",
                    extra={
                        'summary': {
                            'total_checks': summary_details['total_checks'],
                            'passed_checks': summary_details['passed_checks'],
                            'system_status': summary_details['system_status']
                        }
                    }
                )
            
            # Log detailed debug info
            logger.debug(
                f"System checks detailed results",
                extra={'all_results': check_results_detail}
            )
        
        return checks
    
    except Exception as e:
        # Create error result
        error_result = CheckResult(
            passed=False,
            message="[bold red]System checks failed to complete[/bold red]",
            level=CheckLevel.CRITICAL,
            details={
                'error': str(e),
                'error_type': type(e).__name__,
                'completed_checks': list(checks.keys()),
                'traceback': traceback.format_exc(),
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
        ).with_exception(e)
        
        checks['system_error'] = error_result
        
        if logger:
            logger.critical(
                "Fatal error during system checks",
                exc_info=True,
                extra={
                    'completed_checks': list(checks.keys()),
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            )
        
        return checks

def display_check_results(checks: Dict[str, CheckResult], logger: logging.Logger = None):
    """Display check results in a styled table with formatting"""
    try:
        # Create main results table
        result_table = Table(
            title="SYSTEM DIAGNOSTICS REPORT",
            box=box.ROUNDED,
            header_style="bold yellow",
            border_style="bright_blue",
            title_style="bold cyan",
            title_justify="left",
            show_lines=False,
            expand=True,
            width=min(130, console.width - 4)
        )
        
        # Configure columns
        result_table.add_column("Check", style="bold cyan", width=22, no_wrap=True)
        result_table.add_column("Status", width=10, justify="center")
        result_table.add_column("Level", width=12, justify="center")
        result_table.add_column("Details", style="bold cyan", min_width=60, max_width=85, overflow="fold")
        
        # Process checks by level
        for level in [CheckLevel.CRITICAL, CheckLevel.IMPORTANT, CheckLevel.INFORMATIONAL]:
            # Filter checks for this level, excluding summary/error entries
            level_checks = [(name, check) for name, check in checks.items() if check.level == level and name not in ["system_summary", "system_error"]]
            
            if not level_checks:
                continue
            
            # Add section header
            level_colors = {
                CheckLevel.CRITICAL: ("bold white on red", "CRITICAL"),
                CheckLevel.IMPORTANT: ("bold white on yellow", "IMPORTANT"),
                CheckLevel.INFORMATIONAL: ("bold white on blue", "INFORMATIONAL")
            }
            
            level_style, level_name = level_colors.get(level, ("bold white on green", "UNKNOWN"))
            
            result_table.add_row(
                Text("", style=level_style),
                Text("", style=level_style),
                Text("", style=level_style),
                Text(f" {level_name} CHECKS ", style=level_style),
                style=level_style
            )
            
            # Add checks for this level
            for name, check in level_checks:
                # Determine status styling
                if check.passed:
                    status_style = "bold green"
                    status_text = "PASS"
                else:
                    if level == CheckLevel.CRITICAL:
                        status_style = "bold white on red"
                        status_text = "FAIL"
                    else:
                        status_style = "bold yellow"
                        status_text = "WARN"
                
                # Format check name
                check_name = name.replace("_", " ").title()
                
                # Format details based on check type
                details_lines = []
                
                # Add main message (strip formatting if needed)
                clean_message = check.message.replace("[bold green]", "").replace("[/bold green]", "")\
                                            .replace("[bold red]", "").replace("[/bold red]", "")\
                                            .replace("[bold yellow]", "").replace("[/bold yellow]", "")
                details_lines.append(f"[bold white]{clean_message}[/bold white]")
                
                # Add specific details based on check type
                if check.details:
                    if isinstance(check.details, dict):
                        # Python version details
                        if name == 'python_version':
                            if 'current_version' in check.details and 'required_major_minor' in check.details:
                                details_lines.append(f"Version: [bold green]{check.details['current_version']}[/bold green] (requires [bold yellow]{'.'.join(map(str, check.details['required_major_minor']))}+)[/bold yellow]")
                        
                        # Torch details
                        elif name == 'torch_available':
                            if 'version' in check.details:
                                details_lines.append(f"PyTorch: [bold green]{check.details['version']}[/bold green]")
                        
                        # CUDA details
                        elif name == 'cuda_available':
                            if check.details.get('cuda_available'):
                                gpu_info = check.details.get('gpus', [])
                                if gpu_info:
                                    gpu = gpu_info[0]
                                    details_lines.append(f"[bold green]{gpu.get('name', 'GPU')}[/bold green], [bold green]{gpu.get('total_memory_gb', 0):.1f}[/bold green]GB VRAM")
                            else:
                                details_lines.append("[bold green]CPU mode only[/bold green]")
                        
                        # Package versions summary
                        elif name == 'package_versions':
                            summary = check.details.get('summary', {})
                            if summary:
                                details_lines.append(f"[bold green]{summary.get('passed', 0)}[/bold green]/[bold yellow]{summary.get('total_packages', 0)}[/bold yellow] packages satisfied")
                        
                        # Directory access summary
                        elif name == 'directory_access':
                            summary = check.details.get('summary', {})
                            if summary:
                                details_lines.append(f"[bold green]{summary.get('accessible_directories', 0)}[/bold green]/[bold yellow]{summary.get('total_directories', 0)}[/bold yellow] directories accessible")
                        
                        # Disk space details
                        elif name == 'disk_space':
                            if 'free_gb' in check.details and 'required_gb' in check.details:
                                free = check.details['free_gb']
                                required = check.details['required_gb']
                                percent = (free / required * 100) if required > 0 else 0
                                details_lines.append(f"[bold green]{free:.1f}[/bold green]/[bold yellow]{required:.0f}[/bold yellow]GB ([bold green]{percent:.0f}%[/bold green] of requirement)")
                        
                        # CPU cores
                        elif name == 'cpu_cores':
                            if 'logical_cores' in check.details:
                                logical = check.details['logical_cores']
                                physical = check.details.get('physical_cores')
                                if physical:
                                    details_lines.append(f"[bold green]{logical}[/bold green] logical, [bold green]{physical}[/bold green] physical cores")
                                else:
                                    details_lines.append(f"[bold green]{logical}[/bold green] logical cores")
                        
                        # System RAM
                        elif name == 'system_ram':
                            if 'total_gb' in check.details and 'available_gb' in check.details:
                                total = check.details['total_gb']
                                available = check.details['available_gb']
                                percent_used = check.details.get('percent_used', 0)
                                details_lines.append(f"[bold green]{total:.1f}GB[/bold green] total, [bold cyan]{available:.1f}GB[/bold cyan] available ([bold yellow]{percent_used:.1f}%[/bold yellow] used)")
                        
                        # System architecture
                        elif name == 'system_architecture':
                            if 'architecture' in check.details and 'system' in check.details:
                                details_lines.append(f"[bold green]{check.details['architecture']}[/bold green] on [bold green]{check.details['system']} {check.details.get('release', '')}[/bold green]")
                        
                        # Logging setup
                        elif name == 'logging_setup':
                            if 'handler_count' in check.details:
                                details_lines.append(f"[bold green]{check.details['handler_count']}[/bold green] handlers configured")
                        
                        # Seed config
                        elif name == 'seed_config':
                            if 'seed_value' in check.details:
                                seed = check.details['seed_value']
                                if seed != 0:
                                    details_lines.append(f"Seed: [bold green]{seed}[/bold green] (reproducible)")
                                else:
                                    details_lines.append(f"Seed: [bold green]{seed}[/bold green] (not set)")
                    
                    elif isinstance(check.details, str):
                        # Add string details (truncated if too long)
                        if len(check.details) > 100:
                            details_lines.append(f"[bold green]{check.details[:100]}...[/bold green]")
                        else:
                            details_lines.append(f"[bold green]{check.details}[/bold green]")
                
                # Add exception if present
                if check.exception:
                    details_lines.append(f"[bold red]Error: {str(check.exception)}[/bold red]")
                
                details_text = "\n".join(details_lines)
                
                # Add row to table
                result_table.add_row(
                    Text(check_name),
                    Text(f"{status_text}", style=status_style),
                    Text(level.name, style="bold red" if level == CheckLevel.CRITICAL else "bold yellow" if level == CheckLevel.IMPORTANT else "bold cyan"),
                    details_text
                )
        
        # Add summary if present
        if "system_summary" in checks:
            summary = checks["system_summary"]
            summary_details = summary.details if isinstance(summary.details, dict) else {}
            
            # Determine summary styling
            if summary.passed:
                summary_style = "bold white on green"
            else:
                summary_style = "bold white on red"
            
            # Create summary text
            checks_run = summary_details.get('total_checks', 0)
            passed_checks = summary_details.get('passed_checks', 0)
            critical_failures = summary_details.get('critical_failures', 0)
            system_status = summary_details.get('system_status', 'UNKNOWN')
            
            summary_text = (f"{passed_checks}/{checks_run} checks passed | {critical_failures} critical failures | Status: {system_status}")
            
            result_table.add_row(
                Text("SYSTEM SUMMARY", style=summary_style),
                Text("", style=summary_style),
                Text("", style=summary_style),
                Text(summary_text, style=summary_style),
                style=summary_style
            )
        
        # Add system error if present
        if "system_error" in checks:
            error = checks["system_error"]
            error_details = error.details if isinstance(error.details, dict) else {}
            
            result_table.add_row(
                Text("FATAL ERROR", style="bold white on red"),
                Text("ERROR", style="bold white on red"),
                Text("", style="bold white on red"),
                Text(
                    f"{error.message}\n"
                    f"[yellow]{error_details.get('error', 'Unknown error')}[/yellow]\n"
                    f"Completed: {', '.join(error_details.get('completed_checks', []))}",
                    style="bold white on red"
                ),
                style="bold white on red"
            )
        
        # Print the main table
        console.print(result_table)
        
        # Log summary to logger if provided
        if logger:
            summary = checks.get("system_summary")
            if summary and isinstance(summary.details, dict):
                logger.debug(f"System diagnostics: {summary.details.get('passed_checks', 0)}/{summary.details.get('total_checks', 0)} checks passed, {summary.details.get('critical_failures', 0)} critical failures, Status: {summary.details.get('system_status', 'UNKNOWN')}")
            
            # Log critical failures
            critical_failures = [
                (name, check) for name, check in checks.items()
                if not check.passed and check.level == CheckLevel.CRITICAL
                and name not in ["system_summary", "system_error"]
            ]
            
            for name, check in critical_failures:
                logger.error(f"Critical check failed: {name} - {check.message}")
            
            # Log system error
            if "system_error" in checks:
                error = checks["system_error"]
                error_msg = error.details.get('error', 'Unknown error') if isinstance(error.details, dict) else str(error.details)
                logger.critical(f"System checks failed: {error_msg}")
    
    except Exception as e:
        error_msg = f"Failed to display check results: {str(e)}"
        print(Fore.RED + Style.BRIGHT + f"Error: {error_msg}" + Style.RESET_ALL)
        
        if logger:
            logger.critical(f"Display check results failed: {error_msg}", exc_info=True)

# Individual check implementations
def check_python_version(min_version: Tuple[int, int] = (3, 8)) -> CheckResult:
    """Check if Python version meets minimum requirements."""
    try:
        current = tuple(map(int, platform.python_version().split('.')[:2]))
        passed = current >= min_version
        
        if passed:
            message = f"[bold green]Python {platform.python_version()} (requires >= {'.'.join(map(str, min_version))})[/bold green]"
        else:
            message = f"[bold red]Python {platform.python_version()} below required {'.'.join(map(str, min_version))}[/bold red]"
        
        return CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.CRITICAL,
            details={
                "current_version": platform.python_version(),
                "current_major_minor": current,
                "required_major_minor": min_version,
                "version_check": "current >= required"
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Could not determine Python version[/bold red]",
            level=CheckLevel.CRITICAL
        ).with_exception(e)

def check_torch() -> CheckResult:
    """Check if PyTorch is available and functional."""
    try:
        # Basic functionality test
        test_tensor = torch.zeros(1)
        passed = test_tensor is not None
        
        return CheckResult(
            passed=passed,
            message=f"[bold green]PyTorch {torch.__version__} available[/bold green]",
            level=CheckLevel.CRITICAL,
            details={
                "version": torch.__version__,
                "backend": "functional",
                "test_operation": "torch.zeros(1)"
            }
        )
    except ImportError as e:
        return CheckResult(
            passed=False,
            message="[bold red]PyTorch not installed[/bold red]",
            level=CheckLevel.CRITICAL,
            details={"error_type": "ImportError"}
        ).with_exception(e)
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]PyTorch not functional[/bold red]",
            level=CheckLevel.CRITICAL,
            details={"error_type": type(e).__name__}
        ).with_exception(e)

def check_cuda() -> CheckResult:
    """Check CUDA availability and GPU properties."""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1e9,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                })
            
            primary_gpu = gpu_info[0]
            message = f"[bold green]CUDA available ({primary_gpu['name']}, {primary_gpu['total_memory_gb']:.1f}GB)[/bold green]"
            if device_count > 1:
                message += f" + {device_count - 1} more GPU(s)"
            
            return CheckResult(
                passed=True,
                message=message,
                level=CheckLevel.INFORMATIONAL,
                details={
                    "cuda_available": True,
                    "device_count": device_count,
                    "gpus": gpu_info,
                    "current_device": torch.cuda.current_device() if device_count > 0 else None
                }
            )
        else:
            return CheckResult(
                passed=False,
                message="[bold yellow]CUDA not available - Using CPU[/bold yellow]",
                level=CheckLevel.INFORMATIONAL,
                details={
                    "cuda_available": False,
                    "device_count": 0,
                    "note": "Running in CPU mode"
                }
            )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]CUDA check failed[/bold red]",
            level=CheckLevel.INFORMATIONAL,
            details={"error_type": type(e).__name__}
        ).with_exception(e)

def check_package_versions_wrapper(logger: logging.Logger) -> CheckResult:
    """Wrapper for package version checks with detailed results."""
    try:
        # Collect version data using the check_versions function
        all_ok, version_data = check_versions(logger)
        
        # Count statuses for summary
        passed_count = sum(1 for _, _, _, is_ok, _ in version_data if is_ok)
        total_count = len(version_data)
        failed_count = total_count - passed_count
        
        if all_ok:
            message = f"[bold green]All {total_count} package versions satisfied[/bold green]"
        elif failed_count > 0 and passed_count > 0:
            message = f"[bold yellow]{passed_count}/{total_count} packages satisfied ({failed_count} issues)[/bold yellow]"
        else:
            message = f"[bold red]{failed_count}/{total_count} packages have issues[/bold red]"
        
        # Create structured details
        package_details = []
        for pkg, current_ver, min_ver, is_ok, error in version_data:
            package_details.append({
                "package": pkg,
                "current_version": current_ver,
                "required_version": min_ver,
                "status": "satisfied" if is_ok else "failed",
                "error": error,
                "check_passed": is_ok
            })
        
        return CheckResult(
            passed=all_ok,
            message=message,
            level=CheckLevel.IMPORTANT,
            details={
                "check_type": "package_versions",
                "overall_passed": all_ok,
                "summary": {
                    "total_packages": total_count,
                    "passed": passed_count,
                    "failed": failed_count,
                    "success_rate": (passed_count / total_count) * 100 if total_count > 0 else 0
                },
                "packages": package_details,
                "requirements": {
                    'torch': '1.10.0',
                    'torchvision': '0.11.0',
                    'scikit-learn': '1.0.0',
                    'imbalanced-learn': '0.9.0',
                    'pandas': '1.3.0',
                    'numpy': '1.21.0',
                    'matplotlib': '3.5.0',
                    'seaborn': '0.11.0'
                }
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Package version check failed[/bold red]",
            level=CheckLevel.IMPORTANT,
            details={
                "check_type": "package_versions",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        ).with_exception(e)

def check_directory_access_wrapper(logger: logging.Logger) -> CheckResult:
    """Check accessibility of required directories with testing."""
    try:
        # Get directories using setup_directories function
        dirs = setup_directories(logger)
        problematic = []
        access_details = {}
        
        for name, path in dirs.items():
            dir_details = {
                "name": name,
                "path": str(path),
                "absolute_path": str(path.absolute()),
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else False,
                "accessible": False,
                "permissions": {},
                "error": None,
                "tests": {}
            }
            
            try:
                # Test 1: Check if path exists
                if not path.exists():
                    # Try to create it
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        dir_details["tests"]["created"] = True
                    except Exception as e:
                        problematic.append(f"{name}: Directory does not exist and cannot be created - {str(e)}")
                        dir_details["error"] = f"Creation failed: {str(e)}"
                        access_details[name] = dir_details
                        continue
                
                # Test 2: Check if it's a directory
                if not path.is_dir():
                    problematic.append(f"{name}: Path exists but is not a directory")
                    dir_details["error"] = "Not a directory"
                    access_details[name] = dir_details
                    continue
                
                # Test 3: Check read permission
                try:
                    list(path.iterdir())
                    dir_details["tests"]["readable"] = True
                    dir_details["permissions"]["read"] = True
                except PermissionError:
                    problematic.append(f"{name}: No read permission")
                    dir_details["permissions"]["read"] = False
                
                # Test 4: Check write permission
                try:
                    test_file = path / f".permission_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    test_content = f"Test write access at {datetime.datetime.now().isoformat()}"
                    test_file.write_text(test_content)
                    dir_details["tests"]["write_test_file_created"] = True
                    
                    # Test 5: Check read back
                    read_content = test_file.read_text()
                    dir_details["tests"]["read_test_file_successful"] = (read_content == test_content)
                    
                    # Test 6: Clean up
                    test_file.unlink()
                    dir_details["tests"]["cleanup_successful"] = True
                    
                    dir_details["permissions"]["write"] = True
                    dir_details["permissions"]["delete"] = True
                except PermissionError as e:
                    problematic.append(f"{name}: No write permission - {str(e)}")
                    dir_details["permissions"]["write"] = False
                    dir_details["permissions"]["delete"] = False
                    dir_details["error"] = f"Write permission denied: {str(e)}"
                
                # Test 7: Check execute/traverse permission (for directories)
                try:
                    Path(path).resolve()
                    dir_details["permissions"]["execute"] = True
                except PermissionError:
                    problematic.append(f"{name}: No execute/traverse permission")
                    dir_details["permissions"]["execute"] = False
                
                # Determine overall accessibility
                if all(dir_details["permissions"].values()):
                    dir_details["accessible"] = True
                    dir_details["access_level"] = "full"
                elif dir_details["permissions"].get("read") and dir_details["permissions"].get("write"):
                    dir_details["accessible"] = True
                    dir_details["access_level"] = "read_write"
                elif dir_details["permissions"].get("read"):
                    dir_details["accessible"] = True
                    dir_details["access_level"] = "read_only"
                else:
                    dir_details["accessible"] = False
                    dir_details["access_level"] = "none"
                
                access_details[name] = dir_details
                
            except Exception as e:
                error_msg = f"{name}: Unexpected error - {str(e)}"
                problematic.append(error_msg)
                dir_details["error"] = str(e)
                dir_details["accessible"] = False
                dir_details["access_level"] = "error"
                access_details[name] = dir_details
        
        # Calculate summary statistics
        total_dirs = len(dirs)
        accessible_dirs = sum(1 for details in access_details.values() if details.get("accessible", False))
        read_write_dirs = sum(1 for details in access_details.values() if details.get("access_level") == "full" or details.get("access_level") == "read_write")
        read_only_dirs = sum(1 for details in access_details.values() if details.get("access_level") == "read_only")
        
        overall_passed = len(problematic) == 0
        
        if overall_passed:
            message = f"[bold green]All {total_dirs} directories fully accessible[/bold green]"
        elif read_write_dirs == total_dirs:
            message = f"[bold green]All {total_dirs} directories have read/write access[/bold green]"
        elif accessible_dirs > 0 and accessible_dirs < total_dirs:
            message = f"[bold yellow]{accessible_dirs}/{total_dirs} directories accessible ({len(problematic)} issues)[/bold yellow]"
        else:
            message = f"[bold red]{len(problematic)}/{total_dirs} directory access issues[/bold red]"
        
        return CheckResult(
            passed=overall_passed,
            message=message,
            level=CheckLevel.IMPORTANT,
            details={
                "check_type": "directory_access",
                "overall_passed": overall_passed,
                "summary": {
                    "total_directories": total_dirs,
                    "accessible_directories": accessible_dirs,
                    "read_write_directories": read_write_dirs,
                    "read_only_directories": read_only_dirs,
                    "inaccessible_directories": total_dirs - accessible_dirs,
                    "issues_count": len(problematic)
                },
                "directory_details": access_details,
                "issues": problematic if problematic else None
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Directory check failed[/bold red]",
            level=CheckLevel.IMPORTANT,
            details={
                "check_type": "directory_access",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        ).with_exception(e)

def check_disk_space(min_gb: int = 5) -> CheckResult:
    """Check available disk space."""
    try:
        usage = psutil.disk_usage(".")
        free_gb = usage.free / (1024 ** 3)
        passed = free_gb >= min_gb
        
        if passed:
            message = f"[bold green]{free_gb:.1f}GB free (needs >= {min_gb}GB)[/bold green]"
        else:
            message = f"[bold red]{free_gb:.1f}GB free (needs >= {min_gb}GB)[/bold red]"
        
        return CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.IMPORTANT,
            details={
                "free_gb": free_gb,
                "required_gb": min_gb,
                "total_gb": usage.total / (1024 ** 3),
                "used_gb": usage.used / (1024 ** 3),
                "percent_used": usage.percent,
                "path": "."
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Could not check disk space[/bold red]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "check_type": "disk_space",
                "error_type": type(e).__name__
            }
        ).with_exception(e)

def check_cpu_cores() -> CheckResult:
    """Check available CPU cores."""
    try:
        cores = os.cpu_count() or 1
        return CheckResult(
            passed=True,
            message=f"[bold green]{cores} logical cores available[/bold green]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "logical_cores": cores,
                "physical_cores": psutil.cpu_count(logical=False) if hasattr(psutil, 'cpu_count') else None
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Could not check CPU cores[/bold red]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "check_type": "cpu_cores",
                "error_type": type(e).__name__
            }
        ).with_exception(e)

def check_system_ram() -> CheckResult:
    """Check system RAM availability."""
    try:
        vm = psutil.virtual_memory()
        ram_gb = vm.total / (1024 ** 3)
        available_gb = vm.available / (1024 ** 3)
        
        return CheckResult(
            passed=True,
            message=f"[bold green]{ram_gb:.1f}GB system RAM ({available_gb:.1f}GB available)[/bold green]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "total_gb": ram_gb,
                "available_gb": available_gb,
                "used_gb": vm.used / (1024 ** 3),
                "percent_used": vm.percent,
                "thresholds": {
                    "low": vm.total * 0.2 / (1024 ** 3),
                    "medium": vm.total * 0.5 / (1024 ** 3),
                    "high": vm.total * 0.8 / (1024 ** 3)
                }
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Could not check system RAM[/bold red]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "check_type": "system_ram",
                "error_type": type(e).__name__
            }
        ).with_exception(e)

def check_system_architecture() -> CheckResult:
    """Check system architecture."""
    try:
        arch = platform.machine()
        processor = platform.processor()
        system = platform.system()
        release = platform.release()
        
        return CheckResult(
            passed=True,
            message=f"[bold green]{arch} architecture ({system} {release})[/bold green]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "architecture": arch,
                "processor": processor,
                "system": system,
                "release": release,
                "version": platform.version()
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Could not check system architecture[/bold red]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "check_type": "system_architecture",
                "error_type": type(e).__name__
            }
        ).with_exception(e)

def check_logging_setup(logger: logging.Logger) -> CheckResult:
    """Check if logging is properly configured."""
    try:
        if logger.handlers:
            handler_types = [type(h).__name__ for h in logger.handlers]
            handler_levels = [h.level for h in logger.handlers]
            
            return CheckResult(
                passed=True,
                message=f"[bold green]Logging configured ({', '.join(handler_types)})[/bold green]",
                level=CheckLevel.INFORMATIONAL,
                details={
                    "logger_name": logger.name,
                    "logger_level": logging.getLevelName(logger.level),
                    "handler_count": len(logger.handlers),
                    "handler_types": handler_types,
                    "handler_levels": [logging.getLevelName(level) for level in handler_levels],
                    "has_handlers": True
                }
            )
        else:
            return CheckResult(
                passed=False,
                message="[bold yellow]Logging not configured (no handlers)[/bold yellow]",
                level=CheckLevel.IMPORTANT,
                details={
                    "logger_name": logger.name,
                    "logger_level": logging.getLevelName(logger.level) if hasattr(logger, 'level') else 'NOTSET',
                    "handler_count": 0,
                    "has_handlers": False,
                    "note": "Logger exists but has no handlers configured"
                }
            )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Logging check failed[/bold red]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "check_type": "logging_setup",
                "error_type": type(e).__name__
            }
        ).with_exception(e)

def check_seed_config() -> CheckResult:
    """Check reproducibility seed configuration."""
    try:
        seed = int(os.environ.get('PYTHONHASHSEED', '0'))
        passed = seed != 0
        
        if passed:
            message = f"[bold green]Reproducibility seed set to {seed}[/bold green]"
        else:
            message = "[bold yellow]Reproducibility seed not set[/bold yellow]"
        
        return CheckResult(
            passed=passed,
            message=message,
            level=CheckLevel.INFORMATIONAL,
            details={
                "seed_value": seed,
                "seed_set": seed != 0,
                "environment_variable": "PYTHONHASHSEED",
                "note": "Set PYTHONHASHSEED for reproducible hashing"
            }
        )
    except Exception as e:
        return CheckResult(
            passed=False,
            message="[bold red]Seed check failed[/bold red]",
            level=CheckLevel.INFORMATIONAL,
            details={
                "check_type": "seed_config",
                "error_type": type(e).__name__
            }
        ).with_exception(e)

# System and environment configuration
def configure_system() -> None:
    """Configure system settings for optimal performance."""
    # Disable verbose logging for libraries
    
    # TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
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
    'max_samples': 50000,
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
INFO_DIR = ARTIFACTS_DIR = DOCS_DIR = DATASETS_DIR = None

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Configure logging with a single log file and proper handler management."""
    # Determine log_dir (default: script's directory / logs)
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    
    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # Rest of the logging setup
    logger.setLevel(logging.DEBUG)
    
    # Log filename
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
        'datasets': base_dir / "datasets",
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
    global INFO_DIR, ARTIFACTS_DIR, DOCS_DIR, DATASETS_DIR

    try:
        directories = setup_directories(logger)

        MODEL_DIR = directories['models']
        LOG_DIR = directories['logs']
        DATA_DIR = directories['data']
        DATASETS_DIR = directories['datasets']
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

def check_versions(logger: logging.Logger) -> Tuple[bool, List[Tuple[str, Optional[str], str, bool, Optional[str]]]]:
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
    all_ok = True

    for pkg, min_ver in requirements.items():
        try:
            current_ver = importlib.metadata.version(pkg)
            is_ok = (pkg_version.parse(current_ver) >= pkg_version.parse(min_ver))
            version_data.append((pkg, current_ver, min_ver, is_ok, None))

            if not is_ok:
                logger.warning(f"Version mismatch: {pkg} {current_ver} (needs >= {min_ver})")
                all_ok = False
        
        except importlib.metadata.PackageNotFoundError as e:
            version_data.append((pkg, None, min_ver, False, f"Package not found: {str(e)}"))
            logger.warning(f"Package not found: {pkg}")
            all_ok = False
        
        except Exception as e:
            version_data.append((pkg, None, min_ver, False, str(e)))
            logger.warning(f"Package not found: {pkg} - {str(e)}")
            all_ok = False

    # Create rich table
    table = Table(title="Package Versions", box=box.ROUNDED, title_justify="left", title_style="bold yellow")
    table.add_column("Package", style="bold cyan", no_wrap=True)
    table.add_column("Installed", style="bold magenta")
    table.add_column("Required", style="bold green")
    table.add_column("Status", style="bold yellow", justify="center")

    # Add rows to table
    for pkg, current_ver, min_ver, is_ok, error in version_data:
        if current_ver:
            status = Text("OK", style="bold green") if is_ok else Text(f"FAILED", style="bold red")
        else:
            status = Text("MISSING", style="bold red")
            current_ver = Text("N/A", style="italic")
        
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
    debug_output = ["Package Versions"]
    for pkg, current_ver, min_ver, is_ok, error in version_data:
        if current_ver:
            status = "[OK]" if is_ok else f"[FAIL] (needs >= {min_ver})"
            line = f"{pkg} {current_ver} (>= {min_ver}) {status}"
        else:
            line = f"{pkg} [N/A] (>= {min_ver}) [MISSING]"
        debug_output.append(line)
    
    logger.debug("\n".join(debug_output))

    return all_ok, version_data

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
TAB_WIDTH = 5  # assumed terminal tab width

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
        'epochs': 5,
        'early_stopping': 3
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

def select_config_preset() -> Optional[Dict[str, Any]]:
    """Interactive preset selection with rich formatting"""
    
    preset_table = Table(
        title="\nAvailable Preset Configurations",
        box=box.ROUNDED,
        title_style="bold yellow",
        title_justify="left",
        header_style="bold cyan",
        border_style="blue",
        show_header=True,
        show_lines=True
    )
    
    preset_table.add_column("Option", style="bold white", width=8)
    preset_table.add_column("Preset", style="bold yellow", width=12)
    preset_table.add_column("Model", style="bold green", width=10)
    preset_table.add_column("Description", style="bold dim", width=40)
    
    preset_table.add_row("1", "stability", "simple", "Conservative settings for stable training")
    preset_table.add_row("2", "performance", "ensemble", "Optimized for best performance")
    preset_table.add_row("3", "baseline", "standard", "Standard configuration baseline")
    preset_table.add_row("4", "debug", "simple", "Fast training for debugging")
    preset_table.add_row("5", "current", "varies", "Use current configuration")
    preset_table.add_row("6", "custom", "varies", "Use custom configuration")
    preset_table.add_row("0", "Exit", "none", "Exit preset configuration", style="bold red")
    
    console.print(preset_table)
    
    try:
        choice = input(Fore.YELLOW + Style.BRIGHT + "\nSelect an option (0-6): " + Style.RESET_ALL).strip()
        
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
        elif choice == '6':
            configure_training()
        elif choice == '0':
            print(Fore.RED + Style.BRIGHT + "Exiting preset selection." + Style.RESET_ALL)
            # Return to main menu
            return 'exit'
        else:
            print(Fore.YELLOW + Style.BRIGHT + "Invalid selection. Using current configuration." + Style.RESET_ALL)
            return None
    
    except KeyboardInterrupt:
        print(Fore.RED + Style.BRIGHT + "Selection cancelled." + Style.RESET_ALL)
        return None

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

# Main configuration and setup function
def initialize_system():
    """Centralized system initialization with single logging setup"""
    # Basic configuration
    configure_system()
    configure_visualization()
    set_seed(42)
    
    # Early logging setup
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir)
    
    # Run loading screen with checks
    if not loading_screen(logger):
        logger.error(Fore.RED + Style.BRIGHT + "CRITICAL SYSTEM CHECKS FAILED! - Cannot continue" + Style.RESET_ALL)
        sys.exit(1)
    
    # Continue with normal setup if checks passed
    if not check_versions(logger):
        logger.error(Fore.RED + Style.BRIGHT + "Some package requirements not met!" + Style.RESET_ALL)
    
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

def get_autocast_context(device, mixed_precision=True, enabled=True):
    """
    Get appropriate autocast context for different PyTorch versions and devices.
    
    Args:
        device: torch.device object
        mixed_precision: bool, whether mixed precision is enabled
        enabled: bool, whether autocast should be enabled
    
    Returns:
        Context manager for autocast or nullcontext
    """
    if not mixed_precision or not enabled:
        return nullcontext()
    
    if device.type == 'cuda':
        try:
            # Try new API first (PyTorch 1.10+)
            return torch.amp.autocast(device_type='cuda', enabled=True)
        except TypeError:
            try:
                # Fallback for older PyTorch versions
                return torch.amp.autocast()
            except Exception:
                return nullcontext()
    else:
        try:
            # Try CPU autocast (PyTorch 1.10+)
            return torch.amp.autocast(device_type='cpu', enabled=True)
        except (TypeError, AttributeError):
            # CPU autocast not available or different API
            return nullcontext()

class ProgressHelper:
    """Reusable factory for creating consistent alive_progress bars."""

    def __init__(self, titles: list[str] = None):
        """
        Initialize helper with optional list of titles.
        
        Args:
            titles: Optional list of title strings (not needed for simple approach)
        """
        # Titles parameter is kept for backward compatibility but not used
        self.titles = titles if titles else []

    def bar(self, title: str, total: int, unit: str):
        """
        Return a configured alive_progress bar with consistent formatting.
        
        Args:
            title: The title to display
            total: Total number of items to process
            unit: Unit name for the progress bar
            
        Returns:
            Configured alive_bar context manager
        """
        return alive_bar(total, title=title, unit=unit, length=25, elapsed=True, title_length=30)

def print_color(message: str, color: str = 'white', style: str = 'bright'):
    """Helper function for colored output."""
    color_map = {
        'white': Fore.WHITE,
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN
    }
    style_map = {
        'normal': Style.NORMAL,
        'bright': Style.BRIGHT,
        'dim': Style.DIM
    }
    color_code = color_map.get(color, Fore.WHITE)
    style_code = style_map.get(style, Style.NORMAL)
    print(f"{style_code}{color_code}{message}{Style.RESET_ALL}")

# Data preprocessing and validation
def get_preprocessing_outputs(
    config_path: Optional[str] = None,
    base_results_dir: Optional[Path] = None,
    base_preprocessing_dir: Optional[Path] = None,
    interactive: bool = True
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Get preprocessing outputs location with support for run-based directory structure.
    
    This function helps locate preprocessing outputs including the preprocessed dataset .csv file
    and the preprocessing artifacts .pkl file based on various input configurations.
    
    Args:
        config_path: Path to either:
                     - A specific test run directory (e.g., "results/preprocessing/run_001/")
                     - A results directory (e.g., "results/preprocessing/")
                     - A preprocessing run directory (e.g., "datasets/preprocessing/run_001/")
                     - A specific summary file
        base_results_dir: Base directory for test results (default: "results/preprocessing")
        base_preprocessing_dir: Base directory for preprocessing outputs (default: "datasets/preprocessing")
        interactive: If True, allows user to select from multiple matches
    
    Returns:
        Tuple containing:
        - Path to the preprocessed dataset CSV file (or None if not found)
        - Path to the preprocessing artifacts PKL file (or None if not found)
        - Configuration dictionary from the test run (or None if not found)
        - Summary dictionary from preprocessing (or None if not found)
    
    Example usage in another script:
        csv_path, pkl_path, test_config, preprocess_summary = get_preprocessing_outputs(
            config_path="results/preprocessing/run_001/",
            interactive=False
        )
        if csv_path:
            print(f"Found preprocessed dataset: {csv_path}")
            df = pd.read_csv(csv_path)
        if pkl_path:
            print(f"Found preprocessing artifacts: {pkl_path}")
            artifacts = joblib.load(pkl_path)
    """
    try:
        script_dir = Path(__file__).resolve().parent
        max_rows_test_results_dir = script_dir / "results" / "preprocessing"
        preprocessing_outputs_dir = script_dir / "datasets" / "preprocessing"
        
        if base_results_dir is None:
            base_results_dir = max_rows_test_results_dir
        
        if base_preprocessing_dir is None:
            base_preprocessing_dir = preprocessing_outputs_dir

        if config_path is None:
            config_path = base_results_dir
        
        config_file = Path(config_path)
        
        # Dictionary to store found configuration
        test_config = None
        preprocessing_summary = None
        preprocessing_run_id = None
        preprocessing_run_number = None
        
        # Check if the provided path is a specific test run directory
        if config_file.is_dir() and (config_file.name.startswith('run_') or (config_file / f"max_rows_testing_summary_{config_file.name}.json").exists()):
            # Direct test run directory provided
            test_run_dir = config_file
            run_id = test_run_dir.name
            print_color("\nPREPROCESSING OUTPUTS LOCATOR", 'magenta')
            print_color("-" * 50, 'magenta')
            print_color(f"Processing test run directory:", 'magenta')
            print_color(f"  ├─ Directory: {Fore.WHITE + Style.BRIGHT}{test_run_dir}", 'magenta')
            print_color(f"  └─ Run ID: {Fore.YELLOW + Style.BRIGHT}{run_id}", 'magenta')
            print_color("-" * 50, 'magenta')
            
            # Load test configuration from the test run
            summary_file = test_run_dir / f"max_rows_testing_summary_{run_id}.json"
            
            if summary_file.exists():
                with open(summary_file) as f:
                    test_config = json.load(f)
                
                # Extract run information from test config
                test_run_info = test_config.get('run_info', {})
                test_run_id = test_run_info.get('run_id', run_id)
                test_run_number = test_run_info.get('run_number', int(run_id.split('_')[1]) if '_' in run_id else 0)
                
                print_color(f"\nTest configuration loaded:", 'green')
                print_color(f"  ├─ Test Run ID: {Fore.YELLOW + Style.BRIGHT}{test_run_id}", 'green')
                print_color(f"  ├─ Test Run Number: {Fore.YELLOW + Style.BRIGHT}{test_run_number}", 'green')
                print_color(f"  └─ Summary file: {Fore.MAGENTA + Style.BRIGHT}{summary_file}", 'green')
                
                # Look for corresponding preprocessing run
                # Preprocessing runs are stored in base_preprocessing_dir with same run number
                preprocessing_dir = Path(base_preprocessing_dir)
                
                if preprocessing_dir.exists():
                    # Find preprocessing run with same number
                    preprocessing_run_pattern = f"run_{test_run_number:03d}"
                    preprocessing_run_dirs = list(preprocessing_dir.glob(f"{preprocessing_run_pattern}*"))
                    
                    if preprocessing_run_dirs:
                        # Use the first match (should be only one)
                        preprocessing_run_dir = preprocessing_run_dirs[0]
                        preprocessing_run_id = preprocessing_run_dir.name
                        
                        print_color(f"\nFound corresponding preprocessing run:", 'green')
                        print_color(f"  ├─ Directory: {Fore.WHITE + Style.BRIGHT}{preprocessing_run_dir}", 'green')
                        print_color(f"  └─ Run ID: {Fore.YELLOW + Style.BRIGHT}{preprocessing_run_id}", 'green')
                        
                        # Look for preprocessing summary file
                        dataset_name = Path(test_config['dataset_info']['filepath']).stem
                        summary_pattern = f"{dataset_name}_preprocessing_summary_{preprocessing_run_id}.json"
                        summary_files = list(preprocessing_run_dir.glob(f"*{summary_pattern}*"))
                        
                        if summary_files:
                            # Load preprocessing summary
                            with open(summary_files[0]) as f:
                                preprocessing_summary = json.load(f)
                            
                            print_color(f"\nPreprocessing summary loaded:", 'green')
                            print_color(f"  ├─ Dataset name: {Fore.YELLOW + Style.BRIGHT}{dataset_name}", 'green')
                            print_color(f"  └─ Summary file: {Fore.MAGENTA + Style.BRIGHT}{summary_files[0]}", 'green')
                            
                            # Look for preprocessed dataset CSV
                            csv_pattern = f"{dataset_name}_preprocessed_dataset.csv"
                            csv_files = list(preprocessing_run_dir.glob(f"*{csv_pattern}*"))
                            
                            # Look for preprocessed dataset PKL
                            pkl_pattern = f"{dataset_name}_preprocessing_artifacts.pkl"
                            pkl_files = list(preprocessing_run_dir.glob(f"*{pkl_pattern}*"))
                            
                            if csv_files and pkl_files:
                                csv_path = str(csv_files[0])
                                pkl_path = str(pkl_files[0])
                                
                                print_color(f"\nPreprocessing outputs found:", 'green')
                                print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'green')
                                print_color(f"  └─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'green')
                                
                                return csv_path, pkl_path, test_config, preprocessing_summary
                            
                            elif csv_files:
                                csv_path = str(csv_files[0])
                                print_color(f"\nWarning: CSV file found but PKL artifacts missing", 'yellow')
                                print_color(f"  └─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'yellow')
                                return csv_path, None, test_config, preprocessing_summary
                            
                            elif pkl_files:
                                pkl_path = str(pkl_files[0])
                                print_color(f"\nWarning: PKL artifacts found but CSV file missing", 'yellow')
                                print_color(f"  └─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'yellow')
                                return None, pkl_path, test_config, preprocessing_summary
                            
                            else:
                                print_color(f"\nError: No preprocessing outputs found in directory", 'red')
                                print_color(f"  ├─ Searched for CSV pattern: {Fore.YELLOW + Style.BRIGHT}*{csv_pattern}*", 'red')
                                print_color(f"  └─ Searched for PKL pattern: {Fore.YELLOW + Style.BRIGHT}*{pkl_pattern}*", 'red')
                                return None, None, test_config, preprocessing_summary
                        else:
                            print_color(f"\nError: No preprocessing summary file found", 'red')
                            print_color(f"  ├─ Expected pattern: {Fore.YELLOW + Style.BRIGHT}*{summary_pattern}*", 'red')
                            print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{preprocessing_run_dir}", 'red')
                            return None, None, test_config, None
                    else:
                        print_color(f"\nError: No corresponding preprocessing run found", 'red')
                        print_color(f"  ├─ Test run number: {Fore.YELLOW + Style.BRIGHT}{test_run_number}", 'red')
                        print_color(f"  ├─ Expected pattern: {Fore.YELLOW + Style.BRIGHT}{preprocessing_run_pattern}*", 'red')
                        print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{preprocessing_dir}", 'red')
                        return None, None, test_config, None
                else:
                    print_color(f"\nError: Preprocessing directory not found", 'red')
                    print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{base_preprocessing_dir}", 'red')
                    return None, None, test_config, None
            else:
                print_color(f"\nError: Test summary file not found", 'red')
                print_color(f"  ├─ Expected file: {Fore.MAGENTA + Style.BRIGHT}{summary_file}", 'red')
                print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{test_run_dir}", 'red')
                return None, None, None, None
        
        # Check if path points to results directory (search for latest run)
        elif config_file.is_dir():
            results_dir = config_file
            print_color("\nPREPROCESSING OUTPUTS LOCATOR", 'magenta')
            print_color("-" * 50, 'magenta')
            print_color(f"Processing results directory:", 'magenta')
            print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{results_dir}", 'magenta')
            print_color("-" * 50, 'magenta')
            
            # Find all test run directories
            test_run_dirs = sorted(
                [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                key=lambda x: int(x.name.split('_')[1]) if len(x.name.split('_')) > 1 and x.name.split('_')[1].isdigit() else 0,
                reverse=True
            )
            
            if not test_run_dirs:
                print_color(f"\nError: No test runs found in directory", 'red')
                print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{results_dir}", 'red')
                return None, None, None, None
            
            total_test_runs = Fore.YELLOW + Style.BRIGHT + f"{len(test_run_dirs)}" + Style.RESET_ALL
            print_color(f"\nTest Runs Found: {total_test_runs}", 'green')
            
            # If interactive mode, let user choose
            if interactive and len(test_run_dirs) > 1:
                for i, run_dir in enumerate(test_run_dirs[:10], 1):
                    run_id = run_dir.name
                    print_color(f"{i:2d}. {run_id}", 'white')
                
                if len(test_run_dirs) > 10:
                    print_color(f"\n... and {len(test_run_dirs) - 10} more runs", 'cyan')
                
                print_color(f"\n{len(test_run_dirs) + 1:2d}. Use latest run", 'cyan')
                print_color(f"0. Cancel", 'red')
                
                choice = input(Fore.YELLOW + Style.BRIGHT + f"\nSelect test run (1-{len(test_run_dirs) + 1}, 0 to cancel): " + Style.RESET_ALL).strip()
                
                if choice == '0':
                    print_color("\nSelection cancelled", 'yellow')
                    return None, None, None, None
                elif choice == str(len(test_run_dirs) + 1):
                    # Use latest run
                    selected_run_dir = test_run_dirs[0]
                    print_color(f"\nUsing latest run: {Fore.YELLOW + Style.BRIGHT}{selected_run_dir.name}", 'green')
                elif choice.isdigit() and 1 <= int(choice) <= len(test_run_dirs):
                    selected_run_dir = test_run_dirs[int(choice) - 1]
                    print_color(f"\nSelected run: {Fore.YELLOW + Style.BRIGHT}{selected_run_dir.name}", 'green')
                else:
                    print_color("\nInvalid selection, using latest run", 'yellow')
                    selected_run_dir = test_run_dirs[0]
            else:
                # Use latest run
                selected_run_dir = test_run_dirs[0]
                print_color(f"\nUsing latest run: {Fore.YELLOW + Style.BRIGHT}{selected_run_dir.name}", 'green')
            
            # Recursively call with selected run directory
            return get_preprocessing_outputs(
                config_path=str(selected_run_dir),
                base_results_dir=base_results_dir,
                base_preprocessing_dir=base_preprocessing_dir,
                interactive=False  # Don't recurse infinitely
            )
        
        # Check if path is a preprocessing run directory directly
        elif config_file.is_dir() and config_file.parent.name == "preprocessing":
            # This might already be a preprocessing run directory
            preprocessing_run_dir = config_file
            preprocessing_run_id = preprocessing_run_dir.name
            print_color("\nPREPROCESSING OUTPUTS LOCATOR", 'magenta')
            print_color("-" * 50, 'magenta')
            print_color(f"Processing preprocessing run directory directly:", 'magenta')
            print_color(f"  ├─ Directory: {Fore.WHITE + Style.BRIGHT}{preprocessing_run_dir}", 'magenta')
            print_color(f"  └─ Run ID: {Fore.YELLOW + Style.BRIGHT}{preprocessing_run_id}", 'magenta')
            print_color("-" * 50, 'magenta')
            
            # Look for any preprocessing summary file
            summary_files = list(preprocessing_run_dir.glob("*_preprocessing_summary_*.json"))
            
            if summary_files:
                # Load the first summary file found
                with open(summary_files[0]) as f:
                    preprocessing_summary = json.load(f)
                
                print_color(f"\nPreprocessing summary loaded:", 'green')
                print_color(f"  └─ Summary file: {Fore.MAGENTA + Style.BRIGHT}{summary_files[0]}", 'green')
                
                # Extract dataset name from summary
                dataset_name = Path(preprocessing_summary['preprocessing_summary']['input_file']).stem
                
                print_color(f"\nDataset information:", 'green')
                print_color(f"  ├─ Dataset name: {Fore.YELLOW + Style.BRIGHT}{dataset_name}", 'green')
                print_color(f"  └─ Input file: {Fore.MAGENTA + Style.BRIGHT}{preprocessing_summary['preprocessing_summary']['input_file']}", 'green')
                
                # Look for preprocessed dataset CSV
                csv_pattern = f"{dataset_name}_preprocessed_dataset.csv"
                csv_files = list(preprocessing_run_dir.glob(f"*{csv_pattern}*"))

                # Look for preprocessed dataset PKL
                pkl_pattern = f"{dataset_name}_preprocessing_artifacts.pkl"
                pkl_files = list(preprocessing_run_dir.glob(f"*{pkl_pattern}*"))
                
                if csv_files and pkl_files:
                    csv_path = str(csv_files[0])
                    pkl_path = str(pkl_files[0])
                    
                    print_color(f"\nPreprocessing outputs found:", 'green')
                    print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'green')
                    print_color(f"  └─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'green')
                    
                    # Try to find corresponding test run configuration
                    test_run_number = preprocessing_summary['preprocessing_summary'].get('run_number')
                    test_run_id = preprocessing_summary['preprocessing_summary'].get('run_id')
                    
                    if test_run_number:
                        test_run_dir = Path(base_results_dir) / f"run_{test_run_number:03d}"
                        test_summary_file = test_run_dir / f"max_rows_testing_summary_run_{test_run_number:03d}.json"
                        
                        if test_summary_file.exists():
                            with open(test_summary_file) as f:
                                test_config = json.load(f)
                            
                            print_color(f"\nCorresponding test run found:", 'green')
                            print_color(f"  ├─ Test run number: {Fore.YELLOW + Style.BRIGHT}{test_run_number}", 'green')
                            print_color(f"  ├─ Test run ID: {Fore.YELLOW + Style.BRIGHT}{test_run_id}", 'green')
                            print_color(f"  └─ Test summary: {Fore.MAGENTA + Style.BRIGHT}{test_summary_file}", 'green')
                    
                    return csv_path, pkl_path, test_config, preprocessing_summary
                
                elif csv_files:
                    csv_path = str(csv_files[0])
                    print_color(f"\nWarning: CSV file found but PKL artifacts missing", 'yellow')
                    print_color(f"  └─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'yellow')
                    return csv_path, None, test_config, preprocessing_summary
                
                elif pkl_files:
                    pkl_path = str(pkl_files[0])
                    print_color(f"\nWarning: PKL artifacts found but CSV file missing", 'yellow')
                    print_color(f"  └─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'yellow')
                    return None, pkl_path, test_config, preprocessing_summary
                
                else:
                    print_color(f"\nError: No preprocessing outputs found in directory", 'red')
                    print_color(f"  ├─ Searched for CSV pattern: {Fore.YELLOW + Style.BRIGHT}*{csv_pattern}*", 'red')
                    print_color(f"  └─ Searched for PKL pattern: {Fore.YELLOW + Style.BRIGHT}*{pkl_pattern}*", 'red')
                    return None, None, test_config, preprocessing_summary
            else:
                print_color(f"\nError: No preprocessing summary file found", 'red')
                print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{preprocessing_run_dir}", 'red')
                return None, None, None, None
        
        # Check if path is a specific preprocessing summary file
        elif config_file.exists() and config_file.is_file() and "_preprocessing_summary_" in config_file.name:
            # Load preprocessing summary directly
            with open(config_file) as f:
                preprocessing_summary = json.load(f)
            print_color("\nPREPROCESSING OUTPUTS LOCATOR", 'magenta')
            print_color("-" * 50, 'magenta')
            print_color(f"Processing preprocessing summary file directly:", 'magenta')
            print_color(f"  └─ File: {Fore.MAGENTA + Style.BRIGHT}{config_file}", 'magenta')
            print_color("-" * 50, 'magenta')
            
            # Extract output directory from summary
            output_dir = preprocessing_summary['preprocessing_summary']['output_directory']
            preprocessing_run_dir = Path(output_dir)
            
            print_color(f"\nOutput directory from summary:", 'green')
            print_color(f"  └─ Directory: {Fore.WHITE + Style.BRIGHT}{preprocessing_run_dir}", 'green')
            
            # Look for preprocessed dataset CSV
            dataset_name = Path(preprocessing_summary['preprocessing_summary']['input_file']).stem
            csv_pattern = f"{dataset_name}_preprocessed_dataset.csv"
            csv_files = list(preprocessing_run_dir.glob(f"*{csv_pattern}*"))

            # Look for preprocessed dataset PKL
            pkl_pattern = f"{dataset_name}_preprocessing_artifacts.pkl"
            pkl_files = list(preprocessing_run_dir.glob(f"*{pkl_pattern}*"))
            
            if csv_files and pkl_files:
                csv_path = str(csv_files[0])
                pkl_path = str(pkl_files[0])
                
                print_color(f"\nPreprocessing outputs found:", 'green')
                print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'green')
                print_color(f"  └─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'green')
                
                # Try to find corresponding test run configuration
                test_run_number = preprocessing_summary['preprocessing_summary'].get('run_number')
                if test_run_number:
                    test_run_dir = Path(base_results_dir) / f"run_{test_run_number:03d}"
                    test_summary_file = test_run_dir / f"max_rows_testing_summary_run_{test_run_number:03d}.json"
                    
                    if test_summary_file.exists():
                        with open(test_summary_file) as f:
                            test_config = json.load(f)
                        
                        print_color(f"\nCorresponding test run found:", 'green')
                        print_color(f"  └─ Test summary: {Fore.MAGENTA + Style.BRIGHT}{test_summary_file}", 'green')
                
                return csv_path, pkl_path, test_config, preprocessing_summary
            
            elif csv_files:
                csv_path = str(csv_files[0])
                print_color(f"\nWarning: CSV file found but PKL artifacts missing", 'yellow')
                print_color(f"  └─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'yellow')
                return csv_path, None, test_config, preprocessing_summary
            
            elif pkl_files:
                pkl_path = str(pkl_files[0])
                print_color(f"\nWarning: PKL artifacts found but CSV file missing", 'yellow')
                print_color(f"  └─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'yellow')
                return None, pkl_path, test_config, preprocessing_summary
            
            else:
                print_color(f"\nError: No preprocessing outputs found in directory", 'red')
                print_color(f"  ├─ Directory: {Fore.WHITE + Style.BRIGHT}{preprocessing_run_dir}", 'red')
                print_color(f"  ├─ CSV pattern: {Fore.YELLOW + Style.BRIGHT}*{csv_pattern}*", 'red')
                print_color(f"  └─ PKL pattern: {Fore.YELLOW + Style.BRIGHT}*{pkl_pattern}*", 'red')
                return None, None, test_config, preprocessing_summary
        
        # If we get here, we haven't found anything
        print_color("\nCOULD NOT LOCATE PREPROCESSING OUTPUTS", 'red')
        
        print_color("\nPossible reasons:", 'yellow')
        print_color("  ├─ No preprocessing has been run yet", 'yellow')
        print_color("  ├─ The provided path doesn't contain valid run data", 'yellow')
        print_color("  ├─ Directory structure doesn't match expected pattern", 'yellow')
        print_color("  └─ Permissions issues accessing directories", 'yellow')
        
        print_color("\nExpected directory structure:", 'cyan')
        print_color("  ├─ Test runs: {base_results_dir}/run_001/", 'white')
        print_color("  ├─ Preprocessing runs: {base_preprocessing_dir}/run_001/", 'white')
        print_color("  └─ Current path provided: {config_path}", 'white')
        
        if interactive:
            # Offer to browse directories
            response = input(Fore.YELLOW + Style.BRIGHT + "\nBrowse for preprocessing outputs manually? (Y/n): " + Style.RESET_ALL).strip().lower()
            
            if response in ['y', 'yes', '']:
                print_color("\nLooking for preprocessing directories...", 'cyan')
                
                preprocessing_dir = Path(base_preprocessing_dir)
                if preprocessing_dir.exists():
                    preprocessing_runs = sorted(
                        [d for d in preprocessing_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                        key=lambda x: int(x.name.split('_')[1]) if len(x.name.split('_')) > 1 and x.name.split('_')[1].isdigit() else 0,
                        reverse=True
                    )
                    
                    if preprocessing_runs:
                        print_color(f"\nFound {len(preprocessing_runs)} preprocessing runs:", 'green')
                        for i, run_dir in enumerate(preprocessing_runs[:10], 1):
                            # Check what's inside each run
                            csv_files = list(run_dir.glob("*_preprocessed_dataset.csv"))
                            pkl_files = list(run_dir.glob("*_preprocessing_artifacts.pkl"))
                            summary_files = list(run_dir.glob("*_preprocessing_summary_*.json"))
                            
                            csv_status = "CSV: Found" if csv_files else "CSV: Missing"
                            pkl_status = "PKL: Found" if pkl_files else "PKL: Missing"
                            summary_status = "Summary: Found" if summary_files else "Summary: Missing"
                            
                            print_color(f"{i:2d}. {run_dir.name} [{csv_status}, {pkl_status}, {summary_status}]", 'white')
                        
                        if len(preprocessing_runs) > 10:
                            print_color(f"\n... and {len(preprocessing_runs) - 10} more runs", 'cyan')
                        
                        choice = input(Fore.YELLOW + Style.BRIGHT + f"\nSelect preprocessing run (1-{min(10, len(preprocessing_runs))}, 0 to cancel): " + Style.RESET_ALL).strip()
                        
                        if choice.isdigit() and 1 <= int(choice) <= min(10, len(preprocessing_runs)):
                            selected_run_dir = preprocessing_runs[int(choice) - 1]
                            print_color(f"\nSelected run: {selected_run_dir.name}", 'green')
                            return get_preprocessing_outputs(
                                config_path=str(selected_run_dir),
                                base_results_dir=base_results_dir,
                                base_preprocessing_dir=base_preprocessing_dir,
                                interactive=False  # Don't recurse
                            )
                        else:
                            print_color("\nInvalid selection or cancelled", 'yellow')
                    else:
                        print_color(f"\nNo preprocessing runs found in directory:", 'red')
                        print_color(f"  └─ {preprocessing_dir}", 'red')
                else:
                    print_color(f"\nPreprocessing directory does not exist:", 'red')
                    print_color(f"  └─ {preprocessing_dir}", 'red')
        
        return None, None, None, None
    
    except Exception as e:
        print_color(f"\nError locating preprocessing outputs:", 'red')
        print_color(f"  ├─ Error type: {type(e).__name__}", 'red')
        print_color(f"  └─ Error message: {str(e)}", 'red')
        return None, None, None, None

def check_preprocessing_outputs(
    logger: logging.Logger,
    verbose: Optional[bool] = False,
    strict: bool = False,
    min_csv_size: int = 1024,
    min_pkl_size: int = 128,
    validate_csv: bool = True,
    validate_pickle: bool = True
) -> bool:
    """Verify preprocessing outputs with optional validation with progress tracking.
    
    Args:
        logger: Configured logger instance for logging messages
        strict: Enable content validation (default: False)
        min_csv_size: Minimum CSV file size in bytes
        min_pkl_size: Minimum pickle file size in bytes
        validate_csv: Perform CSV content checks
        validate_pickle: Perform pickle structure checks
        
    Returns:
        bool: True if all files exist (and are valid in strict mode)
        
    Raises:
        RuntimeWarning: For suspicious but accepted files
    """
    # Track validation progress and statistics
    validation_stats = {
        'stage': 'Initializing',
        'files_checked': 0,
        'files_passed': 0,
        'files_failed': 0,
        'warnings_issued': 0,
        'validation_stages_passed': 0,
        'validation_stages_failed': 0,
        'detailed_timings': {},
        'detailed_results': {},
        'success_rate': 0.0
    }
    
    # Initialize return values
    config_path = None
    base_results_dir = None
    base_preprocessing_dir = None
    csv_path_return = None
    pkl_path_return = None
    
    # Use get_preprocessing_outputs to locate files
    try:
        print_color(f"\nStarting preprocessing outputs validation...", 'yellow')
        
        validation_stats['stage'] = 'Locating preprocessing outputs'

        # Get preprocessing outputs using the helper function
        csv_path, pkl_path, test_config, preprocessing_summary = get_preprocessing_outputs(
            config_path=config_path,
            base_results_dir=base_results_dir,
            base_preprocessing_dir=base_preprocessing_dir,
            interactive=True
        )
        
        # Store the paths for return
        csv_path_return = csv_path
        pkl_path_return = pkl_path
        
        # Check if we found the required files
        if not csv_path and not pkl_path:
            print_color("\nError: Could not locate preprocessing outputs", 'red')
            print_color("Validation cannot proceed without locating files", 'red')
            return False, None, None
        
        # Prepare required_files dictionary based on located files
        required_files = {}
        if csv_path:
            required_files["preprocessed_dataset.csv"] = {
                "path": Path(csv_path),
                "min_size": min_csv_size,
                "checks": ["header", "delimiter"] if validate_csv else [],
                "description": "Preprocessed dataset CSV",
                "validation_stages": ["existence", "size", "header", "format", "sample_data"] if validate_csv else ["existence", "size"]
            }
        
        if pkl_path:
            required_files["preprocessing_artifacts.pkl"] = {
                "path": Path(pkl_path),
                "min_size": min_pkl_size,
                "required_keys": ["feature_names", "scaler"] if validate_pickle else [],
                "description": "Preprocessing artifacts pickle",
                "validation_stages": ["existence", "size", "load", "keys", "structure"] if validate_pickle else ["existence", "size"]
            }
        
        if not required_files:
            print_color("\nError: No preprocessing files to validate", 'red')
            return False, None, None
        
        all_valid = True
        
        validation_mode = Fore.RED + Style.BRIGHT + 'STRICT' + Style.RESET_ALL if strict else Fore.YELLOW + Style.BRIGHT + 'BASIC' + Style.RESET_ALL
        csv_validation = Fore.GREEN + Style.BRIGHT + 'ENABLED' + Style.RESET_ALL if validate_csv else Fore.RED + Style.BRIGHT + 'DISABLED' + Style.RESET_ALL
        pkl_validation = Fore.GREEN + Style.BRIGHT + 'ENABLED' + Style.RESET_ALL if validate_pickle else Fore.RED + Style.BRIGHT + 'DISABLED' + Style.RESET_ALL

        print_color(f"\nValidation mode: {validation_mode}", 'cyan')
        print_color(f"CSV validation: {csv_validation}", 'cyan')
        print_color(f"Pickle validation: {pkl_validation}", 'cyan')
        if csv_path:
            print_color(f"CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'cyan')
        if pkl_path:
            print_color(f"PKL file: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'cyan')
        
        try:
            # Progress helper: define all stage titles
            titles = [
                "Initializing Validation System",
                "File Validation Loop",
                "Finalization and Reporting"
            ]
            progress = ProgressHelper(titles)
            
            # STAGE 1: Initialization and setup
            with progress.bar("Initializing Validation System", total=2, unit="steps") as init_bar:
                
                # STAGE 1.1
                init_bar.text = "Configuring validation parameters..."
                config_start = time.time()
                
                # Calculate total work units for progress tracking
                total_files = len(required_files)
                total_stages = sum(len(file_info['validation_stages']) for file_info in required_files.values())
                
                # Initialize tracking for each file
                for filename, file_info in required_files.items():
                    validation_stats['detailed_results'][filename] = {
                        'stages_passed': [],
                        'stages_failed': [],
                        'warnings': [],
                        'errors': [],
                        'file_size': 0,
                        'validation_time': 0,
                        'file_path': str(file_info['path'])
                    }
                
                config_time = time.time() - config_start
                validation_stats['detailed_timings']['configuration'] = config_time
                init_bar.text = f"Validation system configured ({total_files} files, {total_stages} stages)"
                init_bar()
                
                # STAGE 1.2
                init_bar.text = "Preparing file validation queues..."
                queue_start = time.time()
                
                # Create a list of validation tasks
                validation_tasks = []
                for filename, requirements in required_files.items():
                    for stage in requirements['validation_stages']:
                        validation_tasks.append((filename, stage))
                
                queue_time = time.time() - queue_start
                validation_stats['detailed_timings']['queue_preparation'] = queue_time
                init_bar.text = f"Validation queues prepared ({len(validation_tasks)} tasks)"
                init_bar()
            
            # STAGE 2: File validation loop
            with progress.bar("File Validation Loop", total=len(validation_tasks), unit="stages") as file_bar:
                
                # Process each file validation task
                for filename, stage_name in validation_tasks:
                    requirements = required_files[filename]
                    filepath = requirements['path']
                    
                    # Track if this is the first stage for this file
                    is_first_stage = stage_name == requirements['validation_stages'][0]
                    if is_first_stage:
                        validation_stats['files_checked'] += 1
                        file_start_time = time.time()
                        file_bar.text = f"Checking {requirements['description'][:20]} at {filepath.name}"
                    
                    # STAGE 2.1: File existence check
                    if stage_name == 'existence':
                        file_bar.text = f"Checking existence: {filename}"
                        existence_start = time.time()
                        
                        if not filepath.exists():
                            error_msg = f"Missing required file: {filename}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('existence')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            validation_stats['files_failed'] += 1
                            file_bar.text = f"Missing: {filename}"
                        else:
                            validation_stats['detailed_results'][filename]['stages_passed'].append('existence')
                            validation_stats['validation_stages_passed'] += 1
                            file_bar.text = f"Exists: {filename}"
                        
                        existence_time = time.time() - existence_start
                        validation_stats['detailed_timings'][f'{filename}_existence'] = existence_time
                        file_bar()
                        continue
                    
                    # Skip remaining stages if file doesn't exist
                    if 'existence' in validation_stats['detailed_results'][filename]['stages_failed']:
                        validation_stats['detailed_results'][filename]['stages_failed'].append(stage_name)
                        validation_stats['validation_stages_failed'] += 1
                        file_bar.text = f"Skipping {stage_name}: {filename}"
                        file_bar()
                        continue
                    
                    # STAGE 2.2: File size validation
                    if stage_name == 'size':
                        file_bar.text = f"Checking size: {filename}"
                        size_start = time.time()
                        
                        file_size = filepath.stat().st_size
                        
                        if file_size >= 1024**3:
                            file_size_display = f"{file_size / 1024**3:.2f} GB"
                        elif file_size >= 1024**2:
                            file_size_display = f"{file_size / 1024**3:.2f} MB"
                        elif file_size >= 1024:
                            file_size_display = f"{file_size / 1024**3:.2f} KB"
                        else:
                            file_size_display = f"{file_size} bytes"
                        
                        validation_stats['detailed_results'][filename]['file_size'] = file_size
                        validation_stats['detailed_results'][filename]['file_size_display'] = file_size_display
                        
                        if file_size < requirements["min_size"]:
                            msg = f"File appears small: {filename} ({file_size_display})"
                            if strict:
                                validation_stats['detailed_results'][filename]['errors'].append(msg)
                                validation_stats['detailed_results'][filename]['stages_failed'].append('size')
                                validation_stats['validation_stages_failed'] += 1
                                all_valid = False
                                file_bar.text = f"Size issue: {file_size_display}"
                            else:
                                validation_stats['detailed_results'][filename]['warnings'].append(msg)
                                validation_stats['detailed_results'][filename]['stages_passed'].append('size')
                                validation_stats['validation_stages_passed'] += 1
                                validation_stats['warnings_issued'] += 1
                                file_bar.text = f"Small file: {file_size_display}"
                        else:
                            validation_stats['detailed_results'][filename]['stages_passed'].append('size')
                            validation_stats['validation_stages_passed'] += 1
                            file_bar.text = f"Size OK: {file_size_display}"
                        
                        size_time = time.time() - size_start
                        validation_stats['detailed_timings'][f'{filename}_size'] = size_time
                        file_bar()
                        continue
                    
                    # Skip content validation in non-strict mode
                    if not strict:
                        validation_stats['files_passed'] += 1
                        file_bar.text = f"Basic check passed: {filename}"
                        file_bar()
                        continue
                    
                    # STAGE 2.3: CSV header validation
                    if stage_name == 'header' and filename.endswith('.csv') and validate_csv:
                        file_bar.text = f"Checking CSV header: {filename}"
                        header_start = time.time()
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                header = f.readline()
                                if not header.strip():
                                    error_msg = f"Empty CSV file: {filename}"
                                    validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                                    validation_stats['detailed_results'][filename]['stages_failed'].append('header')
                                    validation_stats['validation_stages_failed'] += 1
                                    all_valid = False
                                    file_bar.text = f"Empty CSV: {filename}"
                                else:
                                    validation_stats['detailed_results'][filename]['stages_passed'].append('header')
                                    validation_stats['validation_stages_passed'] += 1
                                    file_bar.text = f"Header OK: {filename}"
                        except UnicodeDecodeError:
                            error_msg = f"Invalid CSV encoding: {filename}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('header')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            file_bar.text = f"Encoding error: {filename}"
                        except Exception as e:
                            error_msg = f"CSV header check failed: {filename} - {str(e)}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('header')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            file_bar.text = f"Header check failed: {filename}"
                        
                        header_time = time.time() - header_start
                        validation_stats['detailed_timings'][f'{filename}_header'] = header_time
                        file_bar()
                        continue
                    
                    # STAGE 2.4: CSV format validation
                    if stage_name == 'format' and filename.endswith('.csv') and validate_csv:
                        file_bar.text = f"Checking CSV format: {filename}"
                        format_start = time.time()
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                header = f.readline()
                                if len(header.split(',')) < 2:
                                    error_msg = f"Invalid CSV format in: {filename}"
                                    validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                                    validation_stats['detailed_results'][filename]['stages_failed'].append('format')
                                    validation_stats['validation_stages_failed'] += 1
                                    all_valid = False
                                    file_bar.text = f"Format error: {filename}"
                                else:
                                    validation_stats['detailed_results'][filename]['stages_passed'].append('format')
                                    validation_stats['validation_stages_passed'] += 1
                                    file_bar.text = f"Format OK: {filename}"
                        except Exception as e:
                            error_msg = f"CSV format check failed: {filename} - {str(e)}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('format')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            file_bar.text = f"Format check failed: {filename}"
                        
                        format_time = time.time() - format_start
                        validation_stats['detailed_timings'][f'{filename}_format'] = format_time
                        file_bar()
                        continue
                    
                    # STAGE 2.5: CSV sample data validation
                    if stage_name == 'sample_data' and filename.endswith('.csv') and validate_csv:
                        file_bar.text = f"Checking sample data: {filename}"
                        sample_start = time.time()
                        
                        try:
                            sample_df = pd.read_csv(filepath, nrows=10)
                            if sample_df.empty:
                                error_msg = f"CSV contains no data: {filename}"
                                validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                                validation_stats['detailed_results'][filename]['stages_failed'].append('sample_data')
                                validation_stats['validation_stages_failed'] += 1
                                all_valid = False
                                file_bar.text = f"No data: {filename}"
                            else:
                                validation_stats['detailed_results'][filename]['stages_passed'].append('sample_data')
                                validation_stats['validation_stages_passed'] += 1
                                file_bar.text = f"Sample data OK: {filename}"
                        except Exception as e:
                            error_msg = f"CSV sample read failed: {filename} - {str(e)}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('sample_data')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            file_bar.text = f"Sample read failed: {filename}"
                        
                        sample_time = time.time() - sample_start
                        validation_stats['detailed_timings'][f'{filename}_sample_data'] = sample_time
                        file_bar()
                        continue
                    
                    # STAGE 2.6: Pickle load validation
                    if stage_name == 'load' and filename.endswith('.pkl') and validate_pickle:
                        file_bar.text = f"Loading pickle: {filename}"
                        load_start = time.time()
                        
                        try:
                            with open(filepath, 'rb') as f:
                                data = joblib.load(f)
                                validation_stats['detailed_results'][filename]['stages_passed'].append('load')
                                validation_stats['validation_stages_passed'] += 1
                                file_bar.text = f"Pickle loaded: {filename}"
                        except Exception as e:
                            error_msg = f"Pickle load failed: {filename} - {str(e)}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('load')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            file_bar.text = f"Load failed: {filename}"
                        
                        load_time = time.time() - load_start
                        validation_stats['detailed_timings'][f'{filename}_load'] = load_time
                        file_bar()
                        continue
                    
                    # STAGE 2.7: Pickle key validation
                    if stage_name == 'keys' and filename.endswith('.pkl') and validate_pickle:
                        file_bar.text = f"Checking required keys: {filename}"
                        keys_start = time.time()
                        
                        try:
                            with open(filepath, 'rb') as f:
                                data = joblib.load(f)
                                missing_keys = []
                                for key in requirements["required_keys"]:
                                    if key not in data:
                                        missing_keys.append(key)
                                
                                if missing_keys:
                                    error_msg = f"Missing keys {missing_keys} in: {filename}"
                                    validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                                    validation_stats['detailed_results'][filename]['stages_failed'].append('keys')
                                    validation_stats['validation_stages_failed'] += 1
                                    all_valid = False
                                    file_bar.text = f"Missing keys: {filename}"
                                else:
                                    validation_stats['detailed_results'][filename]['stages_passed'].append('keys')
                                    validation_stats['validation_stages_passed'] += 1
                                    file_bar.text = f"Keys OK: {filename}"
                        except Exception as e:
                            error_msg = f"Key validation failed: {filename} - {str(e)}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('keys')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            file_bar.text = f"Key check failed: {filename}"
                        
                        keys_time = time.time() - keys_start
                        validation_stats['detailed_timings'][f'{filename}_keys'] = keys_time
                        file_bar()
                        continue
                    
                    # STAGE 2.8: Pickle structure validation
                    if stage_name == 'structure' and filename.endswith('.pkl') and validate_pickle:
                        file_bar.text = f"Checking structure: {filename}"
                        structure_start = time.time()
                        
                        try:
                            with open(filepath, 'rb') as f:
                                data = joblib.load(f)
                                
                                structure_valid = True
                                
                                # Validate feature_names
                                if "feature_names" in data:
                                    if not isinstance(data["feature_names"], list) or len(data["feature_names"]) == 0:
                                        error_msg = f"Invalid feature_names in: {filename}"
                                        validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                                        validation_stats['detailed_results'][filename]['stages_failed'].append('structure')
                                        validation_stats['validation_stages_failed'] += 1
                                        all_valid = False
                                        structure_valid = False
                                
                                # Validate scaler
                                if "scaler" in data and data["scaler"] is not None:
                                    if not hasattr(data["scaler"], "transform"):
                                        error_msg = f"Invalid scaler object in: {filename}"
                                        validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                                        validation_stats['detailed_results'][filename]['stages_failed'].append('structure')
                                        validation_stats['validation_stages_failed'] += 1
                                        all_valid = False
                                        structure_valid = False
                                
                                if structure_valid:
                                    validation_stats['detailed_results'][filename]['stages_passed'].append('structure')
                                    validation_stats['validation_stages_passed'] += 1
                                    file_bar.text = f"Structure OK: {filename}"
                                else:
                                    validation_stats['detailed_results'][filename]['stages_failed'].append('structure')
                                    validation_stats['validation_stages_failed'] += 1
                                    file_bar.text = f"Structure invalid: {filename}"
                        
                        except Exception as e:
                            error_msg = f"Structure validation failed: {filename} - {str(e)}"
                            validation_stats['detailed_results'][filename]['errors'].append(error_msg)
                            validation_stats['detailed_results'][filename]['stages_failed'].append('structure')
                            validation_stats['validation_stages_failed'] += 1
                            all_valid = False
                            file_bar.text = f"Structure check failed: {filename}"
                        
                        structure_time = time.time() - structure_start
                        validation_stats['detailed_timings'][f'{filename}_structure'] = structure_time
                        file_bar()
                        continue
                    
                    # Update file validation time if this is the last stage for this file
                    if stage_name == requirements['validation_stages'][-1]:
                        validation_stats['detailed_results'][filename]['validation_time'] = time.time() - file_start_time
                        
                        # Determine final file status
                        if all_valid or (not strict and filepath.exists()):
                            validation_stats['files_passed'] += 1
                            file_bar.text = f"Validation passed: {filename}"
                        else:
                            validation_stats['files_failed'] += 1
                            file_bar.text = f"Validation failed: {filename}"
            
            # STAGE 3: Finalization and reporting
            with progress.bar("Finalization and Reporting", total=2, unit="steps") as final_bar:
                
                # STAGE 3.1
                final_bar.text = "Generating validation report..."
                report_start = time.time()
                
                # Calculate overall statistics
                total_stages_passed = validation_stats['validation_stages_passed']
                total_stages_failed = validation_stats['validation_stages_failed']
                total_stages = total_stages_passed + total_stages_failed
                success_rate = (total_stages_passed / total_stages) * 100 if total_stages > 0 else 0
                validation_stats['success_rate'] = success_rate
                
                # Display summary
                print_color(f"\nPreprocessing outputs validation completed:", 'green')
                print_color(f"  ├─ Files checked: {Fore.YELLOW + Style.BRIGHT}{validation_stats['files_checked']}", 'green')
                print_color(f"  ├─ Files passed: {Fore.CYAN + Style.BRIGHT}{validation_stats['files_passed']}", 'green')
                print_color(f"  ├─ Files failed: {Fore.RED + Style.BRIGHT}{validation_stats['files_failed']}", 'green')
                print_color(f"  ├─ Validation stages: {Fore.YELLOW + Style.BRIGHT}{total_stages}", 'green')
                print_color(f"  │   ├─ Passed stages: {Fore.CYAN + Style.BRIGHT}{total_stages_passed} passed", 'green')
                print_color(f"  │   └─ Failed stages: {Fore.RED + Style.BRIGHT}{total_stages_failed} failed", 'green')
                print_color(f"  ├─ Overall success rate: {Fore.CYAN + Style.BRIGHT}{success_rate:.1f}%", 'green')
                print_color(f"  └─ Warnings issued: {Fore.YELLOW + Style.BRIGHT}{validation_stats['warnings_issued']}", 'green')
                
                # Display detailed results for each file
                for filename, results in validation_stats['detailed_results'].items():
                    status = Fore.GREEN + Style.BRIGHT + "PASSED" if len(results['stages_failed']) == 0 else Fore.RED + Style.BRIGHT + "FAILED"
                    stage_count = len(results['stages_passed']) + len(results['stages_failed'])
                    print_color(f"File checked: {Fore.MAGENTA + Style.BRIGHT}{filename} {status}", 'green')
                    print_color(f"  ├─ File size: {Fore.CYAN + Style.BRIGHT}{results['file_size_display']}", 'green')
                    print_color(f"  ├─ Passed stages: {Fore.YELLOW + Style.BRIGHT}{len(results['stages_passed'])}/{stage_count} stages", 'green')
                    print_color(f"  └─ Validation time: {Fore.CYAN + Style.BRIGHT}{results['validation_time']:.2f}s", 'green')
                    
                    # Display individual stage results for failed files
                    if results['stages_failed']:
                        print_color(f"Failed stages: {Fore.RED + Style.BRIGHT}{', '.join(results['stages_failed'])}", 'green')
                    if results['warnings']:
                        for warning in results['warnings']:
                            print_color(f"Warning: {Fore.YELLOW + Style.BRIGHT}{warning}", 'green')
                    if results['errors']:
                        for error in results['errors']:
                            print_color(f"Error: {Fore.RED + Style.BRIGHT}{error}", 'green')
                
                report_time = time.time() - report_start
                validation_stats['detailed_timings']['report_generation'] = report_time
                final_bar.text = "Validation report generated"
                final_bar()
                
                # STAGE 3.2
                final_bar.text = "Finalizing validation process..."
                finalize_start = time.time()
                if all_valid:
                    final_bar.text = "All outputs validated successfully"
                    print_color("\nAll preprocessing outputs validated successfully", 'green')
                else:
                    final_bar.text = "Validation failed"
                    print_color("\nPreprocessing outputs validation failed", 'red')
                
                # Log detailed timings in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Validation timings:")
                    for stage, timing in validation_stats['detailed_timings'].items():
                        logger.debug(f"├─ {stage}: {timing:.3f}s")
                    total_time = sum(validation_stats['detailed_timings'].values())
                    logger.debug(f"└─ Total validation time: {total_time:.3f}s")
                
                finalize_time = time.time() - finalize_start
                validation_stats['detailed_timings']['finalization'] = finalize_time
                final_bar()
        
        except Exception as e:
            error_context = f"(stage: {validation_stats.get('stage', 'unknown')})"
            logger.error(f"Validation process failed {error_context}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            logger.error(f"Validation process failed at stage: {validation_stats.get('stage', 'unknown')}")
            logger.error(f"Error: {str(e)}")
            
            raise RuntimeError("Preprocessing outputs validation failed") from e
        
        return all_valid, csv_path_return, pkl_path_return
        
    except Exception as e:
        print_color(f"\nError during preprocessing outputs validation:", 'red')
        print_color(f"  ├─ Error type: {type(e).__name__}", 'red')
        print_color(f"  └─ Error message: {str(e)}", 'red')
        logger.error(f"Preprocessing outputs validation failed: {str(e)}")
        return False, None, None

def run_preprocessing(
    logger: logging.Logger,
    verbose: Optional[bool] = False,
    timeout_minutes: float = 30.0,
    cleanup: bool = False,
    strict_output_check: bool = True,
    reproducible: bool = True,
    debug: bool = False
) -> bool:
    """Execute preprocessing with controls and progress tracking.
    
    Args:
        logger: Configured logger instance for logging messages
        verbose: Enable detailed logging
        timeout_minutes: Maximum runtime in minutes
        cleanup: Remove existing output files
        strict_output_check: Use strict validation
        reproducible: Set PYTHONHASHSEED for reproducibility
        debug: Enable verbose debugging output
    
    Returns:
        bool: True if preprocessing succeeded
    
    Raises:
        RuntimeError: For unrecoverable failures
        FileNotFoundError: If script is missing
    """
    # Track preprocessing progress and statistics
    preprocessing_stats = {
        'stage': 'Initializing',
        'start_time': time.time(),
        'current_step': 0,
        'total_steps': 6,
        'successful_steps': 0,
        'failed_steps': 0,
        'warnings_issued': 0,
        'files_cleaned': 0,
        'output_lines_processed': 0,
        'error_lines_found': 0,
        'warning_lines_found': 0,
        'execution_time': 0,
        'memory_usage_mb': 0,
        'detailed_timings': {}
    }
    
    # Initialize return values
    csv_path_return = None
    pkl_path_return = None

    try:
        print_color("\n" + "-" * 50, 'magenta')
        print_color("PREPROCESSING PIPELINE", 'magenta')
        print_color("-" * 50, 'magenta')
        print_color(f"  ├─ Timeout: {Fore.GREEN + Style.BRIGHT}{timeout_minutes} minutes", 'magenta')
        print_color(f"  ├─ Cleanup: {Fore.GREEN + Style.BRIGHT}{'Enabled' if cleanup else 'Disabled'}", 'magenta')
        print_color(f"  ├─ Strict validation: {Fore.GREEN + Style.BRIGHT}{'Enabled' if strict_output_check else 'Disabled'}", 'magenta')
        print_color(f"  └─ Reproducible: {Fore.GREEN + Style.BRIGHT}{'Enabled' if reproducible else 'Disabled'}", 'magenta')
        
        # Progress helper: define all stage titles
        titles = [
            "Initial Setup and Validation",
            "Execution Phase",
            "Post-Processing and Validation",
            "Final Summary"
        ]
        progress = ProgressHelper(titles)

        # STAGE 1: Initial Setup and Validation
        with progress.bar("Initial Setup and Validation", total=3, unit="steps") as setup_bar:
            
            # STAGE 1.1: Script Validation
            setup_bar.text = "Validating preprocessing script..."
            script_validation_start = time.time()
            
            preprocessing_stats['stage'] = "Script Validation"
            preprocessing_stats['current_step'] = 1
            
            base_dir = Path(__file__).resolve().parent
            script_path = Path(base_dir / "preprocessor.py")
            
            if not script_path.exists():
                error_msg = f"Preprocessing script not found at {script_path.absolute()}"
                preprocessing_stats['failed_steps'] += 1
                setup_bar.text = "Script not found"
                print_color(f"\nError: {error_msg}", 'red')
                raise FileNotFoundError(f"preprocessor.py not found at {script_path.absolute()}")
            
            script_validation_time = time.time() - script_validation_start
            preprocessing_stats['detailed_timings']['script_validation'] = script_validation_time
            preprocessing_stats['successful_steps'] += 1
            setup_bar.text = "Script validated"
            setup_bar()
            
            # STAGE 1.2: Cleanup Preparation
            setup_bar.text = "Preparing cleanup..."
            cleanup_prep_start = time.time()
            
            preprocessing_stats['stage'] = "Cleanup Preparation"
            preprocessing_stats['current_step'] = 2
            
            files_to_cleanup = []
            
            if cleanup:
                setup_bar.text = "Checking existing preprocessing output files..."
                
                # Use get_preprocessing_outputs to locate the output files
                csv_path, pkl_path, test_config, preprocessing_summary = get_preprocessing_outputs(
                    config_path=None,
                    base_results_dir=None,
                    base_preprocessing_dir=None,
                    interactive=True
                )
                
                # Add CSV file to cleanup list if found
                if csv_path:
                    csv_file = Path(csv_path)
                    if csv_file.exists():
                        files_to_cleanup.append(csv_file)
                        setup_bar.text = f"Found existing CSV file for cleanup: {csv_file.name}"
                
                # Add PKL file to cleanup list if found
                if pkl_path:
                    pkl_file = Path(pkl_path)
                    if pkl_file.exists():
                        files_to_cleanup.append(pkl_file)
                        setup_bar.text = f"Found existing PKL file for cleanup: {pkl_file.name}"
                
                # Also check default locations as fallback
                output_files = [
                    Path(DATASETS_DIR / "preprocessed_dataset.csv"),
                    Path(DATASETS_DIR / "preprocessing_artifacts.pkl")
                ]
                
                for fpath in output_files:
                    if fpath.exists() and fpath not in files_to_cleanup:
                        files_to_cleanup.append(fpath)
                        setup_bar.text = f"Found additional file for cleanup: {fpath.name}"
            
            cleanup_prep_time = time.time() - cleanup_prep_start
            preprocessing_stats['detailed_timings']['cleanup_prep'] = cleanup_prep_time
            preprocessing_stats['successful_steps'] += 1
            setup_bar.text = f"Cleanup prepared ({len(files_to_cleanup)} files)"
            setup_bar()
            
            # STAGE 1.3: Environment Setup
            setup_bar.text = "Setting up environment..."
            env_setup_start = time.time()
            
            preprocessing_stats['stage'] = "Environment Setup"
            preprocessing_stats['current_step'] = 3
            
            env = os.environ.copy()
            timeout_seconds = int(timeout_minutes * 60)
            
            if reproducible:
                env["PYTHONHASHSEED"] = "42"
                setup_bar.text = "Set PYTHONHASHSEED=42 (reproducible)"
            else:
                setup_bar.text = "Environment configured (standard)"
            
            env_setup_time = time.time() - env_setup_start
            preprocessing_stats['detailed_timings']['env_setup'] = env_setup_time
            preprocessing_stats['successful_steps'] += 1
            setup_bar.text = "Environment configured"
            setup_bar()

        # STAGE 2: Execution Phase
        with progress.bar("Execution Phase", total=2, unit="steps") as exec_bar:
            
            # STAGE 2.1: File Cleanup
            if cleanup and files_to_cleanup:
                exec_bar.text = "Cleaning up previous outputs..."
                cleanup_start = time.time()
                
                preprocessing_stats['stage'] = "File Cleanup"
                
                for fpath in files_to_cleanup:
                    try:
                        if fpath.exists():
                            fpath.unlink(missing_ok=True)
                            preprocessing_stats['files_cleaned'] += 1
                            exec_bar.text = f"Removed: {fpath.name}"
                    except Exception as e:
                        preprocessing_stats['warnings_issued'] += 1
                        print_color(f"\nWarning: Failed to remove {fpath.name}: {e}", 'yellow')
                
                cleanup_time = time.time() - cleanup_start
                preprocessing_stats['detailed_timings']['file_cleanup'] = cleanup_time
                exec_bar.text = f"Cleanup completed ({preprocessing_stats['files_cleaned']} files)"
                exec_bar()
            else:
                exec_bar.text = "No cleanup required"
                exec_bar()
            
            # STAGE 2.2: Script Execution
            exec_bar.text = "Executing preprocessing script..."
            execution_start = time.time()
            
            preprocessing_stats['stage'] = "Script Execution"
            preprocessing_stats['current_step'] = 4
            
            # Execution statistics tracking
            execution_stats = {
                'start_time': time.time(),
                'stdout_lines': 0,
                'stderr_lines': 0,
                'progress_indicators': 0,
                'current_operation': 'Starting execution...'
            }
            
            try:
                # Start the subprocess
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Real-time output processing
                stdout_lines = []
                stderr_lines = []
                start_time = time.time()
                timeout_time = start_time + timeout_seconds
                
                # Function to update progress based on output
                def update_progress_from_output(line: str):
                    line_lower = line.lower()
                    
                    # Progress indicators
                    if any(indicator in line_lower for indicator in ['progress', 'completed', 'step', 'processing', 'loading', 'saving']):
                        execution_stats['progress_indicators'] += 1
                    
                    # Update current operation
                    if 'loading' in line_lower:
                        execution_stats['current_operation'] = 'Loading data'
                    elif 'processing' in line_lower:
                        execution_stats['current_operation'] = 'Processing features'
                    elif 'saving' in line_lower:
                        execution_stats['current_operation'] = 'Saving outputs'
                    elif 'cleaning' in line_lower:
                        execution_stats['current_operation'] = 'Data cleaning'
                    elif 'normalizing' in line_lower:
                        execution_stats['current_operation'] = 'Feature normalization'
                    
                    # Update progress bar text
                    exec_bar.text = f"{execution_stats['current_operation']} ({execution_stats['progress_indicators']} steps)"
                
                # Read output in real-time
                while process.poll() is None:
                    # Check timeout
                    if time.time() > timeout_time:
                        process.terminate()
                        raise subprocess.TimeoutExpired(script_path, timeout_seconds)
                    
                    # Read stdout
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        stdout_lines.append(stdout_line)
                        execution_stats['stdout_lines'] += 1
                        preprocessing_stats['output_lines_processed'] += 1
                        
                        line = stdout_line.strip()
                        if line:
                            if line.startswith("ERROR"):
                                preprocessing_stats['error_lines_found'] += 1
                                print_color(f"\nError: {line}", 'red')
                            elif line.startswith("WARNING"):
                                preprocessing_stats['warning_lines_found'] += 1
                                preprocessing_stats['warnings_issued'] += 1
                                print_color(f"\nWarning: {line}", 'yellow')
                            else:
                                print_color(f"\nPassed: {line[:200]}", 'green')
                            
                            update_progress_from_output(line)
                    
                    # Read stderr
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        stderr_lines.append(stderr_line)
                        execution_stats['stderr_lines'] += 1
                        print_color(f"\nSTDERR: {stderr_line.strip()}", 'red')
                    
                    time.sleep(0.1)
                
                # Get remaining output
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout_lines.extend(remaining_stdout.splitlines())
                if remaining_stderr:
                    stderr_lines.extend(remaining_stderr.splitlines())
                
                # Check return code
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, script_path, ''.join(stdout_lines), ''.join(stderr_lines))
                
                # Execution completed successfully
                execution_time = time.time() - execution_start
                preprocessing_stats['execution_time'] = execution_time
                preprocessing_stats['detailed_timings']['script_execution'] = execution_time
                
                exec_bar.text = f"Execution completed ({execution_time:.1f}s)"
                exec_bar()
            
            except subprocess.TimeoutExpired:
                execution_time = time.time() - execution_start
                preprocessing_stats['execution_time'] = execution_time
                preprocessing_stats['detailed_timings']['script_execution'] = execution_time
                exec_bar.text = f"Timeout after {execution_time:.1f}s"
                exec_bar()
                raise
            
            except Exception as e:
                execution_time = time.time() - execution_start
                preprocessing_stats['execution_time'] = execution_time
                preprocessing_stats['detailed_timings']['script_execution'] = execution_time
                exec_bar.text = f"Execution failed"
                exec_bar()
                raise
        
        # STAGE 3: Post-Processing and Validation
        with progress.bar("Post-Processing and Validation", total=2, unit="steps") as post_bar:
            
            # STAGE 3.1: Output Processing
            post_bar.text = "Processing execution results..."
            output_processing_start = time.time()
            
            preprocessing_stats['stage'] = "Post-Processing"
            preprocessing_stats['current_step'] = 5
            
            # Log information about located outputs
            if csv_path:
                csv_file = Path(csv_path)
                if csv_file.exists():
                    file_size = csv_file.stat().st_size
            
            if pkl_path:
                pkl_file = Path(pkl_path)
                if pkl_file.exists():
                    file_size = pkl_file.stat().st_size
            
            output_processing_time = time.time() - output_processing_start
            preprocessing_stats['detailed_timings']['output_processing'] = output_processing_time
            preprocessing_stats['successful_steps'] += 1
            post_bar.text = f"Output processed"
            post_bar()
            
            # STAGE 3.2: Output Validation
            post_bar.text = "Validating preprocessing outputs..."
            validation_start = time.time()
            
            validation_passed = check_preprocessing_outputs(
                logger=logger,
                strict=strict_output_check
            )
            
            validation_time = time.time() - validation_start
            preprocessing_stats['detailed_timings']['output_validation'] = validation_time
            
            if not validation_passed:
                preprocessing_stats['failed_steps'] += 1
                post_bar.text = f"Validation failed"
                post_bar()
                print_color(f"\nError: Output validation failed", 'red')
                return False, None, None
            
            preprocessing_stats['successful_steps'] += 1
            post_bar.text = f"Validation passed"
            post_bar()

        # STAGE 4: Final Summary
        with progress.bar("Final Summary", total=2, unit="steps") as summary_bar:
            
            # STAGE 4.1: Statistics Calculation
            summary_bar.text = "Calculating final statistics..."
            stats_start = time.time()
            
            preprocessing_stats['stage'] = "Completion"
            preprocessing_stats['current_step'] = 6
            
            total_time = time.time() - preprocessing_stats['start_time']
            preprocessing_stats['detailed_timings']['total'] = total_time
            
            stats_time = time.time() - stats_start
            preprocessing_stats['detailed_timings']['stats_calculation'] = stats_time
            preprocessing_stats['successful_steps'] += 1
            summary_bar.text = "Statistics calculated"
            summary_bar()
            
            # STAGE 4.2: Final Reporting
            summary_bar.text = "Generating final report..."
            report_start = time.time()
            
            # Log summary
            print_color("\nPreprocessing completed successfully!", 'green')
            print_color(f"  ├─ Total time: {Fore.YELLOW + Style.BRIGHT}{total_time:.2f}s", 'green')
            print_color(f"  ├─ Steps completed: {Fore.CYAN + Style.BRIGHT}{preprocessing_stats['successful_steps']}/{preprocessing_stats['total_steps']}", 'green')
            print_color(f"  ├─ Execution time: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['execution_time']:.2f}s", 'green')
            print_color(f"  ├─ Files cleaned: {Fore.CYAN + Style.BRIGHT}{preprocessing_stats['files_cleaned']}", 'green')
            print_color(f"  ├─ Output lines: {Fore.BLUE + Style.BRIGHT}{preprocessing_stats['output_lines_processed']}", 'green')
            print_color(f"  ├─ Warnings: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['warning_lines_found']}", 'green')
            print_color(f"  └─ Errors: {Fore.RED + Style.BRIGHT}{preprocessing_stats['error_lines_found']}", 'green')
            
            # Get final output locations for summary
            csv_path, pkl_path, test_config, preprocessing_summary = get_preprocessing_outputs(
                config_path=None,
                base_results_dir=None,
                base_preprocessing_dir=None,
                interactive=False
            )
            
            # Log output file locations
            if csv_path:
                csv_file = Path(csv_path)
                if csv_file.exists():
                    file_size = csv_file.stat().st_size
                    print_color(f"\nPreprocessing outputs:", 'green')
                    print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'green')
                    print_color(f"  │   └─ Size: {Fore.YELLOW + Style.BRIGHT}{file_size} bytes", 'green')
            
            if pkl_path:
                pkl_file = Path(pkl_path)
                if pkl_file.exists():
                    file_size = pkl_file.stat().st_size
                    print_color(f"  └─ PKL file: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'green')
                    print_color(f"      └─ Size: {Fore.YELLOW + Style.BRIGHT}{file_size} bytes", 'green')
            
            # Log detailed timings in debug mode
            if debug:
                print_color(f"\nDetailed timings:", 'green')
                timing_items = list(preprocessing_stats['detailed_timings'].items())
                for i, (stage, timing) in enumerate(timing_items):
                    prefix = "  ├─ " if i < len(timing_items) - 1 else "  └─ "
                    print_color(f"{prefix}{stage}: {Fore.WHITE + Style.BRIGHT}{timing:.2f}s", 'green')
            
            report_time = time.time() - report_start
            preprocessing_stats['detailed_timings']['final_report'] = report_time
            preprocessing_stats['successful_steps'] += 1
            summary_bar.text = f"Preprocessing pipeline completed"
            summary_bar()
        
        return True, csv_path_return, pkl_path_return

    except subprocess.TimeoutExpired:
        preprocessing_stats['failed_steps'] += 1
        preprocessing_stats['stage'] = "Timeout"
        print_color(f"\nPreprocessing statistics error after timeout: {Fore.YELLOW + Style.BRIGHT}{timeout_minutes} minutes", 'red')
        print_color(f"  ├─ Stage: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['stage']}", 'red')
        print_color(f"  ├─ Steps completed: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['successful_steps']}/{preprocessing_stats['total_steps']}", 'red')
        print_color(f"  ├─ Execution time: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats.get('execution_time', 0):.2f}s", 'red')
        print_color(f"  └─ Output lines processed: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['output_lines_processed']}", 'red')
        
        return False, None, None

    except subprocess.CalledProcessError as e:
        preprocessing_stats['failed_steps'] += 1
        preprocessing_stats['stage'] = "Execution Error"
        
        print_color(f"\nProcess failed with exit code {Fore.YELLOW + Style.BRIGHT}{e.returncode}", 'red')
        print_color(f"  ├─ Error output:", 'red')
        # Show last 20 lines
        for line in e.stderr.splitlines()[-20:]:
            # Truncate very long lines
            if len(line) > 200:
                print_color(f"  │   └─ {line[:200]}...", 'red')
            else:
                print_color(f"  │   └─ {line}", 'red')
        
        print_color(f"  ├─ Failed at stage: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['stage']}", 'red')
        print_color(f"  └─ Steps completed: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['successful_steps']}/{preprocessing_stats['total_steps']}", 'red')
        
        return False, None, None

    except Exception as e:
        preprocessing_stats['failed_steps'] += 1
        preprocessing_stats['stage'] = "Unexpected Error"
        
        print_color(f"\nUnexpected error: {Fore.YELLOW + Style.BRIGHT}{type(e).__name__}", 'red')
        print_color(f"  ├─ Error details: {Fore.YELLOW + Style.BRIGHT}{str(e)}", 'red')
        print_color(f"  ├─ Failed at stage: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['stage']}", 'red')
        print_color(f"  └─ Steps completed: {Fore.YELLOW + Style.BRIGHT}{preprocessing_stats['successful_steps']}/{preprocessing_stats['total_steps']}", 'red')
        
        if debug:
            print_color(f"\nStack trace:", 'red')
            logger.error(traceback.format_exc())
        
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
            "3. Enable debug mode for details"
        ]
    }
    logger.warning("Troubleshooting steps:")
    for step in guides.get(error_type, guides["unexpected"]):
        logger.warning(f"  {step}")

def log_error_output(logger: logging.Logger, stderr: str, use_color: bool) -> None:
    """Log error output with proper formatting and truncation."""
    red = Fore.RED if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    logger.error(f"{red}Error Output{reset}")
    # Show last 20 lines
    for line in stderr.splitlines()[-20:]:
        # Truncate very long lines
        if len(line) > 200:
            logger.error(f"{red}{line[:200]}...{reset}")
        else:
            logger.error(f"{red}{line}{reset}")

def display_data_loading_header(filepath: str) -> None:
    """Display data loading header."""
    print_color("\nDATA PREPROCESSING PIPELINE", 'magenta')
    print_color("-" * 50, 'magenta')
    print_color(f"Preprocessed dataset:", 'magenta')
    print_color(f"  └─ {Fore.YELLOW + Style.BRIGHT}{filepath}", 'magenta')
    print_color("-" * 50, 'magenta')

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
    console.print("\n")
    table = Table(
        title="Class Distribution Analysis",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold yellow",
        title_justify="left",
        border_style="cyan",
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
    filepath: Path = None,
    strict: bool = True,
    use_color: bool = True,
    required_keys: List[str] = None,
    validate_scaler: bool = True,
    verbose: Optional[bool] = False,
    debug: bool = False
) -> Dict:
    """Load preprocessing artifacts with validation and progress tracking.
    
    Args:
        filepath: Path to artifacts file
        strict: Enable strict validation
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
    # Determine filepath (default: script's directory / models/preprocessing_artifacts.pkl)
    if filepath is None:
        filepath = Path(DATASETS_DIR / "preprocessing_artifacts.pkl")

    # Setup styling
    red = Fore.RED if use_color else ""
    yellow = Fore.YELLOW if use_color else ""
    green = Fore.GREEN if use_color else ""
    blue = Fore.BLUE if use_color else ""
    cyan = Fore.CYAN if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    
    # Default required keys
    if required_keys is None:
        required_keys = ["feature_names", "scaler"]
    
    # Track loading progress and statistics
    loading_stats = {
        'stage': 'Initializing',
        'current_step': 0,
        'total_steps': 6,
        'successful_steps': 0,
        'failed_steps': 0,
        'warnings_issued': 0,
        'validation_passed': 0,
        'validation_failed': 0,
        'file_size': 0,
        'load_time': 0,
        'validation_time': 0,
        'detailed_timings': {},
        'artifacts_loaded': False,
        'feature_count': 0,
        'scaler_type': None
    }
    
    try:
        # Progress helper: define all stage titles
        titles = [
            "Initial Setup and File Validation",
            "Finalization and Result Preparation"
        ]
        progress = ProgressHelper(titles)
        
        # STAGE 1: Initial Setup and File Validation
        with progress.bar("Initial Setup and File Validation", total=6, unit="steps") as load_bar:
            
            # STAGE 1.1: File Existence Check
            load_bar.text = "Checking file existence..."
            file_check_start = time.time()
            
            loading_stats['stage'] = "File Check"
            loading_stats['current_step'] = 1
            
            if not filepath.exists():
                error_msg = f"Artifacts file not found: {filepath}"
                loading_stats['failed_steps'] += 1
                load_bar.text = f"{red}File not found{reset}"
                raise FileNotFoundError(error_msg)
            
            # Get file size
            loading_stats['file_size'] = filepath.stat().st_size
            file_check_time = time.time() - file_check_start
            loading_stats['detailed_timings']['file_check'] = file_check_time
            loading_stats['successful_steps'] += 1
            load_bar.text = f"{green}File found ({loading_stats['file_size']} bytes){reset}"
            load_bar()
            
            # STAGE 1.2: File Loading
            load_bar.text = "Loading artifacts file..."
            load_start = time.time()
            
            loading_stats['stage'] = "Loading"
            loading_stats['current_step'] = 2
            
            artifacts = None
            
            try:
                # Load with warning suppression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    if verbose and use_color:
                        logger.info(f"{blue}Loading artifacts from {filepath}{reset}")
                    elif verbose:
                        logger.info(f"Loading artifacts from {filepath}")
                    
                    artifacts = joblib.load(filepath)
                    loading_stats['artifacts_loaded'] = True
                    
            except Exception as load_error:
                loading_stats['failed_steps'] += 1
                load_bar.text = f"{red}Load failed{reset}"
                load_bar()
                raise RuntimeError(f"Failed to load artifacts file: {str(load_error)}") from load_error
            
            load_time = time.time() - load_start
            loading_stats['load_time'] = load_time
            loading_stats['detailed_timings']['file_load'] = load_time
            loading_stats['successful_steps'] += 1
            load_bar.text = f"{green}File loaded ({load_time:.2f}s){reset}"
            load_bar()
            
            # STAGE 1.3: Type Validation
            load_bar.text = "Validating artifact structure..."
            type_validation_start = time.time()
            
            loading_stats['stage'] = "Type Validation"
            loading_stats['current_step'] = 3
            
            if not isinstance(artifacts, dict):
                loading_stats['failed_steps'] += 1
                load_bar.text = f"{red}Invalid structure{reset}"
                load_bar()
                raise ValueError("Artifacts must be a dictionary")
            
            type_validation_time = time.time() - type_validation_start
            loading_stats['detailed_timings']['type_validation'] = type_validation_time
            loading_stats['successful_steps'] += 1
            loading_stats['validation_passed'] += 1
            load_bar.text = f"{green}Valid dictionary structure{reset}"
            load_bar()
            
            # STAGE 1.4: Key Validation
            load_bar.text = "Checking required keys..."
            key_validation_start = time.time()
            
            loading_stats['stage'] = "Key Validation"
            loading_stats['current_step'] = 4
            
            missing_keys = [k for k in required_keys if k not in artifacts]
            
            if missing_keys:
                loading_stats['failed_steps'] += 1
                loading_stats['validation_failed'] += 1
                load_bar.text = f"{red}Missing keys{reset}"
                load_bar()
                raise KeyError(f"Missing required keys: {missing_keys}")
            
            key_validation_time = time.time() - key_validation_start
            loading_stats['detailed_timings']['key_validation'] = key_validation_time
            loading_stats['successful_steps'] += 1
            loading_stats['validation_passed'] += 1
            load_bar.text = f"{green}All required keys present{reset}"
            load_bar()
            
            # STAGE 1.5: Scaler Validation
            load_bar.text = "Validating scaler object..."
            scaler_validation_start = time.time()
            
            loading_stats['stage'] = "Scaler Validation"
            loading_stats['current_step'] = 5
            
            if validate_scaler and "scaler" in artifacts:
                scaler = artifacts["scaler"]
                valid_scalers = (MinMaxScaler, StandardScaler)
                
                if not isinstance(scaler, valid_scalers):
                    loading_stats['failed_steps'] += 1
                    loading_stats['validation_failed'] += 1
                    load_bar.text = f"{red}Invalid scaler type{reset}"
                    load_bar()
                    raise TypeError(f"Invalid scaler type: {type(scaler).__name__}")
                
                loading_stats['scaler_type'] = type(scaler).__name__
                scaler_validation_time = time.time() - scaler_validation_start
                loading_stats['detailed_timings']['scaler_validation'] = scaler_validation_time
                loading_stats['successful_steps'] += 1
                loading_stats['validation_passed'] += 1
                load_bar.text = f"{green}Valid scaler ({loading_stats['scaler_type']}){reset}"
            else:
                loading_stats['scaler_type'] = 'None'
                loading_stats['successful_steps'] += 1
                load_bar.text = f"{yellow}Scaler validation skipped{reset}"
                loading_stats['warnings_issued'] += 1
            
            load_bar()
            
            # STAGE 1.6: Feature Names Processing
            load_bar.text = "Processing feature names..."
            feature_processing_start = time.time()
            
            loading_stats['stage'] = "Feature Processing"
            loading_stats['current_step'] = 6
            
            # Version-aware feature names
            feature_names = artifacts["feature_names"]
            
            if hasattr(artifacts.get("scaler"), 'feature_names_in_'):
                feature_names = artifacts["scaler"].feature_names_in_.tolist()
                if verbose:
                    if use_color:
                        logger.info(f"{blue}Using scaler-derived feature names{reset}")
                    else:
                        logger.info("Using scaler-derived feature names")
                load_bar.text = f"{green}Using scaler feature names{reset}"
            else:
                load_bar.text = f"{green}Using artifact feature names{reset}"
            
            loading_stats['feature_count'] = len(feature_names)
            feature_processing_time = time.time() - feature_processing_start
            loading_stats['detailed_timings']['feature_processing'] = feature_processing_time
            loading_stats['successful_steps'] += 1
            loading_stats['validation_passed'] += 1
            load_bar.text = f"{green}Features processed ({loading_stats['feature_count']} features){reset}"
            load_bar()
            
            # Final progress update for stage 1
            load_bar.text = f"{green}Artifact validation completed{reset}"

        # STAGE 2: Finalization and Result Preparation
        with progress.bar("Finalization and Result Preparation", total=2, unit="steps") as final_bar:
            
            # STAGE 2.1: Prepare Result Dictionary
            final_bar.text = "Preparing result dictionary..."
            result_prep_start = time.time()
            
            loading_stats['stage'] = "Finalization"
            
            # Prepare return dict
            result = {
                "feature_names": feature_names,
                "scaler": artifacts.get("scaler"),
                "chunk_size": artifacts.get("chunk_size", 100000),
                "loading_stats": loading_stats  # Include stats for debugging
            }
            
            result_prep_time = time.time() - result_prep_start
            loading_stats['detailed_timings']['result_preparation'] = result_prep_time
            loading_stats['successful_steps'] += 1
            final_bar.text = f"{green}Result dictionary prepared{reset}"
            final_bar()
            
            # STAGE 2.2: Generate Summary Report
            final_bar.text = "Generating summary report..."
            summary_start = time.time()
            
            total_time = sum(loading_stats['detailed_timings'].values())
            loading_stats['detailed_timings']['total'] = total_time
            
            if verbose:
                if use_color:
                    logger.info(f"{green}Artifacts loaded successfully{reset}")
                    logger.info(f"{blue}Loading Summary:{reset}")
                else:
                    logger.info("Artifacts loaded successfully")
                    logger.info("Loading Summary:")
                
                logger.info(f"  - File: {filepath.name}")
                logger.info(f"  - Size: {loading_stats['file_size']} bytes")
                logger.info(f"  - Load time: {loading_stats['load_time']:.2f}s")
                logger.info(f"  - Total time: {total_time:.2f}s")
                logger.info(f"  - Steps completed: {loading_stats['successful_steps']}/{loading_stats['total_steps']}")
                logger.info(f"  - Validations passed: {loading_stats['validation_passed']}")
                logger.info(f"  - Feature count: {loading_stats['feature_count']}")
                logger.info(f"  - Scaler type: {loading_stats['scaler_type']}")
                
                if debug:
                    if use_color:
                        logger.info(f"{blue}Detailed timings:{reset}")
                    else:
                        logger.info("Detailed timings:")
                    for stage, timing in loading_stats['detailed_timings'].items():
                        logger.info(f"  - {stage}: {timing:.2f}s")
                
                if use_color:
                    logger.info(f"{blue}Artifacts loaded successfully: {list(result.keys())}{reset}")
                else:
                    logger.info(f"Artifacts loaded successfully: {list(result.keys())}")
            
            summary_time = time.time() - summary_start
            loading_stats['detailed_timings']['summary_generation'] = summary_time
            loading_stats['successful_steps'] += 1
            final_bar.text = f"{green}Loading completed successfully{reset}"
            final_bar()
        
        return result
        
    except FileNotFoundError as e:
        if use_color:
            logger.error(f"{red}Error: Artifacts file not found: {filepath}{reset}")
        else:
            logger.error(f"Error: Artifacts file not found: {filepath}")
        
        if debug:
            traceback.print_exc()
        raise FileNotFoundError(f"Artifacts file not found: {filepath}") from e
        
    except Exception as e:
        error_type = type(e).__name__
        
        if use_color:
            logger.error(f"{red}Error: Artifact loading failed ({error_type}): {str(e)}{reset}")
        else:
            logger.error(f"Error: Artifact loading failed ({error_type}): {str(e)}")
        
        # Log failure statistics
        if 'loading_stats' in locals():
            if use_color:
                logger.error(f"{red}Failed at stage: {loading_stats['stage']}{reset}")
                logger.error(f"{red}Steps completed: {loading_stats.get('successful_steps', 0)}/{loading_stats.get('total_steps', 0)}{reset}")
                logger.error(f"{red}Validations passed: {loading_stats.get('validation_passed', 0)}{reset}")
            else:
                logger.error(f"Failed at stage: {loading_stats['stage']}")
                logger.error(f"Steps completed: {loading_stats.get('successful_steps', 0)}/{loading_stats.get('total_steps', 0)}")
                logger.error(f"Validations passed: {loading_stats.get('validation_passed', 0)}")
        
        if strict:
            # Display troubleshooting steps
            if use_color:
                print(f"{yellow}Troubleshooting Steps:{reset}")
                print(f"  {cyan}1.{reset} Verify preprocessing script completed successfully")
                print(f"  {cyan}2.{reset} Check artifact file integrity")
                print(f"  {cyan}3.{reset} Validate required keys: {required_keys}")
                print(f"  {cyan}4.{reset} Check sklearn version compatibility")
                print(f"  {cyan}5.{reset} Ensure file is not corrupted")
            else:
                print("Troubleshooting Steps:")
                print("  1. Verify preprocessing script completed successfully")
                print("  2. Check artifact file integrity")
                print("  3. Validate required keys: {required_keys}")
                print("  4. Check sklearn version compatibility")
                print("  5. Ensure file is not corrupted")
            
            if debug:
                traceback.print_exc()
            
            raise RuntimeError(f"Failed to load artifacts: {str(e)}") from e
        else:
            if use_color:
                logger.warning(f"{yellow}Warning: Using partial artifacts with validation disabled{reset}")
            else:
                logger.warning("Warning: Using partial artifacts with validation disabled")
            
            return {
                "feature_names": [],
                "scaler": None,
                "chunk_size": 100000,
                "loading_stats": loading_stats if 'loading_stats' in locals() else {}
            }

def load_and_clean_data(
    filepath: Path,
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
    verbose: Optional[bool] = False,
    safe_dtype_conversion: bool = True,
    sample_size: int = 10000,
    on_bad_lines: str = 'warn',
    float_precision: str = 'high'
) -> pd.DataFrame:
    """Data loader with validation, dtype handling, and progress tracking.
    
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
    blue = Fore.BLUE if use_color else ""
    cyan = Fore.CYAN if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    
    # Statistics tracking
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
        'clean_samples': 0,
        'processing_stages': {
            'file_validation': 0,
            'dtype_inference': 0,
            'chunk_loading': 0,
            'dtype_conversion': 0,
            'duplicate_removal': 0,
            'nan_handling': 0,
            'value_validation': 0,
            'final_assembly': 0
        },
        'detailed_timings': {},
        'memory_usage_mb': 0
    }
    
    chunks = []
    required_cols = feature_names + [label_col]
    
    try:
        # Progress helper: define all stage titles
        titles = [
            "Initial Setup and File Validation",
            "Chunk Processing",
            "Final Assembly and Validation",
            "Final Summary"
        ]
        progress = ProgressHelper(titles)
        
        # Display header
        display_data_loading_header(filepath)
        
        # STAGE 1: Initial Setup and File Validation
        with progress.bar("Initial Setup and File Validation", total=4, unit="steps") as init_bar:
            
            # STAGE 1.1: File Validation
            init_bar.text = "Validating CSV file structure..."
            file_validation_start = time.time()
            
            try:
                with open(filepath, 'r') as f:
                    header = f.readline().strip().split(',')
                    missing_cols = set(required_cols) - set(header)
                    if missing_cols:
                        raise ValueError(f"Missing required columns: {missing_cols}")
                
                stats['processing_stages']['file_validation'] = 1
                file_validation_time = time.time() - file_validation_start
                stats['detailed_timings']['file_validation'] = file_validation_time
                init_bar.text = f"{green}CSV structure validated{reset}"
            except Exception as e:
                stats['processing_stages']['file_validation'] = 0
                init_bar.text = f"{red}CSV validation failed{reset}"
                init_bar()
                raise RuntimeError(f"CSV validation failed: {str(e)}") from e
            
            init_bar()
            
            # STAGE 1.2: Dtype Inference
            init_bar.text = "Analyzing data types..."
            dtype_inference_start = time.time()
            
            dtypes_map = {}
            try:
                sample_df = pd.read_csv(
                    filepath,
                    nrows=sample_size,
                    usecols=required_cols,
                    engine='c',
                    on_bad_lines='warn',
                    float_precision=float_precision
                )
                
                # Build dtype mapping
                for col in sample_df.columns:
                    if col == label_col:
                        dtypes_map[col] = label_dtype
                    elif col in feature_names:
                        dtypes_map[col] = 'float32'
                
                stats['processing_stages']['dtype_inference'] = 1
                dtype_inference_time = time.time() - dtype_inference_start
                stats['detailed_timings']['dtype_inference'] = dtype_inference_time
                init_bar.text = f"{green}Data types analyzed ({len(dtypes_map)} columns){reset}"
                
                if verbose:
                    if use_color:
                        logger.info(f"{blue}Dtype mapping: {dtypes_map}{reset}")
                    else:
                        logger.info(f"Dtype mapping: {dtypes_map}")
                    
            except Exception as e:
                if use_color:
                    logger.error(f"{yellow}Warning: Dtype inference failed, using safe defaults: {str(e)}{reset}")
                else:
                    logger.warning(f"Warning: Dtype inference failed, using safe defaults: {str(e)}")
                
                dtypes_map = {col: 'float32' for col in feature_names}
                dtypes_map[label_col] = label_dtype
                stats['processing_stages']['dtype_inference'] = 0
                init_bar.text = f"{yellow}Using default data types{reset}"
            
            init_bar()
            
            # STAGE 1.3: File Size Estimation
            init_bar.text = "Estimating file size..."
            estimation_start = time.time()
            
            estimated_chunks = 100  # Conservative default
            try:
                total_rows = sum(1 for _ in open(filepath)) - 1  # Subtract header
                estimated_chunks = (total_rows + chunk_size - 1) // chunk_size
                stats['original'] = total_rows
                init_bar.text = f"{green}File estimated ({total_rows:,} rows, {estimated_chunks} chunks){reset}"
            except Exception as e:
                if use_color:
                    logger.warning(f"{yellow}Warning: Could not estimate file size: {str(e)}{reset}")
                else:
                    logger.warning(f"Warning: Could not estimate file size: {str(e)}")
                init_bar.text = f"{yellow}Using conservative chunk estimate{reset}"
            
            estimation_time = time.time() - estimation_start
            stats['detailed_timings']['file_estimation'] = estimation_time
            init_bar()
            
            # STAGE 1.4: Preparation Complete
            init_bar.text = "Preparation completed"
            init_bar.text = f"{green}Ready to process {estimated_chunks} chunks{reset}"
            init_bar()

        # STAGE 2: Chunk Processing
        with progress.bar("Chunk Processing", total=estimated_chunks, unit="chunks") as chunk_bar:
            
            chunk_processing_start = time.time()
            stats['processing_stages']['chunk_loading'] = 1
            
            # Main loading loop with robust dtype handling
            for chunk_idx, chunk in enumerate(pd.read_csv(
                filepath,
                dtype=dtypes_map,
                usecols=required_cols,
                chunksize=chunk_size,
                engine='c',
                na_values=['nan', 'NaN', 'null', 'NULL', '', 'inf', '-inf'],
                keep_default_na=True,
                on_bad_lines=on_bad_lines,
                float_precision=float_precision
            )):
                chunk_start_time = time.time()
                
                if len(chunk) == 0:
                    stats['bad_lines'] += chunk_size
                    chunk_bar.text = f"{yellow}Empty chunk skipped{reset}"
                    continue
                    
                stats['original'] += len(chunk)
                stats['total_chunks'] += 1
                
                current_chunk = chunk_idx + 1
                chunk_bar.text = f"Processing chunk {current_chunk}/{estimated_chunks} ({len(chunk)} rows)"
                
                # STAGE 2.1: Dtype Conversion
                dtype_conversion_start = time.time()
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
                                    if verbose:
                                        if use_color:
                                            logger.warning(f"{yellow}Partial conversion failure in {col}: {converted.isna().sum()} NA values introduced{reset}")
                                        else:
                                            logger.warning(f"Partial conversion failure in {col}: {converted.isna().sum()} NA values introduced")
                                
                                # Then convert to target dtype
                                chunk[col] = converted.astype(dtypes_map.get(col, 'float32'))
                                stats['dtype_conversions'] += 1
                                
                                if verbose:
                                    if use_color:
                                        logger.info(f"{yellow}Converted {col} via safe method{reset}")
                                    else:
                                        logger.info(f"Converted {col} via safe method")
                            except Exception as inner_e:
                                stats['failed_conversions'] += len(chunk)
                                if use_color:
                                    logger.error(f"{yellow}Warning: Failed to convert {col}: {str(inner_e)}{reset}")
                                else:
                                    logger.warning(f"Warning: Failed to convert {col}: {str(inner_e)}")
                                raise ValueError(f"Critical dtype conversion failed for {col}") from inner_e
                        else:
                            raise ValueError(f"Failed to convert {col} to {dtypes_map.get(col)}") from e
                
                dtype_conversion_time = time.time() - dtype_conversion_start
                stats['processing_stages']['dtype_conversion'] = 1
                
                # STAGE 2.2: Duplicate Removal
                duplicate_removal_start = time.time()
                dup_count = chunk.duplicated().sum()
                stats['duplicates'] += dup_count
                chunk = chunk.drop_duplicates()
                duplicate_removal_time = time.time() - duplicate_removal_start
                stats['processing_stages']['duplicate_removal'] = 1
                
                # STAGE 2.3: NaN Handling
                chunk_bar.text = f"Handling missing values (chunk {chunk_idx+1})..."
                nan_handling_start = time.time()

                # Create an explicit copy to avoid SettingWithCopyWarning
                chunk = chunk.copy()

                # Count rows with any NaN values before processing
                nan_rows = chunk.isna().any(axis=1).sum()
                stats['nan_rows'] += nan_rows

                # Fill NaN values in feature columns using .loc for explicit assignment
                for col in feature_names:
                    col_nans = chunk[col].isna().sum()
                    if col_nans > 0:
                        stats['feature_nans'] += col_nans
                        # Use .loc to ensure we're modifying the DataFrame properly
                        chunk.loc[:, col] = chunk[col].fillna(0)
                
                # Handle NaN values in label column
                label_nans = chunk[label_col].isna().sum()
                if label_nans > 0:
                    stats['label_nans'] += label_nans
                    # Create explicit copy after filtering to avoid chained assignment
                    chunk = chunk.dropna(subset=required_cols).copy()
                else:
                    # Also drop any rows with NaN in required columns (defensive)
                    chunk = chunk.dropna(subset=required_cols).copy()
                
                nan_handling_time = time.time() - nan_handling_start
                stats['processing_stages']['nan_handling'] = 1
                
                # STAGE 2.4: Value Range Validation
                value_validation_start = time.time()
                if not keep_extreme_values:
                    invalid_mask = pd.DataFrame(False, index=chunk.index, columns=feature_names)
                    for col in feature_names:
                        invalid_mask[col] = (chunk[col] > max_value) | (chunk[col] < min_value)
                    
                    invalid_count = invalid_mask.any(axis=1).sum()
                    if invalid_count > 0:
                        stats['invalid_values'] += invalid_count
                        chunk = chunk[~invalid_mask.any(axis=1)]
                
                value_validation_time = time.time() - value_validation_start
                stats['processing_stages']['value_validation'] = 1
                
                stats['cleaned'] += len(chunk)
                
                if len(chunk) > 0:
                    chunks.append(chunk)
                    chunk_bar.text = f"{green}Chunk {current_chunk} cleaned ({len(chunk)} rows kept){reset}"
                else:
                    chunk_bar.text = f"{yellow}Chunk {current_chunk} empty after cleaning{reset}"
                
                # Update chunk processing time
                chunk_processing_time = time.time() - chunk_start_time
                
                # Calculate progress percentage
                progress_percent = (current_chunk / estimated_chunks) * 100
                chunk_bar.text = f"Progress: {current_chunk}/{estimated_chunks} chunks ({progress_percent:.1f}%)"
                chunk_bar()
            
            chunk_processing_time = time.time() - chunk_processing_start
            stats['detailed_timings']['chunk_processing'] = chunk_processing_time
            chunk_bar.text = f"{green}All chunks processed ({stats['total_chunks']} chunks){reset}"

        # STAGE 3: Final Assembly and Validation
        with progress.bar("Final Assembly and Validation", total=2, unit="steps") as final_bar:
            
            # STAGE 3.1: Data Assembly
            final_bar.text = "Assembling final dataset..."
            final_assembly_start = time.time()
            
            if not chunks:
                final_bar.text = f"{red}No valid data remaining{reset}"
                final_bar()
                raise ValueError("No valid data remaining after cleaning")
            
            df = pd.concat(chunks, ignore_index=True)
            stats['clean_samples'] = len(df)
            stats['processing_stages']['final_assembly'] = 1
            
            final_assembly_time = time.time() - final_assembly_start
            stats['detailed_timings']['final_assembly'] = final_assembly_time
            final_bar.text = f"{green}Dataset assembled ({stats['clean_samples']:,} rows){reset}"
            final_bar()
            
            # STAGE 3.2: Memory Optimization
            final_bar.text = "Optimizing memory usage..."
            memory_optimization_start = time.time()
            
            # Calculate memory usage
            stats['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            for col in df.select_dtypes(include=['integer']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            optimized_memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            memory_saved = stats['memory_usage_mb'] - optimized_memory_mb
            stats['memory_usage_mb'] = optimized_memory_mb
            
            memory_optimization_time = time.time() - memory_optimization_start
            stats['detailed_timings']['memory_optimization'] = memory_optimization_time
            final_bar.text = f"{green}Memory optimized ({memory_saved:.1f}MB saved){reset}"
            final_bar()

        # STAGE 4: Final Summary
        with progress.bar("Final Summary", total=2, unit="steps") as summary_bar:
            
            # STAGE 4.1: Statistics Calculation
            summary_bar.text = "Calculating final statistics..."
            stats_start = time.time()
            
            total_time = sum(stats['detailed_timings'].values())
            stats['detailed_timings']['total'] = total_time
            
            stats_time = time.time() - stats_start
            stats['detailed_timings']['statistics_calculation'] = stats_time
            summary_bar.text = f"{green}Statistics calculated{reset}"
            summary_bar()
            
            # STAGE 4.2: Summary Reporting
            summary_bar.text = "Generating final report..."
            report_start = time.time()
            
            # Display summary
            if verbose:
                if use_color:
                    logger.info(f"{green}Data loading and cleaning completed successfully{reset}")
                    logger.info(f"{blue}Processing Summary:{reset}")
                else:
                    logger.info("Data loading and cleaning completed successfully")
                    logger.info("Processing Summary:")
                
                logger.info(f"  - Total time: {total_time:.2f}s")
                logger.info(f"  - Original rows: {stats['original']:,}")
                logger.info(f"  - Clean rows: {stats['clean_samples']:,}")
                logger.info(f"  - Chunks processed: {stats['total_chunks']}")
                logger.info(f"  - Memory usage: {stats['memory_usage_mb']:.1f} MB")
                logger.info(f"  - Data quality metrics:")
                logger.info(f"      Duplicates removed: {stats['duplicates']:,}")
                logger.info(f"      NaN rows handled: {stats['nan_rows']:,}")
                logger.info(f"      Invalid values removed: {stats['invalid_values']:,}")
                logger.info(f"      Dtype conversions: {stats['dtype_conversions']:,}")
            
            if debug:
                if use_color:
                    logger.debug(f"{blue}Detailed Processing Stages:{reset}")
                else:
                    logger.debug("Detailed Processing Stages:")
                
                for stage, status in stats['processing_stages'].items():
                    if use_color:
                        status_str = f"{green}PASS{reset}" if status else f"{red}FAIL{reset}"
                    else:
                        status_str = "PASS" if status else "FAIL"
                    logger.debug(f"  - {stage}: {status_str}")
                
                if use_color:
                    logger.debug(f"{blue}Detailed Timings:{reset}")
                else:
                    logger.debug("Detailed Timings:")
                
                for stage, timing in stats['detailed_timings'].items():
                    logger.debug(f"  - {stage}: {timing:.2f}s")
            
            # Display data validation summary
            display_data_validation_summary(stats)
            
            report_time = time.time() - report_start
            stats['detailed_timings']['final_report'] = report_time
            summary_bar.text = f"{green}Data loading completed successfully{reset}"
            summary_bar()
        
        return df
    
    except FileNotFoundError:
        if use_color:
            logger.error(f"{red}Error: Data file not found: {filepath}{reset}")
        else:
            logger.error(f"Error: Data file not found: {filepath}")
        raise RuntimeError(f"Data file not found: {filepath}") from None
    except pd.errors.EmptyDataError:
        if use_color:
            logger.error(f"{red}Error: CSV file is empty{reset}")
        else:
            logger.error("Error: CSV file is empty")
        raise RuntimeError("CSV file is empty") from None
    except Exception as e:
        if use_color:
            logger.error(f"{red}Error: Data loading failed: {str(e)}{reset}")
        else:
            logger.error(f"Error: Data loading failed: {str(e)}")
        
        if debug:
            traceback.print_exc()
        
        raise RuntimeError("Data loading failed") from e

def auto_select_oversampler(
    results: Dict[str, Dict],
    metric_weights: Optional[Dict[str, float]] = None,
    min_validation_acc: float = 0.7,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Automatically select the best oversampling method based on evaluation metrics
    with progress tracking.
    
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
    # Setup styling for colored output
    red = Fore.RED
    yellow = Fore.YELLOW
    green = Fore.GREEN
    blue = Fore.BLUE
    cyan = Fore.CYAN
    reset = Style.RESET_ALL
    
    # Track selection progress and statistics
    selection_stats = {
        'stage': 'Initializing',
        'total_methods': len(results),
        'valid_methods': 0,
        'filtered_methods': 0,
        'metrics_processed': 0,
        'methods_scored': 0,
        'current_method': None,
        'current_metric': None,
        'detailed_timings': {},
        'scoring_breakdown': {}
    }
    
    # Default metric weights if not provided
    default_weights = {
        'val_acc': 0.5,
        'boundary_violation_rate': -0.3,  # Negative because lower is better
        'silhouette_score': 0.15,
        'feature_correlation_diff': -0.05  # Negative because lower is better
    }
    weights = metric_weights or default_weights
    
    try:
        # Progress helper: define all stage titles
        titles = [
            "Initial Setup and Method Filtering",
            "Method Scoring",
            "Final Selection and Reporting",
            "Final Summary"
        ]
        progress = ProgressHelper(titles)
        
        # STAGE 1: Initial Setup and Method Filtering
        with progress.bar("Initial Setup and Method Filtering", total=4, unit="steps") as init_bar:
            
            # STAGE 1.1: Input Validation
            init_bar.text = "Validating input results"
            validation_start = time.time()
            
            selection_stats['stage'] = "Input Validation"
            
            if not results:
                init_bar.text = "No results provided"
                init_bar()
                raise ValueError("No evaluation results provided for oversampler selection")
            
            selection_stats['total_methods'] = len(results)
            validation_time = time.time() - validation_start
            selection_stats['detailed_timings']['input_validation'] = validation_time
            init_bar.text = f"Validated {len(results)} methods"
            init_bar()
            
            # STAGE 1.2: Method Filtering
            init_bar.text = "Filtering methods by accuracy threshold"
            filtering_start = time.time()
            
            selection_stats['stage'] = "Method Filtering"
            
            # Filter methods that meet minimum accuracy threshold
            valid_methods = {}
            filtered_methods = []
            
            for method, metrics in results.items():
                if metrics.get('error'):
                    filtered_methods.append(method)
                    if verbose:
                        logger.error(f"Filtering {method}: has error flag")
                elif metrics.get('val_acc', 0) >= min_validation_acc:
                    valid_methods[method] = metrics
                else:
                    filtered_methods.append(method)
                    if verbose:
                        logger.info(f"Filtering {method}: val_acc {metrics.get('val_acc', 0):.3f} < {min_validation_acc}")
            
            selection_stats['valid_methods'] = len(valid_methods)
            selection_stats['filtered_methods'] = len(filtered_methods)
            
            if not valid_methods:
                init_bar.text = "No valid methods after filtering"
                init_bar()
                raise ValueError(f"No methods met minimum validation accuracy of {min_validation_acc}")
            
            filtering_time = time.time() - filtering_start
            selection_stats['detailed_timings']['method_filtering'] = filtering_time
            init_bar.text = f"Filtered to {len(valid_methods)} valid methods"
            init_bar()
            
            # STAGE 1.3: Metric Statistics Collection
            init_bar.text = "Collecting metric statistics"
            stats_collection_start = time.time()
            
            selection_stats['stage'] = "Statistics Collection"
            
            metric_stats = {}
            selection_stats['metrics_processed'] = len(weights)
            
            # First pass to collect stats for normalization
            for metric in weights.keys():
                selection_stats['current_metric'] = metric
                values = [m.get(metric, np.nan) for m in valid_methods.values()]
                metric_stats[metric] = {
                    'min': np.nanmin(values),
                    'max': np.nanmax(values),
                    'mean': np.nanmean(values),
                    'valid_count': sum(1 for v in values if not np.isnan(v))
                }
                
                if verbose:
                    logger.info(f"Metric {metric}: min={metric_stats[metric]['min']:.3f}, max={metric_stats[metric]['max']:.3f}, valid={metric_stats[metric]['valid_count']}/{len(values)}")
            
            stats_collection_time = time.time() - stats_collection_start
            selection_stats['detailed_timings']['stats_collection'] = stats_collection_time
            init_bar.text = f"Collected stats for {len(weights)} metrics"
            init_bar()
            
            # STAGE 1.4: Weight Normalization
            init_bar.text = "Normalizing metric weights"
            normalization_start = time.time()
            
            selection_stats['stage'] = "Weight Normalization"
            
            # Ensure weights sum to 1
            weight_sum = sum(abs(w) for w in weights.values())
            if weight_sum != 1.0:
                if verbose:
                    logger.info(f"Normalizing weights from {weight_sum:.3f} to 1.0")
                weights = {k: v / weight_sum for k, v in weights.items()}
            
            normalization_time = time.time() - normalization_start
            selection_stats['detailed_timings']['weight_normalization'] = normalization_time
            init_bar.text = "Weights normalized"
            init_bar()

        # STAGE 2: Method Scoring
        method_scores = {}
        selection_stats['methods_scored'] = 0
        
        with progress.bar("Method Scoring", total=len(valid_methods), unit="methods") as scoring_bar:
            
            scoring_start = time.time()
            selection_stats['stage'] = "Method Scoring"
            
            # Score calculation for each method
            for method_idx, (method, metrics) in enumerate(valid_methods.items()):
                selection_stats['current_method'] = method
                selection_stats['methods_scored'] = method_idx + 1
                
                scoring_bar.text = f"Scoring {method}"
                method_scoring_start = time.time()
                
                score = 0
                details = {}
                
                for metric, weight in weights.items():
                    selection_stats['current_metric'] = metric
                    raw_value = metrics.get(metric, metric_stats[metric]['mean'])
                    
                    # Handle NaN values by using metric average
                    if np.isnan(raw_value):
                        raw_value = metric_stats[metric]['mean']
                        if verbose:
                            logger.info(f"Using mean value for {metric} in {method} due to NaN")
                    
                    # Normalize value between 0-1 (except for negative weights)
                    if weight > 0:  # Higher is better
                        norm_value = ((raw_value - metric_stats[metric]['min']) / (metric_stats[metric]['max'] - metric_stats[metric]['min'] + 1e-10))
                    else:  # Lower is better (negative weight)
                        norm_value = 1 - ((raw_value - metric_stats[metric]['min']) / (metric_stats[metric]['max'] - metric_stats[metric]['min'] + 1e-10))
                    
                    # Ensure normalized value is in valid range
                    norm_value = max(0.0, min(1.0, norm_value))
                    
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
                
                method_scoring_time = time.time() - method_scoring_start
                scoring_bar.text = f"Scored {method}: {score:.3f}"
                scoring_bar()
            
            scoring_time = time.time() - scoring_start
            selection_stats['detailed_timings']['method_scoring'] = scoring_time
            scoring_bar.text = "All methods scored"

        # STAGE 3: Final Selection and Reporting
        with progress.bar("Final Selection and Reporting", total=2, unit="steps") as final_bar:
            
            # STAGE 3.1: Best Method Selection
            final_bar.text = "Selecting best method"
            selection_start = time.time()
            
            selection_stats['stage'] = "Best Method Selection"
            
            # Select method with highest score
            best_method = max(method_scores.items(), key=lambda x: x[1]['total_score'])[0]
            best_score = method_scores[best_method]['total_score']
            
            selection_time = time.time() - selection_start
            selection_stats['detailed_timings']['final_selection'] = selection_time
            final_bar.text = f"Selected {best_method} (score: {best_score:.3f})"
            final_bar()
            
            # STAGE 3.2: Results Compilation
            final_bar.text = "Compiling results"
            compilation_start = time.time()
            
            selection_stats['stage'] = "Results Compilation"
            
            # Prepare detailed results
            scoring_details = {
                'method_scores': method_scores,
                'metric_weights': weights,
                'filtered_methods': set(results.keys()) - set(valid_methods.keys()),
                'selection_stats': selection_stats
            }
            
            compilation_time = time.time() - compilation_start
            selection_stats['detailed_timings']['results_compilation'] = compilation_time
            final_bar.text = "Results compiled"
            final_bar()

        # STAGE 4: Final Summary
        with progress.bar("Final Summary", total=2, unit="steps") as summary_bar:
            
            # STAGE 4.1: Statistics Calculation
            summary_bar.text = "Calculating final statistics"
            stats_start = time.time()
            
            selection_stats['stage'] = "Statistics Calculation"
            
            total_time = sum(selection_stats['detailed_timings'].values())
            selection_stats['detailed_timings']['total'] = total_time
            
            stats_time = time.time() - stats_start
            selection_stats['detailed_timings']['stats_calculation'] = stats_time
            summary_bar.text = "Statistics calculated"
            summary_bar()
            
            # STAGE 4.2: Summary Reporting
            summary_bar.text = "Generating final report"
            report_start = time.time()
            
            selection_stats['stage'] = "Summary Reporting"
            
            if verbose:
                # Print selection report
                logger.info(f"\nOversampler Selection Report")
                logger.info(f"Selection Summary:")
                logger.info(f"  - Total methods evaluated: {selection_stats['total_methods']}")
                logger.info(f"  - Valid methods after filtering: {selection_stats['valid_methods']}")
                logger.info(f"  - Methods filtered out: {selection_stats['filtered_methods']}")
                logger.info(f"  - Total selection time: {total_time:.3f}s")
                logger.info(f"  - Selected method: {best_method} (score: {best_score:.3f})")
                
                # Print scoring breakdown table
                logger.info(f"Method Scoring Breakdown:")
                header = f"{'Method':<20} {'Total Score':<12}"
                for metric in weights:
                    header += f" {metric:<15}"
                logger.info(header)
                logger.info("-" * (20 + 12 + len(weights) * 15))
                
                for method, scores in method_scores.items():
                    row = f"{method:<20} {scores['total_score']:<12.3f}"
                    for metric in weights:
                        raw_val = scores['details'][metric]['raw']
                        row += f" {raw_val:<15.3f}"
                    if method == best_method:
                        logger.info(f"{row}")
                    else:
                        logger.info(row)
                
                # Print metric weights
                logger.info(f"Metric Weights Used:")
                for metric, weight in weights.items():
                    direction = "higher better" if weight > 0 else "lower better"
                    logger.info(f"  - {metric}: {abs(weight):.3f} ({direction})")
                
                # Print detailed timings in verbose mode
                logger.info(f"Detailed Timings:")
                for stage, timing in selection_stats['detailed_timings'].items():
                    logger.info(f"  - {stage}: {timing:.3f}s")
            
            report_time = time.time() - report_start
            selection_stats['detailed_timings']['final_report'] = report_time
            summary_bar.text = "Oversampler selection completed"
            summary_bar()
        
        return best_method, scoring_details
        
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {selection_stats.get('stage', 'unknown')}, "
        error_context += f"method: {selection_stats.get('current_method', 'none')}, "
        error_context += f"metric: {selection_stats.get('current_metric', 'none')})"
        
        logger.error(f"Oversampler selection failed{error_context}: {str(e)}")
        
        # Log selection statistics for debugging
        if verbose:
            logger.error(f"Selection statistics:")
            logger.error(f"  - Total methods: {selection_stats.get('total_methods', 0)}")
            logger.error(f"  - Valid methods: {selection_stats.get('valid_methods', 0)}")
            logger.error(f"  - Methods scored: {selection_stats.get('methods_scored', 0)}")
            logger.error(f"  - Metrics processed: {selection_stats.get('metrics_processed', 0)}")
        
        raise

def calculate_optimal_imbalance_threshold(
    df: pd.DataFrame,
    label_col: str = "Label",
    domain: str = "security",
    model_type: str = "neural_network",
    total_samples: Optional[int] = None,
    min_class_samples: Optional[int] = None,
    validation_split: float = 0.2,
    optimize_for: str = "recall",
    budget_constraint: Optional[float] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Determine the optimal imbalance threshold using multiple factors.
    
    Args:
        df: Input DataFrame containing features and labels
        label_col: Name of label column
        domain: Application domain (security, general, fraud, medical, recommender)
        model_type: Type of model (neural_network, random_forest, svm, logistic_regression, xgboost)
        total_samples: Total number of samples (None to compute from df)
        min_class_samples: Minimum class sample count (None to compute from df)
        validation_split: Fraction of data for validation
        optimize_for: Optimization target (recall, precision, f1, balanced_accuracy)
        budget_constraint: Maximum number of synthetic samples to add
        verbose: Enable verbose output
    
    Returns:
        Dictionary containing:
        - optimal_threshold: Recommended imbalance ratio threshold
        - current_ratio: Current imbalance ratio
        - confidence: Confidence score (0-1)
        - reasoning: Explanation of recommendation
        - alternative_thresholds: Other viable options
        - estimated_samples_added: Expected synthetic samples
        - action_recommended: Whether to oversample
        - factors_used: Breakdown of factors applied
        - class_distribution: Current class distribution
        - total_samples: Total sample count
        - min_class_samples: Minimum class sample count
    """
    
    # Get class distribution
    class_counts = df[label_col].value_counts()
    current_ratio = class_counts.max() / class_counts.min()
    total_samples = total_samples or len(df)
    min_class_samples = min_class_samples or class_counts.min()
    
    # Factor 1: Domain-Specific Baselines
    domain_baselines = {
        'security': 2.0,
        'fraud': 3.0,
        'medical': 2.5,
        'general': 10.0,
        'recommender': 20.0
    }
    base_threshold = domain_baselines.get(domain, 10.0)
    
    # Factor 2: Sample Size Adjustment
    if total_samples < 1000:
        size_factor = 0.5
    elif total_samples < 10000:
        size_factor = 0.75
    elif total_samples < 100000:
        size_factor = 1.0
    else:
        size_factor = 1.5
    
    # Factor 3: Minority Class Size
    if min_class_samples < 50:
        minority_factor = 0.3
    elif min_class_samples < 200:
        minority_factor = 0.5
    elif min_class_samples < 1000:
        minority_factor = 0.75
    else:
        minority_factor = 1.0
    
    # Factor 4: Model Type Sensitivity
    model_sensitivity = {
        'neural_network': 0.7,
        'random_forest': 1.2,
        'svm': 0.8,
        'logistic_regression': 0.9,
        'xgboost': 1.1
    }
    model_factor = model_sensitivity.get(model_type, 1.0)
    
    # Factor 5: Optimization Target
    target_factors = {
        'recall': 0.6,
        'precision': 1.4,
        'f1': 1.0,
        'balanced_accuracy': 0.8
    }
    target_factor = target_factors.get(optimize_for, 1.0)
    
    # Calculate optimal threshold
    optimal_threshold = (
        base_threshold * 
        size_factor * 
        minority_factor * 
        model_factor * 
        target_factor
    )
    
    # Clamp to reasonable range
    optimal_threshold = max(1.5, min(optimal_threshold, 50.0))
    
    # Calculate confidence based on data characteristics
    confidence = 1.0
    
    if total_samples < 500:
        confidence *= 0.6
    elif total_samples < 2000:
        confidence *= 0.8
    
    if min_class_samples < 30:
        confidence *= 0.5
    elif min_class_samples < 100:
        confidence *= 0.7
    
    if current_ratio > 100:
        confidence *= 0.7
    elif current_ratio > 50:
        confidence *= 0.85
    
    num_classes = len(class_counts)
    if num_classes > 5:
        confidence *= 0.8
    elif num_classes > 10:
        confidence *= 0.6
    
    confidence = min(1.0, confidence)
    
    # Generate reasoning
    reasoning = []
    reasoning.append(f"Domain '{domain}' baseline: {base_threshold:.1f}")
    
    if size_factor < 1.0:
        reasoning.append("Reduced for small dataset size")
    elif size_factor > 1.0:
        reasoning.append("Increased for large dataset (more robust)")
    
    if minority_factor < 1.0:
        reasoning.append("Reduced for small minority class (needs aggressive balancing)")
    
    if model_factor < 1.0:
        reasoning.append("Reduced for imbalance-sensitive model type")
    elif model_factor > 1.0:
        reasoning.append("Increased for imbalance-robust model type")
    
    if target_factor < 1.0:
        reasoning.append("Reduced to optimize for recall/detection")
    elif target_factor > 1.0:
        reasoning.append("Increased to optimize for precision")
    
    if current_ratio > optimal_threshold:
        reasoning.append(f"Current ratio ({current_ratio:.1f}) exceeds threshold - Oversampling recommended")
    else:
        reasoning.append(f"Current ratio ({current_ratio:.1f}) within threshold - No oversampling needed")
    
    reasoning_text = " | ".join(reasoning)
    
    # Calculate alternative thresholds
    alternatives = {
        'conservative': optimal_threshold * 0.7,
        'aggressive': optimal_threshold * 1.3,
        'minimal': 2.0,
        'none': float('inf')
    }
    
    # Estimate synthetic samples
    estimated_additions = {}
    for name, threshold in {**{'optimal': optimal_threshold}, **alternatives}.items():
        if current_ratio > threshold:
            majority_size = class_counts.max()
            minority_size = class_counts.min()
            target_size = int(majority_size / threshold)
            additions = max(0, target_size - minority_size)
            estimated_additions[name] = additions
        else:
            estimated_additions[name] = 0
    
    # Check budget constraint
    if budget_constraint and estimated_additions['optimal'] > budget_constraint:
        majority_size = class_counts.max()
        minority_size = class_counts.min()
        max_additions = int(budget_constraint)
        target_minority_size = minority_size + max_additions
        adjusted_threshold = majority_size / target_minority_size
        adjusted_threshold = max(1.5, adjusted_threshold)
        optimal_threshold = max(optimal_threshold, adjusted_threshold)
    
    result = {
        'optimal_threshold': round(optimal_threshold, 2),
        'current_ratio': round(current_ratio, 2),
        'confidence': round(confidence, 3),
        'reasoning': reasoning_text,
        'alternative_thresholds': {k: round(v, 2) for k, v in alternatives.items()},
        'estimated_samples_added': estimated_additions,
        'action_recommended': 'oversample' if current_ratio > optimal_threshold else 'none',
        'factors_used': {
            'domain_baseline': base_threshold,
            'size_adjustment': size_factor,
            'minority_adjustment': minority_factor,
            'model_sensitivity': model_factor,
            'optimization_target': target_factor
        },
        'class_distribution': class_counts.to_dict(),
        'total_samples': total_samples,
        'min_class_samples': min_class_samples
    }
    
    if verbose:
        display_threshold_analysis(result)
    
    return result

def display_threshold_analysis(result: Dict[str, Any]) -> None:
    """Display threshold analysis in rich table format."""
    # Main results table
    table = Table(
        title="\n[bold cyan]Intelligent Threshold Analysis[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold yellow",
        title_justify="left"
    )
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Interpretation", style="green")
    
    # Current state
    table.add_row(
        "Current Ratio",
        f"{result['current_ratio']:.2f}:1",
        "Existing imbalance"
    )
    
    # Recommendation
    action_color = "green" if result['action_recommended'] == 'none' else "yellow"
    table.add_row(
        "Optimal Threshold",
        f"[{action_color}]{result['optimal_threshold']:.2f}:1[/{action_color}]",
        f"Recommended maximum"
    )
    
    # Confidence
    conf_color = "green" if result['confidence'] > 0.8 else "yellow" if result['confidence'] > 0.6 else "red"
    table.add_row(
        "Confidence",
        f"[{conf_color}]{result['confidence']:.1%}[/{conf_color}]",
        "Recommendation certainty"
    )
    
    # Action
    table.add_row(
        "Action",
        f"[{action_color}]{result['action_recommended'].upper()}[/{action_color}]",
        "Whether to oversample"
    )
    
    console.print(table)
    
    # Factors table
    factors_table = Table(
        title="\n[bold cyan]Decision Factors[/]",
        box=box.SIMPLE,
        show_header=True,
        title_justify="left"
    )
    factors_table.add_column("Factor", style="cyan")
    factors_table.add_column("Multiplier", style="yellow", justify="right")
    
    for factor, value in result['factors_used'].items():
        factors_table.add_row(factor.replace('_', ' ').title(), f"{value:.2f}x")
    
    console.print(factors_table)
    
    # Alternatives table
    alt_table = Table(
        title="\n[bold cyan]Alternative Thresholds[/]",
        box=box.SIMPLE,
        title_justify="left"
    )
    alt_table.add_column("Strategy", style="cyan")
    alt_table.add_column("Threshold", style="yellow", justify="right")
    alt_table.add_column("Samples Added", style="magenta", justify="right")
    
    for name, threshold in result['alternative_thresholds'].items():
        samples = result['estimated_samples_added'].get(name, 0)
        alt_table.add_row(
            name.title(),
            f"{threshold:.2f}:1",
            f"+{samples:,}"
        )
    
    console.print(alt_table)
    
    # Reasoning
    console.print(f"\n[bold]Reasoning:[/] {result['reasoning']}")

def handle_class_imbalance(
    df: pd.DataFrame,
    artifacts: Dict,
    *,
    oversampler: str = "SMOTE",
    apply_smote: bool = True,
    imbalance_threshold: Optional[float] = None,
    auto_threshold: bool = True,
    domain: str = "security",
    optimize_for: str = "recall",
    label_col: str = "Label",
    sampling_strategy: Union[str, dict] = "auto",
    random_state: int = 42,
    n_jobs: int = -1,
    evaluate_quality: bool = True,
    visualize: bool = True,
    sample_metrics: Optional[int] = None,
    verbose: Optional[bool] = False,
    debug: bool = False
) -> pd.DataFrame:
    """Class imbalance handler with threshold determination and flexible controls.
    
    Args:
        df: Input DataFrame
        artifacts: Preprocessing artifacts dict
        oversampler: Type of oversampler (see available_samplers)
        apply_smote: Whether to apply oversampling
        imbalance_threshold: Ratio to consider imbalance (None for auto-detection)
        auto_threshold: Enable intelligent threshold determination
        domain: Application domain (security, general, fraud, medical, recommender)
        optimize_for: Optimization target (recall, precision, f1, balanced_accuracy)
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
    # Setup styling for colored output
    red = Fore.RED
    yellow = Fore.YELLOW
    green = Fore.GREEN
    blue = Fore.BLUE
    cyan = Fore.CYAN
    reset = Style.RESET_ALL
    
    # Initialize imbalance handling statistics
    imbalance_stats = {
        'stage': 'Initializing',
        'total_samples': len(df),
        'original_classes': 0,
        'imbalance_ratio': 0.0,
        'oversampling_applied': False,
        'new_samples_generated': 0,
        'quality_metrics_calculated': False,
        'visualizations_generated': False,
        'processing_time': 0.0,
        'detailed_timings': {},
        'class_distribution': {},
        'sampler_used': None,
        'quality_scores': {}
    }
    
    try:
        # Progress helper: define all stage titles
        titles = [
            "Initial Setup and Validation",
            "Final Reporting"
        ]
        progress = ProgressHelper(titles)
        
        # STAGE 1: Initial Setup and Validation
        with progress.bar("Initial Setup and Validation", total=4, unit="steps") as init_bar:
            
            # STAGE 1.1: Input Validation
            init_bar.text = "Validating inputs"
            validation_start = time.time()
            
            imbalance_stats['stage'] = "Input Validation"
            
            _validate_inputs(df, artifacts, label_col)
            validation_time = time.time() - validation_start
            imbalance_stats['detailed_timings']['input_validation'] = validation_time
            init_bar.text = "Inputs validated"
            init_bar()
            
            # STAGE 1.2: Class Distribution Analysis
            init_bar.text = "Analyzing class distribution"
            analysis_start = time.time()
            
            imbalance_stats['stage'] = "Distribution Analysis"
            
            # Get class distribution
            class_counts = df[label_col].value_counts()
            min_samples = class_counts.min()
            max_samples = class_counts.max()
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            
            imbalance_stats['original_classes'] = len(class_counts)
            imbalance_stats['imbalance_ratio'] = imbalance_ratio
            imbalance_stats['class_distribution']['original'] = class_counts.to_dict()
            
            analysis_time = time.time() - analysis_start
            imbalance_stats['detailed_timings']['distribution_analysis'] = analysis_time
            init_bar.text = f"Distribution analyzed ({len(class_counts)} classes, ratio: {imbalance_ratio:.1f})"
            init_bar()
            
            # STAGE 1.3: Threshold Determination
            init_bar.text = "Determining optimal threshold"
            threshold_start = time.time()
            
            imbalance_stats['stage'] = "Threshold Determination"
            
            # Intelligent threshold determination
            if auto_threshold and imbalance_threshold is None:
                threshold_analysis = calculate_optimal_imbalance_threshold(
                    df=df,
                    label_col=label_col,
                    domain=domain,
                    model_type="neural_network",
                    optimize_for=optimize_for,
                    verbose=verbose
                )
                
                imbalance_threshold = threshold_analysis['optimal_threshold']
                imbalance_stats['threshold_analysis'] = threshold_analysis
                
                if verbose:
                    logger.info(f"Auto-determined threshold: {imbalance_threshold:.2f}")
                    logger.info(f"Confidence: {threshold_analysis['confidence']:.1%}")
                    logger.info(f"Reasoning: {threshold_analysis['reasoning']}")
            
            elif imbalance_threshold is None:
                # Fallback to domain-specific default
                domain_defaults = {
                    'security': 2.0,
                    'fraud': 3.0,
                    'medical': 2.5,
                    'general': 10.0
                }
                imbalance_threshold = domain_defaults.get(domain, 10.0)
                if verbose:
                    logger.info(f"Using domain-specific default threshold: {imbalance_threshold:.2f}")
            
            threshold_time = time.time() - threshold_start
            imbalance_stats['detailed_timings']['threshold_determination'] = threshold_time
            
            # Display initial analysis
            _display_initial_analysis(class_counts, imbalance_threshold)
            
            # Check if imbalance exceeds threshold
            if imbalance_ratio <= imbalance_threshold:
                threshold_check_time = time.time() - threshold_start
                imbalance_stats['detailed_timings']['threshold_check'] = threshold_check_time
                init_bar.text = "Class distribution within acceptable limits"
                init_bar()
                
                # STAGE 1.4: Early Exit
                init_bar.text = "Preparing early exit"
                early_exit_start = time.time()
                
                init_bar.text = f"Skipping oversampling, returning original data"
                early_exit_time = time.time() - early_exit_start
                imbalance_stats['detailed_timings']['early_exit'] = early_exit_time
                imbalance_stats['detailed_timings']['total'] = sum(imbalance_stats['detailed_timings'].values())
                init_bar()
                
                if verbose:
                    logger.info(f"Class distribution balanced, skipping oversampling")
                    logger.info(f"Class Distribution: ratio {imbalance_ratio:.1f} <= {imbalance_threshold}")
                    for class_label, count in class_counts.items():
                        logger.info(f"  - Class {class_label}: {count:,} samples")
                
                return df
            
            threshold_time = time.time() - threshold_start
            imbalance_stats['detailed_timings']['threshold_check'] = threshold_time
            init_bar.text = f"Imbalance detected (ratio: {imbalance_ratio:.1f} > {imbalance_threshold})"
            init_bar()
            
            # STAGE 1.4: Oversampling Configuration
            init_bar.text = "Checking oversampling configuration"
            config_start = time.time()
            
            imbalance_stats['stage'] = "Configuration Check"
            
            if not apply_smote:
                config_time = time.time() - config_start
                imbalance_stats['detailed_timings']['config_check'] = config_time
                init_bar.text = "Oversampling disabled by configuration"
                init_bar()
                
                early_exit_start = time.time()
                init_bar.text = "Returning original data (oversampling disabled)"
                early_exit_time = time.time() - early_exit_start
                imbalance_stats['detailed_timings']['early_exit'] = early_exit_time
                imbalance_stats['detailed_timings']['total'] = sum(imbalance_stats['detailed_timings'].values())
                init_bar()
                
                if verbose:
                    logger.info(f"Oversampling disabled, returning original data")
                    logger.info(f"Class Distribution: ratio {imbalance_ratio:.1f} > {imbalance_threshold}")
                    for class_label, count in class_counts.items():
                        logger.info(f"  - Class {class_label}: {count:,} samples")
                
                return df
            
            config_time = time.time() - config_start
            imbalance_stats['detailed_timings']['config_check'] = config_time
            init_bar.text = f"Oversampling enabled, preparing {oversampler}"
            init_bar()
        
        # STAGE 2: Oversampling Application
        preparation_start = time.time()
        
        # STAGE 2.1: Sampler Preparation
        imbalance_stats['stage'] = "Sampler Preparation"
        imbalance_stats['sampler_used'] = oversampler
        
        try:
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
            
            preparation_time = time.time() - preparation_start
            imbalance_stats['detailed_timings']['sampler_preparation'] = preparation_time
        
        except Exception as e:
            preparation_time = time.time() - preparation_start
            imbalance_stats['detailed_timings']['sampler_preparation'] = preparation_time
            raise RuntimeError(f"{oversampler} preparation failed: {str(e)}") from e
        
        # STAGE 2.2: Quality Evaluation
        quality_start = time.time()
        
        imbalance_stats['stage'] = "Quality Evaluation"
        
        if evaluate_quality and metrics:
            imbalance_stats['quality_metrics_calculated'] = True
            imbalance_stats['quality_scores'] = metrics
            quality_time = time.time() - quality_start
            imbalance_stats['detailed_timings']['quality_evaluation'] = quality_time
        else:
            quality_time = time.time() - quality_start
            imbalance_stats['detailed_timings']['quality_evaluation'] = quality_time
        
        # STAGE 2.3: Visualization Generation
        visualization_start = time.time()
        
        imbalance_stats['stage'] = "Visualization Generation"
        
        if visualize:
            try:
                # Generate visualizations
                imbalance_stats['visualizations_generated'] = True
                visualization_time = time.time() - visualization_start
                imbalance_stats['detailed_timings']['visualization_generation'] = visualization_time
            except Exception as viz_error:
                visualization_time = time.time() - visualization_start
                imbalance_stats['detailed_timings']['visualization_generation'] = visualization_time
                if verbose:
                    logger.error(f"Visualization error: {str(viz_error)}")
        else:
            visualization_time = time.time() - visualization_start
            imbalance_stats['detailed_timings']['visualization_generation'] = visualization_time
        
        # STAGE 2.4: Results Compilation
        compilation_start = time.time()
        
        imbalance_stats['stage'] = "Results Compilation"
        
        # Update statistics with new distribution
        new_class_counts = balanced_df[label_col].value_counts()
        imbalance_stats['class_distribution']['balanced'] = new_class_counts.to_dict()
        imbalance_stats['new_samples_generated'] = len(balanced_df) - len(df)
        imbalance_stats['oversampling_applied'] = True
        
        compilation_time = time.time() - compilation_start
        imbalance_stats['detailed_timings']['results_compilation'] = compilation_time

        # STAGE 3: Final Reporting
        with progress.bar("Final Reporting", total=2, unit="steps") as report_bar:
            
            # STAGE 3.1: Performance Reporting
            report_bar.text = "Generating performance report"
            reporting_start = time.time()
            
            imbalance_stats['stage'] = "Performance Reporting"
            
            # Report performance
            elapsed = time.time() - validation_start
            imbalance_stats['processing_time'] = elapsed
            
            _report_results(
                original_counts=class_counts,
                new_counts=new_class_counts,
                metrics=metrics,
                elapsed_time=elapsed,
                sampler_name=oversampler
            )
            
            reporting_time = time.time() - reporting_start
            imbalance_stats['detailed_timings']['performance_reporting'] = reporting_time
            report_bar.text = "Performance report generated"
            report_bar()
            
            # STAGE 3.2: Final Summary
            report_bar.text = "Generating final summary"
            summary_start = time.time()
            
            imbalance_stats['stage'] = "Final Summary"
            
            # Display summary
            total_time = sum(imbalance_stats['detailed_timings'].values())
            imbalance_stats['detailed_timings']['total'] = total_time
            
            if verbose:
                logger.info(f"\nClass Imbalance Handling Summary")
                logger.info(f"Processing Overview:")
                logger.info(f"  - Original samples: {len(df):,}")
                logger.info(f"  - Balanced samples: {len(balanced_df):,}")
                logger.info(f"  - New samples generated: {imbalance_stats['new_samples_generated']:,}")
                logger.info(f"  - Original classes: {imbalance_stats['original_classes']}")
                logger.info(f"  - Imbalance ratio: {imbalance_stats['imbalance_ratio']:.1f}")
                logger.info(f"  - Sampler used: {imbalance_stats['sampler_used']}")
                logger.info(f"  - Total processing time: {total_time:.2f}s")
                
                # Display class distribution changes
                logger.info(f"Class Distribution:")
                for class_label, original_count in class_counts.items():
                    new_count = new_class_counts.get(class_label, 0)
                    change = new_count - original_count
                    change_str = f"+{change}" if change > 0 else str(change)
                    logger.info(f"  - Class {class_label}: {original_count:,} -> {new_count:,} ({change_str})")
                
                # Display quality metrics if available
                if imbalance_stats['quality_metrics_calculated'] and metrics:
                    logger.info(f"Quality Metrics:")
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            logger.info(f"  - {metric_name}: {metric_value:.3f}")
            
            # Display detailed timings in debug mode
            if debug:
                logger.debug(f"Detailed Timings:")
                for stage, timing in imbalance_stats['detailed_timings'].items():
                    logger.debug(f"  - {stage}: {timing:.2f}s")
            
            summary_time = time.time() - summary_start
            imbalance_stats['detailed_timings']['final_summary'] = summary_time
            report_bar.text = "Class imbalance handling completed"
            report_bar()
        
        return balanced_df
        
    except Exception as e:
        # Log error with context
        error_context = f"(stage: {imbalance_stats.get('stage', 'unknown')})"
        logger.error(f"Class imbalance handling failed {error_context}: {str(e)}")
        
        if debug:
            traceback.print_exc()
        
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
    imbalance_threshold: float
):
    """Display initial class distribution analysis."""
    table = Table(
        title="\nInitial Class Distribution Analysis",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold yellow",
        title_justify="left",
        border_style="cyan",
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
    console.print()
    
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > imbalance_threshold:
        imbalance_table = Table(
            title="\nClass Imbalance Detected!",
            box=box.ROUNDED,
            style="bold yellow",
            title_style="bold red",
            title_justify="left",
            border_style="yellow",
            show_lines=True,
            show_header=False
        )

        imbalance_table.add_column("Metric", style="bold yellow")
        imbalance_table.add_column("Details", style="bold yellow")
        
        imbalance_table.add_row(
            "Current Imbalance Ratio",
            f"[bold red]{imbalance_ratio:.1f}:1[/bold red]"
        )
        imbalance_table.add_row(
            "Optimal Imbalance Ratio",
            f"[bold green]{imbalance_threshold}:1[/bold green]"
        )
        console.print(imbalance_table)

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
    n_jobs: int = -1,
    auto_optimize: bool = False,
    verbose: Optional[bool] = False,
    optimize_params: Optional[Dict[str, Any]] = None
) -> tuple[pd.DataFrame, Optional[Dict]]:
    """
    Apply oversampling with integrated sampler configuration and quality evaluation.
    
    Args:
        df: Input DataFrame containing features and labels
        artifacts: Dictionary containing feature names and other artifacts
        oversampler: Oversampling method ('SMOTE', 'ADASYN', 'SMOTE+TOMEK', 'Borderline-SMOTE')
        label_col: Name of the label column
        sampling_strategy: Sampling strategy for oversampling
        random_state: Random seed for reproducibility
        evaluate_quality: Whether to evaluate oversampling quality
        visualize: Whether to generate visualizations
        sample_metrics: Optional size for subsampling large datasets for evaluation
        min_samples: Minimum samples required in minority class
        n_jobs: Number of parallel jobs
        auto_optimize: Whether to automatically optimize k_neighbors (default: False)
        optimize_params: Parameters for k_neighbors optimization
    
    Returns:
        Tuple of (balanced_df, metrics) where metrics is None if evaluate_quality=False
    
    Raises:
        ValueError: If invalid parameters or insufficient samples
    """
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track progress and statistics
    progress_stats = {
        'stage': 'Initializing',
        'total_steps': 0,
        'current_step': 0,
        'current_substep': None,
        'detailed_timings': {},
        'optimization_info': {}
    }
    
    # Extract feature matrix and labels
    X = df[artifacts['feature_names']].values
    #y = df[label_col].values
    y = df[label_col].values.astype(np.int64)
    
    # Set default optimization parameters if not provided
    if optimize_params is None:
        optimize_params = {}
    
    try:
        print_color(f"\nStarting Oversampler Application...", 'yellow')

        # Progress helper: define all stage titles
        titles = [
            "Initial Setup",
            "Apply Oversampling",
            "Quality Evaluation",
            "Visualization",
            "Final Summary"
        ]
        progress = ProgressHelper(titles)
        
        # STAGE 1: Initial Setup and Parameter Optimization
        with progress.bar("Initial Setup", total=4, unit="steps") as init_bar:
            
            # STAGE 1.1: Input Validation
            init_bar.text = "Validating input data"
            validation_start = time.time()
            
            progress_stats['stage'] = "Input Validation"
            
            if len(df) == 0:
                init_bar.text = f"Empty DataFrame provided"
                init_bar()
                raise ValueError("Input DataFrame is empty")
            
            if label_col not in df.columns:
                init_bar.text = f"Label column not found"
                init_bar()
                raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
            validation_time = time.time() - validation_start
            progress_stats['detailed_timings']['input_validation'] = validation_time
            init_bar.text = f"Validated {len(df)} samples"
            init_bar()
            
            # STAGE 1.2: k_neighbors Optimization
            init_bar.text = "Configuring parameters"
            optimization_start = time.time()
            
            progress_stats['stage'] = "Parameter Optimization"
            
            # Determine k_neighbors
            if auto_optimize:
                try:
                    # Extract optimization parameters with defaults
                    max_k = optimize_params.get('max_k', 10)
                    n_splits = optimize_params.get('n_splits', 3)
                    metric = optimize_params.get('metric', 'silhouette')
                    verbose = optimize_params.get('verbose', False) if 'verbose' in optimize_params else verbose
                    opt_n_jobs = optimize_params.get('n_jobs', -1)
                    
                    # Input validation for optimization
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
                        k_neighbors = actual_max_k
                        progress_stats['optimization_info'] = {
                            'status': 'limited_by_samples',
                            'minority_count': minority_count,
                            'k_neighbors': k_neighbors
                        }
                        init_bar.text = f"Limited by minority class size: k={k_neighbors}"
                    else:
                        init_bar.text = f"Optimizing k_neighbors (max_k={actual_max_k})"
                        init_bar()
                        
                        # Close current bar for optimization
                        init_bar.text = f"k_neighbors optimization"
                        
                        # Initialize results storage
                        k_values = list(range(3, actual_max_k + 1))
                        results = {
                            'silhouette': {k: [] for k in k_values},
                            'davies_bouldin': {k: [] for k in k_values},
                            'failed_runs': 0
                        }
                        
                        # Cross-validated evaluation with progress tracking
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                        total_folds = n_splits * len(k_values)
                        
                        # Run optimization using existing progress bar
                        init_bar.text = f"Evaluating {len(k_values)} k values across {n_splits} folds"
                        
                        # Initialize results storage
                        k_values = list(range(3, actual_max_k + 1))
                        results = {
                            'silhouette': {k: [] for k in k_values},
                            'davies_bouldin': {k: [] for k in k_values},
                            'failed_runs': 0
                        }
                        
                        # Cross-validated evaluation
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                        total_folds = n_splits * len(k_values)
                        
                        # Update progress bar total and reset to track optimization progress
                        init_bar.total = 4 + total_folds  # Original 4 steps + optimization folds
                        init_bar.n = 4  # Already completed the first 4 steps
                        init_bar.text = f"Evaluating {len(k_values)} k values across {n_splits} folds"
                        init_bar.refresh()
                        
                        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            for k_idx, k in enumerate(k_values):
                                current_fold = fold_idx * len(k_values) + k_idx + 1
                                init_bar.text = f"Fold {fold_idx+1}/{n_splits}, k={k}/{k_values[-1]} ({current_fold}/{total_folds})"
                                
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
                                
                                init_bar()
                        
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
                        elif metric == 'davies_bouldin':
                            best_k = min(mean_scores['davies_bouldin'].items(), key=lambda x: x[1])[0]
                        else:
                            # 'both' - combined score
                            sil_scores = np.array(list(mean_scores['silhouette'].values()))
                            db_scores = np.array(list(mean_scores['davies_bouldin'].values()))
                            
                            norm_sil = (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min())
                            norm_db = 1 - ((db_scores - db_scores.min()) / (db_scores.max() - db_scores.min()))
                            
                            combined_scores = {
                                k: 0.6 * norm_sil[i] + 0.4 * norm_db[i]
                                for i, k in enumerate(k_values)
                            }
                            
                            best_k = max(combined_scores.items(), key=lambda x: x[1])[0]
                        
                        k_neighbors = best_k
                        progress_stats['optimization_info'] = {
                            'status': 'optimized',
                            'k_values_tested': k_values,
                            'best_k': k_neighbors,
                            'failed_runs': results['failed_runs'],
                            'mean_scores': mean_scores
                        }
                        
                        if verbose:
                            logger.info(f"Optimal k_neighbors: {k_neighbors}")
                            logger.info(f"Tested k values: {list(k_values)}")
                
                except Exception as e:
                    if verbose or optimize_params.get('verbose', False):
                        logger.warning(f"k_neighbors optimization failed, using fallback: {str(e)}")
                    k_neighbors = min(3, min_samples - 1)
                    progress_stats['optimization_info'] = {
                        'status': 'optimization_failed',
                        'fallback_k': k_neighbors,
                        'error': str(e)
                    }
                    init_bar.text = f"Using fallback k={k_neighbors}"
            else:
                k_neighbors = min(3, min_samples - 1)
                progress_stats['optimization_info'] = {
                    'status': 'auto_optimize_disabled',
                    'k_neighbors': k_neighbors
                }
                init_bar.text = f"Using default k={k_neighbors}"
            
            optimization_time = time.time() - optimization_start
            progress_stats['detailed_timings']['parameter_optimization'] = optimization_time
            init_bar()
            
            # STAGE 1.3: Sampler Configuration
            init_bar.text = "Configuring sampler"
            sampler_config_start = time.time()
            
            progress_stats['stage'] = "Sampler Configuration"
            
            # Safety check
            if k_neighbors < 1:
                init_bar.text = f"Cannot apply oversampling"
                init_bar()
                raise ValueError(f"Cannot apply oversampling - minority class has only {min_samples} samples")
            
            # Configure and create sampler
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
            
            if oversampler not in samplers:
                init_bar.text = f"Unknown oversampler"
                init_bar()
                raise ValueError(f"Unknown oversampler: {oversampler}. Choose from {list(samplers.keys())}")
            
            sampler = samplers[oversampler]
            
            sampler_config_time = time.time() - sampler_config_start
            progress_stats['detailed_timings']['sampler_configuration'] = sampler_config_time
            init_bar.text = f"Configured {oversampler} sampler"
            init_bar()
            
            # STAGE 1.4: Data Preparation
            init_bar.text = "Preparing data for resampling"
            data_prep_start = time.time()
            
            progress_stats['stage'] = "Data Preparation"
            
            # Get original class distribution
            original_counts = np.bincount(y)
            progress_stats['original_distribution'] = {
                'class_counts': original_counts.tolist(),
                'imbalance_ratio': original_counts.max() / original_counts.min()
            }
            
            data_prep_time = time.time() - data_prep_start
            progress_stats['detailed_timings']['data_preparation'] = data_prep_time
            init_bar.text = f"Data prepared"
            init_bar()

        # STAGE 2: Apply Oversampling
        with progress.bar("Apply Oversampling", total=1, unit="steps") as resample_bar:
            resample_bar.text = f"Applying {oversampler}"
            resampling_start = time.time()
            
            progress_stats['stage'] = "Applying Oversampling"
            
            # Apply oversampling
            X_res, y_res = sampler.fit_resample(
                df[artifacts['feature_names']],
                df[label_col]
            )
            
            # Create balanced DataFrame
            resample_bar.text = "Creating balanced DataFrame"
            balanced_df = pd.DataFrame(X_res, columns=artifacts['feature_names'])
            balanced_df[label_col] = y_res
            
            resampling_time = time.time() - resampling_start
            progress_stats['detailed_timings']['resampling'] = resampling_time
            
            # Get new class distribution
            resample_bar.text = "Calculating new class distribution"
            new_counts = np.bincount(y_res)
            progress_stats['new_distribution'] = {
                'class_counts': new_counts.tolist(),
                'imbalance_ratio': new_counts.max() / new_counts.min(),
                'total_samples': len(y_res)
            }
            
            resample_bar.text = f"Oversampling complete: {len(y_res)} samples generated"
            resample_bar()
        
        # STAGE 3: Quality Evaluation
        metrics = None
        if evaluate_quality:
            with progress.bar("Quality Evaluation", total=6, unit="metrics") as eval_bar:
                eval_bar.text = "Starting quality evaluation"
                evaluation_start = time.time()
                progress_stats['stage'] = "Quality Evaluation"
                metrics = {}
                
                # Get original features as numpy array for efficiency
                eval_bar.text = "Preparing data"
                original_features = df[artifacts['feature_names']].values
                original_labels = df[label_col].values
                
                # Initialize random state
                eval_bar.text = "Initializing random state"
                rng = np.random.RandomState(random_state)
                
                # Determine sample size
                eval_bar.text = "Determining sample size"
                if sample_metrics:
                    target_size = sample_metrics
                else:
                    # Use reasonable default based on dataset size
                    if len(X_res) > 500000:
                        target_size = 100000
                    elif len(X_res) > 200000:
                        target_size = 50000
                    else:
                        target_size = len(X_res)
                
                # Subsample resampled data if requested for large datasets
                if sample_metrics and len(X_res) > target_size:
                    eval_bar.text = f"Subsampling (resampled): {len(X_res):,} -> {target_size:,} samples"
                    res_idx = rng.choice(len(X_res), target_size, replace=False)
                    X_res_sampled = X_res[res_idx]
                    y_res_sampled = y_res[res_idx]
                else:
                    X_res_sampled = X_res
                    y_res_sampled = y_res
                
                # Subsample original data
                if sample_metrics and len(original_features) > target_size:
                    eval_bar.text = f"Subsampling (original): {len(original_features):,} -> {target_size:,} samples"
                    orig_idx = rng.choice(len(original_features), target_size, replace=False)
                    X_orig_sampled = original_features[orig_idx]
                    y_orig_sampled = original_labels[orig_idx]
                else:
                    X_orig_sampled = original_features
                    y_orig_sampled = original_labels
                
                eval_bar.text = "Data sampling complete"
                eval_bar()
                
                # 1. Nearest Neighbor Analysis
                eval_bar.text = f"Nearest neighbor analysis: {len(X_res_sampled):,} samples"
                nn_start = time.time()
                progress_stats['current_substep'] = "Nearest Neighbor Analysis"
                
                try:
                    # Use approximate nearest neighbors for large datasets
                    if len(X_res_sampled) >= 1000000:
                        eval_bar.text = "Approximate nearest neighbors (clustering)"
                        # Determine number of clusters
                        eval_bar.text = "Initializing MiniBatchKMeans model"
                        n_clusters = int(min(50, len(X_res_sampled) // 1000000))
                        n_features = X_res_sampled.shape[1]
                        
                        # Use MiniBatchKMeans for scalability
                        eval_bar.text = f"Mini batch K-means: {n_clusters} clusters, {n_features} features"
                        kmeans = MiniBatchKMeans(
                            n_clusters=n_clusters,
                            init='k-means++',
                            random_state=random_state,
                            batch_size=100000,
                            n_init=3
                        )
                        eval_bar.text = "Predicting labels"
                        cluster_labels = kmeans.fit_predict(X_res_sampled)

                        eval_bar.text = "Computing cluster centers"
                        cluster_centers = kmeans.cluster_centers_
                        
                        # Calculate average distance to cluster center
                        eval_bar.text = "Calculating distances to cluster centers"
                        distances_to_center = []
                        for i in range(n_clusters):
                            eval_bar.text = f"Processing cluster: {i+1}/{n_clusters}"
                            cluster_points = X_res_sampled[cluster_labels == i]
                            if len(cluster_points) > 0:
                                center_dist = euclidean_distances(
                                    cluster_points,
                                    cluster_centers[i].reshape(1, -1)
                                ).flatten()
                                distances_to_center.extend(center_dist)
                        
                        eval_bar.text = "Calculating neighbor distances"
                        metrics['avg_neighbor_distance'] = np.mean(distances_to_center)
                        metrics['neighbor_std'] = np.std(distances_to_center)
                        metrics['nn_samples'] = len(X_res_sampled)
                        metrics['nn_method'] = 'cluster_approximation'
                    else:
                        # Use exact nearest neighbors for smaller datasets
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            eval_bar.text = "Nearest neighbors analysis (exact)"
                            # Use float32 to save memory
                            eval_bar.text = "Initializing float32 arrays"
                            X_res_float32 = X_res_sampled.astype(np.float32)
                            
                            # Create NearestNeighbors model
                            eval_bar.text = f"Building NearestNeighbors model"
                            nbrs = NearestNeighbors(
                                n_neighbors=min(2, len(X_res_sampled)),
                                algorithm='ball_tree',
                                n_jobs=n_jobs
                            ).fit(X_res_float32)
                            
                            # Calculate distances to nearest neighbors
                            eval_bar.text = f"Finding K-neighbors"
                            distances, _ = nbrs.kneighbors(X_res_float32)
                            
                            eval_bar.text = "Calculating neighbor distances"
                            metrics['avg_neighbor_distance'] = np.mean(distances[:, 1])  # Excluding self-distance
                            metrics['neighbor_std'] = np.std(distances[:, 1])
                            metrics['nn_samples'] = len(X_res_sampled)
                            metrics['nn_method'] = 'exact'
                
                except Exception as e:
                    logger.warning(f"Nearest neighbor analysis failed: {str(e)}")
                    metrics['avg_neighbor_distance'] = float('nan')
                    metrics['neighbor_std'] = float('nan')
                    metrics['nn_samples'] = 0
                    metrics['nn_method'] = 'failed'
                
                nn_time = time.time() - nn_start
                progress_stats['detailed_timings']['nearest_neighbor_analysis'] = nn_time
                eval_bar.text = "Neighbor analysis complete"
                eval_bar()
                
                # 2. Boundary Analysis
                eval_bar.text = f"Boundary analysis: {len(X_res_sampled):,} samples"
                boundary_start = time.time()
                progress_stats['current_substep'] = "Boundary Analysis"
                
                try:
                    # Use full sampled dataset for boundary analysis
                    X_boundary = X_res_sampled
                    y_boundary = y_res_sampled
                    
                    # Train SGDClassifier to evaluate boundary violations
                    eval_bar.text = "Training SGDClassifier"
                    clf = SGDClassifier(
                        loss='hinge',
                        penalty='l2',
                        alpha=0.0001,
                        max_iter=1000,
                        tol=1e-3,
                        random_state=random_state,
                        shuffle=True,
                        n_jobs=n_jobs
                    )
                    eval_bar.text = "Fitting linear model"
                    clf.fit(X_boundary.astype(np.float32), y_boundary)
                    
                    eval_bar.text = "Predicting class labels"
                    predictions = clf.predict(X_boundary.astype(np.float32))
                    
                    eval_bar.text = "Evaluating boundary violations"
                    metrics['boundary_violations'] = np.sum(predictions != y_boundary)
                    metrics['boundary_violation_rate'] = metrics['boundary_violations'] / len(predictions)
                    metrics['boundary_samples'] = len(X_boundary)
                    metrics['boundary_method'] = 'sgd_svm_full'
                
                except MemoryError as e:
                    # Fallback to linearSVC training
                    try:
                        # Train linearSVC to evaluate boundary violations
                        eval_bar.text = "Training LinearSVC"
                        clf = LinearSVC(
                            random_state=random_state,
                            max_iter=1000,
                            tol=1e-3,
                            dual=False
                        )
                        eval_bar.text = "Fitting LinearSVC model"
                        clf.fit(X_boundary.astype(np.float32), y_boundary)
                        
                        eval_bar.text = "Predicting class labels"
                        predictions = clf.predict(X_boundary.astype(np.float32))
                        
                        eval_bar.text = "Evaluating boundary violations"
                        metrics['boundary_violations'] = np.sum(predictions != y_boundary)
                        metrics['boundary_violation_rate'] = metrics['boundary_violations'] / len(predictions)
                        metrics['boundary_samples'] = len(X_boundary)
                        metrics['boundary_method'] = 'linear_svm'
                    
                    except Exception as inner_e:
                        logger.warning(f"Incremental boundary analysis failed: {str(inner_e)}")
                        metrics['boundary_violations'] = float('nan')
                        metrics['boundary_violation_rate'] = float('nan')
                        metrics['boundary_samples'] = 0
                        metrics['boundary_method'] = 'failed'
                        eval_bar.text = "Boundary analysis failed"
                
                except Exception as e:
                    logger.warning(f"Boundary analysis failed: {str(e)}")
                    metrics['boundary_violations'] = float('nan')
                    metrics['boundary_violation_rate'] = float('nan')
                    metrics['boundary_samples'] = 0
                    metrics['boundary_method'] = 'failed'
                    eval_bar.text = "Boundary analysis failed"
                
                boundary_time = time.time() - boundary_start
                progress_stats['detailed_timings']['boundary_analysis'] = boundary_time
                eval_bar.text = "Boundary analysis complete"
                eval_bar()
                
                # 3. Distribution Analysis
                eval_bar.text = "Distribution analysis"
                distribution_start = time.time()
                progress_stats['current_substep'] = "Distribution Analysis"
                
                try:
                    eval_bar.text = "Preparing feature distributions"
                    # Compare distributions per feature using full sampled datasets
                    js_divergences = []
                    
                    eval_bar.text = "Converting to numpy arrays"
                    # Convert to numpy arrays if needed and ensure 2D shape
                    if isinstance(X_orig_sampled, pd.DataFrame):
                        X_orig_dist = X_orig_sampled.values
                    else:
                        X_orig_dist = np.asarray(X_orig_sampled)
                    
                    if isinstance(X_res_sampled, pd.DataFrame):
                        X_res_dist = X_res_sampled.values
                    else:
                        X_res_dist = np.asarray(X_res_sampled)
                    
                    # Ensure arrays are 2D
                    eval_bar.text = "Ensuring arrays are 2D shapes"
                    if X_orig_dist.ndim == 1:
                        X_orig_dist = X_orig_dist.reshape(-1, 1)
                    if X_res_dist.ndim == 1:
                        X_res_dist = X_res_dist.reshape(-1, 1)
                    
                    # Get number of features
                    total_features = X_orig_dist.shape[1]
                    
                    # Calculate distributions feature by feature
                    for i in range(total_features):
                        eval_bar.text = f"Calculating distributions: {i+1}/{total_features} features"
                        
                        # Get feature values
                        orig_feature = X_orig_dist[:, i].flatten()
                        res_feature = X_res_dist[:, i].flatten()
                        
                        # Skip features with no variance
                        combined = np.concatenate([orig_feature, res_feature])
                        min_val = np.min(combined)
                        max_val = np.max(combined)
                        
                        if max_val - min_val < 1e-10:
                            js_divergences.append(0.0)
                            continue
                        
                        # Create density histograms with common bins
                        bins = np.linspace(min_val, max_val, 51)
                        
                        try:
                            p_hist, _ = np.histogram(orig_feature, bins=bins, density=True)
                            q_hist, _ = np.histogram(res_feature, bins=bins, density=True)
                            
                            # Add epsilon and normalize
                            epsilon = 1e-10
                            p_hist = (p_hist + epsilon) / (p_hist.sum() + epsilon * len(p_hist))
                            q_hist = (q_hist + epsilon) / (q_hist.sum() + epsilon * len(q_hist))
                            
                            # Calculate Jensen-Shannon divergence
                            m = 0.5 * (p_hist + q_hist)
                            # Add epsilon to avoid log(0)
                            m = m + epsilon
                            kl_pm = np.sum(p_hist * np.log(p_hist / m))
                            kl_qm = np.sum(q_hist * np.log(q_hist / m))
                            js_div = 0.5 * (kl_pm + kl_qm)
                            js_divergences.append(js_div)
                        
                        except MemoryError:
                            # Fallback for large feature vectors: use kernel density estimation
                            try:
                                # Sample if feature vectors are too large
                                sample_size = min(100000, len(orig_feature), len(res_feature))
                                if len(orig_feature) > sample_size:
                                    orig_idx = rng.choice(len(orig_feature), sample_size, replace=False)
                                    orig_feature_sampled = orig_feature[orig_idx]
                                else:
                                    orig_feature_sampled = orig_feature
                                
                                if len(res_feature) > sample_size:
                                    res_idx = rng.choice(len(res_feature), sample_size, replace=False)
                                    res_feature_sampled = res_feature[res_idx]
                                else:
                                    res_feature_sampled = res_feature
                                
                                # Use kernel density estimation
                                kde_orig = gaussian_kde(orig_feature_sampled)
                                kde_res = gaussian_kde(res_feature_sampled)
                                
                                # Evaluate densities on common grid
                                grid = np.linspace(min_val, max_val, 1000)
                                p_dens = kde_orig(grid)
                                q_dens = kde_res(grid)
                                
                                # Normalize
                                p_dens = (p_dens + epsilon) / (p_dens.sum() + epsilon * len(p_dens))
                                q_dens = (q_dens + epsilon) / (q_dens.sum() + epsilon * len(q_dens))
                                
                                m = 0.5 * (p_dens + q_dens) + epsilon
                                kl_pm = np.sum(p_dens * np.log(p_dens / m))
                                kl_qm = np.sum(q_dens * np.log(q_dens / m))
                                js_div = 0.5 * (kl_pm + kl_qm)
                                js_divergences.append(js_div)
                            
                            except Exception as kde_e:
                                logger.warning(f"KDE fallback failed for feature {i}: {str(kde_e)}")
                                js_divergences.append(float('nan'))
                        
                        except Exception as feature_e:
                            logger.warning(f"Feature {i} distribution calculation failed: {str(feature_e)}")
                            js_divergences.append(float('nan'))
                    
                    # Calculate summary statistics, ignoring NaN values
                    valid_js_divergences = [js for js in js_divergences if not np.isnan(js)]
                    
                    if valid_js_divergences:
                        metrics['js_divergence_mean'] = np.mean(valid_js_divergences)
                        metrics['js_divergence_std'] = np.std(valid_js_divergences)
                        metrics['js_divergence_median'] = np.median(valid_js_divergences)
                        metrics['js_divergence_min'] = np.min(valid_js_divergences)
                        metrics['js_divergence_max'] = np.max(valid_js_divergences)
                        metrics['valid_features'] = len(valid_js_divergences)
                        metrics['total_features_analyzed'] = total_features
                        
                        # Normalize JS divergence (range: 0 to log(2))
                        metrics['distribution_divergence'] = min(metrics['js_divergence_mean'] / np.log(2), 1.0)
                    else:
                        metrics['distribution_divergence'] = float('nan')
                        metrics['js_divergence_mean'] = float('nan')
                        metrics['js_divergence_std'] = float('nan')
                        metrics['valid_features'] = 0
                        metrics['total_features_analyzed'] = total_features
                    
                    metrics['distribution_samples'] = len(X_orig_dist)
                    metrics['distribution_method'] = 'full_distribution_js'
                
                except Exception as e:
                    logger.warning(f"Distribution analysis failed: {str(e)}")
                    metrics['distribution_divergence'] = float('nan')
                    metrics['js_divergence_mean'] = float('nan')
                    metrics['js_divergence_std'] = float('nan')
                    metrics['distribution_samples'] = 0
                    metrics['valid_features'] = 0
                    metrics['total_features_analyzed'] = X_orig_sampled.shape[1] if 'X_orig_sampled' in locals() else 0
                    metrics['distribution_method'] = 'failed'
                    eval_bar.text = "Distribution analysis failed"
                
                distribution_time = time.time() - distribution_start
                progress_stats['detailed_timings']['distribution_analysis'] = distribution_time
                eval_bar.text = "Distribution analysis complete"
                eval_bar()
                
                # 4. Cluster Quality
                eval_bar.text = f"Cluster quality analysis: {len(X_res_sampled):,} samples"
                cluster_start = time.time()
                progress_stats['current_substep'] = "Cluster Analysis"
                
                try:
                    # Use full sampled dataset for cluster quality analysis
                    X_cluster = X_res_sampled
                    y_cluster = y_res_sampled
                    cluster_size = len(X_cluster)
                    
                    # Validate minimum requirements for silhouette calculation
                    eval_bar.text = "Validating minimum requirements"
                    unique_classes = np.unique(y_cluster)
                    n_classes = len(unique_classes)
                    
                    # Silhouette requires at least 2 clusters and samples >= clusters
                    if n_classes < 2:
                        logger.warning(f"Cannot calculate silhouette score: only {n_classes} class(es) present")
                        metrics['silhouette_score'] = float('nan')
                        metrics['silhouette_method'] = 'insufficient_classes'
                        metrics['unique_classes'] = n_classes
                        eval_bar.text = f"Skipped: only {n_classes} class"
                    elif cluster_size < n_classes:
                        logger.warning(f"Cannot calculate silhouette score: {cluster_size} samples < {n_classes} classes")
                        metrics['silhouette_score'] = float('nan')
                        metrics['silhouette_method'] = 'insufficient_samples'
                        metrics['unique_classes'] = n_classes
                        eval_bar.text = "Skipped: insufficient samples"
                    else:
                        # Calculate silhouette score
                        try:
                            # For large datasets, use sampling to speed up computation
                            if cluster_size > 100000:
                                eval_bar.text = "Calculating silhouette score: sampled"
                                # Use sample_size parameter for large datasets
                                actual_sample_size = min(100000, cluster_size)
                                eval_bar.text = f"Using sample size: {actual_sample_size:,}"
                                sil_score = silhouette_score(
                                    X_cluster,
                                    y_cluster,
                                    metric='euclidean',
                                    sample_size=actual_sample_size,
                                    random_state=random_state
                                )
                                metrics['silhouette_score'] = sil_score
                                metrics['silhouette_sample_size'] = actual_sample_size
                                metrics['silhouette_method'] = 'sampled'
                            
                            else:
                                eval_bar.text = "Calculating silhouette score: full"
                                # Use full dataset for smaller datasets
                                sil_score = silhouette_score(
                                    X_cluster,
                                    y_cluster,
                                    metric='euclidean',
                                    random_state=random_state
                                )
                                metrics['silhouette_score'] = sil_score
                                metrics['silhouette_method'] = 'full'
                        
                        except MemoryError:
                            # Fallback for extremely large datasets: use stratified sampling
                            try:
                                eval_bar.text = "Calculating silhouette score (stratified sampling)"
                                # Use stratified sampling to ensure class representation
                                sample_per_class = min(5000, cluster_size // n_classes)
                                
                                sampled_indices = []
                                for cls in unique_classes:
                                    cls_indices = np.where(y_cluster == cls)[0]
                                    if len(cls_indices) > sample_per_class:
                                        cls_sample = rng.choice(cls_indices, sample_per_class, replace=False)
                                    else:
                                        cls_sample = cls_indices
                                    sampled_indices.extend(cls_sample)
                                
                                X_sampled = X_cluster[sampled_indices]
                                y_sampled = y_cluster[sampled_indices]
                                
                                sil_score = silhouette_score(
                                    X_sampled,
                                    y_sampled,
                                    metric='euclidean',
                                    random_state=random_state
                                )
                                metrics['silhouette_score'] = sil_score
                                metrics['silhouette_sample_size'] = len(sampled_indices)
                                metrics['silhouette_method'] = 'stratified_sampling'
                            
                            except Exception as sampling_e:
                                logger.warning(f"Stratified sampling failed: {str(sampling_e)}")
                                metrics['silhouette_score'] = float('nan')
                                metrics['silhouette_method'] = 'stratified_sampling_failed'
                                eval_bar.text = "Stratified sampling failed"
                        
                        except ValueError as ve:
                            # Handle specific ValueError cases (e.g., single sample per class)
                            logger.warning(f"Silhouette calculation failed with ValueError: {str(ve)}")
                            
                            try:
                                eval_bar.text = "Calculating silhouette score (cosine metric)"
                                # Try cosine distance for high-dimensional data
                                sil_score_cosine = silhouette_score(
                                    X_cluster,
                                    y_cluster,
                                    metric='cosine',
                                    random_state=random_state
                                )
                                metrics['silhouette_score'] = sil_score_cosine
                                metrics['silhouette_method'] = 'cosine'
                            
                            except Exception as cosine_e:
                                logger.warning(f"Cosine metric failed: {str(cosine_e)}")
                                
                                try:
                                    eval_bar.text = "Calculating silhouette (precomputed sample)"
                                    # Final fallback: use precomputed distances on smaller sample
                                    sample_size_final = min(5000, cluster_size)
                                    sample_idx = rng.choice(cluster_size, sample_size_final, replace=False)
                                    X_sample = X_cluster[sample_idx]
                                    y_sample = y_cluster[sample_idx]
                                    
                                    # Compute pairwise distances
                                    distances = pairwise_distances(X_sample, metric='euclidean')
                                    sil_score_precomputed = silhouette_score(
                                        distances,
                                        y_sample,
                                        metric='precomputed'
                                    )
                                    metrics['silhouette_score'] = sil_score_precomputed
                                    metrics['silhouette_sample_size'] = sample_size_final
                                    metrics['silhouette_method'] = 'precomputed_sample'
                                
                                except Exception as precomp_e:
                                    logger.warning(f"Precomputed fallback failed: {str(precomp_e)}")
                                    metrics['silhouette_score'] = float('nan')
                                    metrics['silhouette_method'] = 'all_methods_failed'
                                    eval_bar.text = "All silhouette methods failed"
                        
                        except Exception as sil_e:
                            # Catch-all for unexpected errors
                            logger.warning(f"Unexpected error in silhouette calculation: {str(sil_e)}")
                            metrics['silhouette_score'] = float('nan')
                            metrics['silhouette_method'] = 'unexpected_error'
                            eval_bar.text = "Silhouette calculation error"
                    
                    # Store cluster analysis metadata
                    metrics['cluster_samples'] = cluster_size
                    metrics['unique_classes'] = n_classes
                    metrics['cluster_method'] = 'comprehensive'
                
                except Exception as e:
                    logger.warning(f"Cluster analysis failed: {str(e)}")
                    metrics['silhouette_score'] = float('nan')
                    metrics['cluster_samples'] = 0
                    metrics['cluster_method'] = 'failed'
                    metrics['unique_classes'] = 0
                    metrics['error_message'] = str(e)
                    eval_bar.text = "Cluster analysis failed"
                
                cluster_time = time.time() - cluster_start
                progress_stats['detailed_timings']['cluster_analysis'] = cluster_time
                eval_bar.text = "Cluster analysis complete"
                eval_bar()
                
                # 5. Class Balance
                eval_bar.text = "Class balance analysis"
                balance_start = time.time()
                progress_stats['current_substep'] = "Balance Analysis"
                
                try:
                    # Calculate on full resampled dataset
                    eval_bar.text = "Balance analysis: resampled dataset"
                    
                    # Convert to integer type before bincount
                    eval_bar.text = "Converting to integer type before bincount"
                    y_res_sampled_int = y_res_sampled.astype(np.int64)
                    full_new_counts = np.bincount(y_res_sampled_int)
                    
                    eval_bar.text = "Calculating imbalance ratio"
                    if len(full_new_counts) > 1:
                        metrics['new_imbalance_ratio'] = full_new_counts.max() / full_new_counts.min()
                        metrics['new_class_counts'] = full_new_counts.tolist()
                    else:
                        metrics['new_imbalance_ratio'] = 1.0
                        metrics['new_class_counts'] = full_new_counts.tolist()
                    
                    # Original dataset balance
                    eval_bar.text = "Balance analysis: original dataset"
                    
                    # Convert to integer type before bincount
                    eval_bar.text = "Converting to integer type before bincount"
                    y_orig_sampled_int = y_orig_sampled.astype(np.int64)
                    full_orig_counts = np.bincount(y_orig_sampled_int)
                    
                    eval_bar.text = "Calculating imbalance ratio"
                    if len(full_orig_counts) > 1:
                        metrics['original_imbalance_ratio'] = full_orig_counts.max() / full_orig_counts.min()
                        metrics['original_class_counts'] = full_orig_counts.tolist()
                    else:
                        metrics['original_imbalance_ratio'] = 1.0
                        metrics['original_class_counts'] = full_orig_counts.tolist()
                    
                    # Calculate balance improvement
                    eval_bar.text = "Calculating balance improvement"
                    if ('original_imbalance_ratio' in metrics and 'new_imbalance_ratio' in metrics and metrics['original_imbalance_ratio'] > 1.0):
                        improvement = (metrics['original_imbalance_ratio'] - metrics['new_imbalance_ratio']) / metrics['original_imbalance_ratio']
                        metrics['balance_improvement'] = max(0.0, min(1.0, improvement))
                    else:
                        metrics['balance_improvement'] = 0.0
                
                except Exception as e:
                    logger.warning(f"Balance analysis failed: {str(e)}")
                    metrics['new_imbalance_ratio'] = float('nan')
                    metrics['original_imbalance_ratio'] = float('nan')
                    metrics['balance_improvement'] = float('nan')
                
                balance_time = time.time() - balance_start
                progress_stats['detailed_timings']['balance_analysis'] = balance_time
                eval_bar.text = "Balance analysis complete"
                eval_bar()
                
                # 6. Feature Correlation Analysis
                eval_bar.text = f"Feature correlation analysis: {X_orig_sampled.shape[1]:,} features"
                correlation_start = time.time()
                progress_stats['current_substep'] = "Correlation Analysis"
                
                try:
                    # Use full sampled datasets for correlation analysis
                    X_orig_corr = X_orig_sampled
                    X_res_corr = X_res_sampled
                    
                    # Check if datasets are large enough for meaningful correlation
                    if len(X_orig_corr) < 2 or len(X_res_corr) < 2:
                        metrics['feature_correlation_diff'] = float('nan')
                        metrics['correlation_samples'] = 0
                        metrics['correlation_method'] = 'insufficient_samples'
                        eval_bar.text = f"Insufficient samples: {len(X_orig_corr):,} original, {len(X_res_corr):,} resampled"
                    else:
                        try:
                            # Calculate correlations incrementally for large datasets: use batch processing
                            eval_bar.text = "Feature correlation analysis: (Incremental)"
                            n_features = X_orig_corr.shape[1]
                            n_samples_orig = len(X_orig_corr)
                            n_samples_res = len(X_res_corr)
                            
                            # Convert to numpy arrays if needed
                            eval_bar.text = "Converting to numpy arrays"
                            if isinstance(X_orig_corr, pd.DataFrame):
                                X_orig_corr = X_orig_corr.values
                            if isinstance(X_res_corr, pd.DataFrame):
                                X_res_corr = X_res_corr.values
                            
                            # Use batched covariance calculation
                            eval_bar.text = "Batched covariance calculation"
                            batch_size = min(100000, n_samples_orig, n_samples_res)
                            n_batches_orig = (n_samples_orig + batch_size - 1) // batch_size
                            n_batches_res = (n_samples_res + batch_size - 1) // batch_size
                            
                            # Initialize covariance matrices
                            eval_bar.text = "Initializing covariance matrices"
                            cov_orig = np.zeros((n_features, n_features))
                            cov_res = np.zeros((n_features, n_features))
                            mean_orig = np.zeros(n_features)
                            mean_res = np.zeros(n_features)
                            
                            # Calculate means for original data
                            for batch_idx in range(n_batches_orig):
                                eval_bar.text = f"Calculating means (original): {batch_idx+1}/{n_batches_orig}"
                                start_idx = batch_idx * batch_size
                                end_idx = min((batch_idx + 1) * batch_size, n_samples_orig)
                                batch_orig = X_orig_corr[start_idx:end_idx]
                                mean_orig += np.sum(batch_orig, axis=0)
                            mean_orig /= n_samples_orig
                            
                            # Calculate means for resampled data
                            for batch_idx in range(n_batches_res):
                                eval_bar.text = f"Calculating means (resampled): {batch_idx+1}/{n_batches_res}"
                                start_idx = batch_idx * batch_size
                                end_idx = min((batch_idx + 1) * batch_size, n_samples_res)
                                batch_res = X_res_corr[start_idx:end_idx]
                                mean_res += np.sum(batch_res, axis=0)
                            mean_res /= n_samples_res
                            
                            # Calculate covariances for original data
                            for batch_idx in range(n_batches_orig):
                                eval_bar.text = f"Calculating covariances (original): {batch_idx+1}/{n_batches_orig}"
                                start_idx = batch_idx * batch_size
                                end_idx = min((batch_idx + 1) * batch_size, n_samples_orig)
                                batch_orig = X_orig_corr[start_idx:end_idx] - mean_orig
                                
                                # Update covariance matrix
                                cov_orig += batch_orig.T @ batch_orig
                            
                            # Calculate covariances for resampled data
                            for batch_idx in range(n_batches_res):
                                eval_bar.text = f"Calculating covariances (resampled): {batch_idx+1}/{n_batches_res}"
                                start_idx = batch_idx * batch_size
                                end_idx = min((batch_idx + 1) * batch_size, n_samples_res)
                                batch_res = X_res_corr[start_idx:end_idx] - mean_res
                                
                                # Update covariance matrix
                                cov_res += batch_res.T @ batch_res
                            
                            # Normalize covariances
                            eval_bar.text = "Normalizing covariance matrices"
                            cov_orig /= (n_samples_orig - 1)
                            cov_res /= (n_samples_res - 1)
                            
                            # Convert to correlation matrices
                            eval_bar.text = "Converting to correlation matrices"
                            std_orig = np.sqrt(np.diag(cov_orig))
                            std_res = np.sqrt(np.diag(cov_res))
                            
                            # Avoid division by zero
                            eval_bar.text = "Avoiding division by zero"
                            std_orig[std_orig == 0] = 1
                            std_res[std_res == 0] = 1
                            
                            # Calculate correlation matrices
                            eval_bar.text = "Calculating correlation matrices"
                            corr_orig = cov_orig / np.outer(std_orig, std_orig)
                            corr_res = cov_res / np.outer(std_res, std_res)
                            
                            # Calculate absolute difference (ignore diagonal)
                            eval_bar.text = "Calculating absolute difference"
                            mask = ~np.eye(n_features, dtype=bool)
                            corr_diff = np.abs(corr_orig[mask] - corr_res[mask])
                            
                            metrics['feature_correlation_diff'] = np.nanmean(corr_diff)
                            metrics['feature_correlation_diff_std'] = np.nanstd(corr_diff)
                            metrics['correlation_samples'] = n_samples_orig
                            metrics['correlation_batches_orig'] = n_batches_orig
                            metrics['correlation_batches_res'] = n_batches_res
                            metrics['correlation_method'] = 'incremental_correlation'
                        
                        except Exception as inc_e:
                            try:
                                # Calculate correlation matrices on full datasets
                                eval_bar.text = "Calculating on original dataset"
                                corr_orig = np.corrcoef(X_orig_corr.T)
                                
                                eval_bar.text = "Calculating on resampled dataset"
                                corr_res = np.corrcoef(X_res_corr.T)
                                
                                # Calculate absolute difference (ignore diagonal)
                                eval_bar.text = "Calculating absolute difference"
                                mask = ~np.eye(corr_orig.shape[0], dtype=bool)
                                corr_diff = np.abs(corr_orig[mask] - corr_res[mask])
                                
                                # Calculate various statistics on the correlation differences
                                eval_bar.text = "Calculating additional statistics"
                                feature_correlation_diff = np.nanmean(corr_diff)
                                feature_correlation_diff_std = np.nanstd(corr_diff)
                                feature_correlation_diff_median = np.nanmedian(corr_diff)
                                feature_correlation_diff_max = np.nanmax(corr_diff)
                                feature_correlation_diff_min = np.nanmin(corr_diff)
                                correlation_samples = len(X_orig_corr)
                                
                                # Store metrics
                                metrics['feature_correlation_diff'] = feature_correlation_diff
                                metrics['feature_correlation_diff_std'] = feature_correlation_diff_std
                                metrics['feature_correlation_diff_median'] = feature_correlation_diff_median
                                metrics['feature_correlation_diff_max'] = feature_correlation_diff_max
                                metrics['feature_correlation_diff_min'] = feature_correlation_diff_min
                                metrics['correlation_samples'] = correlation_samples
                                metrics['correlation_method'] = 'full_correlation'
                                
                                # Calculate correlation preservation rate (differences < threshold)
                                eval_bar.text = "Calculating correlation preservation rate"
                                threshold = 0.1  # 10% difference threshold
                                preserved_correlations = np.sum(corr_diff < threshold)
                                total_correlations = len(corr_diff)
                                correlation_preservation_rate = preserved_correlations / total_correlations
                                correlation_differences_above_threshold = total_correlations - preserved_correlations
                                
                                metrics['correlation_preservation_rate'] = correlation_preservation_rate
                                metrics['correlation_differences_above_threshold'] = correlation_differences_above_threshold
                            
                            except MemoryError:
                                logger.warning(f"Full correlation calculation failed: {str(inc_e)}")
                                metrics['feature_correlation_diff'] = float('nan')
                                metrics['feature_correlation_diff_std'] = float('nan')
                                metrics['correlation_samples'] = 0
                                metrics['correlation_method'] = 'full_failed'
                                eval_bar.text = "Full correlation analysis failed"
                
                except Exception as e:
                    logger.warning(f"Correlation analysis failed: {str(e)}")
                    metrics['feature_correlation_diff'] = float('nan')
                    metrics['feature_correlation_diff_std'] = float('nan')
                    metrics['feature_correlation_diff_median'] = float('nan')
                    metrics['feature_correlation_diff_max'] = float('nan')
                    metrics['feature_correlation_diff_min'] = float('nan')
                    metrics['correlation_samples'] = 0
                    metrics['correlation_method'] = 'failed'
                    metrics['correlation_preservation_rate'] = float('nan')
                    metrics['correlation_differences_above_threshold'] = float('nan')
                    eval_bar.text = "Correlation analysis failed"
                
                correlation_time = time.time() - correlation_start
                progress_stats['detailed_timings']['correlation_analysis'] = correlation_time
                eval_bar.text = "Correlation analysis complete"
                eval_bar()
                
                # Calculate overall quality metrics
                additional_start = time.time()
                
                try:
                    # Calculate overall quality score
                    score_components = []
                    weights = []
                    
                    # Cluster quality
                    if not np.isnan(metrics.get('silhouette_score', float('nan'))):
                        sil_norm = (metrics['silhouette_score'] + 1) / 2  # Convert from [-1, 1] to [0, 1]
                        score_components.append(sil_norm)
                        weights.append(25)
                    
                    # Distribution preservation
                    if not np.isnan(metrics.get('distribution_divergence', float('nan'))):
                        dist_score = 1.0 - min(metrics['distribution_divergence'], 1.0)
                        score_components.append(dist_score)
                        weights.append(20)
                    
                    # Balance improvement
                    if not np.isnan(metrics.get('balance_improvement', float('nan'))):
                        score_components.append(metrics['balance_improvement'])
                        weights.append(25)
                    
                    # Feature correlation
                    if not np.isnan(metrics.get('feature_correlation_diff', float('nan'))):
                        corr_score = 1.0 - min(metrics['feature_correlation_diff'], 1.0)
                        score_components.append(corr_score)
                        weights.append(20)
                    
                    # Boundary quality
                    if not np.isnan(metrics.get('boundary_violation_rate', float('nan'))):
                        boundary_score = 1.0 - min(metrics['boundary_violation_rate'], 1.0)
                        score_components.append(boundary_score)
                        weights.append(10)
                    
                    # Calculate weighted average
                    if score_components and weights:
                        total_weight = sum(weights)
                        weighted_sum = sum(comp * weight for comp, weight in zip(score_components, weights))
                        metrics['overall_quality_score'] = (weighted_sum / total_weight) * 100
                    else:
                        metrics['overall_quality_score'] = float('nan')
                    
                    # Add evaluation metadata
                    metrics['evaluation_sample_size'] = target_size
                    metrics['total_features'] = X_res_sampled.shape[1]
                
                except Exception as e:
                    logger.warning(f"Overall quality calculation failed: {str(e)}")
                    metrics['overall_quality_score'] = float('nan')
                    metrics['quality_assessment'] = 'Failed'
                
                additional_time = time.time() - additional_start
                progress_stats['detailed_timings']['additional_metrics'] = additional_time
                
                evaluation_time = time.time() - evaluation_start
                progress_stats['detailed_timings']['total_evaluation'] = evaluation_time
                eval_bar.text = "Quality evaluation complete"
                eval_bar()
        
        # STAGE 4: Visualization
        if visualize:
            visualization_start = time.time()
            
            progress_stats['stage'] = "Visualization"
            
            # Convert DataFrame back to numpy array for visualization
            X_res_array = X_res.values if isinstance(X_res, pd.DataFrame) else X_res
            
            _visualize_resampling(
                original=df[artifacts['feature_names']],
                resampled=X_res_array,
                labels=y_res,
                sampler_name=oversampler,
                show_plot=VIZ_CONFIG['interactive'],
                save_plot=True,
                progress_bar=True,
                max_samples=VIZ_CONFIG['max_samples'],
                dpi=VIZ_CONFIG['dpi'],
                projections=VIZ_CONFIG['projections']
            )
            
            visualization_time = time.time() - visualization_start
            progress_stats['detailed_timings']['visualization'] = visualization_time

        # STAGE 5: Final Summary
        with progress.bar("Final Summary", total=2, unit="steps") as summary_bar:
            
            # STAGE 5.1: Statistics Calculation
            summary_bar.text = "Calculating final statistics"
            stats_start = time.time()
            
            progress_stats['stage'] = "Statistics Calculation"
            
            total_time = sum(progress_stats['detailed_timings'].values())
            progress_stats['detailed_timings']['total'] = total_time
            
            stats_time = time.time() - stats_start
            progress_stats['detailed_timings']['stats_calculation'] = stats_time
            summary_bar.text = f"Statistics calculated"
            summary_bar()
            
            # STAGE 5.2: Summary Reporting
            summary_bar.text = "Generating final report"
            report_start = time.time()
            
            progress_stats['stage'] = "Summary Reporting"
            
            if verbose or optimize_params.get('verbose', False):
                # Print summary report
                logger.info(f"\nOversampling Completed Successfully!")
                logger.info(f"  - Original samples: {len(df)}")
                logger.info(f"  - Resampled samples: {len(balanced_df)}")
                logger.info(f"  - Oversampler: {oversampler}")
                logger.info(f"  - k_neighbors: {k_neighbors}")
                logger.info(f"  - Total time: {total_time:.3f}s")
                
                if evaluate_quality and metrics:
                    logger.info(f"\nQuality Metrics:")
                    for metric, value in metrics.items():
                        if not np.isnan(value):
                            logger.info(f"  - {metric}: {value:.4f}")
                
                # Print detailed timings in verbose mode
                if optimize_params.get('verbose', False):
                    logger.info(f"\nDetailed Timings:")
                    for stage, timing in progress_stats['detailed_timings'].items():
                        logger.info(f"  - {stage}: {timing:.3f}s")
            
            report_time = time.time() - report_start
            progress_stats['detailed_timings']['final_report'] = report_time
            summary_bar.text = f"Oversampling completed successfully"
            summary_bar()
        
        return balanced_df, metrics
        
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {progress_stats.get('stage', 'unknown')}, "
        error_context += f"substep: {progress_stats.get('current_substep', 'none')})"
        
        logger.error(f"Oversampling failed{error_context}: {str(e)}")
        raise

def _visualize_resampling(
    original: pd.DataFrame,
    resampled: np.ndarray,
    labels: np.ndarray,
    sampler_name: str,
    random_state: int = 42,
    figsize: tuple = (18, 6),
    max_samples: int = 50000,
    dpi: int = 150,
    show_plot: bool = False,
    save_plot: bool = True,
    progress_bar: bool = True,
    projections: List[str] = ['pca', 'tsne', 'umap'],
    interactive_backend: str = 'matplotlib',
    dimensions: int = 2
) -> Optional[Union[plt.Figure, 'go.Figure']]:
    """
    Generate comparative visualizations of resampling results with integrated backend support
    and progress tracking.
    
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
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track visualization progress and statistics
    viz_stats = {
        'stage': 'Initializing',
        'original_samples': len(original),
        'resampled_samples': len(resampled),
        'projections_processed': 0,
        'current_projection': None,
        'backend_used': interactive_backend,
        'dimensions': dimensions,
        'detailed_timings': {},
        'sampling_applied': False
    }
    
    try:
        print_color(f"\nStarting visualization of resampling results...", 'yellow')
        
        # Progress helper: define all stage titles
        titles = [
            "Visualization Setup",
            "Visualization Creation",
            "Save and Display"
        ]
        progress = ProgressHelper(titles)
        
        if progress_bar:
            # STAGE 1: Visualization Setup
            with progress.bar("Visualization Setup", total=4, unit="steps") as setup_bar:
                
                # STAGE 1.1: Input Validation
                setup_bar.text = "Validating input data..."
                validation_start = time.time()
                
                viz_stats['stage'] = "Input Validation"
                
                # Validate dimensions parameter
                if dimensions not in (2, 3):
                    setup_bar.text = f"Invalid dimensions: {dimensions}"
                    setup_bar()
                    raise ValueError("Dimensions must be 2 or 3")
                
                # Validate inputs
                if not isinstance(original, pd.DataFrame):
                    setup_bar.text = "Original data must be DataFrame"
                    setup_bar()
                    raise ValueError("original must be a pandas DataFrame")
                if not isinstance(resampled, np.ndarray):
                    setup_bar.text = "Resampled data must be numpy array"
                    setup_bar()
                    raise ValueError("resampled must be a numpy array")
                if len(original) == 0 or len(resampled) == 0:
                    setup_bar.text = "Input data cannot be empty"
                    setup_bar()
                    raise ValueError("Input data cannot be empty")
                
                viz_stats['original_samples'] = len(original)
                viz_stats['resampled_samples'] = len(resampled)
                
                validation_time = time.time() - validation_start
                viz_stats['detailed_timings']['input_validation'] = validation_time
                setup_bar.text = f"Validated {len(original)} original, {len(resampled)} resampled samples"
                setup_bar()
                
                # STAGE 1.2: Backend Setup
                setup_bar.text = "Setting up visualization backend..."
                backend_start = time.time()
                
                viz_stats['stage'] = "Backend Setup"
                
                # Handle interactive backend
                if interactive_backend.lower() == 'plotly':
                    try:
                        import plotly.express as px
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        viz_stats['backend_used'] = 'plotly'
                    except ImportError:
                        logger.warning("Plotly not available, falling back to matplotlib")
                        interactive_backend = 'matplotlib'
                        viz_stats['backend_used'] = 'matplotlib (fallback)'
                else:
                    viz_stats['backend_used'] = 'matplotlib'
                
                backend_time = time.time() - backend_start
                viz_stats['detailed_timings']['backend_setup'] = backend_time
                setup_bar.text = f"Using {viz_stats['backend_used']} backend"
                setup_bar()
                
                # STAGE 1.3: Data Sampling
                setup_bar.text = "Sampling data if needed..."
                sampling_start = time.time()
                
                viz_stats['stage'] = "Data Sampling"
                
                # Sample original data if too large
                if len(original) > max_samples:
                    rng = np.random.RandomState(random_state)
                    orig_idx = rng.choice(len(original), min(max_samples, len(original)), replace=False)
                    original = original.iloc[orig_idx]
                    viz_stats['sampling_applied'] = True
                    viz_stats['original_samples_after_sampling'] = len(original)
                
                # Sample resampled data if too large
                if len(resampled) > max_samples:
                    rng = np.random.RandomState(random_state)
                    res_idx = rng.choice(len(resampled), min(max_samples, len(resampled)), replace=False)
                    resampled = resampled[res_idx]
                    labels = labels[res_idx]
                    viz_stats['sampling_applied'] = True
                    viz_stats['resampled_samples_after_sampling'] = len(resampled)
                
                sampling_time = time.time() - sampling_start
                viz_stats['detailed_timings']['data_sampling'] = sampling_time
                if viz_stats['sampling_applied']:
                    setup_bar.text = "Applied sampling to large datasets"
                else:
                    setup_bar.text = "No sampling needed"
                setup_bar()
                
                # STAGE 1.4: Projection Setup
                setup_bar.text = "Setting up projections"
                projection_setup_start = time.time()
                
                viz_stats['stage'] = "Projection Setup"
                
                viz_stats['projections_to_process'] = len(projections)
                viz_stats['total_projections'] = len(projections) + 1  # +1 for original data
                
                projection_setup_time = time.time() - projection_setup_start
                viz_stats['detailed_timings']['projection_setup'] = projection_setup_time
                setup_bar.text = f"Setup {len(projections)} projections"
                setup_bar()

            # STAGE 2: Visualization Creation
            # Determine total steps based on backend and dimensions
            if interactive_backend.lower() == 'plotly':
                if dimensions == 3:
                    plot_steps = 7  # For 3D Plotly
                else:
                    plot_steps = 3 + (len(projections) * 2)  # For 2D Plotly
            else:
                if dimensions == 3:
                    plot_steps = 6  # For 3D Matplotlib
                else:
                    plot_steps = 2 + len(projections)  # For 2D Matplotlib
            
            with progress.bar("Visualization Creation", total=plot_steps, unit="steps") as viz_bar:
                
                viz_start = time.time()
                viz_stats['stage'] = "Visualization Creation"
                
                # Create visualizations based on backend
                if interactive_backend.lower() == 'plotly':
                    # Plotly Visualization
                    try:
                        if dimensions == 3:
                            viz_bar.text = "Creating 3D Plotly visualization"
                            figure_start = time.time()
                            
                            # 3D visualization with Plotly
                            fig = make_subplots(
                                rows=1, cols=2,
                                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                                subplot_titles=["Original Data (3D PCA)", f"Resampled Data (3D PCA) - {sampler_name}"]
                            )
                            
                            viz_bar()
                            
                            # Project data to 3D using PCA
                            viz_bar.text = "Computing 3D PCA projection"
                            pca_start = time.time()
                            
                            pca = PCA(n_components=3)
                            orig_proj = pca.fit_transform(original.values)
                            res_proj = pca.transform(resampled)
                            
                            pca_time = time.time() - pca_start
                            viz_stats['detailed_timings']['3d_pca'] = pca_time
                            viz_bar.text = "3D PCA computed"
                            viz_bar()
                            
                            # Original data trace
                            viz_bar.text = "Adding original data trace"
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
                            viz_bar()
                            
                            # Resampled data traces by class
                            viz_bar.text = "Adding resampled data traces"
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
                            
                            viz_bar()
                            
                            # Layout configuration
                            viz_bar.text = "Configuring layout"
                            fig.update_layout(
                                title_text=f"Oversampling Comparison: {sampler_name}",
                                width=1200,
                                height=600
                            )
                            
                            figure_time = time.time() - figure_start
                            viz_stats['detailed_timings']['plotly_3d_figure'] = figure_time
                            viz_bar.text = f"3D Plotly figure created"
                            viz_bar()
                            
                        else:
                            # 2D visualization with Plotly
                            n_cols = len(projections) + 1
                            viz_bar.text = f"Creating 2D Plotly visualization with {n_cols} subplots"
                            figure_start = time.time()
                            
                            fig = make_subplots(
                                rows=1, cols=n_cols,
                                subplot_titles=["Original Data (PCA)"] + 
                                [f"Resampled Data ({m.upper()})" for m in projections]
                            )
                            
                            viz_bar()
                            
                            # Original data (PCA)
                            viz_bar.text = "Projecting original data (PCA)"
                            pca_start = time.time()
                            
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
                            
                            pca_time = time.time() - pca_start
                            viz_stats['detailed_timings']['original_pca'] = pca_time
                            viz_bar.text = "Original PCA projected"
                            viz_bar()
                            
                            # Resampled data projections
                            for i, method in enumerate(projections, 2):
                                viz_stats['current_projection'] = method
                                viz_bar.text = f"Computing {method.upper()} projection"
                                projection_start = time.time()
                                
                                if method == 'pca':
                                    proj = PCA(n_components=2).fit_transform(resampled)
                                elif method == 'tsne':
                                    proj = TSNE(n_components=2).fit_transform(resampled)
                                else:
                                    proj = umap.UMAP(n_components=2).fit_transform(resampled)
                                
                                projection_time = time.time() - projection_start
                                viz_stats['detailed_timings'][f'{method}_projection'] = projection_time
                                viz_stats['projections_processed'] += 1
                                viz_bar.text = f"{method.upper()} computed"
                                viz_bar()
                                
                                # Add traces for each class
                                viz_bar.text = f"Adding {method.upper()} traces"
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
                                viz_bar()
                            
                            # Layout configuration
                            viz_bar.text = "Configuring layout"
                            fig.update_layout(
                                title_text=f"Oversampling Comparison: {sampler_name}",
                                width=300 * n_cols,
                                height=400
                            )
                            
                            figure_time = time.time() - figure_start
                            viz_stats['detailed_timings']['plotly_2d_figure'] = figure_time
                            viz_bar.text = f"2D Plotly figure created"
                            viz_bar()
                        
                        # Record total visualization time
                        total_viz_time = time.time() - viz_start
                        viz_stats['detailed_timings']['total_visualization'] = total_viz_time
                    
                    except Exception as e:
                        viz_bar.text = f"Plotly visualization failed"
                        logger.error(f"Plotly visualization failed: {str(e)}")
                        raise
                
                else:
                    # Matplotlib Visualization
                    try:
                        if dimensions == 3:
                            viz_bar.text = "Creating 3D Matplotlib visualization"
                            figure_start = time.time()
                            
                            # Create figure with 2 columns (original and resampled)
                            fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
                            viz_bar()
                            
                            # Project data to 3D using PCA
                            viz_bar.text = "Computing 3D PCA projection"
                            pca_start = time.time()
                            
                            pca = PCA(n_components=3, random_state=random_state)
                            orig_proj = pca.fit_transform(original.values)
                            res_proj = pca.transform(resampled)
                            
                            pca_time = time.time() - pca_start
                            viz_stats['detailed_timings']['3d_pca'] = pca_time
                            viz_bar.text = "3D PCA computed"
                            viz_bar()
                            
                            # Original data plot
                            viz_bar.text = "Creating original data subplot"
                            ax1 = fig.add_subplot(121, projection='3d')
                            sc1 = ax1.scatter(
                                orig_proj[:, 0], orig_proj[:, 1], orig_proj[:, 2],
                                alpha=0.5, label='Original'
                            )
                            ax1.set_title("Original Data (3D PCA)")
                            ax1.set_xlabel("PC1")
                            ax1.set_ylabel("PC2")
                            ax1.set_zlabel("PC3")
                            viz_bar()
                            
                            # Resampled data plot
                            viz_bar.text = "Creating resampled data subplot"
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
                            viz_bar()
                            
                            figure_time = time.time() - figure_start
                            viz_stats['detailed_timings']['matplotlib_3d_figure'] = figure_time
                            viz_bar.text = f"3D Matplotlib figure created"
                            viz_bar()
                            
                        else:
                            # 2D visualization
                            n_plots = len(projections) + 1
                            viz_bar.text = f"Creating 2D Matplotlib visualization: {n_plots} subplots"
                            figure_start = time.time()
                            
                            fig = plt.figure(figsize=(figsize[0] * n_plots/3, figsize[1]), dpi=dpi)
                            viz_bar()
                            
                            # Original data plot (always PCA)
                            viz_bar.text = "Creating original data subplot (PCA)"
                            plt.subplot(1, n_plots, 1)
                            _plot_projection(
                                original.values,
                                labels=None,
                                title="Original Data (PCA)",
                                method='pca',
                                random_state=random_state
                            )
                            viz_bar()
                            
                            # Resampled data plots
                            for i, method in enumerate(projections, 2):
                                viz_stats['current_projection'] = method
                                viz_bar.text = f"Creating {method.upper()} subplot"
                                projection_start = time.time()
                                
                                plt.subplot(1, n_plots, i)
                                _plot_projection(
                                    resampled,
                                    labels,
                                    title=f"Resampled Data ({method.upper()})\n{sampler_name}",
                                    method=method,
                                    random_state=random_state
                                )
                                
                                projection_time = time.time() - projection_start
                                viz_stats['detailed_timings'][f'{method}_projection'] = projection_time
                                viz_stats['projections_processed'] += 1
                                viz_bar.text = f"{method.upper()} subplot created"
                                viz_bar()
                            
                            figure_time = time.time() - figure_start
                            viz_stats['detailed_timings']['matplotlib_2d_figure'] = figure_time
                            viz_bar.text = f"2D Matplotlib figure created"
                            viz_bar()
                        
                        # Layout and finishing
                        viz_bar.text = "Applying final layout"
                        plt.tight_layout()
                        viz_bar()
                        
                        # Record total visualization time
                        total_viz_time = time.time() - viz_start
                        viz_stats['detailed_timings']['total_visualization'] = total_viz_time
                    
                    except Exception as e:
                        viz_bar.text = "Matplotlib visualization failed"
                        logger.error(f"Matplotlib visualization failed: {str(e)}")
                        raise

            # STAGE 3: Save and Display
            with progress.bar("Save and Display", total=2, unit="steps") as final_bar:
                
                # STAGE 3.1: Save Plot
                if save_plot:
                    final_bar.text = "Saving visualization"
                    save_start = time.time()
                    
                    viz_stats['stage'] = "Saving Plot"
                    
                    dim_suffix = "3d" if dimensions == 3 else "2d"
                    filename = f"oversampling_{sampler_name.lower().replace('+', '_')}_{dim_suffix}"
                    
                    if interactive_backend.lower() == 'plotly':
                        filename += ".html"
                        fig.write_html(filename)
                        final_bar.text = f"Saved interactive plot to {filename}"
                    else:
                        filename += ".png"
                        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
                        final_bar.text = f"Saved static plot to {filename}"
                    
                    save_time = time.time() - save_start
                    viz_stats['detailed_timings']['saving'] = save_time
                    final_bar()
                else:
                    final_bar.text = "Skipping save"
                    final_bar()
                
                # STAGE 3.2: Display Plot
                final_bar.text = "Finalizing visualization"
                display_start = time.time()
                
                viz_stats['stage'] = "Displaying Plot"
                
                if show_plot:
                    if interactive_backend.lower() == 'plotly':
                        fig.show()
                        final_bar.text = "Interactive plot displayed"
                    else:
                        plt.show()
                        final_bar.text = "Static plot displayed"
                    return_value = fig
                else:
                    if interactive_backend.lower() == 'matplotlib':
                        plt.close(fig)
                    final_bar.text = "Visualization completed"
                    return_value = None
                
                display_time = time.time() - display_start
                viz_stats['detailed_timings']['displaying'] = display_time
                final_bar()
        
        else:
            # Without progress bar
            if dimensions not in (2, 3):
                raise ValueError("Dimensions must be 2 or 3")
            
            if not isinstance(original, pd.DataFrame):
                raise ValueError("original must be a pandas DataFrame")
            if not isinstance(resampled, np.ndarray):
                raise ValueError("resampled must be a numpy array")
            if len(original) == 0 or len(resampled) == 0:
                raise ValueError("Input data cannot be empty")
            
            # Handle backend
            if interactive_backend.lower() == 'plotly':
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                except ImportError:
                    logger.warning("Plotly not available, falling back to matplotlib")
                    interactive_backend = 'matplotlib'
            
            # Sample data if needed
            if len(original) > max_samples:
                rng = np.random.RandomState(random_state)
                orig_idx = rng.choice(len(original), min(max_samples, len(original)), replace=False)
                original = original.iloc[orig_idx]
            
            if len(resampled) > max_samples:
                rng = np.random.RandomState(random_state)
                res_idx = rng.choice(len(resampled), min(max_samples, len(resampled)), replace=False)
                resampled = resampled[res_idx]
                labels = labels[res_idx]
            
            # Create visualization without progress bar
            try:
                if interactive_backend.lower() == 'plotly':
                    # Plotly visualization without progress bar
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
                        # 2D visualization with Plotly
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
                            else:
                                proj = umap.UMAP(n_components=2).fit_transform(resampled)
                            
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
                    
                    # Save plot if requested
                    if save_plot:
                        dim_suffix = "3d" if dimensions == 3 else "2d"
                        filename = f"oversampling_{sampler_name.lower().replace('+', '_')}_{dim_suffix}.html"
                        fig.write_html(filename)
                        logger.info(f"Saved interactive visualization to {filename}")
                    
                    # Display plot if requested
                    if show_plot:
                        fig.show()
                        return_value = fig
                    else:
                        return_value = None
                
                else:
                    # Matplotlib visualization without progress bar
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
                        # 2D visualization
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
                    
                    # Save plot if requested
                    if save_plot:
                        dim_suffix = "3d" if dimensions == 3 else "2d"
                        filename = f"oversampling_{sampler_name.lower().replace('+', '_')}_{dim_suffix}.png"
                        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
                        logger.info(f"Saved visualization to {filename}")
                    
                    # Display plot if requested
                    if show_plot:
                        plt.show()
                        return_value = fig
                    else:
                        plt.close(fig)
                        return_value = None
                
                # Record timings for non-progress bar mode
                viz_stats['detailed_timings']['total_visualization'] = time.time() - viz_start
                
            except Exception as e:
                logger.error(f"Visualization failed: {str(e)}")
                if 'fig' in locals() and interactive_backend.lower() == 'matplotlib':
                    plt.close(fig)
                raise
        
        # Calculate total time
        total_time = sum(viz_stats['detailed_timings'].values())
        viz_stats['detailed_timings']['total'] = total_time
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Visualization completed in {total_time:.2f}s")
            logger.info(f"Visualization Statistics:")
            logger.info(f"  - Original samples: {viz_stats['original_samples']}")
            logger.info(f"  - Resampled samples: {viz_stats['resampled_samples']}")
            logger.info(f"  - Projections processed: {viz_stats['projections_processed']}")
            logger.info(f"  - Backend used: {viz_stats['backend_used']}")
            logger.info(f"  - Dimensions: {viz_stats['dimensions']}")
            if viz_stats['sampling_applied']:
                logger.info(f"  - Sampling applied: Yes")
                if 'original_samples_after_sampling' in viz_stats:
                    logger.info(f"    Original after sampling: {viz_stats['original_samples_after_sampling']}")
                if 'resampled_samples_after_sampling' in viz_stats:
                    logger.info(f"    Resampled after sampling: {viz_stats['resampled_samples_after_sampling']}")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Detailed Timings:")
            for stage, timing in viz_stats['detailed_timings'].items():
                logger.debug(f"  - {stage}: {timing:.2f}s")
        
        return return_value if 'return_value' in locals() else None
    
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {viz_stats.get('stage', 'unknown')}, "
        error_context += f"projection: {viz_stats.get('current_projection', 'none')}, "
        error_context += f"backend: {viz_stats.get('backend_used', 'unknown')})"
        
        logger.error(f"Visualization failed{error_context}: {str(e)}")
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
    """Project high-dim data to 2D for visualization with controls.
    
    Args:
        data: Input feature matrix
        labels: Target labels (None for unlabeled data)
        title: Plot title
        method: Projection method ('pca' or 'tsne' or 'umap)
        random_state: Random seed
        alpha: Point transparency
        legend: Whether to show legend
    """
    # Validate method
    if method not in ['pca', 'tsne', 'umap']:
        raise ValueError(f"Invalid projection method: {method}. Choose 'pca', 'tsne', or 'umap'")
    
    # Perform projection
    if method == 'pca':
        proj = PCA(n_components=2, random_state=random_state).fit_transform(data)
    elif method == 'tsne':
        # t-SNE
        proj = TSNE(n_components=2, random_state=random_state).fit_transform(data)
    else:
        # UMAP
        proj = umap.UMAP(n_components=2, random_state=random_state).fit_transform(data)
    
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
    """Display results report."""
    # Class distribution comparison
    table = Table(
        title=f"\n{sampler_name} Oversampling Results, Processing time: {elapsed_time:.2f}s",
        box=box.ROUNDED,
        style="bold green",
        title_justify="left",
        border_style="green",
        show_lines=True
    )
    table.add_column("Class")
    table.add_column("Original Count")
    table.add_column("New Count")
    table.add_column("Change")
    
    for cls in original_counts.index:
        orig = original_counts[cls]
        new = new_counts.get(cls, 0)
        change = f"{(new-orig)/orig:+.2%}" if orig != 0 else "N/A"
        table.add_row(str(cls), str(orig), str(new), change, style="bold yellow")
    
    console.print(table)
    
    # Quality metrics if available
    if metrics:
        metric_table = Table(
            title="\nQuality Metrics",
            box=box.ROUNDED,
            style="bold green",
            title_justify="left",
            border_style="green",
            show_lines=True
        )
        metric_table.add_column("Metric")
        metric_table.add_column("Value")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                metric_table.add_row(name, f"{value:.4f}", style="bold yellow")
            else:
                metric_table.add_row(name, str(value), style="bold yellow")
        
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
    
    # Display comparison
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
    """Display comparison table."""
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
    csv_path: Optional[Path] = None,
    pkl_path: Optional[Path] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """Load and validate training data with configurable enhancements.
    
    Args:
        enhanced: Use improved validation pipeline (default: True)
        **kwargs: Forwarded to helper functions
    
    Returns:
        Tuple of (cleaned DataFrame, preprocessing artifacts)
    
    Raises:
        RuntimeError: If loading fails, with troubleshooting info
    """
    try:
        print_color(f"\nStarting loading and validation of training data...", 'yellow')
        
        enhanced_mode = Fore.GREEN + Style.BRIGHT + 'ENABLED' + Style.RESET_ALL if enhanced else Fore.RED + Style.BRIGHT + 'DISABLED' + Style.RESET_ALL
        
        print_color("\nTRAINING DATA LOADING AND VALIDATION PIPELINE", 'magenta')
        print_color(f"  └─ Enhanced mode: {enhanced_mode}", 'magenta')
        print_color("-" * 50, 'magenta')
        
        # Use get_preprocessing_outputs to locate the output files
        if csv_path or pkl_path is None:
            csv_path, pkl_path, test_config, preprocessing_summary = get_preprocessing_outputs(
                config_path=None,
                base_results_dir=None,
                base_preprocessing_dir=None,
                interactive=True
            )
        
        if not csv_path or not pkl_path:
            csv_status = Fore.GREEN + Style.BRIGHT + 'YES' + Style.RESET_ALL if csv_path else Fore.YELLOW + Style.BRIGHT + 'NO' + Style.RESET_ALL
            pkl_status = Fore.GREEN + Style.BRIGHT + 'YES' + Style.RESET_ALL if pkl_path else Fore.YELLOW + Style.BRIGHT + 'NO' + Style.RESET_ALL
            print_color(f"\nError: Could not locate preprocessing outputs", 'red')
            print_color(f"  ├─ CSV file found: {csv_status}", 'red')
            print_color(f"  └─ PKL file found: {pkl_status}", 'red')
            raise RuntimeError("Preprocessing outputs not found. Run preprocessing first.")
        
        print_color(f"Located preprocessing outputs:", 'green')
        print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_path}", 'green')
        print_color(f"  └─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_path}", 'green')
        
        # Track loading progress and statistics
        loading_stats = {
            'stage': 'Initializing',
            'start_time': time.time(),
            'csv_size_bytes': 0,
            'pkl_size_bytes': 0,
            'rows_loaded': 0,
            'features_count': 0,
            'memory_usage_mb': 0,
            'detailed_timings': {},
            'warnings_issued': 0
        }
        
        if not enhanced:
            # Legacy mode
            print_color(f"\nStarting data loading in legacy mode...", 'yellow')
            
            legacy_start = time.time()
            
            try:
                # Load artifacts
                pkl_file = Path(pkl_path)
                loading_stats['pkl_size_bytes'] = pkl_file.stat().st_size
                
                print_color(f"\nLoading preprocessing artifacts...", 'cyan')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    artifacts = joblib.load(pkl_file)
                
                # Extract feature names
                feature_names = artifacts.get("feature_names", [])
                if not feature_names:
                    print_color(f"\nError: No feature names found in artifacts", 'red')
                    raise ValueError("No feature names found in artifacts")
                
                loading_stats['features_count'] = len(feature_names)
                print_color(f"\nFound {Fore.YELLOW + Style.BRIGHT}{loading_stats['features_count']} features", 'green')
                
                # Chunked loading
                csv_file = Path(csv_path)
                loading_stats['csv_size_bytes'] = csv_file.stat().st_size
                
                chunksize = kwargs.get('chunk_size', 100000)
                df_chunks = []
                total_rows = 0
                
                print_color(f"\nLoading CSV data in chunks...", 'cyan')
                for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize)):
                    rows_before = len(chunk)
                    chunk = chunk.drop_duplicates().dropna(subset=feature_names + ["Label"])
                    rows_after = len(chunk)
                    rows_removed = rows_before - rows_after
                    
                    df_chunks.append(chunk)
                    total_rows += rows_after
                    
                    if rows_removed > 0:
                        loading_stats['warnings_issued'] += 1
                        print_color(f"\nChunk {i+1}: Removed {Fore.YELLOW + Style.BRIGHT}{rows_removed} problematic rows", 'green')
                
                df = pd.concat(df_chunks, ignore_index=True)
                loading_stats['rows_loaded'] = len(df)
                
                # Check memory usage
                loading_stats['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
                
                legacy_time = time.time() - legacy_start
                loading_stats['detailed_timings']['legacy_loading'] = legacy_time
                
                print_color(f"\nLegacy loading completed:", 'green')
                print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_file}", 'green')
                print_color(f"  |   └─ CSV file size: {Fore.YELLOW + Style.BRIGHT}{loading_stats['csv_size_bytes']:,} bytes", 'green')
                print_color(f"  ├─ PKL artifacts: {Fore.MAGENTA + Style.BRIGHT}{pkl_file}", 'green')
                print_color(f"  |   └─ PKL file size: {Fore.YELLOW + Style.BRIGHT}{loading_stats['pkl_size_bytes']:,} bytes", 'green')
                print_color(f"  ├─ Total rows loaded: {Fore.YELLOW + Style.BRIGHT}{loading_stats['rows_loaded']:,}", 'green')
                print_color(f"  ├─ Memory usage: {Fore.YELLOW + Style.BRIGHT}{loading_stats['memory_usage_mb']:.2f} MB", 'green')
                print_color(f"  └─ Loading time: {Fore.YELLOW + Style.BRIGHT}{legacy_time:.2f}s", 'green')
                
                return df, artifacts
                
            except Exception as e:
                print_color(f"\nError in legacy loading mode:", 'red')
                print_color(f"  ├─ Error type: {Fore.YELLOW + Style.BRIGHT}{type(e).__name__}", 'red')
                print_color(f"  └─ Error message: {Fore.WHITE + Style.BRIGHT}{str(e)}", 'red')
                raise
        
        # Enhanced mode
        print_color(f"\nStarting data loading in enhanced mode...", 'yellow')
        
        # STAGE 1.1: File Validation
        loading_stats['stage'] = "File Validation"
        print_color(f"\nValidating preprocessing outputs...", 'cyan')
        validation_start = time.time()
        
        csv_file = Path(csv_path)
        pkl_file = Path(pkl_path)
        
        if not csv_file.exists():
            print_color(f"\nError: CSV file not found: {csv_path}", 'red')
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        if not pkl_file.exists():
            print_color(f"\nError: PKL file not found: {pkl_path}", 'red')
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        csv_file_size = csv_file.stat().st_size
        pkl_file_size = pkl_file.stat().st_size
        
        if csv_file_size >= 1024**3:
            csv_file_size_display = f"{csv_file_size / 1024**3:.2f} GB"
        elif csv_file_size >= 1024**2:
            csv_file_size_display = f"{csv_file_size / 1024**2:.2f} MB"
        elif csv_file_size >= 1024:
            csv_file_size_display = f"{csv_file_size / 1024:.2f} KB"
        else:
            csv_file_size_display = f"{csv_file_size} bytes"
        
        if pkl_file_size >= 1024**3:
            pkl_file_size_display = f"{pkl_file_size / 1024**3:.2f} GB"
        elif pkl_file_size >= 1024**2:
            pkl_file_size_display = f"{pkl_file_size / 1024**2:.2f} MB"
        elif pkl_file_size >= 1024:
            pkl_file_size_display = f"{pkl_file_size / 1024:.2f} KB"
        else:
            pkl_file_size_display = f"{pkl_file_size} bytes"
        
        loading_stats['csv_size_bytes'] = csv_file_size
        loading_stats['pkl_size_bytes'] = pkl_file_size
        loading_stats['csv_file_size_display'] = csv_file_size_display
        loading_stats['pkl_file_size_display'] = pkl_file_size_display
        
        print_color(f"\nPreprocessing outputs found:", 'green')
        print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_file.name}", 'green')
        print_color(f"  |   └─ CSV size: {Fore.CYAN + Style.BRIGHT}{csv_file_size_display}", 'green')
        print_color(f"  └─ PKL file: {Fore.MAGENTA + Style.BRIGHT}{pkl_file.name}", 'green')
        print_color(f"      └─ PKL size: {Fore.CYAN + Style.BRIGHT}{pkl_file_size_display}", 'green')
        
        validation_time = time.time() - validation_start
        loading_stats['detailed_timings']['file_validation'] = validation_time
        
        # STAGE 1.2: Artifacts Loading
        loading_stats['stage'] = "Artifacts Loading"
        print_color(f"\nLoading preprocessing artifacts...", 'yellow')
        artifacts_start = time.time()
        
        try:
            artifacts = load_preprocessing_artifacts(pkl_file, **kwargs)
            
            # Validate artifacts structure
            feature_names = artifacts.get("feature_names", [])
            if not feature_names:
                print_color(f"\nWarning: 'feature_names' not found in artifacts", 'yellow')
                loading_stats['warnings_issued'] += 1
                # Try to extract from other sources
                if "columns" in artifacts:
                    artifacts["feature_names"] = artifacts["columns"]
                    feature_names = artifacts["feature_names"]
            
            loading_stats['features_count'] = len(feature_names)
            
            print_color(f"\nArtifacts loaded successfully", 'green')
            print_color(f"  ├─ Features count: {Fore.YELLOW + Style.BRIGHT}{loading_stats['features_count']}", 'green')
            print_color(f"  └─ Artifacts keys: {Fore.YELLOW + Style.BRIGHT}{', '.join(artifacts.keys())}", 'green')
            
        except Exception as e:
            print_color(f"\nError loading artifacts:", 'red')
            print_color(f"  ├─ Error type: {Fore.YELLOW + Style.BRIGHT}{type(e).__name__}", 'red')
            print_color(f"  └─ Error message: {Fore.WHITE + Style.BRIGHT}{str(e)}", 'red')
            raise
        
        artifacts_time = time.time() - artifacts_start
        loading_stats['detailed_timings']['artifacts_loading'] = artifacts_time
        
        # STAGE 1.3: Configuration Setup
        loading_stats['stage'] = "Configuration"
        print_color(f"\nSetting up data configuration...", 'cyan')
        config_start = time.time()
        
        # Get configuration from kwargs or use defaults
        chunk_size = kwargs.get('chunk_size', 100000)
        validation_mode = kwargs.get('validation_mode', 'strict')
        
        print_color(f"\nData configuration:", 'green')
        print_color(f"  ├─ Chunk size: {Fore.YELLOW + Style.BRIGHT}{chunk_size:,}", 'green')
        print_color(f"  └─ Validation mode: {Fore.YELLOW + Style.BRIGHT}{validation_mode}", 'green')
        
        config_time = time.time() - config_start
        loading_stats['detailed_timings']['configuration'] = config_time
        
        # STAGE 2: Data Loading and Cleaning Phase
        loading_stats['stage'] = "Data Loading and Cleaning"
        print_color(f"\nLoading and cleaning data...", 'yellow')
        data_loading_start = time.time()
        
        try:
            df = load_and_clean_data(
                csv_file,
                artifacts["feature_names"],
                **kwargs
            )
            
            loading_stats['rows_loaded'] = len(df)
            
            print_color(f"\nData loaded and cleaned successfully:", 'green')
            print_color(f"  └─ Rows loaded: {Fore.YELLOW + Style.BRIGHT}{loading_stats['rows_loaded']:,}", 'green')
            
        except Exception as e:
            print_color(f"\nError loading and cleaning data:", 'red')
            print_color(f"  ├─ Error type: {Fore.YELLOW + Style.BRIGHT}{type(e).__name__}", 'red')
            print_color(f"  └─ Error message: {Fore.YELLOW + Style.BRIGHT}{str(e)}", 'red')
            raise
        
        data_loading_time = time.time() - data_loading_start
        loading_stats['detailed_timings']['data_loading_cleaning'] = data_loading_time
        
        # STAGE 2.1: Memory Optimization
        loading_stats['stage'] = "Memory Optimization"
        print_color(f"\nOptimizing memory usage...", 'yellow')
        memory_start = time.time()
        
        try:
            # Calculate initial memory usage
            initial_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                if df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
            
            # Calculate optimized memory usage
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = initial_memory - optimized_memory
            loading_stats['memory_usage_mb'] = optimized_memory
            
            print_color(f"\nMemory optimization completed successfully:", 'green')
            print_color(f"  ├─ Initial memory: {Fore.YELLOW + Style.BRIGHT}{initial_memory:.3f} MB", 'green')
            print_color(f"  ├─ Optimized memory: {Fore.CYAN + Style.BRIGHT}{optimized_memory:.3f} MB", 'green')
            print_color(f"  └─ Memory saved: {Fore.GREEN + Style.BRIGHT}{memory_saved:.3f} MB", 'green')
            
        except Exception as e:
            print_color(f"\nWarning: Memory optimization failed:", 'yellow')
            print_color(f"  └─ Error: {Fore.YELLOW + Style.BRIGHT}{str(e)}", 'yellow')
            loading_stats['warnings_issued'] += 1
            loading_stats['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
        
        memory_time = time.time() - memory_start
        loading_stats['detailed_timings']['memory_optimization'] = memory_time
        
        # STAGE 3: Handle Class Imbalance Phase
        loading_stats['stage'] = "Class Imbalance Handling"
        print_color(f"\nHandling class imbalance...", 'yellow')
        imbalance_start = time.time()
        
        # Initialize imbalance-specific statistics
        imbalance_stats = {
            'original_distribution': {},
            'balanced_distribution': {},
            'original_ratio': 0.0,
            'balanced_ratio': 0.0,
            'threshold_used': None,
            'threshold_method': None,
            'samples_added': 0,
            'balancing_applied': False,
            'balancing_success': False,
            'threshold_analysis': None
        }
        
        try:
            # Capture original distribution before balancing
            if "Label" in df.columns:
                original_counts = df["Label"].value_counts()
                imbalance_stats['original_distribution'] = original_counts.to_dict()
                
                if len(original_counts) > 1:
                    imbalance_stats['original_ratio'] = original_counts.max() / original_counts.min()
                    
                    print_color(f"\nOriginal class distribution:", 'cyan')
                    print_color(f"  ├─ Total samples: {Fore.CYAN + Style.BRIGHT}{len(df):,}", 'cyan')
                    print_color(f"  ├─ Classes: {Fore.YELLOW + Style.BRIGHT}{len(original_counts)}", 'cyan')
                    print_color(f"  └─ Imbalance ratio: {Fore.YELLOW + Style.BRIGHT}{imbalance_stats['original_ratio']:.2f}:1", 'cyan')
            
            # Apply class imbalance handling
            df = handle_class_imbalance(df, artifacts, **kwargs)
            
            # Check class distribution after handling
            if "Label" in df.columns:
                label_counts = df["Label"].value_counts()
                total_samples = len(df)
                
                # Update imbalance statistics
                imbalance_stats['balanced_distribution'] = label_counts.to_dict()
                imbalance_stats['samples_added'] = total_samples - len(pd.DataFrame(imbalance_stats['original_distribution'], index=[0]).T.sum())
                imbalance_stats['balancing_applied'] = imbalance_stats['samples_added'] > 0
                
                if len(label_counts) > 1:
                    imbalance_stats['balanced_ratio'] = label_counts.max() / label_counts.min()
                
                print_color(f"\nTotal samples after balancing: {Fore.CYAN + Style.BRIGHT}{total_samples:,}", 'green')
                
                # Display class distribution with visual indicators
                for idx, (label, count) in enumerate(label_counts.items()):
                    percentage = (count / total_samples) * 100
                    prefix = "  └─ " if idx == len(label_counts) - 1 else "  ├─ "
                    
                    # Add change indicator if balancing was applied
                    if imbalance_stats['balancing_applied']:
                        original_count = imbalance_stats['original_distribution'].get(label, 0)
                        change = count - original_count
                        change_str = f"({Fore.GREEN + Style.BRIGHT}+{change:,}{Style.RESET_ALL})" if change > 0 else ""
                        print_color(f"{prefix}Label '{label}': {Fore.YELLOW + Style.BRIGHT}{count:,} samples ({percentage:.2f}%) {change_str}", 'green')
                    else:
                        print_color(f"{prefix}Label '{label}': {Fore.YELLOW + Style.BRIGHT}{count:,} samples ({percentage:.2f}%)", 'green')
                
                # Calculate balance improvement
                if imbalance_stats['balancing_applied'] and imbalance_stats['original_ratio'] > 1.0:
                    improvement = ((imbalance_stats['original_ratio'] - imbalance_stats['balanced_ratio']) / imbalance_stats['original_ratio']) * 100
                    
                    print_color(f"\nBalance improvement:", 'green')
                    print_color(f"  ├─ Original ratio: {Fore.YELLOW + Style.BRIGHT}{imbalance_stats['original_ratio']:.2f}:1", 'green')
                    print_color(f"  ├─ Balanced ratio: {Fore.CYAN + Style.BRIGHT}{imbalance_stats['balanced_ratio']:.2f}:1", 'green')
                    print_color(f"  ├─ Improvement: {Fore.GREEN + Style.BRIGHT}{improvement:.1f}%", 'green')
                    print_color(f"  └─ Samples added: {Fore.MAGENTA + Style.BRIGHT}{imbalance_stats['samples_added']:,}", 'green')
                    
                    imbalance_stats['balancing_success'] = True
                
                # Check for severe imbalance
                min_percentage = (label_counts.min() / total_samples) * 100
                max_percentage = (label_counts.max() / total_samples) * 100
                
                if min_percentage < 10:
                    severity = "severe" if min_percentage < 5 else "moderate"
                    print_color(f"\n{severity.capitalize()} class imbalance persists:", 'yellow')
                    print_color(f"  ├─ Minority class: {Fore.RED + Style.BRIGHT}{min_percentage:.1f}%", 'yellow')
                    print_color(f"  ├─ Majority class: {Fore.YELLOW + Style.BRIGHT}{max_percentage:.1f}%", 'yellow')
                    print_color(f"  └─ Consider using class weights or additional balancing", 'yellow')
                    loading_stats['warnings_issued'] += 1
                elif imbalance_stats['balancing_applied']:
                    print_color(f"\nClass imbalance addressed successfully", 'green')
                    print_color(f"  ├─ Minority class: {Fore.GREEN + Style.BRIGHT}{min_percentage:.1f}%", 'green')
                    print_color(f"  ├─ Majority class: {Fore.CYAN + Style.BRIGHT}{max_percentage:.1f}%", 'green')
                    print_color(f"  └─ Distribution is now balanced", 'green')
                else:
                    print_color(f"\nNo balancing applied (ratio within threshold)", 'cyan')
                    print_color(f"  ├─ Minority class: {Fore.CYAN + Style.BRIGHT}{min_percentage:.1f}%", 'cyan')
                    print_color(f"  ├─ Majority class: {Fore.YELLOW + Style.BRIGHT}{max_percentage:.1f}%", 'cyan')
                    print_color(f"  └─ Current distribution is acceptable", 'cyan')
            else:
                print_color(f"\n'Label' column not found in data", 'yellow')
                loading_stats['warnings_issued'] += 1
                imbalance_stats['balancing_success'] = False
        
        except Exception as e:
            print_color(f"\nWarning: Class imbalance handling failed:", 'yellow')
            print_color(f"  ├─ Error type: {Fore.RED + Style.BRIGHT}{type(e).__name__}", 'yellow')
            print_color(f"  ├─ Error message: {Fore.YELLOW + Style.BRIGHT}{str(e)}", 'yellow')
            print_color(f"  └─ Continuing without class balancing...", 'yellow')
            loading_stats['warnings_issued'] += 1
            imbalance_stats['balancing_success'] = False
            
            # Log detailed error if verbose mode
            if kwargs.get('verbose', False):
                logger.error(f"Class imbalance handling error details: {traceback.format_exc()}")
        
        imbalance_time = time.time() - imbalance_start
        loading_stats['detailed_timings']['class_imbalance'] = imbalance_time
        
        # Store imbalance statistics in loading_stats for reporting
        loading_stats['imbalance_handling'] = imbalance_stats
        
        # Summary logging
        if kwargs.get('verbose', False) or kwargs.get('debug', False):
            logger.info(f"\nClass Imbalance Handling Summary:")
            logger.info(f"  - Processing time: {imbalance_time:.2f}s")
            logger.info(f"  - Balancing applied: {imbalance_stats['balancing_applied']}")
            logger.info(f"  - Balancing successful: {imbalance_stats['balancing_success']}")
            if imbalance_stats['balancing_applied']:
                logger.info(f"  - Samples added: {imbalance_stats['samples_added']:,}")
                logger.info(f"  - Original ratio: {imbalance_stats['original_ratio']:.2f}:1")
                logger.info(f"  - Balanced ratio: {imbalance_stats['balanced_ratio']:.2f}:1")
        
        # STAGE 3.1: Version-safe scaler handling
        loading_stats['stage'] = "Scaler Handling"
        print_color(f"\nHandling feature scaling...", 'yellow')
        scaler_start = time.time()
        
        if "scaler" in artifacts:
            try:
                feature_names = artifacts["feature_names"]
                
                # Check if features exist in DataFrame
                missing_features = [f for f in feature_names if f not in df.columns]
                if missing_features:
                    print_color(f"\nMissing features for scaling:", 'yellow')
                    print_color(f"  └─ Missing: {Fore.RED + Style.BRIGHT}{', '.join(missing_features[:5])}", 'yellow')
                    if len(missing_features) > 5:
                        print_color(f"      ... and {len(missing_features) - 5} more", 'yellow')
                    loading_stats['warnings_issued'] += 1
                
                # Version-safe scaler handling
                try:
                    if hasattr(artifacts['scaler'], 'feature_names_in_'):
                        feature_map = dict(zip(feature_names, artifacts['scaler'].feature_names_in_))
                        df = df.rename(columns=feature_map)
                        print_color(f"\nRenamed features for scikit-learn scaler compatibility", 'green')
                    
                    test_sample = df[feature_names].iloc[:1]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        artifacts["scaler"].transform(test_sample)
                    
                    # Apply transformation
                    df[feature_names] = artifacts["scaler"].transform(df[feature_names])
                    print_color(f"\nSuccessfully applied scaler transformation", 'green')
                    print_color(f"  └─ Scaled features: {Fore.YELLOW + Style.BRIGHT}{len(feature_names)}", 'green')
                
                except Exception as e:
                    print_color(f"\nScaler error: {str(e)}", 'red')
                    print_color(f"  └─ Creating new scaler...", 'red')
                    loading_stats['warnings_issued'] += 1
                    
                    new_scaler = MinMaxScaler().fit(df[feature_names])
                    artifacts["scaler"] = new_scaler
                    df[feature_names] = new_scaler.transform(df[feature_names])
                    print_color(f"\nNew scaler created and applied successfully", 'yellow')
            
            except Exception as e:
                print_color(f"\nScaler handling failed:", 'yellow')
                print_color(f"  └─ Error: {Fore.YELLOW + Style.BRIGHT}{str(e)}", 'yellow')
                loading_stats['warnings_issued'] += 1
        
        scaler_time = time.time() - scaler_start
        loading_stats['detailed_timings']['scaler_handling'] = scaler_time
        
        # STAGE 3.2: Feature Validation
        loading_stats['stage'] = "Feature Validation"
        print_color(f"\nValidating features...", 'yellow')
        validation_start = time.time()
        
        try:
            # Check for NaN values in features
            feature_columns = [col for col in feature_names if col in df.columns]
            nan_counts = df[feature_columns].isna().sum()
            total_nans = nan_counts.sum()
            
            if total_nans > 0:
                print_color(f"\nFound NaN values in features: {Fore.YELLOW + Style.BRIGHT}{total_nans}", 'green')
                
                # Show columns with most NaNs
                top_nan_cols = nan_counts.nlargest(5)
                for col, count in top_nan_cols.items():
                    percentage = (count / len(df)) * 100
                    prefix = "  └─ " if col == top_nan_cols.index[-1] else "  ├─ "
                    print_color(f"{prefix}{col}: {Fore.YELLOW + Style.BRIGHT}{count} NaNs ({percentage:.1f}%)", 'green')
                
                loading_stats['warnings_issued'] += 1
            else:
                print_color(f"\nNo NaN values found in features", 'white')
            
            # Check for infinite values
            inf_cols = []
            for col in feature_columns:
                if df[col].dtype in ['float32', 'float64']:
                    if np.isinf(df[col]).any():
                        inf_cols.append(col)
            
            if inf_cols:
                print_color(f"\nFound infinite values in features:", 'yellow')
                print_color(f"  └─ Columns with infinite values: {Fore.YELLOW + Style.BRIGHT}{', '.join(inf_cols)}", 'yellow')
                loading_stats['warnings_issued'] += 1
            else:
                print_color(f"\nNo infinite values found", 'white')
            
        except Exception as e:
            print_color(f"\nFeature validation failed:", 'red')
            print_color(f"  └─ Error: {Fore.YELLOW + Style.BRIGHT}{str(e)}", 'red')
            loading_stats['warnings_issued'] += 1
        
        validation_time = time.time() - validation_start
        loading_stats['detailed_timings']['feature_validation'] = validation_time
        
        # STAGE 4: Final Validation and Summary
        loading_stats['stage'] = "Final Checks"
        print_color(f"\nPerforming final checks...", 'yellow')
        final_start = time.time()
        
        # Calculate total time
        total_time = time.time() - loading_stats['start_time']
        loading_stats['detailed_timings']['total'] = total_time
        
        # Summarize warnings
        if loading_stats['warnings_issued'] > 0:
            print_color(f"\nWarning summary:", 'green')
            print_color(f"  └─ Total warnings issued: {Fore.YELLOW + Style.BRIGHT}{loading_stats['warnings_issued']}", 'green')
        else:
            print_color(f"\nNo warnings issued during loading", 'green')
        
        final_time = time.time() - final_start
        loading_stats['detailed_timings']['final_checks'] = final_time
        
        # STAGE 4.1: Summary Report
        loading_stats['stage'] = "Summary Report"
        print_color(f"\nGenerating summary report...", 'yellow')
        report_start = time.time()
        
        print_color(f"\nData validation completed successfully!", 'green')
        print_color(f"  ├─ CSV file: {Fore.MAGENTA + Style.BRIGHT}{csv_file}", 'green')
        print_color(f"  |   └─ CSV file size: {Fore.YELLOW + Style.BRIGHT}{loading_stats['csv_size_bytes']:,} bytes", 'green')
        print_color(f"  ├─ PKL artifacts file: {Fore.MAGENTA + Style.BRIGHT}{pkl_file}", 'green')
        print_color(f"  |   └─ PKL file size: {Fore.YELLOW + Style.BRIGHT}{loading_stats['pkl_size_bytes']:,} bytes", 'green')
        print_color(f"  ├─ Total time: {Fore.CYAN + Style.BRIGHT}{total_time:.2f}s", 'green')
        print_color(f"  ├─ Rows loaded: {Fore.YELLOW + Style.BRIGHT}{loading_stats['rows_loaded']:,}", 'green')
        print_color(f"  ├─ Features count: {Fore.BLUE + Style.BRIGHT}{loading_stats['features_count']}", 'green')
        print_color(f"  └─ Memory usage: {Fore.YELLOW + Style.BRIGHT}{loading_stats['memory_usage_mb']:.3f} MB", 'green')
        
        # Show detailed timings if debug mode
        if kwargs.get('debug', False):
            print_color(f"\nDetailed timings:", 'green')
            timing_items = list(loading_stats['detailed_timings'].items())
            for i, (stage, timing) in enumerate(timing_items):
                prefix = "  ├─ " if i < len(timing_items) - 1 else "  └─ "
                print_color(f"{prefix}{stage}: {Fore.YELLOW + Style.BRIGHT}{timing:.2f}s", 'green')
        
        report_time = time.time() - report_start
        loading_stats['detailed_timings']['report_generation'] = report_time
        
        return df, artifacts
        
    except Exception as e:
        print_color(f"\nData loading failed:", 'red')
        print_color(f"  ├─ Error type: {Fore.YELLOW + Style.BRIGHT}{type(e).__name__}", 'red')
        print_color(f"  ├─ Error message: {Fore.WHITE + Style.BRIGHT}{str(e)}", 'red')
        print_color(f"  └─ Stage: {Fore.YELLOW + Style.BRIGHT}{locals().get('loading_stats', {}).get('stage', 'unknown')}", 'red')
        print_color(f"\nTroubleshooting steps:", 'yellow')
        print_color(f"  ├─ 1. Verify preprocessing outputs exist:", 'yellow')
        print_color(f"  │   ├─ Run get_preprocessing_outputs() to locate files", 'yellow')
        print_color(f"  │   └─ Check file permissions and paths", 'yellow')
        print_color(f"  ├─ 2. Test with enhanced=False for legacy loader", 'yellow')
        print_color(f"  ├─ 3. Check CSV and PKL file integrity:", 'yellow')
        print_color(f"  │   ├─ Verify CSV can be read with pd.read_csv()", 'yellow')
        print_color(f"  │   └─ Verify PKL can be loaded with joblib.load()", 'yellow')
        print_color(f"  └─ 4. Ensure sufficient memory and disk space", 'yellow')
        
        raise RuntimeError("Data loading failed") from e

def create_synthetic_data(
    logger: logging.Logger,
    verbose: Optional[bool] = False
) -> Tuple[pd.DataFrame, Dict]:
    """Generate realistic synthetic data with logging and progress tracking."""

    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL

    # Track synthetic data generation progress and statistics
    generation_stats = {
        'stage': 'Initializing',
        'total_samples': 10000,
        'total_features': 20,
        'classes_generated': 0,
        'samples_generated': 0,
        'features_created': 0,
        'noise_added': False,
        'dataframe_created': False,
        'artifacts_prepared': False,
        'memory_optimized': False,
        'detailed_timings': {},
        'data_quality_metrics': {},
        'memory_usage_mb': 0.0
    }

    print(f"\n{green}Realistic Synthetic Dataset Generation...{reset}")

    try:
        # Progress helper: define all stage titles
        titles = [
            "Initializing Synthetic Data Generation",
            "DataFrame Creation and Optimization",
            "Artifacts Preparation and Finalization"
        ]
        progress = ProgressHelper(titles)

        # STAGE 1
        with progress.bar("Initializing Synthetic Data Generation", total=7, unit="stages") as bar:

            # STAGE 1.1
            bar.text = "Configuring synthetic data parameters..."
            config_start = time.time()

            num_samples = 10000
            num_features = 20
            np.random.seed(42)

            config_time = time.time() - config_start
            generation_stats['detailed_timings']['configuration'] = config_time
            bar.text = f"{green}Configuration set ({num_samples} samples, {num_features} features){reset}"
            bar()

            # STAGE 1.2
            bar.text = "Generating normal class samples..."
            normal_class_start = time.time()

            X_normal = np.random.normal(0.2, 0.1, (num_samples // 2, num_features))
            generation_stats['samples_generated'] += num_samples // 2
            generation_stats['classes_generated'] += 1

            normal_class_time = time.time() - normal_class_start
            generation_stats['detailed_timings']['normal_class_generation'] = normal_class_time
            bar.text = f"{green}Normal class generated ({num_samples//2} samples){reset}"
            bar()

            # STAGE 1.3
            bar.text = "Generating attack class samples..."
            attack_class_start = time.time()

            X_attack = np.random.normal(0.8, 0.1, (num_samples // 2, num_features))
            generation_stats['samples_generated'] += num_samples // 2
            generation_stats['classes_generated'] += 1

            attack_class_time = time.time() - attack_class_start
            generation_stats['detailed_timings']['attack_class_generation'] = attack_class_time
            bar.text = f"{green}Attack class generated ({num_samples//2} samples){reset}"
            bar()

            # STAGE 1.4
            bar.text = "Combining class datasets..."
            combination_start = time.time()

            X = np.vstack([X_normal, X_attack])
            y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

            combination_time = time.time() - combination_start
            generation_stats['detailed_timings']['data_combination'] = combination_time
            bar.text = f"{green}Classes combined ({len(X)} total samples){reset}"
            bar()

            # STAGE 1.5
            bar.text = "Adding realistic noise..."
            noise_start = time.time()

            X += np.random.normal(0, 0.05, X.shape)
            generation_stats['noise_added'] = True

            noise_time = time.time() - noise_start
            generation_stats['detailed_timings']['noise_addition'] = noise_time
            bar.text = f"{green}Realistic noise added{reset}"
            bar()

            # STAGE 1.6
            bar.text = "Clipping data to valid range..."
            clipping_start = time.time()

            X = np.clip(X, 0, 1)

            clipping_time = time.time() - clipping_start
            generation_stats['detailed_timings']['data_clipping'] = clipping_time
            bar.text = f"{green}Data clipped to [0,1] range{reset}"
            bar()

            # STAGE 1.7
            bar.text = "Creating feature names..."
            feature_naming_start = time.time()

            feature_names = [f"feature_{i}" for i in range(num_features)]
            generation_stats['features_created'] = len(feature_names)

            feature_naming_time = time.time() - feature_naming_start
            generation_stats['detailed_timings']['feature_naming'] = feature_naming_time
            bar.text = f"{green}Feature names created ({len(feature_names)} features){reset}"
            bar()

        # STAGE 2
        with progress.bar("DataFrame Creation and Optimization", total=3, unit="steps") as df_bar:

            # 2.1
            df_bar.text = "Creating DataFrame structure..."
            dataframe_start = time.time()

            df = pd.DataFrame(X, columns=feature_names)
            df["Label"] = y.astype("int8")
            generation_stats['dataframe_created'] = True

            dataframe_time = time.time() - dataframe_start
            generation_stats['detailed_timings']['dataframe_creation'] = dataframe_time
            df_bar.text = f"{green}DataFrame created ({len(df)} rows, {len(df.columns)} columns){reset}"
            df_bar()

            # 2.2
            df_bar.text = "Optimizing memory usage..."
            memory_start = time.time()

            initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

            for col in df.select_dtypes(include=["float"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="float")

            optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            generation_stats['memory_usage_mb'] = optimized_memory
            generation_stats['memory_optimized'] = True

            memory_time = time.time() - memory_start
            generation_stats['detailed_timings']['memory_optimization'] = memory_time
            df_bar.text = f"{green}Memory optimized ({initial_memory - optimized_memory:.2f}MB saved){reset}"
            df_bar()

            # 2.3
            df_bar.text = "Assessing data quality..."
            quality_start = time.time()

            generation_stats['data_quality_metrics'] = {
                'has_nans': df.isna().any().any(),
                'min_values': df.min().tolist(),
                'max_values': df.max().tolist(),
                'mean_values': df.mean().tolist(),
                'std_values': df.std().tolist(),
                'class_distribution': df["Label"].value_counts().to_dict(),
                'memory_usage_mb': generation_stats['memory_usage_mb'],
                'shape': df.shape
            }

            quality_time = time.time() - quality_start
            generation_stats['detailed_timings']['quality_assessment'] = quality_time
            df_bar.text = f"{green}Data quality assessed{reset}"
            df_bar()

        # STAGE 3
        with progress.bar("Artifacts Preparation and Finalization", total=2, unit="steps") as final_bar:

            # 3.1
            final_bar.text = "Preparing artifacts..."
            artifacts_start = time.time()

            artifacts = {
                "feature_names": feature_names,
                "scaler": None,
                "feature_importances": np.random.rand(num_features),
                "chunksize": 100000,
                "synthetic": True,
                "class_names": ["Normal", "Attack"],
                "missing_values": 0,
                "data_quality": generation_stats['data_quality_metrics'],
                "generation_stats": generation_stats
            }
            generation_stats['artifacts_prepared'] = True

            artifacts_time = time.time() - artifacts_start
            generation_stats['detailed_timings']['artifacts_preparation'] = artifacts_time
            final_bar.text = f"{green}Artifacts prepared ({len(artifacts)} items){reset}"
            final_bar()

            # 3.2
            final_bar.text = "Generating final summary..."
            summary_start = time.time()

            total_time = sum(generation_stats['detailed_timings'].values())
            generation_stats['detailed_timings']['total'] = total_time

            if verbose:
                logger.info("Synthetic dataset generation completed successfully")
                logger.info("Generation Summary:")
                logger.info(f"  - Total samples: {generation_stats['total_samples']:,}")
                logger.info(f"  - Total features: {generation_stats['total_features']}")
                logger.info(f"  - Classes generated: {generation_stats['classes_generated']}")
                logger.info(f"  - Memory usage: {generation_stats['memory_usage_mb']:.2f} MB")
                logger.info(f"  - Total time: {total_time:.3f}s")
                logger.info(f"Class distribution:\n{df['Label'].value_counts().to_string()}")
                logger.info(f"Feature statistics:\n{df.describe().to_string()}")
                logger.info(f"{blue}Synthetic Data Generation Summary{reset}")
                logger.info(f"{cyan}Dataset Characteristics:{reset}")
                logger.info(f"  - Samples: {len(df):,}")
                logger.info(f"  - Features: {len(feature_names)}")
                logger.info(f"  - Classes: {generation_stats['classes_generated']}")
                logger.info(f"  - Memory: {generation_stats['memory_usage_mb']:.2f} MB")
                logger.info(f"  - Data shape: {df.shape}")
                logger.info(f"{cyan}Quality Metrics:{reset}")
                logger.info(f"  - Missing values: {artifacts['missing_values']}")
                logger.info(f"  - Value range: [0, 1]")
                logger.info(f"  - Class balance: {df['Label'].value_counts().to_dict()}")
                logger.info(f"{cyan}Processing Timings:{reset}")
                for stage, timing in generation_stats['detailed_timings'].items():
                    if stage != "total":
                        logger.info(f"  - {stage}: {timing:.3f}s")
                logger.info(f"  - Total time: {total_time:.3f}s")

            summary_time = time.time() - summary_start
            generation_stats['detailed_timings']['final_summary'] = summary_time
            final_bar.text = f"{green}Synthetic data generation completed{reset}"
            final_bar()

        return df, artifacts

    except Exception as e:
        error_context = f" (stage: {generation_stats.get('stage', 'unknown')})"
        logger.error(f"Synthetic data generation failed{error_context}: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")

        if verbose:
            logger.error(f"{red}Synthetic data generation failed at stage: {generation_stats.get('stage', 'unknown')}{reset}")
            logger.error(f"{red}Error: {str(e)}{reset}")

        raise RuntimeError("Synthetic data generation failed") from e

def prepare_dataloaders(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    batch_size: int = 64,
    test_size: float = 0.2,
    verbose: Optional[bool] = False,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Prepare optimized dataloaders with proper stratification and imbalance handling
    with progress tracking.
    
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
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track dataloader preparation progress and statistics
    preparation_stats = {
        'stage': 'Initializing',
        'total_samples': len(df),
        'features_processed': 0,
        'classes_detected': 0,
        'train_samples': 0,
        'val_samples': 0,
        'imbalance_detected': False,
        'smote_applied': False,
        'sampler_created': False,
        'dataloaders_created': False,
        'class_distribution': {},
        'imbalance_ratio': 0.0,
        'detailed_timings': {},
        'memory_usage_mb': 0.0,
        'device_used': None
    }
    
    try:
        print(f"\n{green}Preparing Optimized Dataloader...{reset}")

        # Progress helper: define all stage titles
        titles = [
            "Initial Setup and Validation",
            "Data Processing and Balancing",
            "Final Dataloader Configuration"
        ]
        progress = ProgressHelper(titles)

        # STAGE 1: Initial Setup and Validation
        with progress.bar("Initial Setup and Validation", total=7, unit="stages") as bar:
            
            # STAGE 1.1: Input Validation
            bar.text = "Validating input data..."
            validation_start = time.time()
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                bar.text = f"{red}Input DataFrame is invalid{reset}"
                raise ValueError("Input DataFrame is empty or invalid")
                
            if 'feature_names' not in artifacts:
                bar.text = f"{red}Missing feature names in artifacts{reset}"
                raise ValueError("Artifacts must contain feature_names")
                
            feature_names = artifacts['feature_names']
            missing_columns = [col for col in feature_names + ['Label'] if col not in df.columns]
            if missing_columns:
                bar.text = f"{red}Missing columns in DataFrame{reset}"
                raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

            validation_time = time.time() - validation_start
            preparation_stats['detailed_timings']['input_validation'] = validation_time
            bar.text = f"{green}Input data validated ({len(df)} samples){reset}"
            bar()
            
            # STAGE 1.2: Device Detection
            bar.text = "Detecting available device..."
            device_start = time.time()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pin_memory = device.type == 'cuda'
            preparation_stats['device_used'] = str(device)
            
            device_time = time.time() - device_start
            preparation_stats['detailed_timings']['device_detection'] = device_time
            bar.text = f"{green}Using device: {device}{reset}"
            bar()
            
            # STAGE 1.3: Feature and Label Extraction
            bar.text = "Extracting features and labels..."
            extraction_start = time.time()
            
            X = df[feature_names].values
            y = df['Label'].values
            preparation_stats['features_processed'] = len(feature_names)
            preparation_stats['classes_detected'] = len(np.unique(y))
            
            extraction_time = time.time() - extraction_start
            preparation_stats['detailed_timings']['feature_extraction'] = extraction_time
            bar.text = f"{green}Features extracted ({len(feature_names)} features){reset}"
            bar()
            
            # STAGE 1.4: Tensor Conversion
            bar.text = "Converting to PyTorch tensors..."
            tensor_start = time.time()
            
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Calculate memory usage
            tensor_memory = (X_tensor.element_size() * X_tensor.nelement() + y_tensor.element_size() * y_tensor.nelement()) / 1024 / 1024
            preparation_stats['memory_usage_mb'] = tensor_memory
            
            tensor_time = time.time() - tensor_start
            preparation_stats['detailed_timings']['tensor_conversion'] = tensor_time
            bar.text = f"{green}Tensors created ({tensor_memory:.2f} MB){reset}"
            bar()
            
            # STAGE 1.5: Stratified Split
            bar.text = "Performing stratified split..."
            split_start = time.time()
            
            sss = StratifiedShuffleSplit(
                n_splits=1, 
                test_size=test_size, 
                random_state=random_state
            )
            train_idx, val_idx = next(sss.split(X, y))
            preparation_stats['train_samples'] = len(train_idx)
            preparation_stats['val_samples'] = len(val_idx)
            
            split_time = time.time() - split_start
            preparation_stats['detailed_timings']['stratified_split'] = split_time
            bar.text = f"{green}Stratified split completed ({len(train_idx)} train, {len(val_idx)} val){reset}"
            bar()
            
            # STAGE 1.6: Class Distribution Analysis
            bar.text = "Analyzing class distribution..."
            analysis_start = time.time()
            
            class_counts = torch.bincount(y_tensor[train_idx])
            preparation_stats['class_distribution']['original'] = class_counts.tolist()
            preparation_stats['imbalance_ratio'] = torch.max(class_counts).item() / torch.min(class_counts).item()
            preparation_stats['imbalance_detected'] = preparation_stats['imbalance_ratio'] > 2.0
            
            analysis_time = time.time() - analysis_start
            preparation_stats['detailed_timings']['distribution_analysis'] = analysis_time
            bar.text = f"{green}Class distribution analyzed (ratio: {preparation_stats['imbalance_ratio']:.1f}){reset}"
            bar()
            
            # Close setup bar
            bar.text = "Starting data processing..."
            bar()
        
        # STAGE 2: Data Processing and Balancing
        with progress.bar("Data Processing and Balancing", total=3, unit='steps') as process_bar:
            
            # STAGE 2.1: Class Imbalance Handling
            process_bar.text = "Handling class imbalance..."
            imbalance_start = time.time()
            
            # Threshold for extreme imbalance
            if torch.min(class_counts) < 1000:
                process_bar.text = "Applying SMOTE for extreme imbalance..."
                preparation_stats['imbalance_detected'] = True
                
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
                    preparation_stats['class_distribution']['after_smote'] = class_counts.tolist()
                    preparation_stats['smote_applied'] = True
                    preparation_stats['train_samples'] = len(train_idx)
                    
                    process_bar.text = f"{green}SMOTE applied (new ratio: {torch.max(class_counts).item() / torch.min(class_counts).item():.1f}){reset}"
                    
                except Exception as e:
                    process_bar.text = f"{yellow}SMOTE failed, using original data{reset}"
                    logger.warning(f"{yellow}Warning: SMOTE failed: {str(e)}{reset}")
                    # Continue with original data
            
            else:
                process_bar.text = f"{green}Class distribution acceptable{reset}"
            
            imbalance_time = time.time() - imbalance_start
            preparation_stats['detailed_timings']['imbalance_handling'] = imbalance_time
            process_bar()
            
            # STAGE 2.2: Sampler Creation
            process_bar.text = "Creating data sampler..."
            sampler_start = time.time()
            
            if torch.min(class_counts) < 1000:
                class_weights = 1.0 / class_counts.float()
                sample_weights = class_weights[y_tensor[train_idx]]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(train_idx),
                    replacement=True
                )
                preparation_stats['sampler_created'] = 'WeightedRandomSampler'
                process_bar.text = f"{green}Weighted sampler created{reset}"
            else:
                sampler = RandomSampler(train_idx)
                preparation_stats['sampler_created'] = 'RandomSampler'
                process_bar.text = f"{green}Random sampler created{reset}"
            
            sampler_time = time.time() - sampler_start
            preparation_stats['detailed_timings']['sampler_creation'] = sampler_time
            process_bar()
            
            # STAGE 2.3: Dataset Creation
            process_bar.text = "Creating Tensor datasets..."
            dataset_start = time.time()
            
            train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
            val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
            
            dataset_time = time.time() - dataset_start
            preparation_stats['detailed_timings']['dataset_creation'] = dataset_time
            process_bar.text = f"{green}Datasets created ({len(train_dataset)} train, {len(val_dataset)} val){reset}"
            process_bar()
        
        # STAGE 3: Dataloader Configuration
        with progress.bar("Dataloader Configuration", total=2, unit='steps') as loader_bar:
            
            # STAGE 3.1: Dataloader Creation
            loader_bar.text = "Creating dataloaders..."
            dataloader_start = time.time()
            
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
            
            preparation_stats['dataloaders_created'] = True
            
            dataloader_time = time.time() - dataloader_start
            preparation_stats['detailed_timings']['dataloader_creation'] = dataloader_time
            loader_bar.text = f"{green}Dataloaders created{reset}"
            loader_bar()
            
            # STAGE 3.2: Final Validation
            loader_bar.text = "Performing final validation..."
            final_start = time.time()
            
            input_size = X.shape[1]
            num_classes = len(class_counts)
            
            # Validate dataloaders
            train_batches = len(train_loader)
            val_batches = len(val_loader)
            
            final_time = time.time() - final_start
            preparation_stats['detailed_timings']['final_validation'] = final_time
            loader_bar.text = f"{green}Final validation completed{reset}"
            loader_bar()
        
        # STAGE 4: Final Summary and Reporting
        total_time = sum(preparation_stats['detailed_timings'].values())
        preparation_stats['detailed_timings']['total'] = total_time
        
        # Display summary
        if verbose:
            logger.info(f"\n{blue}Dataloader Preparation Summary{reset}")
            logger.info(f"{cyan}Data Overview:{reset}")
            logger.info(f"  - Total samples: {preparation_stats['total_samples']:,}")
            logger.info(f"  - Features: {preparation_stats['features_processed']}")
            logger.info(f"  - Classes: {preparation_stats['classes_detected']}")
            logger.info(f"  - Device: {preparation_stats['device_used']}")
            logger.info(f"  - Memory usage: {preparation_stats['memory_usage_mb']:.2f} MB")
            
            logger.info(f"\n{cyan}Split Configuration:{reset}")
            logger.info(f"  - Training samples: {preparation_stats['train_samples']:,}")
            logger.info(f"  - Validation samples: {preparation_stats['val_samples']:,}")
            logger.info(f"  - Test size: {test_size:.1%}")
            logger.info(f"  - Batch size: {batch_size * 2} (2x for training)")
            
            logger.info(f"\n{cyan}Class Distribution:{reset}")
            original_dist = preparation_stats['class_distribution'].get('original', [])
            if preparation_stats['smote_applied']:
                after_smote_dist = preparation_stats['class_distribution'].get('after_smote', [])
                logger.info(f"  - Original: {original_dist}")
                logger.info(f"  - After SMOTE: {after_smote_dist}")
                logger.info(f"  - Imbalance ratio: {preparation_stats['imbalance_ratio']:.1f} -> {max(after_smote_dist)/min(after_smote_dist):.1f}")
            else:
                logger.info(f"  - Distribution: {original_dist}")
                logger.info(f"  - Imbalance ratio: {preparation_stats['imbalance_ratio']:.1f}")
            
            logger.info(f"\n{cyan}Processing Details:{reset}")
            logger.info(f"  - Imbalance detected: {'Yes' if preparation_stats['imbalance_detected'] else 'No'}")
            logger.info(f"  - SMOTE applied: {'Yes' if preparation_stats['smote_applied'] else 'No'}")
            logger.info(f"  - Sampler type: {preparation_stats['sampler_created']}")
            logger.info(f"  - Dataloaders created: {'Yes' if preparation_stats['dataloaders_created'] else 'No'}")
            logger.info(f"  - Total processing time: {total_time:.3f}s")
            
            logger.info(f"\n{cyan}Dataloader Configuration:{reset}")
            logger.info(f"  - Input size: {input_size}")
            logger.info(f"  - Number of classes: {num_classes}")
            logger.info(f"  - Training batches: {train_batches}")
            logger.info(f"  - Validation batches: {val_batches}")
            logger.info(f"  - Pin memory: {pin_memory}")
            
            # Log final success message
            logger.info(f"\n{green}Successfully prepared dataloaders with {len(train_dataset)} training and {len(val_dataset)} validation samples{reset}")
        
        return train_loader, val_loader, input_size, num_classes
        
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {preparation_stats.get('stage', 'unknown')})"
        logger.error(f"{red}Dataloader preparation failed{error_context}: {str(e)}{reset}")
        raise RuntimeError("DataLoader preparation failed") from e

def visualize_data_distribution(
    df: pd.DataFrame,
    filename: Path,
    max_samples: int = 10000,
    verbose: Optional[bool] = False,
    random_state: int = 42
) -> Optional[Path]:
    """
    Visualize data distribution using PCA and save plot with progress tracking.
    
    Args:
        df: DataFrame containing features and labels
        filename: Path to save visualization
        max_samples: Maximum samples to plot (for large datasets)
        random_state: Random seed for sampling
        
    Returns:
        Path to saved visualization or None if failed
        
    Raises:
        ValueError: If input data is invalid
    """
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track visualization progress and statistics
    visualization_stats = {
        'stage': 'Initializing',
        'total_samples': len(df),
        'sampled_samples': 0,
        'features_processed': 0,
        'pca_components': 2,
        'plot_created': False,
        'plot_saved': False,
        'memory_usage_mb': 0.0,
        'detailed_timings': {},
        'data_statistics': {},
        'pca_statistics': {},
        'plot_characteristics': {}
    }
    
    try:
        print(f"\n{green}Creating PCA Visualization of Data Distribution...{reset}")
        
        # Progress helper: define all stage titles
        titles = [
            "Data Preparation and Validation",
            "PCA Transformation and Analysis",
            "PCA Plot Generation and Styling",
            "Final PCA Summary and Reporting"
        ]
        progress = ProgressHelper(titles)
        
        # STAGE 1: Data Preparation and Validation
        with progress.bar("Data Preparation and Validation", total=6, unit="stages") as bar:
            
            # STAGE 1.1: Input Validation
            bar.text = "Validating input data..."
            validation_start = time.time()
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                bar.text = f"{red}Input DataFrame is invalid{reset}"
                raise ValueError("Input DataFrame is empty or invalid")
                
            if 'Label' not in df.columns:
                bar.text = f"{red}Missing Label column{reset}"
                raise ValueError("DataFrame must contain 'Label' column")
            
            validation_time = time.time() - validation_start
            visualization_stats['detailed_timings']['input_validation'] = validation_time
            bar.text = f"{green}Input data validated{reset}"
            bar()
            
            # STAGE 1.2: Data Statistics Calculation
            bar.text = "Calculating data statistics..."
            stats_start = time.time()
            
            # Calculate data statistics
            feature_columns = [col for col in df.columns if col != 'Label']
            visualization_stats['features_processed'] = len(feature_columns)
            visualization_stats['data_statistics'] = {
                'total_samples': len(df),
                'total_features': len(feature_columns),
                'class_distribution': df['Label'].value_counts().to_dict(),
                'feature_shapes': df[feature_columns].shape,
                'data_types': df[feature_columns].dtypes.value_counts().to_dict(),
                'missing_values': df[feature_columns].isnull().sum().sum()
            }
            
            stats_time = time.time() - stats_start
            visualization_stats['detailed_timings']['statistics_calculation'] = stats_time
            bar.text = f"{green}Data statistics calculated{reset}"
            bar()
            
            # STAGE 1.3: Data Sampling
            bar.text = "Sampling data for visualization..."
            sampling_start = time.time()
            
            # Sample data if too large
            if len(df) > max_samples:
                df_sampled = df.sample(max_samples, random_state=random_state)
                visualization_stats['sampled_samples'] = len(df_sampled)
                bar.text = f"{yellow}Data sampled ({len(df_sampled)}/{len(df)} samples){reset}"
            else:
                df_sampled = df
                visualization_stats['sampled_samples'] = len(df_sampled)
                bar.text = f"{green}Using full dataset ({len(df_sampled)} samples){reset}"
            
            sampling_time = time.time() - sampling_start
            visualization_stats['detailed_timings']['data_sampling'] = sampling_time
            bar()
            
            # STAGE 1.4: Feature Preparation
            bar.text = "Preparing features for PCA..."
            feature_start = time.time()
            
            # Prepare data
            X = df_sampled.drop(columns=['Label']).values
            y = df_sampled['Label'].values
            
            # Calculate memory usage
            memory_usage = X.nbytes / 1024 / 1024
            visualization_stats['memory_usage_mb'] = memory_usage
            
            feature_time = time.time() - feature_start
            visualization_stats['detailed_timings']['feature_preparation'] = feature_time
            bar.text = f"{green}Features prepared ({memory_usage:.1f} MB){reset}"
            bar()
            
            # STAGE 1.5: PCA Configuration
            bar.text = "Configuring PCA..."
            pca_config_start = time.time()
            
            # Apply PCA
            pca = PCA(n_components=2)
            
            pca_config_time = time.time() - pca_config_start
            visualization_stats['detailed_timings']['pca_configuration'] = pca_config_time
            bar.text = f"{green}PCA configured (2 components){reset}"
            bar()
            
            # STAGE 1.6: Preparation Completion
            bar.text = "Preparation completion..."
            completion_start = time.time()
            
            # Final preparation steps
            visualization_stats['stage'] = 'Data Preparation Complete'
            
            completion_time = time.time() - completion_start
            visualization_stats['detailed_timings']['preparation_completion'] = completion_time
            bar.text = f"{green}Data preparation completed{reset}"
            bar()
        
        # STAGE 2: PCA Transformation and Analysis
        with progress.bar("PCA Transformation and Analysis", total=3, unit="steps") as pca_bar:
            
            # STAGE 2.1: PCA Fitting
            pca_bar.text = "Fitting PCA transformation..."
            pca_fit_start = time.time()
            
            X_pca = pca.fit_transform(X)
            
            pca_fit_time = time.time() - pca_fit_start
            visualization_stats['detailed_timings']['pca_fitting'] = pca_fit_time
            pca_bar.text = f"{green}PCA transformation fitted{reset}"
            pca_bar()
            
            # STAGE 2.2: PCA Statistics Calculation
            pca_bar.text = "Calculating PCA statistics..."
            pca_stats_start = time.time()
            
            # Calculate PCA statistics
            visualization_stats['pca_statistics'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': sum(pca.explained_variance_ratio_),
                'component_1_variance': pca.explained_variance_ratio_[0],
                'component_2_variance': pca.explained_variance_ratio_[1],
                'pca_components_shape': pca.components_.shape,
                'transformed_data_shape': X_pca.shape
            }
            
            pca_stats_time = time.time() - pca_stats_start
            visualization_stats['detailed_timings']['pca_statistics'] = pca_stats_time
            pca_bar.text = f"{green}PCA statistics calculated{reset}"
            pca_bar()
            
            # STAGE 2.3: Data Range Analysis
            pca_bar.text = "Analyzing data ranges..."
            range_start = time.time()
            
            # Analyze data ranges for plotting
            x_range = (X_pca[:, 0].min(), X_pca[:, 0].max())
            y_range = (X_pca[:, 1].min(), X_pca[:, 1].max())
            visualization_stats['plot_characteristics']['data_ranges'] = {
                'x_range': x_range,
                'y_range': y_range,
                'x_span': x_range[1] - x_range[0],
                'y_span': y_range[1] - y_range[0]
            }
            
            range_time = time.time() - range_start
            visualization_stats['detailed_timings']['range_analysis'] = range_time
            pca_bar.text = f"{green}Data ranges analyzed{reset}"
            pca_bar()
        
        # STAGE 3: PCA Plot Creation and Styling
        with progress.bar("PCA Plot Generation and Styling", total=4, unit="steps") as plot_bar:
            
            # STAGE 3.1: Figure Initialization
            plot_bar.text = "Initializing plot figure..."
            figure_start = time.time()
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            figure_time = time.time() - figure_start
            visualization_stats['detailed_timings']['figure_initialization'] = figure_time
            plot_bar.text = f"{green}Plot figure initialized{reset}"
            plot_bar()
            
            # STAGE 3.2: Scatter Plot Creation
            plot_bar.text = "Creating scatter plot..."
            scatter_start = time.time()
            
            scatter = plt.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=y, alpha=0.5, cmap='viridis',
                edgecolors='w', linewidths=0.5
            )
            
            scatter_time = time.time() - scatter_start
            visualization_stats['detailed_timings']['scatter_creation'] = scatter_time
            plot_bar.text = f"{green}Scatter plot created{reset}"
            plot_bar()
            
            # STAGE 3.3: Plot Styling and Annotations
            plot_bar.text = "Applying plot styling..."
            styling_start = time.time()
            
            plt.title("Data Distribution (PCA)", fontsize=14, fontweight='bold')
            plt.colorbar(scatter, label='Class')
            plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            plt.grid(alpha=0.3)
            
            # Add informative text box
            text_str = f'Samples: {len(X_pca):,}\nFeatures: {len(feature_columns)}\nTotal Variance: {sum(pca.explained_variance_ratio_):.1%}'
            plt.gca().text(0.02, 0.98, text_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            styling_time = time.time() - styling_start
            visualization_stats['detailed_timings']['plot_styling'] = styling_time
            visualization_stats['plot_created'] = True
            plot_bar.text = f"{green}Plot styling applied{reset}"
            plot_bar()
            
            # STAGE 3.4: Plot Saving
            plot_bar.text = "Saving plot to file..."
            saving_start = time.time()
            
            # Ensure directory exists
            filename.parent.mkdir(parents=True, exist_ok=True)
            
            # Save plot
            plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
            plt.close()
            
            saving_time = time.time() - saving_start
            visualization_stats['detailed_timings']['plot_saving'] = saving_time
            visualization_stats['plot_saved'] = True
            plot_bar.text = f"{green}Plot saved successfully{reset}"
            plot_bar()
        
        # STAGE 4: Final Summary and Reporting
        with progress.bar("Final PCA Summary and Reporting", total=2, unit="steps") as final_bar:
            
            # STAGE 4.1: File Validation
            final_bar.text = "Validating saved plot..."
            validation_start = time.time()
            
            # Verify file was created
            if not filename.exists():
                final_bar.text = f"{red}Plot file not found{reset}"
                raise FileNotFoundError(f"Plot file was not created: {filename}")
            
            file_size = filename.stat().st_size / 1024  # Size in KB
            visualization_stats['plot_characteristics']['file_size_kb'] = file_size
            
            validation_time = time.time() - validation_start
            visualization_stats['detailed_timings']['file_validation'] = validation_time
            final_bar.text = f"{green}Plot file validated ({file_size:.1f} KB){reset}"
            final_bar()
            
            # STAGE 4.2: Final Summary
            final_bar.text = "Generating visualization summary..."
            summary_start = time.time()
            
            # Calculate total time from detailed timings (excluding total)
            timing_keys = [k for k in visualization_stats['detailed_timings'].keys() if k != 'total']
            total_time = sum(visualization_stats['detailed_timings'][k] for k in timing_keys)
            visualization_stats['detailed_timings']['total'] = total_time
            
            # Display summary
            if verbose:
                logger.info(f"\n{blue}PCA Visualization Summary{reset}")
                logger.info(f"{cyan}Data Overview:{reset}")
                logger.info(f"  - Original samples: {visualization_stats['data_statistics']['total_samples']:,}")
                logger.info(f"  - Sampled samples: {visualization_stats['sampled_samples']:,}")
                logger.info(f"  - Features: {visualization_stats['features_processed']}")
                logger.info(f"  - Classes: {len(visualization_stats['data_statistics']['class_distribution'])}")
                logger.info(f"  - Class distribution: {visualization_stats['data_statistics']['class_distribution']}")
                
                logger.info(f"\n{cyan}PCA Analysis:{reset}")
                logger.info(f"  - Components: {visualization_stats['pca_components']}")
                logger.info(f"  - Total variance explained: {visualization_stats['pca_statistics']['total_variance_explained']:.1%}")
                logger.info(f"  - Component 1 variance: {visualization_stats['pca_statistics']['component_1_variance']:.1%}")
                logger.info(f"  - Component 2 variance: {visualization_stats['pca_statistics']['component_2_variance']:.1%}")
                
                logger.info(f"\n{cyan}Plot Information:{reset}")
                logger.info(f"  - Output file: {filename}")
                logger.info(f"  - File size: {visualization_stats['plot_characteristics']['file_size_kb']:.1f} KB")
                logger.info(f"  - Plot created: {'Yes' if visualization_stats['plot_created'] else 'No'}")
                logger.info(f"  - Plot saved: {'Yes' if visualization_stats['plot_saved'] else 'No'}")
                logger.info(f"  - Memory usage: {visualization_stats['memory_usage_mb']:.1f} MB")
                
                logger.info(f"\n{cyan}Processing Timings:{reset}")
                for stage, timing in visualization_stats['detailed_timings'].items():
                    if stage != 'total':
                        logger.info(f"  - {stage}: {timing:.3f}s")
                logger.info(f"  - Total time: {total_time:.3f}s")
            
            summary_time = time.time() - summary_start
            visualization_stats['detailed_timings']['final_summary'] = summary_time
            final_bar.text = f"{green}PCA visualization completed successfully{reset}"
            final_bar()
        
        if verbose:
            logger.info(f"{green}Saved PCA visualization of data distribution {magenta}{filename}{reset}")
        return filename
        
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {visualization_stats.get('stage', 'unknown')})"
        logger.error(f"{red}Could not create PCA visualization{error_context}: {str(e)}{reset}")
        return None

# Training and validation functions
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
        self._last_lr = [base_lr]
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self._last_lr = [lr]
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

def train_epoch(
    # Core Training Parameters
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    
    # Basic Training Parameters
    grad_clip: float = GRADIENT_CLIP,
    grad_accum_steps: int = GRADIENT_ACCUMULATION_STEPS,
    scaler: Optional[GradScaler] = None,
    verbose: Optional[bool] = False,
    warmup_scheduler: Optional[Any] = None,
    
    # Progress Parameters
    progress_bar: Optional[bool] = True,
    progress_bar_desc: Optional[str] = "Training Epoch",
    progress_bar_unit: Optional[str] = "batches",
    
    # Metrics Parameters
    track_metrics: Optional[bool] = True,
    log_frequency: Optional[int] = 10,
    
    # Tracking Parameters
    track_detailed_timing: Optional[bool] = False,
    track_gradient_norms: Optional[bool] = False,
    track_memory_usage: Optional[bool] = False,
    track_comprehensive_stats: Optional[bool] = False,
    
    # Error Handling Parameters
    continue_on_error: Optional[bool] = False,
    graceful_degradation: Optional[bool] = True,
    max_retries: Optional[int] = 3,
    
    # Context Parameters
    run_id: Optional[str] = None,
    epoch_number: Optional[int] = None,
    
    # Export Parameters
    save_results: Optional[bool] = None,
    save_metadata: Optional[bool] = None,
    
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    Train model for one epoch with gradient handling, optional mixed precision, and progress tracking.
    
    Args:
        model: Model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Target device (cuda/cpu)
        grad_clip: Maximum gradient norm (default: 1.0)
        grad_accum_steps: Gradient accumulation steps (default: 4)
        scaler: Gradient scaler for mixed precision
        verbose: Enable verbose logging
        warmup_scheduler: Warmup scheduler for learning rate
        
        # Progress Bar Parameters
        progress_bar: Enable alive-progress bar (default: True)
        progress_bar_desc: Description for progress bar (default: "Training Epoch")
        progress_bar_unit: Unit for progress bar (default: "batches")
        
        # Metrics Display
        track_metrics: Track metrics for progress bar display (default: True)
        log_frequency: Frequency for logging (default: 10)
        
        # Enhanced Tracking
        track_detailed_timing: Track detailed timing breakdowns (default: False)
        track_gradient_norms: Track gradient norms during optimization (default: False)
        track_memory_usage: Track memory usage (default: False)
        track_comprehensive_stats: Track training statistics (default: False)
        
        # Error Handling
        continue_on_error: Continue training on batch errors (default: False)
        graceful_degradation: Return partial results on failure (default: True)
        max_retries: Maximum retries for batch errors (default: 3)
        
        # Context
        run_id: Run ID for tracking context
        epoch_number: Epoch number for context
        
        # Export
        save_results: Save comprehensive results to JSON file (default: False)
        save_metadata: Save metadata to JSON file (default: False)
        
    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    
    # Setup colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Initialize default return values
    default_loss = float('inf')
    default_metrics = {
        'loss': default_loss,
        'accuracy': 0.0,
        'num_batches': 0,
        'num_samples': 0,
        'training_completed': False,
        'error': None,
        'progress_bar_used': False,
        'run_id': run_id,
        'epoch_number': epoch_number
    }
    
    # Start timing
    start_time = datetime.datetime.now()
    epoch_start_time = time.time()
    
    # Initialize variables for progress tracking
    num_batches = 0
    num_samples = 0
    total_loss = 0.0
    correct_predictions = 0
    pbar = None
    pbar_context = None
    
    # Track recent metrics for moving averages
    recent_losses = deque(maxlen=10)
    recent_accuracies = deque(maxlen=10)
    recent_batch_times = deque(maxlen=10)
    
    # Tracking variables
    gradient_norms = []
    learning_rates = []
    batch_timings = [] if track_detailed_timing else None
    gradient_accumulations = 0
    gradient_clippings = 0
    optimizer_steps = 0
    warmup_updates = 0
    running_loss = 0.0
    
    # Store outputs for final accuracy calculation
    last_outputs = None
    
    # Export flags
    save_results = save_results if save_results is not None else False
    save_metadata = save_metadata if save_metadata is not None else False
    
    try:
        print(f"{green}\nTraining Epochs...{reset}")
        # Parameter validation
        if model is None:
            raise ValueError("Model is required for training")
        if loader is None:
            raise ValueError("DataLoader is required for training")
        if criterion is None:
            raise ValueError("Loss criterion is required for training")
        if optimizer is None:
            raise ValueError("Optimizer is required for training")
        
        # Set defaults
        progress_bar = progress_bar if progress_bar is not None else True
        progress_bar_desc = progress_bar_desc or "Training Epoch"
        progress_bar_unit = progress_bar_unit or "batch"
        track_metrics = track_metrics if track_metrics is not None else True
        log_frequency = log_frequency or 10
        continue_on_error = continue_on_error or False
        graceful_degradation = graceful_degradation if graceful_degradation is not None else True
        max_retries = max_retries or 3
        
        # Device configuration
        if device is None:
            device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        
        # Set model to training mode
        model.train()
        
        # Initialize training statistics
        training_stats = {
            'stage': 'Initializing',
            'total_batches': len(loader),
            'completed_batches': 0,
            'current_batch': 0,
            'gradient_accumulations': 0,
            'gradient_clippings': 0,
            'optimizer_steps': 0,
            'warmup_updates': 0,
            'total_loss': 0.0,
            'running_loss': 0.0,
            'correct_predictions': 0,
            'total_samples': 0,
            'batch_times': [],
            'gradient_norms': [],
            'learning_rates': [],
            'detailed_timings': {} if track_detailed_timing else None,
            'memory_usage_mb': 0.0 if track_memory_usage else None,
            'device_used': str(device),
            'config': {
                'grad_clip': grad_clip,
                'grad_accum_steps': grad_accum_steps,
                'mixed_precision': scaler is not None,
                'warmup_scheduler': warmup_scheduler is not None,
                'track_detailed_timing': track_detailed_timing,
                'track_gradient_norms': track_gradient_norms,
                'track_memory_usage': track_memory_usage,
                'track_comprehensive_stats': track_comprehensive_stats,
                'save_results': save_results,
                'save_metadata': save_metadata
            }
        }

        # Progress helper: define all stage titles
        titles = [f"{progress_bar_desc if progress_bar_desc else 'Training Epoch'}"]
        
        # Setup progress bar
        if progress_bar:
            try:
                progress = ProgressHelper(titles)
                pbar_context = progress.bar(
                    title=progress_bar_desc if progress_bar_desc else "Training Epoch",
                    total=len(loader),
                    unit=progress_bar_unit
                )
                pbar = pbar_context.__enter__()
            
            except ImportError:
                logger.error(f"{red}alive-progress not available, progress bar disabled{reset}")
                pbar = None
                pbar_context = None
            except Exception as e:
                logger.error(f"{red}Failed to initialize progress bar: {e}{reset}")
                pbar = None
                pbar_context = None
        else:
            if verbose:
                logger.info(f"{yellow}Progress bar disabled{reset}")
        
        # Display initial configuration
        if verbose:
            print(f"\n{cyan}{'-'*40}")
            print(f"{magenta}TRAINING CONFIGURATION")
            print(f"{cyan}{'-'*40}{reset}")
            
            # Configuration details
            config_items = [
                (f"{cyan}├─ Device:{reset}", f"{yellow}{training_stats['device_used']}{reset}"),
                (f"{cyan}├─ Total batches:{reset}", f"{yellow}{training_stats['total_batches']}{reset}"),
                (f"{cyan}├─ Gradient accumulation steps:{reset}", f"{yellow}{grad_accum_steps}{reset}"),
                (f"{cyan}├─ Gradient clipping:{reset}", f"{yellow}{grad_clip}{reset}"),
                (f"{cyan}├─ Mixed precision:{reset}", f"{yellow}{'Enabled' if scaler else 'Disabled'}{reset}"),
                (f"{cyan}├─ Warmup scheduler:{reset}", f"{yellow}{'Enabled' if warmup_scheduler else 'Disabled'}{reset}"),
                (f"{cyan}├─ Criterion:{reset}", f"{yellow}{criterion.__class__.__name__}{reset}"),
                (f"{cyan}├─ Optimizer:{reset}", f"{yellow}{optimizer.__class__.__name__}{reset}"),
            ]
            
            for label, value in config_items:
                print(f"{label} {value}")
            
            if run_id:
                print(f"{cyan}├─ Run ID:{reset} {yellow}{run_id}{reset}")
            if epoch_number is not None:
                print(f"{cyan}├─ Epoch:{reset} {yellow}{epoch_number}{reset}")
            
            print(f"{cyan}├─ Save results:{reset} {yellow}{'Enabled' if save_results else 'Disabled'}{reset}")
            print(f"{cyan}└─ Save metadata:{reset} {yellow}{'Enabled' if save_metadata else 'Disabled'}{reset}")
        
        # Memory pre-allocation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if verbose:
                print(f"{green}CUDA memory cleared{reset}")
        
        # Initialize metrics tracking
        if track_metrics:
            metrics_tracker = {
                'losses': [],
                'accuracies': [],
                'learning_rates': [],
                'batch_times': [],
                'gradient_norms': [] if track_gradient_norms else None,
                'detailed_timings': [] if track_detailed_timing else None
            }
            if verbose:
                print(f"{green}Metrics tracking enabled{reset}")
        
        # Update progress bar with initialization status
        if pbar:
            pbar.text = "Initializing training..."
        
        # Main training loop
        optimizer.zero_grad()
        accumulated_steps = 0
        
        for batch_idx, batch in enumerate(loader):
            batch_start_time = time.time()
            training_stats['current_batch'] = batch_idx + 1
            training_stats['completed_batches'] = batch_idx + 1
            
            retry_count = 0
            batch_successful = False
            
            while not batch_successful and retry_count <= max_retries:
                try:
                    # STAGE 1: Data Transfer
                    data_transfer_start = time.time()
                    
                    # Extract data from batch
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    else:
                        inputs = batch.to(device)
                        targets = inputs  # Autoencoder case
                    
                    data_transfer_time = time.time() - data_transfer_start
                    
                    current_batch_size = inputs.size(0)
                    num_samples += current_batch_size
                    training_stats['total_samples'] = num_samples
                    
                    # STAGE 2: Forward Pass
                    forward_start = time.time()
                    
                    autocast_context = get_autocast_context(device, scaler is not None)
                    
                    with autocast_context:
                        outputs = model(inputs)
                        last_outputs = outputs  # Store for final accuracy calculation
                        
                        # Handle classification vs regression
                        if outputs.dim() > 1 and outputs.size(-1) > 1:  # Classification
                            loss = criterion(outputs, targets)
                            _, predicted = torch.max(outputs.data, 1)
                            batch_correct = (predicted == targets).sum().item()
                            correct_predictions += batch_correct
                            training_stats['correct_predictions'] = correct_predictions
                            batch_accuracy = batch_correct / current_batch_size
                        else:  # Regression or autoencoder
                            loss = criterion(outputs, targets)
                            batch_correct = 0
                            batch_accuracy = 0.0
                        
                        # Scale loss for gradient accumulation
                        loss = loss / grad_accum_steps
                    
                    forward_time = time.time() - forward_start
                    
                    # STAGE 3: Backward Pass
                    backward_start = time.time()
                    
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    backward_time = time.time() - backward_start
                    
                    accumulated_steps += 1
                    
                    # STAGE 4: Gradient Accumulation and Optimization
                    optimization_start = time.time()
                    
                    # Check if it's time for gradient accumulation step
                    if accumulated_steps % grad_accum_steps == 0:
                        gradient_accumulations += 1
                        training_stats['gradient_accumulations'] = gradient_accumulations
                        
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        
                        # Gradient clipping with norm tracking
                        if grad_clip > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            gradient_clippings += 1
                            training_stats['gradient_clippings'] = gradient_clippings
                            
                            if track_gradient_norms:
                                gradient_norms.append(grad_norm.item())
                                training_stats['gradient_norms'].append(grad_norm.item())
                                if track_metrics:
                                    metrics_tracker['gradient_norms'].append(grad_norm.item())
                        else:
                            if track_gradient_norms:
                                grad_norm = torch.sqrt(sum(p.grad.data.norm()**2 for p in model.parameters() if p.grad is not None))
                                gradient_norms.append(grad_norm.item())
                                training_stats['gradient_norms'].append(grad_norm.item())
                                if track_metrics:
                                    metrics_tracker['gradient_norms'].append(grad_norm.item())
                        
                        # Optimizer step
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        optimizer_steps += 1
                        training_stats['optimizer_steps'] = optimizer_steps
                        
                        # Apply warmup if provided
                        if warmup_scheduler:
                            warmup_scheduler.step()
                            warmup_updates += 1
                            training_stats['warmup_updates'] = warmup_updates
                            
                            # Get learning rate safely
                            if hasattr(warmup_scheduler, 'get_last_lr'):
                                current_lr = warmup_scheduler.get_last_lr()[0]
                            else:
                                current_lr = optimizer.param_groups[0]['lr']
                            
                            learning_rates.append(current_lr)
                            training_stats['learning_rates'].append(current_lr)
                        else:
                            current_lr = optimizer.param_groups[0]['lr']
                            learning_rates.append(current_lr)
                            training_stats['learning_rates'].append(current_lr)
                        
                        optimizer.zero_grad()
                    
                    optimization_time = time.time() - optimization_start
                    
                    # Update running loss
                    batch_loss = loss.item() * grad_accum_steps
                    total_loss += batch_loss
                    num_batches += 1
                    
                    training_stats['total_loss'] = total_loss
                    running_loss = total_loss / num_batches
                    training_stats['running_loss'] = running_loss
                    
                    # Store metrics for moving averages
                    if track_metrics:
                        recent_losses.append(batch_loss)
                        if outputs.dim() > 1 and outputs.size(-1) > 1:
                            recent_accuracies.append(batch_accuracy)
                    
                    # Calculate batch time
                    batch_time = time.time() - batch_start_time
                    recent_batch_times.append(batch_time)
                    training_stats['batch_times'].append(batch_time)
                    
                    # Store detailed timings if enabled
                    if track_detailed_timing:
                        batch_timing = {
                            'data_transfer': data_transfer_time,
                            'forward': forward_time,
                            'backward': backward_time,
                            'optimization': optimization_time,
                            'total': batch_time
                        }
                        if batch_timings is not None:
                            batch_timings.append(batch_timing)
                        if track_metrics:
                            metrics_tracker['detailed_timings'].append(batch_timing)
                    
                    # Track metrics
                    if track_metrics:
                        metrics_tracker['losses'].append(batch_loss)
                        if outputs.dim() > 1 and outputs.size(-1) > 1:
                            metrics_tracker['accuracies'].append(batch_accuracy)
                        metrics_tracker['batch_times'].append(batch_time)
                        metrics_tracker['learning_rates'].append(optimizer.param_groups[0]['lr'])
                    
                    batch_successful = True
                    
                    # Memory usage tracking
                    if track_memory_usage and device.type == 'cuda':
                        training_stats['memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
                    
                except Exception as batch_error:
                    retry_count += 1
                    
                    # OOM handling
                    if isinstance(batch_error, RuntimeError) and "out of memory" in str(batch_error).lower():
                        print(f"{red}CUDA OOM at batch {batch_idx}, retry {retry_count}/{max_retries}{reset}")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        if retry_count >= max_retries:
                            if continue_on_error:
                                print(f"{yellow}Skipping batch {batch_idx} after {retry_count} OOM retries{reset}")
                                break
                            elif graceful_degradation:
                                print(f"{red}Batch {batch_idx} failed after {retry_count} OOM retries{reset}")
                                break
                            else:
                                raise
                    
                    elif retry_count >= max_retries:
                        if continue_on_error:
                            print(f"{yellow}Skipping batch {batch_idx} after {retry_count} retries: {batch_error}{reset}")
                            break
                        elif graceful_degradation:
                            print(f"{red}Batch {batch_idx} failed after {retry_count} retries: {batch_error}{reset}")
                            break
                        else:
                            raise
                    else:
                        print(f"{yellow}Batch {batch_idx} failed, retrying ({retry_count}/{max_retries}): {batch_error}{reset}")
                        # Reset gradients for retry
                        optimizer.zero_grad()
            
            # If batch failed and we should continue, skip to next batch
            if not batch_successful:
                if continue_on_error or graceful_degradation:
                    continue
                else:
                    raise RuntimeError(f"Batch {batch_idx} failed after {retry_count} retries")
            
            # Update progress bar
            if pbar:
                # Calculate moving averages
                avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else batch_loss
                
                if outputs.dim() > 1 and outputs.size(-1) > 1 and recent_accuracies:
                    avg_recent_accuracy = sum(recent_accuracies) / len(recent_accuracies)
                else:
                    avg_recent_accuracy = batch_accuracy if outputs.dim() > 1 and outputs.size(-1) > 1 else 0.0
                
                # Calculate ETA
                eta_str = ""
                if recent_batch_times:
                    avg_batch_time = sum(recent_batch_times) / len(recent_batch_times)
                    remaining_batches = len(loader) - batch_idx - 1
                    eta_seconds = remaining_batches * avg_batch_time
                    
                    # Format ETA
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.0f}s"
                    elif eta_seconds < 3600:
                        eta_str = f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Build progress bar text
                progress_parts = []
                
                # Basic progress
                progress_parts.append(f"{batch_idx + 1}/{len(loader)}")
                
                # Loss information
                progress_parts.append(f"Loss: {batch_loss:.4f}")
                progress_parts.append(f"Avg: {avg_recent_loss:.4f}")
                
                # Accuracy if classification
                if outputs.dim() > 1 and outputs.size(-1) > 1:
                    progress_parts.append(f"Acc: {batch_accuracy:.3f}")
                    progress_parts.append(f"Avg: {avg_recent_accuracy:.3f}")
                
                # Training stats
                progress_parts.append(f"LR: {current_lr:.2e}")
                
                if gradient_accumulations > 0:
                    progress_parts.append(f"GA: {gradient_accumulations}")
                
                if track_gradient_norms and gradient_norms:
                    recent_grad_norm = gradient_norms[-1] if gradient_norms else 0
                    progress_parts.append(f"GN: {recent_grad_norm:.2f}")
                
                progress_text = ", ".join(progress_parts)
                pbar.text = progress_text
                pbar()
            
            # Log progress at specified frequency
            if verbose and batch_idx % log_frequency == 0 and batch_idx > 0:
                current_accuracy = correct_predictions / num_samples if outputs.dim() > 1 and outputs.size(-1) > 1 else 0.0
                print(f"\n{cyan}{'-'*40}")
                print(f"{magenta}PROGRESS UPDATE: Batch {batch_idx}/{len(loader)}")
                print(f"{cyan}{'-'*40}{reset}")
                
                print(f"{cyan}├─ Batch Loss:{reset} {yellow}{batch_loss:.6f}{reset}")
                print(f"{cyan}├─ Running Loss:{reset} {yellow}{running_loss:.6f}{reset}")
                
                if outputs.dim() > 1 and outputs.size(-1) > 1:
                    print(f"{cyan}├─ Batch Accuracy:{reset} {yellow}{batch_accuracy:.4f}{reset}")
                    print(f"{cyan}├─ Cumulative Accuracy:{reset} {yellow}{current_accuracy:.4f}{reset}")
                
                print(f"{cyan}├─ Learning Rate:{reset} {yellow}{optimizer.param_groups[0]['lr']:.2e}{reset}")
                print(f"{cyan}├─ Gradient Accumulations:{reset} {yellow}{gradient_accumulations}{reset}")
                
                if track_gradient_norms and gradient_norms:
                    recent_grad_norm = gradient_norms[-1] if gradient_norms else 0
                    print(f"{cyan}├─ Gradient Norm:{reset} {yellow}{recent_grad_norm:.4f}{reset}")
                
                print(f"{cyan}├─ Samples Processed:{reset} {yellow}{num_samples:,}{reset}")
                print(f"{cyan}└─ Time Elapsed:{reset} {yellow}{time.time() - epoch_start_time:.1f}s{reset}")
            
            # Memory management
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        # Handle remaining gradients
        if accumulated_steps % grad_accum_steps != 0:
            if scaler is not None:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        
        # Calculate final metrics
        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else default_loss
        
        # Calculate accuracy
        if last_outputs is not None and last_outputs.dim() > 1 and last_outputs.size(-1) > 1:
            accuracy = correct_predictions / num_samples if num_samples > 0 else 0.0
        else:
            accuracy = 0.0
        
        # Prepare metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_batches': num_batches,
            'num_samples': num_samples,
            'total_epoch_time': total_epoch_time,
            'avg_batch_time': total_epoch_time / num_batches if num_batches > 0 else 0,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'device': str(device),
            'training_completed': True,
            'progress_bar_used': progress_bar and pbar is not None,
            'run_id': run_id,
            'epoch_number': epoch_number,
            'gradient_accumulation_steps': grad_accum_steps,
            'gradient_clip': grad_clip,
            'mixed_precision': scaler is not None,
            'save_results': save_results,
            'save_metadata': save_metadata,
            'timestamp': datetime.datetime.now().isoformat(),
            'epoch_start_time': start_time.isoformat(),
            
            # Training statistics
            'training_stats': {
                'gradient_accumulations': gradient_accumulations,
                'gradient_clippings': gradient_clippings,
                'optimizer_steps': optimizer_steps,
                'warmup_updates': warmup_updates,
                'running_loss': running_loss,
                'correct_predictions': correct_predictions,
                'total_samples': num_samples
            }
        }
        
        # Add tracked metrics if enabled
        if track_metrics:
            metrics['tracked_metrics'] = {
                'losses': metrics_tracker['losses'],
                'accuracies': metrics_tracker['accuracies'] if last_outputs is not None and last_outputs.dim() > 1 and last_outputs.size(-1) > 1 else [],
                'learning_rates': metrics_tracker['learning_rates'],
                'batch_times': metrics_tracker['batch_times']
            }
            
            if metrics_tracker['losses']:
                metrics['loss_stats'] = {
                    'min': min(metrics_tracker['losses']),
                    'max': max(metrics_tracker['losses']),
                    'std': np.std(metrics_tracker['losses']) if len(metrics_tracker['losses']) > 1 else 0,
                    'avg': avg_loss
                }
            
            if track_gradient_norms and gradient_norms:
                metrics['gradient_norms'] = {
                    'values': gradient_norms,
                    'avg': np.mean(gradient_norms) if gradient_norms else 0,
                    'max': max(gradient_norms) if gradient_norms else 0,
                    'min': min(gradient_norms) if gradient_norms else 0
                }
        
        # Add detailed timing analysis if enabled
        if track_detailed_timing and batch_timings:
            avg_data_transfer = np.mean([t['data_transfer'] for t in batch_timings]) if batch_timings else 0
            avg_forward = np.mean([t['forward'] for t in batch_timings]) if batch_timings else 0
            avg_backward = np.mean([t['backward'] for t in batch_timings]) if batch_timings else 0
            avg_optimization = np.mean([t['optimization'] for t in batch_timings]) if batch_timings else 0
            
            metrics['timing_analysis'] = {
                'detailed_timings': batch_timings,
                'avg_data_transfer': avg_data_transfer,
                'avg_forward': avg_forward,
                'avg_backward': avg_backward,
                'avg_optimization': avg_optimization,
                'samples_per_second': num_samples / total_epoch_time if total_epoch_time > 0 else 0,
                'batches_per_second': num_batches / total_epoch_time if total_epoch_time > 0 else 0
            }
        
        # Add memory usage if tracked
        if track_memory_usage and device.type == 'cuda':
            metrics['memory_usage'] = {
                'peak_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
            }
        
        # Save results to JSON file if requested
        if save_results:
            try:
                # Ensure RESULTS_DIR exists
                if RESULTS_DIR is None:
                    print(f"{yellow}RESULTS_DIR not configured, creating default results directory{reset}")
                    base_dir = Path(__file__).resolve().parent
                    results_dir = base_dir / "results"
                    results_dir.mkdir(parents=True, exist_ok=True)
                else:
                    results_dir = RESULTS_DIR
                
                # Generate filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_results_{timestamp}"
                if run_id:
                    filename += f"_{run_id}"
                if epoch_number is not None:
                    filename += f"_epoch{epoch_number:03d}"
                filename += ".json"
                
                # Save comprehensive results
                results_path = results_dir / filename
                
                # Convert metrics to JSON-serializable format
                def convert_for_json(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, torch.Tensor):
                        return obj.cpu().numpy().tolist() if obj.device.type != 'cpu' else obj.numpy().tolist()
                    elif isinstance(obj, datetime.datetime):
                        return obj.isoformat()
                    elif isinstance(obj, Path):
                        return str(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(item) for item in obj]
                    elif isinstance(obj, torch.device):
                        return str(obj)
                    else:
                        return obj
                
                json_metrics = convert_for_json(metrics)
                
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(json_metrics, f, indent=2, ensure_ascii=False)
                
                if verbose:
                    print(f"{green}Saved comprehensive results to: {results_path}{reset}")
                
                # Also save to metrics directory if configured
                if METRICS_DIR is not None:
                    metrics_path = METRICS_DIR / filename
                    with open(metrics_path, 'w', encoding='utf-8') as f:
                        json.dump(json_metrics, f, indent=2, ensure_ascii=False)
                    
                    if verbose:
                        print(f"{green}Also saved to metrics directory: {metrics_path}{reset}")
            
            except Exception as e:
                logger.error(f"{red}Failed to save results: {str(e)}{reset}")
        
        # Save metadata to JSON file if requested
        if save_metadata:
            try:
                # Ensure appropriate directory exists
                if INFO_DIR is None:
                    if verbose:
                        print(f"{yellow}INFO_DIR not configured, creating default info directory{reset}")
                    base_dir = Path(__file__).resolve().parent
                    info_dir = base_dir / "info"
                    info_dir.mkdir(parents=True, exist_ok=True)
                else:
                    info_dir = INFO_DIR
                
                # Generate filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_metadata_{timestamp}"
                if run_id:
                    filename += f"_{run_id}"
                if epoch_number is not None:
                    filename += f"_epoch{epoch_number:03d}"
                filename += ".json"
                
                # Prepare metadata
                metadata = {
                    'run_id': run_id,
                    'epoch_number': epoch_number,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'training_duration_seconds': total_epoch_time,
                    'model_architecture': model.__class__.__name__,
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'model_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'criterion': criterion.__class__.__name__,
                    'optimizer': optimizer.__class__.__name__,
                    'optimizer_config': {
                        'lr': optimizer.param_groups[0].get('lr', 'N/A'),
                        'betas': optimizer.param_groups[0].get('betas', 'N/A'),
                        'eps': optimizer.param_groups[0].get('eps', 'N/A'),
                        'weight_decay': optimizer.param_groups[0].get('weight_decay', 'N/A')
                    },
                    'device': str(device),
                    'dataset_info': {
                        'num_batches': num_batches,
                        'num_samples': num_samples,
                        'avg_batch_size': num_samples / num_batches if num_batches > 0 else 0
                    },
                    'training_config': {
                        'grad_clip': grad_clip,
                        'grad_accum_steps': grad_accum_steps,
                        'mixed_precision': scaler is not None,
                        'warmup_scheduler': warmup_scheduler is not None
                    },
                    'performance_summary': {
                        'final_loss': avg_loss,
                        'final_accuracy': accuracy,
                        'samples_per_second': num_samples / total_epoch_time if total_epoch_time > 0 else 0,
                        'batches_per_second': num_batches / total_epoch_time if total_epoch_time > 0 else 0
                    },
                    'system_info': {
                        'python_version': sys.version,
                        'pytorch_version': torch.__version__,
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
                        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
                    }
                }
                
                # Add memory info if available
                if track_memory_usage and device.type == 'cuda':
                    metadata['memory_info'] = {
                        'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                        'allocated_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024
                    }
                
                # Save metadata
                metadata_path = info_dir / filename
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                if verbose:
                    print(f"{green}Saved metadata to: {metadata_path}{reset}")
                
            except Exception as e:
                logger.error(f"{red}Failed to save metadata: {str(e)}{reset}")
        
        # Final comprehensive summary
        if verbose:
            print(f"\n{cyan}{'-'*40}")
            print(f"{magenta}TRAINING EPOCH SUMMARY")
            print(f"{cyan}{'-'*40}{reset}")
            
            # Context information
            if epoch_number is not None:
                print(f"{cyan}├─ Epoch:{reset} {yellow}{epoch_number}{reset}")
            if run_id:
                print(f"{cyan}├─ Run ID:{reset} {yellow}{run_id}{reset}")
            print(f"{cyan}├─ Model:{reset} {yellow}{model.__class__.__name__}{reset}")
            print(f"{cyan}├─ Device:{reset} {yellow}{str(device)}{reset}")
            print(f"{cyan}└─ Training Completed:{reset} {green}Yes{reset}")
            
            # Performance metrics
            print(f"\n{blue}Performance Metrics:{reset}")
            print(f"{cyan}├─ Average Loss:{reset} {yellow}{avg_loss:.6f}{reset}")
            
            if last_outputs is not None and last_outputs.dim() > 1 and last_outputs.size(-1) > 1:
                print(f"{cyan}├─ Accuracy:{reset} {yellow}{accuracy:.4f}{reset}")
                print(f"{cyan}├─ Correct Predictions:{reset} {yellow}{correct_predictions}/{num_samples}{reset}")
            
            print(f"{cyan}├─ Running Loss:{reset} {yellow}{running_loss:.6f}{reset}")
            print(f"{cyan}├─ Total Samples:{reset} {yellow}{num_samples:,}{reset}")
            print(f"{cyan}├─ Learning Rate:{reset} {yellow}{optimizer.param_groups[0]['lr']:.2e}{reset}")
            print(f"{cyan}└─ Gradient Clip:{reset} {yellow}{grad_clip}{reset}")
            
            # Training statistics
            print(f"\n{blue}Training Statistics:{reset}")
            print(f"{cyan}├─ Total Batches:{reset} {yellow}{num_batches}{reset}")
            print(f"{cyan}├─ Gradient Accumulations:{reset} {yellow}{gradient_accumulations}{reset}")
            print(f"{cyan}├─ Optimizer Steps:{reset} {yellow}{optimizer_steps}{reset}")
            print(f"{cyan}├─ Gradient Clippings:{reset} {yellow}{gradient_clippings}{reset}")
            print(f"{cyan}└─ Warmup Updates:{reset} {yellow}{warmup_updates}{reset}")
            
            # Timing information
            print(f"\n{blue}Timing Information:{reset}")
            print(f"{cyan}├─ Epoch Duration:{reset} {yellow}{total_epoch_time:.2f}s{reset}")
            
            if num_batches > 0:
                print(f"{cyan}├─ Average Batch Time:{reset} {yellow}{total_epoch_time/num_batches:.3f}s{reset}")
                print(f"{cyan}├─ Samples/Second:{reset} {yellow}{num_samples/total_epoch_time:.0f}{reset}")
                print(f"{cyan}└─ Batches/Second:{reset} {yellow}{num_batches/total_epoch_time:.1f}{reset}")
            else:
                print(f"{cyan}└─ No batches completed{reset}")
            
            # Gradient information if tracked
            if track_gradient_norms and gradient_norms:
                print(f"\n{blue}Gradient Analysis:{reset}")
                avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
                max_grad_norm = max(gradient_norms) if gradient_norms else 0
                min_grad_norm = min(gradient_norms) if gradient_norms else 0
                
                # Color coding for gradient norms
                def get_grad_color(value, threshold=10):
                    if value < threshold * 0.5:
                        return green
                    elif value < threshold:
                        return yellow
                    else:
                        return red
                
                print(f"{cyan}├─ Average Gradient Norm:{reset} {get_grad_color(avg_grad_norm)}{avg_grad_norm:.6f}{reset}")
                print(f"{cyan}├─ Maximum Gradient Norm:{reset} {get_grad_color(max_grad_norm)}{max_grad_norm:.6f}{reset}")
                print(f"{cyan}├─ Minimum Gradient Norm:{reset} {get_grad_color(min_grad_norm)}{min_grad_norm:.6f}{reset}")
                print(f"{cyan}└─ Gradient Samples:{reset} {yellow}{len(gradient_norms)}{reset}")
            
            # Memory usage if tracked
            if track_memory_usage and device.type == 'cuda':
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                
                print(f"\n{blue}Memory Ssage:{reset}")
                print(f"{cyan}├─ Peak Memory:{reset} {yellow}{peak_mb:.1f} MB{reset}")
                print(f"{cyan}└─ Current Memory:{reset} {yellow}{allocated_mb:.1f} MB{reset}")
            
            # Detailed timing analysis if enabled
            if track_detailed_timing and batch_timings:
                print(f"\n{blue}Batch Processing Breakdown:{reset}")
                avg_data_transfer = np.mean([t['data_transfer'] for t in batch_timings]) if batch_timings else 0
                avg_forward = np.mean([t['forward'] for t in batch_timings]) if batch_timings else 0
                avg_backward = np.mean([t['backward'] for t in batch_timings]) if batch_timings else 0
                avg_optimization = np.mean([t['optimization'] for t in batch_timings]) if batch_timings else 0
                
                print(f"{cyan}├─ Data Transfer:{reset} {yellow}{avg_data_transfer:.4f}s avg{reset}")
                print(f"{cyan}├─ Forward Pass:{reset} {yellow}{avg_forward:.4f}s avg{reset}")
                print(f"{cyan}├─ Backward Pass:{reset} {yellow}{avg_backward:.4f}s avg{reset}")
                print(f"{cyan}└─ Optimization:{reset} {yellow}{avg_optimization:.4f}s avg{reset}")
            
            # Export status
            if save_results or save_metadata:
                print(f"\n{blue}Export Status:{reset}")
                if save_results:
                    print(f"{cyan}├─ Results saved:{reset} {green}Yes{reset}")
                else:
                    print(f"{cyan}├─ Results saved:{reset} {yellow}No{reset}")
                
                if save_metadata:
                    print(f"{cyan}└─ Metadata saved:{reset} {green}Yes{reset}")
                else:
                    print(f"{cyan}└─ Metadata saved:{reset} {yellow}No{reset}")
            
            # Performance assessment
            print(f"\n{blue}Performance Assessment:{reset}")
            
            # Loss-based assessment
            if avg_loss < 0.1:
                loss_assessment = f"{green}Excellent{reset}"
            elif avg_loss < 0.5:
                loss_assessment = f"{cyan}Good{reset}"
            elif avg_loss < 1.0:
                loss_assessment = f"{yellow}Acceptable{reset}"
            else:
                loss_assessment = f"{red}Needs Improvement{reset}"
            
            print(f"{cyan}├─ Loss Performance:{reset} {loss_assessment}")
            
            # Throughput assessment
            if num_samples / total_epoch_time > 1000:
                throughput_assessment = f"{green}High{reset}"
            elif num_samples / total_epoch_time > 100:
                throughput_assessment = f"{cyan}Moderate{reset}"
            else:
                throughput_assessment = f"{yellow}Low{reset}"
            
            print(f"{cyan}├─ Throughput:{reset} {throughput_assessment} ({num_samples/total_epoch_time:.0f} samples/s)")
            
            # Stability assessment based on gradient norms
            if gradient_norms:
                grad_std = np.std(gradient_norms) if len(gradient_norms) > 1 else 0
                if grad_std < 0.1:
                    stability_assessment = f"{green}Stable{reset}"
                elif grad_std < 0.5:
                    stability_assessment = f"{cyan}Moderately Stable{reset}"
                else:
                    stability_assessment = f"{yellow}Variable{reset}"
                print(f"{cyan}├─ Gradient Stability:{reset} {stability_assessment}")
            
            print(f"{cyan}└─ Overall Status:{reset} {green}Completed Successfully{reset}")
        
        return avg_loss, metrics
        
    except Exception as e:
        # Error context
        error_context = f" (stage: {training_stats.get('stage', 'unknown')}, "
        error_context += f"batch: {training_stats.get('current_batch', 0)}/{training_stats.get('total_batches', 0)})"
        
        print(f"\n{red}{'-'*40}")
        print(f"{magenta}TRAINING FAILED")
        print(f"{red}{'-'*40}{reset}")
        
        print(f"{red}├─ Error:{reset} {yellow}{str(e)}{reset}")
        print(f"{red}├─ Error Type:{reset} {yellow}{type(e).__name__}{reset}")
        print(f"{red}├─ Stage:{reset} {yellow}{training_stats.get('stage', 'unknown')}{reset}")
        print(f"{red}├─ Batch:{reset} {yellow}{training_stats.get('current_batch', 0)}/{training_stats.get('total_batches', 0)}{reset}")
        print(f"{red}└─ Training Failed:{reset} {red}Yes{reset}")
        
        # Calculate partial results
        final_loss = total_loss / max(num_batches, 1) if num_batches > 0 else default_loss
        final_accuracy = correct_predictions / max(num_samples, 1) if num_samples > 0 else 0.0
        
        error_metrics = default_metrics.copy()
        error_metrics.update({
            'loss': final_loss,
            'accuracy': final_accuracy,
            'num_batches': num_batches,
            'num_samples': num_samples,
            'error': str(e),
            'error_type': type(e).__name__,
            'training_failed': True,
            'partial_results': True,
            'error_context': error_context,
            'training_stats': {
                'gradient_accumulations': gradient_accumulations,
                'gradient_clippings': gradient_clippings,
                'optimizer_steps': optimizer_steps,
                'warmup_updates': warmup_updates
            }
        })
        
        # Partial results summary
        if num_batches > 0:
            print(f"\n{blue}Partial Results:{reset}")
            print(f"{cyan}├─ Batches Completed:{reset} {yellow}{num_batches}{reset}")
            print(f"{cyan}├─ Samples Processed:{reset} {yellow}{num_samples:,}{reset}")
            print(f"{cyan}├─ Partial Loss:{reset} {yellow}{final_loss:.6f}{reset}")
            print(f"{cyan}└─ Partial Accuracy:{reset} {yellow}{final_accuracy:.4f}{reset}")
        
        # Save partial results if export was requested
        if save_results:
            try:
                if RESULTS_DIR is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"partial_results_error_{timestamp}"
                    if run_id:
                        filename += f"_{run_id}"
                    filename += ".json"
                    
                    results_path = RESULTS_DIR / filename
                    
                    # Convert to JSON-serializable
                    def convert_for_json(obj):
                        if isinstance(obj, (np.integer, np.floating)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, torch.Tensor):
                            return obj.cpu().numpy().tolist() if obj.device.type != 'cpu' else obj.numpy().tolist()
                        elif isinstance(obj, datetime.datetime):
                            return obj.isoformat()
                        elif isinstance(obj, Path):
                            return str(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_for_json(item) for item in obj]
                        elif isinstance(obj, torch.device):
                            return str(obj)
                        else:
                            return obj
                    
                    json_metrics = convert_for_json(error_metrics)
                    
                    with open(results_path, 'w', encoding='utf-8') as f:
                        json.dump(json_metrics, f, indent=2, ensure_ascii=False)
                    if verbose:
                        print(f"{yellow}Saved partial results to: {results_path}{reset}")
            except Exception as save_error:
                logger.error(f"{red}Failed to save partial results: {save_error}{reset}")
        
        if graceful_degradation:
            print(f"{yellow}Returning partial results due to error{reset}")
            print(f"{yellow}├─ Loss: {final_loss:.6f}{reset}")
            print(f"{yellow}└─ Accuracy: {final_accuracy:.4f}{reset}")
            return final_loss, error_metrics
        else:
            print(f"{red}Returning default values (graceful degradation disabled){reset}")
            return default_loss, error_metrics
    
    finally:
        # Cleanup progress bar
        if pbar_context:
            try:
                pbar_context.__exit__(None, None, None)
            except:
                pass
        
        # Cleanup GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    attack_threshold: float = 0.5,
    verbose: Optional[bool] = False,
    progress_bar: Optional[bool] = True,
    security_metrics: bool = True
) -> Dict[str, Any]:
    """
    Validation with security-focused metrics, threshold tuning,
    and progress tracking.
    
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
        - Validation statistics and timing information
    """
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track validation progress and statistics
    validation_stats = {
        'stage': 'Initializing',
        'total_batches': len(loader),
        'completed_batches': 0,
        'current_batch': 0,
        'total_samples': 0,
        'processed_samples': 0,
        'model_evaluated': False,
        'metrics_calculated': False,
        'security_metrics_computed': False,
        'classification_report_generated': False,
        'detailed_timings': {},
        'batch_processing_times': [],
        'memory_usage_mb': 0.0,
        'device_used': str(device),
        'class_distribution': {},
        'threshold_used': attack_threshold
    }
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

    try:
        print(f"{green}\nValidating Model with Security Metrics...{reset}")
        # Progress helper: define all stage titles
        titles = [
            "Validation Setup",
            "Processing Batches",
            "Calculating Metrics",
            "Finalizing Validation"
        ]

        # Use progress bar only if progress_bar is True
        if progress_bar:
            progress = ProgressHelper(titles)
        else:
            # Create a dummy context manager that does nothing
            class DummyBar:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def __call__(self):
                    pass
                def __setattr__(self, name, value):
                    pass
                def bar(self, *args, **kwargs):
                    return self
            progress = DummyBar()
        
        # STAGE 1: Validation Setup and Configuration
        with progress.bar("Validation Setup", total=5, unit="stages") as setup_bar:
            
            # STAGE 1.1: Model and Device Configuration
            setup_bar.text = "Configuring model and device..."
            config_start = time.time()
            
            # Set model to evaluation mode
            model.eval()
            
            # Display configuration
            if verbose:
                logger.info(f"{cyan}Validation Configuration:{reset}")
                logger.info(f"  - Device: {validation_stats['device_used']}")
                logger.info(f"  - Total batches: {validation_stats['total_batches']}")
                logger.info(f"  - Attack threshold: {attack_threshold}")
                logger.info(f"  - Security metrics: {'Enabled' if security_metrics else 'Disabled'}")
                logger.info(f"  - Class names: {'Provided' if class_names else 'Not provided'}")
                logger.info(f"  - Criterion: {criterion.__class__.__name__}")
            
            config_time = time.time() - config_start
            validation_stats['detailed_timings']['configuration'] = config_time
            setup_bar.text = f"{green}Model and device configured{reset}"
            setup_bar()
            
            # STAGE 1.2: Memory Preparation
            setup_bar.text = "Preparing memory..."
            memory_start = time.time()
            
            # Clear cache and prepare memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            memory_time = time.time() - memory_start
            validation_stats['detailed_timings']['memory_preparation'] = memory_time
            setup_bar.text = f"{green}Memory prepared{reset}"
            setup_bar()
            
            # STAGE 1.3: DataLoader Analysis
            setup_bar.text = "Analyzing DataLoader..."
            analysis_start = time.time()
            
            # Estimate total samples
            if hasattr(loader, 'dataset'):
                validation_stats['total_samples'] = len(loader.dataset)
            else:
                # Estimate based on batch size and number of batches
                validation_stats['total_samples'] = len(loader) * (loader.batch_size if hasattr(loader, 'batch_size') else 64)
            
            analysis_time = time.time() - analysis_start
            validation_stats['detailed_timings']['dataloader_analysis'] = analysis_time
            setup_bar.text = f"{green}DataLoader analyzed ({validation_stats['total_samples']} estimated samples){reset}"
            setup_bar()
            
            # STAGE 1.4: Validation Parameters
            setup_bar.text = "Setting validation parameters..."
            params_start = time.time()
            
            # Initialize tracking variables
            total_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []
            all_logits = []
            
            params_time = time.time() - params_start
            validation_stats['detailed_timings']['parameter_setup'] = params_time
            setup_bar.text = f"{green}Validation parameters set{reset}"
            setup_bar()
            
            # STAGE 1.5: Setup Completion
            setup_bar.text = "Setup completion..."
            completion_start = time.time()
            
            # Final setup steps
            validation_stats['stage'] = 'Setup Complete'
            
            completion_time = time.time() - completion_start
            validation_stats['detailed_timings']['setup_completion'] = completion_time
            setup_bar.text = f"{green}Setup completed{reset}"
            setup_bar()
        
        # STAGE 2: Batch Processing with Progress Tracking
        validation_stats['stage'] = 'Batch Processing'
        
        with progress.bar("Processing Batches", total=validation_stats['total_batches'], unit="batches") as batch_bar:
            
            batch_processing_start = time.time()
            
            with torch.no_grad():
                for batch_idx, (X_batch, y_batch) in enumerate(loader):
                    validation_stats['current_batch'] = batch_idx + 1
                    validation_stats['completed_batches'] = batch_idx + 1
                    batch_start_time = time.time()
                    
                    try:
                        # Data transfer to device
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        
                        # Model inference
                        outputs = model(X_batch)
                        probs = torch.softmax(outputs, dim=1)
                        loss = criterion(outputs, y_batch)
                        
                        # Accumulate loss
                        total_loss += loss.item()
                        
                        # Store all raw outputs
                        all_logits.extend(outputs.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                        all_labels.extend(y_batch.cpu().numpy())
                        
                        # Update sample count
                        batch_samples = y_batch.size(0)
                        validation_stats['processed_samples'] += batch_samples
                        
                        # Calculate batch processing time
                        batch_time = time.time() - batch_start_time
                        validation_stats['batch_processing_times'].append(batch_time)
                        
                        # Update progress with performance metrics
                        avg_batch_time = np.mean(validation_stats['batch_processing_times'][-10:]) if validation_stats['batch_processing_times'] else 0
                        remaining_batches = validation_stats['total_batches'] - batch_idx - 1
                        eta = remaining_batches * avg_batch_time
                        
                        # Format ETA string
                        if eta < 60:
                            eta_str = f"{eta:.0f}s"
                        elif eta < 3600:
                            eta_str = f"{eta/60:.1f}m"
                        else:
                            eta_str = f"{eta/3600:.1f}h"
                        
                        batch_bar.text = (f"Batch {batch_idx + 1}/{validation_stats['total_batches']} | Samples: {validation_stats['processed_samples']} | Batch Time: {batch_time:.3f}s | ETA: {eta_str}")
                        
                    except Exception as batch_error:
                        batch_bar.text = f"{red}Error in batch {batch_idx + 1}{reset}"
                        if verbose:
                            logger.error(f"{yellow}Warning: Batch {batch_idx + 1} processing failed: {str(batch_error)}{reset}")
                            # Continue with next batch
                    
                    batch_bar()
            
            batch_processing_time = time.time() - batch_processing_start
            validation_stats['detailed_timings']['batch_processing'] = batch_processing_time
            validation_stats['model_evaluated'] = True
            batch_bar.text = f"{green}All batches processed{reset}"
        
        # STAGE 3: Metrics Calculation and Analysis
        validation_stats['stage'] = 'Metrics Calculation'
        
        with progress.bar("Calculating Metrics", total=4, unit="steps") as metrics_bar:
            
            # STAGE 3.1: Data Conversion and Preparation
            metrics_bar.text = "Converting data to numpy arrays..."
            conversion_start = time.time()
            
            # Convert to numpy arrays
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            
            # Calculate class distribution
            unique_labels, label_counts = np.unique(all_labels, return_counts=True)
            validation_stats['class_distribution'] = dict(zip(unique_labels.tolist(), label_counts.tolist()))
            
            conversion_time = time.time() - conversion_start
            validation_stats['detailed_timings']['data_conversion'] = conversion_time
            metrics_bar.text = f"{green}Data converted ({len(all_labels)} samples){reset}"
            metrics_bar()
            
            # STAGE 3.2: Predictions and Base Metrics
            metrics_bar.text = "Calculating base metrics..."
            base_metrics_start = time.time()
            
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
                'attack_threshold': attack_threshold,
                'validation_stats': validation_stats
            }
            
            base_metrics_time = time.time() - base_metrics_start
            validation_stats['detailed_timings']['base_metrics'] = base_metrics_time
            metrics_bar.text = f"{green}Base metrics calculated{reset}"
            metrics_bar()
            
            # STAGE 3.3: Advanced Metrics
            metrics_bar.text = "Calculating advanced metrics..."
            advanced_metrics_start = time.time()
            
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
                        'false_negatives': int(fn),
                        'false_positives': int(fp),
                        'true_positives': int(tp),
                        'true_negatives': int(tn),
                        'attack_detection_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                        'false_alarm_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                    })
                    validation_stats['security_metrics_computed'] = True
            else:  # Multiclass
                metrics.update({
                    'val_auc': roc_auc_score(all_labels, all_probs, multi_class='ovr'),
                    'val_ap': average_precision_score(all_labels, all_probs)
                })
            
            advanced_metrics_time = time.time() - advanced_metrics_start
            validation_stats['detailed_timings']['advanced_metrics'] = advanced_metrics_time
            validation_stats['metrics_calculated'] = True
            metrics_bar.text = f"{green}Advanced metrics calculated{reset}"
            metrics_bar()
            
            # STAGE 3.4: Classification Report
            metrics_bar.text = "Generating classification report..."
            report_start = time.time()
            
            # Classification report
            if class_names:
                metrics['report'] = classification_report(
                    all_labels, all_preds,
                    target_names=class_names,
                    digits=4,
                    output_dict=True
                )
                validation_stats['classification_report_generated'] = True
            
            report_time = time.time() - report_start
            validation_stats['detailed_timings']['classification_report'] = report_time
            metrics_bar.text = f"{green}Classification report generated{reset}"
            metrics_bar()
        
        # STAGE 4: Final Summary and Reporting
        with progress.bar("Finalizing Validation", total=2, unit="steps") as final_bar:
            
            # STAGE 4.1: Memory Usage Calculation
            final_bar.text = "Calculating memory usage..."
            memory_calc_start = time.time()
            
            # Calculate memory usage
            if device.type == 'cuda':
                validation_stats['memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            memory_calc_time = time.time() - memory_calc_start
            validation_stats['detailed_timings']['memory_calculation'] = memory_calc_time
            final_bar.text = f"{green}Memory usage calculated{reset}"
            final_bar()
            
            # STAGE 4.2: Final Summary
            final_bar.text = "Generating validation summary..."
            summary_start = time.time()
            
            # Calculate total time from detailed timings (excluding total)
            timing_keys = [k for k in validation_stats['detailed_timings'].keys() if k != 'total']
            total_time = sum(validation_stats['detailed_timings'][k] for k in timing_keys)
            validation_stats['detailed_timings']['total'] = total_time
            
            # Display validation summary
            if verbose:
                logger.info(f"\n{blue}Validation Summary{reset}")
                logger.info(f"{cyan}Performance Metrics:{reset}")
                logger.info(f"  - Validation Loss: {metrics['val_loss']:.6f}")
                logger.info(f"  - Accuracy: {metrics['val_acc']:.4f}")
                
                if 'val_auc' in metrics:
                    logger.info(f"  - AUC: {metrics['val_auc']:.4f}")
                if 'val_ap' in metrics:
                    logger.info(f"  - Average Precision: {metrics['val_ap']:.4f}")
                
                if security_metrics and 'recall' in metrics:
                    logger.info(f"  - Recall: {metrics['recall']:.4f}")
                    logger.info(f"  - Precision: {metrics['precision']:.4f}")
                    logger.info(f"  - F2-Score: {metrics['f2_score']:.4f}")
                    logger.info(f"  - Attack Detection Rate: {metrics['attack_detection_rate']:.4f}")
                    logger.info(f"  - False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
                
                logger.info(f"\n{cyan}Validation Statistics:{reset}")
                logger.info(f"  - Total Samples: {len(all_labels):,}")
                logger.info(f"  - Total Batches: {validation_stats['total_batches']}")
                logger.info(f"  - Classes: {len(validation_stats['class_distribution'])}")
                logger.info(f"  - Class Distribution: {validation_stats['class_distribution']}")
                logger.info(f"  - Attack Threshold: {attack_threshold}")
                logger.info(f"  - Device: {validation_stats['device_used']}")
                logger.info(f"  - Total Time: {total_time:.2f}s")
                
                if validation_stats['memory_usage_mb'] > 0:
                    logger.info(f"  - Peak Memory Usage: {validation_stats['memory_usage_mb']:.1f} MB")
                
                logger.info(f"\n{cyan}Processing Details:{reset}")
                logger.info(f"  - Average Batch Time: {np.mean(validation_stats['batch_processing_times']):.3f}s")
                logger.info(f"  - Model Evaluated: {'Yes' if validation_stats['model_evaluated'] else 'No'}")
                logger.info(f"  - Metrics Calculated: {'Yes' if validation_stats['metrics_calculated'] else 'No'}")
                logger.info(f"  - Security Metrics: {'Yes' if validation_stats['security_metrics_computed'] else 'No'}")
                logger.info(f"  - Classification Report: {'Yes' if validation_stats['classification_report_generated'] else 'No'}")
                
                # Display detailed timings
                logger.info(f"\n{cyan}Detailed Timings:{reset}")
                for stage, timing in validation_stats['detailed_timings'].items():
                    if stage != 'total':
                        logger.info(f"  - {stage}: {timing:.3f}s")
                logger.info(f"  - Total time: {total_time:.3f}s")
            
            summary_time = time.time() - summary_start
            validation_stats['detailed_timings']['final_summary'] = summary_time
            final_bar.text = f"{green}Validation completed successfully{reset}"
            final_bar()
        
        return metrics

    except Exception as e:
        # Log error with context
        error_context = f" (stage: {validation_stats.get('stage', 'unknown')}, "
        error_context += f"batch: {validation_stats.get('current_batch', 0)}/{validation_stats.get('total_batches', 0)})"
        logger.error(f"{red}Validation failed{error_context}: {str(e)}{reset}")
        raise RuntimeError("Validation failed") from e

def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    verbose: Optional[bool] = False,
    progress_bar: Optional[bool] = True,
    metric: str = 'f2'
) -> Tuple[float, float]:
    """
    Find optimal decision threshold for security-focused metrics with progress tracking.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for positive class
        metric: Metric to optimize ('f2', 'recall', 'precision')
        
    Returns:
        Tuple of (optimal_threshold, best_score)
        
    Raises:
        ValueError: If inputs are invalid or metric is unknown
    """
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track threshold optimization progress and statistics
    optimization_stats = {
        'stage': 'Initializing',
        'total_thresholds': 0,
        'evaluated_thresholds': 0,
        'current_threshold': 0.0,
        'best_threshold': 0.5,
        'best_score': 0.0,
        'metric_used': metric,
        'data_statistics': {},
        'score_progression': [],
        'detailed_timings': {},
        'evaluation_results': []
    }
    
    try:
        print(f"{green}\nStarting Threshold Optimization...{reset}")
        # Progress helper: define all stage titles
        titles = [
            "Threshold Optimization",
            "Evaluating Thresholds",
            "Final Analysis"
        ]
        
        # Use progress bar only if progress_bar is True
        if progress_bar:
            progress = ProgressHelper(titles)
        else:
            # Create a dummy context manager that does nothing
            class DummyBar:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def __call__(self):
                    pass
                def __setattr__(self, name, value):
                    pass
                def bar(self, *args, **kwargs):
                    return self
            progress = DummyBar()
        
        # STAGE 1: Input Validation and Setup
        with progress.bar("Threshold Optimization", total=4, unit="stages") as bar:
            
            # STAGE 1.1: Input Validation
            bar.text = "Validating input data..."
            validation_start = time.time()
            
            # Validate inputs
            if y_true is None or y_probs is None:
                bar.text = f"{red}Input arrays cannot be None{reset}"
                raise ValueError("Input arrays cannot be None")
                
            if len(y_true) != len(y_probs):
                bar.text = f"{red}Input arrays must have same length{reset}"
                raise ValueError(f"Input arrays must have same length: y_true={len(y_true)}, y_probs={len(y_probs)}")
                
            if len(y_true) == 0:
                bar.text = f"{red}Input arrays cannot be empty{reset}"
                raise ValueError("Input arrays cannot be empty")
            
            validation_time = time.time() - validation_start
            optimization_stats['detailed_timings']['input_validation'] = validation_time
            bar.text = f"{green}Input data validated ({len(y_true)} samples){reset}"
            bar()
            
            # STAGE 1.2: Data Statistics Calculation
            bar.text = "Calculating data statistics..."
            stats_start = time.time()
            
            # Calculate data statistics
            optimization_stats['data_statistics'] = {
                'total_samples': len(y_true),
                'positive_samples': np.sum(y_true),
                'negative_samples': len(y_true) - np.sum(y_true),
                'positive_ratio': np.sum(y_true) / len(y_true),
                'probability_range': (np.min(y_probs), np.max(y_probs)),
                'probability_mean': np.mean(y_probs),
                'probability_std': np.std(y_probs)
            }
            
            stats_time = time.time() - stats_start
            optimization_stats['detailed_timings']['statistics_calculation'] = stats_time
            bar.text = f"{green}Data statistics calculated{reset}"
            bar()
            
            # STAGE 1.3: Metric Configuration
            bar.text = "Configuring optimization metric..."
            config_start = time.time()
            
            # Validate metric
            valid_metrics = ['f2', 'recall', 'precision']
            if metric not in valid_metrics:
                bar.text = f"{red}Unknown metric: {metric}{reset}"
                raise ValueError(f"Unknown metric: {metric}. Valid metrics: {valid_metrics}")
            
            # Display configuration
            if verbose:
                logger.info(f"{cyan}Threshold Optimization Configuration:{reset}")
                logger.info(f"  - Metric to optimize: {metric}")
                logger.info(f"  - Total samples: {optimization_stats['data_statistics']['total_samples']:,}")
                logger.info(f"  - Positive samples: {optimization_stats['data_statistics']['positive_samples']:,}")
                logger.info(f"  - Positive ratio: {optimization_stats['data_statistics']['positive_ratio']:.3f}")
                logger.info(f"  - Probability range: [{optimization_stats['data_statistics']['probability_range'][0]:.3f}, {optimization_stats['data_statistics']['probability_range'][1]:.3f}]")
            
            config_time = time.time() - config_start
            optimization_stats['detailed_timings']['metric_configuration'] = config_time
            bar.text = f"{green}Metric configuration completed{reset}"
            bar()
            
            # STAGE 1.4: Threshold Range Generation
            bar.text = "Generating threshold range..."
            threshold_start = time.time()
            
            # Generate threshold range
            thresholds = np.arange(0.1, 0.9, 0.05)
            optimization_stats['total_thresholds'] = len(thresholds)
            
            threshold_time = time.time() - threshold_start
            optimization_stats['detailed_timings']['threshold_generation'] = threshold_time
            bar.text = f"{green}Threshold range generated ({len(thresholds)} thresholds){reset}"
            bar()
        
        # STAGE 2: Threshold Evaluation with Progress Tracking
        optimization_stats['stage'] = 'Threshold Evaluation'
        
        with progress.bar("Evaluating Thresholds", total=len(thresholds), unit="thresholds") as threshold_bar:
            
            evaluation_start = time.time()
            best_score = 0.0
            best_threshold = 0.5
            
            for i, threshold in enumerate(thresholds):
                optimization_stats['current_threshold'] = threshold
                optimization_stats['evaluated_thresholds'] = i + 1
                
                threshold_start_time = time.time()
                
                try:
                    # Convert probabilities to binary predictions
                    y_pred = (y_probs >= threshold).astype(int)
                    
                    # Calculate metric score
                    if metric == 'f2':
                        score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                    elif metric == 'recall':
                        score = recall_score(y_true, y_pred, zero_division=0)
                    elif metric == 'precision':
                        score = precision_score(y_true, y_pred, zero_division=0)
                    
                    # Update best score and threshold
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        optimization_stats['best_threshold'] = best_threshold
                        optimization_stats['best_score'] = best_score
                    
                    # Store evaluation results
                    evaluation_result = {
                        'threshold': threshold,
                        'score': score,
                        'predictions_positive': np.sum(y_pred),
                        'predictions_negative': len(y_pred) - np.sum(y_pred),
                        'evaluation_time': time.time() - threshold_start_time
                    }
                    optimization_stats['evaluation_results'].append(evaluation_result)
                    optimization_stats['score_progression'].append(score)
                    
                    # Update progress bar with current best
                    threshold_bar.text = (f"Threshold: {threshold:.3f} | Score: {score:.4f} | Best: {best_threshold:.3f} ({best_score:.4f})")
                    
                except Exception as e:
                    threshold_bar.text = f"{red}Error at threshold {threshold:.3f}{reset}"
                    logger.error(f"{yellow}Warning: Failed to evaluate threshold {threshold:.3f}: {str(e)}{reset}")
                    # Continue with next threshold
                
                threshold_bar()
            
            evaluation_time = time.time() - evaluation_start
            optimization_stats['detailed_timings']['threshold_evaluation'] = evaluation_time
            threshold_bar.text = f"{green}All thresholds evaluated{reset}"
        
        # STAGE 3: Results Analysis and Finalization
        with progress.bar("Final Analysis", total=3, unit="steps") as analysis_bar:
            
            # STAGE 3.1: Results Analysis
            analysis_bar.text = "Analyzing optimization results..."
            analysis_start = time.time()
            
            # Calculate additional statistics
            scores = [result['score'] for result in optimization_stats['evaluation_results']]
            optimization_stats['score_statistics'] = {
                'mean_score': np.mean(scores) if scores else 0,
                'std_score': np.std(scores) if scores else 0,
                'max_score': np.max(scores) if scores else 0,
                'min_score': np.min(scores) if scores else 0,
                'score_range': (np.min(scores), np.max(scores)) if scores else (0, 0)
            }
            
            analysis_time = time.time() - analysis_start
            optimization_stats['detailed_timings']['results_analysis'] = analysis_time
            analysis_bar.text = f"{green}Results analysis completed{reset}"
            analysis_bar()
            
            # STAGE 3.2: Optimal Threshold Selection
            analysis_bar.text = "Selecting optimal threshold..."
            selection_start = time.time()
            
            # Final assignment
            optimization_stats['best_threshold'] = best_threshold
            optimization_stats['best_score'] = best_score
            
            selection_time = time.time() - selection_start
            optimization_stats['detailed_timings']['threshold_selection'] = selection_time
            analysis_bar.text = f"{green}Optimal threshold selected{reset}"
            analysis_bar()
            
            # STAGE 3.3: Final Summary
            analysis_bar.text = "Generating final summary..."
            summary_start = time.time()
            
            # Calculate total time from detailed timings (excluding total)
            timing_keys = [k for k in optimization_stats['detailed_timings'].keys() if k != 'total']
            total_time = sum(optimization_stats['detailed_timings'][k] for k in timing_keys)
            optimization_stats['detailed_timings']['total'] = total_time
            
            # Display summary
            if verbose:
                logger.info(f"\n{blue}Threshold Optimization Summary{reset}")
                logger.info(f"{cyan}Optimal Results:{reset}")
                logger.info(f"  - Best Threshold: {best_threshold:.4f}")
                logger.info(f"  - Best {metric.upper()} Score: {best_score:.4f}")
                logger.info(f"  - Metric Optimized: {metric}")
                
                logger.info(f"\n{cyan}Optimization Statistics:{reset}")
                logger.info(f"  - Thresholds Evaluated: {optimization_stats['evaluated_thresholds']}")
                logger.info(f"  - Score Range: [{optimization_stats['score_statistics']['min_score']:.4f}, {optimization_stats['score_statistics']['max_score']:.4f}]")
                logger.info(f"  - Average Score: {optimization_stats['score_statistics']['mean_score']:.4f}")
                logger.info(f"  - Score Std: {optimization_stats['score_statistics']['std_score']:.4f}")
                logger.info(f"  - Total Processing Time: {total_time:.3f}s")
                
                logger.info(f"\n{cyan}Data Characteristics:{reset}")
                logger.info(f"  - Total Samples: {optimization_stats['data_statistics']['total_samples']:,}")
                logger.info(f"  - Positive Class: {optimization_stats['data_statistics']['positive_samples']:,} ({optimization_stats['data_statistics']['positive_ratio']:.1%})")
                logger.info(f"  - Probability Mean: {optimization_stats['data_statistics']['probability_mean']:.4f}")
                logger.info(f"  - Probability Std: {optimization_stats['data_statistics']['probability_std']:.4f}")
                
                # Display threshold performance insights
                if optimization_stats['evaluation_results']:
                    logger.info(f"\n{cyan}Threshold Performance Insights:{reset}")
                    # Find thresholds within 95% of best score
                    near_optimal_thresholds = [
                        result for result in optimization_stats['evaluation_results']
                        if result['score'] >= best_score * 0.95
                    ]
                    if near_optimal_thresholds:
                        logger.info(f"  - Thresholds within 95% of best score: {len(near_optimal_thresholds)}")
                        threshold_range = [result['threshold'] for result in near_optimal_thresholds]
                        logger.info(f"  - Range: [{min(threshold_range):.3f}, {max(threshold_range):.3f}]")
                
                # Display detailed timings
                logger.info(f"\n{cyan}Detailed Timings:{reset}")
                for stage, timing in optimization_stats['detailed_timings'].items():
                    if stage != 'total':
                        logger.info(f"  - {stage}: {timing:.3f}s")
                logger.info(f"  - Total time: {total_time:.3f}s")
            
            summary_time = time.time() - summary_start
            optimization_stats['detailed_timings']['final_summary'] = summary_time
            analysis_bar.text = f"{green}Optimization completed successfully{reset}"
            analysis_bar()
        
        return best_threshold, best_score
        
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {optimization_stats.get('stage', 'unknown')}, "
        error_context += f"threshold: {optimization_stats.get('current_threshold', 0.0)})"
        
        logger.error(f"{red}Threshold optimization failed{error_context}: {str(e)}{reset}")
        raise

def train_model(
    logger: logging.Logger,
    progress_bar: Optional[bool] = True,
    verbose: Optional[bool] = False,
    use_mock: bool = False,
    
    # Export Parameters
    save_results: Optional[bool] = True,
    save_metadata: Optional[bool] = True,
    save_training_history: Optional[bool] = True,
    
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete training pipeline with error handling, monitoring, and export.
    
    Args:
        logger: Configured logger instance
        progress_bar: Whether to show progress bars
        use_mock: Whether to use synthetic data (default: False)
        save_results: Save training results to JSON file
        save_metadata: Save metadata to JSON file
        save_training_history: Save training history to JSON file
        config: Optional configuration dictionary to override defaults
        
    Returns:
        Dictionary containing training results and metrics
        
    Raises:
        DataPreparationError: If data loading/preprocessing fails
        ModelConfigurationError: If model setup fails
        TrainingExecutionError: If training process fails
        ModelSavingError: If model artifacts cannot be saved
    """
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Set export flags
    save_results = save_results if save_results is not None else False
    save_metadata = save_metadata if save_metadata is not None else False
    save_training_history = save_training_history if save_training_history is not None else False
    
    # Track training progress and statistics
    training_stats = {
        'stage': 'Initializing',
        'current_epoch': 0,
        'total_epochs': 0,
        'current_batch': 0,
        'total_batches': 0,
        'best_epoch': -1,
        'best_val_f2': 0.0,
        'data_statistics': {},
        'model_statistics': {},
        'training_history': [],
        'validation_history': [],
        'detailed_timings': {},
        'export': {
            'save_results': save_results,
            'save_metadata': save_metadata,
            'save_training_history': save_training_history,
            'results_saved': False,
            'metadata_saved': False,
            'training_history_saved': False
        }
    }
    
    # Helper function to safely extract float values
    def safe_extract_float(value, default=float('nan')):
        """Safely extract a float value from PyTorch tensors, numpy arrays, or other types."""
        if value is None:
            return default
        
        # Handle PyTorch tensors
        if hasattr(value, 'item'):
            try:
                return float(value.item())
            except (ValueError, TypeError, RuntimeError):
                return default
        
        # Handle numpy arrays/scalars
        if isinstance(value, np.ndarray):
            try:
                # If it's a single-element array, extract the scalar
                if value.size == 1:
                    return float(value.item())
                # If it's a multi-element array, take the mean or first element
                elif value.size > 1:
                    # Check if it's a numeric array
                    if np.issubdtype(value.dtype, np.number):
                        # Use mean for numeric arrays
                        return float(np.mean(value))
                    else:
                        # For non-numeric, try first element
                        return float(value.flat[0])
                else:
                    return default
            except (ValueError, TypeError, IndexError):
                return default
        
        # Handle regular iterables (lists, tuples)
        if hasattr(value, '__iter__') and not isinstance(value, str):
            try:
                # Convert to list to handle various iterable types
                value_list = list(value)
                if len(value_list) > 0:
                    # Try to extract first element
                    first_elem = value_list[0]
                    # Recursively process in case it's nested
                    return safe_extract_float(first_elem, default)
                return default
            except (ValueError, TypeError, IndexError):
                return default
        
        # Handle dictionaries (extract 'value' key or first numeric value)
        if isinstance(value, dict):
            # Try common keys
            for key in ['value', 'loss', 'accuracy', 'acc', 'score']:
                if key in value:
                    return safe_extract_float(value[key], default)
            # Try to find any numeric value
            for v in value.values():
                try:
                    result = safe_extract_float(v, None)
                    if result is not None:
                        return result
                except (ValueError, TypeError):
                    continue
            return default
        
        # Handle simple numeric types
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Helper function to safely extract metric values
    def get_safe_metric(metrics_dict, key, default=0.0):
        """Safely extract metric value, handling various types."""
        value = metrics_dict.get(key, default)
        return safe_extract_float(value, default)
    
    try:
        # STAGE 1: Initialization and Setup
        training_stats['stage'] = 'Initialization'
        initialization_start = time.time()
        
        # Parameter extraction
        epochs = config.get('epochs', DEFAULT_EPOCHS) if config else DEFAULT_EPOCHS
        training_stats['total_epochs'] = epochs
        
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
            },
            'export_configuration': {
                'save_results': save_results,
                'save_metadata': save_metadata,
                'save_training_history': save_training_history
            }
        }
        
        initialization_time = time.time() - initialization_start
        training_stats['detailed_timings']['initialization'] = initialization_time
        training_stats['stage'] = 'Data Preparation'
        
        # Setup logging
        writer = SummaryWriter(log_dir=run_tb_dir, filename_suffix=f"_{run_id}")
        
        # STAGE 2: Data Preparation
        data_prep_start = time.time()
        
        try:
            if use_mock:
                df, artifacts = create_synthetic_data(logger)
                training_meta['data_source'] = 'synthetic'
            else:
                if not check_preprocessing_outputs(logger):
                    logger.warning("Preprocessing outputs not found")
                    if not run_preprocessing(logger):
                        raise DataPreparationError("Preprocessing failed", original_exception=None)
                
                df, artifacts = load_and_validate_data()
                training_meta['data_source'] = 'real'
                training_meta['original_samples'] = len(df)
            
            # Handle class imbalance
            df = handle_class_imbalance(df, artifacts, apply_smote=True)
            training_meta['final_samples'] = len(df)
            
            # Calculate data statistics for export
            training_stats['data_statistics'] = {
                'original_samples': training_meta.get('original_samples', len(df)),
                'final_samples': len(df),
                'positive_samples': int(df['Label'].sum()),
                'negative_samples': int(len(df) - df['Label'].sum()),
                'positive_ratio': float(df['Label'].mean()),
                'features_count': len(df.columns) - 1,  # Excluding label column
                'class_distribution': {
                    'normal': int((df['Label'] == 0).sum()),
                    'attack': int((df['Label'] == 1).sum())
                }
            }
            
            # Visualize data
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
            
            training_stats['data_statistics'].update({
                'input_size': input_size,
                'num_classes': num_classes,
                'train_batches': len(train_loader),
                'val_batches': len(val_loader),
                'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE) if config else DEFAULT_BATCH_SIZE
            })
        
        except Exception as e:
            error_msg = f"Data preparation failed at stage: {training_meta.get('data_source', 'unknown')}"
            logger.error(f"{error_msg}: {str(e)}")
            raise DataPreparationError(error_msg, original_exception=e) from e
        
        data_prep_time = time.time() - data_prep_start
        training_stats['detailed_timings']['data_preparation'] = data_prep_time
        training_stats['stage'] = 'Model Configuration'
        
        # STAGE 3: Model Configuration
        model_config_start = time.time()
        
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
            
            # Calculate model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            training_stats['model_statistics'] = {
                'model_type': model_type,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'non_trainable_parameters': total_params - trainable_params,
                'model_size_mb': (total_params * 4) / (1024 * 1024),  # Assuming float32 (4 bytes)
                'layers_count': len(list(model.modules())),
                'device': str(device)
            }
            
            # Class weighting for security applications
            class_counts = torch.tensor(df['Label'].value_counts().sort_index().values, dtype=torch.float32)
            class_weights = (1. / class_counts) * (class_counts.sum() / num_classes)
            class_weights = class_weights / class_weights.sum()
            
            training_stats['model_statistics']['class_weights'] = class_weights.cpu().numpy().tolist()
            training_meta['class_weights'] = class_weights.cpu().numpy().tolist()
            
            # Security-aware loss function
            false_negative_cost = config.get('fn_cost', 2.0) if config else 2.0
            criterion = SecurityAwareLoss(
                class_weights=class_weights.to(device),
                false_negative_cost=false_negative_cost
            )
            
            training_stats['model_statistics']['loss_configuration'] = {
                'loss_function': 'SecurityAwareLoss',
                'false_negative_cost': false_negative_cost,
                'class_weights_applied': True
            }
            
            # Optimizer with improved settings
            learning_rate = config.get('learning_rate', LEARNING_RATE) if config else LEARNING_RATE
            weight_decay = config.get('weight_decay', WEIGHT_DECAY) if config else WEIGHT_DECAY
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
            training_stats['model_statistics']['optimizer_configuration'] = {
                'optimizer': 'AdamW',
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'eps': 1e-8,
                'betas': (0.9, 0.999)
            }
            
            # Learning rate warmup scheduler
            warmup_epochs = config.get('warmup_epochs', 5) if config else 5
            base_lr = learning_rate
            
            training_stats['model_statistics']['scheduler_configuration'] = {
                'warmup_epochs': warmup_epochs,
                'base_learning_rate': base_lr,
                'scheduler_type': 'ReduceLROnPlateau'
            }
            
            warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, base_lr)
            
            # Main scheduler
            try:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    patience=config.get('lr_patience', 3) if config else 3,
                    factor=0.5,
                    verbose=True
                )
            except TypeError:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    patience=config.get('lr_patience', 3) if config else 3,
                    factor=0.5
                )
            
            training_stats['model_statistics']['scheduler_configuration'].update({
                'mode': 'max',
                'patience': config.get('lr_patience', 3) if config else 3,
                'factor': 0.5
            })
            
            # Disable mixed precision for CPU or if not available
            use_amp = torch.cuda.is_available() and config.get('mixed_precision', False) if config else False
            scaler = torch.amp.GradScaler(enabled=use_amp)
            
            training_stats['model_statistics']['training_configuration'] = {
                'mixed_precision': use_amp,
                'gradient_clip': config.get('gradient_clip', 1.0) if config else 1.0,
                'grad_accum_steps': config.get('grad_accum_steps', 1) if config else 1,
                'early_stopping_patience': config.get('early_stopping', EARLY_STOPPING_PATIENCE) if config else EARLY_STOPPING_PATIENCE
            }
        
        except Exception as e:
            error_msg = f"Model configuration failed for type: {model_type if 'model_type' in locals() else 'unknown'}"
            logger.error(f"{error_msg}: {str(e)}")
            raise ModelConfigurationError(error_msg, original_exception=e) from e
        
        model_config_time = time.time() - model_config_start
        training_stats['detailed_timings']['model_configuration'] = model_config_time
        training_stats['stage'] = 'Training'
        
        # STAGE 4: Training Loop
        print(f"\n{cyan}{'-'*40}")
        print(f"{magenta}TRAINING PIPELINE OVERVIEW")
        print(f"{cyan}{'-'*40}{reset}")
        
        # Section 1: Run Information
        print(f"{blue}Run Information:{reset}")
        print(f"{cyan}├─ Run ID:{reset} {yellow}{run_id}{reset}")
        print(f"{cyan}├─ Start Time:{reset} {yellow}{training_meta['start_time']}{reset}")
        print(f"{cyan}├─ Export Configuration:{reset}")
        print(f"{cyan}│  ├─ Save Results:{reset} {yellow}{'Enabled' if save_results else 'Disabled'}{reset}")
        print(f"{cyan}│  ├─ Save Metadata:{reset} {yellow}{'Enabled' if save_metadata else 'Disabled'}{reset}")
        print(f"{cyan}│  └─ Save Training History:{reset} {yellow}{'Enabled' if save_training_history else 'Disabled'}{reset}")
        print(f"{cyan}└─ Directories:{reset}")
        print(f"{cyan}   ├─ Logs:{reset} {yellow}{run_log_dir}{reset}")
        print(f"{cyan}   ├─ Figures:{reset} {yellow}{run_figure_dir}{reset}")
        print(f"{cyan}   ├─ Checkpoints:{reset} {yellow}{run_checkpoint_dir}{reset}")
        print(f"{cyan}   ├─ TensorBoard:{reset} {yellow}{run_tb_dir}{reset}")
        print(f"{cyan}   └─ Artifacts:{reset} {yellow}{run_artifact_dir}{reset}")
        
        # Section 2: Data Statistics
        print(f"\n{blue}Data Statistics:{reset}")
        print(f"{cyan}├─ Data Source:{reset} {yellow}{training_meta['data_source']}{reset}")
        print(f"{cyan}├─ Total Samples:{reset} {yellow}{training_stats['data_statistics']['final_samples']:,}{reset}")
        print(f"{cyan}├─ Class Distribution:{reset}")
        print(f"{cyan}│  ├─ Normal Samples:{reset} {yellow}{training_stats['data_statistics']['class_distribution']['normal']:,}{reset}")
        print(f"{cyan}│  └─ Attack Samples:{reset} {yellow}{training_stats['data_statistics']['class_distribution']['attack']:,}{reset}")
        print(f"{cyan}├─ Attack Ratio:{reset} {yellow}{training_stats['data_statistics']['positive_ratio']:.1%}{reset}")
        print(f"{cyan}├─ Input Features:{reset} {yellow}{training_stats['data_statistics']['input_size']}{reset}")
        print(f"{cyan}├─ Number of Classes:{reset} {yellow}{training_stats['data_statistics']['num_classes']}{reset}")
        print(f"{cyan}├─ Training Batches:{reset} {yellow}{training_stats['data_statistics']['train_batches']:,}{reset}")
        print(f"{cyan}└─ Validation Batches:{reset} {yellow}{training_stats['data_statistics']['val_batches']:,}{reset}")
        
        # Section 3: Model Configuration
        print(f"\n{blue}Model Configuration{reset}")
        print(f"{cyan}├─ Model Type:{reset} {yellow}{training_stats['model_statistics']['model_type']}{reset}")
        print(f"{cyan}├─ Model Parameters:{reset}")
        print(f"{cyan}│  ├─ Total:{reset} {yellow}{training_stats['model_statistics']['total_parameters']:,}{reset}")
        print(f"{cyan}│  ├─ Trainable:{reset} {yellow}{training_stats['model_statistics']['trainable_parameters']:,}{reset}")
        print(f"{cyan}│  └─ Model Size:{reset} {yellow}{training_stats['model_statistics']['model_size_mb']:.2f} MB{reset}")
        print(f"{cyan}├─ Class Weights:{reset} {yellow}{training_stats['model_statistics']['class_weights']}{reset}")
        print(f"{cyan}├─ Loss Function:{reset} {yellow}SecurityAwareLoss{reset}")
        print(f"{cyan}├─ False Negative Cost:{reset} {yellow}{training_stats['model_statistics']['loss_configuration']['false_negative_cost']}{reset}")
        print(f"{cyan}├─ Optimizer:{reset} {yellow}AdamW{reset}")
        print(f"{cyan}│  ├─ Learning Rate:{reset} {yellow}{learning_rate:.2e}{reset}")
        print(f"{cyan}│  └─ Weight Decay:{reset} {yellow}{weight_decay:.2e}{reset}")
        print(f"{cyan}├─ Learning Rate Warmup:{reset} {yellow}{warmup_epochs} epochs{reset}")
        print(f"{cyan}├─ Scheduler:{reset} {yellow}ReduceLROnPlateau{reset}")
        print(f"{cyan}│  ├─ Mode:{reset} {yellow}max{reset}")
        print(f"{cyan}│  ├─ Patience:{reset} {yellow}{config.get('lr_patience', 3) if config else 3}{reset}")
        print(f"{cyan}│  └─ Factor:{reset} {yellow}0.5{reset}")
        print(f"{cyan}├─ Mixed Precision:{reset} {yellow}{'Enabled (AMP)' if use_amp else 'Disabled'}{reset}")
        print(f"{cyan}├─ Training Device:{reset} {yellow}{training_stats['model_statistics']['device']}{reset}")
        print(f"{cyan}└─ Early Stopping Patience:{reset} {yellow}{early_stop_patience if 'early_stop_patience' in locals() else config.get('early_stopping', EARLY_STOPPING_PATIENCE) if config else EARLY_STOPPING_PATIENCE}{reset}")
        
        # Section 4: Training Configuration
        print(f"\n{blue}Training Configuration{reset}")
        print(f"{cyan}├─ Total Epochs:{reset} {yellow}{epochs}{reset}")
        print(f"{cyan}├─ Batch Size:{reset} {yellow}{training_stats['data_statistics']['batch_size']}{reset}")
        print(f"{cyan}├─ Warmup Epochs:{reset} {yellow}{warmup_epochs}{reset}")
        print(f"{cyan}├─ Gradient Clip:{reset} {yellow}{config.get('gradient_clip', 1.0) if config else 1.0}{reset}")
        print(f"{cyan}├─ Gradient Accumulation Steps:{reset} {yellow}{config.get('grad_accum_steps', 1) if config else 1}{reset}")
        print(f"{cyan}└─ Progress Bar:{reset} {yellow}{'Enabled' if progress_bar else 'Disabled'}{reset}")
        
        # Section 5: Environment Information
        print(f"\n{blue}Environment Information{reset}")
        print(f"{cyan}├─ Python Version:{reset} {yellow}{training_meta['environment']['python_version']}{reset}")
        print(f"{cyan}├─ PyTorch Version:{reset} {yellow}{training_meta['environment']['pytorch_version']}{reset}")
        print(f"{cyan}└─ Training Device:{reset} {yellow}{training_meta['environment']['device']}{reset}")
        print(f"{cyan}{'-'*40}{reset}")
        
        logger.info(Fore.YELLOW + Style.BRIGHT + "\nStarting Training with Security Enhancements")
        
        # Training loop - Initialize best_metrics
        best_metrics = {
            'epoch': -1,
            'val_loss': float('inf'),
            'val_acc': 0.0,
            'val_auc': 0.0,
            'val_recall': 0.0,
            'val_f2': 0.0,
            'train_loss': float('inf'),
            'train_acc': 0.0,
            'train_loss_history': [],
            'val_loss_history': [],
            'train_acc_history': [],
            'val_acc_history': [],
            'learning_rate_history': [],
            'optimal_threshold': 0.5,
            'epoch_timings': []
        }
        
        early_stop_patience = config.get('early_stopping', EARLY_STOPPING_PATIENCE) if config else EARLY_STOPPING_PATIENCE
        patience_counter = 0
        
        training_start = time.time()
        
        try:
            for epoch in range(epochs):
                training_stats['current_epoch'] = epoch + 1
                training_stats['current_batch'] = 0
                
                epoch_start = time.time()
                
                # Determine if in warmup phase
                in_warmup_phase = epoch < warmup_epochs
                
                # Initialize lr_scale for all cases
                lr_scale = 1.0  # Default for non-warmup
                
                # Learning rate warmup implementation
                if in_warmup_phase:
                    lr_scale = (epoch + 1) / warmup_epochs
                    current_lr = lr_scale * base_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                # Train epoch - NOTE: train_epoch returns (loss, metrics_dict)
                train_result = train_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    grad_clip=config.get('gradient_clip', 1.0) if config else 1.0,
                    grad_accum_steps=config.get('grad_accum_steps', 1) if config else 1,
                    scaler=scaler,
                    warmup_scheduler=warmup_scheduler if in_warmup_phase else None,
                    progress_bar_desc=f"Training Epoch {epoch+1}/{epochs}" if progress_bar else None
                )
                
                # Extract train_loss and train_acc from result
                if isinstance(train_result, tuple) and len(train_result) == 2:
                    train_loss, train_metrics = train_result
                    # Extract accuracy from metrics dictionary
                    train_acc = train_metrics.get('accuracy', 0.0) if isinstance(train_metrics, dict) else 0.0
                else:
                    # Fallback if train_epoch returns different structure
                    train_loss = train_result[0] if isinstance(train_result, (list, tuple)) and len(train_result) > 0 else float('inf')
                    train_acc = 0.0
                
                # Check for training stability
                if not np.isfinite(train_loss):
                    error_msg = f"Training became unstable at epoch {epoch}: loss = {train_loss}"
                    logger.error(error_msg)
                    raise TrainingExecutionError(error_msg, epoch=epoch+1)
                
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
                
                # Track history - Use safe extraction
                train_loss_val = safe_extract_float(train_loss, float('inf'))
                train_acc_val = safe_extract_float(train_acc, 0.0)
                val_loss_val = safe_extract_float(val_metrics.get('val_loss', float('inf')), float('inf'))
                val_acc_val = safe_extract_float(val_metrics.get('val_acc', 0.0), 0.0)
                
                best_metrics['train_loss_history'].append(train_loss_val)
                best_metrics['val_loss_history'].append(val_loss_val)
                best_metrics['train_acc_history'].append(train_acc_val)
                best_metrics['val_acc_history'].append(val_acc_val)
                best_metrics['learning_rate_history'].append(current_lr)
                
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
                        labels_np,
                        attack_probs,
                        metric='f2',
                        verbose=False
                    )
                    
                    val_metrics['optimal_threshold'] = optimal_threshold
                    val_metrics['val_f2'] = f2_score
                
                # Learning rate adjustment after warmup
                if not in_warmup_phase:
                    # Use F2 score for security-focused applications
                    metric_for_scheduler = safe_extract_float(val_metrics.get('val_f2', val_metrics.get('val_acc', 0.0)), 0.0)
                    scheduler.step(metric_for_scheduler)
                    
                    # Log LR changes
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != current_lr:
                        print(f"{cyan}Learning rate reduced: {yellow}Epoch {epoch+1}, Current learning rate{current_lr:.2e} → {green}New learning rate {new_lr:.2e}{reset}")
                
                # Update best metrics (prioritize security metrics)
                is_best = False
                val_f2_val = safe_extract_float(val_metrics.get('val_f2', 0.0), 0.0)
                
                if in_warmup_phase:
                    # During warmup, use loss
                    if val_loss_val < best_metrics['val_loss']:
                        is_best = True
                else:
                    # After warmup, prioritize F2 score for security
                    if val_f2_val > best_metrics.get('val_f2', 0.0):
                        is_best = True
                
                # Record epoch timing
                epoch_time = time.time() - epoch_start
                best_metrics['epoch_timings'].append(epoch_time)
                
                # Create epoch record for training history
                epoch_record = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss_val,
                    'train_accuracy': train_acc_val,
                    'val_loss': val_loss_val,
                    'val_accuracy': val_acc_val,
                    'val_f2': val_f2_val,
                    'val_auc': safe_extract_float(val_metrics.get('val_auc', 0.0), 0.0),
                    'val_recall': safe_extract_float(val_metrics.get('recall', 0.0), 0.0),
                    'learning_rate': current_lr,
                    'optimal_threshold': safe_extract_float(val_metrics.get('optimal_threshold', 0.5), 0.5),
                    'epoch_time_seconds': epoch_time,
                    'in_warmup_phase': in_warmup_phase,
                    'is_best_epoch': is_best
                }
                
                training_stats['training_history'].append(epoch_record)
                
                if is_best:
                    best_metrics.update({
                        'epoch': epoch,
                        'val_loss': val_loss_val,
                        'val_acc': val_acc_val,
                        'val_auc': safe_extract_float(val_metrics.get('val_auc', 0.0), 0.0),
                        'val_recall': safe_extract_float(val_metrics.get('recall', 0.0), 0.0),
                        'val_f2': val_f2_val,
                        'train_loss': train_loss_val,
                        'train_acc': train_acc_val,
                        'learning_rate': current_lr,
                        'optimal_threshold': safe_extract_float(val_metrics.get('optimal_threshold', 0.5), 0.5),
                        'preds': val_metrics.get('preds'),
                        'labels': val_metrics.get('labels'),
                        'probs': val_metrics.get('probs')
                    })
                    
                    training_stats['best_epoch'] = epoch + 1
                    training_stats['best_val_f2'] = val_f2_val
                    
                    patience_counter = 0
                    
                    # Save best model
                    try:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            metrics=best_metrics,
                            filename=run_checkpoint_dir / f"best_model_{run_id}.pth",
                            config=training_meta,
                            verbose=verbose
                        )
                        print(f"{green}Epoch {epoch+1}:{reset} {yellow}New best model saved (F2: {val_f2_val:.4f}){reset}")
                    except Exception as save_error:
                        logger.warning(f"Failed to save checkpoint at epoch {epoch+1}: {str(save_error)}")
                else:
                    patience_counter += 1
                
                # Log epoch progress
                if verbose:
                    epoch_summary = (
                        f"{cyan}├─ Epoch {epoch+1}/{epochs}:{reset} "
                        f"Train Loss: {yellow}{train_loss_val:.4f}{reset}, "
                        f"Train Acc: {yellow}{train_acc_val:.4f}{reset}, "
                        f"Val Loss: {yellow}{val_loss_val:.4f}{reset}, "
                        f"Val Acc: {yellow}{val_acc_val:.4f}{reset}, "
                        f"Val F2: {yellow}{val_f2_val:.4f}{reset}, "
                        f"LR: {yellow}{current_lr:.2e}{reset}"
                    )
                    print(epoch_summary)
                
                # Early stopping
                if patience_counter >= early_stop_patience:
                    print(f"{cyan}\nEarly Stopping Triggered at: Epoch {yellow}{epoch+1}{cyan} after Patience {yellow}{early_stop_patience}{cyan} without improvement{reset}")
                    break
        
        except TrainingExecutionError:
            # Re-raise training execution errors
            raise
        except Exception as e:
            error_msg = f"Training execution failed at epoch {epoch if 'epoch' in locals() else 'unknown'}"
            logger.error(f"{error_msg}: {str(e)}")
            raise TrainingExecutionError(
                error_msg,
                epoch=epoch if 'epoch' in locals() else None,
                original_exception=e
            ) from e
        
        training_time = time.time() - training_start
        training_stats['detailed_timings']['training'] = training_time
        training_stats['stage'] = 'Finalization'
        
        # STAGE 5: Final Evaluation and Reporting
        finalization_start = time.time()
        
        final_epoch = epoch+1 if 'epoch' in locals() else 0
        
        # Use safe_extract_float for all metric extractions
        training_meta.update({
            'end_time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_time': training_time,
            'final_epoch': final_epoch,
            'best_epoch': best_metrics['epoch'],
            'best_val_loss': safe_extract_float(best_metrics.get('val_loss', float('inf')), float('inf')),
            'best_val_acc': safe_extract_float(best_metrics.get('val_acc', 0.0), 0.0),
            'best_val_auc': safe_extract_float(best_metrics.get('val_auc', 0.0), 0.0),
            'best_val_recall': safe_extract_float(best_metrics.get('val_recall', 0.0), 0.0),
            'best_val_f2': safe_extract_float(best_metrics.get('val_f2', 0.0), 0.0),
            'best_learning_rate': safe_extract_float(best_metrics.get('learning_rate', 0.0), 0.0),
            'optimal_attack_threshold': safe_extract_float(best_metrics.get('optimal_threshold', 0.5), 0.5),
            'early_stop': patience_counter >= early_stop_patience,
            'total_train_loss_history_length': len(best_metrics['train_loss_history']),
            'total_val_loss_history_length': len(best_metrics['val_loss_history'])
        })
        
        # Update training stats with final metrics
        training_stats.update({
            'final_epoch': final_epoch,
            'best_metrics': {
                'epoch': best_metrics['epoch'] + 1,
                'val_loss': training_meta['best_val_loss'],
                'val_accuracy': training_meta['best_val_acc'],
                'val_auc': training_meta['best_val_auc'],
                'val_recall': training_meta['best_val_recall'],
                'val_f2': training_meta['best_val_f2'],
                'train_loss': safe_extract_float(best_metrics.get('train_loss', float('inf')), float('inf')),
                'train_accuracy': safe_extract_float(best_metrics.get('train_acc', 0.0), 0.0),
                'optimal_threshold': training_meta['optimal_attack_threshold']
            },
            'early_stopping': {
                'triggered': training_meta['early_stop'],
                'patience': early_stop_patience,
                'patience_counter': patience_counter
            }
        })
        
        # Load best model for final evaluation
        try:
            checkpoint_result = load_checkpoint(
                filename=run_checkpoint_dir / f"best_model_{run_id}.pth",
                model=model,
                device=device
            )
            
            # Unpack the result correctly - load_checkpoint returns (data_dict, error_message)
            checkpoint_data, error_message = checkpoint_result
            
            if checkpoint_data is None or error_message is not None:
                logger.warning(f"Could not load best model state: {error_message}")
            else:
                # Extract components from the checkpoint data dictionary
                model_state = checkpoint_data.get('model_state_dict')
                optim_state = checkpoint_data.get('optimizer_state_dict')
                scheduler_state = checkpoint_data.get('scheduler_state_dict')
                loaded_metrics = checkpoint_data.get('metrics', {})
                meta = checkpoint_data.get('config', {})
                
                # Load model state
                if model_state is not None:
                    model.load_state_dict(model_state)
                    print(f"{green}Best Model state loaded{reset}")
                else:
                    logger.error("Best Model state not found")
                
                # Load optimizer state if available
                if optim_state is not None and optimizer is not None:
                    try:
                        optimizer.load_state_dict(optim_state)
                        print(f"{green}Best Model Optimizer state loaded{reset}")
                    except Exception as opt_error:
                        logger.warning(f"Failed to load optimizer state: {str(opt_error)}")
                
                # Load scheduler state if available
                if scheduler_state is not None and scheduler is not None:
                    try:
                        scheduler.load_state_dict(scheduler_state)
                        print(f"{green}Best Model Scheduler state loaded{reset}")
                    except Exception as sched_error:
                        logger.warning(f"Failed to load scheduler state: {str(sched_error)}")
                
                print(Fore.GREEN + Style.BRIGHT + "Successfully loaded best model checkpoint" + Style.RESET_ALL)
        
        except Exception as e:
            logger.error(f"Failed to load best model: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
        
        # Generate reports
        if 'preds' in best_metrics and 'labels' in best_metrics:
            # Generate and log classification report
            try:
                # Convert to numpy arrays if needed
                labels_to_use = best_metrics['labels']
                preds_to_use = best_metrics['preds']
                
                if hasattr(labels_to_use, 'cpu'):
                    labels_to_use = labels_to_use.cpu().numpy()
                if hasattr(preds_to_use, 'cpu'):
                    preds_to_use = preds_to_use.cpu().numpy()
                
                report = classification_report(
                    labels_to_use,
                    preds_to_use,
                    target_names=['Normal', 'Attack'],
                    digits=4,
                    output_dict=True  # Return as dictionary for export
                )
                
                # Convert report to string for logging
                report_str = classification_report(
                    labels_to_use,
                    preds_to_use,
                    target_names=['Normal', 'Attack'],
                    digits=4
                )
                
                print(f"{green}\nClassification Report:{reset}")
                print(f"{yellow}{report_str}{reset}")
                
                # Store classification report for export
                training_stats['classification_report'] = report
                
            except Exception as e:
                logger.warning(f"Failed to generate classification report: {str(e)}")
            
            # Generate confusion matrix plot
            try:
                # Convert to numpy arrays if needed
                labels_to_use = best_metrics['labels']
                preds_to_use = best_metrics['preds']
                
                if hasattr(labels_to_use, 'cpu'):
                    labels_to_use = labels_to_use.cpu().numpy()
                if hasattr(preds_to_use, 'cpu'):
                    preds_to_use = preds_to_use.cpu().numpy()
                
                cm = confusion_matrix(labels_to_use, preds_to_use)
                
                # Calculate confusion matrix metrics for export
                tn, fp, fn, tp = cm.ravel()
                confusion_matrix_metrics = {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp),
                    'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                }
                
                training_stats['confusion_matrix_metrics'] = confusion_matrix_metrics
                
                # Create confusion matrix visualization
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                cm_path = run_figure_dir / f"confusion_matrix_{run_id}.png"
                plt.savefig(cm_path, bbox_inches='tight')
                plt.close()
                training_meta['confusion_matrix'] = str(cm_path)
                
                print(f"{green}Confusion Matrix Saved to: {reset}{cyan}{cm_path}{reset}")
            
            except Exception as plot_error:
                logger.warning(f"Failed to generate confusion matrix plot: {str(plot_error)}")
        
        # Save final artifacts
        try:
            # Create safe best metrics for artifact saving
            safe_best_metrics = {}
            for key, value in best_metrics.items():
                safe_best_metrics[key] = safe_extract_float(value, value)  # Return original if not convertible
            
            artifacts_saved = save_training_artifacts(
                model=model,
                metrics=safe_best_metrics,
                config=training_meta,
                class_names=['Normal', 'Attack'],
                feature_names=artifacts.get('feature_names')
            )
            
            if not artifacts_saved:
                raise ModelSavingError("Failed to save some training artifacts")
            else:
                print(f"{green}Training artifacts saved successfully{reset}")
        
        except Exception as e:
            error_msg = "Failed to save training artifacts"
            logger.error(f"{error_msg}: {str(e)}")
            raise ModelSavingError(error_msg, original_exception=e) from e
        
        # STAGE 6: Export Results and Metadata
        if save_results or save_metadata or save_training_history:
            
            export_start = time.time()
            
            # Save training results to JSON file
            if save_results:
                try:
                    # Ensure RESULTS_DIR exists
                    if RESULTS_DIR is None:
                        base_dir = Path(__file__).resolve().parent
                        results_dir = base_dir / "results"
                        results_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        results_dir = RESULTS_DIR
                    
                    # Generate filename
                    results_filename = f"training_results_{run_id}.json"
                    results_path = results_dir / results_filename
                    
                    # Prepare results export
                    export_results = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'run_id': run_id,
                        'training_summary': {
                            'total_epochs': epochs,
                            'final_epoch': final_epoch,
                            'best_epoch': training_stats['best_epoch'],
                            'training_time_seconds': training_time,
                            'early_stopping_triggered': training_meta['early_stop'],
                            'early_stopping_patience': early_stop_patience
                        },
                        'performance_metrics': {
                            'best_validation_f2': training_meta['best_val_f2'],
                            'best_validation_accuracy': training_meta['best_val_acc'],
                            'best_validation_auc': training_meta['best_val_auc'],
                            'best_validation_recall': training_meta['best_val_recall'],
                            'best_validation_loss': training_meta['best_val_loss'],
                            'optimal_attack_threshold': training_meta['optimal_attack_threshold'],
                            'best_train_loss': safe_extract_float(best_metrics.get('train_loss', float('inf')), float('inf')),
                            'best_train_accuracy': safe_extract_float(best_metrics.get('train_acc', 0.0), 0.0)
                        },
                        'model_statistics': training_stats['model_statistics'],
                        'data_statistics': training_stats['data_statistics'],
                        'training_configuration': {
                            'model_type': training_stats['model_statistics']['model_type'],
                            'learning_rate': training_stats['model_statistics']['optimizer_configuration']['learning_rate'],
                            'batch_size': training_stats['data_statistics']['batch_size'],
                            'weight_decay': training_stats['model_statistics']['optimizer_configuration']['weight_decay'],
                            'warmup_epochs': training_stats['model_statistics']['scheduler_configuration']['warmup_epochs'],
                            'false_negative_cost': training_stats['model_statistics']['loss_configuration']['false_negative_cost']
                        },
                        'classification_report': training_stats.get('classification_report', {}),
                        'confusion_matrix_metrics': training_stats.get('confusion_matrix_metrics', {}),
                        'environment_info': training_meta['environment'],
                        'export_configuration': {
                            'save_results': save_results,
                            'save_metadata': save_metadata,
                            'save_training_history': save_training_history,
                            'verbose': verbose,
                            'progress_bar': progress_bar,
                            'use_mock_data': use_mock
                        }
                    }
                    
                    with open(results_path, 'w', encoding='utf-8') as f:
                        json.dump(export_results, f, indent=2, ensure_ascii=False)
                    
                    training_stats['export']['results_saved'] = True
                    
                    # Also save to metrics directory if configured
                    if METRICS_DIR is not None:
                        metrics_path = METRICS_DIR / results_filename
                        with open(metrics_path, 'w', encoding='utf-8') as f:
                            json.dump(export_results, f, indent=2, ensure_ascii=False)
                
                except Exception as e:
                    logger.error(f"Failed to save training results: {str(e)}")
            
            # Save metadata to JSON file
            if save_metadata:
                try:
                    # Ensure INFO_DIR exists
                    if INFO_DIR is None:
                        base_dir = Path(__file__).resolve().parent
                        info_dir = base_dir / "info"
                        info_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        info_dir = INFO_DIR
                    
                    # Generate filename
                    metadata_filename = f"training_metadata_{run_id}.json"
                    metadata_path = info_dir / metadata_filename
                    
                    # Prepare metadata export
                    metadata = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'run_id': run_id,
                        'training_summary': {
                            'start_time': training_meta['start_time'],
                            'end_time': training_meta['end_time'],
                            'total_training_time_minutes': training_time / 60,
                            'total_epochs': epochs,
                            'best_epoch': training_stats['best_epoch'],
                            'early_stopping_triggered': training_meta['early_stop']
                        },
                        'performance_summary': {
                            'best_validation_f2': training_meta['best_val_f2'],
                            'best_validation_accuracy': training_meta['best_val_acc'],
                            'optimal_attack_threshold': training_meta['optimal_attack_threshold'],
                            'improvement_over_default': float((training_meta['best_val_f2'] - 0.5) * 100) if training_meta['best_val_f2'] > 0 else 0.0
                        },
                        'data_summary': {
                            'data_source': training_meta['data_source'],
                            'total_samples': training_stats['data_statistics']['final_samples'],
                            'positive_samples': training_stats['data_statistics']['positive_samples'],
                            'positive_ratio': training_stats['data_statistics']['positive_ratio'],
                            'input_size': training_stats['data_statistics']['input_size'],
                            'num_classes': training_stats['data_statistics']['num_classes']
                        },
                        'model_summary': {
                            'model_type': training_stats['model_statistics']['model_type'],
                            'total_parameters': training_stats['model_statistics']['total_parameters'],
                            'trainable_parameters': training_stats['model_statistics']['trainable_parameters'],
                            'model_size_mb': training_stats['model_statistics']['model_size_mb']
                        },
                        'training_configuration': training_meta['config'],
                        'export_configuration': {
                            'save_results': save_results,
                            'save_metadata': save_metadata,
                            'save_training_history': save_training_history,
                            'verbose': verbose,
                            'progress_bar': progress_bar
                        },
                        'system_info': {
                            'python_version': platform.python_version(),
                            'pytorch_version': torch.__version__,
                            'numpy_version': np.__version__,
                            'device': str(device),
                            'timestamp': datetime.datetime.now().isoformat()
                        },
                        'artifacts_location': {
                            'checkpoint': str(run_checkpoint_dir / f"best_model_{run_id}.pth"),
                            'tensorboard_logs': str(run_tb_dir),
                            'figures_directory': str(run_figure_dir),
                            'artifacts_directory': str(run_artifact_dir)
                        }
                    }
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    training_stats['export']['metadata_saved'] = True
                    
                except Exception as e:
                    logger.error(f"Failed to save training metadata: {str(e)}")
            
            # Save training history to JSON file
            if save_training_history:
                try:
                    # Ensure RESULTS_DIR exists
                    if RESULTS_DIR is None:
                        base_dir = Path(__file__).resolve().parent
                        results_dir = base_dir / "results"
                        results_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        results_dir = RESULTS_DIR
                    
                    # Generate filename
                    history_filename = f"training_history_{run_id}.json"
                    history_path = results_dir / history_filename
                    
                    # Prepare training history export
                    training_history_export = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'run_id': run_id,
                        'training_history': training_stats['training_history'],
                        'loss_history': {
                            'train_loss': best_metrics['train_loss_history'],
                            'val_loss': best_metrics['val_loss_history'],
                            'train_accuracy': best_metrics['train_acc_history'],
                            'val_accuracy': best_metrics['val_acc_history']
                        },
                        'learning_rate_history': best_metrics['learning_rate_history'],
                        'epoch_timings': best_metrics['epoch_timings'],
                        'training_statistics': {
                            'average_epoch_time': np.mean(best_metrics['epoch_timings']) if best_metrics['epoch_timings'] else 0,
                            'total_training_time': training_time,
                            'epochs_completed': final_epoch,
                            'early_stopping_epoch': final_epoch if training_meta['early_stop'] else None
                        }
                    }
                    
                    with open(history_path, 'w', encoding='utf-8') as f:
                        json.dump(training_history_export, f, indent=2, ensure_ascii=False)
                    
                    training_stats['export']['training_history_saved'] = True
                    
                except Exception as e:
                    logger.error(f"Failed to save training history: {str(e)}")
            
            export_time = time.time() - export_start
            training_stats['detailed_timings']['export'] = export_time
        
        # Generate final summary
        # Calculate total time from detailed timings
        timing_keys = [k for k in training_stats['detailed_timings'].keys() if k != 'total']
        total_time = sum(training_stats['detailed_timings'][k] for k in timing_keys)
        training_stats['detailed_timings']['total'] = total_time
        
        # Display summary
        print(f"\n{cyan}{'-'*40}")
        print(f"{magenta}TRAINING COMPLETED SUCCESSFULLY")
        print(f"{cyan}{'-'*40}{reset}")
        
        # Training Results
        print(f"{blue}Training Results:{reset}")
        print(f"{cyan}├─ Run ID:{reset} {yellow}{run_id}{reset}")
        print(f"{cyan}├─ Total Time:{reset} {yellow}{total_time/60:.1f} minutes{reset}")
        print(f"{cyan}├─ Epochs Completed:{reset} {yellow}{final_epoch}/{epochs}{reset}")
        print(f"{cyan}├─ Best Epoch:{reset} {yellow}{training_stats['best_epoch']}{reset}")
        print(f"{cyan}└─ Early Stopping:{reset} {yellow}{'Triggered' if training_meta['early_stop'] else 'Not triggered'}{reset}")
        
        # Performance Metrics
        print(f"\n{blue}Performance Metrics:{reset}")
        print(f"{cyan}├─ Best Validation F2-Score:{reset} {yellow}{training_meta['best_val_f2']:.4f}{reset}")
        print(f"{cyan}├─ Best Validation Accuracy:{reset} {yellow}{training_meta['best_val_acc']:.4f}{reset}")
        print(f"{cyan}├─ Best Validation AUC:{reset} {yellow}{training_meta['best_val_auc']:.4f}{reset}")
        print(f"{cyan}├─ Best Validation Recall:{reset} {yellow}{training_meta['best_val_recall']:.4f}{reset}")
        print(f"{cyan}├─ Optimal Attack Threshold:{reset} {yellow}{training_meta['optimal_attack_threshold']:.4f}{reset}")
        print(f"{cyan}└─ Best Validation Loss:{reset} {yellow}{training_meta['best_val_loss']:.6f}{reset}")
        
        # Data Statistics
        print(f"\n{blue}Data Statistics:{reset}")
        print(f"{cyan}├─ Data Source:{reset} {yellow}{training_meta['data_source']}{reset}")
        print(f"{cyan}├─ Total Samples:{reset} {yellow}{training_stats['data_statistics']['final_samples']:,}{reset}")
        print(f"{cyan}├─ Attack Samples:{reset} {yellow}{training_stats['data_statistics']['positive_samples']:,}{reset}")
        print(f"{cyan}├─ Attack Ratio:{reset} {yellow}{training_stats['data_statistics']['positive_ratio']:.1%}{reset}")
        print(f"{cyan}├─ Input Features:{reset} {yellow}{training_stats['data_statistics']['input_size']}{reset}")
        print(f"{cyan}└─ Number of Classes:{reset} {yellow}{training_stats['data_statistics']['num_classes']}{reset}")
        
        # Model Statistics
        print(f"\n{blue}Model Statistics:{reset}")
        print(f"{cyan}├─ Model Type:{reset} {yellow}{training_stats['model_statistics']['model_type']}{reset}")
        print(f"{cyan}├─ Total Parameters:{reset} {yellow}{training_stats['model_statistics']['total_parameters']:,}{reset}")
        print(f"{cyan}├─ Trainable Parameters:{reset} {yellow}{training_stats['model_statistics']['trainable_parameters']:,}{reset}")
        print(f"{cyan}├─ Model Size:{reset} {yellow}{training_stats['model_statistics']['model_size_mb']:.2f} MB{reset}")
        print(f"{cyan}├─ Training Device:{reset} {yellow}{training_stats['model_statistics']['device']}{reset}")
        print(f"{cyan}└─ False Negative Cost:{reset} {yellow}{training_stats['model_statistics']['loss_configuration']['false_negative_cost']}{reset}")
        
        # Performance Assessment
        print(f"\n{blue}Performance Assessment:{reset}")
        
        # F2-score based assessment
        if training_meta['best_val_f2'] > 0.9:
            f2_assessment = f"{green}Excellent{reset}"
        elif training_meta['best_val_f2'] > 0.8:
            f2_assessment = f"{cyan}Good{reset}"
        elif training_meta['best_val_f2'] > 0.7:
            f2_assessment = f"{yellow}Acceptable{reset}"
        else:
            f2_assessment = f"{red}Needs Improvement{reset}"
        
        print(f"{cyan}├─ F2-Score Performance:{reset} {f2_assessment}")
        
        # Accuracy assessment
        if training_meta['best_val_acc'] > 0.95:
            acc_assessment = f"{green}Excellent{reset}"
        elif training_meta['best_val_acc'] > 0.9:
            acc_assessment = f"{cyan}Good{reset}"
        elif training_meta['best_val_acc'] > 0.85:
            acc_assessment = f"{yellow}Acceptable{reset}"
        else:
            acc_assessment = f"{red}Needs Improvement{reset}"
        
        print(f"{cyan}├─ Accuracy Performance:{reset} {acc_assessment}")
        
        # Loss stability assessment
        if len(best_metrics['val_loss_history']) > 1:
            loss_std = np.std(best_metrics['val_loss_history'][-10:]) if len(best_metrics['val_loss_history']) >= 10 else np.std(best_metrics['val_loss_history'])
            if loss_std < 0.01:
                stability_assessment = f"{green}Very Stable{reset}"
            elif loss_std < 0.05:
                stability_assessment = f"{cyan}Stable{reset}"
            elif loss_std < 0.1:
                stability_assessment = f"{yellow}Moderately Stable{reset}"
            else:
                stability_assessment = f"{red}Unstable{reset}"
            
            print(f"{cyan}├─ Loss Stability:{reset} {stability_assessment} (std: {loss_std:.4f})")
        else:
            print(f"{cyan}├─ Loss Stability:{reset} {yellow}Insufficient data{reset}")
        
        print(f"{cyan}└─ Overall Training Status:{reset} {green}Successful{reset}")
        
        # Export Status
        if save_results or save_metadata or save_training_history:
            print(f"\n{blue}Export Status:{reset}")
            
            if save_results:
                export_status = f"{green}Yes{reset}" if training_stats['export']['results_saved'] else f"{red}Failed{reset}"
                print(f"{cyan}├─ Results saved:{reset} {export_status}")
            
            if save_metadata:
                export_status = f"{green}Yes{reset}" if training_stats['export']['metadata_saved'] else f"{red}Failed{reset}"
                print(f"{cyan}├─ Metadata saved:{reset} {export_status}")
            
            if save_training_history:
                export_status = f"{green}Yes{reset}" if training_stats['export']['training_history_saved'] else f"{red}Failed{reset}"
                print(f"{cyan}└─ Training history saved:{reset} {export_status}")
        
        # Artifacts Location
        print(f"\n{blue}Artifacts Location:{reset}")
        print(f"{cyan}├─ Best Model Checkpoint:{reset} {yellow}{run_checkpoint_dir / f'best_model_{run_id}.pth'}{reset}")
        print(f"{cyan}├─ TensorBoard Logs:{reset} {yellow}{run_tb_dir}{reset}")
        print(f"{cyan}├─ Training Figures:{reset} {yellow}{run_figure_dir}{reset}")
        print(f"{cyan}└─ Training Artifacts:{reset} {yellow}{run_artifact_dir}{reset}")
        
        finalization_time = time.time() - finalization_start
        training_stats['detailed_timings']['finalization'] = finalization_time
        
        if verbose:
            print(f"{cyan}└─ Final Summary:{reset} {green}Generated{reset}")
        
        writer.close()
        
        # Prepare final return dictionary
        return {
            'completed': True,
            'run_id': run_id,
            'best_metrics': best_metrics,
            'meta': training_meta,
            'stats': training_stats,
            'artifacts_dir': str(run_artifact_dir),
            'checkpoint_path': str(run_checkpoint_dir / f"best_model_{run_id}.pth")
        }
        
    except DataPreparationError as e:
        logger.error(f"Data preparation failed: {str(e)}")
        if e.original_exception:
            logger.debug(f"Original exception: {traceback.format_exception(type(e.original_exception), e.original_exception, e.original_exception.__traceback__)}")
        
        # Create error export if requested
        if save_results:
            try:
                if RESULTS_DIR is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"training_error_{timestamp}.json"
                    results_path = RESULTS_DIR / filename
                    
                    error_export = {
                        'error_info': {
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'stage': training_stats.get('stage', 'unknown'),
                            'phase': 'data_preparation',
                            'timestamp': datetime.datetime.now().isoformat(),
                        },
                        'partial_results': {
                            'training_stats': training_stats,
                            'data_statistics': training_stats.get('data_statistics', {}),
                        },
                        'training_config': {
                            'use_mock': use_mock,
                            'save_results': save_results,
                            'save_metadata': save_metadata,
                            'save_training_history': save_training_history,
                        }
                    }
                    
                    with open(results_path, 'w', encoding='utf-8') as f:
                        json.dump(error_export, f, indent=2, ensure_ascii=False)
                    
                    print(f"{yellow}Saved error information to: {results_path}{reset}")
            except Exception as save_error:
                logger.error(f"{red}Failed to save error information: {save_error}{reset}")
        
        return {'completed': False, 'error': str(e), 'phase': 'data_preparation'}
        
    except ModelConfigurationError as e:
        logger.error(f"Model configuration failed: {str(e)}")
        if e.original_exception:
            logger.debug(f"Original exception: {traceback.format_exception(type(e.original_exception), e.original_exception, e.original_exception.__traceback__)}")
        
        return {'completed': False, 'error': str(e), 'phase': 'model_configuration'}
        
    except TrainingExecutionError as e:
        logger.error(f"Training execution failed: {str(e)}")
        if e.epoch is not None:
            logger.error(f"Failed at epoch: {e.epoch}")
        if e.original_exception:
            logger.debug(f"Original exception: {traceback.format_exception(type(e.original_exception), e.original_exception, e.original_exception.__traceback__)}")
        
        # Create error export with partial training results if available
        if save_results and 'training_history' in training_stats:
            try:
                if RESULTS_DIR is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"training_partial_results_{timestamp}.json"
                    results_path = RESULTS_DIR / filename
                    
                    partial_export = {
                        'error_info': {
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'stage': training_stats.get('stage', 'unknown'),
                            'failed_epoch': e.epoch,
                            'timestamp': datetime.datetime.now().isoformat(),
                        },
                        'partial_training_results': {
                            'training_stats': training_stats,
                            'data_statistics': training_stats.get('data_statistics', {}),
                            'model_statistics': training_stats.get('model_statistics', {}),
                            'training_history': training_stats.get('training_history', []),
                        },
                        'training_config': training_meta.get('config', {})
                    }
                    
                    with open(results_path, 'w', encoding='utf-8') as f:
                        json.dump(partial_export, f, indent=2, ensure_ascii=False)
                    
                    print(f"{yellow}Saved partial training results to: {results_path}{reset}")
            except Exception as save_error:
                logger.error(f"{red}Failed to save partial training results: {save_error}{reset}")
        
        return {'completed': False, 'error': str(e), 'phase': 'training_execution', 'failed_epoch': e.epoch}
        
    except ModelSavingError as e:
        logger.error(f"Model saving failed: {str(e)}")
        if e.original_exception:
            logger.debug(f"Original exception: {traceback.format_exception(type(e.original_exception), e.original_exception, e.original_exception.__traceback__)}")
        
        return {'completed': False, 'error': str(e), 'phase': 'model_saving'}
        
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        logger.debug(f"Error traceback: {traceback.format_exc()}")
        
        return {'completed': False, 'error': str(e), 'phase': 'unknown'}

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, Any],
    filename: Path,
    config: Dict[str, Any],
    verbose: Optional[bool] = False,
    progress_bar: Optional[bool] = True,
    safe_mode: bool = True
) -> bool:
    """
    Save training checkpoint with verification, SMOTE metrics, and progress tracking.
    
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
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Ensure verbose has a default value
    if verbose is None:
        verbose = False
    
    # Track checkpoint saving progress and statistics
    checkpoint_stats = {
        'stage': 'Initializing',
        'checkpoint_size_bytes': 0,
        'model_parameters': 0,
        'optimizer_states': 0,
        'metrics_count': 0,
        'config_entries': 0,
        'environment_info': 0,
        'checksum_calculated': False,
        'file_verified': False,
        'safe_mode_used': safe_mode,
        'detailed_timings': {},
        'checkpoint_components': {},
        'memory_usage_mb': 0.0
    }
    
    try:
        print(f"\n{green}Starting Checkpoint Saving...{reset}")
        # Progress helper: define all stage titles
        titles = [
            "Preparing Checkpoint",
            "Saving Checkpoint",
            "Finalizing Checkpoint"
        ]
        
        # Use progress bar only if progress_bar is True
        if progress_bar:
            progress = ProgressHelper(titles)
        else:
            # Create a dummy context manager that does nothing
            class DummyBar:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def __call__(self):
                    pass
                def __setattr__(self, name, value):
                    pass
                def bar(self, *args, **kwargs):
                    return self
            progress = DummyBar()
        
        # STAGE 1: Checkpoint Data Preparation
        with progress.bar("Preparing Checkpoint", total=6, unit="stages") as bar:
            
            # STAGE 1.1: Model State Collection
            bar.text = "Collecting model state..."
            model_start = time.time()
            
            # Count model parameters
            model_parameters = sum(p.numel() for p in model.parameters())
            checkpoint_stats['model_parameters'] = model_parameters
            
            # Get model state dict
            model_state_dict = model.state_dict()
            checkpoint_stats['checkpoint_components']['model'] = {
                'parameters': model_parameters,
                'state_dict_keys': len(model_state_dict),
                'model_class': model.__class__.__name__
            }
            
            model_time = time.time() - model_start
            checkpoint_stats['detailed_timings']['model_state_collection'] = model_time
            bar.text = f"{green}Model state collected ({model_parameters:,} parameters){reset}"
            bar()
            
            # STAGE 1.2: Optimizer State Collection
            bar.text = "Collecting optimizer state..."
            optimizer_start = time.time()
            
            # Get optimizer state dict
            optimizer_state_dict = optimizer.state_dict()
            checkpoint_stats['optimizer_states'] = len(optimizer_state_dict)
            checkpoint_stats['checkpoint_components']['optimizer'] = {
                'state_dict_keys': len(optimizer_state_dict),
                'optimizer_class': optimizer.__class__.__name__,
                'learning_rate': optimizer.param_groups[0]['lr'] if optimizer.param_groups else 'unknown'
            }
            
            optimizer_time = time.time() - optimizer_start
            checkpoint_stats['detailed_timings']['optimizer_state_collection'] = optimizer_time
            bar.text = f"{green}Optimizer state collected{reset}"
            bar()
            
            # STAGE 1.3: Metrics and Configuration Processing
            bar.text = "Processing metrics and configuration..."
            metrics_start = time.time()
            
            # Process metrics (convert numpy arrays to lists)
            processed_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, np.ndarray):
                    processed_metrics[k] = v.tolist()
                else:
                    processed_metrics[k] = v
            
            checkpoint_stats['metrics_count'] = len(processed_metrics)
            checkpoint_stats['config_entries'] = len(config)
            
            checkpoint_stats['checkpoint_components']['metrics'] = {
                'total_metrics': len(processed_metrics),
                'metric_names': list(processed_metrics.keys()),
                'has_smote_metrics': any('smote' in k.lower() or 'oversample' in k.lower() for k in processed_metrics.keys())
            }
            
            metrics_time = time.time() - metrics_start
            checkpoint_stats['detailed_timings']['metrics_processing'] = metrics_time
            bar.text = f"{green}Metrics processed ({len(processed_metrics)} metrics){reset}"
            bar()
            
            # STAGE 1.4: Environment Information Collection
            bar.text = "Collecting environment information..."
            env_start = time.time()
            
            # Collect environment information
            environment_info = {
                'numpy_version': np.__version__,
                'pytorch_version': torch.__version__,
                'python_version': platform.python_version(),
                'device': str(next(model.parameters()).device),
                'timestamp': datetime.datetime.now().isoformat(),
                'hostname': platform.node(),
                'platform': platform.platform()
            }
            checkpoint_stats['environment_info'] = len(environment_info)
            
            env_time = time.time() - env_start
            checkpoint_stats['detailed_timings']['environment_collection'] = env_time
            bar.text = f"{green}Environment information collected{reset}"
            bar()
            
            # STAGE 1.5: SMOTE Metrics Preparation
            bar.text = "Preparing SMOTE metrics..."
            smote_start = time.time()
            
            # Prepare SMOTE metrics
            smote_metrics = {
                'oversampler': config.get('oversampler', 'SMOTE'),
                'k_neighbors': config.get('k_neighbors', 3),
                'feature_correlation_diff': metrics.get('feature_correlation_diff'),
                'neighbor_ratio': metrics.get('avg_neighbor_distance'),
                'minority_class_size': metrics.get('minority_class_size'),
                'synthetic_samples_generated': metrics.get('synthetic_samples_generated'),
                'imbalance_ratio': metrics.get('imbalance_ratio'),
                'class_distribution': metrics.get('class_distribution', {})
            }
            
            smote_time = time.time() - smote_start
            checkpoint_stats['detailed_timings']['smote_metrics_preparation'] = smote_time
            bar.text = f"{green}SMOTE metrics prepared{reset}"
            bar()
            
            # STAGE 1.6: Checkpoint Assembly
            bar.text = "Assembling checkpoint data..."
            assembly_start = time.time()
            
            # Assemble complete checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'metrics': processed_metrics,
                'config': config,
                'environment': environment_info,
                'smote_metrics': smote_metrics,
                'checkpoint_metadata': {
                    'creation_time': datetime.datetime.now().isoformat(),
                    'total_parameters': model_parameters,
                    'safe_mode': safe_mode,
                    'pytorch_version': torch.__version__
                }
            }
            
            # Add scheduler if available
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                checkpoint_stats['checkpoint_components']['scheduler'] = {
                    'scheduler_class': scheduler.__class__.__name__,
                    'has_state_dict': True
                }
            
            assembly_time = time.time() - assembly_start
            checkpoint_stats['detailed_timings']['checkpoint_assembly'] = assembly_time
            bar.text = f"{green}Checkpoint data assembled{reset}"
            bar()
        
        # STAGE 2: File Operations and Serialization
        with progress.bar("Saving Checkpoint", total=4, unit="steps") as save_bar:
            
            # STAGE 2.1: Directory Preparation
            save_bar.text = "Preparing output directory..."
            dir_start = time.time()
            
            # Create full path
            full_path = Path(filename).absolute()
            
            # Create parent directory if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            dir_time = time.time() - dir_start
            checkpoint_stats['detailed_timings']['directory_preparation'] = dir_time
            save_bar.text = f"{green}Output directory prepared{reset}"
            save_bar()
            
            # STAGE 2.2: Checkpoint Serialization
            save_bar.text = "Serializing checkpoint data..."
            serialization_start = time.time()
            
            # Serialize checkpoint
            if safe_mode:
                torch.save(
                    checkpoint,
                    full_path,
                    _use_new_zipfile_serialization=True,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL
                )
                save_bar.text = f"{green}Checkpoint serialized (safe mode){reset}"
            else:
                torch.save(checkpoint, full_path)
                save_bar.text = f"{green}Checkpoint serialized (standard mode){reset}"
            
            serialization_time = time.time() - serialization_start
            checkpoint_stats['detailed_timings']['checkpoint_serialization'] = serialization_time
            save_bar()
            
            # STAGE 2.3: Checksum Calculation
            save_bar.text = "Calculating file checksum..."
            checksum_start = time.time()
            
            # Calculate and verify checksum
            file_bytes = full_path.read_bytes()
            checksum = hashlib.md5(file_bytes).hexdigest()
            
            # Save checksum to separate file
            with open(f"{full_path}.md5", 'w') as f:
                f.write(checksum)
            
            checkpoint_stats['checkpoint_size_bytes'] = len(file_bytes)
            checkpoint_stats['checksum_calculated'] = True
            
            checksum_time = time.time() - checksum_start
            checkpoint_stats['detailed_timings']['checksum_calculation'] = checksum_time
            save_bar.text = f"{green}Checksum calculated and saved{reset}"
            save_bar()
            
            # STAGE 2.4: File Verification
            save_bar.text = "Verifying saved checkpoint..."
            verification_start = time.time()
            
            # Verify file was saved correctly
            if not full_path.exists():
                save_bar.text = f"{red}Checkpoint file not found{reset}"
                raise FileNotFoundError(f"Checkpoint file was not created: {full_path}")
            
            # Verify checksum file
            checksum_path = Path(f"{full_path}.md5")
            if not checksum_path.exists():
                save_bar.text = f"{red}Checksum file not found{reset}"
                raise FileNotFoundError(f"Checksum file was not created: {checksum_path}")
            
            # Verify checksum matches
            with open(checksum_path, 'r') as f:
                saved_checksum = f.read().strip()
            
            if saved_checksum != checksum:
                save_bar.text = f"{red}Checksum verification failed{reset}"
                raise ValueError(f"Checksum mismatch: calculated {checksum}, saved {saved_checksum}")
            
            checkpoint_stats['file_verified'] = True
            
            verification_time = time.time() - verification_start
            checkpoint_stats['detailed_timings']['file_verification'] = verification_time
            save_bar.text = f"{green}Checkpoint verified successfully{reset}"
            save_bar()
        
        # STAGE 3: Final Summary and Reporting
        with progress.bar("Finalizing Checkpoint", total=2, unit="steps") as final_bar:
            
            # STAGE 3.1: Memory Usage Calculation
            final_bar.text = "Calculating memory usage..."
            memory_start = time.time()
            
            # Estimate memory usage
            if hasattr(torch.cuda, 'max_memory_allocated'):
                checkpoint_stats['memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            memory_time = time.time() - memory_start
            checkpoint_stats['detailed_timings']['memory_calculation'] = memory_time
            final_bar.text = f"{green}Memory usage calculated{reset}"
            final_bar()
            
            # STAGE 3.2: Final Summary
            final_bar.text = "Generating checkpoint summary..."
            summary_start = time.time()
            
            # Calculate total time from detailed timings (excluding total)
            timing_keys = [k for k in checkpoint_stats['detailed_timings'].keys() if k != 'total']
            total_time = sum(checkpoint_stats['detailed_timings'][k] for k in timing_keys)
            checkpoint_stats['detailed_timings']['total'] = total_time
            
            # Display summary
            if verbose:
                logger.info(f"\n{blue}Checkpoint Saving Summary{reset}")
                logger.info(f"{cyan}Checkpoint Information:{reset}")
                logger.info(f"  - File: {full_path}")
                logger.info(f"  - Size: {checkpoint_stats['checkpoint_size_bytes'] / 1024 / 1024:.2f} MB")
                logger.info(f"  - Epoch: {epoch}")
                logger.info(f"  - Checksum: {checksum}")
                logger.info(f"  - Safe mode: {'Enabled' if safe_mode else 'Disabled'}")
                
                logger.info(f"\n{cyan}Model and Training Details:{reset}")
                logger.info(f"  - Model parameters: {checkpoint_stats['model_parameters']:,}")
                logger.info(f"  - Optimizer: {checkpoint_stats['checkpoint_components']['optimizer']['optimizer_class']}")
                logger.info(f"  - Learning rate: {checkpoint_stats['checkpoint_components']['optimizer']['learning_rate']}")
                logger.info(f"  - Metrics saved: {checkpoint_stats['metrics_count']}")
                logger.info(f"  - Config entries: {checkpoint_stats['config_entries']}")
                
                logger.info(f"\n{cyan}SMOTE Metrics:{reset}")
                logger.info(f"  - Oversampler: {smote_metrics['oversampler']}")
                logger.info(f"  - K-neighbors: {smote_metrics['k_neighbors']}")
                if smote_metrics['synthetic_samples_generated']:
                    logger.info(f"  - Synthetic samples: {smote_metrics['synthetic_samples_generated']}")
                
                logger.info(f"\n{cyan}Processing Details:{reset}")
                logger.info(f"  - Total time: {total_time:.3f}s")
                logger.info(f"  - File verified: {'Yes' if checkpoint_stats['file_verified'] else 'No'}")
                logger.info(f"  - Checksum validated: {'Yes' if checkpoint_stats['checksum_calculated'] else 'No'}")
                if checkpoint_stats['memory_usage_mb'] > 0:
                    logger.info(f"  - Peak memory usage: {checkpoint_stats['memory_usage_mb']:.1f} MB")
                
                # Display detailed timings
                logger.info(f"\n{cyan}Detailed Timings:{reset}")
                for stage, timing in checkpoint_stats['detailed_timings'].items():
                    if stage != 'total':
                        logger.info(f"  - {stage}: {timing:.3f}s")
                logger.info(f"  - Total time: {total_time:.3f}s")
            
            summary_time = time.time() - summary_start
            checkpoint_stats['detailed_timings']['final_summary'] = summary_time
            final_bar.text = f"{green}Checkpoint saved successfully{reset}"
            final_bar()

            logger.info(f"{green}Checkpoint saved successfully to {magenta}{full_path}{reset}")
            logger.info(f"{green}Checksum: {magenta}{checksum}{reset}")
        
        print(f"{green}Checkpoint saved successfully to {magenta}{full_path}{reset}")
        print(f"{green}Checksum: {magenta}{checksum}{reset}")
        return True
        
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {checkpoint_stats.get('stage', 'unknown')})"
        logger.error(f"{red}Failed to save checkpoint{error_context}: {str(e)}{reset}")
        return False

def load_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    device: torch.device = torch.device('cpu'),
    verbose: Optional[bool] = False,
    progress_bar: Optional[bool] = True,
    verify: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load training checkpoint with verification and progress tracking.
    
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
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track checkpoint loading progress and statistics
    loading_stats = {
        'stage': 'Initializing',
        'checkpoint_size_bytes': 0,
        'file_exists': False,
        'checksum_verified': False,
        'checksum_available': False,
        'model_loaded': False,
        'optimizer_loaded': False,
        'scheduler_loaded': False,
        'metrics_processed': False,
        'safe_load_successful': False,
        'fallback_used': False,
        'detailed_timings': {},
        'checkpoint_components': {},
        'memory_usage_mb': 0.0,
        'compatibility_info': {}
    }
    
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
    
    try:
        print(f"\n{green}Loading Checkpoint...{reset}")
        # Progress helper: define all stage titles
        titles = [
            "Checkpoint Setup",
            "Checksum Verify",
            "Loading Checkpoint",
            "Finalizing Checkpoint"
        ]
        
        # Use progress bar only if progress_bar is True
        if progress_bar:
            progress = ProgressHelper(titles)
        else:
            # Create a dummy context manager that does nothing
            class DummyBar:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def __call__(self):
                    pass
                def __setattr__(self, name, value):
                    pass
                def bar(self, *args, **kwargs):
                    return self
            progress = DummyBar()
        
        # STAGE 1: Initial Setup and File Validation
        with progress.bar("Checkpoint Setup", total=5, unit="stages") as bar:
            
            # STAGE 1.1: File Path Resolution
            bar.text = "Resolving checkpoint path..."
            path_start = time.time()
            
            # Create full path
            full_path = Path(filename).absolute()
            
            path_time = time.time() - path_start
            loading_stats['detailed_timings']['path_resolution'] = path_time
            bar.text = f"{green}Path resolved: {full_path}{reset}"
            bar()
            
            # STAGE 1.2: File Existence Check
            bar.text = "Checking file existence..."
            existence_start = time.time()
            
            if not full_path.exists():
                bar.text = f"{red}Checkpoint file not found{reset}"
                raise FileNotFoundError(f"Checkpoint file not found: {full_path}")
            
            loading_stats['file_exists'] = True
            loading_stats['checkpoint_size_bytes'] = full_path.stat().st_size
            
            existence_time = time.time() - existence_start
            loading_stats['detailed_timings']['file_existence_check'] = existence_time
            bar.text = f"{green}File found ({loading_stats['checkpoint_size_bytes'] / 1024 / 1024:.2f} MB){reset}"
            bar()
            
            # STAGE 1.3: Checksum Verification Setup
            bar.text = "Setting up checksum verification..."
            checksum_setup_start = time.time()
            
            checksum_path = full_path.with_suffix('.md5')
            loading_stats['checksum_available'] = checksum_path.exists()
            
            checksum_setup_time = time.time() - checksum_setup_start
            loading_stats['detailed_timings']['checksum_setup'] = checksum_setup_time
            
            if verify and loading_stats['checksum_available']:
                bar.text = f"{green}Checksum file available{reset}"
            elif verify and not loading_stats['checksum_available']:
                bar.text = f"{yellow}Checksum file not available{reset}"
            else:
                bar.text = f"{yellow}Checksum verification disabled{reset}"
            bar()
            
            # STAGE 1.4: Device Configuration
            bar.text = "Configuring target device..."
            device_start = time.time()
            
            loading_stats['compatibility_info']['target_device'] = str(device)
            if model is not None:
                loading_stats['compatibility_info']['model_class'] = model.__class__.__name__
                loading_stats['compatibility_info']['model_device'] = str(next(model.parameters()).device)
            
            device_time = time.time() - device_start
            loading_stats['detailed_timings']['device_configuration'] = device_time
            bar.text = f"{green}Device configured: {device}{reset}"
            bar()
            
            # STAGE 1.5: Memory Preparation
            bar.text = "Preparing memory..."
            memory_start = time.time()
            
            # Clear cache and prepare memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            memory_time = time.time() - memory_start
            loading_stats['detailed_timings']['memory_preparation'] = memory_time
            bar.text = f"{green}Memory prepared{reset}"
            bar()
        
        # STAGE 2: Checksum Verification
        if verify and loading_stats['checksum_available']:
            with progress.bar("Checksum Verify", total=2, unit="steps") as checksum_bar:
                
                # STAGE 2.1: Checksum Calculation
                checksum_bar.text = "Calculating file checksum..."
                checksum_calc_start = time.time()
                
                with open(checksum_path, 'r') as f:
                    expected_checksum = f.read().strip()
                
                file_bytes = full_path.read_bytes()
                actual_checksum = hashlib.md5(file_bytes).hexdigest()
                
                checksum_calc_time = time.time() - checksum_calc_start
                loading_stats['detailed_timings']['checksum_calculation'] = checksum_calc_time
                checksum_bar.text = f"{green}Checksum calculated{reset}"
                checksum_bar()
                
                # STAGE 2.2: Checksum Validation
                checksum_bar.text = "Validating checksum..."
                checksum_validation_start = time.time()
                
                if expected_checksum != actual_checksum:
                    checksum_validation_time = time.time() - checksum_validation_start
                    loading_stats['detailed_timings']['checksum_validation'] = checksum_validation_time
                    checksum_bar.text = f"{red}Checksum mismatch{reset}"
                    return None, "Checksum verification failed"
                
                loading_stats['checksum_verified'] = True
                checksum_validation_time = time.time() - checksum_validation_start
                loading_stats['detailed_timings']['checksum_validation'] = checksum_validation_time
                checksum_bar.text = f"{green}Checksum validated successfully{reset}"
                checksum_bar()
        
        # STAGE 3: Checkpoint Loading
        checkpoint = None
        with progress.bar("Loading Checkpoint", total=4, unit="steps") as load_bar:
            
            load_bar.text = "Starting checkpoint loading..."
            
            # STAGE 3.1: Safe Load Attempt
            load_bar.text = "Attempting safe load..."
            safe_load_start = time.time()
            
            try:
                checkpoint = torch.load(full_path, map_location=device, weights_only=False)
                loading_stats['safe_load_successful'] = True
                loading_stats['checkpoint_components']['safe_load'] = True
                load_bar.text = f"{green}Safe load successful{reset}"
                
            except Exception as safe_load_error:
                safe_load_time = time.time() - safe_load_start
                loading_stats['detailed_timings']['safe_load_attempt'] = safe_load_time
                loading_stats['checkpoint_components']['safe_load'] = False
                load_bar.text = f"{yellow}Safe load failed, trying fallback{reset}"
                logger.warning(f"{yellow}Safe load warning: {str(safe_load_error)}{reset}")
                
                # Fallback load attempt
                fallback_start = time.time()
                try:
                    checkpoint = torch.load(full_path, map_location=device)
                    loading_stats['fallback_used'] = True
                    loading_stats['checkpoint_components']['fallback_load'] = True
                    load_bar.text = f"{green}Fallback load successful{reset}"
                except Exception as fallback_error:
                    fallback_time = time.time() - fallback_start
                    loading_stats['detailed_timings']['fallback_load_attempt'] = fallback_time
                    load_bar.text = f"{red}All load attempts failed{reset}"
                    raise fallback_error
            
            safe_load_time = time.time() - safe_load_start
            loading_stats['detailed_timings']['safe_load_attempt'] = safe_load_time
            load_bar()
            
            # STAGE 3.2: Checkpoint Structure Validation
            load_bar.text = "Validating checkpoint structure..."
            validation_start = time.time()
            
            required_keys = {'epoch', 'model_state_dict', 'metrics'}
            if not required_keys.issubset(checkpoint.keys()):
                missing = required_keys - checkpoint.keys()
                validation_time = time.time() - validation_start
                loading_stats['detailed_timings']['structure_validation'] = validation_time
                load_bar.text = f"{red}Missing required keys{reset}"
                raise ValueError(f"Checkpoint missing required keys: {missing}")
            
            # Record checkpoint components
            loading_stats['checkpoint_components'].update({
                'epoch': checkpoint.get('epoch', -1),
                'has_model_state': 'model_state_dict' in checkpoint,
                'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
                'has_scheduler_state': 'scheduler_state_dict' in checkpoint,
                'has_metrics': 'metrics' in checkpoint,
                'has_config': 'config' in checkpoint,
                'has_environment': 'environment' in checkpoint,
                'has_smote_metrics': 'smote_metrics' in checkpoint
            })
            
            validation_time = time.time() - validation_start
            loading_stats['detailed_timings']['structure_validation'] = validation_time
            load_bar.text = f"{green}Checkpoint structure validated{reset}"
            load_bar()
            
            # STAGE 3.3: Model State Loading
            load_bar.text = "Loading model state..."
            model_load_start = time.time()
            
            if model is not None and 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    loading_stats['model_loaded'] = True
                    loading_stats['checkpoint_components']['model_parameters'] = sum(p.numel() for p in model.parameters())
                    load_bar.text = f"{green}Model state loaded{reset}"
                except Exception as model_error:
                    loading_stats['model_loaded'] = False
                    load_bar.text = f"{red}Model state loading failed{reset}"
                    print(f"{yellow}Model loading warning: {str(model_error)}{reset}")
                    # Continue without model loading
            else:
                load_bar.text = f"{yellow}Model loading skipped{reset}"
            
            model_load_time = time.time() - model_load_start
            loading_stats['detailed_timings']['model_loading'] = model_load_time
            load_bar()
            
            # STAGE 3.4: Metrics Processing
            load_bar.text = "Processing metrics..."
            metrics_start = time.time()
            
            # Convert lists back to numpy arrays
            metrics = checkpoint.get('metrics', {})
            processed_metrics_count = 0
            
            for k, v in metrics.items():
                if isinstance(v, list):
                    metrics[k] = np.array(v)
                    processed_metrics_count += 1
            
            loading_stats['metrics_processed'] = True
            loading_stats['checkpoint_components']['processed_metrics_count'] = processed_metrics_count
            loading_stats['checkpoint_components']['total_metrics_count'] = len(metrics)
            
            metrics_time = time.time() - metrics_start
            loading_stats['detailed_timings']['metrics_processing'] = metrics_time
            load_bar.text = f"{green}Metrics processed ({len(metrics)} total){reset}"
            load_bar()
        
        # STAGE 4: Final Assembly and Reporting
        with progress.bar("Finalizing Checkpoint", total=2, unit="steps") as final_bar:
            
            # STAGE 4.1: Memory Usage Calculation
            final_bar.text = "Calculating memory usage..."
            memory_calc_start = time.time()
            
            if device.type == 'cuda':
                loading_stats['memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            memory_calc_time = time.time() - memory_calc_start
            loading_stats['detailed_timings']['memory_calculation'] = memory_calc_time
            final_bar.text = f"{green}Memory usage calculated{reset}"
            final_bar()
            
            # STAGE 4.2: Final Summary
            final_bar.text = "Generating loading summary..."
            summary_start = time.time()
            
            # Calculate total time from detailed timings (excluding total)
            timing_keys = [k for k in loading_stats['detailed_timings'].keys() if k != 'total']
            total_time = sum(loading_stats['detailed_timings'][k] for k in timing_keys)
            loading_stats['detailed_timings']['total'] = total_time
            
            # Prepare return values (updated to match the function's signature)
            return_data = {
                'model_state_dict': checkpoint.get('model_state_dict'),
                'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
                'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
                'metrics': metrics,
                'config': checkpoint.get('config', {}),
                'environment': checkpoint.get('environment', {}),
                'smote_metrics': checkpoint.get('smote_metrics', {}),
                'epoch': checkpoint.get('epoch', -1),
                'checkpoint_metadata': checkpoint.get('checkpoint_metadata', {}),
                'training_meta': checkpoint.get('training_meta', {})
            }
            
            # Display summary
            if verbose:
                logger.info(f"\n{blue}Checkpoint Loading Summary{reset}")
                logger.info(f"{cyan}File Information:{reset}")
                logger.info(f"  - File: {full_path}")
                logger.info(f"  - Size: {loading_stats['checkpoint_size_bytes'] / 1024 / 1024:.2f} MB")
                logger.info(f"  - Epoch: {checkpoint.get('epoch', 'Unknown')}")
                logger.info(f"  - Checksum verified: {'Yes' if loading_stats['checksum_verified'] else 'No'}")
                logger.info(f"  - Safe load: {'Yes' if loading_stats['safe_load_successful'] else 'No'}")
                logger.info(f"  - Fallback used: {'Yes' if loading_stats['fallback_used'] else 'No'}")
                
                logger.info(f"\n{cyan}Loaded Components:{reset}")
                logger.info(f"  - Model state: {'Yes' if loading_stats['model_loaded'] else 'No'}")
                logger.info(f"  - Optimizer state: {'Yes' if loading_stats['checkpoint_components']['has_optimizer_state'] else 'No'}")
                logger.info(f"  - Scheduler state: {'Yes' if loading_stats['checkpoint_components']['has_scheduler_state'] else 'No'}")
                logger.info(f"  - Metrics: {loading_stats['checkpoint_components']['total_metrics_count']}")
                logger.info(f"  - Config: {'Yes' if loading_stats['checkpoint_components']['has_config'] else 'No'}")
                logger.info(f"  - SMOTE metrics: {'Yes' if loading_stats['checkpoint_components']['has_smote_metrics'] else 'No'}")
                
                if loading_stats['model_loaded']:
                    logger.info(f"  - Model parameters: {loading_stats['checkpoint_components']['model_parameters']:,}")
                
                logger.info(f"\n{cyan}Processing Details:{reset}")
                logger.info(f"  - Total time: {total_time:.3f}s")
                logger.info(f"  - Target device: {loading_stats['compatibility_info']['target_device']}")
                if loading_stats['memory_usage_mb'] > 0:
                    logger.info(f"  - Peak memory usage: {loading_stats['memory_usage_mb']:.1f} MB")
                
                # Display detailed timings
                logger.info(f"\n{cyan}Detailed Timings:{reset}")
                for stage, timing in loading_stats['detailed_timings'].items():
                    if stage != 'total':
                        logger.info(f"  - {stage}: {timing:.3f}s")
                logger.info(f"  - Total time: {total_time:.3f}s")
            
            summary_time = time.time() - summary_start
            loading_stats['detailed_timings']['final_summary'] = summary_time
            final_bar.text = f"{green}Checkpoint loading completed successfully{reset}"
            final_bar()
        
        logger.info(f"{green}Loaded checkpoint from {magenta}{full_path}{reset} {green}(epoch {checkpoint.get('epoch', 'Unknown')}){reset}")
        
        return return_data, None
        
    except Exception as e:
        # Log error with context
        error_context = f" (stage: {loading_stats.get('stage', 'unknown')})"
        error_message = f"Failed to load checkpoint from {filename}: {str(e)}"
        
        logger.error(f"{red}Checkpoint loading failed{error_context}: {str(e)}{reset}")
        return None, error_message

def save_training_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    verbose: Optional[bool] = False,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Save all training artifacts including model, metrics, configuration, visualizations,
    and SMOTE evaluation metrics with progress tracking.
    
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
    # Setup styling for colored output
    red = Fore.RED + Style.BRIGHT
    yellow = Fore.YELLOW + Style.BRIGHT
    green = Fore.GREEN + Style.BRIGHT
    blue = Fore.BLUE + Style.BRIGHT
    cyan = Fore.CYAN + Style.BRIGHT
    magenta = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    
    # Track artifact saving progress and statistics
    artifact_stats = {
        'stage': 'Initializing',
        'total_artifacts': 0,
        'saved_artifacts': 0,
        'failed_artifacts': 0,
        'model_parameters': 0,
        'metrics_count': 0,
        'config_entries': 0,
        'archive_created': False,
        'confusion_matrix_created': False,
        'smote_metrics_saved': False,
        'detailed_timings': {},
        'artifact_sizes': {},
        'memory_usage_mb': 0.0,
        'run_id': ''
    }
    
    saved_artifacts = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    artifact_stats['run_id'] = run_id
    
    try:
        print(f"\n{green}Saving Training Artifacts...{reset}")

        # Progress helper: define all stage titles
        titles = [
            "Artifact Setup",
            "Saving Artifacts",
            "Finalizing Artifacts"
        ]
        progress = ProgressHelper(titles)
        
        # STAGE 1: Initial Setup and Directory Preparation
        with progress.bar("Artifact Setup", total=6, unit="stages") as bar:
            
            # STAGE 1.1: Configuration and Path Setup
            bar.text = "Configuring artifact paths..."
            config_start = time.time()
            
            # Use custom output directory if provided
            artifact_dir = output_dir if output_dir is not None else ARTIFACTS_DIR
            model_dir = output_dir if output_dir is not None else MODEL_DIR
            metrics_dir = output_dir if output_dir is not None else METRICS_DIR
            config_dir = output_dir if output_dir is not None else CONFIG_DIR
            info_dir = output_dir if output_dir is not None else INFO_DIR
            figure_dir = output_dir if output_dir is not None else FIGURE_DIR
            
            config_time = time.time() - config_start
            artifact_stats['detailed_timings']['configuration'] = config_time
            bar.text = f"{green}Artifact paths configured{reset}"
            bar()
            
            # STAGE 1.2: Directory Creation
            bar.text = "Creating output directories..."
            dir_start = time.time()
            
            directories = [model_dir, metrics_dir, config_dir, info_dir, artifact_dir, figure_dir]
            for dir_path in directories:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            dir_time = time.time() - dir_start
            artifact_stats['detailed_timings']['directory_creation'] = dir_time
            bar.text = f"{green}Output directories created ({len(directories)} directories){reset}"
            bar()
            
            # STAGE 1.3: Model Analysis
            bar.text = "Analyzing model structure..."
            model_analysis_start = time.time()
            
            # Count model parameters
            model_parameters = sum(p.numel() for p in model.parameters())
            artifact_stats['model_parameters'] = model_parameters
            artifact_stats['metrics_count'] = len(metrics)
            artifact_stats['config_entries'] = len(config)
            
            model_analysis_time = time.time() - model_analysis_start
            artifact_stats['detailed_timings']['model_analysis'] = model_analysis_time
            bar.text = f"{green}Model analyzed ({model_parameters:,} parameters){reset}"
            bar()
            
            # STAGE 1.4: Memory Preparation
            bar.text = "Preparing memory..."
            memory_start = time.time()
            
            # Clear cache and prepare memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            memory_time = time.time() - memory_start
            artifact_stats['detailed_timings']['memory_preparation'] = memory_time
            bar.text = f"{green}Memory prepared{reset}"
            bar()
            
            # STAGE 1.5: Artifact Planning
            bar.text = "Planning artifact types..."
            planning_start = time.time()
            
            # Calculate total artifacts to save
            total_artifacts = 5  # model, metrics, config, info, archive
            if 'smote_metrics' in metrics or any(k in metrics for k in ['feature_correlation_diff', 'avg_neighbor_distance']):
                total_artifacts += 1
            if 'labels' in metrics and 'preds' in metrics:
                total_artifacts += 1
            
            artifact_stats['total_artifacts'] = total_artifacts
            
            planning_time = time.time() - planning_start
            artifact_stats['detailed_timings']['artifact_planning'] = planning_time
            bar.text = f"{green}Artifact planning completed ({total_artifacts} artifacts){reset}"
            bar()
            
            # STAGE 1.6: Setup Completion
            bar.text = "Setup completion..."
            completion_start = time.time()
            
            # Final setup steps
            artifact_stats['stage'] = 'Setup Complete'
            
            completion_time = time.time() - completion_start
            artifact_stats['detailed_timings']['setup_completion'] = completion_time
            bar.text = f"{green}Setup completed{reset}"
            bar()
        
        # STAGE 2: Artifact Saving with Progress Tracking
        artifact_stats['stage'] = 'Artifact Saving'
        
        with progress.bar("Saving Artifacts", total=total_artifacts+1, unit="artifacts") as artifact_bar:
            
            # STAGE 2.1: Model State Saving
            artifact_bar.text = "Saving model state..."
            model_save_start = time.time()
            
            try:
                model_path = model_dir / f"ids_model_{run_id}.pth"
                torch.save(model.state_dict(), model_path)
                saved_artifacts['model'] = model_path
                artifact_stats['saved_artifacts'] += 1
                artifact_stats['artifact_sizes']['model'] = model_path.stat().st_size
                artifact_bar.text = f"{green}Model state saved ({artifact_stats['model_parameters']:,} parameters){reset}"
            except Exception as model_error:
                artifact_stats['failed_artifacts'] += 1
                artifact_bar.text = f"{red}Model state saving failed{reset}"
                logger.warning(f"{yellow}Model saving warning: {str(model_error)}{reset}")
            
            model_save_time = time.time() - model_save_start
            artifact_stats['detailed_timings']['model_saving'] = model_save_time
            artifact_bar()
            
            # STAGE 2.2: Metrics Saving
            artifact_bar.text = "Saving evaluation metrics..."
            metrics_save_start = time.time()
            
            try:
                metrics_path = metrics_dir / f"ids_model_metrics_{run_id}.json"
                processed_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, np.ndarray):
                        processed_metrics[k] = v.tolist()
                    else:
                        processed_metrics[k] = v
                
                with open(metrics_path, 'w') as f:
                    json.dump(processed_metrics, f, indent=2)
                
                saved_artifacts['metrics'] = metrics_path
                artifact_stats['saved_artifacts'] += 1
                artifact_stats['artifact_sizes']['metrics'] = metrics_path.stat().st_size
                artifact_bar.text = f"{green}Metrics saved ({len(processed_metrics)} metrics){reset}"
            except Exception as metrics_error:
                artifact_stats['failed_artifacts'] += 1
                artifact_bar.text = f"{red}Metrics saving failed{reset}"
                logger.warning(f"{yellow}Metrics saving warning: {str(metrics_error)}{reset}")
            
            metrics_save_time = time.time() - metrics_save_start
            artifact_stats['detailed_timings']['metrics_saving'] = metrics_save_time
            artifact_bar()
            
            # STAGE 2.3: SMOTE Metrics Saving
            artifact_bar.text = "Saving SMOTE metrics..."
            smote_save_start = time.time()
            
            if 'smote_metrics' in metrics or any(k in metrics for k in ['feature_correlation_diff', 'avg_neighbor_distance']):
                try:
                    smote_metrics = {
                        'oversampler': config.get('oversampler', 'SMOTE'),
                        'k_neighbors': config.get('k_neighbors', 3),
                        'feature_correlation_diff': metrics.get('feature_correlation_diff'),
                        'neighbor_ratio': metrics.get('avg_neighbor_distance'),
                        'minority_class_size': metrics.get('minority_class_size'),
                        'synthetic_samples_generated': metrics.get('synthetic_samples_generated'),
                        'timestamp': timestamp
                    }
                    smote_metrics_path = metrics_dir / f"smote_evaluation_{run_id}.json"
                    with open(smote_metrics_path, 'w') as f:
                        json.dump(smote_metrics, f, indent=2)
                    
                    saved_artifacts['smote_metrics'] = smote_metrics_path
                    artifact_stats['saved_artifacts'] += 1
                    artifact_stats['smote_metrics_saved'] = True
                    artifact_stats['artifact_sizes']['smote_metrics'] = smote_metrics_path.stat().st_size
                    artifact_bar.text = f"{green}SMOTE metrics saved{reset}"
                except Exception as smote_error:
                    artifact_stats['failed_artifacts'] += 1
                    artifact_bar.text = f"{red}SMOTE metrics saving failed{reset}"
                    logger.warning(f"{yellow}SMOTE metrics saving warning: {str(smote_error)}{reset}")
            else:
                artifact_bar.text = f"{yellow}SMOTE metrics skipped{reset}"
            
            smote_save_time = time.time() - smote_save_start
            artifact_stats['detailed_timings']['smote_metrics_saving'] = smote_save_time
            artifact_bar()
            
            # STAGE 2.4: Configuration Saving
            artifact_bar.text = "Saving training configuration..."
            config_save_start = time.time()
            
            try:
                config_path = config_dir / f"ids_model_config_{run_id}.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                saved_artifacts['config'] = config_path
                artifact_stats['saved_artifacts'] += 1
                artifact_stats['artifact_sizes']['config'] = config_path.stat().st_size
                artifact_bar.text = f"{green}Configuration saved ({len(config)} entries){reset}"
            except Exception as config_error:
                artifact_stats['failed_artifacts'] += 1
                artifact_bar.text = f"{red}Configuration saving failed{reset}"
                logger.warning(f"{yellow}Configuration saving warning: {str(config_error)}{reset}")
            
            config_save_time = time.time() - config_save_start
            artifact_stats['detailed_timings']['config_saving'] = config_save_time
            artifact_bar()
            
            # STAGE 2.5: Information File Saving
            artifact_bar.text = "Saving model information..."
            info_save_start = time.time()
            
            try:
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
                info_path = info_dir / f"ids_model_info_{run_id}.json"
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
                
                saved_artifacts['info'] = info_path
                artifact_stats['saved_artifacts'] += 1
                artifact_stats['artifact_sizes']['info'] = info_path.stat().st_size
                artifact_bar.text = f"{green}Model information saved{reset}"
            except Exception as info_error:
                artifact_stats['failed_artifacts'] += 1
                artifact_bar.text = f"{red}Information saving failed{reset}"
                logger.warning(f"{yellow}Information saving warning: {str(info_error)}{reset}")
            
            info_save_time = time.time() - info_save_start
            artifact_stats['detailed_timings']['info_saving'] = info_save_time
            artifact_bar()
            
            # STAGE 2.6: Confusion Matrix Creation
            artifact_bar.text = "Creating confusion matrix..."
            cm_start = time.time()
            
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
                    
                    cm_path = figure_dir / f"confusion_matrix_{run_id}.png"
                    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    saved_artifacts['confusion_matrix'] = cm_path
                    artifact_stats['saved_artifacts'] += 1
                    artifact_stats['confusion_matrix_created'] = True
                    artifact_stats['artifact_sizes']['confusion_matrix'] = cm_path.stat().st_size
                    artifact_bar.text = f"{green}Confusion matrix created{reset}"
                except Exception as cm_error:
                    artifact_stats['failed_artifacts'] += 1
                    artifact_bar.text = f"{red}Confusion matrix creation failed{reset}"
                    logger.warning(f"{yellow}Confusion matrix warning: {str(cm_error)}{reset}")
            else:
                artifact_bar.text = f"{yellow}Confusion matrix skipped{reset}"
            
            cm_time = time.time() - cm_start
            artifact_stats['detailed_timings']['confusion_matrix_creation'] = cm_time
            artifact_bar()
            
            # STAGE 2.7: Archive Creation
            artifact_bar.text = "Creating artifact archive..."
            archive_start = time.time()
            
            try:
                archive_path = artifact_dir / f"ids_model_artifacts_{run_id}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    for artifact_name, file_path in saved_artifacts.items():
                        if isinstance(file_path, Path) and file_path.exists():
                            tar.add(file_path, arcname=file_path.name)
                
                saved_artifacts['archive'] = archive_path
                artifact_stats['saved_artifacts'] += 1
                artifact_stats['archive_created'] = True
                artifact_stats['artifact_sizes']['archive'] = archive_path.stat().st_size
                artifact_bar.text = f"{green}Artifact archive created{reset}"
            except Exception as archive_error:
                artifact_stats['failed_artifacts'] += 1
                artifact_bar.text = f"{red}Archive creation failed{reset}"
                logger.warning(f"{yellow}Archive creation warning: {str(archive_error)}{reset}")
            
            archive_time = time.time() - archive_start
            artifact_stats['detailed_timings']['archive_creation'] = archive_time
            artifact_bar()
        
        # STAGE 3: Final Summary and Reporting
        with progress.bar("Finalizing Artifacts", total=2, unit="steps") as final_bar:
            
            # STAGE 3.1: Memory Usage Calculation
            final_bar.text = "Calculating memory usage..."
            memory_calc_start = time.time()
            
            if torch.cuda.is_available():
                artifact_stats['memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            memory_calc_time = time.time() - memory_calc_start
            artifact_stats['detailed_timings']['memory_calculation'] = memory_calc_time
            final_bar.text = f"{green}Memory usage calculated{reset}"
            final_bar()
            
            # STAGE 3.2: Final Summary
            final_bar.text = "Generating artifact summary..."
            summary_start = time.time()
            
            # Calculate total time from detailed timings (excluding total)
            timing_keys = [k for k in artifact_stats['detailed_timings'].keys() if k != 'total']
            total_time = sum(artifact_stats['detailed_timings'][k] for k in timing_keys)
            artifact_stats['detailed_timings']['total'] = total_time
            
            # Display summary
            if verbose:
                logger.info(f"\n{blue}Training Artifacts Saving Summary{reset}")
                logger.info(f"{cyan}Artifact Overview:{reset}")
                logger.info(f"  - Run ID: {run_id}")
                logger.info(f"  - Total artifacts: {artifact_stats['total_artifacts']}")
                logger.info(f"  - Successfully saved: {artifact_stats['saved_artifacts']}")
                logger.info(f"  - Failed to save: {artifact_stats['failed_artifacts']}")
                success_rate = (artifact_stats['saved_artifacts'] / artifact_stats['total_artifacts'] * 100) if artifact_stats['total_artifacts'] > 0 else 0
                logger.info(f"  - Success rate: {success_rate:.1f}%")
                
                logger.info(f"\n{cyan}Artifact Details:{reset}")
                logger.info(f"  - Model parameters: {artifact_stats['model_parameters']:,}")
                logger.info(f"  - Metrics saved: {artifact_stats['metrics_count']}")
                logger.info(f"  - Config entries: {artifact_stats['config_entries']}")
                logger.info(f"  - SMOTE metrics: {'Yes' if artifact_stats['smote_metrics_saved'] else 'No'}")
                logger.info(f"  - Confusion matrix: {'Yes' if artifact_stats['confusion_matrix_created'] else 'No'}")
                logger.info(f"  - Archive created: {'Yes' if artifact_stats['archive_created'] else 'No'}")
                
                logger.info(f"\n{cyan}File Sizes:{reset}")
                total_size = sum(artifact_stats['artifact_sizes'].values()) / 1024 / 1024
                for artifact_type, size_bytes in artifact_stats['artifact_sizes'].items():
                    size_mb = size_bytes / 1024 / 1024
                    logger.info(f"  - {artifact_type}: {size_mb:.2f} MB")
                logger.info(f"  - Total size: {total_size:.2f} MB")
                
                logger.info(f"\n{cyan}Processing Details:{reset}")
                logger.info(f"  - Total time: {total_time:.2f}s")
                logger.info(f"  - Output directory: {artifact_dir}")
                if artifact_stats['memory_usage_mb'] > 0:
                    logger.info(f"  - Peak memory usage: {artifact_stats['memory_usage_mb']:.1f} MB")
                
                # Display saved artifact paths
                logger.info(f"\n{cyan}Saved Artifact Paths:{reset}")
                for artifact_type, path in saved_artifacts.items():
                    status = f"{green}SUCCESS{reset}" if path.exists() else f"{red}MISSING{reset}"
                    size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
                    logger.info(f"  - {artifact_type}: {status} ({size_mb:.2f} MB)")
                    logger.info(f"    {path}")
                
                # Display detailed timings
                logger.info(f"\n{cyan}Detailed Timings:{reset}")
                for stage, timing in artifact_stats['detailed_timings'].items():
                    if stage != 'total':
                        logger.info(f"  - {stage}: {timing:.3f}s")
                logger.info(f"  - Total time: {total_time:.3f}s")
            
            summary_time = time.time() - summary_start
            artifact_stats['detailed_timings']['final_summary'] = summary_time
            final_bar.text = f"{green}Artifact saving completed successfully{reset}"
            final_bar()
        
        if verbose:
            logger.info(f"\n{green}All training artifacts saved successfully for run {magenta}{run_id}{reset}")
        return saved_artifacts

    except Exception as e:
        # Log error with context
        error_context = f" (stage: {artifact_stats.get('stage', 'unknown')})"
        logger.error(f"{red}Artifact saving failed{error_context}: {str(e)}{reset}")
        
        # Cleanup partial saves
        for path in saved_artifacts.values():
            try:
                if isinstance(path, Path) and path.exists():
                    path.unlink()
            except:
                pass
        
        return {}

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
    print(Fore.RED + Style.BRIGHT + "0. Exit")

def interactive_main(
    logger: logging.Logger,
    device: torch.device,
    config: Dict[str, Any],
    directories: Dict[str, Path]
) -> None:
    """Interactive main function with new options"""
    # Clear any residual input buffer from system initialization
    if hasattr(sys.stdin, 'flush'):
        try:
            sys.stdin.flush()
        except:
            pass
    
    # Small delay to ensure all output is complete
    time.sleep(1)

    # Use the initialized objects passed from main
    while True:
        print("\033c", end="")
        banner()
        print_menu()
        
        # Input handling with retry logic
        choice = None
        while not choice:
            try:
                choice = input(Fore.YELLOW + Style.BRIGHT + "\nSelect an option (0-11): ").strip()
                
                # If empty input, retry
                if not choice:
                    continue
            
            except (EOFError, KeyboardInterrupt):
                print(Fore.RED + Style.BRIGHT + "\nExiting...")
                print(Fore.YELLOW + Style.BRIGHT + "Goodbye!")
                return
        
        try:
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
                    print(Fore.GREEN + Style.BRIGHT + "\nStarting training pipeline with selected preset...\n")
                    train_config = preset
                else:
                    print(Fore.GREEN + Style.BRIGHT + "\nStarting training pipeline with current configuration...\n")
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
                    print(Fore.GREEN + Style.BRIGHT + "\nStarting synthetic training with selected preset...")
                    train_config = preset
                else:
                    print(Fore.GREEN + Style.BRIGHT + "\nStarting synthetic training with current configuration...")
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
            
            elif choice == "0":
                print(Fore.RED + Style.BRIGHT + "\nExiting..." + Style.RESET_ALL)
                print(Fore.YELLOW + Style.BRIGHT + "Goodbye!" + Style.RESET_ALL)
                break
                
            else:
                print(Fore.RED + Style.BRIGHT + f"Invalid selection '{choice}'. Choose 0-11." + Style.RESET_ALL)
        
        except KeyboardInterrupt:
            print(Fore.RED + Style.BRIGHT + "\nOperation interrupted by user" + Style.RESET_ALL)
        except Exception as e:
            logger.error(f"Main menu error: {e}", exc_info=True)
            print(Fore.RED + Style.BRIGHT + "\nUnexpected error in main menu: " + Fore.YELLOW + Style.BRIGHT + f"{str(e)}" + Style.RESET_ALL)
        
        # Only continue if not exiting
        if choice != "0":
            try:
                input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
            except (EOFError, KeyboardInterrupt):
                print(Fore.RED + Style.BRIGHT + "\nExiting..." + Style.RESET_ALL)
                print(Fore.YELLOW + Style.BRIGHT + "Goodbye!" + Style.RESET_ALL)
                break

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
        title="Classification Report",
        box=box.ROUNDED,
        header_style="bold cyan",
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
        "Accuracy",
        "",
        "",
        f"{report['accuracy']:.4f}",
        str(report['macro avg']['support']),
        style="bold yellow"
    )
    
    # Add macro avg row
    report_table.add_row(
        "Macro Avg",
        f"{report['macro avg']['precision']:.4f}",
        f"{report['macro avg']['recall']:.4f}",
        f"{report['macro avg']['f1-score']:.4f}",
        str(report['macro avg']['support']),
        style="bold blue"
    )
    
    console.print(report_table)

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
        
        # If no arguments provided, launch interactive mode
        if len(sys.argv) == 1:
            interactive_main(logger, device, directories, config)
            sys.exit(0)

        # Handle SMOTE comparison mode
        if args.compare_oversamplers:
            try:
                logger.info(Fore.CYAN + Style.BRIGHT + "\nOversampler Comparison Mode")
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
            logger.info(Fore.CYAN + Style.BRIGHT + "\nEnhanced Network Intrusion Detection Model Trainer")
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
                logger.info(Fore.GREEN + Style.BRIGHT + "\nTraining Completed Successfully")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Best validation accuracy: {results['metrics']['val_acc']:.2%}")
                logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Best validation AUC: {results['metrics']['val_auc']:.4f}")
                if 'smote_metrics' in results['metrics']:
                    logger.info(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"SMOTE quality score: {results['metrics']['smote_metrics'].get('quality_score', 'N/A')}")
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