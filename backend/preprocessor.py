import os
import sys
import time
import hashlib
import json
import math
import csv
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import psutil
import shutil
import matplotlib.pyplot as plt
import joblib
import warnings
import subprocess
from colorama import Fore, Back, Style, init as colorama_init
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel
from alive_progress import alive_bar
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

# Initialize colorama for colored output
colorama_init(autoreset=True)

# Initialize rich console
console = Console()

# Constants for hybrid system
HYBRID_FEATURE_COUNT = 20
SCALER_RANGE = (0.1, 0.9)
MEMORY_SAFETY_FACTOR = 0.7

class MemoryAwarePreprocessor:
    """A class to test and determine optimal chunk sizes for processing large files
    while maintaining safe memory usage and performance thresholds."""
    
    VERSION = "2.2.0"  # Updated version to include preprocessing
    
    def __init__(self):
        """Initialize the MemoryAwarePreprocessor with default settings."""
        # Memory safety configuration
        self.safety_factor = 0.8  # 20% safety buffer
        self.min_chunk_size = 1000  # Minimum chunk size to attempt
        self.max_ram_usage = 0.7  # Max fraction of available RAM to use
        self.max_history_entries = 50  # Maximum history entries to keep
        
        # Performance monitoring thresholds
        self.performance_thresholds = {
            'memory_spike': 0.9,  # 90% of target memory
            'throughput_drop': 0.7,  # 70% of average throughput
            'cpu_overload': 0.85  # 85% CPU usage
        }
        
        # Color mapping for console output
        self.color_map = {
            'info': Fore.CYAN + Style.BRIGHT,
            'success': Fore.GREEN + Style.BRIGHT,
            'warning': Fore.YELLOW + Style.BRIGHT,
            'error': Fore.RED + Style.BRIGHT,
            'highlight': Fore.MAGENTA + Style.BRIGHT,
            'system': Fore.BLUE + Style.BRIGHT,
            'debug': Fore.WHITE + Style.BRIGHT
        }
        
        # System information
        self.system_info = self._get_system_info()
        
        try:
            # Display initialization banner
            self._display_initialization_banner()
            
            # Initialize and create directories
            self._initialize_directories()
            
            # Set file paths
            self.config_file = self.config_dir / "preprocessing_config.json"
            self.history_file = self.history_dir / "preprocessing_history.json"
            self.log_file = self.log_dir / "preprocessing.log"
            
            # Initialize files
            self._initialize_files()
            
            # Load configuration
            self.config = self.load_config()
            
            # Validate configuration
            self._validate_config()
            
            # Print initialization status
            self._print_init_status()
            
            # Log successful initialization
            self._log_event(f"Initialization completed successfully (v{self.VERSION})", "info")
            
            # Ask user if they want to proceed
            self._confirm_proceed_after_init()
        
        except KeyboardInterrupt:
            self.print_color("\nInitialization cancelled by user.", 'warning')
            self._log_event("Initialization cancelled by user", "warning")
            raise
        except Exception as e:
            error_msg = f"Critical error during initialization: {str(e)}"
            self.print_color(f"\n{error_msg}", 'error')
            self._log_event(error_msg, 'error')
            raise

    def _display_initialization_banner(self) -> None:
        """Display the initialization banner with interactive prompt."""
        try:
            # Console safety checks
            if not hasattr(console, 'width'):
                # Safe default
                console_width = 80
            else:
                # Minimum width
                console_width = max(60, getattr(console, 'width', 80))
            
            console.clear()
            
            # ASCII art banner
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
            
            if banner_width > 80:
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
                console.print("\n" + "=" * min(34, banner_width), style="bold cyan")
                console.print(f"- PREPROCESSOR v{self.VERSION} -".center(min(34, banner_width)), style="bold magenta")
                console.print("Memory-Aware Preprocessing".center(min(34, banner_width)), style="bold cyan")
                console.print("with Adaptive Chunk-Sizing".center(min(34, banner_width)), style="bold cyan")
                console.print("=" * min(34, banner_width), style="bold cyan")
            else:
                # Simple fallback for narrow terminals
                console.print("\n" + "=" * min(40, banner_width), style="bold cyan")
                console.print("GreyChamp | IDS - SYSTEM INITIALIZATION".center(min(40, banner_width)), style="bold yellow")
                console.print("-" * min(40, banner_width), style="bold cyan")
                console.print(f"- PREPROCESSOR v{self.VERSION} -".center(min(40, banner_width)), style="bold magenta")
                console.print("Adaptive Chunk Sizing with Safety Margins".center(min(40, banner_width)), style="bold cyan")
                console.print("Performance Monitoring & Preprocessing".center(min(40, banner_width)), style="bold cyan")
                console.print("=" * min(40, banner_width) + "\n", style="bold cyan")
            
            # Display system check message
            console.print("\n[bold green]Running system checks and Preprocessor initialization...[/bold green]")
            
            # Show loading animation
            with console.status("[bold cyan]Initializing MemoryAwarePreprocessor...[/bold cyan]", spinner="dots"):
                time.sleep(1.5)
        
        except Exception:
            # Fallback simple banner
            self.print_color("\n" + "="*40, 'highlight')
            self.print_color("GreyChamp | IDS - SYSTEM INITIALIZATION".center(40), 'highlight')
            self.print_color("-"*40, 'highlight')
            self.print_color(f"- PREPROCESSOR v{self.VERSION} -".center(40), 'highlight')
            self.print_color("Adaptive Chunk Sizing with Safety Margins".center(40), 'info')
            self.print_color("Performance Monitoring & Preprocessing".center(40), 'info')
            self.print_color("="*40, 'highlight')
            self.print_color("\nInitializing system...", 'warning')

    def _confirm_proceed_after_init(self) -> None:
        """Ask user if they want to proceed after successful initialization."""
        try:
            self.print_color("\n" + "-" * 40, 'success')
            self.print_color("INITIALIZATION COMPLETE", 'success')
            self.print_color("-" * 40, 'success')
            
            self.print_color("System Status:", 'success')
            self.print_color(f"  ├─ Available RAM: {Fore.YELLOW + Style.BRIGHT}{self.system_info['available_ram_gb']:.2f} GB", 'success')
            self.print_color(f"  └─ CPU Cores: {Fore.YELLOW + Style.BRIGHT}{self.system_info['cpu_cores']}", 'success')
            self.print_color(f"\nDirectories initialized:", 'success')
            self.print_color(f"  ├─ Config: {Fore.CYAN + Style.BRIGHT}{self.config_dir}", 'success')
            self.print_color(f"  ├─ Logs: {Fore.CYAN + Style.BRIGHT}{self.log_dir}", 'success')
            self.print_color(f"  ├─ History: {Fore.CYAN + Style.BRIGHT}{self.history_dir}", 'success')
            self.print_color(f"  ├─ Results: {Fore.CYAN + Style.BRIGHT}{self.results_dir}", 'success')
            self.print_color(f"  └─ Models: {Fore.CYAN + Style.BRIGHT}{self.models_dir}", 'success')
            self.print_color(f"\nFiles initialized:", 'success')
            self.print_color(f"  ├─ Configuration file: {Fore.MAGENTA + Style.BRIGHT}{self.config_file}", 'success')
            self.print_color(f"  ├─ Log file: {Fore.MAGENTA + Style.BRIGHT}{self.log_file}", 'success')
            self.print_color(f"  └─ History file: {Fore.MAGENTA + Style.BRIGHT}{self.history_file}", 'success')
            
            self.print_color("\nConfiguration Summary:", 'success')
            self.print_color(f"  ├─ Target memory per chunk: {Fore.YELLOW + Style.BRIGHT}{self.config.get('default_target_mb', 256)} MB", 'success')
            self.print_color(f"  ├─ Minimum chunk size: {Fore.YELLOW + Style.BRIGHT}{self.config.get('default_min_chunk', 1000)} rows", 'success')
            self.print_color(f"  ├─ Output directory: {Fore.MAGENTA + Style.BRIGHT}{self.config.get('default_output_dir', 'results')}", 'success')
            self.print_color(f"  └─ Safety factor: {Fore.YELLOW + Style.BRIGHT}{self.safety_factor}", 'success')
            
            # Interactive prompt
            max_attempts = 3
            user_choice = None
            
            for attempt in range(max_attempts):
                try:
                    prompt = Fore.YELLOW + Style.BRIGHT + "\nProceed with Preprocessor? (Y/n/q): " + Style.RESET_ALL
                    response = input(prompt).strip().lower()
                    
                    if response in ['y', 'yes', '']:
                        user_choice = True
                        break
                    elif response in ['n', 'no', 'q', 'quit']:
                        user_choice = False
                        break
                    else:
                        if attempt < max_attempts - 1:
                            self.print_color("\nPlease enter 'y' for yes, 'n' for no, or 'q' for quit.", 'warning')
                
                except (EOFError, KeyboardInterrupt):
                    user_choice = False
                    break
                except Exception as input_error:
                    if attempt < max_attempts - 1:
                        self.print_color(f"\nInput error: {str(input_error)} - please try again.", 'error')
            
            # Handle user choice
            if user_choice is None:
                user_choice = True  # Default to continue
                self.print_color("\nUsing default choice: continue", 'success')
            
            if user_choice is False:
                self.print_color("\nExiting CSV Chunk Tester...", 'warning')
                self._log_event("User chose to exit after initialization", "info")
                sys.exit(0)
            
            # Safe console clear before proceeding
            try:
                if hasattr(console, 'clear'):
                    console.clear()
                else:
                    # Platform-specific fallback
                    if sys.platform == 'win32':
                        os.system('cls')
                    else:
                        os.system('clear')
            except Exception:
                # Ignore clear failures
                pass
            
            self.print_color("\nProceeding to main system...", 'success')
        
        except KeyboardInterrupt:
            self.print_color("\nInitialization confirmation interrupted by user.", 'warning')
            self._log_event("Initialization confirmation interrupted", "warning")
            sys.exit(0)
        except Exception as e:
            self.print_color(f"\nError in confirmation prompt: {str(e)}", 'warning')
            self.print_color("\nContinuing anyway...", 'success')
            self._log_event(f"Confirmation prompt error: {str(e)}", "warning")
        finally:
            # Clean up input buffer - Windows-safe version
            try:
                # Small delay for any pending I/O
                time.sleep(0.1)

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
                self.print_color(f"\nInput buffer cleanup failed (non-critical): {cleanup_error}", 'error')

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information."""
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'total_ram_gb': mem.total / (1024 ** 3),
            'available_ram_gb': mem.available / (1024 ** 3),
            'ram_used_percent': mem.percent,
            'cpu_cores': psutil.cpu_count(),
            'cpu_usage': psutil.cpu_percent(),
            'disk_total_gb': disk.total / (1024 ** 3),
            'disk_used_gb': disk.used / (1024 ** 3),
            'disk_free_gb': disk.free / (1024 ** 3),
            'timestamp': datetime.now().isoformat()
        }

    def _initialize_directories(self) -> None:
        """Initialize and validate all required directories with interactive prompts."""
        
        # Display directory initialization status
        self.print_color("\nSetting up required directories...", 'warning')
        
        # Get user confirmation for directory creation
        self._confirm_directory_creation()
        
        # resolve config directory relative to this script
        self.config_dir = Path(__file__).resolve().parent / "config"
        self._create_and_validate_directory(
            self.config_dir,
            "Config directory",
            check_execute=True
        )
        
        # Log directory
        self.log_dir = Path(__file__).resolve().parent / "logs"
        self._create_and_validate_directory(
            self.log_dir,
            "Log directory",
            check_execute=False
        )
        
        # History directory
        self.history_dir = Path(__file__).resolve().parent / "history"
        self._create_and_validate_directory(
            self.history_dir,
            "History directory",
            check_execute=True
        )
        
        # Results directory (from config or default)
        self.results_dir = Path(__file__).resolve().parent / "results"
        self._create_and_validate_directory(
            self.results_dir,
            "Results directory",
            check_execute=False
        )
        
        # Models directory for preprocessing outputs
        self.models_dir = Path(__file__).resolve().parent / "models"
        self._create_and_validate_directory(
            self.models_dir,
            "Models directory",
            check_execute=False
        )
        
        # Datasets directory
        self.datasets_dir = Path(__file__).resolve().parent / "datasets"
        self._create_and_validate_directory(
            self.datasets_dir,
            "Datasets directory",
            check_execute=False
        )

    def _confirm_directory_creation(self) -> None:
        """Ask user to confirm directory creation with custom path option."""
        try:
            self.print_color("\nDirectory Setup Configuration:", 'info')
            self.print_color("  ├─ config/     - Configuration files", 'info')
            self.print_color("  ├─ logs/       - Log files", 'info')
            self.print_color("  ├─ history/    - Test history", 'info')
            self.print_color("  ├─ results/    - Test results and outputs", 'info')
            self.print_color("  ├─ models/     - Preprocessing models and artifacts", 'info')
            self.print_color("  └─ datasets/   - CSV datasets", 'info')
            
            # Ask if user wants custom paths
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    response = input(Fore.YELLOW + Style.BRIGHT + "\nUse custom directory paths? (y/N): " + Style.RESET_ALL).strip().lower()
                    
                    if response in ['y', 'yes']:
                        self._setup_custom_directories()
                        return
                    elif response in ['n', 'no', '']:
                        self.print_color("\nUsing default directory paths.", 'success')
                        return
                    else:
                        if attempt < max_attempts - 1:
                            self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                
                except (EOFError, KeyboardInterrupt):
                    self.print_color("\nUsing default directory paths.", 'success')
                    return
                except Exception as input_error:
                    if attempt < max_attempts - 1:
                        self.print_color(f"\nInput error: {str(input_error)} - please try again.", 'error')
            
            self.print_color("\nUsing default directory paths.", 'success')
        
        except KeyboardInterrupt:
            self.print_color("\nDirectory setup interrupted. Using default paths.", 'warning')
        except Exception as e:
            self.print_color(f"\nError in directory confirmation: {str(e)} - Using default paths.", 'error')

    def _setup_custom_directories(self) -> None:
        """Setup custom directory paths based on user input."""
        try:
            self.print_color("\nCustom Directory Configuration:", 'highlight')
            self.print_color("  └─ Enter custom paths (press Enter to use default):", 'info')
            
            # Get custom paths with validation
            base_path = Path(__file__).resolve().parent
            
            # Config directory
            config_input = input(Fore.YELLOW + Style.BRIGHT + f"\nConfig directory [{base_path / 'config'}]: " + Style.RESET_ALL).strip()
            if config_input:
                self.config_dir = Path(config_input).resolve()
            else:
                self.config_dir = base_path / "config"
            
            # Log directory
            log_input = input(Fore.YELLOW + Style.BRIGHT + f"\nLog directory [{base_path / 'logs'}]: " + Style.RESET_ALL).strip()
            if log_input:
                self.log_dir = Path(log_input).resolve()
            else:
                self.log_dir = base_path / "logs"
            
            # History directory
            history_input = input(Fore.YELLOW + Style.BRIGHT + f"\nHistory directory [{base_path / 'history'}]: " + Style.RESET_ALL).strip()
            if history_input:
                self.history_dir = Path(history_input).resolve()
            else:
                self.history_dir = base_path / "history"
            
            # Results directory
            results_input = input(Fore.YELLOW + Style.BRIGHT + f"\nResults directory [{base_path / 'results'}]: " + Style.RESET_ALL).strip()
            if results_input:
                self.results_dir = Path(results_input).resolve()
            else:
                self.results_dir = base_path / "results"
            
            # Models directory
            models_input = input(Fore.YELLOW + Style.BRIGHT + f"\nModels directory [{base_path / 'models'}]: " + Style.RESET_ALL).strip()
            if models_input:
                self.models_dir = Path(models_input).resolve()
            else:
                self.models_dir = base_path / "models"
            
            # Datasets directory
            datasets_input = input(Fore.YELLOW + Style.BRIGHT + f"\nDatasets directory [{base_path / 'datasets'}]: " + Style.RESET_ALL).strip()
            if datasets_input:
                self.datasets_dir = Path(datasets_input).resolve()
            else:
                self.datasets_dir = base_path / "datasets"
            
            # Confirm choices
            self.print_color("\nPaths configured:", 'success')
            self.print_color(f"  ├─ Config: {self.config_dir}", 'debug')
            self.print_color(f"  ├─ Logs: {self.log_dir}", 'debug')
            self.print_color(f"  ├─ History: {self.history_dir}", 'debug')
            self.print_color(f"  ├─ Results: {self.results_dir}", 'debug')
            self.print_color(f"  ├─ Models: {self.models_dir}", 'debug')
            self.print_color(f"  └─ Datasets: {self.datasets_dir}", 'debug')
            
            # Ask for confirmation
            confirm = input(Fore.YELLOW + Style.BRIGHT + "\nConfirm these paths? (Y/n): " + Style.RESET_ALL).strip().lower()
            if confirm in ['n', 'no']:
                self.print_color("\nReverting to default paths.", 'warning')
                self.config_dir = base_path / "config"
                self.log_dir = base_path / "logs"
                self.history_dir = base_path / "history"
                self.results_dir = base_path / "results"
                self.models_dir = base_path / "models"
                self.datasets_dir = base_path / "datasets"
        
        except KeyboardInterrupt:
            self.print_color("\nCustom directory setup cancelled. Using default paths.", 'warning')
            base_path = Path(__file__).resolve().parent
            self.config_dir = base_path / "config"
            self.log_dir = base_path / "logs"
            self.history_dir = base_path / "history"
            self.results_dir = base_path / "results"
            self.models_dir = base_path / "models"
            self.datasets_dir = base_path / "datasets"
        except Exception as e:
            self.print_color(f"\nError in custom directory setup: {str(e)} - Using default paths.", 'error')
            base_path = Path(__file__).resolve().parent
            self.config_dir = base_path / "config"
            self.log_dir = base_path / "logs"
            self.history_dir = base_path / "history"
            self.results_dir = base_path / "results"
            self.models_dir = base_path / "models"
            self.datasets_dir = base_path / "datasets"

    def _create_and_validate_directory(
        self,
        directory: Path,
        name: str,
        check_execute: bool = True
    ) -> None:
        """Helper method to create and validate a directory with user interaction."""
        try:
            # Check if directory already exists
            if directory.exists():
                if not directory.is_dir():
                    self.print_color(f"\n{name} path exists but is not a directory: {directory}", 'warning')
                    
                    # Ask user what to do
                    response = input(Fore.YELLOW + Style.BRIGHT + f"\nRemove existing file and create directory? (Y/n/skip): " + Style.RESET_ALL).strip().lower()
                    
                    if response in ['y', 'yes', '']:
                        directory.unlink()  # Remove file
                        directory.mkdir(parents=True, exist_ok=True)
                        self.print_color(f"\nRemoved file and created directory: {directory}", 'success')
                    elif response in ['s', 'skip']:
                        raise NotADirectoryError(f"\n{name} is a file, not a directory")
                    else:
                        self.print_color("\nCannot proceed without valid directory. Exiting.", 'error')
                        sys.exit(1)
                else:
                    # Directory exists, check permissions
                    self._check_directory_permissions(directory, name, check_execute)
            else:
                # Directory doesn't exist, create it
                self._create_new_directory(directory, name, check_execute)
            
            # Log directory info
            stat = directory.stat()
            self.print_color(f"\n{name} initialized: {Fore.CYAN + Style.BRIGHT}{directory}", 'system')
            self.print_color(f"  ├─ Size: {Fore.CYAN + Style.BRIGHT}{stat.st_size / (1024 ** 2):.2f} MB", 'system')
            self.print_color(f"  ├─ Created: {Fore.CYAN + Style.BRIGHT}{datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}", 'system')
            self.print_color(f"  ├─ Modified: {Fore.CYAN + Style.BRIGHT}{datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}", 'system')
            self.print_color(f"  └─ Permissions: {Fore.CYAN + Style.BRIGHT}{oct(stat.st_mode)}", 'system')
        
        except Exception as e:
            self.print_color(f"\nError initializing {name.lower()}: {str(e)}", 'error')
            
            # Ask user if they want to try alternative
            response = input(Fore.YELLOW + Style.BRIGHT + f"\nTry alternative location for {name}? (Y/n): " + Style.RESET_ALL).strip().lower()
            
            if response in ['y', 'yes', '']:
                alt_path = input(Fore.YELLOW + Style.BRIGHT + f"\nEnter alternative path for {name}: " + Style.RESET_ALL).strip()
                if alt_path:
                    alt_directory = Path(alt_path).resolve()
                    self._create_and_validate_directory(alt_directory, f"Alternative {name}", check_execute)
                else:
                    raise
            else:
                raise

    def _check_directory_permissions(self, directory: Path, name: str, check_execute: bool) -> None:
        """Check directory permissions and handle issues interactively."""
        required_perms = os.R_OK | os.W_OK
        if check_execute:
            required_perms |= os.X_OK
        
        if not os.access(directory, required_perms):
            self.print_color(f"\nInsufficient permissions for {name.lower()}: {directory}", 'warning')
            self.print_color(f"Required permissions: {oct(required_perms)}", 'warning')
            
            # Try to fix permissions
            response = input(Fore.YELLOW + Style.BRIGHT + "\nAttempt to fix permissions? (Y/n/skip): " + Style.RESET_ALL).strip().lower()
            
            if response in ['y', 'yes', '']:
                try:
                    # Try to change permissions (may require admin/sudo)
                    os.chmod(directory, required_perms)
                    self.print_color(f"\nPermissions updated for {directory}", 'success')
                except Exception as perm_error:
                    self.print_color(f"\nCould not fix permissions: {str(perm_error)}", 'error')
                    
                    # Offer alternative
                    response = input(Fore.YELLOW + Style.BRIGHT + "\nTry alternative directory with proper permissions? (Y/n): " + Style.RESET_ALL).strip().lower()
                    
                    if response in ['y', 'yes', '']:
                        raise PermissionError(f"\nInsufficient permissions for {name.lower()}: {directory}")
                    else:
                        self.print_color("\nCannot proceed without proper permissions. Exiting.", 'error')
                        sys.exit(1)
            elif response in ['s', 'skip']:
                self.print_color(f"\nSkipping permission check for {name}. May cause issues later.", 'warning')
            else:
                self.print_color("\nCannot proceed without proper permissions. Exiting.", 'error')
                sys.exit(1)

    def _create_new_directory(self, directory: Path, name: str, check_execute: bool) -> None:
        """Create new directory with user confirmation."""
        self.print_color(f"\n{name} does not exist: {directory}", 'info')
        
        response = input(Fore.YELLOW + Style.BRIGHT + "\nCreate directory? (Y/n): " + Style.RESET_ALL).strip().lower()
        
        if response in ['y', 'yes', '']:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.print_color(f"\n{name} created successfully: {directory}", 'success')
                
                # Set default permissions
                required_perms = os.R_OK | os.W_OK
                if check_execute:
                    required_perms |= os.X_OK
                
                try:
                    os.chmod(directory, required_perms)
                except Exception:
                    pass  # Non-critical if we can't set permissions
            
            except Exception as mkdir_error:
                self.print_color(f"\nFailed to create {name.lower()}: {str(mkdir_error)}", 'error')
                
                # Try parent directory
                parent = directory.parent
                if not parent.exists():
                    response = input(Fore.YELLOW + Style.BRIGHT + f"\nCreate parent directory {parent}? (Y/n): " + Style.RESET_ALL).strip().lower()
                    
                    if response in ['y', 'yes', '']:
                        parent.mkdir(parents=True, exist_ok=True)
                        directory.mkdir(exist_ok=True)
                        self.print_color(f"\nCreated directory hierarchy: {directory}", 'success')
                    else:
                        raise
                else:
                    raise
        else:
            self.print_color(f"\nCannot proceed without {name.lower()}. Exiting.", 'error')
            sys.exit(1)

    def _initialize_files(self) -> None:
        """Initialize required files with default content if needed."""

        self.print_color("\nInitializing required files...", 'warning')
        
        # Show progress
        with console.status("[bold cyan]Setting up configuration files...[/bold cyan]", spinner="dots"):
            time.sleep(0.5)
            
            # Initialize history file
            self._initialize_file(
                self.history_file,
                [],
                "History file"
            )
            
            # Initialize config file
            default_config = {
                'version': self.VERSION,
                'default_output_dir': str(self.results_dir),
                'default_results_dir': str(self.results_dir),
                'default_target_mb': 256,
                'default_min_chunk': self.min_chunk_size,
                'recent_files': [],
                'max_history_entries': self.max_history_entries,
                'last_updated': datetime.now().isoformat(),
                'preprocessing_config': {
                    #'default_output_dir': str(self.models_dir),
                    'default_output_dir': str(self.results_dir),
                    'default_input_dir': str(self.datasets_dir),
                    'default_results_dir': str(self.results_dir),
                    'memory_safety_factor': MEMORY_SAFETY_FACTOR,
                    'hybrid_feature_count': HYBRID_FEATURE_COUNT,
                    'scaler_range': SCALER_RANGE
                }
            }
            self._initialize_file(
                self.config_file,
                default_config,
                "Config file"
            )
            
            # Initialize log file
            self._initialize_file(
                self.log_file,
                f"Memory-Aware Chunk Tester Log - {datetime.now().isoformat()}\n",
                "Log file",
                is_json=False
            )

    def _initialize_file(
        self,
        file_path: Path,
        default_content: Any,
        name: str,
        is_json: bool = True
    ) -> None:
        """Helper method to initialize a file with default content with user interaction."""
        try:
            if not file_path.exists():
                self.print_color(f"\n{name} does not exist: {file_path}", 'info')
                
                response = input(Fore.YELLOW + Style.BRIGHT + f"\nCreate {name}? (Y/n): " + Style.RESET_ALL).strip().lower()
                
                if response in ['y', 'yes', '']:
                    with open(file_path, 'w') as f:
                        if is_json:
                            json.dump(default_content, f, indent=2)
                        else:
                            f.write(str(default_content))
                    self.print_color(f"\n{name} created: {file_path}", 'success')
                else:
                    self.print_color(f"\nCannot proceed without {name}. Exiting.", 'error')
                    sys.exit(1)
            else:
                # File exists, check if it needs repair
                try:
                    if is_json:
                        with open(file_path, 'r') as f:
                            content = json.load(f)
                    else:
                        with open(file_path, 'r') as f:
                            content = f.read()
                    
                    # Validate file is not empty except for history file
                    if is_json and not content and name != "History file":
                        raise ValueError(f"\n{name} is empty")
                
                except Exception as read_error:
                    self.print_color(f"\n{name} appears corrupted or empty: {str(read_error)}", 'warning')
                    
                    response = input(Fore.YELLOW + Style.BRIGHT + f"\nRepair {name} with default content? (Y/n/backup): " + Style.RESET_ALL).strip().lower()
                    
                    if response in ['y', 'yes', '']:
                        # Create backup first
                        backup_path = file_path.with_suffix(f'.backup_{int(time.time())}')
                        try:
                            shutil.copy2(file_path, backup_path)
                            self.print_color(f"\nBackup created: {backup_path}", 'info')
                        except Exception:
                            pass
                        
                        # Write new content
                        with open(file_path, 'w') as f:
                            if is_json:
                                json.dump(default_content, f, indent=2)
                            else:
                                f.write(str(default_content))
                        self.print_color(f"\n{name} repaired: {file_path}", 'success')
                    
                    elif response == 'backup':
                        # Just create backup
                        backup_path = file_path.with_suffix(f'.backup_{int(time.time())}')
                        try:
                            shutil.copy2(file_path, backup_path)
                            self.print_color(f"\nBackup created: {backup_path}", 'success')
                        except Exception as backup_error:
                            self.print_color(f"\nFailed to create backup: {str(backup_error)}", 'error')
                    
                    else:
                        self.print_color(f"\nUsing existing {name} as-is. May cause issues.", 'warning')
            
            # Validate file permissions
            if not os.access(file_path, os.R_OK | os.W_OK):
                self.print_color(f"\nInsufficient permissions for {name.lower()}: {file_path}", 'warning')
                
                response = input(Fore.YELLOW + Style.BRIGHT + f"\nAttempt to fix {name} permissions? (Y/n): " + Style.RESET_ALL).strip().lower()
                
                if response in ['y', 'yes', '']:
                    try:
                        os.chmod(file_path, os.R_OK | os.W_OK)
                        self.print_color(f"\nPermissions updated for {file_path}", 'success')
                    except Exception as perm_error:
                        self.print_color(f"\nCould not fix permissions: {str(perm_error)}", 'error')
                else:
                    self.print_color(f"\nProceeding with limited permissions for {name}.", 'warning')
            
            # Log file info
            stat = file_path.stat()
            self.print_color(f"\n{name} initialized: {Fore.BLUE + Style.BRIGHT}{file_path}", 'info')
            self.print_color(f"  ├─ Size: {Fore.BLUE + Style.BRIGHT}{stat.st_size / 1024:.2f} KB", 'info')
            self.print_color(f"  ├─ Created: {Fore.BLUE + Style.BRIGHT}{datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}", 'info')
            self.print_color(f"  ├─ Modified: {Fore.BLUE + Style.BRIGHT}{datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}", 'info')
            self.print_color(f"  └─ Permissions: {Fore.BLUE + Style.BRIGHT}{oct(stat.st_mode)}", 'info')
        
        except KeyboardInterrupt:
            self.print_color(f"\n{name} initialization interrupted.", 'warning')
            raise
        except Exception as e:
            self.print_color(f"\nError initializing {name.lower()}: {str(e)}", 'error')
            raise

    def _validate_config(self) -> None:
        """Validate the loaded configuration with user interaction for fixes."""
        try:
            self.print_color("\nValidating configuration...", 'warning')
            
            # Ensure required keys exist
            required_keys = [
                'default_output_dir',
                'default_results_dir',
                'default_target_mb',
                'default_min_chunk',
                'recent_files',
                'max_history_entries',
                'preprocessing_config'
            ]
            
            missing_keys = [key for key in required_keys if key not in self.config]
            
            if missing_keys:
                self.print_color(f"\nMissing required config keys: {missing_keys}", 'warning')
                
                response = input(Fore.YELLOW + Style.BRIGHT + "\nFix missing configuration keys? (Y/n): " + Style.RESET_ALL).strip().lower()
                
                if response in ['y', 'yes', '']:
                    for key in missing_keys:
                        if key == 'default_output_dir':
                            self.config[key] = str(self.results_dir)
                        elif key == 'default_results_dir':
                            self.config[key] = str(self.results_dir)
                        elif key == 'default_target_mb':
                            self.config[key] = 256
                        elif key == 'default_min_chunk':
                            self.config[key] = self.min_chunk_size
                        elif key == 'recent_files':
                            self.config[key] = []
                        elif key == 'max_history_entries':
                            self.config[key] = self.max_history_entries
                        elif key == 'preprocessing_config':
                            self.config[key] = {
                                #'default_output_dir': str(self.models_dir),
                                'default_output_dir': str(self.results_dir),
                                'default_input_dir': str(self.datasets_dir),
                                'default_results_dir': str(self.results_dir),
                                'memory_safety_factor': MEMORY_SAFETY_FACTOR,
                                'hybrid_feature_count': HYBRID_FEATURE_COUNT,
                                'scaler_range': SCALER_RANGE
                            }
                    
                    self.save_config(self.config)
                    self.print_color("\nMissing configuration keys fixed.", 'success')
                else:
                    raise KeyError(f"\nMissing required config keys: {missing_keys}")
            
            # Validate values
            validation_errors = []
            
            if not isinstance(self.config['recent_files'], list):
                validation_errors.append("Config 'recent_files' must be a list")
            
            if self.config['max_history_entries'] <= 0:
                validation_errors.append("Config 'max_history_entries' must be positive")
            
            if validation_errors:
                self.print_color(f"\nConfiguration validation errors: {validation_errors}", 'warning')
                
                response = input(Fore.YELLOW + Style.BRIGHT + "\nFix configuration validation errors? (Y/n): " + Style.RESET_ALL).strip().lower()
                
                if response in ['y', 'yes', '']:
                    if not isinstance(self.config['recent_files'], list):
                        self.config['recent_files'] = []
                    
                    if self.config['max_history_entries'] <= 0:
                        self.config['max_history_entries'] = self.max_history_entries
                    
                    self.save_config(self.config)
                    self.print_color("\nConfiguration validation errors fixed.", 'success')
                else:
                    raise ValueError(f"\nConfiguration validation errors: {validation_errors}")
            
            self.print_color("Configuration validated successfully", 'success')
            
        except KeyboardInterrupt:
            self.print_color("\nConfiguration validation interrupted.", 'warning')
            raise
        except Exception as e:
            self.print_color(f"\nConfiguration validation failed: {str(e)}", 'error')
            raise

    def _print_init_status(self) -> None:
        """Print detailed initialization status information."""
        self.print_color("\n" + "-" * 40, 'highlight')
        self.print_color("INITIALIZATION STATUS", 'highlight')
        self.print_color("-" * 40, 'highlight')
        
        # System information
        self.print_color("System Resources:", 'highlight')
        self.print_color(f"  ├─ Total RAM: {Fore.YELLOW + Style.BRIGHT}{self.system_info['total_ram_gb']:.2f} GB", 'success')
        self.print_color(f"  ├─ Available RAM: {Fore.YELLOW + Style.BRIGHT}{self.system_info['available_ram_gb']:.2f} GB", 'success')
        self.print_color(f"  ├─ CPU Cores: {Fore.YELLOW + Style.BRIGHT}{self.system_info['cpu_cores']}", 'success')
        self.print_color(f"  └─ Disk Free: {Fore.YELLOW + Style.BRIGHT}{self.system_info['disk_free_gb']:.2f} GB", 'success')
        
        # Configuration
        self.print_color("Active Configuration:", 'highlight')
        for key, value in self.config.items():
            prefix = "  └─ " if key == list(self.config.keys())[-1] else "  ├─ "
            if key != 'recent_files':  # Skip potentially long list
                if key.endswith('_mb'):
                    self.print_color(f"{prefix}{key.replace('_', ' ').title():<20}: {Fore.CYAN + Style.BRIGHT}{value:.1f} MB", 'success')
                elif key == 'last_updated':
                    self.print_color(f"{prefix}{key.replace('_', ' ').title():<20}: {Fore.CYAN + Style.BRIGHT}{datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M:%S')}", 'success')
                elif key == 'preprocessing_config':
                    self.print_color(f"{prefix}{key.replace('_', ' ').title():<20}:", 'success')
                    for subkey, subvalue in value.items():
                        subprefix = "    └─ " if subkey == list(value.keys())[-1] else "    ├─ "
                        self.print_color(f"{subprefix}{subkey.replace('_', ' ').title():<20}: {Fore.GREEN + Style.BRIGHT}{subvalue}", 'success')
                else:
                    self.print_color(f"{prefix}{key.replace('_', ' ').title():<20}: {Fore.YELLOW + Style.BRIGHT}{value}", 'success')
        
        # Performance settings
        self.print_color("Performance Settings:", 'highlight')
        self.print_color(f"  ├─ Safety Factor: {Fore.YELLOW + Style.BRIGHT}{self.safety_factor}", 'success')
        self.print_color(f"  ├─ Min Chunk Size: {Fore.YELLOW + Style.BRIGHT}{self.min_chunk_size}", 'success')
        self.print_color(f"  └─ Max RAM Usage: {Fore.YELLOW + Style.BRIGHT}{self.max_ram_usage}", 'success')
        
        # Thresholds
        self.print_color("Performance Thresholds:", 'highlight')
        for key, value in self.performance_thresholds.items():
            prefix = "  └─ " if key == list(self.performance_thresholds.keys())[-1] else "  ├─ "
            self.print_color(f"{prefix}{key.replace('_', ' ').title():<20}: {Fore.YELLOW + Style.BRIGHT}{value}", 'success')

    def print_color(self, message: str, level: str = 'info', end: str = '\n') -> None:
        """Print a colored message to the console.
        
        Args:
            message: The message to print
            level: The message level (info, success, warning, error, highlight, system)
            end: Ending character (like in print())
        """
        color = self.color_map.get(level, Fore.WHITE)
        print(f"{color}{message}{Style.RESET_ALL}", end=end)

    def _log_event(self, message: str, level: str) -> None:
        """Log an event to the log file.
        
        Args:
            message: The message to log
            level: The log level (info, warning, error, etc.)
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] [{level.upper()}] {message}\n"
            
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        
        except Exception as e:
            self.print_color(f"\nFailed to log event: {str(e)}", 'error')

    def load_config(self) -> Dict[str, Any]:
        """Load the configuration from file with interactive repair options.
        
        Returns:
            The loaded configuration dictionary
            
        Raises:
            ValueError: If the config file is invalid
            IOError: If there are issues reading the file
        """
        default_config = {
            'version': self.VERSION,
            'default_output_dir': 'results',
            'default_results_dir': 'results',
            'default_target_mb': 256,
            'default_min_chunk': 1000,
            'recent_files': [],
            'max_history_entries': 10,
            'preprocessing_config': {
                #'default_output_dir': 'models',
                'default_output_dir': 'results',
                'default_input_dir': 'datasets',
                'default_results_dir': 'results',
                'memory_safety_factor': MEMORY_SAFETY_FACTOR,
                'hybrid_feature_count': HYBRID_FEATURE_COUNT,
                'scaler_range': SCALER_RANGE
            }
        }
        
        try:
            if not self.config_file.exists():
                self.print_color("\nNo config file found - creating with defaults", 'warning')
                
                # Show default configuration
                self.print_color("\nDefault configuration:", 'info')
                for key, value in default_config.items():
                    prefix = "  └─ " if key == list(default_config.keys())[-1] else "  ├─ "
                    if key != 'recent_files' and key != 'preprocessing_config':
                        self.print_color(f"{prefix}{key}: {value}", 'debug')
                
                response = input(Fore.YELLOW + Style.BRIGHT + "\nCreate configuration file with these defaults? (Y/n): " + Style.RESET_ALL).strip().lower()
                
                if response in ['y', 'yes', '']:
                    self.save_config(default_config)
                    return default_config
                else:
                    self.print_color("\nCannot proceed without configuration. Exiting.", 'error')
                    sys.exit(1)
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
                # Validate and repair config
                repaired = False
                repair_needed = []
                
                for key in default_config:
                    if key not in config:
                        repair_needed.append(f"Missing key: {key}")
                        config[key] = default_config[key]
                        repaired = True
                
                # Validate numerical values
                for key in ['default_target_mb', 'default_min_chunk', 'max_history_entries']:
                    if not isinstance(config[key], int) or config[key] <= 0:
                        repair_needed.append(f"Invalid value for {key}: {config[key]}")
                        config[key] = default_config[key]
                        repaired = True
                
                if repaired:
                    self.print_color(f"\nConfig file needs repair: {', '.join(repair_needed)}", 'warning')
                    
                    response = input(Fore.YELLOW + Style.BRIGHT + "\nRepair configuration file? (Y/n/backup): " + Style.RESET_ALL).strip().lower()
                    
                    if response in ['y', 'yes', '']:
                        self.save_config(config)
                        self.print_color("\nConfig file repaired successfully", 'success')
                    elif response == 'backup':
                        # Create backup before repair
                        backup_path = self.config_file.with_suffix(f'.backup_{int(time.time())}')
                        try:
                            shutil.copy2(self.config_file, backup_path)
                            self.print_color(f"\nBackup created: {backup_path}", 'success')
                            self.save_config(config)
                            self.print_color("\nConfig file repaired with backup", 'success')
                        except Exception as backup_error:
                            self.print_color(f"\nCould not create backup: {str(backup_error)}", 'error')
                            self.save_config(config)
                            self.print_color("\nConfig file repaired without backup", 'success')
                    else:
                        self.print_color("\nUsing unrepaired configuration. May cause issues.", 'warning')
                
                return config
        
        except json.JSONDecodeError as e:
            self.print_color(f"\nConfig file is corrupted: {str(e)}", 'warning')
            
            response = input(Fore.YELLOW + Style.BRIGHT + "\nRecreate config file with defaults? (Y/n/backup): " + Style.RESET_ALL).strip().lower()
            
            if response in ['y', 'yes', '']:
                self.save_config(default_config)
                self.print_color("\nConfig file recreated with defaults", 'success')
                return default_config
            elif response == 'backup':
                # Create backup of corrupted file
                backup_path = self.config_file.with_suffix(f'.corrupted_{int(time.time())}')
                try:
                    shutil.copy2(self.config_file, backup_path)
                    self.print_color(f"\nCorrupted config backed up to: {backup_path}", 'success')
                except Exception:
                    pass
                
                self.save_config(default_config)
                self.print_color("\nConfig file recreated with defaults", 'success')
                return default_config
            else:
                self.print_color("\nCannot proceed with corrupted config. Exiting.", 'error')
                sys.exit(1)
        
        except Exception as e:
            self.print_color(f"\nError loading config: {str(e)} - using defaults", 'error')
            return default_config

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file with error handling and validation."""
        try:
            # Basic validation before saving
            required_keys = ['default_output_dir', 'default_results_dir', 'default_target_mb', 'default_min_chunk', 'recent_files']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                self.print_color(f"\nMissing required config keys: {missing_keys}", 'warning')
                
                # Try to fix missing keys
                for key in missing_keys:
                    if key == 'default_output_dir':
                        config[key] = str(self.results_dir)
                    elif key == 'default_results_dir':
                        config[key] = str(self.results_dir)
                    elif key == 'default_target_mb':
                        config[key] = 256
                    elif key == 'default_min_chunk':
                        config[key] = 1000
                    elif key == 'recent_files':
                        config[key] = []
            
            # Ensure numerical values are valid
            if not isinstance(config['default_target_mb'], int) or config['default_target_mb'] <= 0:
                config['default_target_mb'] = 256
                self.print_color("\nFixed invalid default_target_mb value", 'warning')
            
            if not isinstance(config['default_min_chunk'], int) or config['default_min_chunk'] <= 0:
                config['default_min_chunk'] = 1000
                self.print_color("\nFixed invalid default_min_chunk value", 'warning')
            
            # Add metadata
            config['version'] = self.VERSION
            config['last_updated'] = datetime.now().isoformat()
            
            # Show summary before saving
            self.print_color("\nSaving configuration:", 'info')
            self.config_keys = ['default_output_dir', 'default_results_dir', 'default_target_mb', 'default_min_chunk', 'max_history_entries']
            for i, key in enumerate(self.config_keys):
                if key in config:
                    prefix = "  └─ " if i == len(self.config_keys) - 1 else "  ├─ "
                    self.print_color(f"{prefix}{key.replace('_', ' ').title():<20}: {Fore.GREEN + Style.BRIGHT}{config[key]}", 'info')
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.print_color(f"\nConfiguration saved to {Fore.CYAN + Style.BRIGHT}{self.config_file}", 'success')
        
        except Exception as e:
            self.print_color(f"\nError saving config: {str(e)}", 'error')
            
            # Try to save to alternative location
            response = input(Fore.YELLOW + Style.BRIGHT + "\nSave configuration to alternative location? (Y/n): " + Style.RESET_ALL).strip().lower()
            
            if response in ['y', 'yes', '']:
                alt_path = input(Fore.YELLOW + Style.BRIGHT + "\nEnter alternative path: " + Style.RESET_ALL).strip()
                if alt_path:
                    try:
                        alt_file = Path(alt_path).resolve()
                        with open(alt_file, 'w') as f:
                            json.dump(config, f, indent=2)
                        self.print_color(f"\nConfiguration saved to alternative location: {alt_file}", 'success')
                        self.config_file = alt_file
                    except Exception as alt_error:
                        self.print_color(f"\nFailed to save to alternative location: {str(alt_error)}", 'error')
                        raise
                else:
                    raise
            else:
                raise

    def update_history(self, summary: Dict[str, Any]) -> None:
        """Update test history with new summary, maintaining size limits with atomic writes."""
        max_retries = 3
        retry_delay = 0.1
        
        try:
            history = []
            if self.history_file.exists():
                for attempt in range(max_retries):
                    try:
                        with open(self.history_file, 'r') as f:
                            content = f.read().strip()
                            if not content:
                                history = []
                            else:
                                history = json.loads(content)
                        
                        if not isinstance(history, list):
                            raise ValueError("History is not a list")
                        break
                    
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            self._log_event(f"JSON decode error in history file (attempt {attempt + 1}): {str(e)}", "warning")
                            time.sleep(retry_delay)
                        else:
                            self.print_color(f"\nHistory file corrupted: {str(e)}", 'warning')
                            
                            response = input(Fore.YELLOW + Style.BRIGHT + "\nReset history file? (Y/n/backup): " + Style.RESET_ALL).strip().lower()
                            
                            if response in ['y', 'yes', '']:
                                history = []
                                self.print_color("\nHistory file reset", 'success')
                            elif response == 'backup':
                                backup_path = self.history_file.with_suffix(f'.corrupted_{int(time.time())}')
                                try:
                                    shutil.copy2(self.history_file, backup_path)
                                    self.print_color(f"\nCorrupted history backed up to: {backup_path}", 'success')
                                except Exception:
                                    pass
                                history = []
                                self.print_color("\nHistory file reset with backup", 'success')
                            else:
                                self.print_color("\nCannot proceed with corrupted history. Skipping history update.", 'error')
                                return
                            break
                    
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self._log_event(f"Error reading history (attempt {attempt + 1}): {str(e)}", "warning")
                            time.sleep(retry_delay)
                        else:
                            self.print_color(f"\nError reading history: {str(e)}", 'error')
                            return
            
            required_fields = ['timestamp', 'dataset_info', 'performance_metrics']
            missing_fields = [field for field in required_fields if field not in summary]
            
            if missing_fields:
                self.print_color(f"\nMissing required summary fields: {missing_fields}", 'warning')
                
                for field in missing_fields:
                    if field == 'timestamp':
                        summary[field] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elif field == 'dataset_info':
                        summary[field] = {'error': 'Missing dataset info'}
                    elif field == 'performance_metrics':
                        summary[field] = {'error': 'Missing performance metrics'}
                
                self.print_color("\nMissing fields populated with defaults", 'info')
            
            max_entries = self.config.get('max_history_entries', 10)
            
            if len(history) >= max_entries:
                self.print_color(f"\nHistory file at capacity ({len(history)}/{max_entries} entries)", 'info')
                
                if history:
                    oldest = history[0]
                    oldest_time = oldest.get('timestamp', 'Unknown')
                    self.print_color(f"\nOldest entry: {oldest_time}", 'debug')
                
                response = input(Fore.YELLOW + Style.BRIGHT + "\nRemove oldest entry to make room? (Y/n/skip): " + Style.RESET_ALL).strip().lower()
                
                if response in ['y', 'yes', '']:
                    history = history[1:]
                    self.print_color("\nOldest entry removed", 'success')
                elif response == 'skip':
                    self.print_color("\nSkipping history update - file at capacity", 'warning')
                    return
            
            history.append(summary)
            history = history[-max_entries:]
            
            self.print_color(f"\nHistory file overview:", 'success')
            self.print_color(f"  ├─ New entry timestamp: {Fore.CYAN + Style.BRIGHT}{summary.get('timestamp', 'Unknown')}", 'success')
            self.print_color(f"  ├─ Dataset file: {Fore.MAGENTA + Style.BRIGHT}{summary.get('dataset_info', {}).get('filepath', 'Unknown')}", 'success')
            self.print_color(f"  ├─ History file: {Fore.MAGENTA + Style.BRIGHT}{self.history_file}", 'success')
            self.print_color(f"  └─ Total history entries: {Fore.YELLOW + Style.BRIGHT}{len(history)}/{max_entries}", 'success')
            
            for attempt in range(max_retries):
                try:
                    temp_file = self.history_file.with_suffix('.tmp')
                    with open(temp_file, 'w') as f:
                        json.dump(history, f, indent=2)
                    
                    temp_file.replace(self.history_file)
                    self._log_event(f"History updated successfully with {len(history)} entries", "info")
                    break
                    
                except Exception as write_error:
                    if attempt < max_retries - 1:
                        self._log_event(f"Error writing history (attempt {attempt + 1}): {str(write_error)}", "warning")
                        time.sleep(retry_delay)
                    else:
                        self.print_color(f"\nFailed to save history after {max_retries} attempts: {str(write_error)}", 'error')
                        self._log_event(f"Failed to save history: {str(write_error)}", "error")
        
        except KeyboardInterrupt:
            self.print_color("\nHistory update interrupted by user.", 'warning')
            self._log_event("History update interrupted by user", "warning")
        except Exception as e:
            self.print_color(f"\nError updating history: {str(e)}", 'error')
            self._log_event(f"History update error: {str(e)}", "error")

    def get_total_rows(self, filepath: str) -> int:
        """Count total rows efficiently without loading.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Number of rows in the file (excluding header)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return sum(1 for line in f) - 1  # Subtract header
        except FileNotFoundError:
            raise FileNotFoundError(f"\nFile not found: {filepath}")
        except Exception as e:
            raise IOError(f"\nError reading file {filepath}: {str(e)}")

    def get_available_datasets(self, dataset_dir: Path = None) -> List[str]:
        """List available CSV datasets in the specified directory.
        
        Args:
            dataset_dir: Directory to search for CSV files
            
        Returns:
            List of CSV filenames sorted alphabetically
        """
        try:
            if dataset_dir is None:
                dataset_dir = Path(__file__).resolve().parent / "datasets"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
            return sorted(datasets)
        except FileNotFoundError:
            self.print_color(f"\nDataset directory not found: {dataset_dir}", 'error')
            return []
        except Exception as e:
            self.print_color(f"\nError listing datasets: {str(e)}", 'error')
            return []

    def load_feature_descriptions(self, dataset_path: str, interactive: bool = True) -> Optional[Dict[str, str]]:
        """
        Auto-detect and load feature descriptions from companion file.
        
        Looks for:
        - Same directory as dataset
        - Naming patterns: *_Features.csv, *_features.csv, features.csv
        - Structure: Feature,Description columns
        
        Args:
            dataset_path: Path to the main dataset CSV
            interactive: If True, allows user to select from multiple matches
        
        Returns:
            Dictionary mapping feature names to descriptions, or None if not found
        """
        try:
            dataset_file = Path(dataset_path)
            dataset_dir = dataset_file.parent
            dataset_stem = dataset_file.stem  # e.g., "NF-CSE-CIC-IDS2018"
            
            # Search patterns (in priority order)
            search_patterns = [
                f"{dataset_stem}_Features.csv",
                f"{dataset_stem}_features.csv",
                f"features.csv",
                f"Features.csv",
                f"{dataset_stem}_meta.csv",
                f"{dataset_stem}_metadata.csv"
            ]
            
            # Find all matching files
            matching_files = []
            for pattern in search_patterns:
                candidate = dataset_dir / pattern
                if candidate.exists():
                    matching_files.append(candidate)
            
            # Also search for any CSV files containing "feature" in the name
            for csv_file in dataset_dir.glob("*.csv"):
                if "feature" in csv_file.name.lower() and csv_file not in matching_files:
                    matching_files.append(csv_file)
            
            if not matching_files:
                self.print_color(f"\nNo feature description files found in {dataset_dir}", 'warning')
                return None
            
            feature_desc_file = None
            
            # If only one match, use it automatically
            if len(matching_files) == 1:
                feature_desc_file = matching_files[0]
                self.print_color(f"\nFound feature descriptions: {Fore.MAGENTA + Style.BRIGHT}{feature_desc_file.name}", 'success')
            
            # If multiple matches and interactive mode, let user choose
            elif len(matching_files) > 1 and interactive:
                self.print_color(f"\nMultiple feature description files found:", 'warning')
                for i, file in enumerate(matching_files, 1):
                    file_size = file.stat().st_size / 1024  # KB
                    self.print_color(f"{i}. {file.name} ({file_size:.1f} KB)", 'info')
                
                self.print_color(f"{len(matching_files) + 1}. Skip feature descriptions", 'info')
                self.print_color("0. Cancel", 'error')
                
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        choice = input(Fore.YELLOW + Style.BRIGHT + f"\nSelect feature description file (0-{len(matching_files) + 1}): " + Style.RESET_ALL).strip()
                        
                        if choice == '0':
                            self.print_color("\nFeature description selection cancelled", 'warning')
                            return None
                        
                        if not choice.isdigit():
                            if attempt < max_attempts - 1:
                                self.print_color("\nPlease enter a number.", 'warning')
                                continue
                            else:
                                self.print_color("\nToo many invalid attempts. Skipping feature descriptions.", 'error')
                                return None
                        
                        choice_num = int(choice)
                        
                        if choice_num == len(matching_files) + 1:
                            self.print_color("\nSkipping feature descriptions", 'info')
                            return None
                        
                        if 1 <= choice_num <= len(matching_files):
                            feature_desc_file = matching_files[choice_num - 1]
                            self.print_color(f"\nSelected: {Fore.MAGENTA + Style.BRIGHT}{feature_desc_file.name}", 'success')
                            break
                        else:
                            if attempt < max_attempts - 1:
                                self.print_color(f"\nPlease enter a number between 0 and {len(matching_files) + 1}.", 'warning')
                            else:
                                self.print_color("\nToo many invalid attempts. Using first match.", 'error')
                                feature_desc_file = matching_files[0]
                                break
                    
                    except (EOFError, KeyboardInterrupt):
                        self.print_color("\nSelection cancelled. Using first match.", 'warning')
                        feature_desc_file = matching_files[0]
                        break
                    except Exception as input_error:
                        if attempt < max_attempts - 1:
                            self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                        else:
                            self.print_color("\nToo many errors. Using first match.", 'error')
                            feature_desc_file = matching_files[0]
                            break
            
            # If multiple matches but non-interactive, use first match
            elif len(matching_files) > 1 and not interactive:
                feature_desc_file = matching_files[0]
                self.print_color(f"\nUsing first match: {Fore.MAGENTA + Style.BRIGHT}{feature_desc_file.name}", 'success')
            
            if not feature_desc_file:
                return None
            
            # Load and parse feature descriptions
            features_df = pd.read_csv(feature_desc_file)
            
            # Validate structure
            if 'Feature' not in features_df.columns or 'Description' not in features_df.columns:
                self.print_color(f"\nInvalid format in {feature_desc_file.name} (expected: Feature,Description)", 'warning')
                
                # Show available columns to help user
                self.print_color(f"\nAvailable columns: {', '.join(features_df.columns)}", 'info')
                return None
            
            # Create mapping
            feature_map = dict(zip(features_df['Feature'], features_df['Description']))
            
            self.print_color(f"  ├─ Loaded descriptions for {Fore.YELLOW + Style.BRIGHT}{len(feature_map)} features", 'success')
            self.print_color(f"  └─ Source: {Fore.CYAN + Style.BRIGHT}{feature_desc_file.name}", 'success')
            
            return feature_map
            
        except Exception as e:
            self.print_color(f"\nError loading feature descriptions: {str(e)}", 'warning')
            return None

    def display_features_with_descriptions(
        self,
        features: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        max_display: int = 10
    ) -> None:
        """Display features with their descriptions if available."""
        
        self.print_color(f"\nFeature Overview ({len(features)} total):", 'info')
        
        display_count = min(len(features), max_display)
        
        for i, feature in enumerate(features[:display_count], 1):
            if descriptions and feature in descriptions:
                desc = descriptions[feature]
                # Truncate long descriptions
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                self.print_color(f"  {i:2d}. {Fore.CYAN + Style.BRIGHT}{feature:<30}{Style.RESET_ALL} - {desc}", 'info')
            else:
                self.print_color(f"  {i:2d}. {Fore.CYAN + Style.BRIGHT}{feature}", 'info')
        
        if len(features) > max_display:
            self.print_color(f"  ... and {len(features) - max_display} more features", 'warning')

    def calibrate_memory_usage(self, filepath: str, sample_rows: int = 10000) -> Dict[str, Any]:
        """Determine memory characteristics with safety margins.
        
        Args:
            filepath: Path to the CSV file
            sample_rows: Number of rows to sample for calibration
            
        Returns:
            Dictionary with calibration results including:
            - mem_per_row: Estimated memory per row (MB)
            - throughput: Estimated rows per second
            - status: 'success' or 'error'
        """
        try:
            # Load sample data with monitoring
            mem_before = psutil.virtual_memory().used
            start_time = time.time()
            
            df = pd.read_csv(
                filepath,
                nrows=sample_rows,
                dtype={
                    "IPV4_SRC_ADDR": "category",
                    "IPV4_DST_ADDR": "category"
                },
                engine="c",
                memory_map=True
            )
            
            mem_used = (psutil.virtual_memory().used - mem_before) / (1024 ** 2)
            time_used = time.time() - start_time
            
            return {
                'mem_per_row': (mem_used / sample_rows) * 1.25,  # Add 25% safety margin
                'throughput': sample_rows / time_used,
                'status': 'success',
                'sample_size': sample_rows
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'sample_size': sample_rows
            }

    def calculate_dynamic_chunk_size(
        self,
        mem_per_row: float,
        target_mb: int,
        total_rows: Optional[int] = None
    ) -> int:
        """Calculate chunk size considering current system state.
        
        Args:
            mem_per_row: Memory per row in MB
            target_mb: Target memory usage per chunk in MB
            total_rows: Total rows in file (optional for bounds checking)
        
        Returns:
            Calculated chunk size in rows
        """
        available_mb = psutil.virtual_memory().available / (1024 ** 2)
        target_mb = min(target_mb, available_mb * self.max_ram_usage)
        
        calculated_size = max(
            self.min_chunk_size,
            int((target_mb * self.safety_factor) / mem_per_row)
        )
        
        # If we know total rows, don't return a size larger than remaining rows
        if total_rows is not None:
            return min(calculated_size, total_rows)
        
        return calculated_size

    def process_chunk(
        self,
        start_row: int,
        chunk_size: int,
        mem_per_row_estimate: float,
        filepath: Path = None
    ) -> Dict[str, Any]:
        """Load chunk with memory monitoring and adaptive adjustments.
        
        Args:
            start_row: Starting row number (0-based)
            chunk_size: Number of rows to attempt to load
            mem_per_row_estimate: Estimated memory per row in MB
            filepath: Path to CSV file
            
        Returns:
            Dictionary with processing results including:
            - status: 'success' or error type
            - actual_rows: Rows successfully loaded
            - ram_usage_mb: Memory used
            - load_time_sec: Time taken
            - throughput_rows_sec: Processing speed
        """
        process = psutil.Process()
        result = {
            'start_row': start_row,
            'requested_rows': chunk_size,
            'actual_rows': 0,
            'status': 'error',
            'ram_usage_mb': 0,
            'mem_per_row': 0,
            'load_time_sec': 0,
            'throughput_rows_sec': 0,
            'cpu_percent': 0
        }

        try:
            # Pre-check available memory
            available_mb = psutil.virtual_memory().available / (1024 ** 2)
            estimated_needed = chunk_size * mem_per_row_estimate
            if available_mb < estimated_needed * 1.5:
                raise MemoryError(f"\nInsufficient memory. Available: {available_mb:.2f}MB, Estimated needed: {estimated_needed:.2f}MB")

            # Load data with monitoring
            ram_start = process.memory_info().rss / (1024 ** 2)
            cpu_start = psutil.cpu_percent(interval=0.1)
            time_start = time.time()
            
            df = pd.read_csv(
                filepath,
                skiprows=range(1, start_row + 1),
                nrows=chunk_size,
                dtype={
                    "IPV4_SRC_ADDR": "category",
                    "IPV4_DST_ADDR": "category"
                },
                engine="c",
                memory_map=True
            )
            
            # Calculate metrics
            result['actual_rows'] = len(df)
            result['ram_usage_mb'] = (process.memory_info().rss / (1024 ** 2)) - ram_start
            result['load_time_sec'] = time.time() - time_start
            
            # Safely calculate mem_per_row and throughput
            if result['actual_rows'] > 0:
                result['mem_per_row'] = result['ram_usage_mb'] / result['actual_rows']
                result['throughput_rows_sec'] = result['actual_rows'] / result['load_time_sec']
            else:
                # No rows loaded - this is an error condition
                raise ValueError(f"\nNo rows loaded from chunk starting at row {start_row}")
            
            result['cpu_percent'] = psutil.cpu_percent(interval=0.1) - cpu_start
            result['status'] = 'success'
            
            # Check performance thresholds
            if result['ram_usage_mb'] > self.performance_thresholds['memory_spike'] * self.max_ram_usage * available_mb:
                self.print_color(f"\nMemory spike detected in chunk (used {result['ram_usage_mb']:.2f}MB)", 'warning')
            
            if result['cpu_percent'] > self.performance_thresholds['cpu_overload'] * 100:
                self.print_color(f"\nHigh CPU usage detected ({result['cpu_percent']:.1f}%)", 'warning')
        
        except MemoryError as e:
            result.update({
                'status': 'MemoryError',
                'message': str(e),
                'suggested_chunk_size': int(chunk_size * 0.6)  # Suggest 40% reduction
            })
        except ValueError as e:
            result.update({
                'status': 'ValueError',
                'message': str(e)
            })
        except Exception as e:
            result.update({
                'status': type(e).__name__,
                'message': str(e)
            })
        
        return result

    def generate_plots(self, all_results: List[Dict[str, Any]], output_dir: Path, filepath: Optional[Path] = None) -> Path:
        """Generate performance plots and save to output directory.
        
        Args:
            all_results: List of chunk processing results
            output_dir: Directory to save plots
        """
        try:
            if not all_results:
                self.print_color("\nNo results to generate plots from", 'warning')
                return
            
            # Create output directory if needed
            if output_dir is None:
                output_dir = Path(__file__).resolve().parent / "results"
            elif isinstance(output_dir, str):
                output_dir = Path(output_dir)
            else:
                output_dir = output_dir
            
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data
            chunk_sizes = [r['metrics']['actual_rows'] for r in all_results]
            throughputs = [r['metrics']['throughput_rows_sec'] for r in all_results]
            memory_usage = [r['metrics']['ram_usage_mb'] for r in all_results]
            mem_per_row = [r['metrics']['mem_per_row'] * 1024 for r in all_results]  # Convert to KB
            cpu_usage = [r['metrics'].get('cpu_percent', 0) for r in all_results]
            
            # Create figure with subplots
            fig = plt.figure(figsize=(14, 10))
            
            # Throughput vs Chunk Size
            plt.subplot(2, 2, 1)
            plt.plot(chunk_sizes, throughputs, 'b.-')
            plt.title('Throughput vs Chunk Size')
            plt.xlabel('Chunk Size (rows)')
            plt.ylabel('Throughput (rows/sec)')
            plt.grid(True)
            
            # Memory Usage vs Chunk Size
            plt.subplot(2, 2, 2)
            plt.plot(chunk_sizes, memory_usage, 'r.-')
            plt.title('Memory Usage vs Chunk Size')
            plt.xlabel('Chunk Size (rows)')
            plt.ylabel('Memory Usage (MB)')
            plt.grid(True)
            
            # CPU Usage vs Chunk Size
            plt.subplot(2, 2, 3)
            plt.plot(chunk_sizes, cpu_usage, 'm.-')
            plt.title('CPU Usage vs Chunk Size')
            plt.xlabel('Chunk Size (rows)')
            plt.ylabel('CPU Usage (%)')
            plt.grid(True)
            
            # Throughput vs Memory Usage
            plt.subplot(2, 2, 4)
            plt.scatter(memory_usage, throughputs, c='purple')
            plt.title('Throughput vs Memory Usage')
            plt.xlabel('Memory Usage (MB)')
            plt.ylabel('Throughput (rows/sec)')
            plt.grid(True)

            # Memory per row vs Chunk Size
            plt.figure(figsize=(7, 5))
            plt.plot(chunk_sizes, mem_per_row, 'g.-')
            plt.title('Memory per Row vs Chunk Size')
            plt.xlabel('Chunk Size (rows)')
            plt.ylabel('Memory per Row (KB)')
            plt.grid(True)

            plt.tight_layout()
            
            # Save plots
            if filepath is None:
                plot_filepath = output_dir / "preprocessing_performance_plots.png"
            else:
                plot_filepath = Path(filepath)
            
            plt.savefig(plot_filepath, dpi=150)
            plt.close(fig)

            self.plot_file = plot_filepath
            return plot_filepath
        
        except Exception as e:
            self.print_color(f"\nFailed to generate plots: {str(e)}", 'warning')
            return None
    
    def run_test(
        self,
        filepath: str,
        output_dir: Path = None,
        target_mb: int = 256,
        min_chunk: Optional[int] = None,
        run_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Main CSV testing pipeline with adaptive chunk sizing.
        
        Args:
            filepath: Path to input CSV file
            output_dir: Directory to save results (run-specific directory from interactive_menu)
            target_mb: Target memory usage per chunk in MB
            min_chunk: Minimum chunk size to test
            run_info: Run information passed from interactive_menu containing tracking data
        
        Returns:
            Dictionary with test summary, or None if test failed
        """
        # Setup experiment tracking - use provided run_info or create default
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        
        # Set minimum chunk size
        if min_chunk is not None:
            self.min_chunk_size = min_chunk
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / "results"
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize statistics tracking
        test_statistics = {
            "test_id": "",
            "run_number": 0,
            "start_time": start_time.isoformat(),
            "status": "initializing",
            "total_rows": 0,
            "processed_rows": 0,
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "memory_usage": {
                "target_mb": target_mb,
                "min_chunk_size": self.min_chunk_size,
                "safety_factor": self.safety_factor,
                "max_ram_usage": self.max_ram_usage
            },
            "performance_metrics": {
                "average_throughput": 0,
                "max_throughput": 0,
                "min_throughput": 0,
                "average_ram_usage_mb": 0,
                "max_ram_usage_mb": 0,
                "min_ram_usage_mb": 0,
                "average_cpu_percent": 0,
                "max_cpu_percent": 0,
                "total_processing_time": 0
            },
            "chunk_history": [],
            "system_info": {},
            "calibration_data": {},
            "completion_status": "",
            "end_time": "",
            "elapsed_time_seconds": 0,
            "run_info": run_info if run_info else {}
        }
        
        # Extract run information from interactive_menu or create it
        if run_info and 'run_id' in run_info:
            # Use run information provided by interactive_menu
            run_id = run_info['run_id']
            run_number = run_info.get('run_number', 0)
            run_id_full = run_info.get('run_id_full', run_id)
            timestamp = run_info.get('timestamp', timestamp)
            
            # Update statistics with run information
            test_statistics["test_id"] = run_id_full
            test_statistics["run_number"] = run_number
            test_statistics["run_info"] = run_info
            
            # Update tracker with start of actual processing
            self._update_run_tracker(output_dir.parent, run_number, 'processing_started', {
                'type': 'processing',
                'timestamp': start_time.isoformat()
            })
        else:
            # Fallback: Generate run info locally
            # Get next sequential run number for this output directory
            def get_next_run_number(output_dir: Path) -> int:
                """Get the next sequential run number for the tracking directory.
                
                Args:
                    output_dir (Path): Directory where run tracking is stored.
                
                Returns:
                    int: Next run number.
                """
                run_tracker_file = output_dir / ".preprocessor_run_tracker"
                max_retries = 3
                retry_delay = 0.1
                
                if not run_tracker_file.exists():
                    with open(run_tracker_file, 'w') as f:
                        json.dump({'last_run': 0, 'runs': {}}, f)
                    return 1
                
                for attempt in range(max_retries):
                    try:
                        with open(run_tracker_file, 'r') as f:
                            content = f.read().strip()
                            if not content:
                                tracker = {'last_run': 0, 'runs': {}}
                            else:
                                tracker = json.loads(content)
                        
                        if not isinstance(tracker, dict):
                            raise ValueError("Tracker is not a dictionary")
                        
                        next_run = tracker.get('last_run', 0) + 1
                        
                        tracker['last_run'] = next_run
                        if 'runs' not in tracker:
                            tracker['runs'] = {}
                        
                        tracker['runs'][str(next_run)] = {
                            'timestamp': timestamp,
                            'started': start_time.isoformat(),
                            'filepath': filepath,
                            'target_mb': target_mb,
                            'min_chunk': min_chunk
                        }
                        
                        temp_file = run_tracker_file.with_suffix('.tmp')
                        with open(temp_file, 'w') as f:
                            json.dump(tracker, f, indent=2)
                        
                        temp_file.replace(run_tracker_file)
                        
                        return next_run
                    
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            self.print_color(f"\nTracker file corrupted (attempt {attempt + 1}), retrying...", 'warning')
                            time.sleep(retry_delay)
                        else:
                            self.print_color(f"\nTracker file corrupted: {str(e)}", 'error')
                            backup_file = run_tracker_file.with_suffix(f'.corrupted_{int(time.time())}')
                            try:
                                shutil.copy2(run_tracker_file, backup_file)
                                self.print_color(f"Backup created: {backup_file}", 'info')
                            except Exception:
                                pass
                            
                            existing_runs = len([d for d in output_dir.parent.iterdir() if d.is_dir() and d.name.startswith('run_')])
                            next_run = existing_runs + 1
                            
                            new_tracker = {
                                'last_run': next_run,
                                'runs': {
                                    str(next_run): {
                                        'timestamp': timestamp,
                                        'started': start_time.isoformat(),
                                        'filepath': filepath,
                                        'target_mb': target_mb,
                                        'min_chunk': min_chunk,
                                        'note': 'Tracker reinitialized due to corruption'
                                    }
                                }
                            }
                            
                            with open(run_tracker_file, 'w') as f:
                                json.dump(new_tracker, f, indent=2)
                            
                            self.print_color(f"\nTracker reinitialized with run number: {next_run}", 'success')
                            return next_run
                    
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.print_color(f"\nError reading tracker (attempt {attempt + 1}): {str(e)}", 'warning')
                            time.sleep(retry_delay)
                        else:
                            self.print_color(f"\nFailed to read run tracker: {str(e)}", 'error')
                            existing_runs = len([d for d in output_dir.parent.iterdir() if d.is_dir() and d.name.startswith('run_')])
                            return existing_runs + 1
                
                return 1
            
            # Generate sequential run ID "run_001"
            run_number = get_next_run_number(output_dir)
            run_id = f"run_{run_number:03d}"
            
            # Generate full tracking ID with all details for metadata only
            process_id = os.getpid()
            unique_hash = hashlib.md5(
                f"{timestamp}_{process_id}".encode()
            ).hexdigest()[:4]
            
            # Full ID stored only in metadata, not used for file/directory names
            run_id_full = f"run_{run_number:03d}_{timestamp}_{unique_hash}"
            
            # Create run_info dictionary
            run_info = {
                'run_id': run_id,
                'run_id_full': run_id_full,
                'run_number': run_number,
                'timestamp': timestamp,
                'start_time': start_time.isoformat(),
                'filepath': filepath,
                'target_mb': target_mb,
                'min_chunk': min_chunk
            }
            
            # Update statistics with run information
            test_statistics["test_id"] = run_id_full
            test_statistics["run_number"] = run_number
            test_statistics["run_info"] = run_info
        
        # Display run header
        if run_info and 'run_id' in run_info:
            self.print_color(f"\n" + "-"*40, 'highlight')
            self.print_color(f"TEST RUN: {Fore.YELLOW + Style.BRIGHT}{run_number:03d}", 'highlight')
            # Extract timestamp from run_id_full if available
            if 'run_id_full' in run_info:
                timestamp_str = run_info['run_id_full'].split('_')[2] + "_" + run_info['run_id_full'].split('_')[3]
                parts = run_info['run_id_full'].split('_')
                date_part = parts[2]   # YYYYMMDD
                time_part = parts[3]   # HHMMSS
                formatted_timestamp = (f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}")
                self.print_color(f"Timestamp: {Fore.YELLOW + Style.BRIGHT}{formatted_timestamp}", 'highlight')
            # Extract hash from run_id_full if available
            if 'run_id_full' in run_info:
                unique_hash = run_info['run_id_full'].split('_')[-1]
                self.print_color(f"Unique Hash: {Fore.YELLOW + Style.BRIGHT}{unique_hash}", 'highlight')
            self.print_color("-"*40, 'highlight')
        
        # Get total rows
        try:
            total_rows = self.get_total_rows(filepath)
            test_statistics["total_rows"] = total_rows
            test_statistics["status"] = "row_count_complete"
            self._update_run_tracker(output_dir.parent, run_number, 'row_count_complete', {
                'total_rows': total_rows,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.print_color(f"\nFailed to count rows: {str(e)}", 'error')
            test_statistics["status"] = "failed_row_count"
            test_statistics["end_time"] = datetime.now().isoformat()
            self._save_run_statistics(output_dir, test_statistics)
            self._update_run_tracker(output_dir.parent, run_number, 'failed_row_count', {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
        
        # System info
        sys_info = self._get_system_info()
        test_statistics["system_info"] = sys_info
        
        self.print_color("\nSystem Information:", 'info')
        self.print_color(f"  ├─ CPU Cores: {Fore.YELLOW + Style.BRIGHT}{sys_info['cpu_cores']}", 'system')
        self.print_color(f"  └─ Available RAM: {Fore.YELLOW + Style.BRIGHT}{sys_info['available_ram_gb']:.1f}GB", 'system')
        self.print_color("\nDataset Information:", 'info')
        self.print_color(f"  ├─ Total rows: {Fore.YELLOW + Style.BRIGHT}{total_rows:,}", 'system')
        self.print_color(f"  └─ Target: {Fore.YELLOW + Style.BRIGHT}{target_mb}MB/chunk", 'system')

        # Initial calibration
        self.print_color("\nMeasuring memory characteristics...", 'warning')
        calibration = self.calibrate_memory_usage(filepath)
        if calibration['status'] != 'success':
            self.print_color(f"\nCalibration failed: {calibration.get('message', 'Unknown error')}", 'error')
            test_statistics["status"] = "failed_calibration"
            test_statistics["end_time"] = datetime.now().isoformat()
            self._save_run_statistics(output_dir, test_statistics)
            self._update_run_tracker(output_dir.parent, run_number, 'failed_calibration', {
                'error': calibration.get('message', 'Unknown error'),
                'timestamp': datetime.now().isoformat()
            })
            return None
        
        test_statistics["calibration_data"] = calibration
        test_statistics["status"] = "calibration_complete"
        self._update_run_tracker(output_dir.parent, run_number, 'calibration_complete', {
            'mem_per_row': calibration['mem_per_row'],
            'timestamp': datetime.now().isoformat()
        })

        # Initial chunk size
        chunk_size = self.calculate_dynamic_chunk_size(
            calibration['mem_per_row'],
            target_mb,
            total_rows
        )
        self.print_color(f"\nInitial chunk size: {chunk_size:,} rows (~{target_mb}MB target, {calibration['mem_per_row']:.6f} MB/row)", 'success')

        # Initialize output file (in run-specific directory)
        output_file = output_dir / f"max_rows_test_{run_id}_results.json"
        
        # Initialize the JSON structure with metadata
        test_results_data = {
            "test_metadata": {
                "test_id": run_id_full,
                "run_id": run_id,
                "run_number": run_number,
                "version": self.VERSION,
                "filepath": filepath,
                "total_rows": total_rows,
                "target_mb": target_mb,
                "min_chunk_size": self.min_chunk_size,
                "safety_factor": self.safety_factor,
                "max_ram_usage": self.max_ram_usage,
                "performance_thresholds": self.performance_thresholds,
                "start_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_info": sys_info,
                "calibration_results": calibration,
                "run_info": run_info
            },
            "chunks": []  # Will store all chunk results (both successful and failed)
        }
        
        # Write initial empty structure to file
        with open(output_file, "w") as f:
            json.dump(test_results_data, f, indent=2)
        
        self.print_color(f"\nInitialized results file: {output_file}", 'success')
        test_statistics["status"] = "processing_started"
        self._update_run_tracker(output_dir.parent, run_number, 'processing_started', {
            'chunk_size': chunk_size,
            'timestamp': datetime.now().isoformat()
        })
        
        # Initialize alive_progress bar
        try:
            self.print_color(f"\nStarting processing with {total_rows:,} total rows...", 'warning')
            
            # Process dataset with progress bar
            current_row = 0
            chunk_index = 1
            all_results = []
            performance_history = []
            
            with alive_bar(title='Processing chunks', bar='smooth', spinner='dots_waves2', length=35, unit='chunks', stats=True, enrich_print=False) as bar:
                
                while current_row < total_rows:
                    chunk_end = min(current_row + chunk_size, total_rows)
                    
                    # Update progress bar message
                    bar.text = f"Chunk {chunk_index} | Rows {current_row:,}-{chunk_end:,} | Size: {chunk_size:,} rows"
                    
                    # Update tracker with chunk start
                    self._update_run_tracker(output_dir.parent, run_number, 'chunk_started', {
                        'chunk_index': chunk_index,
                        'chunk_size': chunk_size,
                        'current_row': current_row,
                        'chunk_end': chunk_end,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Process chunk with memory monitoring
                    chunk_result = self.process_chunk(
                        current_row,
                        chunk_size,
                        calibration['mem_per_row'],
                        filepath
                    )
                    
                    # Handle failures
                    if chunk_result['status'] != 'success':
                        bar.text = f"Chunk {chunk_index}: Failed - {chunk_result.get('message', 'Unknown error')[:30]}..."
                        
                        if 'suggested_chunk_size' in chunk_result:
                            new_size = chunk_result['suggested_chunk_size']
                            bar.text = f"Chunk {chunk_index}: Reducing from {chunk_size:,} to {new_size:,} rows"
                            chunk_size = max(new_size, self.min_chunk_size)
                        
                        # Create failed chunk entry
                        failed_chunk_data = {
                            "chunk_index": chunk_index,
                            "start_row": current_row,
                            "requested_chunk_size": chunk_size,
                            "status": "failed",
                            "error_type": chunk_result['status'],
                            "error_message": chunk_result.get('message', 'Unknown error'),
                            "suggested_chunk_size": chunk_result.get('suggested_chunk_size'),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Append failed chunk to results
                        test_results_data["chunks"].append(failed_chunk_data)
                        
                        # Update statistics
                        test_statistics["failed_chunks"] += 1
                        test_statistics["total_chunks"] += 1
                        
                        # Track chunk in statistics history
                        test_statistics["chunk_history"].append({
                            "chunk_index": chunk_index,
                            "status": "failed",
                            "timestamp": datetime.now().isoformat(),
                            "error": chunk_result.get('message', 'Unknown error')
                        })
                        
                        # Update tracker with chunk failure
                        self._update_run_tracker(output_dir.parent, run_number, 'chunk_failed', {
                            'chunk_index': chunk_index,
                            'error': chunk_result.get('message', 'Unknown error'),
                            'suggested_chunk_size': chunk_result.get('suggested_chunk_size'),
                            'new_chunk_size': chunk_size,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Update the output file
                        try:
                            with open(output_file, "w") as f:
                                json.dump(test_results_data, f, indent=2)
                        except Exception as write_error:
                            self._log_event(f"Failed to write chunk {chunk_index} results: {str(write_error)}", "error")
                        
                        # Don't advance progress bar for failed chunks that didn't process rows
                        if chunk_result.get('actual_rows', 0) == 0:
                            continue
                    
                    else:
                        # Create successful chunk entry
                        successful_chunk_data = {
                            "chunk_index": chunk_index,
                            "start_row": current_row,
                            "end_row": current_row + chunk_result["actual_rows"],
                            "requested_chunk_size": chunk_size,
                            "actual_chunk_size": chunk_result["actual_rows"],
                            "status": "success",
                            "metrics": {
                                "ram_usage_mb": chunk_result["ram_usage_mb"],
                                "mem_per_row_mb": chunk_result["mem_per_row"],
                                "load_time_sec": chunk_result["load_time_sec"],
                                "throughput_rows_sec": chunk_result["throughput_rows_sec"],
                                "cpu_percent": chunk_result.get("cpu_percent", 0),
                                "requested_rows": chunk_result["requested_rows"],
                                "actual_rows": chunk_result["actual_rows"]
                            },
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Append successful chunk to results
                        test_results_data["chunks"].append(successful_chunk_data)
                        
                        # Update statistics
                        test_statistics["successful_chunks"] += 1
                        test_statistics["total_chunks"] += 1
                        test_statistics["processed_rows"] = current_row + chunk_result["actual_rows"]
                        
                        # Update performance metrics in statistics
                        test_statistics["performance_metrics"]["total_processing_time"] += chunk_result["load_time_sec"]
                        
                        # Track chunk in statistics history
                        test_statistics["chunk_history"].append({
                            "chunk_index": chunk_index,
                            "status": "success",
                            "rows_processed": chunk_result["actual_rows"],
                            "ram_usage_mb": chunk_result["ram_usage_mb"],
                            "throughput_rows_sec": chunk_result["throughput_rows_sec"],
                            "cpu_percent": chunk_result.get("cpu_percent", 0),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Update tracker with chunk success
                        self._update_run_tracker(output_dir.parent, run_number, 'chunk_completed', {
                            'chunk_index': chunk_index,
                            'rows_processed': chunk_result["actual_rows"],
                            'ram_usage_mb': chunk_result["ram_usage_mb"],
                            'throughput_rows_sec': chunk_result["throughput_rows_sec"],
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Update progress bar with performance stats
                        throughput_str = f"{chunk_result['throughput_rows_sec']:,.0f}"
                        ram_str = f"{chunk_result['ram_usage_mb']:.1f}"
                        cpu_str = f"{chunk_result.get('cpu_percent', 0):.1f}"
                        
                        bar.text = (f"Chunk {chunk_index}: {chunk_result['actual_rows']:,} rows | RAM: {ram_str}MB | CPU: {cpu_str}% | Throughput: {throughput_str} rows/s")
                        
                        # Update tracking lists for summary generation
                        result_data_for_summary = {
                            "filepath": filepath,
                            "chunk_index": chunk_index,
                            "start_row": current_row,
                            "end_row": current_row + chunk_result["actual_rows"],
                            "metrics": chunk_result,
                            "system_info": sys_info,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        all_results.append(result_data_for_summary)
                        
                        performance_history.append({
                            'chunk_size': chunk_size,
                            'throughput': chunk_result['throughput_rows_sec'],
                            'mem_per_row': chunk_result['mem_per_row'],
                            'cpu_percent': chunk_result['cpu_percent']
                        })
                        
                        current_row += chunk_result["actual_rows"]
                        
                        # Dynamic adjustment - only increase if we have at least 3 successful chunks
                        if len(performance_history) > 3:
                            avg_throughput = sum(p['throughput'] for p in performance_history[-3:]) / 3
                            current_throughput = chunk_result['throughput_rows_sec']
                            
                            # Conditions for increasing chunk size:
                            # 1. Current throughput is within 10% of recent average
                            # 2. Memory usage is below 80% of target
                            # 3. CPU usage is below threshold
                            if (current_throughput > avg_throughput * 0.9 and 
                                chunk_result['ram_usage_mb'] < target_mb * 0.8 and
                                chunk_result['cpu_percent'] < self.performance_thresholds['cpu_overload'] * 100):
                                
                                new_size = min(
                                    int(chunk_size * 1.25),  # Max 25% increase
                                    total_rows - current_row
                                )
                                if new_size > chunk_size:
                                    bar.text = f"Chunk {chunk_index}: Increasing size to {new_size:,} rows"
                                    chunk_size = new_size
                        
                        # Update the output file
                        try:
                            with open(output_file, "w") as f:
                                json.dump(test_results_data, f, indent=2)
                        except Exception as write_error:
                            self._log_event(f"Failed to write chunk {chunk_index} results: {str(write_error)}", "error")
                    
                    # Update progress bar
                    bar()
                    chunk_index += 1
                    
                    # Update progress in tracker
                    progress_percent = (current_row / total_rows) * 100
                    self._update_run_tracker(output_dir.parent, run_number, 'progress_update', {
                        'current_row': current_row,
                        'total_rows': total_rows,
                        'progress_percent': progress_percent,
                        'chunks_completed': chunk_index - 1,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Add completion metadata to the results file
            successful_chunks = [c for c in test_results_data["chunks"] if c.get("status") == "success"]
            failed_chunks = [c for c in test_results_data["chunks"] if c.get("status") == "failed"]
            
            test_results_data["test_metadata"]["end_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            test_results_data["test_metadata"]["total_chunks_processed"] = len(test_results_data["chunks"])
            test_results_data["test_metadata"]["successful_chunks"] = len(successful_chunks)
            test_results_data["test_metadata"]["failed_chunks"] = len(failed_chunks)
            test_results_data["test_metadata"]["total_rows_processed"] = current_row
            test_results_data["test_metadata"]["completion_status"] = "complete" if current_row >= total_rows else "partial"
            
            # Calculate performance summary from successful chunks
            if successful_chunks:
                throughput_values = [c["metrics"]["throughput_rows_sec"] for c in successful_chunks if "metrics" in c]
                ram_usage_values = [c["metrics"]["ram_usage_mb"] for c in successful_chunks if "metrics" in c]
                cpu_values = [c["metrics"].get("cpu_percent", 0) for c in successful_chunks if "metrics" in c]
                chunk_sizes = [c["actual_chunk_size"] for c in successful_chunks if "actual_chunk_size" in c]
                
                test_results_data["test_metadata"]["performance_summary"] = {
                    "average_throughput": sum(throughput_values) / len(throughput_values) if throughput_values else 0,
                    "max_throughput": max(throughput_values) if throughput_values else 0,
                    "min_throughput": min(throughput_values) if throughput_values else 0,
                    "average_ram_usage_mb": sum(ram_usage_values) / len(ram_usage_values) if ram_usage_values else 0,
                    "max_ram_usage_mb": max(ram_usage_values) if ram_usage_values else 0,
                    "min_ram_usage_mb": min(ram_usage_values) if ram_usage_values else 0,
                    "average_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "max_cpu_percent": max(cpu_values) if cpu_values else 0,
                    "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                    "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0
                }
                
                # Update statistics with final performance metrics
                test_statistics["performance_metrics"]["average_throughput"] = test_results_data["test_metadata"]["performance_summary"]["average_throughput"]
                test_statistics["performance_metrics"]["max_throughput"] = test_results_data["test_metadata"]["performance_summary"]["max_throughput"]
                test_statistics["performance_metrics"]["min_throughput"] = test_results_data["test_metadata"]["performance_summary"]["min_throughput"]
                test_statistics["performance_metrics"]["average_ram_usage_mb"] = test_results_data["test_metadata"]["performance_summary"]["average_ram_usage_mb"]
                test_statistics["performance_metrics"]["max_ram_usage_mb"] = test_results_data["test_metadata"]["performance_summary"]["max_ram_usage_mb"]
                test_statistics["performance_metrics"]["min_ram_usage_mb"] = test_results_data["test_metadata"]["performance_summary"]["min_ram_usage_mb"]
                test_statistics["performance_metrics"]["average_cpu_percent"] = test_results_data["test_metadata"]["performance_summary"]["average_cpu_percent"]
                test_statistics["performance_metrics"]["max_cpu_percent"] = test_results_data["test_metadata"]["performance_summary"]["max_cpu_percent"]
            
            # Final write with completion metadata
            try:
                with open(output_file, "w") as f:
                    json.dump(test_results_data, f, indent=2)
            except Exception as final_write_error:
                self.print_color(f"\nError writing final results: {str(final_write_error)}", 'error')
                self._log_event(f"Failed to write final results: {str(final_write_error)}", "error")
        
        except ImportError:
            # Fallback to manual progress tracking if alive_progress is not available
            self.print_color("\nNote: Install 'alive-progress' for progress tracking", 'warning')
            self.print_color("Continuing with basic progress reporting...\n", 'warning')
            
            # Process dataset without alive_progress
            current_row = 0
            chunk_index = 1
            all_results = []
            performance_history = []
            
            while current_row < total_rows:
                chunk_end = min(current_row + chunk_size, total_rows)
                
                self.print_color(f"\nChunk {chunk_index}: Rows {current_row:,}-{chunk_end:,} (Size: {chunk_size:,} rows)", 'info')
                
                # Update tracker with chunk start
                self._update_run_tracker(output_dir.parent, run_number, 'chunk_started', {
                    'chunk_index': chunk_index,
                    'chunk_size': chunk_size,
                    'current_row': current_row,
                    'chunk_end': chunk_end,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Process chunk with memory monitoring
                chunk_result = self.process_chunk(
                    current_row,
                    chunk_size,
                    calibration['mem_per_row'],
                    filepath
                )
                
                # Handle failures
                if chunk_result['status'] != 'success':
                    self.print_color(f"\nChunk failed: {chunk_result.get('message', 'Unknown error')}", 'warning')
                    
                    if 'suggested_chunk_size' in chunk_result:
                        new_size = chunk_result['suggested_chunk_size']
                        self.print_color(f"\nReducing chunk size from {chunk_size:,} to {new_size:,}", 'warning')
                        chunk_size = max(new_size, self.min_chunk_size)
                    
                    # Create failed chunk entry
                    failed_chunk_data = {
                        "chunk_index": chunk_index,
                        "start_row": current_row,
                        "requested_chunk_size": chunk_size,
                        "status": "failed",
                        "error_type": chunk_result['status'],
                        "error_message": chunk_result.get('message', 'Unknown error'),
                        "suggested_chunk_size": chunk_result.get('suggested_chunk_size'),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Append failed chunk to results
                    test_results_data["chunks"].append(failed_chunk_data)
                    
                    # Update statistics
                    test_statistics["failed_chunks"] += 1
                    test_statistics["total_chunks"] += 1
                    
                    # Track chunk in statistics history
                    test_statistics["chunk_history"].append({
                        "chunk_index": chunk_index,
                        "status": "failed",
                        "timestamp": datetime.now().isoformat(),
                        "error": chunk_result.get('message', 'Unknown error')
                    })
                    
                    # Update tracker with chunk failure
                    self._update_run_tracker(output_dir.parent, run_number, 'chunk_failed', {
                        'chunk_index': chunk_index,
                        'error': chunk_result.get('message', 'Unknown error'),
                        'suggested_chunk_size': chunk_result.get('suggested_chunk_size'),
                        'new_chunk_size': chunk_size,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Update the output file
                    try:
                        with open(output_file, "w") as f:
                            json.dump(test_results_data, f, indent=2)
                        self.print_color(f"\nLogged failed chunk {chunk_index} to {output_file.name}", 'warning')
                    except Exception as write_error:
                        self.print_color(f"\nWarning: Failed to update results file: {str(write_error)}", 'warning')
                        self._log_event(f"Failed to write chunk {chunk_index} results: {str(write_error)}", "error")
                    
                    continue
                
                # Create successful chunk entry
                successful_chunk_data = {
                    "chunk_index": chunk_index,
                    "start_row": current_row,
                    "end_row": current_row + chunk_result["actual_rows"],
                    "requested_chunk_size": chunk_size,
                    "actual_chunk_size": chunk_result["actual_rows"],
                    "status": "success",
                    "metrics": {
                        "ram_usage_mb": chunk_result["ram_usage_mb"],
                        "mem_per_row_mb": chunk_result["mem_per_row"],
                        "load_time_sec": chunk_result["load_time_sec"],
                        "throughput_rows_sec": chunk_result["throughput_rows_sec"],
                        "cpu_percent": chunk_result.get("cpu_percent", 0),
                        "requested_rows": chunk_result["requested_rows"],
                        "actual_rows": chunk_result["actual_rows"]
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-d %H:%M:%S")
                }
                
                # Append successful chunk to results
                test_results_data["chunks"].append(successful_chunk_data)
                
                # Update statistics
                test_statistics["successful_chunks"] += 1
                test_statistics["total_chunks"] += 1
                test_statistics["processed_rows"] = current_row + chunk_result["actual_rows"]
                
                # Update performance metrics in statistics
                test_statistics["performance_metrics"]["total_processing_time"] += chunk_result["load_time_sec"]
                
                # Track chunk in statistics history
                test_statistics["chunk_history"].append({
                    "chunk_index": chunk_index,
                    "status": "success",
                    "rows_processed": chunk_result["actual_rows"],
                    "ram_usage_mb": chunk_result["ram_usage_mb"],
                    "throughput_rows_sec": chunk_result["throughput_rows_sec"],
                    "cpu_percent": chunk_result.get("cpu_percent", 0),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update tracker with chunk success
                self._update_run_tracker(output_dir.parent, run_number, 'chunk_completed', {
                    'chunk_index': chunk_index,
                    'rows_processed': chunk_result["actual_rows"],
                    'ram_usage_mb': chunk_result["ram_usage_mb"],
                    'throughput_rows_sec': chunk_result["throughput_rows_sec"],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Basic progress display
                progress_percent = (current_row / total_rows) * 100
                self.print_color(f"Progress: {progress_percent:.1f}% ({current_row:,}/{total_rows:,} rows)", 'info')
                
                # Update the output file
                try:
                    with open(output_file, "w") as f:
                        json.dump(test_results_data, f, indent=2)
                    
                    self.print_color(f"\nUpdated {output_file.name} with chunk {chunk_index} | RAM: {chunk_result['ram_usage_mb']:.2f}MB | CPU: {chunk_result['cpu_percent']:.1f}% | Time: {chunk_result['load_time_sec']:.2f}s | Throughput: {chunk_result['throughput_rows_sec']:,.0f} rows/s", 'success')
                except Exception as write_error:
                    self.print_color(f"\nWarning: Failed to update results file: {str(write_error)}", 'warning')
                    self._log_event(f"Failed to write chunk {chunk_index} results: {str(write_error)}", "error")
                
                # Update tracking lists for summary generation
                result_data_for_summary = {
                    "filepath": filepath,
                    "chunk_index": chunk_index,
                    "start_row": current_row,
                    "end_row": current_row + chunk_result["actual_rows"],
                    "metrics": chunk_result,
                    "system_info": sys_info,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_results.append(result_data_for_summary)
                
                performance_history.append({
                    'chunk_size': chunk_size,
                    'throughput': chunk_result['throughput_rows_sec'],
                    'mem_per_row': chunk_result['mem_per_row'],
                    'cpu_percent': chunk_result['cpu_percent']
                })
                
                current_row += chunk_result["actual_rows"]
                chunk_index += 1
                
                # Dynamic adjustment - only increase if we have at least 3 successful chunks
                if len(performance_history) > 3:
                    avg_throughput = sum(p['throughput'] for p in performance_history[-3:]) / 3
                    current_throughput = chunk_result['throughput_rows_sec']
                    
                    # Conditions for increasing chunk size:
                    # 1. Current throughput is within 10% of recent average
                    # 2. Memory usage is below 80% of target
                    # 3. CPU usage is below threshold
                    if (current_throughput > avg_throughput * 0.9 and 
                        chunk_result['ram_usage_mb'] < target_mb * 0.8 and
                        chunk_result['cpu_percent'] < self.performance_thresholds['cpu_overload'] * 100):
                        
                        new_size = min(
                            int(chunk_size * 1.25),  # Max 25% increase
                            total_rows - current_row
                        )
                        self.print_color(f"\nIncreasing chunk size to {new_size:,} rows", 'info')
                        chunk_size = new_size
                
                # Update progress in tracker
                self._update_run_tracker(output_dir.parent, run_number, 'progress_update', {
                    'current_row': current_row,
                    'total_rows': total_rows,
                    'progress_percent': progress_percent,
                    'chunks_completed': chunk_index - 1,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Update final statistics
        end_time = datetime.now()
        test_statistics["end_time"] = end_time.isoformat()
        test_statistics["elapsed_time_seconds"] = (end_time - start_time).total_seconds()
        test_statistics["completion_status"] = "complete" if current_row >= total_rows else "partial"
        test_statistics["status"] = "completed"
        
        # Save statistics to run directory
        self._save_run_statistics(output_dir, test_statistics)
        
        # Update tracker with completion
        self._update_run_tracker(output_dir.parent, run_number, 'completed', {
            'end_time': end_time.isoformat(),
            'elapsed_seconds': (end_time - start_time).total_seconds(),
            'total_rows_processed': current_row,
            'successful_chunks': len(successful_chunks),
            'failed_chunks': len(failed_chunks),
            'completion_status': "complete" if current_row >= total_rows else "partial",
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate summary
        if all_results:
            average_throughput = sum(r["metrics"]["throughput_rows_sec"] for r in all_results) / len(all_results)
            max_chunk_size = max(r["metrics"]["actual_rows"] for r in all_results)
            min_memory_per_row = min(r["metrics"]["mem_per_row"] for r in all_results)
            max_cpu_usage = max(r["metrics"].get("cpu_percent", 0) for r in all_results)
            total_processing_time = sum(r["metrics"]["load_time_sec"] for r in all_results)
        else:
            average_throughput = 0
            max_chunk_size = 0
            min_memory_per_row = 0
            max_cpu_usage = 0
            total_processing_time = 0
        
        summary = {
            "test_id": run_id_full,
            "run_id": run_id,
            "run_number": run_number,
            "version": self.VERSION,
            "system_info": sys_info,
            "dataset_info": {
                "filepath": filepath,
                "total_rows": total_rows,
                "processed_rows": current_row,
                "chunks_tested": len(all_results)
            },
            "performance_metrics": {
                "average_throughput": average_throughput,
                "max_chunk_size": max_chunk_size,
                "min_memory_per_row": min_memory_per_row,
                "max_cpu_usage": max_cpu_usage,
                "total_processing_time": total_processing_time
            },
            "chunk_details": [{
                "chunk_index": r['chunk_index'],
                "rows": f"{r['start_row']}-{r['end_row']}",
                "rows_processed": r["metrics"]["actual_rows"],
                "ram_used_mb": r["metrics"]["ram_usage_mb"],
                "cpu_percent": r["metrics"].get("cpu_percent", 0),
                "time_sec": r["metrics"]["load_time_sec"],
                "throughput_rows_sec": r["metrics"]["throughput_rows_sec"]
            } for r in all_results],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "target_mb": target_mb,
                "min_chunk_size": self.min_chunk_size,
                "safety_factor": self.safety_factor,
                "max_ram_usage": self.max_ram_usage
            },
            "run_info": run_info,
            "run_directory": str(output_dir),
            "results_file": str(output_file)
        }

        # Save summary files to run directory
        summary_file = output_dir / f"max_rows_testing_summary_{run_id}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV version
        csv_file = output_dir / f"max_rows_testing_summary_{run_id}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Metric', 'Value'])
            # Write system info
            writer.writerow(['System Info', ''])
            for key, value in summary['system_info'].items():
                writer.writerow([f"System_{key}", value])
            # Write dataset info
            writer.writerow(['Dataset Info', ''])
            for key, value in summary['dataset_info'].items():
                writer.writerow([f"Dataset_{key}", value])
            # Write performance metrics
            writer.writerow(['Performance Metrics', ''])
            for key, value in summary['performance_metrics'].items():
                writer.writerow([f"Performance_{key}", value])
            # Write configuration
            writer.writerow(['Configuration', ''])
            for key, value in summary['config'].items():
                writer.writerow([f"Config_{key}", value])
            # Write run info
            writer.writerow(['Run Info', ''])
            for key, value in summary['run_info'].items():
                if key != 'start_time':  # Skip ISO timestamp for CSV readability
                    writer.writerow([f"Run_{key}", value])
        
        # Generate plots in run directory
        plot_filepath = output_dir / f"preprocessing_performance_plots_{run_id}.png"
        self.plot_file = self.generate_plots(all_results, output_dir, plot_filepath)
        
        # Update history with run information
        self.update_history(summary)
        
        self.print_color(f"\n" + "-"*40, 'highlight')
        self.print_color("TEST COMPLETED SUCCESSFULLY", 'success')
        self.print_color("-"*40, 'highlight')
        
        self.print_color(f"\nSummary files:", 'success')
        self.print_color(f"   ├─ JSON: {Fore.MAGENTA + Style.BRIGHT}{summary_file}", 'success')
        self.print_color(f"   └─ CSV: {Fore.MAGENTA + Style.BRIGHT}{csv_file}", 'success')
        self.print_color(f"\nRun summary:", 'success')
        self.print_color(f"   ├─ Rows processed: {Fore.YELLOW + Style.BRIGHT}{current_row:,}", 'success')
        self.print_color(f"   ├─ Number of chunks: {Fore.YELLOW + Style.BRIGHT}{len(all_results)}", 'success')
        self.print_color(f"   ├─ Average throughput: {Fore.YELLOW + Style.BRIGHT}{summary['performance_metrics']['average_throughput']:,.0f} rows/sec", 'success')
        self.print_color(f"   ├─ Maximum safe chunk size: {Fore.YELLOW + Style.BRIGHT}{summary['performance_metrics']['max_chunk_size']:,} rows", 'success')
        self.print_color(f"   ├─ Peak CPU usage: {Fore.YELLOW + Style.BRIGHT}{summary['performance_metrics']['max_cpu_usage']:.1f}%", 'success')
        self.print_color(f"   ├─ Run directory: {Fore.WHITE + Style.BRIGHT}{output_dir}", 'success')
        self.print_color(f"   ├─ Final results: {Fore.MAGENTA + Style.BRIGHT}{output_file}", 'success')
        self.print_color(f"   ├─ Statistics: {Fore.MAGENTA + Style.BRIGHT}{self.stats_file}", 'success')
        if self.plot_file:
            self.print_color(f"   └─ Performance plots: {Fore.MAGENTA + Style.BRIGHT}{self.plot_file}", 'success')
        else:
            self.print_color(f"   └─ Performance plots: {Fore.RED + Style.BRIGHT}Failed to generate", 'success')
        
        return summary

    def _update_run_tracker(self, base_output_dir: Path, run_number: int, status: str, data: Dict[str, Any]) -> None:
        """Update the run tracker file with current status using atomic writes."""
        max_retries = 3
        retry_delay = 0.1
        
        try:
            run_tracker_file = base_output_dir / ".preprocessor_run_tracker"
            
            if not run_tracker_file.exists():
                return
            
            for attempt in range(max_retries):
                try:
                    with open(run_tracker_file, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            tracker = {'last_run': 0, 'runs': {}}
                        else:
                            tracker = json.loads(content)
                    
                    if not isinstance(tracker, dict):
                        raise ValueError("Tracker is not a dictionary")
                    if 'runs' not in tracker:
                        tracker['runs'] = {}
                    
                    run_key = str(run_number)
                    if run_key in tracker['runs']:
                        if 'status_updates' not in tracker['runs'][run_key]:
                            tracker['runs'][run_key]['status_updates'] = []
                        
                        tracker['runs'][run_key]['status_updates'].append({
                            'status': status,
                            'timestamp': datetime.now().isoformat(),
                            'data': data
                        })
                        
                        if status in ['completed', 'failed_row_count', 'failed_calibration', 'failed_file_not_found', 'failed_sample_loading', 'failed_no_data', 'stopped_by_user']:
                            tracker['runs'][run_key]['status'] = status
                            tracker['runs'][run_key]['completed'] = datetime.now().isoformat()
                    
                    temp_file = run_tracker_file.with_suffix('.tmp')
                    with open(temp_file, 'w') as f:
                        json.dump(tracker, f, indent=2)
                    
                    temp_file.replace(run_tracker_file)
                    break
                    
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        self._log_event(f"JSON decode error in run tracker (attempt {attempt + 1}): {str(e)}", "warning")
                        time.sleep(retry_delay)
                    else:
                        self._log_event(f"Run tracker corrupted after {max_retries} attempts: {str(e)}", "error")
                        backup_file = run_tracker_file.with_suffix(f'.corrupted_{int(time.time())}')
                        try:
                            shutil.copy2(run_tracker_file, backup_file)
                            self.print_color(f"\nCorrupted tracker backed up to: {backup_file}", 'warning')
                        except Exception:
                            pass
                        
                        new_tracker = {
                            'last_run': run_number,
                            'runs': {
                                str(run_number): {
                                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                                    'started': datetime.now().isoformat(),
                                    'status': status,
                                    'note': 'Tracker reinitialized due to corruption'
                                }
                            }
                        }
                        with open(run_tracker_file, 'w') as f:
                            json.dump(new_tracker, f, indent=2)
                        break
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        self._log_event(f"Error updating run tracker (attempt {attempt + 1}): {str(e)}", "warning")
                        time.sleep(retry_delay)
                    else:
                        self._log_event(f"Failed to update run tracker after {max_retries} attempts: {str(e)}", "error")
                        break
        
        except Exception as e:
            self._log_event(f"Critical error in _update_run_tracker: {str(e)}", "error")

    def _save_run_statistics(self, run_dir: Path, statistics: Dict[str, Any]) -> None:
        """Save run statistics to a JSON file in the run directory."""
        try:
            self.stats_file = run_dir / "preprocessing_statistics.json"
            with open(self.stats_file, "w") as f:
                json.dump(statistics, f, indent=2)
        except Exception as e:
            self.print_color(f"\nFailed to save statistics: {str(e)}", 'error')
            self._log_event(f"Failed to save run statistics: {str(e)}", "error")
    
    def display_banner(self) -> None:
        """Display the interactive console UI banner."""
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

        self.print_color("\n" + "=" * 40, 'info')
        self.print_color("- Interactive Mode -".center(40), 'success')
        self.print_color(f"PREPROCESSOR v{self.VERSION}".center(40), 'highlight')
        self.print_color("=" * 40, 'info')

    def display_main_menu(self) -> None:
        """Display the main menu options."""
        self.print_color("\nAvailable Options:", 'warning')
        self.print_color("1. Run new chunk size test", 'debug')
        self.print_color("2. Run preprocessing pipeline", 'debug')
        self.print_color("3. View recent test history", 'debug')
        self.print_color("4. Configure settings", 'debug')
        self.print_color("5. System information", 'debug')
        self.print_color("0. Exit", 'error')

    def display_test_history(self) -> None:
        """Display recent test history from saved file with enhanced formatting and error handling."""
        max_retries = 3
        retry_delay = 0.1
        
        try:
            if not self.history_file.exists():
                self.print_color("\nNo test history available - no history file found", 'warning')
                return
            
            history = None
            for attempt in range(max_retries):
                try:
                    with open(self.history_file, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            history = []
                        else:
                            history = json.loads(content)
                    
                    if not isinstance(history, list):
                        raise ValueError("History is not a list")
                    break
                
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        self._log_event(f"JSON decode error reading history (attempt {attempt + 1}): {str(e)}", "warning")
                        time.sleep(retry_delay)
                    else:
                        self.print_color("\nTest history file is corrupted", 'warning')
                        backup_path = self.history_file.with_suffix(f'.corrupted_{int(time.time())}')
                        try:
                            shutil.copy2(self.history_file, backup_path)
                            self.print_color(f"Corrupted file backed up to: {backup_path}", 'info')
                        except Exception:
                            pass
                        return
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        self._log_event(f"Error reading history (attempt {attempt + 1}): {str(e)}", "warning")
                        time.sleep(retry_delay)
                    else:
                        self.print_color(f"\nError reading test history: {str(e)}", 'error')
                        return
            
            if history is None or not history:
                self.print_color("\nNo test history found - history file is empty", 'warning')
                return
            
            self.print_color("\n" + "-" * 40, 'highlight')
            self.print_color("Test History (Last 5 Tests)", 'highlight')
            self.print_color("-" * 40, 'highlight')
            
            recent_tests = history[-5:]
            
            for i, test in enumerate(recent_tests, 1):
                try:
                    self.print_color(f"\nTEST {i}:", 'highlight')
                    
                    run_info = test.get('run_info', {})
                    if run_info:
                        run_id = run_info.get('run_id', 'Unknown')
                        run_number = run_info.get('run_number', 'Unknown')
                        self.print_color(f"Run #{run_number}", 'success')
                        self.print_color(f"  └─ Run ID: {Fore.YELLOW + Style.BRIGHT}{run_id}", 'success')
                    
                    timestamp = test.get('timestamp', 'Unknown')
                    self.print_color(f"1. Timestamp: {Fore.YELLOW + Style.BRIGHT}{timestamp}", 'info')
                    
                    dataset_info = test.get('dataset_info', {})
                    filepath = dataset_info.get('filepath', 'Unknown')
                    self.print_color(f"2. Dataset: {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'info')
                    
                    processed_rows = dataset_info.get('processed_rows', 0)
                    chunks_tested = dataset_info.get('chunks_tested', 0)
                    self.print_color(f"3. Rows Processed: {Fore.GREEN + Style.BRIGHT}{processed_rows:,}", 'info')
                    self.print_color(f"    └─ Chunks: {Fore.YELLOW + Style.BRIGHT}{chunks_tested} chunks", 'info')
                    
                    perf_metrics = test.get('performance_metrics', {})
                    avg_throughput = perf_metrics.get('average_throughput', 0)
                    avg_throughput_styled = Fore.GREEN + Style.BRIGHT + f"{avg_throughput:,.0f}"
                    self.print_color(f"4. Avg Throughput: {Fore.GREEN + Style.BRIGHT}{avg_throughput_styled} rows/sec", 'info')
                    
                    max_chunk = perf_metrics.get('max_chunk_size', 0)
                    self.print_color(f"5. Max Chunk Size: {Fore.RED + Style.BRIGHT}{max_chunk:,} rows", 'info')
                    
                    max_cpu = perf_metrics.get('max_cpu_usage', 'N/A')
                    if isinstance(max_cpu, (int, float)):
                        max_cpu_styled = Fore.RED + Style.BRIGHT + f"{max_cpu:.1f}%" if max_cpu > 80 else Fore.GREEN + Style.BRIGHT + f"{max_cpu:.1f}%"
                        self.print_color(f"6. Peak CPU: {max_cpu_styled}", 'info')
                    else:
                        self.print_color(f"6. Peak CPU: {Fore.YELLOW + Style.BRIGHT}{max_cpu}", 'info')
                    
                    total_time = perf_metrics.get('total_processing_time', 0)
                    if isinstance(total_time, (int, float)):
                        self.print_color(f"7. Total Time: {Fore.GREEN + Style.BRIGHT}{total_time:.2f} sec", 'info')
                    else:
                        self.print_color(f"7. Total Time: {Fore.GREEN + Style.BRIGHT}{total_time}", 'info')
                    
                except Exception as display_error:
                    self.print_color(f"\nError displaying test {i}: {str(display_error)}", 'warning')
                    self._log_event(f"Error displaying test entry {i}: {str(display_error)}", "warning")
                    continue
            
            self.print_color("\n" + "-" * 40, 'highlight')
            self.print_color(f"Total entries in history: {Fore.YELLOW + Style.BRIGHT}{len(history)}", 'info')
            self.print_color("-" * 40, 'highlight')
            
        except KeyboardInterrupt:
            self.print_color("\nHistory display interrupted by user.", 'warning')
        except Exception as e:
            self.print_color(f"\nError reading test history: {str(e)}", 'error')
            self._log_event(f"Display history error: {str(e)}", "error")

    def display_system_info(self) -> None:
        """Display detailed system information."""
        self.print_color("\n" + "-" * 40, 'highlight')
        self.print_color("System Information", 'highlight')
        self.print_color("-" * 40, 'highlight')
        
        info = self._get_system_info()
        for key, value in info.items():
            if key.endswith('_gb'):
                self.print_color(f"{key.replace('_', ' ').title():<25}: {Fore.YELLOW + Style.BRIGHT}{value:.2f} GB", 'info')
            elif key == 'timestamp':
                self.print_color(f"{key.replace('_', ' ').title():<25}: {Fore.GREEN + Style.BRIGHT}{datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M:%S')}", 'info')
            else:
                self.print_color(f"{key.replace('_', ' ').title():<25}: {Fore.YELLOW + Style.BRIGHT}{value}", 'info')
        
        self.print_color("-" * 40, 'highlight')

    def display_test_runs(self, results_dir: Optional[str] = None) -> None:
        """Display available test runs in a formatted table."""
        if results_dir is None:
            #results_dir = self.config.get('default_output_dir', 'results')
            results_dir = self.config.get('default_results_dir', 'results')
        
        results_path = Path(results_dir)
        
        if not results_path.exists():
            self.print_color(f"\nResults directory not found: {Fore.YELLOW + Style.BRIGHT}{results_dir}{Style.RESET_ALL}", 'error')
            return
        
        run_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')], key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0, reverse=True)
        
        if not run_dirs:
            self.print_color("\nNo test runs found in results directory", 'warning')
            return
        
        self.print_color("\n" + "-" * 40, 'highlight')
        self.print_color(f"AVAILABLE TEST RUNS {Fore.GREEN + Style.BRIGHT}({len(run_dirs)}){Style.RESET_ALL}", 'highlight')
        self.print_color("-" * 40, 'highlight')
        
        for i, run_dir in enumerate(run_dirs[:10], 1):  # Show first 10 runs
            run_id = run_dir.name
            summary_file = run_dir / f"max_rows_testing_summary_{run_id}.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        config = json.load(f)
                        
                        timestamp = config.get('timestamp', 'Unknown')
                        if timestamp != 'Unknown':
                            try:
                                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                                formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                            except:
                                formatted_time = timestamp
                        else:
                            formatted_time = 'Unknown'
                        
                        self.print_color(f"\n{i}. {run_id}", 'highlight')
                        self.print_color(f"  ├─ Timestamp: {Fore.GREEN + Style.BRIGHT}{formatted_time}", 'info')
                        
                        dataset_info = config.get('dataset_info', {})
                        filepath = dataset_info.get('filepath', 'Unknown')
                        self.print_color(f"  ├─ Dataset: {Fore.MAGENTA + Style.BRIGHT}{Path(filepath).name}", 'info')
                        
                        processed_rows = dataset_info.get('processed_rows', 0)
                        total_rows = dataset_info.get('total_rows', 0)
                        self.print_color(f"  ├─ Rows: {Fore.GREEN + Style.BRIGHT}{processed_rows:,} / {total_rows:,}", 'info')
                        
                        perf_metrics = config.get('performance_metrics', {})
                        max_chunk = perf_metrics.get('max_chunk_size', 0)
                        self.print_color(f"  ├─ Max chunk: {Fore.YELLOW + Style.BRIGHT}{max_chunk:,} rows", 'info')
                        
                        avg_throughput = perf_metrics.get('average_throughput', 0)
                        self.print_color(f"  ├─ Throughput: {Fore.GREEN + Style.BRIGHT}{avg_throughput:,.0f} rows/sec", 'info')
                        
                        max_cpu = perf_metrics.get('max_cpu_usage', 'N/A')
                        if isinstance(max_cpu, (int, float)):
                            cpu_color = Fore.RED + Style.BRIGHT if max_cpu > 80 else Fore.GREEN + Style.BRIGHT
                            self.print_color(f"  └─ Peak CPU: {cpu_color}{max_cpu:.1f}%", 'info')
                        else:
                            self.print_color(f"  └─ Peak CPU: {max_cpu}", 'info')
                
                except Exception as e:
                    self.print_color(f"\n{i}. {run_id} - Error reading summary", 'error')
                    self.print_color(f"  └─ Error: {str(e)[:50]}...", 'debug')
            else:
                self.print_color(f"\n{i}. {run_id} - No summary file", 'warning')
        
        if len(run_dirs) > 10:
            self.print_color(f"\n... and {len(run_dirs) - 10} more runs", 'info')

    def select_test_run_interactively(self, results_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Allow user to select from available test runs."""
        if results_dir is None:
            #results_dir = self.config.get('default_output_dir', 'results')
            results_dir = self.config.get('default_results_dir', 'results')
        
        results_path = Path(results_dir)
        
        if not results_path.exists():
            self.print_color(f"\nResults directory not found: {Fore.YELLOW + Style.BRIGHT}{results_dir}{Style.RESET_ALL}", 'error')
            return None
        
        run_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')], key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0, reverse=True)
        
        if not run_dirs:
            self.print_color("\nNo test runs found", 'warning')
            return None
        
        self.display_test_runs(results_dir)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                choice = input(Fore.YELLOW + Style.BRIGHT + f"\nSelect a test run (1-{min(10, len(run_dirs))}, 0 to cancel): " + Style.RESET_ALL).strip()
                
                if choice == '0':
                    self.print_color("\nSelection cancelled", 'warning')
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= min(10, len(run_dirs)):
                    selected_run_dir = run_dirs[choice_num - 1]
                    run_id = selected_run_dir.name
                    summary_file = selected_run_dir / f"max_rows_testing_summary_{run_id}.json"
                    
                    if summary_file.exists():
                        with open(summary_file) as f:
                            config = json.load(f)
                        
                        selected_run = {
                            'run_id': run_id,
                            'run_number': config.get('run_number', 'Unknown'),
                            'timestamp': config.get('timestamp', 'Unknown'),
                            'filepath': config['dataset_info']['filepath'],
                            'total_rows': config['dataset_info']['total_rows'],
                            'processed_rows': config['dataset_info']['processed_rows'],
                            'max_chunk_size': config['performance_metrics']['max_chunk_size'],
                            'average_throughput': config['performance_metrics']['average_throughput'],
                            'max_cpu_usage': config['performance_metrics'].get('max_cpu_usage', 'N/A'),
                            'run_directory': str(selected_run_dir),
                            'summary_file': str(summary_file)
                        }
                        
                        self.print_color(f"\nSelected: {selected_run['run_id']} (#{selected_run['run_number']})", 'success')
                        return selected_run
                    else:
                        self.print_color(f"\nSummary file not found for run {run_id}", 'error')
                        return None
                else:
                    self.print_color(f"\nPlease enter a number between 0 and {min(10, len(run_dirs))}", 'warning')
            
            except ValueError:
                self.print_color("\nInvalid input. Please enter a number.", 'warning')
            except KeyboardInterrupt:
                self.print_color("\nSelection cancelled", 'warning')
                return None
        
        self.print_color("\nToo many invalid attempts. Returning to menu.", 'error')
        return None

    def get_memory_config(self, config_path: Optional[str] = None, interactive: bool = True) -> Tuple[int, int]:
        """
        Get memory configuration with interactive fallback and support for run-based directory structure.
        Returns (max_rows, chunk_size)
        """
        if config_path is None:
            config_path = self.config.get('default_results_dir', 'results')
        
        config_file = Path(config_path)
        
        try:
            # Check if the provided path is a specific run directory (either by name or by checking for summary file)
            if config_file.is_dir() and (config_file.name.startswith('run_') or (config_file / f"max_rows_testing_summary_{config_file.name}.json").exists()):
                # Direct run directory provided
                run_dir = config_file
                run_id = run_dir.name
                summary_file = run_dir / f"max_rows_testing_summary_{run_id}.json"
                
                if summary_file.exists():
                    with open(summary_file) as f:
                        config = json.load(f)
                        max_rows = config['dataset_info']['total_rows']
                        chunk_size = int(config['performance_metrics']['max_chunk_size'] * self.config['preprocessing_config']['memory_safety_factor'])
                        
                        self.print_color(f"\nUsing tested configuration from {Fore.YELLOW + Style.BRIGHT}{run_id}", 'success')
                        self.print_color(f"  ├─ Run number: {Fore.YELLOW + Style.BRIGHT}{config.get('run_number', 'Unknown')}", 'success')
                        self.print_color(f"  ├─ Test timestamp: {Fore.YELLOW + Style.BRIGHT}{config.get('timestamp', 'Unknown')}", 'success')
                        self.print_color(f"  ├─ Max rows: {Fore.YELLOW + Style.BRIGHT}{max_rows:,}", 'success')
                        self.print_color(f"  └─ Safe chunk size: {Fore.YELLOW + Style.BRIGHT}{chunk_size:,}", 'success')
                        
                        if interactive:
                            response = input(Fore.YELLOW + Style.BRIGHT + "\nUse this configuration? (Y/n): " + Style.RESET_ALL).strip().lower()
                            if response in ['n', 'no']:
                                raise FileNotFoundError("\nUser chose not to use found configuration")
                        
                        return max_rows, chunk_size
                else:
                    self.print_color(f"\nSummary file not found in run directory: {summary_file}", 'warning')
            
            # Check if path points to results directory (search for latest run)
            elif config_file.is_dir():
                results_dir = config_file
                
                # Find all run directories
                run_dirs = sorted(
                    [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                    key=lambda x: int(x.name.split('_')[1]) if len(x.name.split('_')) > 1 and x.name.split('_')[1].isdigit() else 0,
                    reverse=True
                )
                
                if run_dirs:
                    # Use the most recent run
                    latest_run = run_dirs[0]
                    run_id = latest_run.name
                    summary_file = latest_run / f"max_rows_testing_summary_{run_id}.json"
                    
                    if summary_file.exists():
                        with open(summary_file) as f:
                            config = json.load(f)
                            max_rows = config['dataset_info']['total_rows']
                            chunk_size = int(config['performance_metrics']['max_chunk_size'] * self.config['preprocessing_config']['memory_safety_factor'])
                            
                            self.print_color(f"\nUsing latest tested configuration from {Fore.YELLOW + Style.BRIGHT}{run_id}", 'success')
                            self.print_color(f"  ├─ Run number: {Fore.YELLOW + Style.BRIGHT}{config.get('run_number', 'Unknown')}", 'success')
                            self.print_color(f"  ├─ Test timestamp: {Fore.YELLOW + Style.BRIGHT}{config.get('timestamp', 'Unknown')}", 'success')
                            self.print_color(f"  ├─ Max rows: {Fore.YELLOW + Style.BRIGHT}{max_rows:,}", 'success')
                            self.print_color(f"  └─ Safe chunk size: {Fore.YELLOW + Style.BRIGHT}{chunk_size:,}", 'success')
                            
                            if interactive:
                                response = input(Fore.YELLOW + Style.BRIGHT + "\nUse this configuration? (Y/n): " + Style.RESET_ALL).strip().lower()
                                if response in ['n', 'no']:
                                    raise FileNotFoundError("\nUser chose not to use found configuration")
                            
                            return max_rows, chunk_size
            
            # Original path handling for backward compatibility (specific summary file)
            elif config_file.exists() and config_file.is_file():
                with open(config_file) as f:
                    config = json.load(f)
                    max_rows = config['dataset_info']['total_rows']
                    chunk_size = int(config['performance_metrics']['max_chunk_size'] * self.config['preprocessing_config']['memory_safety_factor'])
                    self.print_color(f"\nUsing tested configuration from file:", 'success')
                    self.print_color(f"  ├─ Max rows: {Fore.YELLOW + Style.BRIGHT}{max_rows:,}", 'success')
                    self.print_color(f"  └─ Safe chunk size: {Fore.YELLOW + Style.BRIGHT}{chunk_size:,}", 'success')
                    return max_rows, chunk_size
        
        except Exception as e:
            self.print_color(f"\nConfig load error: {str(e)}", 'error')
            self._log_event(f"Failed to load config from {config_path}: {str(e)}", "error")
        
        if not interactive:
            return 1000000, 100000
        
        self.print_color("\n" + "-" * 40, 'warning')
        self.print_color("SYSTEM CAPACITY DATA NOT AVAILABLE", 'warning')
        self.print_color("-" * 40, 'warning')
        
        self.print_color("\nRunning a capacity test ensures optimal performance and prevents memory errors.", 'info')
        self.print_color("Without tested configuration, you may experience:", 'warning')
        self.print_color("  ├─ Memory overflow crashes", 'warning')
        self.print_color("  ├─ Suboptimal processing speed", 'warning')
        self.print_color("  └─ System instability", 'warning')
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = input(Fore.YELLOW + Style.BRIGHT + "\nRun system capacity test now? (Y/n): " + Style.RESET_ALL).strip().lower()
                
                if response in ['y', 'yes', '']:
                    # Run the test
                    filepath = input(Fore.CYAN + Style.BRIGHT + "\nEnter CSV file path for testing: " + Style.RESET_ALL).strip()
                    if not filepath:
                        self.print_color("\nNo file path provided", 'error')
                        continue
                    
                    if not Path(filepath).exists():
                        self.print_color(f"\nFile not found: {filepath}", 'error')
                        continue
                    
                    # Run test with current settings
                    test_result = self.run_test(
                        filepath=filepath,
                        output_dir=self.config['default_results_dir'],
                        target_mb=self.config['default_target_mb'],
                        min_chunk=self.config['default_min_chunk']
                    )
                    
                    if test_result:
                        try:
                            max_rows = test_result['dataset_info']['total_rows']
                            chunk_size = int(test_result['performance_metrics']['max_chunk_size'] * self.config['preprocessing_config']['memory_safety_factor'])
                            return max_rows, chunk_size
                        except KeyError:
                            self.print_color("\nTest didn't produce valid results", 'error')
                    break
                elif response in ['n', 'no']:
                    self.print_color("\nSkipping capacity test", 'warning')
                    break
                else:
                    if attempt < max_attempts - 1:
                        self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
            except KeyboardInterrupt:
                self.print_color("\nInput cancelled", 'warning')
                break
        
        self.print_color("\n" + "-" * 40, 'error')
        self.print_color("USING DEFAULT VALUES - MAY CAUSE MEMORY ISSUES", 'error')
        self.print_color("-" * 40, 'error')
        
        if interactive:
            self.print_color("\nYou can specify custom limits or accept defaults:", 'info')
            
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    response = input(Fore.YELLOW + Style.BRIGHT + "\nSpecify custom limits? (Y/n): " + Style.RESET_ALL).strip().lower()
                    
                    if response in ['y', 'yes', '']:
                        while True:
                            try:
                                max_rows = int(input(Fore.YELLOW + Style.BRIGHT + "\nEnter max rows to process (default: 1,000,000): " + Style.RESET_ALL) or "1000000")
                                chunk_size = int(input(Fore.YELLOW + Style.BRIGHT + "\nEnter chunk size (default: 100,000): " + Style.RESET_ALL) or "100000")
                                
                                if max_rows > 0 and chunk_size > 0:
                                    self.print_color(f"\nUsing custom configuration:", 'success')
                                    self.print_color(f"  ├─ Max rows: {max_rows:,}", 'success')
                                    self.print_color(f"  └─ Chunk size: {chunk_size:,}", 'success')
                                    return max_rows, chunk_size
                                else:
                                    self.print_color("\nPlease enter positive integers", 'warning')
                            except ValueError:
                                self.print_color("\nInvalid number format", 'error')
                    elif response in ['n', 'no']:
                        break
                    else:
                        if attempt < max_attempts - 1:
                            self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                except KeyboardInterrupt:
                    self.print_color("\nInput cancelled", 'warning')
                    break
        
        default_max = 1000000
        default_chunk = 100000
        self.print_color(f"\nUsing default values:", 'warning')
        self.print_color(f"  ├─ Max rows: {default_max:,}", 'warning')
        self.print_color(f"  └─ Chunk size: {default_chunk:,}", 'warning')
        self.print_color("\nWarning: These values may cause memory issues with large datasets!", 'error')
        
        return default_max, default_chunk

    def select_and_pad_features(self, df: pd.DataFrame, all_features: list, target_count: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """Select and pad features to reach target count for hybrid system compatibility."""
        if target_count is None:
            target_count = self.config['preprocessing_config']['hybrid_feature_count']
        
        current_features = [col for col in all_features if col in df.columns]
        
        if len(current_features) >= target_count:
            selected_features = current_features[:target_count]
            self.print_color(f"\nFeature Selection:", 'info')
            self.print_color(f"  ├─ Using {Fore.GREEN + Style.BRIGHT}{len(selected_features)} features", 'info')
            self.print_color(f"  └─ Truncated from {Fore.YELLOW + Style.BRIGHT}{len(current_features)}", 'info')
            return df[selected_features], selected_features
        else:
            missing_count = target_count - len(current_features)
            self.print_color(f"\nFeature Padding:", 'info')
            self.print_color(f"  ├─ Current features: {Fore.GREEN + Style.BRIGHT}{len(current_features)}", 'info')
            self.print_color(f"  ├─ Target features: {Fore.YELLOW + Style.BRIGHT}{target_count}", 'info')
            self.print_color(f"  └─ Synthetic features adding: {Fore.GREEN + Style.BRIGHT}{missing_count}", 'info')
            
            synthetic_features = []
            for i in range(missing_count):
                synthetic_col = f"synthetic_feature_{i}"
                df[synthetic_col] = 0.0
                synthetic_features.append(synthetic_col)
            
            final_features = current_features + synthetic_features
            return df[final_features], final_features

    def safe_label_encode(self, series: pd.Series, encoder: LabelEncoder) -> pd.Series:
        """Handle unseen labels in LabelEncoder by assigning them to a special category."""
        series_str = series.astype(str)
        encoder_classes = list(encoder.classes_)
        
        try:
            return encoder.transform(series_str)
        except ValueError:
            # Handle unseen labels by assigning them to a new category
            extended_classes = encoder_classes + ['UNSEEN_LABEL']
            extended_encoder = LabelEncoder()
            extended_encoder.fit(extended_classes)
            
            clean_series = series_str.where(series_str.isin(encoder_classes), 'UNSEEN_LABEL')
            return extended_encoder.transform(clean_series)

    def process_chunk_preprocessing(
        self,
        df: pd.DataFrame,
        output_path: Path,
        encoders: Dict[str, Any],
        scaler: Optional[MinMaxScaler],
        label_col: str = "Label"
    ) -> Tuple[pd.DataFrame, List[str], MinMaxScaler]:
        """Process a single chunk of data with all transformations."""
        initial_rows = len(df)
        df.dropna(inplace=True)
        
        if initial_rows != len(df):
            self.print_color(f"  └─ Rows removed with missing values: {Fore.GREEN + Style.BRIGHT}{initial_rows - len(df)}", 'warning')
        
        labels = df[label_col].astype('category').cat.codes
        df.drop(columns=[label_col], inplace=True)
        
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        # IP Address Encoding with unseen label handling
        ip_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
        for col in ip_cols:
            if col in df.columns and col in encoders:
                df[col] = self.safe_label_encode(df[col], encoders[col])
        
        other_categorical = list(set(categorical_cols) - set(ip_cols))
        encoded_feature_names = []
        
        if other_categorical and 'one_hot' in encoders:
            encoded_features = encoders['one_hot'].transform(df[other_categorical])
            encoded_feature_names = encoders['one_hot'].get_feature_names_out(other_categorical).tolist()
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
            df = df.drop(columns=other_categorical)
            df = pd.concat([df, encoded_df], axis=1)
        
        remaining_numeric = [col for col in numeric_cols if col not in ip_cols]
        ip_encoded_cols = [col for col in ip_cols if col in df.columns]
        all_feature_names = remaining_numeric + ip_encoded_cols + encoded_feature_names
        
        df_features, selected_features = self.select_and_pad_features(df, all_feature_names)
        
        if scaler is None:
            scaler_range = tuple(self.config['preprocessing_config']['scaler_range'])
            scaler = MinMaxScaler(feature_range=scaler_range)
            scaler.fit(df_features[selected_features])
        
        df_features[selected_features] = scaler.transform(df_features[selected_features])
        df_features[label_col] = labels
        
        return df_features, selected_features, scaler

    def preprocess_data(
        self,
        filepath: str,
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        interactive: bool = True,
        verbose: bool = False,
        run_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Main preprocessing pipeline with memory-aware chunked processing and run selection."""
        
        # Setup experiment tracking - use provided run_info or create default
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        if output_dir is None:
            preprocessing_config = self.config.get('preprocessing_config', {})
            output_dir = preprocessing_config.get('default_output_dir', 'results')
        
        if config_path is None:
            config_path = self.config.get('default_results_dir', 'results')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessing statistics
        preprocess_statistics = {
            "run_id": "",
            "run_number": 0,
            "start_time": start_time.isoformat(),
            "status": "initializing",
            "input_file": filepath,
            "output_directory": "",
            "total_rows": 0,
            "processed_rows": 0,
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "memory_config": {
                "max_rows": 0,
                "chunk_size": 0,
                "safety_factor": self.config['preprocessing_config']['memory_safety_factor']
            },
            "performance_metrics": {
                "average_processing_time": 0,
                "total_processing_time": 0,
                "average_rows_per_second": 0,
                "max_rows_per_second": 0,
                "min_rows_per_second": 0
            },
            "chunk_history": [],
            "system_info": self._get_system_info(),
            "final_feature_count": 0,
            "completion_status": "",
            "end_time": "",
            "elapsed_time_seconds": 0,
            "run_info": run_info if run_info else {}
        }
        
        # Extract or generate run information
        if run_info and 'run_id' in run_info:
            # Use run information provided by interactive_menu or caller
            run_id = run_info['run_id']
            run_number = run_info.get('run_number', 0)
            run_id_full = run_info.get('run_id_full', run_id)
            timestamp = run_info.get('timestamp', timestamp)
            
            # Update statistics with run information
            preprocess_statistics["run_id"] = run_id_full
            preprocess_statistics["run_number"] = run_number
            preprocess_statistics["run_info"] = run_info
        
        else:
            # Fallback: Generate run info locally
            # Get next sequential run number for this output directory
            def get_next_run_number(output_dir: Path) -> int:
                """Get the next sequential run number for the tracking directory."""
                run_tracker_file = output_dir / ".preprocessor_run_tracker"
                max_retries = 3
                retry_delay = 0.1
                
                if not run_tracker_file.exists():
                    with open(run_tracker_file, 'w') as f:
                        json.dump({'last_run': 0, 'runs': {}}, f)
                    return 1
                
                for attempt in range(max_retries):
                    try:
                        with open(run_tracker_file, 'r') as f:
                            content = f.read().strip()
                            if not content:
                                tracker = {'last_run': 0, 'runs': {}}
                            else:
                                tracker = json.loads(content)
                        
                        if not isinstance(tracker, dict):
                            raise ValueError("Tracker is not a dictionary")
                        
                        next_run = tracker.get('last_run', 0) + 1
                        
                        tracker['last_run'] = next_run
                        if 'runs' not in tracker:
                            tracker['runs'] = {}
                        
                        tracker['runs'][str(next_run)] = {
                            'timestamp': timestamp,
                            'started': start_time.isoformat(),
                            'filepath': filepath,
                            'output_dir': str(output_dir),
                            'config_path': config_path
                        }
                        
                        temp_file = run_tracker_file.with_suffix('.tmp')
                        with open(temp_file, 'w') as f:
                            json.dump(tracker, f, indent=2)
                        
                        temp_file.replace(run_tracker_file)
                        
                        return next_run
                    
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            self.print_color(f"\nTracker file corrupted (attempt {attempt + 1}), retrying...", 'warning')
                            time.sleep(retry_delay)
                        else:
                            self.print_color(f"\nTracker file corrupted: {str(e)}", 'error')
                            backup_file = run_tracker_file.with_suffix(f'.corrupted_{int(time.time())}')
                            try:
                                shutil.copy2(run_tracker_file, backup_file)
                                self.print_color(f"Backup created: {backup_file}", 'info')
                            except Exception:
                                pass
                            
                            existing_runs = len([d for d in output_dir.parent.iterdir() if d.is_dir() and d.name.startswith('run_')])
                            next_run = existing_runs + 1
                            
                            new_tracker = {
                                'last_run': next_run,
                                'runs': {
                                    str(next_run): {
                                        'timestamp': timestamp,
                                        'started': start_time.isoformat(),
                                        'filepath': filepath,
                                        'output_dir': str(output_dir),
                                        'config_path': config_path,
                                        'note': 'Tracker reinitialized due to corruption'
                                    }
                                }
                            }
                            
                            with open(run_tracker_file, 'w') as f:
                                json.dump(new_tracker, f, indent=2)
                            
                            self.print_color(f"\nTracker reinitialized with run number: {next_run}", 'success')
                            return next_run
                    
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.print_color(f"\nError reading tracker (attempt {attempt + 1}): {str(e)}", 'warning')
                            time.sleep(retry_delay)
                        else:
                            self.print_color(f"\nFailed to read run tracker: {str(e)}", 'error')
                            existing_runs = len([d for d in output_dir.parent.iterdir() if d.is_dir() and d.name.startswith('run_')])
                            return existing_runs + 1
                
                return 1
            
            # Generate sequential run ID "run_001"
            run_number = get_next_run_number(output_path)
            run_id = f"run_{run_number:03d}"
            
            # Generate full tracking ID with all details for metadata only
            process_id = os.getpid()
            unique_hash = hashlib.md5(
                f"{timestamp}_{process_id}".encode()
            ).hexdigest()[:4]
            
            # Full ID stored only in metadata, not used for file/directory names
            run_id_full = f"run_{run_number:03d}_{timestamp}_{unique_hash}"
            
            # Create run_info dictionary
            run_info = {
                'run_id': run_id,
                'run_id_full': run_id_full,
                'run_number': run_number,
                'timestamp': timestamp,
                'start_time': start_time.isoformat(),
                'filepath': filepath,
                'output_dir': str(output_path),
                'config_path': config_path
            }
        
        # Create preprocessing-specific output directory using simple run_id
        preprocess_output_dir = output_path / run_id
        preprocess_output_dir.mkdir(parents=True, exist_ok=True)

        # Update statistics with run information
        preprocess_statistics["run_id"] = run_id_full
        preprocess_statistics["run_number"] = run_number
        preprocess_statistics["run_info"] = run_info
        preprocess_statistics["output_directory"] = str(preprocess_output_dir)
        
        self.print_color("\n" + "-" * 40, 'highlight')
        self.print_color("PREPROCESSING PIPELINE", 'highlight')
        self.print_color("-" * 40, 'highlight')
        
        # Display run header
        if run_info and 'run_id' in run_info:
            self.print_color(f"\nPREPROCESSING RUN: {Fore.YELLOW + Style.BRIGHT}{run_number:03d}", 'highlight')
            # Extract timestamp from run_id_full if available
            if 'run_id_full' in run_info:
                parts = run_info['run_id_full'].split('_')
                if len(parts) >= 4:
                    date_part = parts[2]   # YYYYMMDD
                    time_part = parts[3]   # HHMMSS
                    formatted_timestamp = (f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}")
                    self.print_color(f"Timestamp: {Fore.YELLOW + Style.BRIGHT}{formatted_timestamp}", 'highlight')
            # Extract hash from run_id_full if available
            if 'run_id_full' in run_info and len(run_info['run_id_full'].split('_')) >= 5:
                unique_hash = run_info['run_id_full'].split('_')[-1]
                self.print_color(f"Unique Hash: {Fore.YELLOW + Style.BRIGHT}{unique_hash}", 'highlight')
            self.print_color(f"Output Directory: {Fore.CYAN + Style.BRIGHT}{preprocess_output_dir}", 'highlight')
            self.print_color("-" * 40, 'highlight')
        
        # Update tracker with start of processing
        self._update_run_tracker(output_path.parent, run_number, 'preprocessing_started', {
            'type': 'preprocessing',
            'timestamp': start_time.isoformat()
        })

        # Check if config_path points to a specific run directory
        config_file = Path(config_path)
        selected_run = None
        skip_interactive_selection = False
        
        # Skip interactive selection if config_path is a specific run directory
        if config_file.is_dir() and config_file.name.startswith('run_'):
            skip_interactive_selection = True
            run_id = config_file.name
            summary_file = config_file / f"max_rows_testing_summary_{run_id}.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        config = json.load(f)
                    
                    selected_run = {
                        'run_id': run_id,
                        'run_number': config.get('run_number', 'Unknown'),
                        'timestamp': config.get('timestamp', 'Unknown'),
                        'filepath': config['dataset_info']['filepath'],
                        'total_rows': config['dataset_info']['total_rows'],
                        'processed_rows': config['dataset_info']['processed_rows'],
                        'max_chunk_size': config['performance_metrics']['max_chunk_size'],
                        'average_throughput': config['performance_metrics']['average_throughput'],
                        'max_cpu_usage': config['performance_metrics'].get('max_cpu_usage', 'N/A'),
                        'run_directory': str(config_file),
                        'summary_file': str(summary_file)
                    }
                    
                    self.print_color(f"\nUsing test configuration from {Fore.YELLOW + Style.BRIGHT}{run_id}", 'success')
                    self.print_color(f"  ├─ Run #{Fore.YELLOW + Style.BRIGHT}{selected_run['run_number']}", 'success')
                    self.print_color(f"  ├─ Timestamp: {Fore.GREEN + Style.BRIGHT}{selected_run['timestamp']}", 'success')
                    self.print_color(f"  └─ Config source: {Fore.MAGENTA + Style.BRIGHT}{str(config_file)}", 'success')
                
                except Exception as e:
                    self.print_color(f"\nError reading run configuration: {str(e)}", 'warning')
                    skip_interactive_selection = False  # Fall back to interactive selection
        
        # Allow user to select specific test run if interactive and don't have a specific run already
        if not skip_interactive_selection and interactive:
            self.print_color("\nLooking for system capacity test results...", 'warning')
            
            selected_run = self.select_test_run_interactively(config_path)
            
            if selected_run:
                config_path = selected_run['summary_file']
                self.print_color(f"\nUsing configuration from run: {Fore.YELLOW + Style.BRIGHT}{selected_run['run_id']}", 'success')
            else:
                self.print_color("\nNo specific run selected, searching for latest...", 'info')
        
        max_rows, chunk_size = self.get_memory_config(config_path, interactive)
        
        # Update statistics with memory configuration
        preprocess_statistics["memory_config"]["max_rows"] = max_rows
        preprocess_statistics["memory_config"]["chunk_size"] = chunk_size
        preprocess_statistics["total_rows"] = max_rows
        preprocess_statistics["status"] = "configuration_loaded"
        
        # Update tracker with configuration loaded
        self._update_run_tracker(output_path.parent, run_number, 'configuration_loaded', {
            'max_rows': max_rows,
            'chunk_size': chunk_size,
            'timestamp': datetime.now().isoformat()
        })
        
        self.print_color("\n" + "-" * 40, 'info')
        self.print_color("PROCESSING CONFIGURATION", 'info')
        self.print_color("-" * 40, 'info')
        
        self.print_color(f"\nInput file: {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'info')
        self.print_color(f"Output directory: {Fore.MAGENTA + Style.BRIGHT}{preprocess_output_dir}", 'info')
        self.print_color(f"Processing limit: {Fore.YELLOW + Style.BRIGHT}{max_rows:,} rows", 'info')
        self.print_color(f"Chunk size: {Fore.GREEN + Style.BRIGHT}{chunk_size:,} rows", 'info')
        self.print_color(f"Target feature count: {Fore.YELLOW + Style.BRIGHT}{self.config['preprocessing_config']['hybrid_feature_count']}", 'info')
        self.print_color(f"Memory safety factor: {Fore.GREEN + Style.BRIGHT}{self.config['preprocessing_config']['memory_safety_factor']}", 'info')
        
        if selected_run:
            self.print_color(f"Config source: Run {selected_run['run_id']} (#{selected_run['run_number']})", 'success')
        else:
            self.print_color(f"Config source: {Fore.MAGENTA + Style.BRIGHT}{config_path}", 'info')
        
        # Verify file exists
        if not Path(filepath).exists():
            self.print_color(f"\nError: Input file not found: {Fore.YELLOW + Style.BRIGHT}{filepath}", 'error')
            preprocess_statistics["status"] = "failed_file_not_found"
            preprocess_statistics["end_time"] = datetime.now().isoformat()
            self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
            self._update_run_tracker(output_path.parent, run_number, 'failed_file_not_found', {
                'error': f"File not found: {filepath}",
                'timestamp': datetime.now().isoformat()
            })
            return

        # Load feature descriptions after file verification
        if not Path(filepath).exists():
            self.print_color(f"\nInput file not found: {filepath}", 'error')
            return
        
        # Load feature descriptions
        feature_descriptions = self.load_feature_descriptions(filepath, interactive=interactive)
        
        # Initialize encoders on a larger sample to reduce chance of unseen labels
        self.print_color("\n" + "-" * 40, 'info')
        self.print_color("INITIALIZING ENCODERS", 'info')
        self.print_color("-" * 40, 'info')
        
        sample_size = min(50000, chunk_size)
        self.print_color(f"\nLoading sample data ({sample_size:,} rows)...", 'warning')
        
        try:
            sample_df = pd.read_csv(
                filepath,
                nrows=sample_size,
                low_memory=False,
                dtype={col: "category" for col in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]},
                on_bad_lines='warn'
            )
            
            self.print_color(f"\nSample loaded: {Fore.YELLOW + Style.BRIGHT}{len(sample_df):,} rows, {len(sample_df.columns)} columns{Style.RESET_ALL}", 'success')
            
        except Exception as e:
            self.print_color(f"\nError loading sample data: {str(e)}", 'error')
            preprocess_statistics["status"] = "failed_sample_loading"
            preprocess_statistics["end_time"] = datetime.now().isoformat()
            self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
            self._update_run_tracker(output_path.parent, run_number, 'failed_sample_loading', {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return
        
        encoders = {}
        ip_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
        
        for col in ip_cols:
            if col in sample_df.columns:
                le = LabelEncoder()
                le.fit(sample_df[col])
                encoders[col] = le
                joblib.dump(le, preprocess_output_dir / f"{col}_label_encoder.pkl")
                self.print_color(f"  ├─ {col} encoder: {Fore.YELLOW + Style.BRIGHT}{len(le.classes_)} unique values", 'success')
        
        categorical_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()
        other_categorical = list(set(categorical_cols) - set(ip_cols))
        
        if other_categorical:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(sample_df[other_categorical])
            encoders['one_hot'] = encoder
            joblib.dump(encoder, preprocess_output_dir / "one_hot_encoder.pkl")
            self.print_color(f"  ├─ One-hot encoder: {Fore.YELLOW + Style.BRIGHT}{len(other_categorical)} categorical columns", 'success')
        
        self.print_color(f"  └─ Encoders saved to: {Fore.MAGENTA + Style.BRIGHT}{preprocess_output_dir}", 'success')
        
        # Update statistics
        preprocess_statistics["status"] = "encoders_initialized"
        self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
        
        # Update tracker with encoders initialized
        self._update_run_tracker(output_path.parent, run_number, 'encoders_initialized', {
            'timestamp': datetime.now().isoformat()
        })
        
        # Start chunked processing
        self.print_color("\n" + "-" * 40, 'info')
        self.print_color("CHUNKED PROCESSING", 'info')
        self.print_color("-" * 40, 'info')
        
        try:
            self.print_color(f"\nStarting preprocessing with {max_rows:,} row limit...", 'warning')
            
            chunk_reader = pd.read_csv(
                filepath,
                chunksize=chunk_size,
                nrows=max_rows,
                low_memory=False,
                dtype={col: "category" for col in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]},
                on_bad_lines='warn'
            )
            
            processed_chunks = []
            selected_features = []
            scaler = None
            total_rows = 0
            chunk_index = 0
            processing_times = []
            rows_per_second_history = []
            
            # Process with alive_progress bar
            with alive_bar(title='Processing chunks', bar='smooth', spinner='dots_waves2', length=35, unit='chunks', stats=True, enrich_print=False) as bar:
                
                for df_chunk in chunk_reader:
                    chunk_index += 1
                    chunk_start_row = total_rows
                    chunk_end_row = chunk_start_row + len(df_chunk)
                    chunk_start_time = time.time()
                    
                    # Update progress bar message
                    bar.text = f"Chunk {chunk_index} | Rows {chunk_start_row:,}-{chunk_end_row:,} | Size: {len(df_chunk):,} rows"
                    
                    # Update tracker with chunk start
                    self._update_run_tracker(output_path.parent, run_number, 'chunk_started', {
                        'chunk_index': chunk_index,
                        'chunk_size': len(df_chunk),
                        'current_row': chunk_start_row,
                        'chunk_end': chunk_end_row,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    try:
                        df_processed, features, scaler = self.process_chunk_preprocessing(
                            df_chunk,
                            preprocess_output_dir,
                            encoders,
                            scaler
                        )
                        
                        if not selected_features:
                            selected_features = features
                        
                        processed_chunks.append(df_processed)
                        rows_processed = len(df_processed)
                        total_rows += rows_processed
                        
                        # Calculate chunk performance metrics
                        chunk_processing_time = time.time() - chunk_start_time
                        processing_times.append(chunk_processing_time)
                        
                        if chunk_processing_time > 0:
                            rows_per_second = rows_processed / chunk_processing_time
                            rows_per_second_history.append(rows_per_second)
                        else:
                            rows_per_second = 0
                        
                        # Update statistics
                        preprocess_statistics["successful_chunks"] += 1
                        preprocess_statistics["total_chunks"] += 1
                        preprocess_statistics["processed_rows"] = total_rows
                        
                        # Update performance metrics
                        if processing_times:
                            preprocess_statistics["performance_metrics"]["average_processing_time"] = sum(processing_times) / len(processing_times)
                            preprocess_statistics["performance_metrics"]["total_processing_time"] = sum(processing_times)
                        
                        if rows_per_second_history:
                            preprocess_statistics["performance_metrics"]["average_rows_per_second"] = sum(rows_per_second_history) / len(rows_per_second_history)
                            preprocess_statistics["performance_metrics"]["max_rows_per_second"] = max(rows_per_second_history)
                            preprocess_statistics["performance_metrics"]["min_rows_per_second"] = min(rows_per_second_history)
                        
                        # Track chunk in history
                        preprocess_statistics["chunk_history"].append({
                            "chunk_index": chunk_index,
                            "status": "success",
                            "rows_processed": rows_processed,
                            "processing_time_sec": chunk_processing_time,
                            "rows_per_second": rows_per_second,
                            "features_count": len(features),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Update tracker with chunk success
                        self._update_run_tracker(output_path.parent, run_number, 'chunk_completed', {
                            'chunk_index': chunk_index,
                            'rows_processed': rows_processed,
                            'processing_time_sec': chunk_processing_time,
                            'rows_per_second': rows_per_second,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Update progress bar with processing stats
                        bar.text = f"Chunk {chunk_index}: {rows_processed:,} rows | Total: {total_rows:,} | Features: {len(features)}"
                        
                        if verbose and chunk_index <= 3:  # Only show first 3 chunks if verbose
                            self.print_color(f"\n [+] Chunk {chunk_index} Details:", 'success')
                            self.print_color(f"    ├─ Rows processed: {Fore.YELLOW + Style.BRIGHT}{rows_processed:,}", 'success')
                            self.print_color(f"    ├─ Features: {Fore.YELLOW + Style.BRIGHT}{len(features)}", 'success')
                            self.print_color(f"    └─ Sample: {Fore.YELLOW + Style.BRIGHT}{features[:5]}{'...' if len(features) > 5 else ''}", 'success')
                    
                    except Exception as e:
                        bar.text = f"Chunk {chunk_index}: Failed - {str(e)[:30]}..."
                        self.print_color(f"\nFailed to process chunk {chunk_index}: {str(e)}", 'error')
                        
                        # Update statistics
                        preprocess_statistics["failed_chunks"] += 1
                        preprocess_statistics["total_chunks"] += 1
                        
                        preprocess_statistics["chunk_history"].append({
                            "chunk_index": chunk_index,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Update tracker with chunk failure
                        self._update_run_tracker(output_path.parent, run_number, 'chunk_failed', {
                            'chunk_index': chunk_index,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        if interactive:
                            max_attempts = 3
                            for attempt in range(max_attempts):
                                try:
                                    response = input(Fore.YELLOW + Style.BRIGHT + "\nContinue processing? (Y/n): " + Style.RESET_ALL).strip().lower()
                                    if response in ['y', 'yes', '']:
                                        break
                                    elif response in ['n', 'no']:
                                        self.print_color("\nProcessing stopped by user", 'warning')
                                        preprocess_statistics["status"] = "stopped_by_user"
                                        preprocess_statistics["end_time"] = datetime.now().isoformat()
                                        self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
                                        self._update_run_tracker(output_path.parent, run_number, 'stopped_by_user', {
                                            'timestamp': datetime.now().isoformat()
                                        })
                                        return
                                    else:
                                        if attempt < max_attempts - 1:
                                            self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                                except KeyboardInterrupt:
                                    self.print_color("\nProcessing stopped", 'warning')
                                    preprocess_statistics["status"] = "stopped_by_user"
                                    preprocess_statistics["end_time"] = datetime.now().isoformat()
                                    self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
                                    self._update_run_tracker(output_path.parent, run_number, 'stopped_by_user', {
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    return
                        else:
                            # Skip failed chunk and continue
                            bar.text = f"Chunk {chunk_index}: Skipped due to error"
                            continue
                    
                    # Update progress bar
                    bar()
                    
                    # Periodically save statistics (every 10 chunks)
                    if chunk_index % 10 == 0:
                        self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
                        
                        # Update progress in tracker
                        progress_percent = (total_rows / max_rows) * 100 if max_rows > 0 else 0
                        self._update_run_tracker(output_path.parent, run_number, 'progress_update', {
                            'current_row': total_rows,
                            'total_rows': max_rows,
                            'progress_percent': progress_percent,
                            'chunks_completed': chunk_index,
                            'timestamp': datetime.now().isoformat()
                        })
        
        except ImportError:
            # Fallback to manual progress tracking if alive_progress is not available
            self.print_color("\nNote: Install 'alive-progress' for progress tracking", 'warning')
            self.print_color("Continuing with basic progress reporting...\n", 'warning')
            
            chunk_reader = pd.read_csv(
                filepath,
                chunksize=chunk_size,
                nrows=max_rows,
                low_memory=False,
                dtype={col: "category" for col in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]},
                on_bad_lines='warn'
            )
            
            processed_chunks = []
            selected_features = []
            scaler = None
            total_rows = 0
            chunk_index = 0
            processing_times = []
            rows_per_second_history = []
            
            for df_chunk in chunk_reader:
                chunk_index += 1
                chunk_start_row = total_rows
                chunk_end_row = chunk_start_row + len(df_chunk)
                chunk_start_time = time.time()
                
                self.print_color(f"\nChunk {chunk_index}: Processing rows {chunk_start_row:,}-{chunk_end_row:,} ({len(df_chunk):,} rows)...", 'warning')
                
                # Update tracker with chunk start
                self._update_run_tracker(output_path.parent, run_number, 'chunk_started', {
                    'chunk_index': chunk_index,
                    'chunk_size': len(df_chunk),
                    'current_row': chunk_start_row,
                    'chunk_end': chunk_end_row,
                    'timestamp': datetime.now().isoformat()
                })
                
                try:
                    df_processed, features, scaler = self.process_chunk_preprocessing(
                        df_chunk,
                        preprocess_output_dir,
                        encoders,
                        scaler
                    )
                    
                    if not selected_features:
                        selected_features = features
                    
                    processed_chunks.append(df_processed)
                    rows_processed = len(df_processed)
                    total_rows += rows_processed
                    
                    # Calculate performance metrics
                    chunk_processing_time = time.time() - chunk_start_time
                    processing_times.append(chunk_processing_time)
                    
                    if chunk_processing_time > 0:
                        rows_per_second = rows_processed / chunk_processing_time
                        rows_per_second_history.append(rows_per_second)
                    else:
                        rows_per_second = 0
                    
                    # Update statistics
                    preprocess_statistics["successful_chunks"] += 1
                    preprocess_statistics["total_chunks"] += 1
                    preprocess_statistics["processed_rows"] = total_rows
                    
                    # Track chunk in history
                    preprocess_statistics["chunk_history"].append({
                        "chunk_index": chunk_index,
                        "status": "success",
                        "rows_processed": rows_processed,
                        "processing_time_sec": chunk_processing_time,
                        "rows_per_second": rows_per_second,
                        "features_count": len(features),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update tracker with chunk success
                    self._update_run_tracker(output_path.parent, run_number, 'chunk_completed', {
                        'chunk_index': chunk_index,
                        'rows_processed': rows_processed,
                        'processing_time_sec': chunk_processing_time,
                        'rows_per_second': rows_per_second,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Basic progress display
                    progress_percent = (total_rows / max_rows) * 100 if max_rows > 0 else 0
                    self.print_color(f"  [+] Completed: {rows_processed:,} rows | Total: {total_rows:,} | Progress: {progress_percent:.1f}%", 'success')
                    
                    if verbose and chunk_index <= 3:  # Only show first 3 chunks if verbose
                        self.print_color(f"    ├─ Features: {Fore.YELLOW + Style.BRIGHT}{len(features)}", 'success')
                        self.print_color(f"    └─ Sample: {Fore.YELLOW + Style.BRIGHT}{features[:5]}{'...' if len(features) > 5 else ''}", 'success')
                
                except Exception as e:
                    self.print_color(f"\nFailed to process chunk {chunk_index}: {str(e)}", 'error')
                    
                    preprocess_statistics["failed_chunks"] += 1
                    preprocess_statistics["total_chunks"] += 1
                    
                    preprocess_statistics["chunk_history"].append({
                        "chunk_index": chunk_index,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update tracker with chunk failure
                    self._update_run_tracker(output_path.parent, run_number, 'chunk_failed', {
                        'chunk_index': chunk_index,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    if interactive:
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                response = input(Fore.YELLOW + Style.BRIGHT + "\nContinue processing? (Y/n): " + Style.RESET_ALL).strip().lower()
                                if response in ['y', 'yes', '']:
                                    break
                                elif response in ['n', 'no']:
                                    self.print_color("\nProcessing stopped by user", 'warning')
                                    preprocess_statistics["status"] = "stopped_by_user"
                                    preprocess_statistics["end_time"] = datetime.now().isoformat()
                                    self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
                                    self._update_run_tracker(output_path.parent, run_number, 'stopped_by_user', {
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    return
                                else:
                                    if attempt < max_attempts - 1:
                                        self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                            except KeyboardInterrupt:
                                self.print_color("\nProcessing stopped", 'warning')
                                preprocess_statistics["status"] = "stopped_by_user"
                                preprocess_statistics["end_time"] = datetime.now().isoformat()
                                self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
                                self._update_run_tracker(output_path.parent, run_number, 'stopped_by_user', {
                                    'timestamp': datetime.now().isoformat()
                                })
                                return
                    else:
                        continue
                
                # Periodically save statistics (every 10 chunks)
                if chunk_index % 10 == 0:
                    self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
                    
                    # Update progress in tracker
                    progress_percent = (total_rows / max_rows) * 100 if max_rows > 0 else 0
                    self._update_run_tracker(output_path.parent, run_number, 'progress_update', {
                        'current_row': total_rows,
                        'total_rows': max_rows,
                        'progress_percent': progress_percent,
                        'chunks_completed': chunk_index,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Update final performance metrics
            if processing_times:
                preprocess_statistics["performance_metrics"]["average_processing_time"] = sum(processing_times) / len(processing_times)
                preprocess_statistics["performance_metrics"]["total_processing_time"] = sum(processing_times)
            
            if rows_per_second_history:
                preprocess_statistics["performance_metrics"]["average_rows_per_second"] = sum(rows_per_second_history) / len(rows_per_second_history)
                preprocess_statistics["performance_metrics"]["max_rows_per_second"] = max(rows_per_second_history)
                preprocess_statistics["performance_metrics"]["min_rows_per_second"] = min(rows_per_second_history)
        
        if not processed_chunks:
            self.print_color("\nNo data was processed successfully", 'error')
            preprocess_statistics["status"] = "failed_no_data"
            preprocess_statistics["end_time"] = datetime.now().isoformat()
            self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
            self._update_run_tracker(output_path.parent, run_number, 'failed_no_data', {
                'timestamp': datetime.now().isoformat()
            })
            return
        
        # Save artifacts
        artifacts = {
            "scaler": scaler,
            "feature_names": selected_features,
            "feature_descriptions": feature_descriptions,
            "feature_descriptions_file": str(Path(filepath).parent / f"{Path(filepath).stem}_Features.csv") if feature_descriptions else None,
            "total_rows": total_rows,
            "chunks_processed": chunk_index,
            "original_features": list(set(
                sample_df.select_dtypes(include=["object", "category", "int64", "float64"]).columns.tolist()
            )),
            "hybrid_feature_count": self.config['preprocessing_config']['hybrid_feature_count'],
            "input_size": len(selected_features),
            "config_source": config_path,
            "processing_timestamp": datetime.now().isoformat(),
            "memory_config": {
                "max_rows": max_rows,
                "chunk_size": chunk_size,
                "safety_factor": self.config['preprocessing_config']['memory_safety_factor']
            },
            "run_id": run_id_full,
            "run_number": run_number
        }
        
        artifacts_path = preprocess_output_dir / "preprocessing_artifacts.pkl"
        joblib.dump(artifacts, artifacts_path)
        
        # Save final dataframe
        final_df = pd.concat(processed_chunks)
        processed_path = preprocess_output_dir / "preprocessed_dataset.csv"
        final_df.to_csv(processed_path, index=False)
        
        # Update final statistics
        end_time = datetime.now()
        preprocess_statistics["status"] = "completed"
        preprocess_statistics["end_time"] = end_time.isoformat()
        preprocess_statistics["elapsed_time_seconds"] = (end_time - start_time).total_seconds()
        preprocess_statistics["final_feature_count"] = len(selected_features)
        preprocess_statistics["completion_status"] = "complete" if total_rows > 0 else "failed"
        
        # Save final statistics
        self._save_run_statistics(preprocess_output_dir, preprocess_statistics)
        
        # Save configuration summary
        summary = {
            "preprocessing_summary": {
                "version": self.VERSION,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id_full,
                "run_number": run_number,
                "input_file": filepath,
                "output_directory": str(preprocess_output_dir),
                "total_rows_processed": total_rows,
                "chunks_processed": chunk_index,
                "successful_chunks": preprocess_statistics["successful_chunks"],
                "failed_chunks": preprocess_statistics["failed_chunks"],
                "final_feature_count": len(selected_features),
                "performance_metrics": preprocess_statistics["performance_metrics"],
                "memory_config": {
                    "max_rows": max_rows,
                    "chunk_size": chunk_size,
                    "safety_factor": self.config['preprocessing_config']['memory_safety_factor']
                },
                "artifacts_saved": [
                    str(artifacts_path),
                    str(processed_path),
                    str(preprocess_output_dir / "IPV4_SRC_ADDR_label_encoder.pkl"),
                    str(preprocess_output_dir / "IPV4_DST_ADDR_label_encoder.pkl"),
                    str(preprocess_output_dir / "one_hot_encoder.pkl")
                ],
                "config_source": config_path,
                "elapsed_time_seconds": preprocess_statistics["elapsed_time_seconds"],
                "run_info": run_info
            }
        }
        
        summary_path = preprocess_output_dir / f"preprocessing_summary_{run_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update tracker with completion
        self._update_run_tracker(output_path.parent, run_number, 'completed', {
            'end_time': end_time.isoformat(),
            'elapsed_seconds': (end_time - start_time).total_seconds(),
            'total_rows_processed': total_rows,
            'successful_chunks': preprocess_statistics["successful_chunks"],
            'failed_chunks': preprocess_statistics["failed_chunks"],
            'completion_status': "complete" if total_rows > 0 else "failed",
            'timestamp': datetime.now().isoformat()
        })
        
        # Display results
        self.print_color("\n" + "-" * 40, 'success')
        self.print_color("PREPROCESSING COMPLETE", 'success')
        self.print_color("-" * 40, 'success')
        
        self.print_color(f"\nResults Summary:", 'success')
        self.print_color(f"  ├─ Total rows processed: {Fore.YELLOW + Style.BRIGHT}{total_rows:,}", 'success')
        self.print_color(f"  ├─ Chunks processed: {Fore.YELLOW + Style.BRIGHT}{chunk_index}", 'success')
        self.print_color(f"  ├─ Final feature count: {Fore.CYAN + Style.BRIGHT}{len(selected_features)}", 'success')
        
        if feature_descriptions:
            desc_coverage = sum(1 for f in selected_features if f in feature_descriptions)
            self.print_color(f"  ├─ Feature descriptions: {Fore.GREEN + Style.BRIGHT}{desc_coverage}/{len(selected_features)} available", 'success')
        
        self.print_color(f"  ├─ Output directory: {Fore.CYAN + Style.BRIGHT}{preprocess_output_dir.resolve()}", 'success')
        self.print_color(f"  ├─ Processed dataset: {Fore.MAGENTA + Style.BRIGHT}{processed_path.resolve()}", 'success')
        self.print_color(f"  ├─ Artifacts file: {Fore.MAGENTA + Style.BRIGHT}{artifacts_path.resolve()}", 'success')
        self.print_color(f"  └─ Summary file: {Fore.MAGENTA + Style.BRIGHT}{summary_path.resolve()}", 'success')
        
        if verbose and feature_descriptions:
            self.display_features_with_descriptions(selected_features, feature_descriptions)
        elif verbose:
            self.print_color(f"\nSample features:", 'info')
            for i, feature in enumerate(selected_features[:10], 1):
                self.print_color(f"  {i:2d}. {feature}", 'info')
            if len(selected_features) > 10:
                self.print_color(f"  ... and {len(selected_features) - 10} more", 'warning')
        
        self._log_event(f"Preprocessing completed: {total_rows:,} rows, {chunk_index} chunks, {len(selected_features)} features", "info")

    def configure_settings(self) -> None:
        """Configure settings interactively."""
        self.print_color("\n" + "-" * 40, 'highlight')
        self.print_color("CONFIGURE SETTINGS", 'highlight')
        self.print_color("-" * 40, 'highlight')
        
        self.print_color("\nCurrent Configuration:", 'info')
        for key, value in self.config.items():
            if key != 'recent_files' and key != 'preprocessing_config':
                prefix = "  ├─" if key != list(self.config.keys())[-2] else "  └─"
                self.print_color(f"{prefix} {key.replace('_', ' ').title():<25}: {value}", 'info')
        
        self.print_color("\nPreprocessing Configuration:", 'info')
        for key, value in self.config['preprocessing_config'].items():
            prefix = "  ├─" if key != list(self.config['preprocessing_config'].keys())[-1] else "  └─"
            self.print_color(f"{prefix} {key.replace('_', ' ').title():<25}: {value}", 'info')
        
        self.print_color("\nEnter new values (press Enter to keep current):", 'warning')
        
        # Get new output directory for testing
        new_output = self.config['default_output_dir']
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nNew test output directory ({self.config['default_output_dir']}): " + Style.RESET_ALL).strip()
                
                if not user_input:
                    break
                
                # Basic path validation
                invalid_chars = '<>:"|?*'
                if any(char in user_input for char in invalid_chars):
                    self.print_color("\nInvalid characters in path.", 'error')
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise ValueError("\nInvalid path characters")
                
                # Check if parent directory exists
                dir_path = Path(user_input)
                parent = dir_path.parent
                
                try:
                    if not parent.exists():
                        parent.mkdir(parents=True, exist_ok=True)
                        if parent != dir_path:
                            try:
                                parent.rmdir()
                            except:
                                pass
                    new_output = user_input
                    break
                except Exception as dir_error:
                    if attempt < max_attempts - 1:
                        self.print_color(f"\nInvalid directory: {Fore.YELLOW + Style.BRIGHT}{str(dir_error)}{Style.RESET_ALL}", 'error')
                        continue
                    else:
                        raise ValueError(f"\nInvalid directory: {Fore.YELLOW + Style.BRIGHT}{str(dir_error)}{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                self.print_color("\nInput cancelled", 'warning')
                break
            except Exception as input_error:
                if attempt < max_attempts - 1:
                    self.print_color(f"\nInput error: {str(input_error)}", 'error')
        
        # Get new target memory
        new_target = None
        for attempt in range(max_attempts):
            try:
                user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nNew target memory (MB) [{self.config['default_target_mb']}]: " + Style.RESET_ALL).strip()
                
                if not user_input:
                    break
                
                if not user_input.isdigit():
                    self.print_color("\nPlease enter a number.", 'error')
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise ValueError("\nInvalid number")
                
                value = int(user_input)
                if not (1 <= value <= 16384):
                    self.print_color("\nPlease enter a number between 1 and 16384.", 'error')
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise ValueError("\nValue out of range")
                
                new_target = value
                break
            
            except (EOFError, KeyboardInterrupt):
                self.print_color("\nTarget memory input cancelled.", 'warning')
                break
            except Exception as input_error:
                if attempt < max_attempts - 1:
                    self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
        
        # Get new preprocessing output directory
        new_preprocessing_output = self.config['preprocessing_config']['default_output_dir']
        for attempt in range(max_attempts):
            try:
                user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nNew preprocessing output directory ({self.config['preprocessing_config']['default_output_dir']}): " + Style.RESET_ALL).strip()
                
                if not user_input:
                    break
                
                # Basic path validation
                invalid_chars = '<>:"|?*'
                if any(char in user_input for char in invalid_chars):
                    self.print_color("\nInvalid characters in path.", 'error')
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise ValueError("\nInvalid path characters")
                
                # Check if parent directory exists
                dir_path = Path(user_input)
                parent = dir_path.parent
                
                try:
                    if not parent.exists():
                        parent.mkdir(parents=True, exist_ok=True)
                        if parent != dir_path:
                            try:
                                parent.rmdir()
                            except:
                                pass
                    new_preprocessing_output = user_input
                    break
                except Exception as dir_error:
                    if attempt < max_attempts - 1:
                        self.print_color(f"\nInvalid directory: {Fore.YELLOW + Style.BRIGHT}{str(dir_error)}{Style.RESET_ALL}", 'error')
                        continue
                    else:
                        raise ValueError(f"\nInvalid directory: {Fore.YELLOW + Style.BRIGHT}{str(dir_error)}{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                self.print_color("\nInput cancelled", 'warning')
                break
            except Exception as input_error:
                if attempt < max_attempts - 1:
                    self.print_color(f"\nInput error: {str(input_error)}", 'error')
        
        # Get new memory safety factor
        new_safety = None
        for attempt in range(max_attempts):
            try:
                user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nNew memory safety factor (0.1-0.9) [{self.config['preprocessing_config']['memory_safety_factor']}]: " + Style.RESET_ALL).strip()
                
                if not user_input:
                    break
                
                value = float(user_input)
                if not (0.1 <= value <= 0.9):
                    self.print_color("\nPlease enter a value between 0.1 and 0.9.", 'error')
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise ValueError("\nValue out of range")
                
                new_safety = value
                break
            
            except ValueError:
                self.print_color("\nInvalid number format.", 'error')
                if attempt < max_attempts - 1:
                    continue
                else:
                    break
            except KeyboardInterrupt:
                self.print_color("\nInput cancelled", 'warning')
                break
        
        # Get new hybrid feature count
        new_features = None
        for attempt in range(max_attempts):
            try:
                user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nNew hybrid feature count (1-100) [{self.config['preprocessing_config']['hybrid_feature_count']}]: " + Style.RESET_ALL).strip()
                
                if not user_input:
                    break
                
                value = int(user_input)
                if not (1 <= value <= 100):
                    self.print_color("\nPlease enter a value between 1 and 100.", 'error')
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise ValueError("\nValue out of range")
                
                new_features = value
                break
            
            except ValueError:
                self.print_color("\nInvalid number format.", 'error')
                if attempt < max_attempts - 1:
                    continue
                else:
                    break
            except KeyboardInterrupt:
                self.print_color("\nInput cancelled", 'warning')
                break
        
        # Show changes
        self.print_color("\nConfiguration Changes:", 'highlight')
        changes_made = False
        
        if new_output != self.config.get('default_output_dir'):
            self.print_color(f"\nNew Test output directory: {Fore.MAGENTA + Style.BRIGHT}{new_output}{Style.RESET_ALL}", 'success')
            changes_made = True
        
        if new_target is not None and new_target != self.config.get('default_target_mb'):
            self.print_color(f"\nNew Target memory: {Fore.CYAN + Style.BRIGHT}{new_target} MB{Style.RESET_ALL}", 'success')
            changes_made = True
        
        if new_preprocessing_output != self.config['preprocessing_config'].get('default_output_dir'):
            self.print_color(f"\nNew Preprocessing output directory: {Fore.MAGENTA + Style.BRIGHT}{new_preprocessing_output}{Style.RESET_ALL}", 'success')
            changes_made = True
        
        if new_safety is not None and new_safety != self.config['preprocessing_config'].get('memory_safety_factor'):
            self.print_color(f"\nNew Memory safety factor: {Fore.CYAN + Style.BRIGHT}{new_safety}{Style.RESET_ALL}", 'success')
            changes_made = True
        
        if new_features is not None and new_features != self.config['preprocessing_config'].get('hybrid_feature_count'):
            self.print_color(f"\nNew Hybrid feature count: {Fore.CYAN + Style.BRIGHT}{new_features}{Style.RESET_ALL}", 'success')
            changes_made = True
        
        if not changes_made:
            try:
                self.print_color("\nNo changes made.", 'warning')
            except (EOFError, KeyboardInterrupt):
                self.print_color("\nContinuing...", 'warning')
            return
        
        # Confirm save
        confirm_attempts = 3
        confirmed = False
        
        for attempt in range(confirm_attempts):
            try:
                confirm = input(Fore.YELLOW + Style.BRIGHT + "\nSave these changes? (Y/n): " + Style.RESET_ALL).strip().lower()
                
                if confirm in ['y', 'yes', '']:
                    confirmed = True
                    break
                elif confirm in ['n', 'no']:
                    self.print_color("\nChanges discarded.", 'warning')
                    break
                else:
                    if attempt < confirm_attempts - 1:
                        self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
            except (EOFError, KeyboardInterrupt):
                self.print_color("\nConfirmation cancelled.", 'warning')
                break
        
        if confirmed:
            if new_output != self.config.get('default_output_dir'):
                self.config['default_output_dir'] = new_output
            if new_target is not None:
                self.config['default_target_mb'] = new_target
            if new_preprocessing_output != self.config['preprocessing_config'].get('default_output_dir'):
                self.config['preprocessing_config']['default_output_dir'] = new_preprocessing_output
            if new_safety is not None:
                self.config['preprocessing_config']['memory_safety_factor'] = new_safety
            if new_features is not None:
                self.config['preprocessing_config']['hybrid_feature_count'] = new_features
            
            self.save_config(self.config)
            self.print_color("\nConfiguration updated successfully!", 'success')

    def interactive_menu(self) -> None:
        """Interactive console menu for the application."""
        config = self.load_config()
        datasets = self.get_available_datasets()
        
        while True:
            try:
                self.display_banner()
                self.display_main_menu()
                
                # Menu choice input with validation
                max_attempts = 3
                choice = None
                
                for attempt in range(max_attempts):
                    try:
                        choice_input = input(Fore.YELLOW + Style.BRIGHT + "\nSelect an option (0-5): " + Style.RESET_ALL).strip()
                        
                        if not choice_input:
                            self.print_color("\nPlease enter a choice.", 'warning')
                            continue
                            
                        if choice_input in ['0', '1', '2', '3', '4', '5']:
                            choice = choice_input
                            break
                        else:
                            if attempt < max_attempts - 1:
                                self.print_color(f"\nInvalid choice '{choice_input}'. Please enter 0-5.", 'error')
                    except (EOFError, KeyboardInterrupt):
                        self.print_color("\nInput cancelled. Returning to menu.", 'warning')
                        choice = '0'
                        break
                    except Exception as input_error:
                        if attempt < max_attempts - 1:
                            self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                
                if choice is None:
                    self.print_color("\nToo many invalid attempts. Returning to menu.", 'error')
                    continue
                
                if choice == '1':  # Run new test
                    try:
                        self.print_color("\nAvailable Datasets:", 'warning')
                        
                        if not datasets:
                            self.print_color("\nNo datasets found in 'datasets' directory.", 'warning')
                            self.print_color("Please place CSV files in the 'datasets' folder or enter a custom path.", 'warning')
                        
                        for i, dataset in enumerate(datasets, 1):
                            self.print_color(f"{i}. {dataset}", 'debug')
                        
                        last_option = len(datasets) + 1
                        self.print_color(f"{last_option}. Enter custom file path", 'info')
                        self.print_color("0. Return to main menu", 'error')
                        
                        # Get dataset choice
                        dataset_choice = None
                        filepath = None
                        
                        for attempt in range(max_attempts):
                            try:
                                prompt = Fore.YELLOW + Style.BRIGHT + f"\nSelect dataset (0-{last_option}): " + Style.RESET_ALL
                                choice_input = input(prompt).strip().lower()
                                
                                if choice_input == '0':
                                    self.print_color("\nReturning to main menu...", 'warning')
                                    break
                                
                                elif choice_input.isdigit():
                                    choice_num = int(choice_input)
                                    
                                    if 1 <= choice_num <= len(datasets):
                                        filepath = os.path.join(Path(__file__).resolve().parent, "datasets", datasets[choice_num - 1])
                                        break
                                    
                                    elif choice_num == last_option:
                                        # Get custom file path
                                        custom_path = input(Fore.YELLOW + Style.BRIGHT + "\nEnter CSV file path: " + Style.RESET_ALL).strip()
                                        if not custom_path:
                                            self.print_color("\nNo path provided.", 'warning')
                                            if attempt < max_attempts - 1:
                                                continue
                                            else:
                                                raise ValueError("\nNo file path provided")
                                        
                                        if not os.path.exists(custom_path):
                                            self.print_color(f"\nFile not found: {custom_path}", 'error')
                                            if attempt < max_attempts - 1:
                                                continue
                                            else:
                                                raise FileNotFoundError(f"\nFile not found: {custom_path}")
                                        
                                        if not custom_path.lower().endswith('.csv'):
                                            confirm = input(Fore.YELLOW + Style.BRIGHT + f"\nFile '{os.path.basename(custom_path)}' doesn't have .csv extension. Continue anyway? (y/N): " + Style.RESET_ALL).strip().lower()
                                            if confirm not in ['y', 'yes']:
                                                if attempt < max_attempts - 1:
                                                    continue
                                                else:
                                                    raise ValueError("\nFile doesn't have .csv extension")
                                        
                                        filepath = custom_path
                                        break
                                    
                                    else:
                                        if attempt < max_attempts - 1:
                                            self.print_color(f"\nPlease enter a number between 0 and {last_option}.", 'warning')
                                
                                else:
                                    if attempt < max_attempts - 1:
                                        self.print_color(f"\nPlease enter a number between 0 and {last_option}.", 'warning')
                            
                            except (EOFError, KeyboardInterrupt):
                                self.print_color("\nDataset selection cancelled.", 'warning')
                                break
                            except Exception as input_error:
                                if attempt < max_attempts - 1:
                                    self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                        
                        if not filepath:
                            if choice_input != '0':
                                self.print_color("\nCould not determine file path. Returning to menu.", 'error')
                            continue
                        
                        self.print_color(f"\nSelected file:", 'success')
                        self.print_color(f"  └─ {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'success')
                        
                        self.print_color(f"\nCurrent test max rows configuration:", 'warning')
                        self.print_color(f"  ├─ Output directory: {Fore.GREEN + Style.BRIGHT}{config['default_output_dir']}", 'info')
                        self.print_color(f"  ├─ Target memory: {Fore.YELLOW + Style.BRIGHT}{config['default_target_mb']} MB", 'info')
                        self.print_color(f"  └─ Minimum chunk size: {Fore.YELLOW + Style.BRIGHT}{config['default_min_chunk']:,} rows", 'info')
                        
                        self.print_color("\nConfiguration Options:", 'warning')
                        self.print_color("1. Proceed with current configuration", 'debug')
                        self.print_color("2. Customize configuration", 'debug')
                        self.print_color("0. Return to previous menu", 'error')
                        
                        config_choice = None
                        for attempt in range(max_attempts):
                            try:
                                config_choice = input(Fore.YELLOW + Style.BRIGHT + "\nSelect option (0-2): " + Style.RESET_ALL).strip()
                                
                                if config_choice in ['0', '1', '2']:
                                    break
                                else:
                                    if attempt < max_attempts - 1:
                                        self.print_color("\nPlease enter 0, 1, or 2.", 'warning')
                            except (EOFError, KeyboardInterrupt):
                                self.print_color("\nSelection cancelled.", 'warning')
                                config_choice = '0'
                                break
                        
                        if config_choice == '0':
                            self.print_color("\nReturning to dataset selection...", 'warning')
                            continue
                        
                        # Initialize with current config values
                        output_dir = config['default_output_dir']
                        target_mb = config['default_target_mb']
                        min_chunk = config['default_min_chunk']
                        
                        if config_choice == '2':  # Customize configuration
                            self.print_color("\nCustomize current configuration", 'success')
                            self.print_color("  └─ Enter new values (press Enter to keep current):", 'success')
                            
                            # Get output directory with validation
                            output_dir_valid = False
                            
                            for attempt in range(max_attempts):
                                try:
                                    user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nOutput directory [{config['default_output_dir']}]: " + Style.RESET_ALL).strip()
                                    
                                    if not user_input:
                                        output_dir = config['default_output_dir']
                                        output_dir_valid = True
                                        break
                                    
                                    # Basic path validation
                                    invalid_chars = '<>:"|?*'
                                    if any(char in user_input for char in invalid_chars):
                                        self.print_color(f"\nInvalid characters in path.", 'error')
                                        if attempt < max_attempts - 1:
                                            continue
                                        else:
                                            raise ValueError("\nInvalid path characters")
                                    
                                    # Check if parent directory exists
                                    dir_path = Path(user_input)
                                    parent = dir_path.parent
                                    
                                    try:
                                        if not parent.exists():
                                            # Try to create parent to test permissions
                                            parent.mkdir(parents=True, exist_ok=True)
                                            # Clean up if we created it
                                            if parent != dir_path:
                                                try:
                                                    parent.rmdir()
                                                except:
                                                    pass
                                        output_dir = user_input
                                        output_dir_valid = True
                                        break
                                    except Exception as dir_error:
                                        if attempt < max_attempts - 1:
                                            self.print_color(f"\nInvalid directory: {str(dir_error)}", 'error')
                                            continue
                                        else:
                                            raise ValueError(f"\nInvalid directory: {str(dir_error)}")
                                
                                except (EOFError, KeyboardInterrupt):
                                    self.print_color("\nOutput directory selection cancelled.", 'warning')
                                    break
                                except Exception as input_error:
                                    if attempt < max_attempts - 1:
                                        self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                            
                            if not output_dir_valid:
                                self.print_color("\nUsing current output directory.", 'success')
                                output_dir = config['default_output_dir']
                            
                            # Get target memory with validation
                            target_mb_valid = False
                            
                            for attempt in range(max_attempts):
                                try:
                                    user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nTarget memory per chunk (MB) [{config['default_target_mb']}]: " + Style.RESET_ALL).strip()
                                    
                                    if not user_input:
                                        target_mb = config['default_target_mb']
                                        target_mb_valid = True
                                        break
                                    
                                    if not user_input.isdigit():
                                        self.print_color("\nPlease enter a number.", 'error')
                                        if attempt < max_attempts - 1:
                                            continue
                                        else:
                                            raise ValueError("\nInvalid number")
                                    
                                    value = int(user_input)
                                    if not (1 <= value <= 16384):  # Reasonable range 1MB-16GB
                                        self.print_color("\nPlease enter a number between 1 and 16384.", 'error')
                                        if attempt < max_attempts - 1:
                                            continue
                                        else:
                                            raise ValueError("\nValue out of range")
                                    
                                    target_mb = value
                                    target_mb_valid = True
                                    break
                                
                                except (EOFError, KeyboardInterrupt):
                                    self.print_color("\nTarget memory input cancelled.", 'warning')
                                    break
                                except Exception as input_error:
                                    if attempt < max_attempts - 1:
                                        self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                            
                            if not target_mb_valid:
                                self.print_color("\nUsing current target memory.", 'success')
                                target_mb = config['default_target_mb']
                            
                            # Get minimum chunk size with validation
                            min_chunk_valid = False
                            
                            for attempt in range(max_attempts):
                                try:
                                    user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nMinimum chunk size [{config['default_min_chunk']}]: " + Style.RESET_ALL).strip()
                                    
                                    if not user_input:
                                        min_chunk = config['default_min_chunk']
                                        min_chunk_valid = True
                                        break
                                    
                                    if not user_input.isdigit():
                                        self.print_color("\nPlease enter a number.", 'error')
                                        if attempt < max_attempts - 1:
                                            continue
                                        else:
                                            raise ValueError("\nInvalid number")
                                    
                                    value = int(user_input)
                                    if not (1 <= value <= 1000000):  # Reasonable range
                                        self.print_color("\nPlease enter a number between 1 and 1,000,000.", 'error')
                                        if attempt < max_attempts - 1:
                                            continue
                                        else:
                                            raise ValueError("\nValue out of range")
                                    
                                    min_chunk = value
                                    min_chunk_valid = True
                                    break
                                
                                except (EOFError, KeyboardInterrupt):
                                    self.print_color("\nMinimum chunk size input cancelled.", 'warning')
                                    break
                                except Exception as input_error:
                                    if attempt < max_attempts - 1:
                                        self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                            
                            if not min_chunk_valid:
                                self.print_color("\nUsing current minimum chunk size.", 'success')
                                min_chunk = config['default_min_chunk']
                        
                        # Confirm parameters before running
                        self.print_color("\nTest Parameters", 'highlight')
                        self.print_color(f"  ├─ File: {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'info')
                        self.print_color(f"  ├─ Output directory: {Fore.CYAN + Style.BRIGHT}{output_dir}", 'info')
                        self.print_color(f"  ├─ Target memory: {Fore.YELLOW + Style.BRIGHT}{target_mb} MB", 'info')
                        self.print_color(f"  └─ Minimum chunk size: {Fore.YELLOW + Style.BRIGHT}{min_chunk:,} rows", 'info')
                        
                        confirm_attempts = 3
                        confirmed = False
                        
                        for attempt in range(confirm_attempts):
                            try:
                                confirm = input(Fore.YELLOW + Style.BRIGHT + "\nRun test with these parameters? (Y/n): " + Style.RESET_ALL).strip().lower()
                                
                                if confirm in ['y', 'yes', '']:
                                    confirmed = True
                                    break
                                elif confirm in ['n', 'no']:
                                    self.print_color("\nTest cancelled.", 'warning')
                                    break
                                else:
                                    if attempt < confirm_attempts - 1:
                                        self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                            except (EOFError, KeyboardInterrupt):
                                self.print_color("\nConfirmation cancelled.", 'warning')
                                break
                        
                        if not confirmed:
                            continue
                        
                        # Update config with new values
                        config['default_output_dir'] = output_dir
                        config['default_target_mb'] = target_mb
                        config['default_min_chunk'] = min_chunk
                        
                        if filepath not in config['recent_files']:
                            config['recent_files'].append(filepath)
                            # Keep recent files list manageable
                            if len(config['recent_files']) > 10:
                                config['recent_files'] = config['recent_files'][-10:]
                        
                        self.save_config(config)
                        
                        # Setup run tracking before calling run_test()
                        start_time = datetime.now()
                        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                        
                        # Get next sequential run number for this output directory
                        def get_next_run_number(output_dir: Path) -> int:
                            """Get the next sequential run number for the tracking directory with error handling."""
                            run_tracker_file = Path(output_dir) / ".preprocessor_run_tracker"
                            max_retries = 3
                            retry_delay = 0.1
                            
                            if not run_tracker_file.exists():
                                with open(run_tracker_file, 'w') as f:
                                    json.dump({'last_run': 0, 'runs': {}}, f)
                                return 1
                            
                            for attempt in range(max_retries):
                                try:
                                    with open(run_tracker_file, 'r') as f:
                                        content = f.read().strip()
                                        if not content:
                                            tracker = {'last_run': 0, 'runs': {}}
                                        else:
                                            tracker = json.loads(content)
                                    
                                    if not isinstance(tracker, dict):
                                        raise ValueError("Tracker is not a dictionary")
                                    
                                    next_run = tracker.get('last_run', 0) + 1
                                    
                                    tracker['last_run'] = next_run
                                    if 'runs' not in tracker:
                                        tracker['runs'] = {}
                                    
                                    tracker['runs'][str(next_run)] = {
                                        'timestamp': timestamp,
                                        'started': start_time.isoformat(),
                                        'filepath': filepath,
                                        'target_mb': target_mb,
                                        'min_chunk': min_chunk
                                    }
                                    
                                    temp_file = run_tracker_file.with_suffix('.tmp')
                                    with open(temp_file, 'w') as f:
                                        json.dump(tracker, f, indent=2)
                                    
                                    temp_file.replace(run_tracker_file)
                                    
                                    return next_run
                                
                                except json.JSONDecodeError as e:
                                    if attempt < max_retries - 1:
                                        self.print_color(f"\nTracker file corrupted (attempt {attempt + 1}), retrying...", 'warning')
                                        time.sleep(retry_delay)
                                    else:
                                        self.print_color(f"\nTracker file corrupted: {str(e)}", 'error')
                                        backup_file = run_tracker_file.with_suffix(f'.corrupted_{int(time.time())}')
                                        try:
                                            shutil.copy2(run_tracker_file, backup_file)
                                            self.print_color(f"Backup created: {backup_file}", 'info')
                                        except Exception:
                                            pass
                                        
                                        existing_runs = len([d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith('run_')])
                                        next_run = existing_runs + 1
                                        
                                        new_tracker = {
                                            'last_run': next_run,
                                            'runs': {
                                                str(next_run): {
                                                    'timestamp': timestamp,
                                                    'started': start_time.isoformat(),
                                                    'filepath': filepath,
                                                    'target_mb': target_mb,
                                                    'min_chunk': min_chunk,
                                                    'note': 'Tracker reinitialized due to corruption'
                                                }
                                            }
                                        }
                                        
                                        with open(run_tracker_file, 'w') as f:
                                            json.dump(new_tracker, f, indent=2)
                                        
                                        self.print_color(f"\nTracker reinitialized with run number: {next_run}", 'success')
                                        return next_run
                                
                                except Exception as e:
                                    if attempt < max_retries - 1:
                                        self.print_color(f"\nError reading tracker (attempt {attempt + 1}): {str(e)}", 'warning')
                                        time.sleep(retry_delay)
                                    else:
                                        self.print_color(f"\nFailed to read run tracker: {str(e)}", 'error')
                                        existing_runs = len([d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith('run_')])
                                        return existing_runs + 1
                            
                            return 1
                        
                        # Generate run ID and create run directory
                        run_number = get_next_run_number(Path(output_dir))
                        run_id = f"run_{run_number:03d}"
                        
                        # Generate full tracking ID for metadata
                        process_id = os.getpid()
                        unique_hash = hashlib.md5(
                            f"{timestamp}_{process_id}".encode()
                        ).hexdigest()[:4]
                        run_id_full = f"run_{run_number:03d}_{timestamp}_{unique_hash}"
                        
                        # Create run-specific output directory
                        run_output_dir = Path(output_dir) / run_id
                        run_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Run the test with run-specific directory
                        self.min_chunk_size = min_chunk
                        summary = self.run_test(
                            filepath=filepath,
                            output_dir=run_output_dir,
                            target_mb=target_mb,
                            min_chunk=min_chunk,
                            run_info={
                                'run_id': run_id,
                                'run_id_full': run_id_full,
                                'run_number': run_number,
                                'timestamp': timestamp,
                                'start_time': start_time.isoformat()
                            }
                        )
                        
                        # Post-test options loop
                        if summary:  # Only show options if test completed successfully
                            post_test_loop = True
                            while post_test_loop:
                                try:
                                    self.print_color("\nWhat would you like to do next?", 'warning')
                                    self.print_color("1. Run preprocessing pipeline with this test configuration", 'debug')
                                    self.print_color("2. Run another chunk size test", 'debug')
                                    self.print_color("0. Return to main menu", 'error')
                                    
                                    post_test_choice = None
                                    for attempt in range(max_attempts):
                                        try:
                                            post_test_choice = input(Fore.YELLOW + Style.BRIGHT + "\nSelect option (0-2): " + Style.RESET_ALL).strip()
                                            
                                            if post_test_choice in ['0', '1', '2']:
                                                break
                                            else:
                                                if attempt < max_attempts - 1:
                                                    self.print_color("\nPlease enter 0, 1, or 2.", 'warning')
                                        except (EOFError, KeyboardInterrupt):
                                            self.print_color("\nSelection cancelled.", 'warning')
                                            post_test_choice = '0'
                                            break
                                    
                                    if post_test_choice == '1':  # Run preprocessing pipeline
                                        try:
                                            # Display test configuration
                                            self.print_color(f"\nUsing test configuration from {Fore.YELLOW + Style.BRIGHT}{run_id}", 'success')
                                            self.print_color(f"  ├─ Dataset: {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'info')
                                            self.print_color(f"  ├─ Max chunk size: {Fore.YELLOW + Style.BRIGHT}{summary['performance_metrics']['max_chunk_size']:,} rows", 'info')
                                            self.print_color(f"  └─ Config source: {Fore.MAGENTA + Style.BRIGHT}{str(run_output_dir)}", 'info')
                                            
                                            preprocessing_config = self.config.get('preprocessing_config', {})
                                            self.print_color(f"\nCurrent Preprocessing Configuration:", 'info')
                                            self.print_color(f"  ├─ Output directory: {Fore.CYAN + Style.BRIGHT}{preprocessing_config.get('default_output_dir', 'results')}", 'info')
                                            self.print_color(f"  ├─ Memory safety factor: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('memory_safety_factor', MEMORY_SAFETY_FACTOR)}", 'info')
                                            self.print_color(f"  ├─ Hybrid feature count: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('hybrid_feature_count', HYBRID_FEATURE_COUNT)}", 'info')
                                            self.print_color(f"  └─ Scaler range: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('scaler_range', SCALER_RANGE)}", 'info')
                                            
                                            self.print_color("\nConfiguration Options:", 'warning')
                                            self.print_color("1. Proceed with current configuration", 'debug')
                                            self.print_color("2. Customize configuration", 'debug')
                                            self.print_color("0. Return to previous menu", 'error')
                                            
                                            preprocess_config_choice = None
                                            for attempt in range(max_attempts):
                                                try:
                                                    preprocess_config_choice = input(Fore.YELLOW + Style.BRIGHT + "\nSelect option (0-2): " + Style.RESET_ALL).strip()
                                                    
                                                    if preprocess_config_choice in ['0', '1', '2']:
                                                        break
                                                    else:
                                                        if attempt < max_attempts - 1:
                                                            self.print_color("\nPlease enter 0, 1, or 2.", 'warning')
                                                except (EOFError, KeyboardInterrupt):
                                                    self.print_color("\nSelection cancelled.", 'warning')
                                                    preprocess_config_choice = '0'
                                                    break
                                            
                                            if preprocess_config_choice == '0':
                                                self.print_color("\nReturning to post-test menu...", 'warning')
                                                continue
                                            
                                            # Initialize with current config values
                                            preprocess_output_dir = preprocessing_config.get('default_output_dir', 'results')
                                            config_source = str(run_output_dir)  # Use the test run directory
                                            verbose_mode = True
                                            
                                            if preprocess_config_choice == '2':  # Customize configuration
                                                self.print_color("\nCustomize Configuration Selected", 'success')
                                                self.print_color("  └─ Enter new values (press Enter to keep current):", 'success')
                                                
                                                # Get preprocessing output directory with validation
                                                output_dir_valid = False
                                                
                                                for attempt in range(max_attempts):
                                                    try:
                                                        user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nPreprocessing output directory [{preprocessing_config.get('default_output_dir', 'results')}]: " + Style.RESET_ALL).strip()
                                                        
                                                        if not user_input:
                                                            preprocess_output_dir = preprocessing_config.get('default_output_dir', 'results')
                                                            output_dir_valid = True
                                                            break
                                                        
                                                        # Basic path validation
                                                        invalid_chars = '<>:"|?*'
                                                        if any(char in user_input for char in invalid_chars):
                                                            self.print_color(f"\nInvalid characters in path.", 'error')
                                                            if attempt < max_attempts - 1:
                                                                continue
                                                            else:
                                                                raise ValueError("\nInvalid path characters")
                                                        
                                                        # Check if parent directory exists
                                                        dir_path = Path(user_input)
                                                        parent = dir_path.parent
                                                        
                                                        try:
                                                            if not parent.exists():
                                                                parent.mkdir(parents=True, exist_ok=True)
                                                                if parent != dir_path:
                                                                    try:
                                                                        parent.rmdir()
                                                                    except:
                                                                        pass
                                                            preprocess_output_dir = user_input
                                                            output_dir_valid = True
                                                            break
                                                        except Exception as dir_error:
                                                            if attempt < max_attempts - 1:
                                                                self.print_color(f"\nInvalid directory: {str(dir_error)}", 'error')
                                                                continue
                                                            else:
                                                                raise ValueError(f"\nInvalid directory: {str(dir_error)}")
                                                    
                                                    except (EOFError, KeyboardInterrupt):
                                                        self.print_color("\nOutput directory selection cancelled.", 'warning')
                                                        break
                                                    except Exception as input_error:
                                                        if attempt < max_attempts - 1:
                                                            self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                                                
                                                if not output_dir_valid:
                                                    self.print_color("\nUsing current output directory.", 'success')
                                                    preprocess_output_dir = preprocessing_config.get('default_output_dir', 'results')
                                                
                                                # Get verbose mode preference
                                                for attempt in range(max_attempts):
                                                    try:
                                                        verbose_input = input(Fore.YELLOW + Style.BRIGHT + f"\nEnable verbose output? (Y/n): " + Style.RESET_ALL).strip().lower()
                                                        
                                                        if verbose_input in ['y', 'yes', '']:
                                                            verbose_mode = True
                                                            break
                                                        elif verbose_input in ['n', 'no']:
                                                            verbose_mode = False
                                                            break
                                                        else:
                                                            if attempt < max_attempts - 1:
                                                                self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                                                    
                                                    except (EOFError, KeyboardInterrupt):
                                                        self.print_color("\nVerbose mode input cancelled.", 'warning')
                                                        break
                                            
                                            # Confirm parameters before running preprocessing
                                            self.print_color("\nPreprocessing Parameters", 'highlight')
                                            self.print_color(f"  ├─ Input file: {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'info')
                                            self.print_color(f"  ├─ Output directory: {Fore.CYAN + Style.BRIGHT}{preprocess_output_dir}", 'info')
                                            self.print_color(f"  ├─ Config source: {Fore.MAGENTA + Style.BRIGHT}{config_source}", 'info')
                                            self.print_color(f"  ├─ Memory safety factor: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('memory_safety_factor', MEMORY_SAFETY_FACTOR)}", 'info')
                                            self.print_color(f"  ├─ Hybrid feature count: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('hybrid_feature_count', HYBRID_FEATURE_COUNT)}", 'info')
                                            self.print_color(f"  ├─ Scaler range: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('scaler_range', SCALER_RANGE)}", 'info')
                                            self.print_color(f"  └─ Verbose mode: {Fore.YELLOW + Style.BRIGHT}{verbose_mode}", 'info')
                                            
                                            confirm_attempts = 3
                                            confirmed = False
                                            
                                            for attempt in range(confirm_attempts):
                                                try:
                                                    confirm = input(Fore.YELLOW + Style.BRIGHT + "\nRun preprocessing with these parameters? (Y/n): " + Style.RESET_ALL).strip().lower()
                                                    
                                                    if confirm in ['y', 'yes', '']:
                                                        confirmed = True
                                                        break
                                                    elif confirm in ['n', 'no']:
                                                        self.print_color("\nPreprocessing cancelled.", 'warning')
                                                        break
                                                    else:
                                                        if attempt < confirm_attempts - 1:
                                                            self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                                                except (EOFError, KeyboardInterrupt):
                                                    self.print_color("\nConfirmation cancelled.", 'warning')
                                                    break
                                            
                                            if not confirmed:
                                                continue
                                            
                                            # Run preprocessing
                                            self.preprocess_data(
                                                filepath=filepath,
                                                output_dir=preprocess_output_dir,
                                                config_path=config_source,
                                                interactive=True,
                                                verbose=verbose_mode
                                            )
                                            
                                            # After preprocessing completes, return to post-test menu
                                            try:
                                                input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                                            except (EOFError, KeyboardInterrupt):
                                                self.print_color("\nContinuing...", 'warning')
                                        
                                        except KeyboardInterrupt:
                                            self.print_color("\nPreprocessing setup cancelled.", 'warning')
                                        except Exception as e:
                                            self.print_color(f"\nError setting up preprocessing: {str(e)}", 'error')
                                            self._log_event(f"Preprocessing setup error: {str(e)}", "error")
                                            try:
                                                input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                                            except (EOFError, KeyboardInterrupt):
                                                self.print_color("\nContinuing...", 'warning')
                                    
                                    elif post_test_choice == '2':  # Run another test
                                        self.print_color("\nRestarting chunk size test...", 'warning')
                                        post_test_loop = False  # Exit post-test loop to restart test selection
                                        # The outer continue will restart from dataset selection
                                    
                                    elif post_test_choice == '0':  # Return to main menu
                                        self.print_color("\nReturning to main menu...", 'warning')
                                        post_test_loop = False
                                        break
                                
                                except KeyboardInterrupt:
                                    self.print_color("\nPost-test menu interrupted.", 'warning')
                                    post_test_loop = False
                                    break
                                except Exception as e:
                                    self.print_color(f"\nError in post-test menu: {str(e)}", 'error')
                                    self._log_event(f"Post-test menu error: {str(e)}", "error")
                                    try:
                                        input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                                    except (EOFError, KeyboardInterrupt):
                                        self.print_color("\nContinuing...", 'warning')
                                    post_test_loop = False
                                    break
                        else:
                            # Test failed, just wait for user acknowledgment
                            try:
                                input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                            except (EOFError, KeyboardInterrupt):
                                self.print_color("\nContinuing...", 'warning')
                    
                    except KeyboardInterrupt:
                        self.print_color("\nTest setup cancelled.", 'warning')
                    except Exception as e:
                        self.print_color(f"\nError setting up test: {str(e)}", 'error')
                        self._log_event(f"Test setup error: {str(e)}", "error")
                        try:
                            input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                        except (EOFError, KeyboardInterrupt):
                            self.print_color("\nContinuing...", 'warning')
                
                elif choice == '2':  # Run preprocessing pipeline
                    try:
                        # Get datasets from default input directory in config
                        preprocessing_config = self.config.get('preprocessing_config', {})
                        default_input_dir = preprocessing_config.get('default_input_dir', 'datasets')
                        datasets_dir = Path(__file__).resolve().parent / default_input_dir
                        
                        filepath = None
                        
                        # List available datasets
                        if datasets_dir.exists():
                            datasets = [f for f in datasets_dir.iterdir() if f.suffix == '.csv']
                            
                            if datasets:
                                self.print_color("\nAvailable datasets:", 'info')
                                for i, dataset in enumerate(datasets, 1):
                                    self.print_color(f"{i}. {dataset.name}", 'debug')
                                
                                self.print_color(f"{len(datasets) + 1}. Enter custom file path", 'info')
                                self.print_color("0. Return to main menu", 'error')
                                
                                for attempt in range(max_attempts):
                                    try:
                                        prompt = Fore.YELLOW + Style.BRIGHT + f"\nSelect dataset (0-{len(datasets) + 1}): " + Style.RESET_ALL
                                        choice_input = input(prompt).strip().lower()
                                        
                                        if choice_input == '0':
                                            self.print_color("\nReturning to main menu...", 'warning')
                                            break
                                        
                                        elif choice_input.isdigit():
                                            choice_num = int(choice_input)
                                            
                                            if 1 <= choice_num <= len(datasets):
                                                filepath = str(datasets[choice_num - 1])
                                                break
                                            
                                            elif choice_num == len(datasets) + 1:
                                                custom_path = input(Fore.CYAN + Style.BRIGHT + "\nEnter CSV file path: " + Style.RESET_ALL).strip()
                                                if not custom_path:
                                                    self.print_color("\nNo path provided.", 'warning')
                                                    if attempt < max_attempts - 1:
                                                        continue
                                                    else:
                                                        raise ValueError("\nNo file path provided")
                                                
                                                if not Path(custom_path).exists():
                                                    self.print_color(f"\nFile not found: {custom_path}", 'error')
                                                    if attempt < max_attempts - 1:
                                                        continue
                                                    else:
                                                        raise FileNotFoundError(f"\nFile not found: {custom_path}")
                                                
                                                if not custom_path.lower().endswith('.csv'):
                                                    confirm = input(Fore.YELLOW + Style.BRIGHT + f"\nFile '{os.path.basename(custom_path)}' doesn't have .csv extension. Continue anyway? (y/N): " + Style.RESET_ALL).strip().lower()
                                                    if confirm not in ['y', 'yes']:
                                                        if attempt < max_attempts - 1:
                                                            continue
                                                        else:
                                                            raise ValueError("\nFile doesn't have .csv extension")
                                                
                                                filepath = custom_path
                                                break
                                            else:
                                                if attempt < max_attempts - 1:
                                                    self.print_color(f"\nPlease enter a number between 0 and {len(datasets) + 1}.", 'warning')
                                        
                                        else:
                                            if attempt < max_attempts - 1:
                                                self.print_color(f"\nPlease enter a number between 0 and {len(datasets) + 1}.", 'warning')
                                    
                                    except (EOFError, KeyboardInterrupt):
                                        self.print_color("\nDataset selection cancelled.", 'warning')
                                        break
                                    except Exception as input_error:
                                        if attempt < max_attempts - 1:
                                            self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                                
                                if not filepath:
                                    if choice_input != '0':
                                        self.print_color("\nCould not determine file path. Returning to menu.", 'error')
                                    continue
                                
                            else:
                                self.print_color(f"\nNo datasets found in default directory", 'warning')
                                filepath = input(Fore.CYAN + Style.BRIGHT + "\nEnter CSV file path: " + Style.RESET_ALL).strip()
                                
                                if not filepath:
                                    self.print_color("\nNo path provided.", 'warning')
                                    continue
                                
                                if not Path(filepath).exists():
                                    self.print_color(f"\nFile not found: {filepath}", 'error')
                                    continue
                        else:
                            self.print_color(f"\nDatasets directory not found: {datasets_dir}", 'warning')
                            filepath = input(Fore.CYAN + Style.BRIGHT + "\nEnter CSV file path: " + Style.RESET_ALL).strip()
                            
                            if not filepath:
                                self.print_color("\nNo path provided.", 'warning')
                                continue
                            
                            if not Path(filepath).exists():
                                self.print_color(f"\nFile not found: {filepath}", 'error')
                                continue
                        
                        # Display selected file and current configuration
                        self.print_color(f"\nSelected file:", 'success')
                        self.print_color(f"  └─ {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'success')
                        
                        preprocessing_config = self.config.get('preprocessing_config', {})
                        self.print_color(f"\nCurrent Preprocessing Configuration:", 'info')
                        self.print_color(f"  ├─ Output directory: {Fore.CYAN + Style.BRIGHT}{preprocessing_config.get('default_output_dir', 'results')}", 'info')
                        self.print_color(f"  ├─ Config source: {Fore.MAGENTA + Style.BRIGHT}{self.config.get('default_results_dir', 'results')}", 'info')
                        self.print_color(f"  ├─ Memory safety factor: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('memory_safety_factor', MEMORY_SAFETY_FACTOR)}", 'info')
                        self.print_color(f"  ├─ Hybrid feature count: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('hybrid_feature_count', HYBRID_FEATURE_COUNT)}", 'info')
                        self.print_color(f"  └─ Scaler range: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('scaler_range', SCALER_RANGE)}", 'info')
                        
                        self.print_color("\nConfiguration Options:", 'warning')
                        self.print_color("1. Proceed with current configuration", 'debug')
                        self.print_color("2. Customize configuration", 'debug')
                        self.print_color("0. Return to previous menu", 'error')
                        
                        config_choice = None
                        for attempt in range(max_attempts):
                            try:
                                config_choice = input(Fore.YELLOW + Style.BRIGHT + "\nSelect option (0-2): " + Style.RESET_ALL).strip()
                                
                                if config_choice in ['0', '1', '2']:
                                    break
                                else:
                                    if attempt < max_attempts - 1:
                                        self.print_color("\nPlease enter 0, 1, or 2.", 'warning')
                            except (EOFError, KeyboardInterrupt):
                                self.print_color("\nSelection cancelled.", 'warning')
                                config_choice = '0'
                                break
                        
                        if config_choice == '0':
                            self.print_color("\nReturning to dataset selection...", 'warning')
                            continue
                        
                        # Initialize with current config values
                        preprocess_output_dir = preprocessing_config.get('default_output_dir', 'results')
                        config_source = self.config.get('default_results_dir', 'results')
                        verbose_mode = True
                        
                        if config_choice == '2':  # Customize configuration
                            self.print_color("\nCustomize Configuration", 'success')
                            self.print_color("  └─ Enter new values (press Enter to keep current):", 'success')
                            
                            # Get preprocessing output directory with validation
                            output_dir_valid = False
                            
                            for attempt in range(max_attempts):
                                try:
                                    user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nPreprocessing output directory [{preprocessing_config.get('default_output_dir', 'results')}]: " + Style.RESET_ALL).strip()
                                    
                                    if not user_input:
                                        preprocess_output_dir = preprocessing_config.get('default_output_dir', 'results')
                                        output_dir_valid = True
                                        break
                                    
                                    # Basic path validation
                                    invalid_chars = '<>:"|?*'
                                    if any(char in user_input for char in invalid_chars):
                                        self.print_color(f"\nInvalid characters in path.", 'error')
                                        if attempt < max_attempts - 1:
                                            continue
                                        else:
                                            raise ValueError("\nInvalid path characters")
                                    
                                    # Check if parent directory exists
                                    dir_path = Path(user_input)
                                    parent = dir_path.parent
                                    
                                    try:
                                        if not parent.exists():
                                            # Try to create parent to test permissions
                                            parent.mkdir(parents=True, exist_ok=True)
                                            # Clean up if we created it
                                            if parent != dir_path:
                                                try:
                                                    parent.rmdir()
                                                except:
                                                    pass
                                        preprocess_output_dir = user_input
                                        output_dir_valid = True
                                        break
                                    except Exception as dir_error:
                                        if attempt < max_attempts - 1:
                                            self.print_color(f"\nInvalid directory: {str(dir_error)}", 'error')
                                            continue
                                        else:
                                            raise ValueError(f"\nInvalid directory: {str(dir_error)}")
                                
                                except (EOFError, KeyboardInterrupt):
                                    self.print_color("\nOutput directory selection cancelled.", 'warning')
                                    break
                                except Exception as input_error:
                                    if attempt < max_attempts - 1:
                                        self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                            
                            if not output_dir_valid:
                                self.print_color("\nUsing current output directory.", 'success')
                                preprocess_output_dir = preprocessing_config.get('default_output_dir', 'results')
                            
                            # Get config source directory with validation
                            config_source_valid = False
                            
                            for attempt in range(max_attempts):
                                try:
                                    user_input = input(Fore.YELLOW + Style.BRIGHT + f"\nConfig source directory (test results) [{self.config.get('default_results_dir', 'results')}]: " + Style.RESET_ALL).strip()
                                    
                                    if not user_input:
                                        config_source = self.config.get('default_results_dir', 'results')
                                        config_source_valid = True
                                        break
                                    
                                    # Basic path validation
                                    invalid_chars = '<>:"|?*'
                                    if any(char in user_input for char in invalid_chars):
                                        self.print_color(f"\nInvalid characters in path.", 'error')
                                        if attempt < max_attempts - 1:
                                            continue
                                        else:
                                            raise ValueError("\nInvalid path characters")
                                    
                                    # Check if directory exists
                                    dir_path = Path(user_input)
                                    if dir_path.exists() and dir_path.is_dir():
                                        config_source = user_input
                                        config_source_valid = True
                                        break
                                    else:
                                        if attempt < max_attempts - 1:
                                            self.print_color(f"\nDirectory does not exist: {user_input}", 'error')
                                            continue
                                        else:
                                            raise ValueError(f"\nDirectory does not exist: {user_input}")
                                
                                except (EOFError, KeyboardInterrupt):
                                    self.print_color("\nConfig source selection cancelled.", 'warning')
                                    break
                                except Exception as input_error:
                                    if attempt < max_attempts - 1:
                                        self.print_color(f"\nInput error: {str(input_error)}. Please try again.", 'error')
                            
                            if not config_source_valid:
                                self.print_color("\nUsing current config source.", 'success')
                                config_source = self.config.get('default_results_dir', 'results')
                            
                            # Get verbose mode preference
                            for attempt in range(max_attempts):
                                try:
                                    verbose_input = input(Fore.YELLOW + Style.BRIGHT + f"\nEnable verbose output? (Y/n): " + Style.RESET_ALL).strip().lower()
                                    
                                    if verbose_input in ['y', 'yes', '']:
                                        verbose_mode = True
                                        break
                                    elif verbose_input in ['n', 'no']:
                                        verbose_mode = False
                                        break
                                    else:
                                        if attempt < max_attempts - 1:
                                            self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                                
                                except (EOFError, KeyboardInterrupt):
                                    self.print_color("\nVerbose mode input cancelled.", 'warning')
                                    break
                        
                        # Confirm parameters before running preprocessing
                        self.print_color("\nPreprocessing Parameters", 'highlight')
                        self.print_color(f"  ├─ Input file: {Fore.MAGENTA + Style.BRIGHT}{filepath}", 'info')
                        self.print_color(f"  ├─ Output directory: {Fore.CYAN + Style.BRIGHT}{preprocess_output_dir}", 'info')
                        self.print_color(f"  ├─ Config source: {Fore.MAGENTA + Style.BRIGHT}{config_source}", 'info')
                        self.print_color(f"  ├─ Memory safety factor: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('memory_safety_factor', MEMORY_SAFETY_FACTOR)}", 'info')
                        self.print_color(f"  ├─ Hybrid feature count: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('hybrid_feature_count', HYBRID_FEATURE_COUNT)}", 'info')
                        self.print_color(f"  ├─ Scaler range: {Fore.YELLOW + Style.BRIGHT}{preprocessing_config.get('scaler_range', SCALER_RANGE)}", 'info')
                        self.print_color(f"  └─ Verbose mode: {Fore.YELLOW + Style.BRIGHT}{verbose_mode}", 'info')
                        
                        confirm_attempts = 3
                        confirmed = False
                        
                        for attempt in range(confirm_attempts):
                            try:
                                confirm = input(Fore.YELLOW + Style.BRIGHT + "\nRun preprocessing with these parameters? (Y/n): " + Style.RESET_ALL).strip().lower()
                                
                                if confirm in ['y', 'yes', '']:
                                    confirmed = True
                                    break
                                elif confirm in ['n', 'no']:
                                    self.print_color("\nPreprocessing cancelled.", 'warning')
                                    break
                                else:
                                    if attempt < confirm_attempts - 1:
                                        self.print_color("\nPlease enter 'y' for yes or 'n' for no.", 'warning')
                            except (EOFError, KeyboardInterrupt):
                                self.print_color("\nConfirmation cancelled.", 'warning')
                                break
                        
                        if not confirmed:
                            continue
                        
                        # Run preprocessing
                        self.preprocess_data(
                            filepath=filepath,
                            output_dir=preprocess_output_dir,
                            config_path=config_source,
                            interactive=True,
                            verbose=verbose_mode
                        )
                        
                        # Wait for continue
                        try:
                            input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                        except (EOFError, KeyboardInterrupt):
                            self.print_color("\nContinuing...", 'warning')
                    
                    except KeyboardInterrupt:
                        self.print_color("\nPreprocessing setup cancelled.", 'warning')
                    except Exception as e:
                        self.print_color(f"\nError setting up preprocessing: {str(e)}", 'error')
                        self._log_event(f"Preprocessing setup error: {str(e)}", "error")
                        try:
                            input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                        except (EOFError, KeyboardInterrupt):
                            self.print_color("\nContinuing...", 'warning')
                
                elif choice == '3':  # View history
                    self.display_test_history()
                    try:
                        input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                    except (EOFError, KeyboardInterrupt):
                        self.print_color("\nContinuing...", 'warning')
                
                elif choice == '4':  # Configure settings
                    self.configure_settings()
                    try:
                        input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                    except (EOFError, KeyboardInterrupt):
                        self.print_color("\nContinuing...", 'warning')
                
                elif choice == '5':  # System information
                    self.display_system_info()
                    try:
                        input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                    except (EOFError, KeyboardInterrupt):
                        self.print_color("\nContinuing...", 'warning')
                
                elif choice == '0':  # Exit
                    self.print_color("\nExiting... Thank you for using CSV Chunk Tester!", 'warning')
                    self._log_event("User exited interactive menu", "info")
                    break
            
            except KeyboardInterrupt:
                self.print_color("\nMenu interrupted. Returning to main menu...", 'warning')
                self._log_event("Menu interrupted by user", "warning")
                continue
            except Exception as e:
                self.print_color(f"\nUnexpected error in menu: {str(e)}", 'error')
                self._log_event(f"Menu error: {str(e)}", "error")
                try:
                    input(Fore.YELLOW + Style.BRIGHT + "\nPress Enter to continue..." + Style.RESET_ALL)
                except (EOFError, KeyboardInterrupt):
                    self.print_color("\nContinuing...", 'warning')

def main():
    """Main entry point for the application."""
    tester = MemoryAwarePreprocessor()
    
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        # Use argparse for command-line mode
        parser = argparse.ArgumentParser(
            description="Adaptive Preprocessing with CSV Chunk Tester - Optimizes chunk sizes and preprocesses data",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--file", 
            default="datasets/NF-CSE-CIC-IDS2018.csv",
            help="Path to input CSV file"
        )
        parser.add_argument(
            "--output",
            default="results",
            help="Output directory for test results"
        )
        parser.add_argument(
            "--mb",
            type=int,
            default=256,
            help="Target memory usage per chunk (in MB)"
        )
        parser.add_argument(
            "--min-chunk",
            type=int,
            default=1000,
            help="Minimum chunk size to test"
        )
        parser.add_argument(
            "--preprocess",
            action="store_true",
            help="Run preprocessing pipeline instead of chunk testing"
        )
        parser.add_argument(
            "--preprocess-output",
            default="models",
            help="Output directory for preprocessed data"
        )
        parser.add_argument(
            "--config",
            help="Path to test results directory or specific summary file for preprocessing"
        )
        parser.add_argument(
            "--non-interactive",
            action="store_true",
            help="Disable all interactive prompts"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Show detailed processing information"
        )
        
        args = parser.parse_args()
        
        if args.preprocess:
            # Run preprocessing pipeline
            tester.preprocess_data(
                filepath=args.file,
                output_dir=args.preprocess_output,
                config_path=args.config,
                interactive=not args.non_interactive,
                verbose=args.verbose
            )
        else:
            # Run chunk size test
            tester.min_chunk_size = args.min_chunk
            tester.run_test(args.file, args.output, args.mb, args.min_chunk)
    else:
        # No command line args, launch interactive mode
        tester.interactive_menu()

if __name__ == "__main__":
    main()