import os
import sys
import time
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
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init as colorama_init

# Initialize colorama for colored output
colorama_init(autoreset=True)

class MemoryAwareChunkTester:
    """A class to test and determine optimal chunk sizes for processing large files
    while maintaining safe memory usage and performance thresholds."""
    
    VERSION = "2.1.0"
    
    def __init__(self):
        """Initialize the MemoryAwareChunkTester with default settings."""
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
            'info': Fore.CYAN,
            'success': Fore.GREEN,
            'warning': Fore.YELLOW,
            'error': Fore.RED,
            'highlight': Fore.MAGENTA,
            'system': Fore.BLUE,
            'debug': Fore.WHITE
        }
        
        # System information
        self.system_info = self._get_system_info()
        
        try:
            # Initialize and create directories
            self._initialize_directories()
            
            # Set file paths
            self.config_file = self.config_dir / "memory_aware_chunk_tester_config.json"
            self.history_file = self.history_dir / "memory_aware_chunk_tester_history.json"
            self.log_file = self.log_dir / "memory_aware_chunk_tester.log"
            
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
            
        except Exception as e:
            error_msg = f"Critical error during initialization: {str(e)}"
            self.print_color(error_msg, 'error')
            self._log_event(error_msg, 'error')
            raise

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
        """Initialize and validate all required directories."""
        self.print_color("Initializing Memory-Aware CSV Chunk Tester...", 'system')
        
        # Main configuration directory
        self.config_dir = Path("config").resolve()
        self._create_and_validate_directory(
            self.config_dir,
            "Config directory",
            check_execute=True
        )
        
        # Log directory
        self.log_dir = Path("logs").resolve()
        self._create_and_validate_directory(
            self.log_dir,
            "Log directory",
            check_execute=False
        )
        
        # History directory
        self.history_dir = self.log_dir.resolve()
        self._create_and_validate_directory(
            self.history_dir,
            "History directory",
            check_execute=True
        )
        
        # Results directory (from config or default)
        self.results_dir = Path("results").resolve()
        self._create_and_validate_directory(
            self.results_dir,
            "Results directory",
            check_execute=False
        )
        
        self.print_color("Directories initialized successfully", 'success')

    def _create_and_validate_directory(
        self,
        directory: Path,
        name: str,
        check_execute: bool = True
    ) -> None:
        """Helper method to create and validate a directory."""
        try:
            # Create directory if it doesn't exist
            directory.mkdir(parents=True, exist_ok=True)
            
            # Validate directory existence
            if not directory.exists():
                raise FileNotFoundError(f"{name} path does not exist: {directory}")
            if not directory.is_dir():
                raise NotADirectoryError(f"{name} is not a directory: {directory}")
                
            # Check permissions
            required_perms = os.R_OK | os.W_OK
            if check_execute:
                required_perms |= os.X_OK
                
            if not os.access(directory, required_perms):
                raise PermissionError(
                    f"Insufficient permissions for {name.lower()}: {directory}. "
                    f"Required: {oct(required_perms)}"
                )
            
            # Log directory info
            stat = directory.stat()
            self.print_color(f"{name} initialized: {directory}", 'system')
            self.print_color(f"  - Size: {stat.st_size / (1024 ** 2):.2f} MB", 'system')
            self.print_color(
                f"  - Created: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
                'system'
            )
            self.print_color(
                f"  - Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
                'system'
            )
            self.print_color(f"  - Permissions: {oct(stat.st_mode)}", 'system')
            
        except Exception as e:
            self.print_color(f"Error initializing {name.lower()}: {str(e)}", 'error')
            raise

    def _initialize_files(self) -> None:
        """Initialize required files with default content if needed."""
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
            'default_target_mb': 256,
            'default_min_chunk': self.min_chunk_size,
            'recent_files': [],
            'max_history_entries': self.max_history_entries,
            'last_updated': datetime.now().isoformat()
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
        
        self.print_color("All files initialized successfully", 'success')

    def _initialize_file(
        self,
        file_path: Path,
        default_content: Any,
        name: str,
        is_json: bool = True
    ) -> None:
        """Helper method to initialize a file with default content."""
        try:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    if is_json:
                        json.dump(default_content, f, indent=2)
                    else:
                        f.write(str(default_content))
                
            # Validate file permissions
            if not os.access(file_path, os.R_OK | os.W_OK):
                raise PermissionError(f"Insufficient permissions for {name.lower()}: {file_path}")
            
            # Log file info
            stat = file_path.stat()
            self.print_color(f"{name} initialized: {file_path}", 'system')
            self.print_color(f"  - Size: {stat.st_size / 1024:.2f} KB", 'system')
            self.print_color(
                f"  - Created: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
                'system'
            )
            self.print_color(
                f"  - Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
                'system'
            )
            self.print_color(f"  - Permissions: {oct(stat.st_mode)}", 'system')
            
        except Exception as e:
            self.print_color(f"Error initializing {name.lower()}: {str(e)}", 'error')
            raise

    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        # Ensure required keys exist
        required_keys = [
            'default_output_dir',
            'default_target_mb',
            'default_min_chunk',
            'recent_files',
            'max_history_entries'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config key: {key}")
                
        # Validate values
        if not isinstance(self.config['recent_files'], list):
            raise ValueError("Config 'recent_files' must be a list")
            
        if self.config['max_history_entries'] <= 0:
            raise ValueError("Config 'max_history_entries' must be positive")
            
        self.print_color("Configuration validated successfully", 'success')

    def _print_init_status(self) -> None:
        """Print detailed initialization status information."""
        self.print_color("\nMemory-Aware CSV Chunk Tester Initialization Status", 'highlight')
        self.print_color("=" * 60, 'system')
        self.print_color(f"Version: {self.VERSION}", 'info')
        
        # System information
        self.print_color("\nSystem Resources:", 'info')
        self.print_color(f"  - Total RAM: {self.system_info['total_ram_gb']:.2f} GB", 'system')
        self.print_color(f"  - Available RAM: {self.system_info['available_ram_gb']:.2f} GB", 'system')
        self.print_color(f"  - CPU Cores: {self.system_info['cpu_cores']}", 'system')
        self.print_color(f"  - Disk Free: {self.system_info['disk_free_gb']:.2f} GB", 'system')
        
        # Configuration
        self.print_color("\nActive Configuration:", 'info')
        for key, value in self.config.items():
            if key != 'recent_files':  # Skip potentially long list
                self.print_color(f"  - {key}: {value}", 'system')
        
        # Performance settings
        self.print_color("\nPerformance Settings:", 'info')
        self.print_color(f"  - Safety Factor: {self.safety_factor}", 'system')
        self.print_color(f"  - Min Chunk Size: {self.min_chunk_size}", 'system')
        self.print_color(f"  - Max RAM Usage: {self.max_ram_usage}", 'system')
        
        # Thresholds
        self.print_color("\nPerformance Thresholds:", 'info')
        for key, value in self.performance_thresholds.items():
            self.print_color(f"  - {key}: {value}", 'system')
        
        self.print_color("\nInitialization complete", 'success')

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
            print(f"{Fore.RED}Failed to log event: {str(e)}{Style.RESET_ALL}")

    def load_config(self) -> Dict[str, Any]:
        """Load the configuration from file.
        
        Returns:
            The loaded configuration dictionary
            
        Raises:
            ValueError: If the config file is invalid
            IOError: If there are issues reading the file
        """
        default_config = {
            'version': self.VERSION,
            'default_output_dir': 'results',
            'default_target_mb': 256,
            'default_min_chunk': 1000,
            'recent_files': [],
            'max_history_entries': 10
        }
        
        try:
            if not self.config_file.exists():
                self.print_color("No config file found - creating with defaults", 'warning')
                self.save_config(default_config)
                return default_config
                
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
                # Validate and repair config
                repaired = False
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                        repaired = True
                        
                # Validate numerical values
                for key in ['default_target_mb', 'default_min_chunk', 'max_history_entries']:
                    if not isinstance(config[key], int) or config[key] <= 0:
                        config[key] = default_config[key]
                        repaired = True
                        
                if repaired:
                    self.print_color("Config file was repaired - some values were invalid", 'warning')
                    self.save_config(config)
                    
                return config
                
        except json.JSONDecodeError:
            self.print_color("Config file is corrupted - recreating with defaults", 'warning')
            self.save_config(default_config)
            return default_config
        except Exception as e:
            self.print_color(f"Error loading config: {str(e)} - using defaults", 'error')
            return default_config

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file with error handling and validation."""
        try:
            # Basic validation before saving
            required_keys = ['default_output_dir', 'default_target_mb', 'default_min_chunk', 'recent_files']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
                    
            # Ensure numerical values are valid
            if not isinstance(config['default_target_mb'], int) or config['default_target_mb'] <= 0:
                config['default_target_mb'] = 256
            if not isinstance(config['default_min_chunk'], int) or config['default_min_chunk'] <= 0:
                config['default_min_chunk'] = 1000
                
            # Add metadata
            config['version'] = self.VERSION
            config['last_updated'] = datetime.now().isoformat()
                
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.print_color(f"Error saving config: {str(e)}", 'error')
            raise

    def update_history(self, summary: Dict[str, Any]) -> None:
        """Update test history with new summary, maintaining size limits."""
        try:
            history = []
            if self.history_file.exists():
                try:
                    with open(self.history_file, 'r') as f:
                        history = json.load(f)
                        if not isinstance(history, list):
                            raise json.JSONDecodeError("History is not a list", "", 0)
                except json.JSONDecodeError:
                    self.print_color("History file corrupted - resetting", 'warning')
                    history = []
                except Exception as e:
                    self.print_color(f"Error reading history: {str(e)}", 'error')
                    return
            
            # Ensure summary contains required fields
            required_fields = ['timestamp', 'dataset_info', 'performance_metrics']
            for field in required_fields:
                if field not in summary:
                    raise ValueError(f"Missing required summary field: {field}")
            
            # Get max history entries from config
            max_entries = self.config.get('max_history_entries', 10)
            
            # Add new entry and maintain size
            history.append(summary)
            history = history[-max_entries:]
            
            # Save back to file
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.print_color(f"Error updating history: {str(e)}", 'error')
            raise

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
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise IOError(f"Error reading file {filepath}: {str(e)}")

    def get_available_datasets(self, dataset_dir: str = "datasets") -> List[str]:
        """List available CSV datasets in the specified directory.
        
        Args:
            dataset_dir: Directory to search for CSV files
            
        Returns:
            List of CSV filenames sorted alphabetically
        """
        try:
            datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
            return sorted(datasets)
        except FileNotFoundError:
            return []
        except Exception as e:
            self.print_color(f"Error listing datasets: {str(e)}", 'error')
            return []

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
        filepath: str,
        start_row: int,
        chunk_size: int,
        mem_per_row_estimate: float
    ) -> Dict[str, Any]:
        """Load chunk with memory monitoring and adaptive adjustments.
        
        Args:
            filepath: Path to CSV file
            start_row: Starting row number (0-based)
            chunk_size: Number of rows to attempt to load
            mem_per_row_estimate: Estimated memory per row in MB
            
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
                raise MemoryError(
                    f"Insufficient memory. Available: {available_mb:.2f}MB, "
                    f"Estimated needed: {estimated_needed:.2f}MB"
                )

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
            result['mem_per_row'] = result['ram_usage_mb'] / result['actual_rows']
            result['throughput_rows_sec'] = result['actual_rows'] / result['load_time_sec']
            result['cpu_percent'] = psutil.cpu_percent(interval=0.1) - cpu_start
            result['status'] = 'success'
            
            # Check performance thresholds
            if result['ram_usage_mb'] > self.performance_thresholds['memory_spike'] * self.max_ram_usage * available_mb:
                self.print_color(
                    f"WARNING: Memory spike detected in chunk (used {result['ram_usage_mb']:.2f}MB)",
                    'warning'
                )
            
            if result['cpu_percent'] > self.performance_thresholds['cpu_overload'] * 100:
                self.print_color(
                    f"WARNING: High CPU usage detected ({result['cpu_percent']:.1f}%)",
                    'warning'
                )
            
        except MemoryError as e:
            result.update({
                'status': 'MemoryError',
                'message': str(e),
                'suggested_chunk_size': int(chunk_size * 0.6)  # Suggest 40% reduction
            })
        except Exception as e:
            result.update({
                'status': type(e).__name__,
                'message': str(e)
            })
            
        return result

    def generate_plots(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        """Generate performance plots and save to output directory.
        
        Args:
            all_results: List of chunk processing results
            output_dir: Directory to save plots
        """
        try:
            if not all_results:
                self.print_color("No results to generate plots from", 'warning')
                return

            # Prepare data
            chunk_sizes = [r['metrics']['actual_rows'] for r in all_results]
            throughputs = [r['metrics']['throughput_rows_sec'] for r in all_results]
            memory_usage = [r['metrics']['ram_usage_mb'] for r in all_results]
            mem_per_row = [r['metrics']['mem_per_row'] * 1024 for r in all_results]  # Convert to KB
            cpu_usage = [r['metrics'].get('cpu_percent', 0) for r in all_results]
            
            # Create figure with subplots
            plt.figure(figsize=(14, 10))
            
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
            
            # Save plots
            plot_file = Path(output_dir) / "performance_plots.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()
            
            self.print_color(f"[PLOTS] Performance plots saved to {plot_file}", 'success')
        except Exception as e:
            self.print_color(f"[WARNING] Failed to generate plots: {str(e)}", 'warning')

    def run_test(
        self,
        filepath: str,
        output_dir: str = "results",
        target_mb: int = 256,
        min_chunk: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Main testing pipeline with adaptive chunk sizing.
        
        Args:
            filepath: Path to input CSV file
            output_dir: Directory to save results
            target_mb: Target memory usage per chunk in MB
            min_chunk: Minimum chunk size to test
            
        Returns:
            Dictionary with test summary, or None if test failed
        """
        # Set minimum chunk size
        if min_chunk is not None:
            self.min_chunk_size = min_chunk
            
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get total rows
        try:
            total_rows = self.get_total_rows(filepath)
        except Exception as e:
            self.print_color(f"[ERROR] Failed to count rows: {str(e)}", 'error')
            return None
        
        # System info
        sys_info = self._get_system_info()
        self.print_color(f"[SYSTEM] CPU Cores: {sys_info['cpu_cores']} | "
              f"Available RAM: {sys_info['available_ram_gb']:.1f}GB", 'system')
        self.print_color(f"[DATASET] Total rows: {total_rows:,} | Target: {target_mb}MB/chunk", 'info')

        # Initial calibration
        self.print_color("\n[CALIBRATION] Measuring memory characteristics...", 'info')
        calibration = self.calibrate_memory_usage(filepath)
        if calibration['status'] != 'success':
            self.print_color(f"[ERROR] Calibration failed: {calibration.get('message', 'Unknown error')}", 'error')
            return None

        # Initial chunk size
        chunk_size = self.calculate_dynamic_chunk_size(
            calibration['mem_per_row'], 
            target_mb,
            total_rows
        )
        self.print_color(f"[OPTIMIZED] Initial chunk size: {chunk_size:,} rows "
              f"(~{target_mb}MB target, {calibration['mem_per_row']:.6f} MB/row)", 'success')

        # Process dataset
        current_row = 0
        chunk_index = 1
        all_results = []
        performance_history = []
        
        while current_row < total_rows:
            chunk_end = min(current_row + chunk_size, total_rows)
            output_file = Path(output_dir) / f"max_rows_test-{chunk_index}.json"
            
            self.print_color(f"\n[PROCESSING] Chunk {chunk_index}: Rows {current_row:,}-{chunk_end:,} "
                  f"(Size: {chunk_size:,} rows)", 'info')
            
            # Process chunk with memory monitoring
            chunk_result = self.process_chunk(
                filepath,
                current_row,
                chunk_size,
                calibration['mem_per_row']
            )
            
            # Handle failures
            if chunk_result['status'] != 'success':
                self.print_color(f"[WARNING] Chunk failed: {chunk_result.get('message', 'Unknown error')}", 'warning')
                
                if 'suggested_chunk_size' in chunk_result:
                    new_size = chunk_result['suggested_chunk_size']
                    self.print_color(f"[ADJUSTING] Reducing chunk size from {chunk_size:,} to {new_size:,}", 'warning')
                    chunk_size = max(new_size, self.min_chunk_size)
                continue
            
            # Save successful chunk results
            result_data = {
                "filepath": filepath,
                "chunk_index": chunk_index,
                "start_row": current_row,
                "end_row": current_row + chunk_result["actual_rows"],
                "metrics": chunk_result,
                "system_info": sys_info,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(output_file, "w") as f:
                json.dump(result_data, f, indent=2)
                
            self.print_color(f"[SUCCESS] Saved {output_file} | "
                  f"RAM: {chunk_result['ram_usage_mb']:.2f}MB | "
                  f"CPU: {chunk_result['cpu_percent']:.1f}% | "
                  f"Time: {chunk_result['load_time_sec']:.2f}s | "
                  f"Throughput: {chunk_result['throughput_rows_sec']:,.0f} rows/s", 'success')
            
            # Update tracking
            all_results.append(result_data)
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
                    self.print_color(f"[OPTIMIZING] Increasing chunk size to {new_size:,} rows", 'info')
                    chunk_size = new_size

        # Generate comprehensive summary
        summary = {
            "version": self.VERSION,
            "system_info": sys_info,
            "dataset_info": {
                "filepath": filepath,
                "total_rows": total_rows,
                "processed_rows": current_row,
                "chunks_tested": len(all_results)
            },
            "performance_metrics": {
                "average_throughput": sum(
                    r["metrics"]["throughput_rows_sec"] for r in all_results
                ) / len(all_results),
                "max_chunk_size": max(
                    r["metrics"]["actual_rows"] for r in all_results
                ),
                "min_memory_per_row": min(
                    r["metrics"]["mem_per_row"] for r in all_results
                ),
                "max_cpu_usage": max(
                    r["metrics"].get("cpu_percent", 0) for r in all_results
                ),
                "total_processing_time": sum(
                    r["metrics"]["load_time_sec"] for r in all_results
                )
            },
            "chunk_details": [{
                "file": f"max_rows_test-{r['chunk_index']}.json",
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
            }
        }

        # Save summary files
        summary_file = Path(output_dir) / "testing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        # Save CSV version
        csv_file = Path(output_dir) / "testing_summary.csv"
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
        
        # Generate plots
        self.generate_plots(all_results, output_dir)
        
        # Update history
        self.update_history(summary)
        
        self.print_color(f"\n[SUMMARY] Saved to {summary_file} and {csv_file}", 'success')
        self.print_color(f"1. Processed {current_row:,} rows in {len(all_results)} chunks", 'info')
        self.print_color(f"2. Average throughput: {summary['performance_metrics']['average_throughput']:,.0f} rows/sec", 'info')
        self.print_color(f"3. Maximum safe chunk size: {summary['performance_metrics']['max_chunk_size']:,} rows", 'info')
        self.print_color(f"4. Peak CPU usage: {summary['performance_metrics']['max_cpu_usage']:.1f}%", 'info')
        
        return summary

    def display_banner(self) -> None:
        """Display the interactive console UI banner."""
        self.print_color("\n" + "=" * 60, 'highlight')
        self.print_color(f" MEMORY-AWARE CSV CHUNK TESTER v{self.VERSION} ", 'highlight')
        self.print_color("=" * 60, 'highlight')
        self.print_color("Adaptive CSV processing with memory safety", 'system')
        self.print_color("=" * 60 + "\n", 'highlight')

    def display_main_menu(self) -> None:
        """Display the main menu options."""
        self.print_color("\nMAIN MENU:", 'highlight')
        self.print_color("1. Run new test")
        self.print_color("2. View recent test history")
        self.print_color("3. Configure settings")
        self.print_color("4. System information")
        self.print_color("5. Exit\n")

    def display_test_history(self) -> None:
        """Display recent test history from saved file with enhanced formatting."""
        try:
            if not self.history_file.exists():
                self.print_color("\nNo test history available - no history file found", 'warning')
                return
                
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            if not history:
                self.print_color("\nNo test history found - history file is empty", 'warning')
                return
                
            self.print_color("\n" + "=" * 60, 'highlight')
            self.print_color(" RECENT TEST HISTORY (LAST 5 TESTS) ", 'highlight')
            self.print_color("=" * 60, 'highlight')
            
            for i, test in enumerate(history[-5:], 1):  # Show last 5 tests
                self.print_color(f"\nTEST {i}:", 'highlight')
                self.print_color(f"1. Timestamp: {test['timestamp']}", 'info')
                self.print_color(f"2. Dataset: {test['dataset_info']['filepath']}", 'info')
                self.print_color(f"3. Rows: {test['dataset_info']['processed_rows']:,} in {test['dataset_info']['chunks_tested']} chunks", 'info')
                self.print_color(f"4. Avg Throughput: {test['performance_metrics']['average_throughput']:,.0f} rows/sec", 'info')
                self.print_color(f"5. Max Chunk Size: {test['performance_metrics']['max_chunk_size']:,} rows", 'info')
                self.print_color(f"6. Peak CPU: {test['performance_metrics'].get('max_cpu_usage', 'N/A')}%", 'info')
                self.print_color(f"7. Total Time: {test['performance_metrics']['total_processing_time']:.2f} sec", 'info')
            
            self.print_color("\n" + "=" * 60, 'highlight')
            
        except json.JSONDecodeError:
            self.print_color("\nWarning: Test history file is corrupted", 'warning')
        except Exception as e:
            self.print_color(f"\nError reading test history: {str(e)}", 'error')

    def display_system_info(self) -> None:
        """Display detailed system information."""
        self.print_color("\n" + "=" * 60, 'highlight')
        self.print_color(" SYSTEM INFORMATION ", 'highlight')
        self.print_color("=" * 60, 'highlight')
        
        info = self._get_system_info()
        for key, value in info.items():
            if key.endswith('_gb'):
                self.print_color(f"{key.replace('_', ' ').title():<25}: {value:.2f} GB", 'info')
            elif key == 'timestamp':
                self.print_color(f"{key.replace('_', ' ').title():<25}: {datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M:%S')}", 'info')
            else:
                self.print_color(f"{key.replace('_', ' ').title():<25}: {value}", 'info')
        
        self.print_color("=" * 60, 'highlight')

    def interactive_menu(self) -> None:
        """Interactive console menu for the application."""
        self.display_banner()
        
        config = self.load_config()
        datasets = self.get_available_datasets()
        
        while True:
            self.display_main_menu()
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':  # Run new test
                self.print_color("\nAVAILABLE DATASETS:", 'highlight')
                for i, dataset in enumerate(datasets, 1):
                    self.print_color(f"{i}. {dataset}", 'info')
                
                dataset_choice = input("\nSelect dataset (number) or enter path: ").strip()
                try:
                    if dataset_choice.isdigit():
                        filepath = os.path.join("datasets", datasets[int(dataset_choice)-1])
                    else:
                        filepath = dataset_choice
                    
                    # Get parameters with defaults from config
                    output_dir = input(f"Output directory [{config['default_output_dir']}]: ").strip() or config['default_output_dir']
                    target_mb = int(input(f"Target memory per chunk (MB) [{config['default_target_mb']}]: ").strip() or config['default_target_mb'])
                    min_chunk = int(input(f"Minimum chunk size [{config['default_min_chunk']}]: ").strip() or config['default_min_chunk'])
                    
                    # Update config with new values
                    config['default_output_dir'] = output_dir
                    config['default_target_mb'] = target_mb
                    config['default_min_chunk'] = min_chunk
                    if filepath not in config['recent_files']:
                        config['recent_files'].append(filepath)
                    self.save_config(config)
                    
                    # Run the test
                    self.min_chunk_size = min_chunk
                    self.run_test(filepath, output_dir, target_mb, min_chunk)
                    
                    input("\nPress Enter to continue...")
                except (IndexError, ValueError) as e:
                    self.print_color(f"Invalid input: {str(e)}", 'error')
            
            elif choice == '2':  # View history
                self.display_test_history()
                input("\nPress Enter to continue...")
            
            elif choice == '3':  # Configure settings
                self.print_color("\nCURRENT CONFIGURATION:", 'highlight')
                for key, value in config.items():
                    if key != 'recent_files':  # Skip potentially long list
                        self.print_color(f"{key}: {value}", 'info')
                
                new_output = input(f"\nNew output directory [{config['default_output_dir']}]: ").strip()
                if new_output:
                    config['default_output_dir'] = new_output
                
                new_target = input(f"New target memory (MB) [{config['default_target_mb']}]: ").strip()
                if new_target:
                    config['default_target_mb'] = int(new_target)
                
                new_min = input(f"New minimum chunk size [{config['default_min_chunk']}]: ").strip()
                if new_min:
                    config['default_min_chunk'] = int(new_min)
                
                self.save_config(config)
                self.print_color("Configuration updated successfully!", 'success')
                input("\nPress Enter to continue...")
            
            elif choice == '4':  # System information
                self.display_system_info()
                input("\nPress Enter to continue...")
            
            elif choice == '5':  # Exit
                self.print_color("Exiting... Thank you for using Memory-Aware CSV Chunk Tester!", 'highlight')
                break
            
            else:
                self.print_color("Invalid choice. Please try again.", 'error')

def main():
    """Main entry point for the application."""
    tester = MemoryAwareChunkTester()
    
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        # Use argparse for command-line mode
        parser = argparse.ArgumentParser(
            description="Adaptive CSV Chunk Tester - Optimizes chunk sizes for memory safety and throughput",
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
        
        args = parser.parse_args()
        
        tester.min_chunk_size = args.min_chunk
        tester.run_test(args.file, args.output, args.mb, args.min_chunk)
    else:
        # No command line args, launch interactive mode
        tester.interactive_menu()

if __name__ == "__main__":
    main()