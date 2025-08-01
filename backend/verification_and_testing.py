import asyncio
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
import json
import os
import logging
import shutil
from filelock import FileLock
import re
from colorama import Fore, Style, init
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import threading
import itertools
import time
from alive_progress import alive_bar
import tqdm
from tqdm.asyncio import tqdm
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from tabulate import tabulate

# Get virtual environment's Python path
VENV_PYTHON = Path(sys.executable)

init(autoreset=True)

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = "verification_testing.log"
LOG_FILE = LOG_DIR / LOG_FILE_NAME
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger(__name__)

# Configuration files directory
CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE_NAME = "verification_testing_config.json"
CONFIG_PATH = CONFIG_DIR / CONFIG_FILE_NAME

# History directory
HISTORY_DIR = Path("logs")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE_NAME = "verification_testing_history.json"
HISTORY_PATH = HISTORY_DIR / HISTORY_FILE_NAME
HISTORY_LOCK = FileLock(str(HISTORY_PATH) + ".lock")

# Summary files
SUMMARY_DIR = Path("reports")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE_NAME = "verification_testing_summary"
SUMMARY_MD = SUMMARY_DIR / (SUMMARY_FILE_NAME + ".md")
SUMMARY_HTML = SUMMARY_DIR / (SUMMARY_FILE_NAME + ".html")

# Default configuration values
DEFAULT_CONFIG = {
    "model_path": "models/ids_model.onnx",
    "dataset_path": "models/preprocessed_dataset.csv",
    # Default number of days to keep
    "cleanup_days": 7,
    "_last_modified": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Steps to run
STEPS = [
    {"label": "Input Validation", "command": [str(VENV_PYTHON), "check.py"]},
    {"label": "Output Validation", "command": [str(VENV_PYTHON), "check2.py", "--model", "{model_path}"]},
    {"label": "Inference + Evaluation", "command": [str(VENV_PYTHON), "test_onnx_model.py", "--model", "{model_path}", "--validate", "--benchmark"]},
    {"label": "Quick Sanity Check", "command": [str(VENV_PYTHON), "verify.py", "--model", "{model_path}"]}
]

class OutputFormat(Enum):
    MARKDOWN = auto()
    HTML = auto()
    TERMINAL = auto()

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

def validate_path(path: str) -> bool:
    """Validate that a path exists"""
    return Path(path).exists()

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent command injection"""
    return re.sub(r'[;&|$]', '', input_str).strip()

def load_config() -> Dict[str, Any]:
    """Load configuration from file with error handling"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = json.load(f)
                # Validate paths in config
                for path_key in ['model_path', 'dataset_path']:
                    if path_key in config and not validate_path(config[path_key]):
                        logger.warning(f"Configured path does not exist: {config[path_key]}")
                return config
        else:
            with open(CONFIG_PATH, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            return DEFAULT_CONFIG
    except json.JSONDecodeError:
        logger.error("Config file is corrupted, creating new one")
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return DEFAULT_CONFIG

def save_history(entry: Dict[str, Any]) -> None:
    """Save history entry with file locking"""
    try:
        with HISTORY_LOCK:
            history = []
            if HISTORY_PATH.exists():
                try:
                    with open(HISTORY_PATH) as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    logger.error("History file corrupted, starting fresh")
                    history = []

            history.append(entry)
            
            with open(HISTORY_PATH, "w") as f:
                json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}")

def write_summary(history: List[Dict[str, Any]], format: OutputFormat = OutputFormat.MARKDOWN) -> None:
    """Write summary in the specified format with enhanced presentation"""
    try:
        # Filter history to get most recent run of each step
        seen_labels = set()
        filtered_history = []
        for entry in reversed(history):
            if entry['label'] not in seen_labels:
                seen_labels.add(entry['label'])
                filtered_history.append(entry)
                if len(seen_labels) == len(STEPS):
                    break
        filtered_history = list(reversed(filtered_history))

        if format == OutputFormat.MARKDOWN:
            _write_markdown_summary(filtered_history)
        elif format == OutputFormat.HTML:
            _write_html_summary(filtered_history)
        elif format == OutputFormat.TERMINAL:
            _display_rich_summary(filtered_history)
            
    except Exception as e:
        logger.error(f"Error writing summary: {str(e)}")

def _write_markdown_summary(filtered_history: List[Dict[str, Any]]) -> None:
    """Write markdown summary with multiple formatting options and robust fallbacks"""
    try:
        # Header content
        header_content = [
            "# Verification & Testing Run Summary",
            "",
            f"**Last Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # Try tabulate first (prettiest format)
        try:
            headers = ["Step", "Status", "Duration", "Timestamp"]
            rows = [
                [
                    entry['label'],
                    f"**{entry['status']}**" if entry['status'] == "SUCCESS" else entry['status'],
                    entry['duration'],
                    entry['timestamp']
                ]
                for entry in filtered_history
            ]
            
            table = tabulate(rows, headers, tablefmt="pipe")
            content = header_content + [table]
            
        except ImportError:
            # Fallback to manual alignment with emoji indicators
            try:
                max_label_len = max(len(entry['label']) for entry in filtered_history)
                max_status_len = max(len(entry['status']) for entry in filtered_history)
                max_duration_len = max(len(entry['duration']) for entry in filtered_history)
                
                # Ensure minimum column widths
                max_label_len = max(max_label_len, len("Step"))
                max_status_len = max(max_status_len, len("Status"))
                max_duration_len = max(max_duration_len, len("Duration"))
                
                # Build header
                header = (
                    f"| {'Step'.ljust(max_label_len)} | "
                    f"{'Status'.ljust(max_status_len)} | "
                    f"{'Duration'.ljust(max_duration_len)} | "
                    "Timestamp           |"
                )
                separator = (
                    f"|{'-' * (max_label_len + 2)}|"
                    f"{'-' * (max_status_len + 2)}|"
                    f"{'-' * (max_duration_len + 2)}|"
                    "--------------------|"
                )
                
                # Build rows
                rows = []
                for entry in filtered_history:
                    status = (
                        f"✅ {entry['status']}" 
                        if entry['status'] == "SUCCESS" 
                        else f"❌ {entry['status']}"
                    )
                    rows.append(
                        f"| {entry['label'].ljust(max_label_len)} | "
                        f"{status.ljust(max_status_len)} | "
                        f"{entry['duration'].ljust(max_duration_len)} | "
                        f"{entry['timestamp']} |"
                    )
                
                table = "\n".join([header, separator] + rows)
                content = header_content + [table]
                
            except Exception as fallback_error:
                logger.error(f"Manual formatting failed: {str(fallback_error)}")
                # Ultimate fallback to basic format
                content = header_content + [
                    "| Step | Status | Duration | Timestamp |",
                    "|------|--------|----------|-----------|"
                ] + [
                    f"| {entry['label']} | {entry['status']} | {entry['duration']} | {entry['timestamp']} |"
                    for entry in filtered_history
                ]

        # Write to file
        with open(SUMMARY_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")
            
    except Exception as e:
        logger.error(f"Error writing summary: {str(e)}")
        raise

def _write_simple_markdown(filtered_history: List[Dict[str, Any]]) -> None:
    """Fallback markdown writer without tabulate"""
    lines = [
        "# Verification & Testing Run Summary",
        "",
        f"**Last Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Step | Status | Duration | Timestamp |",
        "|------|--------|----------|-----------|"
    ]
    
    for entry in filtered_history:
        lines.append(f"| {entry['label']} | {entry['status']} | {entry['duration']} | {entry['timestamp']} |")

    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines))

def _write_html_summary(filtered_history: List[Dict[str, Any]]) -> None:
    """Write HTML version with styling"""
    status_colors = {
        "SUCCESS": "green",
        "WARNING": "orange",
        "FAIL": "red",
        "ERROR": "red"
    }

    rows = []
    for entry in filtered_history:
        status = entry['status'].split()[0]
        color = status_colors.get(status, "black")
        rows.append(f"""
            <tr>
                <td>{entry['label']}</td>
                <td style="color: {color}; font-weight: bold">{entry['status']}</td>
                <td>{entry['duration']}</td>
                <td>{entry['timestamp']}</td>
            </tr>
        """)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Verification Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            th, td {{ padding: 12px 15px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #e9ecef; }}
            .last-run {{ color: #6c757d; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Verification & Testing Run Summary</h1>
        <p class="last-run"><strong>Last Run:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <table>
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </body>
    </html>
    """

    with open(SUMMARY_HTML, "w") as f:
        f.write(html)

def _display_rich_summary(filtered_history: List[Dict[str, Any]]) -> None:
    """Display rich terminal output with proper ANSI handling and relative timestamps"""
    try:
        console = Console()
        table = Table(
            title="Verification & Testing Run Summary",
            box=box.ROUNDED,
            header_style="bold magenta",
            title_style="bold yellow",
            show_lines=True
        )
        
        table.add_column("Step", style="bold cyan", width=25)
        table.add_column("Status", justify="left", style="bold", width=12)
        table.add_column("Duration", justify="left", style="bold", width=10)
        table.add_column("Timestamp", style="bold dim", width=35)
        
        status_styles = {
            "SUCCESS": "bold green",
            "WARNING": "bold yellow",
            "FAIL": "bold red",
            "ERROR": "bold red"
        }
        
        current_time = datetime.now()
        
        for entry in filtered_history:
            # Clean status text by removing markdown formatting
            raw_status = entry['status'].replace('**', '').replace('*', '').strip()
            status_key = raw_status.split()[0]  # Get first word for styling
            
            # Parse timestamp and calculate relative time
            timestamp_str = entry['timestamp']
            try:
                # Handle different timestamp formats (with or without microseconds)
                if ':' in timestamp_str and timestamp_str.count(':') > 2:
                    # Timestamp with microseconds (e.g., "2025-08-01 00:12:16:149907")
                    dt = datetime.strptime(':'.join(timestamp_str.split(':')[:3]), '%Y-%m-%d %H:%M:%S')
                else:
                    # Standard timestamp (e.g., "2025-08-01 00:12:16")
                    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                time_diff = current_time - dt
                total_seconds = int(time_diff.total_seconds())
                
                # Calculate relative time string
                if time_diff.days > 365:
                    years = time_diff.days // 365
                    relative_time = f"{years} year{'s' if years > 1 else ''} ago"
                elif time_diff.days > 30:
                    months = time_diff.days // 30
                    relative_time = f"{months} month{'s' if months > 1 else ''} ago"
                elif time_diff.days > 0:
                    relative_time = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
                elif total_seconds >= 3600:
                    hours = total_seconds // 3600
                    relative_time = f"{hours} hour{'s' if hours > 1 else ''} ago"
                elif total_seconds >= 60:
                    minutes = total_seconds // 60
                    relative_time = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                elif total_seconds >= 5:
                    # Only show exact seconds if divisible by 5, otherwise show rounded
                    if total_seconds % 5 == 0:
                        relative_time = f"{total_seconds} seconds ago"
                    else:
                        rounded_seconds = 5 * round(total_seconds / 5)
                        relative_time = f"~{rounded_seconds} seconds ago"
                elif total_seconds > 1:
                    # Show exact count for 2-4 seconds
                    relative_time = f"{total_seconds} second{'s' if total_seconds > 1 else ''} ago"
                else:
                    relative_time = "just now"
                
                timestamp_display = f"{timestamp_str.split('.')[0]}\n[dim italic]({relative_time})"
                
            except ValueError:
                # Fallback if parsing fails
                timestamp_display = timestamp_str
            
            status_style = status_styles.get(status_key, "bold")
            table.add_row(
                f"[cyan]{entry['label']}",
                f"[{status_style}]{raw_status}",
                f"[bold]{entry['duration']}",
                timestamp_display
            )
        
        console.print(table)
        
        last_run_text = Text.assemble(
            ("Last Run: ", "bold white"),
            (f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}", "bold magenta")
        )
        console.print(last_run_text)
        
    except ImportError:
        logger.warning("rich not available, falling back to simple print")
        _write_simple_markdown(filtered_history)
        with open(SUMMARY_MD) as f:
            print(f.read())

def show_last_summary(format: OutputFormat = OutputFormat.TERMINAL) -> None:
    """Show the last summary in the specified format"""
    try:
        if not HISTORY_PATH.exists():
            print(Fore.RED + Style.BRIGHT + "\nNo history found. Run at least one step first.")
            return

        with open(HISTORY_PATH) as f:
            history = json.load(f)
        
        if format == OutputFormat.HTML:
            write_summary(history, OutputFormat.HTML)
            import webbrowser
            webbrowser.open(f"file://{SUMMARY_HTML.resolve()}")
        elif format == OutputFormat.TERMINAL:
            _display_rich_summary(_read_history_from_md())
        else:
            write_summary(history, OutputFormat.MARKDOWN)
            with open(SUMMARY_MD) as f:
                print(f.read())
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Error showing summary: {str(e)}")

def _read_history_from_md() -> List[Dict[str, Any]]:
    """Helper to read history from markdown file"""
    history = []
    if SUMMARY_MD.exists():
        with open(SUMMARY_MD) as f:
            for line in f:
                line = line.strip()
                # Skip separator lines and header rows
                if (line.startswith('|') and not line.startswith('|---') 
                    and 'Step' not in line and 'Status' not in line
                    and ':--------' not in line):  # Add this to catch separator patterns
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if len(parts) == 4:
                        history.append({
                            'label': parts[0],
                            'status': parts[1],
                            'duration': parts[2],
                            'timestamp': parts[3]
                        })
    return history

def banner() -> None:
    """Print banner"""
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "      IDS | VERIFICATION & TESTING SUITE".center(60))
    print(Fore.CYAN + Style.BRIGHT + "=" * 60 + Style.RESET_ALL)

def print_menu() -> None:
    """Print menu options"""
    print(Fore.YELLOW + Style.BRIGHT + "\nAvailable Options:")
    print(Fore.WHITE + Style.BRIGHT + "1. Run Input Validation" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "        (check.py)")
    print(Fore.WHITE + Style.BRIGHT + "2. Run Output Validation" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "       (check2.py)")
    print(Fore.WHITE + Style.BRIGHT + "3. Run Inference + Evaluation" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "  (test_onnx_model.py)")
    print(Fore.WHITE + Style.BRIGHT + "4. Run Quick Sanity Check" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "      (verify.py)")
    print(Fore.WHITE + Style.BRIGHT + "5. Run All Tests Sequentially" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "  (sync mode)")
    print(Fore.WHITE + Style.BRIGHT + "6. Run All Tests Concurrently" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "  (async mode)")
    print(Fore.WHITE + Style.BRIGHT + "7. Edit Config")
    print(Fore.WHITE + Style.BRIGHT + "8. Show Config")
    print(Fore.WHITE + Style.BRIGHT + "9. Show Last Summary" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "           (console)")
    print(Fore.WHITE + Style.BRIGHT + "10. Show Last HTML Summary" + Fore.LIGHTGREEN_EX + Style.BRIGHT + "     (browser)")
    print(Fore.WHITE + Style.BRIGHT + "11. Cleanup Old Results")
    print(Fore.RED + Style.BRIGHT + "12. Exit")

def substitute_command(cmd: List[str], config: Dict[str, Any]) -> List[str]:
    """Substitute variables in command with config values"""
    return [part.format(**config) if isinstance(part, str) else part for part in cmd]

def run_step_sync(label: str, command: List[str], config: Dict[str, Any]) -> bool:
    """Run a step synchronously with alive-progress indication"""
    print(Fore.MAGENTA + Style.BRIGHT + f"\n=== {label} ===")
    logger.info(f"Starting {label}")
    start = datetime.now()

    entry = {
        "label": label,
        "timestamp": start.strftime("%Y-%m-%d %H:%M:%S:%f"),
        "command": " ".join(command),
        "status": "FAILED",
        "duration": "0s"
    }

    try:
        # Validate paths in command
        for part in command:
            if part.startswith("--") and part[2:] in config and not validate_path(config[part[2:]]):
                raise ConfigError(f"Path does not exist: {config[part[2:]]}")

        try:
            # Start the progress bar
            with alive_bar(
                manual=False,
                title=f"{Fore.YELLOW}{Style.BRIGHT}[progress] {label}",
                bar='smooth',
                spinner='dots_waves',
                stats=False,
                monitor=False
            ) as bar:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                while proc.poll() is int | None:
                    bar()
                    time.sleep(0.2)

                stdout, stderr = proc.communicate()

                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, command, stdout, stderr)

                result = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)

        except ImportError:
            print(Fore.YELLOW + Style.BRIGHT + "[progress] Running...", end='', flush=True)
            result = subprocess.run(command, check=True, capture_output=True, text=True)

        duration = datetime.now() - start
        entry.update({
            "status": "SUCCESS",
            "duration": str(duration).split(":")[2],
            "output": result.stdout
        })

        print(Fore.GREEN + Style.BRIGHT + f"\r[success] {label} completed successfully in {entry['duration']}")
        logger.info(f"{label} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        duration = datetime.now() - start
        entry.update({
            "duration": str(duration).split(".")[0],
            "error": e.stderr
        })
        print(Fore.RED + Style.BRIGHT + f"\r[error] {label} failed after {entry['duration']}")
        logger.error(f"{label} failed: {e.stderr}")
        return False

    except ConfigError as e:
        duration = datetime.now() - start
        entry.update({
            "duration": str(duration).split(".")[0],
            "error": str(e)
        })
        print(Fore.RED + Style.BRIGHT + f"\r[error] {label} configuration error: {e}")
        logger.error(f"{label} configuration error: {e}")
        return False

    except Exception as e:
        duration = datetime.now() - start
        entry.update({
            "duration": str(duration).split(".")[0],
            "error": str(e)
        })
        print(Fore.RED + Style.BRIGHT + f"\r[error] {label} unexpected error: {e}")
        logger.error(f"{label} unexpected error: {str(e)}")
        traceback.print_exc()
        return False

    finally:
        save_history(entry)

async def run_step_async(label: str, command: List[str], config: Dict[str, Any]) -> bool:
    """Run a step asynchronously with alive-progress indication"""
    print(Fore.MAGENTA + Style.BRIGHT + f"\n>>> {label} started")
    logger.info(f"Starting {label} (async)")
    start = datetime.now()

    entry = {
        "label": label,
        "timestamp": start.strftime("%Y-%m-%d %H:%M:%S:%f"),
        "command": " ".join(command),
        "status": "FAILED",
        "duration": "0s"
    }

    try:
        # Validate paths in command
        for part in command:
            if part.startswith("--") and part[2:] in config and not validate_path(config[part[2:]]):
                raise ConfigError(f"Path does not exist: {config[part[2:]]}")

        try:
            # Create async progress bar
            with tqdm(
                total=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                desc=f"{Fore.WHITE}{Style.BRIGHT}[progress] {label}",
                ncols=80
            ) as pbar:
                # Start the subprocess
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Simulate progress while process is running
                while proc.returncode is None:
                    # Update progress in 5% increments
                    if pbar.n < 95:  # Don't go to 100% until done
                        pbar.update(5)
                    await asyncio.sleep(0.1)  # Update interval
                
                # Get final output
                stdout, stderr = await proc.communicate()
                pbar.update(100 - pbar.n)  # Complete to 100%

            # with alive_bar(
            #     manual=False,
            #     title=f"{Fore.YELLOW}{Style.BRIGHT}[progress] {label}",
            #     bar='smooth',
            #     spinner='dots_waves',
            #     stats=False,
            #     monitor=False
            # ) as bar:
            #     proc = await asyncio.create_subprocess_exec(
            #         *command,
            #         stdout=asyncio.subprocess.PIPE,
            #         stderr=asyncio.subprocess.PIPE
            #     )

            #     while await proc.wait() is None:
            #         bar()
            #         await asyncio.sleep(0.2)

            #     stdout, stderr = await proc.communicate()

        except ImportError:
            print(Fore.YELLOW + Style.BRIGHT + f"[progress] {label} running...", end='', flush=True)
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

        duration = datetime.now() - start
        entry["duration"] = str(duration).split(".")[0]

        if proc.returncode == 0:
            entry.update({
                "status": "SUCCESS",
                "output": stdout.decode()
            })
            print(Fore.GREEN + Style.BRIGHT + f"\r[success] {label} completed in {entry['duration']}")
            logger.info(f"{label} (async) completed successfully")
            return True
        else:
            entry["error"] = stderr.decode()
            print(Fore.RED + Style.BRIGHT + f"\r[error] {label} failed after {entry['duration']}")
            logger.error(f"{label} (async) failed: {entry['error']}")
            return False

    except ConfigError as e:
        duration = datetime.now() - start
        entry.update({
            "duration": str(duration).split(".")[0],
            "error": str(e)
        })
        print(Fore.RED + Style.BRIGHT + f"\r[error] {label} configuration error: {e}")
        logger.error(f"{label} (async) configuration error: {e}")
        return False

    except Exception as e:
        duration = datetime.now() - start
        entry.update({
            "duration": str(duration).split(".")[0],
            "error": str(e)
        })
        print(Fore.RED + Style.BRIGHT + f"\r[error] {label} unexpected error: {e}")
        logger.error(f"{label} (async) unexpected error: {str(e)}")
        traceback.print_exc()
        return False

    finally:
        save_history(entry)

async def run_all_parallel(config: Dict[str, Any]) -> None:
    """Run all steps in parallel"""
    print(Fore.CYAN + "\n[INFO] Running all steps concurrently...\n")
    print(Fore.YELLOW + "This may take a few minutes depending on your system and model size.\n")
    
    config = load_config()
    config['model_path'] = config.get('model_path', DEFAULT_CONFIG['model_path'])
    config['dataset_path'] = config.get('dataset_path', DEFAULT_CONFIG['dataset_path'])
    
    tasks = []
    for step in STEPS:
        command = substitute_command(step["command"], config)
        tasks.append(run_step_async(step["label"], command, config))
    
    await asyncio.gather(*tasks)
    
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        write_summary(history, OutputFormat.MARKDOWN)
        write_summary(history, OutputFormat.HTML)
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")

def edit_config() -> None:
    """Edit configuration with interactive table and validation"""
    try:
        config = load_config()
        
        while True:
            # Clear screen and show config
            print("\033c", end="")
            show_config(edit_mode=True)
            
            # Get user selection
            print(Fore.WHITE + Style.BRIGHT + "\n" + "-" * 40)
            print(Fore.MAGENTA + Style.BRIGHT + "Edit options:")
            # Subtract 1 to exclude _last_modified
            print(Fore.CYAN + Style.BRIGHT + "1-{0} to edit setting".format(len(config)-1))
            print(Fore.YELLOW + Style.BRIGHT + "s" + Fore.WHITE + Style.BRIGHT + " - Save and exit")
            print(Fore.YELLOW + Style.BRIGHT + "q" + Fore.WHITE + Style.BRIGHT + " - Quit without saving")
            choice = input(Fore.WHITE + Style.BRIGHT + "\nSelect option: ").strip().lower()
            
            if choice == 's':
                try:
                    # Update last modified timestamp
                    config['_last_modified'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    with open(CONFIG_PATH, "w") as f:
                        json.dump(config, f, indent=2)
                    print(Fore.GREEN + "\n[success] Configuration saved.\n")
                    break
                except Exception as e:
                    logger.error(f"Error saving config: {str(e)}")
                    print(Fore.RED + "\nError saving configuration!")
                    input("Press Enter to continue...")
                    
            elif choice == 'q':
                print(Fore.YELLOW + "\nDiscarding changes...")
                break
                
            # Adjust for _last_modified
            elif choice.isdigit() and 1 <= int(choice) <= len(config)-1:
                # Get only editable keys (exclude _last_modified)
                editable_keys = [k for k in config.keys() if not k.startswith('_')]
                key = editable_keys[int(choice)-1]
                current_val = config[key]
                
                while True:
                    new_val = input(f"\nNew value for {key} [{current_val}]: ").strip()
                    # User pressed enter without input
                    if not new_val:
                        break
                        
                    new_val = sanitize_input(new_val)
                    
                    # Special handling for paths
                    if key.endswith('_path'):
                        if not validate_path(new_val):
                            print(Fore.RED + Style.BRIGHT + f"[WARNING] Path does not exist: {new_val}")
                            print(Fore.YELLOW + Style.BRIGHT + "a - Accept anyway")
                            print(Fore.YELLOW + Style.BRIGHT + "r - Re-enter value")
                            print(Fore.YELLOW + Style.BRIGHT + "c - Cancel")
                            path_choice = input(Fore.WHITE + Style.BRIGHT + "Choice [a/r/c]: ").strip().lower()
                            
                            if path_choice == 'a':
                                config[key] = new_val
                                break
                            elif path_choice == 'r':
                                continue
                            else:
                                break
                    
                    # Special validation for cleanup_days
                    elif key == 'cleanup_days':
                        try:
                            days = int(new_val)
                            if days <= 0:
                                raise ValueError("Days must be positive")
                            config[key] = days
                            break
                        except ValueError:
                            print(Fore.RED + Style.BRIGHT + "Invalid number of days. Must be a positive integer.")
                            continue
                    
                    config[key] = new_val
                    break
            else:
                print(Fore.RED + Style.BRIGHT + "Invalid selection!")
                input(Fore.WHITE + Style.BRIGHT + "Press Enter to continue...")
                
    except Exception as e:
        logger.error(f"Error in edit_config: {str(e)}")
        print(Fore.RED + Style.BRIGHT + "\nError in configuration editor!")
        traceback.print_exc()

def show_config(edit_mode: bool = False) -> None:
    """Display configuration with enhanced formatting and validation
    
    Args:
        edit_mode: If True, shows numeric prefixes for editing
    """
    try:
        config = load_config()
        print(Fore.CYAN + "\n[CONFIGURATION]")
        print(Fore.BLUE + f"Config file: {CONFIG_PATH.resolve()}")
        
        # Display last modified time if available
        last_modified = config.get('_last_modified', 'Unknown')
        print(Fore.BLUE + f"Last modified: {last_modified}\n")
        
        # Prepare validation indicators (only for real settings, not metadata)
        path_status = {}
        for key, value in config.items():
            if key.endswith('_path') and not key.startswith('_'):
                path_status[key] = "✓" if validate_path(value) else "✗"
        
        # Get only editable config items (exclude metadata keys starting with _)
        display_config = {k: v for k, v in config.items() if not k.startswith('_')}
        
        # Try rich formatting first
        try:
            console = Console()
            table = Table(
                title="Current configuration",
                box=box.ROUNDED,
                header_style="bold magenta",
                title_style="bold yellow",
                show_header=True,
                highlight=edit_mode
            )
            
            if edit_mode:
                table.add_column("#", style="dim", width=3)
            
            table.add_column("Setting", style="bold cyan", width=20)
            table.add_column("Value", style="bold green", overflow="fold")
            table.add_column("Status", justify="center", width=6) if path_status else None
            
            for i, (key, value) in enumerate(display_config.items(), 1):
                row = []
                if edit_mode:
                    row.append(str(i))
                
                row.append(key)
                
                # Highlight non-default values
                if value != DEFAULT_CONFIG.get(key):
                    val_text = Text(str(value), style="bold green")
                else:
                    val_text = Text(str(value))
                row.append(val_text)
                
                # Add validation status
                if key in path_status:
                    status = Text(path_status[key], 
                                style="bold green" if path_status[key] == "✓" else "bold red")
                    row.append(status)
                
                table.add_row(*row)
            
            console.print(table)
            return
            
        except ImportError:
            # Fall through to next formatting option
            pass
        
        # Try tabulate formatting
        try:
            headers = ["#"] if edit_mode else []
            headers += ["Setting", "Value"]
            if path_status:
                headers.append("Status")
            
            rows = []
            for i, (key, value) in enumerate(display_config.items(), 1):
                row = [str(i)] if edit_mode else []
                row.append(key)
                
                # Highlight non-default values
                if value != DEFAULT_CONFIG.get(key):
                    row.append(f"*{value}*")
                else:
                    row.append(value)
                
                # Add validation status
                if key in path_status:
                    row.append(path_status[key])
                
                rows.append(row)
            
            print(tabulate(rows, headers, tablefmt="pretty"))
            return
            
        except ImportError:
            # Fall through to basic formatting
            pass
        
        # Basic formatting with alignment
        max_key_len = max(len(key) for key in display_config.keys())
        for i, (key, value) in enumerate(display_config.items(), 1):
            prefix = f"{i}. " if edit_mode else ""
            status = f" [{path_status[key]}]" if key in path_status else ""
            modified = " *" if value != DEFAULT_CONFIG.get(key) else ""
            print(f"{prefix}{key.ljust(max_key_len)} : {value}{modified}{status}")
            
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Error displaying config: {str(e)}")
        print(Fore.RED + Style.BRIGHT + "\nError displaying configuration. Check logs for details.")

def cleanup_old_results() -> None:
    """Cleanup old result files with configurable retention period and rich UI"""
    try:
        config = load_config()
        default_days = config.get('cleanup_days', 7)
        
        # First scan to show available results in rich table
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH) as f:
                history = json.load(f)
            
            if history:
                # Get all timestamps and last cleanup info
                timestamps = []
                last_cleanup = None
                for entry in history:
                    timestamp_str = entry.get('timestamp', '')
                    if timestamp_str:
                        try:
                            # Try standard format first
                            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            timestamps.append(dt)
                        except ValueError:
                            try:
                                # Handle timestamps with microseconds
                                parts = timestamp_str.split(':')
                                if len(parts) >= 3:
                                    dt = datetime.strptime(':'.join(parts[:3]), '%Y-%m-%d %H:%M:%S')
                                    timestamps.append(dt)
                            except (ValueError, IndexError):
                                continue
                    # Check if this entry is a cleanup record
                    if entry.get('action') in ['started', 'completed']:
                        last_cleanup = entry.get('timestamp')
                
                if timestamps:
                    oldest = min(timestamps)
                    newest = max(timestamps)
                    age_days = (datetime.now() - oldest).days
                    
                    # Create rich table for available results
                    console = Console()
                    table = Table(
                        title="[bold]Available Results Summary[/bold]",
                        box=box.ROUNDED,
                        header_style="bold blue",
                        title_style="bold green"
                    )
                    
                    table.add_column("Metric", style="cyan", width=25)
                    table.add_column("Value", style="magenta")
                    
                    table.add_row("Total Entries", f"[bold]{len(history)}")
                    table.add_row("Oldest Entry", 
                                f"[bold]{oldest.strftime('%Y-%m-%d %H:%M:%S')} "
                                f"[dim]({age_days} days old)")
                    table.add_row("Newest Entry", f"[bold]{newest.strftime('%Y-%m-%d %H:%M:%S')}")
                    if last_cleanup:
                        table.add_row("Last Cleanup", f"[bold yellow]{last_cleanup}")
                    else:
                        table.add_row("Last Cleanup", "[dim]No cleanup records found")
                    
                    console.print(table)
        
        # Get number of days from user
        days_input = input(
            Fore.WHITE + Style.BRIGHT + 
            f"\nEnter number of days to keep (default {default_days}): "
        ).strip()
        
        try:
            days_to_keep = int(days_input) if days_input else default_days
        except ValueError:
            print(Fore.RED + Style.BRIGHT + "Invalid input. Using default value.")
            days_to_keep = default_days
        
        now = datetime.now()
        cutoff = now.timestamp() - (days_to_keep * 86400)
        
        # Preview what will be removed in rich table
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH) as f:
                history = json.load(f)
            
            removable_entries = []
            removable_reports = 0
            report_files = list(SUMMARY_DIR.glob("verification_testing_summary*"))
            
            for entry in history:
                timestamp_str = entry.get('timestamp', '')
                if timestamp_str:
                    try:
                        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            parts = timestamp_str.split(':')
                            if len(parts) >= 3:
                                dt = datetime.strptime(':'.join(parts[:3]), '%Y-%m-%d %H:%M:%S')
                        except (ValueError, IndexError):
                            continue
                    
                    if dt.timestamp() <= cutoff:
                        removable_entries.append({
                            'label': entry.get('label', 'Unknown'),
                            'timestamp': timestamp_str,
                            'status': entry.get('status', 'Unknown')
                        })
            
            for report_file in report_files:
                try:
                    file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_time.timestamp() < cutoff:
                        removable_reports += 1
                except Exception:
                    continue
            
            # Create rich preview table
            console = Console()
            preview_table = Table(
                title=f"[bold]Cleanup Preview (Keeping last {days_to_keep} days)[/bold]",
                box=box.ROUNDED,
                header_style="bold yellow",
                title_style="bold red"
            )
            
            preview_table.add_column("Type", style="cyan")
            preview_table.add_column("Count", style="magenta", justify="right")
            preview_table.add_column("Details", style="dim")
            
            preview_table.add_row(
                "History Entries", 
                f"[bold red]{len(removable_entries)}",
                f"Oldest: {removable_entries[0]['timestamp'] if removable_entries else 'N/A'}"
            )
            preview_table.add_row(
                "Report Files",
                f"[bold red]{removable_reports}",
                f"Located in: {str(SUMMARY_DIR)}"
            )
            
            # Add sample of entries to be removed
            if removable_entries:
                sample_table = Table(
                    title="[dim]Sample of entries to be removed",
                    box=box.SIMPLE,
                    show_header=True,
                    show_lines=False
                )
                sample_table.add_column("Label", style="cyan")
                sample_table.add_column("Timestamp", style="magenta")
                sample_table.add_column("Status", style="red")
                
                for entry in removable_entries[:3]:  # Show first 3 as sample
                    sample_table.add_row(
                        entry['label'],
                        entry['timestamp'],
                        entry['status']
                    )
                if len(removable_entries) > 3:
                    sample_table.add_row(
                        f"... {len(removable_entries)-3} more", 
                        "", 
                        ""
                    )
                
                preview_table.add_row("", "", sample_table)
            
            console.print(preview_table)
            
            # Get confirmation
            confirm = input(Fore.WHITE + Style.BRIGHT + "\nProceed with cleanup? (y/n): ").strip().lower()
            if confirm != 'y':
                print(Fore.YELLOW + "Cleanup canceled")
                return
        
        # Log cleanup attempt
        cleanup_log = {
            "timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
            "days_to_keep": days_to_keep,
            "action": "started"
        }
        logger.info(f"Cleanup started: {json.dumps(cleanup_log)}")
        
        # Actual cleanup process (same as before)
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH) as f:
                history = json.load(f)
            
            filtered_history = []
            removed_count = 0
            
            for entry in history:
                try:
                    timestamp_str = entry.get('timestamp', '')
                    if not timestamp_str:
                        filtered_history.append(entry)
                        continue
                        
                    try:
                        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            parts = timestamp_str.split(':')
                            if len(parts) >= 3:
                                dt = datetime.strptime(':'.join(parts[:3]), '%Y-%m-%d %H:%M:%S')
                            else:
                                raise ValueError("Invalid timestamp format")
                        except (ValueError, IndexError):
                            filtered_history.append(entry)
                            continue
                    
                    if dt.timestamp() > cutoff:
                        filtered_history.append(entry)
                    else:
                        removed_count += 1
                except Exception:
                    filtered_history.append(entry)
                    continue
            
            if removed_count > 0:
                with open(HISTORY_PATH, "w") as f:
                    json.dump(filtered_history, f, indent=2)
        
        # Cleanup report files
        report_files = list(SUMMARY_DIR.glob("verification_testing_summary*"))
        removed_reports = 0
        for report_file in report_files:
            try:
                file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                if file_time.timestamp() < cutoff:
                    report_file.unlink()
                    removed_reports += 1
            except Exception:
                continue
        
        # Log cleanup completion
        cleanup_log.update({
            "action": "completed",
            "removed_history_entries": removed_count,
            "removed_reports": removed_reports,
            "status": "success"
        })
        logger.info(f"Cleanup completed: {json.dumps(cleanup_log)}")
        
        # Show results in rich table
        result_table = Table(
            title="[bold green]Cleanup Results[/bold green]",
            box=box.ROUNDED,
            header_style="bold blue"
        )
        
        result_table.add_column("Item", style="cyan")
        result_table.add_column("Removed", style="magenta", justify="right")
        result_table.add_column("Retained", style="green", justify="right")
        
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH) as f:
                current_history = json.load(f)
            retained_entries = len(current_history)
        else:
            retained_entries = 0
        
        result_table.add_row(
            "History Entries",
            f"[bold red]{removed_count}",
            f"[bold green]{retained_entries}"
        )
        result_table.add_row(
            "Report Files",
            f"[bold red]{removed_reports}",
            f"[bold green]{len(report_files) - removed_reports}"
        )
        result_table.add_row(
            "Retention Period",
            f"[dim]{days_to_keep} days",
            f"[dim]{now.strftime('%Y-%m-%d')}"
        )
        
        console.print(result_table)
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        error_table = Table(
            title="[bold red]Cleanup Error[/bold red]",
            box=box.ROUNDED,
            header_style="bold white",
            title_style="bold red"
        )
        error_table.add_column("Error Details", style="red")
        error_table.add_row(str(e))
        
        console = Console()
        console.print(error_table)
        print(Fore.RED + "\nCheck logs for more details.")

def main() -> None:
    """Main function"""
    config = load_config()
    banner()
    
    while True:
        print_menu()
        choice = input(Fore.WHITE + Style.BRIGHT + "\nSelect an option (1–12): ").strip()
        
        if choice == "1":
            run_step_sync("Input Validation", [str(VENV_PYTHON), "check.py"], config)
        elif choice == "2":
            cmd = substitute_command([str(VENV_PYTHON), "check2.py", "--model", "{model_path}"], config)
            run_step_sync("Output Validation", cmd, config)
        elif choice == "3":
            cmd = substitute_command(
                [str(VENV_PYTHON), "test_onnx_model.py", "--model", "{model_path}", "--validate", "--benchmark"], 
                config
            )
            run_step_sync("Inference + Evaluation", cmd, config)
        elif choice == "4":
            cmd = substitute_command([str(VENV_PYTHON), "verify.py", "--model", "{model_path}"], config)
            run_step_sync("Quick Sanity Check", cmd, config)
        elif choice == "5":
            print(Fore.YELLOW + "\nRunning all steps sequentially...")
            for step in STEPS:
                cmd = substitute_command(step["command"], config)
                if run_step_sync(step["label"], cmd, config):
                    try:
                        with open(HISTORY_PATH) as f:
                            history = json.load(f)
                        write_summary(history, OutputFormat.MARKDOWN)
                        write_summary(history, OutputFormat.HTML)
                    except Exception as e:
                        logger.error(Fore.RED + Style.BRIGHT + f"Error updating summary: {str(e)}")
        elif choice == "6":
            print(Fore.YELLOW + Style.BRIGHT + "\nRunning all steps concurrently...")
            asyncio.run(run_all_parallel(config))
        elif choice == "7":
            edit_config()
            # Reload config after editing
            config = load_config()
        elif choice == "8":
            show_config()
        elif choice == "9":
            print(Fore.CYAN + Style.BRIGHT + "\n[LAST SUMMARY]")
            show_last_summary(OutputFormat.TERMINAL)
        elif choice == "10":
            print(Fore.CYAN + Style.BRIGHT + "\n[OPENING LAST HTML SUMMARY...]")
            show_last_summary(OutputFormat.HTML)
        elif choice == "11":
            cleanup_old_results()
        elif choice == "12":
            print(Fore.CYAN + Style.BRIGHT + "\nExiting... Goodbye!")
            break
        else:
            print(Fore.RED + Style.BRIGHT + "Invalid selection. Choose 1 – 12.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + Style.BRIGHT + "\n\nInterrupted by user. Exiting...")
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error: {str(e)}")
        traceback.print_exc()