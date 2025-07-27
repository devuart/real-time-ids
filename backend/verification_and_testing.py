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
    "dataset_path": "models/preprocessed_dataset.csv"
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
            from tabulate import tabulate
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
    """Display rich terminal output with proper ANSI handling"""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.text import Text
        
        console = Console()
        table = Table(
            title="Verification & Testing Run Summary",
            box=box.ROUNDED,
            header_style="bold magenta",
            title_style="bold yellow",
            show_lines=True
        )
        
        table.add_column("Step", style="bold cyan", width=25)
        table.add_column("Status", justify="center", style="bold", width=12)
        table.add_column("Duration", justify="right", style="bold", width=10)
        table.add_column("Timestamp", style="bold dim", width=20)
        
        status_styles = {
            "SUCCESS": "bold green",
            "WARNING": "bold yellow",
            "FAIL": "bold red",
            "ERROR": "bold red"
        }
        
        for entry in filtered_history:
            status = entry['status'].split()[0]
            status_style = status_styles.get(status, "bold")
            table.add_row(
                f"[cyan]{entry['label']}",
                f"[{status_style}]{entry['status']}",
                f"[bold]{entry['duration']}",
                f"[dim]{entry['timestamp']}"
            )
        
        console.print(table)
        
        # Create styled text that matches Fore.WHITE + Style.BRIGHT + dim timestamp
        last_run_text = Text.assemble(
            ("Last Run: ", "bold white"),
            (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "bold magenta")
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
    print(Fore.WHITE + Style.BRIGHT + "9. Show Last Summary")
    print(Fore.WHITE + Style.BRIGHT + "10. Show Last HTML Summary")
    print(Fore.WHITE + Style.BRIGHT + "11. Cleanup Old Results")
    print(Fore.RED + Style.BRIGHT + "12. Exit")

def substitute_command(cmd: List[str], config: Dict[str, Any]) -> List[str]:
    """Substitute variables in command with config values"""
    return [part.format(**config) if isinstance(part, str) else part for part in cmd]

def run_step_sync(label: str, command: List[str], config: Dict[str, Any]) -> bool:
    """Run a step synchronously with simulated progress"""
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
            from tqdm import tqdm
            import time
            
            # Create progress bar with custom format
            with tqdm(
                total=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                desc=f"{Fore.YELLOW}{Style.BRIGHT}[progress] {label}",
                ncols=80
            ) as pbar:
                # Start the subprocess
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Simulate progress while process is running
                while proc.poll() is None:
                    # Update progress in 5% increments
                    if pbar.n < 95:  # Don't go to 100% until done
                        pbar.update(5)
                    time.sleep(0.1)  # Update interval
                
                # Get final output
                stdout, stderr = proc.communicate()
                pbar.update(100 - pbar.n)  # Complete to 100%
                
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, command, stdout, stderr)
                
                result = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
                
        except ImportError:
            # Fallback to original behavior if tqdm not available
            print(Fore.YELLOW + Style.BRIGHT + "[progress] Running...", end='', flush=True)
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        duration = datetime.now() - start
        entry.update({
            "status": "SUCCESS",
            "duration": str(duration).split(".")[0],
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
    """Run a step asynchronously with simulated progress"""
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
            from tqdm.asyncio import tqdm
            import time
            
            # Create async progress bar
            with tqdm(
                total=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                desc=f"{Fore.YELLOW}{Style.BRIGHT}[progress] {label}",
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
                
        except ImportError:
            # Fallback to original behavior
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
            print("\033c", end="")  # Clear terminal
            show_config(edit_mode=True)
            
            # Get user selection
            print(Fore.WHITE + Style.BRIGHT + "\n" + "-" * 40)
            print(Fore.MAGENTA + Style.BRIGHT + "Edit options:")
            print(Fore.CYAN + Style.BRIGHT + "1-{0} to edit setting".format(len(config)))
            print(Fore.YELLOW + Style.BRIGHT + "s" + Fore.WHITE + Style.BRIGHT + " - Save and exit")
            print(Fore.YELLOW + Style.BRIGHT + "q" + Fore.WHITE + Style.BRIGHT + " - Quit without saving")
            choice = input(Fore.WHITE + Style.BRIGHT + "\nSelect option: ").strip().lower()
            
            if choice == 's':
                try:
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
                
            elif choice.isdigit() and 1 <= int(choice) <= len(config):
                key = list(config.keys())[int(choice)-1]
                current_val = config[key]
                
                while True:
                    new_val = input(f"\nNew value for {key} [{current_val}]: ").strip()
                    if not new_val:  # User pressed enter without input
                        break
                        
                    new_val = sanitize_input(new_val)
                    
                    # Special handling for paths
                    if key.endswith('_path'):
                        if not validate_path(new_val):
                            print(Fore.RED + Style.BRIGHT + f"⚠ Path does not exist: {new_val}")
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
                    
                    config[key] = new_val
                    break
            else:
                print(Fore.RED + Style.BRIGHT + "Invalid selection!")
                input(Fore.WHITE + Style.BRIGHT + "Press Enter to continue...")
                
    except Exception as e:
        logger.error(f"Error in edit_config: {str(e)}")
        print(Fore.RED + "\nError in configuration editor!")
        traceback.print_exc()

def show_config(edit_mode: bool = False) -> None:
    """Display configuration with enhanced formatting and validation
    
    Args:
        edit_mode: If True, shows numeric prefixes for editing
    """
    try:
        config = load_config()
        print(Fore.CYAN + "\n[CONFIGURATION]")
        print(Fore.BLUE + f"Config file: {CONFIG_PATH.resolve()}\n")
        
        # Prepare validation indicators
        path_status = {}
        for key, value in config.items():
            if key.endswith('_path'):
                path_status[key] = "✓" if validate_path(value) else "✗"
        
        # Try rich formatting first
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            from rich.text import Text
            
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
            
            for i, (key, value) in enumerate(config.items(), 1):
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
            pass  # Fall through to next formatting option
        
        # Try tabulate formatting
        try:
            from tabulate import tabulate
            
            headers = ["#"] if edit_mode else []
            headers += ["Setting", "Value"]
            if path_status:
                headers.append("Status")
            
            rows = []
            for i, (key, value) in enumerate(config.items(), 1):
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
            pass  # Fall through to basic formatting
        
        # Basic formatting with alignment
        max_key_len = max(len(key) for key in config.keys())
        for i, (key, value) in enumerate(config.items(), 1):
            prefix = f"{i}. " if edit_mode else ""
            status = f" [{path_status[key]}]" if key in path_status else ""
            modified = " *" if value != DEFAULT_CONFIG.get(key) else ""
            print(f"{prefix}{key.ljust(max_key_len)} : {value}{modified}{status}")
            
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Error displaying config: {str(e)}")
        print(Fore.RED + Style.BRIGHT + "\nError displaying configuration. Check logs for details.")

def cleanup_old_results(days_to_keep: int = 7) -> None:
    """Cleanup old result files"""
    try:
        now = datetime.now()
        cutoff = now.timestamp() - (days_to_keep * 86400)
        
        # Cleanup old history entries
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH) as f:
                history = json.load(f)
            
            filtered_history = [
                entry for entry in history
                if datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S').timestamp() > cutoff
            ]
            
            if len(filtered_history) < len(history):
                with open(HISTORY_PATH, "w") as f:
                    json.dump(filtered_history, f, indent=2)
                print(Fore.GREEN + f"Removed {len(history) - len(filtered_history)} old history entries")
        
        # Cleanup old report files
        report_files = list(SUMMARY_DIR.glob("verification_testing_summary*"))
        for report_file in report_files:
            file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
            if file_time.timestamp() < cutoff:
                report_file.unlink()
                print(Fore.YELLOW + f"Removed old report: {report_file.name}")
        
        print(Fore.GREEN + "Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

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
            config = load_config()  # Reload config after editing
        elif choice == "8":
            show_config()
        elif choice == "9":
            print(Fore.CYAN + Style.BRIGHT + "\n[LAST SUMMARY]")
            show_last_summary(OutputFormat.TERMINAL)
        elif choice == "10":
            print(Fore.CYAN + Style.BRIGHT + "\n[LAST HTML SUMMARY]")
            show_last_summary(OutputFormat.HTML)
        elif choice == "11":
            try:
                days = int(input(Fore.WHITE + Style.BRIGHT + "Enter number of days to keep (default 7): ") or "7")
                cleanup_old_results(days)
            except ValueError:
                print(Fore.RED + Style.BRIGHT + "Invalid input. Please enter a number.")
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