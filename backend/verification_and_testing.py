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
from typing import Dict, List, Optional, Any

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
# Ensure the config directory exists
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
# Configuration file name
CONFIG_FILE_NAME = "verification_testing_config.json"
# Path to the configuration file
CONFIG_PATH = CONFIG_DIR / CONFIG_FILE_NAME

# History directory
HISTORY_DIR = Path("logs")
# Ensure the history directory exists
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
# History file names
HISTORY_FILE_NAME = "verification_testing_history.json"
# Path to the history file
HISTORY_PATH = HISTORY_DIR / HISTORY_FILE_NAME
# History file lock
HISTORY_LOCK = FileLock(str(HISTORY_PATH) + ".lock")

# Summary Markdown file
SUMMARY_DIR = Path("reports")
# Ensure the summary directory exists
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
# Summary Markdown file name
SUMMARY_FILE_NAME = "verification_testing_summary.md"
# Path to the summary Markdown file
SUMMARY_MD = SUMMARY_DIR / SUMMARY_FILE_NAME

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

def write_summary_md(history: List[Dict[str, Any]]) -> None:
    """Write summary markdown file"""
    try:
        lines = [
            "# Verification & Testing Run Summary",
            "",
            f"**Last Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| Step | Status | Duration | Timestamp |",
            "|------|--------|----------|-----------|"
        ]
        
        # Show last run of each step type
        seen_labels = set()
        filtered_history = []
        for entry in reversed(history):
            if entry['label'] not in seen_labels:
                seen_labels.add(entry['label'])
                filtered_history.append(entry)
                if len(seen_labels) == len(STEPS):
                    break
        
        for entry in reversed(filtered_history):
            lines.append(f"| {entry['label']} | {entry['status']} | {entry['duration']} | {entry['timestamp']} |")

        with open(SUMMARY_MD, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.error(f"Error writing summary: {str(e)}")

def banner() -> None:
    """Print banner"""
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print("      IDS | VERIFICATION & TESTING SUITE".center(60))
    print("=" * 60 + Style.RESET_ALL)

def print_menu() -> None:
    """Print menu options"""
    print(Fore.YELLOW + "\nAvailable Options:")
    print("1. Run Input Validation        (check.py)")
    print("2. Run Output Validation       (check2.py)")
    print("3. Run Inference + Evaluation  (test_onnx_model.py)")
    print("4. Run Quick Sanity Check      (verify.py)")
    print("5. Run All Tests Sequentially  (sync mode)")
    print("6. Run All Tests Concurrently  (async mode)")
    print("7. Edit Config")
    print("8. Show Config")
    print("9. Show Last Summary")
    print("10. Cleanup Old Results")
    print("11. Exit")

def substitute_command(cmd: List[str], config: Dict[str, Any]) -> List[str]:
    """Substitute variables in command with config values"""
    return [part.format(**config) if isinstance(part, str) else part for part in cmd]

def run_step_sync(label: str, command: List[str], config: Dict[str, Any]) -> bool:
    """Run a step synchronously with progress indication"""
    print(Fore.MAGENTA + f"\n=== {label} ===")
    logger.info(f"Starting {label}")
    start = datetime.now()
    
    entry = {
        "label": label,
        "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(command),
        "status": "FAILED",
        "duration": "0s"
    }
    
    try:
        # Validate paths in command
        for part in command:
            if part.startswith("--") and part[2:] in config and not validate_path(config[part[2:]]):
                raise ConfigError(f"Path does not exist: {config[part[2:]]}")
        
        print(Fore.YELLOW + "[progress] Running...", end='', flush=True)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        duration = datetime.now() - start
        entry["status"] = "SUCCESS"
        entry["duration"] = str(duration).split(".")[0]
        entry["output"] = result.stdout
        
        print(Fore.GREEN + "\r[success] " + f"{label} completed successfully in {entry['duration']}")
        logger.info(f"{label} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        duration = datetime.now() - start
        entry["duration"] = str(duration).split(".")[0]
        entry["error"] = e.stderr
        
        print(Fore.RED + "\r[error] " + f"{label} failed after {entry['duration']}")
        logger.error(f"{label} failed: {e.stderr}")
        return False
    except ConfigError as e:
        duration = datetime.now() - start
        entry["duration"] = str(duration).split(".")[0]
        entry["error"] = str(e)
        
        print(Fore.RED + "\r[error] " + f"{label} configuration error: {e}")
        logger.error(f"{label} configuration error: {e}")
        return False
    except Exception as e:
        duration = datetime.now() - start
        entry["duration"] = str(duration).split(".")[0]
        entry["error"] = str(e)
        
        print(Fore.RED + "\r[error] " + f"{label} unexpected error: {e}")
        logger.error(f"{label} unexpected error: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        save_history(entry)

async def run_step_async(label: str, command: List[str], config: Dict[str, Any]) -> bool:
    """Run a step asynchronously with progress indication"""
    print(Fore.MAGENTA + f"\n>>> {label} started")
    logger.info(f"Starting {label} (async)")
    start = datetime.now()
    
    entry = {
        "label": label,
        "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(command),
        "status": "FAILED",
        "duration": "0s"
    }
    
    try:
        # Validate paths in command
        for part in command:
            if part.startswith("--") and part[2:] in config and not validate_path(config[part[2:]]):
                raise ConfigError(f"Path does not exist: {config[part[2:]]}")
        
        print(Fore.YELLOW + f"[progress] {label} running...", end='', flush=True)
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        duration = datetime.now() - start
        entry["duration"] = str(duration).split(".")[0]
        
        if proc.returncode == 0:
            entry["status"] = "SUCCESS"
            entry["output"] = stdout.decode()
            
            print(Fore.GREEN + f"\r[success] {label} completed in {entry['duration']}")
            logger.info(f"{label} (async) completed successfully")
            return True
        else:
            entry["error"] = stderr.decode()
            
            print(Fore.RED + f"\r[error] {label} failed after {entry['duration']}")
            logger.error(f"{label} (async) failed: {entry['error']}")
            return False
    except ConfigError as e:
        duration = datetime.now() - start
        entry["duration"] = str(duration).split(".")[0]
        entry["error"] = str(e)
        
        print(Fore.RED + f"\r[error] {label} configuration error: {e}")
        logger.error(f"{label} (async) configuration error: {e}")
        return False
    except Exception as e:
        duration = datetime.now() - start
        entry["duration"] = str(duration).split(".")[0]
        entry["error"] = str(e)
        
        print(Fore.RED + f"\r[error] {label} unexpected error: {e}")
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
        write_summary_md(history)
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")

def show_last_summary() -> None:
    """Show the last summary"""
    try:
        if SUMMARY_MD.exists():
            print(Fore.CYAN + "\n--- Summary Markdown Output ---\n")
            with open(SUMMARY_MD) as f:
                print(f.read())
        else:
            print(Fore.RED + "\nNo summary found. Run at least one step first.")
    except Exception as e:
        logger.error(f"Error showing summary: {str(e)}")

def edit_config() -> None:
    """Edit configuration with input validation"""
    config = load_config()
    print(Fore.CYAN + "\n[CONFIGURATION]")
    
    for key, value in config.items():
        while True:
            new_val = input(f"{key} [{value}]: ").strip()
            if not new_val:
                break
                
            new_val = sanitize_input(new_val)
            if key.endswith('_path'):
                if not validate_path(new_val):
                    print(Fore.RED + f"Path does not exist: {new_val}")
                    continue
            
            config[key] = new_val
            break
    
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(Fore.GREEN + "[success] Configuration saved.\n")
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")

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
        report_files = list(SUMMARY_DIR.glob("verification_testing_summary_*.md"))
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
        choice = input(Fore.WHITE + "\nSelect an option (1–11): ").strip()
        
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
                        write_summary_md(history)
                    except Exception as e:
                        logger.error(f"Error updating summary: {str(e)}")
        elif choice == "6":
            print(Fore.YELLOW + "\nRunning all steps concurrently...")
            asyncio.run(run_all_parallel(config))
        elif choice == "7":
            edit_config()
            config = load_config()  # Reload config after editing
        elif choice == "8":
            print(Fore.CYAN + "\n[CONFIGURATION]")
            with open(CONFIG_PATH) as f:
                config = json.load(f)
                for key, value in config.items():
                    print(f"{key}: {value}")
        elif choice == "9":
            print(Fore.CYAN + "\n[LAST SUMMARY]")
            show_last_summary()
        elif choice == "10":
            try:
                days = int(input("Enter number of days to keep (default 7): ") or "7")
                cleanup_old_results(days)
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a number.")
        elif choice == "11":
            print(Fore.CYAN + "\nExiting... Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid selection. Choose 1 – 11.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "\n\nInterrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()