import subprocess
import traceback
import json
import datetime
import asyncio
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

# Directory for storing configuration files
CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "verification_testing_config.json"
CONFIG_FILE = str(CONFIG_FILE)

# Directory for storing history logs of verification and testing runs
HISTORY_LOG_DIR = Path("logs")
HISTORY_LOG_DIR.mkdir(exist_ok=True)
HISTORY_LOG = HISTORY_LOG_DIR / "verification_testing_history.log"
HISTORY_LOG = str(HISTORY_LOG)

# Report file for summary of verification and testing results
REPORT_FILE_DIR = Path("reports")
REPORT_FILE_DIR.mkdir(exist_ok=True)
REPORT_FILE = REPORT_FILE_DIR / "verification_testing_report.md"
REPORT_FILE = str(REPORT_FILE)

DEFAULT_CONFIG = {
    "model_path": "models/ids_model.onnx",
    "dataset_path": "datasets/NF-CSE-CIC-IDS2018.csv"
}

STEPS = [
    {"label": "Input Validation", "command": "python check.py"},
    {"label": "Output Validation", "command": "python check2.py --model {model_path}"},
    {"label": "Full Inference Test", "command": "python test_onnx_model.py --model {model_path} --validate --benchmark"},
    {"label": "Quick Sanity Check", "command": "python verify.py --model {model_path}"}
]

results = []

def load_config():
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    else:
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

def save_log(entry):
    timestamp = datetime.datetime.now().isoformat()
    with open(HISTORY_LOG, "a") as f:
        f.write(f"[{timestamp}] {entry}\n")

def save_report():
    with open(REPORT_FILE, "w") as f:
        f.write("# IDS Verification Summary\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().isoformat()}\n\n")
        for result in results:
            f.write(f"## {result['label']}\n")
            f.write(f"- Status: {'[Success]' if result['success'] else '[Failed]'}\n")
            f.write(f"- Timestamp: {result['timestamp']}\n")
            if not result['success']:
                f.write(f"- Error: `{result['error']}`\n")
            f.write("\n")
    print(Fore.GREEN + f"[success] Report saved to {REPORT_FILE}")

def banner():
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print("      IDS | VERIFICATION & TESTING SUITE".center(60))
    print("=" * 60 + Style.RESET_ALL)

def print_menu():
    print(Fore.YELLOW + "\nAvailable Options:")
    print("1. Run Input Validation        (check.py)")
    print("2. Run Output Validation       (check2.py)")
    print("3. Run Inference + Evaluation  (test_onnx_model.py)")
    print("4. Run Quick Sanity Check      (verify.py)")
    print("5. Run All Tests Sequentially  (1, 2, 3, 4)")
    print("6. Run All Tests in Parallel   (1 ∥ 4)")
    print("7. View Config / Edit Paths")
    print("8. Exit")

async def run_step_async(label, command):
    print(Fore.MAGENTA + f"\n=== {label} ===")
    timestamp = datetime.datetime.now().isoformat()
    try:
        process = await asyncio.create_subprocess_shell(command)
        await process.communicate()
        success = process.returncode == 0
        if success:
            print(Fore.GREEN + f"\n[success] {label} completed successfully")
        else:
            print(Fore.RED + f"\n[error] {label} failed (exit code {process.returncode})")
        results.append({"label": label, "timestamp": timestamp, "success": success, "error": ""})
        save_log(f"{label}: {'Success' if success else 'Failed'}")
    except Exception as e:
        print(Fore.RED + f"\n[error] {label} crashed: {str(e)}")
        traceback.print_exc()
        results.append({"label": label, "timestamp": timestamp, "success": False, "error": str(e)})
        save_log(f"{label}: Failed with error")

def run_step_sync(label, command):
    print(Fore.MAGENTA + f"\n=== {label} ===")
    timestamp = datetime.datetime.now().isoformat()
    try:
        subprocess.run(command, shell=True, check=True)
        print(Fore.GREEN + f"\n[success] {label} completed successfully")
        results.append({"label": label, "timestamp": timestamp, "success": True, "error": ""})
        save_log(f"{label}: Success")
    except subprocess.CalledProcessError as e:
        print(Fore.RED + f"\n[error] {label} failed")
        traceback.print_exc()
        results.append({"label": label, "timestamp": timestamp, "success": False, "error": str(e)})
        save_log(f"{label}: Failed")

def edit_config():
    config = load_config()
    print(Fore.CYAN + "\n[EDIT CONFIG]")
    for key in config:
        val = input(f"{key} [{config[key]}]: ").strip()
        if val:
            config[key] = val
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(Fore.GREEN + "[success] Config updated.")

def substitute_command(cmd, config):
    return cmd.format(**config)

def main():
    config = load_config()
    banner()
    while True:
        print_menu()
        choice = input(Fore.WHITE + "\nSelect an option (1 – 8): ").strip()

        if choice in ["1", "2", "3", "4"]:
            step = STEPS[int(choice)-1]
            cmd = substitute_command(step["command"], config)
            run_step_sync(step["label"], cmd)
        elif choice == "5":
            for step in STEPS:
                cmd = substitute_command(step["command"], config)
                run_step_sync(step["label"], cmd)
            save_report()
        elif choice == "6":
            async def run_all_parallel():
                await asyncio.gather(*[
                    run_step_async(step["label"], substitute_command(step["command"], config))
                    for step in STEPS
                ])
                save_report()
            asyncio.run(run_all_parallel())
        elif choice == "7":
            edit_config()
            config = load_config()
        elif choice == "8":
            print(Fore.CYAN + "\nExiting. Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid selection. Choose 1 – 8.")

if __name__ == "__main__":
    main()
