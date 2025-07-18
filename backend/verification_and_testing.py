import subprocess
import traceback
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

MODEL_PATH = "models/ids_model.onnx"

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
    print("5. Run All Tests Sequentially")
    print("6. Exit")

def run_step(label, command):
    print(Fore.MAGENTA + f"\n=== {label} ===")
    try:
        subprocess.run(command, shell=True, check=True)
        print(Fore.GREEN + f"\n[success] {label} completed successfully")
    except subprocess.CalledProcessError as e:
        print(Fore.RED + f"\n[error] {label} failed")
        traceback.print_exc()

def main():
    banner()
    while True:
        print_menu()
        choice = input(Fore.WHITE + "\nSelect an option (1-6): ").strip()

        if choice == "1":
            run_step("Input Validation", "python check.py")
        elif choice == "2":
            run_step("Output Validation", f"python check2.py --model {MODEL_PATH}")
        elif choice == "3":
            run_step("Full Inference Test", f"python test_onnx_model.py --model {MODEL_PATH} --validate --benchmark")
        elif choice == "4":
            run_step("Quick Sanity Check", f"python verify.py --model {MODEL_PATH}")
        elif choice == "5":
            run_step("Input Validation", "python check.py")
            run_step("Output Validation", f"python check2.py --model {MODEL_PATH}")
            run_step("Full Inference Test", f"python test_onnx_model.py --model {MODEL_PATH} --validate --benchmark")
            run_step("Quick Sanity Check", f"python verify.py --model {MODEL_PATH}")
        elif choice == "6":
            print(Fore.CYAN + "\nExiting. Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid selection. Please choose 1–6.")

if __name__ == "__main__":
    main()
