import os
import sys
import subprocess
import time
import json

# Helper function for background process handling
def start_process(command, desc):
    """ Starts a subprocess in the background and detaches it. """
    try:
        print(f"[info] Starting {desc}...")
#       subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(command)
        print(f"[success] {desc} started successfully.")
    except Exception as e:
        print(f"[error] Error starting {desc}: {e}")

def start_backend():
    """ Starts the IDS backend server in the background """
    start_process(["python3", "backend/api.py"], "IDS Backend Server")

def start_dashboard():
    """ Starts the IDS dashboard server in the background """
    start_process(["python3", "dashboard/app.py"], "IDS Dashboard")

def start_websocket():
    """ Starts the WebSocket server for real-time logs """
    start_process(["python3", "backend/ws_server.py"], "WebSocket Server")    

def run_attack_simulation():
    """ Allows users to select and run attack simulations """
    try:
        from backend.attack_simulation import main as attack_simulation
        attack_simulation()
    except ImportError:
        print("[error] Attack simulation module not found.")

def view_logs():
    """ View the latest IDS logs (Supports JSON and plain text) """
    log_file = "logs/ids_logs.json"
    if not os.path.exists(log_file):
        print("[error] No logs found.")
        return
    
    try:
        with open(log_file, "r") as file:
            logs = file.read().strip()

            # Try JSON format first
            try:
                logs_json = json.loads(logs)
                for log in logs_json:
                    print(f"{log.get('timestamp', 'Unknown Time')} - {log.get('event', 'Unknown Event')} (Source: {log.get('source', 'Unknown')})")
            except json.JSONDecodeError:
                # Print as raw text if JSON parsing fails
                print(logs)
    except Exception as e:
        print(f"[error] Error reading logs: {e}")

def run_performance_test():
    """ Runs the IDS performance grading and generates reports """
    try:
        from backend.performance_grader import main as performance_grader
        performance_grader()
    except ImportError:
        print("[error] Performance grading module not found.")

def main():
    while True:
        print("""
    ==========================
    IDS Command-Line Console
    ==========================
    [1] Start IDS Backend
    [2] Start IDS Dashboard
    [3] Start WebSocket Server
    [4] Run Attack Simulation
    [5] View Logs
    [6] Run Performance Test
    [7] Exit
    """)

        try:
            choice = int(input("Select an option: ").strip())
            if choice == 1:
                start_backend()
            elif choice == 2:
                start_dashboard()
            elif choice == 3:
                start_websocket()
            elif choice == 4:
                run_attack_simulation()
            elif choice == 5:
                view_logs()
            elif choice == 6:
                run_performance_test()
            elif choice == 7:
                print("Exiting IDS console...")
                break
            else:
                print("[error] Invalid option. Try again.")
        except ValueError:
            print("[error] Please enter a valid number.")

if __name__ == "__main__":
    main()
