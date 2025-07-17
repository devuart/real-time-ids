import subprocess
import time
import os
import socket

WS_SERVER_PORT = 5003  # WebSocket Server Port
running_environment = None

def is_process_running(process_name):
    """Check if a process is running."""
    try:
        result = subprocess.run(["pgrep", "-f", process_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip() != ""
    except Exception:
        return False

def stop_process(process_name):
    """Stop a running process by name."""
    try:
        subprocess.run(["pkill", "-f", process_name])
        print(f"[success] Stopped {process_name}.")
    except Exception:
        print(f"[warning] Failed to stop {process_name}.")

def is_port_in_use(port):
    """Check if a given port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0

def check_services_status():
    """Check if relevant services are running."""
    ws_status = is_process_running("backend/ws_server.py")
    ws_client_status = is_process_running("backend/websocket_handler.py")

    if ws_status or ws_client_status:
        print("\n[info] Services Running:")
        if ws_status:
            print("   - WebSocket Server (ws_server.py)")
        if ws_client_status:
            print("   - WebSocket Client (websocket_handler.py)")
        return True
    else:
        print("\n[warning] No relevant services are running.")
        return False

def check_conflict_status():
    """Check if relevant services are running."""
    ws_status = is_process_running("backend/ws_server.py")
    ws_client_status = is_process_running("backend/websocket_handler.py")

    if ws_status or ws_client_status:
        print("\n[info] Services Running:")
        if ws_status:
            print("   - WebSocket Server (ws_server.py)")
        if ws_client_status:
            print("   - WebSocket Client (websocket_handler.py)")
        return True
    else:
        print("\n[warning] No conflicting services detected.")
        return False        

def stop_services():
    """Stop all relevant services."""
    print("\n[warning] Stopping Services...")
    stop_process("backend/ws_server.py")
    stop_process("backend/websocket_handler.py")
    print("[success] All relevant services stopped.")

def validate_environment():
    """Validate environment and check for conflicts."""
    global running_environment

    print("\n[info] Validating Environment...")
    
    # Check for running services
    if check_conflict_status():
        print("\n[warning] Conflicting services detected!")
        print("1. Stop Conflicting Services")
        print("2. Retry Validation")
        print("3. Return to Dashboard")
        print("4. Exit")

        choice = input("\n[info] Select an option: ")
        if choice == "1":
            stop_services()
            return validate_environment()
        elif choice == "2":
            return validate_environment()
        elif choice == "3":
            return main_dashboard()
        elif choice == "4":
            exit(0)
        else:
            print("[warning] Invalid choice. Returning to Dashboard.")
            return main_dashboard()
    
    print("\n[success] Validation successful!")
    print("\n1. Proceed to Testing")
    print("2. Redo Validation")
    print("3. Return to Dashboard")
    print("4. Exit")

    choice = input("\n[info] Select an option: ")
    if choice == "1":
        start_test()
    elif choice == "2":
        return validate_environment()
    elif choice == "3":
        return main_dashboard()
    elif choice == "4":
        exit(0)
    else:
        print("[warning] Invalid choice. Returning to Dashboard.")
        return main_dashboard()

def start_test():
    """Run both scripts and handle conflicts."""
    print("\n[info] Running IDS Test...")

    if is_port_in_use(WS_SERVER_PORT):
        print(f"[error] WebSocket Server failed to start! Port {WS_SERVER_PORT} is already in use.")
        print("\n1. Stop Conflicting Services")
        print("2. Return to Dashboard")
        print("3. Exit")

        choice = input("\n[info] Select an option: ")
        if choice == "1":
            stop_services()
            return start_test()
        elif choice == "2":
            return main_dashboard()
        elif choice == "3":
            exit(0)
        else:
            print("[warning] Invalid choice. Returning to Dashboard.")
            return main_dashboard()

    print("\n[info] Starting WebSocket Server...")
    ws_server = subprocess.Popen(["python3", "backend/ws_server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)

    if ws_server.poll() is not None:
        print("[error] WebSocket Server failed to start.")
        return handle_test_failure()

    print("[success] WebSocket Server started successfully!")

    print("\n[info] Starting WebSocket Client...")
    ws_client = subprocess.Popen(["python3", "backend/websocket_handler.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)

    if ws_client.poll() is not None:
        ws_server.terminate()
        print("[error] WebSocket Client failed to start.")
        return handle_test_failure()

    print("\n[success] Test was successful! IDS is running.")
    test_success_options()

def handle_test_failure():
    """Handle test failure and provide options."""
    print("\n[error] Test failed due to conflicts or missing components.")
    print("1. Retry Testing")
    print("2. Return to Dashboard")
    print("3. Exit")

    choice = input("\n[info] Select an option: ")
    if choice == "1":
        start_test()
    elif choice == "2":
        main_dashboard()
    elif choice == "3":
        exit(0)
    else:
        print("[warning] Invalid choice. Returning to Dashboard.")
        main_dashboard()

def test_success_options():
    """Handle post-test validation and prompt user for next steps."""
    print("\n[success] Test completed successfully!")

    # Automatically re-validate to prevent conflicts in the next run
    print("\n[info] Re-validating environment to ensure no conflicts before running another test...")
    validate_and_fix_conflicts()

    print("\n1. Run Test Again")
    print("2. Return to Dashboard")
    print("3. Exit")

    choice = input("\n[info] Select an option: ")

    if choice == "1":
        start_test()
    elif choice == "2":
        main_dashboard()
    elif choice == "3":
        exit(0)
    else:
        print("[warning] Invalid choice. Returning to Dashboard.")
        main_dashboard()

def validate_and_fix_conflicts():
    """Validate the environment and automatically resolve conflicts if found."""
    conflicts = check_services_status()
    
    if conflicts:
        print("\n[warning] Conflicting services detected! Stopping them now...")
        stop_services()
        print("[success] Conflicts resolved. Ready for the next test.")
    else:
        print("\n[success] No conflicts detected. System is ready.")


def main_dashboard():
    """Display the main dashboard and handle user choices."""
    global running_environment

    while True:
        print("\n[info] IDS Testing Dashboard")
        print("1. Run in Non-Docker Environment")
        print("2. Run in Docker Environment")
        print("3. Check Status of Services")
        print("4. Stop Services (if running)")                
        print("5. Exit")

        choice = input("\n🔷 Select an option: ")

        if choice in ["1", "2"]:
            running_environment = choice
            validate_environment()                    
        elif choice == "3":
            check_services_status()            
        elif choice == "4":
            stop_services()
        elif choice == "5":
            exit(0)
        else:
            print("[warning] Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main_dashboard()
