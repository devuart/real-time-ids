import requests
import json
import time
import websocket
import threading
import scapy.all as scapy
import os
import subprocess
import re
import ipaddress
import logging
#from scapy.all import ARP, Ether, srp, send, IP, TCP
#from scapy.all import ARP, Ether, srp, conf, getmacbyip
#from scapy.all import send, IP, TCP, ARP, Ether, srp, conf, getmacbyip

# Configurations
API_URL = "http://localhost:5001"
WEBSOCKET_URL = "ws://localhost:5003"
#LOG_FILE_PATH = "backend/logs/alerts.log"
JWT_TOKEN = None  # This will be set after login

# Logging Setup
LOG_DIR = "backend/logs"
LOG_FILE = os.path.join(LOG_DIR, "alerts.log")
LOG_JSON_FILE = os.path.join(LOG_DIR, "attack_logs.json")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def login():
    """Authenticate and retrieve JWT token."""
    global JWT_TOKEN
    print("\n[info] Logging in...")
    response = requests.post(f"{API_URL}/login", json={"username": "admin", "password": "password123"})
    
    if response.status_code == 200:
        JWT_TOKEN = response.json().get("access_token")
        print("[success] Login successful!")
    else:
        print(f"[error] Login failed: {response.text}")
        exit(1)

def get_network_range():
    """Extracts available network ranges from active interfaces and allows user selection."""
    print("\n[info] Detecting available network interfaces...\n")

    try:
        # Run ifconfig or ip a based on availability
        result = subprocess.run(["ip", "-o", "-f", "inet", "addr"], capture_output=True, text=True)
        if not result.stdout.strip():
            result = subprocess.run(["ifconfig"], capture_output=True, text=True)

        # Parse IP addresses and subnet masks
        interfaces = []
        for line in result.stdout.splitlines():
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)/(\d+)', line)  # Match CIDR notation (e.g., 192.168.1.10/24)
            if match:
                ip = match.group(1)
                cidr = match.group(2)
                network = f"{ip}/{cidr}"

                # Convert to network range
                net = ipaddress.IPv4Network(network, strict=False)
                interfaces.append(str(net))

        if not interfaces:
            print("[error] No valid network interfaces found!")
            return None

        print("[success] Available Networks:")
        for i, net in enumerate(interfaces, 1):
            print(f"[{i}] {net}")

        choice = input("\n[config] Select a network range for scanning (Enter number): ").strip()
        return interfaces[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(interfaces) else None

    except Exception as e:
        print(f"[error] Error detecting network range: {e}")
        return None

def get_target_ip():
    """Scans the selected network for available hosts and lets the user pick a target."""
    network_range = get_network_range()
    
    if not network_range:
        return None

    print(f"\n[info] Scanning network {network_range} for active hosts...")

    try:
        # Run Nmap to detect live hosts
        result = subprocess.run(["sudo", "nmap", "-sn", network_range], capture_output=True, text=True)

        # Extract only live hosts (ignoring network/broadcast addresses)
        ips = re.findall(r"Nmap scan report for ([\d\.]+)\nHost is up", result.stdout)

        if not ips:
            print("[error] No active hosts found!")
            return None

        print("\n[success] Available Targets:")
        for i, ip in enumerate(ips, 1):
            print(f"[{i}] {ip}")

        choice = input("\n[config] Select a target IP (Enter number): ").strip()
        return ips[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(ips) else None

    except Exception as e:
        print(f"[error] Error scanning network: {e}")
        return None


def test_api_endpoints(target_ip):
    """Test key API endpoints."""
    headers = {"Authorization": f"Bearer {JWT_TOKEN}"}

    print("\n[info] Starting IDS...")
    response = requests.get(f"{API_URL}/start-ids", headers=headers)
    print(f"[info] Start IDS Response: {response.json()}")

    print(f"\n[info] Simulating attack on {target_ip} (port scan)...")
    response = requests.post(f"{API_URL}/simulate-attack", 
                             json={"attack_type": "port_scan", "intensity": 5, "target_ip": target_ip}, 
                             headers=headers)
    print(f"[config] Attack Simulation Response: {response.json()}")

    print("\n[info] Grading IDS performance...")
    response = requests.get(f"{API_URL}/grade-ids", headers=headers)
    print(f"[info] IDS Performance Score: {response.json()}")

def on_message(ws, message):
    """Handle incoming WebSocket messages."""
    print(f"[warning] WebSocket Alert: {message}")

def test_websocket():
    """Test WebSocket communication."""
    print("\n[info] Connecting to WebSocket...")
    ws = websocket.WebSocketApp(WEBSOCKET_URL, on_message=on_message)
    thread = threading.Thread(target=ws.run_forever, daemon=True)
    thread.start()
    
    # Wait a few seconds to capture alerts
    time.sleep(5)
    ws.close()
    print("[success] WebSocket test completed.")

from scapy.all import ARP, Ether, srp, conf, getmacbyip

def get_mac_address(target_ip):
    """Get the MAC address of a target IP, falling back to the default gateway if needed."""
    print(f"\n[info] Resolving MAC address for {target_ip}...")

    # First, try to resolve via Scapy's `getmacbyip`
    mac_address = getmacbyip(target_ip)
    if mac_address:
        print(f"[success] Target MAC Address: {mac_address}")
        return mac_address

    # If that fails, use ARP to manually request the MAC address
    print(f"[warning] Warning: Target unreachable. Attempting ARP request...")

    arp_request = ARP(pdst=target_ip)
    broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = broadcast / arp_request
    answered, _ = srp(packet, timeout=3, verbose=False)

    if answered:
        mac_address = answered[0][1].hwsrc
        print(f"[success] Target MAC Address (ARP Resolved): {mac_address}")
        return mac_address

    # If still not found, fallback to default gateway
    gateway_ip = conf.route.route("0.0.0.0")[2]
    gateway_mac = getmacbyip(gateway_ip)

    if gateway_mac:
        print(f"[warning] Warning: Using Gateway {gateway_ip} MAC: {gateway_mac}")
        return gateway_mac

    print("[error] ERROR: Could not resolve MAC address.")
    return None

from scapy.all import sendp, IP, TCP

def simulate_network_traffic(target_ip):
    """Send crafted packets to simulate attacks."""
    print(f"\n[info] Sending simulated attack packets to {target_ip}...")

    mac_address = get_mac_address(target_ip)

    if not mac_address:
        print(f"[error] ERROR: Could not resolve MAC address for {target_ip}. Aborting packet transmission.")
        return

    print(f"[success] Target MAC Address: {mac_address}")

    # Select the correct interface (e.g., eth0)
    iface = conf.iface

    # Simulating a SYN flood attack
    for _ in range(5):
        sendp(Ether(dst=mac_address) / IP(dst=target_ip) / TCP(dport=80, flags="S"), iface=iface, verbose=False)

    print("[success] Attack packets sent.")

def check_logs():
    """Verify if attack logs were recorded."""
    print("\n[info] Checking logs for recorded attacks...")
    
    if not os.path.exists(LOG_JSON_FILE):
        print("[error] Log file not found!")
        return
    
    with open(LOG_JSON_FILE, "r") as file:
        logs = file.readlines()[-5:]  # Read last 5 logs
        for log in logs:
            print(f"[info] {log.strip()}")

    print("[success] Log verification complete.")

def main():
    """Run all IDS tests sequentially."""
    print("[info] Running IDS tests...")

    login()

    # Select target IP
    target_ip = get_target_ip()
    if not target_ip:
        print("[error] No valid target selected. Exiting...")
        return

    test_api_endpoints(target_ip)
    test_websocket()
    simulate_network_traffic(target_ip)
    check_logs()

    print("\n[success] **All tests completed successfully!**")

if __name__ == "__main__":
    main()
