from scapy.all import sniff, IP, TCP, UDP, Raw
import joblib
import numpy as np
import pandas as pd
import socket
import struct
import os
import json
import logging
import re
from datetime import datetime, timedelta
from collections import defaultdict
from flask_socketio import SocketIO
from flask import Flask
import onnxruntime as ort
from backend.database import log_attack

# Flask App for WebSockets
app = Flask(__name__)
socketio = SocketIO(app)

# Load trained ML model and preprocessors
MODEL_PATH = "models/ids_model.onnx"
THRESHOLD_PATH = "models/anomaly_threshold.pkl"
ENCODER_PATH = "models/one_hot_encoder.pkl"
SCALER_PATH = "models/second_scaler.pkl"

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Load other necessary models
anomaly_threshold = joblib.load(THRESHOLD_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# Logging Setup
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "alerts.log")
LOG_JSON_FILE = os.path.join(LOG_DIR, "attack_logs.json")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Port scan detection parameters
scan_activity = defaultdict(list)
SCAN_THRESHOLD = 10
SCAN_WINDOW = timedelta(seconds=5)

# Malicious pattern detection (Deep Packet Inspection)
MALICIOUS_PATTERNS = [
    rb'GET /etc/passwd',
    rb'SQL SELECT .* FROM',
    rb'admin.*password',
]

def ip_to_numeric(ip):
    """ Convert IP to a numeric format (handles both IPv4 and IPv6). """
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return int.from_bytes(socket.inet_pton(socket.AF_INET6, ip), "big")

def extract_features(packet):
    """ Extracts network features from a packet. """
    if not IP in packet:
        return None  # Skip non-IP packets

    ip_src = ip_dst = src_port = dst_port = tcp_flags = ttl = proto = packet_length = 0

    if IP in packet:
        ip_src = ip_to_numeric(packet[IP].src)
        ip_dst = ip_to_numeric(packet[IP].dst)
        proto = packet[IP].proto
        ttl = packet[IP].ttl

    if TCP in packet:
        tcp_flags = int(packet[TCP].flags)
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    packet_length = len(packet)

    features_dict = {
        "ip_src": ip_src, "ip_dst": ip_dst, "src_port": src_port, "dst_port": dst_port,
        "tcp_flags": tcp_flags, "packet_length": packet_length, "ttl": ttl, "proto": proto
    }

    features_df = pd.DataFrame([features_dict])
    features_df = features_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    features_df[scaler.feature_names_in_] = scaler.transform(features_df)
    return features_df

def deep_packet_inspection(packet):
    """ Checks packet payload for known attack patterns. """
    if packet.haslayer(Raw):
        payload = packet[Raw].load
        for pattern in MALICIOUS_PATTERNS:
            if re.search(pattern, payload):
                return True, "Potential Exploit Detected"
    return False, None

def detect_scan(src_ip, dst_port):
    """ Detects potential port scans by tracking connection attempts. """
    now = datetime.now()
    scan_activity[src_ip].append((now, dst_port))
    scan_activity[src_ip] = [(t, p) for t, p in scan_activity[src_ip] if now - t <= SCAN_WINDOW]

    if len(set(p for _, p in scan_activity[src_ip])) > SCAN_THRESHOLD:
        scan_log_msg = f"[warning] Port scan detected from {src_ip}! Scanned ports: {set(p for _, p in scan_activity[src_ip])}"
        logging.warning(scan_log_msg)
        print(scan_log_msg)
        socketio.emit("new_alert", {"source_ip": src_ip, "attack_type": "Port Scan"})
        save_log_to_json({"source_ip": src_ip, "attack_type": "Port Scan"})
        return True
    return False

def save_log_to_json(data):
    """ Saves detected attacks to JSON logs. """
    try:
        logs = []
        if os.path.exists(LOG_JSON_FILE) and os.path.getsize(LOG_JSON_FILE) > 0:
            with open(LOG_JSON_FILE, "r") as file:
                logs = json.load(file)

        logs.append(data)

        with open(LOG_JSON_FILE, "w") as file:
            json.dump(logs, file, indent=4)

        print(f"[success] Attack logged: {data}")

    except (json.JSONDecodeError, IOError) as e:
        print(f"[error] JSON Logging Error: {e}")

# Avoid duplicate alerts
recent_alerts = {}

def packet_callback(packet):
    try:
        features_df = extract_features(packet)
        if features_df is None:
            return  # Skip non-IP packets

        input_data = features_df.to_numpy().astype(np.float32)
        predictions = session.run(None, {input_name: input_data})[0]

        # Extract probabilities
        probability_normal, probability_attack = predictions[0]
        predicted_class = 1 if probability_attack > 0.7 else 0  # Adjust threshold

        # Get source & destination IPs
        src_ip = packet[IP].src if IP in packet else "N/A"
        dst_ip = packet[IP].dst if IP in packet else "N/A"
        protocol = packet[IP].proto if IP in packet else "N/A"
        attack_type = "Normal"

        # Prevent duplicate alerts
        now = datetime.now()
        alert_cooldown = 10 if attack_type == "Port Scan" else 5
        if src_ip in recent_alerts and (now - recent_alerts[src_ip]).total_seconds() < alert_cooldown:
            return  
        recent_alerts[src_ip] = now  

        # Check if packet is suspicious
        suspicious, dpi_attack = deep_packet_inspection(packet)

        # Detect anomalies or known attacks
        if predicted_class == 1 or suspicious:
            attack_type = dpi_attack if suspicious else "Anomaly Detected"
            log_attack(src_ip, dst_ip, protocol, attack_type)
            save_log_to_json({"source_ip": src_ip, "destination_ip": dst_ip, "protocol": protocol, "attack_type": attack_type})
            socketio.emit("new_alert", {"source_ip": src_ip, "destination_ip": dst_ip, "protocol": protocol, "attack_type": attack_type})
            alert_msg = f"[ALERT] {attack_type} from {src_ip} to {dst_ip}"
            logging.warning(alert_msg)
            print(alert_msg)

    except Exception as e:
        print(f"Error: {e}")

def start_sniffing():
    sniff(prn=packet_callback, store=0)

if __name__ == "__main__":
    start_sniffing()
