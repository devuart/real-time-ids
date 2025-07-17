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
from database import log_attack

# Flask App for WebSockets
app = Flask(__name__)
socketio = SocketIO(app)

# Load trained ML model and preprocessors
MODEL_PATH = "models/ids_model.onnx"
THRESHOLD_PATH = "models/anomaly_threshold.pkl"
IDS_MODEL_PATH = "models/ids_model.pkl"
ENCODER_PATH = "models/one_hot_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Load other necessary models
anomaly_threshold = joblib.load(THRESHOLD_PATH)
ids_model, feature_names = joblib.load(IDS_MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "ids_log.txt")
scan_log_file = os.path.join(log_dir, "scan_log.txt")
LOG_FILE_JSON = "logs/attack_logs.json"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
scan_logger = logging.getLogger("scan_logger")
scan_logger.setLevel(logging.INFO)
scan_handler = logging.FileHandler(scan_log_file)
scan_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
scan_logger.addHandler(scan_handler)

# Port scan detection parameters
scan_activity = defaultdict(list)
SCAN_THRESHOLD = 10
SCAN_WINDOW = timedelta(seconds=5)

# Malicious pattern detection (DPI)
MALICIOUS_PATTERNS = [
    rb'GET /etc/passwd',
    rb'SQL SELECT .* FROM',
    rb'admin.*password',
]

def ip_to_numeric(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def extract_features(packet):
    """ Extracts network features from a packet. """
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
    features_df = features_df.reindex(columns=feature_names, fill_value=0)
    features_df[feature_names] = scaler.transform(features_df)
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
        scan_logger.info(scan_log_msg)
        socketio.emit("new_alert", {"source_ip": src_ip, "attack_type": "Port Scan"})
        return True
    return False

def save_log_to_json(data):
    logs = []
    if os.path.exists(LOG_FILE_JSON):
        with open(LOG_FILE_JSON, "r") as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
    logs.append(data)
    with open(LOG_FILE_JSON, "w") as file:
        json.dump(logs, file, indent=4)

def packet_callback(packet):
    try:
        features_df = extract_features(packet)
        
        # Run ONNX model inference
        input_data = features_df.to_numpy().astype(np.float32)
        predictions = session.run(None, {input_name: input_data})[0]
        anomaly_score = np.mean(np.power(input_data - predictions, 2))

        src_ip = packet[IP].src if IP in packet else "N/A"
        dst_ip = packet[IP].dst if IP in packet else "N/A"
        attack_type = "Normal"

        if TCP in packet and detect_scan(src_ip, packet[TCP].dport):
            return
        
        suspicious, dpi_attack = deep_packet_inspection(packet)
        if anomaly_score > anomaly_threshold or suspicious or ids_model.predict(features_df)[0] == 1:
            attack_type = dpi_attack if suspicious else "Anomaly Detected"
            log_attack(src_ip, dst_ip, packet.proto, attack_type)
            save_log_to_json({"source_ip": src_ip, "destination_ip": dst_ip, "protocol": packet.proto, "attack_type": attack_type})
            socketio.emit("new_alert", {"source_ip": src_ip, "destination_ip": dst_ip, "protocol": packet.proto, "attack_type": attack_type})
            print(f"[ALERT] {attack_type} from {src_ip} to {dst_ip}")
    except Exception as e:
        print(f"Error: {e}")

def start_sniffing():
    sniff(prn=packet_callback, store=0)

if __name__ == "__main__":
    start_sniffing()
