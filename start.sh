#!/bin/bash

echo "[info] Starting Real-Time IDS..."

# Load environment variables from .env file
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        # Ignore empty lines and comment lines
        if [[ ! "$key" =~ ^# && -n "$key" ]]; then
            # Strip possible inline comments
            value=$(echo "$value" | cut -d '#' -f1 | xargs)
            export "$key=$value"
        fi
    done < .env
    echo "[success] Environment variables loaded."
else
    echo "[warning] Warning: .env file not found. Using default settings."
fi

# Activate virtual environment (if available)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[success] Virtual environment activated."
else
    echo "[warning] No virtual environment found. Running without it."
fi

# Start the backend services
echo "[info] Starting API and WebSocket Server..."
sudo python3 backend/api.py & echo $! >> ids_pids.txt & 
sudo python3 backend/websocket.py & echo $! >> ids_pids.txt &

# Start the packet sniffer (requires sudo for raw packet capture)
echo "[info] Starting Packet Sniffer..."
sudo python3 backend/ids_sniffer.py & echo $! >> ids_pids.txt &

# Start the dashboard
echo "[info] Starting Dashboard..."
sudo python3 dashboard/app.py & echo $! >> ids_pids.txt &

# Wait for processes to keep running
wait
