#!/bin/bash

echo "[warning] Stopping Real-Time IDS..."

if [ -f ids_pids.txt ]; then
    while read pid; do
        sudo kill "$pid"
    done < ids_pids.txt
    rm ids_pids.txt
    echo "[success] Stopped successfully."
else
    echo "[warning] No PID file found. Stopping by process search..."
    pids=$(pgrep -f "python3 backend/api.py|python3 backend/websocket.py|python3 backend/ids_sniffer.py|python3 dashboard/app.py")
    
    if [ -n "$pids" ]; then
        sudo kill $pids
        echo "[success] Stopped successfully."
    else
        echo "[warning] No running IDS processes found."
    fi
fi
