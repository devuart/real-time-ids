from flask import Flask, render_template, jsonify
import requests
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSocket support
CORS(app)  # Allow cross-origin requests

# API & WebSocket URLs (Docker service names)
API_URL = "http://ids_backend:5001/logs"  # Logs API
WEBSOCKET_URL = "ws://ids_websocket:5003"  # WebSocket connection

def get_attack_logs():
    """ Fetch attack logs from IDS API """
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching attack logs: {e}")
        return []  # Return empty list if API fails

@app.route("/")
def index():
    logs = get_attack_logs()
    return render_template("index.html", logs=logs, websocket_url=WEBSOCKET_URL)

@socketio.on("connect")
def handle_connect():
    """ Notify client on WebSocket connection """
    print("Client connected")

@socketio.on("fetch_logs")
def send_logs():
    """ Fetch & emit logs dynamically """
    logs = get_attack_logs()
    socketio.emit("logs_update", logs)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5002, debug=True)
