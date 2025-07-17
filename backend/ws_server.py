from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS
import logging
#from database import SessionLocal, AttackLog
from backend.database import SessionLocal, AttackLog
import faulthandler
faulthandler.enable()

# Initialize Flask app and WebSocket server
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("websocket")

# Track active clients
active_clients = set()

@socketio.on("connect")
def handle_connect():
    """Handles WebSocket connections."""
    active_clients.add(request.sid)
    logger.info(f"Client {request.sid} connected.")
    socketio.emit("status", {"message": "Connected to WebSocket"})

@socketio.on("disconnect")
def handle_disconnect():
    """Handles WebSocket disconnections."""
    active_clients.discard(request.sid)
    logger.info(f"Client {request.sid} disconnected.")

@socketio.on("fetch_logs")
def send_logs():
    """Emits real-time logs to connected clients."""
    try:
        logs = fetch_latest_logs()
        socketio.emit("logs_update", logs)
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")

@socketio.on("broadcast_message")
def broadcast_message(data):
    """Broadcasts messages to all connected clients."""
    message = data.get("message", "")
    socketio.emit("new_alert", {"message": message})
    logger.info(f"Broadcasted message: {message}")

def fetch_latest_logs():
    """Fetches the latest attack logs from the database."""
    session = SessionLocal()
    try:
        logs = session.query(AttackLog).order_by(AttackLog.timestamp.desc()).limit(10).all()
        return [{
            "timestamp": log.timestamp.isoformat(),
            "event": log.attack_type,
            "source": log.source_ip
        } for log in logs]
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []
    finally:
        session.close()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5003, debug=True)
