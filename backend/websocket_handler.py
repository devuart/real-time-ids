#import ws_server
from backend import ws_server
import websocket  # Import from websocket-client package
import threading
import time
from flask_socketio import emit
#from . import socketio  # Ensure socketio is imported from __init__.py
from backend import socketio

MAX_RETRIES = 10  # Maximum reconnection attempts
INITIAL_BACKOFF = 5  # Initial wait time before reconnecting

class WebSocketClient:
    def __init__(self, url="ws://localhost:5001/logs"):
        self.url = url
        self.ws = None
        self.reconnect_attempts = 0
        self.backoff_time = INITIAL_BACKOFF
        self.thread = None
        self.stop_event = threading.Event()

    def on_message(self, ws, message):
        print(f"\n[info] {message}")

    def on_error(self, ws, error):
        print(f"[warning] WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("\n[error] Disconnected from real-time logs.")
        if not self.stop_event.is_set():
            self.reconnect()

    def on_open(self, ws):
        print("[success] Connected to WebSocket server!")

    def connect(self):
        """Establish WebSocket connection."""
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()

    def reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self.reconnect_attempts >= MAX_RETRIES:
            print("[error] Max reconnection attempts reached. Exiting.")
            return

        self.reconnect_attempts += 1
        print(f"[info] Reconnecting... (Attempt {self.reconnect_attempts}/{MAX_RETRIES}) in {self.backoff_time}s")
        time.sleep(self.backoff_time)
        
        self.backoff_time = min(self.backoff_time * 2, 60)  # Cap backoff at 60s
        self.connect()

    def stop(self):
        """Stop the WebSocket connection gracefully."""
        self.stop_event.set()
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join()

# Added WebSocket Broadcast Function
def broadcast_message(message):
    """Broadcast a message to WebSocket clients."""
    try:
        print(f"[WebSocket] {message}")  # Debugging log
        socketio.emit("new_alert", {"message": message}, namespace="/")
    except Exception as e:
        print(f"[warning] WebSocket Broadcast Error: {e}")

if __name__ == "__main__":
    client = WebSocketClient()
    try:
        client.connect()
        while True:
            time.sleep(1)  # Keep script running
    except KeyboardInterrupt:
        print("\n[warning] Stopping WebSocket client.")
        client.stop()
