# backend/__init__.py
from flask import Flask
from flask_socketio import SocketIO
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize WebSocket (only needed when running WebSocket server)
socketio = SocketIO(app, cors_allowed_origins="*")

# Import modules AFTER app initialization to avoid circular imports
from backend import api, database, websocket_handler

# Ensure tables are created using database.py's engine
with app.app_context():
    database.Base.metadata.create_all(database.engine)
