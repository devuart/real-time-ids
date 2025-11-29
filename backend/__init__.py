"""
Backend Initialization Module
Initializes Flask app, SocketIO, database, and API endpoints.
This is the main entry point for the IDS backend system.
"""

from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from config import Config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO if not Config.DEBUG_MODE else logging.DEBUG,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("IDS-Backend")

# ================================
# Flask App Initialization
# ================================

logger.info("Initializing Flask application...")

# Create Flask app instance
app = Flask(__name__)
app.config.from_object(Config)

# Configure Flask-specific settings
app.config["JWT_SECRET_KEY"] = Config.JWT_SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = Config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = Config.SQLALCHEMY_TRACK_MODIFICATIONS

# Enable CORS for frontend communication
CORS(
    app,
    origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

logger.info(f"Flask app configured with database: {Config.SQLALCHEMY_DATABASE_URI}")

# ================================
# SocketIO Initialization
# ================================

logger.info("Initializing SocketIO...")

# Create single SocketIO instance for the entire application
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",  # Use threading for better compatibility
    logger=Config.DEBUG_MODE,  # Enable SocketIO logging in debug mode
    engineio_logger=Config.DEBUG_MODE,
    ping_timeout=60,
    ping_interval=25
)

logger.info(f"SocketIO initialized on port {Config.WEBSOCKET_PORT}")

# ================================
# Database Initialization
# ================================

logger.info("Setting up database...")

# Import database module (lazy import to avoid circular dependencies)
from backend import database

# Create database tables within Flask application context
with app.app_context():
    try:
        database.Base.metadata.create_all(database.engine)
        logger.info("Database tables created successfully")
        
        # Test database connection
        if database.test_database_connection():
            logger.info("Database connection verified")
        else:
            logger.warning("Database connection test failed")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# ================================
# API Initialization
# ================================

logger.info("Initializing REST API endpoints...")

# Import and initialize API (lazy import to avoid circular dependencies)
from backend.api import init_api

try:
    api = init_api(app)
    logger.info("REST API endpoints registered successfully")
except Exception as e:
    logger.error(f"API initialization error: {e}")

# ================================
# WebSocket Routes Registration
# ================================

logger.info("Registering WebSocket event handlers...")

# Import WebSocket handler (lazy import)
from backend import websocket_handler

# Register WebSocket events
@socketio.on("connect")
def handle_connect():
    """Handle client WebSocket connection."""
    from flask import request
    logger.info(f"WebSocket client connected: {request.sid}")
    socketio.emit("status", {
        "message": "Connected to IDS WebSocket",
        "timestamp": database.datetime.datetime.now().isoformat()
    })

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client WebSocket disconnection."""
    from flask import request
    logger.info(f"WebSocket client disconnected: {request.sid}")

@socketio.on("fetch_logs")
def handle_fetch_logs(data=None):
    """
    Fetch recent attack logs and send to client.
    
    Args:
        data: Optional dictionary with 'limit' parameter
    """
    try:
        limit = data.get("limit", 10) if data else 10
        logs = database.get_recent_attacks(limit=limit)
        
        socketio.emit("logs_update", {
            "logs": [log.to_dict() for log in logs],
            "timestamp": database.datetime.datetime.now().isoformat()
        })
        
        logger.debug(f"Sent {len(logs)} attack logs to client")
        
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        socketio.emit("error", {
            "message": "Failed to fetch logs",
            "error": str(e)
        })

@socketio.on("fetch_statistics")
def handle_fetch_statistics():
    """Fetch and send attack statistics to client."""
    try:
        stats = database.get_attack_statistics()
        
        socketio.emit("statistics_update", {
            "statistics": stats,
            "timestamp": database.datetime.datetime.now().isoformat()
        })
        
        logger.debug("Sent attack statistics to client")
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        socketio.emit("error", {
            "message": "Failed to fetch statistics",
            "error": str(e)
        })

@socketio.on("broadcast_test")
def handle_broadcast_test(data):
    """Test WebSocket broadcasting."""
    message = data.get("message", "Test broadcast") if data else "Test broadcast"
    
    websocket_handler.broadcast_message({
        "type": "test",
        "message": message,
        "timestamp": database.datetime.datetime.now().isoformat()
    })
    
    logger.debug(f"Broadcast test message: {message}")

logger.info("WebSocket event handlers registered successfully")

# ================================
# Health Check Endpoint
# ================================

@app.route("/")
def index():
    """Root endpoint - API health check."""
    return {
        "status": "online",
        "service": "Real-Time IDS Backend",
        "version": "1.0.0",
        "endpoints": {
            "api": "/login, /logs, /start-ids, /stop-ids, /simulate-attack, /grade-ids, /generate-report, /health",
            "websocket": f"ws://{Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}"
        }
    }, 200

# ================================
# Export Objects
# ================================

# Export app and socketio for use in other modules
__all__ = ['app', 'socketio', 'database', 'api']

logger.info("Backend initialization complete")
logger.info(f"Debug mode: {Config.DEBUG_MODE}")
logger.info(f"WebSocket host: {Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}")

# ================================
# Standalone Execution
# ================================

if __name__ == "__main__":
    """
    Run the Flask app with SocketIO in standalone mode.
    For production, use a WSGI server like gunicorn with eventlet/gevent.
    """
    logger.info("=" * 70)
    logger.info("Starting IDS Backend Server in STANDALONE mode")
    logger.info("=" * 70)
    logger.info(f"REST API: http://{Config.WEBSOCKET_HOST}:5001")
    logger.info(f"WebSocket: ws://{Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}")
    logger.info("=" * 70)
    
    try:
        # Run with SocketIO (handles both HTTP and WebSocket)
        socketio.run(
            app,
            host=Config.WEBSOCKET_HOST,
            port=Config.WEBSOCKET_PORT,
            debug=Config.DEBUG_MODE,
            use_reloader=Config.DEBUG_MODE,
            log_output=Config.DEBUG_MODE
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server stopped")