import os
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")

if not os.path.exists(env_path):
    print("[info] Warning: .env file not found! Using default settings.")

load_dotenv(env_path)


def str_to_bool(value):
    """Convert string values to boolean."""
    return str(value).lower() in ["true", "1", "yes"]


class Config:
    """Configuration settings for the IDS system."""

    # Ensure necessary directories exist
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # Database Configuration (Use absolute path)
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'logs', 'ids_logs.db')}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # WebSocket Configuration
    WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", "5003"))
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.getenv("SOCKETIO_CORS_ALLOWED_ORIGINS", "*")

    # Logging Configuration
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", os.path.join(BASE_DIR, "logs", "alerts.log"))
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "5000000"))  # Convert string to integer
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))  # Convert string to integer

    # Machine Learning Model Paths
    AUTOENCODER_MODEL_PATH = os.getenv("AUTOENCODER_MODEL_PATH", os.path.join(BASE_DIR, "models", "autoencoder.h5"))
    IDS_MODEL_PATH = os.getenv("IDS_MODEL_PATH", os.path.join(BASE_DIR, "models", "ids_model.pkl"))

    # Flask API Security
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    if not JWT_SECRET_KEY:
        print("[info] Warning: JWT_SECRET_KEY is not set! Generating a temporary key.")
        JWT_SECRET_KEY = os.urandom(24).hex()  # Temporary key (only for dev)

    # IDS Packet Sniffer Settings
    PACKET_SNIFF_INTERFACE = os.getenv("PACKET_SNIFF_INTERFACE", "eth0")
    PACKET_SNIFF_FILTER = os.getenv("PACKET_SNIFF_FILTER", "tcp or udp")

    # Attack Detection Thresholds
    ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))

    # Debug Mode (Proper Boolean Parsing)
    DEBUG_MODE = str_to_bool(os.getenv("DEBUG_MODE", "False"))

    # Print configurations in debug mode
    if DEBUG_MODE:
        print("[info] Running in DEBUG mode")
        print(f"[info] Database: {SQLALCHEMY_DATABASE_URI}")
        print(f"[info] WebSocket: {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
