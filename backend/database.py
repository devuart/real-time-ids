from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import logging
import os
from logging.handlers import RotatingFileHandler

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Database Configuration
DATABASE_URL = "sqlite:///logs/ids_logs.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Define Attack Log Table
class AttackLog(Base):
    __tablename__ = "attack_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    source_ip = Column(String, nullable=False)
    destination_ip = Column(String, nullable=False)
    protocol = Column(String, nullable=False)
    attack_type = Column(String, nullable=False)

# Create the attack_logs table
Base.metadata.create_all(engine)

# Rotating Log Handler for alerts.log
log_handler = RotatingFileHandler("logs/alerts.log", maxBytes=5_000_000, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
log_handler.setFormatter(formatter)

logger = logging.getLogger("IDSLogger")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Function to log an attack in the database
def log_attack(source_ip, destination_ip, protocol, attack_type):
    """ Logs attack details into the database and alerts WebSocket clients. """
    try:
        from backend import websocket_handler  # Local import to avoid circular dependency

        with SessionLocal() as session:
            attack = AttackLog(
                source_ip=source_ip,
                destination_ip=destination_ip,
                protocol=protocol,
                attack_type=attack_type
            )
            session.add(attack)
            session.commit()

        # Log attack to rotating log file
        logger.info(f"Attack from {source_ip} to {destination_ip} | Protocol: {protocol} | Type: {attack_type}")

        # Emit WebSocket event
        websocket_handler.broadcast_message({
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "protocol": protocol,
            "attack_type": attack_type
        })

    except Exception as db_error:
        logger.error(f"[ERROR] Failed to log attack: {db_error}")


# Example Usage: Logging an Attack (for testing only)
if __name__ == "__main__":
    log_attack("192.168.1.100", "192.168.1.1", "TCP", "Port Scan")
