"""
Database Models and Logging for IDS
Provides SQLAlchemy models, database session management, and attack logging.
Uses lazy imports to avoid circular dependencies.
Includes development-only utilities for localhost testing.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import logging
import os
from logging.handlers import RotatingFileHandler
from config import Config

# Ensure logs directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)

# Database Configuration
# Use Config class for database URL
DATABASE_URL = Config.SQLALCHEMY_DATABASE_URI

Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,  # Verify connections before using
    echo=Config.DEBUG_MODE  # SQL logging in debug mode
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False  # Prevent detached instance errors
)

# Database Models
class AttackLog(Base):
    """
    Attack log table for storing intrusion detection events.
    
    Attributes:
        id: Unique identifier (auto-increment)
        timestamp: Attack detection time (UTC)
        source_ip: Attacker's IP address
        destination_ip: Target IP address
        protocol: Network protocol (TCP/UDP/ICMP)
        attack_type: Classification (Port Scan, DoS, etc.)
    """
    __tablename__ = "attack_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.now, nullable=False)
    source_ip = Column(String(45), nullable=False)
    destination_ip = Column(String(45), nullable=False)
    protocol = Column(String(10), nullable=False)
    attack_type = Column(String(50), nullable=False)
    
    def __repr__(self):
        return (
            f"<AttackLog(id={self.id}, "
            f"timestamp={self.timestamp}, "
            f"source={self.source_ip}, "
            f"type={self.attack_type})>"
        )
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "protocol": self.protocol,
            "attack_type": self.attack_type
        }

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Logging Configuration
# Use Config for log file path
log_file_path = Config.LOG_FILE_PATH

# Rotating log handler for alerts.log
log_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=Config.LOG_MAX_BYTES,
    backupCount=Config.LOG_BACKUP_COUNT
)
log_handler.setLevel(logging.INFO)

# Define log format
formatter = logging.Formatter(
    "%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_handler.setFormatter(formatter)

# Create logger instance
logger = logging.getLogger("IDSLogger")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Prevent duplicate logs in console
logger.propagate = False

# Add console handler in debug mode
if Config.DEBUG_MODE:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Core Database Operations
def log_attack(source_ip, destination_ip, protocol, attack_type):
    """
    Log attack details to database and broadcast via WebSocket.
    
    Args:
        source_ip: Attacker's IP address
        destination_ip: Target IP address
        protocol: Network protocol (TCP/UDP/ICMP)
        attack_type: Attack classification
        
    Returns:
        AttackLog: Created database record (or None on failure)
    """
    attack_record = None
    
    try:
        # Create database session
        session = SessionLocal()
        
        # Create attack log entry
        attack_record = AttackLog(
            source_ip=source_ip,
            destination_ip=destination_ip,
            protocol=protocol,
            attack_type=attack_type
        )
        
        # Save to database
        session.add(attack_record)
        session.commit()
        session.refresh(attack_record)  # Get ID after commit
        
        # Log to file
        logger.info(f"Attack detected: {attack_type} | {source_ip} -> {destination_ip} | Protocol: {protocol}")
        
        # Close session
        session.close()
        
        # Lazy import to avoid circular dependency
        # Only import when needed (after database operation completes)
        try:
            from backend.websocket_handler import broadcast_message
            
            # Broadcast to WebSocket clients
            broadcast_message({
                "type": "attack_detected",
                "timestamp": attack_record.timestamp.isoformat(),
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "protocol": protocol,
                "attack_type": attack_type,
                "status": "alert"
            })
        except ImportError as import_err:
            logger.warning(f"WebSocket handler not available: {import_err}")
        except Exception as ws_err:
            logger.warning(f"WebSocket broadcast failed: {ws_err}")
        
        return attack_record
        
    except Exception as db_error:
        logger.error(f"Failed to log attack: {db_error}")
        
        # Rollback on error
        if 'session' in locals():
            session.rollback()
            session.close()
        
        return None

def get_recent_attacks(limit=100, offset=0, attack_type=None):
    """
    Retrieve recent attack logs from database.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip (pagination)
        attack_type: Filter by attack type (optional)
        
    Returns:
        list: List of AttackLog objects
    """
    session = SessionLocal()
    try:
        query = session.query(AttackLog).order_by(AttackLog.timestamp.desc())
        
        # Apply filter if specified
        if attack_type:
            query = query.filter(AttackLog.attack_type == attack_type)
        
        # Apply pagination
        attacks = query.limit(limit).offset(offset).all()
        
        return attacks
        
    except Exception as e:
        logger.error(f"Failed to retrieve attacks: {e}")
        return []
    finally:
        session.close()

def get_attack_statistics():
    """
    Calculate attack statistics from database.
    
    Returns:
        dict: Attack statistics (total count, unique IPs, protocol breakdown)
    """
    session = SessionLocal()
    try:
        from sqlalchemy import func
        
        # Total attacks
        total_attacks = session.query(func.count(AttackLog.id)).scalar()
        
        # Unique source IPs
        unique_sources = session.query(
            func.count(func.distinct(AttackLog.source_ip))
        ).scalar()
        
        # Protocol breakdown
        protocol_stats = session.query(
            AttackLog.protocol,
            func.count(AttackLog.id)
        ).group_by(AttackLog.protocol).all()
        
        # Attack type breakdown
        attack_type_stats = session.query(
            AttackLog.attack_type,
            func.count(AttackLog.id)
        ).group_by(AttackLog.attack_type).all()
        
        return {
            "total_attacks": total_attacks or 0,
            "unique_sources": unique_sources or 0,
            "protocols": {proto: count for proto, count in protocol_stats},
            "attack_types": {atype: count for atype, count in attack_type_stats}
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate statistics: {e}")
        return {
            "total_attacks": 0,
            "unique_sources": 0,
            "protocols": {},
            "attack_types": {}
        }
    finally:
        session.close()

def clear_old_logs(days=30):
    """
    Delete attack logs older than specified days.
    
    Args:
        days: Number of days to retain (default: 30)
        
    Returns:
        int: Number of deleted records
    """
    session = SessionLocal()
    try:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        deleted_count = session.query(AttackLog).filter(
            AttackLog.timestamp < cutoff_date
        ).delete()
        
        session.commit()
        
        logger.info(f"Deleted {deleted_count} old attack logs (older than {days} days)")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to clear old logs: {e}")
        session.rollback()
        return 0
    finally:
        session.close()

# Database Health Check
def test_database_connection():
    """
    Test database connectivity.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        session = SessionLocal()
        from sqlalchemy import text
        session.execute(text("SELECT 1"))
        session.close()
        logger.info("Database connection test: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Database connection test: FAILED - {e}")
        return False

# Development-Only Utilities
def reset_database():
    """
    Reset database for clean testing (DEVELOPMENT ONLY).
    
    Drops all tables and recreates them.
    Only works when DEBUG_MODE=True.
    
    Returns:
        bool: True if reset successful, False otherwise
        
    Raises:
        PermissionError: If called when DEBUG_MODE=False
        
    Example:
        >>> from backend.database import reset_database
        >>> reset_database()
        Database reset complete (DEBUG MODE)
        True
    """
    if not Config.DEBUG_MODE:
        raise PermissionError(
            "Database reset only allowed in DEBUG_MODE. "
            "Set DEBUG_MODE=True in .env to enable."
        )
    
    try:
        logger.warning("Resetting database (DEBUG MODE)...")
        
        # Drop all tables
        Base.metadata.drop_all(engine)
        logger.debug("All tables dropped")
        
        # Recreate all tables
        Base.metadata.create_all(engine)
        logger.debug("All tables recreated")
        
        logger.warning("Database reset complete (DEBUG MODE)")
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False

def seed_test_data(count=10):
    """
    Add sample attack data for testing (DEVELOPMENT ONLY).
    
    Creates realistic test attack logs with various types and protocols.
    Only works when DEBUG_MODE=True.
    
    Args:
        count: Number of test attacks to generate (default: 10)
        
    Returns:
        list: List of created AttackLog objects
        
    Example:
        >>> from backend.database import seed_test_data
        >>> attacks = seed_test_data(5)
        Seeded 5 test attacks
        >>> len(attacks)
        5
    """
    if not Config.DEBUG_MODE:
        logger.warning("seed_test_data() only available in DEBUG_MODE")
        return []
    
    import random
    
    # Sample attack data templates
    attack_templates = [
        ("192.168.1.{}", "192.168.1.1", "TCP", "Port Scan"),
        ("10.0.0.{}", "10.0.0.1", "UDP", "DDoS Attack"),
        ("172.16.0.{}", "172.16.0.1", "ICMP", "Ping Flood"),
        ("192.168.100.{}", "192.168.100.1", "TCP", "SQL Injection"),
        ("10.10.10.{}", "10.10.10.1", "TCP", "Brute Force"),
        ("172.20.0.{}", "172.20.0.1", "UDP", "DNS Amplification"),
        ("192.168.50.{}", "192.168.50.1", "TCP", "XSS Attack"),
        ("10.100.0.{}", "10.100.0.1", "TCP", "SYN Flood"),
    ]
    
    created_attacks = []
    
    try:
        logger.info(f"Seeding {count} test attacks...")
        
        for i in range(count):
            # Select random attack template
            template = random.choice(attack_templates)
            
            # Generate random source IP
            src_ip = template[0].format(random.randint(100, 200))
            dst_ip = template[1]
            protocol = template[2]
            attack_type = template[3]
            
            # Create attack log
            attack = log_attack(src_ip, dst_ip, protocol, attack_type)
            
            if attack:
                created_attacks.append(attack)
                logger.debug(f"  [{i+1}/{count}] {attack_type}: {src_ip} → {dst_ip}")
        
        logger.info(f"Seeded {len(created_attacks)} test attacks")
        return created_attacks
        
    except Exception as e:
        logger.error(f"Seed data generation failed: {e}")
        return created_attacks

def get_attack_count_by_type():
    """
    Get attack count grouped by type (DEVELOPMENT HELPER).
    
    Returns:
        dict: Attack types with their counts
        
    Example:
        >>> from backend.database import get_attack_count_by_type
        >>> counts = get_attack_count_by_type()
        >>> print(counts)
        {'Port Scan': 15, 'DDoS Attack': 10, ...}
    """
    session = SessionLocal()
    try:
        from sqlalchemy import func
        
        attack_counts = session.query(
            AttackLog.attack_type,
            func.count(AttackLog.id).label('count')
        ).group_by(AttackLog.attack_type).all()
        
        return {attack_type: count for attack_type, count in attack_counts}
        
    except Exception as e:
        logger.error(f"Failed to get attack counts: {e}")
        return {}
    finally:
        session.close()

def get_attack_count_by_ip():
    """
    Get attack count grouped by source IP (DEVELOPMENT HELPER).
    
    Returns:
        dict: Source IPs with their attack counts
        
    Example:
        >>> from backend.database import get_attack_count_by_ip
        >>> counts = get_attack_count_by_ip()
        >>> print(counts)
        {'192.168.1.100': 5, '10.0.0.50': 3, ...}
    """
    session = SessionLocal()
    try:
        from sqlalchemy import func
        
        ip_counts = session.query(
            AttackLog.source_ip,
            func.count(AttackLog.id).label('count')
        ).group_by(AttackLog.source_ip).all()
        
        return {ip: count for ip, count in ip_counts}
        
    except Exception as e:
        logger.error(f"Failed to get IP counts: {e}")
        return {}
    finally:
        session.close()

def truncate_database():
    """
    Delete all attack logs without dropping tables (DEVELOPMENT ONLY).
    
    Faster than reset_database() when you just want to clear data.
    Only works when DEBUG_MODE=True.
    
    Returns:
        int: Number of deleted records
        
    Example:
        >>> from backend.database import truncate_database
        >>> deleted = truncate_database()
        Truncated 50 attack logs
        >>> deleted
        50
    """
    if not Config.DEBUG_MODE:
        raise PermissionError("Database truncate only allowed in DEBUG_MODE")
    
    session = SessionLocal()
    try:
        logger.warning("Truncating attack logs (DEBUG MODE)...")
        
        deleted_count = session.query(AttackLog).delete()
        session.commit()
        
        logger.warning(f"Truncated {deleted_count} attack logs")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Truncate failed: {e}")
        session.rollback()
        return 0
    finally:
        session.close()

def export_database_to_json(output_file="logs/database_export.json"):
    """
    Export all attack logs to JSON file (DEVELOPMENT HELPER).
    
    Args:
        output_file: Path to output JSON file
        
    Returns:
        bool: True if export successful
        
    Example:
        >>> from backend.database import export_database_to_json
        >>> export_database_to_json("backup.json")
        Exported 100 attack logs to backup.json
        True
    """
    import json
    
    try:
        attacks = get_recent_attacks(limit=10000)  # Get all attacks
        
        attack_data = [attack.to_dict() for attack in attacks]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(attack_data, f, indent=2)
        
        logger.info(f"Exported {len(attack_data)} attack logs to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False

def import_database_from_json(input_file="logs/database_export.json"):
    """
    Import attack logs from JSON file (DEVELOPMENT HELPER).
    
    Args:
        input_file: Path to input JSON file
        
    Returns:
        int: Number of imported records
        
    Example:
        >>> from backend.database import import_database_from_json
        >>> count = import_database_from_json("backup.json")
        Imported 100 attack logs
        >>> count
        100
    """
    import json
    
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return 0
    
    try:
        with open(input_file, 'r') as f:
            attack_data = json.load(f)
        
        imported_count = 0
        
        for data in attack_data:
            attack = log_attack(
                source_ip=data['source_ip'],
                destination_ip=data['destination_ip'],
                protocol=data['protocol'],
                attack_type=data['attack_type']
            )
            if attack:
                imported_count += 1
        
        logger.info(f"Imported {imported_count} attack logs")
        return imported_count
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return 0

# Initialization
if __name__ == "__main__":
    """
    Standalone testing and database initialization.
    """
    print("=" * 60)
    print("IDS Database Module - Standalone Test")
    print("=" * 60)
    
    # Test 1: Database connection
    print("\n[1/7] Testing database connection...")
    if test_database_connection():
        print("✓ Database connection successful")
    else:
        print("✗ Database connection failed")
        exit(1)
    
    # Test 2: Create test attack log
    print("\n[2/7] Creating test attack log...")
    test_record = log_attack(
        source_ip="192.168.1.100",
        destination_ip="192.168.1.1",
        protocol="TCP",
        attack_type="Port Scan (Test)"
    )
    
    if test_record:
        print(f"✓ Test attack logged: ID={test_record.id}")
    else:
        print("✗ Failed to log test attack")
    
    # Test 3: Retrieve recent attacks
    print("\n[3/7] Retrieving recent attacks...")
    recent_attacks = get_recent_attacks(limit=5)
    print(f"✓ Found {len(recent_attacks)} recent attacks:")
    for attack in recent_attacks:
        print(f"  - {attack}")
    
    # Test 4: Get statistics
    print("\n[4/7] Calculating attack statistics...")
    stats = get_attack_statistics()
    print(f"✓ Statistics:")
    print(f"  - Total attacks: {stats['total_attacks']}")
    print(f"  - Unique sources: {stats['unique_sources']}")
    print(f"  - Protocols: {stats['protocols']}")
    print(f"  - Attack types: {stats['attack_types']}")
    
    # Test 5: Development utilities (if DEBUG_MODE enabled)
    if Config.DEBUG_MODE:
        print("\n[5/7] Testing development utilities...")
        
        print("  - Testing seed_test_data()...")
        seeded = seed_test_data(count=3)
        print(f"  ✓ Seeded {len(seeded)} test attacks")
        
        print("  - Testing get_attack_count_by_type()...")
        type_counts = get_attack_count_by_type()
        print(f"  ✓ Attack type counts: {type_counts}")
        
        print("  - Testing export_database_to_json()...")
        if export_database_to_json("logs/test_export.json"):
            print("  ✓ Database exported successfully")
        
        print("\n[6/7] Testing truncate (WARNING: This will delete data)...")
        user_input = input("  Do you want to test truncate? (yes/no): ")
        if user_input.lower() == 'yes':
            deleted = truncate_database()
            print(f"  ✓ Truncated {deleted} records")
        else:
            print("  - Truncate test skipped")
        
        print("\n[7/7] Testing reset_database (WARNING: This will reset DB)...")
        user_input = input("  Do you want to test reset? (yes/no): ")
        if user_input.lower() == 'yes':
            if reset_database():
                print("  ✓ Database reset successful")
        else:
            print("  - Reset test skipped")
    else:
        print("\n[5/7] Skipping development utilities (DEBUG_MODE=False)")
        print("[6/7] Skipped")
        print("[7/7] Skipped")
    
    print("\n" + "=" * 60)
    print("Database test completed successfully!")
    print("=" * 60)