"""
WebSocket Message Broadcasting and Client Management
Provides utility functions to broadcast messages to all connected clients.
Designed to work with the unified SocketIO instance in backend/__init__.py
"""

import logging
import sys
import os

# Configure logger
logger = logging.getLogger("WebSocket-Handler")

def broadcast_message(message):
    """
    Broadcast a message to all connected WebSocket clients.
    
    Args:
        message: Dictionary containing message data
                Expected format:
                {
                    "type": "attack_detected|system|test|performance|report",
                    "message": "Human-readable message",
                    "timestamp": "ISO format timestamp",
                    "status": "alert|info|warning|success|error",
                    "details": {...}  # Optional additional data
                }
    
    Example:
        broadcast_message({
            "type": "attack_detected",
            "message": "Port scan detected from 192.168.1.100",
            "timestamp": "2025-11-27T12:00:00",
            "status": "alert",
            "details": {
                "source_ip": "192.168.1.100",
                "destination_ip": "192.168.1.1",
                "protocol": "TCP",
                "attack_type": "Port Scan"
            }
        })
    
    Note:
        Uses lazy import of socketio to avoid circular dependencies.
        The socketio instance is imported from backend.__init__.py
    """
    try:
        # Lazy import to avoid circular dependency
        # This import happens at runtime, not at module load time
        from backend import socketio
        
        # Validate message structure
        if not isinstance(message, dict):
            logger.warning(f"Invalid message type: {type(message)}. Expected dict.")
            return
        
        # Ensure required fields exist
        if "type" not in message:
            message["type"] = "unknown"
        if "status" not in message:
            message["status"] = "info"
        
        # Emit to all connected clients via the 'new_alert' event
        socketio.emit("new_alert", message, namespace="/")
        
        # Log broadcast details
        msg_type = message.get('type', 'unknown')
        msg_content = message.get('message', 'No message')
        logger.debug(f"Broadcast: [{msg_type}] {msg_content}")
        
    except ImportError as import_err:
        logger.error(f"SocketIO not available: {import_err}")
        logger.error("Ensure backend.__init__.py has been imported and socketio is initialized")
    except Exception as e:
        logger.error(f"Broadcast failed: {e}")

def broadcast_attack_alert(source_ip, destination_ip, protocol, attack_type):
    """
    Convenience function to broadcast attack detection alerts.
    
    Args:
        source_ip: Attacker's IP address
        destination_ip: Target IP address
        protocol: Network protocol (TCP/UDP/ICMP)
        attack_type: Type of attack detected
    
    Example:
        broadcast_attack_alert(
            "192.168.1.100", 
            "192.168.1.1", 
            "TCP", 
            "Port Scan"
        )
    """
    import datetime
    
    broadcast_message({
        "type": "attack_detected",
        "message": f"{attack_type} detected: {source_ip} -> {destination_ip}",
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "alert",
        "details": {
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "protocol": protocol,
            "attack_type": attack_type
        }
    })

def broadcast_system_status(message, status="info"):
    """
    Convenience function to broadcast system status messages.
    
    Args:
        message: Status message to broadcast
        status: Message severity (info, warning, error, success)
    
    Example:
        broadcast_system_status("IDS Started!", status="success")
    """
    import datetime
    
    broadcast_message({
        "type": "system",
        "message": message,
        "timestamp": datetime.datetime.now().isoformat(),
        "status": status
    })

def broadcast_statistics(stats):
    """
    Convenience function to broadcast attack statistics.
    
    Args:
        stats: Dictionary containing attack statistics
               Expected keys: total_attacks, unique_sources, protocols, attack_types
    
    Example:
        broadcast_statistics({
            "total_attacks": 150,
            "unique_sources": 25,
            "protocols": {"TCP": 100, "UDP": 50},
            "attack_types": {"Port Scan": 75, "DoS": 75}
        })
    """
    import datetime
    
    broadcast_message({
        "type": "statistics",
        "message": f"Attack statistics updated: {stats.get('total_attacks', 0)} total attacks",
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "info",
        "details": stats
    })

def get_active_clients():
    """
    Get count of active WebSocket clients.
    
    Returns:
        int: Number of connected clients (or -1 if unavailable)
    
    Example:
        client_count = get_active_clients()
        print(f"Active clients: {client_count}")
    """
    try:
        from backend import socketio
        from flask import request
        
        # Get all connected sessions
        # Note: This is a simplified approach; actual implementation may vary
        # based on Flask-SocketIO version
        return len(socketio.server.manager.rooms.get('/', {}).keys())
        
    except Exception as e:
        logger.warning(f"Failed to get active client count: {e}")
        return -1

# Standalone Testing
if __name__ == "__main__":
    """
    Standalone testing mode for websocket_handler.
    Tests broadcasting without running the full server.
    """
    import datetime
    
    # Add parent directory to path for imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    print("=" * 70)
    print("WebSocket Handler - Standalone Test")
    print("=" * 70)
    
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    )
    
    print("\nNote: These tests will show import errors unless the backend server is running.")
    print("To test with actual broadcasts, run: python -m backend")
    print("Then in another terminal: python -c \"from backend.websocket_handler import broadcast_system_status; broadcast_system_status('CLI Test')\"")
    
    print("\n[1/5] Testing broadcast_message()...")
    broadcast_message({
        "type": "test",
        "message": "Test broadcast message",
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "info"
    })
    print("✓ Broadcast test completed (check logs for errors)")
    
    print("\n[2/5] Testing broadcast_attack_alert()...")
    broadcast_attack_alert(
        source_ip="192.168.1.100",
        destination_ip="192.168.1.1",
        protocol="TCP",
        attack_type="Port Scan (Test)"
    )
    print("✓ Attack alert test completed")
    
    print("\n[3/5] Testing broadcast_system_status()...")
    broadcast_system_status("Test system status message", status="success")
    print("✓ System status test completed")
    
    print("\n[4/5] Testing broadcast_statistics()...")
    broadcast_statistics({
        "total_attacks": 100,
        "unique_sources": 25,
        "protocols": {"TCP": 70, "UDP": 30},
        "attack_types": {"Port Scan": 50, "DoS": 50}
    })
    print("✓ Statistics broadcast test completed")
    
    print("\n[5/5] Testing get_active_clients()...")
    client_count = get_active_clients()
    print(f"✓ Active clients: {client_count}")
    
    print("\n" + "=" * 70)
    print("WebSocket Handler tests completed!")
    print("Note: Actual broadcasts require running backend server (python -m backend)")
    print("=" * 70)