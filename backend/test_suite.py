"""
Test Suite & Development Tools for IDS Backend
Combines testing, debugging, database management, and WebSocket utilities.
Tests all major components: Config, Database, API, WebSocket, Integration
Includes real-time WebSocket connection testing with Socket.IO client.
Provides development-only utilities for localhost testing and debugging.
"""

import sys
import os
import time
import json
import requests
import random
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import socketio client for WebSocket tests
try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

# Color Codes for Terminal Output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_test(name, status, message=""):
    """Print formatted test result."""
    status_symbol = f"{Colors.GREEN}✓{Colors.ENDC}" if status else f"{Colors.RED}✗{Colors.ENDC}"
    print(f"{status_symbol} {name}", end="")
    if message:
        print(f" - {message}")
    else:
        print()

def print_section(title):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}\n")

def print_header(title):
    """Print formatted section header (alias for print_section)."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")

def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")

def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.ENDC} {message}")

def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.ENDC}  {message}")

def print_info(message):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ{Colors.ENDC}  {message}")

# Test 1: Configuration Loading
def test_config():
    """Test configuration loading."""
    print_section("Test 1: Configuration Loading")
    
    try:
        from config import Config
        
        # Test required config values
        tests = [
            ("JWT_SECRET_KEY exists", Config.JWT_SECRET_KEY is not None),
            ("DATABASE_URL configured", Config.SQLALCHEMY_DATABASE_URI is not None),
            ("WEBSOCKET_PORT is integer", isinstance(Config.WEBSOCKET_PORT, int)),
            ("LOG_FILE_PATH exists", Config.LOG_FILE_PATH is not None),
            ("DEBUG_MODE is boolean", isinstance(Config.DEBUG_MODE, bool)),
        ]
        
        for test_name, result in tests:
            print_test(test_name, result)
        
        # Display config summary
        print(f"\n{Colors.YELLOW}Configuration Summary:{Colors.ENDC}")
        print(f"  Database: {Config.SQLALCHEMY_DATABASE_URI}")
        print(f"  WebSocket: {Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}")
        print(f"  Debug Mode: {Config.DEBUG_MODE}")
        print(f"  Log File: {Config.LOG_FILE_PATH}")
        
        return all(result for _, result in tests)
        
    except Exception as e:
        print_test("Config loading", False, str(e))
        return False

# Test 2: Database Operations
def test_database():
    """Test database connectivity and operations."""
    print_section("Test 2: Database Operations")
    
    try:
        from backend import database
        
        # Test 1: Connection
        conn_success = database.test_database_connection()
        print_test("Database connection", conn_success)
        
        if not conn_success:
            return False
        
        # Test 2: Create attack log
        test_attack = database.log_attack(
            source_ip="192.168.1.100",
            destination_ip="192.168.1.1",
            protocol="TCP",
            attack_type="Test Suite - Port Scan"
        )
        print_test("Create attack log", test_attack is not None, f"ID: {test_attack.id if test_attack else 'None'}")
        
        # Test 3: Retrieve recent attacks
        recent = database.get_recent_attacks(limit=5)
        print_test("Retrieve recent attacks", isinstance(recent, list), f"Found {len(recent)} records")
        
        # Test 4: Get statistics
        stats = database.get_attack_statistics()
        print_test("Get attack statistics", isinstance(stats, dict), f"Total: {stats.get('total_attacks', 0)}")
        
        # Test 5: Test to_dict() method
        if recent:
            dict_test = recent[0].to_dict()
            print_test("AttackLog.to_dict()", isinstance(dict_test, dict), f"Keys: {list(dict_test.keys())}")
        
        return True
        
    except Exception as e:
        print_test("Database operations", False, str(e))
        return False

# Test 3: REST API Endpoints
def test_api(base_url="http://localhost:5003"):
    """Test REST API endpoints."""
    print_section("Test 3: REST API Endpoints")
    
    print(f"{Colors.YELLOW}Note: Server must be running on {base_url}{Colors.ENDC}\n")
    
    try:
        # Test 1: Root endpoint
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            root_success = response.status_code == 200
            print_test("Root endpoint (/)", root_success, f"Status: {response.status_code}")
            if root_success:
                print(f"  {Colors.CYAN}Response:{Colors.ENDC} {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print_test("Root endpoint (/)", False, str(e))
            return False
        
        # Test 2: Health check
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            health_success = response.status_code == 200
            print_test("Health check (/health)", health_success, f"Status: {response.status_code}")
            if health_success:
                data = response.json()
                print(f"  Database: {data.get('database')}")
                print(f"  IDS Running: {data.get('ids_running')}")
                print(f"  API Version: {data.get('api_version')}")
        except Exception as e:
            print_test("Health check (/health)", False, str(e))
        
        # Test 3: Login endpoint
        try:
            response = requests.post(
                f"{base_url}/login",
                json={"username": "admin", "password": "password123"},
                timeout=5
            )
            login_success = response.status_code == 200
            print_test("Login (/login)", login_success, f"Status: {response.status_code}")
            
            if login_success:
                token = response.json().get("access_token")
                print(f"  Token: {token[:50]}..." if token else "  No token")
                
                # Test 4: Protected endpoint with token
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(f"{base_url}/logs", headers=headers, timeout=5)
                logs_success = response.status_code == 200
                print_test("Get logs (/logs)", logs_success, f"Status: {response.status_code}")
                
                if logs_success:
                    data = response.json()
                    print(f"  Total logs: {data.get('total', 0)}")
                    print(f"  Returned: {len(data.get('logs', []))}")
        except Exception as e:
            print_test("Login endpoint", False, str(e))
        
        return True
        
    except Exception as e:
        print_test("API tests", False, str(e))
        return False

# Test 4: WebSocket Handler Functions
def test_websocket_handler():
    """Test WebSocket handler functions."""
    print_section("Test 4: WebSocket Handler Functions")
    
    try:
        from backend import websocket_handler
        
        print(f"{Colors.YELLOW}Testing broadcast functions (server must be running):{Colors.ENDC}")
        
        # Test 1: broadcast_message
        websocket_handler.broadcast_message({
            "type": "test",
            "message": "Test Suite Broadcast",
            "timestamp": datetime.now().isoformat(),
            "status": "info"
        })
        print_test("broadcast_message()", True, "Function executed")
        
        # Test 2: broadcast_attack_alert
        websocket_handler.broadcast_attack_alert(
            "10.0.0.1", "10.0.0.2", "TCP", "Test Suite Attack"
        )
        print_test("broadcast_attack_alert()", True, "Function executed")
        
        # Test 3: broadcast_system_status
        websocket_handler.broadcast_system_status("Test Suite Status", "info")
        print_test("broadcast_system_status()", True, "Function executed")
        
        # Test 4: broadcast_statistics
        websocket_handler.broadcast_statistics({
            "total_attacks": 100,
            "unique_sources": 25,
            "protocols": {"TCP": 70, "UDP": 30}
        })
        print_test("broadcast_statistics()", True, "Function executed")
        
        return True
        
    except Exception as e:
        print_test("WebSocket handler", False, str(e))
        return False

# Test 5: Live WebSocket Connection Test
def test_websocket_connection(base_url="http://127.0.0.1:5003", timeout=10):
    """Test real-time WebSocket connection with Socket.IO client."""
    print_section("Test 5: Live WebSocket Connection")
    
    if not SOCKETIO_AVAILABLE:
        print(f"{Colors.YELLOW}Skipping live WebSocket tests (python-socketio not installed){Colors.ENDC}")
        print(f"  Install with: {Colors.CYAN}pip install python-socketio[client]{Colors.ENDC}\n")
        return False
    
    print(f"{Colors.YELLOW}Testing real-time WebSocket connectivity to {base_url}{Colors.ENDC}\n")
    
    # Test results storage
    test_results = {
        'connected': False,
        'status_received': False,
        'logs_received': False,
        'stats_received': False,
        'broadcast_received': False,
        'messages': []
    }
    
    # Create Socket.IO client
    sio = socketio.Client()
    
    # Event handlers
    @sio.event
    def connect():
        test_results['connected'] = True
        print_test("WebSocket connection", True, f"Connected (SID: {sio.sid})")
        
        # Request logs
        time.sleep(0.5)
        sio.emit('fetch_logs', {'limit': 5})
        
        # Request statistics
        time.sleep(0.5)
        sio.emit('fetch_statistics')
        
        # Test broadcast
        time.sleep(0.5)
        sio.emit('broadcast_test', {'message': 'Test Suite - WebSocket Live Test'})
    
    @sio.event
    def disconnect():
        print(f"  {Colors.CYAN}Disconnected from WebSocket{Colors.ENDC}")
    
    @sio.event
    def status(data):
        test_results['status_received'] = True
        test_results['messages'].append(('status', data))
        print_test("Status message received", True, f"{data.get('message', '')}")
    
    @sio.event
    def new_alert(data):
        test_results['broadcast_received'] = True
        test_results['messages'].append(('new_alert', data))
        alert_type = data.get('type', 'unknown')
        message = data.get('message', 'No message')
        print_test("Alert broadcast received", True, f"[{alert_type}] {message}")
    
    @sio.event
    def logs_update(data):
        test_results['logs_received'] = True
        test_results['messages'].append(('logs_update', data))
        logs_count = len(data.get('logs', []))
        print_test("Logs update received", True, f"{logs_count} logs")
        
        # Display first 3 logs
        if logs_count > 0:
            print(f"  {Colors.CYAN}Recent attacks:{Colors.ENDC}")
            for log in data.get('logs', [])[:3]:
                print(f"    - [{log['attack_type']}] {log['source_ip']} → {log['destination_ip']}")
    
    @sio.event
    def statistics_update(data):
        test_results['stats_received'] = True
        test_results['messages'].append(('statistics_update', data))
        stats = data.get('statistics', {})
        total = stats.get('total_attacks', 0)
        sources = stats.get('unique_sources', 0)
        print_test("Statistics update received", True, f"Total: {total}, Sources: {sources}")
    
    @sio.event
    def error(data):
        error_msg = data.get('message', 'Unknown error')
        print_test("WebSocket error", False, error_msg)
    
    # Attempt connection
    try:
        print(f"{Colors.CYAN}Connecting to {base_url}...{Colors.ENDC}")
        sio.connect(base_url)
        
        # Wait for events with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.5)
            
            # Check if all expected events received
            if (test_results['status_received'] and 
                test_results['logs_received'] and 
                test_results['stats_received'] and 
                test_results['broadcast_received']):
                break
        
        # Disconnect
        sio.disconnect()
        time.sleep(0.5)
        
        # Summary
        print(f"\n{Colors.CYAN}WebSocket Test Summary:{Colors.ENDC}")
        print(f"  Connected: {Colors.GREEN if test_results['connected'] else Colors.RED}{test_results['connected']}{Colors.ENDC}")
        print(f"  Status messages: {Colors.GREEN if test_results['status_received'] else Colors.RED}{test_results['status_received']}{Colors.ENDC}")
        print(f"  Logs updates: {Colors.GREEN if test_results['logs_received'] else Colors.RED}{test_results['logs_received']}{Colors.ENDC}")
        print(f"  Statistics: {Colors.GREEN if test_results['stats_received'] else Colors.RED}{test_results['stats_received']}{Colors.ENDC}")
        print(f"  Broadcasts: {Colors.GREEN if test_results['broadcast_received'] else Colors.RED}{test_results['broadcast_received']}{Colors.ENDC}")
        print(f"  Total messages: {len(test_results['messages'])}")
        
        # Return success if all tests passed
        return all([
            test_results['connected'],
            test_results['status_received'],
            test_results['logs_received'],
            test_results['stats_received'],
            test_results['broadcast_received']
        ])
        
    except Exception as e:
        print_test("WebSocket connection test", False, str(e))
        return False

# Test 6: Integration Test
def test_integration():
    """Test database → WebSocket integration."""
    print_section("Test 6: Integration Test")
    
    print(f"{Colors.YELLOW}This test logs an attack and verifies database + broadcast{Colors.ENDC}\n")
    
    try:
        from backend import database
        
        # Create attack log (should trigger WebSocket broadcast)
        attack = database.log_attack(
            source_ip="172.16.0.100",
            destination_ip="172.16.0.1",
            protocol="UDP",
            attack_type="Integration Test - DDoS"
        )
        
        print_test("Attack logged to database", attack is not None, f"ID: {attack.id if attack else 'None'}")
        
        # Verify in database
        recent = database.get_recent_attacks(limit=1)
        db_verified = len(recent) > 0 and recent[0].id == (attack.id if attack else None)
        print_test("Database verification", db_verified)
        
        # Note about WebSocket broadcast
        print(f"\n  {Colors.CYAN}WebSocket broadcast verification:{Colors.ENDC}")
        print(f"     If Test 5 passed, broadcasts are working correctly")
        
        return True
        
    except Exception as e:
        print_test("Integration test", False, str(e))
        return False

# Test 7: File Structure
def test_file_structure():
    """Verify all required files exist."""
    print_section("Test 7: File Structure Verification")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(base_dir, "backend")
    
    required_files = [
        ("backend/.env", os.path.join(backend_dir, ".env")),
        ("backend/__init__.py", os.path.join(backend_dir, "__init__.py")),
        ("backend/__main__.py", os.path.join(backend_dir, "__main__.py")),
        ("backend/api.py", os.path.join(backend_dir, "api.py")),
        ("backend/config.py", os.path.join(backend_dir, "config.py")),
        ("backend/database.py", os.path.join(backend_dir, "database.py")),
        ("backend/websocket_handler.py", os.path.join(backend_dir, "websocket_handler.py")),
    ]
    
    all_exist = True
    for file_name, file_path in required_files:
        exists = os.path.exists(file_path)
        print_test(file_name, exists, "" if exists else f"Missing: {file_path}")
        all_exist = all_exist and exists
    
    # Check for removed files
    print(f"\n{Colors.CYAN}Deprecated files (should be removed):{Colors.ENDC}")
    deprecated_files = [
        ("ws_server.py", os.path.join(backend_dir, "ws_server.py")),
        ("test_websocket_client.py", os.path.join(backend_dir, "test_websocket_client.py")),
        ("test_websocket.html", os.path.join(backend_dir, "test_websocket.html")),
        ("dev_tools.py", os.path.join(backend_dir, "dev_tools.py")),  # Now deprecated
    ]
    
    for file_name, file_path in deprecated_files:
        removed = not os.path.exists(file_path)
        print_test(f"{file_name} removed", removed, "✓ Removed" if removed else "✗ Still exists (can be deleted)")
    
    # Check directories
    print(f"\n{Colors.CYAN}Required directories:{Colors.ENDC}")
    logs_dir = os.path.join(backend_dir, "logs")
    models_dir = os.path.join(backend_dir, "models")
    
    print_test("logs/ directory", os.path.exists(logs_dir))
    print_test("models/ directory", os.path.exists(models_dir))
    
    return all_exist

# Development Tools
# Development Tools 1: Attack Generation Functions
def generate_random_attack():
    """Generate a single random attack log for testing."""
    from backend import database
    
    attack_types = [
        "Port Scan", "DDoS", "SQL Injection", "XSS", 
        "Brute Force", "Man-in-the-Middle", "DNS Poisoning",
        "SYN Flood", "Ping Flood", "Buffer Overflow"
    ]
    protocols = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS"]
    
    # Generate random IPs
    src_ip = f"192.168.{random.randint(1, 255)}.{random.randint(100, 200)}"
    dst_ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 50)}"
    protocol = random.choice(protocols)
    attack_type = random.choice(attack_types)
    
    attack = database.log_attack(src_ip, dst_ip, protocol, attack_type)
    
    if attack:
        print_success(f"Generated: [{attack_type}] {src_ip} → {dst_ip} ({protocol})")
    else:
        print_error("Failed to generate attack")
    
    return attack

def generate_bulk_attacks(count=10, delay=0.1):
    """Generate multiple random attacks with delay between each."""
    print_header(f"Generating {count} Random Attacks")
    
    created_attacks = []
    start_time = time.time()
    
    for i in range(count):
        attack = generate_random_attack()
        if attack:
            created_attacks.append(attack)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print_info(f"Progress: {i + 1}/{count} attacks generated")
        
        time.sleep(delay)
    
    elapsed_time = time.time() - start_time
    
    print_success(f"Generated {len(created_attacks)} attacks in {elapsed_time:.2f} seconds")
    return created_attacks

# Development Tools 2: WebSocket Testing Functions
def stress_test_websocket(count=10, delay=1):
    """Stress test WebSocket broadcasts by generating attacks."""
    print_header(f"WebSocket Stress Test ({count} broadcasts)")
    
    print_warning("Ensure backend server is running: python -m backend")
    print_info("WebSocket clients will receive real-time alerts\n")
    
    successful_broadcasts = 0
    
    for i in range(count):
        attack = generate_random_attack()
        
        if attack:
            successful_broadcasts += 1
            print_info(f"[{i+1}/{count}] Broadcast: {attack.attack_type}")
        else:
            print_error(f"[{i+1}/{count}] Broadcast failed")
        
        time.sleep(delay)
    
    print_success(f"Stress test complete! {successful_broadcasts}/{count} successful broadcasts")

def test_websocket_events():
    """Test all WebSocket event types."""
    from backend import websocket_handler, database
    
    print_header("Testing WebSocket Events")
    
    print_warning("Ensure backend server is running: python -m backend\n")
    
    # Test 1: Attack alert
    print_info("Test 1: Broadcasting attack alert...")
    websocket_handler.broadcast_attack_alert(
        "192.168.1.100", "192.168.1.1", "TCP", "Test Suite - Attack Alert"
    )
    print_success("Attack alert broadcasted")
    time.sleep(1)
    
    # Test 2: System status
    print_info("Test 2: Broadcasting system status...")
    websocket_handler.broadcast_system_status(
        "Test Suite - System Operational", status="success"
    )
    print_success("System status broadcasted")
    time.sleep(1)
    
    # Test 3: Statistics
    print_info("Test 3: Broadcasting statistics...")
    stats = database.get_attack_statistics()
    websocket_handler.broadcast_statistics(stats)
    print_success("Statistics broadcasted")
    time.sleep(1)
    
    # Test 4: Custom message
    print_info("Test 4: Broadcasting custom message...")
    websocket_handler.broadcast_message({
        "type": "test",
        "message": "Test Suite Custom Test Message",
        "timestamp": datetime.now().isoformat(),
        "status": "info",
        "details": {"test_mode": True, "source": "test_suite.py"}
    })
    print_success("Custom message broadcasted")
    
    print_success("\nAll WebSocket event tests completed!")

# Development Tools 3: Database Management Functions
def print_database_stats():
    """Print comprehensive database statistics."""
    from backend import database
    
    print_header("Database Statistics")
    
    # Get statistics
    stats = database.get_attack_statistics()
    type_counts = database.get_attack_count_by_type()
    ip_counts = database.get_attack_count_by_ip()
    
    # General stats
    print(f"{Colors.BOLD}General Statistics:{Colors.ENDC}")
    print(f"  Total Attacks:    {Colors.CYAN}{stats['total_attacks']}{Colors.ENDC}")
    print(f"  Unique Sources:   {Colors.CYAN}{stats['unique_sources']}{Colors.ENDC}")
    
    # Protocol breakdown
    print(f"\n{Colors.BOLD}Protocol Breakdown:{Colors.ENDC}")
    if stats['protocols']:
        for proto, count in sorted(stats['protocols'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_attacks'] * 100) if stats['total_attacks'] > 0 else 0
            print(f"  {proto.ljust(10)}: {Colors.GREEN}{count}{Colors.ENDC} ({percentage:.1f}%)")
    else:
        print(f"  {Colors.YELLOW}No protocol data{Colors.ENDC}")
    
    # Attack type breakdown
    print(f"\n{Colors.BOLD}Attack Type Breakdown:{Colors.ENDC}")
    if type_counts:
        for atype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / stats['total_attacks'] * 100) if stats['total_attacks'] > 0 else 0
            print(f"  {atype.ljust(25)}: {Colors.GREEN}{count}{Colors.ENDC} ({percentage:.1f}%)")
    else:
        print(f"  {Colors.YELLOW}No attack type data{Colors.ENDC}")
    
    # Top attacking IPs
    print(f"\n{Colors.BOLD}Top 10 Attacking IPs:{Colors.ENDC}")
    if ip_counts:
        for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {ip.ljust(15)}: {Colors.RED}{count}{Colors.ENDC} attacks")
    else:
        print(f"  {Colors.YELLOW}No IP data{Colors.ENDC}")
    
    print()

def show_recent_attacks(limit=10):
    """Display recent attack logs in a formatted table."""
    from backend import database
    
    print_header(f"Recent {limit} Attacks")
    
    attacks = database.get_recent_attacks(limit=limit)
    
    if not attacks:
        print_warning("No attacks found in database")
        return
    
    # Print table header
    print(f"{Colors.BOLD}{'ID':<5} {'Timestamp':<20} {'Source IP':<15} {'Dest IP':<15} {'Protocol':<8} {'Attack Type':<20}{Colors.ENDC}")
    print("-" * 100)
    
    # Print attacks
    for attack in attacks:
        timestamp = attack.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{attack.id:<5} {timestamp:<20} {attack.source_ip:<15} {attack.destination_ip:<15} {attack.protocol:<8} {attack.attack_type:<20}")
    
    print()

def reset_database_interactive():
    """Interactive database reset with confirmation prompt."""
    from backend import database
    from config import Config
    
    print_header("Reset Database")
    
    if not Config.DEBUG_MODE:
        print_error("Database reset only allowed in DEBUG_MODE")
        print_info("Set DEBUG_MODE=True in .env to enable")
        return
    
    print_warning("This will DELETE ALL attack logs and recreate tables!")
    print_info("Current database stats:")
    stats = database.get_attack_statistics()
    print(f"  - Total attacks: {stats['total_attacks']}")
    print(f"  - Unique sources: {stats['unique_sources']}\n")
    
    confirmation = input(f"{Colors.YELLOW}Type 'RESET' to confirm: {Colors.ENDC}")
    
    if confirmation == "RESET":
        if database.reset_database():
            print_success("Database reset complete!")
        else:
            print_error("Database reset failed!")
    else:
        print_info("Reset cancelled")

def truncate_database_interactive():
    """Interactive database truncation with confirmation prompt."""
    from backend import database
    from config import Config
    
    print_header("Truncate Database")
    
    if not Config.DEBUG_MODE:
        print_error("Database truncate only allowed in DEBUG_MODE")
        print_info("Set DEBUG_MODE=True in .env to enable")
        return
    
    stats = database.get_attack_statistics()
    
    print_warning(f"This will DELETE {stats['total_attacks']} attack logs!")
    print_info("Table structure will be preserved\n")
    
    confirmation = input(f"{Colors.YELLOW}Type 'DELETE' to confirm: {Colors.ENDC}")
    
    if confirmation == "DELETE":
        deleted = database.truncate_database()
        print_success(f"Truncated {deleted} attack logs!")
    else:
        print_info("Truncate cancelled")

def seed_database_interactive(count=None):
    """Interactive database seeding with custom count."""
    from backend import database
    from config import Config
    
    print_header("Seed Database with Test Data")
    
    if not Config.DEBUG_MODE:
        print_error("Database seeding only allowed in DEBUG_MODE")
        print_info("Set DEBUG_MODE=True in .env to enable")
        return
    
    if count is None:
        try:
            count = int(input(f"{Colors.CYAN}How many test attacks to generate? (default: 10): {Colors.ENDC}") or "10")
        except ValueError:
            print_error("Invalid count, using default: 10")
            count = 10
    
    print_info(f"Generating {count} test attacks...")
    
    attacks = database.seed_test_data(count=count)
    
    if attacks:
        print_success(f"Seeded {len(attacks)} test attacks!")
        print_database_stats()
    else:
        print_error("Failed to seed database")

def export_database_interactive():
    """Interactive database export to JSON file."""
    from backend import database
    
    print_header("Export Database to JSON")
    
    default_file = f"logs/database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename = input(f"{Colors.CYAN}Export filename (default: {default_file}): {Colors.ENDC}") or default_file
    
    print_info(f"Exporting database to {filename}...")
    
    if database.export_database_to_json(filename):
        print_success(f"Database exported to {filename}")
        
        # Show file size
        if os.path.exists(filename):
            size_kb = os.path.getsize(filename) / 1024
            print_info(f"File size: {size_kb:.2f} KB")
    else:
        print_error("Export failed!")

def import_database_interactive():
    """Interactive database import from JSON file."""
    from backend import database
    
    print_header("Import Database from JSON")
    
    filename = input(f"{Colors.CYAN}Import filename: {Colors.ENDC}")
    
    if not filename:
        print_error("No filename provided")
        return
    
    if not os.path.exists(filename):
        print_error(f"File not found: {filename}")
        return
    
    print_warning("This will ADD imported attacks to existing database")
    confirmation = input(f"{Colors.YELLOW}Continue? (yes/no): {Colors.ENDC}")
    
    if confirmation.lower() == "yes":
        print_info(f"Importing from {filename}...")
        count = database.import_database_from_json(filename)
        
        if count > 0:
            print_success(f"Imported {count} attack logs!")
            print_database_stats()
        else:
            print_error("Import failed or no records imported")
    else:
        print_info("Import cancelled")

# Development Tools 4: System Information Functions
def show_system_info():
    """Display system and configuration information."""
    from config import Config
    
    print_header("IDS System Information")
    
    print(f"{Colors.BOLD}Backend Configuration:{Colors.ENDC}")
    print(f"  Database:     {Colors.CYAN}{Config.SQLALCHEMY_DATABASE_URI}{Colors.ENDC}")
    print(f"  WebSocket:    {Colors.CYAN}{Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}{Colors.ENDC}")
    print(f"  Debug Mode:   {Colors.GREEN if Config.DEBUG_MODE else Colors.RED}{Config.DEBUG_MODE}{Colors.ENDC}")
    print(f"  Log File:     {Colors.CYAN}{Config.LOG_FILE_PATH}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Python Environment:{Colors.ENDC}")
    print(f"  Version:      {Colors.CYAN}{sys.version.split()[0]}{Colors.ENDC}")
    print(f"  Executable:   {Colors.CYAN}{sys.executable}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Directory Structure:{Colors.ENDC}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "models")
    
    print(f"  Backend:      {Colors.CYAN}{base_dir}{Colors.ENDC}")
    print(f"  Logs:         {Colors.GREEN if os.path.exists(logs_dir) else Colors.RED}{logs_dir}{Colors.ENDC}")
    print(f"  Models:       {Colors.GREEN if os.path.exists(models_dir) else Colors.RED}{models_dir}{Colors.ENDC}")
    
    # Database file size
    db_file = os.path.join(base_dir, "logs", "ids_logs.db")
    if os.path.exists(db_file):
        size_kb = os.path.getsize(db_file) / 1024
        print(f"\n{Colors.BOLD}Database File:{Colors.ENDC}")
        print(f"  Path:         {Colors.CYAN}{db_file}{Colors.ENDC}")
        print(f"  Size:         {Colors.CYAN}{size_kb:.2f} KB{Colors.ENDC}")
    
    print()

def show_help():
    """Display help information for all commands."""
    print_header("IDS Test Suite & Development Tools - Help")
    
    print(f"{Colors.BOLD}Testing Commands:{Colors.ENDC}")
    print(f"  {Colors.CYAN}test{Colors.ENDC}                - Run all test suites")
    print(f"  {Colors.CYAN}test-api{Colors.ENDC}            - Test REST API only")
    print(f"  {Colors.CYAN}test-ws{Colors.ENDC}             - Test WebSocket only")
    print(f"  {Colors.CYAN}test-db{Colors.ENDC}             - Test database only")
    
    print(f"\n{Colors.BOLD}Attack Generation:{Colors.ENDC}")
    print(f"  {Colors.CYAN}random{Colors.ENDC}              - Generate single random attack")
    print(f"  {Colors.CYAN}bulk <count>{Colors.ENDC}        - Generate multiple attacks (default: 10)")
    print(f"  {Colors.CYAN}stress <count>{Colors.ENDC}      - WebSocket stress test (default: 10)")
    
    print(f"\n{Colors.BOLD}Database Management:{Colors.ENDC}")
    print(f"  {Colors.CYAN}stats{Colors.ENDC}               - Show database statistics")
    print(f"  {Colors.CYAN}recent <limit>{Colors.ENDC}      - Show recent attacks (default: 10)")
    print(f"  {Colors.CYAN}seed <count>{Colors.ENDC}        - Seed test data (requires DEBUG_MODE)")
    print(f"  {Colors.CYAN}reset{Colors.ENDC}               - Reset database (requires DEBUG_MODE)")
    print(f"  {Colors.CYAN}truncate{Colors.ENDC}            - Truncate database (requires DEBUG_MODE)")
    print(f"  {Colors.CYAN}export{Colors.ENDC}              - Export database to JSON")
    print(f"  {Colors.CYAN}import{Colors.ENDC}              - Import database from JSON")
    
    print(f"\n{Colors.BOLD}WebSocket Testing:{Colors.ENDC}")
    print(f"  {Colors.CYAN}ws-test{Colors.ENDC}             - Test all WebSocket events")
    print(f"  {Colors.CYAN}stress <count>{Colors.ENDC}      - Stress test WebSocket broadcasts")
    
    print(f"\n{Colors.BOLD}System Information:{Colors.ENDC}")
    print(f"  {Colors.CYAN}info{Colors.ENDC}                - Show system information")
    print(f"  {Colors.CYAN}help{Colors.ENDC}                - Show this help message")
    
    print(f"\n{Colors.BOLD}Examples:{Colors.ENDC}")
    print(f"  {Colors.GREEN}python backend/test_suite.py test{Colors.ENDC}            # Run all tests")
    print(f"  {Colors.GREEN}python backend/test_suite.py random{Colors.ENDC}          # Generate random attack")
    print(f"  {Colors.GREEN}python backend/test_suite.py bulk 50{Colors.ENDC}         # Generate 50 attacks")
    print(f"  {Colors.GREEN}python backend/test_suite.py stress 20{Colors.ENDC}       # Stress test")
    print(f"  {Colors.GREEN}python backend/test_suite.py stats{Colors.ENDC}           # Show statistics")
    print(f"  {Colors.GREEN}python backend/test_suite.py seed 100{Colors.ENDC}        # Seed 100 attacks")
    
    print()

# Main Test Runner
def run_all_tests():
    """Run all test suites."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}IDS Backend - Test Suite{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.CYAN}Python version: {sys.version.split()[0]}{Colors.ENDC}")
    print(f"{Colors.CYAN}Socket.IO client: {'✓ Installed' if SOCKETIO_AVAILABLE else '✗ Not installed'}{Colors.ENDC}\n")
    
    results = {}
    
    # Run tests in order
    results['config'] = test_config()
    results['file_structure'] = test_file_structure()
    results['database'] = test_database()
    results['websocket_handler'] = test_websocket_handler()
    results['api'] = test_api()
    results['websocket_live'] = test_websocket_connection() if SOCKETIO_AVAILABLE else None
    results['integration'] = test_integration()
    
    # Summary
    print_section("Test Summary")
    
    # Calculate stats (exclude None results)
    valid_results = {k: v for k, v in results.items() if v is not None}
    total = len(valid_results)
    passed = sum(1 for v in valid_results.values() if v)
    failed = total - passed
    
    print(f"{Colors.BOLD}Total Tests: {total}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Passed: {passed}{Colors.ENDC}")
    print(f"{Colors.RED}✗ Failed: {failed}{Colors.ENDC}")
    print(f"{Colors.CYAN}Success Rate: {passed/total*100:.1f}%{Colors.ENDC}\n")
    
    # Detailed results
    print(f"{Colors.BOLD}Detailed Results:{Colors.ENDC}")
    for test_name, success in results.items():
        if success is None:
            status = f"{Colors.YELLOW}SKIP{Colors.ENDC}"
        else:
            status = f"{Colors.GREEN}PASS{Colors.ENDC}" if success else f"{Colors.RED}FAIL{Colors.ENDC}"
        print(f"  {test_name.ljust(20)}: {status}")
    
    # Recommendations
    if not SOCKETIO_AVAILABLE:
        print(f"\n{Colors.YELLOW}Recommendation:{Colors.ENDC}")
        print(f"   Install python-socketio for live WebSocket tests:")
        print(f"   {Colors.CYAN}pip install python-socketio[client]{Colors.ENDC}")
    
    if failed > 0:
        print(f"\n{Colors.RED}Some tests failed. Check output above for details.{Colors.ENDC}")
    else:
        print(f"\n{Colors.GREEN}All tests passed successfully!{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")
    
    return failed == 0

# Main Entry Point
def main():
    """Main entry point for test suite and development tools."""
    from config import Config
    
    # Check DEBUG_MODE for destructive operations
    if not Config.DEBUG_MODE:
        print_warning("Running in PRODUCTION mode")
        print_info("Some features require DEBUG_MODE=True in .env\n")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        # Default behavior: run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
    
    command = sys.argv[1].lower()
    
    try:
        # Testing commands
        if command == "test":
            success = run_all_tests()
            sys.exit(0 if success else 1)
        
        elif command == "test-api":
            success = test_api()
            sys.exit(0 if success else 1)
        
        elif command == "test-ws":
            success = test_websocket_connection() if SOCKETIO_AVAILABLE else False
            sys.exit(0 if success else 1)
        
        elif command == "test-db":
            success = test_database()
            sys.exit(0 if success else 1)
        
        # Attack generation commands
        elif command == "random":
            generate_random_attack()
        
        elif command == "bulk":
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            generate_bulk_attacks(count)
        
        elif command == "stress":
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            stress_test_websocket(count)
        
        # Database management commands
        elif command == "stats":
            print_database_stats()
        
        elif command == "recent":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            show_recent_attacks(limit)
        
        elif command == "seed":
            count = int(sys.argv[2]) if len(sys.argv) > 2 else None
            seed_database_interactive(count)
        
        elif command == "reset":
            reset_database_interactive()
        
        elif command == "truncate":
            truncate_database_interactive()
        
        elif command == "export":
            export_database_interactive()
        
        elif command == "import":
            import_database_interactive()
        
        # WebSocket testing commands
        elif command == "ws-test":
            test_websocket_events()
        
        # System information commands
        elif command == "info":
            show_system_info()
        
        elif command == "help":
            show_help()
        
        else:
            print_error(f"Unknown command: {command}")
            print_info("Run 'python backend/test_suite.py help' for usage information")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print_warning("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        if Config.DEBUG_MODE:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()