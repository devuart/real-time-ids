"""
REST API for IDS Management
Provides endpoints for authentication, attack logs, IDS control, 
attack simulation, performance grading, and report generation.
"""

import threading
from flask import jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_restful import Api, Resource
from werkzeug.security import generate_password_hash, check_password_hash

# Import backend modules (lazy imports for heavy modules)
from backend.database import SessionLocal, AttackLog

# User Authentication (temporary in-memory storage)
# TODO: Replace with proper database-backed user management
users = {"admin": generate_password_hash("password123")}

# Global IDS control flag to prevent multiple instances
ids_running = False
ids_thread = None

def init_api(app):
    """
    Initialize Flask-JWT and Flask-RESTful API.
    Called from backend/__init__.py or main app entry point.
    
    Args:
        app: Flask application instance
        
    Returns:
        api: Configured Flask-RESTful API instance
    """
    # Configure JWT
    jwt = JWTManager(app)
    
    # Initialize Flask-RESTful
    api = Api(app)
    
    # Register API Resources
    api.add_resource(Login, "/login")
    api.add_resource(AttackLogs, "/logs")
    api.add_resource(StartIDS, "/start-ids")
    api.add_resource(StopIDS, "/stop-ids")
    api.add_resource(SimulateAttack, "/simulate-attack")
    api.add_resource(GradeIDS, "/grade-ids")
    api.add_resource(GenerateReport, "/generate-report")
    api.add_resource(HealthCheck, "/health")
    
    return api

# API Resource Classes
class Login(Resource):
    """User authentication endpoint to obtain JWT token."""
    
    def post(self):
        """
        Authenticate user and return JWT token.
        
        Request Body:
            {
                "username": "admin",
                "password": "password123"
            }
            
        Returns:
            200: {"access_token": "eyJ0eXAi..."}
            401: {"error": "Invalid credentials"}
        """
        data = request.get_json()
        
        if not data:
            return {"error": "No data provided"}, 400
            
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return {"error": "Username and password required"}, 400

        if username in users and check_password_hash(users[username], password):
            token = create_access_token(identity=username)
            return {
                "access_token": token,
                "username": username
            }, 200

        return {"error": "Invalid credentials"}, 401

class AttackLogs(Resource):
    """Fetch attack logs from database with JWT authentication."""
    
    @jwt_required()
    def get(self):
        """
        Retrieve attack logs with optional filtering.
        
        Query Parameters:
            limit: Maximum number of logs to return (default: 100)
            offset: Number of logs to skip (default: 0)
            attack_type: Filter by attack type (optional)
            
        Returns:
            200: List of attack log objects
            500: {"error": "Database error: ..."}
        """
        # Parse query parameters
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)
        attack_type = request.args.get("attack_type", None)
        
        session = SessionLocal()
        try:
            query = session.query(AttackLog).order_by(AttackLog.timestamp.desc())
            
            # Apply filters
            if attack_type:
                query = query.filter(AttackLog.attack_type == attack_type)
            
            # Apply pagination
            logs = query.limit(limit).offset(offset).all()
            total_count = query.count()
            
            return {
                "logs": [{
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "source_ip": log.source_ip,
                    "destination_ip": log.destination_ip,
                    "protocol": log.protocol,
                    "attack_type": log.attack_type
                } for log in logs],
                "total": total_count,
                "limit": limit,
                "offset": offset
            }, 200
            
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}, 500
        finally:
            session.close()

class StartIDS(Resource):
    """Start real-time Intrusion Detection System."""
    
    @jwt_required()
    def post(self):
        """
        Start packet sniffing and intrusion detection.
        
        Request Body (optional):
            {
                "interface": "eth0",  # Network interface to monitor
                "filter": "tcp or udp"  # BPF filter expression
            }
            
        Returns:
            200: {"message": "IDS Started!", "status": "running"}
            409: {"message": "IDS is already running!", "status": "running"}
            500: {"error": "Failed to start IDS: ..."}
        """
        global ids_running, ids_thread
        
        if ids_running:
            return {
                "message": "IDS is already running!",
                "status": "running"
            }, 409

        try:
            # Lazy import to avoid circular dependencies
            from backend import ids_sniffer
            from backend.websocket_handler import broadcast_message
            #from backend.websocket_handler import broadcast_system_status
            
            # Get optional configuration
            data = request.get_json() or {}
            interface = data.get("interface", None)
            bpf_filter = data.get("filter", None)
            
            # Start IDS in background thread
            ids_thread = threading.Thread(
                target=ids_sniffer.start_sniffing,
                kwargs={"interface": interface, "filter": bpf_filter},
                daemon=True,
                name="IDS-Thread"
            )
            ids_thread.start()
            ids_running = True
            
            # Broadcast status to WebSocket clients
            broadcast_message({
                "type": "system",
                "message": "IDS Started!",
                "status": "success"
            })
            #broadcast_system_status("IDS Started!", status="success")
            
            return {
                "message": "IDS Started!",
                "status": "running",
                "interface": interface or "default",
                "filter": bpf_filter or "default"
            }, 200
            
        except Exception as e:
            return {"error": f"Failed to start IDS: {str(e)}"}, 500

class StopIDS(Resource):
    """Stop the Intrusion Detection System."""
    
    @jwt_required()
    def post(self):
        """
        Stop packet sniffing and intrusion detection.
        
        Returns:
            200: {"message": "IDS Stopped!", "status": "stopped"}
            409: {"message": "IDS is not running!", "status": "stopped"}
        """
        global ids_running
        
        if not ids_running:
            return {
                "message": "IDS is not running!",
                "status": "stopped"
            }, 409

        try:
            # Lazy import
            from backend.websocket_handler import broadcast_message
            #from backend.websocket_handler import broadcast_system_status
            
            # Note: Scapy sniffing requires manual stopping mechanism
            # # TODO: This is a placeholder - implement actual stop logic in ids_sniffer.py
            ids_running = False
            
            # Broadcast status
            broadcast_message({
                "type": "system",
                "message": "IDS Stopped!",
                "status": "warning"
            })
            #broadcast_system_status("IDS Stopped!", status="warning")
            
            return {
                "message": "IDS Stopped!",
                "status": "stopped"
            }, 200
            
        except Exception as e:
            return {"error": f"Failed to stop IDS: {str(e)}"}, 500

class SimulateAttack(Resource):
    """Trigger attack simulation for testing IDS."""
    
    # Define valid attack types
    VALID_ATTACKS = ["port_scan", "dos", "sql_injection", "syn_flood"]
    
    @jwt_required()
    def post(self):
        """
        Simulate various types of attacks.
        
        Request Body:
            {
                "attack_type": "port_scan",  # Required
                "target_ip": "127.0.0.1",    # Required
                "intensity": 5,              # 1-10 (default: 5)
                "duration": 10               # Seconds (optional)
            }
            
        Returns:
            200: {"message": "Attack simulation started", "details": {...}}
            400: {"error": "Invalid attack type/parameters"}
            500: {"error": "Attack simulation error: ..."}
        """
        data = request.get_json()
        
        if not data:
            return {"error": "No data provided"}, 400
        
        # Validate required fields
        attack_type = data.get("attack_type")
        target_ip = data.get("target_ip")
        
        if not attack_type or not target_ip:
            return {
                "error": "attack_type and target_ip are required"
            }, 400
        
        # Validate attack type
        if attack_type not in self.VALID_ATTACKS:
            return {
                "error": f"Invalid attack type. Choose from {self.VALID_ATTACKS}"
            }, 400
        
        # Validate intensity
        intensity = data.get("intensity", 5)
        if not (1 <= intensity <= 10):
            return {"error": "Intensity must be between 1 and 10"}, 400
        
        try:
            # Lazy imports
            from backend import attack_simulation
            from backend.websocket_handler import broadcast_message
            
            # Run attack simulation
            result = attack_simulation.run_attack(
                attack_type=attack_type,
                target_ip=target_ip,
                options={
                    "intensity": intensity,
                    "duration": data.get("duration", 10)
                }
            )
            
            # Broadcast notification
            broadcast_message({
                "type": "attack_simulation",
                "message": f"Simulated {attack_type} attack on {target_ip}",
                "details": {
                    "attack_type": attack_type,
                    "target": target_ip,
                    "intensity": intensity
                },
                "status": "info"
            })
            
            return {
                "message": "Attack simulation started",
                "details": result
            }, 200
            
        except Exception as e:
            return {
                "error": f"Attack simulation error: {str(e)}"
            }, 500

class GradeIDS(Resource):
    """Evaluate IDS performance and accuracy."""
    
    @jwt_required()
    def get(self):
        """
        Calculate IDS performance metrics.
        
        Returns:
            200: {
                "accuracy": 0.95,
                "detection_rate": 0.92,
                "false_positive_rate": 0.05,
                "response_time_avg": 0.003,
                "total_attacks": 1000,
                "detected_attacks": 920
            }
            500: {"error": "Performance grading error: ..."}
        """
        try:
            # Lazy import
            from backend import performance_grader
            from backend.websocket_handler import broadcast_message
            
            # Calculate performance metrics
            score_data = performance_grader.evaluate_performance()
            
            # Broadcast notification
            broadcast_message({
                "type": "performance",
                "message": f"IDS Performance: {score_data.get('accuracy', 0):.2%}",
                "status": "info"
            })
            
            return score_data, 200
            
        except Exception as e:
            return {
                "error": f"Performance grading error: {str(e)}"
            }, 500

class GenerateReport(Resource):
    """Generate IDS security report in various formats."""
    
    VALID_FORMATS = ["json", "csv", "pdf"]
    
    @jwt_required()
    def get(self):
        """
        Export attack logs and analytics.
        
        Query Parameters:
            format: Report format (json/csv/pdf, default: json)
            start_date: Filter start date (ISO format, optional)
            end_date: Filter end date (ISO format, optional)
            
        Returns:
            200: Report data (format-dependent)
            400: {"error": "Invalid format"}
            500: {"error": "Report generation error: ..."}
        """
        format_type = request.args.get("format", "json").lower()
        
        if format_type not in self.VALID_FORMATS:
            return {
                "error": f"Invalid format. Choose from {self.VALID_FORMATS}"
            }, 400
        
        try:
            # Lazy import
            from backend import report_generator
            from backend.websocket_handler import broadcast_message
            
            # Get optional date filters
            start_date = request.args.get("start_date")
            end_date = request.args.get("end_date")
            
            # Generate report
            response = report_generator.generate_report(
                format_type=format_type,
                start_date=start_date,
                end_date=end_date
            )
            
            # Broadcast notification
            broadcast_message({
                "type": "report",
                "message": f"Generated {format_type.upper()} report",
                "status": "success"
            })
            
            return response
            
        except Exception as e:
            return {
                "error": f"Report generation error: {str(e)}"
            }, 500

class HealthCheck(Resource):
    """API health check endpoint."""
    
    def get(self):
        """
        Check API and IDS status.
        
        Returns:
            200: {
                "status": "healthy",
                "ids_running": true,
                "database": "connected",
                "websocket": "active"
            }
        """
        global ids_running
        
        # Check database connection
        try:
            from sqlalchemy import text
            session = SessionLocal()
            session.execute(text("SELECT 1"))
            session.close()
            db_status = "connected"
        except Exception:
            db_status = "disconnected"
        
        return {
            "status": "healthy",
            "ids_running": ids_running,
            "database": db_status,
            "websocket": "active",
            "api_version": "1.0.0"
        }, 200

# Standalone Execution
if __name__ == "__main__":
    """
    Run API server in standalone mode for development/testing.
    For production, use backend/__init__.py or main entry point.
    """
    from flask import Flask
    from flask_cors import CORS
    from config import Config
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["JWT_SECRET_KEY"] = Config.JWT_SECRET_KEY
    
    # Enable CORS
    CORS(app, origins=["http://localhost:3000"])
    
    # Initialize API
    init_api(app)
    
    # Run server
    print("[INFO] Starting API server in standalone mode...")
    print(f"[INFO] Server running on http://localhost:5001")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=Config.DEBUG_MODE
    )