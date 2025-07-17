import os
import threading
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_restful import Api, Resource
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Import backend modules
from backend.database import SessionLocal, AttackLog
from backend import ids_sniffer, attack_simulation, performance_grader, report_generator, websocket_handler

# Flask Setup
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET", "super-secret-key")
jwt = JWTManager(app)
api = Api(app)
CORS(app, origins=["http://localhost:3000", "https://your-production-domain.com"])

# User Authentication (temporary in-memory storage)
users = {"admin": generate_password_hash("password123")}

class Login(Resource):
    """ User login to obtain JWT token """
    def post(self):
        data = request.get_json()
        username, password = data.get("username"), data.get("password")

        if username in users and check_password_hash(users[username], password):
            token = create_access_token(identity=username)
            return jsonify({"access_token": token})

        return jsonify({"error": "Invalid credentials"}), 401

class AttackLogs(Resource):
    """ Fetch attack logs with JWT authentication """
    @jwt_required()
    def get(self):
        session = SessionLocal()
        try:
            logs = session.query(AttackLog).all()
            return jsonify([{
                "timestamp": log.timestamp.isoformat(),
                "source_ip": log.source_ip,
                "destination_ip": log.destination_ip,
                "protocol": log.protocol,
                "attack_type": log.attack_type
            } for log in logs])
        except Exception as e:
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            session.close()

# Global IDS control to prevent multiple instances
ids_running = False  

class StartIDS(Resource):
    """ Start real-time Intrusion Detection """
    def get(self):
        global ids_running
        if not ids_running:
            threading.Thread(target=ids_sniffer.start_sniffing, daemon=True).start()
            ids_running = True
            websocket_handler.broadcast_message("[success] IDS Started!")
            return jsonify({"message": "IDS Started!"})

        return jsonify({"message": "[warning] IDS is already running!"})

class StopIDS(Resource):
    """ Stop the IDS (Placeholder for future implementation) """
    def get(self):
        global ids_running
        if ids_running:
            ids_running = False
            websocket_handler.broadcast_message("[warning] IDS Stopped!")
            return jsonify({"message": "IDS Stopped!"})

        return jsonify({"message": "[warning] IDS is not running!"})

# Define attack types & validation
VALID_ATTACKS = ["port_scan", "dos", "sql_injection"]

class SimulateAttack(Resource):
    """ Trigger an attack simulation """
    @jwt_required()
    def post(self):
        data = request.get_json()
        attack_type = data.get("attack_type", "port_scan")
        intensity = data.get("intensity", 5)
        target_ip = data.get("target_ip", "127.0.0.1")

        if attack_type not in VALID_ATTACKS:
            return jsonify({"error": f"Invalid attack type. Choose from {VALID_ATTACKS}"}), 400

        if not (1 <= intensity <= 10):
            return jsonify({"error": "Intensity must be between 1 and 10"}), 400

        try:
            result = attack_simulation.run_attack(attack_type, target_ip, {"intensity": intensity})
            websocket_handler.broadcast_message(f"[info] Simulated {attack_type} attack on {target_ip} (Intensity {intensity})")
            return jsonify({"message": "Attack simulation started", "details": result})
        except Exception as e:
            return jsonify({"error": f"Attack simulation error: {str(e)}"}), 500

class GradeIDS(Resource):
    """ Evaluate IDS performance """
    @jwt_required()
    def get(self):
        try:
            score_data = performance_grader.evaluate_performance()
            websocket_handler.broadcast_message(f"[info] IDS Performance Score: {score_data.get_json()}")
            return score_data
        except Exception as e:
            return jsonify({"error": f"Performance grading error: {str(e)}"}), 500

class GenerateReport(Resource):
    """ Generate IDS security report """
    @jwt_required()
    def get(self):
        try:
            format_type = request.args.get("format", "json").lower()
            response = report_generator.generate_report(format_type)
            websocket_handler.broadcast_message(f"[info] Report Generated: {response.get_json()}")
            return response
        except Exception as e:
            return jsonify({"error": f"Report generation error: {str(e)}"}), 500

# Register API Endpoints
api.add_resource(Login, "/login")
api.add_resource(AttackLogs, "/logs")
api.add_resource(StartIDS, "/start-ids")
api.add_resource(StopIDS, "/stop-ids")
api.add_resource(SimulateAttack, "/simulate-attack")
api.add_resource(GradeIDS, "/grade-ids")
api.add_resource(GenerateReport, "/generate-report")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
