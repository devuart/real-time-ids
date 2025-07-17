import json
import csv
import os
from fpdf import FPDF
from flask import jsonify

# Default report directory
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def calculate_score(detected_attacks, actual_attacks, response_times, log_completeness):
    """Calculates the IDS performance score based on accuracy, response time, and log completeness."""
    if actual_attacks == 0:
        return jsonify({"error": "No attacks simulated, unable to calculate score."}), 400

    accuracy_score = (detected_attacks / actual_attacks) * 100 if actual_attacks else 0
    log_score = (log_completeness / actual_attacks) * 100 if actual_attacks else 0
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    # Weighted scoring system
    total_score = (accuracy_score * 0.5) + ((100 - avg_response_time) * 0.3) + (log_score * 0.2)
    return round(total_score, 2), accuracy_score, avg_response_time, log_score

def generate_report(results, format_type):
    """Generates a report in JSON, CSV, or PDF format."""
    filename = os.path.join(REPORT_DIR, f"performance_report.{format_type}")

    try:
        if format_type == "json":
            with open(filename, "w") as json_file:
                json.dump(results, json_file, indent=4)
        elif format_type == "csv":
            with open(filename, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Metric", "Value"])
                for key, value in results.items():
                    writer.writerow([key, value])
        elif format_type == "pdf":
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, "IDS Performance Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            for key, value in results.items():
                pdf.cell(0, 10, f"{key}: {value}", ln=True)
            pdf.output(filename)
        else:
            return jsonify({"error": f"Unsupported format: {format_type}. Use 'json', 'csv', or 'pdf'."}), 400
        
        return jsonify({"message": f"Report saved as {filename}", "file_path": filename})
    
    except Exception as e:
        return jsonify({"error": f"Error generating {format_type.upper()} report: {str(e)}"}), 500

def evaluate_performance():
    """API wrapper to evaluate IDS performance and return JSON response."""
    try:
        detected_attacks = 95  # Example: Retrieved from logs
        actual_attacks = 100    # Example: Simulated total
        log_completeness = 90   # Example: Logged attacks
        response_times = [0.5, 1.2, 0.8, 0.6, 1.0]  # Example: Retrieved from logs

        score, accuracy, avg_time, log_score = calculate_score(
            detected_attacks, actual_attacks, response_times, log_completeness
        )

        results = {
            "Total Score": score,
            "Detection Accuracy": f"{accuracy:.2f}%",
            "Average Response Time": f"{avg_time:.2f} sec",
            "Log Completeness": f"{log_score:.2f}%"
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    """Interactive CLI for generating performance reports manually."""
    print("Enter IDS Performance Data:")
    
    try:
        detected_attacks = int(input("Detected Attacks: ").strip())
        actual_attacks = int(input("Total Attacks Simulated: ").strip())
        log_completeness = int(input("Number of Logged Attacks: ").strip())

        response_times = []
        num_responses = int(input("Number of response times to enter: ").strip())

        for i in range(num_responses):
            response_time = float(input(f"Enter response time {i+1}: ").strip())
            response_times.append(response_time)

        score, accuracy, avg_time, log_score = calculate_score(
            detected_attacks, actual_attacks, response_times, log_completeness
        )

        results = {
            "Total Score": score,
            "Detection Accuracy": f"{accuracy:.2f}%",
            "Average Response Time": f"{avg_time:.2f} sec",
            "Log Completeness": f"{log_score:.2f}%"
        }

        print("\nGenerated Performance Results:")
        for key, value in results.items():
            print(f"{key}: {value}")

        format_type = input("\nSelect report format (json/csv/pdf): ").strip().lower()
        response = generate_report(results, format_type)
        print(response.get_json())  # Print response for debugging

    except ValueError:
        print("[error] Invalid input! Please enter numerical values.")
