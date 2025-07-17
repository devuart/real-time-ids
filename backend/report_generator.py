import json
import csv
import os
from fpdf import FPDF

LOG_FILE = "logs/ids_logs.json"  # Log file location

def load_logs(log_file=LOG_FILE):
    """Loads logs from a JSON file."""
    try:
        with open(log_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("[errror] Log file not found.")
        return []
    except json.JSONDecodeError:
        print("[errror] Error decoding JSON.")
        return []

def confirm_overwrite(filename):
    """Checks if a file exists and prompts for overwrite confirmation."""
    if os.path.exists(filename):
        overwrite = input(f"[⚠️] {filename} already exists. Overwrite? (y/n): ").strip().lower()
        return overwrite == 'y'
    return True

def export_to_json(logs, output_file="ids_report.json"):
    """Exports logs to a JSON file."""
    if confirm_overwrite(output_file):
        try:
            with open(output_file, "w") as file:
                json.dump(logs, file, indent=4)
            print(f"[success] Report saved as {output_file}")
        except Exception as e:
            print(f"[errror] Error saving JSON report: {e}")

def export_to_csv(logs, output_file="ids_report.csv"):
    """Exports logs to a CSV file."""
    if not logs:
        print("[errror] No logs to export.")
        return
    
    if confirm_overwrite(output_file):
        try:
            keys = logs[0].keys()
            with open(output_file, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(logs)
            print(f"[success] Report saved as {output_file}")
        except Exception as e:
            print(f"[errror] Error saving CSV report: {e}")

def export_to_pdf(logs, output_file="ids_report.pdf"):
    """Exports logs to a PDF file with improved formatting."""
    if not logs:
        print("[errror] No logs to export.")
        return
    
    if confirm_overwrite(output_file):
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, "Intrusion Detection System Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Detected Events:", ln=True)
            pdf.ln(5)
            pdf.set_font("Arial", size=10)

            for log in logs:
                pdf.multi_cell(0, 7, f"[{log['timestamp']}] {log['event']} (Source: {log['source']})", border=0)
                pdf.ln(2)
            
            pdf.output(output_file)
            print(f"[success] Report saved as {output_file}")
        except Exception as e:
            print(f"[errror] Error saving PDF report: {e}")

def generate_reports():
    """Allows users to select a report format and generate the report."""
    logs = load_logs()
    if not logs:
        print("[errror] No logs available for report generation.")
        return

    print("\nSelect Report Format:")
    print("[1] JSON")
    print("[2] CSV")
    print("[3] PDF")
    print("[4] Generate All Formats")
    try:
        choice = int(input("Enter your choice: ").strip())

        if choice == 1:
            export_to_json(logs)
        elif choice == 2:
            export_to_csv(logs)
        elif choice == 3:
            export_to_pdf(logs)
        elif choice == 4:
            export_to_json(logs)
            export_to_csv(logs)
            export_to_pdf(logs)
        else:
            print("[errror] Invalid choice. Please enter a number between 1-4.")
    except ValueError:
        print("[errror] Invalid input! Please enter a number.")

if __name__ == "__main__":
    generate_reports()
