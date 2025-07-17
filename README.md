Real-Time Intrusion Detection System (IDS)

Overview

This project is a real-time intrusion detection system (IDS) that captures network packets, detects anomalies using a machine learning model, and logs attacks into a database. It provides:

Real-time packet sniffing

Deep packet inspection for known attack patterns

Machine learning-based anomaly detection

A REST API for attack logs

A WebSocket-enabled dashboard for real-time alerts

Secure authentication using JWT

Project Structure

real-time-ids/
│── backend/
│   ├── ids_sniffer.py   # Real-time packet capture & detection
│   ├── deep_learning.py # Autoencoder-based anomaly detection
│   ├── database.py      # Logs attack records into SQLite/MySQL
│   ├── api.py           # REST API for alerts
│── dashboard/
│   ├── app.py          # Flask server for the UI
│   ├── templates/
│   │   ├── index.html  # IDS alert dashboard
│   ├── static/
│   │   ├── styles.css  # Custom styling
│── models/
│   ├── ids_model.pkl   # Trained ML model
│   ├── autoencoder.h5  # Deep learning model
│── logs/
│   ├── ids_logs.db     # SQLite database for attack logs
│   ├── alerts.log      # Log file for alerts
│── config.py           # Configuration settings
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

Installation

1. Clone the Repository

git clone <https://github.com/your-username/real-time-ids.git>
cd real-time-ids

2. Set Up a Virtual Environment (Optional)

python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Train the Deep Learning Model (Optional)

The IDS uses an autoencoder-based anomaly detection model. If you haven't trained the model, run:

python backend/deep_learning.py

This will generate models/autoencoder.h5.

5. Start the API Server

python backend/api.py

6. Start the Dashboard

python dashboard/app.py

Then open <http://127.0.0.1:5002/> in your browser.

Usage

Start IDS Sniffing

By default, the IDS starts sniffing packets when backend/ids_sniffer.py runs.

You can start IDS from the API endpoint:

curl -X GET <http://127.0.0.1:5001/start-ids>

Authenticate and Fetch Attack Logs

Login to get JWT token

curl -X POST <http://127.0.0.1:5001/login> -H "Content-Type: application/json" -d '{"username": "admin", "password": "password123"}'

Copy the access_token from the response.

Get attack logs

curl -X GET <http://127.0.0.1:5001/logs> -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

Monitor Logs via Web Dashboard

The web dashboard at <http://127.0.0.1:5002/> displays attack logs dynamically.

Future Improvements

Implement more sophisticated feature extraction for ML detection.

Support additional logging backends (e.g., Elasticsearch, Splunk).

Add more attack pattern rules in deep packet inspection.

Author

Your Name

GitHub: your-username

License

This project is licensed under the MIT License.
