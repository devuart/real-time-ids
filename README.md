# Real-Time Intrusion Detection System (IDS)

A real-time network intrusion detection system that combines signature-based detection with machine learning for anomaly detection.

## Key Features

- 🕵️‍♂️ **Real-time packet sniffing** with deep packet inspection
- 🤖 **Hybrid detection** (signature + ML-based anomaly detection)
- 📊 **Web dashboard** with real-time WebSocket updates
- 🔐 **JWT authentication** for secure API access
- 📦 **Multiple database support** (SQLite/MySQL)
- 🚨 **Automated alerting** with detailed attack logging

## Project Structure

```
real-time-ids/
├── backend/
│   ├── ids_sniffer.py        # Packet capture & detection
│   ├── deep_learning.py      # Autoencoder anomaly detection
│   ├── database.py           # Attack log database handler
│   ├── api.py                # REST API (FastAPI/Flask)
│   └── auth.py               # JWT authentication
├── dashboard/
│   ├── app.py                # Dashboard server
│   ├── templates/
│   │   └── index.html        # Dashboard UI
│   └── static/
│       ├── styles.css        # Custom CSS
│       └── script.js         # WebSocket client
├── models/
│   ├── ids_model.pkl         # Trained ML model
│   └── autoencoder.h5        # Deep learning model
├── logs/
│   ├── ids_logs.db           # Attack database
│   └── alerts.log            # Text-based logs
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## Installation

### Prerequisites
- Python 3.8+
- libpcap (for packet capture)
- Root/admin privileges (for sniffing)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/devuart/real-time-ids.git
   cd real-time-ids
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML model (optional)**
   ```bash
   python backend/deep_learning.py
   ```

## Usage

### Start Services

1. **Launch the API server**
   ```bash
   python backend/api.py
   ```
   API runs on `http://127.0.0.1:5001`

2. **Start the dashboard**
   ```bash
   python dashboard/app.py
   ```
   Dashboard available at `http://127.0.0.1:5002`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/start-ids` | GET | Start intrusion detection |
| `/stop-ids` | GET | Stop detection |
| `/login` | POST | Authenticate (JWT) |
| `/logs` | GET | Get attack logs |
| `/stats` | GET | System statistics |

**Example Requests:**
```bash
# Start IDS
curl -X GET http://127.0.0.1:5001/start-ids

# Authenticate
curl -X POST http://127.0.0.1:5001/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password123"}'

# Get logs (with JWT)
curl -X GET http://127.0.0.1:5001/logs \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Configuration

Edit `config.py` to customize:
- Network interface for sniffing
- Detection sensitivity thresholds
- Database connection settings
- JWT secret key

## Future Improvements

- [ ] Add support for Elasticsearch logging
- [ ] Implement distributed detection
- [ ] Add more attack signatures
- [ ] Containerize with Docker
- [ ] CI/CD pipeline integration

## Troubleshooting

**Common Issues:**
1. **Permission denied for packet capture**
   - Run with sudo/Admin privileges
   - Or: `sudo setcap cap_net_raw,cap_net_admin+eip $(which python)`

2. **Missing dependencies**
   - On Ubuntu: `sudo apt install libpcap-dev`
   - On Windows: Install WinPcap/Npcap

## License

MIT License - See [LICENSE](LICENSE) for details.

## Author
 
GitHub: https://github.com/devuart