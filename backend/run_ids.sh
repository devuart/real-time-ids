#!/bin/bash

# API Base URL
BASE_URL="http://localhost:5001"

# Admin Credentials
USERNAME="admin"
PASSWORD="password123"

# Obtain JWT Token
echo "[info] Logging in..."
TOKEN=$(curl -s -X POST "$BASE_URL/login" -H "Content-Type: application/json" -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}" | jq -r .access_token)

# Check if login was successful
if [[ "$TOKEN" == "null" || -z "$TOKEN" ]]; then
  echo "[errror] Login failed. Check credentials or API status."
  exit 1
fi
echo "[success] Authentication successful! Token received."

# Start IDS
echo "[info] Starting IDS..."
curl -s "$BASE_URL/start-ids" | jq

# Simulate Attack
ATTACK_TYPE="port_scan"
INTENSITY=5

echo "[info] Simulating attack: $ATTACK_TYPE (Intensity: $INTENSITY)..."
curl -s -X POST "$BASE_URL/simulate-attack" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"attack_type\": \"$ATTACK_TYPE\", \"intensity\": $INTENSITY}" | jq

echo "[success] All actions completed."
