#!/bin/bash

# train.sh - Unified training script for IDS
# Usage: ./train.sh [chunked|full|mock]

MODE=$1
PYTHON=python

if [[ -z "$MODE" ]]; then
  echo "Usage: ./train.sh [chunked|full|mock]"
  exit 1
fi

case "$MODE" in
  chunked)
    echo "[INFO] Training using chunked dataset loading..."
    $PYTHON train_model_chunked.py
    ;;
  full)
    echo "[INFO] Training using full dataset..."
    $PYTHON train_model_full.py
    ;;
  mock)
    echo "[INFO] Training using mock synthetic data..."
    $PYTHON train_model_full.py --use-mock
    ;;
  *)
    echo "[ERROR] Invalid mode: $MODE"
    echo "Valid options are: chunked, full, mock"
    exit 1
    ;;
esac
