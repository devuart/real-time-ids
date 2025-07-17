# Makefile for IDS Model Training

.PHONY: all chunked full mock clean

PYTHON := python
CHUNKED_SCRIPT := train_model_chunked.py
FULL_SCRIPT := train_model_full.py

all: help

help:
	@echo ""
	@echo "IDS Model Trainer - Makefile Options"
	@echo "-------------------------------------"
	@echo "make chunked     - Train using chunked loading"
	@echo "make full        - Train using full dataset"
	@echo "make mock        - Train using synthetic mock data"
	@echo "make clean       - Delete logs and model files"
	@echo ""

chunked:
	$(PYTHON) $(CHUNKED_SCRIPT)

full:
	$(PYTHON) $(FULL_SCRIPT)

mock:
	$(PYTHON) $(FULL_SCRIPT) --use-mock

clean:
	rm -rf logs/ models/*.pth __pycache__
