# IDS Backend - Development Makefile
# Combines ML training, testing, and development tools

# Configuration
PYTHON := python
BACKEND_DIR := backend
TEST_SUITE := backend/test_suite.py
VENV_DIR := venv
REQUIREMENTS := requirements.txt

# ML Training Script
TRAIN_SCRIPT := backend/train_model.py

# Windows-specific commands
ifeq ($(OS),Windows_NT)
	VENV_ACTIVATE := $(VENV_DIR)\Scripts\activate.bat
	RM := del /Q
	RMDIR := rmdir /S /Q
	MKDIR := mkdir
	TYPE := type
	COPY := copy
	SHELL := cmd.exe
else
	VENV_ACTIVATE := source $(VENV_DIR)/bin/activate
	RM := rm -f
	RMDIR := rm -rf
	MKDIR := mkdir -p
	TYPE := cat
	COPY := cp
endif

# Help & Information
.PHONY: help
help:
	@echo.
	@echo ============================================
	@echo IDS Backend - Makefile Commands
	@echo ============================================
	@echo.
	@echo ML MODEL TRAINING:
	@echo   make train          - Train model with default settings
	@echo   make train-mock     - Train using synthetic mock data
	@echo   make train-debug    - Quick training run for debugging
	@echo   make train-interactive - Launch interactive training mode
	@echo   make train-stability - Run stability test
	@echo   make train-progressive - Run progressive training pipeline
	@echo   make clean-models   - Delete training logs and model files
	@echo.
	@echo SETUP COMMANDS:
	@echo   make setup          - Initialize development environment
	@echo   make install        - Install/update dependencies
	@echo   make venv           - Create virtual environment
	@echo.
	@echo SERVER COMMANDS:
	@echo   make dev            - Start development server
	@echo   make run            - Start server (alias for dev)
	@echo   make debug          - Start server with debug logging
	@echo.
	@echo TESTING COMMANDS:
	@echo   make test           - Run all test suites
	@echo   make test-api       - Test REST API only
	@echo   make test-ws        - Test WebSocket only
	@echo   make test-db        - Test database only
	@echo.
	@echo DATABASE COMMANDS:
	@echo   make seed           - Seed database with test data (10 attacks)
	@echo   make seed-100       - Seed 100 test attacks
	@echo   make stats          - Show database statistics
	@echo   make recent         - Show recent attacks (20)
	@echo   make reset          - Reset database (requires DEBUG_MODE)
	@echo   make truncate       - Clear all data (requires DEBUG_MODE)
	@echo   make export         - Export database to JSON
	@echo   make import         - Import database from JSON
	@echo.
	@echo DEVELOPMENT TOOLS:
	@echo   make random         - Generate random attack
	@echo   make bulk           - Generate 10 random attacks
	@echo   make bulk-50        - Generate 50 random attacks
	@echo   make stress         - WebSocket stress test (10 broadcasts)
	@echo   make stress-20      - WebSocket stress test (20 broadcasts)
	@echo   make ws-test        - Test WebSocket events
	@echo.
	@echo UTILITY COMMANDS:
	@echo   make clean          - Remove logs and database
	@echo   make clean-all      - Remove logs, database, and cache
	@echo   make logs           - View recent log entries
	@echo   make info           - Show system information
	@echo.
	@echo QUALITY ASSURANCE:
	@echo   make lint           - Run code linter
	@echo   make format         - Format code with black
	@echo   make check          - Run all quality checks
	@echo.
	@echo ADVANCED COMMANDS:
	@echo   make backup         - Create database backup
	@echo   make restore        - Restore database from backup
	@echo   make monitor        - Monitor real-time logs
	@echo   make quickstart     - Clean, seed, and start server
	@echo   make restart        - Restart development server
	@echo.

# ML Model Training Commands
.PHONY: train train-mock train-debug train-interactive train-stability train-progressive clean-models

train:
	@echo Training model with default settings...
	$(PYTHON) $(TRAIN_SCRIPT)

train-mock:
	@echo Training model using synthetic mock data...
	$(PYTHON) $(TRAIN_SCRIPT) --use-mock

train-debug:
	@echo Running quick debug training...
	$(PYTHON) $(TRAIN_SCRIPT) --use-mock --epochs 10 --debug

train-interactive:
	@echo Launching interactive training mode...
	$(PYTHON) $(TRAIN_SCRIPT) --interactive

train-stability:
	@echo Running stability test...
	$(PYTHON) $(TRAIN_SCRIPT) --use-mock --epochs 10

train-progressive:
	@echo Running progressive training pipeline...
	@echo This will run: Stability -> Baseline -> Performance
	$(PYTHON) $(TRAIN_SCRIPT)

clean-models:
	@echo Cleaning ML training artifacts...
ifeq ($(OS),Windows_NT)
	@if exist logs $(RMDIR) logs
	@if exist figures $(RMDIR) figures
	@if exist tensorboard $(RMDIR) tensorboard
	@if exist checkpoints $(RMDIR) checkpoints
	@if exist artifacts $(RMDIR) artifacts
	@if exist models\*.pth $(RM) models\*.pth
	@if exist __pycache__ $(RMDIR) __pycache__
else
	$(RM) -r logs/
	$(RM) -r figures/
	$(RM) -r tensorboard/
	$(RM) -r checkpoints/
	$(RM) -r artifacts/
	$(RM) models/*.pth
	$(RM) -r __pycache__
endif
	@echo ML artifacts cleaned!

# Setup & Installation
.PHONY: setup install venv

setup: venv install
	@echo.
	@echo ============================================
	@echo Setup complete! Development environment ready.
	@echo ============================================
	@echo Run 'make dev' to start the server.
	@echo Run 'make test' to run test suite.
	@echo Run 'make train' to train the model.
	@echo Run 'make help' to see all commands.
	@echo ============================================

venv:
	@echo Creating virtual environment...
	$(PYTHON) -m venv $(VENV_DIR)
	@echo Virtual environment created.

install:
	@echo Installing dependencies...
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)
	@echo Dependencies installed.

# Server Commands
.PHONY: dev run debug

dev:
	@echo.
	@echo ======================================
	@echo Starting IDS Development Server
	@echo ======================================
	@echo REST API:  http://127.0.0.1:5003
	@echo WebSocket: ws://127.0.0.1:5003
	@echo ======================================
	@echo.
	$(PYTHON) -m $(BACKEND_DIR)

run: dev

debug:
	@echo Starting server with debug logging...
ifeq ($(OS),Windows_NT)
	@set DEBUG_MODE=True && $(PYTHON) -m $(BACKEND_DIR)
else
	DEBUG_MODE=True $(PYTHON) -m $(BACKEND_DIR)
endif

# Testing Commands
.PHONY: test test-api test-ws test-db

test:
	@echo.
	@echo ======================================
	@echo Running Test Suite
	@echo ======================================
	$(PYTHON) $(TEST_SUITE) test

test-api:
	@echo Testing REST API endpoints...
	$(PYTHON) $(TEST_SUITE) test-api

test-ws:
	@echo Testing WebSocket connectivity...
	$(PYTHON) $(TEST_SUITE) test-ws

test-db:
	@echo Testing database operations...
	$(PYTHON) $(TEST_SUITE) test-db

# Database Commands
.PHONY: seed seed-100 stats recent reset truncate export import

seed:
	@echo Seeding database with 10 test attacks...
	$(PYTHON) $(TEST_SUITE) seed 10

seed-100:
	@echo Seeding database with 100 test attacks...
	$(PYTHON) $(TEST_SUITE) seed 100

stats:
	@echo.
	@echo ======================================
	@echo Database Statistics
	@echo ======================================
	$(PYTHON) $(TEST_SUITE) stats

recent:
	@echo.
	@echo ======================================
	@echo Recent 20 Attacks
	@echo ======================================
	$(PYTHON) $(TEST_SUITE) recent 20

reset:
	@echo.
	@echo ======================================
	@echo WARNING: Database Reset
	@echo ======================================
	@echo This will DELETE ALL database records!
	@echo Requires DEBUG_MODE=True
	@echo.
	$(PYTHON) $(TEST_SUITE) reset

truncate:
	@echo.
	@echo ======================================
	@echo WARNING: Database Truncate
	@echo ======================================
	@echo This will DELETE ALL database records!
	@echo Requires DEBUG_MODE=True
	@echo.
	$(PYTHON) $(TEST_SUITE) truncate

export:
	@echo Exporting database to JSON...
	$(PYTHON) $(TEST_SUITE) export

import:
	@echo Importing database from JSON...
	$(PYTHON) $(TEST_SUITE) import

# Development Tools
.PHONY: random bulk bulk-50 stress stress-20 ws-test

random:
	@echo Generating random attack...
	$(PYTHON) $(TEST_SUITE) random

bulk:
	@echo Generating 10 random attacks...
	$(PYTHON) $(TEST_SUITE) bulk 10

bulk-50:
	@echo Generating 50 random attacks...
	$(PYTHON) $(TEST_SUITE) bulk 50

stress:
	@echo Running WebSocket stress test (10 broadcasts)...
	$(PYTHON) $(TEST_SUITE) stress 10

stress-20:
	@echo Running WebSocket stress test (20 broadcasts)...
	$(PYTHON) $(TEST_SUITE) stress 20

ws-test:
	@echo Testing all WebSocket event types...
	$(PYTHON) $(TEST_SUITE) ws-test

# Utility Commands
.PHONY: clean clean-all logs info

clean:
	@echo.
	@echo ======================================
	@echo Cleaning Database and Logs
	@echo ======================================
ifeq ($(OS),Windows_NT)
	@if exist $(BACKEND_DIR)\logs\ids_logs.db $(RM) $(BACKEND_DIR)\logs\ids_logs.db
	@if exist $(BACKEND_DIR)\logs\alerts.log $(RM) $(BACKEND_DIR)\logs\alerts.log*
	@if exist $(BACKEND_DIR)\logs\database_export.json $(RM) $(BACKEND_DIR)\logs\database_export.json
else
	$(RM) $(BACKEND_DIR)/logs/ids_logs.db
	$(RM) $(BACKEND_DIR)/logs/alerts.log*
	$(RM) $(BACKEND_DIR)/logs/database_export.json
endif
	@echo Clean complete!

clean-all: clean
	@echo Removing cache and temporary files...
ifeq ($(OS),Windows_NT)
	@if exist $(BACKEND_DIR)\__pycache__ $(RMDIR) $(BACKEND_DIR)\__pycache__
	@if exist .pytest_cache $(RMDIR) .pytest_cache
	@if exist .coverage $(RM) .coverage
else
	$(RMDIR) $(BACKEND_DIR)/__pycache__
	$(RMDIR) .pytest_cache
	$(RM) .coverage
endif
	@echo All clean!

logs:
	@echo.
	@echo ======================================
	@echo Recent Log Entries
	@echo ======================================
ifeq ($(OS),Windows_NT)
	@$(TYPE) $(BACKEND_DIR)\logs\alerts.log | more
else
	tail -n 50 $(BACKEND_DIR)/logs/alerts.log
endif

info:
	@echo.
	@echo ======================================
	@echo System Information
	@echo ======================================
	$(PYTHON) $(TEST_SUITE) info

# Quality Assurance
.PHONY: lint format check

lint:
	@echo Running code linter (flake8)...
	$(PYTHON) -m flake8 $(BACKEND_DIR) --max-line-length=120 --exclude=__pycache__,venv

format:
	@echo Formatting code with black...
	$(PYTHON) -m black $(BACKEND_DIR) --line-length=120

check: lint
	@echo.
	@echo ======================================
	@echo Quality Checks Complete
	@echo ======================================
	@echo All checks passed!

# Advanced Commands
.PHONY: backup restore monitor quickstart restart

backup:
	@echo Creating database backup...
ifeq ($(OS),Windows_NT)
	@if not exist $(BACKEND_DIR)\logs\backups $(MKDIR) $(BACKEND_DIR)\logs\backups
	@$(COPY) $(BACKEND_DIR)\logs\ids_logs.db $(BACKEND_DIR)\logs\backups\ids_logs_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.db
else
	$(MKDIR) $(BACKEND_DIR)/logs/backups
	$(COPY) $(BACKEND_DIR)/logs/ids_logs.db $(BACKEND_DIR)/logs/backups/ids_logs_backup_$(shell date +%Y%m%d_%H%M%S).db
endif
	@echo Backup created!

restore:
	@echo.
	@echo ======================================
	@echo Restore Database from Backup
	@echo ======================================
ifeq ($(OS),Windows_NT)
	@dir /B $(BACKEND_DIR)\logs\backups\*.db
	@echo.
	@set /p BACKUP_FILE="Enter backup filename: " && $(COPY) $(BACKEND_DIR)\logs\backups\%BACKUP_FILE% $(BACKEND_DIR)\logs\ids_logs.db
else
	@ls -1 $(BACKEND_DIR)/logs/backups/*.db
	@echo.
	@read -p "Enter backup filename: " BACKUP_FILE && $(COPY) $(BACKEND_DIR)/logs/backups/$$BACKUP_FILE $(BACKEND_DIR)/logs/ids_logs.db
endif
	@echo Database restored!

monitor:
	@echo.
	@echo ======================================
	@echo Monitoring Real-Time Logs
	@echo ======================================
	@echo Press Ctrl+C to stop
	@echo.
ifeq ($(OS),Windows_NT)
	@powershell -Command "Get-Content $(BACKEND_DIR)\logs\alerts.log -Wait"
else
	tail -f $(BACKEND_DIR)/logs/alerts.log
endif

quickstart: clean seed dev

restart:
	@echo Restarting development server...
ifeq ($(OS),Windows_NT)
	@taskkill /F /IM python.exe /T 2>nul || echo No Python process found
	@timeout /t 2 /nobreak >nul
else
	@pkill -f "python -m backend" || echo "No Python process found"
	@sleep 2
endif
	@$(MAKE) dev

# CI/CD Commands (for future use)
.PHONY: ci build deploy

ci: install test lint
	@echo.
	@echo ======================================
	@echo CI Pipeline Complete
	@echo ======================================

build:
	@echo Building production package...
	$(PYTHON) -m build

deploy:
	@echo.
	@echo ======================================
	@echo Production Deployment
	@echo ======================================
	@echo Production deployment requires manual configuration
	@echo Please review deployment checklist:
	@echo   1. Set DEBUG_MODE=False
	@echo   2. Update JWT_SECRET_KEY
	@echo   3. Configure production database
	@echo   4. Set CORS origins
	@echo   5. Enable HTTPS/WSS
	@echo ======================================

# Default Target
.DEFAULT_GOAL := help