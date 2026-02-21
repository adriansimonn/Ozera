.PHONY: help install test clean format lint type-check

help:
	@echo "Ozera Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install backend dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Lint code with flake8"
	@echo "  make type-check   - Type check with mypy"
	@echo "  make check        - Run format, lint, and type-check"
	@echo ""
	@echo "Development:"
	@echo "  make verify       - Verify model configs work"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove build artifacts"

# Installation
install:
	@echo "Setting up Python virtual environment..."
	cd backend && python -m venv venv
	@echo "Installing dependencies..."
	cd backend && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt -r requirements-dev.txt
	@echo ""
	@echo " Installation complete!"
	@echo ""
	@echo "To activate the virtual environment:"
	@echo "  cd backend && source venv/bin/activate"
	@echo ""
	@echo "To verify setup:"
	@echo "  make verify"

# Testing
test:
	@echo "Running tests with coverage..."
	cd backend && . venv/bin/activate && pytest -v

# Code Quality
format:
	@echo "Formatting code..."
	cd backend && . venv/bin/activate && black . && isort .
	@echo " Code formatted"

lint:
	@echo "Linting code..."
	cd backend && . venv/bin/activate && flake8 .
	@echo " Linting passed"

type-check:
	@echo "Type checking..."
	cd backend && . venv/bin/activate && mypy .
	@echo " Type checking passed"

check: format lint type-check
	@echo " All checks passed"

# Development helpers
verify:
	@echo "Verifying model configurations..."
	cd backend && . venv/bin/activate && python core/transformer/config.py
	@echo ""
	@echo " Model configs verified"

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo " Cleanup complete"
