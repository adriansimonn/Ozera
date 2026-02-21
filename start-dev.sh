#!/bin/bash
# Development startup script for Ozera

set -e

echo "Starting Ozera Development Environment..."
echo ""

# Check for required environment variables
if [ ! -f "backend/.env" ]; then
    echo "Warning: backend/.env not found. Copy backend/.env.example to backend/.env and configure it."
fi

# Check if Modal is configured (for cloud training)
echo "[0/3] Checking Modal configuration..."
if command -v modal &> /dev/null; then
    echo "Modal CLI installed"
else
    echo "Warning: Modal CLI not installed. Cloud training will not work."
    echo "  Install with: pip install modal"
fi

# Start backend API
echo ""
echo "[1/3] Starting backend API server..."
./venv/bin/python backend/scripts/run_api.py --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to be ready..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "Backend API is running on http://localhost:8000"
else
    echo "Backend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend
echo ""
echo "[2/3] Starting frontend dev server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

echo ""
echo "[3/3] Cloud Training Status"
echo "  Modal app deployment: Run 'modal deploy backend/services/modal_worker.py' to deploy"
echo "  Required secrets: Set up 'ozera-secrets' in Modal dashboard with:"
echo "    - BACKEND_URL: Your backend URL (use ngrok for local dev)"
echo "    - MODAL_WEBHOOK_SECRET: Shared secret for webhook auth"

echo ""
echo "=========================================="
echo "Ozera Development Environment Ready!"
echo "=========================================="
echo ""
echo "Backend API:  http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo "Frontend:     http://localhost:5173"
echo ""
echo "For cloud training to work:"
echo "  1. Deploy Modal app: modal deploy services/modal_worker.py"
echo "  2. Expose backend: ngrok http 8000"
echo "  3. Update BACKEND_URL in Modal secrets"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Wait for user interrupt
wait
