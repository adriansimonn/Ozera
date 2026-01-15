"""
Start the Ozera inference API server.

Usage:
    python scripts/run_api.py --host 0.0.0.0 --port 8000
"""

import argparse
import uvicorn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Run Ozera API server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    print(f"Starting Ozera API server on {args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
