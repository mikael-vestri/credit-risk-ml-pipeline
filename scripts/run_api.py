"""
Script to run the FastAPI application.

Usage:
    python scripts/run_api.py

The API will be available at http://localhost:8000
API documentation will be available at http://localhost:8000/docs
"""

import sys
from pathlib import Path

import uvicorn

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

if __name__ == "__main__":
    print("=" * 80)
    print("CREDIT RISK PREDICTION API")
    print("=" * 80)
    print("\nStarting FastAPI server...")
    print("\n📍 IMPORTANT: Use these URLs in your browser:")
    print("   • API Base:        http://localhost:8000")
    print("   • API Docs:         http://localhost:8000/docs")
    print("   • Alternative Docs: http://localhost:8000/redoc")
    print("   • Health Check:     http://localhost:8000/health")
    print("\n⚠️  Note: Do NOT use '0.0.0.0:8000' in browser - use 'localhost:8000' instead")
    print("\nPress CTRL+C to stop the server")
    print("=" * 80)

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
