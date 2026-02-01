"""
Pytest configuration: ensure src is on PYTHONPATH for all tests.
"""
import sys
from pathlib import Path

# Add src to path once at session start so all tests can import api, models, etc.
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
