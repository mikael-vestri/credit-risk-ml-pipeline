"""
Pytest configuration: ensure src is on PYTHONPATH for all tests.
"""
import sys
from pathlib import Path

# Add src to path once at session start so all tests can import api, models, etc.
_root = Path(__file__).resolve().parent.parent
_src = (_root / "src").resolve()
# Always insert at 0 so our src is found before any other 'models' etc.
sys.path.insert(0, str(_src))
