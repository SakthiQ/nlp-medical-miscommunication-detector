"""Maintenance tools. Run from the repo root, e.g. `python -m backend.tools.kb_candidates`."""

import sys
from pathlib import Path

# Make `app` importable when run as `python -m backend.tools.<tool>` from the repo root.
_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
