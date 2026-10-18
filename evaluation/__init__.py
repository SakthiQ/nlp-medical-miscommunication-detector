"""Evaluation scripts and datasets. Run from the repo root, e.g. `python -m evaluation.ner_eval`."""

import sys
from pathlib import Path

# The backend package (`app`) lives in backend/; make it importable from the repo root.
_BACKEND = str(Path(__file__).resolve().parents[1] / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
