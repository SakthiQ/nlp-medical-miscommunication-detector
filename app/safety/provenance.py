"""Provenance / audit logging.

Per the project brief (Phase 6): "Log every validation event with a provenance record
(original text, entities, retrieved chunk IDs, LLM output, validation flags) for
clinician auditability."

This is a real, deliberate exception to the "nothing is stored" claim made elsewhere in
this project (see README section 3, Step 1, and section 12). It exists because an
explanation system that can be wrong needs a record someone can go back and check — that
is the whole point of clinician auditability. The trade-off is documented, not hidden:

- Logging is opt-in, off by default. Set `AUDIT_LOG_PATH` to enable it.
- Each line is one JSON record, appended, never overwritten.
- A logging failure must never break a patient-facing request: every write is
  best-effort, wrapped so an I/O error becomes a warning, not a 500.
- This is a local file for a prototype, not a production audit trail. A real deployment
  needs access control, retention limits, and encryption at rest for this file — none of
  which exist here. See README section 12 (Known limitations).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from app.models import Explanation, FlaggedTerm, RetrievedChunk
from app.safety.validator import ValidationResult

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "").strip() or None

_write_lock = Lock()


def record_provenance(
    *,
    original_text: str,
    terms: list[FlaggedTerm],
    retrieved: dict[str, list[RetrievedChunk]],
    explanation: Explanation,
    validation: ValidationResult,
    fallback_used: bool,
    generation_error: str | None,
    timings_ms: dict[str, float],
    path: str | None = AUDIT_LOG_PATH,
) -> str | None:
    """Append one provenance record. Returns the record id, or None if logging is off.

    Never raises: a write failure is logged as a warning and the caller proceeds as if
    logging were disabled, because an audit trail must not be able to take the product
    down.
    """
    if not path:
        return None

    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "original_text": original_text,
        "entities": [
            {
                "text": term.text,
                "label": term.label,
                "start": term.start,
                "end": term.end,
                "status": term.status,
                "kb_id": term.kb_id,
                "negated": term.negated,
            }
            for term in terms
        ],
        "retrieved_chunk_ids": sorted({chunk.kb_id for chunks in retrieved.values() for chunk in chunks}),
        "generator": explanation.generator,
        "llm_output": [
            {"text": s.text, "kb_ids": s.kb_ids, "kind": s.kind} for s in explanation.sentences
        ],
        "final_text": validation.final_text,
        "validation_passed": validation.passed,
        "validation_flags": validation.flags,
        "fallback_used": fallback_used,
        "generation_error": generation_error,
        "timings_ms": timings_ms,
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        target = Path(path)
        with _write_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except OSError as exc:
        logger.warning("Provenance logging failed, continuing without it: %s", exc)
        return None

    return record_id
