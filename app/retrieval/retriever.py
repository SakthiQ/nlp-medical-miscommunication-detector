"""Slide stage 4 — retrieval (the RAG evidence layer).

Every explainable KB entry is embedded with a sentence-transformer and stored in a FAISS
inner-product index over normalised vectors, so scores are cosine similarities.
Commonly understood entries (fever, headache) are not indexed.

What retrieval may do was decided by measurement (see eval/retrieval_eval.py):
- Paraphrases find the right entry well: 12/12 lay-language queries have the correct
  entry in their top 3.
- But unrelated terms can outscore genuine paraphrases. "renal" scores 0.634 against
  hydronephrosis and "Urinalysis" 0.539, while the correct match for "reduced kidney
  filtering" scores 0.412. Embedding similarity measures relatedness, not equivalence:
  "renal" is about the kidney, but it is not hydronephrosis. No threshold separates the
  two, so a semantic match is never used to explain a term to a patient.

Hence two entry points:
- `for_term` returns the evidence an explanation may use: only the KB entry the term is
  linked to by exact dictionary match.
- `suggest` returns nearest entries for unvetted terms, for a human curator to confirm
  (by adding an alias) or reject. It feeds the curation queue, not the patient.
"""

from __future__ import annotations

import os

from app.models import FlaggedTerm, RetrievedChunk
from app.retrieval.knowledge_base import KBEntry, KnowledgeBase

IMPLEMENTATION = "real"

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def entry_text(entry: KBEntry) -> str:
    """The text embedded for an entry. Including the definition measured best
    (Recall@3 12/12 vs 11/12 for term and aliases alone)."""
    return f"{entry.term}. Also called: {', '.join(entry.aliases)}. {entry.plain_definition}"


def _chunk(entry: KBEntry, score: float, match: str) -> RetrievedChunk:
    return RetrievedChunk(
        kb_id=entry.id,
        term=entry.term,
        plain_definition=entry.plain_definition,
        clinical_context=entry.clinical_context,
        source_citation=entry.source_citation,
        source_url=entry.source_url,
        score=round(score, 4),
        match=match,
    )


class Retriever:
    def __init__(self, kb: KnowledgeBase, model_name: str = MODEL_NAME) -> None:
        import faiss  # heavy imports; only paid when the retriever is built
        from sentence_transformers import SentenceTransformer

        self.kb = kb
        self.model_name = model_name
        self._entries = kb.explainable
        # Local cache first: loading online checks Hugging Face for newer files on every
        # start, which measured 6.8 s against 0.7 s from the cache. Online only if the
        # model isn't cached yet.
        try:
            self._model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except (OSError, ValueError):
            self._model = SentenceTransformer(model_name, device="cpu")
        vectors = self._encode([entry_text(entry) for entry in self._entries])
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def _encode(self, texts: list[str]):
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

    def embed(self, texts: list[str]):
        """Normalised embeddings for arbitrary text, for callers outside retrieval proper
        (currently: the semantic grounding advisory check in app/safety/semantic.py)."""
        return self._encode(texts)

    def for_term(self, term: FlaggedTerm) -> list[RetrievedChunk]:
        """Evidence an explanation may use: only the KB entry the term is linked to."""
        if term.kb_id is None:
            return []
        entry = self.kb.get(term.kb_id)
        return [_chunk(entry, 1.0, "exact")] if entry is not None else []

    def suggest(self, texts: list[str], k: int = 3) -> list[list[RetrievedChunk]]:
        """Nearest KB entries for each text, best first. For curators, never patients."""
        texts = [text for text in texts if text.strip()]
        if not texts:
            return []
        k = min(k, len(self._entries))
        scores, ids = self._index.search(self._encode(texts), k)
        return [
            [_chunk(self._entries[i], float(score), "semantic") for score, i in zip(row_scores, row_ids) if i >= 0]
            for row_scores, row_ids in zip(scores, ids)
        ]

    def search(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        results = self.suggest([query], k)
        return results[0] if results else []
