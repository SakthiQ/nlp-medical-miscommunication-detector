"""Draft new knowledge-base entries from MedlinePlus Health Topics, for human review.

    python -m tools.kb_candidates               # build or refresh data/kb_candidates.json
    python -m tools.kb_candidates --download    # fetch the newest MedlinePlus XML first
    python -m tools.kb_candidates --promote     # copy approved candidates into the KB

Nothing here writes a definition. The KB's rule is that a human curates every
definition (data/README.md), so this tool only does the mechanical parts:

1. Collect the terms worth adding: every coverage gap in data/eval_corpus.json, plus
   data/kb_wishlist.txt.
2. Look each one up in the MedlinePlus Health Topics XML (public domain, downloaded to
   data/raw/, which is not committed) by topic title, "also called" and "see" names.
3. Write data/kb_candidates.json: per term, a draft entry with the citation, URL and
   suggested aliases filled in, an empty `plain_definition` and `label`, and the
   MedlinePlus text alongside as reference material.

A reviewer writes the definition, sets `"approved": true`, and runs `--promote`. Only
then does anything reach data/knowledge_base.json, and only after the combined KB passes
the same validation the app runs at startup (unique ids, unambiguous aliases).

Re-running keeps every candidate a reviewer has touched (approved, definition written,
notes added, or promoted) and refreshes only the untouched ones.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import (
    DEFAULT_KB_PATH,
    KBEntry,
    KnowledgeBase,
    KnowledgeBaseFile,
    load_knowledge_base,
    normalize,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
WISHLIST_PATH = ROOT / "data" / "kb_wishlist.txt"
CANDIDATES_PATH = ROOT / "data" / "kb_candidates.json"

XML_INDEX_URL = "https://medlineplus.gov/xml.html"
_XML_NAME_RE = re.compile(r"mplus_topics_(\d{4}-\d{2}-\d{2})\.xml")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

LABELS = {"DISEASE", "SYMPTOM", "TEST", "DRUG", "BIOMARKER", "ANATOMY", "DESCRIPTOR"}
SUMMARY_EXCERPT_CHARS = 700
MAX_SUGGESTIONS = 3


@dataclass
class Topic:
    title: str
    url: str
    meta_desc: str
    summary: str
    also_called: list[str] = field(default_factory=list)  # synonyms
    see_references: list[str] = field(default_factory=list)  # cross-links, often a different thing
    groups: list[str] = field(default_factory=list)
    mesh: list[str] = field(default_factory=list)
    primary_institute: str | None = None


# --- MedlinePlus XML ------------------------------------------------------------------


def download_latest(raw_dir: Path = RAW_DIR) -> Path:
    """Download the newest dated Health Topics XML listed on medlineplus.gov/xml.html."""
    request = urllib.request.Request(XML_INDEX_URL, headers={"User-Agent": "Mozilla/5.0"})
    index = urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")
    names = sorted(set(_XML_NAME_RE.findall(index)))
    if not names:
        raise RuntimeError(f"no mplus_topics_<date>.xml link found on {XML_INDEX_URL}")
    target = raw_dir / f"mplus_topics_{names[-1]}.xml"
    if not target.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://medlineplus.gov/xml/{target.name}"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        target.write_bytes(urllib.request.urlopen(request, timeout=300).read())
    return target


def newest_xml(raw_dir: Path = RAW_DIR) -> Path:
    files = sorted(raw_dir.glob("mplus_topics_*.xml"))
    if not files:
        raise FileNotFoundError(
            f"no mplus_topics_<date>.xml in {raw_dir}; run with --download first"
        )
    return files[-1]


def _plain(text: str | None) -> str:
    # full-summary holds escaped HTML: ElementTree unescapes it once into tags as text.
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


def parse_topics(path: Path) -> list[Topic]:
    """English health topics only. ElementTree does not fetch the external DTD."""
    topics = []
    for element in ET.parse(path).getroot().iter("health-topic"):
        if element.get("language") != "English":
            continue
        institute = element.find("primary-institute")
        topics.append(
            Topic(
                title=element.get("title", ""),
                url=element.get("url", ""),
                meta_desc=element.get("meta-desc", ""),
                summary=_plain(element.findtext("full-summary")),
                also_called=[e.text.strip() for e in element.iter("also-called") if e.text],
                see_references=[e.text.strip() for e in element.iter("see-reference") if e.text],
                groups=[e.text for e in element.iter("group") if e.text],
                mesh=[e.text for e in element.iter("descriptor") if e.text],
                primary_institute=institute.text if institute is not None else None,
            )
        )
    return topics


# --- Matching -------------------------------------------------------------------------


def wanted_terms(wishlist_path: Path = WISHLIST_PATH) -> dict[str, dict]:
    """normalized term -> {"term": surface, "requested_by": [...]}, gaps first."""
    wanted: dict[str, dict] = {}

    def add(term: str, reason: str) -> None:
        item = wanted.setdefault(normalize(term), {"term": term.strip(), "requested_by": []})
        if reason not in item["requested_by"]:
            item["requested_by"].append(reason)

    for snippet in load_corpus().snippets:
        for gold in snippet.gap_terms:
            add(gold.surface, f"corpus gap: {snippet.id}")
    if wishlist_path.exists():
        for line in wishlist_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                add(line, "wishlist")
    return wanted


def _index(topics: list[Topic], related: bool = False) -> dict[str, list[Topic]]:
    """Name -> topics. By title and "also called" names, or (related=True) by "see"
    cross-references, which point at a broader or different topic: "hypokalemia" is a
    see-reference on the Potassium topic, whose names include "Hyperkalemia"."""
    index: dict[str, list[Topic]] = {}
    for topic in topics:
        for name in topic.see_references if related else [topic.title, *topic.also_called]:
            bucket = index.setdefault(normalize(name), [])
            if topic not in bucket:
                bucket.append(topic)
    return index


def _lookup(index: dict[str, list[Topic]], key: str) -> list[Topic]:
    # MedlinePlus titles are often plural ("Urinary Tract Infections", "Kidney Cysts").
    return index.get(key) or index.get(key + "s") or index.get(key.removesuffix("s")) or []


def _partial(key: str, topics: list[Topic]) -> list[Topic]:
    """Topics with the term as a whole word or phrase inside one of their names."""
    pattern = re.compile(rf"\b{re.escape(key)}\b")
    hits = [t for t in topics if any(pattern.search(normalize(n)) for n in [t.title, *t.also_called])]
    return hits[:MAX_SUGGESTIONS]


def build_candidate(key: str, wanted: dict, kb: KnowledgeBase, index: dict, related_index: dict,
                    topics: list[Topic]) -> dict:
    candidate = {
        "wanted": wanted["term"],
        "requested_by": wanted["requested_by"],
        "status": "",
        "approved": False,
        "entry": None,
        "reference": None,
        "alias_conflicts": {},
        "alias_warnings": [],
        "suggestions": [],
        "notes": "",
    }
    existing = kb.lookup(wanted["term"])
    if existing is not None:
        candidate["status"] = f"already_in_kb: {existing.id}"
        return candidate

    matches, status = _lookup(index, key), "matched"
    if not matches:
        matches, status = _lookup(related_index, key), "related_topic"
    if not matches:
        partial = _partial(key, topics)
        candidate["status"] = "partial" if partial else "not_found"
        candidate["suggestions"] = [{"title": t.title, "url": t.url} for t in partial]
        return candidate

    topic, others = matches[0], matches[1:]
    # A related topic is about something broader (Sodium for hyponatremia), so its names
    # are not synonyms of the wanted term and are not offered as aliases.
    names = [wanted["term"], topic.title, *topic.also_called] if status == "matched" else [wanted["term"]]
    aliases, conflicts = [], {}
    for name in names:
        owner = kb.lookup(name)
        if owner is not None:
            conflicts[name] = owner.id  # would make the KB ambiguous; reviewer decides
        elif normalize(name) not in {normalize(a) for a in aliases}:
            aliases.append(name)

    candidate["status"] = status
    candidate["entry"] = {
        "term": wanted["term"],
        "aliases": aliases,
        "label": "",
        "plain_definition": "",
        "clinical_context": "",
        "source_citation": f"MedlinePlus Health Topics - '{topic.title}'",
        "source_url": topic.url,
        "is_common": False,
    }
    candidate["reference"] = {
        "meta_desc": topic.meta_desc,
        "summary_excerpt": topic.summary[:SUMMARY_EXCERPT_CHARS],
        "groups": topic.groups,
        "mesh": topic.mesh,
        "primary_institute": topic.primary_institute,
        "see_also_not_aliases": topic.see_references,
    }
    candidate["alias_conflicts"] = conflicts
    candidate["alias_warnings"] = [
        f"{a!r} is very short; it may match unrelated text" for a in aliases if len(normalize(a)) <= 2
    ]
    candidate["suggestions"] = [{"title": t.title, "url": t.url} for t in others[:MAX_SUGGESTIONS]]
    return candidate


def _touched(candidate: dict) -> bool:
    entry = candidate.get("entry") or {}
    return bool(
        candidate.get("approved")
        or candidate.get("promoted_as")
        or candidate.get("notes")
        or entry.get("plain_definition")
    )


def build_candidates(xml_path: Path, existing: dict | None = None, kb: KnowledgeBase | None = None,
                     wishlist_path: Path = WISHLIST_PATH) -> dict:
    kb = kb or load_knowledge_base()
    topics = parse_topics(xml_path)
    index, related_index = _index(topics), _index(topics, related=True)
    kept = {normalize(c["wanted"]): c for c in (existing or {}).get("candidates", []) if _touched(c)}

    candidates = []
    for key, wanted in wanted_terms(wishlist_path).items():
        candidates.append(kept.pop(key, None) or build_candidate(key, wanted, kb, index, related_index, topics))
    candidates.extend(kept.values())  # reviewed candidates whose term left the lists

    return {
        "source": f"MedlinePlus Health Topics XML ({xml_path.name}), public domain",
        "generated": date.today().isoformat(),
        "instructions": (
            "For each 'matched' or 'related_topic' candidate: check that the cited topic "
            "really covers the term ('related_topic' means it was only cross-referenced), "
            "remove any alias that is not the same thing as the term, write entry.plain_definition (1-2 plain sentences, "
            "own words, no interpretation) and entry.clinical_context, pick entry.label from "
            f"{sorted(LABELS)}, check the aliases, then set approved to true and run "
            "`python -m tools.kb_candidates --promote`."
        ),
        "candidates": candidates,
    }


# --- Promotion ------------------------------------------------------------------------


def _next_id(entries: list[KBEntry]) -> int:
    numbers = [int(m.group(1)) for e in entries if (m := re.fullmatch(r"kb-(\d+)", e.id))]
    return max(numbers, default=0) + 1


def promote(candidates: dict, kb_path: Path = DEFAULT_KB_PATH) -> list[str]:
    """Append approved, unpromoted candidates to the KB. Returns the new ids.

    Validates every candidate first and the combined KB second; nothing is written
    unless all of it passes.
    """
    kb_file = KnowledgeBaseFile.model_validate(json.loads(kb_path.read_text(encoding="utf-8")))
    entries = list(kb_file.entries)
    next_number = _next_id(entries)
    new_ids, promoted = [], []

    for candidate in candidates["candidates"]:
        if not candidate.get("approved") or candidate.get("promoted_as"):
            continue
        entry = candidate.get("entry") or {}
        missing = [f for f in ("plain_definition", "clinical_context", "source_citation") if not str(entry.get(f, "")).strip()]
        if entry.get("label") not in LABELS:
            missing.append(f"label (one of {sorted(LABELS)})")
        if missing:
            raise ValueError(f"{candidate['wanted']!r} is approved but missing: {', '.join(missing)}")
        new_id = f"kb-{next_number:03d}"
        next_number += 1
        entries.append(KBEntry(id=new_id, **entry))
        new_ids.append(new_id)
        promoted.append(candidate)

    if not new_ids:
        return []
    KnowledgeBase(entries=entries)  # raises KnowledgeBaseError on a duplicate id or ambiguous alias
    kb_file.entries = entries
    kb_path.write_text(dump_kb(kb_file.model_dump()), encoding="utf-8", newline="\n")
    for candidate, new_id in zip(promoted, new_ids):
        candidate["promoted_as"] = new_id
    return new_ids


def dump_kb(data: dict) -> str:
    """Serialize in the KB file's house style: 2-space indent, aliases on one line."""
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return re.sub(
        r'"aliases": \[\s*([^\]]*?)\s*\]',
        lambda m: '"aliases": [' + re.sub(r",\s+", ", ", m.group(1)) + "]",
        text,
    ) + "\n"


# --- CLI ------------------------------------------------------------------------------


def _summary(candidates: dict) -> str:
    counts: dict[str, int] = {}
    for c in candidates["candidates"]:
        status = c["status"].split(":")[0]
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--download", action="store_true", help="fetch the newest MedlinePlus XML into data/raw/ first")
    parser.add_argument("--promote", action="store_true", help="copy approved candidates into data/knowledge_base.json")
    args = parser.parse_args(argv)

    existing = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8")) if CANDIDATES_PATH.exists() else None

    if args.promote:
        if existing is None:
            sys.exit(f"{CANDIDATES_PATH} does not exist; build it first")
        new_ids = promote(existing)
        CANDIDATES_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        if new_ids:
            print(f"Added {len(new_ids)} entries: {', '.join(new_ids)}.")
            print("Next: add an eval-corpus snippet using each new term (data/README.md rule 4), "
                  "then run `python -m pytest tests/test_knowledge_base.py`.")
        else:
            print("Nothing to promote: no approved, unpromoted candidates.")
        return

    xml_path = download_latest() if args.download else newest_xml()
    candidates = build_candidates(xml_path, existing)
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {CANDIDATES_PATH.relative_to(ROOT)} from {xml_path.name}: {_summary(candidates)}.")


if __name__ == "__main__":
    main()
