"""Build the imported (unreviewed) knowledge-base tier from the NCI Thesaurus.

    python -m backend.tools.ncit_import --download   # fetch Thesaurus.FLAT.zip, then build
    python -m backend.tools.ncit_import              # build from the copy in curation/raw/ncit/

Writes knowledge_base/imported/ncit.json.gz: one entry per NCIt concept that a patient
report could plausibly contain (conditions, findings, symptoms, tests, results, anatomy),
in the same shape as knowledge_base.json entries plus `"tier": "imported"`.

This tier is NOT reviewed. NCIt definitions are written for clinicians and researchers,
not patients, and nobody has checked them for this use. The curated knowledge_base.json
stays the reviewed tier and always wins: a concept whose name or alias it already covers
is skipped. Nothing reads this file yet; wiring it into the pipeline (with the answer
marked as unreviewed) is a separate step.

License: the NCI Thesaurus is released under CC BY 4.0
(https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.htm). Attribution and a
description of the changes made here are written into the output file and
knowledge_base/imported/README.md, as the license requires.

What the filters remove, and why:
- retired, obsolete, header and provisional concepts, and concepts with no definition
- semantic types outside report vocabulary: genes, proteins, chemicals, organisms,
  research and administrative concepts; drugs too unless --include-drugs (explaining
  medicines sits next to treatment advice, which the validator forbids)
- research grading and staging vocabulary (CTCAE grades, AJCC/FIGO stages, protein and
  gene variants, questionnaire items): not report language, and CTCAE definitions read
  as treatment instructions ("Moderate symptoms; corticosteroids indicated")
- definitions that trip the safety validator's forbidden patterns (app.safety.validator),
  since the fallback template shows definitions verbatim; and definitions over
  MAX_DEFINITION_CHARS, which are almost always multi-part research criteria
- aliases that are too short, are everyday words (knowledge_base/common_words.txt), are
  shared by more than one imported concept, or belong to the reviewed KB
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.retrieval.knowledge_base import DEFAULT_KB_PATH, KBEntry, load_knowledge_base, normalize
from app.safety.validator import FORBIDDEN_PATTERNS

from tools.kb_candidates import _get

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "knowledge_base" / "curation" / "raw" / "ncit"
FLAT_ZIP = RAW_DIR / "Thesaurus.FLAT.zip"
COMMON_WORDS_PATH = ROOT / "knowledge_base" / "common_words.txt"
OUTPUT_PATH = ROOT / "knowledge_base" / "imported" / "ncit.json.gz"

FLAT_URL = "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Thesaurus.FLAT.zip"
README_URL = "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ReadMe.txt"
LICENSE_URL = "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.htm"
CONCEPT_URL = "https://evsexplore.semantics.cancer.gov/evsexplore/concept/ncit/{code}"

# NCIt semantic type -> KB label. Checked in this order, so a concept that is both a
# Finding and a Laboratory or Test Result is labeled BIOMARKER.
SEMANTIC_TYPES = {
    "Sign or Symptom": "SYMPTOM",
    "Laboratory or Test Result": "BIOMARKER",
    "Laboratory Procedure": "TEST",
    "Diagnostic Procedure": "TEST",
    "Disease or Syndrome": "DISEASE",
    "Neoplastic Process": "DISEASE",
    "Pathologic Function": "DISEASE",
    "Injury or Poisoning": "DISEASE",
    "Congenital Abnormality": "DISEASE",
    "Acquired Abnormality": "DISEASE",
    "Anatomical Abnormality": "DISEASE",
    "Mental or Behavioral Dysfunction": "DISEASE",
    "Finding": "DISEASE",
    "Anatomical Structure": "ANATOMY",
    "Body Part, Organ, or Organ Component": "ANATOMY",
    "Body Location or Region": "ANATOMY",
    "Body Space or Junction": "ANATOMY",
    "Body System": "ANATOMY",
    "Tissue": "ANATOMY",
}
DRUG_TYPE = "Pharmacologic Substance"

# Research grading/staging vocabulary, coding variants and study documents, matched on the
# preferred name. Coding variants ("Felty's Syndrome, Right Elbow", "... pTis TNM Finding
# v6 and v7") exist for billing and registries; a report says the plain condition.
EXCLUDED_NAME_RE = re.compile(
    r"\bCTCAE\b|\bAJCC\b|\bFIGO\b|\bTNM\b|\bv\d+(\.\d+)? Stage\b|\bStage [0-9IVX]+[A-C]?\d?\b|"
    r"\bProtein Variant\b|\bGene (Mutation|Variant|Amplification)\b|\bGrade \d\b|"
    r"\bFusion (Positive|Negative)\b|\bMutation\b|"
    r"\bQuestion\b|\bQuestionnaire\b|\bScore\b|\bScale\b|\bItem\b|"
    r"\bTrial\b|\bTMF\b|\bSection\b|\bCase Report Form\b|"
    r", (Right|Left|Bilateral)\b|\bUnspecified\b",
    re.I,
)
# Definitions of ICD-style coding concepts ("Evidence of X in the right elbow.") and of
# staging categories copied from the AJCC manual.
EXCLUDED_DEFINITION_RE = re.compile(r"^Evidence of\b|\(from AJCC\b", re.I)
# A one-word alias is kept only if it is the concept's own preferred name ("Embolism") or
# an abbreviation ("DVT"). Other one-word synonyms are mostly data-coding values: NCIt
# lists "complete" for Complete Pharyngeal Contraction and "history" for Personal Medical
# History, which would match ordinary words in every report.
ABBREVIATION_RE = re.compile(r"[A-Z][A-Z0-9]{1,4}")
STATUS_EXCLUDED = {"Retired_Concept", "Obsolete_Concept", "Header_Concept", "Provisional_Concept"}
MAX_DEFINITION_CHARS = 600
MIN_ALIAS_CHARS = 3


@dataclass
class Concept:
    code: str
    names: list[str]          # preferred name first, then synonyms
    definition: str
    semantic_types: list[str]
    status: set[str]


def download(raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "ReadMe.txt").write_bytes(_get(README_URL))
    (raw_dir / "ThesaurusTermsofUse.htm").write_bytes(_get(LICENSE_URL))
    target = raw_dir / FLAT_ZIP.name
    target.write_bytes(_get(FLAT_URL, timeout=600))
    return target


def read_version(raw_dir: Path = RAW_DIR) -> str:
    readme = raw_dir / "ReadMe.txt"
    if readme.exists():
        first = readme.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if first and re.fullmatch(r"\d{2}\.\d{2}[a-z]", first[0].strip()):
            return first[0].strip()
    return "unknown"


def parse_flat(lines) -> list[Concept]:
    """Rows: code, IRI, parents, synonyms, definition, display name, status, semantic types, subsets."""
    concepts = []
    for line in lines:
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 8:
            continue
        names = [n.strip() for n in fields[3].split("|") if n.strip()]
        if not names:
            continue
        concepts.append(Concept(
            code=fields[0],
            names=list(dict.fromkeys(names)),
            definition=re.sub(r"\s+", " ", fields[4]).strip(),
            semantic_types=[t for t in fields[7].split("|") if t],
            status={s for s in fields[6].split("|") if s},
        ))
    return concepts


def read_flat_zip(path: Path = FLAT_ZIP) -> list[Concept]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("Thesaurus.txt") as raw:
            return parse_flat(io.TextIOWrapper(raw, encoding="utf-8"))


def _label(concept: Concept, include_drugs: bool) -> str | None:
    for semantic_type, label in SEMANTIC_TYPES.items():
        if semantic_type in concept.semantic_types:
            return label
    if include_drugs and DRUG_TYPE in concept.semantic_types:
        return "DRUG"
    return None


def _clean_definition(text: str) -> str:
    # NCIt appends source tags like "(NCI)" to some definitions.
    return re.sub(r"\s*\((NCI|NCI-GLOSS|CDISC|FDA)\)\s*$", "", text).strip()


def build(concepts: list[Concept], reviewed_kb=None, common_words: set[str] | None = None,
          include_drugs: bool = False) -> tuple[list[dict], Counter]:
    """Return (entries, drop counts). Entries match KBEntry plus `tier` and `ncit_code`."""
    reviewed_kb = reviewed_kb if reviewed_kb is not None else load_knowledge_base()
    common_words = common_words or set()
    dropped: Counter = Counter()

    kept: list[tuple[Concept, str, str]] = []
    for concept in concepts:
        if concept.status & STATUS_EXCLUDED:
            dropped["inactive concept"] += 1
            continue
        if not concept.definition:
            dropped["no definition"] += 1
            continue
        label = _label(concept, include_drugs)
        if label is None:
            dropped["semantic type outside report vocabulary"] += 1
            continue
        if EXCLUDED_NAME_RE.search(concept.names[0]):
            dropped["grading, staging or questionnaire item"] += 1
            continue
        definition = _clean_definition(concept.definition)
        if EXCLUDED_DEFINITION_RE.search(definition):
            dropped["coding or staging concept"] += 1
            continue
        if len(definition) > MAX_DEFINITION_CHARS:
            dropped["definition too long"] += 1
            continue
        if any(p.search(definition) for p in FORBIDDEN_PATTERNS.values()):
            dropped["definition trips a validator pattern"] += 1
            continue
        if any(reviewed_kb.lookup(name) for name in concept.names):
            dropped["already covered by the reviewed KB"] += 1
            continue
        kept.append((concept, label, definition))

    # An alias shared by two imported concepts would make dictionary matching ambiguous.
    owners: dict[str, set[str]] = defaultdict(set)
    for concept, _, _ in kept:
        for name in concept.names:
            owners[normalize(name)].add(concept.code)

    entries = []
    for concept, label, definition in kept:
        aliases = []
        preferred = normalize(concept.names[0])
        for name in concept.names:
            key = normalize(name)
            if (len(key) < MIN_ALIAS_CHARS or key.isdigit() or key in common_words
                    or len(owners[key]) > 1 or key in {normalize(a) for a in aliases}):
                continue
            if " " not in key and key != preferred and not ABBREVIATION_RE.fullmatch(name):
                continue  # one-word synonym that is not the concept's own name
            aliases.append(name)
        if not aliases:
            dropped["no unambiguous name left"] += 1
            continue
        entry = {
            "id": f"ncit-{concept.code}",
            "term": aliases[0],
            "aliases": aliases,
            "label": label,
            "plain_definition": definition,
            "clinical_context": "",
            "source_citation": f"NCI Thesaurus (CC BY 4.0) - '{concept.names[0]}' ({concept.code})",
            "source_url": CONCEPT_URL.format(code=concept.code),
            "is_common": False,
            "tier": "imported",
            "ncit_code": concept.code,
        }
        KBEntry.model_validate(entry)
        entries.append(entry)
    return entries, dropped


def write_output(entries: list[dict], dropped: Counter, version: str, path: Path = OUTPUT_PATH) -> None:
    labels = Counter(e["label"] for e in entries)
    document = {
        "schema_version": "1.0",
        "tier": "imported",
        "reviewed": False,
        "source": f"NCI Thesaurus {version}, flat file ({FLAT_URL})",
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "attribution": ("Contains definitions and synonyms from the NCI Thesaurus, produced by the "
                        "Enterprise Vocabulary Services group, National Cancer Institute. "
                        f"Terms of use: {LICENSE_URL}"),
        "changes": ("Filtered to report-relevant semantic types; research grading and staging "
                    "concepts removed; definitions that are long or match the safety validator's "
                    "forbidden patterns removed; '(NCI)'-style source tags stripped; ambiguous, "
                    "short and everyday-word synonyms removed; concepts already in the reviewed "
                    "knowledge base skipped. Definitions are otherwise unchanged and unreviewed."),
        "generated": date.today().isoformat(),
        "counts": {"entries": len(entries), "by_label": dict(sorted(labels.items())),
                   "dropped": dict(dropped.most_common())},
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # mtime=0 keeps the gzip bytes identical between runs on the same input.
    with gzip.GzipFile(path, "wb", mtime=0) as out:
        out.write(data)


def load_imported(path: Path = OUTPUT_PATH) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _common_words(path: Path = COMMON_WORDS_PATH) -> set[str]:
    if not path.exists():
        return set()
    return {normalize(w) for w in path.read_text(encoding="utf-8").splitlines()
            if w.strip() and not w.startswith("#")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--download", action="store_true", help="fetch the current NCIt flat file first")
    parser.add_argument("--include-drugs", action="store_true", help="also import Pharmacologic Substance concepts")
    args = parser.parse_args(argv)

    if args.download:
        download()
    if not FLAT_ZIP.exists():
        raise SystemExit(f"{FLAT_ZIP} not found; run with --download")
    entries, dropped = build(read_flat_zip(), load_knowledge_base(DEFAULT_KB_PATH), _common_words(),
                             include_drugs=args.include_drugs)
    version = read_version()
    write_output(entries, dropped, version)
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"NCIt {version}: wrote {len(entries)} entries to {OUTPUT_PATH.relative_to(ROOT)} ({size_mb:.1f} MB)")
    print("by label:", dict(sorted(Counter(e['label'] for e in entries).items())))
    print("dropped:", dict(dropped.most_common()))


if __name__ == "__main__":
    main()
