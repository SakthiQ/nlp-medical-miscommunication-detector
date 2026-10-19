"""tools/ncit_import.py: building the imported (unreviewed) KB tier from the NCI Thesaurus.

Uses hand-written rows in the NCIt flat-file format, not the 90 MB download, so these
tests run offline.
"""

from __future__ import annotations

import gzip
import json

import pytest

from app.retrieval.knowledge_base import KBEntry, load_knowledge_base
from tools.kb_candidates import dump_kb
from tools.ncit_import import build, load_imported, parse_flat, write_output

# code, IRI, parents, synonyms, definition, display name, status, semantic types, subsets
ROWS = [
    ("C1", "", "", "Pneumothorax|Collapsed lung, air", "An abnormal collection of air in the pleural space. (NCI)", "", "", "Disease or Syndrome", ""),
    ("C2", "", "", "Bronchiectasis", "Permanent widening of the bronchi.", "", "Retired_Concept", "Disease or Syndrome", ""),
    ("C3", "", "", "Kinase Gene", "A gene.", "", "", "Gene or Genome", ""),
    ("C4", "", "", "Grade 2 Rash, CTCAE", "Moderate symptoms; corticosteroids indicated.", "", "", "Finding", ""),
    ("C5", "", "", "Felty's Syndrome, Right Elbow", "Evidence of Felty's syndrome in the right elbow.", "", "", "Disease or Syndrome", ""),
    ("C6", "", "", "Hemangioma", "", "", "", "Neoplastic Process", ""),
    ("C7", "", "", "Bad Advice Finding", "You should stop taking the medicine.", "", "", "Finding", ""),
    ("C8", "", "", "Complete Pharyngeal Contraction|complete", "A finding of complete pharyngeal contraction.", "", "", "Finding", ""),
    ("C9", "", "", "Embolism", "The blockage of a blood vessel.", "", "", "Disease or Syndrome", ""),
    ("C10", "", "", "Deep Vein Clot|DVC|Shared Name", "A clot in a deep vein.", "", "", "Disease or Syndrome", ""),
    ("C11", "", "", "Other Clot|Shared Name", "A clot elsewhere.", "", "", "Disease or Syndrome", ""),
    ("C12", "", "", "Steatosis of Liver|fatty liver", "Fat in the liver.", "", "", "Disease or Syndrome", ""),
    ("C13", "", "", "Aspirin", "A pain medicine.", "", "", "Pharmacologic Substance", ""),
    ("C14", "", "", "Ferritin Measurement", "The determination of the amount of ferritin in a sample.", "", "", "Laboratory Procedure", ""),
]
LINES = ["\t".join(row) + "\n" for row in ROWS]

SMALL_KB = {
    "schema_version": "1.0",
    "entries": [
        {"id": "kb-001", "term": "hepatic steatosis", "aliases": ["fatty liver", "hepatic steatosis"],
         "label": "DISEASE", "plain_definition": "Extra fat has built up inside the liver.",
         "clinical_context": "", "source_citation": "s", "source_url": None, "is_common": False},
    ],
}


@pytest.fixture
def reviewed_kb(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(dump_kb(SMALL_KB), encoding="utf-8")
    return load_knowledge_base(path)


@pytest.fixture
def built(reviewed_kb):
    entries, dropped = build(parse_flat(LINES), reviewed_kb, common_words={"history"})
    return {e["ncit_code"]: e for e in entries}, dropped


def test_keeps_report_vocabulary_with_attribution_and_kb_shape(built):
    entries, _ = built
    pneumothorax = entries["C1"]
    assert pneumothorax["id"] == "ncit-C1"
    assert pneumothorax["label"] == "DISEASE"
    assert pneumothorax["tier"] == "imported"
    assert pneumothorax["plain_definition"] == "An abnormal collection of air in the pleural space."  # "(NCI)" stripped
    assert pneumothorax["source_citation"] == "NCI Thesaurus (CC BY 4.0) - 'Pneumothorax' (C1)"
    assert pneumothorax["source_url"].endswith("/C1")
    for entry in entries.values():
        KBEntry.model_validate(entry)
    assert entries["C14"]["label"] == "TEST"


def test_filters_drop_what_does_not_belong_in_patient_reports(built):
    entries, dropped = built
    assert "C2" not in entries   # retired
    assert "C3" not in entries   # gene
    assert "C4" not in entries   # CTCAE grade (reads as treatment instructions)
    assert "C5" not in entries   # laterality coding variant
    assert "C6" not in entries   # no definition
    assert "C7" not in entries   # definition trips the validator's advice/treatment patterns
    assert "C12" not in entries  # "fatty liver" is already in the reviewed KB
    assert "C13" not in entries  # drugs are off by default
    assert dropped["already covered by the reviewed KB"] == 1


def test_one_word_aliases_must_be_the_concepts_own_name_or_an_abbreviation(built):
    entries, _ = built
    assert entries["C8"]["aliases"] == ["Complete Pharyngeal Contraction"]  # "complete" dropped
    assert entries["C9"]["aliases"] == ["Embolism"]                         # its own name: kept
    assert "DVC" in entries["C10"]["aliases"]                                # abbreviation: kept


def test_alias_shared_by_two_concepts_is_dropped_from_both(built):
    entries, _ = built
    assert "Shared Name" not in entries["C10"]["aliases"]
    assert "Shared Name" not in entries["C11"]["aliases"]


def test_drugs_only_with_the_flag(reviewed_kb):
    entries, _ = build(parse_flat(LINES), reviewed_kb, include_drugs=True)
    assert {e["ncit_code"]: e["label"] for e in entries}["C13"] == "DRUG"


def test_output_is_gzipped_json_with_license_and_is_reproducible(built, tmp_path):
    entries, dropped = built
    path = tmp_path / "ncit.json.gz"
    write_output(list(entries.values()), dropped, "26.09d", path)
    first = path.read_bytes()
    document = load_imported(path)
    assert document["tier"] == "imported" and document["reviewed"] is False
    assert "CC BY 4.0" in document["license"]
    assert "National Cancer Institute" in document["attribution"]
    assert document["counts"]["entries"] == len(entries)
    assert json.loads(gzip.decompress(first))["entries"][0]["id"].startswith("ncit-")
    write_output(list(entries.values()), dropped, "26.09d", path)
    assert path.read_bytes() == first  # same input, same day: byte-identical (gzip mtime fixed)
