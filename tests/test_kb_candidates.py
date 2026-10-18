"""tools/kb_candidates.py: drafting KB entries from MedlinePlus, and promoting approved ones.

Uses a tiny hand-written XML in the MedlinePlus format, not the 30 MB download, so these
tests run offline.
"""

from __future__ import annotations

import json

import pytest

from app.retrieval.knowledge_base import DEFAULT_KB_PATH, KnowledgeBaseError, load_knowledge_base
from tools.kb_candidates import build_candidates, dump_kb, parse_lab_tests, parse_topics, promote

XML = """<?xml version="1.0" encoding="UTF-8"?>
<health-topics total="4">
<health-topic title="Urinary Tract Infections" url="https://medlineplus.gov/urinarytractinfections.html"
    meta-desc="A UTI is an infection in the urinary tract." language="English" id="1">
  <also-called>UTI</also-called>
  <also-called>Fatty liver</also-called>
  <see-reference>Pyelonephritis</see-reference>
  <full-summary>&lt;p&gt;Infections of the &lt;b&gt;urinary tract&lt;/b&gt;.&lt;/p&gt;</full-summary>
  <group url="u" id="1">Kidneys and Urinary System</group>
  <mesh-heading><descriptor id="D1">Urinary Tract Infections</descriptor></mesh-heading>
</health-topic>
<health-topic title="Potassium" url="https://medlineplus.gov/potassium.html" meta-desc="" language="English" id="2">
  <also-called>Hyperkalemia</also-called>
  <see-reference>Hypokalemia</see-reference>
  <full-summary>&lt;p&gt;Potassium is a mineral.&lt;/p&gt;</full-summary>
</health-topic>
<health-topic title="Infecciones urinarias" url="https://medlineplus.gov/spanish/x.html" meta-desc="" language="Spanish" id="3">
  <also-called>UTI</also-called>
</health-topic>
</health-topics>
"""


@pytest.fixture
def xml_path(tmp_path):
    path = tmp_path / "mplus_topics_2026-01-01.xml"
    path.write_text(XML, encoding="utf-8")
    return path


@pytest.fixture
def wishlist(tmp_path):
    path = tmp_path / "wishlist.txt"
    path.write_text("# comment\nurinary tract infection\nhypokalemia\nALT\nzzz unknown\n", encoding="utf-8")
    return path


def _by_term(candidates):
    return {c["wanted"]: c for c in candidates["candidates"]}


def test_parse_keeps_english_and_separates_synonyms_from_cross_references(xml_path):
    topics = parse_topics(xml_path)
    assert [t.title for t in topics] == ["Urinary Tract Infections", "Potassium"]
    uti = topics[0]
    assert uti.also_called == ["UTI", "Fatty liver"]
    assert uti.see_references == ["Pyelonephritis"]
    assert uti.summary == "Infections of the urinary tract ."


def test_build_matches_plural_titles_and_drafts_an_unapproved_entry(xml_path, wishlist):
    candidate = _by_term(build_candidates(xml_path, wishlist_path=wishlist))["urinary tract infection"]
    assert candidate["status"] == "matched"
    assert candidate["approved"] is False
    entry = candidate["entry"]
    assert entry["plain_definition"] == "" and entry["label"] == ""
    assert entry["source_citation"] == "MedlinePlus Health Topics - 'Urinary Tract Infections'"
    assert entry["source_url"] == "https://medlineplus.gov/urinarytractinfections.html"
    assert entry["aliases"] == ["urinary tract infection", "Urinary Tract Infections", "UTI"]
    assert candidate["reference"]["see_also_not_aliases"] == ["Pyelonephritis"]


def test_alias_already_owned_by_another_entry_is_reported_not_suggested(xml_path, wishlist):
    candidate = _by_term(build_candidates(xml_path, wishlist_path=wishlist))["urinary tract infection"]
    assert candidate["alias_conflicts"] == {"Fatty liver": "kb-001"}
    assert "Fatty liver" not in candidate["entry"]["aliases"]


def test_cross_reference_match_borrows_no_aliases(xml_path, wishlist):
    # "Hypokalemia" is only a see-reference on Potassium, whose synonyms include its
    # opposite, Hyperkalemia. That must never be offered as an alias.
    candidate = _by_term(build_candidates(xml_path, wishlist_path=wishlist))["hypokalemia"]
    assert candidate["status"] == "related_topic"
    assert candidate["entry"]["aliases"] == ["hypokalemia"]


def test_existing_and_unknown_terms(xml_path, wishlist):
    candidates = _by_term(build_candidates(xml_path, wishlist_path=wishlist))
    assert candidates["ALT"]["status"] == "already_in_kb: kb-002"
    assert candidates["zzz unknown"]["status"] == "not_found"
    assert candidates["zzz unknown"]["entry"] is None


def test_rebuild_keeps_reviewed_candidates(xml_path, wishlist):
    first = build_candidates(xml_path, wishlist_path=wishlist)
    _by_term(first)["urinary tract infection"]["entry"]["plain_definition"] = "Reviewer text."
    second = build_candidates(xml_path, existing=first, wishlist_path=wishlist)
    assert _by_term(second)["urinary tract infection"]["entry"]["plain_definition"] == "Reviewer text."


# --- Medical Tests --------------------------------------------------------------------

LAB_INDEX = """
<a href="https://medlineplus.gov/lab-tests/potassium-blood-test/">Potassium Blood Test</a>
<a href="https://medlineplus.gov/lab-tests/white-blood-cell-wbc-in-stool/">White Blood Cell (WBC) in Stool</a>
<a href="https://medlineplus.gov/lab-tests/white-blood-count-wbc/">White Blood Count (WBC)</a>
<a href="https://medlineplus.gov/some-other-page/">Not a lab test</a>
"""


def test_parse_lab_tests_derives_bare_and_abbreviated_names():
    tests = {t.title: t for t in parse_lab_tests(LAB_INDEX)}
    assert set(tests) == {"Potassium Blood Test", "White Blood Cell (WBC) in Stool", "White Blood Count (WBC)"}
    assert tests["Potassium Blood Test"].also_called == ["Potassium"]
    assert tests["White Blood Count (WBC)"].also_called == ["White Blood Count", "WBC"]
    assert tests["White Blood Count (WBC)"].source == "MedlinePlus Medical Tests"


def test_lab_tests_take_precedence_and_cite_medical_tests(xml_path, tmp_path):
    # "potassium" is also a Health Topic (the mineral); in a report it is a test result.
    wishlist = tmp_path / "w.txt"
    wishlist.write_text("potassium\nWBC\n", encoding="utf-8")
    fetched = []

    def fake_fetch(topic):
        fetched.append(topic.title)
        topic.meta_desc = f"About {topic.title}."

    candidates = _by_term(build_candidates(xml_path, wishlist_path=wishlist,
                                           lab_tests=parse_lab_tests(LAB_INDEX), fetch_details=fake_fetch))
    potassium = candidates["potassium"]
    assert potassium["entry"]["source_citation"] == "MedlinePlus Medical Tests - 'Potassium Blood Test'"
    assert potassium["reference"]["meta_desc"] == "About Potassium Blood Test."
    # Two tests are abbreviated WBC: the general one (shorter title) is chosen.
    assert candidates["WBC"]["entry"]["source_url"] == "https://medlineplus.gov/lab-tests/white-blood-count-wbc/"
    assert candidates["WBC"]["suggestions"][0]["title"] == "White Blood Cell (WBC) in Stool"
    assert fetched == ["Potassium Blood Test", "White Blood Count (WBC)"]


# --- promotion ------------------------------------------------------------------------


@pytest.fixture
def kb_copy(tmp_path):
    path = tmp_path / "knowledge_base.json"
    path.write_text(DEFAULT_KB_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _approved(**overrides):
    entry = {
        "term": "urinary tract infection",
        "aliases": ["urinary tract infection", "UTI"],
        "label": "DISEASE",
        "plain_definition": "An infection in the parts of the body that make and carry urine.",
        "clinical_context": "Often found on a urine test.",
        "source_citation": "MedlinePlus Health Topics - 'Urinary Tract Infections'",
        "source_url": "https://medlineplus.gov/urinarytractinfections.html",
        "is_common": False,
    }
    entry.update(overrides)
    return {"wanted": entry["term"], "approved": True, "entry": entry}


def test_promote_appends_approved_entries_with_the_next_id(kb_copy):
    candidates = {"candidates": [_approved(), {"wanted": "x", "approved": False, "entry": None}]}
    assert promote(candidates, kb_copy) == ["kb-047"]
    kb = load_knowledge_base(kb_copy)
    assert kb.lookup("UTI").id == "kb-047"
    assert candidates["candidates"][0]["promoted_as"] == "kb-047"
    assert promote(candidates, kb_copy) == []  # already promoted: not added twice


def test_promote_refuses_an_incomplete_entry(kb_copy):
    before = kb_copy.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="plain_definition"):
        promote({"candidates": [_approved(plain_definition="")]}, kb_copy)
    with pytest.raises(ValueError, match="label"):
        promote({"candidates": [_approved(label="")]}, kb_copy)
    assert kb_copy.read_text(encoding="utf-8") == before


def test_promote_refuses_an_ambiguous_alias_and_writes_nothing(kb_copy):
    before = kb_copy.read_text(encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="fatty liver"):
        promote({"candidates": [_approved(aliases=["urinary tract infection", "fatty liver"])]}, kb_copy)
    assert kb_copy.read_text(encoding="utf-8") == before


def test_dump_kb_reproduces_the_committed_file_exactly():
    raw = DEFAULT_KB_PATH.read_text(encoding="utf-8")
    assert dump_kb(json.loads(raw)) == raw
