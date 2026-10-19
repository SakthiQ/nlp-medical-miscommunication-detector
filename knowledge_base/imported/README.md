# Imported knowledge base tier (unreviewed)

`ncit.json.gz` holds definitions and synonyms imported in bulk from the **NCI Thesaurus**,
one entry per concept that a patient report could plausibly contain: conditions,
findings, symptoms, tests, test results and anatomy.

**Nothing here has been reviewed.** NCIt definitions are written for clinicians and
researchers, not patients. The curated `../knowledge_base.json` is the reviewed tier and
always takes precedence; concepts it already covers are skipped at import. The app does
not read this file yet. When it does, answers drawn from it must be marked as unreviewed.

## Rebuild

```bash
python -m backend.tools.ncit_import --download   # fetch the current NCIt flat file, then build
python -m backend.tools.ncit_import              # rebuild from knowledge_base/curation/raw/ncit/
```

The importer's docstring (`backend/tools/ncit_import.py`) lists every filter and why it
exists. The output file records its NCIt version, entry counts by label, and how many
concepts each filter dropped.

## Attribution and license

Contains definitions and synonyms from the NCI Thesaurus, produced by the Enterprise
Vocabulary Services group of the Center for Biomedical Informatics and Information
Technology, National Cancer Institute, USA. The NCI Thesaurus is released under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
([terms of use](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.htm)).
"NCI Thesaurus" is a trademark of the NCI; this file is a derived subset, not the NCI
Thesaurus itself.

**Changes made** (as CC BY 4.0 requires us to indicate): filtered to report-relevant
semantic types; research grading and staging concepts, coding variants and study
documents removed; definitions that are long or match the safety validator's forbidden
patterns removed; trailing source tags such as "(NCI)" stripped; ambiguous, short and
everyday-word synonyms removed; concepts already in the reviewed knowledge base skipped.
Definitions are otherwise unchanged.
