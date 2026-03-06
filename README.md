#  🧬 ClinIQ
#  Medical Report Summarization & Information Extraction

> An end-to-end NLP pipeline that automatically summarizes medical reports,
> extracts clinical entities, maps ICD-10 codes, and generates structured clinical notes.

---

## 📁 Project Structure

```
medical_report_nlp/
│
├── data/
│   ├── raw/                   # Original unprocessed reports
│   ├── processed/             # Cleaned, tokenized data
│   └── sample_reports/        # Synthetic reports for testing
│
├── modules/
│   ├── __init__.py
│   ├── preprocessor.py        # MODULE A — Part 1: Text cleaning & normalization
│   ├── ner_extractor.py       # MODULE A — Part 2: Named Entity Recognition
│   ├── summarizer.py          # MODULE B — Extractive + Abstractive summarization
│   ├── relationship_extractor.py  # MODULE C — Symptom-diagnosis relationships
│   └── icd_mapper.py          # MODULE D — ICD-10 code mapping
│
├── utils/
│   ├── __init__.py
│   ├── medical_abbreviations.py   # 200+ medical abbreviation mappings
│   └── helpers.py                 # Shared utility functions
│
├── streamlit_app/
│   └── app.py                 # Full Streamlit UI
│
├── tests/
│   └── test_module_a.py       # Unit tests for preprocessing + NER
│
├── notebooks/
│   └── exploration.ipynb      # EDA and experimentation
│
├── requirements.txt
└── README.md
```

---


---

## 🧠 Models Used

| Module | Model | Source |
|--------|-------|--------|
| NER | `en_ner_bc5cdr_md` | scispaCy |
| NER (extended) | `dmis-lab/biobert-base-cased-v1.2` | HuggingFace |
| Summarization | `facebook/bart-large-cnn` | HuggingFace |
| ICD Mapping | `emilyalsentzer/Bio_ClinicalBERT` | HuggingFace |

---

## 📊 Datasets

| Dataset | Usage | Access |
|---------|-------|--------|
| PubMed Abstracts | Summarization testing | Free via HuggingFace |
| i2b2 2010/2012 | NER gold labels | Free registration |
| Synthetic Reports | Development & demo | Included in repo |

---

## 🔐 HIPAA Disclaimer

This tool is for **educational and portfolio purposes only**.
- No real patient data is stored or transmitted
- All demo data is synthetic or publicly de-identified
- Do not use with real Protected Health Information (PHI)

---
