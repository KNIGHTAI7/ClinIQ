#  🧬 ClinIQ
# 🏥 Medical Report Summarization & Information Extraction

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

## ⚙️ Setup Instructions

### Step 1 — Clone and create virtual environment
```bash
git clone <your-repo-url>
cd medical_report_nlp
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Install scispaCy medical models
```bash
# Small model (faster, less accurate)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# Large model (recommended for production)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz

# BC5CDR model — specialized for disease & chemical NER
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
```

### Step 4 — Download NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

### Step 5 — Run tests to verify setup
```bash
python -m pytest tests/ -v
```

### Step 6 — Launch Streamlit app
```bash
streamlit run streamlit_app/app.py
```

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

## 🚀 Deployment

```bash
# Deploy to Streamlit Cloud
# 1. Push to GitHub
# 2. Go to share.streamlit.io
# 3. Connect your repo → set main file: streamlit_app/app.py
```
