# streamlit_app/app.py
# ─────────────────────────────────────────────────────────────────────────────
# ClinIQ — AI-Powered Medical Report Intelligence
# Full Streamlit UI connecting all 4 NLP modules
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modules.preprocessor import MedicalPreprocessor
from modules.ner_extractor import MedicalNERExtractor, ENTITY_COLORS
from modules.summarizer import MedicalSummarizer
from modules.relationship_extractor import MedicalRelationshipExtractor
from modules.icd_mapper import ICD10Mapper

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ClinIQ — Medical Report Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — White & Blue Clinical Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Fonts ── */
html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e8eeff; }
::-webkit-scrollbar-thumb { background: #2563eb; border-radius: 3px; }

/* ══════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #dbeafe !important;
}
/* All sidebar text dark */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #1e3a5f !important; }
/* Sidebar selectbox */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #f0f4ff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 8px !important;
    color: #1e3a5f !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #1e3a5f !important; }
/* Dropdown popup options */
[data-baseweb="popover"] ul, [data-baseweb="popover"] li {
    background-color: #ffffff !important; color: #1e3a5f !important;
}
[data-baseweb="popover"] li:hover { background-color: #dbeafe !important; }

.sidebar-brand {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem; font-weight: 700;
    color: #2563eb !important;
    letter-spacing: -1px;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 2px solid #dbeafe;
    margin-bottom: 1.5rem;
}
.sidebar-section {
    font-size: 0.72rem; font-weight: 700;
    color: #2563eb !important;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin: 1.2rem 0 0.6rem 0;
    background: #eff6ff !important;
    padding: 4px 8px; border-radius: 6px;
}

/* ══════════════════════════════════════
   EXPANDERS — most critical fix
   ══════════════════════════════════════ */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #dbeafe !important;
    border-radius: 12px !important;
}
/* Expander header row */
[data-testid="stExpander"] > details > summary {
    background-color: #ffffff !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] > details > summary:hover {
    background-color: #eff6ff !important;
}
[data-testid="stExpander"] > details > summary * {
    color: #1e40af !important;
    font-weight: 600 !important;
}
/* Expander body */
[data-testid="stExpander"] > details > div,
[data-testid="stExpander"] > details > div * {
    background-color: #ffffff !important;
    color: #334155 !important;
}
/* Chevron arrow */
[data-testid="stExpander"] svg { fill: #2563eb !important; color: #2563eb !important; }

/* ══════════════════════════════════════
   TABS
   ══════════════════════════════════════ */
[data-testid="stTabs"] [role="tablist"] {
    background-color: #ffffff !important;
    border-bottom: 2px solid #dbeafe !important;
    border-radius: 12px 12px 0 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #64748b !important;
    font-weight: 600 !important;
    background-color: transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
}
[data-testid="stTabs"] button[role="tab"] p { color: inherit !important; }
[data-testid="stTabs"] [role="tabpanel"] { background-color: #f0f4ff !important; padding-top: 1rem !important; }

/* ══════════════════════════════════════
   METRICS
   ══════════════════════════════════════ */
[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #dbeafe !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] p { color: #64748b !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
[data-testid="stMetricValue"] { color: #1d4ed8 !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stMetricValue"] div { color: #1d4ed8 !important; }

/* ══════════════════════════════════════
   INPUTS
   ══════════════════════════════════════ */
.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #1e293b !important;
}
.stTextArea textarea:focus { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important; }
.stTextArea label p { color: #1e293b !important; }
[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}
[data-testid="stTextInput"] label p { color: #1e293b !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background-color: #ffffff !important; border: 1.5px solid #bfdbfe !important; border-radius: 8px !important; }
[data-testid="stSelectbox"] label p { color: #1e293b !important; }

/* ══════════════════════════════════════
   FILE UPLOADER
   ══════════════════════════════════════ */
[data-testid="stFileUploader"] > div {
    background-color: #ffffff !important;
    border: 2px dashed #93c5fd !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #1e40af !important;
}
[data-testid="stFileUploader"] small { color: #64748b !important; }

/* ══════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important; }
.stButton > button p { color: #ffffff !important; }
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(16,185,129,0.3) !important;
}

/* ══════════════════════════════════════
   ALERTS
   ══════════════════════════════════════ */
.stAlert > div { border-radius: 10px !important; }
[data-testid="stAlert"] p { color: inherit !important; }

/* ══════════════════════════════════════
   PROGRESS BAR
   ══════════════════════════════════════ */
.stProgress > div { background-color: #dbeafe !important; border-radius: 4px !important; }
.stProgress > div > div { background: linear-gradient(to right, #2563eb, #0ea5e9) !important; border-radius: 4px !important; }

/* ══════════════════════════════════════
   CUSTOM COMPONENTS
   ══════════════════════════════════════ */
.cliniq-hero {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 40%, #0ea5e9 100%);
    border-radius: 24px; padding: 2.8rem 3rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 20px 60px rgba(37,99,235,0.25);
}
.cliniq-hero::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.cliniq-hero::after {
    content: ''; position: absolute; bottom: -30%; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(14,165,233,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.cliniq-logo { font-family: 'Space Mono', monospace !important; font-size: 3.2rem; font-weight: 700; color: #ffffff; letter-spacing: -2px; line-height: 1; margin-bottom: 0.3rem; }
.cliniq-logo span { color: #bfdbfe; }
.cliniq-tagline { font-size: 1.05rem; color: #bfdbfe; font-weight: 400; margin-bottom: 1.5rem; }
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.hero-badge { background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25); color: #ffffff; padding: 5px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 500; }

.cliniq-card { background: #ffffff; border: 1px solid #dbeafe; border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem; box-shadow: 0 2px 12px rgba(37,99,235,0.06); }
.card-title { font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #2563eb; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 8px; }
.card-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(to right, #dbeafe, transparent); }

.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.2rem 0; }
.stat-item { background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 1px solid #bfdbfe; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
.stat-number { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #1d4ed8; line-height: 1; }
.stat-label { font-size: 0.72rem; color: #64748b; margin-top: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

.entity-tag { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; margin: 3px; }
.entity-disease { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.entity-drug    { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.entity-symptom { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.entity-lab     { background: #ede9fe; color: #4c1d95; border: 1px solid #c4b5fd; }
.entity-vital   { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.entity-proc    { background: #ecfdf5; color: #065f46; border: 1px solid #6ee7b7; }

.icd-row { display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: #f8faff; border: 1px solid #dbeafe; border-radius: 10px; margin-bottom: 8px; }
.icd-row:hover { background: #eff6ff; }
.icd-code { font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #1d4ed8; background: #dbeafe; padding: 3px 10px; border-radius: 6px; min-width: 62px; text-align: center; }
.icd-desc { font-size: 0.85rem; color: #334155; flex: 1; }
.icd-conf { font-size: 0.72rem; color: #64748b; font-weight: 600; background: #f1f5f9; padding: 2px 8px; border-radius: 10px; }
.icd-cat  { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; color: #2563eb; background: #eff6ff; padding: 2px 8px; border-radius: 10px; }

.relation-row { display: flex; align-items: center; gap: 8px; padding: 9px 14px; background: #f8faff; border: 1px solid #dbeafe; border-radius: 10px; margin-bottom: 7px; font-size: 0.84rem; }
.relation-e1 { font-weight: 600; color: #1e40af; background: #dbeafe; padding: 2px 10px; border-radius: 6px; white-space: nowrap; }
.relation-arrow { color: #94a3b8; font-size: 1rem; }
.relation-badge { font-family: 'Space Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; white-space: nowrap; }
.relation-e2 { font-weight: 500; color: #334155; flex: 1; }

.summary-box { background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%); border-left: 4px solid #2563eb; border-radius: 0 12px 12px 0; padding: 1.2rem 1.6rem; margin: 0.8rem 0; font-size: 0.95rem; line-height: 1.75; color: #1e293b; }
.tldr-box { background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%); border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1rem; color: #ffffff; font-size: 0.9rem; line-height: 1.6; font-weight: 500; box-shadow: 0 4px 20px rgba(37,99,235,0.2); }

.soap-section { background: #ffffff; border: 1px solid #dbeafe; border-radius: 10px; padding: 1rem 1.4rem; margin-bottom: 10px; }
.soap-label { font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.soap-content { font-size: 0.88rem; color: #334155; line-height: 1.65; white-space: pre-wrap; }

.hipaa-banner { background: linear-gradient(135deg, #fff7ed, #fef3c7); border: 1px solid #fbbf24; border-radius: 10px; padding: 0.8rem 1.2rem; font-size: 0.82rem; color: #78350f; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px; }
</style>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached so it loads only once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    preprocessor  = MedicalPreprocessor(expand_abbreviations=True)
    ner           = MedicalNERExtractor(use_transformer=False)
    summarizer    = MedicalSummarizer(use_abstractive=False)
    rel_extractor = MedicalRelationshipExtractor()
    icd_mapper    = ICD10Mapper()
    return preprocessor, ner, summarizer, rel_extractor, icd_mapper

# ─────────────────────────────────────────────────────────────────────────────
# Sample Reports
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_REPORTS = {
    "STEMI Patient": """CHIEF COMPLAINT:
58-year-old male complains of chest pain and shortness of breath for 2 hours.

HISTORY OF PRESENT ILLNESS:
Patient is a 58-year-old male with past medical history of type 2 diabetes mellitus,
hypertension, and hyperlipidemia. He presented to the emergency department via ambulance
with acute onset substernal chest pain radiating to the left arm and jaw, associated with
diaphoresis, nausea, and shortness of breath. Blood pressure 145/92 mmHg, heart rate
102 bpm, respiratory rate 18 rpm, oxygen saturation 96%, temperature 98.6°F.

ALLERGIES: Penicillin (rash), Sulfonamides (angioedema)

LABORATORY RESULTS:
Troponin I critically elevated at 2.4 ng/mL. HbA1c 9.2% indicating poorly controlled
diabetes. LDL cholesterol 142 mg/dL. WBC 11.2 K/uL. Sodium 138 mEq/L. Creatinine 1.1 mg/dL.

ASSESSMENT AND PLAN:
1. Inferior STEMI — Percutaneous coronary intervention performed. Drug-eluting stent placed.
   Started on Aspirin 325 mg for myocardial infarction. Clopidogrel 75 mg daily.
   Atorvastatin 80 mg for hyperlipidemia. Metoprolol 50 mg twice daily.
2. Type 2 Diabetes Mellitus — HbA1c 9.2%, poorly controlled. Continue Metformin 500 mg twice daily.
   Added Empagliflozin 10 mg daily for cardiovascular protection.
3. Hypertension — Continue Lisinopril 10 mg daily, target blood pressure < 130/80 mmHg.
4. Obstructive sleep apnea — Continue CPAP therapy nightly.""",

    "Diabetic Patient": """CHIEF COMPLAINT:
45-year-old female with poorly controlled diabetes and fatigue.

HISTORY OF PRESENT ILLNESS:
Patient is a 45-year-old female with known type 1 diabetes mellitus presenting for
routine follow-up. She reports fatigue, polyuria, and polydipsia for 3 weeks.
Blood pressure 128/82 mmHg, heart rate 78 bpm, oxygen saturation 99%.

LABORATORY RESULTS:
HbA1c 10.4% indicating very poorly controlled diabetes. Fasting blood glucose 312 mg/dL.
Creatinine 1.4 mg/dL with eGFR 52 indicating chronic kidney disease stage 3.
TSH 6.2 indicating hypothyroidism. Hemoglobin 10.8 g/dL indicating anemia.

MEDICATIONS:
Insulin glargine 20 units at bedtime, Insulin lispro sliding scale.
Levothyroxine 50 mcg daily for hypothyroidism. Iron supplements 325 mg daily.

ASSESSMENT AND PLAN:
1. Type 1 Diabetes — poorly controlled. Increase insulin glargine to 26 units.
   Refer to endocrinology. Target HbA1c < 7.0%.
2. Chronic kidney disease stage 3 — diabetic nephropathy. Monitor creatinine monthly.
   Restrict protein intake. Nephrology referral.
3. Hypothyroidism — TSH elevated. Increase Levothyroxine 75 mcg daily.
4. Anemia — iron deficiency anemia. Continue iron supplementation.""",

    "Pneumonia Patient": """CHIEF COMPLAINT:
67-year-old male with fever, cough, and shortness of breath for 4 days.

HISTORY OF PRESENT ILLNESS:
67-year-old male with history of COPD and hypertension presents with worsening
cough productive of yellow sputum, fever of 101.8°F, chills, and dyspnea.
Blood pressure 138/88 mmHg, heart rate 96 bpm, respiratory rate 22 rpm,
oxygen saturation 91% on room air, temperature 101.8°F.

LABORATORY RESULTS:
WBC 16.4 K/uL indicating leukocytosis. CRP 84 mg/L markedly elevated.
Procalcitonin 2.1 ng/mL indicating bacterial infection. Creatinine 1.0 mg/dL.

ASSESSMENT AND PLAN:
1. Community acquired pneumonia — Chest X-ray confirms right lower lobe consolidation.
   Start Azithromycin 500 mg daily for pneumonia. Ceftriaxone 1g IV daily.
   Supplemental oxygen to maintain saturation above 94%.
2. COPD exacerbation — Albuterol nebulization every 4 hours. Prednisone 40 mg daily.
3. Hypertension — Continue Amlodipine 5 mg daily.""",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def render_entity_tags(entities: list, category: str, css_class: str, icon: str):
    if not entities:
        st.markdown(f"<span style='color:#94a3b8;font-size:0.82rem'>No {category} detected</span>", unsafe_allow_html=True)
        return
    tags_html = "".join([f'<span class="entity-tag {css_class}">{e}</span>' for e in entities[:12]])
    st.markdown(f"<div style='line-height:2.2'>{icon} {tags_html}</div>", unsafe_allow_html=True)

def render_relation_badge(relation_type: str) -> str:
    colors = {
        "SYMPTOM_OF":              ("background:#fee2e2;color:#991b1b",),
        "TREATS":                  ("background:#d1fae5;color:#065f46",),
        "TREATED_BY":              ("background:#d1fae5;color:#065f46",),
        "INDICATES":               ("background:#ede9fe;color:#4c1d95",),
        "CAUSED_BY":               ("background:#ffedd5;color:#9a3412",),
        "POTENTIAL_COMPLICATION_OF":("background:#fef9c3;color:#713f12",),
        "ALLERGIC_TO":             ("background:#fee2e2;color:#7f1d1d",),
        "DIAGNOSED_WITH":          ("background:#dcfce7;color:#14532d",),
    }
    style = colors.get(relation_type, ("background:#dbeafe;color:#1e40af",))[0]
    return f'<span class="relation-badge" style="{style}">{relation_type.replace("_"," ")}</span>'

def confidence_bar(confidence: float) -> str:
    """Render a clean CSS progress bar instead of unicode block chars."""
    pct = int(confidence * 100)
    color = "#16a34a" if pct >= 90 else "#2563eb" if pct >= 70 else "#f59e0b"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px">'
        f'<span style="display:inline-block;width:60px;height:7px;background:#e2e8f0;'
        f'border-radius:4px;overflow:hidden">'
        f'<span style="display:block;width:{pct}%;height:100%;background:{color};'
        f'border-radius:4px"></span></span>'
        f'<span style="font-size:0.75rem;font-weight:700;color:{color}">{pct}%</span>'
        f'</span>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🧬 ClinIQ</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Configuration</div>', unsafe_allow_html=True)

    summary_length = st.selectbox(
        "Summary Length",
        ["short", "medium", "detailed"],
        index=1,
        help="Controls how long the generated summary will be"
    )

    use_section_aware = st.toggle("Section-Aware Summary", value=True,
        help="Summarize each clinical section separately")

    st.markdown('<div class="sidebar-section">📋 Sample Reports</div>', unsafe_allow_html=True)

    sample_choice = st.selectbox(
        "Load Sample Report",
        ["— Select —"] + list(SAMPLE_REPORTS.keys()),
    )

    st.markdown('<div class="sidebar-section">ℹ️ Pipeline</div>', unsafe_allow_html=True)

    modules = [
        ("🔧", "Preprocessor",   "Cleaning, abbreviation expansion, section detection"),
        ("🏷️", "NER Extractor",  "Diseases, drugs, symptoms, lab values"),
        ("📝", "Summarizer",     "Extractive + SOAP clinical note"),
        ("🔗", "Relationships",  "Symptom→Disease, Drug→Treats triplets"),
        ("🏥", "ICD-10 Mapper",  "Diagnosis code assignment"),
    ]
    for icon, name, desc in modules:
        st.markdown(f"""
        <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:10px">
            <span style="font-size:1rem">{icon}</span>
            <div>
                <div style="font-size:0.82rem;font-weight:600;color:#1e40af">{name}</div>
                <div style="font-size:0.72rem;color:#64748b">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#94a3b8;line-height:1.6'>
    ⚠️ <b>HIPAA Notice</b><br>
    For educational use only.<br>
    No real patient data stored.<br>
    All processing is local.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="cliniq-hero">
    <div class="cliniq-logo">Clin<span>IQ</span></div>
    <div class="cliniq-tagline">AI-Powered Medical Report Intelligence · Summarize · Extract · Analyze</div>
    <div class="hero-badges">
        <span class="hero-badge">🧬 NER Extraction</span>
        <span class="hero-badge">📝 Auto Summarization</span>
        <span class="hero-badge">🔗 Relation Mapping</span>
        <span class="hero-badge">🏥 ICD-10 Coding</span>
        <span class="hero-badge">📋 SOAP Note Generation</span>
        <span class="hero-badge">⚡ scispaCy BC5CDR</span>
    </div>
</div>
""", unsafe_allow_html=True)

# HIPAA Banner
st.markdown("""
<div class="hipaa-banner">
    ⚠️ <strong>Educational Use Only</strong> — This tool is for demonstration and portfolio purposes.
    Do not process real Protected Health Information (PHI). All analysis runs locally.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📄 Input Medical Report")

# Populate from sample if selected
default_text = ""
if sample_choice != "— Select —":
    default_text = SAMPLE_REPORTS[sample_choice]

# ── Input Mode Tabs ────────────────────────────────────────────────────────
input_tab1, input_tab2 = st.tabs(["✏️ Paste Text", "📁 Upload File"])

report_text = ""

with input_tab1:
    report_text_paste = st.text_area(
        label="Paste your medical report here",
        value=default_text,
        height=220,
        placeholder="Paste a clinical note, discharge summary, or medical report here...\n\nOr select a sample from the sidebar →",
        label_visibility="collapsed"
    )
    report_text = report_text_paste

with input_tab2:
    st.markdown("""
    <div style="font-size:0.85rem;color:#64748b;margin-bottom:0.8rem">
        Upload a <strong style="color:#1e40af">.txt</strong> or
        <strong style="color:#1e40af">.pdf</strong> file containing a medical report.
        PDF text extraction is automatic.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your medical report file here",
        type=["txt", "pdf"],
        label_visibility="collapsed",
        help="Accepts .txt and .pdf files. Max 10MB."
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "txt":
            report_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"✅ Loaded `{uploaded_file.name}` — {len(report_text.split())} words")

        elif file_type == "pdf":
            with st.spinner("📑 Extracting text from PDF..."):
                try:
                    import pdfplumber
                    import io
                    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                        pdf_text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text += page_text + "\n"
                    if pdf_text.strip():
                        report_text = pdf_text
                        st.success(f"✅ Extracted `{uploaded_file.name}` — {len(report_text.split())} words, {len(pdf.pages)} pages")
                    else:
                        st.warning("⚠️ PDF appears to be scanned/image-based. Text extraction returned empty. Try a text-based PDF or paste the text manually.")
                except ImportError:
                    st.error("❌ pdfplumber not installed. Run: `pip install pdfplumber`")
                except Exception as e:
                    st.error(f"❌ PDF extraction failed: {e}")

        # Preview extracted text
        if report_text:
            with st.expander("👁️ Preview extracted text", expanded=False):
                st.markdown(f"""
                <div style="background:#f8faff;border:1px solid #dbeafe;border-radius:10px;
                            padding:1rem;font-size:0.82rem;font-family:'Space Mono',monospace;
                            color:#334155;white-space:pre-wrap;max-height:200px;overflow-y:auto">
                {report_text[:1500]}{'...' if len(report_text) > 1500 else ''}
                </div>
                """, unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    analyze_btn = st.button("🔍 Analyze Report", use_container_width=True)
with col_btn2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn and report_text.strip():
    # Load models
    with st.spinner("Loading ClinIQ models..."):
        preprocessor, ner, summarizer, rel_extractor, icd_mapper = load_models()

    # Progress bar
    progress = st.progress(0, text="🔧 Preprocessing report...")
    t_start = time.time()

    # ── Module A: Preprocess ──────────────────────────────────────────────
    preprocessed = preprocessor.process(report_text)
    progress.progress(20, text="🏷️ Running Named Entity Recognition...")

    # ── Module A: NER ─────────────────────────────────────────────────────
    ner_result = ner.extract(preprocessed.clean_text)
    progress.progress(45, text="📝 Generating summaries...")

    # ── Module B: Summarize ───────────────────────────────────────────────
    summary_result = summarizer.summarize(
        preprocessed, ner_result,
        length=summary_length,
        section_aware=use_section_aware
    )
    progress.progress(65, text="🔗 Extracting relationships...")

    # ── Module C: Relationships ───────────────────────────────────────────
    rel_result = rel_extractor.extract(preprocessed, ner_result)
    progress.progress(85, text="🏥 Mapping ICD-10 codes...")

    # ── Module D: ICD-10 ──────────────────────────────────────────────────
    icd_result = icd_mapper.map(ner_result)
    progress.progress(100, text="✅ Analysis complete!")

    t_elapsed = time.time() - t_start
    time.sleep(0.3)
    progress.empty()

    # ─────────────────────────────────────────────────────────────────────
    # Stats Row
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-item">
            <div class="stat-number">{preprocessed.word_count}</div>
            <div class="stat-label">Words Input</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{summary_result.summary_word_count}</div>
            <div class="stat-label">Summary Words</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{int((1-summary_result.compression_ratio)*100)}%</div>
            <div class="stat-label">Compression</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{ner_result.entity_count}</div>
            <div class="stat-label">Entities Found</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{len(ner_result.diseases)}</div>
            <div class="stat-label">Diagnoses</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{len(ner_result.medications)}</div>
            <div class="stat-label">Medications</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{rel_result.total_relations}</div>
            <div class="stat-label">Relations</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{icd_result.mapped_count}</div>
            <div class="stat-label">ICD-10 Codes</div>
        </div>
    </div>
    <div style="text-align:right;font-size:0.75rem;color:#94a3b8;margin-bottom:1rem">
        ⚡ Analyzed in {t_elapsed:.2f}s · Model: {ner_result.model_used}
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # Main Tabs
    # ─────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Summary",
        "🏷️ Entities",
        "🔗 Relationships",
        "🏥 ICD-10 Codes",
        "📋 Clinical Note",
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1 — SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        # TL;DR
        st.markdown(f"""
        <div class="cliniq-card">
            <div class="card-title">⚡ TL;DR — One-Line Summary</div>
            <div class="tldr-box">{summary_result.short_summary}</div>
        </div>
        """, unsafe_allow_html=True)

        # Main summary
        st.markdown(f"""
        <div class="cliniq-card">
            <div class="card-title">📝 AI Summary</div>
            <div class="summary-box">{summary_result.abstractive_summary}</div>
        </div>
        """, unsafe_allow_html=True)

        # Section summaries
        if summary_result.section_summaries:
            st.markdown('<div class="cliniq-card"><div class="card-title">📋 Section-by-Section Breakdown</div>', unsafe_allow_html=True)
            for section_name, section_text in summary_result.section_summaries.items():
                with st.expander(f"📌 {section_name.title()}", expanded=False):
                    st.markdown(f'<div class="summary-box">{section_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Vitals if found
        if ner_result.vitals:
            st.markdown('<div class="cliniq-card"><div class="card-title">🩺 Extracted Vital Signs</div>', unsafe_allow_html=True)
            vcols = st.columns(len(ner_result.vitals))
            for i, (vital, value) in enumerate(ner_result.vitals.items()):
                with vcols[i]:
                    st.metric(vital, value)
            st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2 — ENTITIES
    # ═══════════════════════════════════════════════════════════════════
    with tab2:
        col_l, col_r = st.columns(2)

        with col_l:
            # Diseases
            st.markdown('<div class="cliniq-card"><div class="card-title">🦠 Diseases & Diagnoses</div>', unsafe_allow_html=True)
            render_entity_tags(ner_result.diseases, "diseases", "entity-disease", "")
            st.markdown('</div>', unsafe_allow_html=True)

            # Symptoms
            st.markdown('<div class="cliniq-card"><div class="card-title">😷 Symptoms</div>', unsafe_allow_html=True)
            render_entity_tags(ner_result.symptoms, "symptoms", "entity-symptom", "")
            st.markdown('</div>', unsafe_allow_html=True)

            # Procedures
            st.markdown('<div class="cliniq-card"><div class="card-title">🔬 Procedures</div>', unsafe_allow_html=True)
            render_entity_tags(ner_result.procedures, "procedures", "entity-proc", "")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            # Medications
            st.markdown('<div class="cliniq-card"><div class="card-title">💊 Medications</div>', unsafe_allow_html=True)
            if ner_result.medications:
                for med in ner_result.medications[:10]:
                    name_part  = f'<span style="font-weight:700;color:#0f172a;font-size:0.88rem">{med.name}</span>' if med.name else ""
                    dose_part  = f'<span style="color:#1d4ed8;font-weight:600;background:#dbeafe;padding:2px 8px;border-radius:6px;font-size:0.8rem">{med.dosage}</span>' if med.dosage else ""
                    freq_part  = f'<span style="color:#64748b;font-size:0.78rem">{med.frequency}</span>' if med.frequency else ""
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;padding:9px 14px;
                                background:#f8faff;border:1px solid #dbeafe;
                                border-radius:10px;margin-bottom:6px">
                        <span style="font-size:1.1rem">💊</span>
                        {name_part} {dose_part} {freq_part}
                    </div>""", unsafe_allow_html=True)
            else:
                render_entity_tags(ner_result.drugs, "drugs", "entity-drug", "")
            st.markdown('</div>', unsafe_allow_html=True)

            # Lab Values
            st.markdown('<div class="cliniq-card"><div class="card-title">🧪 Lab Values</div>', unsafe_allow_html=True)
            if ner_result.lab_values:
                for lab in ner_result.lab_values[:8]:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:6px 12px;background:#f8faff;border:1px solid #dbeafe;
                                border-radius:8px;margin-bottom:6px;font-size:0.84rem">
                        <span style="font-weight:600;color:#1e40af">{lab.test_name}</span>
                        <span style="font-family:'Space Mono',monospace;color:#334155">
                            {lab.value} {lab.unit}
                        </span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#94a3b8;font-size:0.82rem'>No lab values extracted</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Preprocessor stats
        with st.expander("🔧 Preprocessor Details", expanded=False):
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Words", preprocessed.word_count)
            pcol2.metric("Sections", len(preprocessed.sections))
            pcol3.metric("Abbrevs Expanded", preprocessed.abbreviations_expanded)
            if preprocessed.sections:
                st.markdown("**Detected Sections:**")
                st.markdown(", ".join([f"`{s}`" for s in preprocessed.sections.keys()]))

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3 — RELATIONSHIPS
    # ═══════════════════════════════════════════════════════════════════
    with tab3:
        if rel_result.total_relations == 0:
            st.info("No relationships extracted. Try a longer, more detailed report.")
        else:
            st.markdown(f"""
            <div style="margin-bottom:1rem;font-size:0.85rem;color:#64748b">
                Found <b style="color:#2563eb">{rel_result.total_relations}</b> relations
                across <b style="color:#2563eb">{len(rel_result.relation_types_found)}</b> types
                · Sources: rule-based + dependency parsing + knowledge base
            </div>
            """, unsafe_allow_html=True)

            sections = [
                ("🦠 Symptom → Disease",      rel_result.symptom_of,      "SYMPTOM_OF"),
                ("💊 Drug Treats Disease",     rel_result.treats,           "TREATS"),
                ("🔬 Lab Indicates Disease",   rel_result.indicates,        "INDICATES"),
                ("🔴 Allergies",               rel_result.allergic_to,      "ALLERGIC_TO"),
                ("⚡ Potential Complications", rel_result.complication_of,  "POTENTIAL_COMPLICATION_OF"),
                ("✅ Diagnoses Confirmed",     [r for r in rel_result.relations
                                               if r.relation == "DIAGNOSED_WITH"], "DIAGNOSED_WITH"),
            ]

            for title, rels, rel_type in sections:
                if not rels:
                    continue
                st.markdown(f'<div class="cliniq-card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
                seen = set()
                for r in rels[:10]:
                    key = (r.entity1.lower(), r.entity2.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    badge = render_relation_badge(r.relation)
                    st.markdown(f"""
                    <div class="relation-row">
                        <span class="relation-e1">{r.entity1[:30]}</span>
                        <span class="relation-arrow">──►</span>
                        {badge}
                        <span class="relation-arrow">──►</span>
                        <span class="relation-e2">{r.entity2[:40]}</span>
                        <span style="font-size:0.72rem;color:#94a3b8;margin-left:auto">
                            {r.confidence:.0%} · {r.source}
                        </span>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4 — ICD-10
    # ═══════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown(f"""
        <div style="margin-bottom:1rem;font-size:0.85rem;color:#64748b">
            Mapped <b style="color:#2563eb">{icd_result.mapped_count}</b> diagnoses to ICD-10-CM codes
            · Categories: {', '.join(f'<b>{c}</b>' for c in icd_result.categories_found)}
        </div>
        """, unsafe_allow_html=True)

        for m in icd_result.mappings:
            if not m.icd_code:
                continue
            st.markdown(f"""
            <div class="icd-row">
                <span class="icd-code">{m.icd_code}</span>
                <div style="flex:1;min-width:0">
                    <span style="font-weight:600;color:#0f172a;font-size:0.88rem">{m.diagnosis}</span>
                    <span style="color:#94a3b8;font-size:0.78rem;margin-left:6px">· {m.description}</span>
                </div>
                <span class="icd-cat">{m.category}</span>
                {confidence_bar(m.confidence)}
            </div>""", unsafe_allow_html=True)

        if icd_result.unmapped:
            st.markdown("---")
            st.markdown(f"""
            <div style="background:#fff7ed;border:1.5px solid #fb923c;border-radius:10px;
                        padding:0.9rem 1.2rem;font-size:0.85rem;color:#7c2d12;
                        display:flex;align-items:flex-start;gap:10px;margin-top:0.5rem">
                <span style="font-size:1.1rem;flex-shrink:0">⚠️</span>
                <div>
                    <span style="font-weight:700;color:#9a3412">Could not map to ICD-10:</span>
                    <span style="color:#7c2d12;margin-left:6px">{', '.join(icd_result.unmapped)}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # ICD Search
        st.markdown("---")
        st.markdown("#### 🔍 ICD-10 Code Search")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input("Search ICD-10 database", placeholder="e.g. diabetes, heart failure, pneumonia...")
        with search_col2:
            n_results = st.selectbox("Results", [3, 5, 10], index=1)

        if search_query:
            from modules.icd_mapper import ICD10Mapper as _M
            _mapper = icd_mapper
            results = _mapper.search_by_keyword(search_query, top_n=n_results)
            if results:
                for r in results:
                    st.markdown(f"""
                    <div class="icd-row">
                        <span class="icd-code">{r.icd_code}</span>
                        <span class="icd-desc">{r.description}</span>
                        <span class="icd-cat">{r.category}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No ICD-10 codes found for that search term.")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 5 — CLINICAL NOTE
    # ═══════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="cliniq-card"><div class="card-title">📋 Auto-Generated SOAP Clinical Note</div>', unsafe_allow_html=True)

        # Parse the clinical note into sections
        note = summary_result.clinical_note
        soap_map = {
            "[S] SUBJECTIVE":  ("🗣️ Subjective", "#3b82f6"),
            "[O] OBJECTIVE":   ("🩺 Objective",   "#10b981"),
            "[A] ASSESSMENT":  ("🎯 Assessment",  "#ef4444"),
            "[P] PLAN":        ("📋 Plan",        "#8b5cf6"),
            "[SUMMARY]":       ("💡 Summary",     "#f59e0b"),
        }

        # Split note by section
        current_section = None
        current_content = []
        all_sections = []

        for line in note.split("\n"):
            matched = False
            for key in soap_map:
                if line.strip().startswith(key):
                    if current_section:
                        all_sections.append((current_section, "\n".join(current_content)))
                    current_section = key
                    current_content = []
                    matched = True
                    break
            if not matched and current_section and "═" not in line and "─" not in line and "AUTO-GENERATED" not in line and "AI-GENERATED" not in line and "Generated:" not in line:
                current_content.append(line)

        if current_section:
            all_sections.append((current_section, "\n".join(current_content)))

        if all_sections:
            for section_key, content in all_sections:
                label, color = soap_map.get(section_key, (section_key, "#2563eb"))
                clean_content = content.strip()
                if clean_content:
                    st.markdown(f"""
                    <div class="soap-section" style="border-left:3px solid {color}">
                        <div class="soap-label" style="color:{color}">{label}</div>
                        <div class="soap-content">{clean_content}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="summary-box"><pre style="font-size:0.82rem">{note}</pre></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="⬇️ Download Clinical Note (.txt)",
            data=summary_result.clinical_note,
            file_name=f"cliniq_note_{preprocessed.report_id}.txt",
            mime="text/plain",
        )

        st.markdown("""
        <div style="margin-top:1rem;padding:10px 14px;background:#fff7ed;border:1px solid #fbbf24;
                    border-radius:8px;font-size:0.78rem;color:#92400e">
            ⚠️ <strong>Physician Review Required</strong> — This AI-generated note is for reference only
            and must be reviewed and verified by a licensed clinician before use in any clinical context.
        </div>
        """, unsafe_allow_html=True)

elif analyze_btn and not report_text.strip():
    st.warning("⚠️ Please paste a medical report or select a sample from the sidebar.")

# ─────────────────────────────────────────────────────────────────────────────
# Empty State
# ─────────────────────────────────────────────────────────────────────────────

else:
    if not report_text.strip():
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;background:#ffffff;
                    border:2px dashed #bfdbfe;border-radius:20px;margin-top:1rem">
            <div style="font-size:3rem;margin-bottom:1rem">🧬</div>
            <div style="font-size:1.3rem;font-weight:700;color:#1e40af;margin-bottom:0.5rem">
                Ready to Analyze
            </div>
            <div style="font-size:0.9rem;color:#64748b;max-width:400px;margin:0 auto">
                Paste a clinical note or discharge summary above, or select
                a sample report from the sidebar to get started.
            </div>
            <div style="margin-top:2rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
                <div style="text-align:center">
                    <div style="font-size:1.5rem">📝</div>
                    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">AI Summary</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.5rem">🏷️</div>
                    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">NER</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.5rem">🔗</div>
                    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">Relations</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.5rem">🏥</div>
                    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">ICD-10</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.5rem">📋</div>
                    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">SOAP Note</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
