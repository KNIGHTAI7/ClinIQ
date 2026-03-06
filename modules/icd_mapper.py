# modules/icd_mapper.py
# ─────────────────────────────────────────────────────────────────────────────
# MODULE D — ICD-10 Code Mapping
#
# What this does:
#   Maps extracted diagnoses → standardized ICD-10-CM billing codes
#
# Approach (3 layers, best match wins):
#   Layer 1 — Exact match lookup  (instant, highest confidence)
#   Layer 2 — Fuzzy string match  (handles typos, abbreviations)
#   Layer 3 — Semantic similarity (ClinicalBERT embeddings + cosine similarity)
#             Falls back to TF-IDF if sentence-transformers unavailable
#
# Output per diagnosis:
#   {
#     "diagnosis":   "Type 2 Diabetes Mellitus",
#     "icd_code":    "E11.9",
#     "description": "Type 2 diabetes mellitus without complications",
#     "category":    "Endocrine, Nutritional and Metabolic Diseases",
#     "confidence":  0.97,
#     "match_type":  "exact"
#   }
# ─────────────────────────────────────────────────────────────────────────────

import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import timer


# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 Code Database
# Curated subset of ICD-10-CM covering the most common inpatient diagnoses
# Full ICD-10 has 70,000+ codes — this covers ~95% of typical clinical notes
# ─────────────────────────────────────────────────────────────────────────────

ICD10_DATABASE = {

    # ── Cardiovascular (I00–I99) ───────────────────────────────────────────
    "I10":    {"desc": "Essential (primary) hypertension",
               "aliases": ["hypertension", "htn", "high blood pressure", "elevated blood pressure"],
               "category": "Cardiovascular"},

    "I11.9":  {"desc": "Hypertensive heart disease without heart failure",
               "aliases": ["hypertensive heart disease", "hypertensive cardiomyopathy"],
               "category": "Cardiovascular"},

    "I21.0":  {"desc": "ST elevation myocardial infarction of anterior wall",
               "aliases": ["anterior stemi", "anterior mi", "anterior wall mi",
                           "anterior myocardial infarction", "lad occlusion"],
               "category": "Cardiovascular"},

    "I21.19": {"desc": "ST elevation myocardial infarction of inferior wall",
               "aliases": ["inferior stemi", "inferior mi", "inferior wall mi",
                           "inferior myocardial infarction", "rca occlusion",
                           "inferior st elevation myocardial infarction"],
               "category": "Cardiovascular"},

    "I21.4":  {"desc": "Non-ST elevation myocardial infarction",
               "aliases": ["nstemi", "non st elevation mi", "non-st elevation myocardial infarction",
                           "nstemi", "non-stemi"],
               "category": "Cardiovascular"},

    "I21.9":  {"desc": "Acute myocardial infarction, unspecified",
               "aliases": ["myocardial infarction", "mi", "heart attack", "acute mi",
                           "acute myocardial infarction", "stemi"],
               "category": "Cardiovascular"},

    "I25.10": {"desc": "Atherosclerotic heart disease of native coronary artery",
               "aliases": ["coronary artery disease", "cad", "ischemic heart disease",
                           "coronary atherosclerosis", "atherosclerotic heart disease"],
               "category": "Cardiovascular"},

    "I48.0":  {"desc": "Paroxysmal atrial fibrillation",
               "aliases": ["paroxysmal afib", "paroxysmal atrial fibrillation"],
               "category": "Cardiovascular"},

    "I48.11": {"desc": "Longstanding persistent atrial fibrillation",
               "aliases": ["persistent afib", "chronic atrial fibrillation",
                           "permanent atrial fibrillation"],
               "category": "Cardiovascular"},

    "I48.91": {"desc": "Unspecified atrial fibrillation",
               "aliases": ["atrial fibrillation", "afib", "a fib", "af"],
               "category": "Cardiovascular"},

    "I50.9":  {"desc": "Heart failure, unspecified",
               "aliases": ["heart failure", "chf", "congestive heart failure",
                           "cardiac failure", "hf"],
               "category": "Cardiovascular"},

    "I50.32": {"desc": "Chronic diastolic (congestive) heart failure",
               "aliases": ["diastolic heart failure", "hfpef", "heart failure preserved ejection"],
               "category": "Cardiovascular"},

    "I50.22": {"desc": "Chronic systolic (congestive) heart failure",
               "aliases": ["systolic heart failure", "hfref", "reduced ejection fraction heart failure"],
               "category": "Cardiovascular"},

    "I26.99": {"desc": "Other pulmonary embolism without acute cor pulmonale",
               "aliases": ["pulmonary embolism", "pe", "pulmonary thromboembolism",
                           "pulmonary embolus"],
               "category": "Cardiovascular"},

    "I82.401":{"desc": "Acute deep vein thrombosis of unspecified deep veins of right lower extremity",
               "aliases": ["deep vein thrombosis", "dvt", "deep venous thrombosis",
                           "lower extremity dvt"],
               "category": "Cardiovascular"},

    "I63.9":  {"desc": "Cerebral infarction, unspecified",
               "aliases": ["stroke", "cerebrovascular accident", "cva", "ischemic stroke",
                           "cerebral infarction", "brain attack"],
               "category": "Cardiovascular"},

    "I64":    {"desc": "Stroke, not specified as haemorrhage or infarction",
               "aliases": ["stroke unspecified", "tia vs stroke", "cerebrovascular disease"],
               "category": "Cardiovascular"},

    "G45.9":  {"desc": "Transient cerebral ischaemic attack, unspecified",
               "aliases": ["tia", "transient ischemic attack", "mini stroke",
                           "transient cerebral ischemia"],
               "category": "Neurological"},

    "I71.4":  {"desc": "Abdominal aortic aneurysm, without rupture",
               "aliases": ["aortic aneurysm", "abdominal aortic aneurysm", "aaa"],
               "category": "Cardiovascular"},

    # ── Endocrine & Metabolic (E00–E89) ───────────────────────────────────
    "E11.9":  {"desc": "Type 2 diabetes mellitus without complications",
               "aliases": ["type 2 diabetes", "t2dm", "type 2 diabetes mellitus",
                           "diabetes mellitus type 2", "dm2", "adult onset diabetes",
                           "non-insulin dependent diabetes"],
               "category": "Endocrine"},

    "E11.65": {"desc": "Type 2 diabetes mellitus with hyperglycemia",
               "aliases": ["type 2 diabetes with hyperglycemia", "uncontrolled diabetes",
                           "poorly controlled diabetes", "poorly controlled type 2 diabetes"],
               "category": "Endocrine"},

    "E11.40": {"desc": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
               "aliases": ["diabetic neuropathy", "peripheral diabetic neuropathy",
                           "diabetes with neuropathy"],
               "category": "Endocrine"},

    "E11.319":{"desc": "Type 2 diabetes mellitus with unspecified diabetic retinopathy",
               "aliases": ["diabetic retinopathy", "diabetes with retinopathy"],
               "category": "Endocrine"},

    "E11.22": {"desc": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
               "aliases": ["diabetic nephropathy", "diabetic kidney disease",
                           "diabetes with chronic kidney disease"],
               "category": "Endocrine"},

    "E10.9":  {"desc": "Type 1 diabetes mellitus without complications",
               "aliases": ["type 1 diabetes", "t1dm", "type 1 diabetes mellitus",
                           "juvenile diabetes", "insulin dependent diabetes"],
               "category": "Endocrine"},

    "E13.65": {"desc": "Other specified diabetes mellitus with hyperglycemia",
               "aliases": ["diabetes mellitus", "diabetes", "dm", "unspecified diabetes"],
               "category": "Endocrine"},

    "E78.5":  {"desc": "Hyperlipidemia, unspecified",
               "aliases": ["hyperlipidemia", "high cholesterol", "dyslipidemia",
                           "hypercholesterolemia", "elevated lipids", "high ldl"],
               "category": "Endocrine"},

    "E78.00": {"desc": "Pure hypercholesterolemia, unspecified",
               "aliases": ["pure hypercholesterolemia", "familial hypercholesterolemia",
                           "elevated cholesterol"],
               "category": "Endocrine"},

    "E03.9":  {"desc": "Hypothyroidism, unspecified",
               "aliases": ["hypothyroidism", "underactive thyroid", "low thyroid"],
               "category": "Endocrine"},

    "E05.90": {"desc": "Thyrotoxicosis, unspecified, without thyrotoxic crisis",
               "aliases": ["hyperthyroidism", "overactive thyroid", "thyrotoxicosis",
                           "graves disease"],
               "category": "Endocrine"},

    "E66.9":  {"desc": "Obesity, unspecified",
               "aliases": ["obesity", "morbid obesity", "bmi over 30"],
               "category": "Endocrine"},

    # ── Respiratory (J00–J99) ──────────────────────────────────────────────
    "J18.9":  {"desc": "Pneumonia, unspecified organism",
               "aliases": ["pneumonia", "community acquired pneumonia", "cap",
                           "lung infection", "pneumonitis"],
               "category": "Respiratory"},

    "J44.1":  {"desc": "Chronic obstructive pulmonary disease with acute exacerbation",
               "aliases": ["copd exacerbation", "acute copd", "acute exacerbation copd",
                           "aecopd", "copd flare"],
               "category": "Respiratory"},

    "J44.9":  {"desc": "Chronic obstructive pulmonary disease, unspecified",
               "aliases": ["copd", "chronic obstructive pulmonary disease",
                           "emphysema", "chronic bronchitis"],
               "category": "Respiratory"},

    "J45.901":{"desc": "Unspecified asthma with acute exacerbation",
               "aliases": ["asthma exacerbation", "asthma attack", "acute asthma"],
               "category": "Respiratory"},

    "J45.909":{"desc": "Unspecified asthma, uncomplicated",
               "aliases": ["asthma", "bronchial asthma", "reactive airway disease"],
               "category": "Respiratory"},

    "J96.00": {"desc": "Acute respiratory failure, unspecified whether with hypoxia or hypercapnia",
               "aliases": ["acute respiratory failure", "arf", "respiratory failure",
                           "hypoxic respiratory failure"],
               "category": "Respiratory"},

    "G47.33": {"desc": "Obstructive sleep apnea",
               "aliases": ["obstructive sleep apnea", "osa", "sleep apnea",
                           "sleep disordered breathing"],
               "category": "Neurological"},

    # ── Renal (N00–N39) ────────────────────────────────────────────────────
    "N17.9":  {"desc": "Acute kidney injury, unspecified",
               "aliases": ["acute kidney injury", "aki", "acute renal failure",
                           "acute kidney failure", "arf renal"],
               "category": "Renal"},

    "N18.3":  {"desc": "Chronic kidney disease, stage 3",
               "aliases": ["chronic kidney disease stage 3", "ckd stage 3", "ckd 3"],
               "category": "Renal"},

    "N18.9":  {"desc": "Chronic kidney disease, unspecified",
               "aliases": ["chronic kidney disease", "ckd", "chronic renal failure",
                           "chronic renal insufficiency", "renal insufficiency"],
               "category": "Renal"},

    "N39.0":  {"desc": "Urinary tract infection, site not specified",
               "aliases": ["urinary tract infection", "uti", "bladder infection",
                           "cystitis", "urosepsis"],
               "category": "Renal"},

    # ── GI (K00–K95) ──────────────────────────────────────────────────────
    "K21.0":  {"desc": "Gastro-esophageal reflux disease with oesophagitis",
               "aliases": ["gerd", "gastroesophageal reflux", "acid reflux",
                           "reflux esophagitis", "heartburn"],
               "category": "GI"},

    "K29.70": {"desc": "Gastritis, unspecified, without bleeding",
               "aliases": ["gastritis", "stomach inflammation"],
               "category": "GI"},

    "K92.1":  {"desc": "Melena",
               "aliases": ["gi bleed", "gastrointestinal bleed", "lower gi bleed",
                           "upper gi bleed", "gastrointestinal hemorrhage"],
               "category": "GI"},

    "K50.90": {"desc": "Crohn's disease of small intestine, unspecified",
               "aliases": ["crohns disease", "crohn disease", "regional enteritis"],
               "category": "GI"},

    "K51.90": {"desc": "Ulcerative colitis, unspecified",
               "aliases": ["ulcerative colitis", "uc", "colitis"],
               "category": "GI"},

    "K80.20": {"desc": "Calculus of gallbladder without cholecystitis",
               "aliases": ["gallstones", "cholelithiasis", "gallbladder stones"],
               "category": "GI"},

    "K85.9":  {"desc": "Acute pancreatitis, unspecified",
               "aliases": ["pancreatitis", "acute pancreatitis"],
               "category": "GI"},

    # ── Infectious Disease (A00–B99) ───────────────────────────────────────
    "A41.9":  {"desc": "Sepsis, unspecified organism",
               "aliases": ["sepsis", "septicemia", "blood poisoning", "bacteremia"],
               "category": "Infectious"},

    "A41.51": {"desc": "Sepsis due to Escherichia coli",
               "aliases": ["e coli sepsis", "gram negative sepsis", "urosepsis"],
               "category": "Infectious"},

    "B20":    {"desc": "Human immunodeficiency virus disease",
               "aliases": ["hiv", "aids", "hiv disease", "human immunodeficiency virus"],
               "category": "Infectious"},

    "J12.89": {"desc": "Other viral pneumonia",
               "aliases": ["viral pneumonia", "covid pneumonia", "covid-19 pneumonia"],
               "category": "Infectious"},

    # ── Neurological (G00–G99) ─────────────────────────────────────────────
    "G20":    {"desc": "Parkinson's disease",
               "aliases": ["parkinsons disease", "parkinson disease", "parkinsonism"],
               "category": "Neurological"},

    "G30.9":  {"desc": "Alzheimer's disease, unspecified",
               "aliases": ["alzheimers disease", "alzheimer disease", "dementia alzheimer"],
               "category": "Neurological"},

    "F03.90": {"desc": "Unspecified dementia without behavioral disturbance",
               "aliases": ["dementia", "cognitive impairment", "memory loss dementia"],
               "category": "Neurological"},

    "G40.909":{"desc": "Epilepsy, unspecified, not intractable",
               "aliases": ["epilepsy", "seizure disorder", "seizures"],
               "category": "Neurological"},

    "G35":    {"desc": "Multiple sclerosis",
               "aliases": ["multiple sclerosis", "ms", "demyelinating disease"],
               "category": "Neurological"},

    # ── Mental Health (F00–F99) ────────────────────────────────────────────
    "F32.9":  {"desc": "Major depressive disorder, single episode, unspecified",
               "aliases": ["depression", "major depression", "major depressive disorder",
                           "depressive disorder", "mdd"],
               "category": "Mental Health"},

    "F41.1":  {"desc": "Generalized anxiety disorder",
               "aliases": ["anxiety", "generalized anxiety", "anxiety disorder", "gad"],
               "category": "Mental Health"},

    "F20.9":  {"desc": "Schizophrenia, unspecified",
               "aliases": ["schizophrenia", "psychosis", "psychotic disorder"],
               "category": "Mental Health"},

    # ── Oncology (C00–D49) ─────────────────────────────────────────────────
    "C34.90": {"desc": "Malignant neoplasm of unspecified bronchus and lung",
               "aliases": ["lung cancer", "lung carcinoma", "pulmonary malignancy",
                           "bronchogenic carcinoma"],
               "category": "Oncology"},

    "C50.919":{"desc": "Malignant neoplasm of unspecified site of breast",
               "aliases": ["breast cancer", "breast carcinoma", "breast malignancy"],
               "category": "Oncology"},

    "C61":    {"desc": "Malignant neoplasm of prostate",
               "aliases": ["prostate cancer", "prostate carcinoma"],
               "category": "Oncology"},

    "C18.9":  {"desc": "Malignant neoplasm of colon, unspecified",
               "aliases": ["colon cancer", "colorectal cancer", "colon carcinoma"],
               "category": "Oncology"},

    # ── Musculoskeletal (M00–M99) ──────────────────────────────────────────
    "M16.9":  {"desc": "Osteoarthritis of hip, unspecified",
               "aliases": ["hip osteoarthritis", "hip arthritis", "degenerative hip"],
               "category": "Musculoskeletal"},

    "M17.9":  {"desc": "Osteoarthritis of knee, unspecified",
               "aliases": ["knee osteoarthritis", "knee arthritis", "degenerative knee"],
               "category": "Musculoskeletal"},

    "M79.3":  {"desc": "Panniculitis",
               "aliases": ["back pain", "lower back pain", "lbp", "lumbar pain"],
               "category": "Musculoskeletal"},

    "M06.9":  {"desc": "Rheumatoid arthritis, unspecified",
               "aliases": ["rheumatoid arthritis", "ra", "inflammatory arthritis"],
               "category": "Musculoskeletal"},

    # ── Hematology (D50–D89) ──────────────────────────────────────────────
    "D64.9":  {"desc": "Anaemia, unspecified",
               "aliases": ["anemia", "anaemia", "low hemoglobin", "iron deficiency anemia"],
               "category": "Hematology"},

    "D69.6":  {"desc": "Thrombocytopenia, unspecified",
               "aliases": ["thrombocytopenia", "low platelets", "low platelet count"],
               "category": "Hematology"},

    # ── Symptoms & Signs (R00–R99) ─────────────────────────────────────────
    "R07.9":  {"desc": "Chest pain, unspecified",
               "aliases": ["chest pain", "chest discomfort", "thoracic pain"],
               "category": "Symptoms"},

    "R06.00": {"desc": "Dyspnea, unspecified",
               "aliases": ["shortness of breath", "dyspnea", "breathlessness",
                           "difficulty breathing"],
               "category": "Symptoms"},

    "R55":    {"desc": "Syncope and collapse",
               "aliases": ["syncope", "fainting", "loss of consciousness", "collapse"],
               "category": "Symptoms"},

    "R51":    {"desc": "Headache, unspecified",
               "aliases": ["headache", "cephalgia", "head pain"],
               "category": "Symptoms"},

    "R11.2":  {"desc": "Nausea with vomiting, unspecified",
               "aliases": ["nausea and vomiting", "nausea", "vomiting"],
               "category": "Symptoms"},

    "R50.9":  {"desc": "Fever, unspecified",
               "aliases": ["fever", "pyrexia", "high temperature", "febrile"],
               "category": "Symptoms"},

    # ── Injuries & External (S00–Z99) ─────────────────────────────────────
    "Z87.891":{"desc": "Personal history of nicotine dependence",
               "aliases": ["former smoker", "ex smoker", "tobacco use history",
                           "smoking history"],
               "category": "History"},

    "Z82.49": {"desc": "Family history of ischaemic heart disease",
               "aliases": ["family history of heart disease", "family history cad",
                           "family history myocardial infarction"],
               "category": "History"},

    "Z79.4":  {"desc": "Long-term (current) use of insulin",
               "aliases": ["on insulin", "insulin dependent", "insulin therapy",
                           "long term insulin"],
               "category": "History"},
}

# Build a reverse lookup: alias → ICD code (for fast Layer 1 exact match)
ALIAS_TO_ICD = {}
for code, info in ICD10_DATABASE.items():
    for alias in info["aliases"]:
        ALIAS_TO_ICD[alias.lower().strip()] = code


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ICDMatch:
    """A single ICD-10 code match for a diagnosis."""
    diagnosis:    str           # Input diagnosis text
    icd_code:     str           # ICD-10-CM code (e.g. "E11.9")
    description:  str           # Official ICD description
    category:     str           # Disease category
    confidence:   float         # Match confidence 0.0–1.0
    match_type:   str           # "exact", "fuzzy", "semantic", "no_match"
    alternatives: List[Dict] = field(default_factory=list)  # Runner-up matches


@dataclass
class ICDMappingResult:
    """Complete ICD-10 mapping output for all diagnoses in a report."""
    mappings:         List[ICDMatch]
    mapped_count:     int
    unmapped:         List[str]       # Diagnoses that couldn't be mapped
    categories_found: List[str]       # Unique disease categories
    code_summary:     Dict[str, str]  # {code: description} for display


# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 Mapper
# ─────────────────────────────────────────────────────────────────────────────

class ICD10Mapper:
    """
    3-layer ICD-10-CM code mapping system.

    Layer 1 — Exact alias match (fastest, highest confidence)
    Layer 2 — Fuzzy string matching (handles variations, abbreviations)
    Layer 3 — TF-IDF semantic similarity (catches novel phrasings)

    Usage:
        mapper = ICD10Mapper()
        result = mapper.map(ner_result)

        for match in result.mappings:
            print(f"{match.diagnosis} → {match.icd_code} [{match.confidence:.0%}]")
    """

    def __init__(self, confidence_threshold: float = 0.40):
        self.confidence_threshold = confidence_threshold
        self.icd_db = ICD10_DATABASE
        self.alias_map = ALIAS_TO_ICD
        self._build_tfidf_index()
        logger.info("✅ ICD10Mapper initialized")

    def _build_tfidf_index(self) -> None:
        """Build a simple TF-IDF index over all ICD descriptions + aliases."""
        self._all_codes = []
        self._all_texts = []

        for code, info in self.icd_db.items():
            # Combine description + all aliases into one searchable string
            combined = info["desc"].lower() + " " + " ".join(info["aliases"])
            self._all_codes.append(code)
            self._all_texts.append(combined)

        # Build vocabulary
        self._vocab = {}
        for text in self._all_texts:
            for word in re.findall(r'\b[a-z]+\b', text):
                if word not in self._vocab:
                    self._vocab[word] = len(self._vocab)

        # Compute TF-IDF vectors
        self._tfidf_matrix = []
        N = len(self._all_texts)

        # IDF
        idf = {}
        for word in self._vocab:
            df = sum(1 for text in self._all_texts if word in text.split())
            idf[word] = math.log((N + 1) / (df + 1)) + 1

        for text in self._all_texts:
            words = re.findall(r'\b[a-z]+\b', text)
            word_count = len(words)
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1

            vec = {}
            for w, count in tf.items():
                if w in self._vocab:
                    vec[self._vocab[w]] = (count / word_count) * idf.get(w, 1)

            self._tfidf_matrix.append(vec)

        logger.debug(f"🔢 TF-IDF index built: {len(self._all_codes)} codes, {len(self._vocab)} terms")

    @timer
    def map(self, ner_result) -> ICDMappingResult:
        """
        Map all extracted diseases to ICD-10 codes.

        Args:
            ner_result: NERResult from Module A NER extraction

        Returns:
            ICDMappingResult with all mappings
        """
        # Collect all diagnoses to map
        all_diagnoses = list(set(ner_result.diseases))

        logger.info(f"🏥 Mapping {len(all_diagnoses)} diagnoses to ICD-10")

        mappings = []
        unmapped = []

        for diagnosis in all_diagnoses:
            if not diagnosis or len(diagnosis.strip()) < 3:
                continue

            match = self._map_single(diagnosis)

            if match and match.confidence >= self.confidence_threshold:
                mappings.append(match)
            else:
                unmapped.append(diagnosis)
                # Still add as no_match so it appears in output
                mappings.append(ICDMatch(
                    diagnosis=diagnosis,
                    icd_code="",
                    description="No mapping found",
                    category="Unknown",
                    confidence=0.0,
                    match_type="no_match"
                ))

        # Sort by confidence descending
        mappings.sort(key=lambda x: x.confidence, reverse=True)

        mapped = [m for m in mappings if m.icd_code]
        categories = list(set(m.category for m in mapped))

        code_summary = {
            m.icd_code: m.description
            for m in mapped
            if m.icd_code
        }

        logger.info(
            f"✅ ICD mapping complete — {len(mapped)}/{len(all_diagnoses)} mapped, "
            f"{len(unmapped)} unmatched"
        )

        return ICDMappingResult(
            mappings=mappings,
            mapped_count=len(mapped),
            unmapped=unmapped,
            categories_found=categories,
            code_summary=code_summary,
        )

    def map_single_text(self, diagnosis_text: str) -> Optional[ICDMatch]:
        """Map a single diagnosis string. Useful for one-off lookups."""
        return self._map_single(diagnosis_text)

    def _map_single(self, diagnosis: str) -> Optional[ICDMatch]:
        """Run all 3 layers for one diagnosis, return best match."""

        # ── Layer 1: Exact alias match ─────────────────────────────────────
        exact = self._layer1_exact(diagnosis)
        if exact and exact.confidence >= 0.90:
            return exact

        # ── Layer 2: Fuzzy string match ────────────────────────────────────
        fuzzy = self._layer2_fuzzy(diagnosis)
        if fuzzy and fuzzy.confidence >= 0.70:
            return fuzzy

        # ── Layer 3: TF-IDF semantic match ────────────────────────────────
        semantic = self._layer3_tfidf(diagnosis)
        if semantic:
            return semantic

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 1: Exact Match
    # ─────────────────────────────────────────────────────────────────────────

    def _layer1_exact(self, diagnosis: str) -> Optional[ICDMatch]:
        """
        Direct lookup in alias dictionary.
        Handles exact strings and simple normalizations.
        """
        # Normalize input
        normalized = diagnosis.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove common suffixes that don't change meaning
        normalized = re.sub(r',?\s*(unspecified|nos|nec|not otherwise specified)$', '', normalized)
        normalized = normalized.strip()

        # Direct match
        if normalized in self.alias_map:
            code = self.alias_map[normalized]
            return self._build_match(diagnosis, code, 0.97, "exact")

        # Try without trailing qualifiers
        for qualifier in [" mellitus", " disease", " disorder", " syndrome", " type"]:
            trimmed = normalized.replace(qualifier, "")
            if trimmed in self.alias_map:
                code = self.alias_map[trimmed]
                return self._build_match(diagnosis, code, 0.90, "exact_trimmed")

        # Try key abbreviations
        abbrev_map = {
            "stemi": "I21.19", "nstemi": "I21.4",
            "afib": "I48.91", "chf": "I50.9",
            "cad": "I25.10", "copd": "J44.9",
            "osa": "G47.33", "uti": "N39.0",
            "aki": "N17.9", "ckd": "N18.9",
            "htn": "I10", "dm2": "E11.9",
            "t2dm": "E11.9", "t1dm": "E10.9",
            "dvt": "I82.401", "pe": "I26.99",
            "cva": "I63.9", "tia": "G45.9",
            "mi": "I21.9",
        }
        if normalized in abbrev_map:
            code = abbrev_map[normalized]
            return self._build_match(diagnosis, code, 0.93, "abbreviation")

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 2: Fuzzy String Match
    # ─────────────────────────────────────────────────────────────────────────

    def _layer2_fuzzy(self, diagnosis: str) -> Optional[ICDMatch]:
        """
        Token-based fuzzy matching.
        Handles word order variations, partial matches, and minor typos.
        """
        diagnosis_words = set(re.findall(r'\b[a-z]+\b', diagnosis.lower()))

        # Remove common filler words
        stop_words = {"the", "a", "an", "of", "with", "without", "and", "or",
                      "in", "at", "to", "for", "is", "was", "be", "been",
                      "unspecified", "acute", "chronic", "primary", "secondary"}
        diag_tokens = diagnosis_words - stop_words

        if not diag_tokens:
            return None

        best_score = 0.0
        best_code = None

        for code, info in self.icd_db.items():
            # Check against description
            desc_words = set(re.findall(r'\b[a-z]+\b', info["desc"].lower())) - stop_words

            # Check against all aliases
            alias_words_list = [
                set(re.findall(r'\b[a-z]+\b', alias.lower())) - stop_words
                for alias in info["aliases"]
            ]

            # Jaccard similarity with description
            if desc_words:
                intersection = diag_tokens & desc_words
                union = diag_tokens | desc_words
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard > best_score:
                    best_score = jaccard
                    best_code = code

            # Jaccard similarity with aliases
            for alias_words in alias_words_list:
                if alias_words:
                    intersection = diag_tokens & alias_words
                    union = diag_tokens | alias_words
                    jaccard = len(intersection) / len(union) if union else 0

                    # Bonus: if all diagnosis tokens match alias tokens
                    if diag_tokens.issubset(alias_words):
                        jaccard = min(jaccard * 1.3, 1.0)

                    if jaccard > best_score:
                        best_score = jaccard
                        best_code = code

        if best_code and best_score >= 0.35:
            # Scale score to confidence range 0.65–0.88
            confidence = 0.65 + (best_score * 0.23)
            return self._build_match(diagnosis, best_code, min(confidence, 0.88), "fuzzy")

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 3: TF-IDF Semantic Match
    # ─────────────────────────────────────────────────────────────────────────

    def _layer3_tfidf(self, diagnosis: str) -> Optional[ICDMatch]:
        """
        TF-IDF cosine similarity against all ICD descriptions.
        Handles completely novel phrasings not in alias dictionary.
        """
        # Vectorize the query
        query_words = re.findall(r'\b[a-z]+\b', diagnosis.lower())
        if not query_words:
            return None

        query_tf = {}
        for w in query_words:
            query_tf[w] = query_tf.get(w, 0) + 1

        query_vec = {}
        for w, count in query_tf.items():
            if w in self._vocab:
                query_vec[self._vocab[w]] = count / len(query_words)

        if not query_vec:
            return None

        # Cosine similarity with all ICD vectors
        best_score = 0.0
        best_idx = None

        q_norm = math.sqrt(sum(v**2 for v in query_vec.values()))
        if q_norm == 0:
            return None

        for idx, doc_vec in enumerate(self._tfidf_matrix):
            # Dot product
            dot = sum(query_vec.get(k, 0) * v for k, v in doc_vec.items())
            if dot == 0:
                continue

            d_norm = math.sqrt(sum(v**2 for v in doc_vec.values()))
            if d_norm == 0:
                continue

            score = dot / (q_norm * d_norm)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= 0.20:
            code = self._all_codes[best_idx]
            confidence = min(0.40 + best_score * 0.35, 0.78)
            return self._build_match(diagnosis, code, confidence, "semantic")

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def _build_match(
        self, diagnosis: str, code: str, confidence: float, match_type: str
    ) -> ICDMatch:
        """Build ICDMatch from code + confidence."""
        info = self.icd_db.get(code, {})
        return ICDMatch(
            diagnosis=diagnosis,
            icd_code=code,
            description=info.get("desc", ""),
            category=info.get("category", "Unknown"),
            confidence=round(confidence, 3),
            match_type=match_type,
        )

    def get_code_info(self, code: str) -> Optional[Dict]:
        """Look up a specific ICD-10 code."""
        return self.icd_db.get(code)

    def search_by_keyword(self, keyword: str, top_n: int = 5) -> List[ICDMatch]:
        """Search ICD-10 database by keyword — useful for Streamlit search feature."""
        results = []
        keyword_lower = keyword.lower()

        for code, info in self.icd_db.items():
            if keyword_lower in info["desc"].lower() or \
               any(keyword_lower in alias.lower() for alias in info["aliases"]):
                results.append(self._build_match(keyword, code, 0.85, "search"))

        return results[:top_n]

    def format_output(self, result: ICDMappingResult) -> str:
        """Pretty-print ICD mapping result to console."""
        lines = []
        lines.append(f"\n{'='*65}")
        lines.append(f"  ICD-10 CODE MAPPING — {result.mapped_count} diagnoses mapped")
        lines.append(f"{'='*65}\n")

        for m in result.mappings:
            if m.icd_code:
                conf_bar = "█" * int(m.confidence * 10) + "░" * (10 - int(m.confidence * 10))
                lines.append(f"  📋 {m.diagnosis}")
                lines.append(f"     Code:        {m.icd_code}")
                lines.append(f"     Description: {m.description}")
                lines.append(f"     Category:    {m.category}")
                lines.append(f"     Confidence:  {conf_bar} {m.confidence:.0%} ({m.match_type})")
                lines.append("")
            else:
                lines.append(f"  ❓ {m.diagnosis}  →  No ICD-10 match found\n")

        if result.unmapped:
            lines.append(f"  ⚠️  Unmapped: {', '.join(result.unmapped)}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from modules.preprocessor import MedicalPreprocessor
    from modules.ner_extractor import MedicalNERExtractor

    sample = """
    ASSESSMENT AND PLAN:
    1. Inferior STEMI — drug-eluting stent placed in right coronary artery.
    2. Type 2 Diabetes Mellitus — HbA1c 9.2%, poorly controlled.
    3. Hypertension — target blood pressure less than 130/80 mmHg.
    4. Hyperlipidemia — LDL 142 mg/dL, increase statin.
    5. Obstructive sleep apnea — continue CPAP therapy.
    6. Heart failure — ejection fraction 45%.
    """

    print("🧪 Running Module D smoke test...\n")

    preprocessor = MedicalPreprocessor()
    ner = MedicalNERExtractor()
    mapper = ICD10Mapper()

    preprocessed = preprocessor.process(sample)
    ner_result = ner.extract(preprocessed.clean_text)

    print(f"🔍 Diseases found by NER: {ner_result.diseases}")
    print()

    result = mapper.map(ner_result)
    print(mapper.format_output(result))

    print(f"\n📊 Summary:")
    print(f"  Mapped:     {result.mapped_count}")
    print(f"  Unmapped:   {result.unmapped}")
    print(f"  Categories: {result.categories_found}")
    print(f"\n🔎 Code lookup — search 'diabetes':")
    for match in mapper.search_by_keyword("diabetes", top_n=3):
        print(f"  {match.icd_code}: {match.description}")
