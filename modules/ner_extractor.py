# modules/ner_extractor.py
# ─────────────────────────────────────────────────────────────────────────────
# MODULE A — PART 2: Medical Named Entity Recognition (NER)
#
# What this does:
#   1. Loads scispaCy + BioBERT models for medical NER
#   2. Extracts 9 entity types: DISEASE, DRUG, DOSAGE, SYMPTOM,
#      PROCEDURE, LAB_VALUE, BODY_PART, PERSON, DATE
#   3. Deduplicates and normalizes entity text
#   4. Applies rule-based extraction for vitals, medications, lab values
#   5. Returns structured NERResult object
#
# Models Used:
#   Primary : scispaCy  en_ner_bc5cdr_md  (disease + chemical NER)
#   Extended: HuggingFace dslim/bert-base-NER (general NER)
#   Fallback: spaCy en_core_web_sm (if medical model not installed)
# ─────────────────────────────────────────────────────────────────────────────

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import timer


# ─────────────────────────────────────────────────────────────────────────────
# Entity Type Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from scispaCy / model labels → our standardized labels
ENTITY_LABEL_MAP = {
    # scispaCy BC5CDR labels
    "DISEASE":       "DISEASE",
    "CHEMICAL":      "DRUG",

    # scispaCy sci_lg labels
    "CANCER":        "DISEASE",
    "CELL_TYPE":     "BODY_PART",
    "CELL_LINE":     "BODY_PART",
    "DNA":           "LAB_VALUE",
    "PROTEIN":       "DRUG",
    "RNA":           "LAB_VALUE",

    # BioBERT / general NER labels
    "B-Disease":     "DISEASE",
    "I-Disease":     "DISEASE",
    "B-Chemical":    "DRUG",
    "I-Chemical":    "DRUG",

    # spaCy standard labels (fallback)
    "PERSON":        "PERSON",
    "DATE":          "DATE",
    "GPE":           "LOCATION",
    "ORG":           "ORGANIZATION",
    "CARDINAL":      "NUMBER",
}

# Colors for Streamlit display (entity type → hex color)
ENTITY_COLORS = {
    "DISEASE":      "#FF6B6B",   # Red
    "DRUG":         "#4ECDC4",   # Teal
    "DOSAGE":       "#45B7D1",   # Blue
    "SYMPTOM":      "#FFA07A",   # Orange
    "PROCEDURE":    "#98D8C8",   # Green
    "LAB_VALUE":    "#DDA0DD",   # Purple
    "BODY_PART":    "#F0E68C",   # Yellow
    "VITAL":        "#87CEEB",   # Sky Blue
    "PERSON":       "#D3D3D3",   # Gray
    "DATE":         "#C8A2C8",   # Lilac
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A single extracted named entity."""
    text: str                  # The entity text as found in document
    label: str                 # Standardized entity type (DISEASE, DRUG, etc.)
    start: int                 # Character start position in text
    end: int                   # Character end position in text
    confidence: float = 1.0    # Confidence score (0.0 - 1.0)
    normalized: str = ""       # Normalized/canonical form
    source: str = "model"      # "model", "rule", or "pattern"


@dataclass
class MedicationEntity:
    """Structured medication with all components."""
    name: str
    dosage: str = ""
    frequency: str = ""
    route: str = ""
    raw_text: str = ""


@dataclass
class LabValue:
    """Structured lab result."""
    test_name: str
    value: str
    unit: str = ""
    flag: str = ""     # "HIGH", "LOW", "NORMAL"
    raw_text: str = ""


@dataclass
class NERResult:
    """Complete NER output for a medical report."""
    entities: List[Entity]
    diseases: List[str]
    drugs: List[str]
    symptoms: List[str]
    procedures: List[str]
    body_parts: List[str]
    dates: List[str]

    # Structured extractions
    medications: List[MedicationEntity]
    lab_values: List[LabValue]
    vitals: Dict[str, str]

    # Metadata
    model_used: str = ""
    entity_count: int = 0
    raw_entity_spans: List = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# NER Extractor Class
# ─────────────────────────────────────────────────────────────────────────────

class MedicalNERExtractor:
    """
    Production-grade Medical NER system with multiple model support.

    Priority:
      1. scispaCy BC5CDR (best for disease + drug NER)
      2. scispaCy sci_lg (broader medical NER)
      3. HuggingFace BioBERT (transformer-based, high accuracy)
      4. spaCy en_core_web_sm (fallback — limited medical awareness)
      + Rule-based patterns (always runs, catches what models miss)

    Usage:
        extractor = MedicalNERExtractor()
        result = extractor.extract(preprocessed_report.clean_text)

        # Access specific entity types
        print(result.diseases)
        print(result.medications)
        print(result.lab_values)
    """

    def __init__(
        self,
        use_transformer: bool = False,    # Set True for BioBERT (slower but accurate)
        use_gpu: bool = False,
        confidence_threshold: float = 0.5
    ):
        self.use_transformer = use_transformer
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.nlp = None
        self.transformer_pipeline = None
        self.model_name = ""

        self._compile_patterns()
        self._load_model()

    def _load_model(self) -> None:
        """Load best available NLP model, falling back gracefully."""

        # Attempt 1: scispaCy BC5CDR (disease + chemical NER)
        try:
            import spacy
            self.nlp = spacy.load("en_ner_bc5cdr_md")
            self.model_name = "scispaCy BC5CDR (Disease + Drug NER)"
            logger.info(f"✅ Loaded: {self.model_name}")
            return
        except OSError:
            logger.warning("⚠️  en_ner_bc5cdr_md not found")

        # Attempt 2: scispaCy large science model
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_lg")
            self.model_name = "scispaCy Large Science Model"
            logger.info(f"✅ Loaded: {self.model_name}")
            return
        except OSError:
            logger.warning("⚠️  en_core_sci_lg not found")

        # Attempt 3: scispaCy small science model
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
            self.model_name = "scispaCy Small Science Model"
            logger.info(f"✅ Loaded: {self.model_name}")
            return
        except OSError:
            logger.warning("⚠️  en_core_sci_sm not found")

        # Attempt 4: Standard spaCy (limited medical awareness)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.model_name = "spaCy English (fallback)"
            logger.warning(f"⚠️  Using fallback model: {self.model_name}")
            return
        except OSError:
            logger.warning("⚠️  en_core_web_sm not found")
        except ImportError:
            logger.warning("⚠️  spaCy not installed")

        # Attempt 5: Pure regex — no spaCy at all (Streamlit Cloud safe)
        self.nlp = None
        self.model_name = "Rule-Based Regex NER (Cloud Mode)"
        logger.warning("⚠️  Running in regex-only mode — no spaCy model loaded")

        # Attempt 6: HuggingFace BioBERT (no spaCy needed)
        if self.use_transformer:
            self._load_transformer()

    def _load_transformer(self) -> None:
        """Load HuggingFace BioBERT for NER."""
        try:
            from transformers import pipeline
            model_id = "d4data/biomedical-ner-all"
            self.transformer_pipeline = pipeline(
                "ner",
                model=model_id,
                aggregation_strategy="simple",
                device=0 if self.use_gpu else -1
            )
            self.model_name = f"HuggingFace: {model_id}"
            logger.info(f"✅ Loaded transformer NER: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load transformer NER: {e}")

    def _compile_patterns(self) -> None:
        """Compile rule-based patterns for entities models often miss."""

        # Medication + dosage pattern
        # NO re.IGNORECASE — drug names must be truly capitalized (proper nouns)
        # This prevents "on", "and", "cholesterol" from matching as drug names
        # Units use explicit alternation for both short and expanded forms
        self.med_pattern = re.compile(
            r'(?<!\w)'                                          # Not preceded by word char
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+'      # Capitalized drug name ONLY
            r'(\d+(?:\.\d+)?)\s*'                              # Numeric dosage
            r'(mg|mcg|g|ml|mEq|IU|milligrams?|micrograms?|grams?|milliliters?|units?)\s*'
            r'(once|twice|three times|daily|weekly|monthly|q\d+h|prn|stat)?'
            r'(?:\s+(by mouth|intravenous|oral|iv|po|im|sc|topical))?'
            # No IGNORECASE flag here
        )

        # Words that look like drugs but are NOT medications
        self.drug_blocklist = {
            "oxygen", "air", "water", "blood", "glucose", "cholesterol",
            "sodium", "potassium", "calcium", "magnesium", "protein",
            "hemoglobin", "bilirubin", "creatinine", "lactate", "albumin",
            "carbon", "nitrogen", "phosphorus", "iron", "zinc", "fiber",
            "the", "a", "an", "this", "that", "he", "she", "patient",
            "room", "normal", "saline", "contrast", "dye", "food",
        }

        # Words that are symptoms, NOT diagnoses
        self.symptom_blocklist = {
            "pain", "chest pain", "shortness of breath", "dyspnea",
            "nausea", "vomiting", "diarrhea", "headache", "dizziness",
            "fatigue", "weakness", "fever", "chills", "cough", "wheezing",
            "diaphoresis", "sweating", "swelling", "edema", "palpitations",
            "syncope", "confusion", "anxiety", "numbness", "tingling",
            "blurred vision", "loss of appetite", "constipation",
        }

        # Lab value pattern
        # Matches: "Sodium 138 mEq/L" / "HbA1c 9.2%" / "WBC 11.2 K/uL"
        self.lab_pattern = re.compile(
            r'\b(Sodium|Potassium|Chloride|Glucose|BUN|Creatinine|Calcium|'
            r'Magnesium|Hemoglobin|Hematocrit|Platelets|WBC|'
            r'ALT|AST|ALP|Bilirubin|INR|HbA1c|TSH|Troponin|BNP|CRP|ESR|'
            r'LDL|HDL|Triglycerides|Albumin|Lactate|Procalcitonin)\s+'
            r'(?:of\s+)?(\d+(?:\.\d+)?)\s*([a-zA-Z\/\%µg]*)',
            re.IGNORECASE
        )

        # Vital signs pattern
        self.vitals_pattern = {
            "Blood Pressure":    re.compile(r'\bblood pressure[:\s]+(\d{2,3}/\d{2,3})\s*(?:mmHg)?', re.IGNORECASE),
            "Heart Rate":        re.compile(r'\bheart rate[:\s]+(\d{2,3})\s*(?:bpm)?', re.IGNORECASE),
            "Respiratory Rate":  re.compile(r'\brespiratory rate[:\s]+(\d{1,2})\s*(?:rpm|breaths)?', re.IGNORECASE),
            "Temperature":       re.compile(r'\btemperature[:\s]+(\d{2,3}\.?\d*\s*°?[FCfc]?)', re.IGNORECASE),
            "Oxygen Saturation": re.compile(r'\boxygen saturation[:\s]+(\d{2,3})\s*(?:%)?', re.IGNORECASE),
            "BMI":               re.compile(r'\bbody mass index[:\s]+(\d{2,3}\.?\d*)', re.IGNORECASE),
            "Weight":            re.compile(r'\bweight[:\s]+(\d{2,3}\.?\d*\s*(?:kg|lbs)?)', re.IGNORECASE),
        }

        # Symptom keywords (rule-based catch for common symptoms)
        self.symptom_keywords = [
            "chest pain", "shortness of breath", "dyspnea", "nausea", "vomiting",
            "diarrhea", "headache", "dizziness", "fatigue", "weakness", "fever",
            "chills", "cough", "wheezing", "palpitations", "syncope", "edema",
            "swelling", "pain", "ache", "discomfort", "numbness", "tingling",
            "blurred vision", "confusion", "anxiety", "depression", "insomnia",
            "abdominal pain", "back pain", "joint pain", "muscle pain", "sore throat",
            "runny nose", "nasal congestion", "loss of appetite", "weight loss",
            "weight gain", "polyuria", "polydipsia", "polyphagia", "constipation"
        ]

        # Procedure keywords
        self.procedure_keywords = [
            "echocardiogram", "electrocardiogram", "coronary angiography",
            "percutaneous coronary intervention", "coronary artery bypass",
            "computed tomography", "magnetic resonance imaging", "ultrasound",
            "chest x-ray", "lumbar puncture", "colonoscopy", "endoscopy",
            "biopsy", "surgery", "intubation", "catheterization", "hemodialysis",
            "blood transfusion", "chemotherapy", "radiation therapy"
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Main Extraction Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    @timer
    def extract(self, text: str) -> NERResult:
        """
        Run full NER pipeline on preprocessed medical text.

        Combines model-based NER with rule-based pattern matching.
        Rule-based always runs to catch things models miss.

        Args:
            text: Clean, preprocessed medical report text

        Returns:
            NERResult with all extracted entities organized by type
        """
        logger.info(f"🔍 Running NER on {len(text.split())} words")

        all_entities: List[Entity] = []

        # ── Run model-based NER ────────────────────────────────────────────
        if self.nlp:
            model_entities = self._run_spacy_ner(text)
            all_entities.extend(model_entities)
            logger.info(f"🤖 Model extracted {len(model_entities)} entities")

        if self.transformer_pipeline:
            transformer_entities = self._run_transformer_ner(text)
            all_entities.extend(transformer_entities)
            logger.info(f"🧠 Transformer extracted {len(transformer_entities)} entities")

        # ── Run rule-based extraction ──────────────────────────────────────
        rule_entities = self._run_rule_based(text)
        all_entities.extend(rule_entities)
        logger.info(f"📏 Rules extracted {len(rule_entities)} entities")

        # ── Deduplicate & filter ───────────────────────────────────────────
        all_entities = self._deduplicate(all_entities)
        all_entities = [e for e in all_entities if e.confidence >= self.confidence_threshold]

        # ── Extract structured items ───────────────────────────────────────
        medications = self._extract_medications(text)
        lab_values = self._extract_lab_values(text)
        vitals = self._extract_vitals(text)
        symptoms = self._extract_symptoms(text)
        procedures = self._extract_procedures(text)

        # ── Organize by entity type ────────────────────────────────────────
        diseases   = self._filter_by_label(all_entities, "DISEASE")
        drugs      = [d for d in self._filter_by_label(all_entities, "DRUG")
                      if d.lower() not in self.drug_blocklist and len(d) > 2]
        body_parts = self._filter_by_label(all_entities, "BODY_PART")
        dates      = self._filter_by_label(all_entities, "DATE")

        result = NERResult(
            entities=all_entities,
            diseases=diseases,
            drugs=drugs,
            symptoms=symptoms,
            procedures=procedures,
            body_parts=body_parts,
            dates=dates,
            medications=medications,
            lab_values=lab_values,
            vitals=vitals,
            model_used=self.model_name,
            entity_count=len(all_entities),
        )

        logger.info(
            f"✅ NER complete — {len(diseases)} diseases, {len(drugs)} drugs, "
            f"{len(medications)} medications, {len(lab_values)} lab values"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Model-Based NER
    # ─────────────────────────────────────────────────────────────────────────

    def _run_spacy_ner(self, text: str) -> List[Entity]:
        """Run spaCy/scispaCy NER model."""
        entities = []
        try:
            # Process in chunks to avoid max token length issues
            chunks = self._chunk_text(text, max_chars=100000)
            offset = 0

            for chunk in chunks:
                doc = self.nlp(chunk)
                for ent in doc.ents:
                    # Map model label to our standardized label
                    label = ENTITY_LABEL_MAP.get(ent.label_, ent.label_)

                    entities.append(Entity(
                        text=ent.text.strip(),
                        label=label,
                        start=ent.start_char + offset,
                        end=ent.end_char + offset,
                        confidence=0.85,   # spaCy doesn't return confidence by default
                        source="spacy"
                    ))
                offset += len(chunk)
        except Exception as e:
            logger.error(f"❌ spaCy NER failed: {e}")

        return entities

    def _run_transformer_ner(self, text: str) -> List[Entity]:
        """Run HuggingFace transformer NER pipeline."""
        entities = []
        try:
            # HuggingFace pipeline has 512 token limit — chunk accordingly
            chunks = self._chunk_text(text, max_chars=1500)  # ~500 tokens
            offset = 0

            for chunk in chunks:
                results = self.transformer_pipeline(chunk)
                for r in results:
                    label = ENTITY_LABEL_MAP.get(r.get("entity_group", ""), r.get("entity_group", "UNKNOWN"))
                    entities.append(Entity(
                        text=r.get("word", "").strip(),
                        label=label,
                        start=r.get("start", 0) + offset,
                        end=r.get("end", 0) + offset,
                        confidence=float(r.get("score", 0.5)),
                        source="transformer"
                    ))
                offset += len(chunk)
        except Exception as e:
            logger.error(f"❌ Transformer NER failed: {e}")

        return entities

    # ─────────────────────────────────────────────────────────────────────────
    # Rule-Based Extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _run_rule_based(self, text: str) -> List[Entity]:
        """
        Pattern-based NER for high-precision extraction of structured entities.
        These rules catch what statistical models often miss.
        """
        entities = []
        text_lower = text.lower()

        # Extract symptom mentions
        for symptom in self.symptom_keywords:
            if symptom in text_lower:
                idx = text_lower.find(symptom)
                while idx != -1:
                    entities.append(Entity(
                        text=text[idx:idx+len(symptom)],
                        label="SYMPTOM",
                        start=idx,
                        end=idx+len(symptom),
                        confidence=0.9,
                        source="rule"
                    ))
                    idx = text_lower.find(symptom, idx + 1)

        # Extract procedure mentions
        for proc in self.procedure_keywords:
            if proc in text_lower:
                idx = text_lower.find(proc)
                entities.append(Entity(
                    text=text[idx:idx+len(proc)],
                    label="PROCEDURE",
                    start=idx,
                    end=idx+len(proc),
                    confidence=0.9,
                    source="rule"
                ))

        # Extract date patterns
        date_pattern = re.compile(
            r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
            r'|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4})',
            re.IGNORECASE
        )
        for match in date_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0).strip(),
                label="DATE",
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                source="rule"
            ))

        return entities

    # ─────────────────────────────────────────────────────────────────────────
    # Structured Extractions
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_medications(self, text: str) -> List[MedicationEntity]:
        """Extract structured medication records with dosage and frequency."""
        medications = []
        seen = set()

        for match in self.med_pattern.finditer(text):
            name = match.group(1).strip()
            raw_unit = match.group(3).strip().lower()
            # Normalize expanded unit names back to standard abbreviations
            unit_map = {
                "milligrams": "mg", "milligram": "mg",
                "micrograms": "mcg", "microgram": "mcg",
                "grams": "g", "gram": "g",
                "milliliters": "ml", "milliliter": "ml",
                "units": "units", "unit": "units",
            }
            unit = unit_map.get(raw_unit, raw_unit)
            dosage = f"{match.group(2)} {unit}" if match.group(2) else ""
            frequency = match.group(4) or ""
            route = match.group(5) or ""

            # Skip blocklisted substance names
            if name.lower() in self.drug_blocklist:
                continue

            # Skip single words that are clearly not drug names
            if len(name) < 4:
                continue

            # Skip names that start with common English words (caught by IGNORECASE before)
            skip_starters = {"the", "on", "and", "or", "of", "in", "at", "to",
                             "for", "by", "an", "as", "is", "it", "he", "she",
                             "was", "has", "had", "are", "were", "been", "with",
                             "from", "this", "that", "these", "those", "continue",
                             "started", "added", "hold", "resume", "take", "use"}
            if name.lower().split()[0] in skip_starters:
                continue

            # Skip if looks like a sentence starter, not a drug name
            if name.lower() in ["the", "a", "an", "this", "that", "he", "she", "patient"]:
                continue

            # Deduplicate by name
            if name.lower() not in seen:
                seen.add(name.lower())
                medications.append(MedicationEntity(
                    name=name,
                    dosage=dosage.strip(),
                    frequency=frequency.strip(),
                    route=route.strip(),
                    raw_text=match.group(0).strip()
                ))

        return medications

    def _extract_lab_values(self, text: str) -> List[LabValue]:
        """Extract structured lab results with values and units."""
        lab_values = []
        seen = set()

        for match in self.lab_pattern.finditer(text):
            test_name = match.group(1).strip()
            value = match.group(2).strip()
            unit = match.group(3).strip() if match.group(3) else ""

            if test_name.lower() not in seen:
                seen.add(test_name.lower())
                lab_values.append(LabValue(
                    test_name=test_name,
                    value=value,
                    unit=unit,
                    raw_text=match.group(0).strip()
                ))

        return lab_values

    def _extract_vitals(self, text: str) -> Dict[str, str]:
        """Extract vital signs as key-value dictionary."""
        vitals = {}
        for vital_name, pattern in self.vitals_pattern.items():
            match = pattern.search(text)
            if match:
                vitals[vital_name] = match.group(1).strip()
        return vitals

    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptom mentions from text."""
        text_lower = text.lower()
        found = []
        for symptom in self.symptom_keywords:
            if symptom in text_lower:
                found.append(symptom.title())
        return list(dict.fromkeys(found))  # Preserve order, deduplicate

    def _extract_procedures(self, text: str) -> List[str]:
        """Extract medical procedure mentions."""
        text_lower = text.lower()
        found = []
        for proc in self.procedure_keywords:
            if proc in text_lower:
                found.append(proc.title())
        return list(dict.fromkeys(found))

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_by_label(self, entities: List[Entity], label: str) -> List[str]:
        """Return unique text values for entities of a given label."""
        seen = set()
        result = []
        for e in entities:
            if e.label == label and e.text.lower() not in seen:
                seen.add(e.text.lower())
                result.append(e.text)
        return result

    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities (same text + label)."""
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower().strip(), e.label)
            if key not in seen and len(e.text.strip()) > 1:
                seen.add(key)
                unique.append(e)
        return unique

    def _chunk_text(self, text: str, max_chars: int = 100000) -> List[str]:
        """Split text into chunks for models with token/char limits."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            # Try to split at sentence boundary
            if end < len(text):
                period_idx = text.rfind('.', start, end)
                if period_idx > start:
                    end = period_idx + 1
            chunks.append(text[start:end])
            start = end

        return chunks

    def get_entity_summary(self, result: NERResult) -> Dict:
        """Return a summary dict for display/logging."""
        return {
            "total_entities": result.entity_count,
            "model_used": result.model_used,
            "diseases": result.diseases,
            "drugs": result.drugs,
            "symptoms": result.symptoms[:5],      # Top 5
            "procedures": result.procedures[:5],
            "medications_structured": [
                {
                    "name": m.name,
                    "dosage": m.dosage,
                    "frequency": m.frequency
                }
                for m in result.medications
            ],
            "lab_values": [
                {
                    "test": l.test_name,
                    "value": f"{l.value} {l.unit}".strip()
                }
                for l in result.lab_values
            ],
            "vitals": result.vitals
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    58-year-old male with history of type 2 diabetes mellitus and hypertension
    presents with chest pain and shortness of breath for 2 hours.
    Blood pressure 145/92 mmHg, heart rate 102 bpm, oxygen saturation 96%.

    Electrocardiogram shows ST elevation in leads II, III, aVF consistent with
    inferior STEMI. Troponin 2.4 ng/mL, HbA1c 9.2%.

    Started on Aspirin 325 mg orally once, Heparin 5000 units intravenous,
    Metoprolol 25 mg twice daily, and Atorvastatin 80 mg at bedtime.

    Plan: Percutaneous coronary intervention, continue diabetes management,
    monitor blood pressure.
    """

    print("🧪 Testing NER Extractor...\n")
    extractor = MedicalNERExtractor(use_transformer=False)
    result = extractor.extract(sample)

    print(f"Model: {result.model_used}")
    print(f"Total entities: {result.entity_count}")
    print(f"\n🦠 Diseases:    {result.diseases}")
    print(f"💊 Drugs:       {result.drugs}")
    print(f"😷 Symptoms:    {result.symptoms}")
    print(f"🔬 Lab Values:  {[(l.test_name, l.value, l.unit) for l in result.lab_values]}")
    print(f"💉 Medications: {[(m.name, m.dosage, m.frequency) for m in result.medications]}")
    print(f"🩺 Vitals:      {result.vitals}")
    print(f"🏥 Procedures:  {result.procedures}")
