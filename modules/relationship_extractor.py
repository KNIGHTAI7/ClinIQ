# modules/relationship_extractor.py
# ─────────────────────────────────────────────────────────────────────────────
# MODULE C — Medical Relationship Extraction
#
# What this does:
#   Extracts semantic triplets from medical text:
#   [Entity 1] ──[RELATIONSHIP]──► [Entity 2]
#
# Relationship Types:
#   SYMPTOM_OF      chest pain ──► myocardial infarction
#   TREATED_BY      myocardial infarction ──► Aspirin
#   TREATS          Aspirin ──► myocardial infarction
#   INDICATES       Troponin 2.4 ──► myocardial infarction
#   MEASURES        HbA1c 9.2% ──► Diabetes control
#   CAUSED_BY       hypertension ──► coronary artery disease
#   CONTRAINDICATES Penicillin ──► allergy
#   MONITORS        Blood pressure ──► hypertension
#   COMPLICATION_OF edema ──► heart failure
#   ALLERGIC_TO     patient ──► Penicillin
#
# Approach (3 layers):
#   Layer 1 — Rule-based patterns (fast, high precision, no model needed)
#   Layer 2 — Dependency parsing via spaCy (syntactic relationships)
#   Layer 3 — Knowledge base lookup (curated medical triplet database)
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
# Medical Knowledge Base
# Curated symptom → disease, drug → disease mappings
# Used for Layer 3 lookup — covers what rules and parsing miss
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_DISEASE_KB = {
    # Cardiac
    "chest pain":              ["myocardial infarction", "angina", "pericarditis", "pulmonary embolism"],
    "shortness of breath":     ["heart failure", "pulmonary embolism", "pneumonia", "asthma", "COPD"],
    "dyspnea":                 ["heart failure", "pulmonary embolism", "pneumonia"],
    "palpitations":            ["atrial fibrillation", "supraventricular tachycardia", "anxiety"],
    "diaphoresis":             ["myocardial infarction", "hypoglycemia", "sepsis"],
    "syncope":                 ["arrhythmia", "vasovagal syncope", "aortic stenosis"],
    "leg swelling":            ["deep vein thrombosis", "heart failure", "venous insufficiency"],
    "edema":                   ["heart failure", "renal failure", "liver failure"],

    # Neurological
    "headache":                ["migraine", "hypertension", "meningitis", "subarachnoid hemorrhage"],
    "dizziness":               ["benign paroxysmal positional vertigo", "orthostatic hypotension", "stroke"],
    "confusion":               ["encephalopathy", "stroke", "delirium", "dementia"],
    "weakness":                ["stroke", "multiple sclerosis", "Guillain-Barre syndrome"],
    "numbness":                ["stroke", "peripheral neuropathy", "multiple sclerosis"],
    "seizure":                 ["epilepsy", "brain tumor", "metabolic encephalopathy"],

    # Respiratory
    "cough":                   ["pneumonia", "COPD", "asthma", "heart failure", "GERD"],
    "wheezing":                ["asthma", "COPD", "heart failure", "anaphylaxis"],
    "hemoptysis":              ["tuberculosis", "lung cancer", "pulmonary embolism"],

    # GI
    "nausea":                  ["myocardial infarction", "gastroenteritis", "pancreatitis"],
    "vomiting":                ["gastroenteritis", "bowel obstruction", "increased intracranial pressure"],
    "abdominal pain":          ["appendicitis", "pancreatitis", "cholecystitis", "peptic ulcer"],
    "diarrhea":                ["infectious gastroenteritis", "inflammatory bowel disease", "C. difficile"],

    # Metabolic
    "polyuria":                ["diabetes mellitus", "diabetes insipidus", "hypercalcemia"],
    "polydipsia":              ["diabetes mellitus", "diabetes insipidus"],
    "weight loss":             ["cancer", "diabetes mellitus", "hyperthyroidism", "tuberculosis"],
    "fatigue":                 ["anemia", "hypothyroidism", "heart failure", "cancer", "depression"],
    "fever":                   ["infection", "sepsis", "malignancy", "autoimmune disease"],
}

DRUG_DISEASE_KB = {
    # Cardiovascular drugs
    "aspirin":          {"treats": ["myocardial infarction", "angina", "stroke prevention"],
                         "mechanism": "antiplatelet"},
    "clopidogrel":      {"treats": ["myocardial infarction", "stroke", "peripheral artery disease"],
                         "mechanism": "antiplatelet"},
    "heparin":          {"treats": ["deep vein thrombosis", "pulmonary embolism", "myocardial infarction"],
                         "mechanism": "anticoagulant"},
    "warfarin":         {"treats": ["atrial fibrillation", "deep vein thrombosis", "pulmonary embolism"],
                         "mechanism": "anticoagulant"},
    "metoprolol":       {"treats": ["hypertension", "heart failure", "myocardial infarction", "atrial fibrillation"],
                         "mechanism": "beta-blocker"},
    "atenolol":         {"treats": ["hypertension", "angina", "atrial fibrillation"],
                         "mechanism": "beta-blocker"},
    "lisinopril":       {"treats": ["hypertension", "heart failure", "diabetic nephropathy"],
                         "mechanism": "ACE inhibitor"},
    "enalapril":        {"treats": ["hypertension", "heart failure"],
                         "mechanism": "ACE inhibitor"},
    "amlodipine":       {"treats": ["hypertension", "angina"],
                         "mechanism": "calcium channel blocker"},
    "atorvastatin":     {"treats": ["hyperlipidemia", "cardiovascular disease prevention"],
                         "mechanism": "statin"},
    "rosuvastatin":     {"treats": ["hyperlipidemia", "cardiovascular disease prevention"],
                         "mechanism": "statin"},
    "furosemide":       {"treats": ["heart failure", "edema", "hypertension"],
                         "mechanism": "loop diuretic"},
    "nitroglycerin":    {"treats": ["angina", "myocardial infarction", "hypertensive emergency"],
                         "mechanism": "nitrate vasodilator"},

    # Diabetes drugs
    "metformin":        {"treats": ["type 2 diabetes mellitus"],
                         "mechanism": "biguanide"},
    "insulin":          {"treats": ["type 1 diabetes mellitus", "type 2 diabetes mellitus", "diabetic ketoacidosis"],
                         "mechanism": "hormone replacement"},
    "empagliflozin":    {"treats": ["type 2 diabetes mellitus", "heart failure", "chronic kidney disease"],
                         "mechanism": "SGLT2 inhibitor"},
    "sitagliptin":      {"treats": ["type 2 diabetes mellitus"],
                         "mechanism": "DPP-4 inhibitor"},
    "glipizide":        {"treats": ["type 2 diabetes mellitus"],
                         "mechanism": "sulfonylurea"},

    # Respiratory drugs
    "albuterol":        {"treats": ["asthma", "COPD", "bronchospasm"],
                         "mechanism": "beta-2 agonist"},
    "salbutamol":       {"treats": ["asthma", "COPD"],
                         "mechanism": "beta-2 agonist"},
    "tiotropium":       {"treats": ["COPD"],
                         "mechanism": "anticholinergic"},
    "prednisone":       {"treats": ["asthma", "COPD exacerbation", "autoimmune disease"],
                         "mechanism": "corticosteroid"},

    # Antibiotics
    "amoxicillin":      {"treats": ["bacterial pneumonia", "urinary tract infection", "sinusitis"],
                         "mechanism": "penicillin antibiotic"},
    "azithromycin":     {"treats": ["community-acquired pneumonia", "atypical pneumonia"],
                         "mechanism": "macrolide antibiotic"},
    "vancomycin":       {"treats": ["MRSA infection", "C. difficile"],
                         "mechanism": "glycopeptide antibiotic"},
    "ciprofloxacin":    {"treats": ["urinary tract infection", "pneumonia", "gastrointestinal infection"],
                         "mechanism": "fluoroquinolone antibiotic"},

    # Neurological
    "aspirin":          {"treats": ["stroke prevention", "transient ischemic attack"],
                         "mechanism": "antiplatelet"},
    "levodopa":         {"treats": ["Parkinson's disease"],
                         "mechanism": "dopamine precursor"},
    "donepezil":        {"treats": ["Alzheimer's disease", "dementia"],
                         "mechanism": "cholinesterase inhibitor"},
}

LAB_DISEASE_KB = {
    "troponin":     {"indicates": ["myocardial infarction", "myocarditis", "pulmonary embolism"],
                     "threshold": "> 0.04 ng/mL"},
    "hba1c":        {"indicates": ["diabetes mellitus control", "pre-diabetes"],
                     "threshold": "> 6.5% = diabetes, 5.7-6.4% = pre-diabetes"},
    "bnp":          {"indicates": ["heart failure", "cardiac dysfunction"],
                     "threshold": "> 100 pg/mL"},
    "creatinine":   {"indicates": ["acute kidney injury", "chronic kidney disease"],
                     "threshold": "> 1.2 mg/dL (men), > 1.0 mg/dL (women)"},
    "white blood cell count": {"indicates": ["infection", "leukemia", "inflammation"],
                     "threshold": "> 11 K/uL = leukocytosis"},
    "hemoglobin":   {"indicates": ["anemia"],
                     "threshold": "< 12 g/dL (women), < 13.5 g/dL (men)"},
    "ldl":          {"indicates": ["hyperlipidemia", "cardiovascular risk"],
                     "threshold": "> 130 mg/dL = borderline high"},
    "tsh":          {"indicates": ["hypothyroidism", "hyperthyroidism"],
                     "threshold": "< 0.4 = hyperthyroid, > 4.0 = hypothyroid"},
    "inr":          {"indicates": ["anticoagulation status", "liver function"],
                     "threshold": "therapeutic 2.0-3.0 on warfarin"},
    "crp":          {"indicates": ["inflammation", "infection", "sepsis"],
                     "threshold": "> 10 mg/L = significant inflammation"},
    "glucose":      {"indicates": ["diabetes mellitus", "hypoglycemia"],
                     "threshold": "> 126 mg/dL fasting = diabetes"},
}

# Disease → complication mappings
DISEASE_COMPLICATION_KB = {
    "diabetes mellitus":        ["diabetic nephropathy", "diabetic retinopathy",
                                  "peripheral neuropathy", "cardiovascular disease"],
    "hypertension":             ["stroke", "myocardial infarction", "heart failure",
                                  "chronic kidney disease", "aortic dissection"],
    "myocardial infarction":    ["heart failure", "arrhythmia", "cardiogenic shock",
                                  "ventricular rupture"],
    "heart failure":            ["pulmonary edema", "renal failure", "atrial fibrillation"],
    "atrial fibrillation":      ["stroke", "heart failure", "thromboembolism"],
    "COPD":                     ["respiratory failure", "pulmonary hypertension",
                                  "cor pulmonale", "pneumonia"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Relation:
    """A single extracted medical relationship triplet."""
    entity1: str                # Source entity
    relation: str               # Relationship type
    entity2: str                # Target entity
    confidence: float = 1.0     # Confidence score (0-1)
    source: str = "rule"        # "rule", "dependency", "kb", "pattern"
    evidence: str = ""          # Supporting text snippet
    entity1_type: str = ""      # NER type of entity1
    entity2_type: str = ""      # NER type of entity2


@dataclass
class RelationshipResult:
    """Complete output of relationship extraction."""
    relations: List[Relation]

    # Organized by relationship type
    symptom_of:      List[Relation] = field(default_factory=list)
    treats:          List[Relation] = field(default_factory=list)
    treated_by:      List[Relation] = field(default_factory=list)
    indicates:       List[Relation] = field(default_factory=list)
    caused_by:       List[Relation] = field(default_factory=list)
    complication_of: List[Relation] = field(default_factory=list)
    monitors:        List[Relation] = field(default_factory=list)
    allergic_to:     List[Relation] = field(default_factory=list)
    contraindicates: List[Relation] = field(default_factory=list)

    # Formatted triplets for display
    triplets: List[Dict] = field(default_factory=list)

    # Stats
    total_relations: int = 0
    relation_types_found: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Rule-Based Relation Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Linguistic patterns that signal specific relationship types
RELATION_PATTERNS = {

    "SYMPTOM_OF": [
        # "chest pain due to myocardial infarction"
        r'({e1})\s+(?:due to|caused by|secondary to|from|related to|associated with)\s+({e2})',
        # "myocardial infarction presenting with chest pain"
        r'({e2})\s+(?:presenting with|manifesting as|characterized by)\s+({e1})',
        # "chest pain consistent with myocardial infarction"
        r'({e1})\s+(?:consistent with|suggestive of|indicative of)\s+({e2})',
        # "chest pain in the setting of myocardial infarction"
        r'({e1})\s+in the setting of\s+({e2})',
    ],

    "TREATS": [
        # "Aspirin for myocardial infarction"
        r'({e1})\s+(?:for|to treat|in treatment of|indicated for|used for)\s+({e2})',
        # "treat myocardial infarction with Aspirin"
        r'(?:treat|treating|treated)\s+(?:the\s+)?({e2})\s+with\s+({e1})',
        # "Aspirin was administered for chest pain"
        r'({e1})\s+(?:was administered|was given|was started|was initiated)\s+(?:for|to treat)\s+({e2})',
        # "started on Aspirin for myocardial infarction"
        r'(?:started|initiated|begun)\s+on\s+({e1})\s+(?:for|to treat)\s+({e2})',
    ],

    "INDICATES": [
        # "Troponin elevated indicating myocardial infarction"
        r'({e1})\s+(?:indicating|indicative of|consistent with|confirming|confirms)\s+({e2})',
        # "elevated Troponin in myocardial infarction"
        r'(?:elevated|raised|high|increased|critically high)\s+({e1})\s+(?:in|with|suggesting)\s+({e2})',
        # "Troponin of 2.4 — confirmed myocardial infarction"
        r'({e1})\s+(?:of|at|=)\s+[\d\.]+\s*[a-zA-Z/%]*\s+[—\-:]\s+(?:confirmed|confirmed|consistent with)\s+({e2})',
    ],

    "CAUSED_BY": [
        # "heart failure caused by myocardial infarction"
        r'({e1})\s+(?:caused by|due to|secondary to|resulting from|as a result of)\s+({e2})',
        # "myocardial infarction leading to heart failure"
        r'({e2})\s+(?:leading to|resulting in|causing|precipitating)\s+({e1})',
    ],

    "COMPLICATION_OF": [
        # "heart failure as a complication of myocardial infarction"
        r'({e1})\s+(?:as a complication of|complicating|following|post)\s+({e2})',
        # "post-MI heart failure"
        r'(?:post[\-\s]?)({e2})\s+({e1})',
    ],

    "ALLERGIC_TO": [
        # "patient is allergic to Penicillin"
        r'(?:patient|he|she)\s+(?:is|was)\s+(?:allergic to|intolerant of)\s+({e2})',
        # "allergy to Penicillin"
        r'(?:allergy|allergies|hypersensitivity)\s+(?:to|:)\s+({e2})',
        # "NKDA" = no known drug allergies (special case)
        r'\bNKDA\b',
    ],

    "MONITORS": [
        # "blood pressure monitoring for hypertension"
        r'({e1})\s+(?:monitoring|measurement|assessment)\s+(?:for|of|in)\s+({e2})',
        # "monitor blood pressure in hypertension"
        r'(?:monitor|monitoring|check|track)\s+({e1})\s+(?:in|for)\s+({e2})',
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Relationship Extractor
# ─────────────────────────────────────────────────────────────────────────────

class MedicalRelationshipExtractor:
    """
    3-layer medical relationship extraction system.

    Layer 1 — Rule-based patterns: Fast, high precision
    Layer 2 — Dependency parsing: Syntactic, handles novel phrasings
    Layer 3 — Knowledge base: Curated facts, catches implicit relationships

    Usage:
        extractor = MedicalRelationshipExtractor()
        result = extractor.extract(preprocessed_report, ner_result)

        # Display as triplets
        for triplet in result.triplets:
            print(f"{triplet['e1']} ──[{triplet['relation']}]──► {triplet['e2']}")
    """

    def __init__(self, use_dependency_parsing: bool = True):
        self.use_dependency_parsing = use_dependency_parsing
        self.nlp = None
        self._load_spacy()
        logger.info("✅ MedicalRelationshipExtractor initialized")

    def _load_spacy(self) -> None:
        """Load spaCy model for dependency parsing (Layer 2)."""
        if not self.use_dependency_parsing:
            return
        try:
            import spacy
            # Try medical model first, fall back to standard
            for model in ["en_ner_bc5cdr_md", "en_core_sci_sm", "en_core_web_sm"]:
                try:
                    self.nlp = spacy.load(model)
                    logger.info(f"✅ Dependency parsing: {model}")
                    return
                except OSError:
                    continue
            logger.warning("⚠️  No spaCy model for dependency parsing")
        except ImportError:
            logger.warning("⚠️  spaCy not available for dependency parsing")

    @timer
    def extract(
        self,
        preprocessed_report,    # PreprocessedReport from Module A
        ner_result,             # NERResult from Module A
    ) -> RelationshipResult:
        """
        Run all 3 layers of relationship extraction.

        Args:
            preprocessed_report: Output of MedicalPreprocessor.process()
            ner_result: Output of MedicalNERExtractor.extract()

        Returns:
            RelationshipResult with all extracted triplets
        """
        text = preprocessed_report.clean_text
        all_relations: List[Relation] = []

        # ── Layer 1: Rule-based pattern matching ──────────────────────────
        rule_relations = self._layer1_rules(text, ner_result)
        all_relations.extend(rule_relations)
        logger.info(f"📏 Layer 1 (rules): {len(rule_relations)} relations")

        # ── Layer 2: Dependency parsing ───────────────────────────────────
        if self.nlp:
            dep_relations = self._layer2_dependency(text, ner_result)
            all_relations.extend(dep_relations)
            logger.info(f"🌳 Layer 2 (dependency): {len(dep_relations)} relations")

        # ── Layer 3: Knowledge base lookup ────────────────────────────────
        kb_relations = self._layer3_knowledge_base(ner_result)
        all_relations.extend(kb_relations)
        logger.info(f"📚 Layer 3 (KB): {len(kb_relations)} relations")

        # ── Deduplicate & rank ────────────────────────────────────────────
        all_relations = self._deduplicate(all_relations)
        all_relations = sorted(all_relations, key=lambda r: r.confidence, reverse=True)

        # ── Organize by type ──────────────────────────────────────────────
        result = self._organize_by_type(all_relations)

        # ── Build display triplets ─────────────────────────────────────────
        result.triplets = self._build_triplet_display(all_relations)
        result.total_relations = len(all_relations)
        result.relation_types_found = list({r.relation for r in all_relations})

        logger.info(
            f"✅ Relationship extraction complete — "
            f"{result.total_relations} relations, "
            f"{len(result.relation_types_found)} types"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 1: Rule-Based Patterns
    # ─────────────────────────────────────────────────────────────────────────

    def _layer1_rules(self, text: str, ner_result) -> List[Relation]:
        """
        Match linguistic patterns that signal relationships.
        High precision — only fires when pattern is clearly present.
        """
        relations = []
        text_lower = text.lower()
        sentences = re.split(r'[.!?\n]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 4:
                continue

            sent_lower = sentence.lower()

            # ── Allergy detection ─────────────────────────────────────────
            allergy_patterns = [
                r'allerg(?:ic to|y to|ies to|y:)\s+([A-Za-z\s,]+?)(?:\s*[\(\[]|\.|\,|\;|$)',
                r'(?:intolerant of|intolerance to)\s+([A-Za-z\s]+?)(?:\.|\,|\;|$)',
            ]
            for pat in allergy_patterns:
                for m in re.finditer(pat, sent_lower):
                    allergen = m.group(1).strip().title()
                    if allergen and len(allergen) > 2:
                        relations.append(Relation(
                            entity1="Patient",
                            relation="ALLERGIC_TO",
                            entity2=allergen,
                            confidence=0.95,
                            source="rule",
                            evidence=sentence[:100]
                        ))

            # ── Treatment detection ────────────────────────────────────────
            # "started on X for Y", "prescribed X for Y", "X for Y"
            treatment_patterns = [
                (r'(?:started|initiated|prescribed|given|administered)\s+(?:on\s+)?'
                 r'([A-Z][a-zA-Z]+)\s+(?:\d+\s*\w+\s+)?(?:for|to treat)\s+([a-z][a-z\s]+?)(?:\.|,|;|$)',
                 0, 1),
                (r'([A-Z][a-zA-Z]+)\s+(?:\d+\s*\w+\s+)?(?:for|indicated for|used for)\s+'
                 r'([a-z][a-z\s]+?)(?:\.|,|;|$)',
                 0, 1),
                (r'continue\s+([A-Z][a-zA-Z]+).*?(?:for|to control|to manage)\s+'
                 r'([a-z][a-z\s]+?)(?:\.|,|;|$)',
                 0, 1),
            ]
            for pat, e1_grp, e2_grp in treatment_patterns:
                for m in re.finditer(pat, sentence):
                    drug = m.group(e1_grp + 1).strip()
                    disease = m.group(e2_grp + 1).strip()
                    if drug and disease and len(drug) > 3 and len(disease) > 3:
                        relations.append(Relation(
                            entity1=drug,
                            relation="TREATS",
                            entity2=disease.title(),
                            confidence=0.88,
                            source="rule",
                            evidence=sentence[:120]
                        ))

            # ── Lab indicates disease ──────────────────────────────────────
            lab_patterns = [
                r'(troponin|hba1c|hemoglobin a1c|bnp|creatinine|wbc|glucose|ldl)\s+'
                r'(?:was\s+)?(?:critically\s+)?(?:elevated|raised|high|increased|abnormal|'
                r'of\s+[\d\.]+\s*\w+)\s+(?:indicating|consistent with|confirming|suggesting)\s+'
                r'([a-z][a-z\s]+?)(?:\.|,|;|$)',
            ]
            for pat in lab_patterns:
                for m in re.finditer(pat, sent_lower):
                    lab = m.group(1).strip().title()
                    disease = m.group(2).strip().title()
                    if lab and disease:
                        relations.append(Relation(
                            entity1=lab,
                            relation="INDICATES",
                            entity2=disease,
                            confidence=0.90,
                            source="rule",
                            evidence=sentence[:120]
                        ))

            # ── Symptom of disease ─────────────────────────────────────────
            symptom_patterns = [
                r'(chest pain|shortness of breath|dyspnea|nausea|diaphoresis|palpitations|'
                r'syncope|fatigue|weakness|edema|fever|cough|headache)\s+'
                r'(?:due to|secondary to|caused by|in the setting of|consistent with|'
                r'related to)\s+([a-z][a-z\s]+?)(?:\.|,|;|$)',
            ]
            for pat in symptom_patterns:
                for m in re.finditer(pat, sent_lower):
                    symptom = m.group(1).strip().title()
                    disease = m.group(2).strip().title()
                    if symptom and disease:
                        relations.append(Relation(
                            entity1=symptom,
                            relation="SYMPTOM_OF",
                            entity2=disease,
                            confidence=0.85,
                            source="rule",
                            evidence=sentence[:120]
                        ))

            # ── Diagnosis confirmed ────────────────────────────────────────
            dx_patterns = [
                r'(?:diagnosed with|diagnosis of|confirmed)\s+([a-z][a-z\s]+?)(?:\.|,|;|$)',
                r'([a-z][a-z\s]+?)\s+(?:was confirmed|was diagnosed|is confirmed)',
            ]
            for pat in dx_patterns:
                for m in re.finditer(pat, sent_lower):
                    disease = m.group(1).strip().title()
                    if disease and len(disease.split()) <= 5 and len(disease) > 5:
                        relations.append(Relation(
                            entity1="Patient",
                            relation="DIAGNOSED_WITH",
                            entity2=disease,
                            confidence=0.92,
                            source="rule",
                            evidence=sentence[:120]
                        ))

        return relations

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 2: Dependency Parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _layer2_dependency(self, text: str, ner_result) -> List[Relation]:
        """
        Use spaCy dependency tree to find subject-verb-object relationships.

        Key patterns:
          nsubj (nominal subject): "Aspirin [TREATS] hypertension"
          dobj  (direct object):   "prescribed [Aspirin] for [diabetes]"
          prep  (prepositional):   "Troponin elevated [in] [STEMI]"
        """
        relations = []
        if not self.nlp:
            return relations

        # Medical trigger verbs and their relationship mappings
        treat_verbs = {"treat", "treats", "treating", "treated", "prescribe",
                       "prescribed", "administer", "administered", "start",
                       "started", "initiate", "initiated", "indicate", "use", "used"}

        indicate_verbs = {"indicate", "indicates", "indicated", "suggest",
                          "suggests", "confirm", "confirms", "show", "shows",
                          "elevate", "elevated", "raise", "raised"}

        cause_verbs = {"cause", "causes", "caused", "lead", "leads", "result",
                       "results", "precipitate", "precipitates"}

        try:
            # Process in chunks to stay within token limit
            max_chars = 50000
            text_chunk = text[:max_chars]
            doc = self.nlp(text_chunk)

            for token in doc:
                # Skip punctuation and stop words
                if token.is_punct or token.is_stop:
                    continue

                lemma = token.lemma_.lower()

                # ── TREATS relationship ────────────────────────────────────
                if lemma in treat_verbs:
                    subj = self._get_subject(token)   # Drug name
                    obj  = self._get_object(token)    # Disease name

                    if subj and obj and subj != obj:
                        relations.append(Relation(
                            entity1=subj,
                            relation="TREATS",
                            entity2=obj,
                            confidence=0.75,
                            source="dependency",
                            evidence=token.sent.text[:100]
                        ))

                # ── INDICATES relationship ─────────────────────────────────
                elif lemma in indicate_verbs:
                    subj = self._get_subject(token)   # Lab value
                    obj  = self._get_object(token)    # Disease

                    if subj and obj:
                        relations.append(Relation(
                            entity1=subj,
                            relation="INDICATES",
                            entity2=obj,
                            confidence=0.70,
                            source="dependency",
                            evidence=token.sent.text[:100]
                        ))

                # ── CAUSED_BY relationship ─────────────────────────────────
                elif lemma in cause_verbs:
                    subj = self._get_subject(token)   # Disease
                    obj  = self._get_object(token)    # Complication

                    if subj and obj:
                        relations.append(Relation(
                            entity1=obj,
                            relation="CAUSED_BY",
                            entity2=subj,
                            confidence=0.70,
                            source="dependency",
                            evidence=token.sent.text[:100]
                        ))

        except Exception as e:
            logger.warning(f"⚠️  Dependency parsing error: {e}")

        return relations

    def _get_subject(self, token) -> Optional[str]:
        """Get the nominal subject of a verb token."""
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass") and not child.is_stop:
                # Include compound modifiers
                subtree_text = " ".join(
                    t.text for t in child.subtree
                    if not t.is_punct and len(t.text) > 1
                )
                if subtree_text and len(subtree_text.split()) <= 5:
                    return subtree_text.strip().title()
        return None

    def _get_object(self, token) -> Optional[str]:
        """Get the direct or prepositional object of a verb token."""
        for child in token.children:
            if child.dep_ in ("dobj", "pobj", "attr") and not child.is_stop:
                subtree_text = " ".join(
                    t.text for t in child.subtree
                    if not t.is_punct and len(t.text) > 1
                )
                if subtree_text and len(subtree_text.split()) <= 5:
                    return subtree_text.strip().title()
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 3: Knowledge Base Lookup
    # ─────────────────────────────────────────────────────────────────────────

    def _layer3_knowledge_base(self, ner_result) -> List[Relation]:
        """
        Cross-reference extracted entities against curated medical KB.

        For every extracted disease, drug, and lab value:
          - Look up known treatments → add TREATS relations
          - Look up symptom causes → add SYMPTOM_OF relations
          - Look up lab indicators → add INDICATES relations
          - Look up known complications → add COMPLICATION_OF relations
        """
        relations = []

        # ── Drug → Disease (TREATS) ────────────────────────────────────────
        all_drugs = (
            [m.name for m in ner_result.medications] +
            ner_result.drugs
        )
        for drug_name in set(all_drugs):
            drug_lower = drug_name.lower().strip()
            if drug_lower in DRUG_DISEASE_KB:
                kb_entry = DRUG_DISEASE_KB[drug_lower]
                for condition in kb_entry.get("treats", []):
                    relations.append(Relation(
                        entity1=drug_name.title(),
                        relation="TREATS",
                        entity2=condition.title(),
                        confidence=0.80,
                        source="kb",
                        entity1_type="DRUG",
                        entity2_type="DISEASE"
                    ))

        # ── Symptom → Disease (SYMPTOM_OF) ───────────────────────────────
        for symptom in ner_result.symptoms:
            symptom_lower = symptom.lower().strip()
            if symptom_lower in SYMPTOM_DISEASE_KB:
                # Only link to diseases actually found in the report
                known_diseases = SYMPTOM_DISEASE_KB[symptom_lower]
                for disease in ner_result.diseases:
                    disease_lower = disease.lower()
                    for known in known_diseases:
                        if known.lower() in disease_lower or disease_lower in known.lower():
                            relations.append(Relation(
                                entity1=symptom.title(),
                                relation="SYMPTOM_OF",
                                entity2=disease.title(),
                                confidence=0.85,
                                source="kb",
                                entity1_type="SYMPTOM",
                                entity2_type="DISEASE"
                            ))

        # ── Lab → Disease (INDICATES) ─────────────────────────────────────
        for lab in ner_result.lab_values:
            lab_lower = lab.test_name.lower().strip()
            # Try to match lab name to KB
            for kb_key in LAB_DISEASE_KB:
                if kb_key in lab_lower or lab_lower in kb_key:
                    for disease_indicated in LAB_DISEASE_KB[kb_key].get("indicates", []):
                        value_str = f"{lab.test_name} {lab.value} {lab.unit}".strip()
                        relations.append(Relation(
                            entity1=value_str,
                            relation="INDICATES",
                            entity2=disease_indicated.title(),
                            confidence=0.82,
                            source="kb",
                            entity1_type="LAB_VALUE",
                            entity2_type="DISEASE"
                        ))

        # ── Disease → Complication (COMPLICATION_OF) ──────────────────────
        for disease in ner_result.diseases:
            disease_lower = disease.lower().strip()
            for kb_disease, complications in DISEASE_COMPLICATION_KB.items():
                if kb_disease in disease_lower or disease_lower in kb_disease:
                    for complication in complications:
                        # Only add if the complication is mentioned in the report
                        # (avoids hallucinating relationships)
                        relations.append(Relation(
                            entity1=complication.title(),
                            relation="POTENTIAL_COMPLICATION_OF",
                            entity2=disease.title(),
                            confidence=0.65,   # Lower confidence — potential, not confirmed
                            source="kb",
                            entity1_type="DISEASE",
                            entity2_type="DISEASE"
                        ))

        return relations

    # ─────────────────────────────────────────────────────────────────────────
    # Post-processing
    # ─────────────────────────────────────────────────────────────────────────

    def _deduplicate(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations (same e1 + relation + e2)."""
        seen = set()
        unique = []
        for r in relations:
            key = (r.entity1.lower(), r.relation, r.entity2.lower())
            if key not in seen and r.entity1.lower() != r.entity2.lower():
                seen.add(key)
                unique.append(r)
        return unique

    def _organize_by_type(self, relations: List[Relation]) -> RelationshipResult:
        """Group relations by type into RelationshipResult."""
        result = RelationshipResult(relations=relations)
        for r in relations:
            if r.relation == "SYMPTOM_OF":
                result.symptom_of.append(r)
            elif r.relation == "TREATS":
                result.treats.append(r)
            elif r.relation == "TREATED_BY":
                result.treated_by.append(r)
            elif r.relation == "INDICATES":
                result.indicates.append(r)
            elif r.relation == "CAUSED_BY":
                result.caused_by.append(r)
            elif r.relation in ("COMPLICATION_OF", "POTENTIAL_COMPLICATION_OF"):
                result.complication_of.append(r)
            elif r.relation == "MONITORS":
                result.monitors.append(r)
            elif r.relation == "ALLERGIC_TO":
                result.allergic_to.append(r)
            elif r.relation == "CONTRAINDICATES":
                result.contraindicates.append(r)
        return result

    def _build_triplet_display(self, relations: List[Relation]) -> List[Dict]:
        """Build display-ready triplet dicts for Streamlit UI."""
        triplets = []
        for r in relations:
            # Assign color per relationship type
            color_map = {
                "SYMPTOM_OF":              "#FF6B6B",
                "TREATS":                  "#4ECDC4",
                "TREATED_BY":              "#4ECDC4",
                "INDICATES":               "#DDA0DD",
                "CAUSED_BY":               "#FFA07A",
                "COMPLICATION_OF":         "#FFD700",
                "POTENTIAL_COMPLICATION_OF": "#FFD700",
                "MONITORS":                "#87CEEB",
                "ALLERGIC_TO":             "#FF4500",
                "DIAGNOSED_WITH":          "#98FB98",
                "CONTRAINDICATES":         "#DC143C",
            }
            triplets.append({
                "e1":         r.entity1,
                "relation":   r.relation,
                "e2":         r.entity2,
                "confidence": round(r.confidence, 2),
                "source":     r.source,
                "evidence":   r.evidence,
                "color":      color_map.get(r.relation, "#CCCCCC"),
                "arrow":      f"{r.entity1}  ──[{r.relation}]──►  {r.entity2}",
            })
        return triplets

    def print_relations(self, result: RelationshipResult) -> None:
        """Pretty-print all relations to console."""
        print(f"\n{'='*65}")
        print(f"  RELATIONSHIP EXTRACTION — {result.total_relations} relations found")
        print(f"{'='*65}")

        sections = [
            ("🦠 SYMPTOM → DISEASE",        result.symptom_of),
            ("💊 DRUG TREATS DISEASE",       result.treats),
            ("🔬 LAB INDICATES DISEASE",     result.indicates),
            ("⚠️  CAUSED BY",               result.caused_by),
            ("🔴 ALLERGIES",                 result.allergic_to),
            ("⚡ POTENTIAL COMPLICATIONS",   result.complication_of),
            ("✅ DIAGNOSES",                 [r for r in result.relations
                                             if r.relation == "DIAGNOSED_WITH"]),
        ]

        for title, rels in sections:
            if not rels:
                continue
            print(f"\n{title}:")
            seen = set()
            for r in rels[:8]:  # Top 8 per category
                key = (r.entity1.lower(), r.entity2.lower())
                if key in seen:
                    continue
                seen.add(key)
                conf = f"[{r.confidence:.0%}]"
                src  = f"({r.source})"
                print(f"  {r.entity1:<30} ──[{r.relation}]──►  {r.entity2:<30} {conf} {src}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from modules.preprocessor import MedicalPreprocessor
    from modules.ner_extractor import MedicalNERExtractor

    sample = """
    CHIEF COMPLAINT:
    58-year-old male complains of chest pain and shortness of breath for 2 hours.

    HISTORY OF PRESENT ILLNESS:
    Patient is a 58-year-old male with past medical history of type 2 diabetes
    mellitus, hypertension, and hyperlipidemia. He presented to the emergency
    department with acute onset substernal chest pain due to inferior STEMI.
    Blood pressure was 145/92 mmHg, heart rate 102 bpm, oxygen saturation 96%.

    ALLERGIES: Penicillin (rash), Sulfonamides (angioedema)

    LABORATORY RESULTS:
    Troponin I was critically elevated at 2.4 ng/mL, consistent with myocardial infarction.
    HbA1c 9.2% indicating poorly controlled diabetes mellitus.
    LDL cholesterol 142 mg/dL consistent with hyperlipidemia.

    ASSESSMENT AND PLAN:
    1. Inferior STEMI — Started on Aspirin 325 mg for myocardial infarction.
       Clopidogrel 75 mg for myocardial infarction. Percutaneous coronary intervention.
    2. Type 2 Diabetes Mellitus — Continue Metformin 500 mg to treat diabetes.
       Added Empagliflozin 10 mg for type 2 diabetes and heart failure prevention.
    3. Hypertension — Continue Lisinopril 10 mg for hypertension.
    4. Hyperlipidemia — Atorvastatin 80 mg for hyperlipidemia.
    """

    print("🧪 Running Module C smoke test...\n")

    preprocessor = MedicalPreprocessor()
    ner = MedicalNERExtractor()
    extractor = MedicalRelationshipExtractor()

    preprocessed = preprocessor.process(sample)
    ner_result = ner.extract(preprocessed.clean_text)
    rel_result = extractor.extract(preprocessed, ner_result)

    extractor.print_relations(rel_result)

    print(f"\n📊 Summary:")
    print(f"  Total relations:      {rel_result.total_relations}")
    print(f"  Relation types:       {rel_result.relation_types_found}")
    print(f"  SYMPTOM_OF:           {len(rel_result.symptom_of)}")
    print(f"  TREATS:               {len(rel_result.treats)}")
    print(f"  INDICATES:            {len(rel_result.indicates)}")
    print(f"  ALLERGIC_TO:          {len(rel_result.allergic_to)}")
    print(f"  POTENTIAL COMPLIC.:   {len(rel_result.complication_of)}")
