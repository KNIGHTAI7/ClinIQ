# modules/summarizer.py
# ─────────────────────────────────────────────────────────────────────────────
# MODULE B — Medical Report Summarization Pipeline
#
# Two-Stage Approach:
#   STAGE 1 — Extractive (TextRank)
#     → Scores every sentence by medical importance
#     → Picks top N% sentences — reduces text before feeding to transformer
#     → No model download needed, runs instantly
#
#   STAGE 2 — Abstractive (BART / T5)
#     → Takes extractive output as input (smaller = better quality)
#     → Generates fluent, condensed, human-readable summary
#     → Model: facebook/bart-large-cnn (best for medical summarization)
#
#   BONUS — Section-Aware Summarization
#     → Summarizes each clinical section separately (HPI, Assessment, Plan)
#     → Then combines into one master summary
#
#   BONUS — Clinical Note Generator
#     → Uses NER output + summary to fill a structured SOAP note template
# ─────────────────────────────────────────────────────────────────────────────

import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import timer, count_words, truncate_text


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# High-value medical keywords — sentences containing these score higher
CLINICAL_KEYWORDS = [
    # Diagnosis & Assessment
    "diagnosis", "diagnosed", "assessment", "impression", "consistent with",
    "suggestive of", "confirmed", "ruled out", "presents with",

    # Critical findings
    "critical", "severe", "acute", "emergent", "urgent", "significant",
    "elevated", "decreased", "abnormal", "positive", "negative",

    # Treatment
    "treatment", "plan", "prescribed", "started", "initiated", "administered",
    "referred", "scheduled", "recommended", "ordered",

    # Cardiac
    "myocardial infarction", "stemi", "nstemi", "chest pain", "troponin",
    "ejection fraction", "ecg", "electrocardiogram", "angiography",

    # Common diagnoses
    "diabetes", "hypertension", "pneumonia", "heart failure", "sepsis",
    "stroke", "cancer", "infection", "fracture", "pulmonary embolism",

    # Lab values (always clinically relevant)
    "hemoglobin", "creatinine", "troponin", "hba1c", "sodium", "potassium",
    "white blood cell", "platelet", "glucose", "bilirubin",

    # Medications (always relevant)
    "aspirin", "heparin", "insulin", "metformin", "lisinopril", "statin",
    "antibiotic", "warfarin", "metoprolol", "furosemide",
]

# Sections ordered by clinical importance for summarization priority
SECTION_PRIORITY = {
    "assessment and plan": 1.0,
    "assessment":          1.0,
    "plan":                0.95,
    "impression":          0.95,
    "history of present illness": 0.85,
    "hpi":                 0.85,
    "chief complaint":     0.80,
    "laboratory results":  0.75,
    "imaging":             0.70,
    "procedures":          0.70,
    "physical examination": 0.60,
    "medications":         0.65,
    "discharge summary":   0.90,
    "past medical history": 0.50,
    "social history":      0.30,
    "family history":      0.30,
    "review of systems":   0.40,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentenceScore:
    """A sentence with its clinical importance score."""
    text: str
    score: float
    section: str = ""
    position: int = 0      # Original position in document


@dataclass
class SummaryResult:
    """Complete output of the summarization pipeline."""
    # Core summaries
    extractive_summary: str          # Best sentences selected by TextRank
    abstractive_summary: str         # AI-generated fluent summary
    short_summary: str               # 1-2 sentence TL;DR
    section_summaries: Dict[str, str] # Per-section summaries

    # Clinical note
    clinical_note: str               # Structured SOAP-style note

    # Stats
    original_word_count: int
    summary_word_count: int
    compression_ratio: float         # e.g. 0.15 = 85% compression
    model_used: str = ""
    extractive_sentences_used: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Extractive Summarizer (TextRank + Medical Scoring)
# ─────────────────────────────────────────────────────────────────────────────

class ExtractiveSummarizer:
    """
    Medical TextRank — scores sentences by clinical importance.

    Scoring factors:
      1. TF-IDF style keyword frequency
      2. Clinical keyword presence (diagnosis, treatment, etc.)
      3. Section importance (Assessment > Social History)
      4. Sentence position (first/last sentences tend to be important)
      5. Sentence length (very short/long sentences penalized)
    """

    def __init__(self):
        self.clinical_keywords = CLINICAL_KEYWORDS

    def summarize(
        self,
        sentences: List[str],
        section_name: str = "",
        compression_ratio: float = 0.35,
        min_sentences: int = 2,
        max_sentences: int = 15
    ) -> Tuple[str, List[SentenceScore]]:
        """
        Select top sentences by clinical importance score.

        Args:
            sentences: List of sentences from preprocessor
            section_name: Clinical section (affects scoring weight)
            compression_ratio: Fraction of sentences to keep (0.35 = keep 35%)
            min_sentences: Always return at least this many sentences
            max_sentences: Never return more than this many sentences

        Returns:
            (summary_text, scored_sentences)
        """
        if not sentences:
            return "", []

        # Score all sentences
        scored = [
            SentenceScore(
                text=sent,
                score=self._score_sentence(sent, i, len(sentences), section_name),
                section=section_name,
                position=i
            )
            for i, sent in enumerate(sentences)
            if len(sent.split()) >= 5  # Skip very short sentences
        ]

        if not scored:
            return " ".join(sentences[:min_sentences]), []

        # Determine how many sentences to keep
        n_keep = max(
            min_sentences,
            min(max_sentences, int(len(scored) * compression_ratio))
        )

        # Sort by score, take top N, then re-sort by original position
        # (re-sorting by position preserves narrative flow)
        top_sentences = sorted(scored, key=lambda x: x.score, reverse=True)[:n_keep]
        top_sentences = sorted(top_sentences, key=lambda x: x.position)

        summary = " ".join(s.text for s in top_sentences)
        return summary, scored

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
        section: str
    ) -> float:
        """
        Compute clinical importance score for a single sentence.
        Returns float between 0.0 and 1.0
        """
        score = 0.0
        sent_lower = sentence.lower()
        words = sentence.split()
        n_words = len(words)

        # ── Factor 1: Clinical keyword presence ───────────────────────────
        keyword_hits = sum(1 for kw in self.clinical_keywords if kw in sent_lower)
        keyword_score = min(keyword_hits / 3.0, 1.0)   # Cap at 1.0
        score += keyword_score * 0.40                   # 40% weight

        # ── Factor 2: Contains numbers/values (lab results, vitals) ───────
        number_count = len(re.findall(r'\d+\.?\d*', sentence))
        number_score = min(number_count / 5.0, 1.0)
        score += number_score * 0.20                    # 20% weight

        # ── Factor 3: Section importance ──────────────────────────────────
        section_weight = SECTION_PRIORITY.get(section.lower(), 0.5)
        score += section_weight * 0.20                  # 20% weight

        # ── Factor 4: Sentence position ───────────────────────────────────
        # First and last sentences are usually most important
        if total > 0:
            rel_pos = position / total
            if rel_pos <= 0.15 or rel_pos >= 0.85:
                position_score = 0.8
            elif rel_pos <= 0.30:
                position_score = 0.6
            else:
                position_score = 0.3
        else:
            position_score = 0.5
        score += position_score * 0.10                  # 10% weight

        # ── Factor 5: Sentence length (penalize extremes) ─────────────────
        if 8 <= n_words <= 35:
            length_score = 1.0
        elif n_words < 8:
            length_score = n_words / 8.0
        else:
            length_score = max(0.3, 35.0 / n_words)
        score += length_score * 0.10                    # 10% weight

        return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Abstractive Summarizer (BART)
# ─────────────────────────────────────────────────────────────────────────────

class AbstractiveSummarizer:
    """
    HuggingFace BART-based abstractive summarizer.

    Model: facebook/bart-large-cnn
    Why BART:
      - Best-in-class for summarization tasks
      - Pre-trained on CNN/DailyMail news (transfers well to clinical text)
      - Handles up to 1024 tokens
      - Available free via HuggingFace

    Fallback: If BART unavailable, returns extractive summary
    """

    # Length presets (min_length, max_length in tokens)
    LENGTH_PRESETS = {
        "short":    (30,  80),    # 1-2 sentences
        "medium":   (60,  150),   # 3-4 sentences
        "detailed": (100, 250),   # 5-7 sentences
    }

    def __init__(self, model_name: str = "facebook/bart-large-cnn", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Load BART model. Falls back gracefully if unavailable."""
        try:
            from transformers import pipeline
            import torch

            device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1

            logger.info(f"⏳ Loading {self.model_name} (first time = ~1GB download)...")
            self.pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=device,
                # Optimization: use smaller model variant if memory is tight
            )
            logger.info(f"✅ Abstractive model loaded: {self.model_name}")

        except ImportError:
            logger.error("❌ transformers not installed. Run: pip install transformers torch")
        except Exception as e:
            logger.warning(f"⚠️  Could not load {self.model_name}: {e}")
            logger.info("💡 Falling back to extractive-only summarization")

    def summarize(
        self,
        text: str,
        length: str = "medium",
        custom_min: Optional[int] = None,
        custom_max: Optional[int] = None,
    ) -> str:
        """
        Generate abstractive summary using BART.

        Args:
            text: Input text (ideally already extractive-filtered, < 1024 tokens)
            length: "short" | "medium" | "detailed"
            custom_min/max: Override preset token lengths

        Returns:
            Generated summary string, or input text if model unavailable
        """
        if not text.strip():
            return ""

        if self.pipeline is None:
            logger.warning("⚠️  Abstractive model not loaded — returning extractive summary")
            return text

        # Get length settings
        min_len, max_len = self.LENGTH_PRESETS.get(length, self.LENGTH_PRESETS["medium"])
        if custom_min:
            min_len = custom_min
        if custom_max:
            max_len = custom_max

        # BART max input is 1024 tokens (~750 words) — truncate if needed
        input_text = truncate_text(text, max_words=700)

        try:
            result = self.pipeline(
                input_text,
                min_length=min_len,
                max_length=max_len,
                do_sample=False,        # Deterministic output
                num_beams=4,            # Beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=3, # Avoid repetitive phrases
            )
            summary = result[0]["summary_text"].strip()
            logger.info(f"✅ Abstractive summary: {count_words(summary)} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Abstractive summarization failed: {e}")
            return text  # Return extractive as fallback


# ─────────────────────────────────────────────────────────────────────────────
# Main Summarization Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MedicalSummarizer:
    """
    Full medical summarization pipeline combining extractive + abstractive.

    Usage:
        summarizer = MedicalSummarizer()

        # Summarize preprocessed report
        result = summarizer.summarize(preprocessed_report, ner_result)

        print(result.abstractive_summary)
        print(result.clinical_note)
        print(result.section_summaries["assessment and plan"])
    """

    def __init__(
        self,
        use_abstractive: bool = True,
        model_name: str = "facebook/bart-large-cnn",
        use_gpu: bool = False,
        summary_length: str = "medium"
    ):
        self.summary_length = summary_length
        self.extractive = ExtractiveSummarizer()

        if use_abstractive:
            self.abstractive = AbstractiveSummarizer(model_name, use_gpu)
        else:
            self.abstractive = None
            logger.info("ℹ️  Abstractive summarization disabled — extractive only")

        logger.info("✅ MedicalSummarizer initialized")

    @timer
    def summarize(
        self,
        preprocessed_report,        # PreprocessedReport from Module A
        ner_result=None,            # NERResult from Module A (optional but enhances output)
        length: str = "medium",     # "short" | "medium" | "detailed"
        section_aware: bool = True  # Summarize each section separately
    ) -> SummaryResult:
        """
        Full summarization pipeline.

        Pipeline:
          1. Extract key sentences per section (Extractive)
          2. Combine section extracts by priority order
          3. Run BART on combined extract (Abstractive)
          4. Generate 1-sentence TL;DR
          5. Generate structured clinical note
          6. Return SummaryResult

        Args:
            preprocessed_report: Output from MedicalPreprocessor.process()
            ner_result: Output from MedicalNERExtractor.extract() (optional)
            length: Summary length preset
            section_aware: If True, summarize each clinical section separately

        Returns:
            SummaryResult with all summaries and clinical note
        """
        original_wc = preprocessed_report.word_count
        logger.info(f"📝 Summarizing {original_wc}-word report (length={length})")

        # ── Step 1: Section-aware extractive summarization ─────────────────
        section_summaries = {}
        combined_extract_parts = []

        if section_aware and preprocessed_report.sections:
            # Sort sections by clinical priority
            sorted_sections = sorted(
                preprocessed_report.sections.items(),
                key=lambda x: SECTION_PRIORITY.get(x[0].lower(), 0.5),
                reverse=True
            )

            for section_name, section in sorted_sections:
                if not section.sentences:
                    continue

                # Higher priority sections get higher compression (keep more)
                priority = SECTION_PRIORITY.get(section_name.lower(), 0.5)
                compression = 0.5 if priority >= 0.8 else 0.3

                extract, _ = self.extractive.summarize(
                    section.sentences,
                    section_name=section_name,
                    compression_ratio=compression,
                    min_sentences=1,
                    max_sentences=6
                )

                if extract:
                    section_summaries[section_name] = extract
                    # Add high-priority sections to combined extract
                    if priority >= 0.5:
                        combined_extract_parts.append(extract)

        else:
            # No sections — summarize all sentences directly
            extract, _ = self.extractive.summarize(
                preprocessed_report.sentences,
                compression_ratio=0.40,
                min_sentences=3,
                max_sentences=12
            )
            combined_extract_parts.append(extract)

        # Combine all section extracts into one passage for BART
        combined_extract = " ".join(combined_extract_parts)

        logger.info(f"📋 Extractive output: {count_words(combined_extract)} words "
                    f"({len(section_summaries)} sections)")

        # ── Step 2: Abstractive summarization ─────────────────────────────
        if self.abstractive and self.abstractive.pipeline:
            abstractive_summary = self.abstractive.summarize(
                combined_extract, length=length
            )
        else:
            # Fallback: use extractive as the "abstractive" summary
            abstractive_summary = truncate_text(combined_extract, max_words=120)

        # ── Step 3: Generate 1-sentence TL;DR ─────────────────────────────
        short_summary = self._generate_short_summary(
            abstractive_summary, preprocessed_report, ner_result
        )

        # ── Step 4: Generate Clinical Note ────────────────────────────────
        clinical_note = self._generate_clinical_note(
            preprocessed_report, ner_result, abstractive_summary
        )

        # ── Step 5: Build result ───────────────────────────────────────────
        summary_wc = count_words(abstractive_summary)
        compression = round(summary_wc / original_wc, 2) if original_wc > 0 else 0

        logger.info(
            f"✅ Summary complete — {summary_wc} words "
            f"({int((1 - compression) * 100)}% compression)"
        )

        return SummaryResult(
            extractive_summary=combined_extract,
            abstractive_summary=abstractive_summary,
            short_summary=short_summary,
            section_summaries=section_summaries,
            clinical_note=clinical_note,
            original_word_count=original_wc,
            summary_word_count=summary_wc,
            compression_ratio=compression,
            model_used=self.abstractive.model_name if self.abstractive else "extractive-only",
            extractive_sentences_used=len(combined_extract_parts),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TL;DR Generator
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_short_summary(
        self,
        abstractive_summary: str,
        preprocessed_report,
        ner_result=None
    ) -> str:
        """
        Generate a 1-2 sentence TL;DR.

        Strategy:
          - If NER available: construct from patient demographics + primary diagnosis
          - Else: take first 2 sentences of abstractive summary
        """
        if ner_result:
            parts = []

            # Patient demographics (from vitals/entities)
            vitals = ner_result.vitals if ner_result.vitals else {}

            # Primary diagnosis
            if ner_result.diseases:
                primary_dx = ner_result.diseases[0]
                parts.append(f"Primary diagnosis: {primary_dx}.")

            # Key symptoms
            if ner_result.symptoms:
                top_symptoms = ", ".join(ner_result.symptoms[:3])
                parts.append(f"Presenting symptoms: {top_symptoms}.")

            # Key medications
            if ner_result.medications:
                top_meds = ", ".join([m.name for m in ner_result.medications[:3]])
                parts.append(f"Medications: {top_meds}.")

            if parts:
                return " ".join(parts)

        # Fallback: first 2 sentences of abstractive summary
        sentences = re.split(r'(?<=[.!?])\s+', abstractive_summary)
        return " ".join(sentences[:2]) if sentences else abstractive_summary[:200]

    # ─────────────────────────────────────────────────────────────────────────
    # Clinical Note Generator
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_clinical_note(
        self,
        preprocessed_report,
        ner_result,
        summary: str
    ) -> str:
        """
        Generate a structured SOAP-style clinical note from extracted data.

        SOAP = Subjective, Objective, Assessment, Plan
        This is the output that looks most impressive in demos.
        """
        from datetime import datetime
        lines = []

        lines.append("=" * 65)
        lines.append("         AUTO-GENERATED CLINICAL NOTE (AI-ASSISTED)")
        lines.append(f"         Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 65)

        # ── SUBJECTIVE (Chief Complaint + History) ─────────────────────────
        lines.append("\n[S] SUBJECTIVE")
        lines.append("─" * 40)

        # Chief complaint from section
        cc_section = preprocessed_report.sections.get("chief complaint", None)
        if cc_section:
            lines.append(f"Chief Complaint: {cc_section.clean_text[:200].strip()}")
        elif ner_result and ner_result.symptoms:
            complaints = ", ".join(ner_result.symptoms[:4])
            lines.append(f"Chief Complaint: {complaints}")
        else:
            lines.append("Chief Complaint: See original report")

        # HPI summary
        hpi_section = preprocessed_report.sections.get(
            "history of present illness",
            preprocessed_report.sections.get("hpi", None)
        )
        if hpi_section and hpi_section.sentences:
            hpi_extract = " ".join(hpi_section.sentences[:3])
            lines.append(f"\nHistory: {truncate_text(hpi_extract, 80)}")

        # ── OBJECTIVE (Vitals + Labs) ──────────────────────────────────────
        lines.append("\n[O] OBJECTIVE")
        lines.append("─" * 40)

        # Vitals
        if ner_result and ner_result.vitals:
            lines.append("Vital Signs:")
            for vital, value in ner_result.vitals.items():
                lines.append(f"  • {vital}: {value}")
        else:
            vitals = preprocessed_report.sections.get("vital signs", None)
            if vitals:
                lines.append(f"Vitals: {vitals.clean_text[:150].strip()}")

        # Lab values
        if ner_result and ner_result.lab_values:
            lines.append("\nKey Lab Values:")
            for lab in ner_result.lab_values[:8]:  # Top 8 labs
                unit_str = f" {lab.unit}" if lab.unit else ""
                lines.append(f"  • {lab.test_name}: {lab.value}{unit_str}")

        # ── ASSESSMENT (Diagnoses + ICD) ───────────────────────────────────
        lines.append("\n[A] ASSESSMENT")
        lines.append("─" * 40)

        if ner_result and ner_result.diseases:
            lines.append("Diagnoses:")
            # Filter out pure symptoms and body substances from disease list
            symptom_words = {
                "pain", "chest pain", "shortness", "breath", "nausea",
                "vomiting", "diarrhea", "headache", "dizziness", "fatigue",
                "weakness", "fever", "cough", "diaphoresis", "sweating",
                "swelling", "edema", "syncope", "confusion", "palpitation",
                "oxygen", "cholesterol", "glucose", "sodium", "potassium",
                "air", "blood", "room", "water", "protein",
            }
            true_diseases = [
                d for d in ner_result.diseases
                if not any(sw in d.lower() for sw in symptom_words)
                and len(d.split()) <= 6   # Ignore overly long false positives
            ]
            display_diseases = true_diseases if true_diseases else ner_result.diseases[:4]
            for i, disease in enumerate(display_diseases[:6], 1):
                lines.append(f"  {i}. {disease}")
        else:
            assess_section = preprocessed_report.sections.get(
                "assessment and plan",
                preprocessed_report.sections.get("assessment", None)
            )
            if assess_section:
                lines.append(truncate_text(assess_section.clean_text, 100))

        # ── PLAN (Medications + Procedures) ───────────────────────────────
        lines.append("\n[P] PLAN")
        lines.append("─" * 40)

        # Medications — use structured medications (rule-based, always works)
        if ner_result and ner_result.medications:
            lines.append("Medications:")
            for med in ner_result.medications[:8]:
                med_str = f"  • {med.name}"
                if med.dosage:
                    med_str += f" {med.dosage}"
                if med.frequency:
                    med_str += f" {med.frequency}"
                if med.route:
                    med_str += f" ({med.route})"
                lines.append(med_str)
        elif ner_result and ner_result.drugs:
            lines.append("Medications:")
            for drug in ner_result.drugs[:8]:
                lines.append(f"  • {drug}")

        if ner_result and ner_result.procedures:
            lines.append("\nProcedures:")
            for proc in ner_result.procedures[:5]:
                lines.append(f"  • {proc}")

        # ── SUMMARY ───────────────────────────────────────────────────────
        if summary:
            lines.append("\n[SUMMARY]")
            lines.append("─" * 40)
            lines.append(truncate_text(summary, 80))

        # ── Footer ────────────────────────────────────────────────────────
        lines.append("\n" + "=" * 65)
        lines.append("⚠️  AI-GENERATED NOTE — Requires physician review before use")
        lines.append("⚠️  NOT for clinical decision-making without verification")
        lines.append("=" * 65)

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self, result: SummaryResult) -> Dict:
        """Return summary statistics for display."""
        return {
            "original_words":      result.original_word_count,
            "summary_words":       result.summary_word_count,
            "compression_ratio":   f"{result.compression_ratio:.0%}",
            "compression_percent": f"{int((1 - result.compression_ratio) * 100)}% shorter",
            "sections_summarized": len(result.section_summaries),
            "model_used":          result.model_used,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test — No model download needed (extractive only)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from modules.preprocessor import MedicalPreprocessor
    from modules.ner_extractor import MedicalNERExtractor

    sample = """
    CHIEF COMPLAINT:
    58-year-old male complains of chest pain and shortness of breath for 2 hours.

    HISTORY OF PRESENT ILLNESS:
    Patient is a 58-year-old male with past medical history of type 2 diabetes
    mellitus, hypertension, and hyperlipidemia. He presented to the emergency
    department via ambulance with acute onset substernal chest pain radiating to
    the left arm and jaw, associated with diaphoresis, nausea, and shortness of
    breath. Blood pressure was 145/92 mmHg, heart rate 102 bpm, oxygen saturation
    96% on room air.

    LABORATORY RESULTS:
    Troponin I was critically elevated at 2.4 ng/mL. HbA1c was 9.2% indicating
    poor diabetic control. LDL cholesterol was 142 mg/dL.

    ASSESSMENT AND PLAN:
    1. Inferior STEMI — Percutaneous coronary intervention performed successfully.
       Drug-eluting stent placed in right coronary artery. Started on Aspirin
       325 mg, Clopidogrel 75 mg, Atorvastatin 80 mg, and Metoprolol 50 mg.
    2. Type 2 Diabetes Mellitus — HbA1c 9.2%, poorly controlled. Added
       Empagliflozin 10 mg daily for cardioprotective benefit.
    3. Hypertension — Continue Lisinopril 10 mg daily, target BP < 130/80 mmHg.
    """

    print("🧪 Running Module B smoke test...\n")

    # Run Module A first
    preprocessor = MedicalPreprocessor()
    ner = MedicalNERExtractor()

    preprocessed = preprocessor.process(sample)
    ner_result = ner.extract(preprocessed.clean_text)

    # Run Module B — extractive only (no model download for smoke test)
    summarizer = MedicalSummarizer(use_abstractive=False)
    result = summarizer.summarize(preprocessed, ner_result, length="medium")

    print("=" * 60)
    print("📋 EXTRACTIVE SUMMARY:")
    print(result.extractive_summary[:400])

    print("\n" + "=" * 60)
    print("📝 SECTION SUMMARIES:")
    for section, summary in result.section_summaries.items():
        print(f"\n  [{section.upper()}]")
        print(f"  {summary[:200]}")

    print("\n" + "=" * 60)
    print("⚡ TL;DR:")
    print(result.short_summary)

    print("\n" + "=" * 60)
    print("🏥 CLINICAL NOTE:")
    print(result.clinical_note)

    print("\n" + "=" * 60)
    stats = summarizer.get_stats(result)
    print("📊 STATS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
