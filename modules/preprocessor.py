# modules/preprocessor.py
# ─────────────────────────────────────────────────────────────────────────────
# MODULE A — PART 1: Medical Text Preprocessing Pipeline
#
# What this does:
#   1. Accepts raw medical text (string, file path, or PDF)
#   2. Detects & parses clinical sections (HPI, Assessment, Plan, etc.)
#   3. Expands medical abbreviations
#   4. Normalizes text (units, numbers, dates)
#   5. Tokenizes into sentences with medical-aware segmentation
#   6. Returns structured, clean text ready for NER + Summarization
# ─────────────────────────────────────────────────────────────────────────────

import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from loguru import logger

# Add parent dir to path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.medical_abbreviations import MEDICAL_ABBREVIATIONS, SECTION_HEADERS
from utils.helpers import clean_whitespace, count_words, timer


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MedicalSection:
    """Represents a single parsed section from a clinical note."""
    name: str                           # e.g. "History of Present Illness"
    raw_text: str                       # Original unprocessed text
    clean_text: str                     # Preprocessed text
    sentences: List[str] = field(default_factory=list)  # Tokenized sentences
    word_count: int = 0


@dataclass
class PreprocessedReport:
    """Full output of the preprocessing pipeline."""
    report_id: str
    raw_text: str                       # Original input (never modified)
    clean_text: str                     # Full cleaned text
    sections: Dict[str, MedicalSection] # Parsed sections
    sentences: List[str]                # All sentences (flat list)
    word_count: int
    abbreviations_expanded: int         # Count of abbreviations replaced
    source_type: str                    # "text", "file", "pdf"
    metadata: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Medical Preprocessor Class
# ─────────────────────────────────────────────────────────────────────────────

class MedicalPreprocessor:
    """
    Production-grade medical text preprocessor.

    Usage:
        preprocessor = MedicalPreprocessor()
        result = preprocessor.process(text)

        # Access clean text
        print(result.clean_text)

        # Access specific sections
        if "assessment" in result.sections:
            print(result.sections["assessment"].clean_text)

        # Access all sentences
        for sentence in result.sentences:
            print(sentence)
    """

    def __init__(self, expand_abbreviations: bool = True, verbose: bool = False):
        self.expand_abbreviations = expand_abbreviations
        self.verbose = verbose
        self.abbreviation_map = MEDICAL_ABBREVIATIONS
        self.section_headers = SECTION_HEADERS

        # Compiled regex patterns (compile once for performance)
        self._compile_patterns()

        logger.info("✅ MedicalPreprocessor initialized")

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns used during processing."""

        # Pattern: detect section headers (case-insensitive, followed by colon/newline)
        header_pattern = "|".join(
            re.escape(h) for h in sorted(self.section_headers, key=len, reverse=True)
        )
        self.section_pattern = re.compile(
            rf'(?:^|\n)\s*({header_pattern})\s*[:\-]?\s*\n?',
            re.IGNORECASE | re.MULTILINE
        )

        # Pattern: vital signs (e.g., "BP 120/80 mmHg", "HR: 88 bpm")
        self.vitals_pattern = re.compile(
            r'\b(BP|HR|RR|Temp|SpO2|O2 sat|BMI|Wt|Ht)\s*[:\-]?\s*'
            r'(\d+[\./]?\d*\s*(?:mmHg|bpm|rpm|°[CF]|%|kg|lbs|cm|in)?)',
            re.IGNORECASE
        )

        # Pattern: medication dosages (e.g., "Metformin 500mg", "Aspirin 81 mg")
        self.dosage_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|mEq|units?|IU|mmol)\b',
            re.IGNORECASE
        )

        # Pattern: dates in various formats
        self.date_pattern = re.compile(
            r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'           # MM/DD/YYYY
            r'|(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
            r'\w*\.?\s+\d{1,2},?\s+\d{4})\b',                   # Month DD, YYYY
            re.IGNORECASE
        )

        # Pattern: lab values (e.g., "Na 138 mEq/L", "WBC 11.2 K/uL")
        self.lab_pattern = re.compile(
            r'\b(WBC|RBC|Hgb|Hct|Plt|Na|K|Cl|CO2|BUN|Cr|Glucose|Ca|Mg|'
            r'ALT|AST|ALP|TBili|INR|PT|PTT|HbA1c|TSH|LDL|HDL|TG|CK|Trop|BNP|CRP|ESR)\s*'
            r'[:\-]?\s*(\d+(?:\.\d+)?)\s*([a-zA-Z\/\%µ]*)',
            re.IGNORECASE
        )

        # Pattern: remove irrelevant formatting artifacts
        self.artifact_pattern = re.compile(
            r'(?:Page\s+\d+\s+of\s+\d+)'        # Page numbers
            r'|(?:CONFIDENTIAL)'                   # Headers
            r'|(?:[-=_]{4,})'                     # Divider lines
            r'|(?:\[\s*\])'                        # Empty checkboxes
            r'|(?:\*{3,})',                        # Star separators
            re.IGNORECASE
        )

        # Pattern: medical sentence boundaries (more nuanced than standard)
        # Avoids splitting on common medical abbreviations like "Dr.", "Fig.", "vs."
        self.sentence_end_pattern = re.compile(
            r'(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bvs)'
            r'(?<!\b[A-Z])(?<!\d)'                # Not after uppercase single letter or digit
            r'[.!?]\s+(?=[A-Z])',                  # Period + space + capital
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────

    @timer
    def process(
        self,
        input_data: Union[str, Path],
        source_type: str = "auto",
        report_id: Optional[str] = None
    ) -> PreprocessedReport:
        """
        Main preprocessing pipeline. Accepts raw text, file path, or PDF path.

        Args:
            input_data: Raw text string or path to .txt / .pdf file
            source_type: "text", "file", "pdf", or "auto" (auto-detect)
            report_id: Optional custom ID; auto-generated if None

        Returns:
            PreprocessedReport with all processed fields populated
        """
        # ── Step 1: Load raw text ──────────────────────────────────────────
        raw_text, detected_source = self._load_input(input_data, source_type)
        if not raw_text:
            raise ValueError("Input text is empty or could not be read.")

        logger.info(f"📄 Loaded report ({detected_source}) — {count_words(raw_text)} words")

        # ── Step 2: Basic cleaning ─────────────────────────────────────────
        text = self._basic_clean(raw_text)

        # ── Step 3: Expand abbreviations ───────────────────────────────────
        text, abbrev_count = self._expand_abbreviations(text)
        logger.info(f"🔤 Expanded {abbrev_count} abbreviations")

        # ── Step 4: Normalize medical text ────────────────────────────────
        text = self._normalize(text)

        # ── Step 5: Parse sections ─────────────────────────────────────────
        sections = self._parse_sections(text)
        logger.info(f"📋 Detected {len(sections)} sections: {list(sections.keys())}")

        # ── Step 6: Sentence tokenization ─────────────────────────────────
        sentences = self._tokenize_sentences(text)
        logger.info(f"✂️  Extracted {len(sentences)} sentences")

        # ── Step 7: Build output ───────────────────────────────────────────
        if report_id is None:
            import hashlib
            report_id = hashlib.md5(raw_text.encode()).hexdigest()[:10].upper()

        return PreprocessedReport(
            report_id=report_id,
            raw_text=raw_text,
            clean_text=text,
            sections=sections,
            sentences=sentences,
            word_count=count_words(text),
            abbreviations_expanded=abbrev_count,
            source_type=detected_source,
            metadata={
                "original_word_count": count_words(raw_text),
                "section_count": len(sections),
                "sentence_count": len(sentences),
            }
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Input Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_input(
        self, input_data: Union[str, Path], source_type: str
    ) -> Tuple[str, str]:
        """Detect input type and load text accordingly."""

        if source_type == "text" or (
            isinstance(input_data, str) and source_type == "auto"
            and len(input_data) > 100  # Long string = raw text, not a file path
        ):
            return input_data, "text"

        path = Path(str(input_data))

        if not path.exists():
            # If path doesn't exist, treat as raw text
            return str(input_data), "text"

        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path), "pdf"
        else:
            # .txt or any other text file
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), "file"

    def _load_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using pdfplumber with fallback to PyMuPDF."""
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"📑 Extracted PDF via pdfplumber ({len(text)} chars)")
        except ImportError:
            logger.warning("pdfplumber not available, trying PyMuPDF...")
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(pdf_path))
                for page in doc:
                    text += page.get_text() + "\n"
                logger.info(f"📑 Extracted PDF via PyMuPDF ({len(text)} chars)")
            except ImportError:
                raise ImportError(
                    "Neither pdfplumber nor PyMuPDF installed. "
                    "Run: pip install pdfplumber"
                )

        if not text.strip():
            logger.warning("⚠️  PDF text extraction returned empty — may be a scanned PDF")
            logger.info("💡 Tip: Install pytesseract for OCR support on scanned PDFs")

        return text

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Basic Cleaning
    # ─────────────────────────────────────────────────────────────────────────

    def _basic_clean(self, text: str) -> str:
        """Remove artifacts, normalize whitespace, fix encoding."""

        # Remove PDF/formatting artifacts
        text = self.artifact_pattern.sub(' ', text)

        # Fix common OCR errors in medical text
        text = text.replace('|', 'I')      # | often misread as I
        text = text.replace('0', 'O') if False else text  # Disabled: too aggressive

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Normalize whitespace
        text = clean_whitespace(text)

        return text

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Abbreviation Expansion
    # ─────────────────────────────────────────────────────────────────────────

    def _expand_abbreviations(self, text: str) -> Tuple[str, int]:
        """
        Replace medical abbreviations with full forms.
        Uses word-boundary matching to avoid partial replacements.
        Returns (expanded_text, count_of_replacements)
        """
        if not self.expand_abbreviations:
            return text, 0

        count = 0

        # ── Protect lab units BEFORE abbreviation expansion ───────────────
        # These short abbreviations must NOT be expanded when used as units
        # e.g. "ng/mL" should NOT become "nasogastric/milliliters"
        # Strategy: temporarily replace unit patterns with placeholders
        LAB_UNIT_PATTERN = re.compile(
            r'(\d+\.?\d*)\s*(ng|pg|µg|ug|mEq|mIU|mcg|IU)\s*(/\s*(?:mL|L|dL|uL|kg))',
            re.IGNORECASE
        )
        placeholders = {}
        def protect_unit(match):
            key = f"__LABUNIT{len(placeholders)}__"
            placeholders[key] = match.group(0)
            return key
        text = LAB_UNIT_PATTERN.sub(protect_unit, text)

        # Sort by length (longest first) to avoid partial matches
        sorted_abbrevs = sorted(
            self.abbreviation_map.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for abbrev, expansion in sorted_abbrevs:
            # Create pattern with word boundaries, case-insensitive
            # Handle abbreviations with special chars like "c/o", "w/"
            escaped = re.escape(abbrev)
            pattern = rf'\b{escaped}\b'

            # Count matches before replacement
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
                count += matches

        # ── Restore protected lab units ────────────────────────────────────
        for key, original in placeholders.items():
            text = text.replace(key, original)

        return text, count

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Normalization
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """
        Normalize medical-specific patterns for consistent downstream processing.
        Preserves clinical meaning while standardizing format.
        """

        # Normalize vital sign formats → consistent "BP: 120/80 mmHg"
        text = self._normalize_vitals(text)

        # Normalize dosages → consistent "500 mg" (add space between number and unit)
        text = re.sub(
            r'(\d+(?:\.\d+)?)(mg|mcg|g|ml|mEq|IU|mmol)',
            r'\1 \2',
            text, flags=re.IGNORECASE
        )

        # Normalize dates → standardize to Month DD, YYYY where possible
        # (keep as-is for now; NER will handle date extraction)

        # Normalize temperature formats → "98.6°F" or "37.0°C"
        # NOTE: Must NOT match "A1c" or "HbA1c" — use negative lookbehind for letters
        text = re.sub(r'(?<!\w)(\d+\.?\d*)\s*°?\s*([FC])\b(?![\w])', r'\1°\2', text)

        # Normalize percentage → "XX%" only when written as words, not already %
        text = re.sub(r'(\d+)\s+percent\b(?!\s*%)', r'\1%', text, flags=re.IGNORECASE)

        # Fix split lab results (e.g., "Na+ 138" → "Sodium 138")
        lab_normalizations = {
            r'\bNa\+?\b': 'Sodium',
            r'\bK\+?\b': 'Potassium',
            r'\bCl-?\b': 'Chloride',
            r'\bHgb\b': 'Hemoglobin',
            r'\bHct\b': 'Hematocrit',
            r'\bPlt\b': 'Platelets',
        }
        for pattern, replacement in lab_normalizations.items():
            text = re.sub(pattern, replacement, text)

        # Normalize "positive" / "negative" shorthand
        text = re.sub(r'\b\(\+\)', '(positive)', text)
        text = re.sub(r'\b\(-\)', '(negative)', text)
        text = re.sub(r'\b\+ve\b', 'positive', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\-ve\b', 'negative', text, flags=re.IGNORECASE)

        return text

    def _normalize_vitals(self, text: str) -> str:
        """Standardize vital sign representation."""
        # Ensure BP always has proper format — only when mmHg not already present
        text = re.sub(
            r'\bblood pressure\s+(\d{2,3})\s*/\s*(\d{2,3})(?!\s*mmHg)',
            r'blood pressure \1/\2 mmHg',
            text, flags=re.IGNORECASE
        )
        # Ensure SpO2 is followed by % — ONLY when % is NOT already there
        # Use a function to avoid double-appending
        def add_percent(m):
            val = m.group(1)
            after = m.group(2)
            if after.startswith('%'):
                return m.group(0)   # already has %, leave unchanged
            return f'oxygen saturation {val}%'

        text = re.sub(
            r'\boxygen saturation\s+(\d{2,3})(.*?)(?=\s|$)',
            add_percent,
            text, flags=re.IGNORECASE
        )
        return text

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Section Parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_sections(self, text: str) -> Dict[str, MedicalSection]:
        """
        Detect and extract named clinical sections from the report.

        Handles various formats:
        - "CHIEF COMPLAINT:" (uppercase with colon)
        - "Chief Complaint" (title case, no colon)
        - "HPI:" (abbreviation)
        """
        sections = {}

        # Find all section header positions
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            # No sections detected — treat entire text as one section
            logger.warning("⚠️  No clinical sections detected — processing as unstructured text")
            raw = text.strip()
            sections["full_report"] = MedicalSection(
                name="Full Report",
                raw_text=raw,
                clean_text=raw,
                sentences=self._tokenize_sentences(raw),
                word_count=count_words(raw)
            )
            return sections

        # Extract text between consecutive section headers
        for i, match in enumerate(matches):
            section_name = match.group(1).strip().lower()

            # Section content: from end of this header to start of next header
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section_text = text[start:end].strip()

            if not section_text:
                continue

            sentences = self._tokenize_sentences(section_text)

            sections[section_name] = MedicalSection(
                name=section_name.title(),
                raw_text=section_text,
                clean_text=section_text,
                sentences=sentences,
                word_count=count_words(section_text)
            )

        return sections

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Sentence Tokenization
    # ─────────────────────────────────────────────────────────────────────────

    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Medical-aware sentence tokenizer.

        Standard tokenizers (NLTK, spaCy) often fail on clinical notes because:
        - Abbreviations like "Dr." and "vs." cause false splits
        - Bullet points / numbered lists aren't proper sentences
        - Lab results span multiple lines

        This tokenizer handles all those edge cases.
        """
        if not text.strip():
            return []

        sentences = []

        # Split on bullet points / numbered lists first
        bullet_pattern = re.compile(r'\n\s*(?:[-•*]\s+|\d+[\.\)]\s+)')
        bullet_items = bullet_pattern.split(text)

        for item in bullet_items:
            item = item.strip()
            if not item:
                continue

            # Split each bullet item on sentence boundaries
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', item)

            for part in parts:
                part = part.strip()
                if len(part.split()) >= 3:  # Minimum 3 words to be a sentence
                    sentences.append(part)

        # If no splits happened, return the text as one sentence
        if not sentences:
            sentences = [text.strip()]

        return sentences

    # ─────────────────────────────────────────────────────────────────────────
    # Utility: Extract Vitals
    # ─────────────────────────────────────────────────────────────────────────

    def extract_vitals(self, text: str) -> Dict[str, str]:
        """
        Quick extraction of structured vital signs from text.
        Returns dict like: {"Blood Pressure": "120/80 mmHg", "HR": "88 bpm"}
        """
        vitals = {}

        vital_patterns = {
            "Blood Pressure": r'(?:blood pressure|BP)\s*[:\-]?\s*(\d{2,3}/\d{2,3}(?:\s*mmHg)?)',
            "Heart Rate": r'(?:heart rate|HR|pulse)\s*[:\-]?\s*(\d{2,3}(?:\s*bpm)?)',
            "Respiratory Rate": r'(?:respiratory rate|RR)\s*[:\-]?\s*(\d{1,2}(?:\s*(?:rpm|breaths/min))?)',
            "Temperature": r'(?:temperature|Temp)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?(?:\s*°?[FCfc])?)',
            "Oxygen Saturation": r'(?:oxygen saturation|SpO2|O2 sat)\s*[:\-]?\s*(\d{2,3}(?:\s*%)?)',
            "BMI": r'(?:BMI|body mass index)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)',
            "Weight": r'(?:weight|Wt)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?(?:\s*(?:kg|lbs))?)',
            "Height": r'(?:height|Ht)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?(?:\s*(?:cm|in|ft))?)',
        }

        for vital_name, pattern in vital_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vitals[vital_name] = match.group(1).strip()

        return vitals

    # ─────────────────────────────────────────────────────────────────────────
    # Utility: Summary Stats
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self, report: PreprocessedReport) -> Dict:
        """Return preprocessing statistics for display/debugging."""
        return {
            "report_id": report.report_id,
            "source_type": report.source_type,
            "original_word_count": report.metadata.get("original_word_count", 0),
            "cleaned_word_count": report.word_count,
            "sections_detected": list(report.sections.keys()),
            "sentence_count": len(report.sentences),
            "abbreviations_expanded": report.abbreviations_expanded,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with a synthetic medical note
    sample = """
    CHIEF COMPLAINT:
    58 y/o M c/o CP and SOB for the past 2 hours.

    HISTORY OF PRESENT ILLNESS:
    Pt is a 58-year-old male with PMH of CAD, HTN, and T2DM who presents to the ED
    with acute onset CP radiating to the left arm. BP 145/92 mmHg, HR 102 bpm,
    RR 18 rpm, SpO2 96% on room air, Temp 98.6°F.

    MEDICATIONS:
    Metformin 500mg bid, Lisinopril 10mg qd, Atorvastatin 40mg qhs, Aspirin 81mg qd.

    ASSESSMENT AND PLAN:
    1. Inferior STEMI - Activate cath lab, start heparin IV, aspirin 325mg stat.
    2. T2DM - Hold Metformin, monitor glucose q6h.
    3. HTN - Continue Lisinopril, target BP < 130/80.
    """

    print("🧪 Running preprocessor smoke test...\n")
    preprocessor = MedicalPreprocessor(verbose=True)
    result = preprocessor.process(sample)

    print(f"\n✅ Report ID: {result.report_id}")
    print(f"📊 Word count: {result.word_count}")
    print(f"🔤 Abbreviations expanded: {result.abbreviations_expanded}")
    print(f"📋 Sections: {list(result.sections.keys())}")
    print(f"✂️  Sentences: {len(result.sentences)}")
    print(f"\n📝 Clean Text (first 300 chars):\n{result.clean_text[:300]}...")

    print("\n💉 Extracted Vitals:")
    vitals = preprocessor.extract_vitals(result.clean_text)
    for k, v in vitals.items():
        print(f"  {k}: {v}")
