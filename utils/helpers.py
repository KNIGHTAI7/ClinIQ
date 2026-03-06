# utils/helpers.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared utility functions used across all modules
# ─────────────────────────────────────────────────────────────────────────────

import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure loguru logger with file + console output."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"medical_nlp_{datetime.now().strftime('%Y%m%d')}.log"

    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
    )

    # File handler
    logger.add(
        sink=str(log_file),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timer Decorator
# ─────────────────────────────────────────────────────────────────────────────

def timer(func):
    """Decorator to measure and log execution time of any function."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"⏱️  {func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Text Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Return word count of text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Rough sentence count using punctuation."""
    return len(re.findall(r'[.!?]+', text))


def truncate_text(text: str, max_words: int = 512) -> str:
    """Truncate text to max_words while preserving sentence boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Try to end at a sentence boundary
    last_period = truncated.rfind(".")
    if last_period > len(truncated) * 0.7:
        return truncated[:last_period + 1]
    return truncated + "..."


def clean_whitespace(text: str) -> str:
    """Normalize whitespace — collapse multiple spaces/newlines."""
    text = re.sub(r'\n{3,}', '\n\n', text)   # Max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)         # Collapse multiple spaces
    text = re.sub(r'\t', ' ', text)             # Replace tabs with spaces
    return text.strip()


def remove_special_chars(text: str, keep_medical: bool = True) -> str:
    """
    Remove special characters while preserving medically relevant symbols.
    keep_medical=True preserves: /, %, +, -, =, <, >  (used in lab values)
    """
    if keep_medical:
        # Keep alphanumeric + medical punctuation + spaces + newlines
        text = re.sub(r'[^\w\s\.\,\:\;\!\?\-\/\%\+\=\<\>\(\)\[\]]', ' ', text)
    else:
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
    return clean_whitespace(text)


def is_valid_medical_text(text: str, min_words: int = 20) -> bool:
    """Check if text is long enough and likely medical content."""
    if count_words(text) < min_words:
        return False
    # Check for at least some medical indicator words
    medical_indicators = [
        "patient", "diagnosis", "treatment", "medication",
        "symptom", "history", "blood", "pain", "mg", "doctor"
    ]
    text_lower = text.lower()
    matches = sum(1 for word in medical_indicators if word in text_lower)
    return matches >= 2


# ─────────────────────────────────────────────────────────────────────────────
# File Utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_text_file(filepath: str) -> str:
    """Read a text file with error handling."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as formatted JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file into dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report_id(text: str) -> str:
    """Generate a unique ID from report text using MD5 hash."""
    return hashlib.md5(text.encode()).hexdigest()[:12].upper()


# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_entity_output(entities: List[Dict]) -> str:
    """Format extracted entities as readable text for display."""
    if not entities:
        return "No entities found."

    output = []
    # Group by entity type
    entity_groups: Dict[str, List[str]] = {}
    for entity in entities:
        label = entity.get("label", "UNKNOWN")
        text = entity.get("text", "")
        if label not in entity_groups:
            entity_groups[label] = []
        if text not in entity_groups[label]:
            entity_groups[label].append(text)

    for label, items in sorted(entity_groups.items()):
        output.append(f"\n🏷️  {label}:")
        for item in items:
            output.append(f"   • {item}")

    return "\n".join(output)


def format_clinical_note(extracted_data: Dict[str, Any]) -> str:
    """
    Generate a structured clinical note from extracted entities.
    This is the final output template filled with NER results.
    """
    note = []
    note.append("=" * 60)
    note.append("AUTO-GENERATED CLINICAL NOTE")
    note.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    note.append("=" * 60)

    # Patient Info
    patient_info = extracted_data.get("patient_info", {})
    note.append(f"\nPATIENT: {patient_info.get('age', 'N/A')} {patient_info.get('gender', '')}")

    # Chief Complaint
    symptoms = extracted_data.get("symptoms", [])
    if symptoms:
        note.append(f"\nCHIEF COMPLAINT:\n  {', '.join(symptoms[:3])}")

    # Diagnoses
    diagnoses = extracted_data.get("diagnoses", [])
    if diagnoses:
        note.append("\nASSESSMENT / DIAGNOSES:")
        for i, dx in enumerate(diagnoses, 1):
            icd = dx.get("icd_code", "")
            note.append(f"  {i}. {dx.get('text', '')}  {f'[{icd}]' if icd else ''}")

    # Medications
    medications = extracted_data.get("medications", [])
    if medications:
        note.append("\nMEDICATIONS:")
        for med in medications:
            dosage = med.get("dosage", "")
            freq = med.get("frequency", "")
            note.append(f"  • {med.get('name', '')} {dosage} {freq}".strip())

    # Vitals
    vitals = extracted_data.get("vitals", {})
    if vitals:
        note.append("\nVITALS:")
        for vital, value in vitals.items():
            note.append(f"  {vital}: {value}")

    # Summary
    summary = extracted_data.get("summary", "")
    if summary:
        note.append(f"\nSUMMARY:\n  {summary}")

    note.append("\n" + "=" * 60)
    note.append("⚠️  AI-GENERATED — Requires physician review before use")
    note.append("=" * 60)

    return "\n".join(note)
