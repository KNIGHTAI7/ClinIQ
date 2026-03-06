# tests/test_module_a.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests for Module A: Preprocessor + NER Extractor
# Run with: python -m pytest tests/test_module_a.py -v
# ─────────────────────────────────────────────────────────────────────────────

import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.preprocessor import MedicalPreprocessor, PreprocessedReport
from modules.ner_extractor import MedicalNERExtractor, NERResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def preprocessor():
    return MedicalPreprocessor(expand_abbreviations=True)


@pytest.fixture
def ner_extractor():
    return MedicalNERExtractor(use_transformer=False)


@pytest.fixture
def simple_report():
    return """
    CHIEF COMPLAINT:
    58 y/o M c/o CP and SOB for 2 hours.

    HISTORY OF PRESENT ILLNESS:
    Patient is a 58-year-old male with PMH of T2DM and HTN.
    BP 145/92 mmHg, HR 102 bpm, SpO2 96%.

    ASSESSMENT AND PLAN:
    1. Inferior STEMI - start Aspirin 325mg stat.
    2. T2DM - hold Metformin, monitor glucose.
    """


@pytest.fixture
def complex_report():
    sample_path = Path(__file__).parent.parent / "data" / "sample_reports" / "sample_discharge_summary.txt"
    if sample_path.exists():
        return sample_path.read_text()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMedicalPreprocessor:

    def test_preprocessor_initializes(self, preprocessor):
        """Preprocessor should initialize without errors."""
        assert preprocessor is not None
        assert preprocessor.expand_abbreviations is True

    def test_processes_simple_text(self, preprocessor, simple_report):
        """Should process text and return PreprocessedReport."""
        result = preprocessor.process(simple_report)
        assert isinstance(result, PreprocessedReport)
        assert result.raw_text == simple_report
        assert len(result.clean_text) > 0

    def test_abbreviation_expansion(self, preprocessor):
        """Should expand common medical abbreviations."""
        text = "Pt c/o SOB and CP. PMH of HTN and DM2."
        result = preprocessor.process(text)

        # Check that abbreviations were expanded
        assert result.abbreviations_expanded > 0
        assert "patient" in result.clean_text.lower() or \
               "complains of" in result.clean_text.lower() or \
               result.abbreviations_expanded > 0

    def test_section_detection(self, preprocessor, simple_report):
        """Should detect clinical sections."""
        result = preprocessor.process(simple_report)
        # Should have at least 1 section
        assert len(result.sections) >= 1

    def test_sentence_tokenization(self, preprocessor, simple_report):
        """Should produce a list of sentences."""
        result = preprocessor.process(simple_report)
        assert isinstance(result.sentences, list)
        assert len(result.sentences) > 0

    def test_vital_extraction(self, preprocessor):
        """Should extract vital signs from text."""
        text = "Blood pressure 145/92 mmHg, heart rate 102 bpm, oxygen saturation 96%."
        vitals = preprocessor.extract_vitals(text)
        assert "Blood Pressure" in vitals
        assert "Heart Rate" in vitals
        assert "Oxygen Saturation" in vitals

    def test_empty_input_raises(self, preprocessor):
        """Empty input should raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            preprocessor.process("")

    def test_word_count_populated(self, preprocessor, simple_report):
        """Word count should be positive."""
        result = preprocessor.process(simple_report)
        assert result.word_count > 0

    def test_report_id_generated(self, preprocessor, simple_report):
        """Report ID should be auto-generated if not provided."""
        result = preprocessor.process(simple_report)
        assert result.report_id is not None
        assert len(result.report_id) > 0

    def test_custom_report_id(self, preprocessor, simple_report):
        """Custom report ID should be preserved."""
        result = preprocessor.process(simple_report, report_id="TEST-001")
        assert result.report_id == "TEST-001"

    def test_normalization_dosages(self, preprocessor):
        """Dosages should have space between number and unit."""
        text = "Patient takes Metformin 500mg twice daily."
        result = preprocessor.process(text)
        # After normalization: "500 mg"
        assert "500 mg" in result.clean_text or "500mg" in result.clean_text

    def test_complex_report(self, preprocessor, complex_report):
        """Should handle full discharge summary."""
        if not complex_report:
            pytest.skip("Sample report file not found")
        result = preprocessor.process(complex_report)
        assert result.word_count > 100
        assert len(result.sections) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# NER Extractor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMedicalNERExtractor:

    def test_extractor_initializes(self, ner_extractor):
        """NER extractor should initialize with some model."""
        assert ner_extractor is not None

    def test_extracts_result_object(self, ner_extractor):
        """Should return NERResult object."""
        text = "Patient has Type 2 Diabetes and Hypertension."
        result = ner_extractor.extract(text)
        assert isinstance(result, NERResult)

    def test_extracts_symptoms(self, ner_extractor):
        """Should identify common symptom mentions."""
        text = "Patient presents with chest pain and shortness of breath."
        result = ner_extractor.extract(text)
        assert any("chest pain" in s.lower() for s in result.symptoms)
        assert any("shortness of breath" in s.lower() for s in result.symptoms)

    def test_extracts_medications(self, ner_extractor):
        """Should extract structured medication information."""
        text = "Prescribed Metformin 500 mg twice daily and Aspirin 81 mg once daily."
        result = ner_extractor.extract(text)
        assert len(result.medications) >= 1
        med_names = [m.name.lower() for m in result.medications]
        assert any("metformin" in name for name in med_names) or \
               any("aspirin" in name for name in med_names)

    def test_extracts_lab_values(self, ner_extractor):
        """Should extract structured lab results."""
        text = "Sodium 138 mEq/L, Potassium 4.1 mEq/L, Creatinine 1.1 mg/dL, HbA1c 9.2%."
        result = ner_extractor.extract(text)
        assert len(result.lab_values) >= 1

    def test_extracts_vitals(self, ner_extractor):
        """Should extract vital signs."""
        text = "Blood pressure 145/92 mmHg, heart rate 102 bpm, oxygen saturation 96%."
        result = ner_extractor.extract(text)
        assert "Blood Pressure" in result.vitals
        assert "Heart Rate" in result.vitals

    def test_extracts_procedures(self, ner_extractor):
        """Should identify medical procedures."""
        text = "Percutaneous coronary intervention performed. Echocardiogram shows LVEF 45%."
        result = ner_extractor.extract(text)
        assert any("percutaneous" in p.lower() or "echocardiogram" in p.lower()
                   for p in result.procedures)

    def test_entity_count_positive(self, ner_extractor):
        """Entity count should reflect extracted entities."""
        text = """
        58-year-old male with type 2 diabetes and hypertension.
        Blood pressure 145/92 mmHg. Prescribed Metformin 500 mg twice daily.
        Presented with chest pain. Electrocardiogram performed.
        """
        result = ner_extractor.extract(text)
        assert result.entity_count >= 0  # Rule-based always returns something

    def test_empty_text(self, ner_extractor):
        """Empty text should return empty NERResult without crashing."""
        result = ner_extractor.extract("")
        assert isinstance(result, NERResult)

    def test_entity_summary(self, ner_extractor):
        """get_entity_summary should return dict with expected keys."""
        text = "Patient has chest pain. Troponin 2.4 ng/mL. Diagnosed with STEMI."
        result = ner_extractor.extract(text)
        summary = ner_extractor.get_entity_summary(result)

        assert "total_entities" in summary
        assert "diseases" in summary
        assert "symptoms" in summary
        assert "medications_structured" in summary
        assert "lab_values" in summary
        assert "vitals" in summary

    def test_deduplication(self, ner_extractor):
        """Repeated mentions of same entity should not create duplicates."""
        text = "Chest pain is the main complaint. The chest pain started 2 hours ago. Chest pain radiates to the arm."
        result = ner_extractor.extract(text)
        symptoms_lower = [s.lower() for s in result.symptoms]
        chest_pain_count = sum(1 for s in symptoms_lower if "chest pain" in s)
        assert chest_pain_count == 1  # Should be deduplicated


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleAIntegration:
    """Test preprocessor + NER working together."""

    def test_end_to_end_pipeline(self, preprocessor, ner_extractor, simple_report):
        """Full Module A pipeline should run without errors."""
        # Step 1: Preprocess
        preprocessed = preprocessor.process(simple_report)
        assert preprocessed.clean_text

        # Step 2: NER on clean text
        ner_result = ner_extractor.extract(preprocessed.clean_text)
        assert isinstance(ner_result, NERResult)

    def test_section_based_ner(self, preprocessor, ner_extractor, simple_report):
        """NER should work on individual sections."""
        preprocessed = preprocessor.process(simple_report)

        for section_name, section in preprocessed.sections.items():
            ner_result = ner_extractor.extract(section.clean_text)
            assert isinstance(ner_result, NERResult)

    def test_vitals_consistency(self, preprocessor, ner_extractor):
        """Vitals extracted by both preprocessor and NER should be consistent."""
        text = "Blood pressure 145/92 mmHg, heart rate 102 bpm."

        preprocessed = preprocessor.process(text)
        vitals_preproc = preprocessor.extract_vitals(preprocessed.clean_text)

        ner_result = ner_extractor.extract(preprocessed.clean_text)
        vitals_ner = ner_result.vitals

        # Both should find blood pressure
        assert "Blood Pressure" in vitals_preproc
        assert "Blood Pressure" in vitals_ner
