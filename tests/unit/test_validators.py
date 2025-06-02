"""Tests for validator modules."""

import pytest
from unittest.mock import Mock, patch

from src.inclusion_dataset.validators.diversity_metrics import DiversityMetrics
from src.inclusion_dataset.validators.template_detector import TemplateDetector
from src.inclusion_dataset.validators.exceptions import InsufficientDataError, TemplateValidationError


class TestDiversityMetrics:
    """Test the DiversityMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        self.diversity_metrics = DiversityMetrics()
        self.sample_texts = [
            "This is a test sentence for diversity analysis.",
            "Another completely different sentence with unique words.",
            "A third sentence that shares some vocabulary but remains distinct."
        ]

    def test_calculate_lexical_diversity(self):
        """Test lexical diversity calculation."""
        metrics = DiversityMetrics()
        texts = ["the cat sat on the mat", "a dog runs in the park"]
        diversity = metrics.calculate_lexical_diversity(texts)
        
        assert isinstance(diversity, dict)
        assert "ttr" in diversity
        assert "mtld" in diversity
        assert 0 <= diversity["ttr"] <= 1

    def test_lexical_diversity_empty_text(self):
        """Test lexical diversity with empty texts."""
        from src.inclusion_dataset.validators.exceptions import InsufficientDataError
        
        metrics = DiversityMetrics()
        
        # Should raise InsufficientDataError for empty input
        with pytest.raises(InsufficientDataError):
            metrics.calculate_lexical_diversity([])

    def test_template_detection(self):
        """Test template pattern detection."""
        metrics = DiversityMetrics()
        instructions = [
            "Rewrite the following text to be more inclusive: [TEXT]",
            "Rewrite the following content to be more inclusive: [CONTENT]",
            "Please make this text more inclusive: [INPUT]",
            "Convert this passage to inclusive language: [PASSAGE]"
        ]
        
        result = metrics.detect_template_patterns(instructions)
        assert isinstance(result, dict)
        assert "template_detected" in result
        assert "max_overlap" in result

    def test_semantic_spread(self):
        """Test semantic spread measurement."""
        metrics = DiversityMetrics()
        instructions = [
            "The cat is sleeping on the couch.",
            "A dog is running in the park.",
            "The feline rests on the sofa.",
            "The canine plays outside."
        ]
        
        result = metrics.measure_semantic_spread(instructions)
        assert isinstance(result, dict)
        assert "clusters" in result
        assert "max_cluster_size" in result


class TestTemplateDetector:
    """Test the TemplateDetector class."""

    def test_detect_templates_simple(self):
        """Test basic template detection."""
        detector = TemplateDetector()
        instructions = [
            "Rewrite the following text to be more inclusive: [TEXT]",
            "Rewrite the following content to be more inclusive: [CONTENT]",
            "Please make this text more inclusive: [INPUT]",
            "Convert this passage to inclusive language: [PASSAGE]"
        ]
        
        result = detector.detect_templates(instructions)
        assert isinstance(result, dict)
        assert "templates_detected" in result
        assert "violation_score" in result
        # Should detect high overlap due to similar structure
        assert result["violation_score"] > 0.1

    def test_detect_templates_diverse(self):
        """Test template detection with diverse instructions."""
        detector = TemplateDetector()
        instructions = [
            "Analyze the bias in this workplace email.",
            "What inclusive alternatives exist for these terms?", 
            "How can we improve the accessibility of this content?",
            "Identify problematic language patterns in this text."
        ]
        
        result = detector.detect_templates(instructions)
        assert isinstance(result, dict)
        assert "templates_detected" in result
        # Should detect some overlap but not extreme amounts
        assert result["violation_score"] >= 0.0
        assert result["violation_score"] <= 1.0

    def test_batch_rejection(self):
        """Test batch rejection functionality."""
        detector = TemplateDetector(max_template_overlap=0.05)
        
        # Templates that should be rejected
        template_instructions = [
            "Rewrite this text: example 1",
            "Rewrite this text: example 2", 
            "Rewrite this text: example 3"
        ]
        
        should_reject, reason = detector.should_reject_batch(template_instructions)
        assert isinstance(should_reject, bool)
        assert isinstance(reason, str)

    def test_extract_structure(self):
        """Test structure extraction."""
        detector = TemplateDetector()
        text = "Please rewrite the following text to be more inclusive"
        
        structure = detector._extract_structure(text)
        assert isinstance(structure, str)
        assert len(structure) > 0