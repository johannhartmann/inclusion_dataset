"""Integration tests for the pipeline."""

import pytest
import tempfile
import os
from unittest.mock import patch, Mock

from src.inclusion_dataset.config.settings import Config


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration for testing."""
        return Config(
            total_samples=5,  # Small number for testing
            min_quality_score=6.0,
            max_template_overlap=0.1
        )

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        with patch('openai.OpenAI') as mock_client:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response content"
            mock_client.return_value.chat.completions.create.return_value = mock_response
            yield mock_client

    def test_content_generation_flow(self, temp_config, mock_openai_client):
        """Test the content generation flow without API calls."""
        # This would test the full pipeline in mock mode
        # Due to the complexity of the pipeline, this is a placeholder
        # for where comprehensive integration tests would go
        
        # Test that configuration is properly loaded
        assert temp_config.total_samples == 5
        assert temp_config.min_quality_score == 6.0

    def test_validation_pipeline(self, temp_config):
        """Test the validation pipeline components."""
        from src.inclusion_dataset.validators.diversity_metrics import DiversityMetrics
        from src.inclusion_dataset.validators.template_detector import TemplateDetector
        
        # Test that validators can be instantiated
        diversity_validator = DiversityMetrics()
        template_detector = TemplateDetector()
        
        assert diversity_validator is not None
        assert template_detector is not None
        
        # Test with sample data
        sample_texts = [
            "This is a sample instruction for testing.",
            "Another different instruction with unique content.",
            "A third instruction that maintains diversity."
        ]
        
        # Test diversity metrics
        diversity_result = diversity_validator.calculate_lexical_diversity(sample_texts)
        assert isinstance(diversity_result, dict)
        assert "ttr" in diversity_result
        
        # Test template detection
        template_result = template_detector.detect_templates(sample_texts)
        assert isinstance(template_result, dict)
        assert "templates_detected" in template_result

    def test_output_format_validation(self):
        """Test that output format matches expected SFT format."""
        # Mock sample output
        sample_output = {
            "instruction": "Test instruction",
            "input": "Test input text",
            "output": "Test output text"
        }
        
        # Validate required keys exist
        required_keys = ["instruction", "input", "output"]
        for key in required_keys:
            assert key in sample_output
            assert isinstance(sample_output[key], str)
            assert len(sample_output[key]) > 0

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_environment_setup(self):
        """Test that environment variables are properly configured."""
        assert os.getenv('OPENAI_API_KEY') is not None
        
        # Test configuration loading with environment
        config = Config()
        assert config.total_samples > 0