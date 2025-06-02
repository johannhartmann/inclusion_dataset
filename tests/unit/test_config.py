"""Tests for configuration modules."""

import pytest
from pydantic import ValidationError

from src.inclusion_dataset.config.settings import Config
from src.inclusion_dataset.config.domains import DomainConfig
from src.inclusion_dataset.config.bias_types import BiasTypeConfig


class TestConfig:
    """Test the Config class."""

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = Config()
        assert config.total_samples == 20000
        assert config.min_quality_score == 7.0
        assert config.max_template_overlap == 0.05

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid transformation ratio
        with pytest.raises(ValidationError):
            Config(transformation_ratio=1.5)

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = Config(
            total_samples=1000,
            min_quality_score=8.0,
            max_template_overlap=0.03
        )
        assert config.total_samples == 1000
        assert config.min_quality_score == 8.0
        assert config.max_template_overlap == 0.03


class TestDomainConfig:
    """Test the domain configuration."""

    def test_domains_structure(self):
        """Test that domains have required structure."""
        domains = DomainConfig.DOMAIN_CONFIGS
        assert isinstance(domains, dict)
        assert len(domains) > 0
        
        for domain, domain_config in domains.items():
            assert domain_config.typical_contexts is not None
            assert domain_config.professional_roles is not None
            assert domain_config.common_scenarios is not None

    def test_domain_content_types(self):
        """Test that domain content has correct types."""
        domains = DomainConfig.DOMAIN_CONFIGS
        for domain, domain_config in domains.items():
            assert isinstance(domain_config.typical_contexts, list)
            assert isinstance(domain_config.professional_roles, list)
            assert isinstance(domain_config.common_scenarios, list)
            
            # Check that lists are not empty
            assert len(domain_config.typical_contexts) > 0
            assert len(domain_config.professional_roles) > 0
            assert len(domain_config.common_scenarios) > 0


class TestBiasTypesConfig:
    """Test the bias types configuration."""

    def test_bias_types_structure(self):
        """Test that bias types have required structure."""
        bias_configs = BiasTypeConfig.BIAS_CONFIGS
        assert isinstance(bias_configs, dict)
        assert len(bias_configs) > 0
        
        for bias_type, bias_config in bias_configs.items():
            assert bias_config.problematic_language is not None
            assert bias_config.transformation_strategies is not None

    def test_bias_patterns_structure(self):
        """Test that bias patterns have correct structure."""
        bias_configs = BiasTypeConfig.BIAS_CONFIGS
        for bias_type, bias_config in bias_configs.items():
            patterns = bias_config.problematic_language
            assert isinstance(patterns, dict)
            
            # Check that time epochs exist
            from src.inclusion_dataset.config.settings import TimeEpoch
            for epoch in TimeEpoch:
                assert epoch in patterns, f"Missing epoch {epoch} in bias type {bias_type}"
                assert isinstance(patterns[epoch], list)
                assert len(patterns[epoch]) > 0