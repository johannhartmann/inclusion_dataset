#!/usr/bin/env python3
"""Test script to verify project structure without API calls."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports() -> bool:
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from inclusion_dataset.config.settings import Config

        print("✅ Config import successful")

        from inclusion_dataset.config.domains import DomainConfig

        print("✅ DomainConfig import successful")

        from inclusion_dataset.config.bias_types import BiasTypeConfig

        print("✅ BiasTypeConfig import successful")

        from inclusion_dataset.validators.diversity_metrics import DiversityMetrics

        print("✅ DiversityMetrics import successful")

        from inclusion_dataset.validators.template_detector import TemplateDetector

        print("✅ TemplateDetector import successful")

        # Test basic functionality
        config = Config()
        print(f"✅ Config created with {config.total_samples} total samples")

        diversity = DiversityMetrics()
        print("✅ DiversityMetrics initialized")

        detector = TemplateDetector()
        print("✅ TemplateDetector initialized")

        # Test domain config
        domains = DomainConfig.get_all_domains()
        print(f"✅ Found {len(domains)} domains: {[d.value for d in domains]}")

        # Test bias config
        bias_types = BiasTypeConfig.get_all_bias_types()
        print(f"✅ Found {len(bias_types)} bias types: {[b.value for b in bias_types]}")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_diversity_metrics() -> bool:
    """Test diversity metrics without API."""
    print("\nTesting diversity metrics...")

    try:
        from inclusion_dataset.validators.diversity_metrics import DiversityMetrics

        diversity = DiversityMetrics()

        # Test sample texts
        sample_texts = [
            "Wandle den folgenden Text in inklusive Sprache um.",
            "Überarbeite diesen Text für mehr Gender-Gerechtigkeit.",
            "Verbessere die Formulierung hinsichtlich Barrierefreiheit.",
            "Analysiere den Text auf diskriminierende Sprache.",
            "Bewerte die Inklusion in diesem Textabschnitt.",
        ]

        # Test lexical diversity
        results = diversity.calculate_lexical_diversity(sample_texts)
        print(f"✅ Lexical diversity TTR: {results['ttr']:.3f}")

        # Test template detection
        template_results = diversity.detect_template_patterns(sample_texts)
        print(f"✅ Template detection: {template_results['template_detected']}")

        # Test semantic spread
        semantic_results = diversity.measure_semantic_spread(sample_texts)
        print(f"✅ Semantic clusters: {semantic_results['clusters']}")

        # Test pragmatic functions
        pragmatic_results = diversity.validate_pragmatic_functions(sample_texts)
        print(
            f"✅ Pragmatic functions: {pragmatic_results['used_functions']}/{pragmatic_results['total_functions']}"
        )

        return True

    except Exception as e:
        print(f"❌ Diversity metrics test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_template_detector() -> bool:
    """Test template detector."""
    print("\nTesting template detector...")

    try:
        from inclusion_dataset.validators.template_detector import TemplateDetector

        detector = TemplateDetector()

        # Test with obvious templates
        template_samples = [
            "Wandle um: Text A",
            "Wandle um: Text B",
            "Wandle um: Text C",
            "Verbessere: Content X",
            "Verbessere: Content Y",
        ]

        results = detector.detect_templates(template_samples)
        print(f"✅ Template detection result: {results['templates_detected']}")
        print(f"✅ Violation score: {results['violation_score']:.3f}")
        print(f"✅ Found {len(results['patterns'])} patterns")

        return True

    except Exception as e:
        print(f"❌ Template detector test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_content_combinations() -> bool:
    """Test content matrix combinations."""
    print("\nTesting content combinations...")

    try:
        # We can't test the full pipeline without OpenAI, but we can test the combination logic
        from inclusion_dataset.config.settings import Config

        config = Config()
        total_combinations = config.content_matrix_size
        print(f"✅ Total content matrix combinations: {total_combinations}")
        print(f"✅ Domains: {len(config.domains)}")
        print(f"✅ Bias types: {len(config.bias_types)}")
        print(f"✅ Time epochs: {len(config.time_epochs)}")
        print(f"✅ Formality levels: {len(config.formality_levels)}")

        return True

    except Exception as e:
        print(f"❌ Content combinations test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("🧪 Testing inclusion dataset project structure\n")

    tests = [
        test_imports,
        test_diversity_metrics,
        test_template_detector,
        test_content_combinations,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Project structure is working correctly.")
        print("\n📝 Next steps:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Install distilabel and other ML dependencies")
        print("3. Run full dry run with: python scripts/dry_run.py --samples 3")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
