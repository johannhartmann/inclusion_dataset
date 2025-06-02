"""Constants for the inclusion dataset pipeline to eliminate magic numbers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricsThresholds:
    """Thresholds for diversity and quality metrics."""

    # Lexical diversity thresholds
    TTR_THRESHOLD: float = 0.72
    MIN_UNIQUE_UNIGRAMS: float = 0.8
    MIN_UNIQUE_BIGRAMS: float = 0.9
    MIN_UNIQUE_TRIGRAMS: float = 0.95

    # Template detection thresholds
    MAX_TEMPLATE_OVERLAP: float = 0.05
    MIN_PATTERN_FREQUENCY_RATIO: float = 0.05
    MIN_PATTERN_OCCURRENCE_RATIO: float = 0.08
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.8

    # Quality thresholds
    MIN_QUALITY_SCORE: float = 7.0
    MIN_SEMANTIC_DISTANCE: float = 0.3


@dataclass(frozen=True)
class ProcessingLimits:
    """Limits for processing operations."""

    # Pattern detection limits
    MIN_PATTERN_LENGTH: int = 3
    MAX_PATTERNS_RETURNED: int = 10
    MIN_PATTERN_WORD_COUNT: int = 1

    # Text processing limits
    MIN_TEXT_LENGTH: int = 150
    MAX_TEXT_LENGTH: int = 300
    MIN_TOKEN_COUNT_FOR_MTLD: int = 10

    # Clustering limits
    MIN_CLUSTERS: int = 2
    MAX_CLUSTERS: int = 20
    MAX_CLUSTER_SIZE_RATIO: float = 0.08

    # Batch processing
    DEFAULT_BATCH_SIZE: int = 25
    MAX_RETRIES: int = 3


@dataclass(frozen=True)
class PatternConstants:
    """Constants for pattern matching and detection."""

    # Common pattern lengths for analysis
    MIN_PREFIX_LENGTH: int = 10
    MAX_PREFIX_LENGTH: int = 50
    MIN_SUFFIX_LENGTH: int = 10
    MAX_SUFFIX_LENGTH: int = 50

    # N-gram analysis
    UNIGRAM_SIZE: int = 1
    BIGRAM_SIZE: int = 2
    TRIGRAM_SIZE: int = 3

    # Function diversity requirements
    MIN_PRAGMATIC_FUNCTIONS: int = 12
    MAX_EXAMPLES_PER_PATTERN: int = 3


@dataclass(frozen=True)
class ValidationConstants:
    """Constants for validation processes."""

    # Minimum sample sizes
    MIN_INSTRUCTIONS_FOR_ANALYSIS: int = 2
    MIN_TEXTS_FOR_DIVERSITY: int = 1
    MIN_SAMPLES_FOR_CLUSTERING: int = 2

    # Confidence levels
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.6
    LOW_CONFIDENCE_THRESHOLD: float = 0.4

    # Error tolerance
    FLOATING_POINT_TOLERANCE: float = 1e-6


@dataclass(frozen=True)
class ContentMatrixConstants:
    """Constants for content matrix generation."""

    # Matrix dimensions (from actual config)
    EXPECTED_DOMAINS: int = 8
    EXPECTED_BIAS_TYPES: int = 6
    EXPECTED_TIME_EPOCHS: int = 4
    EXPECTED_FORMALITY_LEVELS: int = 5

    # Calculated total combinations
    TOTAL_COMBINATIONS: int = (
        EXPECTED_DOMAINS
        * EXPECTED_BIAS_TYPES
        * EXPECTED_TIME_EPOCHS
        * EXPECTED_FORMALITY_LEVELS
    )

    # Sampling ratios
    MIN_SAMPLES_PER_COMBINATION: int = 1
    MAX_SAMPLES_PER_COMBINATION: int = 3


# Instantiate constants for easy importing
METRICS = MetricsThresholds()
LIMITS = ProcessingLimits()
PATTERNS = PatternConstants()
VALIDATION = ValidationConstants()
CONTENT_MATRIX = ContentMatrixConstants()
