"""Main configuration settings for the inclusion dataset pipeline."""

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    TRANSFORMATION = "transformation"
    EVALUATION = "evaluation"


class TimeEpoch(str, Enum):
    ERA_1990 = "1990s"
    ERA_2000 = "2000s"
    ERA_2010 = "2010s"
    ERA_CURRENT = "current"


class Domain(str, Enum):
    WORKPLACE = "workplace"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    MEDIA = "media"
    LEGAL = "legal"
    SOCIAL = "social"
    TECHNOLOGY = "technology"
    FINANCE = "finance"


class BiasType(str, Enum):
    GENDER = "gender"
    DISABILITY = "disability"
    ETHNICITY = "ethnicity"
    AGE = "age"
    SOCIOECONOMIC = "socioeconomic"
    RELIGION = "religion"


class FormalityLevel(str, Enum):
    FORMAL = "formal"
    SEMI_FORMAL = "semi_formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    COLLOQUIAL = "colloquial"


class Config(BaseModel):
    """Main configuration class for the inclusion dataset pipeline."""

    # Dataset parameters
    total_samples: int = Field(
        default=20000, description="Total number of samples to generate"
    )
    transformation_ratio: float = Field(
        default=0.5, description="Ratio of transformation vs evaluation tasks"
    )
    min_text_length: int = Field(
        default=150, description="Minimum text length in words"
    )
    max_text_length: int = Field(
        default=300, description="Maximum text length in words"
    )

    # Model configuration
    teacher_model: str = Field(
        default="gpt-4o", description="Teacher model for generation"
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Quality control
    min_quality_score: float = Field(
        default=7.0, description="Minimum quality score for acceptance"
    )
    batch_size: int = Field(default=25, description="Batch size for API calls")
    max_retries: int = Field(
        default=3, description="Maximum retries for failed API calls"
    )

    # Diversity requirements
    lexical_diversity_threshold: float = Field(
        default=0.4, description="Minimum TTR threshold"
    )
    min_unique_unigrams: float = Field(
        default=0.8, description="Minimum unique unigram ratio"
    )
    min_unique_bigrams: float = Field(
        default=0.9, description="Minimum unique bigram ratio"
    )
    min_unique_trigrams: float = Field(
        default=0.95, description="Minimum unique trigram ratio"
    )
    max_template_overlap: float = Field(
        default=0.05, description="Maximum allowed template overlap"
    )
    min_semantic_distance: float = Field(
        default=0.3, description="Minimum cosine distance between instructions"
    )

    # Content matrix dimensions
    domains: List[Domain] = Field(default_factory=lambda: list(Domain))
    bias_types: List[BiasType] = Field(default_factory=lambda: list(BiasType))
    time_epochs: List[TimeEpoch] = Field(default_factory=lambda: list(TimeEpoch))
    formality_levels: List[FormalityLevel] = Field(
        default_factory=lambda: list(FormalityLevel)
    )

    # File paths
    output_dir: str = Field(
        default="data/output", description="Output directory for generated data"
    )
    log_dir: str = Field(default="logs", description="Directory for log files")
    cache_dir: str = Field(
        default="data/cache", description="Directory for caching intermediate results"
    )

    # Export settings
    train_split: float = Field(default=0.8, description="Training set ratio")
    val_split: float = Field(default=0.1, description="Validation set ratio")
    test_split: float = Field(default=0.1, description="Test set ratio")

    @field_validator("transformation_ratio")
    @classmethod
    def validate_transformation_ratio(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("transformation_ratio must be between 0.0 and 1.0")
        return v

    @field_validator("train_split", "val_split", "test_split")
    @classmethod
    def validate_splits(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Split ratios must be between 0.0 and 1.0")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def set_openai_api_key(cls, v: Optional[str]) -> Optional[str]:
        return v or os.getenv("OPENAI_API_KEY")

    def model_post_init(self, __context: Any) -> None:
        """Validate that splits sum to 1.0"""
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")

    @property
    def transformation_samples(self) -> int:
        """Calculate number of transformation samples."""
        return int(self.total_samples * self.transformation_ratio)

    @property
    def evaluation_samples(self) -> int:
        """Calculate number of evaluation samples."""
        return self.total_samples - self.transformation_samples

    @property
    def content_matrix_size(self) -> int:
        """Calculate total combinations in content matrix."""
        return (
            len(self.domains)
            * len(self.bias_types)
            * len(self.time_epochs)
            * len(self.formality_levels)
        )

    def get_train_samples(self) -> int:
        """Get number of training samples."""
        return int(self.total_samples * self.train_split)

    def get_val_samples(self) -> int:
        """Get number of validation samples."""
        return int(self.total_samples * self.val_split)

    def get_test_samples(self) -> int:
        """Get number of test samples."""
        return int(self.total_samples * self.test_split)
