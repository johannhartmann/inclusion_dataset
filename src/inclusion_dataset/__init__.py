"""Inclusion Dataset - Distilabel pipeline for inclusive German language SFT dataset generation."""

__version__ = "0.1.0"
__author__ = "Dataset Generator"
__email__ = "dev@example.com"

from .config.settings import Config
from .pipeline.main import InclusionDatasetPipeline

__all__ = ["InclusionDatasetPipeline", "Config"]
