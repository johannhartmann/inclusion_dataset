"""Semantic analysis functionality extracted from DiversityMetrics."""

from collections import Counter
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.constants import LIMITS, METRICS, VALIDATION
from .exceptions import InsufficientDataError, SemanticAnalysisError


class SemanticAnalyzer:
    """Handles semantic analysis operations for text collections."""

    def __init__(self, stopwords: Optional[Set[Any]] = None):
        """Initialize semantic analyzer.

        Args:
            stopwords: Set of stopwords for text processing
        """
        self.stopwords = stopwords or set()

    def measure_semantic_spread(
        self, instructions: List[str], max_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Measure semantic spread using clustering.

        Args:
            instructions: List of instructions to analyze
            max_clusters: Maximum number of clusters (default from constants)

        Returns:
            Dictionary with semantic spread metrics

        Raises:
            InsufficientDataError: If insufficient data for analysis
            SemanticAnalysisError: If clustering fails
        """
        if not instructions:
            raise InsufficientDataError(
                "No instructions provided for semantic analysis"
            )

        if len(instructions) < VALIDATION.MIN_SAMPLES_FOR_CLUSTERING:
            return self._single_cluster_result()

        max_clusters = max_clusters or LIMITS.MAX_CLUSTERS

        try:
            # Vectorize instructions
            vectorizer = self._create_vectorizer()
            X = vectorizer.fit_transform(instructions)

            # Perform clustering
            cluster_results = self._perform_clustering(X, instructions, max_clusters)

            # Calculate similarity metrics
            similarity_metrics = self._calculate_similarity_metrics(X)

            return {**cluster_results, **similarity_metrics}

        except Exception as e:
            raise SemanticAnalysisError(f"Semantic spread measurement failed: {e}")

    def _create_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with appropriate settings."""
        return TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.stopwords) if self.stopwords else None,
            ngram_range=(1, 2),
        )

    def _perform_clustering(
        self, X: Any, instructions: List[str], max_clusters: int
    ) -> Dict[str, Any]:
        """Perform K-means clustering on vectorized instructions."""
        try:
            # Determine optimal number of clusters
            n_clusters = min(max_clusters, len(instructions) // 2, LIMITS.MAX_CLUSTERS)

            if n_clusters < LIMITS.MIN_CLUSTERS:
                n_clusters = LIMITS.MIN_CLUSTERS

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Calculate cluster metrics
            cluster_counts = Counter(cluster_labels)
            max_cluster_size = max(cluster_counts.values()) / len(instructions)

            return {
                "clusters": n_clusters,
                "max_cluster_size": max_cluster_size,
                "cluster_distribution": dict(cluster_counts),
                "silhouette_compatible": True,
            }

        except ValueError as e:
            # Handle case where all instructions are identical
            if "identical" in str(e).lower():
                return self._single_cluster_result()
            raise SemanticAnalysisError(f"Clustering failed: {e}")

    def _calculate_similarity_metrics(self, X: Any) -> Dict[str, float]:
        """Calculate average pairwise similarity metrics."""
        try:
            similarity_matrix = cosine_similarity(X)

            # Get upper triangle of similarity matrix (excluding diagonal)
            triu_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities = similarity_matrix[triu_indices]

            avg_distance = 1.0 - np.mean(similarities) if len(similarities) > 0 else 0.0

            return {"avg_distance": float(avg_distance)}

        except Exception as e:
            raise SemanticAnalysisError(f"Similarity calculation failed: {e}")

    def _single_cluster_result(self) -> Dict[str, Any]:
        """Return result for single cluster case."""
        return {
            "clusters": 1,
            "max_cluster_size": 1.0,
            "avg_distance": 0.0,
            "cluster_distribution": {0: 1},
            "silhouette_compatible": True,
        }

    def calculate_semantic_distance(self, text1: str, text2: str) -> float:
        """Calculate semantic distance between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Semantic distance (0.0 to 1.0)

        Raises:
            SemanticAnalysisError: If distance calculation fails
        """
        if not text1 or not text2:
            return 1.0  # Maximum distance for empty texts

        try:
            vectorizer = self._create_vectorizer()
            vectors = vectorizer.fit_transform([text1, text2])

            if vectors.shape[0] < 2:
                return 1.0

            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]
            return float(1.0 - similarity)

        except Exception as e:
            raise SemanticAnalysisError(f"Semantic distance calculation failed: {e}")

    def validate_semantic_diversity(
        self, instructions: List[str], min_distance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate that instructions meet semantic diversity requirements.

        Args:
            instructions: List of instructions to validate
            min_distance: Minimum required average distance

        Returns:
            Validation results
        """
        min_distance = min_distance or METRICS.MIN_SEMANTIC_DISTANCE

        if len(instructions) < VALIDATION.MIN_INSTRUCTIONS_FOR_ANALYSIS:
            return {
                "valid": True,
                "avg_distance": 1.0,
                "meets_requirement": True,
                "message": "Insufficient data for semantic diversity validation",
            }

        try:
            spread_results = self.measure_semantic_spread(instructions)
            avg_distance = spread_results["avg_distance"]
            meets_requirement = avg_distance >= min_distance

            return {
                "valid": meets_requirement,
                "avg_distance": avg_distance,
                "required_distance": min_distance,
                "meets_requirement": meets_requirement,
                "cluster_count": spread_results["clusters"],
                "max_cluster_size": spread_results["max_cluster_size"],
                "message": (
                    "Semantic diversity requirements met"
                    if meets_requirement
                    else f"Semantic diversity too low: {avg_distance:.3f} < {min_distance:.3f}"
                ),
            }

        except Exception as e:
            raise SemanticAnalysisError(f"Semantic diversity validation failed: {e}")
