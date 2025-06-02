"""Diversity metrics calculation and validation."""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
import numpy as np
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from ..config.constants import LIMITS, METRICS, PATTERNS, VALIDATION
from .exceptions import (
    ConfigurationError,
    DiversityValidationError,
    InsufficientDataError,
    LexicalAnalysisError,
)
from .pragmatic_analyzer import PragmaticAnalyzer
from .semantic_analyzer import SemanticAnalyzer


class DiversityMetrics:
    """Calculate and validate diversity metrics for generated instructions and texts."""

    def __init__(self, language: str = "german"):
        """Initialize diversity metrics calculator.

        Args:
            language: Language for text processing (default: german)

        Raises:
            ConfigurationError: If language setup fails
        """
        self.language = language
        self._ensure_nltk_data()

        # Setup stopwords
        try:
            self.stopwords = self._load_stopwords()
        except Exception as e:
            raise ConfigurationError(f"Failed to load stopwords for {language}: {e}")

        # Initialize specialized analyzers
        self.semantic_analyzer = SemanticAnalyzer(self.stopwords)
        self.pragmatic_analyzer = PragmaticAnalyzer()

    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded.

        Raises:
            ConfigurationError: If NLTK data cannot be downloaded
        """
        required_data = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/stopwords", "stopwords"),
        ]

        for data_path, download_name in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                try:
                    nltk.download(download_name)
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to download NLTK data '{download_name}': {e}"
                    )

    def _load_stopwords(self) -> set:
        """Load stopwords for the configured language.

        Returns:
            Set of stopwords

        Raises:
            ConfigurationError: If stopwords cannot be loaded
        """
        try:
            return set(stopwords.words("german"))
        except LookupError:
            # Try to download and load again
            try:
                nltk.download("stopwords")
                return set(stopwords.words("german"))
            except Exception as e:
                raise ConfigurationError(f"Cannot load German stopwords: {e}")

    def calculate_lexical_diversity(self, texts: List[str]) -> Dict[str, float]:
        """Calculate lexical diversity metrics.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with diversity metrics

        Raises:
            InsufficientDataError: If no texts provided
            LexicalAnalysisError: If analysis fails
        """
        if not texts:
            raise InsufficientDataError(
                "No texts provided for lexical diversity analysis"
            )

        try:
            # Combine and tokenize texts
            combined_text = " ".join(texts)
            tokens = self._tokenize_text(combined_text)

            if not tokens:
                return self._empty_diversity_result()

            # Calculate core metrics
            types = set(tokens)
            ttr = self._calculate_ttr(tokens, types)
            mtld = self._calculate_mtld(tokens)
            maas_index = self._calculate_maas_index(tokens, types)

            return {
                "ttr": ttr,
                "mtld": mtld,
                "maas_index": maas_index,
                "total_tokens": len(tokens),
                "unique_tokens": len(types),
                "meets_ttr_threshold": ttr >= METRICS.TTR_THRESHOLD,
            }

        except Exception as e:
            raise LexicalAnalysisError(f"Lexical diversity calculation failed: {e}")

    def _calculate_ttr(self, tokens: List[str], types: Set[str]) -> float:
        """Calculate Type-Token Ratio.

        Args:
            tokens: List of tokens
            types: Set of unique tokens

        Returns:
            TTR score
        """
        return len(types) / len(tokens) if tokens else 0.0

    def _empty_diversity_result(self) -> Dict[str, float]:
        """Return empty result for cases with no tokens."""
        return {
            "ttr": 0.0,
            "mtld": 0.0,
            "maas_index": 0.0,
            "total_tokens": 0,
            "unique_tokens": 0,
            "meets_ttr_threshold": False,
        }

    def detect_template_patterns(self, instructions: List[str]) -> Dict[str, Any]:
        """Detect template patterns in instructions.

        Args:
            instructions: List of instructions to analyze

        Returns:
            Dictionary with template detection results

        Raises:
            InsufficientDataError: If no instructions provided
            DiversityValidationError: If template detection fails
        """
        if not instructions:
            raise InsufficientDataError(
                "No instructions provided for template detection"
            )

        try:
            # Calculate n-gram overlaps using constants
            overlaps = self._calculate_ngram_overlaps(instructions)

            # Detect common patterns
            patterns = self._find_common_patterns(instructions)

            # Determine if templates are detected
            max_overlap = max(overlaps.values())
            template_detected = max_overlap > METRICS.MAX_TEMPLATE_OVERLAP

            return {
                "template_detected": template_detected,
                "max_overlap": max_overlap,
                "threshold": METRICS.MAX_TEMPLATE_OVERLAP,
                **overlaps,
                "patterns": patterns,
                "unique_instructions": len(set(instructions)),
                "total_instructions": len(instructions),
            }

        except Exception as e:
            raise DiversityValidationError(f"Template pattern detection failed: {e}")

    def _calculate_ngram_overlaps(self, instructions: List[str]) -> Dict[str, float]:
        """Calculate overlaps for different n-gram sizes.

        Args:
            instructions: List of instructions

        Returns:
            Dictionary with overlap scores for each n-gram size
        """
        return {
            "unigram_overlap": self._calculate_ngram_overlap(
                instructions, PATTERNS.UNIGRAM_SIZE
            ),
            "bigram_overlap": self._calculate_ngram_overlap(
                instructions, PATTERNS.BIGRAM_SIZE
            ),
            "trigram_overlap": self._calculate_ngram_overlap(
                instructions, PATTERNS.TRIGRAM_SIZE
            ),
        }

    def measure_semantic_spread(
        self, instructions: List[str], max_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Measure semantic spread using clustering.

        Delegates to SemanticAnalyzer for implementation.

        Args:
            instructions: List of instructions to analyze
            max_clusters: Maximum number of clusters to use

        Returns:
            Dictionary with semantic spread metrics
        """
        return self.semantic_analyzer.measure_semantic_spread(
            instructions, max_clusters
        )

    def validate_pragmatic_functions(self, instructions: List[str]) -> Dict[str, Any]:
        """Validate pragmatic function diversity in instructions.

        Delegates to PragmaticAnalyzer for implementation.

        Args:
            instructions: List of instructions to analyze

        Returns:
            Dictionary with pragmatic function analysis
        """
        return self.pragmatic_analyzer.validate_pragmatic_functions(instructions)

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text and filter stopwords.

        Args:
            text: Text to tokenize

        Returns:
            List of filtered tokens
        """
        tokens = word_tokenize(text.lower(), language=self.language)
        # Filter out punctuation and stopwords
        filtered_tokens = [
            token for token in tokens if token.isalnum() and token not in self.stopwords
        ]
        return filtered_tokens

    def _calculate_mtld(self, tokens: List[str], ttr_threshold: Optional[float] = None) -> float:
        """Calculate Measure of Textual Lexical Diversity (MTLD).

        Args:
            tokens: List of tokens
            ttr_threshold: TTR threshold for segmentation (default from constants)

        Returns:
            MTLD score

        Raises:
            LexicalAnalysisError: If MTLD calculation fails
        """
        if not tokens:
            return 0.0

        ttr_threshold = ttr_threshold or METRICS.TTR_THRESHOLD

        try:
            # Calculate MTLD in both directions and average
            forward_mtld = self._calculate_directional_mtld(tokens, ttr_threshold)
            backward_mtld = self._calculate_directional_mtld(
                tokens[::-1], ttr_threshold
            )

            return (forward_mtld + backward_mtld) / 2

        except Exception as e:
            raise LexicalAnalysisError(f"MTLD calculation failed: {e}")

    def _calculate_directional_mtld(self, tokens: List[str], threshold: float) -> float:
        """Calculate MTLD in one direction.

        Args:
            tokens: List of tokens
            threshold: TTR threshold for segmentation

        Returns:
            Directional MTLD score
        """
        if len(tokens) < LIMITS.MIN_TOKEN_COUNT_FOR_MTLD:
            return len(tokens)

        segments = 0
        current_segment = []

        for token in tokens:
            current_segment.append(token)

            if len(current_segment) >= LIMITS.MIN_TOKEN_COUNT_FOR_MTLD:
                segment_ttr = self._calculate_segment_ttr(current_segment)

                if segment_ttr <= threshold:
                    segments += 1
                    current_segment = []

        # Handle remaining tokens
        if current_segment:
            remaining_ttr = self._calculate_segment_ttr(current_segment)
            segments += remaining_ttr / threshold if threshold > 0 else 0.0

        return len(tokens) / segments if segments > 0 else len(tokens)

    def _calculate_segment_ttr(self, segment: List[str]) -> float:
        """Calculate TTR for a token segment.

        Args:
            segment: List of tokens in segment

        Returns:
            TTR score for segment
        """
        if not segment:
            return 0.0

        types = set(segment)
        return len(types) / len(segment)

    def _calculate_maas_index(self, tokens: List[str], types: Set[str]) -> float:
        """Calculate Maas Index.

        Args:
            tokens: List of tokens
            types: Set of unique tokens

        Returns:
            Maas Index score
        """
        if not tokens or not types:
            return 0.0

        try:
            maas = (math.log(len(tokens)) - math.log(len(types))) / (
                math.log(len(tokens)) ** 2
            )
            return maas
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _calculate_ngram_overlap(self, texts: List[str], n: int) -> float:
        """Calculate n-gram overlap percentage.

        Args:
            texts: List of texts
            n: N-gram size

        Returns:
            Overlap percentage (0.0 to 1.0)

        Raises:
            LexicalAnalysisError: If n-gram calculation fails
        """
        if not texts:
            return 0.0

        try:
            all_ngrams = self._extract_all_ngrams(texts, n)

            if not all_ngrams:
                return 0.0

            return self._calculate_overlap_ratio(all_ngrams)

        except Exception as e:
            raise LexicalAnalysisError(
                f"N-gram overlap calculation failed for n={n}: {e}"
            )

    def _extract_all_ngrams(self, texts: List[str], n: int) -> List[tuple]:
        """Extract all n-grams from a list of texts.

        Args:
            texts: List of texts
            n: N-gram size

        Returns:
            List of n-gram tuples
        """
        all_ngrams = []

        for text in texts:
            tokens = self._tokenize_text(text)
            if len(tokens) >= n:
                ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
                all_ngrams.extend(ngrams)

        return all_ngrams

    def _calculate_overlap_ratio(self, ngrams: List[tuple]) -> float:
        """Calculate overlap ratio from n-gram list.

        Args:
            ngrams: List of n-gram tuples

        Returns:
            Overlap ratio (1 - unique/total)
        """
        if not ngrams:
            return 0.0

        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counts)

        # Calculate overlap as 1 - (unique / total)
        return 1.0 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0

    def _find_common_patterns(
        self, instructions: List[str], min_length: int = 3
    ) -> List[str]:
        """Find common patterns in instructions.

        Args:
            instructions: List of instructions
            min_length: Minimum pattern length in characters

        Returns:
            List of common patterns
        """
        patterns = []

        if len(instructions) < 2:
            return patterns

        # Simple pattern detection: find common prefixes/suffixes
        for i in range(len(instructions)):
            for j in range(i + 1, len(instructions)):
                inst1, inst2 = instructions[i].lower(), instructions[j].lower()

                # Find common prefix
                prefix = ""
                for k in range(min(len(inst1), len(inst2))):
                    if inst1[k] == inst2[k]:
                        prefix += inst1[k]
                    else:
                        break

                if len(prefix) >= min_length and prefix not in patterns:
                    patterns.append(prefix.strip())

                # Find common suffix
                suffix = ""
                for k in range(1, min(len(inst1), len(inst2)) + 1):
                    if inst1[-k] == inst2[-k]:
                        suffix = inst1[-k] + suffix
                    else:
                        break

                if len(suffix) >= min_length and suffix not in patterns:
                    patterns.append(suffix.strip())

        # Filter out very common patterns (likely stopwords/articles)
        filtered_patterns = [
            p
            for p in patterns
            if len(p.split()) > 1
            and not all(word in self.stopwords for word in p.split())
        ]

        return filtered_patterns[:10]  # Return top 10 patterns

    def calculate_syntactic_diversity(self, instructions: List[str]) -> Dict[str, Any]:
        """Calculate syntactic diversity metrics.

        Args:
            instructions: List of instructions to analyze

        Returns:
            Dictionary with syntactic diversity metrics
        """
        if not instructions:
            return {
                "sentence_patterns": 0,
                "avg_sentence_length": 0.0,
                "complexity_distribution": {},
            }

        sentence_patterns = set()
        sentence_lengths = []
        complexity_counts = defaultdict(int)

        for instruction in instructions:
            sentences = sent_tokenize(instruction, language=self.language)

            for sentence in sentences:
                tokens = word_tokenize(sentence, language=self.language)
                sentence_lengths.append(len(tokens))

                # Simple pattern detection based on POS structure
                pattern = self._extract_sentence_pattern(sentence)
                sentence_patterns.add(pattern)

                # Classify sentence complexity
                complexity = self._classify_sentence_complexity(sentence)
                complexity_counts[complexity] += 1

        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0

        return {
            "unique_sentence_patterns": len(sentence_patterns),
            "avg_sentence_length": avg_sentence_length,
            "complexity_distribution": dict(complexity_counts),
            "total_sentences": sum(complexity_counts.values()),
        }

    def _extract_sentence_pattern(self, sentence: str) -> str:
        """Extract simplified sentence pattern.

        Args:
            sentence: Sentence to analyze

        Returns:
            Pattern string
        """
        # Simple pattern based on punctuation and sentence structure
        pattern = ""

        if sentence.strip().endswith("?"):
            pattern += "QUESTION"
        elif sentence.strip().endswith("!"):
            pattern += "EXCLAMATION"
        else:
            pattern += "STATEMENT"

        # Add complexity marker
        if "," in sentence:
            pattern += "_COMPLEX"

        # Add modal verbs or imperatives
        lower_sentence = sentence.lower()
        if any(
            word in lower_sentence
            for word in ["können", "sollen", "müssen", "dürfen", "mögen"]
        ):
            pattern += "_MODAL"

        if re.search(r"^[A-ZÄÖÜ].*[a-zäöü].*[!.]$", sentence.strip()):
            pattern += "_IMPERATIVE"

        return pattern

    def _classify_sentence_complexity(self, sentence: str) -> str:
        """Classify sentence complexity.

        Args:
            sentence: Sentence to classify

        Returns:
            Complexity classification
        """
        word_count = len(word_tokenize(sentence, language=self.language))
        comma_count = sentence.count(",")

        if word_count <= 5:
            return "simple"
        elif word_count <= 15 and comma_count <= 1:
            return "medium"
        elif word_count <= 25 and comma_count <= 3:
            return "complex"
        else:
            return "very_complex"
