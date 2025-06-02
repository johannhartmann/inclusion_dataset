"""Common pattern extraction logic."""

from collections import defaultdict
from typing import Dict, List, Optional

from ..config.constants import METRICS, PATTERNS


class PatternExtractor:
    """Utility class for extracting common patterns from text collections."""

    @staticmethod
    def find_prefix_patterns(
        instructions: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_frequency_ratio: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Find common prefix patterns in instructions.

        Args:
            instructions: List of instructions to analyze
            min_length: Minimum prefix length (default from constants)
            max_length: Maximum prefix length (default from constants)
            min_frequency_ratio: Minimum frequency ratio (default from constants)

        Returns:
            Dictionary mapping prefixes to lists of matching instructions
        """
        min_length = min_length or PATTERNS.MIN_PREFIX_LENGTH
        max_length = max_length or PATTERNS.MAX_PREFIX_LENGTH
        min_frequency_ratio = min_frequency_ratio or METRICS.MIN_PATTERN_FREQUENCY_RATIO

        prefix_groups = defaultdict(list)

        for instruction in instructions:
            for length in range(min_length, min(max_length, len(instruction))):
                prefix = instruction[:length].strip()
                if len(prefix) >= min_length:
                    prefix_groups[prefix].append(instruction)

        # Filter by frequency
        min_occurrences = max(2, len(instructions) * min_frequency_ratio)
        return {
            prefix: instances
            for prefix, instances in prefix_groups.items()
            if len(instances) >= min_occurrences
        }

    @staticmethod
    def find_suffix_patterns(
        instructions: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_frequency_ratio: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Find common suffix patterns in instructions.

        Args:
            instructions: List of instructions to analyze
            min_length: Minimum suffix length (default from constants)
            max_length: Maximum suffix length (default from constants)
            min_frequency_ratio: Minimum frequency ratio (default from constants)

        Returns:
            Dictionary mapping suffixes to lists of matching instructions
        """
        min_length = min_length or PATTERNS.MIN_SUFFIX_LENGTH
        max_length = max_length or PATTERNS.MAX_SUFFIX_LENGTH
        min_frequency_ratio = min_frequency_ratio or METRICS.MIN_PATTERN_FREQUENCY_RATIO

        suffix_groups = defaultdict(list)

        for instruction in instructions:
            for length in range(min_length, min(max_length, len(instruction))):
                suffix = instruction[-length:].strip()
                if len(suffix) >= min_length:
                    suffix_groups[suffix].append(instruction)

        # Filter by frequency
        min_occurrences = max(2, len(instructions) * min_frequency_ratio)
        return {
            suffix: instances
            for suffix, instances in suffix_groups.items()
            if len(instances) >= min_occurrences
        }

    @staticmethod
    def find_common_substrings(
        text1: str, text2: str, min_length: Optional[int] = None
    ) -> List[str]:
        """Find common substrings between two texts.

        Args:
            text1: First text
            text2: Second text
            min_length: Minimum substring length (default from constants)

        Returns:
            List of common substrings
        """
        from ..config.constants import LIMITS
        min_length = min_length or LIMITS.MIN_PATTERN_LENGTH
        common_parts = []

        # Find common prefix
        prefix = ""
        for i in range(min(len(text1), len(text2))):
            if text1[i] == text2[i]:
                prefix += text1[i]
            else:
                break

        if len(prefix) >= min_length:
            common_parts.append(prefix.strip())

        # Find common suffix
        suffix = ""
        for i in range(1, min(len(text1), len(text2)) + 1):
            if text1[-i] == text2[-i]:
                suffix = text1[-i] + suffix
            else:
                break

        if len(suffix) >= min_length and suffix not in common_parts:
            common_parts.append(suffix.strip())

        return common_parts

    @staticmethod
    def extract_structural_pattern(text: str) -> str:
        """Extract structural pattern from text by replacing content with placeholders.

        Args:
            text: Text to analyze

        Returns:
            Structural pattern string
        """
        import re

        # Convert to lowercase for pattern matching
        pattern = text.lower()

        # Replace specific content with placeholders
        replacements = [
            (r"\b\d+\b", "[NUM]"),
            (r"\b[a-z]+ung\b", "[NOUN_UNG]"),  # German -ung nouns
            (r"\b[a-z]+heit\b", "[NOUN_HEIT]"),  # German -heit nouns
            (r"\b[a-z]+keit\b", "[NOUN_KEIT]"),  # German -keit nouns
            (r"\b[a-z]+isch\b", "[ADJ_ISCH]"),  # German -isch adjectives
            (r"\b[a-z]+lich\b", "[ADJ_LICH]"),  # German -lich adjectives
            (r'"[^"]*"', "[QUOTE]"),  # Quoted content
            (r"'[^']*'", "[QUOTE]"),  # Single quoted content
            (r"\b[a-z]{8,}\b", "[TECH_TERM]"),  # Long technical terms
        ]

        for regex, replacement in replacements:
            pattern = re.sub(regex, replacement, pattern)

        return pattern
