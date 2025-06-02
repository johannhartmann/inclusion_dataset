"""Template detection and anti-template validation."""

import re
from typing import List, Dict, Set, Tuple, Any
from collections import Counter, defaultdict
import difflib
from dataclasses import dataclass

from ..config.constants import METRICS, PATTERNS, VALIDATION
from .exceptions import TemplateValidationError, InsufficientDataError


@dataclass
class TemplatePattern:
    """Represents a detected template pattern."""
    pattern: str
    frequency: int
    examples: List[str]
    confidence: float


class TemplateDetector:
    """Detect and prevent template-based instruction generation."""
    
    def __init__(self, max_template_overlap: float = None):
        """Initialize template detector.
        
        Args:
            max_template_overlap: Maximum allowed template overlap (default from constants)
        """
        self.max_template_overlap = max_template_overlap or METRICS.MAX_TEMPLATE_OVERLAP
        self.detected_patterns = []
    
    def detect_templates(self, instructions: List[str]) -> Dict[str, Any]:
        """Comprehensive template detection.
        
        Args:
            instructions: List of instructions to analyze
            
        Returns:
            Dictionary with template detection results
        """
        if len(instructions) < 2:
            return {
                "templates_detected": False,
                "violation_score": 0.0,
                "patterns": [],
                "recommendations": []
            }
        
        # Multiple detection methods
        exact_duplicates = self._find_exact_duplicates(instructions)
        structural_patterns = self._find_structural_patterns(instructions)
        prefix_suffix_patterns = self._find_prefix_suffix_patterns(instructions)
        placeholder_patterns = self._find_placeholder_patterns(instructions)
        semantic_templates = self._find_semantic_templates(instructions)
        
        # Combine all patterns
        all_patterns = (
            exact_duplicates + structural_patterns + 
            prefix_suffix_patterns + placeholder_patterns + semantic_templates
        )
        
        # Calculate violation score
        violation_score = self._calculate_violation_score(instructions, all_patterns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violation_score, all_patterns)
        
        return {
            "templates_detected": violation_score > self.max_template_overlap,
            "violation_score": violation_score,
            "max_allowed_overlap": self.max_template_overlap,
            "patterns": [self._pattern_to_dict(p) for p in all_patterns],
            "exact_duplicates": len(exact_duplicates),
            "structural_patterns": len(structural_patterns),
            "prefix_suffix_patterns": len(prefix_suffix_patterns),
            "placeholder_patterns": len(placeholder_patterns),
            "semantic_templates": len(semantic_templates),
            "recommendations": recommendations,
            "total_instructions": len(instructions),
            "unique_instructions": len(set(instructions))
        }
    
    def _find_exact_duplicates(self, instructions: List[str]) -> List[TemplatePattern]:
        """Find exact duplicate instructions.
        
        Args:
            instructions: List of instructions
            
        Returns:
            List of duplicate patterns
        """
        patterns = []
        instruction_counts = Counter(instructions)
        
        for instruction, count in instruction_counts.items():
            if count > 1:
                patterns.append(TemplatePattern(
                    pattern=f"EXACT_DUPLICATE: {instruction[:50]}...",
                    frequency=count,
                    examples=[instruction],
                    confidence=1.0
                ))
        
        return patterns
    
    def _find_structural_patterns(self, instructions: List[str]) -> List[TemplatePattern]:
        """Find structural template patterns.
        
        Args:
            instructions: List of instructions
            
        Returns:
            List of structural patterns
        """
        patterns = []
        
        # Convert instructions to structural patterns
        structural_forms = []
        for instruction in instructions:
            structure = self._extract_structure(instruction)
            structural_forms.append((structure, instruction))
        
        # Count structural patterns
        structure_counts = Counter([s[0] for s in structural_forms])
        
        for structure, count in structure_counts.items():
            if count > max(2, len(instructions) * 0.1):  # More than 10% or at least 3
                examples = [inst for struct, inst in structural_forms if struct == structure][:3]
                patterns.append(TemplatePattern(
                    pattern=f"STRUCTURAL: {structure}",
                    frequency=count,
                    examples=examples,
                    confidence=min(1.0, count / len(instructions))
                ))
        
        return patterns
    
    def _find_prefix_suffix_patterns(self, instructions: List[str]) -> List[TemplatePattern]:
        """Find common prefix/suffix patterns.
        
        Args:
            instructions: List of instructions
            
        Returns:
            List of prefix/suffix patterns
        """
        patterns = []
        
        # Find common prefixes
        from .pattern_extractor import PatternExtractor
        
        prefix_groups = PatternExtractor.find_prefix_patterns(instructions)
        for prefix, instances in prefix_groups.items():
            patterns.append(TemplatePattern(
                pattern=f"PREFIX: {prefix}",
                frequency=len(instances),
                examples=instances[:PATTERNS.MAX_EXAMPLES_PER_PATTERN],
                confidence=len(instances) / len(instructions)
            ))
        
        # Find common suffixes
        suffix_groups = PatternExtractor.find_suffix_patterns(instructions)
        for suffix, instances in suffix_groups.items():
            patterns.append(TemplatePattern(
                pattern=f"SUFFIX: {suffix}",
                frequency=len(instances),
                examples=instances[:PATTERNS.MAX_EXAMPLES_PER_PATTERN],
                confidence=len(instances) / len(instructions)
            ))
        
        return patterns
    
    def _find_placeholder_patterns(self, instructions: List[str]) -> List[TemplatePattern]:
        """Find placeholder-based template patterns.
        
        Args:
            instructions: List of instructions
            
        Returns:
            List of placeholder patterns
        """
        patterns = []
        
        # Convert instructions to placeholder patterns
        placeholder_patterns = []
        for instruction in instructions:
            pattern = self._create_placeholder_pattern(instruction)
            placeholder_patterns.append((pattern, instruction))
        
        # Count placeholder patterns
        pattern_counts = Counter([p[0] for p in placeholder_patterns])
        
        threshold = max(VALIDATION.MIN_INSTRUCTIONS_FOR_ANALYSIS, 
                       len(instructions) * METRICS.MIN_PATTERN_OCCURRENCE_RATIO)
        
        for pattern, count in pattern_counts.items():
            if count > threshold and pattern != instruction:
                examples = [inst for pat, inst in placeholder_patterns if pat == pattern][:PATTERNS.MAX_EXAMPLES_PER_PATTERN]
                patterns.append(TemplatePattern(
                    pattern=f"PLACEHOLDER: {pattern}",
                    frequency=count,
                    examples=examples,
                    confidence=count / len(instructions)
                ))
        
        return patterns
    
    def _find_semantic_templates(self, instructions: List[str]) -> List[TemplatePattern]:
        """Find semantic template patterns using similarity.
        
        Args:
            instructions: List of instructions
            
        Returns:
            List of semantic patterns
        """
        patterns = []
        similarity_threshold = 0.8
        
        # Group highly similar instructions
        similar_groups = []
        processed = set()
        
        for i, inst1 in enumerate(instructions):
            if i in processed:
                continue
                
            similar_group = [inst1]
            for j, inst2 in enumerate(instructions[i+1:], i+1):
                if j in processed:
                    continue
                    
                similarity = difflib.SequenceMatcher(None, inst1.lower(), inst2.lower()).ratio()
                if similarity > similarity_threshold:
                    similar_group.append(inst2)
                    processed.add(j)
            
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
                processed.add(i)
        
        # Create patterns for similar groups
        for group in similar_groups:
            if len(group) > max(2, len(instructions) * 0.05):
                # Find common core
                common_core = self._find_common_subsequence(group)
                patterns.append(TemplatePattern(
                    pattern=f"SEMANTIC: {common_core[:50]}...",
                    frequency=len(group),
                    examples=group[:3],
                    confidence=len(group) / len(instructions)
                ))
        
        return patterns
    
    def _extract_structure(self, instruction: str) -> str:
        """Extract structural pattern from instruction.
        
        Args:
            instruction: Instruction text
            
        Returns:
            Structural pattern string
        """
        # Replace content words with placeholders
        structure = instruction.lower()
        
        # Replace specific content with placeholders
        structure = re.sub(r'\b\d+\b', '[NUM]', structure)
        structure = re.sub(r'\b[a-z]+ung\b', '[NOUN_UNG]', structure)  # German -ung nouns
        structure = re.sub(r'\b[a-z]+heit\b', '[NOUN_HEIT]', structure)  # German -heit nouns
        structure = re.sub(r'\b[a-z]+keit\b', '[NOUN_KEIT]', structure)  # German -keit nouns
        structure = re.sub(r'\b[a-z]+isch\b', '[ADJ_ISCH]', structure)  # German -isch adjectives
        structure = re.sub(r'\b[a-z]+lich\b', '[ADJ_LICH]', structure)  # German -lich adjectives
        
        # Replace quoted content
        structure = re.sub(r'"[^"]*"', '[QUOTE]', structure)
        structure = re.sub(r"'[^']*'", '[QUOTE]', structure)
        
        # Replace technical terms (assuming they're longer words)
        structure = re.sub(r'\b[a-z]{8,}\b', '[TECH_TERM]', structure)
        
        return structure
    
    def _create_placeholder_pattern(self, instruction: str) -> str:
        """Create placeholder pattern from instruction.
        
        Args:
            instruction: Instruction text
            
        Returns:
            Pattern with placeholders
        """
        pattern = instruction
        
        # Replace numbers
        pattern = re.sub(r'\b\d+\b', '[NUMBER]', pattern)
        
        # Replace quoted strings
        pattern = re.sub(r'"[^"]*"', '[QUOTED_TEXT]', pattern)
        pattern = re.sub(r"'[^']*'", '[QUOTED_TEXT]', pattern)
        
        # Replace capitalized words (likely proper nouns)
        pattern = re.sub(r'\b[A-ZÄÖÜ][a-zäöüß]+\b', '[PROPER_NOUN]', pattern)
        
        # Replace domain-specific terms
        pattern = re.sub(r'\b(?:Gender|Diversity|Inklusion|Behinderung|Migration)\b', '[DOMAIN_TERM]', pattern)
        
        # Replace long technical words
        pattern = re.sub(r'\b[a-zäöüß]{10,}\b', '[TECHNICAL_TERM]', pattern)
        
        return pattern
    
    def _find_common_subsequence(self, strings: List[str]) -> str:
        """Find longest common subsequence among strings.
        
        Args:
            strings: List of strings
            
        Returns:
            Common subsequence
        """
        if not strings:
            return ""
        
        if len(strings) == 1:
            return strings[0]
        
        # Start with first string
        common = strings[0]
        
        for string in strings[1:]:
            # Find common subsequence
            matcher = difflib.SequenceMatcher(None, common, string)
            matches = matcher.get_matching_blocks()
            
            # Reconstruct common parts
            new_common = ""
            for match in matches:
                if match.size > 3:  # Only consider substantial matches
                    new_common += common[match.a:match.a + match.size]
            
            common = new_common
            
            if len(common) < 5:  # Stop if too little commonality
                break
        
        return common.strip()
    
    def _calculate_violation_score(self, instructions: List[str], patterns: List[TemplatePattern]) -> float:
        """Calculate overall template violation score.
        
        Args:
            instructions: List of instructions
            patterns: List of detected patterns
            
        Returns:
            Violation score (0.0 to 1.0)
        """
        if not instructions or not patterns:
            return 0.0
        
        # Weight patterns by confidence and frequency
        total_violations = 0.0
        
        for pattern in patterns:
            # Weight by frequency and confidence
            violation_weight = pattern.frequency * pattern.confidence
            total_violations += violation_weight
        
        # Normalize by total number of instructions
        violation_score = total_violations / len(instructions)
        
        return min(1.0, violation_score)
    
    def _generate_recommendations(self, violation_score: float, patterns: List[TemplatePattern]) -> List[str]:
        """Generate recommendations for reducing template usage.
        
        Args:
            violation_score: Current violation score
            patterns: Detected patterns
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if violation_score <= self.max_template_overlap:
            recommendations.append("✓ Template diversity is within acceptable limits")
            return recommendations
        
        recommendations.append(f"⚠ Template violation detected: {violation_score:.1%} > {self.max_template_overlap:.1%}")
        
        # Specific recommendations based on pattern types
        pattern_types = Counter([p.pattern.split(':')[0] for p in patterns])
        
        if pattern_types.get('EXACT_DUPLICATE', 0) > 0:
            recommendations.append("• Remove exact duplicate instructions")
        
        if pattern_types.get('STRUCTURAL', 0) > 0:
            recommendations.append("• Vary sentence structures (questions, statements, imperatives)")
        
        if pattern_types.get('PREFIX', 0) > 0:
            recommendations.append("• Use diverse instruction openings instead of repeated prefixes")
        
        if pattern_types.get('SUFFIX', 0) > 0:
            recommendations.append("• Vary instruction endings and closing phrases")
        
        if pattern_types.get('PLACEHOLDER', 0) > 0:
            recommendations.append("• Implement truly dynamic content generation")
        
        if pattern_types.get('SEMANTIC', 0) > 0:
            recommendations.append("• Increase semantic diversity in instruction meanings")
        
        # General recommendations
        recommendations.extend([
            "• Implement anti-template constraints in generation",
            "• Use contextual instruction derivation instead of templates",
            "• Increase randomization in instruction formulation",
            "• Monitor n-gram diversity during generation"
        ])
        
        return recommendations
    
    def _pattern_to_dict(self, pattern: TemplatePattern) -> Dict[str, Any]:
        """Convert TemplatePattern to dictionary.
        
        Args:
            pattern: TemplatePattern instance
            
        Returns:
            Dictionary representation
        """
        return {
            "pattern": pattern.pattern,
            "frequency": pattern.frequency,
            "examples": pattern.examples,
            "confidence": pattern.confidence
        }
    
    def should_reject_batch(self, instructions: List[str]) -> Tuple[bool, str]:
        """Determine if a batch should be rejected due to template violations.
        
        Args:
            instructions: List of instructions to check
            
        Returns:
            Tuple of (should_reject, reason)
        """
        results = self.detect_templates(instructions)
        
        if results["templates_detected"]:
            violation_score = results["violation_score"]
            return True, f"Template violation: {violation_score:.1%} exceeds limit of {self.max_template_overlap:.1%}"
        
        return False, "No template violations detected"