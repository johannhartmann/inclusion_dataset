"""Pragmatic function analysis extracted from DiversityMetrics."""

import re
from typing import List, Dict, Any
from collections import defaultdict

from ..config.constants import PATTERNS, VALIDATION
from .exceptions import PragmaticAnalysisError, InsufficientDataError


class PragmaticAnalyzer:
    """Handles pragmatic function analysis for instructions."""
    
    def __init__(self):
        """Initialize pragmatic analyzer."""
        self.function_patterns = self._get_pragmatic_patterns()
    
    def validate_pragmatic_functions(self, instructions: List[str]) -> Dict[str, Any]:
        """Validate pragmatic function diversity in instructions.
        
        Args:
            instructions: List of instructions to analyze
            
        Returns:
            Dictionary with pragmatic function analysis
            
        Raises:
            InsufficientDataError: If no instructions provided
            PragmaticAnalysisError: If analysis fails
        """
        if not instructions:
            raise InsufficientDataError("No instructions provided for pragmatic analysis")
        
        try:
            function_counts = self._count_function_occurrences(instructions)
            diversity_metrics = self._calculate_diversity_metrics(function_counts, instructions)
            
            return {
                "total_functions": len(self.function_patterns),
                "used_functions": len(function_counts),
                "function_diversity": diversity_metrics["diversity"],
                "balance_ratio": diversity_metrics["balance"],
                "function_counts": dict(function_counts),
                "unidentified_instructions": diversity_metrics["unidentified"],
                "meets_minimum_requirement": len(function_counts) >= PATTERNS.MIN_PRAGMATIC_FUNCTIONS
            }
            
        except Exception as e:
            raise PragmaticAnalysisError(f"Pragmatic function validation failed: {e}")
    
    def _get_pragmatic_patterns(self) -> Dict[str, List[str]]:
        """Get pragmatic function patterns.
        
        Returns:
            Dictionary mapping function names to regex patterns
        """
        return {
            "directive_command": [
                r"\b(wandle|ändere|überarbeite|korrigiere|verbessere)\b",
                r"\b(mach|erstelle|schreib|formuliere)\b"
            ],
            "polite_request": [
                r"\b(bitte|könnten Sie|würden Sie|möchten Sie)\b",
                r"\b(wäre es möglich|könntest du)\b"
            ],
            "expert_consultation": [
                r"\b(bewerte|analysiere|prüfe|beurteile)\b",
                r"\b(empfehle|schlage vor|rate)\b"
            ],
            "peer_support": [
                r"\b(hilf|unterstütze|begleite)\b",
                r"\b(gemeinsam|zusammen|miteinander)\b"
            ],
            "quality_control": [
                r"\b(kontrolliere|überprüfe|validiere)\b",
                r"\b(standards|qualität|richtlinien)\b"
            ],
            "learning_goal": [
                r"\b(lerne|verstehe|erkenne)\b",
                r"\b(bildung|wissen|kompetenz)\b"
            ],
            "problem_identification": [
                r"\b(identifiziere|erkenne|finde)\b.*\b(problem|fehler|issue)\b",
                r"\b(wo|was|welche).*\b(problematisch|schwierig)\b"
            ],
            "improvement_focus": [
                r"\b(optimiere|verbessere|entwickle weiter)\b",
                r"\b(besser|effektiver|inklusiver)\b"
            ],
            "target_adaptation": [
                r"\b(zielgruppe|publikum|leser)\b",
                r"\b(anpassen|ausrichten|orientieren)\b"
            ],
            "standard_alignment": [
                r"\b(standard|norm|richtlinie|regel)\b",
                r"\b(entsprechen|einhalten|befolgen)\b"
            ],
            "awareness_building": [
                r"\b(bewusstsein|aufmerksamkeit|sensibil)\b",
                r"\b(erkennen|wahrnehmen|verstehen)\b"
            ],
            "correction_task": [
                r"\b(korrigiere|berichtige|stelle richtig)\b",
                r"\b(falsch|incorrect|ungenau)\b"
            ]
        }
    
    def _count_function_occurrences(self, instructions: List[str]) -> Dict[str, int]:
        """Count occurrences of each pragmatic function.
        
        Args:
            instructions: List of instructions to analyze
            
        Returns:
            Dictionary with function counts
        """
        function_counts = defaultdict(int)
        
        for instruction in instructions:
            instruction_lower = instruction.lower()
            for function_name, patterns in self.function_patterns.items():
                if self._instruction_matches_function(instruction_lower, patterns):
                    function_counts[function_name] += 1
                    break  # Count each function only once per instruction
        
        return function_counts
    
    def _instruction_matches_function(self, instruction: str, patterns: List[str]) -> bool:
        """Check if instruction matches any pattern for a function.
        
        Args:
            instruction: Instruction text (lowercase)
            patterns: List of regex patterns to match
            
        Returns:
            True if instruction matches any pattern
        """
        return any(re.search(pattern, instruction) for pattern in patterns)
    
    def _calculate_diversity_metrics(self, function_counts: Dict[str, int], 
                                   instructions: List[str]) -> Dict[str, float]:
        """Calculate diversity and balance metrics.
        
        Args:
            function_counts: Dictionary with function counts
            instructions: Original instructions list
            
        Returns:
            Dictionary with calculated metrics
        """
        total_functions = len(self.function_patterns)
        used_functions = len(function_counts)
        
        # Calculate diversity ratio
        diversity = used_functions / total_functions if total_functions > 0 else 0.0
        
        # Calculate balance ratio
        if function_counts:
            values = list(function_counts.values())
            max_count = max(values)
            min_count = min(values)
            balance = min_count / max_count if max_count > 0 else 0.0
        else:
            balance = 0.0
        
        # Calculate unidentified instructions
        total_identified = sum(function_counts.values())
        unidentified = len(instructions) - total_identified
        
        return {
            "diversity": diversity,
            "balance": balance,
            "unidentified": unidentified
        }
    
    def get_function_coverage_report(self, instructions: List[str]) -> Dict[str, Any]:
        """Generate detailed coverage report for pragmatic functions.
        
        Args:
            instructions: List of instructions to analyze
            
        Returns:
            Detailed coverage report
        """
        if not instructions:
            return {"error": "No instructions provided"}
        
        try:
            analysis = self.validate_pragmatic_functions(instructions)
            
            # Calculate coverage percentages
            total_instructions = len(instructions)
            coverage_by_function = {}
            
            for func_name, count in analysis["function_counts"].items():
                coverage_by_function[func_name] = {
                    "count": count,
                    "percentage": (count / total_instructions) * 100
                }
            
            # Identify missing functions
            used_functions = set(analysis["function_counts"].keys())
            all_functions = set(self.function_patterns.keys())
            missing_functions = all_functions - used_functions
            
            return {
                "summary": {
                    "total_instructions": total_instructions,
                    "functions_used": analysis["used_functions"],
                    "functions_available": analysis["total_functions"],
                    "diversity_score": analysis["function_diversity"],
                    "balance_score": analysis["balance_ratio"]
                },
                "coverage_by_function": coverage_by_function,
                "missing_functions": list(missing_functions),
                "unidentified_instructions": analysis["unidentified_instructions"],
                "meets_requirements": analysis["meets_minimum_requirement"]
            }
            
        except Exception as e:
            raise PragmaticAnalysisError(f"Coverage report generation failed: {e}")