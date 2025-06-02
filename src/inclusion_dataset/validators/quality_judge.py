"""LLM-as-a-Judge quality validation."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from openai import OpenAI
from ..config.settings import Config


@dataclass
class QualityAssessment:
    """Quality assessment result."""
    overall_score: float
    instruction_text_fit: float
    response_quality: float
    consistency: float
    completeness: float
    explanation: str
    feedback: List[str]
    passed: bool


class QualityJudge:
    """LLM-based quality assessment for generated samples."""
    
    def __init__(self, config: Config):
        """Initialize quality judge.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.min_score = config.min_quality_score
        
        # Quality assessment prompt template
        self.assessment_prompt = """Du bist ein Experte für inklusive Sprache und Qualitätsbewertung von SFT-Datasets.

Bewerte das folgende Beispiel nach diesen Kriterien:

1. **Instruction-Text-Passung** (1-10): Passt die Aufgabenstellung zum Input-Text?
2. **Response-Qualität** (1-10): Ist die Antwort korrekt, hilfreich und vollständig?
3. **Konsistenz** (1-10): Stimmen Bewertung und Erklärung überein?
4. **Vollständigkeit** (1-10): Wurden alle relevanten Inklusions-Aspekte berücksichtigt?

**BEISPIEL:**
Instruction: {instruction}
Input: {input_text}
Output: {output}

**BEWERTUNGSFORMAT:**
```json
{{
  "instruction_text_fit": <score 1-10>,
  "response_quality": <score 1-10>, 
  "consistency": <score 1-10>,
  "completeness": <score 1-10>,
  "overall_score": <average of above scores>,
  "explanation": "<kurze Begründung der Bewertung>",
  "feedback": ["<konkreter Verbesserungsvorschlag 1>", "<konkreter Verbesserungsvorschlag 2>"],
  "passed": <true if overall_score >= {min_score} else false>
}}
```

Sei streng aber fair in der Bewertung. Achte besonders auf:
- Realistische und praktikable Verbesserungsvorschläge
- Korrekte Identifikation von Bias-Problemen
- Angemessene Sprache und Ton
- Fachliche Richtigkeit"""
    
    def assess_sample(self, instruction: str, input_text: str, output: str, 
                     metadata: Optional[Dict] = None) -> QualityAssessment:
        """Assess quality of a single sample.
        
        Args:
            instruction: The instruction text
            input_text: The input text
            output: The model's output
            metadata: Optional metadata about the sample
            
        Returns:
            QualityAssessment object
        """
        try:
            prompt = self.assessment_prompt.format(
                instruction=instruction,
                input_text=input_text,
                output=output,
                min_score=self.min_score
            )
            
            response = self.client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für Qualitätsbewertung von inklusiven Sprachdaten."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            assessment_data = self._extract_json_from_response(response_text)
            
            if not assessment_data:
                # Fallback assessment if JSON parsing fails
                return self._create_fallback_assessment(instruction, input_text, output)
            
            return QualityAssessment(
                overall_score=assessment_data.get("overall_score", 0.0),
                instruction_text_fit=assessment_data.get("instruction_text_fit", 0.0),
                response_quality=assessment_data.get("response_quality", 0.0),
                consistency=assessment_data.get("consistency", 0.0),
                completeness=assessment_data.get("completeness", 0.0),
                explanation=assessment_data.get("explanation", ""),
                feedback=assessment_data.get("feedback", []),
                passed=assessment_data.get("passed", False)
            )
            
        except Exception as e:
            print(f"Error in quality assessment: {e}")
            return self._create_fallback_assessment(instruction, input_text, output)
    
    def assess_batch(self, samples: List[Dict[str, str]]) -> List[QualityAssessment]:
        """Assess quality of multiple samples.
        
        Args:
            samples: List of samples with 'instruction', 'input', 'output' keys
            
        Returns:
            List of QualityAssessment objects
        """
        assessments = []
        
        for sample in samples:
            assessment = self.assess_sample(
                instruction=sample.get("instruction", ""),
                input_text=sample.get("input", ""),
                output=sample.get("output", ""),
                metadata=sample.get("meta", {})
            )
            assessments.append(assessment)
        
        return assessments
    
    def filter_by_quality(self, samples: List[Dict], assessments: List[QualityAssessment]) -> Tuple[List[Dict], List[Dict]]:
        """Filter samples by quality assessment.
        
        Args:
            samples: List of original samples
            assessments: List of quality assessments
            
        Returns:
            Tuple of (passed_samples, failed_samples)
        """
        passed_samples = []
        failed_samples = []
        
        for sample, assessment in zip(samples, assessments):
            if assessment.passed and assessment.overall_score >= self.min_score:
                passed_samples.append(sample)
            else:
                # Add failure reason to sample
                sample_with_reason = sample.copy()
                sample_with_reason["rejection_reason"] = assessment.explanation
                sample_with_reason["quality_score"] = assessment.overall_score
                failed_samples.append(sample_with_reason)
        
        return passed_samples, failed_samples
    
    def generate_quality_report(self, assessments: List[QualityAssessment]) -> Dict[str, Any]:
        """Generate comprehensive quality report.
        
        Args:
            assessments: List of quality assessments
            
        Returns:
            Quality report dictionary
        """
        if not assessments:
            return {"error": "No assessments provided"}
        
        # Calculate overall statistics
        overall_scores = [a.overall_score for a in assessments]
        instruction_fit_scores = [a.instruction_text_fit for a in assessments]
        response_quality_scores = [a.response_quality for a in assessments]
        consistency_scores = [a.consistency for a in assessments]
        completeness_scores = [a.completeness for a in assessments]
        
        passed_count = sum(1 for a in assessments if a.passed)
        pass_rate = passed_count / len(assessments)
        
        # Score distributions
        score_ranges = {
            "excellent (9-10)": sum(1 for score in overall_scores if score >= 9),
            "good (7-8.9)": sum(1 for score in overall_scores if 7 <= score < 9),
            "acceptable (5-6.9)": sum(1 for score in overall_scores if 5 <= score < 7),
            "poor (3-4.9)": sum(1 for score in overall_scores if 3 <= score < 5),
            "very_poor (0-2.9)": sum(1 for score in overall_scores if score < 3)
        }
        
        # Common feedback themes
        all_feedback = []
        for assessment in assessments:
            all_feedback.extend(assessment.feedback)
        
        feedback_themes = self._analyze_feedback_themes(all_feedback)
        
        return {
            "total_samples": len(assessments),
            "passed_samples": passed_count,
            "failed_samples": len(assessments) - passed_count,
            "pass_rate": pass_rate,
            "average_scores": {
                "overall": sum(overall_scores) / len(overall_scores),
                "instruction_text_fit": sum(instruction_fit_scores) / len(instruction_fit_scores),
                "response_quality": sum(response_quality_scores) / len(response_quality_scores),
                "consistency": sum(consistency_scores) / len(consistency_scores),
                "completeness": sum(completeness_scores) / len(completeness_scores)
            },
            "score_distribution": score_ranges,
            "common_feedback_themes": feedback_themes,
            "recommendations": self._generate_improvement_recommendations(assessments)
        }
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from LLM response.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed JSON dictionary or None
        """
        try:
            # Look for JSON code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Look for JSON without code blocks
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _create_fallback_assessment(self, instruction: str, input_text: str, output: str) -> QualityAssessment:
        """Create fallback assessment when LLM assessment fails.
        
        Args:
            instruction: Instruction text
            input_text: Input text  
            output: Output text
            
        Returns:
            Fallback QualityAssessment
        """
        # Simple heuristic-based assessment
        score = 5.0  # Neutral score
        
        # Check basic requirements
        if len(instruction.strip()) < 5:
            score -= 2.0
        if len(output.strip()) < 10:
            score -= 2.0
        if not any(word in output.lower() for word in ["inklusiv", "gender", "neutral", "barrierefrei"]):
            score -= 1.0
        
        score = max(1.0, min(10.0, score))
        passed = score >= self.min_score
        
        return QualityAssessment(
            overall_score=score,
            instruction_text_fit=score,
            response_quality=score,
            consistency=score,
            completeness=score,
            explanation="Automatische Bewertung (LLM-Judge nicht verfügbar)",
            feedback=["Manuelle Überprüfung empfohlen"],
            passed=passed
        )
    
    def _analyze_feedback_themes(self, feedback_list: List[str]) -> Dict[str, int]:
        """Analyze common themes in feedback.
        
        Args:
            feedback_list: List of feedback strings
            
        Returns:
            Dictionary of themes and their frequencies
        """
        themes = {
            "gender_language": 0,
            "bias_detection": 0,
            "language_clarity": 0,
            "completeness": 0,
            "practical_suggestions": 0,
            "tone_appropriateness": 0,
            "technical_accuracy": 0
        }
        
        theme_keywords = {
            "gender_language": ["gender", "geschlecht", "maskulin", "feminin", "neutral"],
            "bias_detection": ["bias", "vorurteil", "stereotyp", "diskrimini"],
            "language_clarity": ["klar", "verständlich", "deutlich", "sprache"],
            "completeness": ["vollständig", "komplett", "alle aspekte", "umfassend"],
            "practical_suggestions": ["konkret", "praktisch", "umsetzbar", "beispiel"],
            "tone_appropriateness": ["ton", "stil", "angemessen", "höflich"],
            "technical_accuracy": ["fachlich", "korrekt", "richtig", "genau"]
        }
        
        for feedback in feedback_list:
            feedback_lower = feedback.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in feedback_lower for keyword in keywords):
                    themes[theme] += 1
        
        return themes
    
    def _generate_improvement_recommendations(self, assessments: List[QualityAssessment]) -> List[str]:
        """Generate improvement recommendations based on assessments.
        
        Args:
            assessments: List of quality assessments
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Calculate average scores
        avg_scores = {
            "instruction_fit": sum(a.instruction_text_fit for a in assessments) / len(assessments),
            "response_quality": sum(a.response_quality for a in assessments) / len(assessments),
            "consistency": sum(a.consistency for a in assessments) / len(assessments),
            "completeness": sum(a.completeness for a in assessments) / len(assessments)
        }
        
        # Generate specific recommendations
        if avg_scores["instruction_fit"] < 7.0:
            recommendations.append("Improve instruction-text alignment by better contextual analysis")
        
        if avg_scores["response_quality"] < 7.0:
            recommendations.append("Enhance response quality with more specific and actionable suggestions")
        
        if avg_scores["consistency"] < 7.0:
            recommendations.append("Ensure consistency between problem identification and solutions")
        
        if avg_scores["completeness"] < 7.0:
            recommendations.append("Address all relevant inclusion aspects systematically")
        
        # Pass rate recommendations
        pass_rate = sum(1 for a in assessments if a.passed) / len(assessments)
        if pass_rate < 0.8:
            recommendations.append("Overall pass rate is low - review generation parameters")
        
        if not recommendations:
            recommendations.append("Quality metrics are within acceptable ranges")
        
        return recommendations