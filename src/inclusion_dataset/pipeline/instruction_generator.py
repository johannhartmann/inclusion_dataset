"""Contextual instruction generation without templates."""

import random
import re
from typing import Dict, List, Any, Optional
from openai import OpenAI

from ..config.settings import Config, TaskType
from ..config.bias_types import BiasTypeConfig


class InstructionGenerator:
    """Generate contextual instructions derived from text analysis."""
    
    def __init__(self, config: Config):
        """Initialize instruction generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        
        # Instruction analysis and generation prompt
        self.analysis_prompt = """Du bist ein Experte für inklusive Sprache und kontextuelle Textanalyse.

Analysiere den folgenden Text und erstelle eine passende Aufgabenstellung:

**TEXT ZU ANALYSIEREN:**
"{text}"

**KONTEXT:**
- Bereich: {domain}
- Bias-Fokus: {bias_type}
- Aufgabenart: {task_type}

**AUFGABE:**
1. Analysiere den Text auf sprachliche Eigenschaften:
   - Anredeformen und Register
   - Fachvokabular und Sprachstil
   - Identifiziere konkrete problematische Aspekte
   - Bestimme Zielgruppe und Kommunikationskontext

2. Leite daraus eine kontextuelle Aufgabenstellung ab (KEIN Template verwenden!):
   - Basierend auf den identifizierten konkreten Problemen
   - Passend zum Register und Stil des Originaltexts
   - Fokussiert auf die spezifischen {bias_type}-Aspekte
   - {task_instruction}

**FORMAT:**
```json
{{
  "text_analysis": {{
    "register": "identified language register",
    "address_form": "Du/Sie/etc.",
    "domain_vocabulary": ["term1", "term2", "term3"],
    "problematic_aspects": ["specific issue 1", "specific issue 2"],
    "target_audience": "identified audience",
    "communication_context": "context description"
  }},
  "instruction": "Generated contextual instruction based on analysis",
  "response": "{response_format}"
}}
```

Wichtig: Die Instruction muss aus der Textanalyse abgeleitet werden, nicht aus vorgefertigten Templates!"""
        
        self.transformation_instruction = "Erstelle eine vollständige inklusive Umformulierung des Textes"
        self.evaluation_instruction = "Bewerte den Text auf einer Skala von 1-10 bezüglich inklusiver Sprache und erkläre konkrete Verbesserungsmöglichkeiten"
    
    def generate_instruction(self, text: str, domain: str, bias_type: str, 
                           task_type: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate contextual instruction for given text.
        
        Args:
            text: Input text to generate instruction for
            domain: Domain context
            bias_type: Bias type focus
            task_type: Task type (transformation/evaluation)
            metadata: Optional metadata
            
        Returns:
            Dictionary with instruction, input, output, and metadata
        """
        try:
            # Determine task instruction based on type
            if task_type == TaskType.TRANSFORMATION.value:
                task_instruction = self.transformation_instruction
                response_format = "Vollständige inklusive Umformulierung mit kurzer Erklärung der Änderungen"
            else:
                task_instruction = self.evaluation_instruction
                response_format = "Bewertung (1-10) + strukturierte Erklärung (Problem/Begründung/Vorschlag)"
            
            # Generate contextual instruction
            prompt = self.analysis_prompt.format(
                text=text,
                domain=domain,
                bias_type=bias_type,
                task_type=task_type,
                task_instruction=task_instruction,
                response_format=response_format
            )
            
            response = self.client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für kontextuelle Aufgabenerstellung ohne Templates."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            parsed_response = self._parse_instruction_response(response_text)
            
            if parsed_response:
                # Generate the actual response/output
                output = self._generate_response(
                    text=text,
                    instruction=parsed_response["instruction"],
                    task_type=task_type,
                    analysis=parsed_response.get("text_analysis", {})
                )
                
                return {
                    "instruction": parsed_response["instruction"],
                    "input": text,
                    "output": output,
                    "metadata": {
                        "domain": domain,
                        "bias_type": bias_type,
                        "task_type": task_type,
                        "text_analysis": parsed_response.get("text_analysis", {}),
                        "generation_method": "contextual_analysis"
                    }
                }
            else:
                # Fallback instruction generation
                return self._generate_fallback_instruction(text, domain, bias_type, task_type)
                
        except Exception as e:
            return self._generate_fallback_instruction(text, domain, bias_type, task_type, str(e))
    
    def _parse_instruction_response(self, response_text: str) -> Optional[Dict]:
        """Parse the instruction generation response.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed response dictionary or None
        """
        try:
            # Extract JSON from response
            import json
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
            
        except (json.JSONDecodeError, Exception):
            return None
    
    def _generate_response(self, text: str, instruction: str, task_type: str, 
                          analysis: Dict) -> str:
        """Generate the response/output for the instruction.
        
        Args:
            text: Original text
            instruction: Generated instruction
            task_type: Task type
            analysis: Text analysis
            
        Returns:
            Generated response
        """
        try:
            if task_type == TaskType.TRANSFORMATION.value:
                response_prompt = f"""Folge dieser Anweisung für den gegebenen Text:

Anweisung: {instruction}

Text: {text}

Erstelle eine vollständige inklusive Umformulierung mit kurzer Erklärung der wichtigsten Änderungen."""
            else:
                response_prompt = f"""Folge dieser Anweisung für den gegebenen Text:

Anweisung: {instruction}

Text: {text}

Bewerte den Text (1-10) und erkläre strukturiert: Problem → Begründung → Verbesserungsvorschlag"""
            
            response = self.client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für inklusive Sprache."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.6,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback response
            if task_type == TaskType.TRANSFORMATION.value:
                return f"Umformulierte Version: {text} (Anpassungen für inklusive Sprache erforderlich)"
            else:
                return "Bewertung: 5/10. Verbesserungen in der inklusiven Sprache möglich."
    
    def _generate_fallback_instruction(self, text: str, domain: str, bias_type: str, 
                                     task_type: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Generate fallback instruction when AI generation fails.
        
        Args:
            text: Input text
            domain: Domain context
            bias_type: Bias type
            task_type: Task type
            error: Optional error message
            
        Returns:
            Fallback instruction dictionary
        """
        # Simple contextual fallback based on analysis
        problematic_terms = self._simple_bias_detection(text, bias_type)
        
        if task_type == TaskType.TRANSFORMATION.value:
            if problematic_terms:
                instruction = f"Überarbeite den Text und verbessere die Aspekte bezüglich {bias_type}, insbesondere: {', '.join(problematic_terms[:3])}"
                output = f"Überarbeitete Version: {text} (Verbesserungen bei: {', '.join(problematic_terms[:2])})"
            else:
                instruction = f"Überarbeite den Text für mehr inklusive Sprache im Bereich {bias_type}"
                output = f"Überarbeitete Version mit inklusiverer Sprache: {text}"
        else:
            if problematic_terms:
                instruction = f"Bewerte den Text bezüglich {bias_type} und identifiziere Verbesserungsmöglichkeiten"
                output = f"Bewertung: 6/10. Problematische Aspekte: {', '.join(problematic_terms[:2])}. Empfehlung: Verwendung inklusiverer Alternativen."
            else:
                instruction = f"Bewerte den Text hinsichtlich inklusiver Sprache im Bereich {bias_type}"
                output = "Bewertung: 7/10. Text ist weitgehend angemessen, kleinere Verbesserungen möglich."
        
        return {
            "instruction": instruction,
            "input": text,
            "output": output,
            "metadata": {
                "domain": domain,
                "bias_type": bias_type,
                "task_type": task_type,
                "generation_method": "fallback",
                "generation_error": error,
                "detected_terms": problematic_terms
            }
        }
    
    def _simple_bias_detection(self, text: str, bias_type: str) -> List[str]:
        """Simple bias detection for fallback.
        
        Args:
            text: Text to analyze
            bias_type: Bias type to look for
            
        Returns:
            List of potentially problematic terms
        """
        text_lower = text.lower()
        problematic_terms = []
        
        # Get some problematic language patterns for the bias type
        bias_config = BiasTypeConfig.get_bias_config(bias_type)
        
        # Check for indicators
        for indicator in bias_config.subtle_indicators[:3]:
            # Simple keyword matching
            keywords = indicator.lower().split()[:2]  # First 2 words as keywords
            for keyword in keywords:
                if keyword in text_lower and keyword not in problematic_terms:
                    problematic_terms.append(keyword)
        
        # Check some specific patterns based on bias type
        if bias_type == "gender":
            gender_patterns = ["der arzt", "die krankenschwester", "herr", "dame"]
            for pattern in gender_patterns:
                if pattern in text_lower:
                    problematic_terms.append(pattern)
        
        elif bias_type == "age":
            age_patterns = ["jung", "alt", "senior", "rentner"]
            for pattern in age_patterns:
                if pattern in text_lower:
                    problematic_terms.append(pattern)
        
        return problematic_terms[:3]  # Return max 3 terms