"""Content generation for diverse text samples."""

import itertools
import random
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from ..config.bias_types import BiasTypeConfig
from ..config.domains import DomainConfig
from ..config.settings import (
    BiasType,
    Config,
    Domain,
    FormalityLevel,
    TaskType,
    TimeEpoch,
)


class ContentGenerator:
    """Generate diverse content following the content matrix approach."""

    def __init__(self, config: Config):
        """Initialize content generator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

        # Content generation prompt template
        self.content_prompt = """Du bist ein Experte für realistische Textgenerierung in verschiedenen gesellschaftlichen Kontexten.

Erstelle einen authentischen deutschen Text mit folgenden Spezifikationen:

**Kontext:**
- Bereich: {domain}
- Zeitperiode: {time_epoch}
- Formalitätsgrad: {formality}
- Bias-Typ (zu integrieren): {bias_type}

**Spezifische Anforderungen:**
- Textlänge: {min_length}-{max_length} Wörter
- Der Text soll {bias_description} enthalten
- Verwende {language_register}
- Integriere typische Situationen: {scenarios}
- Sprachstil der {time_epoch}: {time_specific_language}

**Wichtig:**
- Der Text muss realistisch und wie echte Kommunikation klingen
- Bias soll natürlich eingebaut sein, nicht konstruiert wirken
- Verwende authentische Sprache der gewählten Zeitperiode
- Der Text soll Verbesserungspotential haben, aber nicht übertrieben problematisch sein

Generiere NUR den Text, ohne Kommentare oder Erklärungen."""

    def generate_content(
        self,
        domain: str,
        bias_type: str,
        time_epoch: str,
        formality: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Generate content for given parameters.

        Args:
            domain: Domain context
            bias_type: Type of bias to include
            time_epoch: Time period
            formality: Formality level
            task_type: Task type (transformation/evaluation)

        Returns:
            Dictionary with generated content and metadata
        """
        # Get domain and bias configurations
        domain_enum = Domain(domain)
        bias_enum = BiasType(bias_type)
        epoch_enum = TimeEpoch(time_epoch)
        formality_enum = FormalityLevel(formality)

        domain_config = DomainConfig.get_domain_config(domain_enum)
        bias_config = BiasTypeConfig.get_bias_config(bias_enum)

        # Prepare prompt parameters
        prompt_params = {
            "domain": domain,
            "time_epoch": time_epoch,
            "formality": formality,
            "bias_type": bias_type,
            "min_length": self.config.min_text_length,
            "max_length": self.config.max_text_length,
            "bias_description": bias_config.description,
            "language_register": domain_config.language_registers.get(
                formality_enum, "Standard"
            ),
            "scenarios": ", ".join(
                random.sample(
                    domain_config.common_scenarios,
                    min(3, len(domain_config.common_scenarios)),
                )
            ),
            "time_specific_language": ", ".join(
                BiasTypeConfig.get_problematic_language(bias_enum, epoch_enum)[:3]
            ),
        }

        try:
            # Generate content
            prompt = self.content_prompt.format(**prompt_params)

            response = self.client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für realistische Textgenerierung.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=800,
            )

            generated_text = (response.choices[0].message.content or "").strip()

            # Validate text length
            word_count = len(generated_text.split())
            if word_count < self.config.min_text_length:
                # Extend text if too short
                generated_text = self._extend_text(generated_text, prompt_params)
            elif word_count > self.config.max_text_length:
                # Truncate if too long
                words = generated_text.split()
                generated_text = " ".join(words[: self.config.max_text_length])

            return {
                "text": generated_text,
                "metadata": {
                    "domain": domain,
                    "bias_type": bias_type,
                    "time_epoch": time_epoch,
                    "formality": formality,
                    "task_type": task_type,
                    "word_count": len(generated_text.split()),
                    "generation_params": prompt_params,
                },
            }

        except Exception as e:
            # Fallback content generation
            return self._generate_fallback_content(prompt_params, str(e))

    def get_content_combinations(self) -> List[Dict[str, Any]]:
        """Get all possible content matrix combinations.

        Returns:
            List of all combinations
        """
        combinations = []

        # Get all possible combinations
        for domain in self.config.domains:
            for bias_type in self.config.bias_types:
                for time_epoch in self.config.time_epochs:
                    for formality in self.config.formality_levels:
                        for task_type in [TaskType.TRANSFORMATION, TaskType.EVALUATION]:
                            combinations.append(
                                {
                                    "domain": domain,
                                    "bias_type": bias_type,
                                    "time_epoch": time_epoch,
                                    "formality": formality,
                                    "task_type": task_type,
                                }
                            )

        # Shuffle for diversity
        random.shuffle(combinations)
        return combinations

    def _extend_text(self, text: str, params: Dict[str, Any]) -> str:
        """Extend text if it's too short.

        Args:
            text: Original text
            params: Generation parameters

        Returns:
            Extended text
        """
        extension_prompt = f"""Erweitere den folgenden Text um weitere Details und Kontext, aber behalte den ursprünglichen Inhalt und Stil bei:

"{text}"

Füge weitere {params['min_length'] - len(text.split())} Wörter hinzu, die zum Kontext ({params['domain']}, {params['time_epoch']}) passen."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Du erweiterst Texte kontextuell passend.",
                    },
                    {"role": "user", "content": extension_prompt},
                ],
                temperature=0.7,
                max_tokens=400,
            )

            return (response.choices[0].message.content or "").strip()

        except Exception:
            # Simple fallback: repeat with variation
            return (
                text
                + f" Weitere Details zu diesem Thema im Kontext von {params['domain']} sind besonders relevant."
            )

    def _generate_fallback_content(
        self, params: Dict[str, Any], error: str
    ) -> Dict[str, Any]:
        """Generate fallback content when API fails.

        Args:
            params: Generation parameters
            error: Error message

        Returns:
            Fallback content dictionary
        """
        # Simple template-based fallback
        fallback_templates = {
            Domain.WORKPLACE: "In unserem Unternehmen arbeiten verschiedene Teams zusammen. Die Führungskräfte treffen wichtige Entscheidungen für die Zukunft.",
            Domain.EDUCATION: "In der Schule lernen die Schüler verschiedene Fächer. Die Lehrer unterstützen dabei die individuelle Entwicklung.",
            Domain.HEALTHCARE: "Im Krankenhaus kümmern sich die Ärzte und Pflegekräfte um die Patienten. Die Behandlung erfolgt nach modernsten Standards.",
            Domain.MEDIA: "In den Medien werden aktuelle Ereignisse berichtet. Journalisten recherchieren und präsentieren die Nachrichten.",
        }

        domain_key = Domain(params["domain"])
        base_text = fallback_templates.get(
            domain_key, "Dies ist ein Beispieltext für die Generierung."
        )

        return {
            "text": base_text,
            "metadata": {
                "domain": params["domain"],
                "bias_type": params["bias_type"],
                "time_epoch": params["time_epoch"],
                "formality": params["formality"],
                "task_type": params.get("task_type", "transformation"),
                "word_count": len(base_text.split()),
                "generation_error": error,
                "fallback_used": True,
            },
        }
