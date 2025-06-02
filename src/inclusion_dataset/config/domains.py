"""Domain-specific configuration and content templates."""

from typing import Dict, List, Optional

from pydantic import BaseModel

from .settings import Domain, FormalityLevel


class DomainContent(BaseModel):
    """Content configuration for a specific domain."""

    domain: Domain
    typical_contexts: List[str]
    professional_roles: List[str]
    common_scenarios: List[str]
    language_registers: Dict[FormalityLevel, str]
    bias_indicators: List[str]


class DomainConfig:
    """Configuration for domain-specific content generation."""

    DOMAIN_CONFIGS = {
        Domain.WORKPLACE: DomainContent(
            domain=Domain.WORKPLACE,
            typical_contexts=[
                "Stellenausschreibungen und Personalsuche",
                "Mitarbeiterbewertungen und Feedback",
                "Team-Meetings und Projektbesprechungen",
                "Unternehmenskommunikation und HR-Richtlinien",
                "Beförderungen und Karriereentwicklung",
                "Diversity und Inklusion Initiativen",
                "Arbeitszeugnisse und Empfehlungsschreiben",
                "Weiterbildung und Schulungsmaßnahmen",
            ],
            professional_roles=[
                "Geschäftsführung",
                "Personalleitung",
                "Teamleitung",
                "Projektmanagement",
                "Fachkräfte",
                "Auszubildende",
                "Assistenz",
                "Beratung",
                "Entwicklung",
                "Vertrieb",
                "Marketing",
                "Administration",
            ],
            common_scenarios=[
                "Bewerbungsgespräch führen",
                "Leistungsbeurteilung erstellen",
                "Teamzusammenstellung planen",
                "Konflikt zwischen Kollegen lösen",
                "Neue Mitarbeiter einarbeiten",
                "Arbeitszeiten flexibilisieren",
                "Betriebsversammlung moderieren",
                "Kundenbeziehung pflegen",
            ],
            language_registers={
                FormalityLevel.FORMAL: "Sehr geehrte Damen und Herren, Geschäftsleitung, offizieller Ton",
                FormalityLevel.SEMI_FORMAL: "Liebe Kolleginnen und Kollegen, Teamleitung, professioneller Ton",
                FormalityLevel.INFORMAL: "Hallo Team, lockere Büroatmosphäre, direkter Ton",
                FormalityLevel.TECHNICAL: "Fachbegriffe, Prozessbeschreibungen, methodischer Ton",
                FormalityLevel.COLLOQUIAL: "Pausengespräche, interne Kommunikation, persönlicher Ton",
            },
            bias_indicators=[
                "Geschlechtsspezifische Berufsbezeichnungen",
                "Altersdiskriminierung bei Stellenausschreibungen",
                "Kulturelle Vorurteile in der Teamzusammenstellung",
                "Sozialer Status und Bildungshintergrund",
                "Behinderung und Barrierefreiheit",
                "Religiöse Feiertage und Praktiken",
            ],
        ),
        Domain.EDUCATION: DomainContent(
            domain=Domain.EDUCATION,
            typical_contexts=[
                "Schulische Bewertungen und Zeugnisse",
                "Eltern-Lehrer-Gespräche",
                "Unterrichtsmaterialien und Curricula",
                "Universitäre Aufnahmeprüfungen",
                "Bildungsberatung und Laufbahnplanung",
                "Inklusive Pädagogik und Förderprogramme",
                "Lehrer-Fortbildungen",
                "Bildungspolitische Diskussionen",
            ],
            professional_roles=[
                "Lehrkräfte",
                "Schulleitung",
                "Bildungsberater",
                "Sozialpädagogen",
                "Universitätsdozenten",
                "Studienberater",
                "Eltern",
                "Schüler",
                "Studierende",
                "Bildungspolitiker",
                "Psychologen",
                "Therapeuten",
            ],
            common_scenarios=[
                "Schülerleistung bewerten",
                "Individuelle Förderung planen",
                "Elterngespräch führen",
                "Unterricht differenziert gestalten",
                "Mobbing-Situation bearbeiten",
                "Inklusion von Schülern mit Behinderung",
                "Mehrsprachigkeit im Klassenzimmer",
                "Berufsorientierung anbieten",
            ],
            language_registers={
                FormalityLevel.FORMAL: "Amtliche Bescheide, Zeugnisse, offizielle Schulkommunikation",
                FormalityLevel.SEMI_FORMAL: "Elternbriefe, Lehrerkonferenzen, pädagogische Berichte",
                FormalityLevel.INFORMAL: "Klassenzimmergespräche, Pausenaufsicht, lockere Beratung",
                FormalityLevel.TECHNICAL: "Pädagogische Fachbegriffe, Diagnostik, Methodik",
                FormalityLevel.COLLOQUIAL: "Schülerjargon, Pausenhofgespräche, familiäre Atmosphäre",
            },
            bias_indicators=[
                "Geschlechtsspezifische Fächerwahl und Erwartungen",
                "Kulturelle und sprachliche Vorurteile",
                "Sozioökonomische Benachteiligung",
                "Lernbehinderung und besondere Bedürfnisse",
                "Migrationshintergrund und Integration",
                "Religiöse Unterschiede im Schulalltag",
            ],
        ),
        Domain.HEALTHCARE: DomainContent(
            domain=Domain.HEALTHCARE,
            typical_contexts=[
                "Patientenaufklärung und Anamnese",
                "Arzt-Patient-Kommunikation",
                "Pflegedokumentation",
                "Therapiepläne und Behandlungsempfehlungen",
                "Gesundheitsvorsorge und Prävention",
                "Notfallsituationen und Erstversorgung",
                "Rehabilitation und Nachsorge",
                "Gesundheitspolitische Maßnahmen",
            ],
            professional_roles=[
                "Ärzte",
                "Pflegekräfte",
                "Therapeuten",
                "Heilpraktiker",
                "Patienten",
                "Angehörige",
                "Verwaltung",
                "Apotheker",
                "Sozialarbeiter",
                "Psychologen",
                "Rettungsdienst",
                "Krankenkassen",
            ],
            common_scenarios=[
                "Diagnose erklären",
                "Behandlungsplan besprechen",
                "Medikamentenaufklärung durchführen",
                "Pflegebedürftigkeit einschätzen",
                "Präventionsmaßnahmen empfehlen",
                "Notfallversorgung koordinieren",
                "Rehabilitation begleiten",
                "Patientenrechte aufklären",
            ],
            language_registers={
                FormalityLevel.FORMAL: "Medizinische Berichte, Gutachten, amtliche Bescheinigungen",
                FormalityLevel.SEMI_FORMAL: "Arztbriefe, Pflegeplanung, Therapieempfehlungen",
                FormalityLevel.INFORMAL: "Patientengespräche, Beratung, empathische Kommunikation",
                FormalityLevel.TECHNICAL: "Medizinische Fachterminologie, Diagnostik, Behandlungsprotokolle",
                FormalityLevel.COLLOQUIAL: "Alltägliche Gesundheitstipps, Patientenbetreuung, persönliche Gespräche",
            },
            bias_indicators=[
                "Geschlechtsspezifische Symptombewertung",
                "Kulturelle Unterschiede in der Schmerzwahrnehmung",
                "Altersbedingte Behandlungsunterschiede",
                "Sozioökonomische Faktoren bei der Versorgung",
                "Behinderung und Zugänglichkeit zu Behandlung",
                "Religiöse Überzeugungen und medizinische Eingriffe",
            ],
        ),
        Domain.MEDIA: DomainContent(
            domain=Domain.MEDIA,
            typical_contexts=[
                "Nachrichtenberichterstattung",
                "Journalistische Interviews",
                "Redaktionelle Kommentare",
                "Social Media Content",
                "Werbung und Marketing",
                "Dokumentationen und Features",
                "Pressekonferenzen",
                "Medienkritik und Analysen",
            ],
            professional_roles=[
                "Journalisten",
                "Redakteure",
                "Moderatoren",
                "Reporter",
                "Fotografen",
                "Kameraleute",
                "Produzenten",
                "Medienmacher",
                "Pressesprecher",
                "PR-Manager",
                "Influencer",
                "Blogger",
            ],
            common_scenarios=[
                "Nachricht verfassen",
                "Interview führen",
                "Kommentar schreiben",
                "Social Media Post erstellen",
                "Werbekampagne entwickeln",
                "Dokumentation produzieren",
                "Pressekonferenz moderieren",
                "Medienbeitrag kritisieren",
            ],
            language_registers={
                FormalityLevel.FORMAL: "Offizielle Pressemitteilungen, Nachrichten, seriöse Berichterstattung",
                FormalityLevel.SEMI_FORMAL: "Interviews, Features, journalistische Kommentare",
                FormalityLevel.INFORMAL: "Social Media, Blogs, lockere Gespräche",
                FormalityLevel.TECHNICAL: "Medienwissenschaftliche Analysen, Fachterminologie",
                FormalityLevel.COLLOQUIAL: "Boulevardjournalismus, Unterhaltung, populäre Medien",
            },
            bias_indicators=[
                "Geschlechtsstereotype in der Berichterstattung",
                "Kulturelle Vorurteile in der Darstellung",
                "Altersdiskriminierung in der Medienrepräsentation",
                "Soziale Schicht und Medienaufmerksamkeit",
                "Behinderung in der Mediendarstellung",
                "Religiöse Minderheiten in den Medien",
            ],
        ),
    }

    @classmethod
    def get_domain_config(cls, domain: Domain) -> DomainContent:
        """Get configuration for a specific domain."""
        return cls.DOMAIN_CONFIGS.get(domain, cls.DOMAIN_CONFIGS[Domain.WORKPLACE])

    @classmethod
    def get_all_domains(cls) -> List[Domain]:
        """Get list of all available domains."""
        return list(cls.DOMAIN_CONFIGS.keys())

    @classmethod
    def get_contexts_for_domain(cls, domain: Domain) -> List[str]:
        """Get typical contexts for a domain."""
        return cls.get_domain_config(domain).typical_contexts

    @classmethod
    def get_roles_for_domain(cls, domain: Domain) -> List[str]:
        """Get professional roles for a domain."""
        return cls.get_domain_config(domain).professional_roles

    @classmethod
    def get_language_register(cls, domain: Domain, formality: FormalityLevel) -> str:
        """Get language register description for domain and formality level."""
        config = cls.get_domain_config(domain)
        return config.language_registers.get(
            formality, "Standard professional communication"
        )
