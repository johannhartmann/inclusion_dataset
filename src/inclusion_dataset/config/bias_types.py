"""Bias type configuration and content templates."""

from typing import Dict, List, Tuple

from pydantic import BaseModel

from .settings import BiasType, TimeEpoch


class BiasContent(BaseModel):
    """Content configuration for a specific bias type."""

    bias_type: BiasType
    description: str
    common_manifestations: List[str]
    problematic_language: Dict[TimeEpoch, List[str]]
    subtle_indicators: List[str]
    transformation_strategies: List[str]


class BiasTypeConfig:
    """Configuration for bias-specific content generation."""

    BIAS_CONFIGS = {
        BiasType.GENDER: BiasContent(
            bias_type=BiasType.GENDER,
            description="Geschlechtsbezogene Vorurteile und Stereotype",
            common_manifestations=[
                "Generisches Maskulinum ohne Alternativen",
                "Geschlechtsspezifische Berufsbezeichnungen",
                "Traditionelle Rollenverteilungen",
                "Ungleiche Behandlung in Führungspositionen",
                "Körperbezogene Beschreibungen",
                "Annahmen über Fähigkeiten und Interessen",
            ],
            problematic_language={
                TimeEpoch.ERA_1990: [
                    "Krankenschwester",
                    "Sekretärin",
                    "Hausfrau",
                    "der Arzt",
                    "der Manager",
                    "der Ingenieur",
                    "schwaches Geschlecht",
                    "Frauensache",
                ],
                TimeEpoch.ERA_2000: [
                    "Mannschaft",
                    "jedermann",
                    "Fachmann",
                    "Dame/Herr",
                    "Mädchen für alles",
                    "Work-Life-Balance nur für Frauen",
                ],
                TimeEpoch.ERA_2010: [
                    "guys (für gemischte Gruppen)",
                    "manpower",
                    "chairman",
                    "spokesman",
                    "Frauen in Führung (als Besonderheit dargestellt)",
                ],
                TimeEpoch.ERA_CURRENT: [
                    "subtile Geschlechterzuschreibungen",
                    "implizite Annahmen über Karrierewünsche",
                    "unbewusste Sprache in neutralen Kontexten",
                ],
            },
            subtle_indicators=[
                "Verwendung von 'natürlich' bei Geschlechterrollen",
                "Unterschiedliche Adjektive für gleiche Eigenschaften",
                "Fokus auf Aussehen bei Frauen vs. Leistung bei Männern",
                "Annahme heteronormativer Beziehungen",
                "Verniedlichung durch Diminutive",
                "Unterschiedliche Erwartungen an emotionale Reaktionen",
            ],
            transformation_strategies=[
                "Gender-inklusive Sprache verwenden",
                "Neutrale Berufsbezeichnungen wählen",
                "Gleichwertige Beschreibungen für alle Geschlechter",
                "Vielfältige Rollenmodelle zeigen",
                "Stereotype bewusst aufbrechen",
                "Intersektionale Perspektiven berücksichtigen",
            ],
        ),
        BiasType.DISABILITY: BiasContent(
            bias_type=BiasType.DISABILITY,
            description="Vorurteile gegenüber Menschen mit Behinderungen",
            common_manifestations=[
                "Defizitorientierte Sprache",
                "Mitleidsvolle oder heroisierte Darstellung",
                "Annahme der Hilfsbedürftigkeit",
                "Fokus auf die Behinderung statt auf die Person",
                "Ausschluss aus 'normalen' Aktivitäten",
                "Medizinische Pathologisierung",
            ],
            problematic_language={
                TimeEpoch.ERA_1990: [
                    "Krüppel",
                    "Invalide",
                    "Behinderte",
                    "an den Rollstuhl gefesselt",
                    "leidet an",
                    "normal vs. behindert",
                    "gesund vs. krank",
                ],
                TimeEpoch.ERA_2000: [
                    "Behinderter",
                    "geistig behindert",
                    "körperlich behindert",
                    "trotz Behinderung",
                    "Opfer von",
                    "betroffen von",
                ],
                TimeEpoch.ERA_2010: [
                    "special needs",
                    "differently abled",
                    "handicapped",
                    "wheelchair-bound",
                    "inspirational",
                    "brave",
                ],
                TimeEpoch.ERA_CURRENT: [
                    "subtile Abwertung durch Infantilisierung",
                    "Überbetonung der 'Normalität'",
                    "Vermeidung des Themas als Form der Diskriminierung",
                ],
            },
            subtle_indicators=[
                "Verwendung euphemistischer Begriffe",
                "Annahme eingeschränkter Fähigkeiten",
                "Fokus auf Inspiration statt Gleichberechtigung",
                "Ausschluss aus Entscheidungsprozessen",
                "Überfürsorglichkeit und Bevormundung",
                "Reduktion auf die Behinderung",
            ],
            transformation_strategies=[
                "People-First oder Identity-First Sprache",
                "Fokus auf Fähigkeiten und Beiträge",
                "Barrierefreiheit als Standard mitdenken",
                "Selbstbestimmung und Autonomie betonen",
                "Vielfalt als Normalität darstellen",
                "Gesellschaftliche Barrieren thematisieren",
            ],
        ),
        BiasType.ETHNICITY: BiasContent(
            bias_type=BiasType.ETHNICITY,
            description="Ethnische und kulturelle Vorurteile und Stereotype",
            common_manifestations=[
                "Kulturelle Verallgemeinerungen",
                "Exotisierung und Othering",
                "Annahme mangelnder Deutschkenntnisse",
                "Reduktion auf Herkunftskultur",
                "Diskriminierung bei Namen",
                "Mikroaggressionen im Alltag",
            ],
            problematic_language={
                TimeEpoch.ERA_1990: [
                    "Ausländer",
                    "Gastarbeiter",
                    "deutsche vs. ausländische Kinder",
                    "Überfremdung",
                    "Integration verweigern",
                    "typisch türkisch/russisch etc.",
                ],
                TimeEpoch.ERA_2000: [
                    "Migrationshintergrund (als Defizit)",
                    "nicht-deutsche Herkunft",
                    "kulturelle Eigenarten",
                    "mangelnde Integrationswilligkeit",
                ],
                TimeEpoch.ERA_2010: [
                    "Menschen mit Migrationshintergrund (Dauerbetonung)",
                    "kulturell bedingt",
                    "Parallelgesellschaften",
                    "gescheiterte Integration",
                ],
                TimeEpoch.ERA_CURRENT: [
                    "subtile Othering-Strategien",
                    "woher kommst du wirklich?",
                    "du sprichst aber gut Deutsch",
                    "unbewusste Vorurteile in der Sprache",
                ],
            },
            subtle_indicators=[
                "Ständige Betonung der 'anderen' Herkunft",
                "Annahme kultureller Uniformität",
                "Unterschiedliche Standards für 'Deutsche' und 'Andere'",
                "Exotisierung von Namen und Traditionen",
                "Defizitorientierte Beschreibungen",
                "Instrumentalisierung für Diversitäts-PR",
            ],
            transformation_strategies=[
                "Individualität vor Gruppenzugehörigkeit",
                "Vielfalt als gesellschaftliche Realität",
                "Mehrfachzugehörigkeiten anerkennen",
                "Gleichwertige Behandlung aller Kulturen",
                "Systemische Barrieren benennen",
                "Empowerment statt Defizitorientierung",
            ],
        ),
        BiasType.AGE: BiasContent(
            bias_type=BiasType.AGE,
            description="Altersbezogene Vorurteile und Diskriminierung",
            common_manifestations=[
                "Stereotype über ältere Arbeitnehmer",
                "Jugendlichkeit als Standard",
                "Annahmen über technische Fähigkeiten",
                "Altersgrenzen ohne Sachgrund",
                "Generationenkonflikt-Narrative",
                "Infantilisierung oder Überforderungsannahmen",
            ],
            problematic_language={
                TimeEpoch.ERA_1990: [
                    "alte Knacker",
                    "Rentnerdasein",
                    "altmodisch",
                    "nicht mehr zeitgemäß",
                    "junge Wilde vs. alte Hasen",
                    "Generationenkonflikt",
                ],
                TimeEpoch.ERA_2000: [
                    "Silver Ager",
                    "Generation Golf",
                    "Digital Natives vs. Digital Immigrants",
                    "Wissenstransfer vor Rente",
                    "zu alt für Umschulung",
                ],
                TimeEpoch.ERA_2010: [
                    "Millennials vs. Boomer",
                    "Demografischer Wandel als Problem",
                    "Digitale Kompetenz altersabhängig",
                    "Flexibilität der Jungen",
                ],
                TimeEpoch.ERA_CURRENT: [
                    "ok boomer",
                    "entitled millennials",
                    "Gen Z vs. alle anderen",
                    "subtile Altersdiskriminierung in Stellenanzeigen",
                ],
            },
            subtle_indicators=[
                "Annahmen über Lernfähigkeit nach Alter",
                "Unterschiedliche Erwartungen an Flexibilität",
                "Technologie-Kompetenz als Altersfrage",
                "Karriereentwicklung mit Alterslimits",
                "Gesundheit und Leistungsfähigkeit verknüpft",
                "Weisheit vs. Innovation als Altersstereotyp",
            ],
            transformation_strategies=[
                "Altersvielfalt als Bereicherung darstellen",
                "Kompetenzen unabhängig vom Alter bewerten",
                "Lebenslanges Lernen für alle Altersgruppen",
                "Flexible Arbeitsmodelle generationsübergreifend",
                "Mentoring in beide Richtungen",
                "Intersektionale Altersbetrachtung",
            ],
        ),
        BiasType.SOCIOECONOMIC: BiasContent(
            bias_type=BiasType.SOCIOECONOMIC,
            description="Vorurteile basierend auf sozialer Schicht und wirtschaftlicher Lage",
            common_manifestations=[
                "Bildungsabschluss als Persönlichkeitsmerkmal",
                "Armut als individuelle Schuld",
                "Mittelschicht als Norm",
                "Klassistische Sprache",
                "Unterschiedliche Behandlung nach Status",
                "Zugang zu Ressourcen als Selbstverständlichkeit",
            ],
            problematic_language={
                TimeEpoch.ERA_1990: [
                    "einfache Leute",
                    "niedere Schichten",
                    "Unterschicht",
                    "Bildungsferne",
                    "sozial schwach",
                    "bessere Kreise",
                ],
                TimeEpoch.ERA_2000: [
                    "bildungsfern",
                    "Hartz IV Empfänger",
                    "Prekariat",
                    "Sozialschmarotzer",
                    "leistungsbereit vs. leistungsunwillig",
                ],
                TimeEpoch.ERA_2010: [
                    "Modernisierungsverlierer",
                    "abgehängt",
                    "Bildungsverlierer",
                    "systemrelevant (nur in Krisen)",
                    "verdient es nicht besser",
                ],
                TimeEpoch.ERA_CURRENT: [
                    "selbst schuld an der Situation",
                    "hätte ja studieren können",
                    "working poor",
                    "subtile Klassendiskriminierung",
                ],
            },
            subtle_indicators=[
                "Annahmen über Intelligenz basierend auf Abschlüssen",
                "Kultureller Kapital als Bewertungsmaßstab",
                "Networking-Vorteile als Selbstverständlichkeit",
                "Sprache und Habitus als Klassifikation",
                "Zugang zu Technologie und Ressourcen",
                "Freizeit und Lebensstil als Statusmerkmal",
            ],
            transformation_strategies=[
                "Vielfalt der Bildungswege anerkennen",
                "Strukturelle Barrieren thematisieren",
                "Klassismus als Diskriminierungsform benennen",
                "Würde unabhängig von wirtschaftlicher Lage",
                "Verschiedene Lebensrealitäten respektieren",
                "Systemische Ungleichheit adressieren",
            ],
        ),
        BiasType.RELIGION: BiasContent(
            bias_type=BiasType.RELIGION,
            description="Religiöse Vorurteile und Diskriminierung",
            common_manifestations=[
                "Christliche Normen als Standard",
                "Islamophobie und Antisemitismus",
                "Säkularismus als Überlegenheit",
                "Religiöse Praxis als rückständig",
                "Missionarische Haltungen",
                "Ausschluss religiöser Minderheiten",
            ],
            problematic_language={
                TimeEpoch.ERA_1990: [
                    "christlich-abendländische Kultur",
                    "Ungläubige",
                    "Heiden",
                    "fundamentalistisch",
                    "fanatisch",
                    "religiöse Spinner",
                ],
                TimeEpoch.ERA_2000: [
                    "Islamismus-Verdacht",
                    "nicht in unsere Kultur passend",
                    "religiöse Eigenarten",
                    "Aufklärung vs. Religion",
                ],
                TimeEpoch.ERA_2010: [
                    "Parallelgesellschaften durch Religion",
                    "Religion als Integrationshemmnis",
                    "moderne vs. traditionelle Werte",
                    "Religionsfreiheit aber...",
                ],
                TimeEpoch.ERA_CURRENT: [
                    "subtile Islamophobie",
                    "Antisemitismus in neuem Gewand",
                    "Religion als Privatsache (Ausgrenzung)",
                    "Token-Diversität in religiösen Fragen",
                ],
            },
            subtle_indicators=[
                "Christliche Feiertage als Standard",
                "Säkulare Überlegenheitsgefühle",
                "Religiöse Symbole als Problem",
                "Missionarische Tendenzen",
                "Exotisierung östlicher Religionen",
                "Annahme der Rückständigkeit",
            ],
            transformation_strategies=[
                "Religiöse Vielfalt als Bereicherung",
                "Gleichberechtigte Behandlung aller Religionen",
                "Säkularität ohne Religionsfeindlichkeit",
                "Interreligiöser Dialog auf Augenhöhe",
                "Religionsfreiheit als Grundrecht",
                "Spiritualität in verschiedenen Formen respektieren",
            ],
        ),
    }

    @classmethod
    def get_bias_config(cls, bias_type: BiasType) -> BiasContent:
        """Get configuration for a specific bias type."""
        return cls.BIAS_CONFIGS.get(bias_type, cls.BIAS_CONFIGS[BiasType.GENDER])

    @classmethod
    def get_all_bias_types(cls) -> List[BiasType]:
        """Get list of all available bias types."""
        return list(cls.BIAS_CONFIGS.keys())

    @classmethod
    def get_problematic_language(
        cls, bias_type: BiasType, epoch: TimeEpoch
    ) -> List[str]:
        """Get problematic language for specific bias type and time epoch."""
        config = cls.get_bias_config(bias_type)
        return config.problematic_language.get(epoch, [])

    @classmethod
    def get_transformation_strategies(cls, bias_type: BiasType) -> List[str]:
        """Get transformation strategies for a specific bias type."""
        return cls.get_bias_config(bias_type).transformation_strategies

    @classmethod
    def get_subtle_indicators(cls, bias_type: BiasType) -> List[str]:
        """Get subtle bias indicators for a specific bias type."""
        return cls.get_bias_config(bias_type).subtle_indicators
