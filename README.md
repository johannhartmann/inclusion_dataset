# Inclusion Dataset Generator

A distilabel pipeline for generating inclusive German language supervised fine-tuning (SFT) datasets. This system creates 20,000 high-quality instruction-input-output samples focused on teaching German language models (8-21B parameters) inclusive language practices across 6 bias dimensions.

## Features

- **Multi-dimensional bias coverage**: Gender, disability, ethnicity, age, socioeconomic status, and religion
- **Systematic content generation**: 960 unique combinations across domains, bias types, time epochs, and formality levels
- **Anti-template enforcement**: Prevents template-based generation through diversity metrics
- **Quality validation**: LLM-as-a-Judge validation with comprehensive quality scoring
- **Epoch-aware language**: Generates period-appropriate language patterns from 1990s to current
- **Production-ready pipeline**: Built on distilabel for scalable dataset generation

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd inclusion_dataset

# Install with pip
pip install -r requirements.txt

# Or with pipenv
pipenv install
```

### Environment Setup

Create a `.env` file with your API key:

```bash
OPENAI_API_KEY=your_key_here
TOTAL_SAMPLES=20000
MIN_QUALITY_SCORE=7.0
MAX_TEMPLATE_OVERLAP=0.05
```

Install required NLTK data:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### Running the Pipeline

**Test project structure (no API calls):**
```bash
python scripts/test_structure.py
```

**Dry run with API:**
```bash
python scripts/dry_run.py --samples 3
```

**Run without API calls (mock mode):**
```bash
python scripts/dry_run.py --samples 3 --no-api
```

**Full pipeline:**
```bash
python -m src.inclusion_dataset.pipeline.main
```

## Architecture

### Core Components

- **Content Generator** (`src/inclusion_dataset/pipeline/content_generator.py`): Creates authentic domain-specific texts with natural bias integration
- **Instruction Generator** (`src/inclusion_dataset/pipeline/instruction_generator.py`): Derives contextual instructions through text analysis
- **Quality Judge** (`src/inclusion_dataset/validators/quality_judge.py`): LLM-as-a-Judge validation with 7/10 minimum score
- **Diversity Metrics** (`src/inclusion_dataset/validators/diversity_metrics.py`): Enforces lexical diversity and semantic clustering
- **Template Detector** (`src/inclusion_dataset/validators/template_detector.py`): Prevents template-based generation

### Configuration System

- **Settings** (`src/inclusion_dataset/config/settings.py`): Main configuration with Pydantic models
- **Domains** (`src/inclusion_dataset/config/domains.py`): Domain-specific contexts, roles, and scenarios
- **Bias Types** (`src/inclusion_dataset/config/bias_types.py`): Time-period-specific language patterns

### Content Matrix

The system generates content using systematic combinations:
- **8 domains**: Education, healthcare, workplace, social media, news, literature, legal, technology
- **6 bias types**: Gender, disability, ethnicity, age, socioeconomic status, religion
- **4 time epochs**: 1990s, 2000s, 2010s, current
- **5 formality levels**: Very formal to very informal

This results in **960 unique combinations** ensuring comprehensive coverage.

## Quality Standards

### Anti-Template Enforcement
- Automatic rejection if template overlap > 5%
- N-gram diversity: >80% unigrams, >90% bigrams, >95% trigrams unique
- Semantic clustering: Max 20 clusters, no cluster >8% of data
- Pragmatic functions: Must use 12+ communication functions

### Content Requirements
- **Domain authenticity**: Use domain-specific language registers and scenarios
- **Time-aware bias**: Integrate era-appropriate problematic language naturally
- **Text length**: 150-300 words per input text
- **LLM validation**: Minimum 7/10 score for acceptance

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/inclusion_dataset

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

Test the pipeline without API calls:

```bash
python scripts/dry_run.py --samples 5 --no-api --verbose
```

## Output Format

The pipeline generates datasets in standard SFT JSONL format:

```json
{
  "instruction": "Rewrite the following text to use more inclusive language...",
  "input": "The original text with biased language patterns...",
  "output": "The corrected text with inclusive alternatives..."
}
```

## Configuration

Key configuration options in `.env`:

- `TOTAL_SAMPLES`: Total number of samples to generate (default: 20000)
- `MIN_QUALITY_SCORE`: Minimum LLM judge score (default: 7.0)
- `MAX_TEMPLATE_OVERLAP`: Maximum allowed template similarity (default: 0.05)
- `OPENAI_API_KEY`: Required for LLM-based generation and validation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Johann-Peter Hartmann (johann-peter.hartmann@mayflower.de)

