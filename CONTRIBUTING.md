# Contributing to Inclusion Dataset Generator

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the inclusion dataset generator.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- OpenAI API key (for testing with real API calls)

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/johann-peter-hartmann/inclusion-dataset.git
   cd inclusion-dataset
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Standards

### Style Guidelines

This project follows PEP 8 and uses automated formatting tools:

- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking

### Running Code Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### Type Hints

All new code should include type hints. Use the following patterns:

```python
from typing import List, Dict, Optional, Union

def process_data(items: List[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Process data with proper type hints."""
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/inclusion_dataset --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run tests without API calls
python scripts/dry_run.py --samples 3 --no-api
```

### Writing Tests

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Mock external dependencies**: Use `unittest.mock` for API calls

Example test structure:

```python
import pytest
from unittest.mock import Mock, patch

class TestMyComponent:
    def test_basic_functionality(self):
        """Test basic functionality with clear assertions."""
        component = MyComponent()
        result = component.process("test input")
        assert result == "expected output"
    
    @patch('module.external_dependency')
    def test_with_mocked_dependency(self, mock_dep):
        """Test with mocked external dependencies."""
        mock_dep.return_value = "mocked response"
        # Test implementation
```

## Contributing Guidelines

### Issue Reporting

When reporting issues, please include:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Environment details** (Python version, OS, etc.)
5. **Relevant logs or error messages**

### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code standards above

3. **Add or update tests** for your changes

4. **Update documentation** if necessary

5. **Run the full test suite** and ensure all checks pass:
   ```bash
   pytest
   pre-commit run --all-files
   ```

6. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "feat: add new diversity metric calculation"
   ```

7. **Push to your fork** and create a pull request

### Commit Message Format

Use conventional commit format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

Example:
```
feat: add semantic clustering validation

- Implement sentence transformer-based clustering
- Add configurable cluster size limits  
- Include cluster distribution metrics
```

### Pull Request Requirements

Your PR must:

- [ ] Pass all existing tests
- [ ] Include tests for new functionality
- [ ] Follow code style guidelines
- [ ] Include clear documentation
- [ ] Have a descriptive title and description
- [ ] Reference any related issues

## Architecture Guidelines

### Adding New Components

When adding new components to the pipeline:

1. **Follow existing patterns** in the codebase
2. **Use Pydantic models** for configuration
3. **Include comprehensive logging** with appropriate levels
4. **Implement error handling** with meaningful messages
5. **Add configuration options** to `settings.py`

### Validator Guidelines

New validators should:

- Inherit from appropriate base classes
- Include configurable thresholds
- Provide detailed feedback on failures
- Support both single items and batch processing

### Configuration Management

- Add new settings to `PipelineConfig` in `settings.py`
- Use environment variables for sensitive data
- Provide sensible defaults
- Include validation with clear error messages

## Documentation

### Code Documentation

- Use clear, descriptive docstrings
- Follow Google or NumPy docstring format
- Include parameter and return type information
- Provide usage examples for complex functions

### README Updates

When adding features, update:

- Feature descriptions
- Configuration options
- Usage examples
- Installation requirements

## Getting Help

- **Issues**: Create an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact johann-peter.hartmann@mayflower.de for major contributions

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful and inclusive in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.