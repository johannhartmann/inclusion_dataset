"""Custom exceptions for the inclusion dataset validators."""


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class DiversityValidationError(ValidationError):
    """Raised when diversity metrics validation fails."""
    pass


class TemplateValidationError(ValidationError):
    """Raised when template detection validation fails."""
    pass


class SemanticAnalysisError(ValidationError):
    """Raised when semantic analysis operations fail."""
    pass


class LexicalAnalysisError(ValidationError):
    """Raised when lexical analysis operations fail."""
    pass


class PragmaticAnalysisError(ValidationError):
    """Raised when pragmatic function analysis fails."""
    pass


class InsufficientDataError(ValidationError):
    """Raised when insufficient data is provided for analysis."""
    pass


class ConfigurationError(ValidationError):
    """Raised when configuration parameters are invalid."""
    pass