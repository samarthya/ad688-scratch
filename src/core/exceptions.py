"""
Custom exception hierarchy for job market analysis.

Provides specific exception types for different error scenarios
in the data processing and analysis pipeline.
"""


class JobMarketAnalysisError(Exception):
    """Base exception for all job market analysis errors."""
    pass


class DataValidationError(JobMarketAnalysisError):
    """Raised when data validation fails."""
    pass


class DataLoadingError(JobMarketAnalysisError):
    """Raised when data loading fails."""
    pass


class ProcessingError(JobMarketAnalysisError):
    """Raised when data processing fails."""
    pass


class VisualizationError(JobMarketAnalysisError):
    """Raised when chart generation or export fails."""
    pass


class ConfigurationError(JobMarketAnalysisError):
    """Raised when configuration is invalid or missing."""
    pass
