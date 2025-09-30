"""
Data Processing Module

This module contains classes and functions for processing and analyzing
job market data, including data cleaning, feature engineering, and
statistical analysis capabilities.

Classes:
    JobMarketDataProcessor: Main data processing and cleaning engine
    SparkJobAnalyzer: Spark-based big data analysis tools
    SalaryProcessor: Specialized salary analysis and processing (optional)
    FullDatasetProcessor: Large-scale dataset processing utilities

Functions:
    preprocess_data: Basic data preprocessing utilities
"""

# Import core classes
from .enhanced_processor import JobMarketDataProcessor
from .spark_analyzer import SparkJobAnalyzer
from .full_dataset_processor import FullDatasetProcessor

# Import optional classes with graceful fallback
try:
    from .salary_processor import SalaryProcessor
    _salary_processor_available = True
except ImportError:
    SalaryProcessor = None
    _salary_processor_available = False

__all__ = [
    "JobMarketDataProcessor",
    "SparkJobAnalyzer",
    "FullDatasetProcessor",
]

# Add optional classes to exports if available
if _salary_processor_available:
    __all__.append("SalaryProcessor")