"""
Core module for job market analytics.

This module provides the core analysis and processing capabilities
for the job market analytics system.
"""

from .analyzer import SparkJobAnalyzer, create_spark_analyzer
from .processor import JobMarketDataProcessor

__all__ = [
    'SparkJobAnalyzer',
    'create_spark_analyzer',
    'JobMarketDataProcessor'
]
