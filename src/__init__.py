"""
Job Market Analysis Package

A comprehensive data analysis package for job market research and insights.
Provides tools for data processing, visualization, and statistical analysis
of job posting data from various sources.

Modules:
    data: Data processing and analysis modules
    visualization: Chart and plot generation modules  
    config: Configuration and mapping modules
    utilities: Utility functions and helpers

Author: Saurabh Sharma
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Saurabh Sharma"

# Import main classes for easy access
from .data.enhanced_processor import JobMarketDataProcessor
from .data.spark_analyzer import SparkJobAnalyzer
from .visualization.quarto_charts import QuartoChartExporter

__all__ = [
    "JobMarketDataProcessor",
    "SparkJobAnalyzer", 
    "QuartoChartExporter",
]