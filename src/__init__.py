"""
Job Market Analytics Package

A comprehensive system for analyzing job market data using PySpark,
with focus on salary disparity analysis and interactive visualizations.

Main Components:
- Core: Business logic and analysis engines
- Data: Data loading, validation, and transformation
- Visualization: Chart generation and export
- Config: Configuration and settings management
"""

from .core.analyzer import SparkJobAnalyzer, create_spark_analyzer
from .core.processor import JobMarketDataProcessor
from .visualization.charts import QuartoChartExporter
from .config.settings import get_settings

__version__ = "2.0.0"
__author__ = "Saurabh Sharma"

# Main exports for easy importing
__all__ = [
    "SparkJobAnalyzer",
    "create_spark_analyzer",
    "JobMarketDataProcessor",
    "QuartoChartExporter",
    "get_settings"
]
