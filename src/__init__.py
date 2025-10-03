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

# Import main classes for easy access - make PySpark imports optional
try:
    from .core import SparkJobAnalyzer, create_spark_analyzer, JobMarketDataProcessor
    from .ml import SalaryDisparityAnalyzer
    from .utils.spark_utils import create_spark_session
    _PYSPARK_AVAILABLE = True
except ImportError:
    # PySpark not available - define dummy classes or skip
    _PYSPARK_AVAILABLE = False
    SparkJobAnalyzer = None
    create_spark_analyzer = None
    JobMarketDataProcessor = None
    SalaryDisparityAnalyzer = None
    create_spark_session = None

from .visualization.charts import SalaryVisualizer, QuartoChartExporter

__all__ = [
    "SalaryVisualizer",
    "QuartoChartExporter",
]

# Add PySpark classes only if available
if _PYSPARK_AVAILABLE:
    __all__.extend([
        "SparkJobAnalyzer",
        "create_spark_analyzer",
        "JobMarketDataProcessor",
        "SalaryDisparityAnalyzer",
        "create_spark_session",
    ])