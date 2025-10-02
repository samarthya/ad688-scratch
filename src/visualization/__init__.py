"""
Visualization Module

This module provides comprehensive visualization capabilities for job market
data analysis, including interactive charts, static plots, and Quarto-compatible
exports.

Classes:
    QuartoChartExporter: Main chart generation for Quarto integration
    SalaryDisparityChartConfig: Configuration for salary-focused charts

Modules:
    plots: General plotting utilities
    simple_plots: Basic chart generation
    chart_config: Chart styling and configuration
    quarto_charts: Quarto-specific chart exports
"""

from .charts import SalaryVisualizer, QuartoChartExporter

__all__ = [
    "SalaryVisualizer",
    "QuartoChartExporter",
]