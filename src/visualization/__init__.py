"""
Visualization Module

This module provides comprehensive visualization capabilities for job market
data analysis, including interactive charts, static plots, and Quarto-compatible
exports.

Functions:
    display_figure: Utility function for displaying and saving Plotly figures

Classes:
    SalaryVisualizer: Unified salary visualization class
    QuartoChartExporter: Main chart generation for Quarto integration

Modules:
    charts: Unified chart generation
    theme: Visualization themes and styling
"""

from .charts import SalaryVisualizer, QuartoChartExporter, display_figure

__all__ = [
    "display_figure",
    "SalaryVisualizer",
    "QuartoChartExporter",
]