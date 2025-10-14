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
    PresentationCharts: Charts for presentations and reports

Modules:
    charts: Unified chart generation
    theme: Visualization themes and styling
    ml_charts: Machine learning visualization utilities
"""

from .charts import SalaryVisualizer, QuartoChartExporter, display_figure
from .presentation_charts import PresentationCharts
from .ml_charts import (
    create_ml_performance_comparison,
    create_feature_importance_chart,
    create_predicted_vs_actual_plot,
    create_confusion_matrix_heatmap,
    generate_representative_predictions,
    get_default_ml_results,
    get_default_feature_importance
)

__all__ = [
    "display_figure",
    "SalaryVisualizer",
    "QuartoChartExporter",
    "PresentationCharts",
    # ML visualization functions
    "create_ml_performance_comparison",
    "create_feature_importance_chart",
    "create_predicted_vs_actual_plot",
    "create_confusion_matrix_heatmap",
    "generate_representative_predictions",
    "get_default_ml_results",
    "get_default_feature_importance",
]