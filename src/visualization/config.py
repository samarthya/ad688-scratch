"""
Chart configuration for job market analytics.

This module provides chart styling and configuration options
for consistent visualization across the system.
"""

from typing import Dict, Any


class ChartConfig:
    """
    Chart configuration class.

    Provides styling and configuration options for charts.
    """

    def __init__(self):
        """Initialize with default configuration."""
        self.default_style = {
            'figure_size': (10, 6),
            'color_palette': 'husl',
            'font_size': 12,
            'title_size': 14
        }

    def get_style(self) -> Dict[str, Any]:
        """Get chart style configuration."""
        return self.default_style.copy()
