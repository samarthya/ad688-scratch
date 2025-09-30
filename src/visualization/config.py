"""
Chart configuration and styling for job market analysis.

Provides centralized configuration for all chart types with consistent
styling and formatting across the visualization system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.settings import get_settings


@dataclass
class ChartConfig:
    """Centralized chart configuration and styling."""

    # Color schemes
    color_palettes = {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'salary': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        'disparity': ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#3498DB'],
        'industry': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    }

    # Chart dimensions
    default_width: int = 800
    default_height: int = 600

    # Font settings
    title_font_size: int = 16
    axis_font_size: int = 12
    legend_font_size: int = 11

    # Layout settings
    margin_settings = {
        'l': 50, 'r': 50, 't': 80, 'b': 50
    }

    def get_color_palette(self, palette_name: str = 'primary') -> List[str]:
        """Get color palette by name."""
        return self.color_palettes.get(palette_name, self.color_palettes['primary'])

    def create_readable_bar_chart(self, data, x_col: str, y_col: str,
                                title: str, color_col: str = None) -> go.Figure:
        """Create a readable bar chart with consistent styling."""
        try:
            # Create bar chart
            if color_col:
                fig = px.bar(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=title,
                    color_continuous_scale='Viridis'
                )
            else:
                fig = px.bar(
                    data,
                    x=x_col,
                    y=y_col,
                    title=title,
                    color=y_col,
                    color_continuous_scale='Blues'
                )

            # Update layout for readability
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': self.title_font_size}
                },
                xaxis_title=x_col,
                yaxis_title=y_col,
                width=self.default_width,
                height=self.default_height,
                margin=self.margin_settings,
                font=dict(size=self.axis_font_size),
                showlegend=True
            )

            # Rotate x-axis labels if needed
            if len(data[x_col].unique()) > 5:
                fig.update_xaxes(tickangle=45)

            return fig

        except Exception as e:
            raise ValueError(f"Failed to create bar chart: {str(e)}")

    def create_salary_disparity_chart(self, data, x_col: str, y_col: str,
                                    title: str) -> go.Figure:
        """Create salary disparity chart with annotations."""
        try:
            # Create base chart
            fig = self.create_readable_bar_chart(data, x_col, y_col, title)

            # Add disparity annotations
            if len(data) >= 2:
                min_salary = data[y_col].min()
                max_salary = data[y_col].max()
                disparity_ratio = max_salary / min_salary if min_salary > 0 else 0

                # Add disparity ratio annotation
                fig.add_annotation(
                    text=f"<b>Salary Gap:</b> {disparity_ratio:.1f}x difference",
                    xref="paper", yref="paper",
                    x=0.5, y=0.95,
                    showarrow=False,
                    font=dict(size=12, color="red"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1
                )

                # Add salary range annotation
                fig.add_annotation(
                    text=f"<b>Range:</b> ${min_salary:,.0f} - ${max_salary:,.0f}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.88,
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )

            return fig

        except Exception as e:
            raise ValueError(f"Failed to create salary disparity chart: {str(e)}")

    def create_comparison_chart(self, data, x_col: str, y_col: str,
                              group_col: str, title: str) -> go.Figure:
        """Create comparison chart for grouped data."""
        try:
            # Create grouped bar chart
            fig = px.bar(
                data,
                x=x_col,
                y=y_col,
                color=group_col,
                title=title,
                barmode='group',
                color_discrete_sequence=self.get_color_palette('salary')
            )

            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': self.title_font_size}
                },
                xaxis_title=x_col,
                yaxis_title=y_col,
                width=self.default_width,
                height=self.default_height,
                margin=self.margin_settings,
                font=dict(size=self.axis_font_size),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            return fig

        except Exception as e:
            raise ValueError(f"Failed to create comparison chart: {str(e)}")

    def configure_matplotlib_style(self):
        """Configure matplotlib style for consistent plots."""
        plt.style.use('default')
        sns.set_palette("husl")

        # Set default figure size
        plt.rcParams['figure.figsize'] = (self.default_width/100, self.default_height/100)
        plt.rcParams['font.size'] = self.axis_font_size
        plt.rcParams['axes.titlesize'] = self.title_font_size
        plt.rcParams['axes.labelsize'] = self.axis_font_size
        plt.rcParams['legend.fontsize'] = self.legend_font_size


# Global chart configuration instance
chart_config = ChartConfig()
