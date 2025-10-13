"""
Consistent Visualization Theme for Job Market Analysis

This module provides a unified theme and styling configuration for all
visualizations in the job market analysis project.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative, sequential
import matplotlib.pyplot as plt
import seaborn as sns

# Import logger for controlled output
from src.utils.logger import get_logger
logger = get_logger(level="WARNING")

class JobMarketTheme:
    """
    Unified theme configuration for job market visualizations.

    Provides consistent colors, fonts, and styling across all charts
    to maintain professional appearance and brand consistency.
    """

    # Color palette
    PRIMARY_COLORS = {
        'blue': '#1f77b4',
        'orange': '#ff7f0e',
        'green': '#2ca02c',
        'red': '#d62728',
        'purple': '#9467bd',
        'brown': '#8c564b',
        'pink': '#e377c2',
        'gray': '#7f7f7f',
        'olive': '#bcbd22',
        'cyan': '#17becf'
    }

    # Extended color palette for categorical data
    CATEGORICAL_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    # Sequential color scales
    SALARY_SCALE = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    EXPERIENCE_SCALE = ['#fff5eb', '#fed7aa', '#fdbe85', '#fdae61', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']

    # Font configuration
    FONT_FAMILY = 'Arial, sans-serif'
    TITLE_FONT_SIZE = 18
    AXIS_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 11

    # Layout configuration
    PLOT_BGCOLOR = 'white'
    PAPER_BGCOLOR = 'white'
    MARGIN = dict(l=50, r=50, t=80, b=50)

    @classmethod
    def get_plotly_layout(cls, title="", width=800, height=600):
        """Get standard Plotly layout configuration."""
        return {
            'title': {
                'text': title,
                'font': {'size': cls.TITLE_FONT_SIZE, 'family': cls.FONT_FAMILY}
            },
            'font': {'family': cls.FONT_FAMILY, 'size': cls.AXIS_FONT_SIZE},
            'plot_bgcolor': cls.PLOT_BGCOLOR,
            'paper_bgcolor': cls.PAPER_BGCOLOR,
            'margin': cls.MARGIN,
            'width': width,
            'height': height,
            'showlegend': True,
            'legend': {
                'font': {'size': cls.LEGEND_FONT_SIZE, 'family': cls.FONT_FAMILY}
            }
        }

    @classmethod
    def get_salary_colors(cls, n_colors=10):
        """Get color palette optimized for salary visualizations."""
        return cls.CATEGORICAL_COLORS[:n_colors]

    @classmethod
    def get_experience_colors(cls, n_colors=5):
        """Get color palette for experience level visualizations."""
        return cls.CATEGORICAL_COLORS[:n_colors]

    @classmethod
    def get_industry_colors(cls, industries):
        """Get color mapping for industry visualizations."""
        colors = {}
        for i, industry in enumerate(industries):
            colors[industry] = cls.CATEGORICAL_COLORS[i % len(cls.CATEGORICAL_COLORS)]
        return colors

    @classmethod
    def apply_salary_theme(cls, fig, title="", chart_type="bar"):
        """Apply consistent theme to a salary-related chart."""
        layout_config = cls.get_plotly_layout(title)

        # Customize based on chart type
        if chart_type == "histogram":
            layout_config['xaxis'] = {
                'title': 'Salary ($)',
                'tickformat': '$,.0f',
                'gridcolor': '#e0e0e0'
            }
            layout_config['yaxis'] = {
                'title': 'Number of Jobs',
                'gridcolor': '#e0e0e0'
            }
        elif chart_type == "box":
            layout_config['yaxis'] = {
                'title': 'Salary ($)',
                'tickformat': '$,.0f',
                'gridcolor': '#e0e0e0'
            }
        elif chart_type == "scatter":
            layout_config['xaxis'] = {'gridcolor': '#e0e0e0'}
            layout_config['yaxis'] = {
                'title': 'Salary ($)',
                'tickformat': '$,.0f',
                'gridcolor': '#e0e0e0'
            }

        fig.update_layout(**layout_config)
        return fig

    @classmethod
    def apply_industry_theme(cls, fig, title="Industry Analysis"):
        """Apply theme to industry-related charts."""
        layout_config = cls.get_plotly_layout(title)
        layout_config['xaxis'] = {
            'tickangle': -45,
            'gridcolor': '#e0e0e0'
        }
        layout_config['yaxis'] = {
            'title': 'Salary ($)',
            'tickformat': '$,.0f',
            'gridcolor': '#e0e0e0'
        }

        fig.update_layout(**layout_config)
        return fig

    @classmethod
    def apply_experience_theme(cls, fig, title="Experience Analysis"):
        """Apply theme to experience-related charts."""
        layout_config = cls.get_plotly_layout(title)
        layout_config['xaxis'] = {
            'title': 'Experience Level',
            'gridcolor': '#e0e0e0'
        }
        layout_config['yaxis'] = {
            'title': 'Salary ($)',
            'tickformat': '$,.0f',
            'gridcolor': '#e0e0e0'
        }

        fig.update_layout(**layout_config)
        return fig

    @classmethod
    def apply_geographic_theme(cls, fig, title="Geographic Analysis"):
        """Apply theme to geographic charts."""
        layout_config = cls.get_plotly_layout(title)
        layout_config['xaxis'] = {
            'tickangle': -45,
            'gridcolor': '#e0e0e0'
        }
        layout_config['yaxis'] = {
            'title': 'Salary ($)',
            'tickformat': '$,.0f',
            'gridcolor': '#e0e0e0'
        }

        fig.update_layout(**layout_config)
        return fig

    @classmethod
    def get_matplotlib_style(cls):
        """Get matplotlib style configuration."""
        return {
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'text.color': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        }

    @classmethod
    def setup_matplotlib(cls):
        """Setup matplotlib with consistent styling."""
        plt.style.use('default')
        for key, value in cls.get_matplotlib_style().items():
            plt.rcParams[key] = value

    @classmethod
    def get_seaborn_style(cls):
        """Get seaborn style configuration."""
        return {
            'style': 'whitegrid',
            'palette': cls.CATEGORICAL_COLORS[:10],
            'font_scale': 1.1,
            'rc': {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            }
        }

    @classmethod
    def setup_seaborn(cls):
        """Setup seaborn with consistent styling."""
        sns.set_style("whitegrid")
        sns.set_palette(cls.CATEGORICAL_COLORS[:10])
        sns.set_context("notebook", font_scale=1.1)

# Convenience functions for common styling tasks
def apply_salary_theme(fig, title="", chart_type="bar"):
    """Apply salary theme to a Plotly figure."""
    return JobMarketTheme.apply_salary_theme(fig, title, chart_type)

def apply_industry_theme(fig, title="Industry Analysis"):
    """Apply industry theme to a Plotly figure."""
    return JobMarketTheme.apply_industry_theme(fig, title)

def apply_experience_theme(fig, title="Experience Analysis"):
    """Apply experience theme to a Plotly figure."""
    return JobMarketTheme.apply_experience_theme(fig, title)

def apply_geographic_theme(fig, title="Geographic Analysis"):
    """Apply geographic theme to a Plotly figure."""
    return JobMarketTheme.apply_geographic_theme(fig, title)

def setup_plotting_environment():
    """Setup the complete plotting environment with consistent styling."""
    JobMarketTheme.setup_matplotlib()
    JobMarketTheme.setup_seaborn()
    logger.info("[OK] Plotting environment configured with JobMarketTheme")
