"""
Unified Chart Generation for Job Market Analysis

This module provides comprehensive chart generation capabilities,
consolidating functionality from multiple visualization modules.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional, Union

from src.config.settings import get_settings
from src.visualization.config import ChartConfig
from src.core.exceptions import VisualizationError

# Configure matplotlib
plt.style.use('default')
sns.set_palette("husl")


class QuartoChartExporter:
    """
    Unified chart generation for Quarto integration.

    Provides comprehensive chart generation with multiple export formats
    and consistent styling across all visualizations.
    """

    def __init__(self, output_dir: str = None):
        """Initialize chart exporter with configuration."""
        self.settings = get_settings()
        self.output_dir = Path(output_dir or self.settings.figures_path)
        self.output_dir.mkdir(exist_ok=True)
        self.chart_registry = []
        self.chart_config = ChartConfig()

    def create_experience_salary_chart(self, data: pd.DataFrame,
                                     title: str = "Salary by Experience Level") -> Dict[str, str]:
        """Create experience-based salary analysis chart."""
        try:
            # Create Plotly bar chart
            fig = px.bar(
                data,
                x='experience_level',
                y='median_salary',
                title=title,
                color='median_salary',
                color_continuous_scale='Viridis'
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Experience Level",
                yaxis_title="Median Salary ($)",
                title_x=0.5,
                width=self.settings.chart_width,
                height=self.settings.chart_height
            )

            # Add disparity annotation
            if len(data) >= 2:
                min_salary = data['median_salary'].min()
                max_salary = data['median_salary'].max()
                disparity_ratio = max_salary / min_salary if min_salary > 0 else 0

                fig.add_annotation(
                    text=f"<b>Experience Gap:</b> {disparity_ratio:.1f}x salary difference",
                    xref="paper", yref="paper",
                    x=0.5, y=0.95,
                    showarrow=False,
                    font=dict(size=12, color="red")
                )

            # Export chart
            chart_files = self._export_chart(fig, "experience_salary_analysis", title)
            return chart_files

        except Exception as e:
            raise VisualizationError(f"Failed to create experience salary chart: {str(e)}")

    def create_industry_salary_chart(self, data: pd.DataFrame,
                                   title: str = "Salary by Industry") -> Dict[str, str]:
        """Create industry-based salary analysis chart."""
        try:
            # Create horizontal bar chart for better readability
            fig = px.bar(
                data,
                x='median_salary',
                y='industry',
                orientation='h',
                title=title,
                color='median_salary',
                color_continuous_scale='Blues'
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Median Salary ($)",
                yaxis_title="Industry",
                title_x=0.5,
                width=self.settings.chart_width,
                height=self.settings.chart_height
            )

            # Export chart
            chart_files = self._export_chart(fig, "industry_salary_analysis", title)
            return chart_files

        except Exception as e:
            raise VisualizationError(f"Failed to create industry salary chart: {str(e)}")

    def create_geographic_salary_chart(self, data: pd.DataFrame,
                                     title: str = "Salary by Location") -> Dict[str, str]:
        """Create geographic salary analysis chart."""
        try:
            # Create bar chart with location data
            fig = px.bar(
                data,
                x='state',
                y='median_salary',
                title=title,
                color='median_salary',
                color_continuous_scale='Reds'
            )

            # Update layout
            fig.update_layout(
                xaxis_title="State",
                yaxis_title="Median Salary ($)",
                title_x=0.5,
                width=self.settings.chart_width,
                height=self.settings.chart_height,
                xaxis_tickangle=-45
            )

            # Export chart
            chart_files = self._export_chart(fig, "geographic_salary_analysis", title)
            return chart_files

        except Exception as e:
            raise VisualizationError(f"Failed to create geographic salary chart: {str(e)}")

    def create_ai_analysis_chart(self, data: pd.DataFrame,
                               title: str = "AI vs Traditional Roles") -> Dict[str, str]:
        """Create AI/ML role analysis chart."""
        try:
            # Create grouped bar chart
            fig = px.bar(
                data,
                x='ai_related',
                y='median_salary',
                title=title,
                color='ai_related',
                color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'}
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Role Type",
                yaxis_title="Median Salary ($)",
                title_x=0.5,
                width=self.settings.chart_width,
                height=self.settings.chart_height
            )

            # Add AI premium annotation
            if len(data) >= 2:
                ai_salary = data[data['ai_related'] == True]['median_salary'].iloc[0] if True in data['ai_related'].values else 0
                traditional_salary = data[data['ai_related'] == False]['median_salary'].iloc[0] if False in data['ai_related'].values else 0

                if ai_salary > 0 and traditional_salary > 0:
                    premium = ((ai_salary - traditional_salary) / traditional_salary) * 100
                    fig.add_annotation(
                        text=f"<b>AI Premium:</b> {premium:.1f}% higher salary",
                        xref="paper", yref="paper",
                        x=0.5, y=0.95,
                        showarrow=False,
                        font=dict(size=12, color="green")
                    )

            # Export chart
            chart_files = self._export_chart(fig, "ai_analysis", title)
            return chart_files

        except Exception as e:
            raise VisualizationError(f"Failed to create AI analysis chart: {str(e)}")

    def create_remote_work_chart(self, data: pd.DataFrame,
                               title: str = "Remote Work Impact") -> Dict[str, str]:
        """Create remote work analysis chart."""
        try:
            # Create bar chart
            fig = px.bar(
                data,
                x='remote_allowed',
                y='median_salary',
                title=title,
                color='remote_allowed',
                color_discrete_map={True: '#95E1D3', False: '#F38BA8'}
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Remote Work Allowed",
                yaxis_title="Median Salary ($)",
                title_x=0.5,
                width=self.settings.chart_width,
                height=self.settings.chart_height
            )

            # Export chart
            chart_files = self._export_chart(fig, "remote_work_analysis", title)
            return chart_files

        except Exception as e:
            raise VisualizationError(f"Failed to create remote work chart: {str(e)}")

    def _export_chart(self, fig, chart_name: str, title: str) -> Dict[str, str]:
        """Export chart in multiple formats."""
        try:
            # Generate file paths
            html_path = self.output_dir / f"{chart_name}.html"
            png_path = self.output_dir / f"{chart_name}.png"
            svg_path = self.output_dir / f"{chart_name}.svg"

            # Export HTML (interactive)
            fig.write_html(str(html_path))

            # Export PNG (static)
            fig.write_image(str(png_path), width=self.settings.chart_width, height=self.settings.chart_height)

            # Export SVG (vector)
            fig.write_image(str(svg_path), format="svg", width=self.settings.chart_width, height=self.settings.chart_height)

            # Register chart
            chart_info = {
                "name": chart_name,
                "title": title,
                "type": "plotly",
                "files": {
                    "html": str(html_path),
                    "png": str(png_path),
                    "svg": str(svg_path)
                }
            }

            self.chart_registry.append(chart_info)

            return chart_info

        except Exception as e:
            raise VisualizationError(f"Failed to export chart {chart_name}: {str(e)}")

    def export_chart_registry(self) -> str:
        """Export chart registry as JSON."""
        try:
            registry_path = self.output_dir / "chart_registry.json"

            registry_data = {
                "charts": self.chart_registry,
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "total_charts": len(self.chart_registry)
            }

            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)

            return str(registry_path)

        except Exception as e:
            raise VisualizationError(f"Failed to export chart registry: {str(e)}")


class SalaryVisualizer:
    """
    Pandas-based visualization utilities for backward compatibility.

    Provides simple visualization functions for quick analysis and
    development workflows.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with pandas DataFrame."""
        self.df = df

    def get_industry_salary_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Get industry salary analysis."""
        try:
            industry_stats = self.df.groupby('industry')['salary_avg_imputed'].agg([
                'count', 'mean', 'median', 'min', 'max', 'std'
            ]).round(0)

            industry_stats.columns = ['job_count', 'avg_salary', 'median_salary', 'min_salary', 'max_salary', 'std_salary']
            industry_stats = industry_stats.sort_values('median_salary', ascending=False).head(top_n)

            return industry_stats

        except Exception as e:
            raise VisualizationError(f"Failed to generate industry analysis: {str(e)}")

    def analyze_experience_salary_progression(self) -> pd.DataFrame:
        """Analyze salary progression by experience level."""
        try:
            exp_stats = self.df.groupby('experience_level')['salary_avg_imputed'].agg([
                'count', 'mean', 'median', 'min', 'max'
            ]).round(0)

            exp_stats.columns = ['job_count', 'avg_salary', 'median_salary', 'min_salary', 'max_salary']
            exp_stats = exp_stats.sort_values('median_salary', ascending=False)

            return exp_stats

        except Exception as e:
            raise VisualizationError(f"Failed to generate experience analysis: {str(e)}")

    def get_location_salary_analysis(self, top_n: int = 15) -> pd.DataFrame:
        """Get location-based salary analysis."""
        try:
            # Extract state from location
            self.df['state'] = self.df['location'].str.split(',').str[1].str.strip()

            location_stats = self.df.groupby('state')['salary_avg_imputed'].agg([
                'count', 'mean', 'median'
            ]).round(0)

            location_stats.columns = ['job_count', 'avg_salary', 'median_salary']
            location_stats = location_stats.sort_values('median_salary', ascending=False).head(top_n)

            return location_stats

        except Exception as e:
            raise VisualizationError(f"Failed to generate location analysis: {str(e)}")

    def calculate_ai_skill_premiums(self) -> pd.DataFrame:
        """Calculate AI/ML skill premiums."""
        try:
            ai_stats = self.df.groupby('ai_related')['salary_avg_imputed'].agg([
                'count', 'mean', 'median'
            ]).round(0)

            ai_stats.columns = ['job_count', 'avg_salary', 'median_salary']

            # Calculate premium
            if len(ai_stats) >= 2:
                ai_salary = ai_stats.loc[True, 'median_salary'] if True in ai_stats.index else 0
                traditional_salary = ai_stats.loc[False, 'median_salary'] if False in ai_stats.index else 0

                if ai_salary > 0 and traditional_salary > 0:
                    premium = ((ai_salary - traditional_salary) / traditional_salary) * 100
                    ai_stats['premium_percentage'] = [premium if idx else 0 for idx in ai_stats.index]

            return ai_stats

        except Exception as e:
            raise VisualizationError(f"Failed to calculate AI skill premiums: {str(e)}")
