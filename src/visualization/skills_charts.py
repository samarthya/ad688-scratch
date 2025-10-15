"""
Skills Visualization Charts

This module provides visualization capabilities for skills analysis,
including geographic skills maps, salary correlation charts, and trend analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(level="WARNING")


class SkillsVisualizer:
    """
    Visualization class for skills analysis.

    Provides comprehensive charting capabilities for skills data,
    including geographic distribution, salary correlation, and trend analysis.
    """

    def __init__(self, skills_data: Dict[str, Any]):
        """
        Initialize skills visualizer with analysis data.

        Args:
            skills_data: Dictionary containing skills analysis results
        """
        self.skills_data = skills_data

    def create_top_skills_chart(self, top_n: int = 15) -> go.Figure:
        """
        Create horizontal bar chart of top skills by frequency.

        Args:
            top_n: Number of top skills to display

        Returns:
            Plotly figure object
        """
        if 'top_skills' not in self.skills_data or self.skills_data['top_skills'].empty:
            return go.Figure()

        df = self.skills_data['top_skills'].head(top_n)

        fig = go.Figure(data=go.Bar(
            y=df['skill'],
            x=df['frequency'],
            orientation='h',
            marker=dict(
                color=df['frequency'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Frequency")
            ),
            text=[f"{freq:,}" for freq in df['frequency']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Frequency: %{x:,}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=df['percentage']
        ))

        fig.update_layout(
            title=f"Top {top_n} Most In-Demand Technical Skills",
            xaxis_title="Number of Job Postings",
            yaxis_title="Technical Skills",
            height=max(400, top_n * 25),
            margin=dict(l=200, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_skills_salary_correlation_chart(self, top_n: int = 20) -> go.Figure:
        """
        Create scatter plot showing skills vs salary correlation.

        Args:
            top_n: Number of top skills to display

        Returns:
            Plotly figure object
        """
        if 'salary_correlation' not in self.skills_data or self.skills_data['salary_correlation'].empty:
            return go.Figure()

        df = self.skills_data['salary_correlation'].head(top_n)

        fig = go.Figure(data=go.Scatter(
            x=df['frequency'],
            y=df['median_salary'],
            mode='markers+text',
            marker=dict(
                size=df['frequency'] / df['frequency'].max() * 50 + 10,
                color=df['median_salary'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Median Salary ($)"),
                line=dict(width=1, color='white')
            ),
            text=df['skill'],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Frequency: %{x:,}<br>Median Salary: $%{y:,.0f}<br>Mean Salary: $%{customdata[0]:,.0f}<extra></extra>',
            customdata=df[['mean_salary', 'salary_std']].values
        ))

        fig.update_layout(
            title="Skills vs Salary Correlation (Size = Frequency)",
            xaxis_title="Job Posting Frequency",
            yaxis_title="Median Salary ($)",
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_geographic_skills_heatmap(self, top_cities: int = 10, top_skills: int = 15) -> go.Figure:
        """
        Create heatmap showing skills demand by city.

        Args:
            top_cities: Number of top cities to include
            top_skills: Number of top skills to include

        Returns:
            Plotly figure object
        """
        if 'geographic_analysis' not in self.skills_data:
            return go.Figure()

        geo_data = self.skills_data['geographic_analysis']

        # Get top cities and skills
        cities = list(geo_data['top_cities'].keys())[:top_cities]

        # Get all unique skills across cities
        all_skills = set()
        for city_data in geo_data['city_skills'].values():
            all_skills.update(city_data['top_skills'].keys())

        # Get top skills overall
        top_skills_list = list(all_skills)[:top_skills]

        # Create matrix
        matrix_data = []
        for city in cities:
            city_skills = geo_data['city_skills'].get(city, {}).get('top_skills', {})
            row = [city_skills.get(skill, 0) for skill in top_skills_list]
            matrix_data.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=top_skills_list,
            y=cities,
            colorscale='Blues',
            hovertemplate='<b>%{y}</b><br>Skill: %{x}<br>Frequency: %{z}<extra></extra>',
            colorbar=dict(title="Frequency")
        ))

        fig.update_layout(
            title="Skills Demand by Geographic Location",
            xaxis_title="Technical Skills",
            yaxis_title="Cities",
            height=max(400, len(cities) * 30),
            margin=dict(l=100, r=50, t=80, b=100),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_industry_skills_chart(self, industry: str, top_n: int = 10) -> go.Figure:
        """
        Create bar chart of top skills for a specific industry.

        Args:
            industry: Industry name
            top_n: Number of top skills to display

        Returns:
            Plotly figure object
        """
        if 'industry_analysis' not in self.skills_data:
            return go.Figure()

        industry_data = self.skills_data['industry_analysis']

        if industry not in industry_data['industry_skills']:
            return go.Figure()

        skills = industry_data['industry_skills'][industry]['top_skills']
        skills_df = pd.DataFrame(list(skills.items()), columns=['skill', 'frequency']).head(top_n)

        fig = go.Figure(data=go.Bar(
            x=skills_df['skill'],
            y=skills_df['frequency'],
            marker=dict(
                color=skills_df['frequency'],
                colorscale='Viridis'
            ),
            text=skills_df['frequency'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Frequency: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Top Skills in {industry} Industry",
            xaxis_title="Technical Skills",
            yaxis_title="Frequency",
            height=500,
            margin=dict(l=50, r=50, t=80, b=100),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(tickangle=45)
        )

        return fig

    def create_emerging_skills_chart(self, top_n: int = 15) -> go.Figure:
        """
        Create chart showing emerging skills with high salary potential.

        Args:
            top_n: Number of emerging skills to display

        Returns:
            Plotly figure object
        """
        if 'emerging_skills' not in self.skills_data or self.skills_data['emerging_skills'].empty:
            return go.Figure()

        df = self.skills_data['emerging_skills'].head(top_n)

        fig = go.Figure(data=go.Scatter(
            x=df['frequency'],
            y=df['salary_premium'],
            mode='markers+text',
            marker=dict(
                size=df['growth_potential'] * 20,
                color=df['median_salary'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Median Salary ($)"),
                line=dict(width=1, color='white')
            ),
            text=df['skill'],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Frequency: %{x}<br>Salary Premium: $%{y:,.0f}<br>Growth Potential: %{customdata:.2f}%<extra></extra>',
            customdata=df['growth_potential']
        ))

        fig.update_layout(
            title="Emerging Skills: High Salary Potential & Growth",
            xaxis_title="Job Posting Frequency",
            yaxis_title="Salary Premium vs Market Average ($)",
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_skills_trends_by_experience_chart(self) -> go.Figure:
        """
        Create chart showing skills trends by experience level.

        Returns:
            Plotly figure object
        """
        if 'experience_trends' not in self.skills_data:
            return go.Figure()

        exp_data = self.skills_data['experience_trends']

        # Create subplots for each experience level
        levels = list(exp_data.keys())
        n_levels = len(levels)

        fig = make_subplots(
            rows=1, cols=n_levels,
            subplot_titles=levels,
            specs=[[{"type": "bar"}] * n_levels]
        )

        for i, level in enumerate(levels):
            level_skills = exp_data[level]['top_skills']
            skills_df = pd.DataFrame(list(level_skills.items()), columns=['skill', 'frequency']).head(5)

            fig.add_trace(
                go.Bar(
                    x=skills_df['frequency'],
                    y=skills_df['skill'],
                    orientation='h',
                    name=level,
                    showlegend=False
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            title="Top Skills by Experience Level",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_skills_dashboard(self) -> go.Figure:
        """
        Create comprehensive skills dashboard with multiple charts.

        Returns:
            Plotly figure object with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Top Skills by Frequency",
                "Skills vs Salary Correlation",
                "Geographic Skills Heatmap",
                "Emerging Skills"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )

        # Top skills chart
        if 'top_skills' in self.skills_data and not self.skills_data['top_skills'].empty:
            top_skills = self.skills_data['top_skills'].head(10)
            fig.add_trace(
                go.Bar(
                    x=top_skills['frequency'],
                    y=top_skills['skill'],
                    orientation='h',
                    name="Top Skills",
                    showlegend=False
                ),
                row=1, col=1
            )

        # Skills vs salary correlation
        if 'salary_correlation' in self.skills_data and not self.skills_data['salary_correlation'].empty:
            salary_data = self.skills_data['salary_correlation'].head(15)
            fig.add_trace(
                go.Scatter(
                    x=salary_data['frequency'],
                    y=salary_data['median_salary'],
                    mode='markers',
                    name="Salary Correlation",
                    showlegend=False,
                    marker=dict(
                        size=8,
                        color=salary_data['median_salary'],
                        colorscale='Viridis'
                    )
                ),
                row=1, col=2
            )

        # Geographic heatmap (simplified)
        if 'geographic_analysis' in self.skills_data:
            geo_data = self.skills_data['geographic_analysis']
            cities = list(geo_data['top_cities'].keys())[:5]
            skills = list(set().union(*[list(geo_data['city_skills'][city]['top_skills'].keys()) for city in cities]))[:5]

            matrix_data = []
            for city in cities:
                city_skills = geo_data['city_skills'].get(city, {}).get('top_skills', {})
                row = [city_skills.get(skill, 0) for skill in skills]
                matrix_data.append(row)

            fig.add_trace(
                go.Heatmap(
                    z=matrix_data,
                    x=skills,
                    y=cities,
                    colorscale='Blues',
                    showscale=False
                ),
                row=2, col=1
            )

        # Emerging skills
        if 'emerging_skills' in self.skills_data and not self.skills_data['emerging_skills'].empty:
            emerging = self.skills_data['emerging_skills'].head(10)
            fig.add_trace(
                go.Scatter(
                    x=emerging['frequency'],
                    y=emerging['salary_premium'],
                    mode='markers',
                    name="Emerging Skills",
                    showlegend=False,
                    marker=dict(
                        size=emerging['growth_potential'] * 20,
                        color=emerging['median_salary'],
                        colorscale='Viridis'
                    )
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="Comprehensive Skills Analysis Dashboard",
            height=800,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig


def create_skills_visualization(skills_data: Dict[str, Any], chart_type: str = "dashboard") -> go.Figure:
    """
    Create skills visualization based on analysis data.

    Args:
        skills_data: Dictionary containing skills analysis results
        chart_type: Type of chart to create

    Returns:
        Plotly figure object
    """
    visualizer = SkillsVisualizer(skills_data)

    if chart_type == "top_skills":
        return visualizer.create_top_skills_chart()
    elif chart_type == "salary_correlation":
        return visualizer.create_skills_salary_correlation_chart()
    elif chart_type == "geographic":
        return visualizer.create_geographic_skills_heatmap()
    elif chart_type == "emerging":
        return visualizer.create_emerging_skills_chart()
    elif chart_type == "experience_trends":
        return visualizer.create_skills_trends_by_experience_chart()
    else:
        return visualizer.create_skills_dashboard()
