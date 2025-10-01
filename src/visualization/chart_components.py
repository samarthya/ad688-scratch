"""
Reusable chart components for consistent visualization across QMD files.

This module provides standardized chart components that ensure consistent
styling and formatting across all analysis reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class ChartComponents:
    """
    Reusable chart components with consistent styling.

    Provides standardized charts for salary analysis with consistent
    colors, fonts, and formatting across all reports.
    """

    def __init__(self):
        """Initialize chart components with default styling."""
        self.setup_style()

    def setup_style(self):
        """Set up consistent styling for all charts."""
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

        # Set consistent color scheme
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Pink
            'accent': '#F18F01',       # Orange
            'success': '#C73E1D',      # Red
            'info': '#6A994E',         # Green
            'warning': '#F77F00',      # Yellow
            'light': '#F8F9FA',        # Light gray
            'dark': '#212529'          # Dark gray
        }

        # Set plotly color scheme
        self.plotly_colors = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
            '#6A994E', '#F77F00', '#8E44AD', '#E67E22'
        ]

    def create_salary_distribution_chart(self, df: pd.DataFrame, title: str = "Salary Distribution") -> go.Figure:
        """Create a comprehensive salary distribution chart."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Salary Distribution', 'Salary by Experience Level',
                          'Salary by Education Level', 'Salary by Region'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Ensure we have the required columns
        if 'SALARY_AVG' not in df.columns:
            raise ValueError("SALARY_AVG column not found in dataset")

        # 1. Salary histogram
        fig.add_trace(
            go.Histogram(x=df['SALARY_AVG'], nbinsx=30, name='Salary Distribution',
                        marker_color=self.plotly_colors[0], opacity=0.7),
            row=1, col=1
        )

        # 2. Salary by experience level
        if 'experience_level' in df.columns and df['experience_level'].notna().sum() > 0:
            exp_data = df.groupby('experience_level')['SALARY_AVG'].median().sort_values(ascending=False)
            if len(exp_data) > 0:
                fig.add_trace(
                    go.Bar(x=exp_data.index, y=exp_data.values, name='Experience Level',
                          marker_color=self.plotly_colors[1]),
                    row=1, col=2
                )

        # 3. Salary by education level
        if 'education_level' in df.columns and df['education_level'].notna().sum() > 0:
            edu_data = df.groupby('education_level')['SALARY_AVG'].median().sort_values(ascending=False)
            if len(edu_data) > 0:
                fig.add_trace(
                    go.Bar(x=edu_data.index, y=edu_data.values, name='Education Level',
                          marker_color=self.plotly_colors[2]),
                    row=2, col=1
                )

        # 4. Salary by region
        if 'location_readable' in df.columns and df['location_readable'].notna().sum() > 0:
            region_data = df.groupby('location_readable')['SALARY_AVG'].median().sort_values(ascending=False)
            if len(region_data) > 0:
                fig.add_trace(
                    go.Bar(x=region_data.index, y=region_data.values, name='Region',
                          marker_color=self.plotly_colors[3]),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            height=600,
            template='plotly_white',
            font=dict(size=12)
        )

        # Update axes
        fig.update_xaxes(title_text="Salary ($)", row=1, col=1)
        fig.update_xaxes(title_text="Experience Level", row=1, col=2)
        fig.update_xaxes(title_text="Education Level", row=2, col=1)
        fig.update_xaxes(title_text="Region", row=2, col=2)

        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Median Salary ($)", row=1, col=2)
        fig.update_yaxes(title_text="Median Salary ($)", row=2, col=1)
        fig.update_yaxes(title_text="Median Salary ($)", row=2, col=2)

        return fig

    def create_salary_comparison_table(self, data: Dict[str, Any], title: str = "Salary Analysis") -> pd.DataFrame:
        """Create a formatted comparison table."""
        if isinstance(data, dict) and 'analysis' in data:
            df = data['analysis'].copy()
        else:
            df = pd.DataFrame(data)

        # Format numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'salary' in col.lower() or 'wage' in col.lower():
                df[col] = df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
            elif 'count' in col.lower() or 'jobs' in col.lower():
                df[col] = df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            elif 'premium' in col.lower() or 'percent' in col.lower():
                df[col] = df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")

        return df

    def create_skills_analysis_chart(self, skills_data: Dict[str, Any], title: str = "Skills Premium Analysis") -> go.Figure:
        """Create skills premium analysis chart."""
        if 'analysis' not in skills_data:
            raise ValueError("Skills data must contain 'analysis' key")

        df = skills_data['analysis']

        fig = go.Figure()

        # Create horizontal bar chart
        fig.add_trace(go.Bar(
            y=df['Skill'],
            x=df['Premium %'],
            orientation='h',
            marker=dict(
                color=df['Premium %'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Premium %")
            ),
            text=[f"${row['Median Salary']:,.0f}<br>{row['Job Count']:,} jobs"
                  for _, row in df.iterrows()],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>" +
                         "Premium: %{x:+.1f}%<br>" +
                         "Median Salary: $%{customdata[0]:,.0f}<br>" +
                         "Job Count: %{customdata[1]:,}<extra></extra>",
            customdata=df[['Median Salary', 'Job Count']].values
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Salary Premium (%)",
            yaxis_title="Skill Category",
            height=max(400, len(df) * 40),
            template='plotly_white',
            font=dict(size=12)
        )

        return fig

    def create_education_roi_chart(self, roi_data: Dict[str, Any], title: str = "Education ROI Analysis") -> go.Figure:
        """Create education ROI analysis chart."""
        if 'analysis' not in roi_data:
            raise ValueError("ROI data must contain 'analysis' key")

        df = roi_data['analysis']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Investment vs Lifetime Value', 'Break-even Period'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Investment vs Lifetime Value
        fig.add_trace(
            go.Bar(x=df['Education Level'], y=df['Investment'], name='Investment',
                  marker_color=self.plotly_colors[0], opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=df['Education Level'], y=df['Lifetime Value'], name='Lifetime Value',
                  marker_color=self.plotly_colors[1], opacity=0.7),
            row=1, col=1
        )

        # Break-even period
        fig.add_trace(
            go.Bar(x=df['Education Level'], y=df['Break-even (years)'], name='Break-even (years)',
                  marker_color=self.plotly_colors[2]),
            row=1, col=2
        )

        fig.update_layout(
            title=title,
            height=500,
            template='plotly_white',
            font=dict(size=12)
        )

        fig.update_xaxes(title_text="Education Level", row=1, col=1)
        fig.update_xaxes(title_text="Education Level", row=1, col=2)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
        fig.update_yaxes(title_text="Years", row=1, col=2)

        return fig

    def create_geographic_analysis_chart(self, geo_data: pd.DataFrame, title: str = "Geographic Salary Analysis") -> go.Figure:
        """Create geographic salary analysis chart."""
        fig = go.Figure()

        # Create bubble chart: x=job_count, y=median_salary, size=job_count
        fig.add_trace(go.Scatter(
            x=geo_data['Job Count'],
            y=geo_data['Median Salary'],
            mode='markers',
            marker=dict(
                size=geo_data['Job Count'] / 10,  # Scale bubble size
                color=geo_data['Median Salary'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Median Salary ($)"),
                line=dict(width=2, color='white')
            ),
            text=geo_data['Location'],
            hovertemplate="<b>%{text}</b><br>" +
                         "Jobs: %{x:,}<br>" +
                         "Median Salary: $%{y:,.0f}<br>" +
                         "Mean Salary: $%{customdata:,.0f}<extra></extra>",
            customdata=geo_data['Mean Salary']
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Number of Jobs",
            yaxis_title="Median Salary ($)",
            height=500,
            template='plotly_white',
            font=dict(size=12)
        )

        return fig

    def create_summary_stats_cards(self, stats: Dict[str, Any]) -> str:
        """Create summary statistics cards in HTML format."""
        cards_html = """
        <div class="row">
        """

        # Define key metrics to display
        key_metrics = [
            ('Total Jobs', 'total_records', '{:,}'),
            ('Salary Coverage', 'salary_coverage', '{:.1f}%'),
            ('Unique Companies', 'unique_companies', '{:,}'),
            ('Unique Locations', 'unique_locations', '{:,}'),
            ('Education Levels', 'education_levels', '{:,}'),
            ('Experience Levels', 'experience_levels', '{:,}')
        ]

        for i, (label, key, format_str) in enumerate(key_metrics):
            if key in stats:
                value = format_str.format(stats[key])
                color = self.plotly_colors[i % len(self.plotly_colors)]

                cards_html += f"""
                <div class="col-md-4 mb-3">
                    <div class="card" style="border-left: 4px solid {color};">
                        <div class="card-body">
                            <h5 class="card-title text-muted">{label}</h5>
                            <h3 class="card-text" style="color: {color};">{value}</h3>
                        </div>
                    </div>
                </div>
                """

        cards_html += "</div>"
        return cards_html


# Global instance for easy access
chart_components = ChartComponents()
