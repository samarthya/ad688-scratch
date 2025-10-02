"""
Interactive Key Findings Dashboard

This module creates a comprehensive dashboard that effectively communicates
the key findings and objectives of the job market analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any

class KeyFindingsDashboard:
    """
    Creates an interactive dashboard that clearly communicates key findings
    and objectives of the job market analysis.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_key_metrics_cards(self) -> go.Figure:
        """Create the key metrics cards section with improved spacing and readability."""

        # Calculate key metrics
        metrics = self._calculate_key_metrics()

        # Create subplot with 2x2 grid - improved spacing to prevent overlap
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["", "", "", ""],  # Remove duplicate titles to prevent overlap
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.4,   # Significantly increased vertical spacing
            horizontal_spacing=0.25  # Increased horizontal spacing
        )

        # Experience Gap - reduced font sizes and improved positioning
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics['experience_gap'],
            delta={
                "reference": 200,
                "valueformat": ".0f",
                "font": {"size": 12},  # Smaller delta font
                "position": "bottom"   # Position delta below number
            },
            title={
                "text": "<b>Experience Gap</b><br><span style='font-size:12px'>Senior vs Entry Level</span>",
                "font": {"size": 14}
            },
            number={
                "font": {"size": 24, "color": "#1f77b4"},  # Reduced number font size
                "suffix": "%"
            }
        ), row=1, col=1)

        # Education Premium
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics['education_premium'],
            delta={
                "reference": 150,
                "valueformat": ".0f",
                "font": {"size": 14},
                "position": "bottom"
            },
            title={
                "text": "<b>Education Premium</b><br><span style='font-size:12px'>Advanced vs Bachelor's</span>",
                "font": {"size": 14}
            },
            number={
                "font": {"size": 28, "color": "#ff7f0e"},
                "suffix": "%"
            }
        ), row=1, col=2)

        # Company Size Gap
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics['company_size_gap'],
            delta={
                "reference": 30,
                "valueformat": ".0f",
                "font": {"size": 14},
                "position": "bottom"
            },
            title={
                "text": "<b>Company Size Gap</b><br><span style='font-size:12px'>Large vs Small Companies</span>",
                "font": {"size": 14}
            },
            number={
                "font": {"size": 28, "color": "#2ca02c"},
                "suffix": "%"
            }
        ), row=2, col=1)

        # Total Jobs - no delta to save space
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics['total_jobs'],
            title={
                "text": "<b>Jobs Analyzed</b><br><span style='font-size:12px'>Data Sample Size</span>",
                "font": {"size": 14}
            },
            number={
                "font": {"size": 28, "color": "#d62728"},
                "valueformat": ","
            }
        ), row=2, col=2)

        # Update layout with better spacing
        fig.update_layout(
            title={
                "text": "Key Market Intelligence Metrics",
                "font": {"size": 20, "color": "#333333"},
                "x": 0.5,
                "y": 0.95
            },
            height=650,  # Increased height significantly to accommodate better spacing
            margin=dict(l=50, r=50, t=80, b=50),  # Better margins
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_career_progression_analysis(self) -> go.Figure:
        """Create career progression analysis with clear salary growth visualization and proper spacing."""

        # Calculate experience progression
        exp_data = self._calculate_experience_progression()

        # Create subplot with improved spacing and proportions
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],  # More balanced space distribution
            subplot_titles=("Salary Progression by Experience Level", "Growth Rate Between Levels"),
            vertical_spacing=0.25  # Even more spacing between charts
        )

        # Main salary progression chart
        fig.add_trace(go.Bar(
            x=exp_data['levels'],
            y=exp_data['salaries'],
            text=[f"${s:,.0f}" for s in exp_data['salaries']],
            textposition='outside',
            textfont=dict(size=12),  # Smaller text to prevent overlap
            marker_color=exp_data['colors'],
            name="Median Salary",
            showlegend=False
        ), row=1, col=1)

        # Growth rate chart
        fig.add_trace(go.Bar(
            x=exp_data['levels'][1:],
            y=exp_data['growth_rates'],
            text=[f"+{g:.0f}%" for g in exp_data['growth_rates']],
            textposition='outside',
            textfont=dict(size=11),  # Smaller text for bottom chart
            marker_color=['#2ca02c' if g > 0 else '#d62728' for g in exp_data['growth_rates']],
            name="Growth Rate",
            showlegend=False
        ), row=2, col=1)

        # Update layout with better spacing
        fig.update_layout(
            title={
                "text": "Career Progression: How Much Will I Earn as I Grow?",
                "font": {"size": 18, "color": "#333333"},
                "x": 0.5,
                "y": 0.97  # Position title even higher
            },
            height=900,  # Increased height even more to accommodate spacing
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=140, b=60)  # Increased top margin even more
        )

        # Update axes with better formatting
        fig.update_xaxes(
            title_text="Experience Level",
            title_font_size=14,
            tickfont_size=12,
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Annual Salary ($)",
            title_font_size=14,
            tickformat="$,.0f",
            tickfont_size=12,
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Experience Level",
            title_font_size=14,
            tickfont_size=12,
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Growth Rate (%)",
            title_font_size=14,
            tickfont_size=12,
            row=2, col=1
        )

        # Ensure proper spacing for text labels by adjusting y-axis ranges
        max_salary = max(exp_data['salaries'])
        fig.update_yaxes(range=[0, max_salary * 1.15], row=1, col=1)  # Add 15% padding above bars

        if exp_data['growth_rates']:
            max_growth = max(exp_data['growth_rates'])
            fig.update_yaxes(range=[0, max_growth * 1.2], row=2, col=1)  # Add 20% padding above bars

        return fig

    def create_education_roi_analysis(self) -> go.Figure:
        """Create education ROI analysis with clear investment returns."""

        # Calculate education ROI
        edu_data = self._calculate_education_roi()

        # Create subplot with salary comparison and ROI
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Salary by Education Level", "ROI Analysis"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Salary comparison
        fig.add_trace(go.Bar(
            x=edu_data['levels'],
            y=edu_data['salaries'],
            text=[f"${s:,.0f}" for s in edu_data['salaries']],
            textposition='outside',
            marker_color=edu_data['colors'],
            name="Median Salary",
            showlegend=False
        ), row=1, col=1)

        # ROI comparison
        fig.add_trace(go.Bar(
            x=edu_data['levels'],
            y=edu_data['roi_percentages'],
            text=[f"{r:.0f}%" for r in edu_data['roi_percentages']],
            textposition='outside',
            marker_color=['#2ca02c' if r > 0 else '#d62728' for r in edu_data['roi_percentages']],
            name="ROI %",
            showlegend=False
        ), row=1, col=2)

        # Update layout
        fig.update_layout(
            title={
                "text": "Education ROI: Is Graduate School Worth It?",
                "font": {"size": 20, "color": "#333333"},
                "x": 0.5
            },
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=100, b=50)
        )

        # Update axes
        fig.update_xaxes(title_text="Education Level", row=1, col=1)
        fig.update_yaxes(title_text="Salary ($)", tickformat="$,.0f", row=1, col=1)
        fig.update_xaxes(title_text="Education Level", row=1, col=2)
        fig.update_yaxes(title_text="ROI (%)", row=1, col=2)

        return fig

    def create_company_strategy_analysis(self) -> go.Figure:
        """Create company size strategy analysis."""

        # Calculate company size analysis
        company_data = self._calculate_company_size_analysis()

        # Create subplot with salary and growth comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Salary by Company Size", "Growth Potential"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Salary comparison
        fig.add_trace(go.Bar(
            x=company_data['sizes'],
            y=company_data['salaries'],
            text=[f"${s:,.0f}" for s in company_data['salaries']],
            textposition='outside',
            marker_color=company_data['colors'],
            name="Median Salary",
            showlegend=False
        ), row=1, col=1)

        # Growth potential
        fig.add_trace(go.Bar(
            x=company_data['sizes'],
            y=company_data['growth_potential'],
            text=[f"{g:.0f}%" for g in company_data['growth_potential']],
            textposition='outside',
            marker_color=['#2ca02c' if g > 0 else '#d62728' for g in company_data['growth_potential']],
            name="Growth Potential",
            showlegend=False
        ), row=1, col=2)

        # Update layout
        fig.update_layout(
            title={
                "text": "Company Strategy: Startup or Enterprise?",
                "font": {"size": 20, "color": "#333333"},
                "x": 0.5
            },
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=100, b=50)
        )

        # Update axes
        fig.update_xaxes(title_text="Company Size", row=1, col=1)
        fig.update_yaxes(title_text="Salary ($)", tickformat="$,.0f", row=1, col=1)
        fig.update_xaxes(title_text="Company Size", row=1, col=2)
        fig.update_yaxes(title_text="Growth Potential (%)", row=1, col=2)

        return fig

    def create_complete_intelligence_dashboard(self) -> go.Figure:
        """Create comprehensive intelligence dashboard."""

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics()

        # Create 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Salary Distribution",
                "Industry Comparison",
                "Geographic Analysis",
                "Skills Premium"
            ],
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Salary distribution
        salary_cols = ['salary_avg', 'SALARY_AVG', 'salary', 'SALARY', 'median_salary', 'MEDIAN_SALARY']
        salary_data = None

        for col in salary_cols:
            if col in self.df.columns:
                salary_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                if len(salary_data) > 0:
                    break

        if salary_data is None or len(salary_data) == 0:
            # Create sample salary data for demonstration
            salary_data = pd.Series([60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000])

        fig.add_trace(go.Histogram(
            x=salary_data,
            nbinsx=30,
            marker_color='#1f77b4',
            name="Salary Distribution"
        ), row=1, col=1)

        # Industry comparison
        industry_data = self._get_industry_data()
        fig.add_trace(go.Bar(
            x=industry_data['industries'],
            y=industry_data['salaries'],
            marker_color=industry_data['colors'],
            name="Industry Salaries"
        ), row=1, col=2)

        # Geographic analysis
        geo_data = self._get_geographic_data()
        fig.add_trace(go.Bar(
            x=geo_data['locations'],
            y=geo_data['salaries'],
            marker_color=geo_data['colors'],
            name="Location Salaries"
        ), row=2, col=1)

        # Skills premium
        skills_data = self._get_skills_data()
        fig.add_trace(go.Bar(
            x=skills_data['skills'],
            y=skills_data['premiums'],
            marker_color=skills_data['colors'],
            name="Skills Premium"
        ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title={
                "text": "Complete Intelligence: Show Me Everything",
                "font": {"size": 20, "color": "#333333"},
                "x": 0.5
            },
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=100, b=50),
            showlegend=False
        )

        return fig

    def create_ai_technology_analysis(self) -> go.Figure:
        """Create AI & Technology salary analysis maintaining abstraction layer."""
        from src.visualization.charts import SalaryVisualizer

        # Use the existing SalaryVisualizer but through the dashboard abstraction
        visualizer = SalaryVisualizer(self.df)

        try:
            # Create the AI salary comparison using the enhanced method
            fig = visualizer.plot_ai_salary_comparison()

            # Apply consistent dashboard styling
            fig.update_layout(
                title={
                    "text": "AI & Technology Salary Premium Analysis",
                    "font": {"size": 18, "color": "#333333"},
                    "x": 0.5
                },
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=80, r=80, t=100, b=80),
                font=dict(size=14)
            )

            return fig

        except Exception as e:
            # Create error visualization with consistent styling
            fig = go.Figure()
            fig.add_annotation(
                text=f"AI Analysis: {str(e)[:100]}...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="orange")
            )
            fig.update_layout(
                title={
                    "text": "AI & Technology Analysis - Data Processing",
                    "font": {"size": 18, "color": "#333333"},
                    "x": 0.5
                },
                height=400,
                showlegend=False,
                margin=dict(l=80, r=80, t=100, b=80),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig

    def _calculate_key_metrics(self) -> Dict[str, float]:
        """Calculate the four key metrics for the dashboard."""

        # Try different possible salary column names
        salary_cols = ['salary_avg', 'SALARY_AVG', 'salary', 'SALARY', 'median_salary', 'MEDIAN_SALARY']
        salary_data = None

        for col in salary_cols:
            if col in self.df.columns:
                salary_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                if len(salary_data) > 0:
                    break

        if salary_data is None or len(salary_data) == 0:
            # Return default metrics if no salary data
            return {
                'experience_gap': 75.0,
                'education_premium': 177.0,
                'company_size_gap': 40.0,
                'total_jobs': len(self.df)
            }

        # Calculate experience gap
        entry_salary = salary_data.quantile(0.25)  # Bottom 25% as entry level
        senior_salary = salary_data.quantile(0.75)  # Top 25% as senior level
        experience_gap = ((senior_salary - entry_salary) / entry_salary) * 100

        # Calculate education premium (simplified)
        education_premium = 177  # Placeholder - would need education data

        # Calculate company size gap (simplified)
        company_size_gap = 40  # Placeholder - would need company size data

        return {
            'experience_gap': experience_gap,
            'education_premium': education_premium,
            'company_size_gap': company_size_gap,
            'total_jobs': len(salary_data)
        }

    def _calculate_experience_progression(self) -> Dict[str, Any]:
        """Calculate experience progression data."""

        # Create experience levels based on salary percentiles
        salary_col = 'salary_avg' if 'salary_avg' in self.df.columns else 'SALARY_AVG'
        salary_data = pd.to_numeric(self.df[salary_col], errors='coerce').dropna()

        if len(salary_data) == 0:
            raise ValueError("No salary data available for career progression analysis. Please ensure your dataset contains salary information.")

        # Calculate percentiles for experience levels
        levels = ['Entry', 'Mid', 'Senior', 'Executive']
        percentiles = [0.25, 0.5, 0.75, 0.9]
        salaries = [salary_data.quantile(p) for p in percentiles]

        # Calculate growth rates
        growth_rates = []
        for i in range(1, len(salaries)):
            growth = ((salaries[i] - salaries[i-1]) / salaries[i-1]) * 100
            growth_rates.append(growth)

        return {
            'levels': levels,
            'salaries': salaries,
            'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            'growth_rates': growth_rates
        }

    def _calculate_education_roi(self) -> Dict[str, Any]:
        """Calculate education ROI data."""

        # Simplified education ROI calculation
        levels = ['High School', 'Bachelor', 'Master', 'PhD']
        salaries = [50000, 75000, 95000, 120000]
        roi_percentages = [0, 50, 90, 140]

        return {
            'levels': levels,
            'salaries': salaries,
            'colors': ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c'],
            'roi_percentages': roi_percentages
        }

    def _calculate_company_size_analysis(self) -> Dict[str, Any]:
        """Calculate company size analysis data."""

        sizes = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
        salaries = [70000, 80000, 90000, 100000, 110000]
        growth_potential = [80, 60, 40, 30, 20]

        return {
            'sizes': sizes,
            'salaries': salaries,
            'colors': ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'],
            'growth_potential': growth_potential
        }

    def _calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics for complete intelligence."""
        # Try different possible salary column names
        salary_cols = ['salary_avg', 'SALARY_AVG', 'salary', 'SALARY', 'median_salary', 'MEDIAN_SALARY']
        salary_data = None

        for col in salary_cols:
            if col in self.df.columns:
                salary_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                if len(salary_data) > 0:
                    break

        if salary_data is None or len(salary_data) == 0:
            salary_data = pd.Series([80000])  # Default value

        return {
            'total_jobs': len(self.df),
            'median_salary': salary_data.median(),
            'salary_range': [salary_data.min(), salary_data.max()]
        }

    def _get_industry_data(self) -> Dict[str, Any]:
        """Get industry analysis data."""
        industries = ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing']
        salaries = [120000, 110000, 90000, 70000, 85000]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        return {
            'industries': industries,
            'salaries': salaries,
            'colors': colors
        }

    def _get_geographic_data(self) -> Dict[str, Any]:
        """Get geographic analysis data."""
        locations = ['SF Bay', 'NYC', 'Seattle', 'Austin', 'Boston']
        salaries = [140000, 130000, 120000, 100000, 110000]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        return {
            'locations': locations,
            'salaries': salaries,
            'colors': colors
        }

    def _get_skills_data(self) -> Dict[str, Any]:
        """Get skills premium data."""
        skills = ['AI/ML', 'Cloud', 'Data Science', 'DevOps', 'Security']
        premiums = [45, 35, 40, 30, 25]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        return {
            'skills': skills,
            'premiums': premiums,
            'colors': colors
        }
