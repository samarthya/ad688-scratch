"""
Improved chart configuration for better readability and salary disparity theme coherence.
This file provides standardized chart settings focusing on salary disparity visualization.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional

class SalaryDisparityChartConfig:
    """Standardized configuration for salary disparity-focused charts."""
    
    @staticmethod
    def get_standard_layout() -> Dict[str, Any]:
        """Get standard layout configuration for readability."""
        return {
            'height': 900,  # Increased height for better readability
            'width': 1200,  # Standard width
            'title': {
                'font': {'size': 24, 'color': '#2E2E2E'},
                'x': 0.5,  # Center title
                'pad': {'t': 50, 'b': 20}
            },
            'xaxis': {
                'title': {'font': {'size': 16, 'color': '#444444'}},
                'tickfont': {'size': 14, 'color': '#444444'},
                'showgrid': True,
                'gridcolor': '#E0E0E0',
                'gridwidth': 1
            },
            'yaxis': {
                'title': {'font': {'size': 16, 'color': '#444444'}},
                'tickfont': {'size': 14, 'color': '#444444'},
                'showgrid': True,
                'gridcolor': '#E0E0E0',
                'gridwidth': 1,
                'tickformat': '$,.0f'  # Format as currency
            },
            'legend': {
                'font': {'size': 14, 'color': '#444444'},
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': '#CCCCCC',
                'borderwidth': 1
            },
            'template': 'plotly_white',
            'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80}
        }
    
    @staticmethod
    def get_salary_disparity_colors() -> Dict[str, str]:
        """Color scheme emphasizing salary disparities."""
        return {
            'low_salary': '#E74C3C',      # Red for low salaries
            'medium_salary': '#F39C12',   # Orange for medium salaries  
            'high_salary': '#27AE60',     # Green for high salaries
            'undefined': '#95A5A6',       # Gray for undefined/missing data
            'emphasis': '#8E44AD',        # Purple for emphasis
            'background': '#FFFFFF',      # White background
            'text': '#2E2E2E'            # Dark gray text
        }
    
    @staticmethod
    def apply_salary_focus_styling(fig: go.Figure, chart_type: str = 'general') -> go.Figure:
        """Apply salary disparity-focused styling to any Plotly figure."""
        
        layout_config = SalaryDisparityChartConfig.get_standard_layout()
        colors = SalaryDisparityChartConfig.get_salary_disparity_colors()
        
        # Apply standard layout
        fig.update_layout(layout_config)
        
        # Chart-specific adjustments
        if chart_type == 'salary_comparison':
            # Emphasize salary differences with distinct colors
            fig.update_traces(
                marker=dict(
                    line=dict(width=2, color=colors['text']),
                    opacity=0.8
                ),
                textfont=dict(size=14, color=colors['text'])
            )
            
        elif chart_type == 'company_disparity':
            # Show company size impact on salaries
            fig.update_layout(
                annotations=[
                    dict(
                        text="<b>Company Size Salary Impact</b><br>Larger companies typically offer higher compensation",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        font=dict(size=12, color=colors['text']),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=colors['emphasis'],
                        borderwidth=1
                    )
                ]
            )
            
        elif chart_type == 'experience_progression':
            # Highlight experience-based salary growth
            fig.update_layout(
                annotations=[
                    dict(
                        text="<b>Experience Premium</b><br>Salary growth with career progression",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.98, y=0.02, xanchor='right', yanchor='bottom',
                        font=dict(size=12, color=colors['text']),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=colors['emphasis'],
                        borderwidth=1
                    )
                ]
            )
        
        return fig
    
    @staticmethod
    def create_readable_bar_chart(data, x_col: str, y_col: str, title: str, 
                                color_col: Optional[str] = None) -> go.Figure:
        """Create a highly readable bar chart focused on salary disparities."""
        
        colors = SalaryDisparityChartConfig.get_salary_disparity_colors()
        
        if color_col:
            fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.bar(data, x=x_col, y=y_col, title=title)
            # Apply salary-focused color scheme
            fig.update_traces(marker_color=colors['emphasis'])
        
        # Apply standard styling
        fig = SalaryDisparityChartConfig.apply_salary_focus_styling(fig, 'salary_comparison')
        
        return fig
    
    @staticmethod
    def create_readable_box_plot(data, x_col: str, y_col: str, title: str) -> go.Figure:
        """Create a readable box plot for salary distribution analysis."""
        
        colors = SalaryDisparityChartConfig.get_salary_disparity_colors()
        
        fig = px.box(data, x=x_col, y=y_col, title=title)
        
        # Apply salary disparity colors
        fig.update_traces(
            marker_color=colors['emphasis'],
            line_color=colors['text'],
            fillcolor=colors['emphasis']
        )
        
        # Apply standard styling
        fig = SalaryDisparityChartConfig.apply_salary_focus_styling(fig, 'salary_comparison')
        
        return fig