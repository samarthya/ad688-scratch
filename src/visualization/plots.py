"""
Visualization utilities for job market analysis.

This module provides reusable plotting functions and dashboard components
for analyzing salary trends and job market patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional

# Set default styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalaryVisualizer:
    """
    Main class for creating salary and job market visualizations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with processed job data."""
        self.df = df
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_salary_distribution(self, 
                               group_by: str = None,
                               bins: int = 50,
                               interactive: bool = True) -> go.Figure:
        """
        Create salary distribution plots.
        
        Args:
            group_by: Column to group distributions by
            bins: Number of histogram bins
            interactive: Whether to return plotly (True) or matplotlib (False)
        """
        if interactive:
            if group_by:
                fig = px.histogram(
                    self.df, 
                    x='salary_avg',
                    color=group_by,
                    nbins=bins,
                    title=f"Salary Distribution by {group_by.title()}",
                    labels={'salary_avg': 'Average Salary (USD)', 'count': 'Number of Jobs'},
                    opacity=0.7
                )
            else:
                fig = px.histogram(
                    self.df, 
                    x='salary_avg',
                    nbins=bins,
                    title="Overall Salary Distribution",
                    labels={'salary_avg': 'Average Salary (USD)', 'count': 'Number of Jobs'}
                )
            
            fig.update_layout(
                xaxis_title="Average Salary (USD)",
                yaxis_title="Number of Job Postings",
                showlegend=True if group_by else False
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if group_by and group_by in self.df.columns:
                for category in self.df[group_by].unique():
                    subset = self.df[self.df[group_by] == category]
                    ax.hist(subset['salary_avg'], bins=bins, alpha=0.7, 
                           label=category, edgecolor='black', linewidth=0.5)
                ax.legend()
            else:
                ax.hist(self.df['salary_avg'], bins=bins, alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Average Salary (USD)')
            ax.set_ylabel('Number of Job Postings')
            ax.set_title('Salary Distribution')
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def plot_salary_by_category(self, 
                               category: str,
                               top_n: int = 10,
                               horizontal: bool = True) -> go.Figure:
        """
        Create bar plots of salary by category.
        
        Args:
            category: Column to analyze (e.g., 'industry', 'city')
            top_n: Number of top categories to show
            horizontal: Whether to create horizontal bar chart
        """
        # Calculate median salary by category
        salary_by_category = (
            self.df.groupby(category)['salary_avg']
            .agg(['median', 'mean', 'count'])
            .round(0)
            .reset_index()
        )
        
        # Filter to top N by median salary
        top_categories = (
            salary_by_category
            .nlargest(top_n, 'median')
            .sort_values('median', ascending=horizontal)
        )
        
        if horizontal:
            fig = px.bar(
                top_categories,
                x='median',
                y=category,
                title=f"Median Salary by {category.title()} (Top {top_n})",
                labels={'median': 'Median Salary (USD)', category: category.title()},
                color='median',
                color_continuous_scale='viridis',
                text='median'
            )
            
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
        else:
            fig = px.bar(
                top_categories,
                x=category,
                y='median',
                title=f"Median Salary by {category.title()} (Top {top_n})",
                labels={'median': 'Median Salary (USD)', category: category.title()},
                color='median',
                color_continuous_scale='viridis',
                text='median'
            )
            
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
        
        fig.update_layout(showlegend=False)
        return fig
    
    def plot_ai_salary_comparison(self) -> go.Figure:
        """Compare salaries between AI and non-AI roles."""
        
        ai_comparison = (
            self.df.groupby('ai_related')['salary_avg']
            .agg(['median', 'mean', 'std', 'count'])
            .round(0)
        )
        
        # Create box plot
        fig = go.Figure()
        
        for ai_status in [False, True]:
            subset = self.df[self.df['ai_related'] == ai_status]
            label = "AI-Related" if ai_status else "Traditional"
            
            fig.add_trace(go.Box(
                y=subset['salary_avg'],
                name=label,
                boxpoints='outliers',
                marker_color='red' if ai_status else 'blue',
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Salary Comparison: AI vs Traditional Roles",
            yaxis_title="Average Salary (USD)",
            xaxis_title="Role Type",
            showlegend=False
        )
        
        return fig
    
    def plot_experience_salary_trend(self) -> go.Figure:
        """Plot salary trends by experience level."""
        
        exp_salary = (
            self.df.groupby(['experience_level', 'ai_related'])['salary_avg']
            .median()
            .reset_index()
        )
        
        fig = go.Figure()
        
        for ai_status in [False, True]:
            subset = exp_salary[exp_salary['ai_related'] == ai_status]
            label = "AI-Related" if ai_status else "Traditional"
            
            fig.add_trace(go.Scatter(
                x=subset['experience_level'],
                y=subset['salary_avg'],
                mode='lines+markers',
                name=label,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Salary Progression by Experience Level",
            xaxis_title="Experience Level",
            yaxis_title="Median Salary (USD)",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_geographic_heatmap(self, metric: str = 'median') -> go.Figure:
        """Create geographic salary heatmap."""
        
        # Calculate salary statistics by state
        geo_data = (
            self.df.groupby('state')['salary_avg']
            .agg(['median', 'mean', 'count'])
            .reset_index()
        )
        
        # Filter states with sufficient data
        geo_data = geo_data[geo_data['count'] >= 10]
        
        fig = px.choropleth(
            geo_data,
            locations='state',
            color=metric,
            locationmode='USA-states',
            scope='usa',
            title=f"Average Salary by State ({metric.title()})",
            labels={metric: f'{metric.title()} Salary (USD)'},
            color_continuous_scale='viridis'
        )
        
        return fig
    
    def create_correlation_matrix(self) -> go.Figure:
        """Create correlation matrix of numerical variables."""
        
        # Select numerical columns
        numerical_cols = [
            'salary_avg', 'salary_min', 'salary_max', 
            'experience_years', 'ai_related', 'is_remote'
        ]
        
        # Convert boolean to int for correlation
        corr_data = self.df[numerical_cols].copy()
        corr_data['ai_related'] = corr_data['ai_related'].astype(int)
        corr_data['is_remote'] = corr_data['is_remote'].astype(int)
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            colorscale='RdBu_r',
            zmid=0
        )
        
        fig.update_layout(
            title="Correlation Matrix: Salary and Job Characteristics",
            width=700,
            height=600
        )
        
        return fig
    
    def plot_remote_salary_analysis(self) -> go.Figure:
        """Analyze salary differences for remote vs on-site roles."""
        
        # Calculate statistics by remote status and industry
        remote_analysis = (
            self.df.groupby(['industry', 'is_remote'])['salary_avg']
            .median()
            .reset_index()
        )
        
        # Pivot for easier plotting
        remote_pivot = remote_analysis.pivot(
            index='industry', 
            columns='is_remote', 
            values='salary_avg'
        ).reset_index()
        
        remote_pivot.columns = ['Industry', 'On-Site', 'Remote']
        remote_pivot = remote_pivot.dropna()
        
        # Calculate percentage difference
        remote_pivot['Difference'] = (
            (remote_pivot['Remote'] - remote_pivot['On-Site']) / 
            remote_pivot['On-Site'] * 100
        )
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='On-Site',
            x=remote_pivot['Industry'],
            y=remote_pivot['On-Site'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Remote',
            x=remote_pivot['Industry'],
            y=remote_pivot['Remote'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Salary Comparison: Remote vs On-Site by Industry",
            xaxis_title="Industry",
            yaxis_title="Median Salary (USD)",
            barmode='group'
        )
        
        return fig


def create_dashboard_layout(df: pd.DataFrame) -> Dict:
    """
    Create a layout configuration for a Dash dashboard.
    
    Args:
        df: Processed job data
        
    Returns:
        Dictionary with dashboard layout components
    """
    
    visualizer = SalaryVisualizer(df)
    
    # Generate key plots
    plots = {
        'salary_distribution': visualizer.plot_salary_distribution(),
        'industry_comparison': visualizer.plot_salary_by_category('industry'),
        'ai_comparison': visualizer.plot_ai_salary_comparison(),
        'experience_trend': visualizer.plot_experience_salary_trend(),
        'correlation_matrix': visualizer.create_correlation_matrix(),
        'remote_analysis': visualizer.plot_remote_salary_analysis()
    }
    
    return plots


def save_plots_to_files(df: pd.DataFrame, output_dir: str = 'reports/figures/'):
    """
    Generate and save all standard plots to files.
    
    Args:
        df: Processed job data
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = SalaryVisualizer(df)
    
    # Generate and save plots
    plots = {
        'salary_distribution.html': visualizer.plot_salary_distribution(),
        'salary_by_industry.html': visualizer.plot_salary_by_category('industry'),
        'ai_vs_traditional.html': visualizer.plot_ai_salary_comparison(),
        'experience_trends.html': visualizer.plot_experience_salary_trend(),
        'correlation_matrix.html': visualizer.create_correlation_matrix(),
        'remote_work_analysis.html': visualizer.plot_remote_salary_analysis()
    }
    
    for filename, fig in plots.items():
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved plot: {filepath}")


if __name__ == "__main__":
    # Example usage
    from src.data.preprocess_data import JobDataProcessor
    
    # Load and process data
    processor = JobDataProcessor('data/raw/lightcast_job_postings.csv')
    df = processor.process_all()
    
    # Create visualizations
    visualizer = SalaryVisualizer(df)
    
    # Generate sample plots
    print("Generating sample visualizations...")
    
    # Show salary distribution
    fig1 = visualizer.plot_salary_distribution(group_by='ai_related')
    fig1.show()
    
    # Show industry comparison
    fig2 = visualizer.plot_salary_by_category('industry')
    fig2.show()
    
    print("Visualization examples complete.")