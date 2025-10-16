"""
Visualization functions for AI/ML job analysis by location.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional


def create_ai_ml_jobs_by_location_chart(
    city_stats: pd.DataFrame,
    title: str = "AI/ML Jobs by Top 5 Locations (Based on Specialized Skills)"
) -> go.Figure:
    """
    Create grouped bar chart showing AI/ML jobs vs total jobs by location.

    Args:
        city_stats: DataFrame with columns: city_name, total_jobs, ai_ml_jobs, median_salary
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Sort by AI/ML jobs for display
    city_stats = city_stats.sort_values('ai_ml_jobs', ascending=True)

    fig = go.Figure()

    # Add total jobs bars
    fig.add_trace(go.Bar(
        y=city_stats['city_name'],
        x=city_stats['total_jobs'],
        name='Total Jobs',
        orientation='h',
        marker=dict(color='#3498db'),
        text=city_stats['total_jobs'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Total Jobs: %{x:,}<extra></extra>'
    ))

    # Add AI/ML jobs bars
    fig.add_trace(go.Bar(
        y=city_stats['city_name'],
        x=city_stats['ai_ml_jobs'],
        name='AI/ML Jobs',
        orientation='h',
        marker=dict(color='#e74c3c'),
        text=city_stats.apply(lambda row: f"{int(row['ai_ml_jobs']):,} ({row['ai_ml_percentage']:.1f}%)", axis=1),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>AI/ML Jobs: %{x:,}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=city_stats['ai_ml_percentage']
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title="Number of Jobs",
        yaxis_title="",
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=150, r=150, t=100, b=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True
        ),
        yaxis=dict(
            showgrid=False
        )
    )

    return fig


def create_ai_ml_percentage_chart(
    city_stats: pd.DataFrame,
    title: str = "AI/ML Job Concentration by Location"
) -> go.Figure:
    """
    Create bar chart showing AI/ML job percentage by location.

    Args:
        city_stats: DataFrame with columns: city_name, ai_ml_percentage, median_salary
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Sort by percentage
    city_stats = city_stats.sort_values('ai_ml_percentage', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=city_stats['city_name'],
        x=city_stats['ai_ml_percentage'],
        orientation='h',
        marker=dict(
            color=city_stats['ai_ml_percentage'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title=dict(text="AI/ML %", side='right')
            )
        ),
        text=city_stats['ai_ml_percentage'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>AI/ML Jobs: %{x:.1f}%<br>Median Salary: $%{customdata:,.0f}<extra></extra>',
        customdata=city_stats['median_salary']
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title="Percentage of Jobs in AI/ML (%)",
        yaxis_title="",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=150, r=100, t=100, b=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            range=[0, city_stats['ai_ml_percentage'].max() * 1.2]
        ),
        yaxis=dict(
            showgrid=False
        )
    )

    return fig


def create_ai_ml_salary_comparison_chart(
    city_stats: pd.DataFrame,
    title: str = "Median Salary by Location (AI/ML vs All Jobs)"
) -> go.Figure:
    """
    Create chart comparing salaries for AI/ML jobs.

    Args:
        city_stats: DataFrame with city statistics
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Sort by median salary
    city_stats = city_stats.sort_values('median_salary', ascending=True)

    fig = go.Figure()

    # Add salary bars with AI/ML percentage as color
    fig.add_trace(go.Bar(
        y=city_stats['city_name'],
        x=city_stats['median_salary'],
        orientation='h',
        marker=dict(
            color=city_stats['ai_ml_percentage'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title=dict(text="AI/ML<br>Jobs (%)", side='right')
            ),
            line=dict(color='#2c3e50', width=1)
        ),
        text=city_stats['median_salary'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Median Salary: $%{x:,.0f}<br>AI/ML Jobs: %{customdata:.1f}%<extra></extra>',
        customdata=city_stats['ai_ml_percentage']
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title="Median Salary ($)",
        yaxis_title="",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=150, r=120, t=100, b=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            tickformat='$,.0f'
        ),
        yaxis=dict(
            showgrid=False
        )
    )

    return fig


def create_ai_ml_jobs_count_chart(
    city_stats: pd.DataFrame,
    title: str = "AI/ML Jobs Count by Top 5 Locations"
) -> go.Figure:
    """
    Create bar chart showing AI/ML jobs count by location.

    Args:
        city_stats: DataFrame with columns: city_name, ai_ml_jobs
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Sort by AI/ML jobs for display
    city_stats_sorted = city_stats.sort_values('ai_ml_jobs', ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=city_stats_sorted['city_name'],
            y=city_stats_sorted['ai_ml_jobs'],
            marker=dict(color='#e74c3c'),
            text=city_stats_sorted['ai_ml_jobs'].apply(lambda x: f'{int(x):,}'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>AI/ML Jobs: %{y:,}<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title="Location",
        yaxis_title="Number of AI/ML Jobs",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=80, r=80, t=100, b=80),
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig


def create_ai_ml_concentration_chart(
    city_stats: pd.DataFrame,
    title: str = "AI/ML Job Concentration (% of Total Jobs)"
) -> go.Figure:
    """
    Create bar chart showing AI/ML job concentration percentage by location.

    Args:
        city_stats: DataFrame with columns: city_name, ai_ml_percentage
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Sort by AI/ML jobs for consistent display
    city_stats_sorted = city_stats.sort_values('ai_ml_jobs', ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=city_stats_sorted['city_name'],
            y=city_stats_sorted['ai_ml_percentage'],
            marker=dict(color='#3498db'),
            text=city_stats_sorted['ai_ml_percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>AI/ML Concentration: %{y:.1f}%<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title="Location",
        yaxis_title="Percentage of Jobs in AI/ML (%)",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=80, r=80, t=100, b=80),
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig


def create_ai_ml_median_salary_chart(
    city_stats: pd.DataFrame,
    title: str = "Median Salary by Top 5 AI/ML Locations"
) -> go.Figure:
    """
    Create bar chart showing median salary by location with color gradient.

    Args:
        city_stats: DataFrame with columns: city_name, median_salary
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Sort by AI/ML jobs for consistent display
    city_stats_sorted = city_stats.sort_values('ai_ml_jobs', ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=city_stats_sorted['city_name'],
            y=city_stats_sorted['median_salary'],
            marker=dict(
                color=city_stats_sorted['median_salary'],
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title=dict(text="Salary ($)", side='right'))
            ),
            text=city_stats_sorted['median_salary'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Median Salary: $%{y:,.0f}<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title="Location",
        yaxis_title="Median Salary ($)",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=80, r=80, t=100, b=80),
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig


def create_ai_ml_jobs_distribution_chart(
    city_stats: pd.DataFrame,
    title: str = "Total Jobs Distribution Across Top 5 AI/ML Locations"
) -> go.Figure:
    """
    Create pie chart showing total jobs distribution across locations.

    Args:
        city_stats: DataFrame with columns: city_name, total_jobs
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Sort by AI/ML jobs for consistent display
    city_stats_sorted = city_stats.sort_values('ai_ml_jobs', ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=city_stats_sorted['city_name'],
            values=city_stats_sorted['total_jobs'],
            marker=dict(colors=px.colors.qualitative.Set3),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Total Jobs: %{value:,}<br>%{percent}<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=80, r=80, t=100, b=80)
    )

    return fig


def create_ai_ml_combined_dashboard(
    city_stats: pd.DataFrame,
    title: str = "AI/ML Jobs Analysis: Top 5 Locations"
) -> go.Figure:
    """
    Create comprehensive dashboard with multiple AI/ML metrics.

    Args:
        city_stats: DataFrame with city statistics
        title: Main dashboard title

    Returns:
        Plotly Figure with subplots
    """
    # Sort for consistent display
    city_stats_sorted = city_stats.sort_values('ai_ml_jobs', ascending=False)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'AI/ML Jobs Count',
            'AI/ML Job Percentage',
            'Median Salary',
            'Jobs Distribution'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "pie"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # 1. AI/ML Jobs Count (top-left)
    fig.add_trace(
        go.Bar(
            x=city_stats_sorted['city_name'],
            y=city_stats_sorted['ai_ml_jobs'],
            marker=dict(color='#e74c3c'),
            text=city_stats_sorted['ai_ml_jobs'].apply(lambda x: f'{int(x):,}'),
            textposition='outside',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>AI/ML Jobs: %{y:,}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. AI/ML Percentage (top-right)
    fig.add_trace(
        go.Bar(
            x=city_stats_sorted['city_name'],
            y=city_stats_sorted['ai_ml_percentage'],
            marker=dict(color='#3498db'),
            text=city_stats_sorted['ai_ml_percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>AI/ML: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Median Salary (bottom-left)
    fig.add_trace(
        go.Bar(
            x=city_stats_sorted['city_name'],
            y=city_stats_sorted['median_salary'],
            marker=dict(color='#2ecc71'),
            text=city_stats_sorted['median_salary'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Salary: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Total Jobs Distribution (bottom-right)
    fig.add_trace(
        go.Pie(
            labels=city_stats_sorted['city_name'],
            values=city_stats_sorted['total_jobs'],
            marker=dict(colors=px.colors.qualitative.Set3),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Jobs: %{value:,}<br>%{percent}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50')
        ),
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=11)
    )

    # Update axes
    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig

