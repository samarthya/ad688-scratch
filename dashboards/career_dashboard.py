"""
Interactive dashboard application for job market analysis.

This Dash application provides interactive visualizations for exploring
salary trends, job market patterns, and career planning insights.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, date

# Import custom modules
import sys
sys.path.append('..')
from src.data.preprocess_data import JobDataProcessor
from src.visualization.plots import SalaryVisualizer

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Career Analytics Dashboard"

# Load and process data
print("Loading job market data...")
processor = JobDataProcessor('data/raw/lightcast_job_postings.csv')
df = processor.process_all()
visualizer = SalaryVisualizer(df)

# Dashboard styling
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Career Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
        html.P("Interactive analysis of job market trends and salary compensation",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Filter by Industry:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='industry-dropdown',
                options=[{'label': 'All Industries', 'value': 'all'}] + 
                        [{'label': ind, 'value': ind} for ind in sorted(df['industry'].unique())],
                value='all',
                style={'marginBottom': '20px'}
            ),
            
            html.Label("Experience Level:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='experience-dropdown',
                options=[{'label': 'All Levels', 'value': 'all'}] + 
                        [{'label': exp, 'value': exp} for exp in sorted(df['experience_level'].unique())],
                value='all',
                style={'marginBottom': '20px'}
            ),
            
            html.Label("Salary Range:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.RangeSlider(
                id='salary-slider',
                min=df['salary_avg'].min(),
                max=df['salary_avg'].max(),
                value=[df['salary_avg'].min(), df['salary_avg'].max()],
                marks={
                    int(df['salary_avg'].min()): f"${int(df['salary_avg'].min()/1000)}k",
                    int(df['salary_avg'].max()): f"${int(df['salary_avg'].max()/1000)}k"
                },
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 
                 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Main content area
        html.Div([
            # Key metrics cards
            html.Div(id='metrics-cards', style={'marginBottom': '30px'}),
            
            # Chart tabs
            dcc.Tabs(id='chart-tabs', value='salary-distribution', children=[
                dcc.Tab(label='Salary Distribution', value='salary-distribution'),
                dcc.Tab(label='Industry Analysis', value='industry-analysis'),
                dcc.Tab(label='AI vs Traditional', value='ai-comparison'),
                dcc.Tab(label='Geographic Trends', value='geographic-trends'),
                dcc.Tab(label='Remote Work', value='remote-analysis')
            ]),
            
            html.Div(id='chart-content', style={'marginTop': '20px'})
            
        ], style={'width': '70%', 'display': 'inline-block', 'marginLeft': '5%'})
        
    ]),
    
    # Footer
    html.Div([
        html.P(f"Data last updated: {datetime.now().strftime('%Y-%m-%d')} | "
               f"Total job postings: {len(df):,}",
               style={'textAlign': 'center', 'color': '#95a5a6', 'marginTop': '40px'})
    ])
    
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0', 'padding': '20px'})


# Callback for filtering data
@app.callback(
    [Output('metrics-cards', 'children'),
     Output('chart-content', 'children')],
    [Input('industry-dropdown', 'value'),
     Input('experience-dropdown', 'value'),
     Input('salary-slider', 'value'),
     Input('chart-tabs', 'value')]
)
def update_dashboard(selected_industry, selected_experience, salary_range, active_tab):
    # Filter data based on selections
    filtered_df = df.copy()
    
    if selected_industry != 'all':
        filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
    
    if selected_experience != 'all':
        filtered_df = filtered_df[filtered_df['experience_level'] == selected_experience]
    
    filtered_df = filtered_df[
        (filtered_df['salary_avg'] >= salary_range[0]) & 
        (filtered_df['salary_avg'] <= salary_range[1])
    ]
    
    # Create metrics cards
    metrics_cards = create_metrics_cards(filtered_df)
    
    # Create chart based on selected tab
    if active_tab == 'salary-distribution':
        chart = create_salary_distribution_chart(filtered_df)
    elif active_tab == 'industry-analysis':
        chart = create_industry_analysis_chart(filtered_df)
    elif active_tab == 'ai-comparison':
        chart = create_ai_comparison_chart(filtered_df)
    elif active_tab == 'geographic-trends':
        chart = create_geographic_chart(filtered_df)
    elif active_tab == 'remote-analysis':
        chart = create_remote_analysis_chart(filtered_df)
    else:
        chart = html.Div("Select a chart type")
    
    return metrics_cards, chart


def create_metrics_cards(df):
    """Create summary metrics cards."""
    if len(df) == 0:
        return html.Div("No data available for selected filters")
    
    total_jobs = len(df)
    median_salary = df['salary_avg'].median()
    ai_percentage = (df['ai_related'].sum() / len(df)) * 100
    remote_percentage = (df['is_remote'].sum() / len(df)) * 100
    
    cards = html.Div([
        # Total Jobs Card
        html.Div([
            html.H3(f"{total_jobs:,}", style={'color': '#3498db', 'fontSize': '32px', 'margin': '0'}),
            html.P("Total Job Postings", style={'margin': '5px 0', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'backgroundColor': 'white', 'padding': '20px', 
                 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block', 'margin': '1%'}),
        
        # Median Salary Card
        html.Div([
            html.H3(f"${median_salary:,.0f}", style={'color': '#27ae60', 'fontSize': '32px', 'margin': '0'}),
            html.P("Median Salary", style={'margin': '5px 0', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'backgroundColor': 'white', 'padding': '20px', 
                 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block', 'margin': '1%'}),
        
        # AI Jobs Card
        html.Div([
            html.H3(f"{ai_percentage:.1f}%", style={'color': '#e74c3c', 'fontSize': '32px', 'margin': '0'}),
            html.P("AI-Related Jobs", style={'margin': '5px 0', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'backgroundColor': 'white', 'padding': '20px', 
                 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block', 'margin': '1%'}),
        
        # Remote Jobs Card
        html.Div([
            html.H3(f"{remote_percentage:.1f}%", style={'color': '#9b59b6', 'fontSize': '32px', 'margin': '0'}),
            html.P("Remote Jobs", style={'margin': '5px 0', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'backgroundColor': 'white', 'padding': '20px', 
                 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block', 'margin': '1%'})
    ])
    
    return cards


def create_salary_distribution_chart(df):
    """Create salary distribution histogram."""
    if len(df) == 0:
        return html.Div("No data available")
    
    fig = px.histogram(
        df, 
        x='salary_avg', 
        color='ai_related',
        nbins=30,
        title="Salary Distribution",
        labels={'salary_avg': 'Average Salary (USD)', 'count': 'Number of Jobs'},
        color_discrete_map={True: '#e74c3c', False: '#3498db'}
    )
    
    fig.update_layout(height=500, showlegend=True)
    
    return dcc.Graph(figure=fig)


def create_industry_analysis_chart(df):
    """Create industry salary comparison chart."""
    if len(df) == 0:
        return html.Div("No data available")
    
    industry_stats = df.groupby('industry')['salary_avg'].agg(['median', 'count']).reset_index()
    industry_stats = industry_stats[industry_stats['count'] >= 5]  # Filter for sufficient data
    
    fig = px.bar(
        industry_stats.sort_values('median', ascending=True),
        x='median',
        y='industry',
        orientation='h',
        title="Median Salary by Industry",
        labels={'median': 'Median Salary (USD)', 'industry': 'Industry'},
        color='median',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=500, showlegend=False)
    
    return dcc.Graph(figure=fig)


def create_ai_comparison_chart(df):
    """Create AI vs traditional roles comparison."""
    if len(df) == 0:
        return html.Div("No data available")
    
    fig = go.Figure()
    
    for ai_status in [False, True]:
        subset = df[df['ai_related'] == ai_status]
        if len(subset) > 0:
            label = "AI-Related" if ai_status else "Traditional"
            fig.add_trace(go.Box(
                y=subset['salary_avg'],
                name=label,
                boxpoints='outliers',
                marker_color='red' if ai_status else 'blue'
            ))
    
    fig.update_layout(
        title="Salary Comparison: AI vs Traditional Roles",
        yaxis_title="Average Salary (USD)",
        height=500
    )
    
    return dcc.Graph(figure=fig)


def create_geographic_chart(df):
    """Create geographic salary analysis."""
    if len(df) == 0:
        return html.Div("No data available")
    
    geo_stats = df.groupby('city')['salary_avg'].agg(['median', 'count']).reset_index()
    geo_stats = geo_stats[geo_stats['count'] >= 10]  # Filter for sufficient data
    
    fig = px.scatter(
        geo_stats.sort_values('median', ascending=False).head(15),
        x='count',
        y='median',
        size='count',
        hover_name='city',
        title="Job Count vs Median Salary by City (Top 15)",
        labels={'count': 'Number of Jobs', 'median': 'Median Salary (USD)'}
    )
    
    fig.update_layout(height=500)
    
    return dcc.Graph(figure=fig)


def create_remote_analysis_chart(df):
    """Create remote work analysis chart."""
    if len(df) == 0:
        return html.Div("No data available")
    
    remote_stats = df.groupby(['industry', 'is_remote'])['salary_avg'].median().reset_index()
    remote_pivot = remote_stats.pivot(index='industry', columns='is_remote', values='salary_avg').reset_index()
    
    if False in remote_pivot.columns and True in remote_pivot.columns:
        remote_pivot.columns = ['Industry', 'On-Site', 'Remote']
        remote_pivot = remote_pivot.dropna()
        
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
            title="Median Salary: Remote vs On-Site by Industry",
            xaxis_title="Industry",
            yaxis_title="Median Salary (USD)",
            barmode='group',
            height=500
        )
    else:
        # Fallback if not enough remote data
        fig = px.bar(
            df.groupby('is_remote')['salary_avg'].median().reset_index(),
            x='is_remote',
            y='salary_avg',
            title="Overall Salary: Remote vs On-Site"
        )
    
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    print("Starting Career Analytics Dashboard...")
    print("Access the dashboard at: http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)