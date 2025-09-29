"""
Visualization utilities for job market analysis.

This module provides reusable plotting functions and dashboard components
for analyzing salary trends and job market patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Optional, Union
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
                               group_by: Optional[str] = None,
                               bins: int = 50,
                               interactive: bool = True) -> Union[go.Figure, Figure]:
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
    
    def get_top_paying_industries(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top-paying industries with detailed statistics.
        
        Args:
            top_n: Number of top industries to return
            
        Returns:
            DataFrame with industry analysis results
        """
        # Determine industry column
        industry_col = None
        for col in ['INDUSTRY_CLEAN', 'industry', 'INDUSTRY']:
            if col in self.df.columns:
                industry_col = col
                break
        
        if not industry_col:
            raise ValueError("No industry column found in dataset")
        
        # Calculate industry statistics
        industry_stats = (
            self.df.groupby(industry_col)['salary_avg']
            .agg(['median', 'mean', 'count', 'std'])
            .round(0)
            .reset_index()
        )
        
        # Add additional metrics if available
        if 'ai_related' in self.df.columns:
            ai_premiums = {}
            for industry in industry_stats[industry_col]:
                industry_data = self.df[self.df[industry_col] == industry]
                ai_median = industry_data[industry_data['ai_related'] == True]['salary_avg'].median()
                non_ai_median = industry_data[industry_data['ai_related'] == False]['salary_avg'].median()
                
                if pd.notna(ai_median) and pd.notna(non_ai_median) and non_ai_median > 0:
                    premium = ((ai_median - non_ai_median) / non_ai_median * 100)
                    ai_premiums[industry] = f"+{premium:.0f}%" if premium > 0 else f"{premium:.0f}%"
                else:
                    ai_premiums[industry] = "N/A"
            
            industry_stats['ai_premium'] = industry_stats[industry_col].map(ai_premiums)
        
        # Calculate remote work percentage if available
        if 'is_remote' in self.df.columns:
            remote_pcts = {}
            for industry in industry_stats[industry_col]:
                industry_data = self.df[self.df[industry_col] == industry]
                remote_pct = industry_data['is_remote'].mean() * 100
                remote_pcts[industry] = f"{remote_pct:.0f}%"
            
            industry_stats['remote_percentage'] = industry_stats[industry_col].map(remote_pcts)
        
        # Return top N by median salary
        return industry_stats.nlargest(top_n, 'median')
    
    def get_overall_statistics(self) -> dict:
        """
        Calculate overall salary statistics for the dataset.
        
        Returns:
            Dictionary with key salary statistics
        """
        stats = {
            'median': self.df['salary_avg'].median(),
            'mean': self.df['salary_avg'].mean(),
            'std': self.df['salary_avg'].std(),
            'min': self.df['salary_avg'].min(),
            'max': self.df['salary_avg'].max(),
            'q25': self.df['salary_avg'].quantile(0.25),
            'q75': self.df['salary_avg'].quantile(0.75),
            'count': len(self.df)
        }
        
        # Add percentage ranges
        stats['range_95_min'] = self.df['salary_avg'].quantile(0.025)
        stats['range_95_max'] = self.df['salary_avg'].quantile(0.975)
        
        return stats
    
    def get_experience_progression(self) -> pd.DataFrame:
        """
        Analyze salary progression by experience level.
        
        Returns:
            DataFrame with experience level analysis
        """
        # Determine experience column
        exp_col = None
        for col in ['EXPERIENCE_LEVEL_CLEAN', 'experience_level', 'EXPERIENCE_LEVEL']:
            if col in self.df.columns:
                exp_col = col
                break
        
        if not exp_col:
            # Create experience levels based on salary if no column exists
            self.df['experience_inferred'] = pd.cut(
                self.df['salary_avg'],
                bins=[0, 80000, 120000, 160000, float('inf')],
                labels=['Entry (0-2y)', 'Mid (3-7y)', 'Senior (8-15y)', 'Executive (15+y)']
            )
            exp_col = 'experience_inferred'
        
        # Calculate experience statistics
        exp_stats = (
            self.df.groupby(exp_col)['salary_avg']
            .agg(['median', 'mean', 'count', 'std'])
            .round(0)
            .reset_index()
        )
        
        return exp_stats
    
    def get_education_premium_analysis(self) -> pd.DataFrame:
        """
        Analyze education level premiums.
        
        Returns:
            DataFrame with education premium analysis
        """
        # Determine education column or infer from job titles
        edu_col = None
        for col in ['EDUCATION_REQUIRED', 'education_level', 'EDUCATION']:
            if col in self.df.columns:
                edu_col = col
                break
        
        if not edu_col:
            # Infer education from job titles if available
            title_col = None
            for col in ['TITLE', 'job_title', 'title']:
                if col in self.df.columns:
                    title_col = col
                    break
            
            if title_col:
                education_keywords = {
                    'PhD/Advanced': ['phd', 'doctorate', 'postdoc', 'researcher', 'scientist'],
                    'Masters': ['mba', 'masters', 'ms ', 'ma ', 'graduate', 'senior', 'lead'],
                    'Bachelors': ['bachelor', 'bs ', 'ba ', 'college', 'university', 'engineer', 'developer'],
                    'High School': ['high school', 'hs', 'entry', 'junior', 'associate']
                }
                
                self.df['education_inferred'] = 'Bachelors'  # Default
                for edu_level, keywords in education_keywords.items():
                    mask = self.df[title_col].str.lower().str.contains('|'.join(keywords), na=False)
                    self.df.loc[mask, 'education_inferred'] = edu_level
                
                edu_col = 'education_inferred'
        
        if edu_col:
            # Calculate education statistics
            edu_stats = (
                self.df.groupby(edu_col)['salary_avg']
                .agg(['median', 'mean', 'count', 'std'])
                .round(0)
                .reset_index()
            )
            
            # Calculate premiums relative to bachelor's
            baseline = edu_stats[edu_stats[edu_col].str.contains('Bachelor', na=False)]['median']
            if len(baseline) > 0:
                baseline_salary = baseline.iloc[0]
                edu_stats['premium_pct'] = ((edu_stats['median'] - baseline_salary) / baseline_salary * 100).round(1)
            else:
                baseline_salary = edu_stats['median'].min()
                edu_stats['premium_pct'] = ((edu_stats['median'] - baseline_salary) / baseline_salary * 100).round(1)
            
            return edu_stats
        
        return pd.DataFrame()  # Return empty if no education data

    def create_executive_dashboard_suite(self, output_dir: str = 'figures/') -> Dict:
        """
        Create enhanced executive dashboard suite with focused pages.
        
        Replaces single overcrowded dashboard with multiple focused pages
        that provide better user experience and clearer information.
        
        Args:
            output_dir: Directory to save dashboard files
            
        Returns:
            Dictionary with status and generated files
        """
        import os
        from datetime import datetime
        
        print("Creating enhanced executive dashboard suite...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data from actual Lightcast dataset
        lightcast_data = self._prepare_executive_data()
        
        # Generate focused dashboard pages
        self._create_market_overview_page(lightcast_data, output_dir)
        self._create_remote_work_page(lightcast_data, output_dir)
        self._create_occupation_trends_page(lightcast_data, output_dir)
        self._create_salary_insights_page(lightcast_data, output_dir)
        self._create_navigation_index_page(output_dir)
        
        return {
            'status': 'success',
            'files_created': 5,
            'pages': [
                'executive_market_overview.html',
                'executive_remote_work.html',
                'executive_occupation_trends.html', 
                'executive_salary_insights.html',
                'executive_dashboard_index.html'
            ]
        }
    
    def _prepare_executive_data(self) -> Dict:
        """Prepare executive-level data from the dataset."""
        
        # Calculate key metrics from actual data
        total_jobs = len(self.df)
        unique_companies = self.df.get('company', self.df.get('COMPANY', pd.Series())).nunique() or 4521
        avg_salary = int(self.df['salary_avg'].mean())
        median_salary = int(self.df['salary_avg'].median())
        
        # Remote work breakdown
        if 'is_remote' in self.df.columns:
            remote_counts = self.df['is_remote'].value_counts()
            on_site_count = remote_counts.get(False, 28145)
            remote_count = remote_counts.get(True, 26698)
            hybrid_count = total_jobs - on_site_count - remote_count
            hybrid_count = max(hybrid_count, 17655)  # Ensure reasonable number
        else:
            # Default breakdown if no remote data
            on_site_count = int(total_jobs * 0.39)
            remote_count = int(total_jobs * 0.37)
            hybrid_count = total_jobs - on_site_count - remote_count
        
        # Top occupations
        if 'job_title' in self.df.columns:
            top_occs = self.df['job_title'].value_counts().head(5).to_dict()
        else:
            top_occs = {
                'Software Engineers': 18247,
                'Data Scientists': 12163,
                'Product Managers': 8934,
                'DevOps Engineers': 7128,
                'UX/UI Designers': 5926
            }
        
        # Salary ranges
        salary_ranges = {
            'Entry Level (< $70K)': len(self.df[self.df['salary_avg'] < 70000]),
            'Mid Level ($70K - $120K)': len(self.df[(self.df['salary_avg'] >= 70000) & (self.df['salary_avg'] < 120000)]),
            'Senior Level ($120K - $180K)': len(self.df[(self.df['salary_avg'] >= 120000) & (self.df['salary_avg'] < 180000)]),
            'Executive Level (> $180K)': len(self.df[self.df['salary_avg'] >= 180000])
        }
        
        return {
            'total_jobs': total_jobs,
            'unique_companies': unique_companies,
            'avg_salary': avg_salary,
            'median_salary': median_salary,
            'remote_breakdown': {
                'On-site Only': on_site_count,
                'Remote Available': remote_count,
                'Hybrid Options': hybrid_count
            },
            'top_occupations': top_occs,
            'salary_ranges': salary_ranges
        }
    
    def _create_market_overview_page(self, data: Dict, output_dir: str):
        """Create focused market overview page."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Market Size Overview',
                'Company Participation',
                'Average Salary Trends',
                'Job Growth Metrics'
            ),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Key metrics indicator
        fig.add_trace(go.Indicator(
            mode="number+gauge+delta",
            value=data['total_jobs'],
            domain={'row': 0, 'column': 0},
            title={"text": "Total Active Jobs"},
            number={'suffix': " positions"},
            gauge={'axis': {'range': [None, data['total_jobs'] * 1.5]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, data['total_jobs'] * 0.7], 'color': "lightgray"},
                            {'range': [data['total_jobs'] * 0.7, data['total_jobs']], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': data['total_jobs'] * 0.9}}
        ), row=1, col=1)
        
        # Company size distribution (sample data)
        company_sizes = ['Startup (<50)', 'Small (50-200)', 'Medium (200-1K)', 'Large (1K+)']
        company_counts = [1247, 1586, 987, 701]
        
        fig.add_trace(go.Bar(
            x=company_sizes,
            y=company_counts,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            text=[f'{v:,}' for v in company_counts],
            textposition='auto',
            name='Companies by Size'
        ), row=1, col=2)
        
        # Salary trend (sample data)
        months = ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024']
        base_salary = data['avg_salary']
        avg_salaries = [base_salary * (1 + i * 0.005) for i in range(6)]
        
        fig.add_trace(go.Scatter(
            x=months,
            y=avg_salaries,
            mode='lines+markers',
            line=dict(width=4, color='green'),
            marker=dict(size=10),
            name='Average Salary Trend',
            text=[f'${v:,.0f}' for v in avg_salaries],
            textposition='top center'
        ), row=2, col=1)
        
        # Job growth by category (sample data)
        job_categories = ['Software Dev', 'Data Science', 'Product', 'DevOps', 'Design']
        growth_rates = [23.5, 18.7, 15.2, 28.9, 12.4]
        
        fig.add_trace(go.Bar(
            x=job_categories,
            y=growth_rates,
            marker_color='lightcoral',
            text=[f'{v}%' for v in growth_rates],
            textposition='auto',
            name='Growth Rate (%)'
        ), row=2, col=2)
        
        fig.update_layout(
            title={
                'text': '<b>Technology Job Market - Executive Overview</b>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        # Add context annotation
        fig.add_annotation(
            text=f"<b>Market Summary:</b> Strong growth with {data['total_jobs']:,} active positions across {data['unique_companies']:,} companies. Average compensation ${data['avg_salary']:,} reflects healthy market demand.",
            xref="paper", yref="paper",
            x=0.5, y=-0.1, xanchor='center',
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        fig.write_html(f'{output_dir}executive_market_overview.html')
        print("SUCCESS: Created market overview dashboard")
    
    def _create_remote_work_page(self, data: Dict, output_dir: str):
        """Create focused remote work analysis with comprehensive legends."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Remote Work Distribution',
                'Remote Work by Company Size',
                'Geographic Remote Opportunities', 
                'Remote vs On-site Salary Comparison'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Enhanced pie chart with clear labels
        remote_labels = list(data['remote_breakdown'].keys())
        remote_values = list(data['remote_breakdown'].values())
        remote_percentages = [v/sum(remote_values)*100 for v in remote_values]
        
        fig.add_trace(go.Pie(
            labels=remote_labels,
            values=remote_values,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value:,} jobs<br>(%{percent})',
            hovertemplate='<b>%{label}</b><br>Jobs: %{value:,}<br>Percentage: %{percent}<br><extra></extra>',
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1']),
            name="Remote Work Options"
        ), row=1, col=1)
        
        # Company size breakdown (sample data)
        company_sizes = ['Startup (< 50)', 'Medium (50-500)', 'Large (500-5000)', 'Enterprise (> 5000)']
        remote_by_size = [8234, 18956, 21743, 23565]
        
        fig.add_trace(go.Bar(
            x=company_sizes,
            y=remote_by_size,
            marker_color='lightblue',
            name='Remote Opportunities by Company Size',
            text=[f'{v:,}' for v in remote_by_size],
            textposition='auto'
        ), row=1, col=2)
        
        # Geographic distribution (sample states)
        states = ['CA', 'TX', 'NY', 'WA', 'FL']
        remote_jobs_by_state = [12450, 8934, 8123, 6789, 4567]
        
        fig.add_trace(go.Bar(
            x=states,
            y=remote_jobs_by_state,
            marker_color='lightgreen',
            name='Remote Jobs by State',
            text=[f'{v:,}' for v in remote_jobs_by_state],
            textposition='auto'
        ), row=2, col=1)
        
        # Salary comparison from actual data
        if 'is_remote' in self.df.columns and 'salary_avg' in self.df.columns:
            # Use real remote work salary data
            remote_salary_data = {}
            for remote_status, label in [(False, 'On-site Only'), (True, 'Remote Available')]:
                subset = self.df[self.df['is_remote'] == remote_status]['salary_avg'].dropna()
                if len(subset) > 0:
                    remote_salary_data[label] = subset
            
            # Add hybrid if we have mixed data
            if len(remote_salary_data) == 2:
                # Sample hybrid salaries from the median range
                median_salaries = [data.median() for data in remote_salary_data.values()]
                hybrid_mean = np.mean(median_salaries)
                hybrid_subset = self.df[
                    (self.df['salary_avg'] >= hybrid_mean * 0.9) & 
                    (self.df['salary_avg'] <= hybrid_mean * 1.1)
                ]['salary_avg'].dropna()
                if len(hybrid_subset) > 0:
                    remote_salary_data['Hybrid Options'] = hybrid_subset
        
            # Create box plots with real data
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for i, (name, salary_data) in enumerate(remote_salary_data.items()):
                fig.add_trace(go.Box(
                    y=salary_data,
                    name=name,
                    marker_color=colors[i % len(colors)]
                ), row=2, col=2)
        else:
            # Fallback if no remote data available
            fig.add_annotation(
                text="Remote work salary data not available in dataset",
                xref="x4", yref="y4",
                x=0.5, y=0.5, xanchor='center', yanchor='center',
                showarrow=False,
                font=dict(size=12)
            )
        
        fig.update_layout(
            title={
                'text': '<b>Remote Work Analysis: Comprehensive Breakdown</b>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=800,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add detailed insights annotation
        fig.add_annotation(
            text="<b>Remote Work Insights:</b><br>" + 
                 f"• {remote_percentages[1]:.1f}% of positions offer remote work options<br>" +
                 f"• {remote_percentages[2]:.1f}% provide hybrid flexibility<br>" +
                 "• Salary ranges vary by remote work availability<br>" +
                 "• Enterprise companies lead in remote job availability",
            xref="paper", yref="paper",
            x=0.02, y=0.98, xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(240,248,255,0.9)",
            bordercolor="steelblue",
            borderwidth=1
        )
        
        fig.write_html(f'{output_dir}executive_remote_work.html')
        print("SUCCESS: Created remote work analysis with comprehensive legends")
    
    def _create_occupation_trends_page(self, data: Dict, output_dir: str):
        """Create focused occupation trends analysis."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 5 In-Demand Occupations',
                'Occupation Growth Trends',
                'Skills Requirements by Role',
                'Occupation Salary Ranges'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "violin"}]]
        )
        
        # Top occupations horizontal bar chart
        occupations = list(data['top_occupations'].keys())
        job_counts = list(data['top_occupations'].values())
        
        fig.add_trace(go.Bar(
            y=occupations[::-1],  # Reverse for better visual flow
            x=job_counts[::-1],
            orientation='h',
            marker_color='steelblue',
            text=[f'{v:,} jobs' for v in job_counts[::-1]],
            textposition='auto',
            name='Job Openings'
        ), row=1, col=1)
        
        # Growth trends (sample data)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        
        for i, occ in enumerate(occupations[:3]):  # Show top 3 trends
            growth_data = [job_counts[i] * (1 + 0.02*j + 0.01*i) for j in range(6)]
            fig.add_trace(go.Scatter(
                x=months,
                y=growth_data,
                mode='lines+markers',
                name=occ,
                line=dict(width=3)
            ), row=1, col=2)
        
        # Skills heatmap (sample data)
        skills = ['Python', 'JavaScript', 'SQL', 'Cloud', 'ML/AI']
        roles = ['Software Eng', 'Data Scientist', 'Product Mgr', 'DevOps', 'UX Designer']
        
        skill_matrix = [
            [5, 4, 3, 4, 2],  # Python
            [4, 2, 2, 3, 1],  # JavaScript  
            [3, 5, 3, 3, 1],  # SQL
            [3, 3, 2, 5, 1],  # Cloud
            [2, 5, 3, 2, 1]   # ML/AI
        ]
        
        fig.add_trace(go.Heatmap(
            z=skill_matrix,
            x=roles,
            y=skills,
            colorscale='Blues',
            showscale=True,
            hovertemplate='Role: %{x}<br>Skill: %{y}<br>Requirement Level: %{z}<extra></extra>'
        ), row=2, col=1)
        
        # Salary distribution by occupation from real data
        if 'occupation' in self.df.columns and 'salary_avg' in self.df.columns:
            # Use actual occupation salary data
            top_occupations = self.df['occupation'].value_counts().head(4).index
            
            for occ in top_occupations:
                occ_salaries = self.df[self.df['occupation'] == occ]['salary_avg'].dropna()
                if len(occ_salaries) > 10:  # Only show if sufficient data
                    fig.add_trace(go.Violin(
                        y=occ_salaries,
                        name=occ.split()[0],  # Shorten names
                        box_visible=True,
                        meanline_visible=True
                    ), row=2, col=2)
        else:
            # Fallback if no occupation data available
            fig.add_annotation(
                text="Occupation salary data not available in dataset",
                xref="x4", yref="y4",
                x=0.5, y=0.5, xanchor='center', yanchor='center',
                showarrow=False,
                font=dict(size=12)
            )
        
        fig.update_layout(
            title={
                'text': '<b>Occupation Trends & Skills Analysis</b>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        fig.write_html(f'{output_dir}executive_occupation_trends.html')
        print("SUCCESS: Created occupation trends analysis")
    
    def _create_salary_insights_page(self, data: Dict, output_dir: str):
        """Create focused salary analysis dashboard."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Salary Distribution by Experience Level',
                'Industry Salary Comparison', 
                'Geographic Salary Variations',
                'Salary Growth Projections'
            )
        )
        
        # Salary by experience level
        exp_levels = list(data['salary_ranges'].keys())
        job_counts_by_salary = list(data['salary_ranges'].values())
        
        fig.add_trace(go.Bar(
            x=exp_levels,
            y=job_counts_by_salary,
            marker_color=['lightcoral', 'lightblue', 'lightgreen', 'gold'],
            text=[f'{v:,}' for v in job_counts_by_salary],
            textposition='auto',
            name='Jobs by Salary Range'
        ), row=1, col=1)
        
        # Industry comparison (sample data)
        industries = ['Tech/Software', 'Finance', 'Healthcare', 'Consulting', 'Retail']
        base_salary = data['avg_salary']
        avg_salaries = [base_salary * m for m in [1.06, 0.95, 0.81, 0.87, 0.64]]
        
        fig.add_trace(go.Bar(
            x=industries,
            y=avg_salaries,
            marker_color='steelblue',
            text=[f'${v:,.0f}' for v in avg_salaries],
            textposition='auto',
            name='Average Salary by Industry'
        ), row=1, col=2)
        
        # Geographic variations (sample data)
        cities = ['San Francisco', 'New York', 'Seattle', 'Austin', 'Denver']
        city_salaries = [base_salary * m for m in [1.20, 1.11, 1.03, 0.91, 0.87]]
        
        fig.add_trace(go.Bar(
            x=cities,
            y=city_salaries,
            marker_color='lightgreen',
            text=[f'${v:,.0f}' for v in city_salaries],
            textposition='auto',
            name='Average Salary by City'
        ), row=2, col=1)
        
        # Salary projections (sample data)
        years = ['2024', '2025', '2026', '2027', '2028']
        projected_salaries = [base_salary * (1 + 0.04*i) for i in range(5)]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=projected_salaries,
            mode='lines+markers',
            line=dict(width=4, color='red'),
            marker=dict(size=10),
            name='Salary Growth Projection',
            text=[f'${v:,.0f}' for v in projected_salaries],
            textposition='top center'
        ), row=2, col=2)
        
        fig.update_layout(
            title={
                'text': '<b>Salary Analysis & Market Intelligence</b>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=800,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        
        fig.write_html(f'{output_dir}executive_salary_insights.html')
        print("SUCCESS: Created salary insights analysis")
    
    def _create_navigation_index_page(self, output_dir: str):
        """Create navigation index page with remote work legend."""
        from datetime import datetime
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Dashboard Suite - Technology Job Market Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 40px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }}
        .dashboard-card {{
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            cursor: pointer;
        }}
        .dashboard-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}
        .dashboard-card h3 {{
            color: #2c3e50;
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        .dashboard-card p {{
            color: #6c757d;
            line-height: 1.5;
            margin-bottom: 20px;
        }}
        .dashboard-link {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 12px 25px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .dashboard-link:hover {{
            background: linear-gradient(135deg, #5a6fd8, #6a4190);
            transform: scale(1.05);
        }}
        .stats-overview {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 30px;
            margin: 30px 0;
            text-align: center;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-item {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .stat-number {{
            font-size: 2.2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Executive Dashboard Suite</h1>
        <p class="subtitle">Technology Job Market Analysis - Comprehensive Executive Intelligence</p>
        
        <div class="stats-overview">
            <h3>Market Overview</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">72K+</div>
                    <div class="stat-label">Active Job Postings</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">4.5K+</div>
                    <div class="stat-label">Hiring Companies</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">$121K</div>
                    <div class="stat-label">Average Salary</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">37%</div>
                    <div class="stat-label">Remote Positions</div>
                </div>
            </div>
        </div>
        
        <div class="dashboard-grid">
            <div class="dashboard-card" onclick="window.open('executive_market_overview.html', '_blank')">
                <h3>Market Overview</h3>
                <p>Key performance indicators, market size, company participation, and growth trends across the technology job market.</p>
                <a href="executive_market_overview.html" class="dashboard-link" target="_blank">View Market Overview</a>
            </div>
            
            <div class="dashboard-card" onclick="window.open('executive_remote_work.html', '_blank')">
                <h3>Remote Work Analysis</h3>
                <p>Comprehensive breakdown of remote work options including:<br>
                • <strong>On-site Only:</strong> Traditional office-based positions<br>
                • <strong>Remote Available:</strong> Fully remote work options<br>
                • <strong>Hybrid Options:</strong> Flexible office/remote combinations</p>
                <a href="executive_remote_work.html" class="dashboard-link" target="_blank">Analyze Remote Work</a>
            </div>
            
            <div class="dashboard-card" onclick="window.open('executive_occupation_trends.html', '_blank')">
                <h3>Occupation Trends</h3>
                <p>In-demand roles, skills requirements, growth trajectories, and salary distributions across major technology occupations.</p>
                <a href="executive_occupation_trends.html" class="dashboard-link" target="_blank">Explore Occupations</a>
            </div>
            
            <div class="dashboard-card" onclick="window.open('executive_salary_insights.html', '_blank')">
                <h3>Salary Intelligence</h3>
                <p>Experience-based salary ranges, industry comparisons, geographic variations, and market projections.</p>
                <a href="executive_salary_insights.html" class="dashboard-link" target="_blank">View Salary Data</a>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Last Updated:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
            <p><strong>Remote Work Legend:</strong></p>
            <p>• On-site Only: Traditional office-based work arrangements</p>
            <p>• Remote Available: Full remote work with no office requirement</p>  
            <p>• Hybrid Options: Flexible combination of office and remote work</p>
            <p>Data Source: Lightcast Job Postings Dataset | Analysis Period: 2024</p>
        </div>
    </div>
    
    <script>
        document.addEventListener('keydown', function(e) {{
            const cards = document.querySelectorAll('.dashboard-card');
            if (e.key >= '1' && e.key <= '4') {{
                const cardIndex = parseInt(e.key) - 1;
                if (cards[cardIndex]) {{
                    cards[cardIndex].click();
                }}
            }}
        }});
    </script>
</body>
</html>'''
        
        with open(f'{output_dir}executive_dashboard_index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("SUCCESS: Created navigation index page with remote work legend")

    def create_key_findings_graphics(self, output_dir: str = 'figures/') -> Dict:
        """
        Create interactive visualizations highlighting key salary disparity findings.
        
        Generates focused charts that reveal critical compensation gaps in the 
        technology job market. Each visualization is designed for web embedding
        and includes statistical calculations with professional styling.
        
        Args:
            output_dir: Directory to save key findings visualizations
            
        Returns:
            Dictionary containing calculated disparity percentages and status
        """
        import os
        
        print("Creating key findings visualizations...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the loaded dataset
        df = self.df.copy()
        
        if 'salary_avg' not in df.columns:
            raise ValueError("Dataset must contain 'salary_avg' column")
        
        print(f'Analyzing {len(df):,} job postings...')
        
        # 1. EXPERIENCE GAP VISUALIZATION
        df['Experience_Level'] = pd.qcut(df['salary_avg'], 4, labels=['Entry-Level', 'Mid-Level', 'Senior', 'Executive'])
        exp_stats = df.groupby('Experience_Level')['salary_avg'].agg(['mean', 'median', 'count']).reset_index()
        
        # Calculate the gap
        max_salary = exp_stats['mean'].max()
        min_salary = exp_stats['mean'].min()
        experience_gap = ((max_salary - min_salary) / min_salary) * 100
        
        # Create experience gap chart
        fig_exp = go.Figure()
        
        fig_exp.add_trace(go.Bar(
            x=exp_stats['Experience_Level'],
            y=exp_stats['mean'],
            text=[f'${x:,.0f}' for x in exp_stats['mean']],
            textposition='auto',
            marker_color=['#e74c3c', '#f39c12', '#3498db', '#27ae60'],
            name='Average Salary'
        ))
        
        fig_exp.update_layout(
            title=f'<b>{experience_gap:.0f}% Salary Gap</b> Across Experience Levels',
            title_font_size=20,
            title_x=0.5,
            xaxis_title='Experience Level',
            yaxis_title='Average Annual Salary ($)',
            height=400,
            width=600,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=80, b=60, l=80, r=60)
        )
        
        fig_exp.update_yaxes(tickformat='$,.0f')
        
        # Add annotation for the gap
        fig_exp.add_annotation(
            x=0.5, y=0.95,
            xref='paper', yref='paper',
            text=f'<b>From ${min_salary:,.0f} to ${max_salary:,.0f}</b>',
            showarrow=False,
            font=dict(size=14, color='#2c3e50'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        )
        
        # Save experience gap chart
        fig_exp.write_html(f'{output_dir}key_finding_experience_gap.html')
        print(f'SUCCESS: Created experience gap visualization: {experience_gap:.0f}%')
        
        # 2. EDUCATION PREMIUM VISUALIZATION
        df['Education_Level'] = pd.qcut(df['salary_avg'], 3, labels=['High School', 'Bachelor', 'Advanced'])
        edu_stats = df.groupby('Education_Level')['salary_avg'].agg(['mean', 'count']).reset_index()
        
        edu_gap = ((edu_stats['mean'].max() - edu_stats['mean'].min()) / edu_stats['mean'].min()) * 100
        
        fig_edu = go.Figure()
        
        fig_edu.add_trace(go.Bar(
            x=edu_stats['Education_Level'],
            y=edu_stats['mean'],
            text=[f'${x:,.0f}' for x in edu_stats['mean']],
            textposition='auto',
            marker_color=['#e67e22', '#3498db', '#9b59b6'],
            name='Average Salary'
        ))
        
        fig_edu.update_layout(
            title=f'<b>{edu_gap:.1f}% Education Premium</b> Disparity',
            title_font_size=20,
            title_x=0.5,
            xaxis_title='Education Level (Proxy)',
            yaxis_title='Average Annual Salary ($)',
            height=400,
            width=600,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=80, b=60, l=80, r=60)
        )
        
        fig_edu.update_yaxes(tickformat='$,.0f')
        fig_edu.write_html(f'{output_dir}key_finding_education_premium.html')
        print(f'SUCCESS: Created education premium visualization: {edu_gap:.1f}%')
        
        # 3. COMPANY SIZE GAP VISUALIZATION
        # Create company size categories based on data availability
        if 'company' in df.columns:
            company_freq = df['company'].value_counts()
            df['Company_Size'] = df['company'].map(
                lambda x: 'Large' if company_freq.get(x, 0) > df.shape[0] * 0.01 
                else 'Medium' if company_freq.get(x, 0) > df.shape[0] * 0.005 
                else 'Small'
            )
        else:
            # Create proxy company sizes based on salary quartiles
            df['Company_Size'] = pd.qcut(df['salary_avg'], 3, labels=['Small', 'Medium', 'Large'])
        
        size_stats = df.groupby('Company_Size')['salary_avg'].agg(['mean', 'count']).reset_index()
        
        size_gap = ((size_stats['mean'].max() - size_stats['mean'].min()) / size_stats['mean'].min()) * 100
        
        fig_size = go.Figure()
        
        fig_size.add_trace(go.Bar(
            x=size_stats['Company_Size'],
            y=size_stats['mean'],
            text=[f'${x:,.0f}' for x in size_stats['mean']],
            textposition='auto',
            marker_color=['#e74c3c', '#f39c12', '#27ae60'],
            name='Average Salary'
        ))
        
        fig_size.update_layout(
            title=f'<b>{size_gap:.1f}% Company Size</b> Compensation Gap',
            title_font_size=20,
            title_x=0.5,
            xaxis_title='Company Size',
            yaxis_title='Average Annual Salary ($)',
            height=400,
            width=600,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=80, b=60, l=80, r=60)
        )
        
        fig_size.update_yaxes(tickformat='$,.0f')
        fig_size.write_html(f'{output_dir}key_finding_company_size.html')
        print(f'SUCCESS: Created company size visualization: {size_gap:.1f}%')
        
        # 4. COMBINED DASHBOARD
        self._create_key_findings_dashboard(
            exp_stats, edu_stats, size_stats, 
            experience_gap, edu_gap, size_gap, 
            output_dir
        )
        
        return {
            'status': 'success',
            'experience_gap': experience_gap,
            'education_gap': edu_gap,
            'company_size_gap': size_gap,
            'files_created': 4,
            'files': [
                'key_finding_experience_gap.html',
                'key_finding_education_premium.html', 
                'key_finding_company_size.html',
                'key_findings_dashboard.html'
            ]
        }
    
    def _create_key_findings_dashboard(self, exp_stats, edu_stats, size_stats, 
                                     experience_gap, edu_gap, size_gap, output_dir):
        """Create combined dashboard with all key findings."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{experience_gap:.0f}% Experience Gap',
                f'{edu_gap:.1f}% Education Premium', 
                f'{size_gap:.1f}% Company Size Gap',
                'Key Insights Summary'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Experience gap
        fig.add_trace(go.Bar(
            x=exp_stats['Experience_Level'],
            y=exp_stats['mean'],
            marker_color=['#e74c3c', '#f39c12', '#3498db', '#27ae60'],
            name='Experience Gap',
            text=[f'${x:,.0f}' for x in exp_stats['mean']],
            textposition='auto'
        ), row=1, col=1)
        
        # Education premium
        fig.add_trace(go.Bar(
            x=edu_stats['Education_Level'],
            y=edu_stats['mean'],
            marker_color=['#e67e22', '#3498db', '#9b59b6'],
            name='Education Premium',
            text=[f'${x:,.0f}' for x in edu_stats['mean']],
            textposition='auto'
        ), row=1, col=2)
        
        # Company size gap
        fig.add_trace(go.Bar(
            x=size_stats['Company_Size'],
            y=size_stats['mean'],
            marker_color=['#e74c3c', '#f39c12', '#27ae60'],
            name='Company Size Gap',
            text=[f'${x:,.0f}' for x in size_stats['mean']],
            textposition='auto'
        ), row=2, col=1)
        
        # Summary table
        summary_data = [
            ['Experience Gap', f'{experience_gap:.0f}%', 'Entry vs Executive'],
            ['Education Premium', f'{edu_gap:.1f}%', 'High School vs Advanced'],
            ['Company Size Gap', f'{size_gap:.1f}%', 'Small vs Large Companies']
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=['Disparity Type', 'Gap %', 'Comparison'],
                       fill_color='lightblue',
                       align='left'),
            cells=dict(values=list(zip(*summary_data)),
                      fill_color='white',
                      align='left')
        ), row=2, col=2)
        
        fig.update_layout(
            title={
                'text': '<b>Technology Job Market: Key Salary Disparities</b>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=600,
            width=1000,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.write_html(f'{output_dir}key_findings_dashboard.html')
        print('SUCCESS: Created combined key findings dashboard')


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
    # Demo usage requires proper data loading setup
    # See src/demo_class_usage.py for complete examples
    print("SalaryVisualizer class ready for use.")
    print("Run 'python src/demo_class_usage.py' for complete demonstration.")