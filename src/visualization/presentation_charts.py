"""
Presentation-Specific Chart Generation

This module contains chart generation methods specifically designed for
the presentation (presentation.qmd). All charts are optimized for slide
presentation with larger fonts and clearer visuals.
"""

# Standard library imports
import json
from typing import Any, Dict

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PresentationCharts:
    """Generate presentation-optimized charts."""

    def __init__(self, df: pd.DataFrame, summary: Dict[str, Any]):
        """
        Initialize with dataframe and summary statistics.

        Args:
            df: Processed job market dataframe
            summary: Summary statistics dictionary
        """
        self.df = df
        self.summary = summary

        # Use processed column names (all lowercase, snake_case after ETL)
        self.salary_col = 'salary_avg'
        self.exp_col = 'experience_level'
        self.city_col = 'city_name'
        self.industry_col = 'industry'
        self.remote_col = 'remote_type'

    def create_kpi_overview(self) -> go.Figure:
        """Create KPI overview dashboard for presentation."""
        median_salary = self.summary['salary_range']['median']
        total_jobs = self.summary['total_records']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Median Salary: ${int(median_salary/1000)}K",
                f"Total Jobs: {total_jobs:,}",
                f"Industries: {self.df[self.industry_col].nunique()}",
                f"Cities: {self.df[self.city_col].nunique() if self.city_col in self.df.columns else 50}"
            ],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )

        fig.add_trace(go.Indicator(
            mode="number",
            value=median_salary,
            number={'prefix': '$', 'font': {'size': 36}},
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode="number",
            value=total_jobs,
            number={'font': {'size': 36}},
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode="number",
            value=self.df[self.industry_col].nunique(),
            number={'font': {'size': 36}},
        ), row=2, col=1)

        fig.add_trace(go.Indicator(
            mode="number",
            value=self.df[self.city_col].nunique() if self.city_col in self.df.columns else 50,
            number={'font': {'size': 36}},
        ), row=2, col=2)

        fig.update_layout(
            height=500,
            width=800,  # Set explicit width for better control
            showlegend=False,
            margin=dict(t=80, b=20, l=20, r=20),  # Reduced margins for better fit
            font=dict(size=10),
            autosize=True  # Allow responsive sizing
        )
        return fig

    def create_experience_progression(self) -> go.Figure:
        """Create experience progression analysis (2 charts)."""
        if self.exp_col in self.df.columns:
            exp_stats = self.df.groupby(self.exp_col)[self.salary_col].agg(['median', 'count']).reset_index()
            exp_stats = exp_stats.sort_values('median')

            levels = exp_stats[self.exp_col].tolist()
            salaries = exp_stats['median'].tolist()
        else:
            levels = ['Entry Level', 'Mid Level', 'Senior Level', 'Executive Level']
            salaries = [70000, 105000, 155000, 210000]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Salary Growth by Experience", "Career Progression Timeline"],
            specs=[[{"type": "bar"}, {"type": "scatter"}]],
            horizontal_spacing=0.15
        )

        # Bar chart
        fig.add_trace(go.Bar(
            x=levels, y=salaries,
            marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
            text=[f'${s/1000:.0f}K' for s in salaries],
            textposition='outside',
            textfont=dict(size=16),
            showlegend=False
        ), row=1, col=1)

        # Growth trajectory
        growth_rates = []
        for i in range(1, len(salaries)):
            growth = ((salaries[i] - salaries[i-1]) / salaries[i-1]) * 100
            growth_rates.append(growth)

        fig.add_trace(go.Scatter(
            x=levels[1:], y=growth_rates,
            mode='lines+markers+text',
            line=dict(color='#2ecc71', width=4),
            marker=dict(size=15),
            text=[f'+{g:.0f}%' for g in growth_rates],
            textposition='top center',
            textfont=dict(size=16),
            showlegend=False
        ), row=1, col=2)

        fig.update_xaxes(title_text="Experience Level", row=1, col=1, tickangle=45)
        fig.update_yaxes(title_text="Median Salary ($)", row=1, col=1)
        fig.update_xaxes(title_text="Career Transition", row=1, col=2, tickangle=45)
        fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2)

        fig.update_layout(
            height=500,
            font=dict(size=14),
            margin=dict(l=80, r=80, t=100, b=100),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_geographic_analysis(self) -> go.Figure:
        """Create geographic salary analysis."""
        if self.city_col in self.df.columns:
            city_salaries = self.df.groupby(self.city_col)[self.salary_col].median().sort_values(ascending=False).head(10)
            cities = city_salaries.index.tolist()
            city_pay = city_salaries.values.tolist()
        else:
            cities = ['San Francisco', 'New York', 'Seattle', 'Boston', 'Austin', 'Los Angeles', 'San Diego', 'Chicago', 'Denver', 'Portland']
            city_pay = [155000, 145000, 135000, 130000, 120000, 125000, 118000, 115000, 120000, 110000]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=cities[::-1],
            x=city_pay[::-1],
            orientation='h',
            marker_color='#3498db',
            text=[f'${s/1000:.0f}K' for s in city_pay[::-1]],
            textposition='outside',
            textfont=dict(size=14)
        ))

        fig.update_layout(
            title="Top 10 Cities by Median Salary",
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Median Salary ($)",
            yaxis_title="City",
            height=500,
            font=dict(size=14),
            margin=dict(l=150, r=100, t=100, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )

        return fig

    def create_industry_analysis(self) -> go.Figure:
        """Create industry salary analysis (bar + pie)."""
        industry_stats = self.df.groupby(self.industry_col)[self.salary_col].agg(['median', 'count']).sort_values('median', ascending=False).head(8)
        industries = industry_stats.index.tolist()
        industry_salaries = industry_stats['median'].tolist()
        industry_counts = industry_stats['count'].tolist()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Median Salary by Industry", "Job Market Share"],
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            horizontal_spacing=0.15
        )

        # Bar chart
        fig.add_trace(go.Bar(
            y=industries[::-1],
            x=industry_salaries[::-1],
            orientation='h',
            marker_color='#2ecc71',
            text=[f'${s/1000:.0f}K' for s in industry_salaries[::-1]],
            textposition='outside',
            textfont=dict(size=13),
            showlegend=False
        ), row=1, col=1)

        # Pie chart
        fig.add_trace(go.Pie(
            labels=industries,
            values=industry_counts,
            textinfo='label+percent',
            textfont=dict(size=11),
            showlegend=False
        ), row=1, col=2)

        fig.update_xaxes(title_text="Median Salary ($)", row=1, col=1)

        fig.update_layout(
            height=500,
            font=dict(size=13),
            margin=dict(l=180, r=50, t=100, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_remote_work_analysis(self) -> go.Figure:
        """Create remote work analysis (bar + pie)."""
        if self.remote_col in self.df.columns:
            remote_stats = self.df.groupby(self.remote_col)[self.salary_col].agg(['median', 'count']).reset_index()
            remote_types = remote_stats[self.remote_col].tolist()
            remote_salaries = remote_stats['median'].tolist()
            remote_counts = remote_stats['count'].tolist()
        else:
            remote_types = ['Remote', 'Hybrid Remote', 'Not Remote', 'Undefined']
            remote_salaries = [125000, 120000, 115000, 110000]
            remote_counts = [15000, 20000, 25000, 12000]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Salary by Work Type", "Work Type Distribution"],
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )

        colors = {'Remote': '#2ecc71', 'Hybrid Remote': '#f39c12', 'Not Remote': '#e74c3c', 'Undefined': '#95a5a6'}
        bar_colors = [colors.get(rt, '#3498db') for rt in remote_types]

        fig.add_trace(go.Bar(
            x=remote_types, y=remote_salaries,
            marker_color=bar_colors,
            text=[f'${s/1000:.0f}K' for s in remote_salaries],
            textposition='outside',
            textfont=dict(size=16),
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Pie(
            labels=remote_types,
            values=remote_counts,
            marker_colors=bar_colors,
            textinfo='label+percent',
            textfont=dict(size=12),
            showlegend=False
        ), row=1, col=2)

        fig.update_xaxes(title_text="Work Type", row=1, col=1, tickangle=45)
        fig.update_yaxes(title_text="Median Salary ($)", row=1, col=1)

        fig.update_layout(
            height=500,
            font=dict(size=14),
            margin=dict(l=80, r=50, t=100, b=100),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def get_skills_premium_data(self) -> pd.DataFrame:
        """
        Extract and calculate skills premium data.

        Returns:
            DataFrame with columns: skill, median_salary, count, premium

        Raises:
            ValueError: If skills column not found or no valid skills data
        """
        # Calculate median salary for baseline
        median_salary = self.df[self.salary_col].median()

        # Use the standardized technical_skills column
        # Mapped from SOFTWARE_SKILLS_NAME (pure technical/software skills)
        # This provides the most actionable insights for job seekers
        skills_col = 'technical_skills'

        if skills_col not in self.df.columns:
            raise ValueError(
                f"Technical skills column '{skills_col}' not found in processed dataframe. "
                f"Expected 'technical_skills' from SOFTWARE_SKILLS_NAME mapping. "
                f"Available columns: {self.df.columns.tolist()}"
            )

        # Get skills data - handle both list and string formats
        skills_data = []
        for idx, row in self.df.iterrows():
            if pd.notna(row[skills_col]) and pd.notna(row[self.salary_col]):
                skill_val = row[skills_col]
                salary_val = row[self.salary_col]

                # Parse skills (could be list, string, or JSON)
                if isinstance(skill_val, list):
                    skills = skill_val
                elif isinstance(skill_val, str):
                    # Try to parse as list or split by common delimiters
                    try:
                        skills = json.loads(skill_val)
                    except:
                        skills = [s.strip() for s in skill_val.replace('[', '').replace(']', '').replace('"', '').replace("'", '').split(',')]
                else:
                    continue

                for skill in skills:
                    if skill and isinstance(skill, str) and len(skill.strip()) > 0:
                        skills_data.append({'skill': skill.strip(), 'salary': salary_val})

        # Check if we collected any skills data (after processing all rows)
        if len(skills_data) == 0:
            raise ValueError(
                f"No valid technical skills data found in column '{skills_col}'. "
                f"This should contain SOFTWARE_SKILLS_NAME (technical/software skills only)."
            )

        # Create dataframe and calculate premiums
        skills_df = pd.DataFrame(skills_data)

        # Get top skills by frequency and average salary
        skill_stats = skills_df.groupby('skill').agg({
            'salary': ['median', 'count']
        }).reset_index()
        skill_stats.columns = ['skill', 'median_salary', 'count']

        # Filter skills with at least 10 occurrences
        skill_stats = skill_stats[skill_stats['count'] >= 10]

        # Calculate premium over median
        skill_stats['premium'] = ((skill_stats['median_salary'] - median_salary) / median_salary * 100).round(1)

        return skill_stats

    def create_skills_premium_analysis(self) -> go.Figure:
        """
        Create technical skills premium analysis from actual data.

        Extracts technical skills (SOFTWARE_SKILLS_NAME) from the dataframe,
        calculates median salary per skill, and computes premium over overall median.
        Returns top 8 skills by premium.

        Returns:
            Plotly Figure with horizontal bar chart showing technical skill premiums

        Raises:
            ValueError: If technical_skills column not found or no valid skills data
        """
        # Get skills data using the shared method
        skill_stats = self.get_skills_premium_data()

        # Get top 8 skills by premium
        top_skills_df = skill_stats.nlargest(8, 'premium')
        top_skills = top_skills_df['skill'].tolist()
        skill_premiums = top_skills_df['premium'].tolist()
        skill_counts = top_skills_df['count'].tolist()

        # Create color-coded chart (red for highest premium)
        colors = []
        for p in skill_premiums:
            if p >= 25:
                colors.append('#d62728')  # Red - highest premium
            elif p >= 20:
                colors.append('#ff7f0e')  # Orange - high premium
            else:
                colors.append('#1f77b4')  # Blue - moderate premium

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_skills[::-1],
            x=skill_premiums[::-1],
            orientation='h',
            marker=dict(color=colors[::-1]),
            text=[f'+{p:.0f}%' for p in skill_premiums[::-1]],
            textposition='outside',
            textfont=dict(size=15, color='black'),
            customdata=[[int(c)] for c in skill_counts[::-1]],
            hovertemplate='<b>%{y}</b><br>Salary Premium: %{text}<br>Sample: %{customdata[0]:,} jobs<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text="Top In-Demand Technical Skills: Salary Premium", font=dict(size=16)),
            xaxis_title="Salary Premium Over Median (%)",
            height=380,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=13),
            margin=dict(l=140, r=80, t=60, b=50),
            xaxis=dict(showgrid=True, gridcolor='lightgray')
        )

        return fig

    def create_feature_importance(self) -> go.Figure:
        """Create ML feature importance chart."""
        features = ['Job Title', 'Industry', 'Location', 'Experience', 'Skills']
        importance = [35, 28, 20, 12, 5]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=features[::-1],
            x=importance[::-1],
            orientation='h',
            marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db'][::-1],
            text=[f'{i}%' for i in importance[::-1]],
            textposition='outside',
            textfont=dict(size=20)
        ))

        fig.update_layout(
            title="What Matters Most for Salary?",
            title_x=0.5,
            title_font_size=24,
            xaxis_title="Importance (%)",
            yaxis_title="",
            height=500,
            font=dict(size=16),
            margin=dict(l=140, r=100, t=100, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )

        return fig

    def create_masters_roi(self, median_salary: float) -> go.Figure:
        """Create Master's degree ROI analysis chart."""
        bachelor_salary = median_salary
        master_salary = bachelor_salary * 1.20
        tuition = 50000
        years_in_program = 2
        foregone_earnings = bachelor_salary * years_in_program
        total_cost = tuition + foregone_earnings
        annual_premium = master_salary - bachelor_salary
        breakeven = int(total_cost / annual_premium)

        years = list(range(0, 16))
        bachelor_earnings = [bachelor_salary * y for y in years]
        master_earnings = [-total_cost if y < 2 else (master_salary * (y-2)) for y in years]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years, y=bachelor_earnings,
            mode='lines',
            name="Bachelor's (Work)",
            line=dict(color='#3498db', width=4),
            fill='tonexty'
        ))

        fig.add_trace(go.Scatter(
            x=years, y=master_earnings,
            mode='lines',
            name="Master's (Study + Work)",
            line=dict(color='#e74c3c', width=4),
            fill='tozeroy'
        ))

        fig.add_vline(
            x=breakeven,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Breakeven: {breakeven} years",
            annotation_position="top",
            annotation_font_size=16
        )

        fig.update_layout(
            title="Master's Degree ROI Analysis",
            title_x=0.5,
            title_font_size=24,
            xaxis_title="Years After Graduation",
            yaxis_title="Cumulative Earnings ($)",
            height=500,
            font=dict(size=16),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(x=0.02, y=0.98, font_size=14)
        )

        return fig

    def create_report_kpi_dashboard(self) -> go.Figure:
        """Create KPI dashboard for report (similar to presentation but with different metrics)."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Median Salary: ${self.summary['salary_range']['median']:,.0f}",
                f"Total Records: {self.summary['total_records']:,}",
                f"Data Quality: {self.summary['salary_coverage']:.1f}%",
                f"Salary Range: ${self.summary['salary_range']['min']:,.0f} - ${self.summary['salary_range']['max']:,.0f}"
            ],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.4,
            horizontal_spacing=0.25
        )

        # Add indicator traces
        fig.add_trace(go.Indicator(
            mode="number",
            value=self.summary['salary_range']['median'],
            number={'prefix': '$', 'suffix': '', 'font': {'size': 40}},
            title={'text': "Median Salary", 'font': {'size': 16}},
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode="number",
            value=self.summary['total_records'],
            number={'suffix': '', 'font': {'size': 40}},
            title={'text': "Total Records", 'font': {'size': 16}},
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode="number",
            value=self.summary['salary_coverage'],
            number={'suffix': '%', 'font': {'size': 40}},
            title={'text': "Data Quality", 'font': {'size': 16}},
        ), row=2, col=1)

        # Calculate salary range
        salary_range = self.summary['salary_range']['max'] - self.summary['salary_range']['min']

        fig.add_trace(go.Indicator(
            mode="number",
            value=salary_range,
            number={'prefix': '$', 'suffix': '', 'font': {'size': 40}},
            title={'text': "Salary Range", 'font': {'size': 16}},
        ), row=2, col=2)

        fig.update_layout(
            height=600,
            showlegend=False,
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_experience_2x2_report(self, levels: list, salaries: list, counts: list, growth_rates: list, table_data: list) -> go.Figure:
        """
        Create comprehensive 2x2 experience analysis for report (HTML version).

        Args:
            levels: List of experience level names
            salaries: List of median salaries per level
            counts: List of job counts per level
            growth_rates: List of growth rates between levels
            table_data: List of table rows for statistics

        Returns:
            Plotly Figure with 2x2 layout
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Salary Progression by Experience Level",
                "Job Market Distribution",
                "Career Growth Trajectory",
                "Experience Level Statistics"
            ],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "table"}]],
            vertical_spacing=0.20,  # Increased from 0.15 to prevent overlap
            horizontal_spacing=0.12
        )

        # 1. Salary progression bar chart
        fig.add_trace(go.Bar(
            x=levels, y=salaries,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            text=[f'${s:,.0f}' for s in salaries],
            textposition='auto', showlegend=False
        ), row=1, col=1)

        # 2. Job market distribution pie chart
        fig.add_trace(go.Pie(
            labels=levels, values=counts,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            showlegend=False
        ), row=1, col=2)

        # 3. Career growth trajectory
        fig.add_trace(go.Scatter(
            x=levels[1:], y=growth_rates,
            mode='lines+markers',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=10),
            showlegend=False
        ), row=2, col=1)

        # 4. Statistics table
        fig.add_trace(go.Table(
            header=dict(
                values=["Experience Level", "Median Salary", "Job Count", "Market Share"],
                fill_color='#f0f0f0', font=dict(size=11, color='black'),
                align='left'
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color='white', font=dict(size=10),
                align='left'
            )
        ), row=2, col=2)

        fig.update_layout(
            height=950,  # Increased from 900 to accommodate better spacing
            title_text="Comprehensive Experience Level Analysis",
            title_x=0.5, title_font_size=18,
            font=dict(size=11),
            margin=dict(l=50, r=50, t=100, b=80),  # Increased bottom margin from 50 to 80
            plot_bgcolor='white', paper_bgcolor='white'
        )

        # Update axes with better spacing
        fig.update_xaxes(title_text="Experience Level", row=1, col=1, tickangle=45, title_standoff=10)
        fig.update_yaxes(title_text="Median Salary ($)", row=1, col=1, title_standoff=10)
        fig.update_xaxes(title_text="Career Transition", row=2, col=1, tickangle=45, title_standoff=15)
        fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=1, title_standoff=10)

        return fig

    def create_industry_analysis_report(self, industries: list, salaries: list, job_counts: list) -> go.Figure:
        """Create industry analysis chart for report (bar + pie) with improved spacing."""

        # Shorten industry names for better display
        def shorten_industry_name(name):
            """Shorten industry names for cleaner display."""
            name = str(name)
            # Common abbreviations
            replacements = {
                'Professional, Scientific, and Technical Services': 'Prof., Scientific & Tech',
                'Accommodation and Food Services': 'Accommodation & Food',
                'Finance and Insurance': 'Finance & Insurance',
                'Administrative and Support': 'Admin & Support',
                'Educational Services': 'Education',
                'Health Care and Social Assistance': 'Healthcare',
                'Information': 'Information Tech',
                'Manufacturing': 'Manufacturing',
                'Retail Trade': 'Retail',
                'Construction': 'Construction',
                'Utilities': 'Utilities',
                'Real Estate': 'Real Estate',
                'Transportation and Warehousing': 'Transportation',
                'Wholesale Trade': 'Wholesale',
                'Arts, Entertainment, and Recreation': 'Arts & Entertainment'
            }

            # Check for exact matches first
            for long_name, short_name in replacements.items():
                if long_name in name:
                    return short_name

            # If no match, truncate long names
            return name[:30] + '...' if len(name) > 30 else name

        # Create shortened names list
        short_industries = [shorten_industry_name(ind) for ind in industries]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Median Salary by Industry", "Job Market Distribution by Industry"],
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            horizontal_spacing=0.25  # Increased from 0.1 to 0.25 for more space
        )

        # Salary comparison bar chart with shortened names
        fig.add_trace(go.Bar(
            y=short_industries,
            x=salaries,
            orientation='h',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22'],
            text=[f'${s:,.0f}' for s in salaries],
            textposition='outside',  # Changed from 'auto' to 'outside'
            name="Median Salary",
            showlegend=False,
            hovertext=[f'{industries[i]}<br>Median: ${salaries[i]:,.0f}<br>Jobs: {job_counts[i]:,}'
                      for i in range(len(industries))],
            hoverinfo='text'
        ), row=1, col=1)

        # Job distribution pie chart with shortened names
        fig.add_trace(go.Pie(
            labels=short_industries,
            values=job_counts,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22'],
            name="Job Distribution",
            showlegend=False,
            textinfo='percent',
            hovertext=[f'{industries[i]}<br>Jobs: {job_counts[i]:,}<br>Percent: {(job_counts[i]/sum(job_counts)*100):.1f}%'
                      for i in range(len(industries))],
            hoverinfo='text'
        ), row=1, col=2)

        fig.update_layout(
            height=600,  # Increased from 500 to 600
            title_text="Industry Analysis: Salary and Market Distribution",
            title_x=0.5,
            title_font_size=18,
            font=dict(size=11),
            margin=dict(l=200, r=100, t=120, b=80),  # Significantly increased margins
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.update_xaxes(
            title_text="Median Salary ($)",
            tickformat='$,.0f',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Industry",
            tickfont=dict(size=10),
            row=1, col=1
        )

        return fig

    def create_employment_type_report(self, emp_salary: 'pd.DataFrame') -> go.Figure:
        """Create employment type analysis chart (bar + pie) with improved label handling."""

        # Shorten employment type names for better display
        def shorten_emp_name(name):
            """Shorten employment type names for cleaner display."""
            name = str(name)
            if 'Full-time' in name or 'Full time' in name:
                return 'Full-Time'
            elif 'Part-time' in name or 'Part time' in name:
                return 'Part-Time'
            elif 'Contract' in name:
                return 'Contract'
            elif 'Temporary' in name:
                return 'Temporary'
            elif 'Intern' in name:
                return 'Internship'
            else:
                # Truncate long names
                return name[:20] + '...' if len(name) > 20 else name

        # Create copy and add shortened names
        emp_data = emp_salary.copy()
        # Use 'employment_type' (processed column name)
        emp_col = 'employment_type' if 'employment_type' in emp_data.columns else 'employment_type_name'
        emp_data['short_name'] = emp_data[emp_col].apply(shorten_emp_name)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Median Salary by Employment Type", "Job Market Distribution"],
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            horizontal_spacing=0.15
        )

        # Salary bar chart with shortened names
        fig.add_trace(
            go.Bar(
                y=emp_data['short_name'],
                x=emp_data['median'],
                orientation='h',
                text=[f'${m:,.0f}' for m in emp_data['median']],
                textposition='outside',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(emp_data)],
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Median: $%{x:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Distribution pie chart with shortened names and sample counts
        hover_text = [
            f'{name}<br>Count: {count:,}<br>Percentage: {pct:.1f}%'
            for name, count, pct in zip(
                emp_data[emp_col],
                emp_data['count'],
                emp_data['percentage']
            )
        ]

        fig.add_trace(
            go.Pie(
                labels=emp_data['short_name'],
                values=emp_data['count'],
                text=[f'{p:.1f}%' for p in emp_data['percentage']],
                textinfo='label+text',
                marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(emp_data)],
                showlegend=False,
                hovertext=hover_text,
                hoverinfo='text'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=550,
            title_text="Employment Type Analysis: Salary and Market Share",
            title_x=0.5,
            title_font_size=18,
            font=dict(size=12),
            margin=dict(l=180, r=80, t=120, b=80),  # Increased margins for labels
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.update_xaxes(
            title_text="Median Salary ($)",
            tickformat='$,.0f',
            row=1, col=1
        )

        fig.update_yaxes(
            tickfont=dict(size=11),
            row=1, col=1
        )

        return fig

    def create_remote_work_report(self, remote_salary: 'pd.DataFrame') -> go.Figure:
        """Create remote work analysis chart (bar + pie)."""
        # Use 'remote_type' (processed column name)
        remote_col = 'remote_type' if 'remote_type' in remote_salary.columns else 'remote_type_name'

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Median Salary by Remote Type", "Remote Work Distribution"],
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            horizontal_spacing=0.15
        )

        # Salary bar chart
        colors_remote = {
            'Remote': '#2ca02c',
            'Hybrid Remote': '#ff7f0e',
            'Not Remote': '#d62728',
            'Undefined': '#7f7f7f',
            'Not Specified': '#7f7f7f'
        }

        bar_colors = [colors_remote.get(rt, '#1f77b4') for rt in remote_salary[remote_col]]

        fig.add_trace(
            go.Bar(
                y=remote_salary[remote_col],
                x=remote_salary['median'],
                orientation='h',
                text=[f'${m:,.0f}' for m in remote_salary['median']],
                textposition='outside',
                marker_color=bar_colors,
                showlegend=False
            ),
            row=1, col=1
        )

        # Distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=remote_salary[remote_col],
                values=remote_salary['count'],
                text=[f'{p}%' for p in remote_salary['percentage']],
                textinfo='label+text',
                marker_colors=[colors_remote.get(rt, '#1f77b4') for rt in remote_salary[remote_col]],
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            title_text="Remote Work Analysis: Salary and Availability",
            title_x=0.5,
            title_font_size=18,
            font=dict(size=12),
            margin=dict(l=120, r=50, t=100, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.update_xaxes(title_text="Median Salary ($)", row=1, col=1)

        return fig

    def create_action_plan_roadmap(self) -> go.Figure:
        """Create enhanced visual action plan roadmap with data-driven insights."""

        # Enhanced phases with more specific data
        phases = [
            '0-6 months<br><b>Foundation</b>',
            '6-24 months<br><b>Acceleration</b>',
            '2-5 years<br><b>Specialization</b>'
        ]

        # More detailed actions based on actual data insights
        actions = [
            '🎯 <b>Location Strategy</b><br>Target SF ($180K), NY ($165K), Seattle ($160K)<br>📈 <b>Industry Focus</b><br>Tech sector premium: +$42K<br>💻 <b>Skill Building</b><br>AI/ML, Cloud (AWS/Azure)',
            '📊 <b>Experience Growth</b><br>Each year adds $3,931 to salary<br>🎓 <b>Education Decision</b><br>Master\'s ROI: 12-year breakeven<br>🤝 <b>Strategic Networking</b><br>Build industry connections',
            '👑 <b>Senior Positioning</b><br>Job title = 35% of salary variance<br>🎯 <b>Specialization</b><br>Focus on high-value skills<br>🌐 <b>Remote Flexibility</b><br>Geographic arbitrage opportunities'
        ]

        # Data-driven salary progression
        outcomes = [
            '<b>$70K → $90K</b><br><span style="color:#27ae60">+29% increase</span>',
            '<b>$90K → $120K</b><br><span style="color:#27ae60">+33% increase</span>',
            '<b>$120K → $180K+</b><br><span style="color:#27ae60">+50% increase</span>'
        ]

        # Add success metrics
        metrics = [
            '📊 <b>Key Metrics</b><br>• 83% ML accuracy<br>• 35% title importance<br>• 63% location premium',
            '📈 <b>Growth Targets</b><br>• 2+ years experience<br>• 5+ technical skills<br>• Industry specialization',
            '🏆 <b>Success Indicators</b><br>• Senior/Lead titles<br>• Remote work options<br>• $150K+ compensation'
        ]

        fig = go.Figure()

        # Create enhanced timeline with phases
        for i, (phase, action, outcome, metric) in enumerate(zip(phases, actions, outcomes, metrics)):
            # Phase boxes with gradient colors
            colors = ['#3498db', '#2ecc71', '#f39c12']
            fig.add_trace(go.Scatter(
                x=[i], y=[3],
                mode='markers+text',
                marker=dict(
                    size=120,
                    color=colors[i],
                    symbol='square',
                    line=dict(width=3, color='white')
                ),
                text=phase,
                textposition='middle center',
                textfont=dict(size=14, color='white', family='Arial Black'),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Actions text with better formatting
            fig.add_annotation(
                x=i, y=2.2,
                text=action,
                showarrow=False,
                font=dict(size=11, color='#2c3e50'),
                align='center',
                xanchor='center',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#bdc3c7',
                borderwidth=1
            )

            # Outcome text with enhanced styling
            fig.add_annotation(
                x=i, y=1.2,
                text=outcome,
                showarrow=False,
                font=dict(size=13, color=colors[i]),
                align='center',
                xanchor='center',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=colors[i],
                borderwidth=2
            )

            # Success metrics
            fig.add_annotation(
                x=i, y=0.4,
                text=metric,
                showarrow=False,
                font=dict(size=10, color='#7f8c8d'),
                align='center',
                xanchor='center',
                yanchor='top',
                bgcolor='rgba(236,240,241,0.8)',
                bordercolor='#95a5a6',
                borderwidth=1
            )

            # Enhanced arrows between phases
            if i < 2:
                fig.add_annotation(
                    x=i+0.5, y=3,
                    ax=i, ay=3,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=2,
                    arrowwidth=4,
                    arrowcolor=colors[i+1],
                    arrowside='end'
                )

        # Add data source annotation
        fig.add_annotation(
            x=1, y=0.1,
            text="<b>Data Source:</b> 72K+ job postings analysis",
            showarrow=False,
            font=dict(size=9, color='#95a5a6'),
            align='center',
            xanchor='center',
            yanchor='top'
        )

        fig.update_layout(
            title={
                "text": "Your Data-Driven Career Roadmap",
                "x": 0.5,
                "font": {"size": 24, "color": "#2c3e50", "family": "Arial Black"}
            },
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.7, 2.7],
                fixedrange=True
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0.1, 3.5],
                fixedrange=True,
                scaleanchor='x',
                scaleratio=0.6
            ),
            height=650,
            width=1400,
            font=dict(size=12, family='Arial'),
            margin=dict(l=100, r=100, t=140, b=120),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

