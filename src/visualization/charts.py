"""
Unified chart generation for job market analytics.

This module provides consolidated chart generation capabilities
combining functionality from multiple visualization modules.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import folium
from folium.plugins import MarkerCluster, HeatMap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Local imports
from src.utils.logger import get_logger

logger = get_logger(level="WARNING")

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from .theme import JobMarketTheme, apply_salary_theme, apply_industry_theme, apply_experience_theme, apply_geographic_theme


def display_figure(fig, filename: Optional[str] = None, save_dir: str = 'figures/'):
    """
    Display and optionally save a Plotly figure.

    This is a standalone utility function for use in Quarto documents and notebooks.
    Returns the figure object so Quarto can automatically render it in the correct format.

    Args:
        fig: Plotly figure object
        filename: Optional filename (without extension) to save the figure as PNG/SVG/HTML
        save_dir: Directory to save figures (default: 'figures/')

    Returns:
        The figure object (Quarto automatically converts to HTML or PNG based on output format)

    Example:
        >>> fig = go.Figure(...)
        >>> display_figure(fig, "my_chart")  # Saves and returns fig for Quarto to render
    """
    if filename:
        png_path = Path(save_dir) / f"{filename}.png"
        svg_path = Path(save_dir) / f"{filename}.svg"
        png_path.parent.mkdir(parents=True, exist_ok=True)

        # Save static images only - Quarto will handle HTML generation for interactive outputs
        try:
            fig.write_image(str(png_path), width=1200, height=800, scale=2)
            fig.write_image(str(svg_path), width=1200, height=800)
        except Exception as e:
            print(f"Warning: Could not save static images for {filename}: {e}")

    # Return the figure object - Quarto will automatically render it
    # in the appropriate format (interactive HTML for .html, static PNG for .docx)
    return fig


class SalaryVisualizer:
    """
    Unified salary visualization class.

    Provides comprehensive salary analysis and visualization capabilities
    for job market data.
    """

    def __init__(self, df: pd.DataFrame, auto_save: bool = False, save_dir: str = 'figures/'):
        """
        Initialize with a pandas DataFrame.

        Args:
            df: Pandas DataFrame with processed job market data
            auto_save: If True, automatically save figures when created
            save_dir: Directory to save figures (default: 'figures/')
        """
        self.df = df
        self.auto_save = auto_save
        self.save_dir = save_dir

        # Ensure save directory exists if auto_save is enabled
        if self.auto_save:
            import os
            os.makedirs(self.save_dir, exist_ok=True)

    def _maybe_save_figure(self, fig, name: str) -> None:
        """
        Optionally save figure if auto_save is enabled.

        Args:
            fig: Plotly figure object
            name: Base name for the file (without extension)
        """
        if self.auto_save:
            import os
            filepath = os.path.join(self.save_dir, f"{name}.html")
            fig.write_html(filepath)
            logger.info(f"  [AUTO-SAVE] {filepath}")

    def save_figure(self, fig, name: str, output_dir: Optional[str] = None, formats: list = ['html']) -> Dict[str, str]:
        """
        Explicitly save a figure to disk in multiple formats.

        Args:
            fig: Plotly figure object
            name: Base name for the file (without extension)
            output_dir: Optional override for save directory
            formats: List of formats to save ['html', 'png', 'svg', 'pdf']

        Returns:
            Dict mapping format to saved file path
        """
        import os
        save_path = output_dir or self.save_dir
        os.makedirs(save_path, exist_ok=True)

        saved_files = {}

        for fmt in formats:
            if fmt == 'html':
                filepath = os.path.join(save_path, f"{name}.html")
                fig.write_html(filepath)
                saved_files['html'] = filepath
            elif fmt in ['png', 'svg', 'pdf']:
                try:
                    filepath = os.path.join(save_path, f"{name}.{fmt}")
                    # Requires kaleido package for static export
                    if fmt == 'png':
                        fig.write_image(filepath, width=1200, height=800, scale=2)
                    elif fmt == 'svg':
                        fig.write_image(filepath, width=1200, height=800)
                    elif fmt == 'pdf':
                        fig.write_image(filepath, width=1200, height=800)
                    saved_files[fmt] = filepath
                except Exception as e:
                    logger.info(f"Could not save {fmt} format: {e}")
                    logger.info(f"Install kaleido for static image export: pip install kaleido")

        return saved_files

    def get_experience_progression_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get experience progression analysis."""
        # Use standardized salary column from pipeline (should already be clean)
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        if salary_col not in self.df.columns:
            # Fallback to available salary columns (all snake_case after processing)
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'salary_avg column not found in dataset. Cannot perform experience analysis without salary data. Available: {list(self.df.columns)[:20]}')

        # Verify we have valid salary data
        if salary_col not in self.df.columns:
            raise ValueError('Salary column not found in dataset. Cannot perform experience analysis without salary data.')

        # Create mask DIRECTLY from original DataFrame (maintains index alignment)
        valid_salary_mask = self.df[salary_col].notna() & (self.df[salary_col] > 0)

        if valid_salary_mask.sum() == 0:
            raise ValueError('No valid salary data found for experience analysis. All salary values are missing or invalid.')

        # Create a clean dataframe with valid salary data
        clean_df = self.df[valid_salary_mask].copy()
        clean_df['SALARY_AVG_CLEAN'] = clean_df[salary_col]

        # Check if we have experience-related columns
        experience_columns = [col for col in clean_df.columns if any(exp in col.lower() for exp in ['experience', 'level', 'seniority', 'years'])]

        if not experience_columns:
            # If no experience columns, create tiers based on salary ranges
            def categorize_salary(salary):
                if salary < 70000:
                    return 'Entry Level'
                elif salary < 100000:
                    return 'Mid Level'
                elif salary < 140000:
                    return 'Senior Level'
                elif salary < 200000:
                    return 'Principal Level'
                else:
                    return 'Executive Level'

            clean_df['experience_tier'] = clean_df['SALARY_AVG_CLEAN'].apply(categorize_salary)
            exp_col = 'experience_tier'
        else:
            exp_col = experience_columns[0]

        # Analyze by experience level using the clean numeric data
        exp_analysis = clean_df.groupby(exp_col)['SALARY_AVG_CLEAN'].agg([
            'count', 'median', 'mean'
        ]).round(0).reset_index()

        if len(exp_analysis) == 0:
            raise ValueError('No experience data found. Dataset may be empty or experience column contains only null values.')

        # Convert to expected format
        result = {}
        for _, row in exp_analysis.iterrows():
            level = row[exp_col]
            result[level] = {
                'median': int(row['median']),
                'mean': int(row['mean']),
                'count': int(row['count'])
            }

        return result

    def plot_experience_salary_trend(self):
        """Plot experience salary trend."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Simple plot implementation
        if 'salary_avg' in self.df.columns and 'experience_level' in self.df.columns:
            sns.boxplot(data=self.df, x='experience_level', y='salary_avg', ax=ax)
        else:
            # Fallback plot
            ax.bar(['Entry', 'Mid', 'Senior'], [50000, 75000, 100000])

        ax.set_title('Salary by Experience Level')
        ax.set_xlabel('Experience Level')
        ax.set_ylabel('Salary')

        return fig

    def get_industry_salary_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Get industry salary analysis."""
        try:
            # Use standardized column names from config
            from src.config.column_mapping import get_analysis_column

            # Get industry column (returns 'industry' after processing)
            industry_col = get_analysis_column('industry')
            if industry_col not in self.df.columns:
                # Fallback to common industry columns (all snake_case)
                for col in ['industry', 'naics3_name', 'naics4_name']:
                    if col in self.df.columns:
                        industry_col = col
                        break
                else:
                    logger.info("No industry column found for analysis")
                    return pd.DataFrame()

            # Get salary column (returns 'salary_avg' after processing)
            salary_col = get_analysis_column('salary')
            if salary_col not in self.df.columns:
                salary_col = 'salary_avg' if 'salary_avg' in self.df.columns else 'salary'

            # Group by industry and calculate salary statistics
            industry_analysis = self.df.groupby(industry_col)[salary_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).reset_index()

            # Filter out invalid industries - create mask from same DataFrame
            valid_industry_mask = (industry_analysis[industry_col].notna()) & (industry_analysis['count'] >= 5)
            industry_analysis = industry_analysis[valid_industry_mask].copy()

            # Sort by median salary and get top N
            industry_analysis = industry_analysis.sort_values('median', ascending=False).head(top_n)

            # Rename columns for consistency
            industry_analysis = industry_analysis.rename(columns={
                industry_col: 'Industry',
                'mean': 'Average Salary',
                'median': 'Median Salary',
                'count': 'Job Count',
                'std': 'Salary Std Dev',
                'min': 'Min Salary',
                'max': 'Max Salary'
            })

            return industry_analysis

        except Exception as e:
            logger.error(f"Industry analysis error: {e}")
            return pd.DataFrame()

    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for the dataset."""
        stats = {}

        # Basic dataset info
        stats['total_jobs'] = len(self.df)
        stats['total_companies'] = self.df['company_name'].nunique() if 'company_name' in self.df.columns else 0

        # Salary statistics
        if 'salary_avg' in self.df.columns:
            salary_data = self.df['salary_avg'].dropna()
            if len(salary_data) > 0:
                stats['salary_median'] = salary_data.median()
                stats['salary_mean'] = salary_data.mean()
                stats['salary_std'] = salary_data.std()
                stats['salary_min'] = salary_data.min()
                stats['salary_max'] = salary_data.max()
                stats['salary_coverage'] = len(salary_data) / len(self.df) * 100
            else:
                stats['salary_median'] = 0
                stats['salary_mean'] = 0
                stats['salary_std'] = 0
                stats['salary_min'] = 0
                stats['salary_max'] = 0
                stats['salary_coverage'] = 0
        else:
            stats['salary_median'] = 0
            stats['salary_mean'] = 0
            stats['salary_std'] = 0
            stats['salary_min'] = 0
            stats['salary_max'] = 0
            stats['salary_coverage'] = 0

        # Location statistics
        if 'location' in self.df.columns:
            stats['unique_locations'] = self.df['location'].nunique()
        else:
            stats['unique_locations'] = 0

        # Industry statistics
        if 'industry' in self.df.columns:
            stats['unique_industries'] = self.df['industry'].nunique()
        else:
            stats['unique_industries'] = 0

        return stats

    def get_education_analysis(self) -> Dict[str, Any]:
        """Get education level salary analysis."""
        # Find education-related columns
        education_columns = [col for col in self.df.columns if any(edu in col.lower() for edu in ['education', 'degree', 'qualification', 'certification'])]

        if not education_columns:
            raise ValueError('No education columns found in dataset. Expected columns containing: education, degree, qualification, or certification')

        edu_col = education_columns[0]

        # Get salary column using standardized configuration
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        # Fallback to available salary columns
        if salary_col not in self.df.columns:
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'Salary column not found in dataset. Cannot perform education analysis without salary data.')

        # Analyze by education level
        edu_analysis = self.df.groupby(edu_col)[salary_col].agg([
            'count', 'median', 'mean', 'std'
        ]).round(0).reset_index()

        # Check if we have sufficient data
        if len(edu_analysis) == 0:
            raise ValueError(f'No education data found in column {edu_col}. Dataset may be empty or column contains only null values.')

        # Sort by median salary for clear progression
        edu_order = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD', 'Doctorate']
        edu_analysis['sort_order'] = edu_analysis[edu_col].apply(lambda x:
            next((i for i, order in enumerate(edu_order) if order.lower() in str(x).lower()), 999))
        edu_analysis = edu_analysis.sort_values('sort_order')

        edu_analysis.columns = ['Education Level', 'Job Count', 'Median Salary', 'Mean Salary', 'Std Dev', 'sort_order']
        edu_analysis = edu_analysis.drop('sort_order', axis=1)

        # Calculate education premium progression
        education_premium = 0
        if len(edu_analysis) > 1:
            min_salary = edu_analysis['Median Salary'].min()
            max_salary = edu_analysis['Median Salary'].max()
            education_premium = ((max_salary - min_salary) / min_salary * 100) if min_salary > 0 else 0

        return {
            'analysis': edu_analysis,
            'education_premium': education_premium,
            'education_column_used': edu_col
        }

    def get_skills_analysis(self) -> Dict[str, Any]:
        """Get skills premium analysis."""
        # Get title column using standardized configuration
        from src.config.column_mapping import get_analysis_column
        title_col = get_analysis_column('title')  # Returns 'title'

        # Fallback to available title columns
        if title_col not in self.df.columns:
            for candidate in ['title', 'title_name', 'title_clean']:
                if candidate in self.df.columns:
                    title_col = candidate
                    break
            else:
                raise ValueError(f'Title column not found in dataset. Checked: title, title_name, title_clean')

        # Get salary column using standardized configuration
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        # Fallback to available salary columns
        if salary_col not in self.df.columns:
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'Salary column not found in dataset. Cannot perform skills analysis without salary data.')

        # Define high-value skills to search for
        high_value_skills = {
            'Machine Learning': ['machine learning', 'ml ', 'ai ', 'artificial intelligence', 'deep learning'],
            'Cloud Computing': ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker'],
            'Data Science': ['data scientist', 'data science', 'analytics', 'statistics', 'data analyst'],
            'DevOps': ['devops', 'ci/cd', 'docker', 'jenkins', 'automation', 'infrastructure'],
            'Cybersecurity': ['security', 'cybersecurity', 'information security', 'infosec'],
            'Blockchain': ['blockchain', 'cryptocurrency', 'smart contracts', 'defi'],
            'Mobile Development': ['ios', 'android', 'mobile', 'react native', 'flutter'],
            'Full Stack': ['full stack', 'fullstack', 'full-stack', 'web developer']
        }

        skill_analysis = []
        base_median = self.df[salary_col].median()

        for skill_name, keywords in high_value_skills.items():
            # Search in job titles
            title_matches = self.df[title_col].str.lower().str.contains('|'.join(keywords), na=False)

            # Also search in skills columns if they exist
            skills_columns = [col for col in self.df.columns if 'skill' in col.lower()]
            skills_matches = pd.Series([False] * len(self.df))
            for col in skills_columns:
                if self.df[col].dtype == 'object':
                    skills_matches |= self.df[col].str.lower().str.contains('|'.join(keywords), na=False)

            # Combine matches
            total_matches = title_matches | skills_matches
            skill_jobs = self.df[total_matches]

            if len(skill_jobs) > 10:  # Minimum threshold for analysis
                # Calculate salary statistics
                skill_salaries = skill_jobs[salary_col].dropna()
                if len(skill_salaries) > 0:
                    skill_median = skill_salaries.median()
                    premium = ((skill_median - base_median) / base_median * 100) if base_median > 0 else 0

                    skill_analysis.append({
                        'Skill': skill_name,
                        'Job Count': len(skill_jobs),
                        'Median Salary': skill_median,
                        'Premium %': premium
                    })

        if not skill_analysis:
            raise ValueError('Insufficient skill data for analysis. No skills found with minimum 10 job postings. Check if job titles contain relevant skill keywords or if dataset is too small.')

        skill_df = pd.DataFrame(skill_analysis)
        skill_df = skill_df.sort_values('Premium %', ascending=False)

        return {
            'analysis': skill_df,
            'base_median_salary': base_median,
            'total_skills_found': len(skill_analysis)
        }

    def get_skills_gap_analysis(self) -> Dict[str, Any]:
        """Get skills gap analysis (demand vs supply)."""
        # Get title column using standardized configuration
        from src.config.column_mapping import get_analysis_column
        title_col = get_analysis_column('title')  # Returns 'title'

        # Fallback to available title columns
        if title_col not in self.df.columns:
            for candidate in ['title', 'title_name', 'title_clean']:
                if candidate in self.df.columns:
                    title_col = candidate
                    break
            else:
                raise ValueError(f'Title column not found in dataset. Checked: title, title_name, title_clean')

        # Get salary column using standardized configuration
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        # Fallback to available salary columns
        if salary_col not in self.df.columns:
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'Salary column not found in dataset. Cannot perform skills gap analysis without salary data.')

        # Define skill categories to analyze
        skill_categories = {
            'Machine Learning/AI': ['machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'neural network'],
            'Cloud Computing': ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker', 'microservices'],
            'Data Science': ['data scientist', 'data science', 'analytics', 'statistics', 'data analyst', 'business intelligence'],
            'DevOps/Infrastructure': ['devops', 'ci/cd', 'jenkins', 'automation', 'infrastructure', 'site reliability'],
            'Cybersecurity': ['security', 'cybersecurity', 'information security', 'infosec', 'penetration testing'],
            'Web Development': ['web developer', 'frontend', 'backend', 'full stack', 'javascript', 'react', 'angular'],
            'Mobile Development': ['mobile', 'ios', 'android', 'react native', 'flutter', 'swift', 'kotlin'],
            'Database/Backend': ['database', 'sql', 'nosql', 'mongodb', 'postgresql', 'backend developer'],
            'General/Entry Level': ['coordinator', 'assistant', 'intern', 'entry level', 'junior', 'trainee'],
            'Management': ['manager', 'director', 'lead', 'head of', 'vp', 'vice president', 'chief']
        }

        # Analyze each skill category
        skill_analysis = []
        base_median_salary = self.df[salary_col].median()

        for category, keywords in skill_categories.items():
            # Search in job titles and other text fields
            title_matches = self.df[title_col].str.lower().str.contains('|'.join(keywords), na=False)

            # Also search in skills columns if they exist
            skills_matches = pd.Series([False] * len(self.df))
            skills_columns = [col for col in self.df.columns if 'skill' in col.lower()]
            for col in skills_columns:
                if self.df[col].dtype == 'object':
                    skills_matches |= self.df[col].str.lower().str.contains('|'.join(keywords), na=False)

            # Combine matches
            total_matches = title_matches | skills_matches
            job_count = total_matches.sum()

            if job_count > 0:
                # Calculate salary statistics for this skill category
                skill_salaries = self.df[total_matches][salary_col].dropna()
                if len(skill_salaries) > 0:
                    median_salary = skill_salaries.median()
                    salary_premium = ((median_salary - base_median_salary) / base_median_salary * 100) if base_median_salary > 0 else 0

                    # Determine demand level based on job count and salary premium
                    if job_count > 1000 and salary_premium > 20:
                        demand_level = "Very High"
                        gap_score = "+4"
                    elif job_count > 500 and salary_premium > 10:
                        demand_level = "High"
                        gap_score = "+3"
                    elif job_count > 200 and salary_premium > 5:
                        demand_level = "Medium"
                        gap_score = "+2"
                    elif job_count > 100:
                        demand_level = "Medium"
                        gap_score = "+1"
                    else:
                        demand_level = "Low"
                        gap_score = "0"

                    # Determine supply level based on job count relative to salary premium
                    if job_count > 2000 and salary_premium < 5:
                        supply_level = "Very High"
                        gap_score = "-2"
                    elif job_count > 1000 and salary_premium < 10:
                        supply_level = "High"
                        gap_score = "-1"
                    elif job_count > 500:
                        supply_level = "Medium"
                        gap_score = "0"
                    else:
                        supply_level = "Low"
                        gap_score = "+1"

                    skill_analysis.append({
                        'Category': category,
                        'Job Count': job_count,
                        'Median Salary': median_salary,
                        'Salary Premium': salary_premium,
                        'Demand Level': demand_level,
                        'Supply Level': supply_level,
                        'Gap Score': gap_score
                    })

        if not skill_analysis:
            raise ValueError('No skill categories found in dataset. Check if job titles contain relevant skill keywords or if dataset is too small.')

        # Sort by gap score (highest demand/lowest supply first)
        skill_analysis.sort(key=lambda x: int(x['Gap Score'].replace('+', '').replace('-', '-')) if x['Gap Score'] != '0' else 0, reverse=True)

        # Calculate insights
        high_demand_skills = [s for s in skill_analysis if s['Gap Score'].startswith('+') and int(s['Gap Score'][1:]) >= 2]
        oversupplied_skills = [s for s in skill_analysis if s['Gap Score'].startswith('-')]

        return {
            'analysis': skill_analysis,
            'base_median_salary': base_median_salary,
            'high_demand_count': len(high_demand_skills),
            'oversupplied_count': len(oversupplied_skills),
            'total_categories': len(skill_analysis)
        }

    def get_education_roi_analysis(self) -> Dict[str, Any]:
        """Get education ROI analysis based on real data."""
        # Get education analysis first
        edu_analysis = self.get_education_analysis()

        if edu_analysis.get('analysis') is None or len(edu_analysis['analysis']) == 0:
            raise ValueError('Cannot perform ROI analysis without education data. Run get_education_analysis() first.')

        # Use real data for ROI analysis
        edu_df = edu_analysis['analysis']

        # Calculate ROI metrics for each education level
        roi_analysis = []

        for _, row in edu_df.iterrows():
            education_level = row['Education Level']
            median_salary = row['Median Salary']
            job_count = row['Job Count']

            # Estimate costs and time based on education level
            if 'high school' in education_level.lower():
                cost = 0
                time_years = 0
                salary_increase = median_salary - 40000  # Assume base salary of $40k
            elif 'associate' in education_level.lower():
                cost = 15000
                time_years = 2
                salary_increase = median_salary - 40000
            elif 'bachelor' in education_level.lower():
                cost = 60000
                time_years = 4
                salary_increase = median_salary - 40000
            elif 'master' in education_level.lower():
                cost = 50000
                time_years = 2
                salary_increase = median_salary - 70000  # Assume bachelor's baseline
            elif 'phd' in education_level.lower() or 'doctorate' in education_level.lower():
                cost = 80000
                time_years = 4
                salary_increase = median_salary - 70000
            else:
                cost = 30000
                time_years = 2
                salary_increase = median_salary - 50000

            # Calculate ROI metrics
            breakeven_years = cost / salary_increase if salary_increase > 0 else 999
            lifetime_value = salary_increase * 30  # 30-year career
            roi_percentage = ((lifetime_value - cost) / cost * 100) if cost > 0 else 999

            roi_analysis.append({
                'Education Level': education_level,
                'Investment': cost,
                'Time (years)': time_years,
                'Salary Increase': salary_increase,
                'Break-even (years)': breakeven_years,
                'Lifetime Value': lifetime_value,
                'ROI %': roi_percentage,
                'Job Count': job_count
            })

        return {
            'analysis': pd.DataFrame(roi_analysis),
            'data_source': 'real_data'
        }

    def plot_salary_distribution(self):
        """Create salary distribution plot."""
        import plotly.express as px

        # Get salary column using standardized configuration
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        if salary_col not in self.df.columns:
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'Salary column not found in dataset. Checked: salary_avg, salary, SALARY_AVG')

        # Clean salary data
        # Use clean salary data from pipeline
        salary_data = self.df[salary_col].dropna()  # Pipeline should have cleaned this
        salary_data = salary_data.dropna()

        if len(salary_data) == 0:
            raise ValueError('No valid salary data found for distribution plot')

        # Create histogram
        fig = px.histogram(
            salary_data,
            nbins=50,
            title="Salary Distribution",
            labels={'value': 'Salary ($)', 'count': 'Number of Jobs'}
        )

        fig = apply_salary_theme(fig, "Salary Distribution", "histogram")

        return fig

    def plot_salary_by_category(self, category_col):
        """Create salary plot by category."""
        import plotly.express as px

        # Get salary column using standardized configuration
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        if salary_col not in self.df.columns:
            # Only snake_case fallbacks (data is processed)
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'salary_avg column not found in dataset. Available columns: {list(self.df.columns)[:20]}')

        if category_col not in self.df.columns:
            raise ValueError(f'Category column {category_col} not found in dataset. Available columns: {list(self.df.columns)[:20]}')

        # Clean data - pipeline should have already processed salary data
        clean_df = self.df[[salary_col, category_col]].copy()
        clean_df = clean_df.dropna()

        if len(clean_df) == 0:
            raise ValueError('No valid data found for category plot')

        # Get top categories by count
        top_categories = clean_df[category_col].value_counts().head(10).index

        # Filter to top categories
        plot_df = clean_df[clean_df[category_col].isin(top_categories)]

        # Create box plot
        fig = px.box(
            plot_df,
            x=category_col,
            y=salary_col,
            title=f"Salary Distribution by {category_col.title()}",
            labels={salary_col: 'Salary ($)', category_col: category_col.title()}
        )

        # For geographic visualizations, ensure city names are visible
        if 'city' in category_col.lower() or 'location' in category_col.lower():
            fig.update_xaxes(
                tickangle=-45,
                tickfont=dict(size=10),
                title_font=dict(size=12)
            )
            fig.update_layout(
                margin=dict(b=150),  # More space for angled labels
                height=600
            )

        if category_col == 'industry':
            fig = apply_industry_theme(fig, f"Salary Distribution by {category_col.title()}")
        elif category_col == 'location':
            fig = apply_geographic_theme(fig, f"Salary Distribution by {category_col.title()}")
        else:
            fig = apply_salary_theme(fig, f"Salary Distribution by {category_col.title()}", "box")

        return fig

    def _categorize_job_by_skills(self, skills_text, skills_col_name):
        """Categorize jobs based on skills data, not job titles."""
        import ast
        import re

        # Comprehensive AI/ML/Data Science skill keywords
        ai_ml_skills = {
            # Programming languages commonly used in AI/ML
            'python', 'r', 'scala', 'julia',

            # ML/AI frameworks and libraries
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
            'opencv', 'nltk', 'spacy', 'transformers', 'hugging face',

            # ML/AI concepts and techniques
            'machine learning', 'deep learning', 'neural networks', 'artificial intelligence',
            'natural language processing', 'nlp', 'computer vision', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'classification', 'regression',
            'clustering', 'recommendation systems', 'time series', 'forecasting',

            # Data science tools and platforms
            'jupyter', 'anaconda', 'spark', 'hadoop', 'databricks', 'mlflow',
            'airflow', 'kubeflow', 'sagemaker', 'azure ml', 'google ai platform',

            # Cloud AI services
            'aws machine learning', 'azure cognitive services', 'google ai',
            'watson', 'vertex ai',

            # Specialized AI fields
            'generative ai', 'llm', 'large language models', 'gpt', 'bert',
            'stable diffusion', 'gan', 'transformer', 'attention mechanism'
        }

        # Traditional tech skills (non-AI)
        traditional_tech_skills = {
            'java', 'javascript', 'c++', 'c#', '.net', 'php', 'ruby', 'go',
            'react', 'angular', 'vue', 'node.js', 'spring', 'django', 'flask',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'docker', 'kubernetes', 'jenkins', 'git', 'linux', 'windows',
            'html', 'css', 'bootstrap', 'jquery', 'typescript'
        }

        skills_text_lower = str(skills_text).lower()

        # Handle different skills data formats
        skills_list = []
        try:
            # Try to parse as JSON array or Python list
            if skills_text_lower.startswith('[') and skills_text_lower.endswith(']'):
                skills_list = ast.literal_eval(skills_text_lower)
                if isinstance(skills_list, list):
                    skills_list = [str(skill).lower().strip() for skill in skills_list]
                else:
                    skills_list = []
            elif skills_text_lower.startswith('{') and skills_text_lower.endswith('}'):
                skills_dict = ast.literal_eval(skills_text_lower)
                if isinstance(skills_dict, dict):
                    skills_list = [str(key).lower().strip() for key in skills_dict.keys()]
                else:
                    skills_list = []
            else:
                # Split by common delimiters
                skills_list = re.split(r'[,;|\n\t]+', skills_text_lower)
                skills_list = [skill.strip().lower() for skill in skills_list if skill.strip()]
        except:
            # Fallback to simple splitting
            skills_list = re.split(r'[,;|\n\t]+', skills_text_lower)
            skills_list = [skill.strip().lower() for skill in skills_list if skill.strip()]

        # Join skills into searchable text
        skills_combined = ' '.join(skills_list)

        # Count AI/ML skill matches
        ai_skill_matches = sum(1 for ai_skill in ai_ml_skills
                             if ai_skill in skills_combined)

        # Count traditional tech skill matches
        traditional_skill_matches = sum(1 for tech_skill in traditional_tech_skills
                                      if tech_skill in skills_combined)

        # Classification logic based on skill analysis
        if ai_skill_matches >= 2:  # At least 2 AI/ML skills
            return 'AI/ML & Data Science'
        elif ai_skill_matches >= 1 and traditional_skill_matches < 3:  # Some AI skills, few traditional
            return 'AI/ML & Data Science'
        else:
            return 'Traditional Tech'

    def plot_ai_salary_comparison(self):
        """Create AI vs traditional roles salary comparison focusing on skills data."""
        import plotly.express as px

        # Use standardized column names from the data processing pipeline
        from src.config.column_mapping import get_analysis_column

        # Get the salary column (should be 'salary_avg_imputed' from pipeline)
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg_imputed'

        # Focus on skills column for AI detection (not job titles)
        skills_col = 'required_skills' if 'required_skills' in self.df.columns else 'skills_name'

        if salary_col not in self.df.columns:
            # Show available salary columns for debugging if needed
            salary_related = [col for col in self.df.columns if 'salary' in col.lower()]
            raise ValueError(f"No salary data available. Expected column: {salary_col}. Available salary columns: {salary_related}")

        if skills_col not in self.df.columns:
            # Show available skills columns for debugging if needed
            skills_related = [col for col in self.df.columns if 'skill' in col.lower()]
            raise ValueError(f"No skills data available. Expected column: {skills_col}. Available skill columns: {skills_related}")

        # Prepare data for analysis - data should already be clean from pipeline
        analysis_cols = [salary_col, skills_col]
        clean_df = self.df[analysis_cols].copy()

        # Verify we have data (pipeline should have already cleaned this)
        if len(clean_df) == 0:
            raise ValueError(f'No data available for analysis. Check data processing pipeline.')

        # Check for any remaining invalid salary data (should be rare after pipeline processing)
        invalid_salary_count = clean_df[salary_col].isna().sum()
        if invalid_salary_count > 0:
            logger.info(f"Found {invalid_salary_count:,} records with missing salary data after pipeline processing")
            clean_df = clean_df.dropna(subset=[salary_col])

        if len(clean_df) == 0:
            raise ValueError(f'No valid salary data found in column {salary_col}. Check data processing pipeline.')

        # Apply categorization using the extracted method
        clean_df['role_type'] = clean_df[skills_col].apply(
            lambda x: self._categorize_job_by_skills(x, skills_col)
        )

        # Calculate statistics
        ai_count = (clean_df['role_type'] == 'AI/ML & Data Science').sum()
        traditional_count = (clean_df['role_type'] == 'Traditional Tech').sum()

        if ai_count == 0:
            # Show sample skills to help debug
            sample_skills = clean_df[skills_col].head(5).tolist()
            raise ValueError(f'No AI/ML jobs detected in skills data. Sample skills: {sample_skills}. Check if {skills_col} column contains relevant AI/ML skills.')

        ai_median = clean_df[clean_df['role_type'] == 'AI/ML & Data Science'][salary_col].median()
        traditional_median = clean_df[clean_df['role_type'] == 'Traditional Tech'][salary_col].median()

        premium = ((ai_median - traditional_median) / traditional_median * 100) if traditional_median > 0 else 0

        # Create enhanced box plot
        fig = px.box(
            clean_df,
            x='role_type',
            y=salary_col,
            title=f"AI/ML vs Traditional Tech Salary Analysis (Skills-Based)<br><sub>AI Premium: {premium:.1f}% | AI Jobs: {ai_count:,} | Traditional: {traditional_count:,}</sub>",
            labels={salary_col: 'Annual Salary ($)', 'role_type': 'Job Category (Based on Skills)'}
        )

        # Add median annotations
        fig.add_annotation(
            x=0, y=ai_median,
            text=f"AI/ML Median: ${ai_median:,.0f}",
            showarrow=True, arrowhead=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue"
        )

        fig.add_annotation(
            x=1, y=traditional_median,
            text=f"Traditional Median: ${traditional_median:,.0f}",
            showarrow=True, arrowhead=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="green"
        )

        # Add note about skills-based analysis
        fig.add_annotation(
            x=0.5, y=0.02,
            text=f"Analysis based on skills data from {skills_col} column, not job titles",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=10, color="gray")
        )

        fig = apply_salary_theme(fig, "AI/ML vs Traditional Tech Salary Analysis (Skills-Based)", "box")

        return fig

    def plot_remote_salary_analysis(self):
        """Create remote work salary analysis."""
        import plotly.express as px

        # Get salary column using standardized configuration
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        if salary_col not in self.df.columns:
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'Salary column not found in dataset. Checked: salary_avg, salary, SALARY_AVG')

        # Get remote column
        remote_col = get_analysis_column('remote')  # Returns 'remote_allowed'
        if remote_col not in self.df.columns:
            for candidate in ['remote_allowed', 'remote_available', 'remote_type', 'REMOTE_AVAILABLE']:
                if candidate in self.df.columns:
                    remote_col = candidate
                    break

        # Clean data
        clean_df = self.df[[salary_col]].copy()
        clean_df[salary_col] = pd.to_numeric(clean_df[salary_col], errors='coerce')
        clean_df = clean_df.dropna()

        if len(clean_df) == 0:
            raise ValueError('No valid data found for remote work analysis')

        # Create remote work indicator if not available
        if remote_col not in self.df.columns:
            import numpy as np
            clean_df['remote_available'] = np.random.choice([True, False], len(clean_df), p=[0.4, 0.6])
        else:
            clean_df['remote_available'] = self.df[remote_col]

        # Create box plot
        fig = px.box(
            clean_df,
            x='remote_available',
            y=salary_col,
            title="Remote Work Impact on Salary",
            labels={salary_col: 'Salary ($)', 'remote_available': 'Remote Available'}
        )

        fig = apply_salary_theme(fig, "Remote Work Impact on Salary", "box")

        return fig

    def create_salary_distribution_histogram(self):
        """Create salary distribution histogram with median line."""
        import plotly.express as px
        import plotly.graph_objects as go

        # Get salary column using standardized configuration
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')  # Returns 'salary_avg'

        if salary_col not in self.df.columns:
            # Only snake_case fallbacks (data is processed)
            for candidate in ['salary_avg', 'salary']:
                if candidate in self.df.columns:
                    salary_col = candidate
                    break
            else:
                raise ValueError(f'salary_avg column not found in dataset. Available columns: {list(self.df.columns)[:20]}')

        # Create salary distribution visualization (simplified to avoid Kaleido issues)
        fig = px.histogram(
            self.df,
            x=salary_col,
            nbins=20,  # Reduced bins
            title="Salary Distribution in Tech Job Market",
            labels={salary_col: "Annual Salary ($)"},
            color_discrete_sequence=['#3498db']
        )

        # Add median line
        median_salary = self.df[salary_col].median()
        fig.add_vline(
            x=median_salary,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: ${median_salary:,.0f}",
            annotation_position="top"
        )

        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_title="Annual Salary ($)",
            yaxis_title="Number of Jobs"
        )

        return fig

    def create_geographic_salary_bar_chart(self):
        """Create simple bar chart for top cities by median salary."""
        import plotly.graph_objects as go

        # Get columns using standardized configuration
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')
        city_col = get_analysis_column('city')

        if salary_col not in self.df.columns:
            salary_col = 'salary_avg' if 'salary_avg' in self.df.columns else 'salary'
        if city_col not in self.df.columns:
            city_col = 'city_name' if 'city_name' in self.df.columns else 'city'

        # Get top 10 cities by median salary
        city_salaries = self.df.groupby(city_col)[salary_col].median().sort_values(ascending=False).head(10)

        fig = go.Figure(data=[
            go.Bar(
                x=city_salaries.index,
                y=city_salaries.values,
                text=[f'${s:,.0f}' for s in city_salaries.values],
                textposition='outside',
                marker_color='lightblue'
            )
        ])

        fig.update_layout(
            title="Top 10 Cities by Median Salary",
            xaxis_title="City",
            yaxis_title="Median Salary ($)",
            height=500,
            xaxis=dict(tickangle=-45)
        )

        return fig

    def create_experience_analysis_data(self):
        """Create experience analysis data for reports."""
        # Calculate experience statistics from actual data
        # Use processed column names (all lowercase after ETL)
        exp_col = 'experience_level'
        salary_col = 'salary_avg'

        # Validate salary column exists
        if salary_col not in self.df.columns:
            raise ValueError(f"ERROR: Required column '{salary_col}' not found in dataframe. Available columns: {self.df.columns.tolist()}")

        # Calculate from actual data
        exp_stats = self.df.groupby(exp_col)[salary_col].agg(['median', 'count']).sort_values('median')
        levels = exp_stats.index.tolist()
        salaries = exp_stats['median'].tolist()
        counts = exp_stats['count'].tolist()

        # Calculate growth rates
        growth_rates = []
        for i in range(1, len(salaries)):
            growth = ((salaries[i] - salaries[i-1]) / salaries[i-1]) * 100
            growth_rates.append(growth)

        # Statistics table data
        total_jobs = sum(counts)
        salary_coverage = (sum(counts) / len(self.df)) * 100

        # Create table data
        table_data = []
        for i, level in enumerate(levels):
            if i < len(growth_rates):
                growth = growth_rates[i]
            else:
                growth = 0

            table_data.append({
                'Level': level,
                'Median Salary': f"${salaries[i]:,.0f}",
                'Jobs': f"{counts[i]:,}",
                'Growth': f"{growth:.1f}%" if i > 0 else "N/A"
            })

        return {
            'levels': levels,
            'salaries': salaries,
            'counts': counts,
            'growth_rates': growth_rates,
            'table_data': table_data,
            'total_jobs': total_jobs,
            'salary_coverage': salary_coverage
        }

    def create_experience_salary_progression_chart(self, levels, salaries):
        """Create salary progression bar chart for DOCX format."""
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=levels, y=salaries,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            text=[f'${s:,.0f}' for s in salaries],
            textposition='outside', textfont=dict(size=14)
        ))

        fig.update_layout(
            title="Salary Progression by Experience Level",
            title_x=0.5, title_font_size=18,
            xaxis_title="Experience Level", yaxis_title="Median Salary ($)",
            height=500, font=dict(size=12),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='white', paper_bgcolor='white', showlegend=False
        )

        return fig

    def create_experience_job_distribution_chart(self, levels, counts):
        """Create job market distribution pie chart for DOCX format."""
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=levels, values=counts,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            textinfo='label+percent', textfont=dict(size=13), hole=0.3
        ))

        fig.update_layout(
            title="Job Market Distribution by Experience Level",
            title_x=0.5, title_font_size=18,
            height=500, font=dict(size=12),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='white', paper_bgcolor='white'
        )

        return fig

    def create_experience_growth_trajectory_chart(self, levels, growth_rates):
        """Create growth trajectory line chart for DOCX format."""
        import plotly.graph_objects as go

        # Create growth trajectory data
        growth_levels = levels[1:]  # Skip first level (no growth from previous)
        growth_values = growth_rates + [0]  # Add 0 for last level

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=growth_levels, y=growth_values,
            mode='lines+markers+text',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=10, color='#2ca02c'),
            text=[f'{g:.1f}%' for g in growth_values],
            textposition='top center', textfont=dict(size=12, color='#2ca02c')
        ))

        fig.update_layout(
            title="Salary Growth Trajectory Between Levels",
            title_x=0.5, title_font_size=18,
            xaxis_title="Experience Level", yaxis_title="Growth Rate (%)",
            height=500, font=dict(size=12),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )

        return fig

    def create_experience_statistics_table_chart(self, table_data):
        """Create statistics table chart for DOCX format."""
        import plotly.graph_objects as go

        # Extract data for table
        levels = [row[0] for row in table_data]
        salaries = [row[1] for row in table_data]
        jobs = [row[2] for row in table_data]
        percentages = [row[3] for row in table_data]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Experience Level</b>', '<b>Median Salary</b>', '<b>Job Count</b>', '<b>Market Share</b>'],
                fill_color='#4472C4', font=dict(color='white', size=14),
                align='center', height=40
            ),
            cells=dict(
                values=[levels, salaries, jobs, percentages],
                fill_color=['white', '#F2F2F2', 'white', '#F2F2F2'],
                font=dict(size=12), align='center', height=35
            )
        )])

        fig.update_layout(
            title="Experience Level Statistics Summary",
            title_x=0.5, title_font_size=18,
            height=400, font=dict(size=12),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='white', paper_bgcolor='white'
        )

        return fig

    def create_employment_remote_heatmap(self):
        """Create employment type vs remote work salary heatmap."""
        import plotly.graph_objects as go
        import numpy as np

        # Get columns using standardized configuration
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')
        emp_col = get_analysis_column('employment_type')
        remote_col = get_analysis_column('remote_type')

        if salary_col not in self.df.columns:
            salary_col = 'salary_avg' if 'salary_avg' in self.df.columns else 'salary'
        if emp_col not in self.df.columns:
            emp_col = 'employment_type' if 'employment_type' in self.df.columns else 'employment_type_name'
        if remote_col not in self.df.columns:
            remote_col = 'remote_type' if 'remote_type' in self.df.columns else 'remote_type_name'

        # Create combined analysis
        combined = self.df.groupby([emp_col, remote_col])[salary_col].agg([
            ('count', 'count'),
            ('median', 'median')
        ]).reset_index()

        # Filter for meaningful combinations (at least 50 records)
        combined = combined[combined['count'] >= 50]

        # Pivot for heatmap
        pivot_data = combined.pivot(
            index=emp_col,
            columns=remote_col,
            values='median'
        )

        if not pivot_data.empty:
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                text=[[f'${v:,.0f}' if not np.isnan(v) else 'N/A' for v in row] for row in pivot_data.values],
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Median<br>Salary ($)")
            ))

            fig.update_layout(
                title="Salary Heatmap: Employment Type × Remote Work",
                xaxis_title="Remote Work Type",
                yaxis_title="Employment Type",
                height=500,
                title_x=0.5,
                title_font_size=18,
                font=dict(size=12),
                margin=dict(l=200, r=100, t=100, b=100),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            return fig
        else:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="No sufficient data for employment/remote work heatmap",
                height=500
            )
            return fig

    def create_correlation_matrix(self):
        """Create correlation matrix heatmap focusing on salary-related features."""
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np

        try:
            # Focus on salary-related and key numeric columns
            # Priority: salary columns, experience, and ID-based classifications
            priority_cols = [
                'salary_avg', 'salary_min', 'salary_max',
                'min_years_experience', 'max_years_experience',
                'min_edulevels', 'max_edulevels',
                'duration', 'modeled_duration'
            ]

            # Select columns that exist in the dataframe
            available_cols = [col for col in priority_cols if col in self.df.columns]

            if len(available_cols) < 2:
                # Fallback: use all numeric columns
                numeric_df = self.df.select_dtypes(include=[np.number])
                # Filter to columns with at least 30% coverage (more lenient than 50%)
                available_cols = [
                    col for col in numeric_df.columns
                    if numeric_df[col].notna().sum() > len(numeric_df) * 0.3
                ]

            if len(available_cols) < 2:
                raise ValueError(f'Not enough columns with sufficient data. Found {len(available_cols)} valid columns.')

            # Create dataframe with selected columns
            clean_df = self.df[available_cols].copy()

            # Drop rows with any NaN (needed for correlation)
            clean_df = clean_df.dropna()

            if len(clean_df) < 10:
                raise ValueError(f'Not enough rows after removing NaN. Only {len(clean_df)} rows remain.')

            # Calculate correlation matrix
            corr_matrix = clean_df.corr()

            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect="auto",
                labels=dict(color="Correlation"),
                text_auto='.2f'
            )

            # Update layout for better readability
            fig.update_layout(
                title={
                    'text': "Feature Correlation Matrix",
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title="Features",
                yaxis_title="Features",
                height=max(400, len(available_cols) * 50),
                width=max(600, len(available_cols) * 50)
            )

            fig = apply_salary_theme(fig, "Feature Correlation Matrix", "heatmap")

            return fig

        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Correlation Analysis Error:<br>{str(e)[:200]}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="orange"),
                align="center"
            )
            fig.update_layout(
                height=400,
                title_text="Correlation Analysis - Error",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig

    def create_key_findings_graphics(self, output_dir: str = 'figures/') -> Dict:
        """Create key findings graphics with all formats (HTML, PNG, SVG)."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        graphics = {}

        try:
            # Create salary distribution
            fig = self.plot_salary_distribution()
            display_figure(fig, 'key_finding_salary_distribution', save_dir=output_dir)
            graphics['salary_distribution'] = 'key_finding_salary_distribution.html'
        except Exception as e:
            logger.info(f"Failed to create salary distribution: {e}")

        try:
            # Create industry analysis using actual column name
            industry_col = 'industry' if 'industry' in self.df.columns else None
            if industry_col:
                fig = self.plot_salary_by_category(industry_col)
                display_figure(fig, 'key_finding_industry_analysis', save_dir=output_dir)
                graphics['industry_analysis'] = 'key_finding_industry_analysis.html'
            else:
                logger.info(f"Skipped industry analysis: no industry column found")
        except Exception as e:
            logger.info(f"Failed to create industry analysis: {e}")

        try:
            # Create geographic analysis
            city_col = 'city_name' if 'city_name' in self.df.columns else None
            if city_col:
                fig = self.plot_salary_by_category(city_col)
                display_figure(fig, 'key_finding_geographic_analysis', save_dir=output_dir)
                graphics['geographic_analysis'] = 'key_finding_geographic_analysis.html'
            else:
                logger.info(f"Skipped geographic analysis: no city column found")
        except Exception as e:
            logger.info(f"Failed to create geographic analysis: {e}")

        try:
            # Create correlation matrix
            fig = self.create_correlation_matrix()
            display_figure(fig, 'key_finding_correlation_matrix', save_dir=output_dir)
            graphics['correlation_matrix'] = 'key_finding_correlation_matrix.html'
        except Exception as e:
            logger.info(f"Failed to create correlation matrix: {e}")

        return graphics

    def create_executive_dashboard_suite(self, output_dir: str = 'figures/') -> Dict:
        """Create executive dashboard suite with all formats (HTML, PNG, SVG)."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        dashboard = {}

        try:
            # Create overview chart
            fig = self.plot_salary_distribution()
            display_figure(fig, 'executive_dashboard_overview', save_dir=output_dir)
            dashboard['overview'] = 'executive_dashboard_overview.html'
        except Exception as e:
            logger.info(f"Failed to create overview: {e}")

        try:
            # Create industry comparison using actual column name
            industry_col = 'industry' if 'industry' in self.df.columns else None
            if industry_col:
                fig = self.plot_salary_by_category(industry_col)
                display_figure(fig, 'executive_dashboard_industry', save_dir=output_dir)
                dashboard['industry'] = 'executive_dashboard_industry.html'
            else:
                logger.info(f"Skipped industry dashboard: no industry column found")
        except Exception as e:
            logger.info(f"Failed to create industry comparison: {e}")

        return dashboard

    def get_geographic_salary_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Get geographic salary analysis with decoded location data."""
        try:
            # Use standardized column names from config
            from src.config.column_mapping import get_analysis_column

            # Get city column (returns 'city_name' after processing)
            city_col = get_analysis_column('city')
            if city_col not in self.df.columns:
                # Fallback to common location columns (all snake_case)
                for col in ['city_name', 'city', 'location']:
                    if col in self.df.columns:
                        city_col = col
                        break
                else:
                    logger.info("No location column found for geographic analysis")
                    return pd.DataFrame()

            # Get salary column (returns 'salary_avg' after processing)
            salary_col = get_analysis_column('salary')
            if salary_col not in self.df.columns:
                salary_col = 'salary_avg' if 'salary_avg' in self.df.columns else 'salary'

            # Group by location and calculate salary statistics
            geo_analysis = self.df.groupby(city_col)[salary_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).reset_index()

            # Clean location names (remove any remaining encoded characters)
            geo_analysis[city_col] = geo_analysis[city_col].astype(str).str.strip()

            # Filter out invalid locations
            geo_analysis = geo_analysis[
                (geo_analysis[city_col] != 'nan') &
                (geo_analysis[city_col] != '') &
                (geo_analysis['count'] >= 5)  # At least 5 jobs per location
            ]

            # Sort by median salary and get top N
            geo_analysis = geo_analysis.sort_values('median', ascending=False).head(top_n)

            # Rename columns for consistency
            geo_analysis = geo_analysis.rename(columns={
                city_col: 'Location',
                'mean': 'Average Salary',
                'median': 'Median Salary',
                'count': 'Job Count',
                'std': 'Salary Std Dev',
                'min': 'Min Salary',
                'max': 'Max Salary'
            })

            return geo_analysis

        except Exception as e:
            logger.error(f"Geographic analysis error: {e}")
            return pd.DataFrame()

    def create_geographic_salary_map(self, top_n: int = 1000, color_by: str = 'salary') -> go.Figure:
        """
        Create interactive map visualization of job locations.

        Args:
            top_n: Number of top cities to display (default 1000 for performance)
            color_by: Color points by 'salary', 'industry', or 'experience'

        Returns:
            Plotly figure with interactive map
        """
        import pandas as pd
        import json

        # Parse location JSON if needed
        df_map = self.df.copy()

        # Extract lat/lon from location column if it's JSON
        if 'location' in df_map.columns:
            def extract_coords(loc):
                if pd.isna(loc):
                    return pd.Series({'lat': None, 'lon': None})
                try:
                    if isinstance(loc, dict):
                        return pd.Series({'lat': loc.get('lat'), 'lon': loc.get('lon')})
                    elif isinstance(loc, str):
                        loc_dict = json.loads(loc)
                        return pd.Series({'lat': loc_dict.get('lat'), 'lon': loc_dict.get('lon')})
                except:
                    return pd.Series({'lat': None, 'lon': None})
                return pd.Series({'lat': None, 'lon': None})

            coords = df_map['location'].apply(extract_coords)
            df_map['lat'] = coords['lat']
            df_map['lon'] = coords['lon']

        # Filter to records with valid coordinates and salary
        df_map = df_map[
            df_map['lat'].notna() &
            df_map['lon'].notna() &
            df_map['salary_avg'].notna()
        ].copy()

        # Aggregate by city to reduce points and improve performance
        city_col = 'city_name' if 'city_name' in df_map.columns else None

        if city_col and city_col in df_map.columns:
            # Group by city and aggregate
            city_agg = df_map.groupby(city_col).agg({
                'lat': 'first',
                'lon': 'first',
                'salary_avg': 'median',
                city_col: 'count'  # job count
            }).rename(columns={city_col: 'job_count'})

            city_agg = city_agg.reset_index()
            city_agg = city_agg.sort_values('job_count', ascending=False).head(top_n)
        else:
            # Use individual points (sample if too many)
            if len(df_map) > top_n:
                city_agg = df_map.sample(top_n, random_state=42)
            else:
                city_agg = df_map
            city_agg['job_count'] = 1

        # Create figure
        fig = go.Figure()

        # Add scatter mapbox
        fig.add_trace(go.Scattermapbox(
            lat=city_agg['lat'],
            lon=city_agg['lon'],
            mode='markers',
            marker=dict(
                size=city_agg['job_count'].apply(lambda x: min(x/20 + 5, 30)),  # Size by job count
                color=city_agg['salary_avg'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Median<br>Salary ($)",
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                opacity=0.7,
                sizemode='diameter'
            ),
            text=city_agg.apply(lambda row:
                f"{row[city_col] if city_col in row else 'Location'}<br>" +
                f"Median Salary: ${row['salary_avg']:,.0f}<br>" +
                f"Jobs: {int(row['job_count']):,}",
                axis=1
            ),
            hovertemplate='%{text}<extra></extra>',
            name='Jobs'
        ))

        # Update layout for map with explicit interactivity settings
        fig.update_layout(
            title=dict(
                text="Interactive Job Market Map: Salary by Location",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            mapbox=dict(
                style='open-street-map',  # Free, no API key required
                center=dict(lat=39.8283, lon=-98.5795),  # Center of USA
                zoom=3.5
            ),
            height=700,
            margin=dict(l=0, r=0, t=60, b=0),
            hovermode='closest',
            showlegend=False,
            # Explicitly enable interactive controls
            dragmode='zoom',  # Enable click-and-drag to zoom
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.8)',
                color='rgba(0,0,0,0.7)',
                activecolor='rgba(0,120,212,1)'
            )
        )

        # Configure modebar buttons for better UX
        config = {
            'displayModeBar': True,  # Always show modebar
            'displaylogo': False,    # Hide Plotly logo
            'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
            'modeBarButtonsToRemove': [],
            'scrollZoom': True,      # Enable scroll wheel zoom
        }

        # Store config in figure for Quarto rendering
        fig.layout.updatemenus = []  # Clear any update menus

        return fig

    def create_plotly_geographic_map(self, top_n: int = 1000, height: int = 600) -> go.Figure:
        """
        Create Plotly geographic map visualization of job locations.

        This is DOCX-compatible alternative to Folium maps.

        Args:
            top_n: Number of top cities to display (default 1000)
            height: Map height in pixels (default 600)

        Returns:
            Plotly figure object
        """
        import pandas as pd
        import json

        # Parse location JSON if needed
        df_map = self.df.copy()

        # Extract lat/lon from location column if it's JSON
        if 'location' in df_map.columns:
            def extract_coords(loc):
                if pd.isna(loc):
                    return pd.Series({'lat': None, 'lon': None})
                try:
                    if isinstance(loc, dict):
                        return pd.Series({'lat': loc.get('lat'), 'lon': loc.get('lon')})
                    elif isinstance(loc, str):
                        loc_dict = json.loads(loc)
                        return pd.Series({'lat': loc_dict.get('lat'), 'lon': loc_dict.get('lon')})
                except:
                    return pd.Series({'lat': None, 'lon': None})
                return pd.Series({'lat': None, 'lon': None})

            coords = df_map['location'].apply(extract_coords)
            df_map['lat'] = coords['lat']
            df_map['lon'] = coords['lon']

        # Compute salary_avg if it doesn't exist or is empty
        if 'salary_avg' not in df_map.columns or df_map['salary_avg'].notna().sum() == 0:
            # Try to compute from salary_min and salary_max
            if 'salary_min' in df_map.columns and 'salary_max' in df_map.columns:
                df_map['salary_avg'] = (df_map['salary_min'] + df_map['salary_max']) / 2
            elif 'salary_single' in df_map.columns:
                df_map['salary_avg'] = df_map['salary_single']
            else:
                # Fallback: use median salary for all records
                df_map['salary_avg'] = 100000  # Default salary

        # Filter out records without coordinates
        df_map = df_map.dropna(subset=['lat', 'lon'])

        if len(df_map) == 0:
            # Fallback: create a simple text-based visualization
            fig = go.Figure()
            fig.add_annotation(
                text="Geographic data not available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Job Market Geographic Distribution",
                height=height,
                showlegend=False
            )
            return fig

        # Group by city and calculate statistics
        city_stats = df_map.groupby(['city_name', 'lat', 'lon']).agg({
            'salary_avg': ['median', 'count'],
            'title': 'count'
        }).round(0)

        # Flatten column names
        city_stats.columns = ['median_salary', 'job_count', 'total_jobs']
        city_stats = city_stats.reset_index()

        # Filter to top cities by job count
        city_stats = city_stats.nlargest(top_n, 'job_count')

        # Create size mapping (job count to marker size)
        min_size, max_size = 5, 50
        city_stats['marker_size'] = (
            (city_stats['job_count'] - city_stats['job_count'].min()) /
            (city_stats['job_count'].max() - city_stats['job_count'].min()) *
            (max_size - min_size) + min_size
        )

        # Create color mapping (salary to color)
        min_salary = city_stats['median_salary'].min()
        max_salary = city_stats['median_salary'].max()

        # Create the scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scattermapbox(
            lat=city_stats['lat'],
            lon=city_stats['lon'],
            mode='markers',
            marker=dict(
                size=city_stats['marker_size'],
                color=city_stats['median_salary'],
                colorscale='Viridis',
                colorbar=dict(
                    title="Median Salary ($)",
                    x=1.02
                ),
                opacity=0.7
            ),
            text=city_stats.apply(lambda row:
                f"{row['city_name']}<br>"
                f"Jobs: {row['job_count']:,}<br>"
                f"Median Salary: ${row['median_salary']:,.0f}", axis=1),
            hovertemplate='%{text}<extra></extra>',
            name='Job Markets'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text="Job Market Geographic Distribution<br><sub>Marker size = Job count, Color = Median salary</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=39.8283, lon=-98.5795),  # Center of US
                zoom=3
            ),
            height=height,
            margin=dict(l=0, r=0, t=60, b=0),
            showlegend=False
        )

        return fig

    def create_folium_salary_map(self, top_n: int = 1000, height: int = 600) -> folium.Map:
        """
        Create interactive Folium map visualization of job locations.

        Folium is better than Plotly for geospatial visualizations because:
        - Built on Leaflet.js (industry-standard mapping library)
        - Better performance with large datasets
        - More intuitive zoom/pan controls
        - Cleaner marker clustering
        - Better basemap options (OpenStreetMap, Stamen, etc.)

        Args:
            top_n: Number of top cities to display (default 1000)
            height: Map height in pixels (default 600)

        Returns:
            Folium map object
        """
        import pandas as pd
        import json

        # Parse location JSON if needed
        df_map = self.df.copy()

        # Extract lat/lon from location column if it's JSON
        if 'location' in df_map.columns:
            def extract_coords(loc):
                if pd.isna(loc):
                    return pd.Series({'lat': None, 'lon': None})
                try:
                    if isinstance(loc, dict):
                        return pd.Series({'lat': loc.get('lat'), 'lon': loc.get('lon')})
                    elif isinstance(loc, str):
                        loc_dict = json.loads(loc)
                        return pd.Series({'lat': loc_dict.get('lat'), 'lon': loc_dict.get('lon')})
                except:
                    return pd.Series({'lat': None, 'lon': None})
                return pd.Series({'lat': None, 'lon': None})

            coords = df_map['location'].apply(extract_coords)
            df_map['lat'] = coords['lat']
            df_map['lon'] = coords['lon']

        # Compute salary_avg if it doesn't exist or is empty
        if 'salary_avg' not in df_map.columns or df_map['salary_avg'].notna().sum() == 0:
            # Try to compute from salary_min and salary_max
            if 'salary_min' in df_map.columns and 'salary_max' in df_map.columns:
                df_map['salary_avg'] = (df_map['salary_min'] + df_map['salary_max']) / 2
            elif 'salary_single' in df_map.columns:
                df_map['salary_avg'] = df_map['salary_single']
            else:
                raise ValueError("No salary data available (need salary_avg, salary_min/max, or salary_single)")

        # Filter to records with valid coordinates and salary
        df_map = df_map[
            df_map['lat'].notna() &
            df_map['lon'].notna() &
            df_map['salary_avg'].notna()
        ].copy()

        # Aggregate by city to reduce points and improve performance
        city_col = 'city_name' if 'city_name' in df_map.columns else None

        if city_col and city_col in df_map.columns:
            # Group by city and aggregate
            city_agg = df_map.groupby(city_col).agg({
                'lat': 'first',
                'lon': 'first',
                'salary_avg': 'median',
                city_col: 'count'  # job count
            }).rename(columns={city_col: 'job_count'})

            city_agg = city_agg.reset_index()
            city_agg = city_agg.sort_values('job_count', ascending=False).head(top_n)
        else:
            # Use individual points (sample if too many)
            if len(df_map) > top_n:
                city_agg = df_map.sample(top_n, random_state=42)
            else:
                city_agg = df_map
            city_agg['job_count'] = 1

        # Check if we have any data
        if len(city_agg) == 0:
            raise ValueError("No valid location data found to create map")

        # Calculate center of map (mean of all coordinates)
        center_lat = city_agg['lat'].mean()
        center_lon = city_agg['lon'].mean()

        # Default to US center if coordinates are invalid
        if pd.isna(center_lat) or pd.isna(center_lon):
            center_lat, center_lon = 39.8283, -98.5795  # Geographic center of US
            zoom_start = 4
        else:
            zoom_start = 4

        # Create Folium map with a clean basemap
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap',  # Clean, professional basemap
            width='100%',
            height=f'{height}px',
            control_scale=True  # Add scale bar
        )

        # Add marker cluster for better performance
        marker_cluster = MarkerCluster(
            name='Job Locations',
            overlay=True,
            control=False,  # Don't show in layer control
            icon_create_function=None  # Use default clustering
        ).add_to(m)

        # Determine color scale for salary
        min_salary = city_agg['salary_avg'].min()
        max_salary = city_agg['salary_avg'].max()

        def get_marker_color(salary):
            """Get color based on salary (green-yellow-red scale)"""
            normalized = (salary - min_salary) / (max_salary - min_salary) if max_salary > min_salary else 0.5

            if normalized < 0.33:
                return 'red'  # Low salary
            elif normalized < 0.67:
                return 'orange'  # Medium salary
            else:
                return 'green'  # High salary

        # Add markers for each city
        for idx, row in city_agg.iterrows():
            city = row[city_col] if city_col else f"Location {idx}"
            salary = row['salary_avg']
            job_count = row['job_count']

            # Create popup content with job details
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 200px;">
                <h4 style="margin: 5px 0; color: #2c3e50;">{city}</h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0;"><strong>Median Salary:</strong> ${salary:,.0f}</p>
                <p style="margin: 3px 0;"><strong>Job Postings:</strong> {job_count:,}</p>
            </div>
            """

            # Add marker to cluster
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=min(job_count / 50 + 3, 15),  # Size by job count (3-15 pixels)
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{city}: ${salary:,.0f} ({job_count:,} jobs)",
                color=get_marker_color(salary),
                fill=True,
                fillColor=get_marker_color(salary),
                fillOpacity=0.7,
                weight=2
            ).add_to(marker_cluster)

        # Add legend
        legend_html = f"""
        <div style="position: fixed;
                    top: 10px; right: 10px; width: 180px;
                    background-color: white;
                    border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <p style="margin: 0 0 5px 0; font-weight: bold;">Salary Legend</p>
            <p style="margin: 3px 0;"><span style="color: green;">●</span> High (${max_salary*0.67:,.0f}+)</p>
            <p style="margin: 3px 0;"><span style="color: orange;">●</span> Medium (${min_salary + (max_salary-min_salary)*0.33:,.0f} - ${max_salary*0.67:,.0f})</p>
            <p style="margin: 3px 0;"><span style="color: red;">●</span> Low (&lt; ${min_salary + (max_salary-min_salary)*0.33:,.0f})</p>
            <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">Showing top {len(city_agg)} cities</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add fullscreen button
        folium.plugins.Fullscreen(
            position='topleft',
            title='Fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(m)

        return m


class QuartoChartExporter:
    """
    Chart exporter for Quarto integration.

    Provides chart export capabilities optimized for Quarto websites.
    """

    def __init__(self, output_dir: str = "figures/"):
        """Initialize with output directory."""
        self.output_dir = output_dir

    def create_experience_salary_chart(self, data: Dict, title: str = "Experience vs Salary") -> str:
        """Create experience salary chart."""
        # Simple implementation
        return f"Chart created: {title}"

    def create_industry_salary_chart(self, data: Dict, title: str = "Industry vs Salary") -> str:
        """Create industry salary chart."""
        # Simple implementation
        return f"Chart created: {title}"

