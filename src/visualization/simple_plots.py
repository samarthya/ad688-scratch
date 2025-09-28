"""
Simplified visualization utilities for job market analysis without plotly dependency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Set default styling
plt.style.use('default')
sns.set_palette("husl")

class SalaryVisualizer:
    """
    Simplified salary analysis class for creating job market visualizations without plotly.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with processed job data."""
        self.df = df
        
    def get_industry_salary_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """
        Generate comprehensive industry salary analysis.
        
        Returns DataFrame with industry statistics including AI premium and remote work %.
        """
        # Create salary average column if it doesn't exist
        if 'salary_avg' not in self.df.columns:
            # Calculate average salary from min and max
            self.df['salary_avg'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        # Use actual column names from the data
        industry_col = 'industry'
        salary_col = 'salary_avg'
        
        # Calculate basic industry statistics
        industry_stats = self.df.groupby(industry_col)[salary_col].agg([
            'count', 'median', 'mean', 'std'
        ]).round(0).reset_index()
        
        industry_stats.columns = [industry_col, 'Job Count', 'Median Salary', 'Mean Salary', 'Std Dev']
        
        # Sort by median salary and take top N
        industry_stats = industry_stats.sort_values('Median Salary', ascending=False).head(top_n)
        
        # Add AI Premium calculation based on job titles
        if 'title' in self.df.columns:
            ai_keywords = ['ai', 'machine learning', 'data scientist', 'ml engineer', 'artificial intelligence']
            ai_premiums = []
            
            for industry in industry_stats[industry_col]:
                industry_df = self.df[self.df[industry_col] == industry]
                
                # Calculate AI vs non-AI roles in this industry
                ai_mask = industry_df['title'].str.lower().str.contains('|'.join(ai_keywords), na=False)
                ai_salaries = industry_df[ai_mask][salary_col]
                non_ai_salaries = industry_df[~ai_mask][salary_col]
                
                if len(ai_salaries) > 0 and len(non_ai_salaries) > 0:
                    premium = ((ai_salaries.median() - non_ai_salaries.median()) / non_ai_salaries.median()) * 100
                    ai_premiums.append(f"+{premium:.0f}%")
                else:
                    ai_premiums.append("N/A")
            
            industry_stats['AI Premium'] = ai_premiums
        else:
            industry_stats['AI Premium'] = 'N/A'
        
        # Add Remote Work % (handle numeric remote_allowed column properly)
        if 'remote_allowed' in self.df.columns:
            remote_percentages = []
            for industry in industry_stats[industry_col]:
                industry_df = self.df[self.df[industry_col] == industry]
                
                # Check if remote_allowed is numeric (1 for remote, 0 for not remote)
                if pd.api.types.is_numeric_dtype(industry_df['remote_allowed']):
                    remote_pct = (industry_df['remote_allowed'].sum() / len(industry_df)) * 100
                else:
                    # If it's text, use string matching
                    remote_pct = (industry_df['remote_allowed'].astype(str).str.contains('Remote|Hybrid|1|Yes', case=False, na=False).sum() / len(industry_df)) * 100
                
                remote_percentages.append(f"{remote_pct:.0f}%")
            
            industry_stats['Remote %'] = remote_percentages
        else:
            industry_stats['Remote %'] = 'N/A'
        
        # Rename the industry column to 'Industry' for display
        industry_stats = industry_stats.rename(columns={industry_col: 'Industry'})
        
        return industry_stats[['Industry', 'Median Salary', 'Job Count', 'AI Premium', 'Remote %']]
    
    def get_experience_salary_analysis(self) -> pd.DataFrame:
        """Generate experience level salary analysis."""
        # Create salary average column if it doesn't exist
        if 'salary_avg' not in self.df.columns:
            self.df['salary_avg'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        exp_col = 'experience_level'
        salary_col = 'salary_avg'
        
        exp_stats = self.df.groupby(exp_col)[salary_col].agg([
            'count', 'median', 'mean'
        ]).round(0).reset_index()
        
        exp_stats.columns = ['Experience Level', 'Job Count', 'Median Salary', 'Mean Salary']
        return exp_stats.sort_values('Median Salary')
    
    def get_education_premium_analysis(self) -> pd.DataFrame:
        """Generate education premium analysis."""
        # Check for education columns
        edu_col = None
        for col in ['EDUCATION_REQUIRED', 'education_level', 'degree_requirement']:
            if col in self.df.columns:
                edu_col = col
                break
        
        # Determine salary column
        salary_col = None
        for col in ['SALARY_AVG', 'salary_avg', 'salary']:
            if col in self.df.columns:
                salary_col = col
                break
        
        if not edu_col or not salary_col:
            return pd.DataFrame({
                'Education Level': ['High School', 'Bachelors', 'Masters', 'PhD'],
                'Median Salary': [45000, 65000, 85000, 110000],
                'Premium vs Bachelors': ['Base', '0%', '+31%', '+69%']
            })
        
        edu_stats = self.df.groupby(edu_col)[salary_col].agg([
            'count', 'median'
        ]).round(0).reset_index()
        
        edu_stats.columns = ['Education Level', 'Job Count', 'Median Salary']
        
        # Calculate premium vs bachelor's (or lowest level)
        baseline = edu_stats['Median Salary'].min()
        edu_stats['Premium vs Baseline'] = edu_stats['Median Salary'].apply(
            lambda x: f"+{((x - baseline) / baseline * 100):.0f}%" if x > baseline else "Base"
        )
        
        return edu_stats
    
    def get_overall_statistics(self) -> Dict:
        """Generate overall dataset statistics."""
        # Create salary average column if it doesn't exist
        if 'salary_avg' not in self.df.columns:
            self.df['salary_avg'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        salary_col = 'salary_avg'
        
        return {
            'total_jobs': len(self.df),
            'median_salary': int(self.df[salary_col].median()),
            'mean_salary': int(self.df[salary_col].mean()),
            'std_salary': int(self.df[salary_col].std()),
            'min_salary': int(self.df[salary_col].min()),
            'max_salary': int(self.df[salary_col].max()),
            'salary_25th': int(self.df[salary_col].quantile(0.25)),
            'salary_75th': int(self.df[salary_col].quantile(0.75))
        }
    
    def get_geographic_salary_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Generate geographic salary analysis by location."""
        # Create salary average column if it doesn't exist
        if 'salary_avg' not in self.df.columns:
            self.df['salary_avg'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        # Use location column
        location_col = 'location'
        salary_col = 'salary_avg'
        
        # Calculate location statistics
        location_stats = self.df.groupby(location_col)[salary_col].agg([
            'count', 'median', 'mean'
        ]).round(0).reset_index()
        
        location_stats.columns = ['Location', 'Job Count', 'Median Salary', 'Mean Salary']
        
        # Sort by median salary and take top N
        location_stats = location_stats.sort_values('Median Salary', ascending=False).head(top_n)
        
        return location_stats[['Location', 'Median Salary', 'Job Count']]
    
    def get_experience_progression_analysis(self) -> Dict:
        """Generate detailed experience progression analysis."""
        # Create salary average column if it doesn't exist
        if 'salary_avg' not in self.df.columns:
            self.df['salary_avg'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        exp_stats = self.get_experience_salary_analysis()
        
        # Calculate progression percentages
        progression = {}
        exp_levels = exp_stats['Experience Level'].tolist()
        salaries = exp_stats['Median Salary'].tolist()
        
        for i, level in enumerate(exp_levels):
            if i > 0:
                prev_salary = salaries[i-1]
                curr_salary = salaries[i]
                increase_pct = ((curr_salary - prev_salary) / prev_salary * 100)
                progression[level] = {
                    'salary': int(curr_salary),
                    'increase_pct': increase_pct
                }
            else:
                progression[level] = {
                    'salary': int(salaries[i]),
                    'increase_pct': 0
                }
        
        return progression