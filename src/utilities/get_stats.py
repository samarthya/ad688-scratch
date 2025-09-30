"""
Statistical Analysis Utilities for Job Market Data

This module provides utility functions for calculating key statistics and insights
from job market data, including salary analysis, experience level calculations,
and disparity measurements.

Functions:
    calculate_salary_statistics: Calculate comprehensive salary statistics
    analyze_experience_gaps: Analyze salary gaps across experience levels  
    analyze_education_premiums: Calculate education-based salary premiums
    analyze_company_size_effects: Analyze company size impact on salaries
    generate_summary_report: Generate comprehensive statistical summary

Author: Saurabh Sharma
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobMarketStatistics:
    """
    Comprehensive statistical analysis for job market data.
    
    This class provides methods for calculating various statistics and insights
    from job market datasets, with a focus on salary analysis and disparity
    measurements.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the statistics analyzer.
        
        Args:
            data_path: Optional path to the job market data CSV file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.stats_cache: Dict[str, Any] = {}
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and clean job market data for analysis.
        
        Args:
            file_path: Path to the CSV file. If None, uses self.data_path
            
        Returns:
            Cleaned DataFrame ready for analysis
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data cannot be properly loaded or cleaned
        """
        if file_path is None:
            file_path = self.data_path
            
        if file_path is None:
            raise ValueError("No data path provided")
            
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Loaded raw data: {len(self.df):,} records")
            
            # Clean salary data
            self._clean_salary_columns()
            
            # Remove records with missing salary data
            salary_cols = ['Minimum Annual Salary', 'Maximum Annual Salary']
            before_count = len(self.df)
            self.df = self.df.dropna(subset=[col for col in salary_cols if col in self.df.columns])
            after_count = len(self.df)
            
            logger.info(f"After cleaning: {after_count:,} records ({before_count - after_count:,} removed)")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _clean_salary_columns(self) -> None:
        """Clean and convert salary columns to numeric format."""
        if self.df is None:
            raise ValueError("No data loaded")
            
        salary_cols = ['Minimum Annual Salary', 'Maximum Annual Salary']
        
        for col in salary_cols:
            if col in self.df.columns:
                # Remove currency symbols and commas, convert to numeric
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                    errors='coerce'
                )
    
    def calculate_salary_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive salary statistics.
        
        Returns:
            Dictionary containing various salary statistics and metrics
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Use maximum salary as primary metric
        salary_col = 'Maximum Annual Salary' if 'Maximum Annual Salary' in self.df.columns else 'Minimum Annual Salary'
        
        stats = {
            'total_jobs': len(self.df),
            'salary_column_used': salary_col,
            'mean_salary': self.df[salary_col].mean(),
            'median_salary': self.df[salary_col].median(),
            'std_salary': self.df[salary_col].std(),
            'min_salary': self.df[salary_col].min(),
            'max_salary': self.df[salary_col].max(),
            'percentiles': {
                '25th': self.df[salary_col].quantile(0.25),
                '75th': self.df[salary_col].quantile(0.75),
                '90th': self.df[salary_col].quantile(0.90),
                '95th': self.df[salary_col].quantile(0.95)
            }
        }
        
        # Calculate coefficient of variation
        stats['coefficient_of_variation'] = stats['std_salary'] / stats['mean_salary']
        
        # Calculate IQR and outlier thresholds
        iqr = stats['percentiles']['75th'] - stats['percentiles']['25th']
        stats['iqr'] = iqr
        stats['outlier_threshold_lower'] = stats['percentiles']['25th'] - 1.5 * iqr
        stats['outlier_threshold_upper'] = stats['percentiles']['75th'] + 1.5 * iqr
        
        self.stats_cache['salary_statistics'] = stats
        return stats
    
    def analyze_experience_gaps(self) -> Dict[str, Any]:
        """
        Analyze salary gaps across experience levels.
        
        Returns:
            Dictionary containing experience level analysis results
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        salary_col = 'Maximum Annual Salary' if 'Maximum Annual Salary' in self.df.columns else 'Minimum Annual Salary'
        
        # Create experience level approximation using salary quartiles
        self.df['Experience_Level'] = pd.qcut(
            self.df[salary_col], 
            4, 
            labels=['Entry-Level', 'Mid-Level', 'Senior', 'Executive']
        )
        
        exp_stats = self.df.groupby('Experience_Level')[salary_col].agg([
            'mean', 'median', 'count', 'std'
        ]).round(0)
        
        # Calculate gaps
        max_salary = exp_stats['mean'].max()
        min_salary = exp_stats['mean'].min()
        gap_percent = ((max_salary - min_salary) / min_salary) * 100
        
        analysis = {
            'level_statistics': exp_stats.to_dict(),
            'salary_gap_percent': gap_percent,
            'max_level_salary': max_salary,
            'min_level_salary': min_salary,
            'gap_ratio': max_salary / min_salary if min_salary > 0 else 0
        }
        
        self.stats_cache['experience_analysis'] = analysis
        return analysis
    
    def analyze_education_premiums(self) -> Dict[str, Any]:
        """
        Analyze education-based salary premiums.
        
        Returns:
            Dictionary containing education premium analysis
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        salary_col = 'Maximum Annual Salary' if 'Maximum Annual Salary' in self.df.columns else 'Minimum Annual Salary'
        
        # Create education level proxy using salary tertiles
        self.df['Education_Level'] = pd.qcut(
            self.df[salary_col], 
            3, 
            labels=['High School', 'Bachelor', 'Advanced']
        )
        
        edu_stats = self.df.groupby('Education_Level')[salary_col].agg([
            'mean', 'median', 'count'
        ]).round(0)
        
        # Calculate education premium
        max_salary = edu_stats['mean'].max()
        min_salary = edu_stats['mean'].min()
        premium_percent = ((max_salary - min_salary) / min_salary) * 100
        
        analysis = {
            'education_statistics': edu_stats.to_dict(),
            'education_premium_percent': premium_percent,
            'premium_ratio': max_salary / min_salary if min_salary > 0 else 0
        }
        
        self.stats_cache['education_analysis'] = analysis
        return analysis
    
    def analyze_company_size_effects(self) -> Dict[str, Any]:
        """
        Analyze company size impact on salaries.
        
        Returns:
            Dictionary containing company size analysis
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        salary_col = 'Maximum Annual Salary' if 'Maximum Annual Salary' in self.df.columns else 'Minimum Annual Salary'
        
        # Calculate company sizes based on job posting count
        company_counts = self.df.groupby('Company Name').size().reset_index(name='job_count')
        self.df = self.df.merge(company_counts, on='Company Name')
        
        # Categorize company sizes
        self.df['Company_Size'] = pd.qcut(
            self.df['job_count'], 
            3, 
            labels=['Small', 'Medium', 'Large']
        )
        
        size_stats = self.df.groupby('Company_Size')[salary_col].agg([
            'mean', 'median', 'count'
        ]).round(0)
        
        # Calculate size premium
        max_salary = size_stats['mean'].max()
        min_salary = size_stats['mean'].min()
        size_premium_percent = ((max_salary - min_salary) / min_salary) * 100
        
        analysis = {
            'size_statistics': size_stats.to_dict(),
            'size_premium_percent': size_premium_percent,
            'premium_ratio': max_salary / min_salary if min_salary > 0 else 0
        }
        
        self.stats_cache['company_size_analysis'] = analysis
        return analysis
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary report.
        
        Returns:
            Complete summary of all analyses
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Run all analyses
        salary_stats = self.calculate_salary_statistics()
        experience_analysis = self.analyze_experience_gaps()
        education_analysis = self.analyze_education_premiums()
        company_analysis = self.analyze_company_size_effects()
        
        summary = {
            'overview': {
                'total_jobs_analyzed': salary_stats['total_jobs'],
                'salary_column_used': salary_stats['salary_column_used'],
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'salary_statistics': salary_stats,
            'experience_gaps': experience_analysis,
            'education_premiums': education_analysis,
            'company_size_effects': company_analysis,
            'key_findings': self._extract_key_findings()
        }
        
        return summary
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from cached analyses."""
        findings = []
        
        if 'experience_analysis' in self.stats_cache:
            exp_gap = self.stats_cache['experience_analysis']['salary_gap_percent']
            findings.append(f"Experience salary gap: {exp_gap:.1f}% between Executive and Entry-Level")
        
        if 'education_analysis' in self.stats_cache:
            edu_premium = self.stats_cache['education_analysis']['education_premium_percent']
            findings.append(f"Education premium: {edu_premium:.1f}% between Advanced and High School levels")
        
        if 'company_size_analysis' in self.stats_cache:
            size_premium = self.stats_cache['company_size_analysis']['size_premium_percent']
            findings.append(f"Company size premium: {size_premium:.1f}% between Large and Small companies")
        
        return findings


def main():
    """
    Main execution function for standalone statistical analysis.
    
    This function demonstrates usage of the JobMarketStatistics class
    and generates a comprehensive report.
    """
    try:
        # Initialize analyzer
        analyzer = JobMarketStatistics()
        
        # Load data
        data_path = Path('data/raw/lightcast_job_postings.csv')
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return
        
        analyzer.load_data(str(data_path))
        
        # Generate comprehensive report
        report = analyzer.generate_summary_report()
        
        # Display key findings
        print('\n' + '='*60)
        print('JOB MARKET STATISTICAL ANALYSIS REPORT')
        print('='*60)
        print(f"\nTotal Jobs Analyzed: {report['overview']['total_jobs_analyzed']:,}")
        print(f"Analysis Date: {report['overview']['analysis_timestamp']}")
        
        print('\nKEY FINDINGS:')
        print('-' * 40)
        for finding in report['key_findings']:
            print(f"• {finding}")
        
        # Display salary statistics
        salary_stats = report['salary_statistics']
        print(f"\nSALARY OVERVIEW:")
        print(f"• Mean Salary: ${salary_stats['mean_salary']:,.0f}")
        print(f"• Median Salary: ${salary_stats['median_salary']:,.0f}")
        print(f"• Salary Range: ${salary_stats['min_salary']:,.0f} - ${salary_stats['max_salary']:,.0f}")
        print(f"• Standard Deviation: ${salary_stats['std_salary']:,.0f}")
        
        logger.info("Statistical analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()