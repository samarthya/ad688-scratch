"""
Skills Analysis Module

This module provides comprehensive skills analysis capabilities for job market data,
including geographic skills analysis, salary correlation, and demand trends.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter
import re
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(level="WARNING")


class SkillsAnalyzer:
    """
    Comprehensive skills analysis for job market data.

    Analyzes technical skills from the Lightcast dataset to provide insights
    on skill demand, salary correlation, and geographic distribution.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize skills analyzer with job market data.

        Args:
            df: Pandas DataFrame with job market data including technical_skills column
        """
        self.df = df.copy()
        self.skills_data = None
        self._process_skills_data()

    def _process_skills_data(self):
        """Process and clean technical skills data."""
        logger.info("Processing technical skills data...")

        # Parse JSON strings in technical_skills column
        skills_list = []
        for idx, skills_json in enumerate(self.df['technical_skills'].dropna()):
            try:
                if isinstance(skills_json, str):
                    skills = json.loads(skills_json)
                    if isinstance(skills, list):
                        for skill in skills:
                            skills_list.append({
                                'job_id': self.df.iloc[idx]['job_id'],
                                'skill': skill.strip(),
                                'city_name': self.df.iloc[idx]['city_name'],
                                'industry': self.df.iloc[idx]['industry'],
                                'salary_avg': self.df.iloc[idx]['salary_avg'],
                                'experience_level': self.df.iloc[idx]['experience_level'],
                                'remote_type': self.df.iloc[idx]['remote_type']
                            })
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse skills JSON at index {idx}: {e}")
                continue

        self.skills_data = pd.DataFrame(skills_list)
        logger.info(f"Processed {len(self.skills_data):,} skill instances from {len(self.df):,} jobs")

    def get_top_skills_by_frequency(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top skills by frequency across all jobs.

        Args:
            top_n: Number of top skills to return

        Returns:
            DataFrame with skill frequency and percentage
        """
        if self.skills_data is None or len(self.skills_data) == 0:
            return pd.DataFrame()

        skill_counts = self.skills_data['skill'].value_counts()
        total_skills = len(self.skills_data)

        top_skills = skill_counts.head(top_n).reset_index()
        top_skills.columns = ['skill', 'frequency']
        top_skills['percentage'] = (top_skills['frequency'] / total_skills * 100).round(2)

        return top_skills

    def get_skills_by_geography(self, top_cities: int = 10, top_skills_per_city: int = 10) -> Dict[str, Any]:
        """
        Analyze skills demand by geographic location.

        Args:
            top_cities: Number of top cities to analyze
            top_skills_per_city: Number of top skills to show per city

        Returns:
            Dictionary with geographic skills analysis
        """
        if self.skills_data is None or len(self.skills_data) == 0:
            return {}

        # Get top cities by job count
        city_job_counts = self.df['city_name'].value_counts().head(top_cities)

        results = {
            'top_cities': city_job_counts.to_dict(),
            'city_skills': {},
            'summary': {}
        }

        for city in city_job_counts.index:
            city_skills = self.skills_data[self.skills_data['city_name'] == city]

            if len(city_skills) > 0:
                skill_counts = city_skills['skill'].value_counts().head(top_skills_per_city)
                results['city_skills'][city] = {
                    'top_skills': skill_counts.to_dict(),
                    'total_skill_instances': len(city_skills),
                    'unique_skills': city_skills['skill'].nunique()
                }

        # Calculate summary statistics
        results['summary'] = {
            'total_cities_analyzed': len(results['city_skills']),
            'total_unique_skills': self.skills_data['skill'].nunique(),
            'total_skill_instances': len(self.skills_data)
        }

        return results

    def get_skills_salary_correlation(self, min_frequency: int = 50) -> pd.DataFrame:
        """
        Analyze correlation between skills and salary.

        Args:
            min_frequency: Minimum frequency for skill to be included in analysis

        Returns:
            DataFrame with skills and their salary statistics
        """
        if self.skills_data is None or len(self.skills_data) == 0:
            return pd.DataFrame()

        # Filter skills with minimum frequency
        skill_counts = self.skills_data['skill'].value_counts()
        frequent_skills = skill_counts[skill_counts >= min_frequency].index

        # Calculate salary statistics for each skill
        skill_salary_stats = []

        for skill in frequent_skills:
            skill_data = self.skills_data[
                (self.skills_data['skill'] == skill) &
                (self.skills_data['salary_avg'].notna())
            ]

            if len(skill_data) > 0:
                skill_salary_stats.append({
                    'skill': skill,
                    'frequency': len(skill_data),
                    'median_salary': skill_data['salary_avg'].median(),
                    'mean_salary': skill_data['salary_avg'].mean(),
                    'salary_std': skill_data['salary_avg'].std(),
                    'min_salary': skill_data['salary_avg'].min(),
                    'max_salary': skill_data['salary_avg'].max()
                })

        skill_salary_df = pd.DataFrame(skill_salary_stats)
        skill_salary_df = skill_salary_df.sort_values('median_salary', ascending=False)

        return skill_salary_df

    def get_skills_by_industry(self, top_industries: int = 10, top_skills_per_industry: int = 10) -> Dict[str, Any]:
        """
        Analyze skills demand by industry.

        Args:
            top_industries: Number of top industries to analyze
            top_skills_per_industry: Number of top skills to show per industry

        Returns:
            Dictionary with industry skills analysis
        """
        if self.skills_data is None or len(self.skills_data) == 0:
            return {}

        # Get top industries by job count
        industry_job_counts = self.df['industry'].value_counts().head(top_industries)

        results = {
            'top_industries': industry_job_counts.to_dict(),
            'industry_skills': {},
            'summary': {}
        }

        for industry in industry_job_counts.index:
            industry_skills = self.skills_data[self.skills_data['industry'] == industry]

            if len(industry_skills) > 0:
                skill_counts = industry_skills['skill'].value_counts().head(top_skills_per_industry)
                results['industry_skills'][industry] = {
                    'top_skills': skill_counts.to_dict(),
                    'total_skill_instances': len(industry_skills),
                    'unique_skills': industry_skills['skill'].nunique()
                }

        # Calculate summary statistics
        results['summary'] = {
            'total_industries_analyzed': len(results['industry_skills']),
            'total_unique_skills': self.skills_data['skill'].nunique(),
            'total_skill_instances': len(self.skills_data)
        }

        return results

    def get_emerging_skills(self, min_frequency: int = 20, max_frequency: int = 200) -> pd.DataFrame:
        """
        Identify emerging skills (moderate frequency, high salary potential).

        Args:
            min_frequency: Minimum frequency to be considered emerging
            max_frequency: Maximum frequency to avoid over-saturated skills

        Returns:
            DataFrame with emerging skills analysis
        """
        if self.skills_data is None or len(self.skills_data) == 0:
            return pd.DataFrame()

        # Get skills in the emerging frequency range
        skill_counts = self.skills_data['skill'].value_counts()
        emerging_skills = skill_counts[
            (skill_counts >= min_frequency) & (skill_counts <= max_frequency)
        ].index

        # Calculate salary statistics for emerging skills
        emerging_skills_stats = []

        for skill in emerging_skills:
            skill_data = self.skills_data[
                (self.skills_data['skill'] == skill) &
                (self.skills_data['salary_avg'].notna())
            ]

            if len(skill_data) > 0:
                emerging_skills_stats.append({
                    'skill': skill,
                    'frequency': len(skill_data),
                    'median_salary': skill_data['salary_avg'].median(),
                    'mean_salary': skill_data['salary_avg'].mean(),
                    'salary_premium': skill_data['salary_avg'].median() - self.df['salary_avg'].median(),
                    'growth_potential': len(skill_data) / len(self.df) * 100  # Percentage of jobs requiring this skill
                })

        emerging_df = pd.DataFrame(emerging_skills_stats)
        emerging_df = emerging_df.sort_values('salary_premium', ascending=False)

        return emerging_df

    def get_skills_trends_by_experience(self) -> Dict[str, Any]:
        """
        Analyze skills trends by experience level.

        Returns:
            Dictionary with skills analysis by experience level
        """
        if self.skills_data is None or len(self.skills_data) == 0:
            return {}

        experience_levels = self.df['experience_level'].dropna().unique()
        results = {}

        for level in experience_levels:
            level_skills = self.skills_data[self.skills_data['experience_level'] == level]

            if len(level_skills) > 0:
                top_skills = level_skills['skill'].value_counts().head(10)
                results[level] = {
                    'top_skills': top_skills.to_dict(),
                    'total_skill_instances': len(level_skills),
                    'unique_skills': level_skills['skill'].nunique(),
                    'avg_skills_per_job': len(level_skills) / len(self.df[self.df['experience_level'] == level])
                }

        return results

    def get_comprehensive_skills_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive skills analysis report.

        Returns:
            Dictionary with complete skills analysis
        """
        logger.info("Generating comprehensive skills analysis report...")

        report = {
            'overview': {
                'total_jobs': len(self.df),
                'jobs_with_skills': self.df['technical_skills'].notna().sum(),
                'skills_coverage': (self.df['technical_skills'].notna().sum() / len(self.df) * 100),
                'total_skill_instances': len(self.skills_data) if self.skills_data is not None else 0,
                'unique_skills': self.skills_data['skill'].nunique() if self.skills_data is not None else 0
            },
            'top_skills': self.get_top_skills_by_frequency(20),
            'geographic_analysis': self.get_skills_by_geography(10, 10),
            'industry_analysis': self.get_skills_by_industry(10, 10),
            'salary_correlation': self.get_skills_salary_correlation(50),
            'emerging_skills': self.get_emerging_skills(20, 200),
            'experience_trends': self.get_skills_trends_by_experience()
        }

        logger.info("Skills analysis report generated successfully")
        return report


def run_skills_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run comprehensive skills analysis on job market data.

    Args:
        df: Pandas DataFrame with job market data

    Returns:
        Dictionary with complete skills analysis results
    """
    analyzer = SkillsAnalyzer(df)
    return analyzer.get_comprehensive_skills_report()
