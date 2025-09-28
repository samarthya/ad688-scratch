"""
Spark-based Job Market Analysis Module

Provides comprehensive analysis capabilities for the Lightcast job market dataset
using Apache Spark for scalable data processing.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, avg, median, stddev, max, min, when, desc, asc
from typing import Optional

class SparkJobAnalyzer:
    """
    Comprehensive job market analysis using Spark for scalable processing.
    Designed specifically for Lightcast job posting data.
    """
    
    def __init__(self, df: DataFrame):
        """Initialize with Lightcast DataFrame"""
        self.df = df
        
    def analyze_experience_levels(self):
        """Analyze job distribution and salary by experience level"""
        
        if 'experience_level' not in self.df.columns:
            # Create basic experience level if not exists
            self.df = self.df.withColumn(
                'experience_level',
                when(col('experience_required').isNull(), 'Not Specified')
                .otherwise('Specified')
            )
        
        experience_analysis = self.df.groupBy('experience_level').agg(
            count('*').alias('Job Count'),
            avg('salary').alias('Average Salary'),
            median('salary').alias('Median Salary'),
            stddev('salary').alias('Salary Std Dev'),
            min('salary').alias('Min Salary'),
            max('salary').alias('Max Salary')
        ).orderBy(desc('Job Count'))
        
        return experience_analysis
    
    def analyze_by_industry(self):
        """Analyze job distribution and salary by industry"""
        
        # Use industry column or create from available data
        industry_col = 'industry'
        if industry_col not in self.df.columns:
            # Fallback to other industry-related columns
            for alt_col in ['naics2_name', 'sector', 'industry_name']:
                if alt_col in self.df.columns:
                    industry_col = alt_col
                    break
                    
        if industry_col in self.df.columns:
            industry_analysis = self.df.filter(col(industry_col).isNotNull()).groupBy(industry_col).agg(
                count('*').alias('Job Count'),
                avg('salary').alias('Average Salary'),
                median('salary').alias('Median Salary'),
                stddev('salary').alias('Salary Std Dev')
            ).withColumnRenamed(industry_col, 'Industry').orderBy(desc('Job Count'))
            
            return industry_analysis
        else:
            # Return empty result if no industry column available
            return self.df.limit(0).select(
                col('job_title').alias('Industry'),
                count('*').alias('Job Count'),
                avg('salary').alias('Average Salary'),
                median('salary').alias('Median Salary'),
                stddev('salary').alias('Salary Std Dev')
            )
    
    def analyze_by_location(self):
        """Analyze job distribution and salary by location"""
        
        location_col = 'location'
        if location_col not in self.df.columns:
            # Try alternative location columns
            for alt_col in ['city', 'state', 'location_name', 'job_location']:
                if alt_col in self.df.columns:
                    location_col = alt_col
                    break
        
        if location_col in self.df.columns:
            location_analysis = self.df.filter(col(location_col).isNotNull()).groupBy(location_col).agg(
                count('*').alias('Job Count'),
                avg('salary').alias('Average Salary'),
                median('salary').alias('Median Salary'),
                stddev('salary').alias('Salary Std Dev')
            ).withColumnRenamed(location_col, 'Location').orderBy(desc('Job Count'))
            
            return location_analysis
        else:
            # Return empty result if no location column available
            return self.df.limit(0).select(
                col('job_title').alias('Location'),
                count('*').alias('Job Count'),
                avg('salary').alias('Average Salary'),
                median('salary').alias('Median Salary'),
                stddev('salary').alias('Salary Std Dev')
            )
    
    def analyze_skills_demand(self):
        """Analyze skills demand across job postings"""
        
        skills_col = 'skills'
        if skills_col not in self.df.columns:
            # Try alternative skills columns
            for alt_col in ['required_skills', 'skill_requirements', 'skills_required']:
                if alt_col in self.df.columns:
                    skills_col = alt_col
                    break
        
        if skills_col in self.df.columns:
            # This is a simplified analysis - in practice, you'd want to parse skills properly
            skills_analysis = self.df.filter(col(skills_col).isNotNull()).groupBy(skills_col).agg(
                count('*').alias('Demand Count'),
                avg('salary').alias('Average Salary'),
                median('salary').alias('Median Salary')
            ).withColumnRenamed(skills_col, 'Skill').orderBy(desc('Demand Count'))
            
            return skills_analysis.limit(50)  # Top 50 skills
        else:
            # Return empty result if no skills column available
            return self.df.limit(0).select(
                col('job_title').alias('Skill'),
                count('*').alias('Demand Count'),
                avg('salary').alias('Average Salary'),
                median('salary').alias('Median Salary')
            )
    
    def analyze_salary_ranges(self):
        """Analyze salary distribution and ranges"""
        
        if 'salary' not in self.df.columns:
            return None
            
        # Define salary brackets
        salary_brackets = self.df.select(
            when(col('salary') < 50000, 'Under $50K')
            .when((col('salary') >= 50000) & (col('salary') < 75000), '$50K-$75K')
            .when((col('salary') >= 75000) & (col('salary') < 100000), '$75K-$100K')
            .when((col('salary') >= 100000) & (col('salary') < 150000), '$100K-$150K')
            .when((col('salary') >= 150000) & (col('salary') < 200000), '$150K-$200K')
            .when(col('salary') >= 200000, '$200K+')
            .otherwise('Unknown').alias('Salary Range'),
            col('salary')
        ).filter(col('salary').isNotNull())
        
        range_analysis = salary_brackets.groupBy('Salary Range').agg(
            count('*').alias('Job Count'),
            avg('salary').alias('Average Salary'),
            median('salary').alias('Median Salary'),
            min('salary').alias('Range Min'),
            max('salary').alias('Range Max')
        ).orderBy('Average Salary')
        
        return range_analysis
    
    def get_dataset_overview(self):
        """Get comprehensive dataset overview statistics"""
        
        total_records = self.df.count()
        total_columns = len(self.df.columns)
        
        # Check for key columns and their coverage
        key_columns = ['job_title', 'company', 'location', 'salary', 'skills', 'industry']
        column_coverage = {}
        
        for col_name in key_columns:
            if col_name in self.df.columns:
                non_null_count = self.df.filter(col(col_name).isNotNull()).count()
                coverage_pct = (non_null_count / total_records) * 100
                column_coverage[col_name] = {
                    'exists': True,
                    'non_null_count': non_null_count,
                    'coverage_percentage': coverage_pct
                }
            else:
                column_coverage[col_name] = {'exists': False}
        
        overview = {
            'total_records': total_records,
            'total_columns': total_columns,
            'column_coverage': column_coverage
        }
        
        return overview