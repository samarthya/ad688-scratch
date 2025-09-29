"""
Salary Processing Module for Lightcast Data

This module provides salary processing capabilities specifically designed
for the Lightcast job postings dataset with comprehensive salary analysis.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_replace, avg, median, count, stddev, lit
from pyspark.sql.types import DoubleType
from typing import Optional, Dict, List
import re

class SalaryProcessor:
    """
    Process and standardize salary information from Lightcast job postings.
    Handles salary ranges, pay periods, and missing salary data imputation.
    """
    
    def __init__(self, df: DataFrame):
        """Initialize with Lightcast DataFrame"""
        self.df = df
        self.processed_df: Optional[DataFrame] = None
        
    def process_salary_data(self):
        """
        Comprehensive salary processing for Lightcast data.
        Returns DataFrame with standardized salary column.
        """
        
        # Start with the input DataFrame
        working_df = self.df
        
        # Standardize salary columns if they exist
        if 'salary_from' in working_df.columns and 'salary_to' in working_df.columns:
            # Calculate midpoint for salary ranges
            working_df = working_df.withColumn(
                'salary_midpoint',
                (col('salary_from') + col('salary_to')) / 2
            )
            
            # Use midpoint as primary salary, fallback to salary_from
            working_df = working_df.withColumn(
                'salary',
                when(col('salary_midpoint').isNotNull(), col('salary_midpoint'))
                .otherwise(col('salary_from'))
            )
            
        elif 'salary_from' in working_df.columns:
            # Use salary_from as primary salary
            working_df = working_df.withColumn('salary', col('salary_from'))
            
        # Clean and standardize existing salary column if present
        if 'salary' in working_df.columns:
            # Remove common salary text patterns and convert to numeric
            working_df = working_df.withColumn(
                'salary_clean',
                regexp_replace(col('salary').cast('string'), r'[,$]', '')
            ).withColumn(
                'salary_numeric',
                col('salary_clean').cast(DoubleType())
            )
            
            # Filter out unrealistic salaries (likely errors)
            working_df = working_df.withColumn(
                'salary_final',
                when(
                    (col('salary_numeric') >= 20000) & (col('salary_numeric') <= 500000),
                    col('salary_numeric')
                ).otherwise(None)
            )
            
        else:
            # If no salary column exists, create a placeholder
            working_df = working_df.withColumn('salary_final', col('salary_from').cast(DoubleType()))
            
        # Store processed result
        self.processed_df = working_df.withColumn('salary', col('salary_final'))
        
        return self.processed_df
    
    def get_salary_statistics(self):
        """Get comprehensive salary statistics"""
        
        if self.processed_df is None:
            self.process_salary_data()
            
        if self.processed_df is None:
            raise ValueError("Failed to process salary data")
            
        # Calculate statistics on non-null salaries
        salary_stats = self.processed_df.select(
            count('salary').alias('salary_count'),
            count('*').alias('total_count'),
            avg('salary').alias('avg_salary'),
            median('salary').alias('median_salary'),
            stddev('salary').alias('stddev_salary')
        ).collect()[0]
        
        stats_dict = {
            'total_records': salary_stats['total_count'],
            'records_with_salary': salary_stats['salary_count'],
            'salary_coverage_pct': (salary_stats['salary_count'] / salary_stats['total_count']) * 100,
            'average_salary': salary_stats['avg_salary'],
            'median_salary': salary_stats['median_salary'],
            'salary_std_dev': salary_stats['stddev_salary']
        }
        
        return stats_dict
    
    def impute_missing_salaries(self, grouping_columns=['industry', 'experience_level', 'location']):
        """
        Impute missing salaries using hierarchical approach.
        Uses median salaries from similar job characteristics.
        """
        
        if self.processed_df is None:
            self.process_salary_data()
            
        if self.processed_df is None:
            raise ValueError("Failed to process salary data")
            
        working_df = self.processed_df
        
        # Calculate median salaries for different groupings
        available_groupings = [col for col in grouping_columns if col in working_df.columns]
        
        if available_groupings:
            # Create reference medians for imputation
            for grouping in available_groupings:
                median_df = working_df.filter(col('salary').isNotNull()) \
                    .groupBy(grouping) \
                    .agg(median('salary').alias(f'{grouping}_median_salary'))
                
                # Join back to main DataFrame
                working_df = working_df.join(median_df, on=grouping, how='left')
                
            # Impute using hierarchical approach
            imputation_expr = col('salary')
            for grouping in available_groupings:
                median_col = f'{grouping}_median_salary'
                imputation_expr = when(
                    imputation_expr.isNull(), 
                    col(median_col)
                ).otherwise(imputation_expr)
            
            # Apply imputation
            working_df = working_df.withColumn('salary_imputed', imputation_expr)
            
        else:
            # Fallback to global median if no grouping columns available
            global_median = working_df.agg(median('salary')).collect()[0][0]
            working_df = working_df.withColumn(
                'salary_imputed',
                when(col('salary').isNull(), global_median).otherwise(col('salary'))
            )
        
        self.processed_df = working_df
        return working_df
    
    def get_processed_data(self):
        """Return the processed DataFrame with standardized salary data"""
        
        if self.processed_df is None:
            self.process_salary_data()
            
        if self.processed_df is None:
            raise ValueError("Failed to process salary data")
            
        return self.processed_df