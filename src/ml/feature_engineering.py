"""
Feature Engineering for Salary Disparity Analysis

This module provides feature engineering capabilities specifically designed
for salary disparity analysis and machine learning models.
"""

from typing import Dict, List, Optional, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, when, isnan, isnull, regexp_replace, split, size,
    length, upper, lower, trim, lit, concat_ws, coalesce,
    sum as spark_sum, count, avg, max as spark_max, min as spark_min,
    row_number, rank, dense_rank, percent_rank, cume_dist
)

# Import logger for controlled output
from src.utils.logger import get_logger

from pyspark.sql.window import Window
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, MinMaxScaler, PCA
)

from pyspark.ml import Pipeline
import re

logger = get_logger(level="WARNING")

class SalaryDisparityFeatureEngineer:
    """
    Feature engineering specifically for salary disparity analysis.

    Creates features that help identify and analyze salary gaps based on:
    - Demographics (experience, education, location)
    - Job characteristics (title, industry, company size)
    - Skills (AI, technical, soft skills)
    - Work arrangements (remote, employment type)
    - Market factors (location cost, industry growth)
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

        # Define feature categories for salary disparity analysis
        self.feature_categories = {
            'demographic': [
                'experience_level', 'education_level', 'location_tier',
                'age_group', 'gender_indicators'
            ],
            'job_characteristics': [
                'job_title_category', 'industry_category', 'company_size_tier',
                'seniority_level', 'management_level'
            ],
            'skills': [
                'ai_skills_score', 'technical_skills_score', 'soft_skills_score',
                'programming_languages', 'certifications', 'degree_level'
            ],
            'work_arrangement': [
                'remote_type', 'employment_type', 'work_schedule',
                'benefits_level', 'flexibility_score'
            ],
            'market_factors': [
                'location_cost_index', 'industry_growth_rate', 'skill_demand_score',
                'competition_level', 'market_saturation'
            ]
        }

        # Salary disparity specific features
        self.salary_disparity_features = [
            'salary_percentile', 'above_median_salary', 'salary_gap_indicator',
            'experience_salary_ratio', 'education_salary_ratio', 'location_salary_ratio'
        ]

    def create_demographic_features(self, df: DataFrame) -> DataFrame:
        """Create demographic features for salary disparity analysis."""

        # Experience level categorization
        df = df.withColumn(
            'experience_level',
            when(col('EXPERIENCE_LEVEL').isNull(), 'Unknown')
            .when(col('EXPERIENCE_LEVEL').contains('Entry'), 'Entry')
            .when(col('EXPERIENCE_LEVEL').contains('Mid'), 'Mid')
            .when(col('EXPERIENCE_LEVEL').contains('Senior'), 'Senior')
            .when(col('EXPERIENCE_LEVEL').contains('Executive'), 'Executive')
            .otherwise('Other')
        )

        # Education level encoding
        df = df.withColumn(
            'education_level',
            when(col('EDUCATION_LEVEL').isNull(), 0)
            .when(col('EDUCATION_LEVEL').contains('High School'), 1)
            .when(col('EDUCATION_LEVEL').contains('Associate'), 2)
            .when(col('EDUCATION_LEVEL').contains('Bachelor'), 3)
            .when(col('EDUCATION_LEVEL').contains('Master'), 4)
            .when(col('EDUCATION_LEVEL').contains('PhD'), 5)
            .otherwise(0)
        )

        # Location tier classification
        df = df.withColumn(
            'location_tier',
            when(col('CITY').isNull(), 'Unknown')
            .when(col('CITY').contains('New York') | col('CITY').contains('San Francisco') |
                  col('CITY').contains('Los Angeles') | col('CITY').contains('Chicago'), 'Tier 1')
            .when(col('CITY').contains('Boston') | col('CITY').contains('Seattle') |
                  col('CITY').contains('Austin') | col('CITY').contains('Denver'), 'Tier 2')
            .otherwise('Tier 3')
        )

        return df

    def create_job_characteristics_features(self, df: DataFrame) -> DataFrame:
        """Create job characteristic features for salary disparity analysis."""

        # Job title categorization
        df = df.withColumn(
            'job_title_category',
            when(col('TITLE').isNull(), 'Unknown')
            .when(col('TITLE').rlike('(?i)(manager|director|vp|vice president|head|lead)'), 'Management')
            .when(col('TITLE').rlike('(?i)(senior|sr|principal|staff)'), 'Senior')
            .when(col('TITLE').rlike('(?i)(junior|jr|entry|associate)'), 'Junior')
            .when(col('TITLE').rlike('(?i)(intern|trainee|apprentice)'), 'Entry')
            .otherwise('Individual Contributor')
        )

        # Industry categorization
        df = df.withColumn(
            'industry_category',
            when(col('INDUSTRY').isNull(), 'Unknown')
            .when(col('INDUSTRY').rlike('(?i)(technology|software|tech|it)'), 'Technology')
            .when(col('INDUSTRY').rlike('(?i)(finance|banking|financial)'), 'Finance')
            .when(col('INDUSTRY').rlike('(?i)(healthcare|medical|pharma)'), 'Healthcare')
            .when(col('INDUSTRY').rlike('(?i)(consulting|professional services)'), 'Consulting')
            .otherwise('Other')
        )

        # Company size estimation (based on job posting patterns)
        df = df.withColumn(
            'company_size_tier',
            when(col('COMPANY').isNull(), 'Unknown')
            .when(col('COMPANY').rlike('(?i)(inc|corp|corporation|llc)'), 'Large')
            .when(col('COMPANY').rlike('(?i)(startup|start-up|small)'), 'Small')
            .otherwise('Medium')
        )

        return df

    def create_skills_features(self, df: DataFrame) -> DataFrame:
        """Create skills-based features for salary disparity analysis."""

        # AI skills detection
        ai_keywords = ['artificial intelligence', 'machine learning', 'deep learning',
                      'neural network', 'ai', 'ml', 'nlp', 'computer vision', 'data science']

        df = df.withColumn(
            'ai_skills_score',
            sum([
                when(col('TITLE').rlike(f'(?i){keyword}'), 1).otherwise(0)
                for keyword in ai_keywords
            ])
        )

        # Technical skills detection
        tech_keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'azure',
                        'docker', 'kubernetes', 'react', 'angular', 'node.js']

        df = df.withColumn(
            'technical_skills_score',
            sum([
                when(col('TITLE').rlike(f'(?i){keyword}'), 1).otherwise(0)
                for keyword in tech_keywords
            ])
        )

        # Soft skills detection
        soft_skills_keywords = ['communication', 'leadership', 'team', 'collaboration',
                               'problem solving', 'analytical', 'creative', 'strategic']

        df = df.withColumn(
            'soft_skills_score',
            sum([
                when(col('TITLE').rlike(f'(?i){keyword}'), 1).otherwise(0)
                for keyword in soft_skills_keywords
            ])
        )

        return df

    def create_work_arrangement_features(self, df: DataFrame) -> DataFrame:
        """Create work arrangement features for salary disparity analysis."""

        # Remote work classification
        df = df.withColumn(
            'remote_type',
            when(col('REMOTE_TYPE').isNull(), 'Unknown')
            .when(col('REMOTE_TYPE').contains('Remote'), 'Remote')
            .when(col('REMOTE_TYPE').contains('Hybrid'), 'Hybrid')
            .when(col('REMOTE_TYPE').contains('On-site'), 'On-site')
            .otherwise('Unknown')
        )

        # Employment type classification
        df = df.withColumn(
            'employment_type',
            when(col('EMPLOYMENT_TYPE').isNull(), 'Unknown')
            .when(col('EMPLOYMENT_TYPE').contains('Full-time'), 'Full-time')
            .when(col('EMPLOYMENT_TYPE').contains('Part-time'), 'Part-time')
            .when(col('EMPLOYMENT_TYPE').contains('Contract'), 'Contract')
            .otherwise('Unknown')
        )

        return df

    def create_market_factors_features(self, df: DataFrame) -> DataFrame:
        """Create market factor features for salary disparity analysis."""

        # Location cost index (simplified)
        df = df.withColumn(
            'location_cost_index',
            when(col('location_tier') == 'Tier 1', 1.0)
            .when(col('location_tier') == 'Tier 2', 0.8)
            .when(col('location_tier') == 'Tier 3', 0.6)
            .otherwise(0.5)
        )

        # Industry growth rate (simplified)
        df = df.withColumn(
            'industry_growth_rate',
            when(col('industry_category') == 'Technology', 1.2)
            .when(col('industry_category') == 'Finance', 1.1)
            .when(col('industry_category') == 'Healthcare', 1.15)
            .when(col('industry_category') == 'Consulting', 1.05)
            .otherwise(1.0)
        )

        return df

    def create_salary_disparity_features(self, df: DataFrame) -> DataFrame:
        """Create salary disparity specific features."""

        # Calculate salary percentiles
        window_spec = Window.orderBy(col('salary_avg'))
        df = df.withColumn('salary_percentile', percent_rank().over(window_spec))

        # Above median salary indicator
        median_salary = df.select(avg('salary_avg')).collect()[0][0]
        df = df.withColumn('above_median_salary',
                          when(col('salary_avg') > median_salary, 1).otherwise(0))

        # Salary gap indicators
        df = df.withColumn(
            'experience_salary_ratio',
            col('salary_avg') / col('education_level')
        )

        df = df.withColumn(
            'education_salary_ratio',
            col('salary_avg') / col('education_level')
        )

        return df

    def prepare_ml_features(self, df: DataFrame) -> DataFrame:
        """Prepare all features for machine learning models."""

        logger.info("Creating demographic features...")
        df = self.create_demographic_features(df)

        logger.info("Creating job characteristics features...")
        df = self.create_job_characteristics_features(df)

        logger.info("Creating skills features...")
        df = self.create_skills_features(df)

        logger.info("Creating work arrangement features...")
        df = self.create_work_arrangement_features(df)

        logger.info("Creating market factors features...")
        df = self.create_market_factors_features(df)

        logger.info("Creating salary disparity features...")
        df = self.create_salary_disparity_features(df)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns for ML models."""
        all_features = []
        for category, features in self.feature_categories.items():
            all_features.extend(features)
        all_features.extend(self.salary_disparity_features)
        return all_features

    def create_feature_pipeline(self, feature_columns: List[str]) -> Pipeline:
        """Create a Spark ML pipeline for feature processing."""

        # String indexers for categorical features
        categorical_features = [
            'experience_level', 'location_tier', 'job_title_category',
            'industry_category', 'company_size_tier', 'remote_type', 'employment_type'
        ]

        string_indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
            for col in categorical_features if col in feature_columns
        ]

        # One-hot encoders for categorical features
        one_hot_encoders = [
            OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
            for col in categorical_features if col in feature_columns
        ]

        # Vector assembler for all features
        vector_assembler = VectorAssembler(
            inputCols=[col for col in feature_columns if col not in categorical_features] +
                     [f"{col}_encoded" for col in categorical_features if col in feature_columns],
            outputCol="features"
        )

        # Create pipeline
        pipeline = Pipeline(stages=string_indexers + one_hot_encoders + [vector_assembler])

        return pipeline
