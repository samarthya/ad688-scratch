"""
Core Spark-based job market analyzer.

This module provides the main analysis engine for job market data
using Apache Spark for big data processing.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, avg, count, max as spark_max, min as spark_min
from pathlib import Path
import logging

from ..utils.spark_utils import create_spark_session
from ..config.settings import Settings
from ..data.loaders import DataLoader
from ..data.validators import DataValidator

logger = logging.getLogger(__name__)


class SparkJobAnalyzer:
    """
    Core Spark-based analyzer for job market data.

    Provides comprehensive analysis capabilities for salary disparity,
    job market trends, and statistical insights.
    """

    def __init__(self, spark: Optional[SparkSession] = None, settings: Optional[Settings] = None):
        """Initialize the analyzer with Spark session and settings."""
        self.spark = spark or create_spark_session("Job Market Analyzer")
        self.settings = settings or Settings()
        self.data_loader = DataLoader(self.spark, self.settings)
        self.data_validator = DataValidator()
        self.job_data: Optional[DataFrame] = None

    def load_full_dataset(self, data_path: Optional[str] = None, force_raw: bool = False) -> DataFrame:
        """Load the full dataset with automatic fallback strategy."""

        if force_raw or (data_path and not data_path.endswith('.parquet')):
            # Load raw data
            logger.info("Loading raw data...")
            self.job_data = self.data_loader.load_raw_data(data_path)

            # Raw data validation - check for basic structure
            validation_results = self._validate_raw_data(self.job_data)
            if not validation_results['is_valid']:
                logger.error(f"Raw data validation failed: {validation_results['errors']}")
                raise ValueError(f"Raw data validation failed: {validation_results['errors']}")

        else:
            # Load processed data
            logger.info("Loading processed data...")
            self.job_data = self.data_loader.load_processed_data(data_path)

            # Processed data validation - check for analysis columns
            validation_results = self.data_validator.validate_dataset(self.job_data)
            if not validation_results['is_valid']:
                logger.error(f"Data validation failed: {validation_results['errors']}")
                raise ValueError(f"Data validation failed: {validation_results['errors']}")

        logger.info(f"Dataset loaded successfully: {self.job_data.count():,} records")
        return self.job_data

    def _validate_raw_data(self, df: DataFrame) -> Dict[str, Any]:
        """Validate raw data structure for basic requirements."""
        errors = []

        # Check for essential columns
        essential_columns = ['TITLE', 'COMPANY']
        missing_essential = [col for col in essential_columns if col not in df.columns]
        if missing_essential:
            errors.extend([f"Missing essential column: {col}" for col in missing_essential])

        # Check for optional but important columns
        optional_columns = ['SALARY_AVG_IMPUTED', 'INDUSTRY', 'LOCATION']
        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            logger.warning(f"Missing optional columns: {missing_optional}")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': missing_optional
        }

    def get_industry_analysis(self, top_n: int = 10) -> Dict[str, Any]:
        """Get industry salary analysis."""
        if self.job_data is None:
            raise ValueError("No data loaded. Call load_full_dataset() first.")

        # Industry analysis query
        industry_query = """
            SELECT
                COALESCE(INDUSTRY, 'Unknown') as industry,
                COUNT(*) as job_count,
                ROUND(AVG(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as avg_salary,
                ROUND(MIN(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as min_salary,
                ROUND(MAX(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as max_salary
            FROM job_data
            WHERE SALARY_AVG_IMPUTED IS NOT NULL AND SALARY_AVG_IMPUTED > 0
            GROUP BY INDUSTRY
            ORDER BY avg_salary DESC
            LIMIT {}
        """.format(top_n)

        # Create temporary view
        self.job_data.createOrReplaceTempView("job_data")

        # Execute query
        industry_df = self.spark.sql(industry_query)
        industry_results = industry_df.collect()

        return {
            'data': [dict(row.asDict()) for row in industry_results],
            'summary': f"Top {top_n} industries by average salary"
        }

    def get_experience_analysis(self) -> Dict[str, Any]:
        """Get experience level salary analysis."""
        if self.job_data is None:
            raise ValueError("No data loaded. Call load_full_dataset() first.")

        # Experience analysis query
        exp_query = """
            SELECT
                COALESCE(EXPERIENCE_LEVEL, 'Unknown') as experience_level,
                COUNT(*) as job_count,
                ROUND(AVG(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as avg_salary
            FROM job_data
            WHERE SALARY_AVG_IMPUTED IS NOT NULL AND SALARY_AVG_IMPUTED > 0
            GROUP BY EXPERIENCE_LEVEL
            ORDER BY avg_salary DESC
        """

        # Create temporary view
        self.job_data.createOrReplaceTempView("job_data")

        # Execute query
        exp_df = self.spark.sql(exp_query)
        exp_results = exp_df.collect()

        return {
            'data': [dict(row.asDict()) for row in exp_results],
            'summary': "Experience level salary analysis"
        }

    def get_geographic_analysis(self, top_n: int = 10) -> Dict[str, Any]:
        """Get geographic salary analysis."""
        if self.job_data is None:
            raise ValueError("No data loaded. Call load_full_dataset() first.")

        # Geographic analysis query
        geo_query = """
            SELECT
                COALESCE(STATE, 'Unknown') as state,
                COUNT(*) as job_count,
                ROUND(AVG(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as avg_salary
            FROM job_data
            WHERE SALARY_AVG_IMPUTED IS NOT NULL AND SALARY_AVG_IMPUTED > 0
            GROUP BY STATE
            ORDER BY avg_salary DESC
            LIMIT {}
        """.format(top_n)

        # Create temporary view
        self.job_data.createOrReplaceTempView("job_data")

        # Execute query
        geo_df = self.spark.sql(geo_query)
        geo_results = geo_df.collect()

        return {
            'data': [dict(row.asDict()) for row in geo_results],
            'summary': f"Top {top_n} states by average salary"
        }

    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall dataset statistics."""
        if self.job_data is None:
            raise ValueError("No data loaded. Call load_full_dataset() first.")

        # Overall statistics query
        stats_query = """
            SELECT
                COUNT(*) as total_jobs,
                COUNT(DISTINCT COMPANY) as unique_companies,
                COUNT(DISTINCT INDUSTRY) as unique_industries,
                ROUND(AVG(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as avg_salary,
                ROUND(MIN(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as min_salary,
                ROUND(MAX(COALESCE(SALARY_AVG_IMPUTED, 0)), 0) as max_salary
            FROM job_data
            WHERE SALARY_AVG_IMPUTED IS NOT NULL AND SALARY_AVG_IMPUTED > 0
        """

        # Create temporary view
        self.job_data.createOrReplaceTempView("job_data")

        # Execute query
        stats_df = self.spark.sql(stats_query)
        stats_results = stats_df.collect()[0]

        return dict(stats_results.asDict())

    def execute_custom_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a custom SQL query on the dataset."""
        if self.job_data is None:
            raise ValueError("No data loaded. Call load_full_dataset() first.")

        # Create temporary view
        self.job_data.createOrReplaceTempView("job_data")

        # Execute query
        result_df = self.spark.sql(query)
        results = result_df.collect()

        return [dict(row.asDict()) for row in results]


def create_spark_analyzer(data_path: Optional[str] = None, force_raw: bool = False) -> SparkJobAnalyzer:
    """Create a SparkJobAnalyzer with automatic data loading."""
    analyzer = SparkJobAnalyzer()
    analyzer.load_full_dataset(data_path, force_raw)
    return analyzer
