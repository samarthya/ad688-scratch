"""
Unified Spark-based Analysis Engine for Job Market Data

This module provides the main analysis engine for job market data using PySpark,
consolidating functionality from the original SparkJobAnalyzer with improved
error handling and configuration management.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, count, avg, sum as spark_sum, median, desc, asc,
    min as spark_min, max as spark_max, stddev, percentile_approx,
    regexp_replace, trim, upper, lower, lit, current_timestamp,
    expr, year, month, dayofmonth, to_date, split, isnan, isnull
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, DateType
)

from src.config.settings import get_settings
from src.config.schemas import LIGHTCAST_SCHEMA, PROCESSED_SCHEMA
from src.data.loaders import DataLoader
from src.data.validators import DataValidator
from src.utils.spark_utils import create_spark_session, stop_spark_session
from src.core.exceptions import DataLoadingError, DataValidationError, ProcessingError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkJobAnalyzer:
    """
    Unified PySpark-based job market analysis engine.

    This class consolidates all analysis functionality with improved error handling,
    configuration management, and data validation.
    """

    def __init__(self, spark_session: Optional[SparkSession] = None):
        """Initialize analyzer with Spark session and configuration."""
        self.settings = get_settings()

        if spark_session is None:
            self.spark = create_spark_session()
            self._owns_spark = True
        else:
            self.spark = spark_session
            self._owns_spark = False

        # Initialize data access components
        self.data_loader = DataLoader(self.spark)
        self.data_validator = DataValidator()

        self.job_data: Optional[DataFrame] = None
        logger.info(f"SparkJobAnalyzer initialized with Spark {self.spark.version}")

    def load_full_dataset(self, data_path: Optional[str] = None,
                         force_raw: bool = False) -> DataFrame:
        """
        Load the full dataset with flexible data source selection.

        Args:
            data_path: Optional path to specific data file
            force_raw: If True, bypasses processed data and loads from raw Lightcast CSV

        Returns:
            Spark DataFrame with full dataset

        Raises:
            DataLoadingError: If data sources don't exist
            DataValidationError: If data validation fails
        """
        try:
            if force_raw:
                if data_path and Path(data_path).exists():
                    logger.info(f"FORCE RAW MODE: Loading from specified path: {data_path}")
                    if data_path.endswith('.parquet'):
                        self.job_data = self.spark.read.parquet(data_path)
                    else:
                        self.job_data = self.spark.read.option("header", "true").csv(data_path)
                else:
                    logger.info("FORCE RAW MODE: Loading from raw source")
                    self.job_data = self.data_loader.load_raw_data()
            elif data_path and Path(data_path).exists():
                logger.info(f"Loading from specified path: {data_path}")
                if data_path.endswith('.parquet'):
                    self.job_data = self.spark.read.parquet(data_path)
                else:
                    self.job_data = self.spark.read.option("header", "true").csv(data_path)
            else:
                logger.info("Loading with automatic fallback")
                self.job_data = self.data_loader.load_data_with_fallback()

            # Validate loaded data based on data type
            if force_raw or (data_path and not data_path.endswith('.parquet')):
                # Raw data validation - check for basic structure
                validation_results = self._validate_raw_data(self.job_data)
            else:
                # Processed data validation - check for analysis columns
                validation_results = self.data_validator.validate_dataset(self.job_data)

            if not validation_results["is_valid"]:
                raise DataValidationError(f"Data validation failed: {validation_results['errors']}")

            logger.info(f"Dataset loaded successfully: {self.job_data.count():,} records")
            return self.job_data

        except Exception as e:
            if isinstance(e, (DataLoadingError, DataValidationError)):
                raise
            else:
                raise DataLoadingError(f"Failed to load dataset: {str(e)}")

    def _validate_raw_data(self, df: DataFrame) -> Dict[str, Any]:
        """Validate raw data structure for basic requirements."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_metrics": {}
        }

        # Check if dataset is empty
        record_count = df.count()
        if record_count == 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Dataset is empty")
            return validation_results

        validation_results["quality_metrics"]["total_records"] = record_count

        # Check for basic raw data columns (not processed columns)
        raw_data_columns = [
            "TITLE", "COMPANY", "LOCATION", "SALARY_FROM", "SALARY_TO",
            "NAICS2_NAME", "MIN_YEARS_EXPERIENCE", "MAX_YEARS_EXPERIENCE",
            "REMOTE_TYPE_NAME", "EMPLOYMENT_TYPE_NAME"
        ]

        missing_columns = []
        for col_name in raw_data_columns:
            if col_name not in df.columns:
                missing_columns.append(col_name)

        # Only require a few essential columns for raw data
        essential_columns = ["TITLE", "COMPANY"]
        missing_essential = [col for col in essential_columns if col in missing_columns]

        if missing_essential:
            validation_results["is_valid"] = False
            validation_results["errors"].extend([f"Missing essential column: {col}" for col in missing_essential])
        else:
            # Add warnings for missing optional columns
            optional_missing = [col for col in missing_columns if col not in essential_columns]
            if optional_missing:
                validation_results["warnings"].extend([f"Missing optional column: {col}" for col in optional_missing])

        return validation_results

    def get_industry_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Get industry analysis with salary statistics."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            industry_stats = self.job_data.groupBy("industry") \
                .agg(
                    count("*").alias("job_count"),
                    avg("salary_avg_imputed").alias("avg_salary"),
                    median("salary_avg_imputed").alias("median_salary"),
                    min("salary_avg_imputed").alias("min_salary"),
                    max("salary_avg_imputed").alias("max_salary"),
                    stddev("salary_avg_imputed").alias("std_salary")
                ) \
                .filter(col("avg_salary").isNotNull()) \
                .orderBy(desc("median_salary")) \
                .limit(top_n)

            return industry_stats.toPandas()

        except Exception as e:
            raise ProcessingError(f"Failed to generate industry analysis: {str(e)}")

    def get_experience_analysis(self) -> pd.DataFrame:
        """Get experience level analysis with salary progression."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            experience_stats = self.job_data.groupBy("experience_level") \
                .agg(
                    count("*").alias("job_count"),
                    avg("salary_avg_imputed").alias("avg_salary"),
                    median("salary_avg_imputed").alias("median_salary"),
                    min("salary_avg_imputed").alias("min_salary"),
                    max("salary_avg_imputed").alias("max_salary")
                ) \
                .filter(col("avg_salary").isNotNull()) \
                .orderBy(desc("median_salary"))

            return experience_stats.toPandas()

        except Exception as e:
            raise ProcessingError(f"Failed to generate experience analysis: {str(e)}")

    def get_geographic_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Get geographic analysis with location-based salary statistics."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            # Extract state from location
            df_with_state = self.job_data.withColumn(
                "state",
                split(col("location"), ",").getItem(1)
            ).withColumn(
                "state",
                trim(col("state"))
            )

            geo_stats = df_with_state.groupBy("state") \
                .agg(
                    count("*").alias("job_count"),
                    avg("salary_avg_imputed").alias("avg_salary"),
                    median("salary_avg_imputed").alias("median_salary")
                ) \
                .filter(col("avg_salary").isNotNull()) \
                .orderBy(desc("median_salary")) \
                .limit(top_n)

            return geo_stats.toPandas()

        except Exception as e:
            raise ProcessingError(f"Failed to generate geographic analysis: {str(e)}")

    def get_skills_analysis(self, top_n: int = 20) -> pd.DataFrame:
        """Get skills analysis with salary impact."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            # This is a simplified version - full implementation would parse skills
            skills_stats = self.job_data.groupBy("required_skills") \
                .agg(
                    count("*").alias("job_count"),
                    avg("salary_avg_imputed").alias("avg_salary"),
                    median("salary_avg_imputed").alias("median_salary")
                ) \
                .filter(col("avg_salary").isNotNull()) \
                .orderBy(desc("median_salary")) \
                .limit(top_n)

            return skills_stats.toPandas()

        except Exception as e:
            raise ProcessingError(f"Failed to generate skills analysis: {str(e)}")

    def get_ai_analysis(self) -> pd.DataFrame:
        """Get AI/ML role analysis."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            ai_stats = self.job_data.groupBy("ai_related") \
                .agg(
                    count("*").alias("job_count"),
                    avg("salary_avg_imputed").alias("avg_salary"),
                    median("salary_avg_imputed").alias("median_salary")
                ) \
                .filter(col("avg_salary").isNotNull())

            return ai_stats.toPandas()

        except Exception as e:
            raise ProcessingError(f"Failed to generate AI analysis: {str(e)}")

    def get_remote_work_analysis(self) -> pd.DataFrame:
        """Get remote work analysis."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            remote_stats = self.job_data.groupBy("remote_allowed") \
                .agg(
                    count("*").alias("job_count"),
                    avg("salary_avg_imputed").alias("avg_salary"),
                    median("salary_avg_imputed").alias("median_salary")
                ) \
                .filter(col("avg_salary").isNotNull())

            return remote_stats.toPandas()

        except Exception as e:
            raise ProcessingError(f"Failed to generate remote work analysis: {str(e)}")

    def get_overall_statistics(self) -> Dict[str, float]:
        """Get overall dataset statistics."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            stats = self.job_data.select(
                count("*").alias("total_jobs"),
                avg("salary_avg_imputed").alias("avg_salary"),
                median("salary_avg_imputed").alias("median_salary"),
                min("salary_avg_imputed").alias("min_salary"),
                max("salary_avg_imputed").alias("max_salary"),
                stddev("salary_avg_imputed").alias("std_salary")
            ).collect()[0]

            return {
                "total_jobs": stats["total_jobs"],
                "avg_salary": float(stats["avg_salary"]) if stats["avg_salary"] else 0,
                "median_salary": float(stats["median_salary"]) if stats["median_salary"] else 0,
                "min_salary": float(stats["min_salary"]) if stats["min_salary"] else 0,
                "max_salary": float(stats["max_salary"]) if stats["max_salary"] else 0,
                "std_salary": float(stats["std_salary"]) if stats["std_salary"] else 0
            }

        except Exception as e:
            raise ProcessingError(f"Failed to generate overall statistics: {str(e)}")

    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query on the dataset."""
        if self.job_data is None:
            raise ProcessingError("No data loaded. Call load_full_dataset() first.")

        try:
            # Register DataFrame as temporary view
            self.job_data.createOrReplaceTempView("job_postings")

            # Execute query
            result = self.spark.sql(query)
            return result.toPandas()

        except Exception as e:
            raise ProcessingError(f"Failed to execute custom query: {str(e)}")

    def get_df(self) -> Optional[DataFrame]:
        """Get the current DataFrame."""
        return self.job_data

    def stop(self):
        """Stop Spark session if owned by this instance."""
        if self._owns_spark:
            stop_spark_session(self.spark)


def create_spark_analyzer(data_path: Optional[str] = None,
                         force_raw: bool = False) -> SparkJobAnalyzer:
    """
    Factory function to create and initialize SparkJobAnalyzer.

    Args:
        data_path: Optional path to data file
        force_raw: Whether to force loading from raw data

    Returns:
        Initialized SparkJobAnalyzer instance
    """
    analyzer = SparkJobAnalyzer()

    try:
        analyzer.load_full_dataset(data_path, force_raw)
        return analyzer
    except Exception as e:
        logger.error(f"Failed to create analyzer: {str(e)}")
        analyzer.stop()
        raise
