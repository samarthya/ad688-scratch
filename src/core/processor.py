"""
Unified Job Market Data Processor

This module provides comprehensive data processing capabilities for job market analysis,
consolidating functionality from multiple processor classes with improved error handling
and configuration management.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, isnan, isnull, trim, upper, lower,
    count, avg, sum as spark_sum, median, desc,
    lit, current_timestamp, regexp_replace, split,
    coalesce, round, year, month, dayofmonth
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, DateType, TimestampType
)
from pyspark.sql.window import Window

from src.config.settings import get_settings
from src.config.schemas import LIGHTCAST_SCHEMA, PROCESSED_SCHEMA
from src.config.mappings import (
    LIGHTCAST_COLUMN_MAPPING, DERIVED_COLUMNS,
    EXPERIENCE_LEVEL_MAPPING, INDUSTRY_STANDARDIZATION,
    SALARY_RANGES_BY_INDUSTRY
)
from src.data.loaders import DataLoader
from src.data.validators import DataValidator
from src.data.transformers import DataTransformer
from src.utils.spark_utils import create_spark_session, stop_spark_session
from src.core.exceptions import DataLoadingError, DataValidationError, ProcessingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JobMarketDataProcessor:
    """
    Unified processor for comprehensive job market data analysis.

    This class consolidates all data processing functionality with improved
    error handling, configuration management, and data validation.
    """

    def __init__(self, app_name: str = "JobMarketAnalysis"):
        """Initialize processor with Spark session and configuration."""
        self.settings = get_settings()
        self.app_name = app_name

        # Initialize Spark session
        self.spark = create_spark_session(app_name)
        self.spark.sparkContext.setLogLevel("WARN")

        # Initialize data access components
        self.data_loader = DataLoader(self.spark)
        self.data_validator = DataValidator()
        self.data_transformer = DataTransformer()

        # Initialize data containers
        self.df_raw = None
        self.df_processed = None

        logger.info(f"JobMarketDataProcessor initialized: {self.spark.version}")

    def load_data(self, file_path: str, use_sample: bool = False,
                  sample_size: int = 50000) -> DataFrame:
        """
        Load job market data from file or create sample data.

        Args:
            file_path: Path to the data file
            use_sample: Whether to create sample data instead
            sample_size: Size of sample data to create

        Returns:
            Spark DataFrame with loaded data
        """
        try:
            if use_sample:
                logger.info(f"Creating sample data with {sample_size:,} records")
                from src.utils.data_utils import create_sample_data
                self.df_raw = create_sample_data(self.spark, sample_size)
            else:
                logger.info(f"Loading data from {file_path}")
                if file_path.endswith('.parquet'):
                    self.df_raw = self.spark.read.parquet(file_path)
                else:
                    self.df_raw = self.spark.read \
                        .option("header", "true") \
                        .option("inferSchema", "true") \
                        .csv(file_path)

            # Validate loaded data
            validation_results = self.data_validator.validate_dataset(self.df_raw)
            if not validation_results["is_valid"]:
                logger.warning(f"Data validation issues: {validation_results['warnings']}")

            logger.info(f"Data loaded successfully: {self.df_raw.count():,} records")
            return self.df_raw

        except Exception as e:
            raise DataLoadingError(f"Failed to load data: {str(e)}")

    def assess_data_quality(self, df: DataFrame) -> Dict[str, any]:
        """Assess data quality and return comprehensive report."""
        try:
            validation_results = self.data_validator.validate_dataset(df)

            # Add additional quality metrics
            quality_report = {
                "validation_results": validation_results,
                "total_records": df.count(),
                "total_columns": len(df.columns),
                "column_names": df.columns,
                "data_types": dict(df.dtypes),
                "timestamp": datetime.now().isoformat()
            }

            # Add null value analysis
            null_analysis = {}
            for col_name in df.columns:
                null_count = df.filter(col(col_name).isNull()).count()
                total_count = df.count()
                null_analysis[col_name] = {
                    "null_count": null_count,
                    "null_percentage": (null_count / total_count) * 100
                }

            quality_report["null_analysis"] = null_analysis

            return quality_report

        except Exception as e:
            raise ProcessingError(f"Failed to assess data quality: {str(e)}")

    def clean_and_standardize_data(self, df: DataFrame) -> DataFrame:
        """Clean and standardize the dataset."""
        try:
            logger.info("Starting data cleaning and standardization...")

            # Standardize column names
            df_cleaned = self.data_transformer.standardize_columns(df)

            # Clean text data
            df_cleaned = self.data_transformer.clean_text_data(df_cleaned)

            # Remove duplicates
            df_cleaned = self.data_transformer.remove_duplicates(df_cleaned)

            # Validate salary ranges
            df_cleaned = self.data_transformer.validate_salary_ranges(df_cleaned)

            logger.info("Data cleaning completed successfully")
            return df_cleaned

        except Exception as e:
            raise ProcessingError(f"Failed to clean and standardize data: {str(e)}")

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Engineer derived features for analysis."""
        try:
            logger.info("Starting feature engineering...")

            # Apply comprehensive feature engineering
            df_enhanced = self.data_transformer.engineer_features(df)

            logger.info("Feature engineering completed successfully")
            return df_enhanced

        except Exception as e:
            raise ProcessingError(f"Failed to engineer features: {str(e)}")

    def clean_and_process_data_optimized(self, df: DataFrame) -> DataFrame:
        """Optimized data processing pipeline."""
        try:
            logger.info("Starting optimized data processing pipeline...")

            # Step 1: Clean and standardize
            df_processed = self.clean_and_standardize_data(df)

            # Step 2: Engineer features
            df_processed = self.engineer_features(df_processed)

            # Step 3: Final validation
            validation_results = self.data_validator.validate_dataset(df_processed)
            if not validation_results["is_valid"]:
                logger.warning(f"Final validation issues: {validation_results['warnings']}")

            self.df_processed = df_processed
            logger.info("Optimized processing pipeline completed successfully")
            return df_processed

        except Exception as e:
            raise ProcessingError(f"Failed in optimized processing pipeline: {str(e)}")

    def save_processed_data(self, df: DataFrame, output_path: str = None) -> None:
        """Save processed data in multiple formats."""
        try:
            if output_path is None:
                output_path = self.settings.processed_data_path

            logger.info(f"Saving processed data to {output_path}")

            # Save as Parquet (primary format)
            parquet_path = output_path if output_path.endswith('.parquet') else f"{output_path}.parquet"
            self.data_loader.save_data(df, parquet_path, "parquet")

            # Save as CSV (compatibility format)
            csv_path = self.settings.clean_data_path
            self.data_loader.save_data(df, csv_path, "csv")

            # Generate processing report
            self._generate_processing_report(df, output_path)

            logger.info("Processed data saved successfully")

        except Exception as e:
            raise ProcessingError(f"Failed to save processed data: {str(e)}")

    def generate_summary_statistics(self, df: DataFrame) -> Dict[str, any]:
        """Generate comprehensive summary statistics."""
        try:
            logger.info("Generating summary statistics...")

            # Basic statistics
            total_jobs = df.count()

            # Industry analysis
            top_industries = df.groupBy("industry") \
                .count() \
                .orderBy(desc("count")) \
                .limit(10) \
                .collect()

            # AI/ML analysis
            ai_stats = df.groupBy("ai_related") \
                .agg(count("*").alias("count")) \
                .collect()

            # Experience level analysis
            exp_stats = df.groupBy("experience_level") \
                .agg(count("*").alias("count")) \
                .collect()

            # Salary analysis
            salary_stats = df.select(
                avg("salary_avg_imputed").alias("avg_salary"),
                median("salary_avg_imputed").alias("median_salary"),
                min("salary_avg_imputed").alias("min_salary"),
                max("salary_avg_imputed").alias("max_salary")
            ).collect()[0]

            summary = {
                "total_jobs": total_jobs,
                "top_industries": [(row["industry"], row["count"]) for row in top_industries],
                "ai_ml_stats": {row["ai_related"]: {"count": row["count"]} for row in ai_stats},
                "experience_stats": {row["experience_level"]: {"count": row["count"]} for row in exp_stats},
                "salary_stats": {
                    "avg_salary": float(salary_stats["avg_salary"]) if salary_stats["avg_salary"] else 0,
                    "median_salary": float(salary_stats["median_salary"]) if salary_stats["median_salary"] else 0,
                    "min_salary": float(salary_stats["min_salary"]) if salary_stats["min_salary"] else 0,
                    "max_salary": float(salary_stats["max_salary"]) if salary_stats["max_salary"] else 0
                },
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Summary statistics generated successfully")
            return summary

        except Exception as e:
            raise ProcessingError(f"Failed to generate summary statistics: {str(e)}")

    def _generate_processing_report(self, df: DataFrame, output_path: str) -> None:
        """Generate processing report with quality metrics."""
        try:
            report_path = Path(output_path).parent / "processing_report.md"

            # Generate quality assessment
            quality_report = self.assess_data_quality(df)
            summary_stats = self.generate_summary_statistics(df)

            # Create markdown report
            report_content = f"""# Data Processing Report

## Processing Summary
- **Total Records**: {summary_stats['total_jobs']:,}
- **Processing Date**: {summary_stats['timestamp']}
- **Data Quality Score**: {quality_report['validation_results']['quality_metrics'].get('valid_salary_percentage', 0):.1f}%

## Top Industries
"""

            for industry, count in summary_stats['top_industries'][:5]:
                report_content += f"- **{industry}**: {count:,} jobs\n"

            report_content += f"""
## AI/ML Analysis
- **AI/ML Roles**: {summary_stats['ai_ml_stats'].get(True, {}).get('count', 0):,}
- **Traditional Roles**: {summary_stats['ai_ml_stats'].get(False, {}).get('count', 0):,}

## Salary Statistics
- **Average Salary**: ${summary_stats['salary_stats']['avg_salary']:,.0f}
- **Median Salary**: ${summary_stats['salary_stats']['median_salary']:,.0f}
- **Salary Range**: ${summary_stats['salary_stats']['min_salary']:,.0f} - ${summary_stats['salary_stats']['max_salary']:,.0f}

## Data Quality Metrics
- **Valid Salary Data**: {quality_report['validation_results']['quality_metrics'].get('valid_salary_percentage', 0):.1f}%
- **Null Salary Data**: {quality_report['validation_results']['quality_metrics'].get('null_salary_percentage', 0):.1f}%
"""

            with open(report_path, 'w') as f:
                f.write(report_content)

            logger.info(f"Processing report saved: {report_path}")

        except Exception as e:
            logger.warning(f"Failed to generate processing report: {str(e)}")

    def stop_spark(self):
        """Stop Spark session."""
        stop_spark_session(self.spark)
