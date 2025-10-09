"""
Core data processor for job market analytics.

This module provides comprehensive data processing capabilities
for cleaning, transforming, and preparing job market data.
"""

from typing import Dict, List, Optional, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, trim, upper, lower, regexp_replace
from pathlib import Path
import logging

from ..utils.spark_utils import create_spark_session
from ..config.settings import Settings
from ..data.loaders import DataLoader
from ..data.validators import DataValidator
from ..data.transformers import DataTransformer

logger = logging.getLogger(__name__)


class JobMarketDataProcessor:
    """
    Core data processor for job market analytics.

    Provides comprehensive data cleaning, transformation, and preparation
    capabilities for job market analysis.
    """

    def __init__(self, spark: Optional[SparkSession] = None, settings: Optional[Settings] = None):
        """Initialize the processor with Spark session and settings."""
        self.spark = spark or create_spark_session("Job Market Processor")
        self.settings = settings or Settings()
        self.data_loader = DataLoader(self.spark, self.settings)
        self.data_validator = DataValidator()
        self.data_transformer = DataTransformer()

    def load_and_process_data(self, data_path: Optional[str] = None) -> DataFrame:
        """Load and process data with automatic method selection."""

        if data_path is None:
            data_path = self.settings.raw_data_path

        # Check file extension to determine processing method
        if data_path.endswith('.parquet'):
            logger.info("Loading processed Parquet data...")
            df = self.data_loader.load_processed_data(data_path)
            logger.info("Processed data loaded successfully")
            return df
        elif data_path.endswith('.csv'):
            logger.info("Processing raw CSV data...")
            return self.process_raw_data(data_path)
        else:
            # Try processed data first, then fall back to raw processing
            try:
                logger.info("Attempting to load processed data...")
                df = self.data_loader.load_processed_data(data_path)
                logger.info("Processed data loaded successfully")
                return df
            except FileNotFoundError:
                logger.info("No processed data found, processing raw data...")
                return self.process_raw_data(data_path)

    def process_raw_data(self, data_path: Optional[str] = None) -> DataFrame:
        """Process raw data through the complete pipeline."""

        # Load raw data
        logger.info("Loading raw data...")
        df = self.data_loader.load_raw_data(data_path)

        # Clean and standardize
        logger.info("Cleaning and standardizing data...")
        df_clean = self.data_transformer.clean_and_standardize(df)

        # Engineer features
        logger.info("Engineering features...")
        df_enhanced = self.data_transformer.engineer_features(df_clean)

        # Validate processed data
        logger.info("Validating processed data...")
        validation_results = self.data_validator.validate_dataset(df_enhanced)
        if not validation_results['is_valid']:
            logger.warning(f"Data validation warnings: {validation_results['warnings']}")

        # Save processed data
        logger.info("Saving processed data...")
        self.save_processed_data(df_enhanced)

        return df_enhanced

    def clean_and_standardize_data(self, df: DataFrame) -> DataFrame:
        """Clean and standardize the dataset."""
        return self.data_transformer.clean_and_standardize(df)

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Engineer features for analysis."""
        return self.data_transformer.engineer_features(df)

    def assess_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """Assess data quality and return metrics."""
        return self.data_validator.assess_data_quality(df)

    def save_processed_data(self, df: DataFrame, output_path: Optional[str] = None) -> None:
        """Save processed data in multiple formats."""

        if output_path is None:
            output_path = self.settings.processed_data_path

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as Parquet (primary format)
        parquet_path = output_path.replace('.csv', '.parquet')
        df.write.mode("overwrite").option("compression", "snappy").parquet(parquet_path)
        logger.info(f"Processed data saved as Parquet: {parquet_path}")

        # Save as CSV (compatibility)
        if output_path.endswith('.csv'):
            df.write.mode("overwrite").option("header", "true").csv(output_path)
            logger.info(f"Processed data saved as CSV: {output_path}")

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the processing pipeline."""
        return {
            'processor_type': 'JobMarketDataProcessor',
            'spark_version': self.spark.version,
            'settings': {
                'raw_data_path': self.settings.raw_data_path,
                'processed_data_path': self.settings.processed_data_path
            }
        }
