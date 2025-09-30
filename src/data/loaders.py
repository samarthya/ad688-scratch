"""
Data loading utilities for job market analysis.

Provides centralized data loading functionality with proper error handling
and fallback mechanisms.
"""

from pathlib import Path
from typing import Optional, Union

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

from src.config.settings import get_settings
from src.config.schemas import LIGHTCAST_SCHEMA, PROCESSED_SCHEMA
from src.core.exceptions import DataLoadingError, DataValidationError


class DataLoader:
    """Centralized data loading with fallback mechanisms."""

    def __init__(self, spark_session: Optional[SparkSession] = None):
        """Initialize data loader."""
        self.spark = spark_session
        self.settings = get_settings()

    def load_raw_data(self, schema: Optional[StructType] = None) -> DataFrame:
        """Load raw Lightcast data with validation."""
        if not self.settings.raw_data_exists:
            raise DataLoadingError(f"Raw data not found at {self.settings.raw_data_path}")

        try:
            # For raw data, don't enforce schema by default to preserve all columns
            if schema is not None:
                df = self.spark.read \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .schema(schema) \
                    .csv(self.settings.raw_data_path)
            else:
                # Let Spark infer all columns from the raw data
                df = self.spark.read \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .csv(self.settings.raw_data_path)

            # Validate loaded data
            if df.count() == 0:
                raise DataValidationError("Loaded dataset is empty")

            return df

        except Exception as e:
            raise DataLoadingError(f"Failed to load raw data: {str(e)}")

    def load_processed_data(self) -> DataFrame:
        """Load processed Parquet data (preferred)."""
        if not self.settings.processed_data_exists:
            raise DataLoadingError(f"Processed data not found at {self.settings.processed_data_path}")

        try:
            df = self.spark.read.parquet(self.settings.processed_data_path)

            # Validate processed data
            if df.count() == 0:
                raise DataValidationError("Processed dataset is empty")

            return df

        except Exception as e:
            raise DataLoadingError(f"Failed to load processed data: {str(e)}")

    def load_clean_data(self) -> DataFrame:
        """Load clean CSV data (fallback)."""
        if not self.settings.clean_data_exists:
            raise DataLoadingError(f"Clean data not found at {self.settings.clean_data_path}")

        try:
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(self.settings.clean_data_path)

            # Validate clean data
            if df.count() == 0:
                raise DataValidationError("Clean dataset is empty")

            return df

        except Exception as e:
            raise DataLoadingError(f"Failed to load clean data: {str(e)}")

    def load_data_with_fallback(self) -> DataFrame:
        """Load data with automatic fallback: Parquet → CSV → Raw."""
        try:
            # Try processed Parquet first (fastest)
            return self.load_processed_data()
        except DataLoadingError:
            try:
                # Try clean CSV second
                return self.load_clean_data()
            except DataLoadingError:
                # Fall back to raw data
                return self.load_raw_data()

    def save_data(self, df: DataFrame, path: str, format: str = "parquet") -> None:
        """Save DataFrame in specified format."""
        try:
            if format.lower() == "parquet":
                df.write.mode("overwrite").option("compression", "snappy").parquet(path)
            elif format.lower() == "csv":
                df.write.mode("overwrite").option("header", "true").csv(path)
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            raise DataLoadingError(f"Failed to save data: {str(e)}")
