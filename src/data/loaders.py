"""
Data loading utilities for the job market analytics system.
"""

from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from pathlib import Path

from ..config.settings import Settings


class DataLoader:
    """Data loading utilities for job market analytics."""

    def __init__(self, spark: SparkSession, settings: Settings):
        self.spark = spark
        self.settings = settings

    def load_raw_data(self, data_path: Optional[str] = None) -> DataFrame:
        """Load raw data from CSV file."""
        if data_path is None:
            data_path = self.settings.raw_data_path

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Raw data not found at {data_path}")

        # Load raw data without schema to preserve all columns
        df = self.spark.read \
            .csv(data_path, multiLine=True, escape="\"", header=True, inferSchema=True)

        return df

    def load_processed_data(self, data_path: Optional[str] = None) -> DataFrame:
        """Load processed data from Parquet file."""
        if data_path is None:
            data_path = self.settings.processed_data_path

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")

        # Load processed Parquet data
        df = self.spark.read.parquet(data_path)

        return df
