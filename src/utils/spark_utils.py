"""
Spark utilities for job market analysis.

Provides centralized Spark session management and configuration
for the job market analytics system.
"""

from typing import Dict, Optional

from pyspark.sql import SparkSession

from src.config.settings import get_settings
from src.core.exceptions import ConfigurationError


def create_spark_session(app_name: Optional[str] = None) -> SparkSession:
    """Create and configure Spark session."""
    settings = get_settings()

    if app_name is None:
        app_name = settings.spark_app_name

    try:
        builder = SparkSession.builder \
            .appName(app_name) \
            .master(settings.spark_master)

        # Apply individual config settings
        for key, value in settings.spark_config.items():
            builder = builder.config(key, value)

        spark = builder.getOrCreate()
        return spark
    except Exception as e:
        raise ConfigurationError(f"Failed to create Spark session: {str(e)}")


def get_spark_config() -> Dict[str, str]:
    """Get Spark configuration settings."""
    settings = get_settings()
    return settings.spark_config


def stop_spark_session(spark: SparkSession) -> None:
    """Stop Spark session safely."""
    try:
        if spark is not None:
            spark.stop()
    except Exception as e:
        print(f"Warning: Error stopping Spark session: {e}")
