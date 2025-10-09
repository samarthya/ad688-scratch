"""
Spark utilities for the job market analytics system.

This module provides utilities for creating and managing Spark sessions
with appropriate configuration for the job market analysis workload.
"""

from typing import Dict, Any, Optional
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import os


def create_spark_session(app_name: str = "Job Market Analytics",
                        master: str = "local[*]",
                        config: Optional[Dict[str, str]] = None) -> SparkSession:
    """
    Create a Spark session with appropriate configuration for job market analysis.

    Args:
        app_name: Name of the Spark application
        master: Spark master URL (default: local[*])
        config: Additional Spark configuration parameters

    Returns:
        Configured SparkSession
    """

    # Default configuration for job market analysis
    default_config = {
        "spark.app.name": app_name,
        "spark.master": master,
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
        "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "5",
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "256MB"
    }

    # Merge with provided config
    if config:
        default_config.update(config)

    # Create SparkConf
    conf = SparkConf()
    for key, value in default_config.items():
        conf.set(key, value)

    # Create SparkSession
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")

    return spark


def get_spark_config() -> Dict[str, str]:
    """
    Get recommended Spark configuration for job market analysis.

    Returns:
        Dictionary of Spark configuration parameters
    """
    return {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
        "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "5",
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "256MB"
    }


def optimize_spark_for_ml(spark: SparkSession) -> SparkSession:
    """
    Optimize Spark session for machine learning workloads.

    Args:
        spark: Existing SparkSession

    Returns:
        Optimized SparkSession
    """

    # Set ML-specific configurations
    ml_config = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB",
        "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "3",
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "128MB",
        "spark.sql.adaptive.localShuffleReader.enabled": "true",
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "128MB"
    }

    # Apply configurations
    for key, value in ml_config.items():
        spark.conf.set(key, value)

    return spark


def get_memory_config() -> Dict[str, str]:
    """
    Get memory configuration recommendations for Spark.

    Returns:
        Dictionary of memory-related Spark configuration parameters
    """
    return {
        "spark.driver.memory": "4g",
        "spark.driver.maxResultSize": "2g",
        "spark.executor.memory": "4g",
        "spark.executor.memoryFraction": "0.8",
        "spark.storage.memoryFraction": "0.2"
    }


def create_ml_spark_session(app_name: str = "ML Job Market Analytics") -> SparkSession:
    """
    Create a Spark session optimized for machine learning workloads.

    Args:
        app_name: Name of the Spark application

    Returns:
        ML-optimized SparkSession
    """

    # Get base configuration
    config = get_spark_config()

    # Add ML-specific configurations
    ml_config = {
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB",
        "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "3",
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "128MB",
        "spark.sql.adaptive.localShuffleReader.enabled": "true"
    }

    config.update(ml_config)

    # Create session
    spark = create_spark_session(app_name, config=config)

    # Apply additional optimizations
    spark = optimize_spark_for_ml(spark)

    return spark


def stop_spark_session(spark: SparkSession) -> None:
    """
    Safely stop a Spark session.

    Args:
        spark: SparkSession to stop
    """
    if spark:
        spark.stop()
        print("Spark session stopped successfully")


def get_spark_info(spark: SparkSession) -> Dict[str, Any]:
    """
    Get information about the current Spark session.

    Args:
        spark: SparkSession

    Returns:
        Dictionary containing Spark session information
    """
    return {
        "app_name": spark.sparkContext.appName,
        "master": spark.sparkContext.master,
        "version": spark.version,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "total_cores": spark.sparkContext._conf.get("spark.executor.cores", "unknown")
    }
