"""
Settings and configuration management for the job market analytics system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class Settings:
    """Configuration settings for the job market analytics system."""

    # Data paths
    raw_data_path: str = "data/raw/lightcast_job_postings.csv"
    processed_data_path: str = "data/processed/job_market_processed.parquet"
    clean_data_path: str = "data/processed/clean_job_data.csv"
    figures_path: str = "figures/"

    # Spark configuration
    spark_master: str = "local[*]"
    spark_config: Dict[str, str] = None

    def __post_init__(self):
        """Initialize settings with default values."""
        if self.spark_config is None:
            self.spark_config = {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.sql.adaptive.skewJoin.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.sql.execution.arrow.pyspark.enabled": "true",
                "spark.driver.memory": "8g",
                "spark.executor.memory": "8g",
                "spark.executor.cores": "4",
                "spark.cores.max": "8",
                "spark.driver.maxResultSize": "4g",
                "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB"
            }

        # Convert relative paths to absolute paths
        current_dir = Path.cwd()
        project_root = current_dir
        while project_root != project_root.parent:
            if (project_root / "src").exists():
                break
            project_root = project_root.parent

        self.raw_data_path = str(project_root / self.raw_data_path)
        self.processed_data_path = str(project_root / self.processed_data_path)
        self.clean_data_path = str(project_root / self.clean_data_path)
        self.figures_path = str(project_root / self.figures_path)
