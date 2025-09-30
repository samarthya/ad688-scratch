"""
Centralized settings management for job market analysis.

Provides a single source of truth for all configuration settings
including Spark configuration, file paths, and processing parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import os


@dataclass
class Settings:
    """Centralized settings for job market analysis."""

    # Data paths (will be converted to absolute paths in __post_init__)
    raw_data_path: str = "data/raw/lightcast_job_postings.csv"
    processed_data_path: str = "data/processed/job_market_processed.parquet"
    clean_data_path: str = "data/processed/clean_job_data.csv"
    figures_path: str = "figures"

    # Spark configuration
    spark_app_name: str = "JobMarketAnalysis"
    spark_master: str = "local[*]"
    spark_config: Dict[str, str] = None

    # Processing parameters
    sample_size: Optional[int] = None
    max_records: Optional[int] = None

    # Visualization settings
    chart_width: int = 800
    chart_height: int = 600
    chart_theme: str = "plotly_white"

    def __post_init__(self):
        """Initialize default Spark configuration and convert paths to absolute."""
        if self.spark_config is None:
            self.spark_config = {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.sql.execution.arrow.pyspark.enabled": "true",
                "spark.sql.debug.maxToStringFields": "1000"
            }

        # Convert relative paths to absolute paths
        # Find the project root by looking for the src directory
        current_dir = Path.cwd()
        project_root = current_dir

        # Walk up the directory tree to find the project root
        while project_root != project_root.parent:
            if (project_root / "src").exists():
                break
            project_root = project_root.parent

        # Convert relative paths to absolute paths
        self.raw_data_path = str(project_root / self.raw_data_path)
        self.processed_data_path = str(project_root / self.processed_data_path)
        self.clean_data_path = str(project_root / self.clean_data_path)
        self.figures_path = str(project_root / self.figures_path)

    @property
    def raw_data_exists(self) -> bool:
        """Check if raw data file exists."""
        return Path(self.raw_data_path).exists()

    @property
    def processed_data_exists(self) -> bool:
        """Check if processed data exists."""
        return Path(self.processed_data_path).exists()

    @property
    def clean_data_exists(self) -> bool:
        """Check if clean data exists."""
        return Path(self.clean_data_path).exists()


def get_settings() -> Settings:
    """Get application settings with environment variable overrides."""

    settings = Settings()

    # Override with environment variables if present
    if os.getenv("JOB_MARKET_RAW_DATA_PATH"):
        settings.raw_data_path = os.getenv("JOB_MARKET_RAW_DATA_PATH")

    if os.getenv("JOB_MARKET_PROCESSED_DATA_PATH"):
        settings.processed_data_path = os.getenv("JOB_MARKET_PROCESSED_DATA_PATH")

    if os.getenv("JOB_MARKET_SAMPLE_SIZE"):
        settings.sample_size = int(os.getenv("JOB_MARKET_SAMPLE_SIZE"))

    return settings
