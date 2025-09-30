"""
Data utilities for job market analysis.

Provides common data manipulation and validation utilities
for the job market analytics system.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from src.config.settings import get_settings
from src.core.exceptions import DataValidationError


def create_sample_data(spark: SparkSession, sample_size: int = 1000) -> DataFrame:
    """Create sample data for testing and development."""
    settings = get_settings()

    # Create sample job data
    sample_data = []
    industries = ["Technology", "Finance", "Healthcare", "Education", "Retail"]
    experience_levels = ["Entry", "Mid", "Senior", "Executive"]
    employment_types = ["Full-time", "Part-time", "Contract", "Temporary"]
    remote_types = ["Remote", "Hybrid", "On-site"]

    for i in range(sample_size):
        sample_data.append({
            "job_id": f"JOB_{i:06d}",
            "title": f"Sample Job {i}",
            "company": f"Company {i % 50}",
            "location": f"City {i % 20}, State {i % 10}",
            "industry": industries[i % len(industries)],
            "experience_level": experience_levels[i % len(experience_levels)],
            "employment_type": employment_types[i % len(employment_types)],
            "remote_type": remote_types[i % len(remote_types)],
            "salary_avg_imputed": 50000 + (i % 100000),
            "ai_related": i % 3 == 0,
            "remote_allowed": i % 2 == 0
        })

    # Convert to Spark DataFrame
    df = spark.createDataFrame(sample_data)
    return df


def validate_data_paths() -> Dict[str, bool]:
    """Validate that required data paths exist."""
    settings = get_settings()

    return {
        "raw_data_exists": settings.raw_data_exists,
        "processed_data_exists": settings.processed_data_exists,
        "clean_data_exists": settings.clean_data_exists,
        "figures_dir_exists": Path(settings.figures_path).exists()
    }


def ensure_directory_structure() -> None:
    """Ensure required directory structure exists."""
    settings = get_settings()

    directories = [
        "data/raw",
        "data/processed",
        "figures",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_data_summary(df: DataFrame) -> Dict[str, Any]:
    """Get summary statistics for a DataFrame."""
    try:
        summary = {
            "total_records": df.count(),
            "columns": len(df.columns),
            "column_names": df.columns,
            "has_salary_data": "salary_avg_imputed" in df.columns,
            "has_industry_data": "industry" in df.columns,
            "has_experience_data": "experience_level" in df.columns
        }

        # Add null counts for key columns
        if "salary_avg_imputed" in df.columns:
            null_salary_count = df.filter(df.salary_avg_imputed.isNull()).count()
            summary["null_salary_count"] = null_salary_count
            summary["null_salary_percentage"] = (null_salary_count / summary["total_records"]) * 100

        return summary

    except Exception as e:
        raise DataValidationError(f"Failed to generate data summary: {str(e)}")
