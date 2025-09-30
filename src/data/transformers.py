"""
Data transformation utilities for job market analysis.

Provides data cleaning, feature engineering, and transformation
capabilities for the job market analytics system.
"""

from typing import Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, isnan, isnull, trim, upper, lower,
    count, avg, median, lit, current_timestamp,
    regexp_replace, split, coalesce, round
)
from pyspark.sql.window import Window

from src.config.mappings import (
    LIGHTCAST_COLUMN_MAPPING, DERIVED_COLUMNS,
    EXPERIENCE_LEVEL_MAPPING, INDUSTRY_STANDARDIZATION
)
from src.core.exceptions import ProcessingError


class DataTransformer:
    """Data transformation and feature engineering utilities."""

    def __init__(self):
        """Initialize data transformer."""
        self.column_mapping = LIGHTCAST_COLUMN_MAPPING
        self.derived_columns = DERIVED_COLUMNS

    def standardize_columns(self, df: DataFrame) -> DataFrame:
        """Standardize column names using mapping."""
        try:
            # Rename columns according to mapping
            for old_name, new_name in self.column_mapping.items():
                if old_name in df.columns:
                    df = df.withColumnRenamed(old_name, new_name)

            return df
        except Exception as e:
            raise ProcessingError(f"Failed to standardize columns: {str(e)}")

    def clean_text_data(self, df: DataFrame) -> DataFrame:
        """Clean and standardize text data."""
        try:
            # Clean text columns
            text_columns = ["title", "company", "location", "industry", "required_skills"]

            for col_name in text_columns:
                if col_name in df.columns:
                    df = df.withColumn(
                        col_name,
                        when(col(col_name).isNull(), lit("Undefined"))
                        .otherwise(trim(col(col_name)))
                    )

            return df
        except Exception as e:
            raise ProcessingError(f"Failed to clean text data: {str(e)}")

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Engineer derived features."""
        try:
            # Add timestamp columns
            df = df.withColumn("created_at", current_timestamp())
            df = df.withColumn("updated_at", current_timestamp())

            # Calculate salary average with imputation
            df = self._calculate_salary_average(df)

            # Create experience level categories
            df = self._create_experience_levels(df)

            # Detect AI/ML roles
            df = self._detect_ai_roles(df)

            # Standardize remote work flag
            df = self._standardize_remote_work(df)

            # Clean industry names
            df = self._standardize_industries(df)

            return df
        except Exception as e:
            raise ProcessingError(f"Failed to engineer features: {str(e)}")

    def _calculate_salary_average(self, df: DataFrame) -> DataFrame:
        """Calculate salary average with smart imputation."""
        # Use existing salary calculation logic from enhanced_processor.py
        # This is a simplified version - full implementation would be more complex

        df = df.withColumn(
            "salary_avg_imputed",
            when(col("salary_single").isNotNull(), col("salary_single"))
            .when(
                col("salary_min").isNotNull() & col("salary_max").isNotNull(),
                (col("salary_min") + col("salary_max")) / 2
            )
            .when(col("salary_min").isNotNull(), col("salary_min") * 1.125)
            .when(col("salary_max").isNotNull(), col("salary_max") * 0.889)
            .otherwise(lit(75000))  # Default median
        )

        return df

    def _create_experience_levels(self, df: DataFrame) -> DataFrame:
        """Create standardized experience level categories."""
        df = df.withColumn(
            "experience_years",
            when(col("experience_min").isNotNull(), col("experience_min"))
            .when(col("experience_max").isNotNull(), col("experience_max"))
            .otherwise(lit(0))
        )

        df = df.withColumn(
            "experience_level",
            when(col("experience_years") < 2, "Entry")
            .when(col("experience_years") < 5, "Mid")
            .when(col("experience_years") < 10, "Senior")
            .otherwise("Executive")
        )

        return df

    def _detect_ai_roles(self, df: DataFrame) -> DataFrame:
        """Detect AI/ML related roles based on title patterns."""
        ai_patterns = [
            "ai", "artificial intelligence", "machine learning", "ml engineer",
            "data scientist", "data science", "deep learning", "neural network"
        ]

        ai_condition = lit(False)
        for pattern in ai_patterns:
            ai_condition = ai_condition | col("title").rlike(f"(?i).*{pattern}.*")

        df = df.withColumn("ai_related", ai_condition.cast("boolean"))

        return df

    def _standardize_remote_work(self, df: DataFrame) -> DataFrame:
        """Standardize remote work flags."""
        if "remote_type" in df.columns:
            df = df.withColumn(
                "remote_allowed",
                when(
                    col("remote_type").rlike("(?i)(remote|anywhere|wfh|work from home)"),
                    lit(True)
                ).otherwise(lit(False))
            )
        else:
            df = df.withColumn("remote_allowed", lit(False))

        return df

    def _standardize_industries(self, df: DataFrame) -> DataFrame:
        """Standardize industry names."""
        if "industry" not in df.columns:
            return df

        # Apply industry standardization
        industry_condition = col("industry")
        for standard_name, variations in INDUSTRY_STANDARDIZATION.items():
            for variation in variations:
                industry_condition = when(
                    col("industry").rlike(f"(?i).*{variation}.*"),
                    lit(standard_name.title())
                ).otherwise(industry_condition)

        df = df.withColumn("industry_clean", industry_condition)

        return df

    def remove_duplicates(self, df: DataFrame, key_columns: List[str] = None) -> DataFrame:
        """Remove duplicate records."""
        if key_columns is None:
            key_columns = ["title", "company", "location"]

        # Only use columns that exist in the DataFrame
        existing_columns = [col for col in key_columns if col in df.columns]

        if existing_columns:
            return df.dropDuplicates(existing_columns)
        else:
            return df

    def validate_salary_ranges(self, df: DataFrame) -> DataFrame:
        """Validate and clean salary ranges."""
        if "salary_avg_imputed" not in df.columns:
            return df

        # Remove unrealistic salary values
        df = df.filter(
            (col("salary_avg_imputed") >= 20000) &
            (col("salary_avg_imputed") <= 500000)
        )

        return df
