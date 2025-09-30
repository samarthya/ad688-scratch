"""
Data validation utilities for job market analysis.

Provides comprehensive data validation and quality assessment
for the job market analytics system.
"""

from typing import Dict, List, Any

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnan, isnull, count, when

from src.core.exceptions import DataValidationError
from src.config.mappings import ANALYSIS_COLUMNS, SALARY_RANGES_BY_INDUSTRY


class DataValidator:
    """Comprehensive data validation and quality assessment."""

    def __init__(self):
        """Initialize data validator."""
        self.required_columns = list(ANALYSIS_COLUMNS.values())

    def validate_dataset(self, df: DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset validation."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_metrics": {}
        }

        # Check if dataset is empty
        record_count = df.count()
        if record_count == 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Dataset is empty")
            return validation_results

        validation_results["quality_metrics"]["total_records"] = record_count

        # Check required columns
        missing_columns = self._check_required_columns(df)
        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["errors"].extend([f"Missing required column: {col}" for col in missing_columns])

        # Validate salary data
        salary_validation = self._validate_salary_data(df)
        validation_results["quality_metrics"].update(salary_validation)

        if salary_validation.get("invalid_salary_count", 0) > record_count * 0.5:
            validation_results["warnings"].append("High percentage of invalid salary data")

        # Validate categorical data
        categorical_validation = self._validate_categorical_data(df)
        validation_results["quality_metrics"].update(categorical_validation)

        return validation_results

    def _check_required_columns(self, df: DataFrame) -> List[str]:
        """Check for missing required columns."""
        missing_columns = []
        for col_name in self.required_columns:
            if col_name not in df.columns:
                missing_columns.append(col_name)
        return missing_columns

    def _validate_salary_data(self, df: DataFrame) -> Dict[str, Any]:
        """Validate salary data quality."""
        if "salary_avg_imputed" not in df.columns:
            return {"salary_validation": "No salary column found"}

        # Count null salaries
        null_salary_count = df.filter(col("salary_avg_imputed").isNull()).count()
        total_count = df.count()

        # Count invalid salary ranges
        invalid_salary_count = df.filter(
            (col("salary_avg_imputed") < 20000) |
            (col("salary_avg_imputed") > 500000)
        ).count()

        return {
            "null_salary_count": null_salary_count,
            "null_salary_percentage": (null_salary_count / total_count) * 100,
            "invalid_salary_count": invalid_salary_count,
            "invalid_salary_percentage": (invalid_salary_count / total_count) * 100,
            "valid_salary_percentage": ((total_count - null_salary_count - invalid_salary_count) / total_count) * 100
        }

    def _validate_categorical_data(self, df: DataFrame) -> Dict[str, Any]:
        """Validate categorical data quality."""
        validation_results = {}

        categorical_columns = ["industry", "experience_level", "employment_type", "remote_type"]

        for col_name in categorical_columns:
            if col_name in df.columns:
                # Count null values
                null_count = df.filter(col(col_name).isNull()).count()
                total_count = df.count()

                validation_results[f"{col_name}_null_count"] = null_count
                validation_results[f"{col_name}_null_percentage"] = (null_count / total_count) * 100

                # Count unique values
                unique_count = df.select(col_name).distinct().count()
                validation_results[f"{col_name}_unique_values"] = unique_count

        return validation_results

    def validate_salary_ranges(self, df: DataFrame) -> Dict[str, Any]:
        """Validate salary ranges by industry."""
        if "industry" not in df.columns or "salary_avg_imputed" not in df.columns:
            return {"error": "Required columns not found"}

        validation_results = {}

        for industry, (min_salary, max_salary) in SALARY_RANGES_BY_INDUSTRY.items():
            if industry == "default":
                continue

            industry_df = df.filter(col("industry") == industry)
            if industry_df.count() == 0:
                continue

            # Check for outliers
            outliers = industry_df.filter(
                (col("salary_avg_imputed") < min_salary) |
                (col("salary_avg_imputed") > max_salary)
            ).count()

            total_industry = industry_df.count()
            validation_results[industry] = {
                "outlier_count": outliers,
                "outlier_percentage": (outliers / total_industry) * 100,
                "expected_range": (min_salary, max_salary)
            }

        return validation_results
