"""
Data validation utilities for the job market analytics system.
"""

from typing import Dict, List, Any
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnan, isnull


class DataValidator:
    """Data validation utilities for job market analytics."""

    def validate_dataset(self, df: DataFrame) -> Dict[str, Any]:
        """Validate dataset for analysis readiness."""
        errors = []
        warnings = []

        # Check for empty dataset
        if df.count() == 0:
            errors.append("Dataset is empty")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}

        # Check for required columns
        required_columns = ['SALARY_AVG_IMPUTED', 'TITLE', 'COMPANY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.extend([f"Missing required column: {col}" for col in missing_columns])

        # Check data quality
        total_rows = df.count()
        for col_name in df.columns:
            if col_name in ['SALARY_AVG_IMPUTED']:
                null_count = df.filter(isnull(col(col_name)) | isnan(col(col_name))).count()
                null_percentage = (null_count / total_rows) * 100
                if null_percentage > 50:
                    warnings.append(f"High null percentage in {col_name}: {null_percentage:.1f}%")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def assess_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """Assess data quality and return metrics."""
        total_rows = df.count()
        total_cols = len(df.columns)

        # Basic quality metrics
        quality_metrics = {
            'total_records': total_rows,
            'total_columns': total_cols,
            'completeness_score': 0.0
        }

        # Calculate completeness score
        complete_columns = 0
        for col_name in df.columns:
            null_count = df.filter(isnull(col(col_name)) | isnan(col(col_name))).count()
            completeness = 1 - (null_count / total_rows)
            if completeness > 0.8:
                complete_columns += 1

        quality_metrics['completeness_score'] = complete_columns / total_cols

        return quality_metrics
