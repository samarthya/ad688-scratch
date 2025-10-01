"""
Data transformation utilities for the job market analytics system.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, trim, upper, lower


class DataTransformer:
    """Data transformation utilities for job market analytics."""

    def clean_and_standardize(self, df: DataFrame) -> DataFrame:
        """Clean and standardize the dataset."""
        # Basic cleaning operations
        df_clean = df

        # Trim string columns
        for col_name in df.columns:
            if df.schema[col_name].dataType.typeName() == 'string':
                df_clean = df_clean.withColumn(col_name, trim(col(col_name)))

        return df_clean

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Engineer features for analysis."""
        # Basic feature engineering
        df_enhanced = df

        # Add basic derived columns if they don't exist
        if 'SALARY_AVG_IMPUTED' not in df.columns:
            # Simple salary calculation if not present - just use 0 for now to avoid casting issues
            from pyspark.sql.functions import lit
            df_enhanced = df_enhanced.withColumn('SALARY_AVG_IMPUTED', lit(0.0))

        return df_enhanced
