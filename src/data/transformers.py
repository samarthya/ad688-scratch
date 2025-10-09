"""
Data transformation utilities for the job market analytics system.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, trim, upper, lower
import re


class DataTransformer:
    """Data transformation utilities for job market analytics."""

    def clean_and_standardize(self, df: DataFrame) -> DataFrame:
        """Clean and standardize the dataset."""
        # Step 1: Standardize column names to snake_case
        df_clean = self._standardize_column_names(df)

        # Step 2: Trim string columns
        for col_name in df_clean.columns:
            if df_clean.schema[col_name].dataType.typeName() == 'string':
                df_clean = df_clean.withColumn(col_name, trim(col(col_name)))

        return df_clean

    def _standardize_column_names(self, df: DataFrame) -> DataFrame:
        """Convert all column names to snake_case."""
        # Create mapping for all columns to snake_case
        column_mapping = {}

        for col_name in df.columns:
            # Convert to snake_case
            # Handle multiple formats: UPPERCASE, CamelCase, Mixed-Case, etc.
            snake_case = col_name.lower()
            # Replace spaces and hyphens with underscores
            snake_case = snake_case.replace(' ', '_').replace('-', '_')
            # Replace multiple underscores with single
            snake_case = re.sub(r'_+', '_', snake_case)
            # Remove leading/trailing underscores
            snake_case = snake_case.strip('_')

            if snake_case != col_name:
                column_mapping[col_name] = snake_case

        # Apply the mapping
        if column_mapping:
            for old_name, new_name in column_mapping.items():
                df = df.withColumnRenamed(old_name, new_name)

        return df

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Engineer features for analysis."""
        from pyspark.sql.functions import lit, when, expr
        from pyspark.sql.types import DoubleType

        df_enhanced = df

        # Calculate salary_avg from salary_from and salary_to if not already present
        if 'salary_avg' not in df.columns:
            if 'salary_from' in df.columns and 'salary_to' in df.columns:
                # Use try_cast for safe conversion (returns null for invalid values)
                df_enhanced = df_enhanced.withColumn(
                    'salary_from_num',
                    expr("try_cast(salary_from as double)")
                )
                df_enhanced = df_enhanced.withColumn(
                    'salary_to_num',
                    expr("try_cast(salary_to as double)")
                )

                # Compute average where both exist
                df_enhanced = df_enhanced.withColumn(
                    'salary_avg',
                    when(
                        col('salary_from_num').isNotNull() & col('salary_to_num').isNotNull(),
                        (col('salary_from_num') + col('salary_to_num')) / 2
                    ).when(
                        col('salary_from_num').isNotNull(),
                        col('salary_from_num')
                    ).when(
                        col('salary_to_num').isNotNull(),
                        col('salary_to_num')
                    ).otherwise(lit(None).cast(DoubleType()))
                )

                # Drop temporary columns
                df_enhanced = df_enhanced.drop('salary_from_num', 'salary_to_num')

            elif 'salary' in df.columns:
                # Use try_cast for safe conversion
                df_enhanced = df_enhanced.withColumn(
                    'salary_avg',
                    expr("try_cast(salary as double)")
                )
            else:
                # Create placeholder if no salary data
                df_enhanced = df_enhanced.withColumn('salary_avg', lit(None).cast(DoubleType()))

        return df_enhanced
