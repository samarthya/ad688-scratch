"""
Data transformation utilities for the job market analytics system.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, trim, upper, lower, regexp_replace, lit
import re


class DataTransformer:
    """Data transformation utilities for job market analytics."""

    def clean_and_standardize(self, df: DataFrame) -> DataFrame:
        """Clean and standardize the dataset."""
        # Step 1: Standardize column names (apply mapping + snake_case, handle conflicts)
        df_clean = self._standardize_column_names(df)

        # Step 2: Trim string columns
        for col_name in df_clean.columns:
            if df_clean.schema[col_name].dataType.typeName() == 'string':
                df_clean = df_clean.withColumn(col_name, trim(col(col_name)))

        # Step 3: Clean employment type data
        df_clean = self._clean_employment_type(df_clean)

        # Step 4: Clean remote type data
        df_clean = self._clean_remote_type(df_clean)

        return df_clean

    def _clean_employment_type(self, df: DataFrame) -> DataFrame:
        """Clean employment_type_name column - remove special characters and handle missing values"""
        if 'employment_type_name' not in df.columns:
            return df

        # Remove non-ASCII characters (special characters)
        df = df.withColumn(
            'employment_type_name',
            regexp_replace(col('employment_type_name'), '([^\x00-\x7f])', '')
        )

        # Handle missing values: NULL, empty string, [NONE], etc.
        df = df.withColumn(
            'employment_type_name',
            when(
                (col('employment_type_name').isNull()) |
                (trim(col('employment_type_name')) == '') |
                (upper(trim(col('employment_type_name'))) == '[NONE]') |
                (upper(trim(col('employment_type_name'))) == 'NONE') |
                (upper(trim(col('employment_type_name'))) == 'N/A') |
                (upper(trim(col('employment_type_name'))) == 'NA'),
                lit('Undefined')
            ).otherwise(col('employment_type_name'))
        )

        return df

    def _clean_remote_type(self, df: DataFrame) -> DataFrame:
        """Clean remote_type_name column - handle missing values"""
        if 'remote_type_name' not in df.columns:
            return df

        # Remove non-ASCII characters
        df = df.withColumn(
            'remote_type_name',
            regexp_replace(col('remote_type_name'), '([^\x00-\x7f])', '')
        )

        # Handle missing values
        df = df.withColumn(
            'remote_type_name',
            when(
                (col('remote_type_name').isNull()) |
                (trim(col('remote_type_name')) == '') |
                (upper(trim(col('remote_type_name'))) == '[NONE]') |
                (upper(trim(col('remote_type_name'))) == 'NONE') |
                (upper(trim(col('remote_type_name'))) == 'N/A') |
                (upper(trim(col('remote_type_name'))) == 'NA'),
                lit('Undefined')
            ).otherwise(col('remote_type_name'))
        )

        return df

    def _standardize_column_names(self, df: DataFrame) -> DataFrame:
        """Convert all column names using centralized column mapping."""
        # Import centralized column mapping
        from src.config.column_mapping import LIGHTCAST_COLUMN_MAPPING

        # Step 1: Build complete mapping (explicit + snake_case fallback)
        final_mapping = {}

        # Add explicit mappings from LIGHTCAST_COLUMN_MAPPING
        for old_col in df.columns:
            if old_col in LIGHTCAST_COLUMN_MAPPING:
                final_mapping[old_col] = LIGHTCAST_COLUMN_MAPPING[old_col]

        # Step 2: For remaining UPPERCASE columns, add snake_case mapping
        for col_name in df.columns:
            if col_name not in final_mapping and col_name.isupper():
                # Convert to snake_case
                snake_case = col_name.lower()
                # Replace spaces and hyphens with underscores
                snake_case = snake_case.replace(' ', '_').replace('-', '_')
                # Replace multiple underscores with single
                snake_case = re.sub(r'_+', '_', snake_case)
                # Remove leading/trailing underscores
                snake_case = snake_case.strip('_')

                if snake_case != col_name:
                    final_mapping[col_name] = snake_case

        # Step 3: Check for conflicts (multiple columns mapping to same target)
        # If conflict, keep only the first mapping and drop the rest
        target_to_source = {}
        columns_to_drop = []
        for old_col in df.columns:
            if old_col in final_mapping:
                new_col = final_mapping[old_col]
                if new_col in target_to_source:
                    # Conflict! Drop this column
                    columns_to_drop.append(old_col)
                else:
                    target_to_source[new_col] = old_col

        # Drop conflicting columns first
        if columns_to_drop:
            df = df.drop(*columns_to_drop)

        # Step 4: Apply remaining mappings (no conflicts now)
        for old_col in df.columns:
            if old_col in final_mapping:
                df = df.withColumnRenamed(old_col, final_mapping[old_col])

        return df

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Engineer features for analysis with sophisticated salary imputation."""
        from pyspark.sql.functions import lit, when, expr, col, avg, median
        from pyspark.sql.types import DoubleType
        from pyspark.sql.window import Window

        df_enhanced = df

        # Step 1: Calculate initial salary_avg from available salary data
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            # Compute average where both exist, otherwise use the one that exists
            df_enhanced = df_enhanced.withColumn(
                'salary_avg',
                when(
                    col('salary_min').isNotNull() & col('salary_max').isNotNull(),
                    (col('salary_min') + col('salary_max')) / 2
                ).when(
                    col('salary_min').isNotNull(),
                    col('salary_min')
                ).when(
                    col('salary_max').isNotNull(),
                    col('salary_max')
                ).otherwise(lit(None).cast(DoubleType()))
            )
        elif 'salary' in df.columns:
            # Use existing salary column
            df_enhanced = df_enhanced.withColumn('salary_avg', col('salary'))
        else:
            # Create placeholder if no salary data
            df_enhanced = df_enhanced.withColumn('salary_avg', lit(None).cast(DoubleType()))

        # Step 2: Impute missing salaries using location and industry grouping
        if 'industry' in df.columns and 'city_name' in df.columns:
            # Calculate industry-location medians for imputation
            industry_location_medians = df_enhanced.filter(
                col('salary_avg').isNotNull() &
                col('industry').isNotNull() &
                col('city_name').isNotNull()
            ).groupBy('industry', 'city_name').agg(
                median('salary_avg').alias('industry_location_median')
            )

            # Join with medians and impute missing values
            df_enhanced = df_enhanced.join(
                industry_location_medians,
                ['industry', 'city_name'],
                'left'
            )

            # Impute using industry-location median
            df_enhanced = df_enhanced.withColumn(
                'salary_avg',
                when(
                    col('salary_avg').isNull() & col('industry_location_median').isNotNull(),
                    col('industry_location_median')
                ).otherwise(col('salary_avg'))
            )

            # If still missing, use industry median
            industry_medians = df_enhanced.filter(
                col('salary_avg').isNotNull() & col('industry').isNotNull()
            ).groupBy('industry').agg(
                median('salary_avg').alias('industry_median')
            )

            df_enhanced = df_enhanced.join(
                industry_medians,
                'industry',
                'left'
            )

            df_enhanced = df_enhanced.withColumn(
                'salary_avg',
                when(
                    col('salary_avg').isNull() & col('industry_median').isNotNull(),
                    col('industry_median')
                ).otherwise(col('salary_avg'))
            )

            # Clean up temporary columns
            df_enhanced = df_enhanced.drop('industry_location_median', 'industry_median')

        # Step 3: Final fallback - use overall median
        overall_median = df_enhanced.filter(col('salary_avg').isNotNull()).agg(
            median('salary_avg').alias('overall_median')
        ).collect()[0]['overall_median']

        if overall_median is not None:
            df_enhanced = df_enhanced.withColumn(
                'salary_avg',
                when(col('salary_avg').isNull(), lit(overall_median))
                .otherwise(col('salary_avg'))
            )
        else:
            # Ultimate fallback
            df_enhanced = df_enhanced.withColumn(
                'salary_avg',
                when(col('salary_avg').isNull(), lit(75000.0))
                .otherwise(col('salary_avg'))
            )

        # Step 4: Handle missing values in categorical columns
        # Clean remote_type column
        if 'remote_type' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'remote_type',
                when(
                    (col('remote_type').isNull()) |
                    (col('remote_type') == '[None]') |
                    (col('remote_type') == 'None') |
                    (col('remote_type') == 'N/A') |
                    (col('remote_type') == 'NA') |
                    (trim(col('remote_type')) == ''),
                    lit('Undefined')
                ).otherwise(col('remote_type'))
            )

        # Clean employment_type column
        if 'employment_type' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'employment_type',
                when(
                    (col('employment_type').isNull()) |
                    (col('employment_type') == '[None]') |
                    (col('employment_type') == 'None') |
                    (col('employment_type') == 'N/A') |
                    (col('employment_type') == 'NA') |
                    (trim(col('employment_type')) == ''),
                    lit('Undefined')
                ).otherwise(col('employment_type'))
            )

        # Clean industry column
        if 'industry' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'industry',
                when(
                    (col('industry').isNull()) |
                    (col('industry') == '[None]') |
                    (col('industry') == 'None') |
                    (col('industry') == 'N/A') |
                    (col('industry') == 'NA') |
                    (trim(col('industry')) == ''),
                    lit('Undefined')
                ).otherwise(col('industry'))
            )

        # Clean city_name column
        if 'city_name' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'city_name',
                when(
                    (col('city_name').isNull()) |
                    (col('city_name') == '[None]') |
                    (col('city_name') == 'None') |
                    (col('city_name') == 'N/A') |
                    (col('city_name') == 'NA') |
                    (trim(col('city_name')) == ''),
                    lit('Undefined')
                ).otherwise(col('city_name'))
            )

        # Step 5: Create derived columns for analysis
        # Create experience_level from experience_min
        if 'experience_min' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'experience_level',
                when(col('experience_min').isNull(), lit('Unknown'))
                .when(col('experience_min') <= 2, lit('Entry Level (0-2 years)'))
                .when(col('experience_min') <= 5, lit('Mid Level (3-5 years)'))
                .when(col('experience_min') <= 10, lit('Senior Level (6-10 years)'))
                .otherwise(lit('Leadership (10+ years)'))
            )
        else:
            df_enhanced = df_enhanced.withColumn('experience_level', lit('Unknown'))

        # Create experience_years from experience_min
        if 'experience_min' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'experience_years',
                when(col('experience_min').isNull(), lit(2.0))
                .otherwise(col('experience_min').cast('double'))
            )
        else:
            df_enhanced = df_enhanced.withColumn('experience_years', lit(2.0))

        # Create ai_related flag from title_clean (actual job titles)
        if 'title_clean' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'ai_related',
                when(
                    col('title_clean').rlike('(?i)(ai|artificial intelligence|machine learning|ml|data science|data scientist|deep learning|neural network|nlp|natural language)'),
                    lit(True)
                ).otherwise(lit(False))
            )
        else:
            df_enhanced = df_enhanced.withColumn('ai_related', lit(False))

        # Create remote_allowed flag from remote_type
        if 'remote_type' in df_enhanced.columns:
            df_enhanced = df_enhanced.withColumn(
                'remote_allowed',
                when(
                    col('remote_type').rlike('(?i)(remote|hybrid)'),
                    lit(True)
                ).otherwise(lit(False))
            )
        else:
            df_enhanced = df_enhanced.withColumn('remote_allowed', lit(False))

        return df_enhanced
