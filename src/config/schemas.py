"""
Data schemas for job market analysis.

Defines the expected schema for raw Lightcast data and processed data
to ensure consistent data validation and processing.
"""

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, DateType, TimestampType
)

# Raw Lightcast data schema (131 columns)
LIGHTCAST_SCHEMA = StructType([
    StructField("ID", StringType(), True),
    StructField("TITLE", StringType(), True),
    StructField("TITLE_CLEAN", StringType(), True),
    StructField("COMPANY", StringType(), True),
    StructField("LOCATION", StringType(), True),
    StructField("SALARY_FROM", DoubleType(), True),
    StructField("SALARY_TO", DoubleType(), True),
    StructField("SALARY", DoubleType(), True),
    StructField("ORIGINAL_PAY_PERIOD", StringType(), True),
    StructField("NAICS2_NAME", StringType(), True),
    StructField("MIN_YEARS_EXPERIENCE", IntegerType(), True),
    StructField("MAX_YEARS_EXPERIENCE", IntegerType(), True),
    StructField("SKILLS_NAME", StringType(), True),
    StructField("EDUCATION_LEVELS_NAME", StringType(), True),
    StructField("REMOTE_TYPE_NAME", StringType(), True),
    StructField("EMPLOYMENT_TYPE_NAME", StringType(), True),
    # Add other fields as needed...
])

# Processed data schema (analysis-ready)
PROCESSED_SCHEMA = StructType([
    StructField("job_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("title_clean", StringType(), True),
    StructField("company", StringType(), True),
    StructField("location", StringType(), True),
    StructField("salary_min", DoubleType(), True),
    StructField("salary_max", DoubleType(), True),
    StructField("salary_single", DoubleType(), True),
    StructField("pay_period", StringType(), True),
    StructField("industry", StringType(), True),
    StructField("experience_min", IntegerType(), True),
    StructField("experience_max", IntegerType(), True),
    StructField("required_skills", StringType(), True),
    StructField("education_required", StringType(), True),
    StructField("remote_type", StringType(), True),
    StructField("employment_type", StringType(), True),
    # Derived columns
    StructField("salary_avg_imputed", DoubleType(), True),
    StructField("experience_years", IntegerType(), True),
    StructField("ai_related", BooleanType(), True),
    StructField("remote_allowed", BooleanType(), True),
    StructField("experience_level", StringType(), True),
    StructField("industry_clean", StringType(), True),
    StructField("created_at", TimestampType(), True),
    StructField("updated_at", TimestampType(), True)
])
