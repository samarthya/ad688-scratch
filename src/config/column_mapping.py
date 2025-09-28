"""
Centralized Column Mapping Configuration

This module provides standardized column mapping between raw Lightcast data
and processed analysis-ready format. Used consistently across all analysis
components to ensure data consistency.
"""

# Core Column Mapping: Raw Lightcast → Analysis Format
LIGHTCAST_COLUMN_MAPPING = {
    # Core Identification
    'ID': 'job_id',
    'TITLE': 'title',
    'TITLE_CLEAN': 'title_clean', 
    'COMPANY': 'company',
    'LOCATION': 'location',
    
    # Salary Data (Multiple Sources)
    'SALARY_FROM': 'salary_min',
    'SALARY_TO': 'salary_max',
    'SALARY': 'salary_single',
    'ORIGINAL_PAY_PERIOD': 'pay_period',
    
    # Industry & Experience
    'NAICS2_NAME': 'industry',
    'MIN_YEARS_EXPERIENCE': 'experience_min',
    'MAX_YEARS_EXPERIENCE': 'experience_max',
    
    # Skills & Requirements
    'SKILLS_NAME': 'required_skills',
    'EDUCATION_LEVELS_NAME': 'education_required',
    
    # Work Arrangements
    'REMOTE_TYPE_NAME': 'remote_type',
    'EMPLOYMENT_TYPE_NAME': 'employment_type'
}

# Derived columns created during processing
DERIVED_COLUMNS = [
    'salary_avg_imputed',    # Smart salary calculation with imputation
    'experience_years',      # Numeric experience from MIN_YEARS_EXPERIENCE
    'ai_related',           # AI/ML role classification
    'remote_allowed',       # Boolean remote work flag
    'experience_level',     # Standardized experience categories
    'industry_clean'        # Cleaned industry names
]

# Analysis-ready column names (for consistent usage across all modules)
ANALYSIS_COLUMNS = {
    'salary': 'salary_avg_imputed',
    'industry': 'industry', 
    'experience': 'experience_years',
    'title': 'title',
    'location': 'location',
    'remote': 'remote_allowed',
    'employment_type': 'employment_type'
}

# Experience level categorization mapping
EXPERIENCE_CATEGORIES = {
    'entry': 'Entry Level (0-2 years)',
    'mid': 'Mid Level (3-5 years)', 
    'senior': 'Senior Level (6-10 years)',
    'leadership': 'Leadership (10+ years)'
}

def get_analysis_column(column_key):
    """Get standardized analysis column name"""
    return ANALYSIS_COLUMNS.get(column_key, column_key)

def map_lightcast_columns(df):
    """Apply standardized column mapping to raw Lightcast DataFrame"""
    from pyspark.sql.functions import col, when, coalesce, lit
    
    # Check if it's a Spark DataFrame
    if hasattr(df, 'withColumnRenamed'):
        # Spark DataFrame - apply column renaming
        df_mapped = df
        for old_col, new_col in LIGHTCAST_COLUMN_MAPPING.items():
            if old_col in df.columns:
                df_mapped = df_mapped.withColumnRenamed(old_col, new_col)
        
        # Create derived columns for Spark
        if 'salary_min' in df_mapped.columns and 'salary_max' in df_mapped.columns:
            df_mapped = df_mapped.withColumn(
                'salary', 
                (col('salary_min') + col('salary_max')) / 2
            )
        
        if 'experience_min' in df_mapped.columns:
            df_mapped = df_mapped.withColumn(
                'experience_years',
                coalesce(col('experience_min'), lit(2))
            )
                
    else:
        # Pandas DataFrame - apply direct column mapping
        df_mapped = df.rename(columns=LIGHTCAST_COLUMN_MAPPING)
        
        # Create derived columns for pandas
        if 'salary_min' in df_mapped.columns and 'salary_max' in df_mapped.columns:
            df_mapped['salary'] = (df_mapped['salary_min'] + df_mapped['salary_max']) / 2
        
        if 'experience_min' in df_mapped.columns:
            df_mapped['experience_years'] = df_mapped['experience_min'].fillna(2)
            
    return df_mapped

def categorize_experience_level(years_col):
    """Categorize experience years into standard levels - returns Spark expression"""
    from pyspark.sql.functions import when, col, isnan, isnull
    
    return when(
        (col(years_col).isNull()) | (isnan(col(years_col))), 'Unknown'
    ).when(
        col(years_col) <= 2, EXPERIENCE_CATEGORIES['entry']
    ).when(
        col(years_col) <= 5, EXPERIENCE_CATEGORIES['mid']
    ).when(
        col(years_col) <= 10, EXPERIENCE_CATEGORIES['senior']
    ).otherwise(EXPERIENCE_CATEGORIES['leadership'])