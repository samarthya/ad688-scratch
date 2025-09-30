"""
Column mappings and data transformations.

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
EXPERIENCE_LEVEL_MAPPING = {
    'entry': (0, 2),
    'mid': (2, 5),
    'senior': (5, 10),
    'executive': (10, 100)
}

# Industry standardization mapping
INDUSTRY_STANDARDIZATION = {
    'technology': ['tech', 'software', 'computer', 'it', 'information technology'],
    'finance': ['finance', 'banking', 'investment', 'financial'],
    'healthcare': ['health', 'medical', 'healthcare', 'pharmaceutical'],
    'education': ['education', 'university', 'school', 'academic'],
    'retail': ['retail', 'commerce', 'ecommerce', 'shopping'],
    'manufacturing': ['manufacturing', 'production', 'industrial', 'factory']
}

# Salary validation ranges by industry (annual)
SALARY_RANGES_BY_INDUSTRY = {
    'technology': (30000, 500000),
    'finance': (35000, 400000),
    'healthcare': (25000, 300000),
    'education': (20000, 150000),
    'retail': (20000, 100000),
    'manufacturing': (25000, 120000),
    'default': (20000, 500000)
}
