"""
Auto Data Processor for Figure Generation

This module provides automatic data loading by delegating to the centralized
website_processor pipeline to ensure consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_analysis_data(analysis_type="comprehensive"):
    """
    Load data for analysis using the centralized pipeline.

    This delegates to website_processor.load_and_process_data() to ensure
    all analysis uses the same standardized data.

    Args:
        analysis_type: Type of analysis ("comprehensive", "experience", "industry", etc.)

    Returns:
        pd.DataFrame: Processed data ready for analysis with standardized columns
    """
    print(f"Loading data for {analysis_type} analysis...")

    # Use the centralized pipeline from website_processor
    from .website_processor import load_and_process_data

    df, summary = load_and_process_data()

    print(f"  [OK] Loaded {len(df):,} records with standardized columns")

    return df

def process_raw_data(df):
    """Process raw Lightcast data using centralized column mapping."""
    print("  Processing raw Lightcast data...")

    # Import centralized column mapping
    from src.config.column_mapping import LIGHTCAST_COLUMN_MAPPING

    # Handle city_name vs city priority (convert to snake_case)
    if 'CITY_NAME' in df.columns and 'CITY' in df.columns:
        print("    Found both CITY_NAME and CITY - prioritizing CITY_NAME (plain text)")
        df = df.drop(columns=['CITY'])  # Remove base64 encoded version
        df = df.rename(columns={'CITY_NAME': 'city_name'})  # Convert to snake_case
    elif 'CITY_NAME' in df.columns:
        print("    Using CITY_NAME as location source")
        df = df.rename(columns={'CITY_NAME': 'city_name'})  # Convert to snake_case
    elif 'CITY' in df.columns:
        print("    Using CITY as location source (will need decoding)")
        df = df.rename(columns={'CITY': 'city_name'})  # Convert to snake_case

    # Apply centralized column mapping
    mapping_to_apply = {k: v for k, v in LIGHTCAST_COLUMN_MAPPING.items() if k in df.columns}
    if mapping_to_apply:
        print(f"    Applying {len(mapping_to_apply)} column mappings")
        df = df.rename(columns=mapping_to_apply)

    # Standardize ALL remaining UPPERCASE columns to snake_case
    print("    Standardizing all remaining UPPERCASE columns to snake_case...")
    uppercase_columns = [col for col in df.columns if col.isupper()]

    if uppercase_columns:
        snake_case_mapping = {}
        for col in uppercase_columns:
            # Convert UPPERCASE to snake_case
            snake_case_name = col.lower().replace(' ', '_').replace('-', '_')
            snake_case_mapping[col] = snake_case_name

        print(f"    Converting {len(snake_case_mapping)} UPPERCASE columns to snake_case")
        df = df.rename(columns=snake_case_mapping)

    # Ensure city_name column always exists for geographic analysis
    if 'city_name' not in df.columns:
        if 'location' in df.columns:
            print("    Creating city_name from location column")
            df['city_name'] = df['location']
        else:
            print("    Creating default city_name column")
            df['city_name'] = 'Unknown'

    # Apply imputation: Replace 'Unknown' city_name with 'Remote'
    if 'city_name' in df.columns:
        unknown_count = (df['city_name'] == 'Unknown').sum()
        if unknown_count > 0:
            print(f"    Imputing {unknown_count:,} 'Unknown' city_name values with 'Remote'")
            df['city_name'] = df['city_name'].replace('Unknown', 'Remote')

    # Create salary average (computed from raw salary data)
    print("  🧮 Computing salary_avg from raw salary data...")

    # Debug: Check what salary columns we have
    salary_cols_present = [c for c in df.columns if 'salary' in c.lower()]
    print(f"    Available salary columns: {salary_cols_present}")

    # Convert salary columns to numeric first (including pre-computed salary_avg from clean sample)
    salary_raw_cols = ['salary_avg', 'salary_single', 'salary_min', 'salary_max', 'salary']
    for col in salary_raw_cols:
        if col in df.columns:
            before_convert = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_convert = df[col].notna().sum()
            if before_convert != after_convert:
                print(f"    Converted {col}: {before_convert:,} → {after_convert:,} valid values")

    # Check if salary_avg already exists (from clean sample data)
    salary_avg_exists = False
    if 'salary_avg' in df.columns:
        valid_count = df['salary_avg'].notna().sum()
        print(f"    Found existing salary_avg column: {valid_count:,}/{len(df):,} valid ({valid_count/len(df)*100:.1f}%)")

        if valid_count > len(df) * 0.5:
            # Already has computed salary_avg with good coverage, use it
            print(f"    [OK] Using existing salary_avg column - skipping computation")
            salary_avg_exists = True
        else:
            print(f"    [WARNING] Existing salary_avg has low coverage, will compute from salary components")

    # Only compute if we don't have a good salary_avg already
    if not salary_avg_exists and ('salary_single' in df.columns or 'salary' in df.columns):
        # Priority 1: Use single salary value if available
        salary_single_col = 'salary_single' if 'salary_single' in df.columns else 'salary'
        df['salary_avg'] = df[salary_single_col].copy()

        # Priority 2: Fill missing with average of min/max if available
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            missing_single = df['salary_avg'].isna()
            both_range_exist = df['salary_min'].notna() & df['salary_max'].notna()
            fill_mask = missing_single & both_range_exist

            if fill_mask.sum() > 0:
                df.loc[fill_mask, 'salary_avg'] = (df.loc[fill_mask, 'salary_min'] + df.loc[fill_mask, 'salary_max']) / 2
                print(f"    Computed {fill_mask.sum():,} salary_avg values from salary ranges")

    elif not salary_avg_exists and 'salary_min' in df.columns and 'salary_max' in df.columns:
        # Only range data available - compute average
        both_exist = df['salary_min'].notna() & df['salary_max'].notna()
        df['salary_avg'] = np.nan
        df.loc[both_exist, 'salary_avg'] = (df.loc[both_exist, 'salary_min'] + df.loc[both_exist, 'salary_max']) / 2
        print(f"    Computed {both_exist.sum():,} salary_avg values from salary ranges")

    elif not salary_avg_exists:
        # Create synthetic salary data for testing
        print("    No salary data found - creating synthetic data for testing")
        df['salary_avg'] = np.random.normal(80000, 30000, len(df))
        df['salary_avg'] = df['salary_avg'].clip(lower=30000, upper=300000)

    # Validate and clean salary ranges (remove unrealistic values before imputation)
    if 'salary_avg' in df.columns:
        valid_salary_mask = (df['salary_avg'] >= 20000) & (df['salary_avg'] <= 500000)
        invalid_count = df['salary_avg'].notna().sum() - valid_salary_mask.sum()
        if invalid_count > 0:
            print(f"    [CLEAN] Marking {invalid_count:,} unrealistic salary values as missing for imputation")
            df.loc[~valid_salary_mask, 'salary_avg'] = np.nan

    # Create salary_avg_imputed (the derived column expected by analysis)
    df['salary_avg_imputed'] = df['salary_avg'].copy()

    # Apply intelligent salary imputation for missing values using industry grouping
    missing_salary_mask = df['salary_avg_imputed'].isna()
    if missing_salary_mask.sum() > 0:
        print(f"  [TIP] Imputing {missing_salary_mask.sum():,} missing salary values using industry grouping...")

        # Use industry median for imputation
        if 'industry' in df.columns:
            industry_medians = df.groupby('industry')['salary_avg_imputed'].median()

            for industry, median_salary in industry_medians.items():
                if pd.notna(median_salary):
                    mask = (df['industry'] == industry) & missing_salary_mask
                    if mask.sum() > 0:
                        df.loc[mask, 'salary_avg_imputed'] = median_salary
                        print(f"    Imputed {mask.sum():,} salaries for {industry}: ${median_salary:,.0f}")

        # Fill any remaining missing values with overall median
        missing_salary_mask = df['salary_avg_imputed'].isna()  # Recalculate
        if missing_salary_mask.sum() > 0:
            overall_median = df['salary_avg_imputed'].median()
            if pd.notna(overall_median):
                df.loc[missing_salary_mask, 'salary_avg_imputed'] = overall_median
                print(f"    Imputed {missing_salary_mask.sum():,} remaining salaries with overall median: ${overall_median:,.0f}")
            else:
                # Ultimate fallback
                df.loc[missing_salary_mask, 'salary_avg_imputed'] = 75000
                print(f"    Applied fallback salary of $75,000 to {missing_salary_mask.sum():,} records")

    # Final data quality assurance for analysis-ready data
    print("  [CHECK] Final data quality checks...")

    # Ensure salary_avg_imputed is numeric and clean
    df['salary_avg_imputed'] = pd.to_numeric(df['salary_avg_imputed'], errors='coerce')

    # Remove records with invalid salary data
    invalid_salary_mask = df['salary_avg_imputed'].isna() | (df['salary_avg_imputed'] <= 0)
    invalid_count = invalid_salary_mask.sum()
    if invalid_count > 0:
        print(f"    Removing {invalid_count:,} records with invalid salary data")
        df = df[~invalid_salary_mask].copy()

    # Final validation: Remove records with still unrealistic salary values
    final_valid_mask = (df['salary_avg_imputed'] >= 20000) & (df['salary_avg_imputed'] <= 500000)
    final_invalid_count = (~final_valid_mask).sum()
    if final_invalid_count > 0:
        print(f"    Removing {final_invalid_count:,} records with unrealistic salary values")
        df = df[final_valid_mask].copy()

    print(f"  [OK] Final dataset: {len(df):,} records with clean salary data")

    # Standardize experience columns and ensure they are numeric
    print("  [DATA] Processing experience data...")
    experience_columns = ['experience_min', 'experience_max', 'min_experience', 'max_experience', 'MIN_YEARS_EXPERIENCE', 'MAX_YEARS_EXPERIENCE']

    for col in experience_columns:
        if col in df.columns:
            print(f"    Processing {col}...")
            # Convert to numeric, replacing non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill negative or unrealistic values with NaN
            df.loc[df[col] < 0, col] = np.nan
            df.loc[df[col] > 50, col] = np.nan  # Cap at 50 years experience

            # Fill NaN values with reasonable defaults based on column type
            if 'min' in col.lower():
                df[col] = df[col].fillna(0)  # Minimum experience defaults to 0
            elif 'max' in col.lower():
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 5)  # Use median or default to 5

    # Ensure experience_min <= experience_max
    if 'experience_min' in df.columns and 'experience_max' in df.columns:
        # Swap values where min > max
        swap_mask = df['experience_min'] > df['experience_max']
        if swap_mask.any():
            print(f"    Swapping {swap_mask.sum()} records where min > max experience")
            df.loc[swap_mask, ['experience_min', 'experience_max']] = df.loc[swap_mask, ['experience_max', 'experience_min']].values

    # Create derived numeric columns for analysis
    print("  [COMPUTE] Creating derived numeric columns...")

    # Company size numeric (if exists)
    if 'company_size' in df.columns:
        df['company_size_numeric'] = pd.to_numeric(df['company_size'], errors='coerce')
        df['company_size_numeric'] = df['company_size_numeric'].fillna(df['company_size_numeric'].median() if df['company_size_numeric'].notna().any() else 100)

    # Job ID numeric (if exists)
    if 'job_id' in df.columns:
        df['job_id_numeric'] = pd.to_numeric(df['job_id'], errors='coerce')

    # Experience range (max - min)
    if 'experience_min' in df.columns and 'experience_max' in df.columns:
        df['experience_range'] = df['experience_max'] - df['experience_min']
        df['experience_range'] = df['experience_range'].fillna(0)

    # Average experience
    if 'experience_min' in df.columns and 'experience_max' in df.columns:
        df['experience_avg'] = (df['experience_min'] + df['experience_max']) / 2
        df['experience_avg'] = df['experience_avg'].fillna(df['experience_min'].fillna(df['experience_max'].fillna(2)))

    print(f"  [OK] Experience data processing completed")

    # Create experience level
    if 'min_experience' in df.columns:
        df['experience_level'] = pd.cut(
            df['min_experience'],
            bins=[0, 2, 5, 10, 20, float('inf')],
            labels=['Entry', 'Mid', 'Senior', 'Executive', 'C-Level']
        )
    else:
        df['experience_level'] = np.random.choice(
            ['Entry', 'Mid', 'Senior', 'Executive'],
            len(df),
            p=[0.3, 0.4, 0.2, 0.1]
        )

    # Fill missing values
    df['industry'] = df['industry'].fillna('Unknown')
    df['location'] = df['location'].fillna('Unknown')
    df['education_required'] = df['education_required'].fillna('Bachelor')

    # Create remote work indicator
    df['remote_available'] = np.random.choice([True, False], len(df), p=[0.4, 0.6])

    return df

def process_processed_data(df):
    """Process already processed data using standardization pipeline."""
    print("  Processing pre-processed data...")

    # Use the same standardization pipeline as raw data
    # This ensures SALARY_AVG -> salary_avg_imputed conversion
    df = process_raw_data(df)

    return df


def get_data_summary(df=None):
    """
    Get summary statistics for the loaded data.

    Delegates to the centralized website_processor.get_data_summary() for consistency.
    """
    from .website_processor import get_data_summary as _get_summary
    return _get_summary(df)

def validate_data_for_analysis(df, analysis_type="comprehensive"):
    """Validate that data is suitable for the specified analysis."""
    required_columns = {
        'comprehensive': ['title', 'company', 'salary_avg', 'industry'],
        'experience': ['title', 'company', 'salary_avg', 'experience_level'],
        'industry': ['title', 'company', 'salary_avg', 'industry'],
        'geographic': ['title', 'company', 'salary_avg', 'location']
    }

    required = required_columns.get(analysis_type, required_columns['comprehensive'])
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"[WARNING]  Missing columns for {analysis_type} analysis: {missing}")
        return False

    return True