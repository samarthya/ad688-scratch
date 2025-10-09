"""
Comprehensive data cleaning utilities for job market analytics.

This module provides robust data cleaning functions that handle the specific
data quality issues found in the Lightcast job postings dataset.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Optional, Tuple


class JobMarketDataCleaner:
    """
    Comprehensive data cleaner for job market datasets.

    Handles specific data quality issues in the Lightcast dataset including:
    - JSON string parsing for location and education data
    - Salary data validation and imputation
    - Text data cleaning and standardization
    - Missing value handling with intelligent imputation
    """

    def __init__(self):
        """Initialize the data cleaner."""
        self.cleaning_stats = {}

    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean the entire dataset with comprehensive data quality improvements.

        Args:
            df: Raw job market dataset

        Returns:
            Tuple of (cleaned_dataframe, cleaning_statistics)
        """
        print("Starting comprehensive data cleaning...")
        original_shape = df.shape
        df_clean = df.copy()

        # Track cleaning statistics
        stats = {
            'original_shape': original_shape,
            'steps_completed': [],
            'columns_processed': {},
            'rows_removed': 0,
            'values_imputed': 0,
            'columns_removed': 0,
            'columns_added': 0
        }

        # Step 1: Optimize and remove unnecessary columns
        df_clean, column_stats = self._optimize_columns(df_clean)
        stats['steps_completed'].append('column_optimization')
        stats['columns_removed'] = column_stats['columns_removed']
        stats['columns_added'] = column_stats['columns_added']

        # Step 2: Clean basic text columns
        df_clean, text_stats = self._clean_text_columns(df_clean)
        stats['steps_completed'].append('text_cleaning')
        stats['columns_processed'].update(text_stats)

        # Step 3: Parse and clean location data
        df_clean, location_stats = self._clean_location_data(df_clean)
        stats['steps_completed'].append('location_cleaning')
        stats['columns_processed'].update(location_stats)

        # Step 4: Parse and clean education data
        df_clean, education_stats = self._clean_education_data(df_clean)
        stats['steps_completed'].append('education_cleaning')
        stats['columns_processed'].update(education_stats)

        # Step 5: Clean and validate salary data
        df_clean, salary_stats = self._clean_salary_data(df_clean)
        stats['steps_completed'].append('salary_cleaning')
        stats['columns_processed'].update(salary_stats)
        stats['values_imputed'] += salary_stats.get('values_imputed', 0)

        # Step 6: Clean experience data
        df_clean, experience_stats = self._clean_experience_data(df_clean)
        stats['steps_completed'].append('experience_cleaning')
        stats['columns_processed'].update(experience_stats)

        # Step 7: Remove rows with critical missing data
        df_clean, removal_stats = self._remove_invalid_rows(df_clean)
        stats['steps_completed'].append('row_removal')
        stats['rows_removed'] = removal_stats['rows_removed']

        # Step 8: Create derived features
        df_clean, derived_stats = self._create_derived_features(df_clean)
        stats['steps_completed'].append('derived_features')
        stats['columns_processed'].update(derived_stats)

        stats['final_shape'] = df_clean.shape
        stats['cleaning_success'] = True

        print(f"Data cleaning completed:")
        print(f"  Original: {original_shape[0]:,} rows × {original_shape[1]} columns")
        print(f"  Final: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        print(f"  Rows removed: {stats['rows_removed']:,}")
        print(f"  Values imputed: {stats['values_imputed']:,}")

        return df_clean, stats

    def _optimize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize dataset by removing unnecessary columns and adding useful ones."""
        print("  Optimizing columns...")
        original_columns = len(df.columns)

        # Columns to remove (as per user requirements)
        columns_to_remove = [
            'ID',  # Will be replaced with monotonic ID
            'LAST_UPDATED_DATE', 'LAST_UPDATED_TIMESTAMP', 'POSTED',
            'ACTIVE_URLS', 'ACTIVE_SOURCES_INFO', 'MODELED_EXPIRED',
            'COMPANY_IS_STAFFING', 'STATE', 'NAICS2', 'NAICS3', 'NAICS4',
            'NAICS5', 'NAICS6', 'TITLE', 'SKILLS',
            'COMPANY', 'COMPANY_RAW',  # Use COMPANY_NAME instead
            'COUNTY'  # Use COUNTY_NAME instead
        ]

        # Remove columns that exist in the dataset
        existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        df_clean = df.drop(columns=existing_columns_to_remove)

        # Add monotonic ID
        df_clean['ID'] = range(1, len(df_clean) + 1)

        # Filter out duplicates if DUPLICATES column exists and is 1
        if 'DUPLICATES' in df_clean.columns:
            original_length = len(df_clean)
            df_clean = df_clean[df_clean['DUPLICATES'] != 1]
            duplicates_removed = original_length - len(df_clean)
            print(f"    Removed {duplicates_removed:,} duplicate records")
        else:
            duplicates_removed = 0

        # Remove the DUPLICATES column after filtering
        if 'DUPLICATES' in df_clean.columns:
            df_clean = df_clean.drop(columns=['DUPLICATES'])

        final_columns = len(df_clean.columns)
        columns_removed = original_columns - final_columns + 1  # +1 for DUPLICATES column
        columns_added = 1  # For the new ID column

        print(f"    Removed {len(existing_columns_to_remove)} unnecessary columns")
        print(f"    Added monotonic ID column")
        print(f"    Removed {duplicates_removed:,} duplicate records")
        print(f"    Final dataset: {len(df_clean):,} rows × {final_columns} columns")

        return df_clean, {
            'columns_removed': columns_removed,
            'columns_added': columns_added,
            'duplicates_removed': duplicates_removed,
            'original_columns': original_columns,
            'final_columns': final_columns
        }

    def _clean_text_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean basic text columns."""
        print("  Cleaning text columns...")
        stats = {}

        text_columns = ['TITLE_NAME', 'COMPANY_NAME']  # TITLE and COMPANY_RAW removed in optimization

        for col in text_columns:
            if col in df.columns:
                original_nulls = df[col].isna().sum()

                # Clean text data
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['', 'nan', 'None', 'null', 'NULL'], np.nan)

                # Remove rows where critical text columns are null
                if col in ['TITLE', 'COMPANY_NAME']:
                    df[col] = df[col].fillna('Unknown')

                stats[col] = {
                    'original_nulls': original_nulls,
                    'final_nulls': df[col].isna().sum(),
                    'cleaned': original_nulls - df[col].isna().sum()
                }

        return df, stats

    def _clean_location_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse and clean location data from JSON strings."""
        print("  Cleaning location data...")
        stats = {}

        if 'LOCATION' in df.columns:
            # Parse JSON location data
            df['location_parsed'] = df['LOCATION'].apply(self._parse_location_json)

            # Extract location information
            df['city_name'] = df['location_parsed'].apply(lambda x: x.get('city', 'Unknown') if isinstance(x, dict) else 'Unknown')
            df['state_name'] = df['location_parsed'].apply(lambda x: x.get('state', 'Unknown') if isinstance(x, dict) else 'Unknown')
            df['region'] = df['location_parsed'].apply(lambda x: x.get('region', 'Unknown') if isinstance(x, dict) else 'Unknown')
            df['country'] = df['location_parsed'].apply(lambda x: x.get('country', 'US') if isinstance(x, dict) else 'US')

            # Create a readable location string
            df['location_readable'] = df.apply(
                lambda row: f"{row['city_name']}, {row['state_name']}" if row['city_name'] != 'Unknown'
                else f"{row['region']} Region" if row['region'] != 'Unknown'
                else 'Unknown Location',
                axis=1
            )

            stats['location'] = {
                'parsed_successfully': df['location_parsed'].notna().sum(),
                'cities_extracted': df['city_name'].nunique(),
                'states_extracted': df['state_name'].nunique(),
                'regions_extracted': df['region'].nunique()
            }

        return df, stats

    def _parse_location_json(self, location_str: str) -> Dict[str, str]:
        """Parse location JSON string safely."""
        if pd.isna(location_str) or location_str == '':
            return {}

        try:
            # Try to parse as JSON
            location_data = json.loads(location_str)

            # Extract meaningful location information
            result = {}
            if 'lat' in location_data and 'lon' in location_data:
                # This is coordinate data - we'll use a simple mapping for major regions
                lat = location_data['lat']
                lon = location_data['lon']

                # Simple geographic region mapping (US-focused)
                if 24 <= lat <= 49 and -125 <= lon <= -66:  # Continental US
                    if lat >= 40 and lon >= -80:  # Northeast
                        result = {'region': 'Northeast', 'type': 'coordinates'}
                    elif lat >= 40 and lon < -80:  # Midwest
                        result = {'region': 'Midwest', 'type': 'coordinates'}
                    elif lat < 40 and lon >= -100:  # South
                        result = {'region': 'South', 'type': 'coordinates'}
                    else:  # West
                        result = {'region': 'West', 'type': 'coordinates'}
                else:
                    result = {'region': 'Other', 'type': 'coordinates'}
            else:
                # This might be address data
                result = location_data

            return result
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, try to extract city/state from string
            location_str = str(location_str).strip()
            if ',' in location_str:
                parts = location_str.split(',')
                if len(parts) >= 2:
                    return {
                        'city': parts[0].strip(),
                        'state': parts[1].strip(),
                        'type': 'parsed_string'
                    }
            return {'type': 'unparseable', 'raw': location_str}

    def _clean_education_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse and clean education data from JSON arrays."""
        print("  Cleaning education data...")
        stats = {}

        if 'EDUCATION_LEVELS_NAME' in df.columns:
            # Parse education JSON arrays
            df['education_parsed'] = df['EDUCATION_LEVELS_NAME'].apply(self._parse_education_json)

            # Extract primary education level
            df['education_level'] = df['education_parsed'].apply(self._extract_primary_education)

            # Create education categories
            df['education_category'] = df['education_level'].apply(self._categorize_education)

            stats['education'] = {
                'parsed_successfully': df['education_parsed'].notna().sum(),
                'education_levels_found': df['education_level'].nunique(),
                'categories_created': df['education_category'].nunique()
            }

        return df, stats

    def _parse_education_json(self, education_str: str) -> List[str]:
        """Parse education JSON array safely."""
        if pd.isna(education_str) or education_str == '':
            return []

        try:
            education_data = json.loads(education_str)
            if isinstance(education_data, list):
                return [str(item).strip() for item in education_data if str(item).strip()]
            else:
                return [str(education_data).strip()]
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, try to extract from string
            education_str = str(education_str).strip()
            if '[' in education_str and ']' in education_str:
                # Try to extract text between brackets
                matches = re.findall(r'"([^"]+)"', education_str)
                return [match.strip() for match in matches if match.strip()]
            else:
                return [education_str]

    def _extract_primary_education(self, education_list: List[str]) -> str:
        """Extract the primary/highest education level from a list."""
        if not education_list:
            return 'No Education Listed'

        # Education hierarchy (highest to lowest)
        hierarchy = [
            'Doctorate', 'PhD', 'Ph.D.', 'Doctor of Philosophy',
            'Master', 'Master\'s', 'Master\'s degree',
            'Bachelor', 'Bachelor\'s', 'Bachelor\'s degree',
            'Associate', 'Associate\'s', 'Associate\'s degree',
            'High School', 'High school', 'High school diploma',
            'No Education Listed', 'None'
        ]

        for level in hierarchy:
            for edu in education_list:
                if level.lower() in edu.lower():
                    return level

        return education_list[0]  # Return first if no hierarchy match

    def _categorize_education(self, education_level: str) -> str:
        """Categorize education level into broad categories."""
        if pd.isna(education_level):
            return 'Unknown'

        education_level = str(education_level).lower()

        if any(term in education_level for term in ['doctorate', 'phd', 'ph.d.']):
            return 'Doctorate'
        elif any(term in education_level for term in ['master', 'master\'s']):
            return 'Master\'s'
        elif any(term in education_level for term in ['bachelor', 'bachelor\'s']):
            return 'Bachelor\'s'
        elif any(term in education_level for term in ['associate', 'associate\'s']):
            return 'Associate\'s'
        elif any(term in education_level for term in ['high school', 'highschool']):
            return 'High School'
        elif any(term in education_level for term in ['no education', 'none']):
            return 'No Education'
        else:
            return 'Other'

    def _clean_salary_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean and validate salary data with intelligent imputation."""
        print("  Cleaning salary data...")
        stats = {'values_imputed': 0}

        # Convert salary columns to numeric
        salary_columns = ['SALARY_FROM', 'SALARY_TO']
        for col in salary_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create SALARY_AVG column
        if 'SALARY_FROM' in df.columns and 'SALARY_TO' in df.columns:
            # Calculate average where both values exist
            both_exist = df['SALARY_FROM'].notna() & df['SALARY_TO'].notna()
            df.loc[both_exist, 'SALARY_AVG'] = (df.loc[both_exist, 'SALARY_FROM'] + df.loc[both_exist, 'SALARY_TO']) / 2

            # Use single value where only one exists
            only_from = df['SALARY_FROM'].notna() & df['SALARY_TO'].isna()
            df.loc[only_from, 'SALARY_AVG'] = df.loc[only_from, 'SALARY_FROM']

            only_to = df['SALARY_TO'].notna() & df['SALARY_FROM'].isna()
            df.loc[only_to, 'SALARY_AVG'] = df.loc[only_to, 'SALARY_TO']

        # Validate salary ranges
        if 'SALARY_AVG' in df.columns:
            # Remove unrealistic salary values
            valid_salary = (df['SALARY_AVG'] >= 20000) & (df['SALARY_AVG'] <= 500000)
            invalid_count = (~valid_salary).sum()
            df = df[valid_salary]
            stats['invalid_salaries_removed'] = invalid_count

        # Impute missing salaries using education and location
        if 'SALARY_AVG' in df.columns:
            missing_salaries = df['SALARY_AVG'].isna()
            if missing_salaries.sum() > 0:
                df = self._impute_salaries(df)
                stats['values_imputed'] = missing_salaries.sum()

        stats['final_salary_coverage'] = df['SALARY_AVG'].notna().sum() / len(df) * 100
        return df, stats

    def _impute_salaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing salaries using education and location patterns."""
        # Create imputation groups
        imputation_groups = ['education_category', 'city_name', 'state_name']
        available_groups = [col for col in imputation_groups if col in df.columns]

        if available_groups:
            # Calculate median salary for each group
            group_medians = df.groupby(available_groups)['SALARY_AVG'].median()

            # Impute missing values
            missing_mask = df['SALARY_AVG'].isna()
            for idx in df[missing_mask].index:
                row = df.loc[idx]
                group_key = tuple(row[col] for col in available_groups)

                if group_key in group_medians and not pd.isna(group_medians[group_key]):
                    df.loc[idx, 'SALARY_AVG'] = group_medians[group_key]
                else:
                    # Fallback to overall median
                    overall_median = df['SALARY_AVG'].median()
                    if not pd.isna(overall_median):
                        df.loc[idx, 'SALARY_AVG'] = overall_median

        return df

    def _clean_experience_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean and standardize experience data."""
        print("  Cleaning experience data...")
        stats = {}

        experience_columns = ['MIN_YEARS_EXPERIENCE', 'MAX_YEARS_EXPERIENCE']

        for col in experience_columns:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Remove unrealistic values
                valid_experience = (df[col] >= 0) & (df[col] <= 50)
                df.loc[~valid_experience, col] = np.nan

                stats[col] = {
                    'original_nulls': df[col].isna().sum(),
                    'valid_values': valid_experience.sum()
                }

        # Create experience level categories
        if 'MIN_YEARS_EXPERIENCE' in df.columns:
            df['experience_level'] = df['MIN_YEARS_EXPERIENCE'].apply(self._categorize_experience)
            stats['experience_categories'] = df['experience_level'].nunique()

        return df, stats

    def _categorize_experience(self, years: float) -> str:
        """Categorize years of experience into levels."""
        if pd.isna(years):
            return 'Unknown'

        if years == 0:
            return 'Entry Level (0 years)'
        elif years <= 2:
            return 'Junior (1-2 years)'
        elif years <= 5:
            return 'Mid Level (3-5 years)'
        elif years <= 10:
            return 'Senior (6-10 years)'
        elif years <= 15:
            return 'Principal (11-15 years)'
        else:
            return 'Executive (15+ years)'

    def _remove_invalid_rows(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove rows with critical missing data."""
        print("  Removing invalid rows...")
        original_rows = len(df)

        # Remove rows missing critical information
        critical_columns = ['TITLE_NAME', 'COMPANY_NAME']  # Updated after column optimization
        df = df.dropna(subset=critical_columns)

        # Remove rows with no salary information
        if 'SALARY_AVG' in df.columns:
            df = df.dropna(subset=['SALARY_AVG'])

        rows_removed = original_rows - len(df)

        return df, {'rows_removed': rows_removed}

    def _create_derived_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create useful derived features for analysis."""
        print("  Creating derived features...")
        stats = {}

        # Create salary ranges
        if 'SALARY_AVG' in df.columns:
            df['salary_range'] = pd.cut(
                df['SALARY_AVG'],
                bins=[0, 50000, 75000, 100000, 125000, 150000, 200000, float('inf')],
                labels=['<50k', '50-75k', '75-100k', '100-125k', '125-150k', '150-200k', '200k+']
            )
            stats['salary_ranges'] = df['salary_range'].nunique()

        # Create company size categories (if we have company data)
        if 'COMPANY_NAME' in df.columns:
            company_counts = df['COMPANY_NAME'].value_counts()
            df['company_size'] = df['COMPANY_NAME'].map(company_counts).apply(self._categorize_company_size)
            stats['company_sizes'] = df['company_size'].nunique()

        return df, stats

    def _categorize_company_size(self, job_count: int) -> str:
        """Categorize company size based on number of job postings."""
        if pd.isna(job_count):
            return 'Unknown'
        elif job_count == 1:
            return 'Small (1 job)'
        elif job_count <= 5:
            return 'Medium (2-5 jobs)'
        elif job_count <= 20:
            return 'Large (6-20 jobs)'
        else:
            return 'Enterprise (20+ jobs)'
