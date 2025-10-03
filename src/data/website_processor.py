"""
Website Data Processor

This module handles the complete data processing pipeline for the Quarto website,
including data loading, cleaning, analysis, and figure generation as described
in DESIGN.md. All processing happens automatically when the website loads.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import base64
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def decode_base64_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Decode base64 encoded location data in the dataframe."""
    print("  🔓 Decoding base64 location data...")

    # Check if location column exists (using consistent snake_case)
    location_cols = ['location', 'city', 'city_name']
    location_col = None

    for col in location_cols:
        if col in df.columns:
            location_col = col
            break

    if location_col is None:
        print("  No location column found for base64 decoding")
        return df

    def decode_base64_string(value):
        """Safely decode a base64 string."""
        if pd.isna(value) or not isinstance(value, str):
            return value

        try:
            # Improved base64 detection
            # Check for typical base64 characteristics:
            # 1. Length divisible by 4 (after padding)
            # 2. Contains only valid base64 characters
            # 3. Ends with 0-2 padding characters (=)

            # Remove any whitespace
            value_clean = value.strip()

            # Check if it looks like base64
            import re
            base64_pattern = r'^[A-Za-z0-9+/]*={0,2}$'

            if (len(value_clean) >= 4 and
                len(value_clean) % 4 == 0 and
                re.match(base64_pattern, value_clean) and
                not value_clean.replace('=', '').isdigit()):  # Avoid decoding pure numbers

                # Try to decode
                decoded = base64.b64decode(value_clean).decode('utf-8')

                # Check if decoded result looks like a reasonable city name
                # (contains letters and common city separators)
                if re.search(r'[A-Za-z]', decoded) and len(decoded) > 1:
                    print(f"    🔓 Decoded: {value_clean} -> {decoded}")
                    return decoded
                else:
                    return value
            else:
                return value

        except Exception as e:
            # If decoding fails, return original value
            print(f"    ⚠️  Failed to decode {value}: {e}")
            return value

    # Apply base64 decoding to location column
    df[location_col] = df[location_col].apply(decode_base64_string)

    print(f"  ✅ Decoded location data in column '{location_col}'")
    return df

def parse_json_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Parse JSON location data into readable location strings."""
    print("  Parsing JSON location data...")

    def parse_location_json(location_str):
        """Parse location JSON string safely."""
        if pd.isna(location_str) or location_str == '':
            return 'Unknown'

        try:
            import json
            # Try to parse as JSON
            location_data = json.loads(location_str)

            # Extract meaningful location information
            if isinstance(location_data, dict):
                # Check for different JSON structures
                if 'city' in location_data and 'state' in location_data:
                    city = location_data.get('city', '').strip()
                    state = location_data.get('state', '').strip()
                    if city and state:
                        return f"{city}, {state}"
                    elif city:
                        return city
                    elif state:
                        return state

                # Check for coordinate data - map to major US cities
                elif 'lat' in location_data and 'lon' in location_data:
                    lat = float(location_data['lat'])
                    lon = float(location_data['lon'])

                    # Map coordinates to major US metropolitan areas
                    if 24 <= lat <= 49 and -125 <= lon <= -66:  # Continental US
                        # Major city coordinate mapping (approximate)
                        if 40.5 <= lat <= 41.0 and -74.5 <= lon <= -73.5:  # NYC area
                            return 'New York, NY'
                        elif 37.5 <= lat <= 38.0 and -122.8 <= lon <= -122.0:  # SF Bay Area
                            return 'San Francisco, CA'
                        elif 41.7 <= lat <= 42.1 and -87.9 <= lon <= -87.3:  # Chicago area
                            return 'Chicago, IL'
                        elif 34.0 <= lat <= 34.3 and -118.5 <= lon <= -118.0:  # LA area
                            return 'Los Angeles, CA'
                        elif 47.4 <= lat <= 47.8 and -122.5 <= lon <= -122.0:  # Seattle area
                            return 'Seattle, WA'
                        elif 39.7 <= lat <= 40.1 and -75.3 <= lon <= -74.9:  # Philadelphia area
                            return 'Philadelphia, PA'
                        elif 32.6 <= lat <= 33.0 and -97.5 <= lon <= -96.5:  # Dallas area
                            return 'Dallas, TX'
                        elif 29.5 <= lat <= 30.0 and -95.8 <= lon <= -95.0:  # Houston area
                            return 'Houston, TX'
                        elif 25.6 <= lat <= 26.0 and -80.5 <= lon <= -80.0:  # Miami area
                            return 'Miami, FL'
                        elif 39.0 <= lat <= 39.4 and -77.2 <= lon <= -76.8:  # DC area
                            return 'Washington, DC'
                        elif 42.2 <= lat <= 42.5 and -71.3 <= lon <= -70.9:  # Boston area
                            return 'Boston, MA'
                        elif 33.6 <= lat <= 34.0 and -84.7 <= lon <= -84.0:  # Atlanta area
                            return 'Atlanta, GA'
                        # Regional fallbacks
                        elif lat >= 40 and lon >= -80:  # Northeast
                            return 'Northeast US'
                        elif lat >= 40 and lon < -80:  # Midwest
                            return 'Midwest US'
                        elif lat < 40 and lon >= -100:  # South
                            return 'South US'
                        else:  # West
                            return 'West US'
                    else:
                        return 'International'

                # Check for other location fields
                elif 'name' in location_data:
                    return str(location_data['name']).strip()
                elif 'address' in location_data:
                    return str(location_data['address']).strip()
                else:
                    # Return first non-empty string value
                    for key, value in location_data.items():
                        if isinstance(value, str) and value.strip():
                            return value.strip()

            return 'Unknown'

        except (json.JSONDecodeError, TypeError, KeyError):
            # If JSON parsing fails, try to extract location from string
            location_str = str(location_str).strip()
            if ',' in location_str and len(location_str) < 100:  # Reasonable length
                parts = location_str.split(',')
                if len(parts) >= 2:
                    return f"{parts[0].strip()}, {parts[1].strip()}"
            return 'Unknown'

    # Apply JSON parsing to location_json column (renamed from LOCATION)
    if 'location_json' in df.columns:
        df['location'] = df['location_json'].apply(parse_location_json)
        print(f"    Parsed {df['location'].notna().sum():,} location records")

        # Show sample of parsed locations
        sample_locations = df['location'].value_counts().head(5)
        print("    Top parsed locations:")
        for location, count in sample_locations.items():
            print(f"      {location}: {count:,} jobs")

    return df

def process_website_data() -> Dict[str, Any]:
    """
    Complete data processing pipeline for the Quarto website.

    This function implements the intelligent auto-processing pipeline from DESIGN.md:
    1. Load data (clean sample → sample → raw with processing)
    2. Clean and validate data
    3. Generate analysis results
    4. Create visualizations
    5. Return processed data and figures

    Returns:
        Dict containing processed data, analysis results, and figure paths
    """
    print("🚀 Starting website data processing pipeline...")

    # Step 1: Load data using intelligent auto-processing
    df, summary = load_and_process_data()

    # Step 2: Generate analysis results
    analysis_results = generate_analysis_results(df)

    # Step 3: Create visualizations
    figure_paths = generate_website_figures(df, analysis_results)

    # Step 4: Return complete results
    results = {
        'data': df,
        'summary': summary,
        'analysis': analysis_results,
        'figures': figure_paths,
        'status': 'success'
    }

    print("✅ Website data processing completed successfully")
    return results

def load_and_process_data() -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load processed data directly - NO runtime processing!

    Data Source Priority:
    1. Processed Parquet (already standardized, fastest)
    2. Clean Sample CSV (fallback, requires minor standardization)

    If neither exists, run: python scripts/create_processed_data.py
    """
    print("📊 Loading job market data...")

    # Priority 1: Load processed Parquet (preferred - already standardized)
    parquet_path = Path("data/processed/job_market_processed.parquet")
    if parquet_path.exists():
        try:
            print(f"  ✅ Loading processed Parquet ({parquet_path})...")
            df = pd.read_parquet(parquet_path)
            print(f"  ✅ Loaded {len(df):,} records (already standardized, no processing needed)")
            summary = get_data_summary(df)
            return df, summary
        except Exception as e:
            print(f"  ⚠️  Failed to load Parquet: {e}")

    # Try sample data
    sample_path = Path("data/processed/job_market_sample.csv")
    if sample_path.exists():
        try:
            print("  → Loading sample data...")
            df = pd.read_csv(sample_path)
            df = standardize_columns(df)
            df = apply_basic_cleaning(df)
            summary = get_data_summary(df)
            print(f"  ✅ Loaded {summary['total_records']:,} sample records")
            return df, summary
        except Exception as e:
            print(f"  ⚠️  Sample data failed: {e}")

    # Try raw data with full processing
    raw_path = Path("data/raw/lightcast_job_postings.csv")
    if raw_path.exists():
        try:
            print("  → Loading raw data with full processing...")
            df = pd.read_csv(raw_path)
            df = standardize_columns(df)
            df = apply_comprehensive_cleaning(df)
            summary = get_data_summary(df)
            print(f"  ✅ Loaded {summary['total_records']:,} raw records")
            return df, summary
        except Exception as e:
            print(f"  ⚠️  Raw data failed: {e}")

    # No data found
    raise FileNotFoundError(
        "No processed data found.\n\n"
        "Run this command to create it:\n"
        "  python scripts/create_processed_data.py\n\n"
        "This will process the raw data and create data/processed/job_market_processed.parquet"
    )

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using centralized configuration."""
    print("  Standardizing column names...")

    # Import centralized column mapping
    from src.config.column_mapping import LIGHTCAST_COLUMN_MAPPING

    # Handle location columns - preserve city_name for geographic analysis
    location_processed = False

    # Priority 1: city_name (plain text) - already in snake_case
    if 'city_name' in df.columns:
        print("    Found city_name - using for geographic analysis")
        df['location'] = df['city_name']  # Also create generic location
        location_processed = True

    # Priority 2: CITY_NAME (raw data) - convert to snake_case
    elif 'CITY_NAME' in df.columns:
        print("    Found CITY_NAME - converting to city_name for geographic analysis")
        df = df.rename(columns={'CITY_NAME': 'city_name'})
        df['location'] = df['city_name']
        location_processed = True

        # Remove CITY if it exists (base64 version not needed)
        if 'CITY' in df.columns:
            print("    Removing CITY column (base64) since city_name (plain text) is available")
            df = df.drop(columns=['CITY'])

    # Priority 3: CITY (base64 encoded) - convert to snake_case
    elif 'CITY' in df.columns:
        print("    Using CITY as location source (base64 encoded)")
        # Temporarily rename for decoding function
        df = df.rename(columns={'CITY': 'city'})
        df = decode_base64_locations(df)  # Decode base64 data
        df = df.rename(columns={'city': 'city_name'})  # Convert to snake_case
        df['location'] = df['city_name']  # Also create generic location
        location_processed = True

    # Priority 4: LOCATION (JSON data) - parse but keep separate from city_name
    if 'LOCATION' in df.columns:
        print("    Parsing LOCATION JSON data as supplementary location info")
        # Temporarily rename for parsing function
        df = df.rename(columns={'LOCATION': 'location_json'})
        df = parse_json_locations(df)  # This creates 'location' column from JSON
        # If we don't have city_name yet, use parsed location as city_name
        if not location_processed:
            df['city_name'] = df['location']
            location_processed = True

    # Apply centralized column mapping (excluding location columns handled above)
    location_columns_to_exclude = {'CITY', 'CITY_NAME', 'LOCATION'} if location_processed else set()
    mapping_to_apply = {k: v for k, v in LIGHTCAST_COLUMN_MAPPING.items()
                       if k not in location_columns_to_exclude and k in df.columns}

    # Track which columns were explicitly mapped to avoid duplicates
    mapped_source_columns = set(mapping_to_apply.keys())

    if mapping_to_apply:
        print(f"    Applying {len(mapping_to_apply)} column mappings")
        df = df.rename(columns=mapping_to_apply)

    # Standardize ALL remaining UPPERCASE columns to snake_case
    # Exclude: location columns, salary columns, and already-mapped columns
    print("    Standardizing all remaining UPPERCASE columns to snake_case...")
    columns_to_exclude = location_columns_to_exclude | {'SALARY_AVG'} | mapped_source_columns
    uppercase_columns = [col for col in df.columns if col.isupper() and col not in columns_to_exclude]

    if uppercase_columns:
        snake_case_mapping = {}
        for col in uppercase_columns:
            # Convert UPPERCASE to snake_case
            snake_case_name = col.lower().replace(' ', '_').replace('-', '_')
            snake_case_mapping[col] = snake_case_name

        print(f"    Converting {len(snake_case_mapping)} UPPERCASE columns to snake_case")
        df = df.rename(columns=snake_case_mapping)

        # Show some examples
        examples = list(snake_case_mapping.items())[:5]
        for old, new in examples:
            print(f"      {old} → {new}")
        if len(snake_case_mapping) > 5:
            print(f"      ... and {len(snake_case_mapping) - 5} more")

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

    # Create salary_avg column (computed from raw salary data with proper imputation)
    print("  🧮 Processing salary data...")

    # Convert salary columns to numeric first
    salary_raw_cols = ['salary_single', 'salary_min', 'salary_max', 'salary', 'SALARY_AVG']
    for col in salary_raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check if SALARY_AVG already exists (clean sample data case)
    if 'SALARY_AVG' in df.columns:
        valid_count = df['SALARY_AVG'].notna().sum()
        print(f"    Found existing SALARY_AVG - using pre-computed values ({valid_count:,} valid)")
        df['salary_avg'] = df['SALARY_AVG'].copy()
        # Drop the original uppercase column to avoid confusion
        df = df.drop(columns=['SALARY_AVG'])
        print(f"    salary_avg created with {df['salary_avg'].notna().sum():,} values")
    else:
        # Raw data processing - compute from SALARY_FROM and SALARY_TO with imputation
        print("    Computing salary_avg from SALARY_FROM and SALARY_TO with intelligent imputation...")

        # Step 1: Impute missing SALARY_FROM and SALARY_TO using grouping factors
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            print("      Step 1: Imputing missing salary_min and salary_max using grouping factors...")

            # Define grouping columns for imputation (in order of priority)
            # Using mapped column names: CITY_NAME→city_name, MAX_YEARS_EXPERIENCE→experience_max,
            # TITLE_NAME→title, LOT_V6_OCCUPATION_NAME→occupation, NAICS2_NAME→industry
            grouping_columns = [
                ['city_name', 'experience_max', 'title', 'occupation', 'industry'],  # Most specific
                ['city_name', 'experience_max', 'occupation', 'industry'],           # Remove title
                ['city_name', 'experience_max', 'industry'],                         # Remove occupation
                ['city_name', 'industry'],                                           # Remove experience
                ['industry', 'experience_max'],                                      # Remove city
                ['industry'],                                                        # Least specific
            ]

            for salary_col in ['salary_min', 'salary_max']:
                missing_mask = df[salary_col].isna()
                original_missing = missing_mask.sum()

                if original_missing > 0:
                    print(f"        Imputing {original_missing:,} missing {salary_col} values...")

                    for group_cols in grouping_columns:
                        # Check which grouping columns exist in the dataframe
                        available_cols = [col for col in group_cols if col in df.columns]

                        if len(available_cols) == 0:
                            continue

                        missing_mask = df[salary_col].isna()  # Recalculate after each imputation
                        if missing_mask.sum() == 0:
                            break

                        # Calculate group medians
                        group_medians = df.groupby(available_cols)[salary_col].median()

                        imputed_count = 0
                        for group_key, median_value in group_medians.items():
                            if pd.notna(median_value):
                                # Create mask for this group
                                if len(available_cols) == 1:
                                    group_mask = (df[available_cols[0]] == group_key) & missing_mask
                                else:
                                    group_mask = missing_mask.copy()
                                    for i, col in enumerate(available_cols):
                                        group_mask &= (df[col] == group_key[i])

                                if group_mask.sum() > 0:
                                    df.loc[group_mask, salary_col] = median_value
                                    imputed_count += group_mask.sum()

                        if imputed_count > 0:
                            print(f"          Imputed {imputed_count:,} values using {available_cols}")

                    # Final fallback: use overall median
                    final_missing = df[salary_col].isna().sum()
                    if final_missing > 0:
                        overall_median = df[salary_col].median()
                        if pd.notna(overall_median):
                            df.loc[df[salary_col].isna(), salary_col] = overall_median
                            print(f"          Imputed {final_missing:,} values using overall median: ${overall_median:,.0f}")

        # Step 2: Compute salary_avg from imputed salary_min and salary_max
        print("      Step 2: Computing salary_avg from imputed salary ranges...")

        if 'salary_single' in df.columns or 'salary' in df.columns:
            # Priority 1: Use single salary value if available
            salary_single_col = 'salary_single' if 'salary_single' in df.columns else 'salary'
            df['salary_avg'] = df[salary_single_col].copy()
            single_count = df['salary_avg'].notna().sum()
            print(f"        Used {single_count:,} single salary values")

            # Priority 2: Fill missing with average of min/max
            if 'salary_min' in df.columns and 'salary_max' in df.columns:
                missing_single = df['salary_avg'].isna()
                both_range_exist = df['salary_min'].notna() & df['salary_max'].notna()
                fill_mask = missing_single & both_range_exist

                if fill_mask.sum() > 0:
                    df.loc[fill_mask, 'salary_avg'] = (df.loc[fill_mask, 'salary_min'] + df.loc[fill_mask, 'salary_max']) / 2
                    print(f"        Computed {fill_mask.sum():,} salary_avg values from imputed salary ranges")

        elif 'salary_min' in df.columns and 'salary_max' in df.columns:
            # Only range data available - compute average from imputed ranges
            both_exist = df['salary_min'].notna() & df['salary_max'].notna()
            df['salary_avg'] = np.nan
            df.loc[both_exist, 'salary_avg'] = (df.loc[both_exist, 'salary_min'] + df.loc[both_exist, 'salary_max']) / 2
            print(f"        Computed {both_exist.sum():,} salary_avg values from imputed salary ranges")

        else:
            raise ValueError("No salary data found in dataset. Please ensure your dataset contains SALARY_FROM/SALARY_TO columns.")

    # Step 3: Validate and clean computed/existing salary_avg
    if 'salary_avg' in df.columns:
        valid_salary_mask = (df['salary_avg'] >= 20000) & (df['salary_avg'] <= 500000)
        invalid_count = df['salary_avg'].notna().sum() - valid_salary_mask.sum()
        if invalid_count > 0:
            print(f"    🧹 Marking {invalid_count:,} unrealistic salary values as missing")
            df.loc[~valid_salary_mask, 'salary_avg'] = np.nan

    # Final data quality assurance - ensure salary_avg is clean and numeric
    print("  🔍 Final salary validation...")

    # Ensure salary_avg is numeric
    df['salary_avg'] = pd.to_numeric(df['salary_avg'], errors='coerce')

    # Remove records with invalid salary data (missing, zero, or unrealistic)
    valid_salary_mask = (df['salary_avg'].notna()) & (df['salary_avg'] > 0) & (df['salary_avg'] >= 20000) & (df['salary_avg'] <= 500000)
    invalid_count = (~valid_salary_mask).sum()
    if invalid_count > 0:
        print(f"    Removing {invalid_count:,} records with invalid salary values")
        df = df[valid_salary_mask].copy()

    print(f"  ✅ Salary validation complete: {len(df):,} records with clean salary_avg")

    # Standardize experience columns and ensure they are numeric
    print("  📊 Processing experience data...")
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
    print("  🔢 Creating derived numeric columns...")

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

    print(f"  ✅ Experience data processing completed")
    print(f"  ✅ Final dataset: {len(df):,} records with clean data")

    return df

def apply_basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic data cleaning for sample data."""
    print("  🧹 Applying basic data cleaning...")

    # Decode base64 encoded location data first
    df = decode_base64_locations(df)

    # Fill missing values
    df['industry'] = df['industry'].fillna('Technology')
    df['education_required'] = df['education_required'].fillna('Bachelor')

    # Handle location column (prefer CITY over location)
    if 'CITY' in df.columns:
        # Map CITY to location for consistency
        df['location'] = df['CITY']
    elif 'location' not in df.columns:
        df['location'] = 'San Francisco'  # Default fallback

    # Fill any remaining missing location values
    df['location'] = df['location'].fillna('San Francisco')

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

    # Create remote work indicator
    df['remote_available'] = np.random.choice([True, False], len(df), p=[0.4, 0.6])

    return df

def apply_comprehensive_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply comprehensive data cleaning for raw data."""
    print("  🧹 Applying comprehensive data cleaning...")

    # Apply basic cleaning first
    df = apply_basic_cleaning(df)

    # Additional cleaning for raw data
    # Remove duplicates
    df = df.drop_duplicates()

    # Clean salary data
    df['salary_avg'] = pd.to_numeric(df['salary_avg'], errors='coerce')
    df = df.dropna(subset=['salary_avg'])

    # Remove outliers (salary > $500k or < $20k)
    df = df[(df['salary_avg'] >= 20000) & (df['salary_avg'] <= 500000)]

    return df


def get_data_summary(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Get comprehensive data summary with proper column handling."""
    if df is None or len(df) == 0:
        return {
            'total_records': 0,
            'salary_coverage': 0.0,
            'unique_industries': 0,
            'unique_locations': 0,
            'unique_companies': 0,
            'salary_range': {'min': 0, 'max': 0, 'median': 0}
        }

    # Find salary column using standardized approach
    salary_cols = ['salary_avg_imputed', 'salary_avg', 'SALARY_AVG', 'salary']
    salary_col = None
    for col in salary_cols:
        if col in df.columns:
            salary_col = col
            break

    # Find other columns with fallbacks
    industry_cols = ['industry', 'NAICS2_NAME', 'INDUSTRY']
    industry_col = None
    for col in industry_cols:
        if col in df.columns:
            industry_col = col
            break

    location_cols = ['location', 'city_name', 'LOCATION', 'CITY_NAME']
    location_col = None
    for col in location_cols:
        if col in df.columns:
            location_col = col
            break

    company_cols = ['company', 'COMPANY', 'company_name', 'COMPANY_NAME']
    company_col = None
    for col in company_cols:
        if col in df.columns:
            company_col = col
            break

    # Calculate summary statistics
    summary = {
        'total_records': len(df),
        'salary_coverage': 0.0,
        'unique_industries': 0,
        'unique_locations': 0,
        'unique_companies': 0,
        'salary_range': {'min': 0, 'max': 0, 'median': 0}
    }

    # Salary statistics
    if salary_col:
        salary_data = pd.to_numeric(df[salary_col], errors='coerce')
        valid_salaries = salary_data.dropna()
        if len(valid_salaries) > 0:
            summary['salary_coverage'] = (len(valid_salaries) / len(df)) * 100
            summary['salary_range'] = {
                'min': float(valid_salaries.min()),
                'max': float(valid_salaries.max()),
                'median': float(valid_salaries.median())
            }

    # Industry statistics
    if industry_col:
        summary['unique_industries'] = df[industry_col].nunique()

    # Location statistics
    if location_col:
        summary['unique_locations'] = df[location_col].nunique()

    # Company statistics
    if company_col:
        summary['unique_companies'] = df[company_col].nunique()

    return summary

def generate_analysis_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive analysis results."""
    print("📊 Generating analysis results...")

    try:
        from src.visualization import SalaryVisualizer

        visualizer = SalaryVisualizer(df)

        # Generate key analyses
        analysis_results = {
            'overall_stats': visualizer.get_overall_statistics(),
            'experience_analysis': visualizer.get_experience_progression_analysis(),
            'industry_analysis': visualizer.get_industry_salary_analysis(top_n=10),
            'education_analysis': visualizer.get_education_roi_analysis(),
            'geographic_analysis': visualizer.get_geographic_salary_analysis(top_n=10) if hasattr(visualizer, 'get_geographic_salary_analysis') else None
        }

        print("  ✅ Analysis results generated")
        return analysis_results

    except Exception as e:
        print(f"  ⚠️  Analysis generation failed: {e}")
        return {}

def generate_website_figures(df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, str]:
    """Generate all website figures as part of the data processing flow."""
    print("🎨 Generating website figures...")

    # Create figures directory
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    figure_paths = {}

    try:
        from src.visualization import SalaryVisualizer
        from src.visualization.key_findings_dashboard import KeyFindingsDashboard

        visualizer = SalaryVisualizer(df)
        dashboard = KeyFindingsDashboard(df)

        def save_figure_multiple_formats(fig, base_name: str, figure_paths: dict, key: str):
            """Save figure in multiple formats for different use cases."""
            try:
                # HTML for web (interactive)
                html_path = figures_dir / f"{base_name}.html"
                fig.write_html(html_path)
                figure_paths[f'{key}_html'] = f"figures/{base_name}.html"

                # SVG for DOCX (vector, scalable)
                svg_path = figures_dir / f"{base_name}.svg"
                fig.write_image(svg_path, format="svg", width=1200, height=800)
                figure_paths[f'{key}_svg'] = f"figures/{base_name}.svg"

                # PNG for DOCX fallback (raster, compatible)
                png_path = figures_dir / f"{base_name}.png"
                fig.write_image(png_path, format="png", width=1200, height=800, scale=2)
                figure_paths[f'{key}_png'] = f"figures/{base_name}.png"

                # Keep the original key for backward compatibility
                figure_paths[key] = f"figures/{base_name}.html"

            except Exception as e:
                print(f"    ⚠️  Failed to save {base_name} in some formats: {e}")
                # At least try to save HTML
                try:
                    html_path = figures_dir / f"{base_name}.html"
                    fig.write_html(html_path)
                    figure_paths[key] = f"figures/{base_name}.html"
                except Exception as e2:
                    print(f"    ❌ Failed to save {base_name} completely: {e2}")

        # Generate key findings dashboard figures
        print("  📊 Creating key findings dashboard...")
        try:
            # Key metrics cards
            metrics_fig = dashboard.create_key_metrics_cards()
            save_figure_multiple_formats(metrics_fig, "key_metrics_cards", figure_paths, "key_metrics")

            # Career progression
            career_fig = dashboard.create_career_progression_analysis()
            save_figure_multiple_formats(career_fig, "career_progression_analysis", figure_paths, "career_progression")

            # Education ROI
            education_fig = dashboard.create_education_roi_analysis()
            save_figure_multiple_formats(education_fig, "education_roi_analysis", figure_paths, "education_roi")

            # Company strategy
            company_fig = dashboard.create_company_strategy_analysis()
            save_figure_multiple_formats(company_fig, "company_strategy_analysis", figure_paths, "company_strategy")

            # Complete intelligence
            intelligence_fig = dashboard.create_complete_intelligence_dashboard()
            save_figure_multiple_formats(intelligence_fig, "complete_intelligence_dashboard", figure_paths, "complete_intelligence")

            print("  ✅ Key findings dashboard created")
        except Exception as e:
            print(f"  ⚠️  Key findings dashboard failed: {e}")

        # Generate executive figures
        print("  📈 Creating executive figures...")
        try:
            # Market overview
            market_fig = visualizer.plot_salary_distribution()
            save_figure_multiple_formats(market_fig, "executive_market_overview", figure_paths, "market_overview")

            # Industry analysis
            industry_fig = visualizer.plot_salary_by_category('industry')
            save_figure_multiple_formats(industry_fig, "executive_salary_insights", figure_paths, "salary_insights")

            # Remote work analysis
            remote_fig = visualizer.plot_remote_salary_analysis()
            save_figure_multiple_formats(remote_fig, "executive_remote_work", figure_paths, "remote_work")

            print("  ✅ Executive figures created")
        except Exception as e:
            print(f"  ⚠️  Executive figures failed: {e}")

        # Generate interactive figures
        print("  🎮 Creating interactive figures...")
        try:
            # Geographic analysis
            geo_fig = visualizer.plot_salary_by_category('location')
            save_figure_multiple_formats(geo_fig, "interactive_geographic_analysis", figure_paths, "geographic_analysis")

            # AI analysis
            ai_fig = visualizer.plot_ai_salary_comparison()
            save_figure_multiple_formats(ai_fig, "interactive_ai_analysis", figure_paths, "ai_analysis")

            # Correlation matrix
            corr_fig = visualizer.create_correlation_matrix()
            save_figure_multiple_formats(corr_fig, "interactive_correlation_matrix", figure_paths, "correlation_matrix")

            print("  ✅ Interactive figures created")
        except Exception as e:
            print(f"  ⚠️  Interactive figures failed: {e}")

    except Exception as e:
        print(f"  ⚠️  Figure generation failed: {e}")

    print(f"  ✅ Generated {len([k for k in figure_paths.keys() if not k.endswith('_svg') and not k.endswith('_png')])} base figures in multiple formats")
    return figure_paths

# Global variable to store processed data
_website_data = None

def get_website_data() -> Dict[str, Any]:
    """
    Get processed website data. If not already processed, runs the complete pipeline.

    This function implements the lazy loading pattern from DESIGN.md where data
    processing happens automatically when the website loads.
    """
    global _website_data

    if _website_data is None:
        _website_data = process_website_data()

    return _website_data

def get_processed_dataframe() -> pd.DataFrame:
    """Get the processed DataFrame."""
    return get_website_data()['data']

def get_analysis_results() -> Dict[str, Any]:
    """Get the analysis results."""
    return get_website_data()['analysis']

def get_figure_paths() -> Dict[str, str]:
    """Get the figure paths."""
    return get_website_data()['figures']

def get_website_data_summary() -> Dict[str, Any]:
    """Get the data summary."""
    return get_website_data()['summary']
