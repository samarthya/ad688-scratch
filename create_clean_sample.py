#!/usr/bin/env python3
"""
Create a clean sample dataset with proper data cleaning and validation.

This script uses the comprehensive data cleaner to create a high-quality
sample dataset for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json
import re

# Add src to path
sys.path.append('src')

# Import the cleaner class directly
sys.path.append('src/data')
from data_cleaner import JobMarketDataCleaner

def create_clean_sample():
    """Create a clean sample dataset using comprehensive data cleaning."""

    print("=== Creating Clean Sample Dataset ===")

    # Initialize data cleaner
    cleaner = JobMarketDataCleaner()

    # Load raw data
    raw_data_path = Path("data/raw/lightcast_job_postings.csv")
    if not raw_data_path.exists():
        print(f"ERROR: Raw data not found at {raw_data_path}")
        return False

    print(f"Loading raw data from {raw_data_path}...")
    df_raw = pd.read_csv(raw_data_path, low_memory=False)
    print(f"Loaded {len(df_raw):,} rows from raw dataset")

    # Take a random sample for processing
    sample_size = min(10000, len(df_raw))  # Process up to 10k rows
    df_sample = df_raw.sample(n=sample_size, random_state=42)
    print(f"Sampled {len(df_sample):,} rows for processing")

    # Clean the data
    df_clean, cleaning_stats = cleaner.clean_dataset(df_sample)

    # Save the clean sample
    output_path = Path("data/processed/job_market_clean_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving clean sample to {output_path}...")
    df_clean.to_csv(output_path, index=False)

    # Print summary
    print("\n=== CLEANING SUMMARY ===")
    print(f"Original sample: {cleaning_stats['original_shape'][0]:,} rows")
    print(f"Final clean: {cleaning_stats['final_shape'][0]:,} rows")
    print(f"Rows removed: {cleaning_stats['rows_removed']:,}")
    print(f"Values imputed: {cleaning_stats['values_imputed']:,}")
    print(f"Cleaning steps: {', '.join(cleaning_stats['steps_completed'])}")

    # Show data quality metrics
    print("\n=== DATA QUALITY METRICS ===")
    key_columns = ['TITLE', 'COMPANY_NAME', 'location_readable', 'SALARY_AVG',
                   'education_level', 'experience_level']

    for col in key_columns:
        if col in df_clean.columns:
            coverage = df_clean[col].notna().sum() / len(df_clean) * 100
            unique_vals = df_clean[col].nunique()
            print(f"{col}: {coverage:.1f}% coverage, {unique_vals} unique values")

    # Show sample of cleaned data
    print("\n=== SAMPLE OF CLEANED DATA ===")
    display_columns = ['TITLE', 'COMPANY_NAME', 'location_readable', 'SALARY_AVG',
                       'education_level', 'experience_level']
    available_columns = [col for col in display_columns if col in df_clean.columns]

    if available_columns:
        print(df_clean[available_columns].head().to_string(index=False))

    print(f"\nClean sample dataset created successfully at {output_path}")
    return True

if __name__ == "__main__":
    success = create_clean_sample()
    if success:
        print("Clean sample dataset creation completed successfully!")
    else:
        print("Clean sample dataset creation failed!")
        sys.exit(1)
