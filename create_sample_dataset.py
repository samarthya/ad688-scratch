#!/usr/bin/env python3
"""
Create a sample dataset from the full Lightcast dataset for testing purposes.
This script creates a smaller, manageable dataset using random sampling and proper imputation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

def create_sample_dataset():
    """Create a sample dataset from the full Lightcast dataset using random sampling and proper imputation."""

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Paths
    raw_data_path = Path("data/raw/lightcast_job_postings.csv")
    sample_data_path = Path("data/processed/job_market_sample.csv")

    # Create output directory
    sample_data_path.parent.mkdir(parents=True, exist_ok=True)

    print("Reading full dataset...")

    # Read the full dataset in chunks to avoid memory issues
    chunk_size = 10000
    sample_rows = []
    total_rows = 0

    try:
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size):
            total_rows += len(chunk)
            print(f"Processed {total_rows:,} rows...")

            # Sample 1% of each chunk
            sample_size = max(1, len(chunk) // 100)
            sample_chunk = chunk.sample(n=sample_size, random_state=42)
            sample_rows.append(sample_chunk)

            # Stop after processing enough data (limit to ~50k rows total)
            if total_rows >= 500000:  # Process 500k rows from original
                break

    except Exception as e:
        print(f"Error reading dataset: {e}")
        return False

    print(f"Total rows processed: {total_rows:,}")

    # Combine all sample chunks
    print("Combining sample data...")
    sample_df = pd.concat(sample_rows, ignore_index=True)

    print(f"Sample dataset size: {len(sample_df):,} rows")

    # Clean up the sample data
    print("Cleaning sample data...")

    # Remove rows with missing critical columns
    critical_columns = ['TITLE', 'COMPANY_NAME', 'LOCATION']
    sample_df = sample_df.dropna(subset=critical_columns)

    # Convert salary columns to numeric
    salary_columns = ['SALARY_FROM', 'SALARY_TO']
    for col in salary_columns:
        if col in sample_df.columns:
            sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce')

    # Create SALARY_AVG column
    if 'SALARY_AVG' not in sample_df.columns:
        sample_df['SALARY_AVG'] = (sample_df['SALARY_FROM'] + sample_df['SALARY_TO']) / 2

    # Clean up imputation columns - ensure they exist and are clean
    imputation_columns = ['EDUCATION_LEVELS_NAME', 'CITY_NAME', 'NAICS2_NAME', 'STATE_NAME']
    for col in imputation_columns:
        if col in sample_df.columns:
            # Fill missing values with 'Unknown' for imputation
            sample_df[col] = sample_df[col].fillna('Unknown')
        else:
            print(f"Warning: Column {col} not found in dataset")
            sample_df[col] = 'Unknown'

    print("Performing median-based salary imputation...")

    # Create imputation groups based on the specified columns
    sample_df['imputation_group'] = sample_df[imputation_columns].apply(
        lambda x: '|'.join(x.astype(str)), axis=1
    )

    # Calculate median salary for each imputation group
    group_medians = sample_df.groupby('imputation_group')['SALARY_AVG'].median()

    # Impute missing salary values using group medians
    missing_salary_mask = (sample_df['SALARY_AVG'].isna()) | (sample_df['SALARY_AVG'] <= 0)
    print(f"Found {missing_salary_mask.sum():,} rows with missing salary data")

    for idx, row in sample_df[missing_salary_mask].iterrows():
        group = row['imputation_group']
        if group in group_medians and not pd.isna(group_medians[group]) and group_medians[group] > 0:
            # Use group median
            sample_df.loc[idx, 'SALARY_AVG'] = group_medians[group]
            # Estimate SALARY_FROM and SALARY_TO based on median
            median_salary = group_medians[group]
            sample_df.loc[idx, 'SALARY_FROM'] = median_salary * 0.8
            sample_df.loc[idx, 'SALARY_TO'] = median_salary * 1.2
        else:
            # Fallback to overall median if group median not available
            overall_median = sample_df['SALARY_AVG'].median()
            if not pd.isna(overall_median) and overall_median > 0:
                sample_df.loc[idx, 'SALARY_AVG'] = overall_median
                sample_df.loc[idx, 'SALARY_FROM'] = overall_median * 0.8
                sample_df.loc[idx, 'SALARY_TO'] = overall_median * 1.2
            else:
                # Final fallback - use a reasonable default
                sample_df.loc[idx, 'SALARY_AVG'] = 50000
                sample_df.loc[idx, 'SALARY_FROM'] = 40000
                sample_df.loc[idx, 'SALARY_TO'] = 60000

    # Remove the temporary imputation group column
    sample_df = sample_df.drop('imputation_group', axis=1)

    # Save the sample dataset
    print(f"Saving sample dataset to {sample_data_path}...")
    sample_df.to_csv(sample_data_path, index=False)

    print(f"Sample dataset created successfully!")
    print(f"Final size: {len(sample_df):,} rows, {len(sample_df.columns)} columns")
    print(f"Salary data coverage: {(sample_df['SALARY_AVG'] > 0).sum() / len(sample_df) * 100:.1f}%")

    # Show imputation statistics
    print(f"Imputation groups created: {len(group_medians):,}")
    print(f"Rows imputed with group medians: {missing_salary_mask.sum():,}")

    return True

if __name__ == "__main__":
    success = create_sample_dataset()
    if success:
        print("Sample dataset creation completed successfully!")
    else:
        print("Sample dataset creation failed!")
