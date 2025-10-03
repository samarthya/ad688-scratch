#!/usr/bin/env python3
"""
ONE-TIME DATA PROCESSING

Process raw data and save as standardized Parquet.
Run this ONCE, then all analysis uses the parquet.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 60)
    print("CREATING PROCESSED DATA (ONE-TIME PROCESSING)")
    print("=" * 60)

    # Load raw data
    raw_path = Path("data/raw/lightcast_job_postings.csv")
    print(f"\n1. Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"   Loaded {len(df):,} records with {len(df.columns)} columns")

    # Standardize column names: UPPERCASE → snake_case
    print(f"\n2. Standardizing column names...")
    df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    print(f"   All columns converted to snake_case")

    # Process salary
    print(f"\n3. Processing salary data...")

    # Compute salary_avg from salary_from and salary_to
    if 'salary_from' in df.columns and 'salary_to' in df.columns:
        df['salary_from'] = pd.to_numeric(df['salary_from'], errors='coerce')
        df['salary_to'] = pd.to_numeric(df['salary_to'], errors='coerce')

        # Compute average where both exist
        both_exist = df['salary_from'].notna() & df['salary_to'].notna()
        df['salary_avg'] = np.nan
        df.loc[both_exist, 'salary_avg'] = (df.loc[both_exist, 'salary_from'] +
                                             df.loc[both_exist, 'salary_to']) / 2
        print(f"   Computed salary_avg for {both_exist.sum():,} records")

    # Validate salary
    print(f"\n4. Validating salary data...")
    df['salary_avg'] = pd.to_numeric(df['salary_avg'], errors='coerce')

    # Remove records with invalid salary
    valid_salary = (
        df['salary_avg'].notna() &
        (df['salary_avg'] > 0) &
        (df['salary_avg'] >= 20000) &
        (df['salary_avg'] <= 500000)
    )

    invalid_count = (~valid_salary).sum()
    print(f"   Removing {invalid_count:,} records with invalid salary")
    df = df[valid_salary].copy()

    # Basic stats
    print(f"\n5. Final dataset statistics:")
    print(f"   Total records: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Salary range: ${df['salary_avg'].min():,.0f} - ${df['salary_avg'].max():,.0f}")
    print(f"   Median salary: ${df['salary_avg'].median():,.0f}")

    # Save as Parquet
    output_path = Path("data/processed/job_market_processed.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n6. Saving to {output_path}...")
    df.to_parquet(output_path, index=False)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved {len(df):,} records ({file_size:.1f} MB)")

    print("\n" + "=" * 60)
    print("DONE! Processed data ready for use.")
    print("All analysis will now load this parquet directly.")
    print("=" * 60)

if __name__ == "__main__":
    main()

