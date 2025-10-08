#!/usr/bin/env python3
"""
Generate Processed Data for Quarto Website

This script ensures processed Parquet data exists before Quarto rendering.
Run this ONCE before `quarto preview` or `quarto render` for much faster performance.

Usage:
    python scripts/generate_processed_data.py [--force]

Options:
    --force    Force regeneration even if Parquet exists
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description="Generate processed job market data")
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if processed data exists')
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING PROCESSED DATA FOR QUARTO WEBSITE")
    print("=" * 70)

    # Check if processed data already exists
    parquet_path = project_root / "data/processed/job_market_processed.parquet"

    if parquet_path.exists() and not args.force:
        file_size = parquet_path.stat().st_size / (1024*1024)
        print(f"\n[OK] Processed data already exists ({file_size:.1f} MB)")
        print(f"    Location: {parquet_path}")
        print(f"\n    Quarto will load this instantly (1-2 seconds)")
        print(f"    Use --force to regenerate\n")
        return 0

    if args.force:
        print("\n[REGENERATE] Force flag set - regenerating processed data...")
    else:
        print("\n[GENERATE] No processed data found - generating for first time...")

    # Import and run processing
    try:
        from src.data.website_processor import load_and_process_data

        print("\n[START] Starting data processing pipeline...")
        print("        This may take 5-10 minutes for initial processing")
        print("        (Subsequent runs will be instant!)\n")

        df, summary = load_and_process_data()

        print("\n" + "=" * 70)
        print("[SUCCESS] Processed data generated")
        print("=" * 70)
        print(f"   Records: {summary['total_records']:,}")
        print(f"   Salary coverage: {summary['salary_coverage']:.1f}%")
        print(f"   Location: {parquet_path}")

        if parquet_path.exists():
            file_size = parquet_path.stat().st_size / (1024*1024)
            print(f"   File size: {file_size:.1f} MB")

        print("\n[READY] You can now run 'quarto preview' - it will be FAST!")
        print("=" * 70)

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("[ERROR] Failed to generate processed data")
        print("=" * 70)
        print(f"   {str(e)}")
        print("\n[TIP] Check that data/raw/lightcast_job_postings.csv exists")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())

