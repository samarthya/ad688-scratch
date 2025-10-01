"""
Automatic data processing pipeline for Quarto website.

This module provides intelligent data loading that automatically processes
and cleans raw data when clean versions are not available.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple

# Add src to path for imports
sys.path.append('src')
sys.path.append('src/data')

from data_cleaner import JobMarketDataCleaner


class AutoDataProcessor:
    """
    Automatic data processor that intelligently loads and processes data.

    This class provides a single interface for data loading that:
    1. Tries to load clean data first (fastest)
    2. Falls back to sample data if clean data unavailable
    3. Automatically processes raw data if needed
    4. Handles all data cleaning and validation
    """

    def __init__(self):
        """Initialize the auto processor."""
        self.cleaner = JobMarketDataCleaner()
        self.data_cache = {}

    def load_data_for_analysis(self, analysis_type: str = "general") -> pd.DataFrame:
        """
        Load data for analysis with automatic processing.

        Args:
            analysis_type: Type of analysis (for logging purposes)

        Returns:
            Clean pandas DataFrame ready for analysis
        """
        print(f"=== Loading Data for {analysis_type.upper()} Analysis ===")

        # Try different data sources in order of preference
        data_sources = [
            ("Clean Sample Data", "data/processed/job_market_clean_sample.csv"),
            ("Original Sample Data", "data/processed/job_market_sample.csv"),
            ("Raw Data (Auto-Process)", "data/raw/lightcast_job_postings.csv")
        ]

        for source_name, data_path in data_sources:
            try:
                print(f"Trying {source_name}...")
                df = self._load_and_process_data(data_path, source_name)

                if df is not None and len(df) > 0:
                    print(f"✅ Successfully loaded {len(df):,} records from {source_name}")
                    print(f"   Data quality: {df['SALARY_AVG'].notna().sum()/len(df)*100:.1f}% salary coverage")
                    return df
                else:
                    print(f"❌ {source_name} returned empty dataset")

            except Exception as e:
                print(f"❌ Failed to load {source_name}: {str(e)[:100]}...")
                continue

        # If all sources fail, raise an error
        raise RuntimeError("Unable to load any data source. Please check data files and permissions.")

    def _load_and_process_data(self, data_path: str, source_name: str) -> Optional[pd.DataFrame]:
        """Load and process data from a specific source."""
        data_path = Path(data_path)

        if not data_path.exists():
            print(f"   File not found: {data_path}")
            return None

        # Load the data
        df = pd.read_csv(data_path, low_memory=False)
        print(f"   Loaded {len(df):,} raw records")

        # Process based on source type
        if "clean_sample" in str(data_path):
            # Clean data is already processed
            print(f"   Using pre-cleaned data")
            return self._validate_clean_data(df)

        elif "sample" in str(data_path):
            # Sample data needs basic cleaning
            print(f"   Applying basic cleaning to sample data")
            return self._clean_sample_data(df)

        else:
            # Raw data needs full processing
            print(f"   Applying comprehensive cleaning to raw data")
            return self._process_raw_data(df)

    def _validate_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that clean data has required columns."""
        required_columns = ['SALARY_AVG', 'TITLE_NAME', 'COMPANY_NAME']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"   Warning: Clean data missing columns: {missing_columns}")
            # Try to fix common issues
            if 'salary_avg' in df.columns and 'SALARY_AVG' not in df.columns:
                df['SALARY_AVG'] = df['salary_avg']
            if 'title' in df.columns and 'TITLE_NAME' not in df.columns:
                df['TITLE_NAME'] = df['title']
            if 'company_name' in df.columns and 'COMPANY_NAME' not in df.columns:
                df['COMPANY_NAME'] = df['company_name']

        return df

    def _clean_sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning to sample data."""
        # Ensure SALARY_AVG exists
        if 'SALARY_AVG' not in df.columns:
            if 'SALARY_FROM' in df.columns and 'SALARY_TO' in df.columns:
                df['SALARY_AVG'] = (df['SALARY_FROM'] + df['SALARY_TO']) / 2
            else:
                raise ValueError("Cannot create SALARY_AVG - missing salary columns")

        # Clean salary data
        df['SALARY_AVG'] = pd.to_numeric(df['SALARY_AVG'], errors='coerce')
        df = df.dropna(subset=['SALARY_AVG'])

        # Remove unrealistic salary values
        df = df[(df['SALARY_AVG'] >= 20000) & (df['SALARY_AVG'] <= 500000)]

        return df

    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive cleaning to raw data."""
        # Take a sample for processing (to avoid memory issues)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
            print(f"   Sampled {len(df):,} records for processing")

        # Apply comprehensive cleaning
        df_clean, stats = self.cleaner.clean_dataset(df)

        print(f"   Cleaning completed: {stats['final_shape'][0]:,} records")
        print(f"   Values imputed: {stats['values_imputed']:,}")

        return df_clean

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of the loaded data."""
        summary = {
            'total_records': len(df),
            'columns': len(df.columns),
            'salary_coverage': df['SALARY_AVG'].notna().sum() / len(df) * 100 if 'SALARY_AVG' in df.columns else 0,
            'unique_companies': df['COMPANY_NAME'].nunique() if 'COMPANY_NAME' in df.columns else 0,
            'unique_titles': df['TITLE'].nunique() if 'TITLE' in df.columns else 0,
        }

        if 'location_readable' in df.columns:
            summary['unique_locations'] = df['location_readable'].nunique()
        elif 'region' in df.columns:
            summary['unique_regions'] = df['region'].nunique()

        if 'education_level' in df.columns:
            summary['education_levels'] = df['education_level'].nunique()

        if 'experience_level' in df.columns:
            summary['experience_levels'] = df['experience_level'].nunique()

        return summary


# Global instance for easy access
auto_processor = AutoDataProcessor()


def load_analysis_data(analysis_type: str = "general") -> pd.DataFrame:
    """
    Convenience function to load data for analysis.

    Args:
        analysis_type: Type of analysis (for logging)

    Returns:
        Clean pandas DataFrame ready for analysis
    """
    return auto_processor.load_data_for_analysis(analysis_type)


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of the loaded data.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with data summary statistics
    """
    return auto_processor.get_data_summary(df)
