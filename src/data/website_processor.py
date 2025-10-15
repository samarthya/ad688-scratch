"""
Website Data Processor - Hybrid PySpark + Pandas Architecture

This module provides the data processing interface for the Quarto website.

Architecture (Optimized Hybrid):
    Raw CSV (13M rows)
    → PySpark ETL (heavy lifting)
    → PySpark MLlib (ML training)
    → Parquet (~30-50K rows) ← BOUNDARY
    → Pandas (visualization layer)
    → Plotly (interactive charts)

Rationale:
    - PySpark: Excels at processing millions of rows
    - Parquet: Efficient storage boundary (10x compression)
    - Pandas: Excellent for 30-50K rows, better Plotly integration
    - Result: Fast website (1-2 sec load), static deployment, no Spark cluster needed

Trade-off:
    Framework purity (not 100% PySpark) vs. pragmatic performance and deployment simplicity
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import logger (silent by default for Quarto)
from src.utils.logger import get_logger
logger = get_logger(level="WARNING")  # Only show warnings/errors in Quarto

def _generate_common_figures(df: Any) -> None:
    """
    Generate common figures used across multiple QMD pages.

    This function optimizes rendering by:
    1. Generating figures once when data loads
    2. Saving to figures/ directory for reuse
    3. Only regenerating if figures are missing or data is newer

    Args:
        df: Pandas DataFrame with processed data
    """
    from pathlib import Path
    import os

    figures_dir = project_root / "figures"
    parquet_path = project_root / "data/processed/job_market_processed.parquet"

    # Check if figures need to be regenerated
    figures_exist = figures_dir.exists() and len(list(figures_dir.glob("*.html"))) > 0

    if figures_exist and parquet_path.exists():
        # Check if figures are newer than data
        parquet_mtime = parquet_path.stat().st_mtime
        figures_mtime = max([f.stat().st_mtime for f in figures_dir.glob("*.html")], default=0)

        if figures_mtime > parquet_mtime:
            # Figures are up-to-date
            return

    # Generate figures
    logger.info("\n[FIGURES] Generating common visualizations...")
    logger.info(f" Output directory: {figures_dir}")

    try:
        from src.visualization.charts import SalaryVisualizer

        visualizer = SalaryVisualizer(df)

        # Generate key findings graphics
        key_findings = visualizer.create_key_findings_graphics(str(figures_dir))
        logger.info(f" Generated {len(key_findings)} key finding graphics")

        # Generate executive dashboard suite
        dashboard = visualizer.create_executive_dashboard_suite(str(figures_dir))
        logger.info(f" Generated {len(dashboard)} executive dashboard graphics")

        logger.info(f" Total figures saved: {len(key_findings) + len(dashboard)}")

    except Exception as e:
        logger.warning(f" Figure generation failed: {e}")
        # Don't fail the entire load if figure generation fails


def load_and_process_data() -> Tuple[Any, Dict[str, Any]]:
    """
    Load and process job market data using PySpark.

    This function:
    1. Checks if processed Parquet exists
    2. If yes: Load with Pandas (fast - 1-2 seconds)
    3. If no: Process raw CSV with PySpark (5-10 minutes, saves to Parquet)

    Returns:
        Tuple of (DataFrame, summary_dict)
        DataFrame is Pandas for compatibility with visualization code
    """
    import pandas as pd

    # Define paths
    raw_csv_path = project_root / "data/raw/lightcast_job_postings.csv"
    parquet_path = project_root / "data/processed/job_market_processed.parquet"

    # Check if processed data exists (FAST PATH)
    if parquet_path.exists():
        logger.info("Loading job market data...")
        logger.info(f" Loading processed Parquet (fast!)...")
        df = pd.read_parquet(parquet_path)
        logger.info(f" Loaded {len(df):,} records in 1-2 seconds")

        # Generate common figures if they don't exist (OPTIMIZATION)
        _generate_common_figures(df)

        summary = get_data_summary(df)
        return df, summary

    # No processed data - need to run PySpark ETL (SLOW PATH - only runs once!)
    if not raw_csv_path.exists():
        raise FileNotFoundError(
            "No data source found.\n\n"
            "Expected data location:\n"
            f" {raw_csv_path}\n\n"
            "Please ensure raw data file exists in the data/raw/ directory."
        )

    logger.info("Loading job market data...")
    logger.info(f" Processing raw data with PySpark (first time - may take 5-10 minutes)...")
    logger.info(f" Raw CSV: {raw_csv_path.name}")

    # Import PySpark processor
    from src.core.processor import JobMarketDataProcessor
    from src.config.settings import Settings

    # Initialize PySpark processor
    settings = Settings()
    processor = JobMarketDataProcessor(settings=settings)

    logger.info(f" [PySpark] Loading {raw_csv_path.name}...")

    # Process with PySpark (this does all the heavy lifting)
    # The processor automatically saves to Parquet in process_raw_data()
    try:
        # Update settings to point to our desired Parquet output location
        settings.processed_data_path = str(parquet_path)

        spark_df = processor.load_and_process_data(str(raw_csv_path))

        # The processor has already saved to Parquet via process_raw_data()
        # Get record count before stopping Spark
        record_count = spark_df.count()

        # Stop Spark session (don't convert to Pandas - too much memory!)
        processor.spark.stop()

        if parquet_path.exists():
            file_size = parquet_path.stat().st_size / (1024*1024)
            logger.info(f" Saved to {parquet_path.name} ({file_size:.1f} MB)")
            logger.info(f" Processed {record_count:,} records")
        else:
            logger.warning(f" Expected Parquet file not found at {parquet_path}")

        logger.info(f" Next run will load from Parquet instantly!")

        # Now load the Parquet with Pandas (efficient for the filtered/processed data)
        logger.info(f" [Pandas] Loading processed Parquet...")
        df = pd.read_parquet(parquet_path)

        # Generate common figures (OPTIMIZATION)
        _generate_common_figures(df)

        summary = get_data_summary(df)
        logger.info(f" Loaded {len(df):,} records for analysis")

        return df, summary

    except Exception as e:
        logger.error(f" PySpark processing failed: {e}")
        import traceback
        traceback.print_exc()
        # Try to stop Spark if it was started
        try:
            processor.spark.stop()
        except:
            pass
        raise


def decode_numeric_columns(df: Any) -> Any:
    """
    Decode numeric codes to text descriptions for remote_type and employment_type.

    This is a workaround because the processed data has numeric codes but code expects text.

    Args:
        df: Pandas DataFrame with numeric remote_type and employment_type columns

    Returns:
        DataFrame with decoded text columns
    """
    import pandas as pd

    df = df.copy()

    # Decode remote_type (0, 1, 2, 3 → text)
    if 'remote_type' in df.columns and df['remote_type'].dtype in ['float64', 'int64']:
        remote_mapping = {
            0.0: 'Not Specified',
            1.0: 'Remote',
            2.0: 'Not Remote',
            3.0: 'Hybrid Remote'
        }
        df['remote_type'] = df['remote_type'].map(remote_mapping).fillna('Not Specified')

    # Decode employment_type (1, 2, 3 → text)
    if 'employment_type' in df.columns and df['employment_type'].dtype in ['float64', 'int64']:
        employment_mapping = {
            1.0: 'Full-time (> 32 hours)',
            2.0: 'Part-time (< 32 hours)',
            3.0: 'Contract'
        }
        df['employment_type'] = df['employment_type'].map(employment_mapping).fillna('Not Specified')

    return df


def add_experience_level(df: Any) -> Any:
    """
    Add experience_level categorical column from min_years_experience.

    This is calculated on-the-fly from existing columns, not stored in ETL.

    Categories based on industry standards and data distribution:
    - Unknown: NULL values
    - Entry Level: 0-2 years (9.9K records, median $96K)
    - Mid Level: 3-5 years (20.8K records, median $115K)
    - Senior Level: 6-10 years (15.2K records, median $128K)
    - Leadership Level: 10+ years (3.5K records, median $126K)

    Args:
        df: Pandas DataFrame with min_years_experience column

    Returns:
        DataFrame with experience_level column added
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Create experience_level from min_years_experience
    # Bins: Entry (0-2), Mid (3-5), Senior (6-9), Executive (10+)
    # Use the mapped column name from column_mapping.py
    # MIN_YEARS_EXPERIENCE → experience_min
    exp_col = 'experience_min' if 'experience_min' in df.columns else 'min_years_experience'

    df['experience_level'] = pd.cut(
        df[exp_col],
        bins=[-np.inf, 2, 5, 9, np.inf],
        labels=['Entry Level', 'Mid Level', 'Senior Level', 'Executive Level'],
        include_lowest=True
    )

    # Handle NaN values
    df['experience_level'] = df['experience_level'].cat.add_categories(['Unknown'])
    df.loc[df[exp_col].isna(), 'experience_level'] = 'Unknown'

    return df


def compute_salary_avg(df: Any) -> Any:
    """
    Compute salary_avg from available salary columns.

    Priority:
    1. Use salary_single if available
    2. Compute average of salary_min and salary_max
    3. Use salary_min if only that's available

    Args:
        df: Pandas DataFrame with salary columns

    Returns:
        DataFrame with salary_avg column added/updated
    """
    import numpy as np

    # Initialize salary_avg with NaN
    df['salary_avg'] = np.nan

    # Priority 1: Use salary_single
    if 'salary_single' in df.columns:
        mask = df['salary_single'].notna()
        df.loc[mask, 'salary_avg'] = df.loc[mask, 'salary_single']

    # Priority 2: Compute from min/max
    if 'salary_min' in df.columns and 'salary_max' in df.columns:
        mask = df['salary_avg'].isna() & df['salary_min'].notna() & df['salary_max'].notna()
        df.loc[mask, 'salary_avg'] = (df.loc[mask, 'salary_min'] + df.loc[mask, 'salary_max']) / 2

    # Priority 3: Use salary_min if nothing else
    if 'salary_min' in df.columns:
        mask = df['salary_avg'].isna() & df['salary_min'].notna()
        df.loc[mask, 'salary_avg'] = df.loc[mask, 'salary_min']

    return df


def get_processed_dataframe() -> Any:
    """
    Get the processed dataframe for analysis.

    This is the main entry point for Quarto QMD files.
    Returns Pandas DataFrame with all derived columns already created in PySpark ETL.
    Note: All derived columns (salary_avg, experience_level, etc.) are created in PySpark ETL.
    """
    df, _ = load_and_process_data()
    df = decode_numeric_columns(df)  # Decode remote_type and employment_type to text
    # All derived columns (salary_avg, experience_level, experience_years, ai_related, remote_allowed)
    # are already created in PySpark ETL - no need to recompute
    return df


def get_website_data_summary() -> Dict[str, Any]:
    """Get summary statistics for the website."""
    # Load fresh data to ensure summary reflects computed salary_avg
    df = get_processed_dataframe()
    summary = get_data_summary(df)
    return summary


def get_data_summary(df: Any = None) -> Dict[str, Any]:
    """
    Get comprehensive data summary.

    Works with both Spark and Pandas DataFrames.
    """
    if df is None or len(df) == 0:
        return {
            'total_records': 0,
            'salary_coverage': 0.0,
            'unique_industries': 0,
            'unique_locations': 0,
            'unique_companies': 0,
            'salary_range': {'min': 0, 'max': 0, 'median': 0}
        }

    # Check if it's a Spark DataFrame
    if hasattr(df, 'toPandas'):
        df = df.toPandas()

    # Use processed salary column (lowercase after ETL)
    salary_col = 'salary_avg' if 'salary_avg' in df.columns else 'salary'

    # Use processed column names (lowercase after ETL)
    industry_col = 'industry' if 'industry' in df.columns else None
    location_col = 'city_name' if 'city_name' in df.columns else 'location'

    company_col = 'company' if 'company' in df.columns else None

    # Calculate summary statistics
    total_records = len(df)

    salary_coverage = 0.0
    salary_range = {'min': 0, 'max': 0, 'median': 0}

    if salary_col:
        # Ensure salary column is numeric
        import pandas as pd
        if not pd.api.types.is_numeric_dtype(df[salary_col]):
            # Try to convert to numeric
            df[salary_col] = pd.to_numeric(df[salary_col], errors='coerce')

        valid_salaries = df[salary_col].notna().sum()
        salary_coverage = (valid_salaries / total_records * 100) if total_records > 0 else 0.0

        salary_data = df[salary_col].dropna()
        if len(salary_data) > 0:
            try:
                salary_range = {
                    'min': float(salary_data.min()),
                    'max': float(salary_data.max()),
                    'median': float(salary_data.median())
                }
            except (ValueError, TypeError):
                # If conversion still fails, leave as zeros
                pass

    return {
        'total_records': total_records,
        'salary_coverage': salary_coverage,
        'unique_industries': df[industry_col].nunique() if industry_col else 0,
        'unique_locations': df[location_col].nunique() if location_col else 0,
        'unique_companies': df[company_col].nunique() if company_col else 0,
        'salary_range': salary_range
    }


# Additional website functions
def get_website_data() -> Any:
    """
    Get the processed dataframe for website display.
    Alias for get_processed_dataframe() for backward compatibility.
    """
    return get_processed_dataframe()


def get_analysis_results() -> Dict[str, Any]:
    """
    Get analysis results for the website.
    Returns summary statistics and key metrics.
    """
    df = get_processed_dataframe()
    summary = get_data_summary(df)

    # Add additional analysis results
    results = {
        'summary': summary,
        'data_loaded': True,
        'records': len(df),
        'columns': list(df.columns)
    }

    return results


def get_figure_paths() -> Dict[str, str]:
    """
    Get paths to generated figure files.
    Returns dictionary mapping figure names to their file paths.
    """
    from pathlib import Path

    figures_dir = project_root / "figures"

    # Return paths to common figures
    figure_paths = {}

    if figures_dir.exists():
        for fig_file in figures_dir.glob("*.html"):
            fig_name = fig_file.stem
            figure_paths[fig_name] = str(fig_file)

    return figure_paths


# Backward compatibility aliases
load_analysis_data = get_processed_dataframe
get_website_summary = get_website_data_summary
