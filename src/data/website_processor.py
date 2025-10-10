"""
Website Data Processor - PySpark-Based ETL

This module provides the data processing interface for the Quarto website.
It uses PySpark for heavy ETL processing (13M rows) and saves to Parquet.
The website then loads the processed Parquet with Pandas for fast analysis.

Architecture:
    Raw CSV (13M rows) → PySpark ETL → Parquet (~30-50K rows) → Pandas Analysis
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

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
    print("\n[FIGURES] Generating common visualizations...")
    print(f" Output directory: {figures_dir}")

    try:
        from src.visualization.charts import SalaryVisualizer

        visualizer = SalaryVisualizer(df)

        # Generate key findings graphics
        key_findings = visualizer.create_key_findings_graphics(str(figures_dir))
        print(f" Generated {len(key_findings)} key finding graphics")

        # Generate executive dashboard suite
        dashboard = visualizer.create_executive_dashboard_suite(str(figures_dir))
        print(f" Generated {len(dashboard)} executive dashboard graphics")

        print(f" Total figures saved: {len(key_findings) + len(dashboard)}")

    except Exception as e:
        print(f" [WARNING] Figure generation failed: {e}")
        # Don't fail the entire load if figure generation fails
        import traceback
        traceback.print_exc()


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
        print("Loading job market data...")
        print(f" Loading processed Parquet (fast!)...")
        df = pd.read_parquet(parquet_path)
        print(f" Loaded {len(df):,} records in 1-2 seconds")

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

    print("Loading job market data...")
    print(f" Processing raw data with PySpark (first time - may take 5-10 minutes)...")
    print(f" Raw CSV: {raw_csv_path.name}")

    # Import PySpark processor
    from src.core.processor import JobMarketDataProcessor
    from src.config.settings import Settings

    # Initialize PySpark processor
    settings = Settings()
    processor = JobMarketDataProcessor(settings=settings)

    print(f" [PySpark] Loading {raw_csv_path.name}...")

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
            print(f" Saved to {parquet_path.name} ({file_size:.1f} MB)")
            print(f" Processed {record_count:,} records")
        else:
            print(f" [WARNING] Expected Parquet file not found at {parquet_path}")

        print(f" Next run will load from Parquet instantly!")

        # Now load the Parquet with Pandas (efficient for the filtered/processed data)
        print(f" [Pandas] Loading processed Parquet...")
        df = pd.read_parquet(parquet_path)

        # Generate common figures (OPTIMIZATION)
        _generate_common_figures(df)

        summary = get_data_summary(df)
        print(f" Loaded {len(df):,} records for analysis")

        return df, summary

    except Exception as e:
        print(f" [ERROR] PySpark processing failed: {e}")
        import traceback
        traceback.print_exc()
        # Try to stop Spark if it was started
        try:
            processor.spark.stop()
        except:
            pass
        raise


def get_processed_dataframe() -> Any:
    """
    Get the processed dataframe for analysis.

    This is the main entry point for Quarto QMD files.
    Returns Pandas DataFrame for visualization compatibility.
    """
    df, _ = load_and_process_data()
    return df


def get_website_data_summary() -> Dict[str, Any]:
    """Get summary statistics for the website."""
    _, summary = load_and_process_data()
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

    # Find salary column
    salary_col = None
    for col in ['salary_avg', 'salary', 'SALARY_AVG']:
        if col in df.columns:
            salary_col = col
            break

    # Find other columns (handle both snake_case and UPPERCASE)
    industry_col = None
    for col in ['industry', 'INDUSTRY', 'NAICS2_NAME', 'naics2_name']:
        if col in df.columns:
            industry_col = col
            break

    location_col = None
    for col in ['city_name', 'location', 'CITY_NAME', 'LOCATION']:
        if col in df.columns:
            location_col = col
            break

    company_col = None
    for col in ['company', 'company_name', 'COMPANY', 'COMPANY_NAME']:
        if col in df.columns:
            company_col = col
            break

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
