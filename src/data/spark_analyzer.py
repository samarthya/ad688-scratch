"""
PySpark-based SQL Analysis Engine for Job Market Data

This module provides SQL-based analysis using PySpark DataFrames and SQL queries
to replace the pandas-based SalaryVisualizer with proper big data processing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, count, avg, sum as spark_sum, median, desc, asc,
    min as spark_min, max as spark_max, stddev, percentile_approx,
    regexp_replace, trim, upper, lower, lit, current_timestamp,
    expr, year, month, dayofmonth, to_date, split
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, BooleanType, DateType
)
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkJobAnalyzer:
    """
    PySpark-based job market analysis engine using SQL queries and DataFrames.
    
    This class processes the full Lightcast dataset using PySpark for scalable analysis.
    """
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """Initialize with Spark session."""
        if spark_session is None:
            self.spark = SparkSession.builder \
                .appName("JobMarketAnalysis") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark_session
        
        self.job_data: Optional[DataFrame] = None
        logger.info(f"SparkJobAnalyzer initialized with Spark {self.spark.version}")
    
    def get_df(self)-> Optional[DataFrame]:
        return self.job_data
    
    def load_full_dataset(self, data_path: str = "../../data/processed/job_market_processed.parquet", 
                         force_raw: bool = False) -> DataFrame:
        """
        Load the full dataset with flexible data source selection.
        
        Args:
            data_path: Path to the processed parquet data
            force_raw: If True, bypasses processed data and loads from raw Lightcast CSV
            
        Returns:
            Spark DataFrame with full dataset
            
        Raises:
            FileNotFoundError: If data sources don't exist
            Exception: If data loading or validation fails
        """
        if force_raw:
            logger.info("🔄 FORCE RAW MODE: Bypassing processed data, loading from raw source")
            return self._load_raw_data()
        
        # Normal loading hierarchy: Parquet -> CSV -> Raw
        # Try to load from parquet (preferred for performance)
        if Path(data_path).exists():
            logger.info(f"Loading full dataset from {data_path}")
            try:
                self.job_data = self.spark.read.parquet(data_path)
                logger.info(f"Successfully loaded Parquet data: {self.job_data.count():,} records")
            except Exception as e:
                logger.error(f"Failed to load Parquet data from {data_path}: {e}")
                raise Exception(f"Corrupted Parquet data at {data_path}: {e}")
        else:
            # Fallback to processed CSV if parquet doesn't exist
            csv_path = "../../data/processed/clean_job_data.csv"
            if Path(csv_path).exists():
                logger.info(f"Parquet not found, loading from processed CSV: {csv_path}")
                try:
                    self.job_data = self.spark.read \
                        .option("header", "true") \
                        .option("inferSchema", "true") \
                        .csv(csv_path)
                    logger.info(f"Successfully loaded processed CSV data: {self.job_data.count():,} records")
                except Exception as e:
                    logger.error(f"Failed to load processed CSV data from {csv_path}: {e}")
                    raise Exception(f"Corrupted processed CSV data at {csv_path}: {e}")
            else:
                # Final fallback to original raw Lightcast data
                raw_data_path = "../../data/raw/lightcast_job_postings.csv"
                if Path(raw_data_path).exists():
                    logger.warning(f"No processed data found, loading from original raw data: {raw_data_path}")
                    logger.warning("Note: Raw data may require processing for optimal analysis")
                    try:
                        self.job_data = self.spark.read \
                            .option("header", "true") \
                            .option("inferSchema", "true") \
                            .option("multiline", "true") \
                            .option("escape", "\"") \
                            .csv(raw_data_path)
                        logger.info(f"Successfully loaded raw Lightcast data: {self.job_data.count():,} records")
                    except Exception as e:
                        logger.error(f"Failed to load raw data from {raw_data_path}: {e}")
                        raise Exception(f"Corrupted raw data at {raw_data_path}: {e}")
                else:
                    # No data sources available - this is a critical error
                    error_msg = f"No data sources found. Missing: {data_path}, {csv_path}, and {raw_data_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
        
        # Validate loaded data
        self._validate_dataset(self.job_data)
        
        # Register as temporary view for SQL queries
        self.job_data.createOrReplaceTempView("job_postings")
        
        record_count = self.job_data.count()
        logger.info(f"Dataset loaded and validated: {record_count:,} job postings")
        
        return self.job_data
    
    def _load_raw_data(self, raw_data_path: str = "../../data/raw/lightcast_job_postings.csv") -> DataFrame:
        """
        Force load data from raw Lightcast CSV file for development/validation.
        
        Args:
            raw_data_path: Path to raw Lightcast CSV file
            
        Returns:
            Spark DataFrame with raw data
            
        Raises:
            FileNotFoundError: If raw data file doesn't exist
            Exception: If loading fails
        """
        logger.warning("⚠️  DEVELOPER MODE: Loading raw data - processed optimizations bypassed")
        
        if not Path(raw_data_path).exists():
            error_msg = f"Raw data file not found: {raw_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading raw Lightcast data from: {raw_data_path}")
        
        try:
            # Load raw CSV with appropriate options for Lightcast format
            self.job_data = self.spark.read \
                .csv(raw_data_path, 
                     header=True, 
                     inferSchema=True,
                     multiLine=True,
                     escape="\"")
            
            record_count = self.job_data.count()
            col_count = len(self.job_data.columns)
            
            logger.info(f"✅ Raw data loaded: {record_count:,} records, {col_count} columns")
            logger.warning("⚠️  Note: Raw data may have different column names and require processing")
            
            # Use flexible validation for raw data
            self._validate_raw_dataset(self.job_data)
            
            # Register as temporary view
            self.job_data.createOrReplaceTempView("job_postings")
            
            return self.job_data
            
        except Exception as e:
            error_msg = f"Failed to load raw data from {raw_data_path}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _validate_raw_dataset(self, df: DataFrame) -> None:
        """
        Validate raw dataset with more flexible requirements.
        
        Args:
            df: Spark DataFrame to validate
            
        Raises:
            Exception: If critical issues are found
        """
        logger.info("Validating raw dataset (flexible validation)")
        
        # Check if dataset is empty
        record_count = df.count()
        if record_count == 0:
            raise Exception("Dataset is empty - no records found")
        
        # Check for basic Lightcast columns (more flexible)
        basic_columns = ["TITLE", "COMPANY", "LOCATION"]
        missing_basic = [col for col in basic_columns if col not in df.columns]
        
        if len(missing_basic) == len(basic_columns):
            # If ALL basic columns are missing, check for processed column names
            processed_columns = ["title", "company", "location"]
            missing_processed = [col for col in processed_columns if col not in df.columns]
            
            if len(missing_processed) == len(processed_columns):
                raise Exception(f"No recognizable job data columns found. Expected either {basic_columns} or {processed_columns}")
            else:
                logger.info("✅ Detected processed column schema")
        else:
            logger.info("✅ Detected raw Lightcast schema")
        
        # Flexible salary validation
        salary_columns = [col for col in df.columns if 'SALARY' in col.upper() or 'salary' in col.lower()]
        
        if salary_columns:
            logger.info(f"📊 Found salary columns: {salary_columns}")
        else:
            logger.warning("⚠️  No salary columns detected - analysis may be limited")
        
        logger.info(f"Raw dataset validation completed: {record_count:,} records")
    
    def _validate_dataset(self, df: DataFrame) -> None:
        """
        Validate the loaded dataset for data quality and consistency.
        
        Args:
            df: Spark DataFrame to validate
            
        Raises:
            Exception: If critical data quality issues are found
        """
        logger.info("Validating dataset quality and consistency")
        
        # Check if dataset is empty
        record_count = df.count()
        if record_count == 0:
            raise Exception("Dataset is empty - no records found")
        
        # Check for required columns
        required_columns = ["salary_avg_imputed", "industry", "title", "location"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Check for reasonable data distribution
        null_salary_count = df.filter(col("salary_avg_imputed").isNull()).count()
        null_salary_pct = (null_salary_count / record_count) * 100
        
        if null_salary_pct > 90:
            logger.warning(f"High percentage of missing salary data: {null_salary_pct:.1f}%")
        
        # Validate salary ranges
        invalid_salary_count = df.filter(
            (col("salary_avg_imputed") < 10000) | (col("salary_avg_imputed") > 1000000)
        ).count()
        
        if invalid_salary_count > (record_count * 0.1):  # More than 10% invalid salaries
            logger.warning(f"High percentage of invalid salary ranges: {invalid_salary_count} records")
        
        logger.info(f"Dataset validation completed: {record_count:,} records, {null_salary_pct:.1f}% missing salaries")
    
    def get_industry_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get comprehensive industry salary analysis using SQL.
        
        Args:
            top_n: Number of top industries to return
            
        Returns:
            Pandas DataFrame with industry analysis
        """
        query = f"""
        WITH salary_calcs AS (
            SELECT 
                industry,
                salary_avg_imputed as salary_avg,
                title,
                remote_allowed_clean as remote_allowed
            FROM job_postings
            WHERE salary_avg_imputed IS NOT NULL
        ),
        
        industry_stats AS (
            SELECT 
                industry,
                COUNT(*) as job_count,
                ROUND(percentile_approx(salary_avg, 0.5), 0) as median_salary,
                ROUND(AVG(salary_avg), 0) as mean_salary,
                ROUND(STDDEV(salary_avg), 0) as std_salary
            FROM salary_calcs
            GROUP BY industry
        ),
        
        ai_analysis AS (
            SELECT 
                industry,
                COUNT(*) as total_jobs,
                SUM(CASE WHEN LOWER(title) RLIKE '(ai|machine learning|data scientist|ml engineer|artificial intelligence)' 
                    THEN 1 ELSE 0 END) as ai_jobs
            FROM salary_calcs
            GROUP BY industry
        ),
        
        remote_analysis AS (
            SELECT 
                industry,
                COUNT(*) as total_jobs,
                SUM(CASE WHEN remote_allowed = 'Remote' THEN 1 ELSE 0 END) as remote_jobs
            FROM salary_calcs
            GROUP BY industry
        )
        
        SELECT 
            i.industry,
            i.median_salary,
            i.job_count,
            CASE 
                WHEN a.total_jobs > 0 THEN CONCAT(ROUND((a.ai_jobs * 100.0 / a.total_jobs), 0), '%')
                ELSE 'N/A'
            END as ai_premium,
            CASE 
                WHEN r.total_jobs > 0 THEN CONCAT(ROUND((r.remote_jobs * 100.0 / r.total_jobs), 0), '%')
                ELSE 'N/A'
            END as remote_pct
        FROM industry_stats i
        LEFT JOIN ai_analysis a ON i.industry = a.industry
        LEFT JOIN remote_analysis r ON i.industry = r.industry
        ORDER BY i.median_salary DESC
        LIMIT {top_n}
        """
        
        result_df = self.spark.sql(query)
        
        # Convert to pandas for display
        pandas_df = result_df.toPandas()
        pandas_df.columns = ['Industry', 'Median Salary', 'Job Count', 'AI Premium', 'Remote %']
        
        return pandas_df
    
    def get_experience_analysis(self) -> pd.DataFrame:
        """Get experience level salary analysis using SQL."""
        query = """
        SELECT 
            experience_level_clean as experience_level,
            COUNT(*) as job_count,
            ROUND(percentile_approx(salary_avg_imputed, 0.5), 0) as median_salary,
            ROUND(AVG(salary_avg_imputed), 0) as mean_salary
        FROM job_postings
        WHERE salary_avg_imputed IS NOT NULL
        GROUP BY experience_level_clean
        ORDER BY median_salary
        """
        
        result_df = self.spark.sql(query)
        pandas_df = result_df.toPandas()
        pandas_df.columns = ['Experience Level', 'Job Count', 'Median Salary', 'Mean Salary']
        
        return pandas_df
    
    def get_geographic_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Get geographic salary analysis using SQL."""
        query = f"""
        SELECT 
            location,
            COUNT(*) as job_count,
            ROUND(percentile_approx(salary_avg_imputed, 0.5), 0) as median_salary,
            ROUND(AVG(salary_avg_imputed), 0) as mean_salary
        FROM job_postings
        WHERE salary_avg_imputed IS NOT NULL
        GROUP BY location
        HAVING job_count >= 10
        ORDER BY median_salary DESC
        LIMIT {top_n}
        """
        
        result_df = self.spark.sql(query)
        pandas_df = result_df.toPandas()
        pandas_df.columns = ['Location', 'Job Count', 'Median Salary', 'Mean Salary']
        
        return pandas_df
    
    def get_overall_statistics(self) -> Dict:
        """Get overall dataset statistics using SQL."""
        query = """
        SELECT 
            COUNT(*) as total_jobs,
            ROUND(percentile_approx(salary_avg_imputed, 0.5), 0) as median_salary,
            ROUND(AVG(salary_avg_imputed), 0) as mean_salary,
            ROUND(STDDEV(salary_avg_imputed), 0) as std_salary,
            ROUND(MIN(salary_avg_imputed), 0) as min_salary,
            ROUND(MAX(salary_avg_imputed), 0) as max_salary,
            ROUND(percentile_approx(salary_avg_imputed, 0.25), 0) as salary_25th,
            ROUND(percentile_approx(salary_avg_imputed, 0.75), 0) as salary_75th
        FROM job_postings
        WHERE salary_avg_imputed IS NOT NULL
        """
        
        result = self.spark.sql(query).collect()[0]
        
        return {
            'total_jobs': int(result['total_jobs']),
            'median_salary': int(result['median_salary']),
            'mean_salary': int(result['mean_salary']),
            'std_salary': int(result['std_salary']),
            'min_salary': int(result['min_salary']),
            'max_salary': int(result['max_salary']),
            'salary_25th': int(result['salary_25th']),
            'salary_75th': int(result['salary_75th'])
        }
    
    def create_relational_view(self, table_name: str) -> None:
        """
        Create a relational view with normalized schema for advanced analysis.
        
        Args:
            table_name: Name for the SQL view
        """
        query = f"""
        CREATE OR REPLACE TEMPORARY VIEW {table_name} AS
        SELECT 
            job_id,
            title,
            company,
            location,
            SPLIT(location, ', ')[0] as city,
            SPLIT(location, ', ')[1] as state,
            industry,
            experience_level,
            salary_min,
            salary_max,
            (salary_min + salary_max) / 2 as salary_avg,
            remote_allowed,
            posted_date,
            CASE 
                WHEN LOWER(title) RLIKE '(ai|machine learning|data scientist|ml engineer|artificial intelligence)' 
                THEN 1 ELSE 0 
            END as is_ai_role,
            CASE 
                WHEN industry = 'Technology' THEN 1 ELSE 0 
            END as is_tech_industry,
            YEAR(TO_DATE(posted_date, 'yyyy-MM-dd')) as posted_year,
            MONTH(TO_DATE(posted_date, 'yyyy-MM-dd')) as posted_month
        FROM job_postings
        WHERE salary_min IS NOT NULL AND salary_max IS NOT NULL
        """
        
        self.spark.sql(query)
        logger.info(f"Created relational view: {table_name}")
    
    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query and return results as pandas DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            result_df = self.spark.sql(query)
            return result_df.toPandas()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def get_skills_analysis(self, top_n: int = 7) -> pd.DataFrame:
        """
        Get skills analysis with salary premiums using SQL.
        
        Args:
            top_n: Number of top skills to return
            
        Returns:
            Pandas DataFrame with skills analysis
        """
        query = f"""
        WITH skill_extraction AS (
            SELECT 
                title,
                required_skills,
                salary_avg_imputed,
                CASE 
                    WHEN LOWER(title) RLIKE '(machine learning|ml engineer|ai|artificial intelligence)' THEN 'Machine Learning'
                    WHEN LOWER(title) RLIKE '(data scientist|data science|analytics|analyst)' THEN 'Data Science'
                    WHEN LOWER(title) RLIKE '(cloud|aws|azure|gcp|kubernetes|docker)' THEN 'Cloud Architecture'
                    WHEN LOWER(title) RLIKE '(security|cybersecurity|cyber|infosec)' THEN 'Cybersecurity'
                    WHEN LOWER(title) RLIKE '(devops|dev ops|sre|site reliability)' THEN 'DevOps'
                    WHEN LOWER(title) RLIKE '(mobile|ios|android|react native|flutter)' THEN 'Mobile Development'
                    WHEN LOWER(title) RLIKE '(ui|ux|design|designer|user experience)' THEN 'UI/UX Design'
                    WHEN LOWER(title) RLIKE '(full stack|fullstack|full-stack)' THEN 'Full Stack'
                    WHEN LOWER(title) RLIKE '(backend|back-end|back end|api)' THEN 'Backend Development'
                    WHEN LOWER(title) RLIKE '(frontend|front-end|front end|react|vue|angular)' THEN 'Frontend Development'
                    ELSE 'General Tech'
                END as skill_category
            FROM job_postings
            WHERE salary_avg_imputed IS NOT NULL
        ),
        
        skill_stats AS (
            SELECT 
                skill_category,
                COUNT(*) as job_count,
                ROUND(percentile_approx(salary_avg_imputed, 0.5), 0) as median_salary,
                ROUND(AVG(salary_avg_imputed), 0) as mean_salary
            FROM skill_extraction
            WHERE skill_category != 'General Tech'
            GROUP BY skill_category
        ),
        
        baseline AS (
            SELECT ROUND(AVG(salary_avg_imputed), 0) as baseline_salary
            FROM skill_extraction
            WHERE skill_category = 'General Tech'
        )
        
        SELECT 
            s.skill_category,
            s.median_salary,
            s.job_count,
            ROUND(((s.median_salary - b.baseline_salary) / b.baseline_salary * 100), 0) as premium_pct
        FROM skill_stats s
        CROSS JOIN baseline b
        WHERE s.job_count >= 5
        ORDER BY s.median_salary DESC
        LIMIT {top_n}
        """
        
        result_df = self.spark.sql(query)
        pandas_df = result_df.toPandas()
        pandas_df.columns = ['Skill Category', 'Median Salary', 'Jobs Available', 'Premium %']
        
        return pandas_df
    
    def stop(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


# Convenience functions for backward compatibility
def create_spark_analyzer(data_path: Optional[str] = None, force_raw: bool = False) -> SparkJobAnalyzer:
    """
    Create and return a SparkJobAnalyzer instance with loaded data.
    
    Args:
        data_path: Optional path to data file. If None, uses default Parquet path.
        force_raw: If True, forces loading from raw Lightcast CSV for validation
        
    Returns:
        SparkJobAnalyzer instance with loaded data
        
    Raises:
        Exception: If data loading fails
    """
    analyzer = SparkJobAnalyzer()
    if data_path:
        analyzer.load_full_dataset(data_path, force_raw=force_raw)
    else:
        analyzer.load_full_dataset(force_raw=force_raw)
    return analyzer


def create_raw_analyzer() -> SparkJobAnalyzer:
    """
    Create SparkJobAnalyzer that forces loading from raw Lightcast data.
    Useful for development, validation, and debugging.
    
    Returns:
        SparkJobAnalyzer instance with raw data loaded
        
    Raises:
        Exception: If raw data loading fails
    """
    return create_spark_analyzer(force_raw=True)


if __name__ == "__main__":
    # Example usage
    analyzer = create_spark_analyzer()
    
    # Run some example analyses
    print("Industry Analysis:")
    industry_df = analyzer.get_industry_analysis()
    print(industry_df)
    
    print("\nExperience Analysis:")
    exp_df = analyzer.get_experience_analysis()
    print(exp_df)
    
    print("\nOverall Statistics:")
    stats = analyzer.get_overall_statistics()
    for key, value in stats.items():
        print(f"{key}: {value:,}")
    
    analyzer.stop()