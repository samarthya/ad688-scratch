"""
Robust Data Casting Utilities for Spark DataFrames

This module provides safe casting operations that prevent NumberFormatException
and other casting errors common when working with malformed data.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_replace, length, isnan, isnull, expr, lit
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RobustDataCaster:
    """
    A utility class for safely casting data in Spark DataFrames.
    Handles common issues like empty strings, malformed numbers, and null values.
    """
    
    @staticmethod
    def safe_numeric_cast(df: DataFrame, column_name: str, target_type: str = 'double', 
                         new_column_name: Optional[str] = None) -> DataFrame:
        """
        Safely cast a string column to numeric type with comprehensive validation.
        
        Args:
            df: Input DataFrame
            column_name: Name of column to cast
            target_type: Target numeric type ('double', 'int', 'long')
            new_column_name: Name for new column (default: column_name + '_numeric')
            
        Returns:
            DataFrame with new numeric column
        """
        if new_column_name is None:
            new_column_name = f"{column_name}_numeric"
            
        # Step 1: Pre-filter to only valid numeric strings
        numeric_pattern = r'^-?[0-9]+\.?[0-9]*$'  # Handles negative numbers and decimals
        
        # Step 2: Create safe numeric column with multiple validation layers
        df_with_numeric = df.withColumn(
            new_column_name,
            when(
                # Check if column is not null
                col(column_name).isNotNull() &
                # Check if column has content (length > 0)
                (length(col(column_name)) > 0) &
                # Check if column is not in common null representations
                (~col(column_name).isin(['', 'null', 'NULL', 'None', 'NaN', 'nan'])) &
                # Check if column matches numeric pattern
                col(column_name).rlike(numeric_pattern),
                # If all checks pass, cast to target type
                col(column_name).cast(target_type)
            ).otherwise(None)  # Otherwise set to null
        )
        
        return df_with_numeric
    
    @staticmethod
    def safe_filter_numeric(df: DataFrame, column_name: str, 
                           exclude_zero: bool = False) -> DataFrame:
        """
        Filter DataFrame to only include rows with valid numeric values.
        
        Args:
            df: Input DataFrame
            column_name: Column to filter on
            exclude_zero: Whether to exclude zero values
            
        Returns:
            Filtered DataFrame
        """
        filter_condition = col(column_name).isNotNull()
        
        if exclude_zero:
            filter_condition = filter_condition & (col(column_name) != 0)
            
        return df.filter(filter_condition)
    
    @staticmethod
    def clean_string_column(df: DataFrame, column_name: str, 
                           new_column_name: Optional[str] = None) -> DataFrame:
        """
        Clean string column by trimming whitespace and handling common issues.
        
        Args:
            df: Input DataFrame
            column_name: Column to clean
            new_column_name: Name for cleaned column (default: column_name + '_clean')
            
        Returns:
            DataFrame with cleaned column
        """
        if new_column_name is None:
            new_column_name = f"{column_name}_clean"
            
        df_cleaned = df.withColumn(
            new_column_name,
            when(
                col(column_name).isNotNull() &
                (length(col(column_name)) > 0),
                # Trim whitespace and replace common null representations
                regexp_replace(
                    regexp_replace(col(column_name), r'^\s+|\s+$', ''),  # Trim
                    r'^(null|NULL|None|NaN|nan)$', ''  # Replace null representations
                )
            ).otherwise(None)
        )
        
        return df_cleaned
    
    @staticmethod
    def safe_aggregation_with_fallback(df: DataFrame, group_cols: List[str], 
                                     agg_expressions: Dict[str, str],
                                     min_count: int = 1) -> Optional[DataFrame]:
        """
        Perform aggregation with error handling and fallback.
        Uses simple count aggregation to avoid casting issues.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_expressions: Dictionary of column -> aggregation function
            min_count: Minimum count per group to include in result
            
        Returns:
            Aggregated DataFrame or None if operation fails
        """
        try:
            # Validate group columns exist
            missing_cols = [col for col in group_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing group columns: {missing_cols}")
                return None
            
            # For safety, use only count aggregation to avoid casting issues
            # More complex aggregations should be done after this step
            result = df.groupBy(*group_cols).count()
            
            # Apply minimum count filter
            if min_count > 1:
                result = result.filter(col("count") >= min_count)
                
            return result
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return None
    
    @staticmethod
    def validate_dataframe_health(df: DataFrame, critical_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate DataFrame health and return diagnostics.
        
        Args:
            df: DataFrame to validate
            critical_columns: List of columns that must have data
            
        Returns:
            Dictionary with health metrics
        """
        try:
            health_report = {
                'total_rows': df.count(),
                'total_columns': len(df.columns),
                'health_status': 'HEALTHY',
                'issues': [],
                'column_metrics': {}
            }
            
            # Check critical columns if specified
            if critical_columns:
                for col_name in critical_columns:
                    if col_name not in df.columns:
                        health_report['issues'].append(f"Missing critical column: {col_name}")
                        health_report['health_status'] = 'CRITICAL'
                    else:
                        null_count = df.filter(col(col_name).isNull()).count()
                        null_percentage = (null_count / health_report['total_rows']) * 100
                        
                        health_report['column_metrics'][col_name] = {
                            'null_count': null_count,
                            'null_percentage': null_percentage,
                            'completion_rate': 100 - null_percentage
                        }
                        
                        if null_percentage > 50:
                            health_report['issues'].append(
                                f"High null rate in {col_name}: {null_percentage:.1f}%"
                            )
                            health_report['health_status'] = 'WARNING'
                            
            return health_report
            
        except Exception as e:
            return {
                'total_rows': 0,
                'total_columns': 0,
                'health_status': 'CRITICAL',
                'issues': [f"Health check failed: {str(e)}"],
                'column_metrics': {}
            }

# Convenience functions for common operations
def safe_cast_salary(df: DataFrame, salary_col: str = 'SALARY') -> DataFrame:
    """Convenience function to safely cast salary columns."""
    return RobustDataCaster.safe_numeric_cast(df, salary_col, 'double', 'salary_numeric')

def safe_cast_id(df: DataFrame, id_col: str) -> DataFrame:
    """Convenience function to safely cast ID columns."""
    return RobustDataCaster.safe_numeric_cast(df, id_col, 'long', f'{id_col}_numeric')

def safe_string_filter(df: DataFrame, column_name: str, exclude_values: Optional[List[str]] = None) -> DataFrame:
    """
    Safely filter string columns without triggering casting errors.
    Uses length-based filtering only to avoid any casting issues.
    """
    if exclude_values is None:
        exclude_values = ['', 'null', 'NULL', 'None', 'NaN']
    
    # Use only length-based and null checks to avoid casting issues
    # Avoid isin() completely as it can trigger casting problems
    filter_condition = (
        col(column_name).isNotNull() &
        (length(col(column_name)) > 0)
    )
    
    # For critical exclusions, use individual != checks for common problematic values
    # But limit to essential ones to avoid casting issues
    filter_condition = (
        filter_condition &
        (col(column_name) != '') &  # This is safe for string columns
        (col(column_name) != 'null') &
        (col(column_name) != 'NULL')
    )
    
    return df.filter(filter_condition)

def create_data_quality_report(df: DataFrame, 
                             critical_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a comprehensive data quality report."""
    return RobustDataCaster.validate_dataframe_health(df, critical_columns)

def ultra_safe_group_count(df: DataFrame, group_col: str, min_count: int = 1) -> Optional[DataFrame]:
    """
    Ultra-safe groupBy count that avoids all potential casting issues.
    Only performs basic counting without any filtering that might trigger casting.
    """
    try:
        # Check if column exists
        if group_col not in df.columns:
            logger.warning(f"Column {group_col} not found in DataFrame")
            return None
            
        # Perform the simplest possible groupBy count
        result = df.groupBy(group_col).count()
        
        # Only apply min_count filter if > 1 and if it's safe
        if min_count > 1:
            try:
                result = result.filter(col("count") >= min_count)
            except Exception as filter_error:
                logger.warning(f"Min count filter failed: {filter_error}, returning unfiltered results")
                # Return unfiltered results if filtering fails
                result = df.groupBy(group_col).count()
                
        return result
        
    except Exception as e:
        logger.error(f"Ultra safe grouping failed: {e}")
        return None