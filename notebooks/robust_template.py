# Robust Notebook Template and Utilities
# 
# This template provides standardized patterns for safe data processing in Jupyter notebooks
# to prevent common issues like NumberFormatException and casting errors.

import sys
import os
sys.path.append('../src')

# Import core libraries with error handling
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, length, isnan, isnull, expr, lit, regexp_replace
    from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType
    SPARK_AVAILABLE = True
except ImportError:
    print("WARNING: PySpark not available - some functions will be limited")
    SPARK_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("WARNING: Pandas not available")
    PANDAS_AVAILABLE = False

# Try to import robust casting utilities
try:
    from utils.robust_casting import (RobustDataCaster, safe_cast_salary, 
                                     safe_string_filter, ultra_safe_group_count, 
                                     create_data_quality_report)
    ROBUST_UTILS_AVAILABLE = True
    print("Robust casting utilities loaded")
except ImportError:
    print("WARNING: Robust casting utilities not available - using inline functions")
    ROBUST_UTILS_AVAILABLE = False

# Define inline robust functions if utilities not available
def safe_numeric_cast_inline(df, column_name, target_type='double', new_column_name=None):
    """Inline safe numeric casting function."""
    if not SPARK_AVAILABLE:
        raise RuntimeError("PySpark not available")
        
    if new_column_name is None:
        new_column_name = f"{column_name}_numeric"
    
    # Ultra-safe numeric pattern
    numeric_pattern = r'^-?[0-9]+\.?[0-9]*$'
    
    return df.withColumn(
        new_column_name,
        when(
            (col(column_name).isNotNull()) &
            (length(col(column_name)) > 0) &
            (~col(column_name).isin(['', 'null', 'NULL', 'None', 'NaN', 'nan'])) &
            (col(column_name).rlike(numeric_pattern)),
            col(column_name).cast(target_type)
        ).otherwise(None)
    )

def safe_filter_inline(df, column_name):
    """Inline safe filtering function."""
    if not SPARK_AVAILABLE:
        raise RuntimeError("PySpark not available")
        
    return df.filter(
        (col(column_name).isNotNull()) &
        (length(col(column_name)) > 0) &
        (col(column_name) != '') &
        (col(column_name) != 'null') &
        (col(column_name) != 'NULL')
    )

def safe_groupby_count_inline(df, group_col, min_count=1):
    """Inline safe group by count function."""
    if not SPARK_AVAILABLE:
        raise RuntimeError("PySpark not available")
        
    try:
        result = df.groupBy(group_col).count()
        if min_count > 1:
            try:
                result = result.filter(col("count") >= min_count)
            except Exception:
                print(f"   Filter failed, returning unfiltered results")
        return result
    except Exception as e:
        print(f"   Group by failed: {e}")
        return None

def create_health_report_inline(df, critical_columns=None):
    """Inline health report function."""
    if not SPARK_AVAILABLE:
        return {"error": "PySpark not available"}
        
    try:
        report = {
            'total_rows': df.count(),
            'total_columns': len(df.columns),
            'health_status': 'HEALTHY',
            'column_metrics': {}
        }
        
        if critical_columns:
            for col_name in critical_columns:
                if col_name in df.columns:
                    null_count = df.filter(col(col_name).isNull()).count()
                    completion_rate = ((report['total_rows'] - null_count) / report['total_rows']) * 100
                    report['column_metrics'][col_name] = {
                        'completion_rate': completion_rate,
                        'null_count': null_count
                    }
        
        return report
    except Exception as e:
        return {"error": str(e)}

# Universal safe casting function that works with either utility or inline
def safe_cast(df, column_name, target_type='double', new_column_name=None):
    """Universal safe casting function."""
    if ROBUST_UTILS_AVAILABLE:
        return RobustDataCaster.safe_numeric_cast(df, column_name, target_type, new_column_name)
    else:
        return safe_numeric_cast_inline(df, column_name, target_type, new_column_name)

def safe_filter(df, column_name):
    """Universal safe filtering function."""
    if ROBUST_UTILS_AVAILABLE:
        return safe_string_filter(df, column_name)
    else:
        return safe_filter_inline(df, column_name)

def safe_group_count(df, group_col, min_count=1):
    """Universal safe group count function."""
    if ROBUST_UTILS_AVAILABLE:
        return ultra_safe_group_count(df, group_col, min_count)
    else:
        return safe_groupby_count_inline(df, group_col, min_count)

def health_check(df, critical_columns=None):
    """Universal health check function."""
    if ROBUST_UTILS_AVAILABLE:
        return create_data_quality_report(df, critical_columns)
    else:
        return create_health_report_inline(df, critical_columns)

# Notebook execution patterns
def run_with_fallback(operation_name, primary_func, fallback_func=None, *args, **kwargs):
    """
    Execute a function with automatic fallback on failure.
    
    Args:
        operation_name: Description of the operation
        primary_func: Primary function to try
        fallback_func: Function to try if primary fails
        *args, **kwargs: Arguments for the functions
    
    Returns:
        Result of successful function or None
    """
    try:
        print(f"Attempting {operation_name}...")
        result = primary_func(*args, **kwargs)
        print(f"SUCCESS: {operation_name} completed successfully")
        return result
    except Exception as e:
        print(f"ERROR: {operation_name} failed: {e}")
        
        if fallback_func:
            try:
                print(f"Trying fallback for {operation_name}...")
                result = fallback_func(*args, **kwargs)
                print(f"SUCCESS: Fallback {operation_name} completed")
                return result
            except Exception as fallback_error:
                print(f"ERROR: Fallback {operation_name} also failed: {fallback_error}")
        
        return None

def notebook_section(title, func, *args, **kwargs):
    """
    Execute a notebook section with standardized error handling and reporting.
    
    Args:
        title: Section title
        func: Function to execute
        *args, **kwargs: Arguments for the function
        
    Returns:
        Result of function or None if failed
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    try:
        result = func(*args, **kwargs)
        print(f"\nSUCCESS: {title} completed successfully")
        return result
    except Exception as e:
        print(f"\nERROR: {title} failed: {e}")
        print(f"   Continuing to next section...")
        return None

# Data quality checks
def validate_required_columns(df, required_columns):
    """Check if all required columns exist in DataFrame."""
    if not SPARK_AVAILABLE:
        return False
        
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        return False
    print(f"SUCCESS: All required columns present: {required_columns}")
    return True

def quick_validation_check(df, test_columns=None):
    """
    Quick validation check for any DataFrame.
    
    Args:
        df: DataFrame to validate
        test_columns: Columns to test (default: common columns)
        
    Returns:
        Simple validation report
    """
    if not SPARK_AVAILABLE:
        print("Cannot validate - PySpark not available")
        return None
        
    if test_columns is None:
        # Try common column patterns
        all_cols = df.columns
        test_columns = []
        for pattern in ['salary', 'city', 'state', 'title', 'company']:
            matches = [col for col in all_cols if pattern.lower() in col.lower()]
            if matches:
                test_columns.append(matches[0])
    
    try:
        total_rows = df.count()
        print(f"Quick Validation Check:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Total columns: {len(df.columns)}")
        
        if test_columns:
            print(f"  Testing columns: {test_columns}")
            for col_name in test_columns:
                if col_name in df.columns:
                    null_count = df.filter(col(col_name).isNull()).count()
                    completion = ((total_rows - null_count) / total_rows) * 100
                    status = "Good" if completion >= 80 else "Fair" if completion >= 50 else "Poor"
                    print(f"    {col_name}: {completion:.1f}% complete - {status}")
        
        # Test safe casting on numeric-looking columns
        numeric_test_cols = [col for col in df.columns if any(word in col.lower() for word in ['salary', 'price', 'amount', 'count'])]
        if numeric_test_cols:
            test_col = numeric_test_cols[0]
            print(f"  Testing safe casting on: {test_col}")
            try:
                df_test = safe_cast(df, test_col, 'double', f'{test_col}_test')
                valid_count = df_test.filter(col(f'{test_col}_test').isNotNull()).count()
                cast_rate = (valid_count / total_rows) * 100 if total_rows > 0 else 0
                print(f"    Safe casting success rate: {cast_rate:.1f}%")
            except Exception as e:
                print(f"    Safe casting test failed: {e}")
        
        print("  Validation complete")
        return True
        
    except Exception as e:
        print(f"  Validation failed: {e}")
        return False

# Standard notebook patterns
STANDARD_PATTERNS = {
    'safe_salary_cast': """
# Safe salary casting pattern
df_with_salary = safe_cast(df, 'SALARY', 'double', 'salary_numeric')
valid_salaries = df_with_salary.filter(col('salary_numeric').isNotNull())
""",
    
    'safe_location_analysis': """
# Safe location analysis pattern  
location_stats = safe_group_count(df, 'CITY', min_count=10)
if location_stats:
    location_stats.orderBy(col("count").desc()).show(15, truncate=False)
""",
    
    'health_check_pattern': """
# Data health check pattern
health_report = health_check(df, ['SALARY', 'CITY', 'STATE'])
print(f"Health Status: {health_report.get('health_status', 'Unknown')}")
for col_name, metrics in health_report.get('column_metrics', {}).items():
    print(f"  {col_name}: {metrics['completion_rate']:.1f}% complete")
"""
}

print(f"\nROBUST NOTEBOOK TEMPLATE LOADED")
print(f"   Available functions: safe_cast, safe_filter, safe_group_count, health_check")
print(f"   Utility functions: run_with_fallback, notebook_section, validate_required_columns")
print(f"   PySpark available: {SPARK_AVAILABLE}")
print(f"   Robust utilities available: {ROBUST_UTILS_AVAILABLE}")
print(f"Template ready for notebook operations")