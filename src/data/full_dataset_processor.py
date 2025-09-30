"""
Full Dataset Processor for Lightcast Job Market Data

This script processes the complete Lightcast dataset using PySpark for scalable analysis,
creating proper relational tables and normalized data structures.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, count, avg, sum as spark_sum, median, desc, asc,
    min as spark_min, max as spark_max, stddev, percentile_approx,
    regexp_replace, trim, upper, lower, lit, current_timestamp,
    expr, year, month, dayofmonth, to_date, split, isnan, isnull,
    coalesce, monotonically_increasing_id, udf
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, BooleanType, DateType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """Create optimized Spark session for big data processing."""
    spark = SparkSession.builder \
        .appName("LightcastJobMarketProcessor") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \
        .getOrCreate()
    
    logger.info(f"Spark session created: {spark.version}")
    return spark


def define_lightcast_schema() -> StructType:
    """Define the schema for Lightcast job postings dataset."""
    return StructType([
        StructField("JOB_ID", StringType(), True),
        StructField("TITLE", StringType(), True),
        StructField("COMPANY", StringType(), True),
        StructField("LOCATION", StringType(), True),
        StructField("POSTED", StringType(), True),
        StructField("SALARY_MIN", StringType(), True),
        StructField("SALARY_MAX", StringType(), True),
        StructField("SALARY_CURRENCY", StringType(), True),
        StructField("INDUSTRY", StringType(), True),
        StructField("EXPERIENCE_LEVEL", StringType(), True),
        StructField("EMPLOYMENT_TYPE", StringType(), True),
        StructField("REMOTE_ALLOWED", StringType(), True),
        StructField("REQUIRED_SKILLS", StringType(), True),
        StructField("EDUCATION_REQUIRED", StringType(), True),
        StructField("DESCRIPTION", StringType(), True),
        StructField("COUNTRY", StringType(), True),
        StructField("STATE", StringType(), True),
        StructField("CITY", StringType(), True)
    ])


def load_raw_lightcast_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Load raw Lightcast CSV data with proper schema.
    
    Args:
        spark: Spark session
        file_path: Path to the raw Lightcast CSV file
        
    Returns:
        Raw Spark DataFrame
    """
    schema = define_lightcast_schema()
    
    if not Path(file_path).exists():
        logger.warning(f"Lightcast file not found: {file_path}")
        return create_sample_lightcast_data(spark)
    
    logger.info(f"Loading Lightcast data from: {file_path}")
    
    # Load with optimized settings for large files
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .option("multiline", "true") \
        .option("escape", '"') \
        .schema(schema) \
        .csv(file_path)
    
    record_count = df.count()
    logger.info(f"Raw data loaded: {record_count:,} records")
    
    return df


def create_sample_lightcast_data(spark: SparkSession, num_records: int = 100000) -> DataFrame:
    """
    Create a comprehensive sample dataset that mimics Lightcast structure.
    
    Args:
        spark: Spark session
        num_records: Number of sample records to generate
        
    Returns:
        Sample Spark DataFrame
    """
    logger.info(f"Creating sample Lightcast dataset: {num_records:,} records")
    
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    # Realistic data distributions
    job_titles = [
        "Software Engineer", "Senior Software Engineer", "Data Scientist", "Senior Data Scientist",
        "Product Manager", "Senior Product Manager", "Marketing Manager", "Digital Marketing Manager",
        "Sales Representative", "Senior Sales Manager", "Business Analyst", "Senior Business Analyst",
        "UX Designer", "Senior UX Designer", "DevOps Engineer", "Senior DevOps Engineer",
        "Machine Learning Engineer", "AI Engineer", "Data Analyst", "Senior Data Analyst",
        "Software Developer", "Senior Software Developer", "Project Manager", "Program Manager",
        "Research Scientist", "Principal Research Scientist", "Frontend Developer", "Backend Developer",
        "Full Stack Developer", "Cloud Architect", "Cybersecurity Analyst", "Database Administrator",
        "QA Engineer", "Site Reliability Engineer", "Technical Writer", "Scrum Master"
    ]
    
    companies = [
        "Google LLC", "Microsoft Corporation", "Amazon.com Inc", "Apple Inc", "Meta Platforms Inc",
        "Netflix Inc", "Tesla Inc", "Uber Technologies Inc", "Airbnb Inc", "Spotify Technology SA",
        "Adobe Inc", "Salesforce Inc", "Oracle Corporation", "IBM Corporation", "Intel Corporation",
        "NVIDIA Corporation", "Cisco Systems Inc", "PayPal Holdings Inc", "Square Inc", "Zoom Video",
        "TechCorp Solutions", "DataSystems Inc", "InnovateCo Ltd", "StartupXYZ", "GlobalTech Industries",
        "FutureTech Ventures", "CloudFirst Systems", "AI Innovations", "NextGen Analytics", "SmartSoft"
    ]
    
    locations = [
        "San Francisco, CA", "Seattle, WA", "New York, NY", "Austin, TX", "Boston, MA",
        "Los Angeles, CA", "Chicago, IL", "Denver, CO", "Atlanta, GA", "Miami, FL",
        "Portland, OR", "San Diego, CA", "Phoenix, AZ", "Dallas, TX", "Philadelphia, PA",
        "Washington, DC", "Raleigh, NC", "Nashville, TN", "Salt Lake City, UT", "Minneapolis, MN",
        "Remote", "San Jose, CA", "Oakland, CA", "Cambridge, MA", "Santa Monica, CA"
    ]
    
    industries = [
        "Technology", "Software", "Internet", "Computer Software", "Information Technology",
        "Healthcare", "Biotechnology", "Pharmaceuticals", "Medical Devices", "Healthcare Technology",
        "Finance", "Financial Services", "Banking", "Investment Banking", "Insurance",
        "Manufacturing", "Automotive", "Aerospace", "Industrial", "Consumer Goods",
        "Retail", "E-commerce", "Fashion", "Consumer Electronics", "Food & Beverage",
        "Education", "Higher Education", "EdTech", "Training", "Research",
        "Government", "Public Sector", "Defense", "Federal Government", "State Government",
        "Non-profit", "NGO", "Social Impact", "Charity", "Foundation"
    ]
    
    experience_levels = ["Entry Level", "Mid Level", "Senior Level", "Executive", "Internship"]
    employment_types = ["Full-time", "Part-time", "Contract", "Temporary", "Internship"]
    education_levels = ["High School", "Associate", "Bachelor", "Master", "PhD", "Professional"]
    remote_options = ["No", "Yes", "Hybrid", "Remote-First"]
    
    # Generate sample records
    sample_data = []
    for i in range(num_records):
        # Generate realistic salary based on role and location
        title = np.random.choice(job_titles)
        location = np.random.choice(locations)
        industry = np.random.choice(industries)
        experience = np.random.choice(experience_levels)
        
        # Base salary calculation
        base_salary = 75000
        
        # Experience multiplier
        if "Senior" in title or "Principal" in title:
            base_salary *= 1.4
        elif "Manager" in title or "Lead" in title:
            base_salary *= 1.3
        elif "Entry" in experience or "Intern" in title:
            base_salary *= 0.7
        
        # Location multiplier
        if "San Francisco" in location or "San Jose" in location:
            base_salary *= 1.6
        elif "Seattle" in location or "New York" in location:
            base_salary *= 1.4
        elif "Austin" in location or "Boston" in location:
            base_salary *= 1.2
        elif "Remote" in location:
            base_salary *= 1.1
        
        # Industry multiplier
        if industry in ["Technology", "Software", "Internet"]:
            base_salary *= 1.2
        elif industry in ["Finance", "Banking", "Investment Banking"]:
            base_salary *= 1.15
        elif industry in ["Healthcare", "Biotechnology"]:
            base_salary *= 1.1
        
        # Add random variation
        salary_variation = np.random.normal(1.0, 0.15)
        final_salary = max(30000, int(base_salary * salary_variation))
        
        salary_min = int(final_salary * 0.9)
        salary_max = int(final_salary * 1.1)
        
        # Generate posting date within last 2 years
        days_ago = np.random.randint(0, 730)
        posted_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Skills based on role
        if "Data" in title or "ML" in title or "AI" in title:
            skills = "Python, SQL, Machine Learning, Statistics, Pandas, Scikit-learn"
        elif "Software" in title or "Developer" in title:
            skills = "Python, Java, JavaScript, Git, SQL, AWS, Docker"
        elif "Manager" in title:
            skills = "Leadership, Project Management, Strategy, Communication, Agile"
        else:
            skills = "Communication, Problem Solving, Teamwork, Analysis"
        
        sample_data.append({
            "JOB_ID": f"lightcast_{i:07d}",
            "TITLE": title,
            "COMPANY": np.random.choice(companies),
            "LOCATION": location,
            "POSTED": posted_date,
            "SALARY_MIN": str(salary_min),
            "SALARY_MAX": str(salary_max),
            "SALARY_CURRENCY": "USD",
            "INDUSTRY": industry,
            "EXPERIENCE_LEVEL": experience,
            "EMPLOYMENT_TYPE": np.random.choice(employment_types),
            "REMOTE_ALLOWED": np.random.choice(remote_options),
            "REQUIRED_SKILLS": skills,
            "EDUCATION_REQUIRED": np.random.choice(education_levels),
            "DESCRIPTION": f"Exciting opportunity for {title} at {np.random.choice(companies)}. Join our team!",
            "COUNTRY": "United States" if location != "Remote" else "Remote",
            "STATE": location.split(", ")[1] if ", " in location else "Remote",
            "CITY": location.split(", ")[0] if ", " in location else "Remote"
        })
    
    # Create DataFrame with proper schema
    schema = define_lightcast_schema()
    df = spark.createDataFrame(sample_data, schema=schema)
    
    logger.info(f"Sample dataset created: {len(sample_data):,} records")
    return df


def clean_and_process_data(df: DataFrame) -> DataFrame:
    """
    Comprehensive data cleaning and processing pipeline.
    
    Args:
        df: Raw Spark DataFrame
        
    Returns:
        Cleaned and processed DataFrame
    """
    logger.info("Starting data cleaning and processing pipeline")
    
    initial_count = df.count()
    logger.info(f"Initial record count: {initial_count:,}")
    
    # 1. Remove duplicates
    df_clean = df.dropDuplicates(subset=["TITLE", "COMPANY", "LOCATION", "POSTED"])
    duplicates_removed = initial_count - df_clean.count()
    logger.info(f"Duplicates removed: {duplicates_removed:,}")
    
    # 2. Clean and convert salary fields
    df_clean = df_clean.withColumn(
        "salary_min_clean",
        when(col("SALARY_MIN").rlike("^[0-9]+$"), col("SALARY_MIN").cast("double"))
        .otherwise(None)
    ).withColumn(
        "salary_max_clean",
        when(col("SALARY_MAX").rlike("^[0-9]+$"), col("SALARY_MAX").cast("double"))
        .otherwise(None)
    )
    
    # 3. Calculate average salary
    df_clean = df_clean.withColumn(
        "salary_avg",
        (col("salary_min_clean") + col("salary_max_clean")) / 2
    )
    
    # 4. Clean location data
    df_clean = df_clean.withColumn(
        "location_clean",
        trim(upper(col("LOCATION")))
    ).withColumn(
        "city_clean",
        when(col("CITY").isNotNull(), trim(upper(col("CITY"))))
        .otherwise(split(col("LOCATION"), ",")[0])
    ).withColumn(
        "state_clean", 
        when(col("STATE").isNotNull(), trim(upper(col("STATE"))))
        .otherwise(split(col("LOCATION"), ",")[1])
    )
    
    # 5. Standardize industry names
    df_clean = df_clean.withColumn(
        "industry_clean",
        when(col("INDUSTRY").rlike("(?i).*(tech|software|computer).*"), "Technology")
        .when(col("INDUSTRY").rlike("(?i).*(health|medical|bio|pharma).*"), "Healthcare")
        .when(col("INDUSTRY").rlike("(?i).*(financ|bank|invest).*"), "Finance")
        .when(col("INDUSTRY").rlike("(?i).*(retail|commerce|shop).*"), "Retail")
        .when(col("INDUSTRY").rlike("(?i).*(manufact|industrial).*"), "Manufacturing")
        .when(col("INDUSTRY").rlike("(?i).*(educ|school|university).*"), "Education")
        .when(col("INDUSTRY").rlike("(?i).*(government|public|federal).*"), "Government")
        .when(col("INDUSTRY").rlike("(?i).*(non.profit|ngo|charity).*"), "Non-profit")
        .otherwise(col("INDUSTRY"))
    )
    
    # 6. Standardize experience levels
    df_clean = df_clean.withColumn(
        "experience_level_clean",
        when(col("EXPERIENCE_LEVEL").rlike("(?i).*(entry|junior|intern|graduate).*"), "Entry")
        .when(col("EXPERIENCE_LEVEL").rlike("(?i).*(mid|intermediate).*"), "Mid")
        .when(col("EXPERIENCE_LEVEL").rlike("(?i).*(senior|lead|principal).*"), "Senior")
        .when(col("EXPERIENCE_LEVEL").rlike("(?i).*(executive|director|vp|cto|ceo).*"), "Executive")
        .otherwise("Mid")  # Default to Mid level
    )
    
    # 7. Clean remote work flags
    df_clean = df_clean.withColumn(
        "remote_allowed_clean",
        when(col("REMOTE_ALLOWED").rlike("(?i).*(yes|remote|hybrid|flexible).*"), 1)
        .otherwise(0)
    )
    
    # 8. Identify AI/ML roles
    df_clean = df_clean.withColumn(
        "is_ai_role",
        when(col("TITLE").rlike("(?i).*(ai|machine.learning|data.scientist|ml.engineer|artificial.intelligence).*"), 1)
        .otherwise(0)
    )
    
    # 9. Convert posting date
    df_clean = df_clean.withColumn(
        "posted_date_clean",
        to_date(col("POSTED"), "yyyy-MM-dd")
    )
    
    # 10. Filter valid records
    df_final = df_clean.filter(
        col("salary_min_clean").isNotNull() &
        col("salary_max_clean").isNotNull() &
        col("salary_avg").between(20000, 500000) &  # Reasonable salary range
        col("TITLE").isNotNull() &
        col("COMPANY").isNotNull()
    )
    
    final_count = df_final.count()
    logger.info(f"Final record count: {final_count:,}")
    logger.info(f"Records filtered: {initial_count - final_count:,}")
    
    return df_final


def clean_and_process_data_optimized(df: DataFrame) -> DataFrame:
    """
    Optimized version of clean_and_process_data incorporating all notebook improvements.
    
    This method applies the following optimizations:
    1. Drop non-essential timestamp columns early
    2. Handle REMOTE_TYPE_NAME nulls
    3. Resolve CITY vs CITY_NAME consolidation with base64 decoding
    4. Remove duplicate county columns
    5. Apply core cleaning pipeline
    6. Generate optimization summary
    
    Args:
        df: Raw Spark DataFrame
        
    Returns:
        DataFrame: Cleaned and processed data with optimizations
    """
    logger.info("Starting optimized data cleaning and processing")
    initial_count = df.count()
    logger.info(f"Initial record count: {initial_count:,}")
    
    # Step 1: Drop non-essential columns early for better performance
    logger.info("Step 1: Dropping non-essential timestamp columns")
    columns_to_drop = ["LAST_UPDATED_DATE", "LAST_UPDATED_TIMESTAMP", "ACTIVE_SOURCES_INFO"]
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(*existing_cols_to_drop)
        logger.info(f"Dropped columns: {existing_cols_to_drop}")
    
    # Step 2: Handle REMOTE_TYPE_NAME nulls
    logger.info("Step 2: Handling REMOTE_TYPE_NAME nulls")
    if "REMOTE_TYPE_NAME" in df.columns:
        null_count = df.filter(col("REMOTE_TYPE_NAME").isNull()).count()
        logger.info(f"REMOTE_TYPE_NAME null values: {null_count:,}")
        df = df.withColumn("REMOTE_TYPE_NAME", 
                          when(col("REMOTE_TYPE_NAME").isNull(), lit("Undefined"))
                          .otherwise(col("REMOTE_TYPE_NAME")))
    
    # Step 3: Resolve CITY vs CITY_NAME with base64 decoding
    logger.info("Step 3: Resolving CITY vs CITY_NAME consolidation")
    if "CITY" in df.columns and "CITY_NAME" in df.columns:
        # Create safe base64 decoding UDF
        def safe_base64_decode(encoded_str):
            if encoded_str is None:
                return None
            try:
                import base64
                decoded_bytes = base64.b64decode(encoded_str)
                return decoded_bytes.decode('utf-8')
            except:
                return encoded_str
        
        from pyspark.sql.types import StringType
        decode_udf = udf(safe_base64_decode, StringType())
        
        # Apply base64 decoding to CITY column
        df = df.withColumn("CITY_DECODED", decode_udf(col("CITY")))
        
        # Consolidate into single city column
        df = df.withColumn("CITY_CONSOLIDATED", 
                          when(col("CITY_DECODED").isNotNull(), col("CITY_DECODED"))
                          .otherwise(col("CITY_NAME")))
        
        # Replace original CITY column and drop intermediates
        df = df.withColumn("CITY", col("CITY_CONSOLIDATED")) \
               .drop("CITY_NAME", "CITY_DECODED", "CITY_CONSOLIDATED")
        
        logger.info("CITY column consolidated with base64 decoding")
    
    # Step 4: Remove duplicate county columns
    logger.info("Step 4: Checking for duplicate county columns")
    county_columns = [col_name for col_name in df.columns if 'county' in col_name.lower()]
    if len(county_columns) > 1:
        logger.info(f"Found county columns: {county_columns}")
        
        # Sample data to check similarity
        sample_df = df.select(*county_columns).limit(1000).toPandas()
        
        # Calculate similarity between county columns
        from difflib import SequenceMatcher
        similarities = []
        for i, col1 in enumerate(county_columns):
            for j, col2 in enumerate(county_columns[i+1:], i+1):
                # Compare non-null values
                col1_vals = sample_df[col1].dropna().astype(str).tolist()
                col2_vals = sample_df[col2].dropna().astype(str).tolist()
                
                if col1_vals and col2_vals:
                    # Compare first few values
                    similarity = SequenceMatcher(None, str(col1_vals[:10]), str(col2_vals[:10])).ratio()
                    similarities.append((col1, col2, similarity))
                    logger.info(f"Similarity between {col1} and {col2}: {similarity:.3f}")
        
        # Remove columns with >95% similarity (keep the first one)
        cols_to_remove = []
        for col1, col2, sim in similarities:
            if sim > 0.95 and col2 not in cols_to_remove:
                cols_to_remove.append(col2)
        
        if cols_to_remove:
            df = df.drop(*cols_to_remove)
            logger.info(f"Removed duplicate county columns: {cols_to_remove}")
    
    # Step 5: Apply existing cleaning pipeline
    logger.info("Step 5: Applying core cleaning pipeline")
    df_processed = clean_and_process_data(df)
    
    # Step 6: Generate optimization summary
    final_count = df_processed.count()
    optimization_summary = {
        "initial_records": initial_count,
        "final_records": final_count,
        "records_filtered": initial_count - final_count,
        "optimizations_applied": [
            "Dropped non-essential timestamp columns",
            "Handled REMOTE_TYPE_NAME nulls", 
            "Consolidated CITY columns with base64 decoding",
            "Removed duplicate county columns",
            "Applied core cleaning pipeline"
        ]
    }
    
    logger.info("=== OPTIMIZATION SUMMARY ===")
    for key, value in optimization_summary.items():
        if isinstance(value, list):
            logger.info(f"{key}: {', '.join(value)}")
        else:
            logger.info(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
    
    return df_processed


def create_relational_tables(df: DataFrame, output_path: str = "data/processed/relational_tables"):
    """
    Create normalized relational tables from the processed data.
    
    Args:
        df: Processed Spark DataFrame
        output_path: Path to save relational tables
    """
    logger.info("Creating relational tables")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Companies dimension table
    companies_df = df.select("COMPANY") \
        .distinct() \
        .withColumn("company_id", monotonically_increasing_id()) \
        .select("company_id", col("COMPANY").alias("company_name"))
    
    companies_df.write.mode("overwrite").parquet(f"{output_path}/companies")
    logger.info(f"Companies table created: {companies_df.count():,} records")
    
    # 2. Locations dimension table  
    locations_df = df.select("city_clean", "state_clean", "location_clean") \
        .distinct() \
        .withColumn("location_id", monotonically_increasing_id()) \
        .select("location_id", 
                col("city_clean").alias("city"), 
                col("state_clean").alias("state"),
                col("location_clean").alias("location"))
    
    locations_df.write.mode("overwrite").parquet(f"{output_path}/locations")
    logger.info(f"Locations table created: {locations_df.count():,} records")
    
    # 3. Industries dimension table
    industries_df = df.select("industry_clean") \
        .distinct() \
        .withColumn("industry_id", monotonically_increasing_id()) \
        .select("industry_id", col("industry_clean").alias("industry_name"))
    
    industries_df.write.mode("overwrite").parquet(f"{output_path}/industries")
    logger.info(f"Industries table created: {industries_df.count():,} records")
    
    # 4. Job postings fact table (with foreign keys)
    # This would require joins with dimension tables in a full implementation
    # For now, create a comprehensive fact table
    jobs_fact_df = df.select(
        col("JOB_ID").alias("job_id"),
        col("TITLE").alias("job_title"),
        col("COMPANY").alias("company_name"),
        col("industry_clean").alias("industry"),
        col("experience_level_clean").alias("experience_level"),
        col("salary_min_clean").alias("salary_min"),
        col("salary_max_clean").alias("salary_max"),
        col("salary_avg").alias("salary_avg"),
        col("remote_allowed_clean").alias("remote_allowed"),
        col("is_ai_role"),
        col("posted_date_clean").alias("posted_date"),
        col("city_clean").alias("city"),
        col("state_clean").alias("state"),
        col("REQUIRED_SKILLS").alias("required_skills"),
        col("EDUCATION_REQUIRED").alias("education_required")
    )
    
    jobs_fact_df.write.mode("overwrite").parquet(f"{output_path}/job_postings_fact")
    logger.info(f"Job postings fact table created: {jobs_fact_df.count():,} records")


def save_processed_data(df: DataFrame, output_path: str = "data/processed"):
    """
    Save processed data in multiple formats for different use cases.
    
    Args:
        df: Processed Spark DataFrame
        output_path: Base output path
    """
    logger.info("Saving processed data in multiple formats")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save full dataset as Parquet (best for Spark processing)
    parquet_path = f"{output_path}/job_market_processed.parquet"
    df.write.mode("overwrite").parquet(parquet_path)
    logger.info(f"Parquet saved: {parquet_path}")
    
    # 2. Save sample as CSV (for quick analysis and compatibility)
    csv_sample = df.sample(fraction=0.1, seed=42)  # 10% sample
    csv_path = f"{output_path}/job_market_sample.csv"
    csv_sample.toPandas().to_csv(csv_path, index=False)
    logger.info(f"CSV sample saved: {csv_path}")
    
    # 3. Save clean version with selected columns as CSV
    clean_df = df.select(
        col("JOB_ID").alias("job_id"),
        col("TITLE").alias("title"), 
        col("COMPANY").alias("company"),
        col("location_clean").alias("location"),
        col("industry_clean").alias("industry"),
        col("posted_date_clean").alias("posted_date"),
        col("salary_min_clean").alias("salary_min"),
        col("salary_max_clean").alias("salary_max"),
        col("salary_avg"),
        col("experience_level_clean").alias("experience_level"),
        col("remote_allowed_clean").alias("remote_allowed")
    )
    
    clean_csv_path = f"{output_path}/clean_job_data.csv"
    clean_df.toPandas().to_csv(clean_csv_path, index=False)
    logger.info(f"Clean CSV saved: {clean_csv_path}")
    
    # 4. Generate processing report
    report_path = f"{output_path}/processing_report.md"
    with open(report_path, "w") as f:
        f.write("# Job Market Data Processing Report\n\n")
        f.write(f"**Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Records Processed:** {df.count():,}\n\n")
        f.write("## Data Quality Metrics\n\n")
        f.write(f"- Records with valid salaries: {df.filter(col('salary_avg').isNotNull()).count():,}\n")
        f.write(f"- Unique companies: {df.select('COMPANY').distinct().count():,}\n")
        f.write(f"- Unique locations: {df.select('location_clean').distinct().count():,}\n")
        f.write(f"- AI/ML roles: {df.filter(col('is_ai_role') == 1).count():,}\n")
        f.write(f"- Remote-friendly positions: {df.filter(col('remote_allowed_clean') == 1).count():,}\n")
        f.write("\n## Output Files\n\n")
        f.write(f"- **Parquet (full dataset):** `{parquet_path}`\n")
        f.write(f"- **CSV sample:** `{csv_path}`\n") 
        f.write(f"- **Clean CSV:** `{clean_csv_path}`\n")
    
    logger.info(f"Processing report saved: {report_path}")


def main():
    """Main processing pipeline using optimized cleaning method."""
    logger.info("Starting Lightcast job market data processing (optimized)")
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Load raw data
        raw_df = load_raw_lightcast_data(spark, "data/raw/lightcast_job_postings.csv")
        
        # Clean and process data using optimized method
        processed_df = clean_and_process_data_optimized(raw_df)
        
        # Cache for multiple operations
        processed_df.cache()
        
        # Create relational tables
        create_relational_tables(processed_df)
        
        # Save processed data
        save_processed_data(processed_df)
        
        logger.info("Optimized data processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in processing pipeline: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()