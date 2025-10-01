"""
Enhanced Job Market Data Processor with Comprehensive Analytics

This module provides advanced data processing capabilities for job market analysis,
including sophisticated data cleaning, feature engineering, and PySpark-based
big data processing for the Lightcast job postings dataset.

Author: Saurabh Sharma
Version: 2.0.0
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, isnan, isnull, trim, upper, lower, 
    count, avg, sum as spark_sum, median, desc, 
    lit, current_timestamp, regexp_replace, coalesce
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, BooleanType, DateType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JobMarketDataProcessor:
    """
    Enhanced processor for comprehensive job market data analysis.
    
    This class provides advanced data processing capabilities including:
    - Sophisticated data quality assessment and cleaning
    - Multi-strategy missing value imputation
    - Feature engineering for AI/ML role classification
    - Geographic and salary analysis
    - Comprehensive statistical summaries
    """
    
    def __init__(self, app_name: str = "JobMarketAnalysis"):
        """Initialize the processor with Spark session and data schema."""
        
        logger.info("Initializing Enhanced Job Market Data Processor...")
        
        # Initialize Spark session with optimized configuration
        self.spark = SparkSession.builder.appName(app_name).config("spark.sql.adaptive.enabled", "true").config("spark.sql.adaptive.coalescePartitions.enabled", "true").config("spark.sql.adaptive.skewJoin.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true").config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Define comprehensive Lightcast schema
        self.lightcast_schema = StructType([
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
            StructField("CITY", StringType(), True),
            StructField("JOB_FUNCTION", StringType(), True),
            StructField("COMPANY_SIZE", StringType(), True),
            StructField("BENEFITS", StringType(), True)
        ])
        
        # Initialize data containers
        self.df_raw = None
        self.df_processed = None
        
        logger.info(f"Spark session initialized: {self.spark.version}")
        logger.info(f"Schema defined with {len(self.lightcast_schema.fields)} fields")
    
    def load_data(self, file_path: str, use_sample: bool = False, sample_size: int = 50000) -> DataFrame:
        """
        Load job market data from file or create sample data.
        
        Args:
            file_path: Path to the Lightcast CSV file
            use_sample: Whether to create sample data instead of loading from file
            sample_size: Number of sample records to generate (if using sample data)
            
        Returns:
            Spark DataFrame with loaded data
        """
        
        if use_sample or not Path(file_path).exists():
            logger.info("Creating enhanced sample dataset for development...")
            self.df_raw = self._create_enhanced_sample_data(n_samples=sample_size)
        else:
            logger.info(f"Loading data from: {file_path}")
            
            # Load CSV with predefined schema
            self.df_raw = self.spark.read.option("header", "true").option("inferSchema", "false").option("multiline", "true").option("escape", "\"").schema(self.lightcast_schema).csv(file_path)
            
            logger.info(f"Data loaded: {self.df_raw.count():,} records")
        
        return self.df_raw
    
    def load_and_process_data(self, file_path: str, use_sample: bool = False, sample_size: int = 50000) -> DataFrame:
        """
        Load and automatically process data with appropriate method.
        
        Args:
            file_path: Path to the Lightcast CSV file
            use_sample: Whether to create sample data instead of loading from file
            sample_size: Number of sample records to generate (if using sample data)
            
        Returns:
            Processed Spark DataFrame ready for analysis
        """
        # Load the raw data
        df = self.load_data(file_path, use_sample, sample_size)
        
        # Choose appropriate processing method based on data format
        if use_sample or not Path(file_path).exists():
            # Use simple processing for generated sample data
            return self.process_sample_data(df)
        elif "sample" in file_path.lower() or df.count() < 100:  # Detect sample data format
            # Use simple processing for sample data files
            return self.process_sample_data(df)
        else:
            # Use full processing pipeline for real Lightcast data
            return self.clean_and_process_data(df)
    
    def _create_enhanced_sample_data(self, n_samples: int = 50000) -> DataFrame:
        """Create comprehensive sample data with realistic distributions."""
        
        np.random.seed(42)
        logger.info(f"Generating {n_samples:,} sample job records...")
        
        # Enhanced job titles with more realistic distribution
        job_titles = [
            "Software Engineer", "Senior Software Engineer", "Staff Software Engineer",
            "Data Scientist", "Senior Data Scientist", "Principal Data Scientist",
            "Machine Learning Engineer", "ML Engineer", "AI Engineer", 
            "Product Manager", "Senior Product Manager", "Principal Product Manager",
            "Data Analyst", "Business Analyst", "Senior Business Analyst",
            "DevOps Engineer", "Site Reliability Engineer", "Cloud Architect",
            "Frontend Developer", "Backend Developer", "Full Stack Developer",
            "UX Designer", "UI/UX Designer", "Product Designer",
            "Marketing Manager", "Digital Marketing Manager", "Growth Marketing Manager",
            "Sales Representative", "Account Executive", "Sales Manager",
            "Customer Success Manager", "Technical Writer", "QA Engineer",
            "Cybersecurity Analyst", "Information Security Analyst", "CISO",
            "Database Administrator", "System Administrator", "Network Administrator"
        ]
        
        # Company names with size indicators
        companies_data = [
            ("Google", "Large"), ("Microsoft", "Large"), ("Amazon", "Large"), ("Apple", "Large"),
            ("Meta", "Large"), ("Netflix", "Large"), ("Tesla", "Large"), ("Uber", "Large"),
            ("Airbnb", "Large"), ("Spotify", "Medium"), ("Adobe", "Large"), ("Salesforce", "Large"),
            ("Oracle", "Large"), ("IBM", "Large"), ("Intel", "Large"), ("NVIDIA", "Large"),
            ("Startup Alpha", "Small"), ("TechCorp Inc", "Medium"), ("DataSystems LLC", "Medium"),
            ("Innovation Labs", "Small"), ("FutureTech Solutions", "Medium"), ("CloudFirst", "Small")
        ]
        
        # Geographic locations with market tier
        locations_data = [
            ("San Francisco, CA", "CA", "San Francisco", "Tier 1"),
            ("Seattle, WA", "WA", "Seattle", "Tier 1"),
            ("New York, NY", "NY", "New York", "Tier 1"),
            ("Austin, TX", "TX", "Austin", "Tier 2"),
            ("Boston, MA", "MA", "Boston", "Tier 1"),
            ("Los Angeles, CA", "CA", "Los Angeles", "Tier 1"),
            ("Chicago, IL", "IL", "Chicago", "Tier 2"),
            ("Denver, CO", "CO", "Denver", "Tier 2"),
            ("Atlanta, GA", "GA", "Atlanta", "Tier 2"),
            ("Portland, OR", "OR", "Portland", "Tier 2"),
            ("Phoenix, AZ", "AZ", "Phoenix", "Tier 3"),
            ("Dallas, TX", "TX", "Dallas", "Tier 2"),
            ("Miami, FL", "FL", "Miami", "Tier 2"),
            ("Nashville, TN", "TN", "Nashville", "Tier 3"),
            ("Salt Lake City, UT", "UT", "Salt Lake City", "Tier 3")
        ]
        
        # Industry categories
        industries = [
            "Technology", "Healthcare", "Finance", "Manufacturing", "Retail",
            "Education", "Government", "Non-profit", "Entertainment", 
            "Transportation", "Energy", "Real Estate", "Consulting",
            "Media", "Telecommunications", "Aerospace", "Biotechnology"
        ]
        
        # Generate sample records
        sample_records = []
        
        for i in range(n_samples):
            # Select job title and derive characteristics
            title = str(np.random.choice(job_titles))
            
            # Base salary calculation with realistic factors
            base_salary = self._calculate_realistic_salary(title, locations_data)
            
            # Company selection with size correlation
            company, company_size = companies_data[np.random.randint(0, len(companies_data))]
            
            # Location selection
            location_full, state, city, tier = locations_data[np.random.randint(0, len(locations_data))]
            
            # Adjust salary for location tier
            location_multiplier = {"Tier 1": 1.4, "Tier 2": 1.1, "Tier 3": 0.9}[tier]
            adjusted_salary = base_salary * location_multiplier
            
            # Generate salary range
            salary_min = max(25000, int(adjusted_salary * 0.85))
            salary_max = int(adjusted_salary * 1.15)
            
            # Employment type with realistic distribution
            employment_type = str(np.random.choice(
                ["Full-time", "Part-time", "Contract", "Temporary", "Internship"],
                p=[0.75, 0.08, 0.12, 0.03, 0.02]
            ))
            
            # Experience level based on title
            if "Senior" in title or "Staff" in title or "Principal" in title:
                experience_level = str(np.random.choice(["Senior", "Executive"], p=[0.8, 0.2]))
            elif "Manager" in title or "Lead" in title:
                experience_level = str(np.random.choice(["Mid", "Senior", "Executive"], p=[0.3, 0.5, 0.2]))
            else:
                experience_level = str(np.random.choice(["Entry", "Mid", "Senior"], p=[0.4, 0.4, 0.2]))
            
            # Remote work policy based on role type
            if any(keyword in title.lower() for keyword in ["engineer", "developer", "data", "analyst"]):
                remote_options = ["Remote", "Hybrid", "On-site"]
                remote_probs = [0.3, 0.5, 0.2]
            else:
                remote_options = ["Remote", "Hybrid", "On-site"]
                remote_probs = [0.15, 0.35, 0.5]
            
            remote_allowed = str(np.random.choice(remote_options, p=remote_probs))
            
            # Generate posting date in last 12 months
            days_ago = np.random.randint(1, 365)
            posted_date = (datetime.now() - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Create record
            record = {
                "JOB_ID": f"LCS_{i:07d}",
                "TITLE": title,
                "COMPANY": company,
                "LOCATION": location_full,
                "POSTED": posted_date,
                "SALARY_MIN": str(salary_min),
                "SALARY_MAX": str(salary_max),
                "SALARY_CURRENCY": "USD",
                "INDUSTRY": str(np.random.choice(industries)),
                "EXPERIENCE_LEVEL": experience_level,
                "EMPLOYMENT_TYPE": employment_type,
                "REMOTE_ALLOWED": remote_allowed,
                "REQUIRED_SKILLS": self._generate_realistic_skills(title),
                "EDUCATION_REQUIRED": self._determine_education_requirement(title, experience_level),
                "DESCRIPTION": f"Seeking a qualified {title} to join our team...",
                "COUNTRY": "United States",
                "STATE": state,
                "CITY": city,
                "JOB_FUNCTION": self._categorize_job_function(title),
                "COMPANY_SIZE": company_size,
                "BENEFITS": self._generate_benefits()
            }
            
            sample_records.append(record)
        
        # Create Spark DataFrame
        df = self.spark.createDataFrame(sample_records)
        logger.info(f"Enhanced sample data created: {df.count():,} records")
        
        return df
    
    def _calculate_realistic_salary(self, title: str, locations_data: List) -> float:
        """Calculate realistic base salary based on job title."""
        
        # Base salary ranges by role type
        salary_ranges = {
            "entry": (45000, 75000),
            "mid": (75000, 120000),
            "senior": (120000, 180000),
            "staff": (180000, 250000),
            "principal": (200000, 300000),
            "executive": (250000, 500000)
        }
        
        # Determine level from title
        title_lower = title.lower()
        if "principal" in title_lower or "chief" in title_lower:
            level = "principal"
        elif "staff" in title_lower:
            level = "staff"
        elif "senior" in title_lower or "sr" in title_lower:
            level = "senior"
        elif "manager" in title_lower or "lead" in title_lower:
            level = "senior"
        elif any(word in title_lower for word in ["junior", "associate", "intern"]):
            level = "entry"
        else:
            level = "mid"
        
        # Role type multipliers
        if any(keyword in title_lower for keyword in ["data scientist", "ml", "ai", "machine learning"]):
            multiplier = 1.3
        elif "engineer" in title_lower:
            multiplier = 1.15
        elif "manager" in title_lower:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        # Calculate base salary
        min_sal, max_sal = salary_ranges[level]
        base_salary = np.random.uniform(min_sal, max_sal) * multiplier
        
        return base_salary
    
    def _generate_realistic_skills(self, title: str) -> str:
        """Generate realistic skill sets based on job title."""
        
        # Skill categories
        programming_skills = ["Python", "Java", "JavaScript", "C++", "Go", "Rust", "Swift"]
        data_skills = ["SQL", "R", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch"]
        cloud_skills = ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform"]
        soft_skills = ["Communication", "Leadership", "Project Management", "Agile", "Scrum"]
        domain_skills = ["Machine Learning", "Data Analysis", "Statistics", "A/B Testing"]
        
        selected_skills = []
        title_lower = title.lower()
        
        # Add skills based on title
        if "data" in title_lower or "analyst" in title_lower:
            selected_skills.extend(np.random.choice(data_skills, 3, replace=False))
            selected_skills.extend(np.random.choice(domain_skills, 2, replace=False))
        
        if "engineer" in title_lower or "developer" in title_lower:
            selected_skills.extend(np.random.choice(programming_skills, 2, replace=False))
            selected_skills.extend(np.random.choice(cloud_skills, 2, replace=False))
        
        if "manager" in title_lower or "lead" in title_lower:
            selected_skills.extend(np.random.choice(soft_skills, 3, replace=False))
        
        # Add some random skills
        all_skills = programming_skills + data_skills + cloud_skills + soft_skills
        additional_skills = np.random.choice(all_skills, np.random.randint(1, 3), replace=False)
        selected_skills.extend(additional_skills)
        
        # Remove duplicates and return
        unique_skills = list(set(selected_skills))
        return ", ".join(unique_skills[:8])  # Limit to 8 skills
    
    def _determine_education_requirement(self, title: str, experience_level: str) -> str:
        """Determine realistic education requirements."""
        
        title_lower = title.lower()
        
        if "data scientist" in title_lower or "research" in title_lower:
            return str(np.random.choice(["Master", "PhD"], p=[0.7, 0.3]))
        elif "senior" in title_lower or experience_level == "Senior":
            return str(np.random.choice(["Bachelor", "Master"], p=[0.6, 0.4]))
        elif "engineer" in title_lower:
            return str(np.random.choice(["Bachelor", "Master", "Associate"], p=[0.7, 0.2, 0.1]))
        else:
            return str(np.random.choice(["High School", "Associate", "Bachelor"], p=[0.2, 0.3, 0.5]))
    
    def _categorize_job_function(self, title: str) -> str:
        """Categorize job function based on title."""
        
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ["engineer", "developer", "architect"]):
            return "Engineering"
        elif any(keyword in title_lower for keyword in ["data", "analyst", "scientist"]):
            return "Data & Analytics"
        elif "product" in title_lower:
            return "Product Management"
        elif any(keyword in title_lower for keyword in ["marketing", "growth"]):
            return "Marketing"
        elif "sales" in title_lower:
            return "Sales"
        elif any(keyword in title_lower for keyword in ["design", "ux", "ui"]):
            return "Design"
        else:
            return "Other"
    
    def _generate_benefits(self) -> str:
        """Generate realistic benefits package."""
        
        benefits_options = [
            "Health Insurance", "Dental Insurance", "Vision Insurance",
            "401(k) Matching", "Paid Time Off", "Remote Work", 
            "Flexible Hours", "Stock Options", "Bonuses",
            "Professional Development", "Gym Membership", "Free Meals"
        ]
        
        num_benefits = np.random.randint(3, 8)
        selected_benefits = np.random.choice(benefits_options, num_benefits, replace=False)
        
        return ", ".join([str(benefit) for benefit in selected_benefits])
    
    def assess_data_quality(self, df: DataFrame) -> Dict:
        """Comprehensive data quality assessment."""
        
        logger.info("=== CONDUCTING DATA QUALITY ASSESSMENT ===")
        
        total_rows = df.count()
        total_cols = len(df.columns)
        
        logger.info(f"Dataset dimensions: {total_rows:,} rows × {total_cols} columns")
        
        # Missing value analysis
        missing_analysis = {}
        
        for column in df.columns:
            null_count = df.filter(col(column).isNull() | (col(column) == "")).count()
            missing_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            
            missing_analysis[column] = {
                "null_count": null_count,
                "missing_percentage": missing_pct,
                "recommendation": "drop" if missing_pct > 50 else "impute" if missing_pct > 0 else "keep"
            }
        
        # Data type analysis
        schema_info = {field.name: str(field.dataType) for field in df.schema.fields}
        
        # Duplicate analysis
        key_columns = ["TITLE", "COMPANY", "LOCATION", "POSTED"]
        duplicate_count = df.count() - df.dropDuplicates(key_columns).count()
        
        quality_report = {
            "total_records": total_rows,
            "total_columns": total_cols,
            "missing_analysis": missing_analysis,
            "schema_info": schema_info,
            "duplicate_records": duplicate_count,
            "duplicate_percentage": (duplicate_count / total_rows * 100) if total_rows > 0 else 0
        }
        
        # Log summary
        logger.info(f"Missing value analysis completed for {total_cols} columns")
        logger.info(f"Duplicate records found: {duplicate_count:,} ({duplicate_count/total_rows*100:.1f}%)")
        
        return quality_report
    
    def clean_and_process_data(self, df: DataFrame) -> DataFrame:
        """Comprehensive data cleaning and processing pipeline."""
        
        logger.info("=== STARTING DATA CLEANING PIPELINE ===")
        
        initial_count = df.count()
        
        # Step 1: Remove duplicates
        logger.info("Step 1: Removing duplicate records...")
        df_clean = df.dropDuplicates(["TITLE", "COMPANY", "LOCATION", "POSTED"])
        duplicates_removed = initial_count - df_clean.count()
        logger.info(f"   Removed {duplicates_removed:,} duplicate records")
        
        # Step 2: Clean and convert salary fields
        logger.info("Step 2: Processing salary data...")
        
        # Handle different salary column naming conventions
        # Original Lightcast data uses SALARY_FROM/SALARY_TO
        # Sample data uses SALARY_MIN/SALARY_MAX
        if "SALARY_FROM" in df_clean.columns and "SALARY_TO" in df_clean.columns:
            # Lightcast format
            logger.info("   Processing Lightcast format (SALARY_FROM/SALARY_TO)")
            df_clean = df_clean.withColumn(
                "SALARY_MIN_CLEAN", 
                when(col("SALARY_FROM").rlike("^[0-9]+$"), col("SALARY_FROM").cast("double")).otherwise(None)
            ).withColumn(
                "SALARY_MAX_CLEAN", 
                when(col("SALARY_TO").rlike("^[0-9]+$"), col("SALARY_TO").cast("double")).otherwise(None)
            )
        elif "SALARY_MIN" in df_clean.columns and "SALARY_MAX" in df_clean.columns:
            # Sample data format - already numeric strings
            logger.info("   Processing sample data format (SALARY_MIN/SALARY_MAX)")
            df_clean = df_clean.withColumn(
                "SALARY_MIN_CLEAN", 
                when(col("SALARY_MIN").rlike("^[0-9]+$"), col("SALARY_MIN").cast("double")).otherwise(None)
            ).withColumn(
                "SALARY_MAX_CLEAN", 
                when(col("SALARY_MAX").rlike("^[0-9]+$"), col("SALARY_MAX").cast("double")).otherwise(None)
            )
        else:
            logger.warning("   No recognizable salary columns found, creating null columns")
            df_clean = df_clean.withColumn("SALARY_MIN_CLEAN", lit(None).cast("double")).withColumn("SALARY_MAX_CLEAN", lit(None).cast("double"))
        
        # Create average salary column
        df_clean = df_clean.withColumn("SALARY_AVG", (col("SALARY_MIN_CLEAN") + col("SALARY_MAX_CLEAN")) / 2)
        
        # Step 3: Standardize categorical fields
        logger.info("Step 3: Standardizing categorical fields...")
        df_clean = df_clean.withColumn("COMPANY_CLEAN", when(col("COMPANY").isNull() | (col("COMPANY") == "") | (col("COMPANY") == "null"), "Undefined").otherwise(trim(col("COMPANY")))).withColumn("EMPLOYMENT_TYPE_CLEAN", when(col("EMPLOYMENT_TYPE").isNull() | (col("EMPLOYMENT_TYPE") == ""), "Full-time").otherwise(trim(col("EMPLOYMENT_TYPE")))).withColumn("REMOTE_TYPE_CLEAN", when(col("REMOTE_TYPE").isNull(), "Unknown").when(lower(col("REMOTE_TYPE_NAME")).contains("remote"), "Remote").when(lower(col("REMOTE_TYPE_NAME")).contains("hybrid"), "Hybrid").when(lower(col("REMOTE_TYPE_NAME")).contains("on"), "On-site").otherwise("Unknown")).withColumn("EXPERIENCE_CLEAN", 
            when(col("MIN_YEARS_EXPERIENCE").isNull() & col("MAX_YEARS_EXPERIENCE").isNull(), "Unknown")
            .when(col("MIN_YEARS_EXPERIENCE") <= 2, "Entry Level")
            .when(col("MIN_YEARS_EXPERIENCE") <= 5, "Mid Level")
            .when(col("MIN_YEARS_EXPERIENCE") <= 10, "Senior Level")
            .otherwise("Executive Level")
        )
        
        # Step 4: Geographic data cleaning
        logger.info("Step 4: Cleaning geographic data...")
        df_clean = df_clean.withColumn("STATE_CLEAN", when(col("STATE").isNull(), "Unknown").otherwise(trim(upper(col("STATE"))))).withColumn("CITY_CLEAN", when(col("CITY_NAME").isNull(), trim(col("CITY"))).otherwise(trim(col("CITY_NAME"))))
        
        # Step 5: Handle missing salary values with sophisticated imputation
        logger.info("Step 5: Imputing missing salary values...")
        df_clean = self._impute_missing_salaries(df_clean)
        
        # Step 6: Filter unrealistic salaries
        logger.info("Step 6: Filtering unrealistic salary values...")
        df_clean = df_clean.filter(
            (col("SALARY_AVG_IMPUTED") >= 20000) & 
            (col("SALARY_AVG_IMPUTED") <= 500000)
        )
        
        # Step 7: Feature engineering
        logger.info("Step 7: Engineering derived features...")
        df_clean = self._engineer_features(df_clean)
        
        # Step 8: Final column standardization - alias salary columns to lowercase for compatibility
        logger.info("Step 8: Standardizing final column names...")
        df_final = df_clean.select(
            # Salary columns with lowercase aliases for compatibility
            col("SALARY_MIN").alias("salary_min"),
            col("SALARY_MAX").alias("salary_max"),
            col("SALARY_MIN_CLEAN").alias("salary_min_clean"),
            col("SALARY_MAX_CLEAN").alias("salary_max_clean"),
            col("SALARY_AVG").alias("salary_avg"),
            col("SALARY_AVG_IMPUTED").alias("salary_avg_imputed"),
            # All other columns as-is
            *[col(c) for c in df_clean.columns if not c.startswith("SALARY_")]
        )
        
        final_count = df_final.count()
        logger.info(f"Data cleaning completed: {initial_count:,} → {final_count:,} records")
        logger.info(f"Records removed: {initial_count - final_count:,} ({(initial_count - final_count)/initial_count*100:.1f}%)")
        
        self.df_processed = df_final
        return df_final
    
    def process_sample_data(self, df: DataFrame) -> DataFrame:
        """
        Simplified processing for sample data that adds column aliases and basic mapping.
        
        Sample data is already clean and just needs column name standardization
        to match the expected lowercase format for SalaryVisualizer compatibility.
        """
        logger.info("=== PROCESSING SAMPLE DATA ===")
        
        # Add experience_level mapping for compatibility
        df_with_experience = df.withColumn("experience_level", 
            when(col("EXPERIENCE_LEVEL").isNull() | (col("EXPERIENCE_LEVEL") == ""), "Unknown")
            .when(lower(col("EXPERIENCE_LEVEL")).contains("entry") | lower(col("EXPERIENCE_LEVEL")).contains("junior"), "Entry Level")
            .when(lower(col("EXPERIENCE_LEVEL")).contains("mid") | lower(col("EXPERIENCE_LEVEL")).contains("intermediate"), "Mid Level") 
            .when(lower(col("EXPERIENCE_LEVEL")).contains("senior"), "Senior Level")
            .when(lower(col("EXPERIENCE_LEVEL")).contains("lead") | lower(col("EXPERIENCE_LEVEL")).contains("principal") | lower(col("EXPERIENCE_LEVEL")).contains("director"), "Executive Level")
            .otherwise("Mid Level"))
        
        # Simple column aliasing for sample data (which is already clean)
        df_processed = df_with_experience.select(
            # Core salary columns with lowercase aliases for compatibility
            col("SALARY_MIN").alias("salary_min"),
            col("SALARY_MAX").alias("salary_max"),
            col("SALARY_CURRENCY").alias("salary_currency"),
            # Create derived salary columns with safe casting
            coalesce(
                (
                    when(col("SALARY_MIN").rlike("^[0-9]+$"), col("SALARY_MIN").cast("double")).otherwise(lit(None)) +
                    when(col("SALARY_MAX").rlike("^[0-9]+$"), col("SALARY_MAX").cast("double")).otherwise(lit(None))
                ) / 2,
                lit(None)
            ).alias("salary_avg"),
            # Experience level mapping
            col("experience_level"),
            # Key columns with lowercase aliases for SalaryVisualizer compatibility
            col("INDUSTRY").alias("industry"),
            col("EMPLOYMENT_TYPE").alias("employment_type"),
            col("REMOTE_ALLOWED").alias("remote_type"),
            col("EDUCATION_REQUIRED").alias("education_level"),
            col("COMPANY").alias("company"),
            col("LOCATION").alias("location"),
            col("STATE").alias("state"),
            col("CITY").alias("city"),
            col("TITLE").alias("job_title"),
            col("REQUIRED_SKILLS").alias("skills"),
            col("COMPANY_SIZE").alias("company_size"),
            # All other columns as-is
            *[col(c) for c in df_with_experience.columns if c not in ["SALARY_MIN", "SALARY_MAX", "SALARY_CURRENCY", "experience_level", 
                                                                       "INDUSTRY", "EMPLOYMENT_TYPE", "REMOTE_ALLOWED", "EDUCATION_REQUIRED",
                                                                       "COMPANY", "LOCATION", "STATE", "CITY", "TITLE", "REQUIRED_SKILLS", "COMPANY_SIZE"]]
        )
        
        logger.info(f"Sample data processed: {df_processed.count():,} records with standardized columns")
        self.df_processed = df_processed
        return df_processed
    
    def _impute_missing_salaries(self, df: DataFrame) -> DataFrame:
        """Sophisticated missing salary imputation using multiple strategies."""
        
        # Strategy 1: Impute based on location, employment type, and experience level
        location_employment_medians = df.filter(col("SALARY_AVG").isNotNull()).groupBy("STATE_CLEAN", "EMPLOYMENT_TYPE_CLEAN", "EXPERIENCE_CLEAN").agg(median("SALARY_AVG").alias("median_salary_detailed"))
        
        # Strategy 2: Impute based on employment type and experience level only
        employment_experience_medians = df.filter(col("SALARY_AVG").isNotNull()).groupBy("EMPLOYMENT_TYPE_CLEAN", "EXPERIENCE_CLEAN").agg(median("SALARY_AVG").alias("median_salary_employment"))
        
        # Strategy 3: Impute based on experience level only
        experience_medians = df.filter(col("SALARY_AVG").isNotNull()).groupBy("EXPERIENCE_CLEAN").agg(median("SALARY_AVG").alias("median_salary_experience"))
        
        # Strategy 4: Overall median
        overall_median = df.filter(col("SALARY_AVG").isNotNull()).agg(median("SALARY_AVG").alias("overall_median")).collect()[0]["overall_median"]
        
        # Join imputation values
        df_imputed = df.join(location_employment_medians, ["STATE_CLEAN", "EMPLOYMENT_TYPE_CLEAN", "EXPERIENCE_CLEAN"], "left").join(employment_experience_medians, ["EMPLOYMENT_TYPE_CLEAN", "EXPERIENCE_CLEAN"], "left").join(experience_medians, ["EXPERIENCE_CLEAN"], "left")
        
        # Apply imputation in order of preference
        df_imputed = df_imputed.withColumn(
            "SALARY_AVG_IMPUTED",
            when(col("SALARY_AVG").isNotNull(), col("SALARY_AVG"))
            .when(col("median_salary_detailed").isNotNull(), col("median_salary_detailed"))
            .when(col("median_salary_employment").isNotNull(), col("median_salary_employment"))
            .when(col("median_salary_experience").isNotNull(), col("median_salary_experience"))
            .otherwise(lit(overall_median))
        )
        
        # Clean up intermediate columns
        df_imputed = df_imputed.drop(
            "median_salary_detailed", "median_salary_employment", "median_salary_experience"
        )
        
        return df_imputed
    
    def _engineer_features(self, df: DataFrame) -> DataFrame:
        """Engineer additional features for analysis."""
        
        # AI/ML role classification
        ai_keywords_pattern = ".*(machine learning|artificial intelligence|\\bai\\b|\\bml\\b|data scientist|deep learning|neural network|computer vision|nlp|data science).*"
        
        df_features = df.withColumn(
            "IS_AI_ROLE",
            when(lower(col("TITLE")).rlike(ai_keywords_pattern) |
                 lower(col("SKILLS_NAME")).rlike(ai_keywords_pattern) |
                 lower(col("BODY")).rlike(ai_keywords_pattern), True)
            .otherwise(False)
        )
        
        # Experience years mapping
        df_features = df_features.withColumn(
            "EXPERIENCE_YEARS",
            when(col("EXPERIENCE_CLEAN") == "Entry Level", 1)
            .when(col("EXPERIENCE_CLEAN") == "Mid Level", 4)
            .when(col("EXPERIENCE_CLEAN") == "Senior Level", 8)
            .when(col("EXPERIENCE_CLEAN") == "Executive Level", 15)
            .otherwise(3)
        )
        
        # Salary tiers
        df_features = df_features.withColumn(
            "SALARY_TIER",
            when(col("SALARY_AVG_IMPUTED") < 60000, "Low")
            .when(col("SALARY_AVG_IMPUTED") < 100000, "Medium")
            .when(col("SALARY_AVG_IMPUTED") < 150000, "High")
            .otherwise("Very High")
        )
        
        # Job function classification
        df_features = df_features.withColumn(
            "JOB_FUNCTION_DERIVED",
            when(lower(col("TITLE")).rlike(".*(engineer|developer|architect).*"), "Engineering")
            .when(lower(col("TITLE")).rlike(".*(data|analyst|scientist).*"), "Data & Analytics")
            .when(lower(col("TITLE")).rlike(".*product.*"), "Product Management")
            .when(lower(col("TITLE")).rlike(".*(marketing|growth).*"), "Marketing")
            .when(lower(col("TITLE")).rlike(".*sales.*"), "Sales")
            .when(lower(col("TITLE")).rlike(".*(design|ux|ui).*"), "Design")
            .otherwise("Other")
        )
        
        # Add processing timestamp
        df_features = df_features.withColumn("PROCESSED_TIMESTAMP", current_timestamp())
        
        return df_features
    
    def generate_summary_statistics(self, df: DataFrame) -> Dict:
        """Generate comprehensive summary statistics."""
        
        logger.info("=== GENERATING SUMMARY STATISTICS ===")
        
        # Basic counts
        total_jobs = df.count()
        
        # AI/ML statistics
        ai_stats = df.groupBy("IS_AI_ROLE").agg(
            count("*").alias("count"),
            avg("SALARY_AVG_IMPUTED").alias("avg_salary")
        ).collect()
        
        # Remote work statistics
        remote_stats = df.groupBy("REMOTE_ALLOWED_CLEAN").agg(
            count("*").alias("count"),
            avg("SALARY_AVG_IMPUTED").alias("avg_salary")
        ).collect()
        
        # Salary statistics
        salary_stats = df.select("SALARY_AVG_IMPUTED").describe().collect()
        
        # Industry distribution
        industry_stats = df.groupBy("INDUSTRY_CLEAN").agg(count("*").alias("count")).orderBy(desc("count")).limit(10).collect()
        
        summary = {
            "total_jobs": total_jobs,
            "ai_ml_stats": {row["IS_AI_ROLE"]: {"count": row["count"], "avg_salary": row["avg_salary"]} 
                           for row in ai_stats},
            "remote_stats": {row["REMOTE_ALLOWED_CLEAN"]: {"count": row["count"], "avg_salary": row["avg_salary"]} 
                            for row in remote_stats},
            "salary_statistics": {row["summary"]: row["SALARY_AVG_IMPUTED"] for row in salary_stats},
            "top_industries": [(row["INDUSTRY_CLEAN"], row["count"]) for row in industry_stats]
        }
        
        logger.info(f"Summary statistics generated for {total_jobs:,} job records")
        
        return summary
    
    def save_processed_data(self, df: DataFrame, output_path: str = "data/processed/"):
        """Save processed data in multiple formats for different use cases."""
        
        logger.info("=== SAVING PROCESSED DATA ===")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet (most efficient for Spark)
        parquet_path = output_dir / "job_market_processed.parquet"
        df.write.mode("overwrite").option("compression", "snappy").parquet(str(parquet_path))
        logger.info(f"Saved Parquet format: {parquet_path}")
        
        # Save sample as CSV for easy inspection
        sample_df = df.sample(fraction=0.02)  # 2% sample
        csv_path = output_dir / "job_market_sample.csv"
        sample_df.toPandas().to_csv(csv_path, index=False)
        logger.info(f"Saved CSV sample: {csv_path}")
        
        # Save schema information
        schema_path = output_dir / "data_schema.json"
        schema_info = {"schema": df.schema.json(), "column_count": len(df.columns)}
        
        with open(schema_path, "w") as f:
            json.dump(schema_info, f, indent=2)
        logger.info(f"Saved schema information: {schema_path}")
        
        # Generate processing report
        report_path = output_dir / "processing_report.md"
        with open(report_path, "w") as f:
            f.write("# Job Market Data Processing Report\n\n")
            f.write(f"**Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Spark Version**: {self.spark.version}\n")
            f.write(f"**Total Records Processed**: {df.count():,}\n")
            f.write(f"**Output Location**: {output_path}\n\n")
            f.write("## Data Quality Summary\n\n")
            f.write("- Duplicates removed based on job title, company, location, and posting date\n")
            f.write("- Missing salary values imputed using hierarchical median strategy\n")
            f.write("- Categorical fields standardized and cleaned\n")
            f.write("- Derived features engineered for AI/ML role classification\n")
        
        logger.info(f"Processing report saved: {report_path}")
    
    def clean_and_process_data_optimized(self, df: DataFrame) -> DataFrame:
        """
        Optimized data cleaning and processing pipeline with notebook improvements.
        
        This method incorporates all the optimizations discovered in the notebook analysis:
        - Drop non-essential timestamp columns
        - Handle REMOTE_TYPE_NAME nulls
        - Resolve CITY vs CITY_NAME duplication with base64 decoding
        - Remove duplicate county columns
        - Comprehensive data structure optimization
        
        Args:
            df: Raw Spark DataFrame
            
        Returns:
            Optimized and cleaned DataFrame
        """
        import base64
        from pyspark.sql.functions import when, col, isnan, isnull, coalesce, lit, trim, udf
        from pyspark.sql.types import StringType
        
        logger.info("=== STARTING OPTIMIZED DATA CLEANING PIPELINE ===")
        
        original_column_count = len(df.columns)
        original_record_count = df.count()
        
        logger.info(f"BEFORE CLEANING:")
        logger.info(f"   → Columns: {original_column_count}")
        logger.info(f"   → Records: {original_record_count:,}")
        
        # STEP 1: Drop non-essential timestamp/metadata columns
        logger.info("Step 1: Dropping non-essential columns...")
        columns_to_drop = [
            'LAST_UPDATED_DATE',
            'LAST_UPDATED_TIMESTAMP', 
            'ACTIVE_SOURCES_INFO'
        ]
        
        existing_columns_to_drop = [col_name for col_name in columns_to_drop if col_name in df.columns]
        if existing_columns_to_drop:
            df_cleaned = df.drop(*existing_columns_to_drop)
            logger.info(f"   Dropped columns: {existing_columns_to_drop}")
        else:
            df_cleaned = df
            logger.info(f"   No target columns found to drop")
        
        # STEP 2: Handle REMOTE_TYPE_NAME nulls
        logger.info("Step 2: Handling REMOTE_TYPE_NAME nulls...")
        if 'REMOTE_TYPE_NAME' in df_cleaned.columns:
            null_remote_count = df_cleaned.filter(col('REMOTE_TYPE_NAME').isNull()).count()
            null_percentage = (null_remote_count / original_record_count) * 100
            
            logger.info(f"   → REMOTE_TYPE_NAME nulls: {null_remote_count:,} ({null_percentage:.1f}%)")
            
            df_cleaned = df_cleaned.withColumn(
                'REMOTE_TYPE_NAME',
                when(col('REMOTE_TYPE_NAME').isNull(), lit('Undefined'))
                .otherwise(col('REMOTE_TYPE_NAME'))
            )
            logger.info(f"   Nulls replaced with 'Undefined'")
        else:
            logger.info(f"   REMOTE_TYPE_NAME column not found")
        
        # STEP 3: Resolve CITY vs CITY_NAME duplication with base64 decoding
        logger.info("Step 3: Resolving CITY vs CITY_NAME duplication...")
        city_cols = [col_name for col_name in df_cleaned.columns if col_name in ['CITY', 'CITY_NAME']]
        logger.info(f"   → Found city columns: {city_cols}")
        
        if len(city_cols) >= 2:
            # Define base64 decoding UDF
            def safe_base64_decode(encoded_str):
                if not encoded_str:
                    return None
                try:
                    # Simple check for base64-like string
                    if len(encoded_str) > 10 and encoded_str.replace('=', '').replace('+', '').replace('/', '').isalnum():
                        decoded = base64.b64decode(encoded_str).decode('utf-8', errors='ignore')
                        return decoded.strip() if decoded.strip() else None
                    else:
                        return encoded_str
                except:
                    return encoded_str
            
            decode_udf = udf(safe_base64_decode, StringType())
            
            if 'CITY' in city_cols and 'CITY_NAME' in city_cols:
                # Create unified CITY column
                df_cleaned = df_cleaned.withColumn(
                    'CITY_UNIFIED',
                    coalesce(
                        # Priority 1: Use CITY_NAME if not null/empty
                        when(col('CITY_NAME').isNotNull() & (col('CITY_NAME') != ''), col('CITY_NAME')),
                        # Priority 2: Use decoded CITY if CITY_NAME is null/empty
                        decode_udf(col('CITY'))
                    )
                )
                
                # Drop original columns and rename unified column
                df_cleaned = df_cleaned.drop('CITY', 'CITY_NAME').withColumnRenamed('CITY_UNIFIED', 'CITY')
                logger.info(f"   Created unified CITY column from CITY and CITY_NAME")
                
            elif 'CITY_NAME' in city_cols:
                df_cleaned = df_cleaned.withColumnRenamed('CITY_NAME', 'CITY')
                logger.info(f"   Renamed CITY_NAME to CITY")
                
            elif 'CITY' in city_cols:
                df_cleaned = df_cleaned.withColumn('CITY', decode_udf(col('CITY')))
                logger.info(f"   Attempted base64 decoding on CITY column")
        
        # STEP 4: Remove duplicate county columns
        logger.info("Step 4: Removing duplicate county columns...")
        county_id_cols = [col_name for col_name in df_cleaned.columns if col_name in ['COUNTY_OUTGOING', 'COUNTY_INCOMING']]
        county_name_cols = [col_name for col_name in df_cleaned.columns if col_name in ['COUNTY_NAME_OUTGOING', 'COUNTY_NAME_INCOMING']]
        
        # Handle county ID columns
        if len(county_id_cols) >= 2:
            # Check if values are identical
            comparison_df = df_cleaned.select('COUNTY_OUTGOING', 'COUNTY_INCOMING').limit(100)
            identical_count = comparison_df.filter(col('COUNTY_OUTGOING') == col('COUNTY_INCOMING')).count()
            total_sample = comparison_df.count()
            
            if identical_count == total_sample or identical_count / total_sample > 0.95:
                df_cleaned = df_cleaned.drop('COUNTY_INCOMING').withColumnRenamed('COUNTY_OUTGOING', 'COUNTY_ID')
                logger.info(f"   Dropped COUNTY_INCOMING, renamed COUNTY_OUTGOING to COUNTY_ID")
        
        # Handle county name columns  
        if len(county_name_cols) >= 2:
            comparison_df = df_cleaned.select('COUNTY_NAME_OUTGOING', 'COUNTY_NAME_INCOMING').limit(100)
            identical_count = comparison_df.filter(col('COUNTY_NAME_OUTGOING') == col('COUNTY_NAME_INCOMING')).count()
            total_sample = comparison_df.count()
            
            if identical_count == total_sample or identical_count / total_sample > 0.95:
                df_cleaned = df_cleaned.drop('COUNTY_NAME_INCOMING').withColumnRenamed('COUNTY_NAME_OUTGOING', 'COUNTY_NAME')
                logger.info(f"   Dropped COUNTY_NAME_INCOMING, renamed COUNTY_NAME_OUTGOING to COUNTY_NAME")
        
        # STEP 5: Apply existing cleaning pipeline
        logger.info("Step 5: Applying standard data cleaning pipeline...")
        df_cleaned = self.clean_and_process_data(df_cleaned)
        
        # STEP 6: Final optimization summary
        final_column_count = len(df_cleaned.columns)
        final_record_count = df_cleaned.count()
        
        logger.info(f"\nOPTIMIZATION SUMMARY:")
        logger.info(f"   → Columns: {original_column_count} → {final_column_count} (removed {original_column_count - final_column_count})")
        logger.info(f"   → Records: {original_record_count:,} → {final_record_count:,}")
        logger.info(f"   Data cleaning and optimization complete")
        
        self.df_processed = df_cleaned
        return df_cleaned

    def stop_spark(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """Main execution function for standalone processing."""
    
    # Initialize processor
    processor = JobMarketDataProcessor("JobMarketAnalysis_Enhanced")
    
    try:
        # Load data (will create sample data if file doesn't exist)
        df_raw = processor.load_data("data/raw/lightcast_job_postings.csv", use_sample=True)
        
        # Assess data quality
        quality_report = processor.assess_data_quality(df_raw)
        
        # Clean and process data using optimized pipeline
        df_processed = processor.clean_and_process_data_optimized(df_raw)
        
        # Generate summary statistics
        summary_stats = processor.generate_summary_statistics(df_processed)
        
        # Save processed data
        processor.save_processed_data(df_processed)
        
        # Display key results
        print("\n=== PROCESSING COMPLETE ===")
        print(f"Total jobs processed: {summary_stats['total_jobs']:,}")
        
        if True in summary_stats['ai_ml_stats'] and False in summary_stats['ai_ml_stats']:
            ai_count = summary_stats['ai_ml_stats'][True]['count']
            traditional_count = summary_stats['ai_ml_stats'][False]['count']
            print(f"AI/ML roles: {ai_count:,} ({ai_count/summary_stats['total_jobs']*100:.1f}%)")
            print(f"Traditional roles: {traditional_count:,} ({traditional_count/summary_stats['total_jobs']*100:.1f}%)")
        
        print("\nTop Industries:")
        for industry, count in summary_stats['top_industries'][:5]:
            print(f"  {industry}: {count:,}")
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    
    finally:
        # Clean up Spark session
        processor.stop_spark()


if __name__ == "__main__":
    main()