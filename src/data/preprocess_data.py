"""
Advanced Data preprocessing and cleaning module for job market analysis.

This module handles the initial cleaning and processing of the Lightcast 
job postings dataset using PySpark for big data processing, preparing it 
for downstream analysis and visualization.

Author: Saurabh Sharma (Boston University)
Version: 2.0
Date: September 26, 2025
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# PySpark imports
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Traditional data science libraries
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedJobDataProcessor:
    """
    Advanced job market data processor using PySpark for big data processing.
    
    This class provides comprehensive data cleaning, preprocessing, and analysis
    capabilities for the Lightcast job postings dataset.
    """
    
    def __init__(self, data_path: str, spark_config: Optional[Dict] = None):
        """
        Initialize the processor with PySpark configuration.
        
        Args:
            data_path: Path to raw data file
            spark_config: Optional Spark configuration parameters
        """
        self.data_path = Path(data_path)
        self.df_spark = None
        self.df_pandas = None
        
        # Initialize Spark Session
        self.spark = self._initialize_spark(spark_config or {})
        
        # Define Lightcast schema
        self.lightcast_schema = self._define_lightcast_schema()
        
        logger.info(f"Initialized AdvancedJobDataProcessor with Spark {self.spark.version}")
    
    def _initialize_spark(self, config: Dict) -> SparkSession:
        """Initialize Spark session with optimal configurations"""
        
        default_config = {
            "spark.app.name": "JobMarketAnalysis",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.repl.eagerEval.enabled": "true",
            "spark.sql.repl.eagerEval.maxNumRows": "20"
        }
        
        # Merge user config with defaults
        final_config = {**default_config, **config}
        
        # Build Spark session
        builder = SparkSession.builder
        for key, value in final_config.items():
            builder = builder.config(key, value)
            
        return builder.getOrCreate()
    
    def _define_lightcast_schema(self) -> StructType:
        """Define comprehensive schema for Lightcast dataset"""
        
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
            StructField("CITY", StringType(), True),
            # Additional derived fields
            StructField("JOB_FUNCTION", StringType(), True),
            StructField("COMPANY_SIZE", StringType(), True),
            StructField("BENEFITS", StringType(), True)
        ])
        
    def load_data(self) -> pd.DataFrame:
        """Load raw job postings data."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records")
            return self.df
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            # Create sample data for development
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for development purposes."""
        logger.info("Creating sample data for development")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Sample data generation
        titles = [
            'Data Scientist', 'Software Engineer', 'Product Manager', 
            'Marketing Analyst', 'Sales Representative', 'HR Manager',
            'Machine Learning Engineer', 'DevOps Engineer', 'UX Designer',
            'Financial Analyst', 'Operations Manager', 'Customer Success Manager'
        ]
        
        companies = [
            'TechCorp', 'DataSystems', 'InnovateCo', 'GlobalTech', 'StartupXYZ',
            'BigCorp', 'FinanceFirst', 'HealthTech', 'EduTech', 'RetailGiant'
        ]
        
        locations = [
            'San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Austin, TX',
            'Denver, CO', 'Atlanta, GA', 'Chicago, IL', 'Boston, MA',
            'Los Angeles, CA', 'Miami, FL', 'Remote', 'Hybrid'
        ]
        
        industries = [
            'Technology', 'Healthcare', 'Finance', 'Manufacturing',
            'Retail', 'Education', 'Government', 'Non-profit'
        ]
        
        # Generate sample dataset
        sample_data = {
            'job_id': [f"JOB_{i:06d}" for i in range(n_samples)],
            'title': np.random.choice(titles, n_samples),
            'company': np.random.choice(companies, n_samples),
            'location': np.random.choice(locations, n_samples),
            'industry': np.random.choice(industries, n_samples),
            'posted_date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'salary_min': np.random.randint(40000, 120000, n_samples),
            'salary_max': np.random.randint(60000, 200000, n_samples),
            'experience_level': np.random.choice(['Entry', 'Mid', 'Senior', 'Executive'], n_samples),
            'remote_allowed': np.random.choice([True, False], n_samples),
            'education_required': np.random.choice(['Bachelor', 'Master', 'PhD', 'None'], n_samples),
            'required_skills': [self._generate_skills() for _ in range(n_samples)]
        }
        
        self.df = pd.DataFrame(sample_data)
        
        # Ensure salary_max > salary_min
        self.df['salary_max'] = np.maximum(
            self.df['salary_max'], 
            self.df['salary_min'] + 10000
        )
        
        return self.df
    
    def _generate_skills(self) -> str:
        """Generate random skill combinations."""
        all_skills = [
            'Python', 'SQL', 'Machine Learning', 'Data Analysis', 'Communication',
            'JavaScript', 'React', 'Cloud Computing', 'AWS', 'Docker',
            'Project Management', 'Leadership', 'Excel', 'Tableau', 'PowerBI',
            'Java', 'C++', 'Git', 'Agile', 'Scrum'
        ]
        n_skills = np.random.randint(2, 6)
        return ', '.join(np.random.choice(all_skills, n_skills, replace=False))
    
    def clean_salary_data(self) -> pd.DataFrame:
        """Clean and standardize salary information."""
        logger.info("Cleaning salary data")
        
        # Remove rows with missing salary data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['salary_min', 'salary_max'])
        
        # Remove unrealistic salaries (< $20k or > $500k)
        self.df = self.df[
            (self.df['salary_min'] >= 20000) & 
            (self.df['salary_max'] <= 500000)
        ]
        
        # Calculate average salary
        self.df['salary_avg'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        logger.info(f"Removed {initial_count - len(self.df)} records with invalid salaries")
        
        return self.df
    
    def standardize_locations(self) -> pd.DataFrame:
        """Standardize and clean location data."""
        logger.info("Standardizing location data")
        
        # Extract state/region information
        self.df['location_clean'] = self.df['location'].str.strip()
        
        # Identify remote work
        remote_patterns = ['remote', 'work from home', 'telecommute', 'hybrid']
        self.df['is_remote'] = self.df['location_clean'].str.lower().str.contains(
            '|'.join(remote_patterns), na=False
        )
        
        # Extract city and state
        location_parts = self.df['location_clean'].str.split(',', expand=True)
        self.df['city'] = location_parts[0] if len(location_parts.columns) > 0 else ''
        self.df['state'] = location_parts[1].str.strip() if len(location_parts.columns) > 1 else ''
        
        return self.df
    
    def classify_ai_roles(self) -> pd.DataFrame:
        """Classify jobs as AI-related or traditional."""
        logger.info("Classifying AI-related roles")
        
        ai_keywords = [
            'machine learning', 'artificial intelligence', 'ai', 'ml',
            'deep learning', 'neural network', 'nlp', 'computer vision',
            'data science', 'data scientist', 'ml engineer', 'ai engineer'
        ]
        
        # Check title and skills for AI keywords
        ai_pattern = '|'.join(ai_keywords)
        
        self.df['ai_related'] = (
            self.df['title'].str.lower().str.contains(ai_pattern, na=False) |
            self.df['required_skills'].str.lower().str.contains(ai_pattern, na=False)
        )
        
        ai_count = self.df['ai_related'].sum()
        logger.info(f"Identified {ai_count} AI-related positions ({ai_count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def extract_experience_years(self) -> pd.DataFrame:
        """Convert experience levels to numeric years."""
        logger.info("Converting experience levels to years")
        
        experience_mapping = {
            'Entry': 1,
            'Mid': 4,
            'Senior': 8,
            'Executive': 15
        }
        
        self.df['experience_years'] = self.df['experience_level'].map(experience_mapping)
        
        return self.df
    
    def process_all(self) -> pd.DataFrame:
        """Run complete data processing pipeline."""
        logger.info("Starting complete data processing pipeline")
        
        # Load data
        self.load_data()
        
        # Apply all cleaning steps
        self.clean_salary_data()
        self.standardize_locations()
        self.classify_ai_roles()
        self.extract_experience_years()
        
        # Add processed timestamp
        self.df['processed_date'] = datetime.now()
        
        logger.info(f"Processing complete. Final dataset: {len(self.df)} records")
        
        return self.df
    
    def save_processed_data(self, output_path: str) -> None:
        """Save processed data to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
    
    def generate_summary_stats(self) -> dict:
        """Generate summary statistics for the dataset."""
        if self.df is None:
            return {}
        
        stats = {
            'total_records': len(self.df),
            'salary_stats': {
                'mean': self.df['salary_avg'].mean(),
                'median': self.df['salary_avg'].median(),
                'std': self.df['salary_avg'].std(),
                'min': self.df['salary_avg'].min(),
                'max': self.df['salary_avg'].max()
            },
            'ai_percentage': self.df['ai_related'].mean() * 100,
            'remote_percentage': self.df['is_remote'].mean() * 100,
            'top_locations': self.df['city'].value_counts().head().to_dict(),
            'top_industries': self.df['industry'].value_counts().head().to_dict()
        }
        
        return stats


def main():
    """Main function for standalone execution."""
    # Initialize processor
    processor = JobDataProcessor('data/raw/lightcast_job_postings.csv')
    
    # Process data
    df = processor.process_all()
    
    # Save processed data
    processor.save_processed_data('data/processed/clean_job_data.csv')
    
    # Generate and display summary
    summary = processor.generate_summary_stats()
    print("\nDataset Summary:")
    print(f"Total Records: {summary['total_records']:,}")
    print(f"Average Salary: ${summary['salary_stats']['mean']:,.0f}")
    print(f"AI-Related Jobs: {summary['ai_percentage']:.1f}%")
    print(f"Remote Jobs: {summary['remote_percentage']:.1f}%")
    
    return df


if __name__ == "__main__":
    main()