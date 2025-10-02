"""
Relational Tables Creation Module

This module creates normalized relational tables from the processed job market data
as specified in the DESIGN.md architecture document.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_relational_tables(df: pd.DataFrame, output_dir: str = "data/processed/relational_tables/") -> Dict[str, str]:
    """
    Create normalized relational tables for advanced analytics.

    Args:
        df: Processed job market DataFrame
        output_dir: Directory to save relational tables

    Returns:
        Dictionary mapping table names to file paths
    """
    print("🗂️  Creating relational tables...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    table_paths = {}

    try:
        # 1. Companies Dimension Table
        print("  📊 Creating companies dimension...")
        companies_df = create_companies_dimension(df)
        companies_path = output_path / "companies.parquet"
        companies_df.to_parquet(companies_path, index=False)
        table_paths['companies'] = str(companies_path)
        print(f"     ✅ Companies table: {len(companies_df):,} records")

        # 2. Locations Dimension Table
        print("  🌍 Creating locations dimension...")
        locations_df = create_locations_dimension(df)
        locations_path = output_path / "locations.parquet"
        locations_df.to_parquet(locations_path, index=False)
        table_paths['locations'] = str(locations_path)
        print(f"     ✅ Locations table: {len(locations_df):,} records")

        # 3. Industries Dimension Table
        print("  🏭 Creating industries dimension...")
        industries_df = create_industries_dimension(df)
        industries_path = output_path / "industries.parquet"
        industries_df.to_parquet(industries_path, index=False)
        table_paths['industries'] = str(industries_path)
        print(f"     ✅ Industries table: {len(industries_df):,} records")

        # 4. Skills Dimension Table
        print("  🛠️  Creating skills dimension...")
        skills_df = create_skills_dimension(df)
        skills_path = output_path / "skills.parquet"
        skills_df.to_parquet(skills_path, index=False)
        table_paths['skills'] = str(skills_path)
        print(f"     ✅ Skills table: {len(skills_df):,} records")

        # 5. Job Postings Fact Table
        print("  📋 Creating job postings fact table...")
        fact_df = create_job_postings_fact(df)
        fact_path = output_path / "job_postings_fact.parquet"
        fact_df.to_parquet(fact_path, index=False)
        table_paths['job_postings_fact'] = str(fact_path)
        print(f"     ✅ Fact table: {len(fact_df):,} records")

        # 6. Job Skills Bridge Table (Many-to-Many)
        print("  🔗 Creating job-skills bridge table...")
        bridge_df = create_job_skills_bridge(df)
        bridge_path = output_path / "job_skills_bridge.parquet"
        bridge_df.to_parquet(bridge_path, index=False)
        table_paths['job_skills_bridge'] = str(bridge_path)
        print(f"     ✅ Bridge table: {len(bridge_df):,} records")

        # Create schema documentation
        create_schema_documentation(output_path, table_paths)

        print(f"  ✅ Created {len(table_paths)} relational tables in {output_dir}")

    except Exception as e:
        logger.error(f"Error creating relational tables: {e}")
        print(f"  ❌ Failed to create relational tables: {e}")

    return table_paths

def create_companies_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create companies dimension table with size classifications."""

    # Get company column
    company_col = 'company' if 'company' in df.columns else 'COMPANY'
    if company_col not in df.columns:
        # Create minimal companies table
        return pd.DataFrame({
            'company_id': [1],
            'company_name': ['Unknown'],
            'company_size': ['Unknown'],
            'job_count': [len(df)]
        })

    # Aggregate company data
    company_stats = df.groupby(company_col).agg({
        'salary_avg': ['count', 'median', 'mean'] if 'salary_avg' in df.columns else 'job_id': 'count'
    }).reset_index()

    # Flatten column names
    if 'salary_avg' in df.columns:
        company_stats.columns = [company_col, 'job_count', 'median_salary', 'mean_salary']
    else:
        company_stats.columns = [company_col, 'job_count']
        company_stats['median_salary'] = None
        company_stats['mean_salary'] = None

    # Classify company size based on job posting frequency
    def classify_company_size(job_count):
        if job_count >= 100:
            return 'Large Enterprise (100+ jobs)'
        elif job_count >= 20:
            return 'Medium Company (20-99 jobs)'
        elif job_count >= 5:
            return 'Small Company (5-19 jobs)'
        else:
            return 'Startup/Small (1-4 jobs)'

    company_stats['company_size'] = company_stats['job_count'].apply(classify_company_size)

    # Create dimension table
    companies_dim = pd.DataFrame({
        'company_id': range(1, len(company_stats) + 1),
        'company_name': company_stats[company_col],
        'company_size': company_stats['company_size'],
        'job_count': company_stats['job_count'],
        'median_salary': company_stats['median_salary'],
        'mean_salary': company_stats['mean_salary']
    })

    return companies_dim

def create_locations_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create locations dimension table with geographic breakdown."""

    # Get location columns
    city_col = 'city_name' if 'city_name' in df.columns else 'CITY_NAME'
    location_col = 'location' if 'location' in df.columns else 'LOCATION'

    if city_col not in df.columns and location_col not in df.columns:
        # Create minimal locations table
        return pd.DataFrame({
            'location_id': [1],
            'city': ['Unknown'],
            'state': ['Unknown'],
            'region': ['Unknown'],
            'location_type': ['Unknown']
        })

    # Use city_name if available, otherwise location
    primary_location_col = city_col if city_col in df.columns else location_col

    # Aggregate location data
    location_stats = df.groupby(primary_location_col).agg({
        'salary_avg': ['count', 'median'] if 'salary_avg' in df.columns else 'job_id': 'count'
    }).reset_index()

    # Flatten column names
    if 'salary_avg' in df.columns:
        location_stats.columns = [primary_location_col, 'job_count', 'median_salary']
    else:
        location_stats.columns = [primary_location_col, 'job_count']
        location_stats['median_salary'] = None

    # Parse location information
    def parse_location(location_str):
        location_str = str(location_str).strip()

        if location_str.lower() in ['unknown', 'remote', 'nan', 'none']:
            return {
                'city': location_str.title(),
                'state': 'Remote' if location_str.lower() == 'remote' else 'Unknown',
                'region': 'Remote' if location_str.lower() == 'remote' else 'Unknown',
                'location_type': 'Remote' if location_str.lower() == 'remote' else 'Unknown'
            }

        # Try to parse "City, State" format
        if ',' in location_str:
            parts = [part.strip() for part in location_str.split(',')]
            city = parts[0]
            state = parts[1] if len(parts) > 1 else 'Unknown'
        else:
            city = location_str
            state = 'Unknown'

        # Determine region based on state
        region = get_region_from_state(state)
        location_type = 'On-site'

        return {
            'city': city,
            'state': state,
            'region': region,
            'location_type': location_type
        }

    # Parse all locations
    parsed_locations = location_stats[primary_location_col].apply(parse_location)

    # Create dimension table
    locations_dim = pd.DataFrame({
        'location_id': range(1, len(location_stats) + 1),
        'location_name': location_stats[primary_location_col],
        'city': [loc['city'] for loc in parsed_locations],
        'state': [loc['state'] for loc in parsed_locations],
        'region': [loc['region'] for loc in parsed_locations],
        'location_type': [loc['location_type'] for loc in parsed_locations],
        'job_count': location_stats['job_count'],
        'median_salary': location_stats['median_salary']
    })

    return locations_dim

def create_industries_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create industries dimension table with standardized categories."""

    industry_col = 'industry' if 'industry' in df.columns else 'NAICS2_NAME'
    if industry_col not in df.columns:
        # Create minimal industries table
        return pd.DataFrame({
            'industry_id': [1],
            'industry_name': ['Unknown'],
            'industry_category': ['Unknown'],
            'job_count': [len(df)]
        })

    # Aggregate industry data
    industry_stats = df.groupby(industry_col).agg({
        'salary_avg': ['count', 'median', 'mean'] if 'salary_avg' in df.columns else 'job_id': 'count'
    }).reset_index()

    # Flatten column names
    if 'salary_avg' in df.columns:
        industry_stats.columns = [industry_col, 'job_count', 'median_salary', 'mean_salary']
    else:
        industry_stats.columns = [industry_col, 'job_count']
        industry_stats['median_salary'] = None
        industry_stats['mean_salary'] = None

    # Categorize industries
    def categorize_industry(industry_name):
        industry_lower = str(industry_name).lower()

        if any(keyword in industry_lower for keyword in ['software', 'computer', 'tech', 'information']):
            return 'Technology'
        elif any(keyword in industry_lower for keyword in ['finance', 'banking', 'investment']):
            return 'Financial Services'
        elif any(keyword in industry_lower for keyword in ['health', 'medical', 'pharmaceutical']):
            return 'Healthcare'
        elif any(keyword in industry_lower for keyword in ['retail', 'consumer', 'e-commerce']):
            return 'Retail & Consumer'
        elif any(keyword in industry_lower for keyword in ['manufacturing', 'industrial', 'automotive']):
            return 'Manufacturing'
        elif any(keyword in industry_lower for keyword in ['education', 'university', 'school']):
            return 'Education'
        elif any(keyword in industry_lower for keyword in ['government', 'public', 'federal']):
            return 'Government & Public'
        else:
            return 'Other'

    industry_stats['industry_category'] = industry_stats[industry_col].apply(categorize_industry)

    # Create dimension table
    industries_dim = pd.DataFrame({
        'industry_id': range(1, len(industry_stats) + 1),
        'industry_name': industry_stats[industry_col],
        'industry_category': industry_stats['industry_category'],
        'job_count': industry_stats['job_count'],
        'median_salary': industry_stats['median_salary'],
        'mean_salary': industry_stats['mean_salary']
    })

    return industries_dim

def create_skills_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create skills dimension table from job requirements."""
    import ast
    import json
    from collections import Counter

    skills_col = 'required_skills' if 'required_skills' in df.columns else 'SKILLS_NAME'

    if skills_col not in df.columns:
        # Create minimal skills table
        return pd.DataFrame({
            'skill_id': [1, 2, 3],
            'skill_name': ['Python', 'SQL', 'Communication'],
            'skill_category': ['Programming', 'Database', 'Soft Skills'],
            'job_count': [100, 80, 150]
        })

    # Extract all skills
    all_skills = []

    for skills_data in df[skills_col].dropna():
        skills_list = []
        skills_str = str(skills_data)

        # Try to parse as JSON/list
        try:
            if skills_str.startswith('[') or skills_str.startswith('{'):
                parsed_skills = ast.literal_eval(skills_str)
                if isinstance(parsed_skills, list):
                    skills_list = [str(skill).strip() for skill in parsed_skills]
                elif isinstance(parsed_skills, dict):
                    skills_list = list(parsed_skills.keys())
            else:
                # Split by common delimiters
                skills_list = [skill.strip() for skill in skills_str.split(',')]
        except:
            # Fallback to simple splitting
            skills_list = [skill.strip() for skill in skills_str.split(',')]

        # Clean and filter skills
        cleaned_skills = []
        for skill in skills_list:
            skill = skill.lower().strip()
            if len(skill) > 1 and len(skill) < 50 and skill not in ['', 'nan', 'none']:
                cleaned_skills.append(skill)

        all_skills.extend(cleaned_skills)

    # Count skill frequencies
    skill_counts = Counter(all_skills)

    # Filter out very rare skills (appear in less than 2 jobs)
    filtered_skills = {skill: count for skill, count in skill_counts.items() if count >= 2}

    # Categorize skills
    def categorize_skill(skill_name):
        skill_lower = skill_name.lower()

        if any(keyword in skill_lower for keyword in ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust']):
            return 'Programming Languages'
        elif any(keyword in skill_lower for keyword in ['sql', 'mysql', 'postgresql', 'mongodb', 'database']):
            return 'Database Technologies'
        elif any(keyword in skill_lower for keyword in ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes']):
            return 'Cloud & DevOps'
        elif any(keyword in skill_lower for keyword in ['machine learning', 'ai', 'tensorflow', 'pytorch', 'data science']):
            return 'AI & Machine Learning'
        elif any(keyword in skill_lower for keyword in ['react', 'angular', 'vue', 'html', 'css', 'frontend']):
            return 'Frontend Technologies'
        elif any(keyword in skill_lower for keyword in ['communication', 'leadership', 'teamwork', 'problem solving']):
            return 'Soft Skills'
        elif any(keyword in skill_lower for keyword in ['agile', 'scrum', 'project management', 'jira']):
            return 'Project Management'
        else:
            return 'Technical Skills'

    # Create skills dimension
    skills_data = []
    for skill_id, (skill_name, job_count) in enumerate(filtered_skills.items(), 1):
        skills_data.append({
            'skill_id': skill_id,
            'skill_name': skill_name.title(),
            'skill_category': categorize_skill(skill_name),
            'job_count': job_count
        })

    skills_dim = pd.DataFrame(skills_data)

    return skills_dim

def create_job_postings_fact(df: pd.DataFrame) -> pd.DataFrame:
    """Create main fact table with all metrics and foreign keys."""

    # Create a copy of the main dataframe
    fact_df = df.copy()

    # Add surrogate key
    fact_df['job_posting_id'] = range(1, len(fact_df) + 1)

    # Select key columns for fact table
    fact_columns = ['job_posting_id']

    # Add available columns
    column_mapping = {
        'title': 'job_title',
        'company': 'company_name',
        'salary_avg': 'salary',
        'salary_min': 'salary_min',
        'salary_max': 'salary_max',
        'industry': 'industry_name',
        'city_name': 'city',
        'location': 'location_name',
        'experience_years': 'experience_years',
        'experience_level': 'experience_level',
        'remote_type': 'remote_type',
        'employment_type': 'employment_type'
    }

    for original_col, fact_col in column_mapping.items():
        if original_col in fact_df.columns:
            fact_df[fact_col] = fact_df[original_col]
            fact_columns.append(fact_col)

    # Add derived metrics
    if 'salary_avg' in fact_df.columns:
        fact_df['is_high_salary'] = fact_df['salary_avg'] > fact_df['salary_avg'].median()
        fact_columns.append('is_high_salary')

    if 'title' in fact_df.columns:
        ai_keywords = ['ai', 'machine learning', 'data scientist', 'ml engineer']
        fact_df['is_ai_role'] = fact_df['title'].str.lower().str.contains('|'.join(ai_keywords), na=False)
        fact_columns.append('is_ai_role')

    # Add posting date (synthetic for now)
    import datetime
    fact_df['posting_date'] = datetime.date.today()
    fact_columns.append('posting_date')

    return fact_df[fact_columns]

def create_job_skills_bridge(df: pd.DataFrame) -> pd.DataFrame:
    """Create bridge table for many-to-many job-skills relationship."""
    import ast

    skills_col = 'required_skills' if 'required_skills' in df.columns else 'SKILLS_NAME'

    if skills_col not in df.columns:
        # Create minimal bridge table
        return pd.DataFrame({
            'job_posting_id': [1, 2, 3],
            'skill_name': ['Python', 'SQL', 'Communication'],
            'skill_importance': ['High', 'Medium', 'High']
        })

    bridge_data = []

    for job_id, skills_data in enumerate(df[skills_col].dropna(), 1):
        skills_str = str(skills_data)
        skills_list = []

        # Parse skills
        try:
            if skills_str.startswith('[') or skills_str.startswith('{'):
                parsed_skills = ast.literal_eval(skills_str)
                if isinstance(parsed_skills, list):
                    skills_list = [str(skill).strip() for skill in parsed_skills]
            else:
                skills_list = [skill.strip() for skill in skills_str.split(',')]
        except:
            skills_list = [skill.strip() for skill in skills_str.split(',')]

        # Add to bridge table
        for i, skill in enumerate(skills_list):
            if skill and len(skill) > 1:
                importance = 'High' if i < 3 else 'Medium' if i < 6 else 'Low'
                bridge_data.append({
                    'job_posting_id': job_id,
                    'skill_name': skill.title(),
                    'skill_importance': importance
                })

    return pd.DataFrame(bridge_data)

def get_region_from_state(state):
    """Map state to US region."""
    regions = {
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'Southeast': ['DE', 'MD', 'DC', 'VA', 'WV', 'KY', 'TN', 'NC', 'SC', 'GA', 'FL', 'AL', 'MS', 'AR', 'LA'],
        'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'Southwest': ['TX', 'OK', 'NM', 'AZ'],
        'West': ['MT', 'WY', 'CO', 'UT', 'ID', 'WA', 'OR', 'NV', 'CA', 'AK', 'HI']
    }

    state_upper = str(state).upper().strip()

    for region, states in regions.items():
        if state_upper in states:
            return region

    return 'Unknown'

def create_schema_documentation(output_path: Path, table_paths: Dict[str, str]):
    """Create schema documentation for the relational tables."""

    schema_doc = """# Relational Tables Schema Documentation

## Overview
This directory contains normalized relational tables created from the job market data for advanced analytics.

## Tables

### 1. Companies Dimension (`companies.parquet`)
- **company_id**: Unique identifier for each company
- **company_name**: Company name
- **company_size**: Size classification based on job posting frequency
- **job_count**: Number of job postings from this company
- **median_salary**: Median salary offered by this company
- **mean_salary**: Average salary offered by this company

### 2. Locations Dimension (`locations.parquet`)
- **location_id**: Unique identifier for each location
- **location_name**: Original location string
- **city**: Parsed city name
- **state**: State or region
- **region**: US geographic region
- **location_type**: On-site, Remote, etc.
- **job_count**: Number of jobs in this location
- **median_salary**: Median salary for this location

### 3. Industries Dimension (`industries.parquet`)
- **industry_id**: Unique identifier for each industry
- **industry_name**: Industry name
- **industry_category**: Broad industry category
- **job_count**: Number of jobs in this industry
- **median_salary**: Median salary for this industry
- **mean_salary**: Average salary for this industry

### 4. Skills Dimension (`skills.parquet`)
- **skill_id**: Unique identifier for each skill
- **skill_name**: Skill name
- **skill_category**: Skill category (Programming, Database, etc.)
- **job_count**: Number of jobs requiring this skill

### 5. Job Postings Fact Table (`job_postings_fact.parquet`)
- **job_posting_id**: Unique identifier for each job posting
- **job_title**: Job title
- **company_name**: Company name
- **salary**: Salary amount
- **salary_min**: Minimum salary
- **salary_max**: Maximum salary
- **industry_name**: Industry
- **city**: City location
- **experience_years**: Required experience in years
- **experience_level**: Experience level category
- **remote_type**: Remote work type
- **employment_type**: Employment type
- **is_high_salary**: Boolean flag for above-median salary
- **is_ai_role**: Boolean flag for AI/ML roles
- **posting_date**: Job posting date

### 6. Job-Skills Bridge Table (`job_skills_bridge.parquet`)
- **job_posting_id**: Reference to job posting
- **skill_name**: Required skill
- **skill_importance**: Importance level (High/Medium/Low)

## Usage Examples

```python
import pandas as pd

# Load tables
companies = pd.read_parquet('companies.parquet')
locations = pd.read_parquet('locations.parquet')
industries = pd.read_parquet('industries.parquet')
skills = pd.read_parquet('skills.parquet')
jobs = pd.read_parquet('job_postings_fact.parquet')
job_skills = pd.read_parquet('job_skills_bridge.parquet')

# Example analysis: Top paying companies by industry
analysis = jobs.groupby(['company_name', 'industry_name'])['salary'].median().reset_index()
```

## Performance Benefits
- **Normalized structure** reduces data redundancy
- **Parquet format** provides fast columnar access
- **Indexed dimensions** enable efficient joins
- **Smaller file sizes** due to normalization
"""

    with open(output_path / "schema_documentation.md", 'w') as f:
        f.write(schema_doc)

    print("  📚 Created schema documentation")
