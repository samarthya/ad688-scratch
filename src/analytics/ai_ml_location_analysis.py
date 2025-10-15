"""
AI/ML Job Analysis by Location using Specialized Skills

This module identifies AI/ML jobs based on specialized skills keywords
and analyzes their distribution across top locations.
"""

import pandas as pd
import json
from typing import List, Dict, Tuple
import re


# AI/ML Keywords to search in specialized skills
AI_ML_KEYWORDS = [
    # Core AI/ML
    'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
    'ai', 'ml', 'nlp', 'natural language processing', 'computer vision',

    # Data Science
    'data science', 'data scientist', 'predictive modeling', 'statistical modeling',
    'advanced analytics', 'predictive analytics', 'statistical analysis',

    # ML Techniques
    'supervised learning', 'unsupervised learning', 'reinforcement learning',
    'classification', 'regression', 'clustering', 'recommendation system',
    'time series', 'forecasting', 'anomaly detection',

    # ML Frameworks & Tools
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'spark mllib',
    'xgboost', 'lightgbm', 'catboost',

    # ML Operations
    'mlops', 'ml ops', 'model deployment', 'feature engineering',
    'model training', 'hyperparameter tuning',

    # Big Data ML
    'distributed machine learning', 'scalable machine learning',
    'big data analytics', 'data mining',

    # Specific Domains
    'generative ai', 'llm', 'large language model', 'chatbot', 'conversational ai',
    'image recognition', 'speech recognition', 'sentiment analysis'
]


def extract_skills_list(skills_str: str) -> List[str]:
    """
    Extract skills list from JSON string.

    Args:
        skills_str: JSON string containing list of skills

    Returns:
        List of skill strings, or empty list if parsing fails
    """
    if pd.isna(skills_str) or not skills_str:
        return []

    try:
        # Try to parse as JSON
        skills = json.loads(skills_str)
        if isinstance(skills, list):
            return [str(skill).lower() for skill in skills]
        return []
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, return empty list
        return []


def has_ai_ml_skills(skills_list: List[str], keywords: List[str] = AI_ML_KEYWORDS) -> bool:
    """
    Check if any skill in the list matches AI/ML keywords.

    Args:
        skills_list: List of skill strings (lowercase)
        keywords: List of AI/ML keywords to search for

    Returns:
        True if any AI/ML keyword is found
    """
    if not skills_list:
        return False

    # Join all skills into one string for easier searching
    skills_text = ' '.join(skills_list)

    # Check if any keyword appears in skills
    for keyword in keywords:
        if keyword.lower() in skills_text:
            return True

    return False


def get_matching_ai_ml_skills(skills_list: List[str], keywords: List[str] = AI_ML_KEYWORDS) -> List[str]:
    """
    Get list of AI/ML skills that match keywords.

    Args:
        skills_list: List of skill strings (lowercase)
        keywords: List of AI/ML keywords to search for

    Returns:
        List of matching skills
    """
    if not skills_list:
        return []

    matching_skills = []
    for skill in skills_list:
        for keyword in keywords:
            if keyword.lower() in skill.lower():
                matching_skills.append(skill)
                break

    return matching_skills


def analyze_ai_ml_jobs_by_location(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Analyze AI/ML jobs by location using specialized skills.

    Args:
        df: DataFrame with 'specialized_skills' and 'city_name' columns
        top_n: Number of top locations to include

    Returns:
        DataFrame with columns: city_name, total_jobs, ai_ml_jobs, ai_ml_percentage, avg_salary
    """
    # Make a copy to avoid modifying original
    df_analysis = df.copy()

    # Extract specialized skills lists
    print("Extracting specialized skills...")
    df_analysis['skills_list'] = df_analysis['specialized_skills'].apply(extract_skills_list)

    # Identify AI/ML jobs
    print("Identifying AI/ML jobs...")
    df_analysis['is_ai_ml'] = df_analysis['skills_list'].apply(has_ai_ml_skills)

    # Get matching skills for debugging
    df_analysis['ai_ml_skills_found'] = df_analysis['skills_list'].apply(get_matching_ai_ml_skills)

    # Group by city
    print("Analyzing by location...")
    city_stats = df_analysis.groupby('city_name').agg({
        'job_id': 'count',  # Total jobs
        'is_ai_ml': 'sum',  # AI/ML jobs count
        'salary_avg': 'median'  # Median salary
    }).reset_index()

    city_stats.columns = ['city_name', 'total_jobs', 'ai_ml_jobs', 'median_salary']

    # Calculate percentage
    city_stats['ai_ml_percentage'] = (city_stats['ai_ml_jobs'] / city_stats['total_jobs'] * 100).round(1)

    # Sort by total jobs and get top N
    city_stats = city_stats.sort_values('total_jobs', ascending=False).head(top_n)

    # Sort by AI/ML jobs for final output
    city_stats = city_stats.sort_values('ai_ml_jobs', ascending=False)

    return city_stats


def get_top_ai_ml_skills_by_location(df: pd.DataFrame, top_cities: int = 5, top_skills: int = 10) -> Dict[str, List[Tuple[str, int]]]:
    """
    Get top AI/ML skills for each top location.

    Args:
        df: DataFrame with specialized_skills and city_name
        top_cities: Number of top cities to analyze
        top_skills: Number of top skills to return per city

    Returns:
        Dictionary mapping city_name to list of (skill, count) tuples
    """
    # Extract skills
    df_analysis = df.copy()
    df_analysis['skills_list'] = df_analysis['specialized_skills'].apply(extract_skills_list)
    df_analysis['ai_ml_skills_found'] = df_analysis['skills_list'].apply(get_matching_ai_ml_skills)

    # Filter to only AI/ML jobs
    df_ai_ml = df_analysis[df_analysis['ai_ml_skills_found'].apply(len) > 0].copy()

    # Get top cities by AI/ML job count
    top_cities_list = df_ai_ml.groupby('city_name').size().sort_values(ascending=False).head(top_cities).index.tolist()

    results = {}
    for city in top_cities_list:
        city_df = df_ai_ml[df_ai_ml['city_name'] == city]

        # Flatten all AI/ML skills for this city
        all_skills = []
        for skills in city_df['ai_ml_skills_found']:
            all_skills.extend(skills)

        # Count skill frequency
        skill_counts = pd.Series(all_skills).value_counts().head(top_skills)
        results[city] = list(skill_counts.items())

    return results


def create_ai_ml_location_summary(df: pd.DataFrame, top_n: int = 5) -> str:
    """
    Create a text summary of AI/ML jobs by location.

    Args:
        df: DataFrame with job data
        top_n: Number of top locations to include

    Returns:
        Formatted summary string
    """
    stats = analyze_ai_ml_jobs_by_location(df, top_n)

    summary = f"# AI/ML Jobs by Location (Top {top_n} Cities)\n\n"
    summary += "Based on specialized skills analysis:\n\n"

    for idx, row in stats.iterrows():
        summary += f"## {row['city_name']}\n"
        summary += f"- Total Jobs: {row['total_jobs']:,}\n"
        summary += f"- AI/ML Jobs: {int(row['ai_ml_jobs']):,}\n"
        summary += f"- AI/ML Percentage: {row['ai_ml_percentage']:.1f}%\n"
        summary += f"- Median Salary: ${row['median_salary']:,.0f}\n\n"

    return summary

