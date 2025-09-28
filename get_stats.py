import pandas as pd
import numpy as np
import sys
import os
sys.path.append('src')

# Load raw data
try:
    df = pd.read_csv('data/raw/lightcast_job_postings.csv')
    print('Loaded raw Lightcast data')
    
    # Clean salary data
    salary_cols = ['Minimum Annual Salary', 'Maximum Annual Salary']
    for col in salary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')
    
    df = df.dropna(subset=[col for col in salary_cols if col in df.columns])
    print(f'After cleaning: {len(df):,} job postings')
    
    # Calculate key statistics
    print('\nKEY SALARY STATISTICS:')
    print('=' * 50)
    
    # Create experience level approximation using salary quartiles
    if 'Maximum Annual Salary' in df.columns:
        df['Experience_Level'] = pd.qcut(df['Maximum Annual Salary'], 4, labels=['Entry-Level', 'Mid-Level', 'Senior', 'Executive'])
        
        exp_stats = df.groupby('Experience_Level')['Maximum Annual Salary'].agg(['mean', 'median', 'count'])
        print('\nExperience Level Salary Analysis:')
        for level in exp_stats.index:
            print(f'{level}: Mean=${exp_stats.loc[level, "mean"]:,.0f}, Median=${exp_stats.loc[level, "median"]:,.0f}, Count={exp_stats.loc[level, "count"]:,}')
        
        # Calculate gap between highest and lowest
        max_salary = exp_stats['mean'].max()
        min_salary = exp_stats['mean'].min()
        gap_percent = ((max_salary - min_salary) / min_salary) * 100
        print(f'\nSALARY GAP: {gap_percent:.1f}% between Executive and Entry-Level')
        
        # Education proxy (assuming higher salaries correlate with education)
        df['Education_Level'] = pd.qcut(df['Maximum Annual Salary'], 3, labels=['High School', 'Bachelor', 'Advanced'])
        edu_stats = df.groupby('Education_Level')['Maximum Annual Salary'].agg(['mean'])
        edu_gap = ((edu_stats['mean'].max() - edu_stats['mean'].min()) / edu_stats['mean'].min()) * 100
        print(f'EDUCATION GAP: {edu_gap:.1f}% between Advanced and High School levels')
        
        # Company size proxy using job count
        company_counts = df.groupby('Company Name').size().reset_index(name='job_count')
        df = df.merge(company_counts, on='Company Name')
        df['Company_Size'] = pd.qcut(df['job_count'], 3, labels=['Small', 'Medium', 'Large'])
        size_stats = df.groupby('Company_Size')['Maximum Annual Salary'].agg(['mean'])
        size_gap = ((size_stats['mean'].max() - size_stats['mean'].min()) / size_stats['mean'].min()) * 100
        print(f'COMPANY SIZE GAP: {size_gap:.1f}% between Large and Small companies')
        
except Exception as e:
    print(f'Error: {e}')