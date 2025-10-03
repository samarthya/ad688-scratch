# Technical Design & Implementation Guide

**Current Architecture Documentation** - How the job market analytics system works

> See [README.md](README.md) for project overview
> Last Updated: October 2025

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Column Standardization](#column-standardization)
4. [System Components](#system-components)
5. [Usage Patterns](#usage-patterns)
6. [Homepage Metrics](#homepage-metrics)

---

## Architecture Overview

### Core Principle: Process Once, Use Many Times

```bash
ONE-TIME (scripts/create_processed_data.py):
  Raw CSV (72K records) → Process → Parquet (32K clean records)

RUNTIME (All QMD/notebooks):
  Load Parquet → Use directly (NO processing)
```

### Design Principles

1. **Process Once**: Dedicated script processes raw data, saves as Parquet
2. **Load Fast**: Parquet loads instantly vs CSV processing
3. **Standardized Columns**: All snake_case, no runtime mapping
4. **Single Salary Column**: `salary_avg` is source of truth
5. **Simple**: Pandas + Parquet (no distributed processing needed)

### Current Dataset

- **Source**: Lightcast job postings (72,498 raw)
- **Processed**: 32,364 with validated salary data
- **Format**: Parquet (columnar, compressed)
- **Columns**: 132 standardized snake_case
- **Key**: `salary_avg`, `title`, `city_name`, `naics2_name`

---

## Data Processing Pipeline

### ONE-TIME: Create Processed Data

**Script**: `scripts/create_processed_data.py`

**Steps**:

1. Load `data/raw/lightcast_job_postings.csv`
2. Standardize columns to snake_case
3. Compute `salary_avg` from `salary_from`/`salary_to`
4. Impute missing salaries (by city, experience, title, occupation)
5. Validate range (20K-500K USD)
6. Clean experience, location, industry
7. Save `data/processed/job_market_processed.parquet`

### RUNTIME: Load Processed Data

**Module**: `src/data/website_processor.py`

```python
from src.data.website_processor import load_and_process_data

df, summary = load_and_process_data()
# Loads Parquet instantly, no processing
```

---

## Column Standardization

### All columns use snake_case

| Raw | Processed |
|-----|-----------|
| `SALARY_FROM` | `salary_min` |
| `SALARY_TO` | `salary_max` |
| `SALARY_AVG` | `salary_avg` (computed) |
| `TITLE_NAME` | `title` |
| `CITY_NAME` | `city_name` |
| `NAICS2_NAME` | `naics2_name` |
| `SKILLS_NAME` | `skills_name` |

### Standard Column Lookup

**Config**: `src/config/column_mapping.py`

```python
ANALYSIS_COLUMNS = {
    'salary': 'salary_avg',
    'title': 'title',
    'city': 'city_name',
    'industry': 'naics2_name',
}

# Usage
from src.config.column_mapping import get_analysis_column
salary_col = get_analysis_column('salary')  # 'salary_avg'
```

---

## System Components

### Data Processing (`src/data/`)

**`website_processor.py`**:

- `load_and_process_data()` - Loads Parquet
- `standardize_columns()` - Column standardization
- `get_data_summary()` - Data statistics
- `generate_analysis_results()` - Run all analyses
- `generate_website_figures()` - Create visualizations

### Visualization (`src/visualization/`)

**`charts.py` - SalaryVisualizer**:

- `get_experience_progression_analysis()`
- `get_education_analysis()`
- `get_skills_analysis()`
- `get_industry_salary_analysis()`
- `plot_salary_distribution()`
- `plot_ai_salary_comparison()`
- `create_correlation_matrix()`

**`key_findings_dashboard.py` - KeyFindingsDashboard**:

- `create_key_metrics_cards()`
- `create_career_progression_analysis()`
- `create_complete_intelligence_dashboard()`

### Analytics (`src/analytics/`)

**`salary_models.py` - SalaryAnalyticsModels**:

- Multiple linear regression (salary prediction)
- Binary classification (above/below avg)
- Feature engineering and reporting

**`nlp_analysis.py` - JobMarketNLPAnalyzer**:

- Skills extraction
- Topic clustering
- Word cloud generation

---

## Usage Patterns

### In QMD Files

```python
# Load data
from src.data.website_processor import load_and_process_data
df, summary = load_and_process_data()

# Create visualizations
from src.visualization.charts import SalaryVisualizer
visualizer = SalaryVisualizer(df)
fig = visualizer.plot_salary_distribution()
fig.show()
```

### In Notebooks

```python
import pandas as pd

# Load Parquet directly
df = pd.read_parquet('data/processed/job_market_processed.parquet')

# Use standardized columns
salary = df['salary_avg']
city = df['city_name']
```

### Analytics Models

```python
from src.analytics.salary_models import SalaryAnalyticsModels

models = SalaryAnalyticsModels(df)
results = models.run_complete_analysis()
```

---

## Homepage Metrics

### Experience Gap (90%)

```python
entry = df[df['min_years_experience'] <= 2]['salary_avg'].median()
senior = df[df['min_years_experience'] >= 8]['salary_avg'].median()
gap = ((senior - entry) / entry) * 100  # 90%
```

Senior roles pay 90% more than entry-level

### Education Premium (9%)

```python
bachelors = df[df['education'].str.contains('Bachelor')]['salary_avg'].median()
masters = df[df['education'].str.contains('Master')]['salary_avg'].median()
premium = ((masters - bachelors) / bachelors) * 100  # 9%
```

Master's holders earn 9% more than Bachelor's

### Company Size Gap (32%)

```python
small = df[df['company_size'] <= 50]['salary_avg'].median()
large = df[df['company_size'] >= 1000]['salary_avg'].median()
gap = ((large - small) / small) * 100  # 32%
```

Large companies pay 32% more than small

### Salary Growth (3.0x)

```python
entry = df['salary_avg'].quantile(0.25)
peak = df['salary_avg'].quantile(0.75)
growth = peak / entry  # 3.0x
```

Salaries grow 3x from entry to senior

### Remote Availability (25%)

```python
total = len(df)
remote = len(df[df['remote_type'].str.contains('remote', case=False)])
percentage = (remote / total) * 100  # 25%
```

25% of jobs offer remote work

---

## File Structure

```bash
ad688-scratch/
├── data/
│   ├── raw/lightcast_job_postings.csv          # Source (72K)
│   └── processed/job_market_processed.parquet  # Clean (32K) ← USE THIS
│
├── scripts/create_processed_data.py            # ONE-TIME processing
│
├── src/
│   ├── data/website_processor.py               # Data loading
│   ├── config/column_mapping.py                # Column config
│   ├── visualization/charts.py                 # SalaryVisualizer
│   ├── visualization/key_findings_dashboard.py # Dashboards
│   └── analytics/salary_models.py              # ML models
│
├── *.qmd                                        # Quarto pages
├── notebooks/                                   # Jupyter notebooks
└── figures/                                     # Generated charts
```

---

## Quick Start

1. **Create processed data** (one-time):

   ```bash
   python scripts/create_processed_data.py
   ```

2. **Run Quarto**:

   ```bash
   quarto preview --port 4200
   ```

3. **Use in code**:

   ```python
   df = pd.read_parquet('data/processed/job_market_processed.parquet')
   ```

---

**Best Practices**:

- Always load from Parquet
- Use `ANALYSIS_COLUMNS` for consistency
- Keep business logic in Python modules (not QMD files)
- Test with processed data only
