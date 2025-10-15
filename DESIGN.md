# Implementation Design

**Tech Career Intelligence Platform** - Implementation Guide

> This document explains HOW the code is organized. For WHY decisions were made, see [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Quick Reference

| Need to... | Look at... |
|-----------|-----------|
| Load processed data | `src/data/website_processor.py` → `load_and_process_data()` |
| Create visualizations | `src/visualization/charts.py` → `SalaryVisualizer` |
| Run ML models | `src/analytics/salary_models.py` → `SalaryAnalyticsModels` |
| Process raw data | `scripts/generate_processed_data.py` |

---

## Module Organization

```bash
src/
├── data/                    # Data loading & processing
│   ├── loaders.py          # Read CSV/Parquet files
│   ├── transformers.py     # Clean & transform data (PySpark)
│   ├── validators.py       # Data quality checks
│   └── website_processor.py # Main entry point for analysis
│
├── analytics/              # Machine learning & specialized analysis
│   ├── salary_models.py    # Regression & classification (PySpark MLlib)
│   ├── skills_analysis.py  # Technical skills analysis
│   ├── ai_ml_location_analysis.py  # AI/ML job market analysis ✨ NEW
│   └── nlp_analysis.py     # NLP features
│
├── visualization/          # Charts & dashboards
│   ├── charts.py           # Core visualization (SalaryVisualizer)
│   ├── ml_charts.py        # ML model visualizations
│   ├── skills_charts.py    # Skills analysis charts
│   ├── ai_ml_charts.py     # AI/ML location analysis charts ✨ NEW
│   ├── presentation_charts.py  # Presentation-optimized charts
│   └── key_findings_dashboard.py  # Dashboard layouts
│
├── config/                 # Configuration
│   ├── settings.py         # Paths, constants
│   └── column_mapping.py   # Column mappings (SOFTWARE_SKILLS_NAME, SPECIALIZED_SKILLS_NAME)
│
├── ml/                     # Machine learning modules (PySpark MLlib)
│   ├── regression.py       # Linear & Random Forest regression
│   ├── classification.py   # Random Forest & Logistic Regression
│   ├── feature_engineering.py  # Feature transformations
│   ├── evaluation.py       # Model evaluation metrics
│   └── clustering.py       # K-means clustering
│
└── utils/                  # Utilities
    ├── spark_utils.py      # Spark session management
    └── logger.py           # Logging configuration
```

---

## Core Design Patterns

### 1. Data Loading Pattern

```python
# Simple pattern used throughout the codebase
from src.data.website_processor import load_and_process_data

# This handles everything:
# - Check if processed data exists
# - If yes: load Parquet (fast)
# - If no: process raw CSV with PySpark (slow, one-time)
df, summary = load_and_process_data()
```

### 2. Visualization Pattern

```python
# Create visualizations with helper function
from src.visualization.charts import SalaryVisualizer, display_figure

viz = SalaryVisualizer(df)
fig = viz.create_salary_distribution()

# Save in multiple formats (HTML, SVG, PNG)
display_figure(fig, "salary_distribution")
```

### 3. ML Pattern

```python
# Train models (usually done once, results cached)
from src.analytics.salary_models import SalaryAnalyticsModels

models = SalaryAnalyticsModels(df)
results = models.model_1_multiple_linear_regression()

# Results include: R², RMSE, coefficients, etc.
```

---

## Key Classes

### `SalaryVisualizer` (src/visualization/charts.py)

Main visualization class with methods for all chart types:

```python
viz = SalaryVisualizer(df)

# Available methods:
viz.create_salary_distribution()        # Histogram + box plot
viz.create_geographic_analysis()        # Map + bar chart
viz.create_experience_analysis()        # Salary by experience
viz.create_skills_correlation()         # Correlation matrix
viz.create_remote_work_analysis()       # Remote vs. on-site
# ... and more
```

### `SalaryAnalyticsModels` (src/analytics/salary_models.py)

ML models for salary prediction and classification:

```python
models = SalaryAnalyticsModels(df)

# Model 1: Predict salary based on features
regression_results = models.model_1_multiple_linear_regression()
# Returns: R², RMSE, coefficients

# Model 2: Classify above/below average salary
classification_results = models.model_2_above_average_classification()
# Returns: Accuracy, F1, feature importance
```

### `KeyFindingsDashboard` (src/visualization/key_findings_dashboard.py)

Pre-built dashboard layouts for key findings:

```python
dashboard = KeyFindingsDashboard(df)

# Create comprehensive dashboards
fig = dashboard.create_complete_intelligence_dashboard()
fig = dashboard.create_salary_distribution_analysis()
fig = dashboard.create_geographic_analysis()
# ... etc.
```

---

## Data Processing Pipeline

### How Raw Data Becomes Processed Data

1. **scripts/generate_processed_data.py** orchestrates the entire pipeline

2. **src/data/loaders.py** reads the raw CSV using PySpark

```python
   spark_df = DataLoader.load_raw_data(spark, file_path)
   ```

3. **src/data/transformers.py** cleans and transforms:

```python
      transformer = DataTransformer(spark)
      clean_df = transformer.clean_job_postings(spark_df)
      clean_df = transformer.engineer_features(clean_df)
   ```

4. **src/data/validators.py** checks quality:

```python
      validator = DataValidator()
      is_valid, issues = validator.validate_processed_data(clean_df)
   ```

5. **Save to Parquet** for fast loading:

```python
      clean_df.write.parquet("data/processed/job_market_processed.parquet")
   ```

---

## Configuration Management

### settings.py

Central configuration for paths and constants:

```python
from src.config.settings import (
    RAW_DATA_PATH,        # Path to raw CSV
    PROCESSED_DATA_PATH,  # Path to processed Parquet
    FIGURES_OUTPUT_DIR    # Where to save charts
)
```

### column_mapping.py

Standardized column names across the codebase:

```python
from src.config.column_mapping import get_analysis_column

salary_col = get_analysis_column('salary')      # 'salary_avg'
city_col = get_analysis_column('city')          # 'city_name'
industry_col = get_analysis_column('industry')  # 'naics2_name'
```

This abstraction allows changing column names in one place.

---

## Derived Features

### ETL Pipeline (Stored in Parquet)

| Derived Column | Source | Purpose |
|---------------|--------|---------|
| `salary_avg` | `salary_from`, `salary_to` | Average salary for analysis |

**Design Principle**: ETL pipeline only creates fundamental derived columns that are universally needed.

### Analysis-Time Calculation (On-the-Fly)

| Calculated Column | Source | Where Created | Purpose |
|------------------|--------|---------------|---------|
| `experience_level` | `min_years_experience` | `website_processor.add_experience_level()` | Entry/Mid/Senior/Executive categories |

**Design Principle**: Categorical groupings are calculated on-the-fly from existing columns during analysis. This keeps the stored data minimal and uses existing columns consistently.

---

## Logging Strategy

```python
from src.utils.logger import get_logger

# Default: WARNING level (suppresses INFO in Quarto)
logger = get_logger(level="WARNING")

# For debugging
logger = get_logger(level="INFO")

# Usage
logger.info("Processing data...")      # Suppressed in Quarto
logger.warning("Missing column!")      # Always shown
logger.error("Failed to load!")        # Always shown
```

---

## Spark Session Management

```python
from src.utils.spark_utils import create_spark_session

# Create optimized session
spark = create_spark_session("MyApp")

# For ML workloads (more memory)
from src.utils.spark_utils import create_ml_spark_session
spark = create_ml_spark_session("MLApp")

# Clean up when done
spark.stop()
```

---

## Testing Strategy

### Data Quality Checks

Validators ensure data integrity:

```python
from src.data.validators import DataValidator

validator = DataValidator()
is_valid, issues = validator.validate_processed_data(df)

if not is_valid:
    print(f"Data quality issues: {issues}")
```

### Visual QA

All charts save to `figures/` for manual inspection:

```bash
figures/
├── salary_distribution.html    # Interactive
├── salary_distribution.svg     # Vector (for papers)
└── salary_distribution.png     # Raster (for slides)
```

---

## Performance Optimization

### Why It's Fast

1. **One-time ETL**: Process 13M rows once, save to Parquet
2. **Parquet format**: 10x compression, columnar (fast filtering)
3. **Pandas for viz**: 72K rows fits in memory, no Spark overhead
4. **Cached figures**: Charts saved to files, not regenerated

### Memory Management

```python
# PySpark for big data
spark_df = spark.read.csv("13M_rows.csv")  # Distributed
processed = spark_df.filter(...).select(...)
processed.write.parquet("output.parquet")  # Save

# Pandas for small data
df = pd.read_parquet("output.parquet")     # 72K rows, in-memory
viz.create_chart(df)                        # Fast!
```

---

## Common Workflows

### Adding a New Visualization

1. Add method to `SalaryVisualizer` class:

```python
      def create_my_new_chart(self) -> go.Figure:
         fig = go.Figure(...)
         return fig
   ```

2. Use in QMD file:

```python
viz = SalaryVisualizer(df)
      fig = viz.create_my_new_chart()
      display_figure(fig, "my_new_chart")
   ```

### Adding a New Analysis Page

1. Create `my-analysis.qmd`
2. Load data: `df, summary = load_and_process_data()`
3. Create visualizations
4. Add to `_quarto.yml` navbar

### Regenerating Data

```bash
# Only needed when raw CSV changes
python scripts/generate_processed_data.py --force
```

---

## Deployment

The system generates a **static website** (no server needed):

```bash
quarto render
# Output: _salary/ directory

# Deploy anywhere:
# - GitHub Pages
# - Netlify
# - S3 + CloudFront
# - Any static host
```

---

## Troubleshooting

### "Processed data not found"

```bash
python scripts/generate_processed_data.py
```

### "Module not found"

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Best Practices

1. **Use the abstraction layers**: Don't bypass `load_and_process_data()`
2. **Display figures properly**: Always use `display_figure(fig, name)`
3. **Check processed data exists**: Run `generate_processed_data.py` first
4. **Column mapping**: Use `get_analysis_column()` for column names
5. **Logging levels**: Use WARNING for Quarto (suppress INFO noise)

---

## Code Style

- **Docstrings**: All functions have clear descriptions
- **Type hints**: Critical functions use type annotations
- **Imports**: Organized by standard → third-party → local
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes

---

## Documentation

- **ARCHITECTURE.md**: High-level system design (WHY)
- **DESIGN.md**: Implementation patterns (HOW) ← You are here
- **LEARNINGS.md**: Technical insights and trade-offs
- **README.md**: Quick start guide
- **Module docstrings**: Detailed API documentation

---

## Data References & Metric Provenance

**Purpose**: This section provides the factual backing for every number, metric, and statistic presented in the report, presentation, and notebooks. All metrics trace back to the data pipeline and specific calculations.

### Data Pipeline Foundation

#### Source Data

- **File**: `data/raw/lightcast_job_postings.csv`
- **Original Size**: 13M rows (raw job postings)
- **Source**: Lightcast (formerly Emsi Burning Glass) job analytics platform
- **Time Period**: 2024-2025
- **Geographic Coverage**: United States

#### Processed Data

- **File**: `data/processed/job_market_processed.parquet`
- **Processed Size**: ~72,000 rows
- **Columns**: 132 columns (after standardization to snake_case)
- **Processing Script**: `src/data/transformers.py` via `JobMarketDataProcessor`
- **Loading Function**: `src/data/website_processor.get_processed_dataframe()`

#### Key Pipeline Steps

1. **ETL Processing**: `src/core/processor.py` (PySpark)
2. **Column Standardization**: All columns converted to snake_case
3. **Missing Data Handling**: Convert missing values to meaningful "Undefined" categories
4. **Feature Engineering**: Created `salary_avg` from `salary_from` and `salary_to`
5. **Derived Features**: `experience_level` calculated on-the-fly from `min_years_experience`

### Missing Value Handling Strategy

**Philosophy**: Convert missing values to meaningful "Undefined" categories instead of NaN/Null to enable analysis of missing data patterns.

**Implementation**: `src/data/transformers.py` → `_enhance_dataframe()` method

| Column | Missing Values | Replacement | Count | Analysis Value |
|--------|---------------|-------------|-------|----------------|
| `remote_type` | NULL, '[None]', 'None', 'N/A', 'NA', '' | 'Undefined' | 56,614 (78%) | **Key Insight**: Most jobs don't specify remote policy |
| `employment_type` | NULL, '[None]', 'None', 'N/A', 'NA', '' | 'Undefined' | 44 (0.1%) | Small sample, but analyzable |
| `industry` | NULL, '[None]', 'None', 'N/A', 'NA', '' | 'Undefined' | 44 (0.1%) | Small sample, but analyzable |
| `city_name` | NULL, '[None]', 'None', 'N/A', 'NA', '' | 'Undefined' | 0 | All jobs have location data |
| `salary_avg` | NULL, 0, negative values | $75K median | N/A | Imputed for analysis continuity |

**Key Insight**: The "Undefined" remote type represents 78% of all jobs, indicating that most job postings don't clearly specify their remote work policy. This is valuable information for job seekers - they should ask about remote work during interviews since it's often not specified in postings.

### Column Mapping & Data Dictionary

#### Core Columns Used in Analysis

| Logical Name | Actual Column | Type | Source | Purpose |
|-------------|--------------|------|--------|---------|
| Salary (primary) | `salary_avg` | Numeric | Derived: `(salary_from + salary_to) / 2` | Main salary metric |
| Salary Min | `salary_from` | Numeric | Raw data | Lower bound |
| Salary Max | `salary_to` | Numeric | Raw data | Upper bound |
| Industry | `naics2_name` | Categorical | Raw data | Industry classification |
| Experience | `min_years_experience` | Numeric | Raw data | Minimum years required |
| Experience Level | `experience_level` | Categorical | Derived: `pd.cut(min_years_experience)` | Career stage |
| City | `city_name` | Categorical | Raw data | Geographic location |
| Remote Type | `remote_type_name` | Categorical | Raw data | Remote work policy |
| Employment Type | `employment_type_name` | Categorical | Raw data | Full-time/Part-time/etc |
| Job Title | `title` | Text | Raw data | Position name |
| Technical Skills | `technical_skills` | JSON Array | Raw: SOFTWARE_SKILLS_NAME | Pure technical/software skills (Python, SQL, etc.) |

#### Experience Level Derivation

**Function**: `src/data/website_processor.add_experience_level()`

**Logic**:

```python
pd.cut(
    df['min_years_experience'],
    bins=[-np.inf, 2, 5, 9, np.inf],
    labels=['Entry Level', 'Mid Level', 'Senior Level', 'Executive Level']
)
```

**Categories** (aligned with industry standards):

- **Unknown**: NULL values in `min_years_experience` (23,146 records)
- **Entry Level**: 0-2 years (9,925 records with salary, median $96K)
- **Mid Level**: 3-5 years (20,773 records with salary, median $115K)
- **Senior Level**: 6-10 years (15,180 records with salary, median $128K)
- **Leadership Level**: 10+ years (3,474 records with salary, median $126K)

**Reference in Code**: Lines 195-232 in `src/data/website_processor.py`

### Key Metrics & Their Sources

#### Dataset Size Metrics

#### Total Records

- **Value**: ~72,000 job postings
- **Source**: `len(df)` where `df = get_processed_dataframe()`
- **Code Location**: All `.qmd` files that load data
- **Verification**: `python scripts/validate_metrics.py`

#### Salary Coverage

- **Value**: 44.7% of records have salary data
- **Source**: `(df['salary_avg'].notna().sum() / len(df)) * 100`
- **Note**: Original data coverage (no imputation stored in Parquet)
- **Code Location**: `scripts/validate_metrics.py`

#### Industries Covered

- **Value**: 21 unique industries
- **Source**: `df['naics2_name'].nunique()`
- **Column**: `naics2_name` (NAICS 2-digit industry codes)

#### Cities Covered

- **Value**: 3,841 unique cities
- **Source**: `df['city_name'].nunique()`
- **Note**: Includes all US cities with job postings in dataset

#### Salary Metrics

#### Median Salary

- **Value**: $113,490
- **Source**: `df['salary_avg'].median()`
- **Sample Size**: N=32,398 (jobs with salary data)
- **Calculation**: Median of `salary_avg` column (imputed where missing)

#### Experience Gap

- **Value**: 33%
- **Calculation**:

   ```python
   exp_salaries = df.groupby('experience_level')['salary_avg'].median()
   gap = ((exp_salaries.iloc[-1] - exp_salaries.iloc[0]) / exp_salaries.iloc[0]) * 100
   ```

- **Source**: Experience progression analysis
- **Interpretation**: Senior salary is 1.33× Entry level salary

#### Experience Level Medians

- **Source**: `df.groupby('experience_level')['salary_avg'].median()`
- **Calculation Location**: Multiple files (index.qmd, salary-insights.qmd, tech-career-intelligence-report.qmd)
- **Actual Values** (based on bins: 0-2, 3-5, 6-9, 10+):
  - Entry Level (0-2 years): $96,100 (N=9,925)
  - Mid Level (3-5 years): $114,500 (N=20,773)
  - Senior Level (6-10 years): $127,550 (N=15,180)
  - Leadership Level (10+ years): $125,900 (N=3,474)

### Best Practices for Metric Reporting

#### Always Include

1. **The Number**: What is the actual value?
2. **The Source**: Which column(s) does it come from?
3. **The Calculation**: How is it computed?
4. **The Sample Size**: How many data points (N=)?
5. **The Context**: What does this number mean?

#### Example: Correct Reference

[X] **WRONG**: "Experience matters - senior roles earn 85% more"

[OK] **CORRECT**: "Experience matters - leadership roles earn 32% more than entry level (median: $126K vs $96K, N=49,352 jobs with experience data, calculated from `experience_level` grouping of `salary_avg`)"

### Sample Sizes

- **Always Include**: N= for every analysis
- **Example**: "Entry Level median: $96K (N=9,925 jobs)"
- **Why**: Transparency about confidence in estimates

### Validation

All metrics can be validated by running:

```bash
python scripts/validate_metrics.py
```

This script validates 14 key metrics against the actual data pipeline:

- Dataset size metrics (4 checks)
- Salary metrics (3 checks)
- Experience metrics (4 checks)
- Geographic variation (1 check)
- Industry premiums (1 check)
- Remote work data (1 check)

---

## AI/ML Job Market Analysis Module

### Overview

New specialized analysis module that identifies AI/ML jobs based on specialized skills (not just job titles) and analyzes their geographic distribution.

### Module Components

#### Analysis Module: `src/analytics/ai_ml_location_analysis.py`

**Purpose**: Identify AI/ML jobs using specialized skills keywords and analyze by location

**Key Functions**:

```python
def analyze_ai_ml_jobs_by_location(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identifies AI/ML jobs using 40+ keywords in specialized_skills column.

    Returns DataFrame with:
    - city_name
    - total_jobs
    - ai_ml_jobs
    - ai_ml_percentage
    - median_salary
    """
```

**AI/ML Detection Keywords** (40+ keywords):
- **Core AI/ML**: artificial intelligence, machine learning, deep learning, neural network
- **Frameworks**: tensorflow, pytorch, keras, scikit-learn
- **Techniques**: nlp, computer vision, supervised learning, clustering
- **MLOps**: model deployment, feature engineering, mlops
- **Data Science**: data science, predictive modeling, advanced analytics

**Why Specialized Skills?**
- More accurate than job title matching ("Data Scientist" may not do ML)
- Captures actual required expertise
- Identifies roles that genuinely need AI/ML skills

#### Visualization Module: `src/visualization/ai_ml_charts.py`

**Purpose**: Create publication-quality visualizations for AI/ML analysis

**Key Functions**:

```python
def create_ai_ml_jobs_by_location_chart(city_stats: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Total jobs vs AI/ML jobs by city"""

def create_ai_ml_percentage_chart(city_stats: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: AI/ML job concentration (%) by city"""

def create_ai_ml_combined_dashboard(city_stats: pd.DataFrame) -> go.Figure:
    """4-panel dashboard: Count, percentage, salary, distribution"""
```

### Integration Points

#### 1. Column Mapping (`src/config/column_mapping.py`)

```python
LIGHTCAST_COLUMN_MAPPING = {
    ...
    'SPECIALIZED_SKILLS_NAME': 'specialized_skills',  # ← NEW mapping
    ...
}
```

#### 2. Report Integration (`tech-career-intelligence-report.qmd`)

New section added after "Technical Skills Analysis":
- "AI/ML Job Market Analysis by Location"
- 3 visualizations + data-driven insights
- Lines 1283-1406

#### 3. Chart Preview Tool (`scripts/generate_charts_preview.py`)

New chart category:
```bash
python scripts/generate_charts_preview.py --charts ai_ml
```

Generates:
- `preview_ai_ml_jobs_by_location.png`
- `preview_ai_ml_percentage.png`
- `preview_ai_ml_dashboard.png`

### Design Decisions

**Why Separate from General Skills Analysis?**
- AI/ML is a specialized subset requiring different analysis
- Uses different data source (`specialized_skills` vs `technical_skills`)
- Targets specific audience (AI/ML job seekers)

**Why Location-Based?**
- AI/ML jobs are highly concentrated geographically
- Helps job seekers make relocation decisions
- Shows market maturity by city

**Why Not sklearn?**
- Consistent with project architecture (100% PySpark MLlib)
- This is data processing + visualization, no ML modeling needed
- Uses Pandas for analysis (72K rows is small enough)

---

**Questions?** Check the source code - every module and class has detailed docstrings!
