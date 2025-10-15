# Jupyter Notebooks Overview

This directory contains 3 essential notebooks for the Job Market Analytics project.

## Technology Stack

All notebooks use **PySpark MLlib** for machine learning and NLP tasks, aligned with the project's learning objectives:

- **PySpark**: Large-scale ETL and data processing
- **PySpark MLlib**: All ML models (regression, classification, clustering) and NLP (TF-IDF, Word2Vec, tokenization)
- **Plotly**: Interactive visualizations
- **NO scikit-learn**: Removed to maintain consistency with PySpark ecosystem

---

## Notebook 1: Data Pipeline

**File**: `data_processing_pipeline_demo.ipynb`

**Purpose**: Showcase the data processing pipeline from raw to processed data

**Content**:

- Load raw data with PySpark (13M rows from CSV)
- Demonstrate data processing via `JobMarketDataProcessor`
- Display before/after statistics
- Validate processed Parquet output
- Show column standardization (all snake_case)

**Key Features**:

- Uses `multiLine=True, escape="\"", header=True, inferSchema=True` for robust CSV parsing
- Automatic column name standardization to snake_case
- Efficient Parquet output for downstream analysis

---

## Notebook 2: Machine Learning Feature Engineering

**File**: `ml_feature_engineering_lab.ipynb`

**Purpose**: Demonstrate PySpark MLlib feature engineering and transformations

**Content**:

- Load processed data using `get_processed_dataframe()`
- Feature engineering with PySpark MLlib transformers
- Handle missing values and outliers
- Create derived features from salary and experience data
- Demonstrate `VectorAssembler`, `StandardScaler`, `StringIndexer`

**Key Features**:

- 100% PySpark MLlib (no scikit-learn)
- Uses standardized column names: `salary_avg`, `salary_from`, `salary_to`, `min_years_experience`, `naics2_name`
- Automatic data loading from Parquet

---

## Notebook 3: Job Market Skill Analysis (NLP)

**File**: `job_market_skill_analysis.ipynb`

**Purpose**: NLP analysis of job market skills and requirements using PySpark MLlib

**Content**:

- Load processed data using `get_processed_dataframe()`
- Text processing with PySpark MLlib tokenization
- TF-IDF analysis using `HashingTF` and `IDF`
- Word2Vec embeddings for skill similarity
- Topic modeling and skill clustering with PySpark KMeans
- Interactive skill visualizations with Plotly

**Key Features**:

- 100% PySpark MLlib NLP pipeline
- Replaced `TfidfVectorizer` with PySpark `Tokenizer` + `HashingTF` + `IDF`
- Replaced `sklearn.cluster.KMeans` with PySpark `KMeans`
- Uses `Word2Vec` for semantic skill analysis

---

## Data Loading

All notebooks now use the **centralized data loading** pattern:

```python
from src.data.website_processor import get_processed_dataframe

# Load processed data (fast - 1-2 seconds from Parquet)
df = get_processed_dataframe()
```

This ensures:

- Consistent data across all notebooks
- Fast loading from pre-processed Parquet
- Automatic generation if processed data is missing

---

## Visualization

For Plotly visualizations, use the **centralized `display_figure()` utility**:

```python
from src.visualization.charts import display_figure
import plotly.graph_objects as go

# Create a figure
fig = go.Figure(...)

# Display and save automatically
display_figure(fig, "my_chart")  # Saves to figures/my_chart.png
```

**Benefits**:

- Automatic PNG export for reports (high-quality, 1200x800, scale=2)
- Consistent file naming and organization
- Centralized error handling
- Works in both notebooks and Quarto QMD files

**Note**: For matplotlib plots, continue using `plt.show()` as usual.

---

## Standardized Column Names

After PySpark ETL, all columns use **snake_case** naming:

| Logical Name | Actual Column | Description |
|--------------|--------------|-------------|
| `salary` | `salary_avg` | Average salary (primary) |
| `salary_min` | `salary_from` | Minimum salary |
| `salary_max` | `salary_to` | Maximum salary |
| `industry` | `naics2_name` | Industry classification |
| `experience` | `min_years_experience` | Minimum years required |
| `city` | `city_name` | City name |
| `remote` | `remote_type_name` | Remote work type |

**Note**: Use the actual column names directly in notebooks for clarity.

---

## Setup & Usage

### Prerequisites

Ensure processed data exists:

```bash
# Generate processed data (run ONCE, takes 5-10 minutes)
python scripts/generate_processed_data.py
```

### Running Notebooks

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start Jupyter
cd notebooks
jupyter notebook
```

### Notebook Execution Order

1. **data_processing_pipeline_demo.ipynb** - Understand the ETL pipeline
2. **ml_feature_engineering_lab.ipynb** - Learn feature engineering with PySpark MLlib
3. **job_market_skill_analysis.ipynb** - Explore NLP and skill analysis

---

## Architecture Notes

### Why PySpark MLlib?

1. **Scalability**: Handles 13M rows efficiently
2. **Consistency**: Same stack for ETL and ML
3. **Learning Objective**: Master PySpark ecosystem
4. **Production Ready**: Code scales from laptop to cluster

### Data Flow

```
Raw CSV (13M rows)
    ↓ PySpark ETL (5-10 min, runs once)
Processed Parquet (72K rows)
    ↓ Pandas Load (1-2 sec)
Analysis & Visualization
```

### Memory Management

- **PySpark** processes large raw data without memory issues
- **Pandas** loads smaller processed data for fast analysis
- **Parquet** provides efficient columnar storage with compression
