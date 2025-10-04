# Jupyter Notebooks Overview

This directory contains 3 essential notebooks for the Job Market Analytics project.

## Notebook 1: Data Pipeline

**File**: `data_processing_pipeline_demo.ipynb`

**Purpose**: Showcase the data processing pipeline from raw to processed data

**Content**:

- Load raw data
- Show data processing steps via `scripts/create_processed_data.py`
- Display before/after statistics
- Validate processed Parquet output

---

## Notebook 2: Machine Learning Models

**File**: `ml_feature_engineering_lab.ipynb`

**Purpose**: Demonstrate ML models and clustering techniques

**Content**:

- ONET-based KMeans clustering for job segmentation
- Multiple regression with justification
- Classification aligned with topic (AI/remote jobs)
- Visualize key results and metrics

---

## Notebook 3: Predictive Analytics

**File**: `job_market_skill_analysis.ipynb`

**Purpose**: Advanced predictive dashboards and feature engineering

**Content**:

- Predictive dashboards using `PredictiveAnalyticsDashboard`
- Extended feature engineering
- Interactive visualizations

---

## Usage

All notebooks load pre-processed data from `data/processed/job_market_processed.parquet`.

To regenerate processed data:

```bash
python scripts/create_processed_data.py
```

To run notebooks:

```bash
cd notebooks
jupyter notebook
```
