# System Architecture

**Tech Career Intelligence Platform** - Scalable data processing with PySpark, interactive analysis with Pandas + Plotly

> See [DESIGN.md](DESIGN.md) for implementation guide and usage patterns

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Architecture Diagram](#architecture-diagram)
4. [Data Flow](#data-flow)
5. [Module Structure](#module-structure)
6. [Class Diagrams](#class-diagrams)
7. [Deployment Architecture](#deployment-architecture)

---

## System Overview

The Tech Career Intelligence Platform is a **multi-layered data analytics system** that processes millions of job postings to provide actionable career insights through interactive web dashboards and reports.

### Core Architecture Principles

```mermaid
graph LR
    subgraph PRINCIPLE["SEPARATION OF CONCERNS"]
        PYSPARK["PySpark<br/>Heavy ETL<br/>13M rows"]
        PANDAS["Pandas<br/>Analysis<br/>30-50K rows"]
        PLOTLY["Plotly<br/>Interactive<br/>Visualizations"]
        QUARTO["Quarto<br/>Professional<br/>Presentations"]
    end

    PYSPARK -->|"Process & Filter"| PANDAS
    PANDAS -->|"Analyze & Model"| PLOTLY
    PLOTLY -->|"Embed & Display"| QUARTO

    style PRINCIPLE fill:#f5f5f5,stroke:#333,stroke-width:2px
    style PYSPARK fill:#1565c0,stroke:#fff,color:#fff,stroke-width:2px
    style PANDAS fill:#2e7d32,stroke:#fff,color:#fff,stroke-width:2px
    style PLOTLY fill:#6a1b9a,stroke:#fff,color:#fff,stroke-width:2px
    style QUARTO fill:#c62828,stroke:#fff,color:#fff,stroke-width:2px
```

### System Layers

1. **Data Processing Layer** (PySpark)

   - Load and clean 13M row CSV datasets
   - Validate and transform data
   - Engineer features at scale
   - Save to efficient Parquet format

2. **Analysis Layer** (Pandas + PySpark MLlib)

   - Load processed Parquet datasets
   - Statistical analysis and ML models
   - Feature engineering for ML
   - Generate insights and metrics

3. **Visualization Layer** (Plotly)

   - Create interactive charts
   - Generate executive dashboards
   - Export multi-format outputs (HTML/PNG/SVG)
   - Consistent theming and styling

4. **Presentation Layer** (Quarto)

   - Static website generation
   - Dynamic report rendering
   - Embed interactive visualizations
   - Professional documentation

---

## Technology Stack

### Primary Technologies

```mermaid
graph TB
    subgraph PROCESSING[" DATA PROCESSING"]
        SPARK[PySpark 4.0.1<br/>Distributed ETL]
        PARQUET[Apache Parquet<br/>Columnar Storage]
    end

    subgraph ANALYSIS["[DATA] ANALYSIS & ML"]
        PANDAS[Pandas 2.3<br/>Data Analysis]
        MLLIB[PySpark MLlib<br/>Machine Learning]
        NUMPY[NumPy 2.3<br/>Numerical Computing]
    end

    subgraph VISUALIZATION["[CHART] VISUALIZATION"]
        PLOTLY[Plotly 6.3<br/>Interactive Charts]
        MATPLOTLIB[Matplotlib 3.10<br/>Static Plots]
        KALEIDO[Kaleido 1.1<br/>Image Export]
    end

    subgraph PRESENTATION[" PRESENTATION"]
        QUARTO[Quarto<br/>Website Generator]
        JUPYTER[Jupyter Lab<br/>Notebooks]
        DOCX[python-docx<br/>Word Reports]
    end

    PROCESSING --> ANALYSIS
    ANALYSIS --> VISUALIZATION
    VISUALIZATION --> PRESENTATION

    style PROCESSING fill:#1565c0,stroke:#fff,color:#fff,stroke-width:3px
    style ANALYSIS fill:#2e7d32,stroke:#fff,color:#fff,stroke-width:3px
    style VISUALIZATION fill:#6a1b9a,stroke:#fff,color:#fff,stroke-width:3px
    style PRESENTATION fill:#c62828,stroke:#fff,color:#fff,stroke-width:3px
```

### Technology Decision Matrix

| Layer | Technology | Why? | Alternatives Considered |
|-------|-----------|------|------------------------|
| **ETL** | PySpark | 13M rows, distributed processing, lazy evaluation | Pandas (too slow), Dask (less mature) |
| **Storage** | Parquet | Columnar, compressed, fast reads | CSV (slow), HDF5 (not distributed) |
| **Analysis** | Pandas | Rich API, ecosystem, fast for <100K rows | Polars (new), Spark (overkill) |
| **ML** | PySpark MLlib | Scalable, consistent with PySpark architecture | Scikit-learn (not scalable), TensorFlow (overkill) |
| **Charts** | Plotly | Interactive, web-native, rich features | Altair (limited), D3 (complex) |
| **Website** | Quarto | Reproducible, supports Python, professional | R Markdown (R-focused), Sphinx (docs-only) |

---

## Architecture Diagram

### High-Level System Architecture

```mermaid
graph TB
    subgraph INPUT["DATA SOURCES"]
        RAW[(Raw CSV<br/>13M rows<br/>lightcast_job_postings.csv)]
    end

    subgraph ETL["ETL LAYER - PySpark"]
        PROCESSOR[JobMarketDataProcessor<br/>src/core/processor.py]
        LOADER[DataLoader<br/>src/data/loaders.py]
        TRANSFORMER[DataTransformer<br/>src/data/transformers.py]
        VALIDATOR[DataValidator<br/>src/data/validators.py]

        PROCESSOR --> LOADER
        PROCESSOR --> TRANSFORMER
        PROCESSOR --> VALIDATOR
    end

    subgraph STORAGE["DATA STORAGE"]
        PARQUET[(Processed Parquet<br/>30-50K rows<br/>job_market_processed.parquet)]
    end

    subgraph ANALYTICS["ANALYTICS LAYER - Pandas"]
        ANALYZER[SparkJobAnalyzer<br/>src/core/analyzer.py]
        MODELS[SalaryAnalyticsModels<br/>src/analytics/salary_models.py]
        NLP[JobMarketNLPAnalyzer<br/>src/analytics/nlp_analysis.py]
        ML[ML Models<br/>src/ml/*.py]
    end

    subgraph VIZ["VISUALIZATION LAYER - Plotly"]
        CHARTS[SalaryVisualizer<br/>src/visualization/charts.py]
        DASHBOARD[KeyFindingsDashboard<br/>src/visualization/key_findings_dashboard.py]
        THEME[JobMarketTheme<br/>src/visualization/theme.py]
    end

    subgraph OUTPUT["OUTPUT LAYER"]
        QUARTO[Quarto Website<br/>*.qmd files]
        JUPYTER[Jupyter Notebooks<br/>notebooks/*.ipynb]
        REPORTS[Word Reports<br/>*.docx]
    end

    RAW --> PROCESSOR
    PROCESSOR --> PARQUET
    PARQUET --> ANALYZER
    PARQUET --> MODELS
    PARQUET --> NLP
    PARQUET --> ML

    MODELS --> CHARTS
    NLP --> CHARTS
    ML --> CHARTS
    ANALYZER --> DASHBOARD

    CHARTS --> THEME
    DASHBOARD --> THEME

    CHARTS --> QUARTO
    DASHBOARD --> QUARTO
    CHARTS --> JUPYTER
    CHARTS --> REPORTS

    style INPUT fill:#37474f,stroke:#fff,color:#fff,stroke-width:2px
    style ETL fill:#1565c0,stroke:#fff,color:#fff,stroke-width:3px
    style STORAGE fill:#6a1b9a,stroke:#fff,color:#fff,stroke-width:2px
    style ANALYTICS fill:#2e7d32,stroke:#fff,color:#fff,stroke-width:3px
    style VIZ fill:#d84315,stroke:#fff,color:#fff,stroke-width:3px
    style OUTPUT fill:#c62828,stroke:#fff,color:#fff,stroke-width:2px
```

---

## Data Flow

### End-to-End Data Pipeline

```mermaid
flowchart TD
    START([Raw CSV<br/>13M rows]) --> SPARK_READ[PySpark Read CSV<br/>Parallel Loading]

    SPARK_READ --> STANDARDIZE[Standardize Columns<br/>UPPERCASE → snake_case]
    STANDARDIZE --> CLEAN[Clean & Validate<br/>- Salary ranges<br/>- Experience values<br/>- Location data]
    CLEAN --> IMPUTE[Intelligent Imputation<br/>- Salary by city+industry<br/>- Experience defaults]
    IMPUTE --> FILTER[Filter Invalid Records<br/>- salary < 20K or > 500K<br/>- missing critical fields]
    FILTER --> FEATURE[Feature Engineering<br/>- experience_level<br/>- salary_avg<br/>- company_size_numeric]
    FEATURE --> SAVE_PARQUET[Save to Parquet<br/>Columnar + Compressed]

    SAVE_PARQUET --> PARQUET[(Processed Parquet<br/>30-50K clean rows)]

    PARQUET --> PANDAS_READ[Pandas Read Parquet<br/>Fast Load to Memory]

    PANDAS_READ --> BRANCH{Analysis Type?}

    BRANCH -->|Statistical| STATS[Statistical Analysis<br/>- Aggregations<br/>- Distributions<br/>- Correlations]
    BRANCH -->|ML Models| ML[Machine Learning<br/>- Regression<br/>- Classification<br/>- Clustering]
    BRANCH -->|NLP| NLP[NLP Analysis<br/>- Skills extraction<br/>- Topic modeling<br/>- Word clouds]

    STATS --> PLOTLY[Plotly Visualizations]
    ML --> PLOTLY
    NLP --> PLOTLY

    PLOTLY --> EXPORT{Export Format?}

    EXPORT -->|Web| HTML[HTML<br/>Interactive Charts]
    EXPORT -->|Reports| PNG[PNG/SVG<br/>Static Images]

    HTML --> QUARTO[Quarto Website<br/>_salary/]
    PNG --> DOCX[Word Reports<br/>*.docx]
    HTML --> JUPYTER[Jupyter Notebooks<br/>notebooks/]

    style START fill:#37474f,stroke:#fff,color:#fff
    style SPARK_READ fill:#1565c0,stroke:#fff,color:#fff
    style PARQUET fill:#6a1b9a,stroke:#fff,color:#fff
    style PANDAS_READ fill:#2e7d32,stroke:#fff,color:#fff
    style PLOTLY fill:#d84315,stroke:#fff,color:#fff
    style QUARTO fill:#c62828,stroke:#fff,color:#fff
```

### Layer Boundaries and Contracts

```mermaid
graph LR
    subgraph INPUT ["Input Contract"]
        CSV[CSV File<br/>UPPERCASE columns<br/>Raw, unvalidated]
    end

    subgraph SPARK_LAYER ["PySpark Layer"]
        SPARK_DF[Spark DataFrame<br/>snake_case columns<br/>Validated schema]
    end

    subgraph PARQUET_LAYER ["Storage Contract"]
        PARQ[Parquet File<br/>snake_case columns<br/>Clean, validated<br/>30-50K rows]
    end

    subgraph PANDAS_LAYER ["Pandas Layer"]
        PANDAS_DF[Pandas DataFrame<br/>snake_case columns<br/>Ready for analysis]
    end

    subgraph OUTPUT_LAYER ["Output Contract"]
        FIGURES[Plotly Figures<br/>HTML/PNG/SVG<br/>Self-contained]
    end

    CSV -->|spark.read.csv| SPARK_DF
    SPARK_DF -->|.write.parquet| PARQ
    PARQ -->|pd.read_parquet| PANDAS_DF
    PANDAS_DF -->|viz.plot_*| FIGURES

    style INPUT fill:#37474f,stroke:#fff,color:#fff
    style SPARK_LAYER fill:#1565c0,stroke:#fff,color:#fff
    style PARQUET_LAYER fill:#6a1b9a,stroke:#fff,color:#fff
    style PANDAS_LAYER fill:#2e7d32,stroke:#fff,color:#fff
    style OUTPUT_LAYER fill:#d84315,stroke:#fff,color:#fff
```

---

## Module Structure

### Directory Organization

```mermaid
graph TB
    ROOT[src/]

    ROOT --> CONFIG[config/<br/>Configuration]
    ROOT --> CORE[core/<br/>PySpark ETL]
    ROOT --> DATA[data/<br/>Data Utilities]
    ROOT --> ANALYTICS[analytics/<br/>Pandas ML]
    ROOT --> ML[ml/<br/>ML Models]
    ROOT --> VIZ[visualization/<br/>Plotly Charts]
    ROOT --> UTILS[utils/<br/>Helpers]

    CONFIG --> CONFIG1[column_mapping.py<br/>Column standards]
    CONFIG --> CONFIG2[settings.py<br/>App config]

    CORE --> CORE1[processor.py<br/>JobMarketDataProcessor]
    CORE --> CORE2[analyzer.py<br/>SparkJobAnalyzer]

    DATA --> DATA1[loaders.py<br/>DataLoader Spark]
    DATA --> DATA2[transformers.py<br/>DataTransformer Spark]
    DATA --> DATA3[validators.py<br/>DataValidator Spark]
    DATA --> DATA4[website_processor.py<br/>Quarto interface]
    DATA --> DATA5[auto_processor.py<br/>Helper functions]

    ANALYTICS --> ANALYTICS1[salary_models.py<br/>SalaryAnalyticsModels]
    ANALYTICS --> ANALYTICS2[nlp_analysis.py<br/>JobMarketNLPAnalyzer]
    ANALYTICS --> ANALYTICS3[predictive_dashboard.py<br/>PredictiveAnalyticsDashboard]
    ANALYTICS --> ANALYTICS4[docx_report_generator.py<br/>ReportGenerator]

    ML --> ML1[regression.py<br/>Salary prediction]
    ML --> ML2[classification.py<br/>Job categorization]
    ML --> ML3[clustering.py<br/>Market segmentation]
    ML --> ML4[feature_engineering.py<br/>Feature prep]
    ML --> ML5[evaluation.py<br/>Model metrics]

    VIZ --> VIZ1[charts.py<br/>SalaryVisualizer]
    VIZ --> VIZ2[key_findings_dashboard.py<br/>KeyFindingsDashboard]
    VIZ --> VIZ3[theme.py<br/>JobMarketTheme]

    UTILS --> UTILS1[spark_utils.py<br/>Spark helpers]

    style ROOT fill:#1a237e,stroke:#fff,color:#fff,stroke-width:4px
    style CONFIG fill:#f57f17,stroke:#fff,color:#000,stroke-width:2px
    style CORE fill:#1565c0,stroke:#fff,color:#fff,stroke-width:3px
    style DATA fill:#0277bd,stroke:#fff,color:#fff,stroke-width:2px
    style ANALYTICS fill:#2e7d32,stroke:#fff,color:#fff,stroke-width:3px
    style ML fill:#558b2f,stroke:#fff,color:#fff,stroke-width:2px
    style VIZ fill:#d84315,stroke:#fff,color:#fff,stroke-width:3px
    style UTILS fill:#455a64,stroke:#fff,color:#fff,stroke-width:2px
```

### Module Responsibilities

| Module | Responsibility | Primary Technology | Output |
|--------|---------------|-------------------|--------|
| **src/config/** | Configuration management | Python | Settings, mappings |
| **src/core/** | Heavy ETL processing | PySpark | Processed DataFrame |
| **src/data/** | Data loading & utilities | PySpark + Pandas | DataFrames |
| **src/analytics/** | ML models & analysis | PySpark MLlib | Models, insights |
| **src/ml/** | Advanced ML | PySpark MLlib | Trained models |
| **src/visualization/** | Charts & dashboards | Plotly | Figures |
| **src/utils/** | Helper functions | Python | Utilities |

---

## Class Diagrams

### 1. Data Processing Classes (PySpark)

```mermaid
classDiagram
    class JobMarketDataProcessor {
        -spark: SparkSession
        -settings: Settings
        -data_loader: DataLoader
        -data_transformer: DataTransformer
        -data_validator: DataValidator
        +load_and_process_data(path) DataFrame
        +clean_and_standardize_data(df) DataFrame
        +engineer_features(df) DataFrame
        +save_processed_data(df, path) void
        +assess_data_quality(df) Dict
    }

    class SparkJobAnalyzer {
        -spark: SparkSession
        -settings: Settings
        -job_data: DataFrame
        +load_full_dataset(path) DataFrame
        +calculate_salary_statistics() Dict
        +analyze_by_location() DataFrame
        +analyze_by_experience() DataFrame
        +analyze_by_education() DataFrame
        +generate_insights_report() Dict
    }

    class DataLoader {
        -spark: SparkSession
        -settings: Settings
        +load_raw_data(path) DataFrame
        +load_processed_data(path) DataFrame
        +validate_schema(df) bool
    }

    class DataTransformer {
        +clean_and_standardize(df) DataFrame
        +standardize_column_names(df) DataFrame
        +impute_missing_values(df) DataFrame
        +engineer_features(df) DataFrame
        +create_experience_level(df) DataFrame
    }

    class DataValidator {
        +validate_dataset(df) Dict
        +check_required_columns(df) bool
        +validate_salary_data(df) Dict
        +validate_experience_data(df) Dict
        +assess_data_quality(df) Dict
    }

    JobMarketDataProcessor --> DataLoader
    JobMarketDataProcessor --> DataTransformer
    JobMarketDataProcessor --> DataValidator
    SparkJobAnalyzer --> DataLoader
    SparkJobAnalyzer --> DataValidator
```

### 2. Analytics Classes (Pandas + ML)

```mermaid
classDiagram
    class SalaryAnalyticsModels {
        -df: DataFrame
        -models: Dict
        -scalers: Dict
        -encoders: Dict
        +prepare_features() DataFrame
        +model_1_salary_regression(X, y) Dict
        +model_2_above_average_classification(X, y) Dict
        +run_complete_analysis() Dict
        +create_analysis_visualizations(results) List~Figure~
    }

    class JobMarketNLPAnalyzer {
        -df: DataFrame
        -vectorizer: TfidfVectorizer
        -skills_corpus: List
        +extract_skills(text) List~str~
        +create_word_cloud() Figure
        +topic_clustering(n_topics) Dict
        +analyze_skill_trends() DataFrame
        +run_complete_nlp_analysis() Dict
    }

    class PredictiveAnalyticsDashboard {
        -df: DataFrame
        -salary_models: SalaryAnalyticsModels
        -nlp_analyzer: JobMarketNLPAnalyzer
        +create_executive_summary_dashboard() Figure
        +create_model_comparison_dashboard() Figure
        +create_skills_insights_dashboard() Figure
        +generate_comprehensive_report() Dict
    }

    class SalaryRegressionModel {
        -spark: SparkSession
        -models: Dict
        +prepare_regression_data(df) DataFrame
        +train_linear_regression(df) Model
        +train_random_forest(df) Model
        +evaluate_model(model, test_df) Dict
    }

    class JobClassificationModel {
        -spark: SparkSession
        -pipelines: Dict
        +prepare_classification_data(df) DataFrame
        +train_logistic_regression(df) Model
        +train_random_forest_classifier(df) Model
        +evaluate_classification(model, test_df) Dict
    }

    PredictiveAnalyticsDashboard --> SalaryAnalyticsModels
    PredictiveAnalyticsDashboard --> JobMarketNLPAnalyzer
    SalaryAnalyticsModels --> SalaryRegressionModel
    SalaryAnalyticsModels --> JobClassificationModel
```

### 3. Visualization Classes (Plotly)

```mermaid
classDiagram
    class SalaryVisualizer {
        -df: DataFrame
        -theme: JobMarketTheme
        +get_overall_statistics() Dict
        +get_experience_progression_analysis() Dict
        +get_education_roi_analysis() Dict
        +get_industry_salary_analysis(top_n) DataFrame
        +get_geographic_salary_analysis(top_n) DataFrame
        +plot_salary_distribution() Figure
        +plot_experience_salary_trend() Figure
        +plot_salary_by_category(column) Figure
        +plot_ai_salary_comparison() Figure
        +plot_remote_salary_analysis() Figure
        +create_correlation_matrix() Figure
    }

    class KeyFindingsDashboard {
        -df: DataFrame
        -theme: JobMarketTheme
        -key_metrics: Dict
        +create_key_metrics_cards() Figure
        +create_career_progression_analysis() Figure
        +create_education_roi_analysis() Figure
        +create_company_strategy_analysis() Figure
        +create_ai_technology_analysis() Figure
        +create_complete_intelligence_dashboard() Figure
        -_calculate_key_metrics() Dict
        -_get_salary_progression_data() Dict
        -_create_metric_card(metric, value) Figure
    }

    class JobMarketTheme {
        <<static>>
        +PRIMARY_COLORS: Dict
        +CATEGORICAL_COLORS: List
        +SALARY_SCALE: List
        +FONT_FAMILY: str
        +BASE_FONT_SIZE: int
        +get_plotly_layout(title, width, height) Dict
        +get_color_scale(type) List
        +apply_theme_to_figure(fig) Figure
    }

    class QuartoChartExporter {
        +export_for_quarto(fig, path) str
        +save_multiple_formats(fig, base_name) Dict
        +save_html(fig, path) void
        +save_png(fig, path, width, height) void
        +save_svg(fig, path, width, height) void
    }

    SalaryVisualizer --> JobMarketTheme
    KeyFindingsDashboard --> JobMarketTheme
    SalaryVisualizer --> QuartoChartExporter
    KeyFindingsDashboard --> QuartoChartExporter
```

---

## Deployment Architecture

### Local Development

```mermaid
graph TB
    DEV[Developer Machine]

    subgraph LOCAL["Local Environment"]
        VENV[Python Virtual Env<br/>.venv/]
        SPARK_LOCAL[PySpark Local Mode<br/>Single JVM]
        DATA_LOCAL[data/<br/>Local filesystem]
        JUPYTER_LOCAL[Jupyter Lab<br/>localhost:8888]
        QUARTO_LOCAL[Quarto Preview<br/>localhost:4200]
    end

    DEV --> VENV
    VENV --> SPARK_LOCAL
    SPARK_LOCAL --> DATA_LOCAL
    VENV --> JUPYTER_LOCAL
    VENV --> QUARTO_LOCAL

    style DEV fill:#37474f,stroke:#fff,color:#fff
    style LOCAL fill:#1565c0,stroke:#fff,color:#fff,stroke-width:2px
```

### Production/Scaled Deployment (Future)

```mermaid
graph TB
    subgraph COMPUTE["Compute Cluster"]
        SPARK_CLUSTER[PySpark Cluster<br/>Distributed Processing]
        WORKERS[Worker Nodes<br/>Parallel Execution]
    end

    subgraph STORAGE["Data Storage"]
        S3[Cloud Object Storage<br/>S3/Azure Blob]
        PARQUET_CLOUD[Parquet Files<br/>Partitioned by date]
    end

    subgraph WEB["Web Tier"]
        STATIC[Static Site<br/>GitHub Pages/Netlify]
        CDN[CDN<br/>CloudFlare]
    end

    SPARK_CLUSTER --> WORKERS
    WORKERS --> PARQUET_CLOUD
    PARQUET_CLOUD --> S3
    S3 --> STATIC
    STATIC --> CDN

    style COMPUTE fill:#1565c0,stroke:#fff,color:#fff,stroke-width:2px
    style STORAGE fill:#6a1b9a,stroke:#fff,color:#fff,stroke-width:2px
    style WEB fill:#c62828,stroke:#fff,color:#fff,stroke-width:2px
```

---

## Performance Characteristics

### Processing Performance

| Operation | Technology | Dataset Size | Time | Memory |
|-----------|-----------|--------------|------|--------|
| Load raw CSV | PySpark | 13M rows | ~2-3 min | 4-8 GB |
| Clean & transform | PySpark | 13M rows | ~5-10 min | 4-8 GB |
| Save to Parquet | PySpark | 30-50K rows | ~10-30 sec | 2-4 GB |
| Load Parquet | Pandas | 30-50K rows | ~1-2 sec | 500 MB |
| Statistical analysis | Pandas | 30-50K rows | <1 sec | 500 MB |
| ML training | PySpark MLlib | 30-50K rows | 5-30 sec | 1-2 GB |
| Generate chart | Plotly | 30-50K points | 1-5 sec | 200 MB |
| Render Quarto page | Quarto | N/A | 5-15 sec | 500 MB |

### Storage Efficiency

```bash
Raw CSV:        ~2.5 GB (13M rows, 131 columns)
                ↓ PySpark processing + filtering
Processed Parquet: ~120 MB (30-50K rows, 132 columns)
                ↓ 95% size reduction
```

**Parquet Benefits**:

- **Columnar storage**: Read only needed columns
- **Compression**: Built-in snappy/gzip compression
- **Type efficiency**: Proper data types vs. string CSV
- **Fast reads**: Optimized for analytical queries

---

## Security & Data Privacy

### Data Handling

1. **No PII storage**: Job postings are anonymized
2. **Local processing**: All data stays on local machine
3. **No external APIs**: Self-contained processing
4. **Version control**: Data files in `.gitignore`

### Access Control

- **Development**: Local file system permissions
- **Production**: Cloud IAM roles (if deployed)
- **API keys**: Environment variables (not committed)

---

## Monitoring & Observability

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Processing started...")
```

### Metrics to Track

1. **Data Quality**:
   - Missing value percentages
   - Outlier counts
   - Schema violations

2. **Performance**:
   - Processing time per stage
   - Memory usage peaks
   - Parquet file sizes

3. **Model Performance**:
   - R² scores
   - Classification accuracy
   - Feature importance

---

## Future Enhancements

### Scalability

1. **Distributed Spark**: Move to multi-node cluster for larger datasets
2. **Incremental updates**: Process only new/changed data
3. **Data partitioning**: Partition Parquet by date/region
4. **Caching layer**: Redis for frequently accessed aggregations

### Features

1. **Real-time updates**: Stream processing with Spark Streaming
2. **Interactive dashboards**: Add Streamlit/Dash for live exploration
3. **API layer**: REST API for programmatic access
4. **Automated reports**: Scheduled report generation and delivery

### MLOps

1. **Model versioning**: MLflow for experiment tracking
2. **Model registry**: Centralized model storage
3. **A/B testing**: Compare model versions
4. **Monitoring**: Track model performance over time

---

## References

- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Python](https://plotly.com/python/)
- [Quarto](https://quarto.org/)
- [PySpark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)

---

**For implementation details and usage patterns**, see [DESIGN.md](DESIGN.md)

**For project overview and setup**, see [README.md](README.md)
