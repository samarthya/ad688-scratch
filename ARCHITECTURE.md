# System Architecture

**Tech Career Intelligence Platform** - Comprehensive architecture documentation with Mermaid diagrams.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Module Structure](#module-structure)
4. [Class Diagrams](#class-diagrams)
5. [Data Flow](#data-flow)
6. [Quarto Website Integration](#quarto-website-integration)
7. [Jupyter Notebooks](#jupyter-notebooks)
8. [Configuration Management](#configuration-management)

---

## System Overview

The Tech Career Intelligence Platform is a data-driven web application built with **Quarto**, **Python**, and **Plotly** to analyze job market trends, salary patterns, and career progression insights.

### Core Principles

1. **Process Once, Use Many Times**: Raw data is processed once into Parquet format, then loaded directly for all analyses
2. **Abstraction Layer**: All business logic resides in `src/` modules; QMD files are pure presentation layer
3. **Column Standardization**: All processed data uses `snake_case` column names
4. **Configuration-Driven**: Centralized column mapping and settings in `src/config/`

### Technology Stack

```mermaid
graph TB
    A[PRESENTATION LAYER<br/>Quarto HTML + Plotly + Markdown]
    B[ABSTRACTION LAYER<br/>Python Classes<br/>Pandas + NumPy + Scikit-learn]
    C[DATA LAYER<br/>Parquet Processed + CSV Raw]
    
    A --> B
    B --> C
    
    style A fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style B fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style C fill:#f1f8e9,stroke:#33691e,stroke-width:2px
```

---

## Architecture Diagram

```mermaid
graph TB
    subgraph QUARTO["🌐 QUARTO WEBSITE"]
        INDEX[index.qmd<br/>Homepage]
        PIPELINE[data-pipeline.qmd<br/>Pipeline Demo]
        INSIGHTS[salary-insights.qmd<br/>Salary Analysis]
        MARKET[market-dashboard.qmd<br/>Market Overview]
        PREDICT[predictive-analytics.qmd<br/>ML Models]
    end
    
    subgraph PROCESSOR["📊 DATA PROCESSOR"]
        WP[website_processor.py<br/>load_and_process_data<br/>get_processed_dataframe<br/>get_website_data_summary]
    end
    
    subgraph VIZ["📈 VISUALIZATION"]
        SV[SalaryVisualizer<br/>Charts & Analysis]
        KF[KeyFindingsDashboard<br/>Executive Dashboards]
        TH[JobMarketTheme<br/>Styling]
    end
    
    subgraph ANALYTICS["🤖 ANALYTICS"]
        ML[SalaryAnalyticsModels<br/>Regression & Classification]
        NLP[JobMarketNLPAnalyzer<br/>Skills Extraction]
        DASH[PredictiveAnalyticsDashboard<br/>ML Dashboards]
    end
    
    subgraph CONFIG["⚙️ CONFIGURATION"]
        COL[column_mapping.py<br/>LIGHTCAST_COLUMN_MAPPING<br/>ANALYSIS_COLUMNS]
        SET[settings.py<br/>Application Config]
    end
    
    subgraph DATA["💾 DATA LAYER"]
        PARQUET[(job_market_processed.parquet<br/>117.8 MB, 32K records)]
        CSV[(lightcast_job_postings.csv<br/>Raw Data)]
    end
    
    INDEX --> WP
    PIPELINE --> WP
    INSIGHTS --> WP
    MARKET --> WP
    PREDICT --> WP
    
    WP --> VIZ
    WP --> ANALYTICS
    WP --> CONFIG
    
    VIZ --> TH
    ANALYTICS --> ML
    
    CONFIG --> WP
    
    WP --> PARQUET
    PARQUET -.fallback.-> CSV
    
    style QUARTO fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    style PROCESSOR fill:#fff3e0,stroke:#e65100,stroke-width:3px
    style VIZ fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style ANALYTICS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style CONFIG fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style DATA fill:#fce4ec,stroke:#c2185b,stroke-width:3px
```


---

## Module Structure

### Directory Tree

```mermaid
graph LR
    ROOT[src/]
    
    ROOT --> ANALYTICS[analytics/<br/>ML & NLP]
    ROOT --> CONFIG[config/<br/>Settings]
    ROOT --> DATA[data/<br/>Pipeline]
    ROOT --> ML[ml/<br/>Models]
    ROOT --> UTILS[utils/<br/>Helpers]
    ROOT --> VIZ[visualization/<br/>Charts]
    
    ANALYTICS --> A1[salary_models.py]
    ANALYTICS --> A2[nlp_analysis.py]
    ANALYTICS --> A3[predictive_dashboard.py]
    ANALYTICS --> A4[docx_report_generator.py]
    
    CONFIG --> C1[column_mapping.py]
    CONFIG --> C2[settings.py]
    
    DATA --> D1[website_processor.py]
    DATA --> D2[auto_processor.py]
    DATA --> D3[data_cleaner.py]
    DATA --> D4[loaders.py]
    
    ML --> M1[regression.py]
    ML --> M2[classification.py]
    ML --> M3[clustering.py]
    ML --> M4[feature_engineering.py]
    
    VIZ --> V1[charts.py]
    VIZ --> V2[key_findings_dashboard.py]
    VIZ --> V3[theme.py]
    
    style ROOT fill:#1a237e,stroke:#fff,color:#fff,stroke-width:3px
    style ANALYTICS fill:#4a148c,stroke:#fff,color:#fff,stroke-width:2px
    style CONFIG fill:#f57f17,stroke:#fff,color:#fff,stroke-width:2px
    style DATA fill:#01579b,stroke:#fff,color:#fff,stroke-width:2px
    style ML fill:#1b5e20,stroke:#fff,color:#fff,stroke-width:2px
    style VIZ fill:#b71c1c,stroke:#fff,color:#fff,stroke-width:2px
```

---

## Class Diagrams

### 1. Data Processing Classes

```mermaid
classDiagram
    class WebsiteProcessor {
        <<module>>
        +load_and_process_data() tuple
        +get_processed_dataframe() DataFrame
        +get_website_data_summary() Dict
        +standardize_columns(df) DataFrame
        +decode_base64_locations(df) DataFrame
        +parse_json_locations(df) DataFrame
        +generate_website_figures(df) Dict
    }
    
    class AutoProcessor {
        <<module>>
        +load_analysis_data() DataFrame
        +get_data_summary(df) Dict
        +process_raw_data(df) DataFrame
    }
    
    class JobMarketDataCleaner {
        -cleaning_stats: Dict
        +clean_dataset(df) tuple
        +_optimize_columns(df) tuple
        +_clean_text_columns(df) tuple
        +_clean_location_data(df) tuple
        +_clean_salary_data(df) tuple
    }
    
    class DataLoader {
        -spark: SparkSession
        -settings: Settings
        +load_raw_data(path) DataFrame
        +load_processed_data(path) DataFrame
    }
    
    class DataTransformer {
        +clean_and_standardize(df) DataFrame
        +engineer_features(df) DataFrame
    }
    
    class DataValidator {
        +validate_dataset(df) bool
        +check_required_columns(df) bool
        +validate_salary_data(df) bool
    }
    
    WebsiteProcessor ..> AutoProcessor : delegates
    WebsiteProcessor ..> JobMarketDataCleaner : uses
    AutoProcessor ..> DataLoader : uses
    JobMarketDataCleaner ..> DataTransformer : uses
    JobMarketDataCleaner ..> DataValidator : uses
```

### 2. Visualization Classes

```mermaid
classDiagram
    class SalaryVisualizer {
        -df: DataFrame
        +get_experience_progression_analysis() Dict
        +get_education_analysis() Dict
        +get_skills_analysis() Dict
        +get_industry_salary_analysis() DataFrame
        +get_geographic_salary_analysis() DataFrame
        +plot_salary_distribution() Figure
        +plot_salary_by_category(col) Figure
        +plot_ai_salary_comparison() Figure
        +plot_remote_salary_analysis() Figure
        +create_correlation_matrix() Figure
        +create_key_findings_graphics(dir) Dict
    }
    
    class KeyFindingsDashboard {
        -df: DataFrame
        +create_key_metrics_cards() Figure
        +create_career_progression_analysis() Figure
        +create_company_strategy_analysis() Figure
        +create_education_roi_analysis() Figure
        +create_complete_intelligence_dashboard() Figure
        +create_ai_technology_analysis() Figure
        -_calculate_key_metrics() Dict
        -_get_salary_progression_data() Dict
    }
    
    class JobMarketTheme {
        <<static>>
        +PRIMARY_COLORS: Dict
        +CATEGORICAL_COLORS: List
        +SALARY_SCALE: List
        +FONT_FAMILY: str
        +get_plotly_layout(title, w, h) Dict
        +get_matplotlib_style() Dict
    }
    
    class QuartoChartExporter {
        +export_for_quarto(fig, path) void
        +save_multiple_formats(fig, name) void
    }
    
    SalaryVisualizer ..> JobMarketTheme : uses
    KeyFindingsDashboard ..> JobMarketTheme : uses
    SalaryVisualizer ..> QuartoChartExporter : uses
    KeyFindingsDashboard ..> QuartoChartExporter : uses
```

### 3. Analytics Classes

```mermaid
classDiagram
    class SalaryAnalyticsModels {
        -df: DataFrame
        +run_complete_analysis() Dict
        +prepare_features() DataFrame
        +model_1_salary_regression(X) Dict
        +model_2_above_average_classification(X) Dict
        +create_analysis_visualizations(results) List~Figure~
    }
    
    class JobMarketNLPAnalyzer {
        -df: DataFrame
        +extract_skills() List~str~
        +create_word_cloud() Figure
        +topic_clustering() Dict
        +visualize_skill_trends() Figure
    }
    
    class PredictiveAnalyticsDashboard {
        +create_prediction_dashboard(model, X, y) Figure
        +create_feature_importance_chart(model) Figure
    }
    
    class DOCXReportGenerator {
        +generate_comprehensive_report(data) str
        +add_figure_to_doc(doc, fig_path) void
        +add_analysis_section(doc, analysis) void
    }
    
    SalaryAnalyticsModels ..> PredictiveAnalyticsDashboard : uses
    JobMarketNLPAnalyzer ..> DOCXReportGenerator : uses
```

### 4. Configuration System

```mermaid
classDiagram
    class ColumnMapping {
        <<module>>
        +LIGHTCAST_COLUMN_MAPPING: Dict
        +ANALYSIS_COLUMNS: Dict
        +DERIVED_COLUMNS: List
        +EXPERIENCE_CATEGORIES: Dict
        +get_analysis_column(key) str
        +map_lightcast_columns(df) DataFrame
    }
    
    class Settings {
        +raw_data_path: str
        +processed_data_path: str
        +output_dir: str
        +min_salary: int
        +max_salary: int
        +required_columns: List
        +default_chart_height: int
        +test_size: float
        +random_state: int
    }
    
    note for ColumnMapping "Maps:\nUPPERCASE → snake_case\nLogical → Actual columns"
    note for Settings "Centralized\napplication config"
```


---

## Data Flow

### One-Time Processing Flow

```mermaid
flowchart TD
    START([python scripts/create_processed_data.py])
    RAW[(data/raw/lightcast_job_postings.csv<br/>32,364 records<br/>UPPERCASE columns)]
    
    START --> LOAD[Load Raw CSV]
    LOAD --> RAW
    RAW --> STANDARDIZE[standardize_columns]
    
    STANDARDIZE --> MAP[Apply LIGHTCAST_COLUMN_MAPPING<br/>TITLE_NAME → title<br/>COMPANY_NAME → company]
    MAP --> SNAKE[Convert UPPERCASE → snake_case<br/>CITY_NAME → city_name]
    SNAKE --> DECODE[Decode base64 locations<br/>CITY base64 → city_name plain text]
    DECODE --> JSON[Parse JSON locations<br/>LOCATION JSON → coordinates]
    
    JSON --> SALARY[Compute salary_avg]
    
    subgraph SALARY_COMPUTE[" "]
        S1[Check for SALARY_AVG]
        S2{Exists?}
        S3[Use SALARY_AVG directly]
        S4[Compute from salary_min/max]
        S5[Intelligent Imputation<br/>Group by: city, experience,<br/>title, industry]
        
        S1 --> S2
        S2 -->|Yes| S3
        S2 -->|No| S4
        S4 --> S5
    end
    
    SALARY --> SALARY_COMPUTE
    SALARY_COMPUTE --> VALIDATE[Validate salary data<br/>Range: 20K-500K<br/>Remove outliers]
    
    VALIDATE --> EXPERIENCE[Process experience data<br/>experience_min/max → numeric<br/>Derived: avg, range]
    
    EXPERIENCE --> CLEAN[Clean & Standardize<br/>Remove nulls<br/>Trim strings]
    
    CLEAN --> PARQUET[(data/processed/<br/>job_market_processed.parquet<br/>32,364 records<br/>132 snake_case columns<br/>117.8 MB)]
    
    PARQUET --> DONE([✅ Ready for Runtime])
    
    style START fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
    style DONE fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
    style RAW fill:#ff9800,stroke:#e65100,stroke-width:2px
    style PARQUET fill:#2196f3,stroke:#0d47a1,stroke-width:3px,color:#fff
    style SALARY_COMPUTE fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

### Runtime Data Flow (Quarto Website)

```mermaid
flowchart TD
    START([quarto preview])
    QMD[*.qmd files load]
    
    START --> QMD
    QMD --> IMPORT[Import website_processor]
    
    IMPORT --> GET_DF[get_processed_dataframe]
    
    GET_DF --> CHECK{Parquet exists?}
    
    CHECK -->|Yes| LOAD_FAST[Load Parquet<br/>~100ms ⚡]
    CHECK -->|No| LOAD_SLOW[Process Raw CSV<br/>~5s 🐌]
    
    LOAD_FAST --> DF[DataFrame<br/>snake_case columns]
    LOAD_SLOW --> DF
    
    DF --> SUMMARY[get_website_data_summary<br/>total_records, salary_range, etc.]
    
    SUMMARY --> VIZ[Create Visualizations]
    
    subgraph VIZ_PROCESS[" "]
        V1[SalaryVisualizer df]
        V2[get_analysis_column 'salary'<br/>→ Returns 'salary_avg']
        V3[KeyFindingsDashboard df]
        V4[create_key_metrics_cards]
        V5[create_career_progression]
        
        V1 --> V2
        V3 --> V4
        V3 --> V5
    end
    
    VIZ --> VIZ_PROCESS
    
    VIZ_PROCESS --> FIGS[Generate figures/]
    
    subgraph FORMATS[" "]
        F1[*.html - Interactive Plotly]
        F2[*.png - Static for DOCX]
        F3[*.svg - Vector for scaling]
    end
    
    FIGS --> FORMATS
    FORMATS --> RENDER[Render HTML pages]
    RENDER --> OUTPUT[Output to _salary/]
    OUTPUT --> DONE([✅ Website Ready])
    
    style START fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
    style DONE fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
    style LOAD_FAST fill:#4caf50,stroke:#1b5e20,stroke-width:2px
    style LOAD_SLOW fill:#f44336,stroke:#b71c1c,stroke-width:2px
    style VIZ_PROCESS fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style FORMATS fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

### Column Name Resolution Flow

```mermaid
flowchart LR
    QMD[QMD File needs:<br/>'industry' analysis]
    
    QMD --> IMPORT[Import get_analysis_column]
    IMPORT --> CALL[get_analysis_column 'industry']
    CALL --> LOOKUP[Lookup ANALYSIS_COLUMNS]
    
    subgraph MAPPING[" "]
        M1["ANALYSIS_COLUMNS = {<br/>'industry': 'naics2_name'<br/>}"]
    end
    
    LOOKUP --> MAPPING
    MAPPING --> RETURN[Returns: 'naics2_name']
    
    RETURN --> USE[visualizer.plot_salary_by_category<br/>'naics2_name']
    
    USE --> CHECK{Column<br/>exists?}
    CHECK -->|Yes| PLOT[Create Chart ✅]
    CHECK -->|No| ERROR[Error: Column not found ❌]
    
    style QMD fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style MAPPING fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style PLOT fill:#4caf50,stroke:#1b5e20,stroke-width:2px,color:#fff
    style ERROR fill:#f44336,stroke:#b71c1c,stroke-width:2px,color:#fff
```


---

## Quarto Website Integration

### Page Architecture Flow

```mermaid
flowchart TD
    subgraph QMD_STRUCTURE["📄 QMD File Structure"]
        YAML[1. YAML Header<br/>title, format, theme]
        DATA_LOAD[2. Data Loading Block<br/>get_processed_dataframe]
        VIZ_BLOCKS[3. Visualization Blocks<br/>SalaryVisualizer<br/>KeyFindingsDashboard]
    end
    
    YAML --> DATA_LOAD
    DATA_LOAD --> VIZ_BLOCKS
    
    subgraph RULES["✅ QMD Rules"]
        R1[✅ Use abstraction layer classes]
        R2[✅ Use get_analysis_column]
        R3[✅ Keep logic in src/ modules]
        R4[❌ NO groupby, agg in QMD]
        R5[❌ NO data wrangling in QMD]
        R6[❌ NO hardcoded column names]
    end
    
    VIZ_BLOCKS -.follows.-> RULES
    
    style QMD_STRUCTURE fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    style RULES fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style R1 fill:#c8e6c9,stroke:#2e7d32
    style R2 fill:#c8e6c9,stroke:#2e7d32
    style R3 fill:#c8e6c9,stroke:#2e7d32
    style R4 fill:#ffcdd2,stroke:#c62828
    style R5 fill:#ffcdd2,stroke:#c62828
    style R6 fill:#ffcdd2,stroke:#c62828
```

### Page Relationships

```mermaid
graph TB
    HOME[index.qmd<br/>Homepage<br/>Key metrics & navigation]
    
    PIPELINE[data-pipeline.qmd<br/>Pipeline Demo<br/>Processing statistics]
    
    INSIGHTS[salary-insights.qmd<br/>Salary Analysis<br/>Distribution, Experience,<br/>Industry, Geography]
    
    MARKET[market-dashboard.qmd<br/>Market Overview<br/>Executive dashboards,<br/>Geographic trends]
    
    PREDICT[predictive-analytics.qmd<br/>ML Models<br/>Regression, Classification,<br/>Predictions]
    
    REPORT[tech-career-intelligence-report.qmd<br/>DOCX Report<br/>Consolidated analysis]
    
    HOME --> PIPELINE
    HOME --> INSIGHTS
    HOME --> MARKET
    HOME --> PREDICT
    HOME --> REPORT
    
    PIPELINE -.shares data.-> INSIGHTS
    INSIGHTS -.shares data.-> MARKET
    MARKET -.shares data.-> PREDICT
    
    style HOME fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
    style PIPELINE fill:#2196f3,stroke:#0d47a1,stroke-width:2px,color:#fff
    style INSIGHTS fill:#9c27b0,stroke:#4a148c,stroke-width:2px,color:#fff
    style MARKET fill:#ff9800,stroke:#e65100,stroke-width:2px,color:#fff
    style PREDICT fill:#f44336,stroke:#b71c1c,stroke-width:2px,color:#fff
    style REPORT fill:#607d8b,stroke:#263238,stroke-width:2px,color:#fff
```

### Page Descriptions

| File | Purpose | Key Visualizations |
|------|---------|-------------------|
| `index.qmd` | Homepage with key metrics | Overview cards, navigation |
| `data-pipeline.qmd` | Data processing pipeline demo | Pipeline stats, processing flow |
| `salary-insights.qmd` | Comprehensive salary analysis | Distribution, experience, industry, geography |
| `market-dashboard.qmd` | Executive market overview | Geographic trends, market insights, AI analysis |
| `predictive-analytics.qmd` | ML models and predictions | Regression, classification, salary predictions |
| `tech-career-intelligence-report.qmd` | Downloadable DOCX report | Consolidated analysis |

---

## Jupyter Notebooks

### Notebook Structure

```mermaid
graph TD
    subgraph NOTEBOOKS["📓 Jupyter Notebooks"]
        N1[data_processing_pipeline_demo.ipynb<br/>Data Pipeline Showcase]
        N2[ml_feature_engineering_lab.ipynb<br/>ML Models & Analytics]
        N3[job_market_skill_analysis.ipynb<br/>Skills & NLP Analysis]
    end
    
    subgraph N1_SECTIONS["Pipeline Demo Sections"]
        N1S1[1. Load raw data]
        N1S2[2. standardize_columns process]
        N1S3[3. Before/After statistics]
        N1S4[4. Column mapping demo]
        N1S5[5. Export to Parquet]
    end
    
    subgraph N2_SECTIONS["ML Lab Sections"]
        N2S1[1. KMeans clustering]
        N2S2[2. Linear Regression]
        N2S3[3. Classification]
        N2S4[4. Model evaluation]
        N2S5[5. Feature importance]
    end
    
    subgraph N3_SECTIONS["Skills Analysis Sections"]
        N3S1[1. Extract skills from descriptions]
        N3S2[2. Topic clustering TF-IDF, LDA]
        N3S3[3. Word clouds]
        N3S4[4. Skill trends by industry]
        N3S5[5. AI/ML skill detection]
    end
    
    N1 --> N1_SECTIONS
    N2 --> N2_SECTIONS
    N3 --> N3_SECTIONS
    
    style NOTEBOOKS fill:#ff9800,stroke:#e65100,stroke-width:3px
    style N1_SECTIONS fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style N2_SECTIONS fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style N3_SECTIONS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

### Notebook vs QMD Comparison

```mermaid
graph LR
    subgraph JUPYTER["📓 Jupyter Notebooks"]
        J1[Purpose: Exploration]
        J2[Audience: Data Scientists]
        J3[Execution: Manual]
        J4[Data: Raw or Processed]
        J5[Logic: Inline allowed]
    end
    
    subgraph QMD["📄 Quarto QMD Files"]
        Q1[Purpose: Production]
        Q2[Audience: General Users]
        Q3[Execution: Automatic]
        Q4[Data: Processed only]
        Q5[Logic: Abstraction layer only]
    end
    
    J1 -.vs.-> Q1
    J2 -.vs.-> Q2
    J3 -.vs.-> Q3
    J4 -.vs.-> Q4
    J5 -.vs.-> Q5
    
    style JUPYTER fill:#ff9800,stroke:#e65100,stroke-width:2px
    style QMD fill:#2196f3,stroke:#0d47a1,stroke-width:2px,color:#fff
```


---

## Configuration Management

### Column Mapping System

```mermaid
flowchart TD
    RAW[Raw Data<br/>UPPERCASE columns<br/>TITLE_NAME, CITY_NAME]
    
    STEP1[STEP 1:<br/>LIGHTCAST_COLUMN_MAPPING]
    
    subgraph MAPPING1[" "]
        M1["TITLE_NAME → title<br/>COMPANY_NAME → company<br/>CITY_NAME → city_name<br/>NAICS2_NAME → naics2_name"]
    end
    
    RAW --> STEP1
    STEP1 --> MAPPING1
    
    MAPPING1 --> PROCESSED[Processed Data<br/>snake_case columns<br/>title, city_name]
    
    PROCESSED --> STEP2[STEP 2:<br/>ANALYSIS_COLUMNS]
    
    subgraph MAPPING2[" "]
        M2["'salary' → 'salary_avg'<br/>'industry' → 'naics2_name'<br/>'city' → 'city_name'<br/>'experience' → 'experience_years'"]
    end
    
    STEP2 --> MAPPING2
    
    MAPPING2 --> ABSTRACT[Abstraction Layer<br/>get_analysis_column 'salary'<br/>→ Returns 'salary_avg']
    
    ABSTRACT --> USAGE[QMD Files use:<br/>get_analysis_column 'key'<br/>to get actual column name]
    
    style RAW fill:#ff9800,stroke:#e65100,stroke-width:2px
    style PROCESSED fill:#4caf50,stroke:#1b5e20,stroke-width:2px,color:#fff
    style MAPPING1 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style MAPPING2 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style ABSTRACT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style USAGE fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

### Derived Columns Flow

```mermaid
flowchart LR
    subgraph INPUT["Input Columns"]
        I1[salary_min]
        I2[salary_max]
        I3[experience_min]
        I4[experience_max]
        I5[CITY base64]
    end
    
    subgraph PIPELINE["Data Pipeline"]
        P1[Compute salary_avg<br/>salary_min + salary_max / 2]
        P2[Compute experience_avg<br/>experience_min + max / 2]
        P3[Decode CITY<br/>base64 → city_name]
        P4[Create experience_range<br/>max - min]
        P5[Flag ai_related<br/>Check skills for AI/ML]
    end
    
    subgraph OUTPUT["Derived Columns"]
        O1[salary_avg ✨]
        O2[experience_avg ✨]
        O3[city_name ✨]
        O4[experience_range ✨]
        O5[ai_related ✨]
    end
    
    I1 --> P1
    I2 --> P1
    I3 --> P2
    I4 --> P2
    I4 --> P4
    I3 --> P4
    I5 --> P3
    
    P1 --> O1
    P2 --> O2
    P3 --> O3
    P4 --> O4
    P5 --> O5
    
    style INPUT fill:#ffebee,stroke:#c62828,stroke-width:2px
    style PIPELINE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style OUTPUT fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

---

## Summary

### Key Architectural Decisions

```mermaid
mindmap
  root((Tech Career<br/>Intelligence))
    Process Once
      Raw CSV to Parquet
      One-time processing
      Fast runtime loads
    Abstraction Layer
      Zero logic in QMD
      All logic in src/
      Reusable classes
    Column Standards
      All snake_case
      Centralized mapping
      get_analysis_column
    Configuration
      column_mapping.py
      settings.py
      DRY principle
    Modular Design
      data/
      visualization/
      analytics/
      config/
```

### Class Responsibilities

```mermaid
graph TB
    subgraph DATA["📊 Data Classes"]
        D1[website_processor<br/>Load & process data]
        D2[JobMarketDataCleaner<br/>Clean & validate]
        D3[DataLoader<br/>Load from files]
    end
    
    subgraph VIZ["📈 Visualization Classes"]
        V1[SalaryVisualizer<br/>Salary-focused charts]
        V2[KeyFindingsDashboard<br/>Executive dashboards]
        V3[JobMarketTheme<br/>Consistent styling]
    end
    
    subgraph ANALYTICS["🤖 Analytics Classes"]
        A1[SalaryAnalyticsModels<br/>ML models]
        A2[JobMarketNLPAnalyzer<br/>Skills extraction]
        A3[PredictiveAnalyticsDashboard<br/>ML dashboards]
    end
    
    QMD[*.qmd Files] --> DATA
    QMD --> VIZ
    QMD --> ANALYTICS
    
    DATA --> VIZ
    VIZ --> A3
    
    style QMD fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
    style DATA fill:#2196f3,stroke:#0d47a1,stroke-width:2px,color:#fff
    style VIZ fill:#9c27b0,stroke:#4a148c,stroke-width:2px,color:#fff
    style ANALYTICS fill:#ff9800,stroke:#e65100,stroke-width:2px
```

### Data Flow Principles

```mermaid
flowchart TD
    RAW[Raw Data<br/>CSV, UPPERCASE]
    
    PROCESS[One-time Processing<br/>create_processed_data.py]
    
    PARQUET[Processed Data<br/>Parquet, snake_case]
    
    RUNTIME[Runtime Load<br/>Direct, no processing]
    
    ABSTRACTION[Abstraction Layer<br/>SalaryVisualizer<br/>KeyFindingsDashboard]
    
    PRESENTATION[Presentation Layer<br/>QMD files<br/>No business logic]
    
    OUTPUT[HTML Website<br/>_salary/]
    
    RAW --> PROCESS
    PROCESS --> PARQUET
    PARQUET --> RUNTIME
    RUNTIME --> ABSTRACTION
    ABSTRACTION --> PRESENTATION
    PRESENTATION --> OUTPUT
    
    style RAW fill:#ff9800,stroke:#e65100,stroke-width:2px
    style PROCESS fill:#f44336,stroke:#b71c1c,stroke-width:2px,color:#fff
    style PARQUET fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
    style RUNTIME fill:#2196f3,stroke:#0d47a1,stroke-width:2px,color:#fff
    style ABSTRACTION fill:#9c27b0,stroke:#4a148c,stroke-width:2px,color:#fff
    style PRESENTATION fill:#00bcd4,stroke:#006064,stroke-width:2px,color:#fff
    style OUTPUT fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:#fff
```

---

**Last Updated**: October 2025  
**Version**: 3.0 (Mermaid Diagrams Edition)  
**Format**: Markdown with Mermaid flowcharts, class diagrams, and mind maps

