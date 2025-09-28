# Class Architecture: UML Diagram

Visual representation of the job market analysis system's class relationships and data flow.

> **System overview and design philosophy**: See [Technical Design](../DESIGN.md)  
> **Implementation details and usage patterns**: See individual class documentation in `src/`

## Architecture Diagram

> **Note:** If the Mermaid diagram below doesn't display properly in your markdown viewer, see the [Text-Based Diagram](#text-based-class-diagram) section for an alternative representation.

```mermaid
classDiagram
    %% Core Data Processing Classes
    class JobMarketDataProcessor {
        -SparkSession spark
        -StructType lightcast_schema
        -DataFrame df_raw
        -DataFrame df_processed
        +__init__(app_name: str)
        +load_data(file_path: str, use_sample: bool, sample_size: int) DataFrame
        +assess_data_quality() Dict
        +clean_job_data() DataFrame
        +engineer_features() DataFrame
        +export_processed_data(output_path: str) void
        +get_data_summary() Dict
    }

    class AdvancedJobDataProcessor {
        -SparkSession spark
        -str data_path
        -Dict spark_config
        -DataFrame df_raw
        -DataFrame df_clean
        +__init__(data_path: str, spark_config: Dict)
        +load_raw_data() DataFrame
        +perform_data_quality_assessment() Dict
        +execute_comprehensive_cleaning() DataFrame
        +apply_feature_engineering() DataFrame
        +save_processed_data(output_path: str) void
        +generate_processing_report() str
    }

    class SparkJobAnalyzer {
        -SparkSession spark
        -DataFrame job_data
        +__init__(spark_session: SparkSession)
        +load_full_dataset(data_path: str) DataFrame
        +get_industry_analysis(top_n: int) DataFrame
        +get_experience_analysis() DataFrame
        +get_geographic_analysis(top_n: int) DataFrame
        +get_overall_statistics() Dict
        +get_skills_analysis(top_n: int) DataFrame
        +create_relational_view(table_name: str) void
        +execute_custom_query(query: str) DataFrame
        +stop() void
        -_create_sample_data() DataFrame
    }

    %% Visualization Classes
    class SalaryVisualizer {
        -DataFrame df
        +__init__(df: DataFrame)
        +get_industry_salary_analysis(top_n: int) DataFrame
        +get_experience_salary_analysis() DataFrame
        +get_education_premium_analysis() DataFrame
        +get_overall_statistics() Dict
        +get_geographic_salary_analysis(top_n: int) DataFrame
        +get_experience_progression_analysis() Dict
    }

    %% Data Schema Classes (Actual Lightcast Structure)
    class LightcastSchema {
        <<Raw Data - 131 Columns>>
        +ID: StringType
        +TITLE: StringType
        +TITLE_CLEAN: StringType
        +COMPANY: StringType
        +LOCATION: StringType
        +POSTED: StringType
        +SALARY: DoubleType
        +SALARY_FROM: DoubleType
        +SALARY_TO: DoubleType
        +ORIGINAL_PAY_PERIOD: StringType
        +NAICS2_NAME: StringType
        +MIN_YEARS_EXPERIENCE: IntegerType
        +MAX_YEARS_EXPERIENCE: IntegerType
        +SKILLS_NAME: StringType
        +EDUCATION_LEVELS_NAME: StringType
        +REMOTE_TYPE_NAME: StringType
        +EMPLOYMENT_TYPE_NAME: StringType
        +... 113 more columns: StringType
    }

    class ProcessedSchema {
        <<Analysis-Ready Format>>
        +job_id: StringType
        +title: StringType
        +title_clean: StringType
        +company: StringType
        +location: StringType
        +industry: StringType
        +salary_min: DoubleType
        +salary_max: DoubleType
        +salary_single: DoubleType
        +salary_avg_imputed: DoubleType
        +experience_min: IntegerType
        +experience_max: IntegerType
        +experience_level: StringType
        +required_skills: StringType
        +education_required: StringType
        +remote_type: StringType
        +remote_allowed: BooleanType
        +ai_related: BooleanType
        +industry_clean: StringType
    }

    %% Usage/Dependency Relationships
    JobMarketDataProcessor --> LightcastSchema : "uses raw schema"
    JobMarketDataProcessor --> ProcessedSchema : "produces clean schema"
    AdvancedJobDataProcessor --> LightcastSchema : "uses raw schema" 
    AdvancedJobDataProcessor --> ProcessedSchema : "produces clean schema"
    SparkJobAnalyzer --> ProcessedSchema : "analyzes processed data"
    SalaryVisualizer --> ProcessedSchema : "visualizes processed data"
    
    %% Composition Relationships
    SparkJobAnalyzer *-- SparkSession : "owns"
    JobMarketDataProcessor *-- SparkSession : "owns"
    AdvancedJobDataProcessor *-- SparkSession : "owns"
    SalaryVisualizer *-- DataFrame : "operates on"

    %% Inheritance/Implementation
    JobMarketDataProcessor ..|> DataProcessor : "implements"
    AdvancedJobDataProcessor ..|> DataProcessor : "implements"
    
    %% Interface/Abstract Classes
    class DataProcessor {
        <<interface>>
        +load_data() DataFrame
        +clean_data() DataFrame
        +export_data() void
    }

    %% Analysis Pipeline Flow
    class AnalysisPipeline {
        -JobMarketDataProcessor processor
        -SparkJobAnalyzer analyzer
        -SalaryVisualizer visualizer
        +__init__()
        +run_full_pipeline() Dict
        +generate_reports() void
        +export_results() void
    }

    AnalysisPipeline --> JobMarketDataProcessor : "orchestrates"
    AnalysisPipeline --> SparkJobAnalyzer : "orchestrates" 
    AnalysisPipeline --> SalaryVisualizer : "orchestrates"

    %% Notes and Design Patterns
    SparkJobAnalyzer : SQL-based analysis engine
    SparkJobAnalyzer : Replaces pandas operations
    SparkJobAnalyzer : with PySpark SQL queries
    SparkJobAnalyzer : Status: Implemented
    
    SalaryVisualizer : Lightweight visualization
    SalaryVisualizer : Compatible with existing code
    SalaryVisualizer : Falls back gracefully
    SalaryVisualizer : Status: Implemented
    
    JobMarketDataProcessor : Comprehensive processor
    JobMarketDataProcessor : Data quality assessment
    JobMarketDataProcessor : Feature engineering
    JobMarketDataProcessor : Status: Needs salary pipeline
    
    LightcastSchema : Raw 131-column format
    LightcastSchema : SALARY_FROM/TO → salary_avg_imputed
    LightcastSchema : NAICS2_NAME → industry_clean
    LightcastSchema : Status: Documented
```

## Key Relationships

**Data Flow**: `LightcastSchema` (raw 131 columns) → `JobMarketDataProcessor` → `ProcessedSchema` (analysis-ready) → `SparkJobAnalyzer` & `SalaryVisualizer`

**Processing Pipeline**: Raw CSV → Spark processing → salary_avg_imputed calculation → multi-format output

> **Complete implementation details**: See [Technical Design](../DESIGN.md#data-processing-pipeline)  
> **Column mapping specifications**: See [Technical Design](../DESIGN.md#column-mapping--transformation-strategy)

---

## Text-Based Class Diagram

If the Mermaid diagram above doesn't render properly, here's a text-based representation:

```
ANALYSIS LAYER
┌─────────────────────────────────────────────────────────────────────┐
│                         SparkJobAnalyzer                            │
├─────────────────────────────────────────────────────────────────────┤
│ - spark: SparkSession                                               │
│ - job_data: DataFrame                                               │
├─────────────────────────────────────────────────────────────────────┤
│ + get_industry_analysis(top_n: int): DataFrame                      │
│ + get_skills_analysis(top_n: int): DataFrame                        │
│ + get_geographic_analysis(top_n: int): DataFrame                    │
│ + get_overall_statistics(): Dict                                    │
│ + execute_custom_query(query: str): DataFrame                       │
└─────────────────────────────────────────────────────────────────────┘

DATA PROCESSING LAYER
┌─────────────────────────────────────────────────────────────────────┐
│                      JobMarketDataProcessor                         │
├─────────────────────────────────────────────────────────────────────┤
│ - spark: SparkSession                                               │
│ - df_raw: DataFrame                                                 │
│ - df_processed: DataFrame                                           │
├─────────────────────────────────────────────────────────────────────┤
│ + load_data(file_path: str): DataFrame                             │
│ + assess_data_quality(): Dict                                       │
│ + clean_job_data(): DataFrame                                       │
│ + engineer_features(): DataFrame                                    │
│ + export_processed_data(output_path: str): void                     │
└─────────────────────────────────────────────────────────────────────┘

VISUALIZATION LAYER
┌─────────────────────────────────────────────────────────────────────┐
│                         SalaryVisualizer                            │
├─────────────────────────────────────────────────────────────────────┤
│ - df: DataFrame                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ + get_industry_salary_analysis(top_n: int): DataFrame              │
│ + get_experience_salary_analysis(): DataFrame                      │
│ + get_education_premium_analysis(): DataFrame                       │
│ + get_overall_statistics(): Dict                                    │
│ + get_geographic_salary_analysis(top_n: int): DataFrame            │
└─────────────────────────────────────────────────────────────────────┘

RELATIONSHIPS:
SparkJobAnalyzer ──uses──> ProcessedData
JobMarketDataProcessor ──produces──> ProcessedData
SalaryVisualizer ──operates_on──> ProcessedData

DATA FLOW:
Raw Data → JobMarketDataProcessor → SparkJobAnalyzer → SalaryVisualizer → Results
```

---

**For detailed class documentation, usage patterns, and implementation examples**: See [Technical Design](../DESIGN.md#class-architecture--responsibilities)