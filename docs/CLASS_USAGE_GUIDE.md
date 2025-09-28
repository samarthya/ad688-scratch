# Class-Based Job Market Analysis Architecture

## 🎯 Overview

This project follows a **class-based architecture** design pattern that eliminates code duplication and ensures consistent analysis across all Quarto documents and notebooks. Instead of reinventing the wheel with manual functions, we use well-designed classes that handle specific concerns.

## 🏗️ Architecture Diagram

See [class_architecture.md](docs/class_architecture.md) for the complete UML diagram showing class relationships.

## 🔧 Core Classes

### 1. SparkJobAnalyzer 
**Location:** `src/data/spark_analyzer.py`  
**Purpose:** SQL-based analysis engine replacing manual DataFrame operations

```python
from src.data.spark_analyzer import SparkJobAnalyzer, create_spark_analyzer

# Quick initialization
analyzer = create_spark_analyzer()

# Get comprehensive analyses
industry_results = analyzer.get_industry_analysis(top_n=10)
skills_results = analyzer.get_skills_analysis(top_n=7) 
geographic_results = analyzer.get_geographic_analysis(top_n=8)
stats = analyzer.get_overall_statistics()
```

**Key Methods:**
- `get_industry_analysis()` - Industry salary rankings with AI premium
- `get_skills_analysis()` - Skills premium analysis using SQL pattern matching
- `get_geographic_analysis()` - Location-based salary analysis
- `get_experience_analysis()` - Experience level progression
- `get_overall_statistics()` - Comprehensive dataset statistics
- `execute_custom_query()` - Run custom SQL queries

### 2. JobMarketDataProcessor
**Location:** `src/data/enhanced_processor.py`  
**Purpose:** Comprehensive data processing and cleaning

```python
from src.data.enhanced_processor import JobMarketDataProcessor

processor = JobMarketDataProcessor("MyAnalysis")
processed_df = processor.load_data("data/raw/job_data.csv")
quality_report = processor.assess_data_quality()
clean_df = processor.clean_job_data()
```

**Key Methods:**
- `load_data()` - Smart data loading with schema validation
- `assess_data_quality()` - Comprehensive quality assessment
- `clean_job_data()` - Advanced cleaning pipeline
- `engineer_features()` - AI role detection and feature engineering
- `export_processed_data()` - Multi-format data export

### 3. SalaryVisualizer
**Location:** `src/visualization/simple_plots.py`  
**Purpose:** Pandas-based visualization utilities with graceful fallbacks

```python
from src.visualization.simple_plots import SalaryVisualizer

visualizer = SalaryVisualizer(pandas_df)
industry_analysis = visualizer.get_industry_salary_analysis(top_n=10)
experience_analysis = visualizer.get_experience_salary_analysis()
overall_stats = visualizer.get_overall_statistics()
```

**Key Methods:**
- `get_industry_salary_analysis()` - Industry breakdown with AI premiums
- `get_experience_salary_analysis()` - Experience level analysis  
- `get_education_premium_analysis()` - Education ROI analysis
- `get_geographic_salary_analysis()` - Location-based analysis
- `get_overall_statistics()` - Dataset summary statistics

## 📚 Usage in Quarto Documents

### Before (Manual Functions - DON'T DO THIS):
```python
# ❌ Reinventing the wheel
def analyze_industries(df):
    # 50+ lines of manual DataFrame operations
    industry_stats = df.groupBy("industry").agg(...)
    # More complex logic...
    return results

def analyze_skills(df):  
    # Another 50+ lines of similar operations
    # Duplicate error handling, duplicate SQL logic
    pass
```

### After (Class-Based - DO THIS):
```python
# ✅ Using our architecture
import sys
sys.path.append('src')
from data.spark_analyzer import create_spark_analyzer

analyzer = create_spark_analyzer()
industry_results = analyzer.get_industry_analysis(top_n=10)
skills_results = analyzer.get_skills_analysis(top_n=7)
```

## 📝 Updated Quarto Documents

### data-analysis.qmd
- **Before:** 800+ lines with manual functions
- **After:** Uses `SparkJobAnalyzer` and `JobMarketDataProcessor`
- **Benefit:** Eliminated 300+ lines of duplicate code

### salary-analysis.qmd  
- **Before:** Static hardcoded skills table
- **After:** Dynamic `analyzer.get_skills_analysis()` 
- **Benefit:** Real-time analysis instead of fake data

## 🔄 Class Integration Pattern

```python
# Standard integration pattern for all analysis
import sys
sys.path.append('src')

# Import our classes
from data.spark_analyzer import create_spark_analyzer
from data.enhanced_processor import JobMarketDataProcessor  
from visualization.simple_plots import SalaryVisualizer

# Initialize (shares Spark session and data)
analyzer = create_spark_analyzer()
processor = JobMarketDataProcessor("MyAnalysis")
processor.df_raw = analyzer.job_data  # Share data

# Run analyses using class methods
industry_results = analyzer.get_industry_analysis()
skills_results = analyzer.get_skills_analysis()
stats = analyzer.get_overall_statistics()

# Use results in visualizations
fig = px.bar(industry_results, x="Job Count", y="Industry")
```

## 🎯 Design Benefits

### 1. **Single Responsibility**
- `SparkJobAnalyzer` → Analysis logic only
- `JobMarketDataProcessor` → Data processing only  
- `SalaryVisualizer` → Visualization only

### 2. **Don't Repeat Yourself (DRY)**
- Industry analysis: 1 method, used everywhere
- Skills analysis: 1 method, replaces 5+ manual implementations
- Data loading: 1 method, consistent across all documents

### 3. **Interface Segregation**
- Clean method signatures: `get_industry_analysis(top_n=10)`
- Optional parameters with sensible defaults
- Consistent return types (pandas DataFrames)

### 4. **Dependency Inversion**
- Analysis depends on interfaces, not concrete implementations
- Easy to swap PySpark for pandas, or add new data sources
- Testable and mockable for unit tests

### 5. **Error Handling**
- Built-in graceful fallbacks in all methods
- Consistent logging and error reporting
- Try/catch blocks with meaningful fallback data

## 🚀 Quick Start

1. **Run the demo:**
   ```bash
   python src/demo_class_usage.py
   ```

2. **In Quarto documents:**
   ```python
   import sys; sys.path.append('src')
   from data.spark_analyzer import create_spark_analyzer
   analyzer = create_spark_analyzer()
   results = analyzer.get_industry_analysis()
   ```

3. **In Jupyter notebooks:**
   ```python
   %load_ext autoreload
   %autoreload 2
   import sys; sys.path.append('../src')
   from data.spark_analyzer import create_spark_analyzer
   ```

## 📊 Analysis Methods Comparison

| Analysis Type | Manual Approach | Class-Based Approach |
|---------------|----------------|---------------------|
| Industry Analysis | 50+ lines each document | `analyzer.get_industry_analysis()` |
| Skills Analysis | Static hardcoded table | `analyzer.get_skills_analysis()` |
| Data Loading | Multiple functions | `create_spark_analyzer()` |
| Error Handling | Manual try/catch everywhere | Built into class methods |
| Code Maintenance | Update 5+ locations | Update 1 class method |

## 🔧 Extending the Architecture

### Adding New Analysis Methods

1. **Add to SparkJobAnalyzer:**
   ```python
   def get_remote_work_analysis(self, top_n: int = 10) -> pd.DataFrame:
       query = """
       SELECT 
           remote_allowed_clean,
           COUNT(*) as job_count,
           ROUND(percentile_approx(salary_avg_imputed, 0.5), 0) as median_salary
       FROM job_postings
       WHERE salary_avg_imputed IS NOT NULL
       GROUP BY remote_allowed_clean
       ORDER BY median_salary DESC
       """
       return self.spark.sql(query).toPandas()
   ```

2. **Use in all documents:**
   ```python
   remote_results = analyzer.get_remote_work_analysis()
   ```

### Adding New Visualization Methods

1. **Add to SalaryVisualizer:**
   ```python
   def get_company_size_analysis(self) -> pd.DataFrame:
       # Implementation here
       return results_df
   ```

2. **Consistent usage everywhere:**
   ```python
   company_results = visualizer.get_company_size_analysis()
   ```

## 📁 File Organization

```
src/
├── data/
│   ├── spark_analyzer.py          # SparkJobAnalyzer class
│   ├── enhanced_processor.py      # JobMarketDataProcessor class  
│   └── preprocess_data.py         # Legacy preprocessing
├── visualization/
│   ├── simple_plots.py           # SalaryVisualizer class
│   └── plots.py                  # Advanced plotting utilities
└── demo_class_usage.py           # Usage demonstration

docs/
└── class_architecture.md         # Complete UML diagram

# Quarto documents using the classes:
├── data-analysis.qmd             # Uses all 3 classes
├── salary-analysis.qmd           # Uses SparkJobAnalyzer
└── *.qmd                        # All use class-based approach
```

## ✅ Migration Checklist

- [x] Created comprehensive class architecture
- [x] Updated data-analysis.qmd to use classes
- [x] Updated salary-analysis.qmd skills analysis
- [x] Created UML documentation  
- [x] Created usage demonstration
- [x] Eliminated 300+ lines of duplicate code
- [x] Added error handling and fallbacks
- [x] Ensured consistent interfaces across documents

## 🎉 Results

**Before:** 800+ lines of manual functions across documents  
**After:** 3 reusable classes with 50+ methods  
**Maintenance:** Update 1 class instead of 5+ locations  
**Consistency:** Same analysis results across all documents  
**Scalability:** Easy to add new analysis methods  

This architecture ensures we **"drink our own wine"** - using our own well-designed tools instead of reinventing functionality in every document.