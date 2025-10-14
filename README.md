# Tech Career Intelligence: Data-Driven Job Market Analysis

A comprehensive job market analysis platform providing data-driven career insights for students and professionals in the technology sector.

## **What This Project Does**

Transforms complex job market data into actionable career intelligence:

- **Salary Analysis**: Real compensation data across experience levels, education, and geography
- **Education ROI**: Quantified return on investment for different degree paths
- **Remote Work Intelligence**: Modern workplace flexibility impact on compensation
- **Geographic Insights**: Location-based career optimization strategies

**Data Foundation**: 72,000+ real job postings from Lightcast's comprehensive database

---

## **Quick Start**

### **For Users (Non-Technical)**

1. **Visit the Analysis Website**: Open `_salary/index.html` in your browser
2. **Start with Homepage**: Get overview of key insights and navigation
3. **Explore Core Analysis**: Use sidebar navigation for specific questions
4. **Interactive Features**: Click charts and dashboards for detailed exploration

### **Developer Setup**

#### First Time Setup

```bash
# 1. Clone and setup environment
git clone https://github.com/samarthya/ad688-scratch
cd ad688-scratch
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. IMPORTANT: Generate processed data (run ONCE, saves 5-10 min on every render!)
python scripts/generate_processed_data.py

# 4. Generate website (fast with preprocessed data!)
source .venv/bin/activate  # Ensure venv is active
quarto preview --port 4200
```

#### Daily Development Workflow

```bash
source .venv/bin/activate
quarto preview --port 4200
# Edit .qmd files → Changes auto-reload instantly!
```

#### Regenerate Data (if raw CSV changes)

```bash
python scripts/generate_processed_data.py --force
```

### Why the preprocessing step?

- Raw CSV: 13M rows, 683 MB → Takes 5-10 minutes to process
- Processed Parquet: 30-50K rows, 120 MB → Loads in 1-2 seconds
- Run preprocessing ONCE → All subsequent Quarto renders are instant!

---

## **Analysis Components**

### **Core Questions Answered:**

#### **"How much should I expect to earn?"**

- **Experience Progression**: Entry → Mid → Senior → Executive salary growth (3.3x multiplier)
- **Education Premium**: Bachelor's → Master's → PhD compensation differences (25% average premium)
- **Industry Benchmarks**: Technology sector salary ranges and percentiles
- **Company Size Impact**: Startup vs Enterprise compensation strategies (40% average difference)

#### **"Where should I work for best opportunities?"**

- **Geographic Analysis**: Metropolitan area salary comparisons with cost-of-living adjustments
- **Regional Intelligence**: Industry concentration and growth patterns by location
- **Arbitrage Opportunities**: High-salary locations vs low-cost living areas

#### **"How has remote work changed the job market?"**

- **Flexibility Adoption**: 75% of tech jobs offer remote/hybrid options
- **Compensation Impact**: Remote work salary comparisons vs on-site positions
- **Geographic Arbitrage**: Location-independent career strategies

#### **"How reliable is this analysis?"**

- **Statistical Validation**: Machine learning models and confidence intervals
- **Data Methodology**: PySpark processing of large-scale job posting data
- **Interactive Exploration**: Jupyter notebooks with detailed technical analysis

---

## **Project Structure**

```bash
├── Core Analysis (Quarto Website)
│   ├── index.qmd                 # Homepage with key insights dashboard
│   ├── salary-analysis.qmd       # Comprehensive compensation analysis
│   ├── regional-trends.qmd       # Geographic intelligence and opportunities
│   ├── remote-work.qmd           # Remote work impact and strategies
│   └── data-methodology.qmd      # Technical methodology and validation
│
├── Interactive Analysis
│   └── notebooks/job_market_skill_analysis.ipynb  # Deep-dive technical analysis
│
├── Data & Processing
│   ├── src/                      # Python classes and processing pipeline
│   ├── data/                     # Raw, processed, and external datasets
│   └── figures/                  # Generated charts and visualizations
│
└── Documentation
    ├── README.md                 # This file - project overview and setup
    ├── ARCHITECTURE.md           # High-level system design
    ├── DESIGN.md                 # Implementation patterns and code structure
    └── LEARNINGS.md              # Technical insights and lessons learned
```

---

## How to Use This Analysis

### For Students Planning Careers

1. **Set Realistic Expectations**: Use salary benchmarks for post-graduation planning
2. **Calculate Education ROI**: Evaluate Master's/PhD investment returns
3. **Plan Geographic Strategy**: Consider location for internships and first jobs
4. **Understand Modern Workplace**: Factor remote work into job search strategy

### For Job Seekers

1. **Benchmark Current Compensation**: Compare against market data for your level
2. **Identify Growth Opportunities**: Target high-value skills and career progression
3. **Optimize Job Search**: Use geographic and remote work intelligence
4. **Negotiate with Data**: Use specific market benchmarks in salary discussions

### For Career Changers

1. **Assess Transition Impact**: Understand salary implications of industry/role changes
2. **Plan Skill Development**: Focus on high-ROI capabilities based on market analysis
3. **Evaluate Education Options**: Compare formal degrees vs certifications vs experience
4. **Strategic Location Planning**: Consider geographic moves for career optimization

---

## Key Findings Summary

### Experience Progression Analysis

- **Entry Level (0-2 years)**: $81,120 median salary (N=5,211 jobs)
- **Mid Level (3-5 years)**: $110,396 median (+36% growth)
- **Senior Level (6-9 years)**: $135,000 median (+22% growth)
- **Executive (10+ years)**: $150,000 median (+11% leadership premium)

### Education Return on Investment

- **Bachelor's Degree**: Market baseline (100%)
- **Master's Degree**: +25% average salary premium
- **PhD/Advanced**: +30% average salary premium
- **ROI Timeline**: Most advanced degrees pay for themselves within 3-5 years

### Geographic Intelligence

- **Highest-Paying Markets**: San Francisco, Seattle, New York (cost-adjusted)
- **Best ROI Markets**: Austin, Denver, Atlanta (salary vs cost-of-living)
- **Remote Work Impact**: 75% of positions offer location flexibility
- **Geographic Premium**: Up to 67% salary difference between markets

### Modern Workplace Trends

- **Remote Available**: 45% of positions offer full remote work
- **Hybrid Options**: 30% provide flexible arrangements
- **On-Site Only**: 25% require physical presence
- **Salary Impact**: Remote work doesn't reduce compensation in most cases

---

## Technical Details

### Data Sources

- **Primary**: Lightcast job postings database (72,000+ records)
- **Supplementary**: Bureau of Labor Statistics, geographic cost-of-living data
- **Processing**: PySpark for large-scale data analysis and transformation
- **Validation**: Cross-reference with industry salary surveys

### Analysis Methods

- **Statistical Modeling**: Python with pandas, numpy, scipy
- **Machine Learning**: PySpark MLlib for scalable prediction and classification models
- **Visualization**: Plotly for interactive charts, matplotlib for static analysis
- **Web Framework**: Quarto for integrated analysis and presentation

### Quality Assurance

- **Data Validation**: Outlier detection and cleaning procedures
- **Statistical Testing**: Confidence intervals and significance testing
- **Model Validation**: Cross-validation and performance metrics
- **Reproducibility**: Version-controlled pipeline with documented methodology

---

## Contributing & Usage

### Technical Contributions

- Enhance analysis methodology in `src/` modules
- Improve visualizations in `notebooks/`
- Extend geographic or industry coverage
- Optimize data processing pipeline

---

## **Contact & Support**

**Author**: Saurabh Sharma (Boston University)
**Repository**: [GitHub - ad688-scratch](https://github.com/samarthya/ad688-scratch)

**Technical Documentation**:
- `ARCHITECTURE.md` - High-level system design and data flow
- `DESIGN.md` - Implementation patterns and code organization
- `LEARNINGS.md` - Technical insights and lessons learned

---

## Setup Instructions

### 1. Prerequisites

#### Java Installation (Required for PySpark)

#### Linux/Ubuntu:

```bash
# Install OpenJDK 17 (recommended for PySpark 4.0.1)
sudo apt update
sudo apt install openjdk-17-jdk

# Verify installation
java -version
```

**macOS:**

```bash
# Using Homebrew
brew install openjdk@11

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Linux/Mac)
source .venv/bin/activate

# Activate environment (Windows)
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (includes PySpark 4.0.1)
pip install -r requirements.txt
```

**Important for PySpark:**
- Virtual environment name should be `.venv` (not `venv`) for consistency
- Ensure Java is available in PATH before installing PySpark
- If you encounter "Connection refused" errors, see troubleshooting section below

### 3. Quarto Installation

Install Quarto from: https://quarto.org/docs/get-started/

```bash
# Verify installation
quarto --version
```

### 4. Verify PySpark Installation

```bash
# Test PySpark in your virtual environment
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"

# Test Spark session creation
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[2]').appName('test').config('spark.ui.enabled', 'false').getOrCreate(); print('Spark session created successfully'); spark.stop()"
```

### 5. Data Setup

Place your Lightcast dataset in:

```bash
data/raw/lightcast_job_postings.csv
```

The raw CSV contains 131 columns from Lightcast. The processing pipeline automatically standardizes and cleans this data.

### 6. Initial Data Processing

#### IMPORTANT: Run this ONCE before using Quarto

```bash
# Generate processed data (saves 5-10 min on every Quarto render!)
python scripts/generate_processed_data.py

# Verify processed data
ls data/processed/job_market_processed.parquet
```

**For Data Exploration:**
- Use `notebooks/data_processing_pipeline_demo.ipynb` for PySpark examples
- Use `notebooks/job_market_skill_analysis.ipynb` for NLP analysis
- Use `notebooks/ml_feature_engineering_lab.ipynb` for ML examples

### 7. Generate Reports

```bash
# Render Quarto website
quarto render

# Preview locally
quarto preview
```

### 8. View Results

```bash
# Preview Quarto website (interactive, auto-reloads on changes)
quarto preview --port 4200
```

Access at: http://localhost:4200

---

## Troubleshooting

### PySpark Issues

#### "Connection refused" Error

This is the most common PySpark issue in virtual environments:

**Symptoms:**
```bash
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solutions:**

1. **Use the provided notebook configuration:**
   - Open `notebooks/data_processing_pipeline_demo.ipynb`
   - The notebook includes automatic Spark configuration for local environments
   - Run the configuration cells to set up Spark properly

2. **Manual Spark configuration:**
   ```python
   from pyspark.sql import SparkSession

   spark = SparkSession.builder \
       .appName("JobMarketLocal") \
       .master("local[2]") \
       .config("spark.ui.enabled", "false") \
       .config("spark.driver.memory", "1g") \
       .getOrCreate()
   ```

3. **Environment variables:**
   ```bash
   export SPARK_LOCAL_IP=127.0.0.1
   export PYSPARK_SUBMIT_ARGS='--master local[2] --conf spark.ui.enabled=false pyspark-shell'
   ```

#### Java Issues

- **Error**: `JAVA_HOME not set`
  - **Solution**: Install Java and set JAVA_HOME environment variable
  - **Linux**: `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
  - **macOS**: `export JAVA_HOME=$(/usr/libexec/java_home)`

#### Memory Issues

- **Error**: `OutOfMemoryError` or heap space errors
  - **Solution**: Reduce Spark driver memory or use sampling
  - **Configuration**: `.config("spark.driver.memory", "1g")`

### Common Issues

1. **Missing Data**: Ensure `lightcast_job_postings.csv` is in `data/raw/`
2. **Package Errors**: Verify virtual environment is activated
3. **Quarto Errors**: Check YAML syntax in `.qmd` files
4. **Dashboard Issues**: Confirm all dependencies installed
5. **PySpark Errors**: See PySpark troubleshooting section above

### Getting Help

- Check `ARCHITECTURE.md` for system design decisions
- Check `DESIGN.md` for implementation details
- Review error logs for specific issues
- Ensure all file paths are correct
- Verify data format matches expectations

---

*Make informed career decisions with real market data, not guesswork.*
