# Career Analytics Project

## Setup Instructions

### 1. Prerequisites

#### Java Installation (Required for PySpark)

**Linux/Ubuntu:**

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

## Project Structure

```
project-from-scratch/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned data
│   └── external/               # Reference datasets
├── src/
│   ├── data/                   # Data processing scripts
│   ├── analysis/               # Statistical analysis
│   └── visualization/          # Plotting functions
├── notebooks/                  # Jupyter explorations
├── reports/                    # Generated analyses
├── dashboards/                 # Interactive apps
├── docs/                      # Website output
├── csl/                       # Citation styles
├── _quarto.yml                # Quarto configuration
├── requirements.txt           # Python dependencies
└── references.bib             # Bibliography
```

## Key Commands

| Task | Command                                        |
|------|---------                                       |
| **Data Processing** | |
| Generate processed data | `python scripts/generate_processed_data.py` |
| Force regenerate data | `python scripts/generate_processed_data.py --force` |
| Explore PySpark pipeline | `jupyter lab notebooks/data_processing_pipeline_demo.ipynb` |
| Explore NLP analysis | `jupyter lab notebooks/job_market_skill_analysis.ipynb` |
| Explore ML features | `jupyter lab notebooks/ml_feature_engineering_lab.ipynb` |
| **Development** | |
| Install packages | `pip install -r requirements.txt` |
| Test PySpark | `python -c "import pyspark; print(pyspark.__version__)"` |
| **Publishing** | |
| Render website | `quarto render` |
| Preview site | `quarto preview --port 4200` |
| **Troubleshooting** | |
| Clear Quarto cache | `rm -rf _quarto .quarto` |
| Clear Python cache | `find . -type d -name __pycache__ -exec rm -rf {} +` |
| Reset Jupyter kernel | Kernel → Restart & Clear Output |

## Development Workflow

1. **Data Processing**: Run `python scripts/generate_processed_data.py` once
2. **Data Analysis**: Use Jupyter notebooks for exploration
3. **Create Visualizations**: Add functions to `src/visualization/charts.py`
4. **Write Reports**: Author Quarto markdown files (`.qmd`)
5. **Preview Site**: Run `quarto preview --port 4200` (auto-reloads on changes)
6. **Render Site**: Run `quarto render` to build static website
7. **Deploy**: Push to GitHub for automatic Pages deployment

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

- Check project documentation in `docs/`
- Review error logs for specific issues
- Ensure all file paths are correct
- Verify data format matches expectations