# Career Analytics Project

## Setup Instructions

### 1. Prerequisites

#### Java Installation (Required for PySpark)

**Linux/Ubuntu:**

```bash
# Install OpenJDK 11 or 17 (recommended for PySpark 4.0.1)
sudo apt update
sudo apt install openjdk-11-jdk

# Verify installation
java -version
```

**macOS:**
```bash
# Using Homebrew
brew install openjdk@11

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"
```

**Windows:**
- Download and install OpenJDK from [Adoptium](https://adoptium.net/)
- Add Java to system PATH

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

Expected columns:
- job_id, title, company, location, salary_min, salary_max
- posted_date, industry, experience_level, remote_allowed
- required_skills, education_required

### 6. Initial Data Processing

**Option 1: Using PySpark (Recommended for large datasets)**
```bash
# Use the developer validation notebook
jupyter lab notebooks/job_market_analysis_simple.ipynb
```

**Option 2: Direct processing script**
```bash
# Process raw data
python src/data/preprocess_data.py

# Verify processed data
ls data/processed/
```

**For PySpark Processing:**
- The project uses PySpark 4.0.1 for scalable data processing
- Start with `notebooks/job_market_analysis_simple.ipynb` for data validation
- The notebook includes automatic Spark configuration for local environments

### 7. Generate Reports

```bash
# Render Quarto website
quarto render

# Preview locally
quarto preview
```

### 8. Run Interactive Dashboard

```bash
# Start dashboard server
cd dashboards
python career_dashboard.py
```

Access at: http://127.0.0.1:4200

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
| Validate raw data (PySpark) | `jupyter lab notebooks/job_market_analysis_simple.ipynb` |
| Process data (traditional) | `python src/data/preprocess_data.py` |
| Generate key findings charts | `python create_key_findings.py` |
| **Development** | |
| Install packages | `pip install -r requirements.txt` |
| Test PySpark | `python -c "import pyspark; print(pyspark.__version__)"` |
| **Publishing** | |
| Render website | `quarto render` |
| Preview site | `quarto preview` |
| Run dashboard | `python dashboards/career_dashboard.py` |
| **Troubleshooting** | |
| Clear Quarto cache | `quarto clear` |
| Reset Jupyter kernel | Kernel → Restart & Clear Output |

## Development Workflow

1. **Data Analysis**: Use Jupyter notebooks for exploration
2. **Create Visualizations**: Add functions to `src/visualization/plots.py`
3. **Write Reports**: Author Quarto markdown files (`.qmd`)
4. **Update Dashboard**: Modify `dashboards/career_dashboard.py`
5. **Render Site**: Run `quarto render` to build website
6. **Deploy**: Push to GitHub for automatic Pages deployment

## Troubleshooting

### PySpark Issues

#### "Connection refused" Error
This is the most common PySpark issue in virtual environments:

**Symptoms:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solutions:**
1. **Use the provided notebook configuration:**
   - Open `notebooks/job_market_analysis_simple.ipynb`
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