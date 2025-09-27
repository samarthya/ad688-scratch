# Career Analytics Project

## Setup Instructions

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv career_analytics_env

# Activate environment (Linux/Mac)
source career_analytics_env/bin/activate

# Activate environment (Windows)
career_analytics_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Quarto Installation

Install Quarto from: https://quarto.org/docs/get-started/

```bash
# Verify installation
quarto --version
```

### 3. Data Setup

Place your Lightcast dataset in:
```
data/raw/lightcast_job_postings.csv
```

Expected columns:
- job_id, title, company, location, salary_min, salary_max
- posted_date, industry, experience_level, remote_allowed
- required_skills, education_required

### 4. Initial Data Processing

```bash
# Process raw data
python src/data/preprocess_data.py

# Verify processed data
ls data/processed/
```

### 5. Generate Reports

```bash
# Render Quarto website
quarto render

# Preview locally
quarto preview
```

### 6. Run Interactive Dashboard

```bash
# Start dashboard server
cd dashboards
python career_dashboard.py
```

Access at: http://127.0.0.1:8050

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

| Task | Command |
|------|---------|
| Process data | `python src/data/preprocess_data.py` |
| Render website | `quarto render` |
| Preview site | `quarto preview` |
| Run dashboard | `python dashboards/career_dashboard.py` |
| Install packages | `pip install -r requirements.txt` |

## Development Workflow

1. **Data Analysis**: Use Jupyter notebooks for exploration
2. **Create Visualizations**: Add functions to `src/visualization/plots.py`
3. **Write Reports**: Author Quarto markdown files (`.qmd`)
4. **Update Dashboard**: Modify `dashboards/career_dashboard.py`
5. **Render Site**: Run `quarto render` to build website
6. **Deploy**: Push to GitHub for automatic Pages deployment

## Troubleshooting

### Common Issues

1. **Missing Data**: Ensure `lightcast_job_postings.csv` is in `data/raw/`
2. **Package Errors**: Verify virtual environment is activated
3. **Quarto Errors**: Check YAML syntax in `.qmd` files
4. **Dashboard Issues**: Confirm all dependencies installed

### Getting Help

- Check project documentation in `docs/`
- Review error logs for specific issues
- Ensure all file paths are correct
- Verify data format matches expectations