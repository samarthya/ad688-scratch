# Job Market Trends and Salary Analysis

A comprehensive data science project analyzing job market trends, salary compensation, and career planning insights using Python, PySpark, and Pandas.

## Project Overview

This project explores how job seekers can position themselves effectively given changes in hiring trends, salaries, AI adoption, remote work, and gender-based employment patterns. The analysis is designed to be both practical and exploratory, helping students apply data science, analytics, and visualization skills to their own career planning.

## Research Questions

### Primary Focus: Salary and Compensation Trends
- How do salaries differ across AI vs. non-AI careers?
- What regions offer the highest-paying jobs in AI-related and traditional careers?
- Are remote jobs better paying than in-office roles?
- What industries saw the biggest wage growth in 2025?

### Additional Areas of Investigation
- Gender-based employment patterns and salary disparities
- Skills gap analysis and demand forecasting
- Career progression pathways in emerging vs. traditional fields
- Geographic mobility and salary optimization strategies

## Technology Stack

- **Documentation & Reporting**: Quarto Markdown, GitHub Pages
- **Data Processing**: Python, PySpark, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly, Dash
- **Data Source**: Lightcast job postings dataset
- **Bibliography Management**: BibTeX with Econometrica citation style
- **Development Environment**: VS Code, Jupyter Notebooks

## Project Structure

```
├── data/
│   ├── raw/                 # Original datasets (Lightcast job postings)
│   ├── processed/           # Cleaned and transformed data
│   └── external/            # Additional reference datasets
├── src/
│   ├── data/               # Data processing and cleaning scripts
│   ├── analysis/           # Statistical analysis and modeling
│   └── visualization/      # Chart and dashboard creation
├── notebooks/              # Jupyter notebooks for exploratory analysis
├── reports/               # Generated analysis reports
├── dashboards/            # Interactive dashboard applications
├── docs/                  # Generated website output
├── csl/                   # Citation style files
└── references.bib         # Bibliography database
```

## Getting Started

### Prerequisites

- Python 3.8+
- Quarto CLI
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/samarthya/ad688-scratch
cd project-from-scratch
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Quarto extensions:
```bash
quarto install extension quarto-ext/lightbox
```

### Data Setup

1. Place the Lightcast dataset in `data/raw/lightcast_job_postings.csv`
2. Run initial data processing:
```bash
python src/data/preprocess_data.py
```

### Building the Website

```bash
quarto render
quarto preview
```

## Deliverables

1. **Team-Based Career Report**: Structured research report analyzing job market trends
2. **Interactive Data Dashboards**: Python-based visualizations integrated with GitHub Pages
3. **Personal Career Strategy Plans**: Individual 3-step career action plans
4. **Final Presentation Website**: Complete Quarto-based research portfolio

## Contributing

1. Create a feature branch for your analysis area
2. Follow the established coding style and documentation standards
3. Include proper citations for data sources and methodologies
4. Test all visualizations and ensure reproducibility

## Citation

If you use this analysis in your research, please cite:

```bibtex
@misc{jobmarket2025,
  title={Job Market Trends and Salary Analysis: Career Planning in the AI Era},
  author={"Saurabh Sharma"},
  year={2025},
  url={https://github.io/samarthya/ad688scratch/}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Lightcast for providing the job postings dataset
- Quarto development team for the publishing platform
- Open source data science community for tools and methodologies