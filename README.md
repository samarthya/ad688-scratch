# Technology Job Market: Salary Disparity Analysis

A comprehensive data science investigation into **salary disparities** within the technology industry, focusing on compensation inequities across experience levels, education, company sizes, and geographic regions.

## 🚨 Research Objective

This project investigates **systematic salary disparities** in the technology job market to promote equitable compensation practices and inform career planning decisions. We analyze compensation gaps across multiple demographic and professional dimensions.

## 🎯 Primary Research Focus: Salary Disparities

### Key Disparity Investigations
- **Experience Level Gaps**: 197% salary disparity between entry-level and leadership positions
- **Education Premium Inequities**: 64.9% compensation variation across degree levels  
- **Company Size Effects**: 24.3% systematic salary gaps between organization sizes
- **Geographic Inequities**: Regional salary variations beyond cost-of-living adjustments
- **Demographic Factors**: Cross-sectional analysis of wage gap contributors

### Research Questions
- What drives the massive salary progression gaps in technology?
- Do education premiums reflect actual value contribution?
- How do company sizes systematically affect compensation?
- Which geographic regions show the largest pay disparities?
- What interventions can address these systematic inequities?

## 🛠️ Technology Stack

- **Data Processing**: PySpark 4.0.1, Pandas, NumPy for large-scale analysis
- **Visualization**: Plotly, Kaleido 1.1.0 for interactive web visualizations  
- **Documentation**: Quarto website with embedded interactive dashboards
- **Statistical Analysis**: Python scipy, statistical modeling
- **Development**: Jupyter Notebooks, VS Code, Git version control

## 📁 Project Structure (Cleaned & Focused)

```
├── notebooks/
│   └── job_market_skill_analysis.ipynb  # Main disparity analysis
├── src/
│   └── visualization/                    # Reusable plotting utilities
├── figures/                              # Generated interactive visualizations
├── _output/                              # Quarto website output
├── data/
│   ├── processed/                        # Clean datasets and exports
│   └── raw/                              # Original data sources
├── index.qmd                             # Main research homepage
├── salary-analysis.qmd                   # Detailed disparity findings
├── regional-trends.qmd                   # Geographic analysis
├── remote-work.qmd                       # Work arrangement impacts
└── _quarto.yml                           # Website configuration
```

### 🔧 Key Components
- **Interactive Dashboards**: Plotly visualizations with web embedding
- **Salary Disparity Analysis**: Core research in Jupyter notebook
- **Visualization Utilities**: Reusable components from `src/visualization/`
- **Clean Data Pipeline**: PySpark processing with structured outputs

## 🚀 Getting Started

### Prerequisites
- **Python 3.9+** with PySpark support
- **Quarto CLI** for website generation  
- **Git** for version control

### 🔧 Quick Setup
1. **Clone and Setup**:
```bash
git clone https://github.com/samarthya/ad688-scratch
cd project-from-scratch
pip install -r requirements.txt
```

2. **Run the Analysis**:
```bash
# Start Jupyter and open the main notebook
jupyter notebook notebooks/job_market_skill_analysis.ipynb

# Or view the website
quarto preview --port 4200
```

3. **Access Interactive Dashboards**:
   - **Main Analysis**: `http://localhost:4200`
   - **Salary Disparity Dashboard**: `figures/salary_disparity_analysis.html`
   - **Executive Summary**: `figures/executive_dashboard.html`

### 📊 Key Findings Available
- **197% Experience Gap**: Entry vs Leadership salary disparity
- **Interactive Visualizations**: Embedded Plotly dashboards
- **Statistical Analysis**: Quantified inequality measurements
- **Actionable Insights**: Recommendations for organizations and job seekers

## 📈 Research Deliverables

1. **🎯 Salary Disparity Report**: Comprehensive inequity investigation
2. **📊 Interactive Dashboards**: Web-embedded Plotly visualizations  
3. **📝 Research Website**: Quarto-based presentation with findings
4. **⚙️ Reproducible Analysis**: Complete Jupyter notebook workflow

## Contributing

1. Create a feature branch for your analysis area
2. Follow the established coding style and documentation standards
3. Include proper citations for data sources and methodologies
4. Test all visualizations and ensure reproducibility

## Citation

If you use this analysis in your research, please cite:

```bibtex
@misc{salarydisparity2025,
  title={Technology Job Market: Salary Disparity Analysis},
  author={Data Analytics Team},
  year={2025},
  url={https://github.com/samarthya/ad688-scratch},
  note={Investigating compensation inequities in the technology industry}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Lightcast for providing the job postings dataset
- Quarto development team for the publishing platform
- Open source data science community for tools and methodologies