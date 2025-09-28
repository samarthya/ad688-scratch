# Technology Job Market: Salary Disparalysis

A comprehensive data tigation into salary disparities within the technology industry using large-scale job market data (72K+ records) from Lightcast.

## Research Objective

This project investigates systematic salary disparities in the technology job market to promote equitable compensation practices and inform career planning decisions.

### Key Research Areas
- Experience level compensation gaps (197% disparity found)
- Education premium inequities across degree levels
- Company size effects on compensation
- Geographic salary variations
- AI-related role compensation patterns

> **Detailed findings and methodology**: See [Analysis Results](salary-analysis.qmd) and [Technical Design](DESIGN.md)

## Architecture Overview

**Core Technologies**: PySpark 4.0.1 (big data processing), Plotly (interactive visualizations), Quarto (research publication)

**Data Pipeline**: Raw Lightcast CSV (131 columns) → Spark processing → Multi-format output (Parquet/CSV)

> **Complete technical specifications**: See [System Architecture](DESIGN.md) and [Class Design](docs/class_architecture.md)

## Project Organization

**Research Deliverables**
- `salary-analysis.qmd` - Core disparity analysis with interactive dashboards
- `regional-trends.qmd` - Geographic compensation analysis
- `remote-work.qmd` - Work arrangement impact studies

**Data & Code**
- `data/raw/lightcast_job_postings.csv` - Source dataset (72K+ records, 131 columns)
- `src/` - Reusable analysis classes and visualization utilities
- `notebooks/` - Exploratory analysis and feature engineering

> **Detailed structure and data pipeline**: See [Technical Design](DESIGN.md)

## Quick Start

```bash
# Setup
git clone https://github.com/samarthya/ad688-scratch
cd project-from-scratch
pip install -r requirements.txt

# View research website
quarto preview --port 4200

# Or explore analysis notebooks  
jupyter notebook notebooks/job_market_skill_analysis.ipynb
```

**Key Outputs**: Interactive dashboards at `figures/salary_disparity_analysis.html` and research findings in `salary-analysis.qmd`

## Documentation Structure

- **[Technical Design](DESIGN.md)** - Complete system architecture, data pipeline, and implementation guide
- **[Class Architecture](docs/class_architecture.md)** - UML diagrams and visual class relationships  
- **[Analysis Results](salary-analysis.qmd)** - Research findings with interactive visualizations

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