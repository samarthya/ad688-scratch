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

1. **Visit the Analysis Website**: Open `_output/index.html` in your browser
2. **Start with Homepage**: Get overview of key insights and navigation
3. **Explore Core Analysis**: Use sidebar navigation for specific questions
4. **Interactive Features**: Click charts and dashboards for detailed exploration

### **Code setup**

```bash
# 1. Clone and setup environment
git clone https://github.com/samarthya/ad688-scratch
cd project-from-scratch
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Generate website
quarto render

# 4. View website
quarto preview --no-browser
```

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
    ├── DESIGN.md                 # Technical architecture and implementation  
    └── SETUP.md                  # Environment configuration and setup
```

---

## **How to Use This Analysis**

### **For Students Planning Careers:**

1. **Set Realistic Expectations**: Use salary benchmarks for post-graduation planning
2. **Calculate Education ROI**: Evaluate Master's/PhD investment returns  
3. **Plan Geographic Strategy**: Consider location for internships and first jobs
4. **Understand Modern Workplace**: Factor remote work into job search strategy

### **For Job Seekers:**
1. **Benchmark Current Compensation**: Compare against market data for your level
2. **Identify Growth Opportunities**: Target high-value skills and career progression
3. **Optimize Job Search**: Use geographic and remote work intelligence
4. **Negotiate with Data**: Use specific market benchmarks in salary discussions

### **For Career Changers:**

1. **Assess Transition Impact**: Understand salary implications of industry/role changes
2. **Plan Skill Development**: Focus on high-ROI capabilities based on market analysis
3. **Evaluate Education Options**: Compare formal degrees vs certifications vs experience
4. **Strategic Location Planning**: Consider geographic moves for career optimization

---

## **Key Findings Summary**

### **Experience Progression Analysis**

- **Entry Level (0-2 years)**: $65,000 average starting salary
- **Mid Level (3-7 years)**: $85,000 average (+31% growth)  
- **Senior Level (8-15 years)**: $120,000 average (+41% growth)
- **Executive (15+ years)**: $150,000+ average (+25% leadership premium)

### **Education Return on Investment** 

- **Bachelor's Degree**: Market baseline (100%)
- **Master's Degree**: +25% average salary premium
- **PhD/Advanced**: +30% average salary premium
- **ROI Timeline**: Most advanced degrees pay for themselves within 3-5 years

### **Geographic Intelligence**

- **Highest-Paying Markets**: San Francisco, Seattle, New York (cost-adjusted)
- **Best ROI Markets**: Austin, Denver, Atlanta (salary vs cost-of-living)
- **Remote Work Impact**: 75% of positions offer location flexibility
- **Geographic Premium**: Up to 67% salary difference between markets

### **Modern Workplace Trends**

- **Remote Available**: 45% of positions offer full remote work
- **Hybrid Options**: 30% provide flexible arrangements
- **On-Site Only**: 25% require physical presence  
- **Salary Impact**: Remote work doesn't reduce compensation in most cases

---

## **Technical Details**

### **Data Sources**

- **Primary**: Lightcast job postings database (72,000+ records)
- **Supplementary**: Bureau of Labor Statistics, geographic cost-of-living data
- **Processing**: PySpark for large-scale data analysis and transformation
- **Validation**: Cross-reference with industry salary surveys

### **Analysis Methods**

- **Statistical Modeling**: Python with pandas, numpy, scipy
- **Machine Learning**: Scikit-learn for prediction and classification models
- **Visualization**: Plotly for interactive charts, matplotlib for static analysis
- **Web Framework**: Quarto for integrated analysis and presentation

### **Quality Assurance**

- **Data Validation**: Outlier detection and cleaning procedures
- **Statistical Testing**: Confidence intervals and significance testing
- **Model Validation**: Cross-validation and performance metrics
- **Reproducibility**: Version-controlled pipeline with documented methodology

---

## **Contributing & Usage**

### **For Academic Use**

This analysis is designed for educational purposes. Please cite appropriately if used in academic work.

### **For Professional Application** 

The insights are based on real market data and suitable for career planning and salary benchmarking.

### **Technical Contributions**

- Enhance analysis methodology in `src/` modules
- Improve visualizations in `notebooks/` 
- Extend geographic or industry coverage
- Optimize data processing pipeline

---

## **Contact & Support**

**Author**: Saurabh Sharma (Boston University)
**Repository**: [GitHub - ad688-scratch](https://github.com/samarthya/ad688-scratch)  
**Technical Documentation**: See `DESIGN.md` for detailed implementation guide
**Analysis Methodology**: See `salary-analysis.qmd` for detailed methodology and interpretation

---

*Make informed career decisions with real market data, not guesswork.*