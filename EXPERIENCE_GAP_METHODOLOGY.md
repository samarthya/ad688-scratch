# Experience Gap Analysis Methodology

## How We Calculate the Experience Disparity Percentage

### Overview

The experience gap analysis uses salary-based quartiles as a proxy for experience levels, since the Lightcast dataset does not contain explicit experience level information.

### Methodology

#### Step 1: Data Preparation

- **Source**: Lightcast job postings dataset (72,498 records)
- **Salary Field**: Uses `SALARY_TO` column (maximum salary offered)
- **Data Cleaning**: Removes records with missing salary information

#### Step 2: Experience Level Proxy Creation

```python
# Create quartile-based experience levels
df['Experience_Level'] = pd.qcut(df['SALARY_TO'], 4, labels=[
    'Entry-Level',    # Bottom 25% of salaries
    'Mid-Level',      # 25-50% of salaries  
    'Senior',         # 50-75% of salaries
    'Executive'       # Top 25% of salaries
])
```

#### Step 3: Statistical Analysis

- Calculate mean salary for each quartile group
- Identify highest and lowest quartile means
- Apply percentage gap formula

#### Step 4: Gap Calculation Formula

```python
experience_gap = ((max_salary - min_salary) / min_salary) * 100
```

### Results Interpretation

#### Current Results (September 2025)

- **Entry-Level** (Q1): Lowest quartile average salary
- **Executive** (Q4): Highest quartile average salary
- **Calculated Gap**: ~233% disparity between quartiles

#### What This Means

- The top salary quartile earns **233% more** than the bottom quartile
- This represents a **3.33x** salary multiplier from entry to executive levels
- Based on salary distribution rather than actual years of experience

### Important Limitations

#### 1. **Proxy Method**

- Uses salary quartiles as experience proxy
- Not actual experience years or job titles
- Assumes salary correlates with experience level

#### 2. **Data Distribution**

- Quartile method ensures equal sample sizes per "level"
- May not reflect real-world experience distribution
- Sensitive to salary outliers

#### 3. **Industry Variation**

- Single percentage across all tech roles
- Doesn't account for role-specific experience curves
- Geographic and company size variations not isolated

### Validation Approach

#### Cross-Reference Checks

1. **Job Title Analysis**: Compare results with title-based experience inference
2. **Industry Benchmarks**: Validate against published salary surveys
3. **Temporal Consistency**: Track percentage changes over time

#### Alternative Methodologies

- **Years of Experience**: Direct experience data (when available)
- **Job Title Parsing**: Extract seniority from role titles
- **Skills-Based**: Use skill complexity as experience proxy

### Usage in Research

#### Appropriate Applications

- Highlighting overall compensation disparities
- Demonstrating salary distribution inequality  
- Motivating deeper experience-based analysis

#### Reporting Guidelines

- Always specify "salary-quartile based" methodology
- Include confidence intervals and sample sizes
- Reference limitations in interpretation
- Direct readers to detailed methodology documentation

### Technical Implementation

#### Code Location

- **Primary Function**: `create_key_findings_graphics()` in `create_key_findings.py`
- **Visualization**: `figures/key_finding_experience_gap.html`
- **Documentation**: This methodology file

#### Reproducibility

```bash
# Generate current analysis
python create_key_findings.py

# View detailed breakdown
jupyter notebook notebooks/job_market_analysis_simple.ipynb
```

### Future Enhancements

#### Data Integration

- Incorporate actual experience data when available
- Add job title parsing for seniority detection
- Include skill progression analysis

#### Methodological Improvements  

- Multi-dimensional clustering (salary + skills + titles)
- Industry-specific experience curves
- Geographic and temporal adjustments

---

**Note**: This methodology document explains the technical approach used to generate the experience gap percentages displayed on the website. For current results and interactive visualizations, see the [Salary Analysis](salary-analysis.qmd) section of the research website.
