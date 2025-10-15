#!/usr/bin/env python3
"""
Generate preview charts for validation without running full Quarto reports.

This script creates all major visualizations and saves them to the figures/ directory
for quick review and validation.

Usage:
    python scripts/generate_charts_preview.py

    # Or with specific chart types:
    python scripts/generate_charts_preview.py --charts ml presentation skills
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.website_processor import get_processed_dataframe
from src.visualization.charts import display_figure
from src.visualization.ml_charts import (
    create_confusion_matrix_heatmap_docx_optimized,
    create_ml_performance_comparison,
    create_predicted_vs_actual_plot
)
from src.visualization.presentation_charts import PresentationCharts
from src.analytics.skills_analysis import run_skills_analysis
from src.visualization.skills_charts import create_skills_visualization, SkillsVisualizer
from src.analytics.ai_ml_location_analysis import analyze_ai_ml_jobs_by_location
from src.visualization.ai_ml_charts import (
    create_ai_ml_jobs_by_location_chart,
    create_ai_ml_percentage_chart,
    create_ai_ml_combined_dashboard
)

print("=" * 80)
print("CHART GENERATION & VALIDATION TOOL")
print("=" * 80)
print()


def generate_ml_charts(df: pd.DataFrame) -> None:
    """Generate machine learning visualization charts."""
    print("📊 Generating ML Charts...")
    print("-" * 80)

    # 1. Confusion Matrix
    print("  → Confusion Matrix...")
    confusion_data = np.array([
        [43, 7],   # Below-average: 43% correct, 7% incorrect
        [8, 42]    # Above-average: 8% incorrect, 42% correct
    ])
    class_labels = ['Below Average', 'Above Average']

    fig_conf = create_confusion_matrix_heatmap_docx_optimized(
        confusion_data,
        class_labels,
        title='Classification Model: Confusion Matrix (Above/Below Median Salary)',
        colorscale_name='heat_style'
    )
    fig_conf.write_image('figures/preview_ml_confusion_matrix.png', width=900, height=500, scale=2)
    print("     ✓ figures/preview_ml_confusion_matrix.png")

    # 2. Model Performance Comparison
    print("  → Model Performance Comparison...")
    regression_results = {'train_r2': 0.84, 'test_r2': 0.83}
    classification_results = {'train_accuracy': 0.86, 'test_accuracy': 0.85}

    fig_perf = create_ml_performance_comparison(regression_results, classification_results)
    fig_perf.write_image('figures/preview_ml_model_performance.png', width=1000, height=500, scale=2)
    print("     ✓ figures/preview_ml_model_performance.png")

    # 3. Predicted vs Actual (using sample data)
    print("  → Predicted vs Actual Scatter...")
    from src.visualization.ml_charts import generate_representative_predictions
    actual, predicted = generate_representative_predictions(
        median_salary=114000,
        salary_std=35000,
        rmse=19800,
        n_samples=500
    )

    fig_scatter = create_predicted_vs_actual_plot(
        actual,
        predicted,
        title='Salary Predictions: Actual vs Predicted'
    )
    fig_scatter.write_image('figures/preview_ml_predicted_vs_actual.png', width=900, height=600, scale=2)
    print("     ✓ figures/preview_ml_predicted_vs_actual.png")

    print()


def generate_presentation_charts(df: pd.DataFrame) -> None:
    """Generate presentation visualization charts."""
    print("📊 Generating Presentation Charts...")
    print("-" * 80)

    # Get summary data
    from src.data.website_processor import get_website_data_summary
    summary = get_website_data_summary()

    pcharts = PresentationCharts(df, summary)

    # 1. KPI Overview
    print("  → KPI Overview...")
    fig_kpi = pcharts.create_kpi_overview()
    fig_kpi.write_image('figures/preview_presentation_kpi.png', width=1000, height=600, scale=2)
    print("     ✓ figures/preview_presentation_kpi.png")

    # 2. Experience Progression
    print("  → Experience Progression...")
    fig_exp = pcharts.create_experience_progression()
    fig_exp.write_image('figures/preview_presentation_experience.png', width=1000, height=600, scale=2)
    print("     ✓ figures/preview_presentation_experience.png")

    # 3. Geographic Analysis
    print("  → Geographic Analysis...")
    fig_geo = pcharts.create_geographic_analysis()
    fig_geo.write_image('figures/preview_presentation_geographic.png', width=1000, height=600, scale=2)
    print("     ✓ figures/preview_presentation_geographic.png")

    # 4. Industry Analysis
    print("  → Industry Analysis...")
    fig_ind = pcharts.create_industry_analysis()
    fig_ind.write_image('figures/preview_presentation_industry.png', width=1000, height=600, scale=2)
    print("     ✓ figures/preview_presentation_industry.png")

    # 5. Skills Premium
    print("  → Skills Premium Analysis...")
    fig_skills = pcharts.create_skills_premium_analysis()
    fig_skills.write_image('figures/preview_presentation_skills.png', width=1200, height=500, scale=2)
    print("     ✓ figures/preview_presentation_skills.png")

    # 6. Remote Work Analysis
    print("  → Remote Work vs Salary...")
    fig_remote = pcharts.create_remote_work_analysis()
    fig_remote.write_image('figures/preview_presentation_remote.png', width=1000, height=600, scale=2)
    print("     ✓ figures/preview_presentation_remote.png")

    # 7. Action Plan Roadmap
    print("  → Action Plan Roadmap...")
    fig_roadmap = pcharts.create_action_plan_roadmap()
    fig_roadmap.write_image('figures/preview_presentation_roadmap.png', width=1000, height=500, scale=2)
    print("     ✓ figures/preview_presentation_roadmap.png")

    print()


def generate_skills_charts(df: pd.DataFrame) -> None:
    """Generate skills analysis visualization charts."""
    print("📊 Generating Skills Analysis Charts...")
    print("-" * 80)

    # Run skills analysis
    print("  → Running skills analysis...")
    skills_results = run_skills_analysis(df)

    # 1. Geographic Skills Heatmap
    print("  → Geographic Skills Demand...")
    geo_skills_fig = create_skills_visualization(skills_results, "geographic")
    geo_skills_fig.update_layout(
        title="Technical Skills Demand by Geographic Location",
        height=600,
        margin=dict(l=150, r=50, t=100, b=150)
    )
    geo_skills_fig.write_image('figures/preview_skills_geographic.png', width=1200, height=600, scale=2)
    print("     ✓ figures/preview_skills_geographic.png")

    # 2. Skills vs Salary Correlation
    print("  → Skills-Salary Correlation...")
    skills_visualizer = SkillsVisualizer(skills_results)
    skills_salary_fig = skills_visualizer.create_skills_salary_correlation_chart(top_n=5)
    skills_salary_fig.update_layout(
        title="Top 5 Technical Skills vs Salary Correlation",
        height=400
    )
    skills_salary_fig.write_image('figures/preview_skills_salary_correlation.png', width=1000, height=400, scale=2)
    print("     ✓ figures/preview_skills_salary_correlation.png")

    # 3. Emerging Skills
    print("  → Emerging High-Value Skills...")
    emerging_skills_fig = skills_visualizer.create_emerging_skills_chart(top_n=5)
    emerging_skills_fig.update_layout(
        title="Top 5 Emerging High-Value Skills",
        height=400
    )
    emerging_skills_fig.write_image('figures/preview_skills_emerging.png', width=1000, height=400, scale=2)
    print("     ✓ figures/preview_skills_emerging.png")

    print()


def generate_ai_ml_charts(df: pd.DataFrame) -> None:
    """Generate AI/ML jobs by location visualization charts."""
    print("📊 Generating AI/ML Location Analysis Charts...")
    print("-" * 80)

    # Analyze AI/ML jobs by location
    print("  → Analyzing AI/ML jobs by location...")
    city_stats = analyze_ai_ml_jobs_by_location(df, top_n=5)

    # 1. AI/ML Jobs Comparison
    print("  → AI/ML Jobs by Location...")
    fig_jobs = create_ai_ml_jobs_by_location_chart(city_stats)
    fig_jobs.write_image('figures/preview_ai_ml_jobs_by_location.png', width=1000, height=500, scale=2)
    print("     ✓ figures/preview_ai_ml_jobs_by_location.png")

    # 2. AI/ML Percentage
    print("  → AI/ML Job Concentration...")
    fig_pct = create_ai_ml_percentage_chart(city_stats)
    fig_pct.write_image('figures/preview_ai_ml_percentage.png', width=1000, height=500, scale=2)
    print("     ✓ figures/preview_ai_ml_percentage.png")

    # 3. Combined Dashboard
    print("  → AI/ML Combined Dashboard...")
    fig_dash = create_ai_ml_combined_dashboard(city_stats)
    fig_dash.write_image('figures/preview_ai_ml_dashboard.png', width=1200, height=800, scale=2)
    print("     ✓ figures/preview_ai_ml_dashboard.png")

    print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate preview charts for validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all charts
  python scripts/generate_charts_preview.py

  # Generate only ML charts
  python scripts/generate_charts_preview.py --charts ml

  # Generate presentation and skills charts
  python scripts/generate_charts_preview.py --charts presentation skills
        """
    )
    parser.add_argument(
        '--charts',
        nargs='+',
        choices=['ml', 'presentation', 'skills', 'ai_ml', 'all'],
        default=['all'],
        help='Which chart categories to generate (default: all)'
    )

    args = parser.parse_args()

    # Determine which charts to generate
    generate_all = 'all' in args.charts
    generate_ml = generate_all or 'ml' in args.charts
    generate_presentation = generate_all or 'presentation' in args.charts
    generate_skills = generate_all or 'skills' in args.charts
    generate_ai_ml = generate_all or 'ai_ml' in args.charts

    print("Loading processed data...")
    print("-" * 80)

    try:
        df = get_processed_dataframe()
        print(f"✓ Loaded {len(df):,} job postings")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nPlease ensure processed data exists:")
        print("  python scripts/generate_processed_data.py")
        sys.exit(1)

    # Generate requested charts
    try:
        if generate_ml:
            generate_ml_charts(df)

        if generate_presentation:
            generate_presentation_charts(df)

        if generate_skills:
            generate_skills_charts(df)

        if generate_ai_ml:
            generate_ai_ml_charts(df)

        print("=" * 80)
        print("✅ CHART GENERATION COMPLETE")
        print("=" * 80)
        print()
        print("Preview files saved to figures/ directory with 'preview_' prefix")
        print("Review charts and compare with report outputs for validation.")
        print()

    except Exception as e:
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

