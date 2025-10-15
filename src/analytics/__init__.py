"""
Analytics Module for Advanced Job Market Analysis

This module provides advanced analytics capabilities including:
- Machine learning models for salary prediction and job classification
- NLP analysis for skills extraction and clustering
- Predictive dashboards and interactive tools
- Comprehensive reporting and insights

Classes:
    SalaryAnalyticsModels: Two main analytics models (regression + classification)
    JobMarketNLPAnalyzer: NLP analysis for skills and requirements
    PredictiveAnalyticsDashboard: Interactive dashboards and visualizations

Functions:
    create_analytics_report: Generate comprehensive analytics report
    run_predictive_analysis: Execute complete predictive analysis pipeline
"""

from .salary_models import SalaryAnalyticsModels
from .nlp_analysis import JobMarketNLPAnalyzer
from .predictive_dashboard import PredictiveAnalyticsDashboard
from .docx_report_generator import JobMarketReportGenerator, generate_comprehensive_docx_report
from .skills_analysis import SkillsAnalyzer, run_skills_analysis

# Convenience functions following existing patterns
def create_analytics_report(df=None):
    """
    Create comprehensive analytics report using abstraction layer.

    Args:
        df: Optional DataFrame. If None, uses auto data loading.

    Returns:
        Dict with complete analytics results
    """
    dashboard = PredictiveAnalyticsDashboard(df)
    return dashboard.generate_comprehensive_report()

def run_predictive_analysis(df=None):
    """
    Run complete predictive analysis pipeline.

    Args:
        df: Optional DataFrame. If None, uses auto data loading.

    Returns:
        Dict with model results and insights
    """
    models = SalaryAnalyticsModels(df)
    return models.run_complete_analysis()

def run_nlp_analysis(df=None):
    """
    Run complete NLP analysis pipeline.

    Args:
        df: Optional DataFrame. If None, uses auto data loading.

    Returns:
        Dict with NLP results and insights
    """
    nlp_analyzer = JobMarketNLPAnalyzer(df)
    return nlp_analyzer.run_complete_nlp_analysis()

def generate_docx_report(df=None, output_path="job_market_analytics_report.docx"):
    """
    Generate comprehensive DOCX report using abstraction layer.

    Args:
        df: Optional DataFrame. If None, uses auto data loading.
        output_path: Path where to save the DOCX report

    Returns:
        Path to generated report
    """
    return generate_comprehensive_docx_report(df, output_path)

def run_skills_analysis_complete(df=None):
    """
    Run complete skills analysis pipeline.

    Args:
        df: Optional DataFrame. If None, uses auto data loading.

    Returns:
        Dict with skills analysis results and insights
    """
    return run_skills_analysis(df)

__all__ = [
    "SalaryAnalyticsModels",
    "JobMarketNLPAnalyzer",
    "PredictiveAnalyticsDashboard",
    "JobMarketReportGenerator",
    "SkillsAnalyzer",
    "create_analytics_report",
    "run_predictive_analysis",
    "run_nlp_analysis",
    "run_skills_analysis_complete",
    "generate_docx_report",
    "generate_comprehensive_docx_report"
]
