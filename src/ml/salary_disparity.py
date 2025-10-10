"""
Main Salary Disparity Analyzer

This module provides the main interface for conducting comprehensive
salary disparity analysis using machine learning models.
"""

from typing import Dict, List, Optional, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, mean, stddev, count
import numpy as np

from .feature_engineering import SalaryDisparityFeatureEngineer
from .clustering import JobMarketClusterer
from .regression import SalaryRegressionModel
from .classification import JobClassificationModel
from .evaluation import ModelEvaluator


class SalaryDisparityAnalyzer:
    """
    Main analyzer for comprehensive salary disparity analysis.

    Integrates all ML components to provide:
    - KMeans clustering for job market segmentation
    - Multiple regression models for salary prediction
    - Classification models for job categorization
    - Comprehensive evaluation and reporting
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

        # Initialize components
        self.feature_engineer = SalaryDisparityFeatureEngineer(spark)
        self.clusterer = JobMarketClusterer(spark)
        self.regression_model = SalaryRegressionModel(spark)
        self.classification_model = JobClassificationModel(spark)
        self.evaluator = ModelEvaluator(spark)

        # Analysis results
        self.analysis_results = {}
        self.feature_columns = []

    def prepare_analysis_data(self, df: DataFrame) -> DataFrame:
        """Prepare data for comprehensive salary disparity analysis."""

        print("Preparing data for salary disparity analysis...")

        # Apply feature engineering
        df_with_features = self.feature_engineer.prepare_ml_features(df)

        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns()

        print(f"Created {len(self.feature_columns)} features for analysis")

        return df_with_features

    def run_clustering_analysis(self, df: DataFrame, k: int = 5) -> Dict[str, Any]:
        """Run KMeans clustering analysis for job market segmentation."""

        print(f"Running KMeans clustering analysis with k={k}...")

        # Update clusterer with new k
        self.clusterer.k = k

        # Fit clustering model
        self.clusterer.fit_clustering_model(df)

        # Analyze cluster characteristics
        cluster_stats = self.clusterer.analyze_cluster_characteristics(df)

        # Identify salary disparity patterns
        disparity_patterns = self.clusterer.identify_salary_disparity_patterns(df)

        # Get cluster summary
        cluster_summary = self.clusterer.get_cluster_summary()

        results = {
            'cluster_stats': cluster_stats,
            'disparity_patterns': disparity_patterns,
            'cluster_summary': cluster_summary,
            'model': self.clusterer.kmeans_model
        }

        self.analysis_results['clustering'] = results

        return results

    def run_regression_analysis(self, df: DataFrame) -> Dict[str, Any]:
        """Run regression analysis for salary prediction."""

        print("Running regression analysis for salary prediction...")

        # Prepare regression features
        regression_features = [
            'education_level', 'experience_level', 'ai_skills_score',
            'technical_skills_score', 'soft_skills_score', 'location_cost_index',
            'industry_growth_rate', 'salary_percentile'
        ]

        # Train Multiple Linear Regression
        linear_results = self.regression_model.train_linear_regression(
            df, regression_features
        )

        # Train Random Forest Regression
        rf_results = self.regression_model.train_random_forest_regression(
            df, regression_features
        )

        # Compare models
        model_comparison = self.evaluator.compare_models(
            [linear_results['test_metrics'], rf_results['test_metrics']],
            'regression'
        )

        results = {
            'linear_regression': linear_results,
            'random_forest_regression': rf_results,
            'model_comparison': model_comparison
        }

        self.analysis_results['regression'] = results

        return results

    def run_classification_analysis(self, df: DataFrame) -> Dict[str, Any]:
        """Run classification analysis for job categorization."""

        print("Running classification analysis for job categorization...")

        # Prepare classification features
        classification_features = [
            'education_level', 'experience_level', 'ai_skills_score',
            'technical_skills_score', 'soft_skills_score', 'location_cost_index',
            'industry_growth_rate', 'salary_percentile'
        ]

        # Create above-average salary classification
        df_with_above_avg = self.classification_model.create_above_average_salary_classification(
            df, classification_features
        )

        # Train Logistic Regression for above-average salary
        lr_results = self.classification_model.train_logistic_regression(
            df_with_above_avg, classification_features, 'above_average_salary'
        )

        # Train Random Forest for above-average salary
        rf_results = self.classification_model.train_random_forest_classification(
            df_with_above_avg, classification_features, 'above_average_salary'
        )

        # Compare models
        model_comparison = self.evaluator.compare_models(
            [lr_results['test_metrics'], rf_results['test_metrics']],
            'classification'
        )

        results = {
            'logistic_regression': lr_results,
            'random_forest_classification': rf_results,
            'model_comparison': model_comparison
        }

        self.analysis_results['classification'] = results

        return results

    def run_comprehensive_analysis(self, df: DataFrame, k: int = 5) -> Dict[str, Any]:
        """Run comprehensive salary disparity analysis."""

        print("Starting comprehensive salary disparity analysis...")

        # Prepare data
        df_processed = self.prepare_analysis_data(df)

        # Run clustering analysis
        clustering_results = self.run_clustering_analysis(df_processed, k)

        # Run regression analysis
        regression_results = self.run_regression_analysis(df_processed)

        # Run classification analysis
        classification_results = self.run_classification_analysis(df_processed)

        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report()

        # Store all results
        self.analysis_results['comprehensive'] = {
            'clustering': clustering_results,
            'regression': regression_results,
            'classification': classification_results,
            'report': comprehensive_report
        }

        return self.analysis_results['comprehensive']

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""

        report = "SALARY DISPARITY ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"

        # Clustering analysis summary
        if 'clustering' in self.analysis_results:
            clustering = self.analysis_results['clustering']
            report += "1. JOB MARKET SEGMENTATION (KMeans Clustering)\n"
            report += "-" * 45 + "\n"
            report += clustering['cluster_summary'] + "\n"

            # Salary disparity insights
            disparity = clustering['disparity_patterns']
            report += f"Key Insights:\n"
            report += f" • Highest salary cluster: {disparity['highest_salary_cluster']}\n"
            report += f" • Lowest salary cluster: {disparity['lowest_salary_cluster']}\n"
            report += f" • Salary range ratio: {disparity['salary_range_ratio']:.2f}\n"
            report += f" • Average salary ratio: {disparity['avg_salary_ratio']:.2f}\n\n"

        # Regression analysis summary
        if 'regression' in self.analysis_results:
            regression = self.analysis_results['regression']
            report += "2. SALARY PREDICTION MODELS\n"
            report += "-" * 30 + "\n"

            # Linear regression results
            lr_metrics = regression['linear_regression']['test_metrics']
            report += f"Multiple Linear Regression:\n"
            report += f" • R²: {lr_metrics['r2']:.4f}\n"
            report += f" • RMSE: ${lr_metrics['rmse']:,.0f}\n"
            report += f" • MAPE: {lr_metrics['mape_percent']:.2f}%\n\n"

            # Random forest results
            rf_metrics = regression['random_forest_regression']['test_metrics']
            report += f"Random Forest Regression:\n"
            report += f" • R²: {rf_metrics['r2']:.4f}\n"
            report += f" • RMSE: ${rf_metrics['rmse']:,.0f}\n"
            report += f" • MAPE: {rf_metrics['mape_percent']:.2f}%\n\n"

        # Classification analysis summary
        if 'classification' in self.analysis_results:
            classification = self.analysis_results['classification']
            report += "3. JOB CLASSIFICATION MODELS\n"
            report += "-" * 35 + "\n"

            # Logistic regression results
            lr_metrics = classification['logistic_regression']['test_metrics']
            report += f"Logistic Regression (Above-Average Salary):\n"
            report += f" • Accuracy: {lr_metrics['accuracy']:.4f}\n"
            report += f" • F1 Score: {lr_metrics['f1_score']:.4f}\n"
            report += f" • Precision: {lr_metrics['precision']:.4f}\n\n"

            # Random forest results
            rf_metrics = classification['random_forest_classification']['test_metrics']
            report += f"Random Forest Classification (Above-Average Salary):\n"
            report += f" • Accuracy: {rf_metrics['accuracy']:.4f}\n"
            report += f" • F1 Score: {rf_metrics['f1_score']:.4f}\n"
            report += f" • Precision: {rf_metrics['precision']:.4f}\n\n"

        # Recommendations
        report += "4. RECOMMENDATIONS FOR JOB SEEKERS\n"
        report += "-" * 40 + "\n"
        report += "Based on the analysis:\n"
        report += " • Focus on developing AI and technical skills\n"
        report += " • Consider location impact on salary potential\n"
        report += " • Target industries with higher growth rates\n"
        report += " • Leverage education and experience for salary negotiation\n"
        report += " • Consider remote work opportunities for flexibility\n\n"

        report += "5. KEY INSIGHTS\n"
        report += "-" * 15 + "\n"
        report += " • Salary disparities exist across job market segments\n"
        report += " • Skills and location significantly impact salary potential\n"
        report += " • Machine learning models can predict salary with reasonable accuracy\n"
        report += " • Job classification helps identify high-paying opportunities\n"

        return report

    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get feature importance summary across all models."""

        importance_summary = {}

        # Regression feature importance
        if 'regression' in self.analysis_results:
            regression = self.analysis_results['regression']

            # Linear regression coefficients
            lr_importance = regression['linear_regression']['feature_importance']
            importance_summary['linear_regression'] = lr_importance

            # Random forest importance
            rf_importance = regression['random_forest_regression']['feature_importance']
            importance_summary['random_forest_regression'] = rf_importance

        # Classification feature importance
        if 'classification' in self.analysis_results:
            classification = self.analysis_results['classification']

            # Logistic regression coefficients
            lr_importance = classification['logistic_regression']['feature_importance']
            importance_summary['logistic_regression'] = lr_importance

            # Random forest importance
            rf_importance = classification['random_forest_classification']['feature_importance']
            importance_summary['random_forest_classification'] = rf_importance

        return importance_summary

    def export_results(self, output_path: str) -> None:
        """Export analysis results to files."""

        print(f"Exporting results to {output_path}...")

        # Export comprehensive report
        report_path = f"{output_path}/salary_disparity_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.generate_comprehensive_report())

        # Export feature importance
        importance_path = f"{output_path}/feature_importance.json"
        import json
        with open(importance_path, 'w') as f:
            json.dump(self.get_feature_importance_summary(), f, indent=2)

        print(f"Results exported to {output_path}")

    def get_analysis_summary(self) -> str:
        """Get a summary of the analysis results."""

        if not self.analysis_results:
            return "No analysis results available. Please run analysis first."

        summary = "SALARY DISPARITY ANALYSIS SUMMARY\n"
        summary += "=" * 40 + "\n\n"

        summary += f"Analysis Components Completed:\n"
        for component in self.analysis_results.keys():
            summary += f" {component.replace('_', ' ').title()}\n"

        summary += f"\nTotal Features Used: {len(self.feature_columns)}\n"
        summary += f"Feature Categories: {len(self.feature_engineer.feature_categories)}\n"

        return summary
