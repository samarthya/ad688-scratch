"""
Predictive Analytics Dashboard

This module creates comprehensive dashboards combining:
- Multiple Linear Regression results
- Classification model results
- NLP analysis and skills insights
- Interactive prediction tools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from .salary_models import SalaryAnalyticsModels
from .nlp_analysis import JobMarketNLPAnalyzer


class PredictiveAnalyticsDashboard:
    """
    Comprehensive dashboard for predictive analytics results.

    Combines:
    - Salary prediction models
    - Job classification models
    - NLP skills analysis
    - Interactive prediction tools
    """

    def __init__(self, df: pd.DataFrame = None):
        """Initialize with job market data using abstraction layer."""
        if df is None:
            # Use existing data loading abstraction
            from src.data.auto_processor import load_analysis_data
            self.df = load_analysis_data("dashboard")
        else:
            self.df = df

        # Initialize analytics components using abstraction layer
        self.salary_models = SalaryAnalyticsModels(self.df)
        self.nlp_analyzer = JobMarketNLPAnalyzer(self.df)
        self.analytics_results = None

    def create_executive_summary_dashboard(self) -> go.Figure:
        """
        Create executive summary dashboard with key metrics.

        Returns:
            Plotly figure with executive summary
        """
        # Run analytics if not already done
        if self.analytics_results is None:
            self.analytics_results = self.salary_models.run_complete_analysis()

        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Model Performance Overview",
                "Top Salary Drivers",
                "Above-Average Job Predictors",
                "Salary Distribution by Model Predictions",
                "Feature Importance Comparison",
                "Key Insights Summary"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # 1. Model Performance Overview
        performance_data = {
            'Model': ['Regression (R²)', 'Classification (Accuracy)'],
            'Score': [
                self.analytics_results['regression']['test_r2'],
                self.analytics_results['classification']['test_accuracy']
            ]
        }

        fig.add_trace(
            go.Bar(
                x=performance_data['Model'],
                y=performance_data['Score'],
                name="Model Performance",
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[f"{score:.3f}" for score in performance_data['Score']],
                textposition='auto'
            ),
            row=1, col=1
        )

        # 2. Top Salary Drivers (Regression Features)
        reg_features = self.analytics_results['regression']['feature_importance'].head(5)
        fig.add_trace(
            go.Bar(
                x=reg_features['abs_coefficient'],
                y=reg_features['feature'],
                orientation='h',
                name="Salary Drivers",
                marker_color='lightblue'
            ),
            row=1, col=2
        )

        # 3. Above-Average Job Predictors (Classification Features)
        class_features = self.analytics_results['classification']['feature_importance'].head(5)
        fig.add_trace(
            go.Bar(
                x=class_features['importance'],
                y=class_features['feature'],
                orientation='h',
                name="Job Predictors",
                marker_color='lightgreen'
            ),
            row=1, col=3
        )

        # 4. Salary Distribution by Predictions
        reg_predictions = self.analytics_results['regression']['predictions']
        fig.add_trace(
            go.Histogram(
                x=reg_predictions['y_test'],
                name="Actual Salaries",
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=reg_predictions['y_pred'],
                name="Predicted Salaries",
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )

        # 5. Feature Importance Comparison
        # Combine top features from both models
        combined_features = pd.concat([
            reg_features.head(3).assign(model='Regression'),
            class_features.head(3).assign(model='Classification')
        ])

        fig.add_trace(
            go.Bar(
                x=combined_features['feature'],
                y=combined_features.get('abs_coefficient', combined_features.get('importance', 0)),
                name="Combined Importance",
                marker_color=['blue' if m == 'Regression' else 'green'
                             for m in combined_features['model']]
            ),
            row=2, col=2
        )

        # 6. Key Insights Table
        insights_data = self.analytics_results['insights']
        table_data = [
            ["Regression R²", f"{self.analytics_results['regression']['test_r2']:.3f}"],
            ["Classification Accuracy", f"{self.analytics_results['classification']['test_accuracy']:.3f}"],
            ["Salary Threshold", f"${self.analytics_results['classification']['threshold']:,.0f}"],
            ["Top Salary Driver", reg_features.iloc[0]['feature']],
            ["Top Job Predictor", class_features.iloc[0]['feature']]
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=[[row[0] for row in table_data],
                                  [row[1] for row in table_data]])
            ),
            row=2, col=3
        )

        # Update layout
        fig.update_layout(
            title="Predictive Analytics Executive Summary",
            height=800,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_model_comparison_dashboard(self) -> go.Figure:
        """
        Create detailed model comparison dashboard.

        Returns:
            Plotly figure comparing both models
        """
        if self.analytics_results is None:
            self.analytics_results = self.salary_models.run_complete_analysis()

        # Create comparison dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Regression: Prediction vs Actual",
                "Classification: ROC-like Analysis",
                "Feature Importance: Regression Model",
                "Feature Importance: Classification Model"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )

        # Regression scatter plot
        reg_pred = self.analytics_results['regression']['predictions']
        fig.add_trace(
            go.Scatter(
                x=reg_pred['y_test'],
                y=reg_pred['y_pred'],
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )

        # Perfect prediction line
        min_val = min(reg_pred['y_test'].min(), reg_pred['y_pred'].min())
        max_val = max(reg_pred['y_test'].max(), reg_pred['y_pred'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )

        # Classification probability distribution
        class_pred = self.analytics_results['classification']['predictions']
        fig.add_trace(
            go.Scatter(
                x=class_pred['y_test'],
                y=class_pred['y_pred_proba'],
                mode='markers',
                name='Classification Confidence',
                marker=dict(color='green', opacity=0.6)
            ),
            row=1, col=2
        )

        # Feature importance bars
        reg_features = self.analytics_results['regression']['feature_importance'].head(10)
        fig.add_trace(
            go.Bar(
                x=reg_features['feature'],
                y=reg_features['abs_coefficient'],
                name='Regression Features',
                marker_color='lightblue'
            ),
            row=2, col=1
        )

        class_features = self.analytics_results['classification']['feature_importance'].head(10)
        fig.add_trace(
            go.Bar(
                x=class_features['feature'],
                y=class_features['importance'],
                name='Classification Features',
                marker_color='lightgreen'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="Detailed Model Comparison Dashboard",
            height=800,
            showlegend=True
        )

        # Update axes labels
        fig.update_xaxes(title_text="Actual Salary ($)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Salary ($)", row=1, col=1)
        fig.update_xaxes(title_text="Actual Class (0=Below, 1=Above)", row=1, col=2)
        fig.update_yaxes(title_text="Predicted Probability", row=1, col=2)

        return fig

    def create_skills_insights_dashboard(self) -> go.Figure:
        """
        Create skills and NLP insights dashboard.

        Returns:
            Plotly figure with skills analysis
        """
        # Run NLP analysis
        nlp_results = self.nlp_analyzer.run_complete_nlp_analysis()

        # Create skills dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Top In-Demand Skills",
                "Skills by Salary Premium",
                "Skill Clusters Overview",
                "Skills vs Job Count"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "scatter"}]
            ]
        )

        # Top skills
        top_skills = nlp_results['skills_summary'].head(10)
        fig.add_trace(
            go.Bar(
                x=top_skills['skill'],
                y=top_skills['frequency'],
                name='Skill Frequency',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # Skills by salary premium
        if not nlp_results['correlation_analysis'].empty:
            salary_skills = nlp_results['correlation_analysis'].head(10)
            fig.add_trace(
                go.Bar(
                    x=salary_skills['skill'],
                    y=salary_skills['salary_premium'],
                    name='Salary Premium',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )

        # Skill clusters pie chart
        cluster_sizes = [len(cluster['top_skills']) for cluster in nlp_results['clusters'].values()]
        cluster_names = [cluster['description'] for cluster in nlp_results['clusters'].values()]

        fig.add_trace(
            go.Pie(
                labels=cluster_names,
                values=cluster_sizes,
                name="Skill Clusters"
            ),
            row=2, col=1
        )

        # Skills vs job count scatter
        if not nlp_results['correlation_analysis'].empty:
            fig.add_trace(
                go.Scatter(
                    x=nlp_results['correlation_analysis']['job_count'],
                    y=nlp_results['correlation_analysis']['avg_salary'],
                    mode='markers+text',
                    text=nlp_results['correlation_analysis']['skill'],
                    textposition='top center',
                    name='Skills Analysis',
                    marker=dict(
                        size=nlp_results['correlation_analysis']['frequency']/10,
                        color=nlp_results['correlation_analysis']['salary_premium'],
                        colorscale='viridis',
                        showscale=True
                    )
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title="Skills and NLP Insights Dashboard",
            height=800,
            showlegend=True
        )

        # Update axes
        fig.update_xaxes(title_text="Skills", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Skills", row=1, col=2)
        fig.update_yaxes(title_text="Salary Premium ($)", row=1, col=2)
        fig.update_xaxes(title_text="Job Count", row=2, col=2)
        fig.update_yaxes(title_text="Average Salary ($)", row=2, col=2)

        return fig

    def create_interactive_prediction_tool(self) -> Dict[str, Any]:
        """
        Create interactive prediction tool interface.

        Returns:
            Dictionary with prediction tool components
        """
        if self.analytics_results is None:
            self.analytics_results = self.salary_models.run_complete_analysis()

        # Get available options for predictions
        unique_locations = self.df[self.salary_models.location_col].value_counts().head(20).index.tolist()
        unique_titles = self.df['title'].value_counts().head(20).index.tolist()
        unique_industries = self.df['industry'].value_counts().head(15).index.tolist()

        prediction_interface = {
            'input_options': {
                'locations': unique_locations,
                'job_titles': unique_titles,
                'industries': unique_industries,
                'experience_range': [0, 20],
                'skills_range': [1, 15]
            },
            'models': {
                'regression': self.salary_models.models.get('regression'),
                'classification': self.salary_models.models.get('classification')
            },
            'instructions': {
                'regression': "Select job characteristics to predict salary",
                'classification': "Select job characteristics to predict if above-average paying"
            }
        }

        return prediction_interface

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report.

        Returns:
            Dictionary with complete analysis results
        """
        print("[DATA] GENERATING COMPREHENSIVE ANALYTICS REPORT")
        print("=" * 50)

        # Run all analyses
        if self.analytics_results is None:
            self.analytics_results = self.salary_models.run_complete_analysis()

        nlp_results = self.nlp_analyzer.run_complete_nlp_analysis()

        # Create all dashboards
        executive_dashboard = self.create_executive_summary_dashboard()
        model_comparison = self.create_model_comparison_dashboard()
        skills_dashboard = self.create_skills_insights_dashboard()
        prediction_tool = self.create_interactive_prediction_tool()

        # Compile comprehensive report
        comprehensive_report = {
            'executive_summary': {
                'total_jobs_analyzed': len(self.df),
                'regression_r2': self.analytics_results['regression']['test_r2'],
                'classification_accuracy': self.analytics_results['classification']['test_accuracy'],
                'salary_threshold': self.analytics_results['classification']['threshold'],
                'top_salary_driver': self.analytics_results['regression']['feature_importance'].iloc[0]['feature'],
                'top_job_predictor': self.analytics_results['classification']['feature_importance'].iloc[0]['feature']
            },
            'model_results': {
                'regression': self.analytics_results['regression'],
                'classification': self.analytics_results['classification']
            },
            'nlp_analysis': nlp_results,
            'dashboards': {
                'executive_summary': executive_dashboard,
                'model_comparison': model_comparison,
                'skills_insights': skills_dashboard
            },
            'prediction_tool': prediction_tool,
            'insights': self.analytics_results['insights'],
            'recommendations': self._generate_strategic_recommendations()
        }

        print("[OK] Comprehensive report generated successfully!")
        print(f"   [CHART] Regression Model R²: {self.analytics_results['regression']['test_r2']:.3f}")
        print(f"   [TARGET] Classification Accuracy: {self.analytics_results['classification']['test_accuracy']:.3f}")
        print(f"   [CHECK] Skills Analyzed: {nlp_results['insights']['total_unique_skills']:,}")
        print(f"   [DATA] Dashboards Created: {len(comprehensive_report['dashboards'])}")

        return comprehensive_report

    def _generate_strategic_recommendations(self) -> Dict[str, List[str]]:
        """Generate strategic recommendations based on analysis."""
        recommendations = {
            'for_job_seekers': [
                "Focus on skills identified as high salary drivers in the regression model",
                "Target locations and industries flagged as above-average by the classification model",
                "Develop expertise in the top skill clusters identified in NLP analysis",
                "Consider geographic mobility to access higher-paying markets",
                "Build a portfolio that includes the most in-demand technical skills"
            ],
            'for_employers': [
                "Benchmark salaries against model predictions to ensure competitive offers",
                "Focus recruitment in locations identified as talent-rich markets",
                "Invest in training programs for high-value skills identified in analysis",
                "Use classification model insights to optimize job posting strategies",
                "Consider skill premium data when setting compensation bands"
            ],
            'for_policy_makers': [
                "Address geographic salary disparities identified in location analysis",
                "Invest in education programs for high-demand skills clusters",
                "Consider policies to support talent mobility between regions",
                "Use insights to inform workforce development initiatives",
                "Monitor trends in skill premiums for economic planning"
            ]
        }

        return recommendations
