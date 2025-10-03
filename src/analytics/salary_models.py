"""
Two Main Analytics Models for Salary and Compensation Trends

This module implements the core analytics models:
1. Multiple Linear Regression for salary prediction
2. Classification for above-average paying jobs

Each model includes plain-language summaries, feature descriptions,
and implications for job seekers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class SalaryAnalyticsModels:
    """
    Two main analytics models for salary and compensation analysis.

    Model 1: Multiple Linear Regression for salary prediction
    Model 2: Classification for above-average paying jobs
    """

    def __init__(self, df: pd.DataFrame = None):
        """Initialize with processed job market data using abstraction layer."""
        if df is None:
            # Use existing data loading abstraction
            from src.data.auto_processor import load_analysis_data
            self.df = load_analysis_data("analytics")
        else:
            self.df = df.copy()

        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_results = {}

        # Use existing column mapping abstraction
        from src.config.column_mapping import get_analysis_column
        self.salary_col = get_analysis_column('salary')  # 'salary_avg_imputed'
        self.location_col = get_analysis_column('city')  # 'city_name'

        print(f"Initialized with {len(self.df):,} records")
        print(f"Using salary column: {self.salary_col}")
        print(f"Using location column: {self.location_col}")

    def prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for both models.

        Features used:
        - Location (city_name): Geographic salary variations
        - Job Title (title): Role-based compensation differences
        - Industry (industry): Sector-specific pay scales
        - Experience (experience_years): Career progression impact
        - Skills (required_skills): Technical skill premiums
        """
        print("\n=== FEATURE PREPARATION ===")

        # Start with core columns
        feature_df = self.df.copy()

        # Ensure required columns exist
        required_cols = [self.salary_col, self.location_col, 'title', 'industry']
        missing_cols = [col for col in required_cols if col not in feature_df.columns]

        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            # Create placeholder columns
            for col in missing_cols:
                if col == self.salary_col:
                    feature_df[col] = 75000  # Default salary
                else:
                    feature_df[col] = 'Unknown'

        # Clean salary data
        feature_df[self.salary_col] = pd.to_numeric(feature_df[self.salary_col], errors='coerce')
        feature_df = feature_df.dropna(subset=[self.salary_col])
        feature_df = feature_df[feature_df[self.salary_col] > 0]

        # Create experience features
        if 'experience_years' not in feature_df.columns:
            # Create experience from salary percentiles
            salary_percentiles = feature_df[self.salary_col].quantile([0.25, 0.5, 0.75])
            def estimate_experience(salary):
                if salary <= salary_percentiles[0.25]:
                    return 1  # Entry level
                elif salary <= salary_percentiles[0.5]:
                    return 3  # Mid level
                elif salary <= salary_percentiles[0.75]:
                    return 7  # Senior level
                else:
                    return 12  # Executive level

            feature_df['experience_years'] = feature_df[self.salary_col].apply(estimate_experience)

        # Create skills score
        if 'required_skills' in feature_df.columns:
            feature_df['skills_count'] = feature_df['required_skills'].astype(str).str.count(',') + 1
        else:
            feature_df['skills_count'] = 3  # Default skills count

        # Clean categorical variables
        categorical_cols = [self.location_col, 'title', 'industry']
        for col in categorical_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(str).str.strip()
                feature_df[col] = feature_df[col].replace(['nan', 'None', ''], 'Unknown')

        print(f"\n✅ Prepared features for {len(feature_df):,} records")
        print(f"   Salary range: ${feature_df[self.salary_col].min():,.0f} - ${feature_df[self.salary_col].max():,.0f}")
        print(f"   Median salary: ${feature_df[self.salary_col].median():,.0f}")
        print(f"   Unique locations: {feature_df[self.location_col].nunique()}")
        print(f"   Unique titles: {feature_df['title'].nunique()}")
        print(f"   Unique industries: {feature_df['industry'].nunique()}")

        # Check for data quality issues
        if len(feature_df) < 100:
            print(f"\n⚠️ WARNING: Only {len(feature_df)} records available. Models may not perform well.")

        salary_std = feature_df[self.salary_col].std()
        if salary_std < 1000:
            print(f"\n⚠️ WARNING: Very low salary variation (std: ${salary_std:.2f}). Classification may fail.")

        return feature_df

    def model_1_multiple_linear_regression(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        MODEL 1: Multiple Linear Regression for Salary Prediction

        WHAT WE'RE MODELING:
        We're predicting salary based on location, job title, industry, experience, and skills.
        This helps understand which factors most influence compensation.

        FEATURES USED:
        - Location: Geographic cost of living and market demand
        - Job Title: Role complexity and responsibility level
        - Industry: Sector-specific compensation standards
        - Experience Years: Career progression and expertise
        - Skills Count: Technical capability breadth

        WHY THIS MATTERS FOR JOB SEEKERS:
        - Identify high-paying locations to target
        - Understand which skills command salary premiums
        - Quantify the value of experience and specialization
        - Compare compensation across industries and roles
        """
        print("\n=== MODEL 1: MULTIPLE LINEAR REGRESSION ===")
        print("Predicting salary based on location, job title, industry, experience, and skills")

        # Prepare features
        X_features = []
        feature_names = []

        # Encode categorical variables
        categorical_features = [self.location_col, 'title', 'industry']

        for feature in categorical_features:
            if feature in feature_df.columns:
                # Use top categories to avoid too many features
                top_categories = feature_df[feature].value_counts().head(20).index
                for category in top_categories:
                    feature_name = f"{feature}_{category}"
                    feature_df[feature_name] = (feature_df[feature] == category).astype(int)
                    X_features.append(feature_name)
                    feature_names.append(feature_name)

        # Add numerical features
        numerical_features = ['experience_years', 'skills_count']
        for feature in numerical_features:
            if feature in feature_df.columns:
                X_features.append(feature)
                feature_names.append(feature)

        # Prepare data
        X = feature_df[X_features].fillna(0)
        y = feature_df[self.salary_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Feature importance (absolute coefficients)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)

        # Store model
        self.models['regression'] = {
            'model': model,
            'scaler': scaler,
            'features': feature_names
        }

        results = {
            'model_type': 'Multiple Linear Regression',
            'purpose': 'Salary Prediction',
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': feature_importance,
            'predictions': {
                'y_test': y_test,
                'y_pred': y_pred_test
            },
            'sample_size': {
                'train': len(X_train),
                'test': len(X_test)
            }
        }

        self.model_results['regression'] = results

        print(f"✅ Model trained successfully!")
        print(f"   R² Score: {test_r2:.3f} (explains {test_r2*100:.1f}% of salary variance)")
        print(f"   RMSE: ${test_rmse:,.0f} (average prediction error)")
        print(f"   Sample: {len(X_train):,} training, {len(X_test):,} testing")

        return results

    def model_2_above_average_classification(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        MODEL 2: Classification for Above-Average Paying Jobs

        WHAT WE'RE MODELING:
        We're classifying jobs as "above-average" or "below-average" paying based on
        location, title, industry, and skills. This helps identify high-opportunity roles.

        FEATURES USED:
        - Location: High-paying vs. lower-paying markets
        - Job Title: Premium roles vs. standard positions
        - Industry: High-compensation vs. average sectors
        - Experience Level: Senior vs. junior classifications
        - Skills Complexity: Advanced vs. basic skill requirements

        WHY THIS MATTERS FOR JOB SEEKERS:
        - Identify which combinations lead to above-average pay
        - Target high-opportunity locations and industries
        - Understand which skills unlock premium compensation
        - Focus job search on above-average paying role types
        """
        print("\n=== MODEL 2: ABOVE-AVERAGE SALARY CLASSIFICATION ===")
        print("Classifying jobs as above-average or below-average paying")

        # Create target variable (above median salary)
        median_salary = feature_df[self.salary_col].median()
        feature_df['above_average'] = (feature_df[self.salary_col] > median_salary).astype(int)

        print(f"Median salary threshold: ${median_salary:,.0f}")
        print(f"Above-average jobs: {feature_df['above_average'].sum():,} ({feature_df['above_average'].mean()*100:.1f}%)")

        # Prepare features (similar to regression but simplified)
        X_features = []
        feature_names = []

        # Top locations
        top_locations = feature_df[self.location_col].value_counts().head(15).index
        for location in top_locations:
            feature_name = f"location_{location}"
            feature_df[feature_name] = (feature_df[self.location_col] == location).astype(int)
            X_features.append(feature_name)
            feature_names.append(feature_name)

        # Top job titles
        top_titles = feature_df['title'].value_counts().head(15).index
        for title in top_titles:
            feature_name = f"title_{title}"
            feature_df[feature_name] = (feature_df['title'] == title).astype(int)
            X_features.append(feature_name)
            feature_names.append(feature_name)

        # Top industries
        top_industries = feature_df['industry'].value_counts().head(10).index
        for industry in top_industries:
            feature_name = f"industry_{industry}"
            feature_df[feature_name] = (feature_df['industry'] == industry).astype(int)
            X_features.append(feature_name)
            feature_names.append(feature_name)

        # Numerical features
        numerical_features = ['experience_years', 'skills_count']
        for feature in numerical_features:
            if feature in feature_df.columns:
                X_features.append(feature)
                feature_names.append(feature)

        # Prepare data
        X = feature_df[X_features].fillna(0)
        y = feature_df['above_average']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Check if model has both classes
        n_classes = len(model.classes_)
        print(f"Model trained with {n_classes} classes: {model.classes_}")

        if n_classes < 2:
            raise ValueError(f"Classification requires at least 2 classes, but only found {n_classes}. "
                           f"This indicates insufficient salary variation in the data. "
                           f"Median salary: ${median_salary:,.0f}, "
                           f"Salary range: ${feature_df[self.salary_col].min():,.0f} - ${feature_df[self.salary_col].max():,.0f}")

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_accuracy = (y_pred_train == y_train).mean()
        test_accuracy = (y_pred_test == y_test).mean()

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store model
        self.models['classification'] = {
            'model': model,
            'features': feature_names,
            'threshold': median_salary
        }

        results = {
            'model_type': 'Random Forest Classification',
            'purpose': 'Above-Average Salary Prediction',
            'threshold': median_salary,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance,
            'predictions': {
                'y_test': y_test,
                'y_pred': y_pred_test,
                'y_pred_proba': y_pred_proba
            },
            'sample_size': {
                'train': len(X_train),
                'test': len(X_test)
            }
        }

        self.model_results['classification'] = results

        print(f"✅ Model trained successfully!")
        print(f"   Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}% correct predictions)")
        print(f"   Sample: {len(X_train):,} training, {len(X_test):,} testing")

        return results

    def create_model_visualizations(self) -> Dict[str, go.Figure]:
        """Create visualizations for both models."""
        figures = {}

        # Model 1: Regression Results
        if 'regression' in self.model_results:
            reg_results = self.model_results['regression']

            # Prediction vs Actual
            fig_reg = go.Figure()

            y_test = reg_results['predictions']['y_test']
            y_pred = reg_results['predictions']['y_pred']

            fig_reg.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            ))

            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_reg.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            fig_reg.update_layout(
                title=f"Model 1: Salary Prediction Accuracy (R² = {reg_results['test_r2']:.3f})",
                xaxis_title="Actual Salary ($)",
                yaxis_title="Predicted Salary ($)",
                height=500
            )

            figures['regression_accuracy'] = fig_reg

            # Feature Importance
            top_features = reg_results['feature_importance'].head(10)

            fig_importance = go.Figure(go.Bar(
                x=top_features['abs_coefficient'],
                y=top_features['feature'],
                orientation='h',
                marker_color='lightblue'
            ))

            fig_importance.update_layout(
                title="Model 1: Top 10 Most Important Features",
                xaxis_title="Absolute Coefficient Value",
                yaxis_title="Features",
                height=500
            )

            figures['regression_importance'] = fig_importance

        # Model 2: Classification Results
        if 'classification' in self.model_results:
            class_results = self.model_results['classification']

            # Feature Importance
            top_features = class_results['feature_importance'].head(10)

            fig_class_importance = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='lightgreen'
            ))

            fig_class_importance.update_layout(
                title="Model 2: Top 10 Features for Above-Average Salary Classification",
                xaxis_title="Feature Importance",
                yaxis_title="Features",
                height=500
            )

            figures['classification_importance'] = fig_class_importance

            # Prediction Distribution
            y_pred_proba = class_results['predictions']['y_pred_proba']
            y_test = class_results['predictions']['y_test']

            fig_dist = go.Figure()

            # Above-average jobs
            above_avg_proba = y_pred_proba[y_test == 1]
            fig_dist.add_trace(go.Histogram(
                x=above_avg_proba,
                name='Above-Average Jobs',
                opacity=0.7,
                nbinsx=20
            ))

            # Below-average jobs
            below_avg_proba = y_pred_proba[y_test == 0]
            fig_dist.add_trace(go.Histogram(
                x=below_avg_proba,
                name='Below-Average Jobs',
                opacity=0.7,
                nbinsx=20
            ))

            fig_dist.update_layout(
                title="Model 2: Prediction Confidence Distribution",
                xaxis_title="Predicted Probability of Above-Average Salary",
                yaxis_title="Number of Jobs",
                barmode='overlay',
                height=500
            )

            figures['classification_distribution'] = fig_dist

        return figures

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run both models and return comprehensive results."""
        print("🚀 RUNNING COMPLETE SALARY ANALYTICS")
        print("=" * 50)

        # Prepare features
        feature_df = self.prepare_features()

        # Run Model 1: Multiple Linear Regression
        regression_results = self.model_1_multiple_linear_regression(feature_df)

        # Run Model 2: Above-Average Classification
        classification_results = self.model_2_above_average_classification(feature_df)

        # Create visualizations
        figures = self.create_model_visualizations()

        # Summary insights
        insights = self.generate_insights()

        return {
            'regression': regression_results,
            'classification': classification_results,
            'figures': figures,
            'insights': insights,
            'data_summary': {
                'total_records': len(feature_df),
                'salary_range': {
                    'min': feature_df[self.salary_col].min(),
                    'max': feature_df[self.salary_col].max(),
                    'median': feature_df[self.salary_col].median()
                },
                'unique_locations': feature_df[self.location_col].nunique(),
                'unique_titles': feature_df['title'].nunique(),
                'unique_industries': feature_df['industry'].nunique()
            }
        }

    def generate_insights(self) -> Dict[str, List[str]]:
        """Generate plain-language insights for job seekers."""
        insights = {
            'regression_insights': [],
            'classification_insights': [],
            'job_seeker_recommendations': []
        }

        if 'regression' in self.model_results:
            reg_results = self.model_results['regression']
            r2 = reg_results['test_r2']
            top_features = reg_results['feature_importance'].head(5)

            insights['regression_insights'] = [
                f"Our salary prediction model explains {r2*100:.1f}% of salary variation",
                f"Average prediction error is ${reg_results['test_rmse']:,.0f}",
                f"Top salary driver: {top_features.iloc[0]['feature']}",
                f"Model trained on {reg_results['sample_size']['train']:,} job postings"
            ]

        if 'classification' in self.model_results:
            class_results = self.model_results['classification']
            accuracy = class_results['test_accuracy']
            threshold = class_results['threshold']
            top_features = class_results['feature_importance'].head(5)

            insights['classification_insights'] = [
                f"Above-average salary threshold: ${threshold:,.0f}",
                f"Model correctly identifies high-paying jobs {accuracy*100:.1f}% of the time",
                f"Top predictor of above-average pay: {top_features.iloc[0]['feature']}",
                f"Model analyzes {class_results['sample_size']['train']:,} job classifications"
            ]

        insights['job_seeker_recommendations'] = [
            "Focus on locations and industries identified as high-paying",
            "Develop skills that the models identify as salary drivers",
            "Target job titles that consistently appear in above-average classifications",
            "Consider geographic mobility to access higher-paying markets",
            "Build experience in areas the models show have strong salary correlation"
        ]

        return insights
