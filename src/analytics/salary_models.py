"""
Two Main Analytics Models for Salary and Compensation Trends - PySpark MLlib

This module implements the core analytics models using PySpark MLlib:
1. Multiple Linear Regression for salary prediction
2. Random Forest Classification for above-average paying jobs

Refactored to use PySpark MLlib instead of scikit-learn for consistency
with the PySpark-based architecture and learning objectives.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class SalaryAnalyticsModels:
    """
    Two main analytics models for salary and compensation analysis using PySpark MLlib.

    Model 1: Multiple Linear Regression for salary prediction
    Model 2: Random Forest Classification for above-average paying jobs

    All models use PySpark MLlib for distributed machine learning at scale.
    """

    def __init__(self, df: pd.DataFrame = None, spark: SparkSession = None):
        """Initialize with processed job market data."""
        # Initialize Spark session
        if spark is None:
            from src.utils.spark_utils import create_spark_session
            self.spark = create_spark_session("SalaryAnalyticsModels")
        else:
            self.spark = spark

        # Load data
        if df is None:
            from src.data.auto_processor import load_analysis_data
            pandas_df = load_analysis_data("analytics")
        else:
            pandas_df = df.copy()

        # Convert Pandas DataFrame to Spark DataFrame
        self.spark_df = self.spark.createDataFrame(pandas_df)
        self.pandas_df = pandas_df  # Keep for summary stats

        self.models = {}
        self.pipelines = {}
        self.model_results = {}

        # Use existing column mapping abstraction
        from src.config.column_mapping import get_analysis_column
        self.salary_col = get_analysis_column('salary')  # 'salary_avg'
        self.location_col = get_analysis_column('city')  # 'city_name'

        print(f"Initialized with {self.spark_df.count():,} records")
        print(f"Using salary column: {self.salary_col}")
        print(f"Using location column: {self.location_col}")

    def prepare_features(self) -> SparkDataFrame:
        """
        Prepare features for both models using PySpark.

        Features used:
        - Location (city_name): Geographic salary variations
        - Job Title (title): Role-based compensation differences
        - Industry (industry): Sector-specific pay scales
        - Experience (experience_years): Career progression impact
        - Skills (required_skills): Technical skill premiums
        """
        print("\n=== FEATURE PREPARATION (PySpark) ===")

        # Start with Spark DataFrame
        feature_df = self.spark_df

        # Ensure required columns exist
        required_cols = [self.salary_col, self.location_col, 'title', 'industry']
        existing_cols = feature_df.columns

        for col in required_cols:
            if col not in existing_cols:
                print(f"Warning: Missing column {col}, creating placeholder")
                if col == self.salary_col:
                    feature_df = feature_df.withColumn(col, F.lit(75000.0))
                else:
                    feature_df = feature_df.withColumn(col, F.lit('Unknown'))

        # Clean salary data
        feature_df = feature_df.withColumn(
            self.salary_col,
            F.col(self.salary_col).cast('double')
        )
        feature_df = feature_df.filter(
            (F.col(self.salary_col).isNotNull()) &
            (F.col(self.salary_col) > 0)
        )

        # Create experience features if not exists
        if 'experience_years' not in feature_df.columns:
            # Estimate from salary (simple heuristic)
            feature_df = feature_df.withColumn(
                'experience_years',
                F.when(F.col(self.salary_col) < 60000, F.lit(1.0))
                 .when(F.col(self.salary_col) < 90000, F.lit(3.0))
                 .when(F.col(self.salary_col) < 120000, F.lit(7.0))
                 .otherwise(F.lit(12.0))
            )

        # Create skills score if not exists
        if 'required_skills' in feature_df.columns:
            feature_df = feature_df.withColumn(
                'skills_count',
                F.size(F.split(F.col('required_skills'), ','))
            )
        else:
            feature_df = feature_df.withColumn('skills_count', F.lit(3.0))

        # Clean categorical variables
        for col in [self.location_col, 'title', 'industry']:
            if col in feature_df.columns:
                feature_df = feature_df.withColumn(
                    col,
                    F.when(
                        (F.col(col).isNull()) | (F.col(col) == '') | (F.col(col) == 'nan'),
                        F.lit('Unknown')
                    ).otherwise(F.trim(F.col(col)))
                )

        record_count = feature_df.count()
        print(f"\n[OK] Prepared features for {record_count:,} records")

        # Get summary stats using Pandas for display
        salary_stats = feature_df.select(self.salary_col).toPandas()[self.salary_col]
        print(f"   Salary range: ${salary_stats.min():,.0f} - ${salary_stats.max():,.0f}")
        print(f"   Median salary: ${salary_stats.median():,.0f}")

        return feature_df

    def model_1_multiple_linear_regression(self, feature_df: SparkDataFrame = None) -> Dict[str, Any]:
        """
        MODEL 1: Multiple Linear Regression for Salary Prediction (PySpark MLlib)

        WHAT WE'RE MODELING:
        We're predicting salary based on location, job title, industry, experience, and skills
        using PySpark's distributed linear regression.

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
        print("\n=== MODEL 1: MULTIPLE LINEAR REGRESSION (PySpark MLlib) ===")
        print("Predicting salary using distributed linear regression")

        if feature_df is None:
            feature_df = self.prepare_features()

        # Select top categories to avoid too many features
        top_locations = feature_df.groupBy(self.location_col).count().orderBy(F.desc('count')).limit(10)
        top_locations_list = [row[self.location_col] for row in top_locations.collect()]

        top_titles = feature_df.groupBy('title').count().orderBy(F.desc('count')).limit(10)
        top_titles_list = [row['title'] for row in top_titles.collect()]

        top_industries = feature_df.groupBy('industry').count().orderBy(F.desc('count')).limit(10)
        top_industries_list = [row['industry'] for row in top_industries.collect()]

        # Filter to top categories
        feature_df = feature_df.filter(
            F.col(self.location_col).isin(top_locations_list) &
            F.col('title').isin(top_titles_list) &
            F.col('industry').isin(top_industries_list)
        )

        # Create StringIndexers and OneHotEncoders for categorical features
        location_indexer = StringIndexer(inputCol=self.location_col, outputCol='location_index', handleInvalid='keep')
        location_encoder = OneHotEncoder(inputCol='location_index', outputCol='location_vec')

        title_indexer = StringIndexer(inputCol='title', outputCol='title_index', handleInvalid='keep')
        title_encoder = OneHotEncoder(inputCol='title_index', outputCol='title_vec')

        industry_indexer = StringIndexer(inputCol='industry', outputCol='industry_index', handleInvalid='keep')
        industry_encoder = OneHotEncoder(inputCol='industry_index', outputCol='industry_vec')

        # Assemble features
        assembler = VectorAssembler(
            inputCols=['location_vec', 'title_vec', 'industry_vec', 'experience_years', 'skills_count'],
            outputCol='features_raw'
        )

        # Scale features
        scaler = StandardScaler(inputCol='features_raw', outputCol='features', withMean=True, withStd=True)

        # Linear Regression model
        lr = LinearRegression(
            featuresCol='features',
            labelCol=self.salary_col,
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0
        )

        # Create pipeline
        pipeline = Pipeline(stages=[
            location_indexer, location_encoder,
            title_indexer, title_encoder,
            industry_indexer, industry_encoder,
            assembler, scaler, lr
        ])

        # Split data
        train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)

        print(f"Training on {train_df.count():,} records, testing on {test_df.count():,} records")

        # Train model
        model = pipeline.fit(train_df)
        self.pipelines['regression'] = model

        # Make predictions
        train_predictions = model.transform(train_df)
        test_predictions = model.transform(test_df)

        # Evaluate
        evaluator = RegressionEvaluator(
            labelCol=self.salary_col,
            predictionCol='prediction',
            metricName='r2'
        )

        train_r2 = evaluator.evaluate(train_predictions)
        test_r2 = evaluator.evaluate(test_predictions)

        evaluator.setMetricName('rmse')
        train_rmse = evaluator.evaluate(train_predictions)
        test_rmse = evaluator.evaluate(test_predictions)

        # Get feature importances (coefficients from linear regression)
        lr_model = model.stages[-1]
        coefficients = lr_model.coefficients.toArray()

        results = {
            'model_type': 'Multiple Linear Regression (PySpark MLlib)',
            'purpose': 'Salary Prediction',
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'num_features': len(coefficients),
            'intercept': float(lr_model.intercept),
            'sample_size': {
                'train': train_df.count(),
                'test': test_df.count()
            }
        }

        self.model_results['regression'] = results

        print(f"[OK] Model trained successfully!")
        print(f"   R² Score: {test_r2:.3f} (explains {test_r2*100:.1f}% of salary variance)")
        print(f"   RMSE: ${test_rmse:,.0f} (average prediction error)")
        print(f"   Features: {len(coefficients)} encoded features")

        return results

    def model_2_above_average_classification(self, feature_df: SparkDataFrame = None) -> Dict[str, Any]:
        """
        MODEL 2: Random Forest Classification for Above-Average Paying Jobs (PySpark MLlib)

        WHAT WE'RE MODELING:
        We're classifying jobs as "above-average" or "below-average" paying using
        Random Forest, which handles categorical features well and provides feature importance.

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
        print("\n=== MODEL 2: ABOVE-AVERAGE SALARY CLASSIFICATION (PySpark MLlib) ===")
        print("Classifying jobs using Random Forest")

        if feature_df is None:
            feature_df = self.prepare_features()

        # Create target variable (above median salary)
        median_salary = feature_df.approxQuantile(self.salary_col, [0.5], 0.01)[0]
        feature_df = feature_df.withColumn(
            'label',
            F.when(F.col(self.salary_col) > median_salary, 1.0).otherwise(0.0)
        )

        print(f"Median salary threshold: ${median_salary:,.0f}")
        above_avg_count = feature_df.filter(F.col('label') == 1.0).count()
        total_count = feature_df.count()
        print(f"Above-average jobs: {above_avg_count:,} ({above_avg_count/total_count*100:.1f}%)")

        # Select top categories
        top_locations = feature_df.groupBy(self.location_col).count().orderBy(F.desc('count')).limit(10)
        top_locations_list = [row[self.location_col] for row in top_locations.collect()]

        top_titles = feature_df.groupBy('title').count().orderBy(F.desc('count')).limit(10)
        top_titles_list = [row['title'] for row in top_titles.collect()]

        top_industries = feature_df.groupBy('industry').count().orderBy(F.desc('count')).limit(10)
        top_industries_list = [row['industry'] for row in top_industries.collect()]

        # Filter to top categories
        feature_df = feature_df.filter(
            F.col(self.location_col).isin(top_locations_list) &
            F.col('title').isin(top_titles_list) &
            F.col('industry').isin(top_industries_list)
        )

        # Create StringIndexers for categorical features
        location_indexer = StringIndexer(inputCol=self.location_col, outputCol='location_index', handleInvalid='keep')
        title_indexer = StringIndexer(inputCol='title', outputCol='title_index', handleInvalid='keep')
        industry_indexer = StringIndexer(inputCol='industry', outputCol='industry_index', handleInvalid='keep')

        # Assemble features
        assembler = VectorAssembler(
            inputCols=['location_index', 'title_index', 'industry_index', 'experience_years', 'skills_count'],
            outputCol='features'
        )

        # Random Forest Classifier
        rf = RandomForestClassifier(
            featuresCol='features',
            labelCol='label',
            numTrees=50,
            maxDepth=10,
            seed=42
        )

        # Create pipeline
        pipeline = Pipeline(stages=[
            location_indexer, title_indexer, industry_indexer,
            assembler, rf
        ])

        # Split data
        train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)

        print(f"Training on {train_df.count():,} records, testing on {test_df.count():,} records")

        # Train model
        model = pipeline.fit(train_df)
        self.pipelines['classification'] = model

        # Make predictions
        train_predictions = model.transform(train_df)
        test_predictions = model.transform(test_df)

        # Evaluate
        evaluator = MulticlassClassificationEvaluator(
            labelCol='label',
            predictionCol='prediction',
            metricName='accuracy'
        )

        train_accuracy = evaluator.evaluate(train_predictions)
        test_accuracy = evaluator.evaluate(test_predictions)

        evaluator.setMetricName('f1')
        train_f1 = evaluator.evaluate(train_predictions)
        test_f1 = evaluator.evaluate(test_predictions)

        # Get feature importances
        rf_model = model.stages[-1]
        feature_importance = rf_model.featureImportances.toArray()

        results = {
            'model_type': 'Random Forest Classification (PySpark MLlib)',
            'purpose': 'Above-Average Salary Classification',
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            'median_threshold': float(median_salary),
            'num_trees': 50,
            'sample_size': {
                'train': train_df.count(),
                'test': test_df.count()
            }
        }

        self.model_results['classification'] = results

        print(f"[OK] Model trained successfully!")
        print(f"   Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"   F1 Score: {test_f1:.3f}")
        print(f"   Random Forest: 50 trees, max depth 10")

        return results

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run both models and generate comprehensive results.

        Returns complete analysis with both regression and classification results.
        """
        print("\n" + "="*70)
        print("[START] RUNNING COMPLETE SALARY ANALYTICS (PySpark MLlib)")
        print("="*70)

        # Prepare features once
        feature_df = self.prepare_features()

        # Run both models
        regression_results = self.model_1_multiple_linear_regression(feature_df)
        classification_results = self.model_2_above_average_classification(feature_df)

        comprehensive_results = {
            'regression': regression_results,
            'classification': classification_results,
            'technology': 'PySpark MLlib',
            'models_trained': 2
        }

        print("\n" + "="*70)
        print("[OK] COMPLETE ANALYSIS FINISHED")
        print("="*70)
        print(f"Model 1 (Regression): R² = {regression_results['test_r2']:.3f}")
        print(f"Model 2 (Classification): Accuracy = {classification_results['test_accuracy']:.3f}")

        return comprehensive_results

    def create_analysis_visualizations(self, results: Dict[str, Any]) -> List[go.Figure]:
        """
        Create visualizations for model results.

        Returns list of Plotly figures for regression and classification results.
        """
        print("\n=== CREATING VISUALIZATIONS ===")

        figures = []

        # Model Performance Comparison
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Regression R²',
            x=['Training', 'Testing'],
            y=[results['regression']['train_r2'], results['regression']['test_r2']],
            marker_color='#1f77b4'
        ))

        fig.update_layout(
            title='Model 1: Regression Performance (R² Score)',
            yaxis_title='R² Score',
            yaxis_range=[0, 1],
            height=400
        )

        figures.append(fig)

        # Classification Performance
        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            name='Accuracy',
            x=['Training', 'Testing'],
            y=[results['classification']['train_accuracy'], results['classification']['test_accuracy']],
            marker_color='#2ca02c'
        ))

        fig2.update_layout(
            title='Model 2: Classification Performance (Accuracy)',
            yaxis_title='Accuracy',
            yaxis_range=[0, 1],
            height=400
        )

        figures.append(fig2)

        print(f"[OK] Created {len(figures)} visualizations")

        return figures

    def __del__(self):
        """Clean up Spark session on deletion."""
        if hasattr(self, 'spark'):
            try:
                self.spark.stop()
            except:
                pass


# Convenience function for quick analysis
def run_salary_analytics(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Run complete salary analytics using PySpark MLlib.

    Args:
        df: Optional Pandas DataFrame with job market data

    Returns:
        Dictionary with complete analysis results
    """
    models = SalaryAnalyticsModels(df=df)
    results = models.run_complete_analysis()
    return results
