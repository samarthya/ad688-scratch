"""
Regression Models for Salary Prediction

This module provides regression models for predicting salary based on
various features, focusing on salary disparity analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, isnan, isnull
import numpy as np


class SalaryRegressionModel:
    """
    Regression models for salary prediction and disparity analysis.

    Supports:
    - Multiple Linear Regression
    - Random Forest Regression

    Focuses on salary disparity factors like location, job title, and skills.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.models = {}
        self.pipelines = {}
        self.feature_columns = []
        self.target_column = 'SALARY_AVG_IMPUTED'

    def prepare_regression_data(self, df: DataFrame,
                              feature_columns: List[str]) -> DataFrame:
        """Prepare data for regression modeling."""

        # Select features and target
        regression_data = df.select(
            *feature_columns,
            col(self.target_column).alias('label')
        )

        # Remove rows with null values
        regression_data = regression_data.filter(
            col('label').isNotNull() &
            col('label') > 0  # Only positive salaries
        )

        # Remove rows with null feature values
        for col_name in feature_columns:
            regression_data = regression_data.filter(col(col_name).isNotNull())

        return regression_data

    def create_regression_pipeline(self, feature_columns: List[str],
                                 model_type: str = 'linear') -> Pipeline:
        """Create pipeline for regression modeling."""

        # Identify categorical features
        categorical_features = [
            'experience_level', 'location_tier', 'job_title_category',
            'industry_category', 'company_size_tier', 'remote_type', 'employment_type'
        ]

        # String indexers for categorical features
        string_indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
            for col in categorical_features if col in feature_columns
        ]

        # One-hot encoders for categorical features
        one_hot_encoders = [
            OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
            for col in categorical_features if col in feature_columns
        ]

        # Vector assembler for all features
        vector_assembler = VectorAssembler(
            inputCols=[col for col in feature_columns if col not in categorical_features] +
                     [f"{col}_encoded" for col in categorical_features if col in feature_columns],
            outputCol="features"
        )

        # Choose regression model
        if model_type == 'linear':
            regression_model = LinearRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=100,
                regParam=0.01,
                elasticNetParam=0.8
            )
        elif model_type == 'random_forest':
            regression_model = RandomForestRegressor(
                featuresCol="features",
                labelCol="label",
                numTrees=100,
                maxDepth=10,
                seed=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Create pipeline
        pipeline = Pipeline(stages=string_indexers + one_hot_encoders +
                          [vector_assembler, regression_model])

        return pipeline

    def train_linear_regression(self, df: DataFrame,
                              feature_columns: List[str]) -> Dict[str, Any]:
        """Train Multiple Linear Regression model."""

        print("Training Multiple Linear Regression model...")

        # Prepare data
        regression_data = self.prepare_regression_data(df, feature_columns)

        # Create pipeline
        pipeline = self.create_regression_pipeline(feature_columns, 'linear')

        # Split data (70/30)
        train_data, test_data = regression_data.randomSplit([0.7, 0.3], seed=42)

        # Train model
        model = pipeline.fit(train_data)

        # Make predictions
        train_predictions = model.transform(train_data)
        test_predictions = model.transform(test_data)

        # Store model and pipeline
        self.models['linear_regression'] = model
        self.pipelines['linear_regression'] = pipeline
        self.feature_columns = feature_columns

        # Calculate training metrics
        train_metrics = self._calculate_regression_metrics(train_predictions, 'Training')

        # Calculate test metrics
        test_metrics = self._calculate_regression_metrics(test_predictions, 'Test')

        # Get feature importance (for linear regression, use coefficients)
        feature_importance = self._get_linear_regression_coefficients(
            model, feature_columns
        )

        results = {
            'model_type': 'Multiple Linear Regression',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model': model,
            'predictions': test_predictions
        }

        return results

    def train_random_forest_regression(self, df: DataFrame,
                                     feature_columns: List[str]) -> Dict[str, Any]:
        """Train Random Forest Regression model."""

        print("Training Random Forest Regression model...")

        # Prepare data
        regression_data = self.prepare_regression_data(df, feature_columns)

        # Create pipeline
        pipeline = self.create_regression_pipeline(feature_columns, 'random_forest')

        # Split data (70/30)
        train_data, test_data = regression_data.randomSplit([0.7, 0.3], seed=42)

        # Train model
        model = pipeline.fit(train_data)

        # Make predictions
        train_predictions = model.transform(train_data)
        test_predictions = model.transform(test_data)

        # Store model and pipeline
        self.models['random_forest_regression'] = model
        self.pipelines['random_forest_regression'] = pipeline
        self.feature_columns = feature_columns

        # Calculate training metrics
        train_metrics = self._calculate_regression_metrics(train_predictions, 'Training')

        # Calculate test metrics
        test_metrics = self._calculate_regression_metrics(test_predictions, 'Test')

        # Get feature importance
        feature_importance = self._get_random_forest_importance(
            model, feature_columns
        )

        results = {
            'model_type': 'Random Forest Regression',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model': model,
            'predictions': test_predictions
        }

        return results

    def _calculate_regression_metrics(self, predictions: DataFrame,
                                    dataset_type: str) -> Dict[str, float]:
        """Calculate regression metrics for predictions."""

        # Calculate basic metrics
        predictions = predictions.withColumn(
            'residual', col('prediction') - col('label')
        )

        # Calculate RMSE
        rmse = predictions.select(
            (col('residual') ** 2).alias('squared_error')
        ).agg({'squared_error': 'mean'}).collect()[0][0] ** 0.5

        # Calculate MAE
        mae = predictions.select(
            col('residual').alias('abs_error')
        ).agg({'abs_error': 'mean'}).collect()[0][0]

        # Calculate R²
        r2 = self._calculate_r2(predictions)

        # Calculate MAPE
        mape = self._calculate_mape(predictions)

        metrics = {
            'dataset': dataset_type,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'mape_percent': mape * 100
        }

        print(f"{dataset_type} Metrics:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape*100:.2f}%")

        return metrics

    def _calculate_r2(self, predictions: DataFrame) -> float:
        """Calculate R² score."""

        # Calculate total sum of squares
        mean_label = predictions.select('label').agg({'label': 'mean'}).collect()[0][0]
        tss = predictions.select(
            (col('label') - mean_label) ** 2
        ).agg({'(label - mean_label) ** 2': 'sum'}).collect()[0][0]

        # Calculate residual sum of squares
        rss = predictions.select(
            col('residual') ** 2
        ).agg({'residual ** 2': 'sum'}).collect()[0][0]

        # Calculate R²
        r2 = 1 - (rss / tss) if tss > 0 else 0

        return r2

    def _calculate_mape(self, predictions: DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error."""

        # Calculate absolute percentage error
        mape_df = predictions.withColumn(
            'ape',
            col('residual').abs() / col('label')
        )

        # Calculate mean
        mape = mape_df.select('ape').agg({'ape': 'mean'}).collect()[0][0]

        return mape if mape is not None else 0.0

    def _get_linear_regression_coefficients(self, model,
                                          feature_columns: List[str]) -> Dict[str, float]:
        """Get feature coefficients from linear regression model."""

        try:
            # Get the linear regression model from pipeline
            lr_model = model.stages[-1]
            coefficients = lr_model.coefficients

            # Map coefficients to feature names
            feature_importance = {}
            for i, feature in enumerate(feature_columns):
                if i < len(coefficients):
                    feature_importance[feature] = float(coefficients[i])

            # Sort by absolute value
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))

            return sorted_importance

        except Exception as e:
            print(f"Error getting linear regression coefficients: {e}")
            return {}

    def _get_random_forest_importance(self, model,
                                    feature_columns: List[str]) -> Dict[str, float]:
        """Get feature importance from random forest model."""

        try:
            # Get the random forest model from pipeline
            rf_model = model.stages[-1]
            importance = rf_model.featureImportances

            # Map importance to feature names
            feature_importance = {}
            for i, feature in enumerate(feature_columns):
                if i < len(importance):
                    feature_importance[feature] = float(importance[i])

            # Sort by importance
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))

            return sorted_importance

        except Exception as e:
            print(f"Error getting random forest importance: {e}")
            return {}

    def predict_salary(self, df: DataFrame, model_name: str = 'linear_regression') -> DataFrame:
        """Predict salary for new data using trained model."""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        # Use the trained model to make predictions
        predictions = self.models[model_name].transform(df)

        return predictions

    def analyze_salary_disparity(self, predictions: DataFrame,
                               group_by_column: str) -> Dict[str, Any]:
        """Analyze salary disparity across different groups."""

        # Group by specified column and calculate salary statistics
        group_stats = predictions.groupBy(group_by_column).agg(
            {'label': 'mean', 'prediction': 'mean', 'residual': 'mean'}
        ).collect()

        # Calculate disparity metrics
        disparity_analysis = {}
        for row in group_stats:
            group = row[group_by_column]
            actual_salary = row['avg(label)']
            predicted_salary = row['avg(prediction)']
            residual = row['avg(residual)']

            disparity_analysis[group] = {
                'actual_salary': actual_salary,
                'predicted_salary': predicted_salary,
                'residual': residual,
                'prediction_error_percent': (residual / actual_salary) * 100 if actual_salary > 0 else 0
            }

        return disparity_analysis

    def get_model_summary(self, model_name: str = 'linear_regression') -> str:
        """Get a summary of the trained model."""

        if model_name not in self.models:
            return f"Model {model_name} not found."

        model = self.models[model_name]
        model_type = model_name.replace('_', ' ').title()

        summary = f"{model_type} Model Summary\n"
        summary += "=" * 30 + "\n"
        summary += f"Model Type: {model_type}\n"
        summary += f"Features Used: {len(self.feature_columns)}\n"
        summary += f"Feature List: {', '.join(self.feature_columns[:5])}...\n"

        return summary
