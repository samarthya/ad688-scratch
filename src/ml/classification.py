"""
Classification Models for Job Market Analysis

This module provides classification models for analyzing job market patterns
and salary disparity indicators.
"""

from typing import Dict, List, Optional, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, isnan, isnull, mean, stddev
import numpy as np


class JobClassificationModel:
    """
    Classification models for job market analysis.

    Supports:
    - Logistic Regression
    - Random Forest Classification

    Focuses on classifying jobs based on salary levels, job categories,
    and other market indicators.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.models = {}
        self.pipelines = {}
        self.feature_columns = []
        self.label_columns = {}

    def prepare_classification_data(self, df: DataFrame,
                                  feature_columns: List[str],
                                  target_column: str) -> DataFrame:
        """Prepare data for classification modeling."""

        # Select features and target
        classification_data = df.select(
            *feature_columns,
            col(target_column).alias('label')
        )

        # Remove rows with null values
        classification_data = classification_data.filter(
            col('label').isNotNull()
        )

        # Remove rows with null feature values
        for col_name in feature_columns:
            classification_data = classification_data.filter(col(col_name).isNotNull())

        return classification_data

    def create_classification_pipeline(self, feature_columns: List[str],
                                     model_type: str = 'logistic') -> Pipeline:
        """Create pipeline for classification modeling."""

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

        # Choose classification model
        if model_type == 'logistic':
            classification_model = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=100,
                regParam=0.01,
                elasticNetParam=0.8
            )
        elif model_type == 'random_forest':
            classification_model = RandomForestClassifier(
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
                          [vector_assembler, classification_model])

        return pipeline

    def create_salary_level_classification(self, df: DataFrame,
                                         feature_columns: List[str]) -> DataFrame:
        """Create salary level classification target variable."""

        # Calculate salary percentiles
        salary_percentiles = df.select('salary_avg').approxQuantile(
            'salary_avg', [0.25, 0.5, 0.75], 0.1
        )

        # Create salary level categories
        df_with_salary_level = df.withColumn(
            'salary_level',
            when(col('salary_avg') <= salary_percentiles[0], 'Low')
            .when(col('salary_avg') <= salary_percentiles[1], 'Below_Median')
            .when(col('salary_avg') <= salary_percentiles[2], 'Above_Median')
            .otherwise('High')
        )

        return df_with_salary_level

    def create_above_average_salary_classification(self, df: DataFrame,
                                                 feature_columns: List[str]) -> DataFrame:
        """Create above-average salary classification target variable."""

        # Calculate median salary
        median_salary = df.select('salary_avg').agg({'salary_avg': 'median'}).collect()[0][0]

        # Create binary classification
        df_with_above_avg = df.withColumn(
            'above_average_salary',
            when(col('salary_avg') > median_salary, 1).otherwise(0)
        )

        return df_with_above_avg

    def train_logistic_regression(self, df: DataFrame,
                                feature_columns: List[str],
                                target_column: str) -> Dict[str, Any]:
        """Train Logistic Regression model."""

        print(f"Training Logistic Regression model for {target_column}...")

        # Prepare data
        classification_data = self.prepare_classification_data(df, feature_columns, target_column)

        # Create pipeline
        pipeline = self.create_classification_pipeline(feature_columns, 'logistic')

        # Split data (70/30)
        train_data, test_data = classification_data.randomSplit([0.7, 0.3], seed=42)

        # Train model
        model = pipeline.fit(train_data)

        # Make predictions
        train_predictions = model.transform(train_data)
        test_predictions = model.transform(test_data)

        # Store model and pipeline
        model_name = f'logistic_regression_{target_column}'
        self.models[model_name] = model
        self.pipelines[model_name] = pipeline
        self.feature_columns = feature_columns
        self.label_columns[model_name] = target_column

        # Calculate training metrics
        train_metrics = self._calculate_classification_metrics(train_predictions, 'Training')

        # Calculate test metrics
        test_metrics = self._calculate_classification_metrics(test_predictions, 'Test')

        # Get feature importance
        feature_importance = self._get_logistic_regression_coefficients(
            model, feature_columns
        )

        results = {
            'model_type': 'Logistic Regression',
            'target_column': target_column,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model': model,
            'predictions': test_predictions
        }

        return results

    def train_random_forest_classification(self, df: DataFrame,
                                         feature_columns: List[str],
                                         target_column: str) -> Dict[str, Any]:
        """Train Random Forest Classification model."""

        print(f"Training Random Forest Classification model for {target_column}...")

        # Prepare data
        classification_data = self.prepare_classification_data(df, feature_columns, target_column)

        # Create pipeline
        pipeline = self.create_classification_pipeline(feature_columns, 'random_forest')

        # Split data (70/30)
        train_data, test_data = classification_data.randomSplit([0.7, 0.3], seed=42)

        # Train model
        model = pipeline.fit(train_data)

        # Make predictions
        train_predictions = model.transform(train_data)
        test_predictions = model.transform(test_data)

        # Store model and pipeline
        model_name = f'random_forest_classification_{target_column}'
        self.models[model_name] = model
        self.pipelines[model_name] = pipeline
        self.feature_columns = feature_columns
        self.label_columns[model_name] = target_column

        # Calculate training metrics
        train_metrics = self._calculate_classification_metrics(train_predictions, 'Training')

        # Calculate test metrics
        test_metrics = self._calculate_classification_metrics(test_predictions, 'Test')

        # Get feature importance
        feature_importance = self._get_random_forest_importance(
            model, feature_columns
        )

        results = {
            'model_type': 'Random Forest Classification',
            'target_column': target_column,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model': model,
            'predictions': test_predictions
        }

        return results

    def _calculate_classification_metrics(self, predictions: DataFrame,
                                        dataset_type: str) -> Dict[str, float]:
        """Calculate classification metrics for predictions."""

        # Calculate basic metrics
        total_predictions = predictions.count()
        correct_predictions = predictions.filter(col('prediction') == col('label')).count()

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate precision, recall, and F1 for each class
        class_metrics = self._calculate_per_class_metrics(predictions)

        # Calculate weighted averages
        weighted_precision = sum(
            metrics['precision'] * metrics['support']
            for metrics in class_metrics.values()
        ) / sum(metrics['support'] for metrics in class_metrics.values())

        weighted_recall = sum(
            metrics['recall'] * metrics['support']
            for metrics in class_metrics.values()
        ) / sum(metrics['support'] for metrics in class_metrics.values())

        weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0

        metrics = {
            'dataset': dataset_type,
            'accuracy': accuracy,
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1,
            'per_class_metrics': class_metrics
        }

        print(f"{dataset_type} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {weighted_precision:.4f}")
        print(f"  Recall: {weighted_recall:.4f}")
        print(f"  F1 Score: {weighted_f1:.4f}")

        return metrics

    def _calculate_per_class_metrics(self, predictions: DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 for each class."""

        # Get unique classes
        unique_classes = predictions.select('label').distinct().collect()
        class_metrics = {}

        for row in unique_classes:
            class_label = row['label']

            # Calculate true positives, false positives, false negatives
            tp = predictions.filter(
                (col('label') == class_label) & (col('prediction') == class_label)
            ).count()

            fp = predictions.filter(
                (col('label') != class_label) & (col('prediction') == class_label)
            ).count()

            fn = predictions.filter(
                (col('label') == class_label) & (col('prediction') != class_label)
            ).count()

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn

            class_metrics[class_label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            }

        return class_metrics

    def _get_logistic_regression_coefficients(self, model,
                                            feature_columns: List[str]) -> Dict[str, float]:
        """Get feature coefficients from logistic regression model."""

        try:
            # Get the logistic regression model from pipeline
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
            print(f"Error getting logistic regression coefficients: {e}")
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

    def predict_classification(self, df: DataFrame,
                             model_name: str) -> DataFrame:
        """Predict classification for new data using trained model."""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        # Use the trained model to make predictions
        predictions = self.models[model_name].transform(df)

        return predictions

    def analyze_classification_disparity(self, predictions: DataFrame,
                                       group_by_column: str) -> Dict[str, Any]:
        """Analyze classification disparity across different groups."""

        # Group by specified column and calculate classification statistics
        group_stats = predictions.groupBy(group_by_column).agg(
            {'label': 'mean', 'prediction': 'mean'}
        ).collect()

        # Calculate disparity metrics
        disparity_analysis = {}
        for row in group_stats:
            group = row[group_by_column]
            actual_rate = row['avg(label)']
            predicted_rate = row['avg(prediction)']

            disparity_analysis[group] = {
                'actual_rate': actual_rate,
                'predicted_rate': predicted_rate,
                'disparity': abs(actual_rate - predicted_rate)
            }

        return disparity_analysis

    def get_model_summary(self, model_name: str) -> str:
        """Get a summary of the trained model."""

        if model_name not in self.models:
            return f"Model {model_name} not found."

        model = self.models[model_name]
        model_type = model_name.replace('_', ' ').title()
        target_column = self.label_columns.get(model_name, 'Unknown')

        summary = f"{model_type} Model Summary\n"
        summary += "=" * 30 + "\n"
        summary += f"Model Type: {model_type}\n"
        summary += f"Target Column: {target_column}\n"
        summary += f"Features Used: {len(self.feature_columns)}\n"
        summary += f"Feature List: {', '.join(self.feature_columns[:5])}...\n"

        return summary
