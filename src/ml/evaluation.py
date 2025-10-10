"""
Model Evaluation Metrics for Machine Learning Models

This module provides comprehensive evaluation metrics for both regression
and classification models used in salary disparity analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql.functions import col, abs as spark_abs, sqrt, mean, stddev
import numpy as np


class ModelEvaluator:
    """
    Comprehensive model evaluation for salary disparity analysis.

    Provides evaluation metrics for:
    - Regression models: RMSE, R², MAE, MAPE
    - Classification models: Accuracy, F1 Score, Precision, Recall, Confusion Matrix
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

        # Regression evaluators
        self.rmse_evaluator = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse"
        )
        self.r2_evaluator = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="r2"
        )
        self.mae_evaluator = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="mae"
        )

        # Classification evaluators
        self.accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        self.f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        self.precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        self.recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )

    def evaluate_regression_model(self, predictions: DataFrame,
                                model_name: str = "Regression Model") -> Dict[str, float]:
        """Evaluate regression model performance."""

        print(f"Evaluating {model_name}...")

        # Calculate standard metrics
        rmse = self.rmse_evaluator.evaluate(predictions)
        r2 = self.r2_evaluator.evaluate(predictions)
        mae = self.mae_evaluator.evaluate(predictions)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = self._calculate_mape(predictions)

        # Calculate additional metrics
        mse = rmse ** 2
        mape_percent = mape * 100

        evaluation_results = {
            'model_name': model_name,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'mape_percent': mape_percent
        }

        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape_percent:.2f}%")

        return evaluation_results

    def evaluate_classification_model(self, predictions: DataFrame,
                                    model_name: str = "Classification Model") -> Dict[str, float]:
        """Evaluate classification model performance."""

        print(f"Evaluating {model_name}...")

        # Calculate standard metrics
        accuracy = self.accuracy_evaluator.evaluate(predictions)
        f1 = self.f1_evaluator.evaluate(predictions)
        precision = self.precision_evaluator.evaluate(predictions)
        recall = self.recall_evaluator.evaluate(predictions)

        # Calculate confusion matrix
        confusion_matrix = self._calculate_confusion_matrix(predictions)

        evaluation_results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_matrix
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        return evaluation_results

    def _calculate_mape(self, predictions: DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error."""

        # Calculate absolute percentage error for each prediction
        mape_df = predictions.withColumn(
            'ape',
            spark_abs(col('prediction') - col('label')) / col('label')
        )

        # Calculate mean
        mape = mape_df.select(mean('ape')).collect()[0][0]

        return mape if mape is not None else 0.0

    def _calculate_confusion_matrix(self, predictions: DataFrame) -> Dict[str, Any]:
        """Calculate confusion matrix for classification results."""

        # Get unique labels
        unique_labels = predictions.select('label').distinct().collect()
        labels = [row['label'] for row in unique_labels]

        # Calculate confusion matrix
        confusion_matrix = {}
        for actual in labels:
            confusion_matrix[actual] = {}
            for predicted in labels:
                count = predictions.filter(
                    (col('label') == actual) & (col('prediction') == predicted)
                ).count()
                confusion_matrix[actual][predicted] = count

        return confusion_matrix

    def compare_models(self, model_results: List[Dict[str, Any]],
                      metric_type: str = "regression") -> Dict[str, Any]:
        """Compare multiple models based on their performance."""

        if not model_results:
            return {"error": "No model results provided"}

        if metric_type == "regression":
            # Compare regression models
            best_rmse = min(model_results, key=lambda x: x['rmse'])
            best_r2 = max(model_results, key=lambda x: x['r2'])
            best_mae = min(model_results, key=lambda x: x['mae'])

            comparison = {
                'best_rmse': best_rmse,
                'best_r2': best_r2,
                'best_mae': best_mae,
                'all_results': model_results
            }

        elif metric_type == "classification":
            # Compare classification models
            best_accuracy = max(model_results, key=lambda x: x['accuracy'])
            best_f1 = max(model_results, key=lambda x: x['f1_score'])
            best_precision = max(model_results, key=lambda x: x['precision'])
            best_recall = max(model_results, key=lambda x: x['recall'])

            comparison = {
                'best_accuracy': best_accuracy,
                'best_f1': best_f1,
                'best_precision': best_precision,
                'best_recall': best_recall,
                'all_results': model_results
            }

        else:
            comparison = {"error": f"Unknown metric type: {metric_type}"}

        return comparison

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""

        report = f"Model Evaluation Report: {evaluation_results['model_name']}\n"
        report += "=" * 50 + "\n\n"

        if 'rmse' in evaluation_results:
            # Regression model report
            report += "REGRESSION METRICS:\n"
            report += f" RMSE: {evaluation_results['rmse']:.2f}\n"
            report += f" R²: {evaluation_results['r2']:.4f}\n"
            report += f" MAE: {evaluation_results['mae']:.2f}\n"
            report += f" MSE: {evaluation_results['mse']:.2f}\n"
            report += f" MAPE: {evaluation_results['mape_percent']:.2f}%\n\n"

            # Interpretation
            report += "INTERPRETATION:\n"
            if evaluation_results['r2'] > 0.7:
                report += " Good model fit (R² > 0.7)\n"
            elif evaluation_results['r2'] > 0.5:
                report += " Moderate model fit (R² > 0.5)\n"
            else:
                report += " Poor model fit (R² < 0.5)\n"

            if evaluation_results['mape_percent'] < 10:
                report += " Low prediction error (MAPE < 10%)\n"
            elif evaluation_results['mape_percent'] < 20:
                report += " Moderate prediction error (MAPE < 20%)\n"
            else:
                report += " High prediction error (MAPE > 20%)\n"

        elif 'accuracy' in evaluation_results:
            # Classification model report
            report += "CLASSIFICATION METRICS:\n"
            report += f" Accuracy: {evaluation_results['accuracy']:.4f}\n"
            report += f" F1 Score: {evaluation_results['f1_score']:.4f}\n"
            report += f" Precision: {evaluation_results['precision']:.4f}\n"
            report += f" Recall: {evaluation_results['recall']:.4f}\n\n"

            # Interpretation
            report += "INTERPRETATION:\n"
            if evaluation_results['accuracy'] > 0.8:
                report += " High accuracy (> 80%)\n"
            elif evaluation_results['accuracy'] > 0.6:
                report += " Moderate accuracy (> 60%)\n"
            else:
                report += " Low accuracy (< 60%)\n"

            if evaluation_results['f1_score'] > 0.7:
                report += " Good F1 score (> 0.7)\n"
            elif evaluation_results['f1_score'] > 0.5:
                report += " Moderate F1 score (> 0.5)\n"
            else:
                report += " Poor F1 score (< 0.5)\n"

        return report

    def calculate_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for tree-based models."""

        try:
            if hasattr(model, 'featureImportances'):
                # Get feature importance
                importance = model.featureImportances

                # Convert to dictionary
                feature_importance = {}
                for i, feature_name in enumerate(feature_names):
                    if i < len(importance):
                        feature_importance[feature_name] = float(importance[i])

                # Sort by importance
                sorted_importance = dict(sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))

                return sorted_importance
            else:
                return {"error": "Model does not support feature importance"}

        except Exception as e:
            return {"error": f"Error calculating feature importance: {str(e)}"}

    def validate_model_assumptions(self, predictions: DataFrame,
                                 model_type: str = "regression") -> Dict[str, Any]:
        """Validate model assumptions for better interpretation."""

        validation_results = {}

        if model_type == "regression":
            # Check residuals
            residuals = predictions.withColumn(
                'residual',
                col('prediction') - col('label')
            )

            # Calculate residual statistics
            residual_stats = residuals.select(
                mean('residual').alias('mean_residual'),
                stddev('residual').alias('std_residual')
            ).collect()[0]

            validation_results['residual_mean'] = residual_stats['mean_residual']
            validation_results['residual_std'] = residual_stats['std_residual']
            validation_results['residual_mean_interpretation'] = (
                "Residuals should be close to 0 for unbiased predictions"
            )

        return validation_results
