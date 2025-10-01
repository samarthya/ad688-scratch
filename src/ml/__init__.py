"""
Machine Learning Module for Job Market Analytics

This module provides machine learning capabilities for salary disparity analysis,
job market segmentation, and predictive modeling.
"""

from .clustering import JobMarketClusterer
from .regression import SalaryRegressionModel
from .classification import JobClassificationModel
from .feature_engineering import SalaryDisparityFeatureEngineer
from .evaluation import ModelEvaluator
from .salary_disparity import SalaryDisparityAnalyzer

__all__ = [
    'JobMarketClusterer',
    'SalaryRegressionModel',
    'JobClassificationModel',
    'SalaryDisparityFeatureEngineer',
    'ModelEvaluator',
    'SalaryDisparityAnalyzer'
]
