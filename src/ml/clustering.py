"""
KMeans Clustering for Job Market Segmentation

This module provides KMeans clustering capabilities for analyzing job market
segmentation based on salary disparity factors.
"""

from typing import Dict, List, Optional, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, count, avg, max as spark_max, min as spark_min
import numpy as np

# Import logger for controlled output
from src.utils.logger import get_logger
logger = get_logger(level="WARNING")


class JobMarketClusterer:
    """
    KMeans clustering for job market segmentation analysis.

    Uses SOC, NAICS, or ONET job classification as reference labels
    to identify salary disparity patterns across different job segments.
    """

    def __init__(self, spark: SparkSession, k: int = 5):
        self.spark = spark
        self.k = k
        self.kmeans_model = None
        self.pipeline = None
        self.cluster_centers = None
        self.cluster_stats = None

    def prepare_clustering_features(self, df: DataFrame) -> DataFrame:
        """Prepare features specifically for clustering analysis."""

        # Select features for clustering
        clustering_features = [
            'salary_avg', 'education_level', 'experience_level',
            'ai_skills_score', 'technical_skills_score', 'soft_skills_score',
            'location_cost_index', 'industry_growth_rate'
        ]

        # Filter out null values for clustering
        df_clean = df.filter(
            col('salary_avg').isNotNull() &
            col('education_level').isNotNull() &
            col('experience_level').isNotNull()
        )

        return df_clean

    def create_clustering_pipeline(self, feature_columns: List[str]) -> Pipeline:
        """Create pipeline for clustering feature preparation."""

        # Vector assembler for features
        vector_assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="raw_features"
        )

        # Standard scaler for feature normalization
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        # KMeans clustering
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=self.k,
            seed=42,
            maxIter=100
        )

        # Create pipeline
        pipeline = Pipeline(stages=[vector_assembler, scaler, kmeans])

        return pipeline

    def fit_clustering_model(self, df: DataFrame) -> 'JobMarketClusterer':
        """Fit KMeans clustering model to the data."""

        logger.info(f"Fitting KMeans clustering model with k={self.k}...")

        # Prepare features
        df_features = self.prepare_clustering_features(df)

        # Define clustering features
        clustering_features = [
            'salary_avg', 'education_level', 'ai_skills_score',
            'technical_skills_score', 'soft_skills_score', 'location_cost_index'
        ]

        # Create pipeline
        self.pipeline = self.create_clustering_pipeline(clustering_features)

        # Fit model
        self.kmeans_model = self.pipeline.fit(df_features)

        # Get cluster centers
        self.cluster_centers = self.kmeans_model.stages[-1].clusterCenters()

        logger.info(f"Clustering model fitted successfully with {len(self.cluster_centers)} clusters")

        return self

    def predict_clusters(self, df: DataFrame) -> DataFrame:
        """Predict clusters for new data."""

        if self.kmeans_model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare features
        df_features = self.prepare_clustering_features(df)

        # Make predictions
        predictions = self.kmeans_model.transform(df_features)

        return predictions

    def analyze_cluster_characteristics(self, df: DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""

        if self.kmeans_model is None:
            raise ValueError("Model must be fitted before analyzing clusters")

        # Get predictions
        predictions = self.predict_clusters(df)

        # Calculate cluster statistics
        cluster_stats = predictions.groupBy("cluster").agg(
            count("*").alias("job_count"),
            avg("salary_avg").alias("avg_salary"),
            spark_max("salary_avg").alias("max_salary"),
            spark_min("salary_avg").alias("min_salary"),
            avg("education_level").alias("avg_education"),
            avg("ai_skills_score").alias("avg_ai_skills"),
            avg("technical_skills_score").alias("avg_technical_skills"),
            avg("soft_skills_score").alias("avg_soft_skills"),
            avg("location_cost_index").alias("avg_location_cost")
        ).collect()

        # Convert to dictionary
        self.cluster_stats = {}
        for row in cluster_stats:
            cluster_id = row['cluster']
            self.cluster_stats[cluster_id] = {
                'job_count': row['job_count'],
                'avg_salary': row['avg_salary'],
                'max_salary': row['max_salary'],
                'min_salary': row['min_salary'],
                'avg_education': row['avg_education'],
                'avg_ai_skills': row['avg_ai_skills'],
                'avg_technical_skills': row['avg_technical_skills'],
                'avg_soft_skills': row['avg_soft_skills'],
                'avg_location_cost': row['avg_location_cost']
            }

        return self.cluster_stats

    def identify_salary_disparity_patterns(self, df: DataFrame) -> Dict[str, Any]:
        """Identify salary disparity patterns across clusters."""

        if self.cluster_stats is None:
            self.analyze_cluster_characteristics(df)

        # Calculate salary disparity metrics
        salary_ranges = [stats['max_salary'] - stats['min_salary'] for stats in self.cluster_stats.values()]
        avg_salaries = [stats['avg_salary'] for stats in self.cluster_stats.values()]

        # Find clusters with highest salary disparities
        max_disparity_cluster = max(self.cluster_stats.keys(),
                                  key=lambda k: self.cluster_stats[k]['max_salary'] - self.cluster_stats[k]['min_salary'])

        # Find clusters with highest average salaries
        highest_salary_cluster = max(self.cluster_stats.keys(),
                                   key=lambda k: self.cluster_stats[k]['avg_salary'])

        # Find clusters with lowest average salaries
        lowest_salary_cluster = min(self.cluster_stats.keys(),
                                  key=lambda k: self.cluster_stats[k]['avg_salary'])

        disparity_analysis = {
            'max_disparity_cluster': max_disparity_cluster,
            'highest_salary_cluster': highest_salary_cluster,
            'lowest_salary_cluster': lowest_salary_cluster,
            'salary_range_ratio': max(salary_ranges) / min(salary_ranges) if min(salary_ranges) > 0 else 0,
            'avg_salary_ratio': max(avg_salaries) / min(avg_salaries) if min(avg_salaries) > 0 else 0,
            'cluster_characteristics': self.cluster_stats
        }

        return disparity_analysis

    def get_cluster_summary(self) -> str:
        """Get a summary of cluster characteristics."""

        if self.cluster_stats is None:
            return "No cluster analysis available. Please fit the model first."

        summary = f"Job Market Segmentation Analysis (k={self.k})\n"
        summary += "=" * 50 + "\n\n"

        for cluster_id, stats in self.cluster_stats.items():
            summary += f"Cluster {cluster_id}:\n"
            summary += f"  Job Count: {stats['job_count']:,}\n"
            summary += f"  Average Salary: ${stats['avg_salary']:,.0f}\n"
            summary += f"  Salary Range: ${stats['min_salary']:,.0f} - ${stats['max_salary']:,.0f}\n"
            summary += f"  Average Education Level: {stats['avg_education']:.1f}\n"
            summary += f"  AI Skills Score: {stats['avg_ai_skills']:.1f}\n"
            summary += f"  Technical Skills Score: {stats['avg_technical_skills']:.1f}\n"
            summary += f"  Soft Skills Score: {stats['avg_soft_skills']:.1f}\n"
            summary += f"  Location Cost Index: {stats['avg_location_cost']:.2f}\n\n"

        return summary

    def recommend_optimal_k(self, df: DataFrame, max_k: int = 10) -> int:
        """Recommend optimal number of clusters using elbow method."""

        logger.info("Calculating optimal k using elbow method...")

        # Prepare features
        df_features = self.prepare_clustering_features(df)

        # Calculate WCSS for different k values
        wcss_values = []
        k_values = range(2, max_k + 1)

        for k in k_values:
            # Create temporary pipeline
            temp_pipeline = self.create_clustering_pipeline([
                'salary_avg', 'education_level', 'ai_skills_score',
                'technical_skills_score', 'soft_skills_score', 'location_cost_index'
            ])

            # Update KMeans with current k
            temp_pipeline.getStages()[-1].setK(k)

            # Fit model
            temp_model = temp_pipeline.fit(df_features)

            # Calculate WCSS
            predictions = temp_model.transform(df_features)
            wcss = temp_model.stages[-1].computeCost(predictions)
            wcss_values.append(wcss)

            logger.info(f"k={k}: WCSS={wcss:.2f}")

        # Find elbow point (simplified)
        if len(wcss_values) > 1:
            # Calculate second derivative to find elbow
            second_derivatives = []
            for i in range(1, len(wcss_values) - 1):
                second_deriv = wcss_values[i-1] - 2*wcss_values[i] + wcss_values[i+1]
                second_derivatives.append(second_deriv)

            # Find maximum second derivative (elbow point)
            elbow_idx = second_derivatives.index(max(second_derivatives)) + 1
            optimal_k = k_values[elbow_idx]
        else:
            optimal_k = 3  # Default fallback

        logger.info(f"Recommended optimal k: {optimal_k}")
        return optimal_k
