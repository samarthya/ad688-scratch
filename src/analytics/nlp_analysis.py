"""
NLP Analysis for Job Market Skills and Requirements - PySpark MLlib

This module provides Natural Language Processing capabilities using PySpark MLlib:
- Skills extraction and clustering (PySpark KMeans)
- TF-IDF vectorization (PySpark HashingTF + IDF)
- Word embeddings (PySpark Word2Vec)
- Skill trend analysis

Refactored to use PySpark MLlib for consistency with PySpark-based architecture.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re
from collections import Counter
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, Word2Vec, CountVectorizer, StopWordsRemover
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml import Pipeline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Import logger for controlled output
from src.utils.logger import get_logger
logger = get_logger(level="WARNING")

# Handle optional word cloud dependency
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logger.info("Warning: wordcloud not available. Word cloud generation will be skipped.")


class JobMarketNLPAnalyzer:
    """
    NLP Analysis for job market skills and requirements using PySpark MLlib.

    Provides:
    - Skills extraction and frequency analysis
    - TF-IDF based skill clustering (PySpark)
    - Word embeddings with Word2Vec (PySpark)
    - Skill-salary correlation analysis
    """

    def __init__(self, df: pd.DataFrame = None, spark: SparkSession = None):
        """Initialize with job market data."""
        # Initialize Spark session
        if spark is None:
            from src.utils.spark_utils import create_spark_session
            self.spark = create_spark_session("JobMarketNLPAnalyzer")
        else:
            self.spark = spark

        # Load data
        if df is None:
            from src.data.auto_processor import load_analysis_data
            pandas_df = load_analysis_data("nlp")
        else:
            pandas_df = df.copy()

        # Convert to Spark DataFrame
        self.spark_df = self.spark.createDataFrame(pandas_df)
        self.pandas_df = pandas_df  # Keep for some operations

        self.skills_data = None
        self.clusters = None
        self.word2vec_model = None

        # Common tech skills for reference
        self.tech_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'r', 'sql', 'html', 'css', 'php', 'ruby'],
            'data_science': ['machine learning', 'data science', 'analytics', 'statistics', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'nodejs'],
            'tools': ['git', 'jira', 'confluence', 'tableau', 'power bi', 'excel', 'slack']
        }

        logger.info(f"Initialized NLP analyzer with {self.spark_df.count():,} job records (PySpark MLlib)")

    def extract_skills_from_text(self, text_column: str = 'required_skills') -> Dict[str, Any]:
        """
        Extract and process skills from job descriptions using PySpark.

        Args:
            text_column: Column containing skills/requirements text

        Returns:
            Dictionary with skills data and summary
        """
        logger.info(f"\n=== SKILLS EXTRACTION (PySpark) FROM {text_column.upper()} ===")

        # Check if column exists
        if text_column not in self.spark_df.columns:
            logger.warning(f"Warning: {text_column} column not found.")
            available_cols = [col for col in self.spark_df.columns
                            if 'skill' in col.lower() or 'requirement' in col.lower()]
            if available_cols:
                text_column = available_cols[0]
                logger.info(f"Using column: {text_column}")
            else:
                logger.info("No suitable text column found. Creating synthetic data.")
                return self._create_synthetic_skills_data()

        # Filter non-null values
        skills_df = self.spark_df.filter(F.col(text_column).isNotNull())

        # Tokenize skills (split by comma)
        skills_df = skills_df.withColumn(
            'skills_array',
            F.split(F.lower(F.col(text_column)), ',')
        )

        # Explode array to individual skills
        skills_exploded = skills_df.select(
            F.explode('skills_array').alias('skill')
        )

        # Clean skills
        skills_exploded = skills_exploded.withColumn(
            'skill',
            F.trim(F.col('skill'))
        ).filter(
            (F.col('skill') != '') &
            (F.length('skill') > 2) &
            (F.length('skill') < 50)
        )

        # Count skill frequencies
        skill_counts = skills_exploded.groupBy('skill').count().orderBy(F.desc('count'))

        # Convert to Pandas for analysis
        skill_summary_pd = skill_counts.toPandas().rename(columns={'count': 'frequency'})

        logger.info(f"[OK] Extracted {len(skill_summary_pd):,} unique skills")
        logger.info(f"   Most common: {skill_summary_pd.iloc[0]['skill']} ({skill_summary_pd.iloc[0]['frequency']:,} occurrences)")

        self.skills_data = {
            'skill_summary': skill_summary_pd,
            'spark_df': skills_exploded,
            'total_skills': skill_counts.count()
        }

        return self.skills_data

    def cluster_skills_by_topic(self, n_clusters: int = 5, use_tfidf: bool = True) -> Dict[str, Any]:
        """
        Cluster skills into topic groups using PySpark KMeans with TF-IDF.

        Args:
            n_clusters: Number of skill clusters to create
            use_tfidf: Whether to use TF-IDF (True) or Word2Vec (False)

        Returns:
            Dictionary with clustering results
        """
        logger.info(f"\n=== SKILLS CLUSTERING (PySpark KMeans) INTO {n_clusters} TOPICS ===")

        if self.skills_data is None:
            self.extract_skills_from_text()

        skills_df = self.skills_data['spark_df']

        # Create documents (each skill as a document for clustering)
        # Group skills into documents by combining them
        docs_df = skills_df.groupBy('skill').agg(
            F.collect_list('skill').alias('words'),
            F.count('*').alias('frequency')
        )

        if use_tfidf:
            # Use TF-IDF vectorization
            logger.info("Using TF-IDF vectorization...")

            # Convert skill to array of words (tokenize)
            tokenizer = Tokenizer(inputCol='skill', outputCol='words_tokenized')
            docs_df = docs_df.withColumn('words_tokenized', F.split('skill', ' '))

            # Remove stop words
            remover = StopWordsRemover(inputCol='words_tokenized', outputCol='filtered_words')

            # Hash features
            hashingTF = HashingTF(inputCol='filtered_words', outputCol='raw_features', numFeatures=100)

            # IDF
            idf = IDF(inputCol='raw_features', outputCol='features')

            # Pipeline
            pipeline = Pipeline(stages=[remover, hashingTF, idf])

            # Fit and transform
            vectorized_df = pipeline.fit(docs_df).transform(docs_df)

        else:
            # Use Word2Vec
            logger.info("Using Word2Vec embeddings...")

            # Word2Vec requires tokenized text
            word2vec = Word2Vec(
                inputCol='words',
                outputCol='features',
                vectorSize=100,
                minCount=2
            )

            vectorized_df = word2vec.fit(docs_df).transform(docs_df)
            self.word2vec_model = word2vec

        # KMeans clustering
        kmeans = SparkKMeans(k=n_clusters, seed=42, featuresCol='features', predictionCol='cluster')

        logger.info(f"Training KMeans with {n_clusters} clusters...")
        kmeans_model = kmeans.fit(vectorized_df)

        # Get predictions
        predictions = kmeans_model.transform(vectorized_df)

        # Get top skills per cluster
        cluster_summaries = []
        for cluster_id in range(n_clusters):
            cluster_skills = predictions.filter(F.col('cluster') == cluster_id)\
                                       .select('skill', 'frequency')\
                                       .orderBy(F.desc('frequency'))\
                                       .limit(10)\
                                       .toPandas()

            top_skills = cluster_skills['skill'].tolist()
            total_freq = int(cluster_skills['frequency'].sum())

            cluster_summaries.append({
                'cluster_id': cluster_id,
                'top_skills': top_skills[:5],
                'num_skills': len(cluster_skills),
                'total_frequency': total_freq
            })

        self.clusters = cluster_summaries

        logger.info(f"[OK] Created {len(cluster_summaries)} skill clusters")
        for i, cluster in enumerate(cluster_summaries[:3]):
            logger.info(f"   Cluster {i}: {', '.join(cluster['top_skills'][:3])}...")

        return {
            'clusters': cluster_summaries,
            'n_clusters': n_clusters,
            'model': 'PySpark KMeans with TF-IDF' if use_tfidf else 'PySpark KMeans with Word2Vec'
        }

    def analyze_skill_salary_correlation(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze correlation between skills and salary using PySpark.

        Args:
            top_n: Number of top skills to analyze

        Returns:
            DataFrame with skill-salary correlations
        """
        logger.info(f"\n=== SKILL-SALARY CORRELATION ANALYSIS (PySpark) ===")

        if self.skills_data is None:
            self.extract_skills_from_text()

        # Get salary column
        from src.config.column_mapping import get_analysis_column
        salary_col = get_analysis_column('salary')

        # Check if salary column exists
        if salary_col not in self.spark_df.columns:
            logger.warning(f"Warning: Salary column {salary_col} not found")
            return pd.DataFrame()

        # Filter valid salaries and skills
        df_with_salary = self.spark_df.filter(
            (F.col(salary_col).isNotNull()) &
            (F.col(salary_col) > 0) &
            (F.col('required_skills').isNotNull())
        )

        # Explode skills
        df_exploded = df_with_salary.withColumn(
            'skill',
            F.explode(F.split(F.lower(F.col('required_skills')), ','))
        ).withColumn(
            'skill',
            F.trim('skill')
        ).filter(
            (F.col('skill') != '') &
            (F.length('skill') > 2)
        )

        # Calculate average salary per skill
        skill_salary = df_exploded.groupBy('skill').agg(
            F.avg(salary_col).alias('avg_salary'),
            F.count('*').alias('job_count'),
            F.stddev(salary_col).alias('salary_std')
        ).filter(
            F.col('job_count') >= 5  # Minimum frequency
        ).orderBy(
            F.desc('avg_salary')
        ).limit(top_n)

        # Convert to Pandas
        correlation_df = skill_salary.toPandas()

        logger.info(f"[OK] Analyzed salary correlation for {len(correlation_df)} skills")
        if len(correlation_df) > 0:
            top_skill = correlation_df.iloc[0]
            logger.info(f"   Highest paying skill: {top_skill['skill']} (${top_skill['avg_salary']:,.0f})")

        return correlation_df

    def create_word_cloud(self, max_words: int = 100) -> Optional[go.Figure]:
        """
        Create word cloud visualization from skills.

        Note: Still uses wordcloud library (not PySpark) for visualization only.
        """
        if not WORDCLOUD_AVAILABLE:
            logger.info("WordCloud library not available")
            return None

        logger.info(f"\n=== GENERATING WORD CLOUD ===")

        if self.skills_data is None:
            self.extract_skills_from_text()

        # Get skill frequencies from Pandas summary
        skill_summary = self.skills_data['skill_summary']

        # Create word frequency dict
        word_freq = dict(zip(skill_summary['skill'], skill_summary['frequency']))

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words
        ).generate_from_frequencies(word_freq)

        # Convert to Plotly figure
        fig = go.Figure()

        # Convert wordcloud to image
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close()

        # Create Plotly figure with image
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_b64}',
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="below"
            )
        )

        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_layout(
            title='Skills Word Cloud',
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        logger.info(f"[OK] Generated word cloud with {len(word_freq)} skills")

        return fig

    def run_complete_nlp_analysis(self) -> Dict[str, Any]:
        """
        Run complete NLP analysis pipeline using PySpark MLlib.

        Returns comprehensive analysis with all NLP results.
        """
        logger.info("\n" + "="*70)
        logger.info("[START] RUNNING COMPLETE NLP ANALYSIS (PySpark MLlib)")
        logger.info("="*70)

        # Extract skills
        skills_data = self.extract_skills_from_text()

        # Cluster skills
        clusters = self.cluster_skills_by_topic(n_clusters=5)

        # Analyze salary correlation
        salary_corr = self.analyze_skill_salary_correlation(top_n=20)

        # Create word cloud if available
        word_cloud = self.create_word_cloud() if WORDCLOUD_AVAILABLE else None

        results = {
            'skills_extracted': len(skills_data['skill_summary']),
            'clusters': clusters,
            'salary_correlation': salary_corr,
            'word_cloud': word_cloud,
            'technology': 'PySpark MLlib (TF-IDF + KMeans)',
            'top_skills': skills_data['skill_summary'].head(10).to_dict('records')
        }

        logger.info("\n" + "="*70)
        logger.info("[OK] COMPLETE NLP ANALYSIS FINISHED")
        logger.info("="*70)
        logger.info(f"Extracted {results['skills_extracted']} unique skills")
        logger.info(f"Created {clusters['n_clusters']} skill clusters")
        logger.info(f"Analyzed top {len(salary_corr)} high-paying skills")

        return results

    def _create_synthetic_skills_data(self) -> Dict[str, Any]:
        """Create synthetic skills data for testing."""
        logger.info("Creating synthetic skills data...")

        all_skills = []
        for category_skills in self.tech_skills.values():
            all_skills.extend(category_skills)

        # Create frequency distribution
        np.random.seed(42)
        frequencies = np.random.zipf(1.5, len(all_skills))

        skill_summary = pd.DataFrame({
            'skill': all_skills,
            'frequency': frequencies
        }).sort_values('frequency', ascending=False)

        return {
            'skill_summary': skill_summary,
            'total_skills': len(all_skills)
        }

    def __del__(self):
        """Clean up Spark session."""
        if hasattr(self, 'spark'):
            try:
                self.spark.stop()
            except:
                pass


# Convenience function
def run_nlp_analysis(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Run complete NLP analysis using PySpark MLlib.

    Args:
        df: Optional Pandas DataFrame with job market data

    Returns:
        Dictionary with complete NLP analysis results
    """
    analyzer = JobMarketNLPAnalyzer(df=df)
    results = analyzer.run_complete_nlp_analysis()
    return results
