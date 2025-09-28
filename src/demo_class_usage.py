"""
Demonstration of Class-Based Job Market Analysis Architecture

This script showcases how to use our custom classes together to perform
comprehensive job market analysis without reinventing the wheel.

Classes Used:
- SparkJobAnalyzer: SQL-based analysis engine
- JobMarketDataProcessor: Data processing and cleaning
- SalaryVisualizer: Visualization utilities

Author: Job Market Analysis Team
Date: September 27, 2025
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

# Import our custom classes
from data.spark_analyzer import SparkJobAnalyzer, create_spark_analyzer
from data.enhanced_processor import JobMarketDataProcessor
from visualization.simple_plots import SalaryVisualizer

def demonstrate_class_usage():
    """
    Demonstrate integrated usage of all our analysis classes.
    
    This function shows the proper way to use our class-based architecture
    for job market analysis instead of writing functions from scratch.
    """
    
    print("🚀 Job Market Analysis Class Demonstration")
    print("=" * 50)
    
    # 1. Initialize SparkJobAnalyzer (handles Spark session and data loading)
    print("\n1. 📊 Initializing SparkJobAnalyzer...")
    analyzer = create_spark_analyzer()
    
    if analyzer.job_data is not None:
        record_count = analyzer.job_data.count()
        print(f"   ✅ Analyzer ready with {record_count:,} records")
    else:
        print("   ⚠️ No data loaded - using sample data")
    
    # 2. Use JobMarketDataProcessor for advanced processing
    print("\n2. 🔧 Using JobMarketDataProcessor...")
    processor = JobMarketDataProcessor("ClassDemo")
    processor.df_raw = analyzer.job_data  # Share data between classes
    print("   ✅ Processor initialized with shared data")
    
    # 3. Get comprehensive analysis using SparkJobAnalyzer methods
    print("\n3. 📈 Running SparkJobAnalyzer Analysis Methods...")
    
    # Industry analysis
    industry_results = analyzer.get_industry_analysis(top_n=5)
    print(f"   📍 Top 5 Industries by Median Salary:")
    print(industry_results.to_string(index=False))
    
    # Skills analysis
    skills_results = analyzer.get_skills_analysis(top_n=5)
    print(f"\n   💻 Top 5 Skills by Premium:")
    print(skills_results.to_string(index=False))
    
    # Overall statistics
    stats = analyzer.get_overall_statistics()
    print(f"\n   📊 Overall Statistics:")
    print(f"     Total Jobs: {stats['total_jobs']:,}")
    print(f"     Median Salary: ${stats['median_salary']:,}")
    print(f"     Salary Range: ${stats['min_salary']:,} - ${stats['max_salary']:,}")
    
    # 4. Use SalaryVisualizer for pandas-based analysis
    print("\n4. 📊 Using SalaryVisualizer with Sample Data...")
    
    # Convert sample to pandas for SalaryVisualizer
    if analyzer.job_data is not None:
        sample_df = analyzer.job_data.sample(fraction=0.1).toPandas()
    else:
        sample_df = pd.DataFrame()
    
    if len(sample_df) > 0:
        visualizer = SalaryVisualizer(sample_df)
        
        # Get various analyses
        try:
            viz_stats = visualizer.get_overall_statistics()
            print(f"     SalaryVisualizer - Total Jobs: {viz_stats['total_jobs']:,}")
            print(f"     SalaryVisualizer - Median Salary: ${viz_stats['median_salary']:,}")
        except Exception as e:
            print(f"     ⚠️ SalaryVisualizer analysis: {e}")
    
    # 5. Export results using class methods
    print("\n5. 💾 Exporting Results...")
    
    output_dir = Path("data/processed/class_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analysis results
    industry_results.to_csv(output_dir / "industry_analysis.csv", index=False)
    skills_results.to_csv(output_dir / "skills_analysis.csv", index=False)
    
    # Save stats as JSON
    import json
    with open(output_dir / "overall_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"     ✅ Results exported to {output_dir}")
    
    # 6. Demonstrate class integration benefits
    print("\n6. 🎯 Class Integration Benefits:")
    print("   ✅ No duplicate code - single source of truth for analysis methods")
    print("   ✅ Consistent interfaces - same methods across Quarto docs and notebooks") 
    print("   ✅ Error handling - graceful fallbacks built into classes")
    print("   ✅ Scalability - PySpark backend handles large datasets")
    print("   ✅ Maintainability - easy to add new methods to existing classes")
    
    # Clean up
    print("\n7. 🧹 Cleanup...")
    analyzer.stop()
    print("   ✅ Spark session stopped")
    
    return {
        'industry_results': industry_results,
        'skills_results': skills_results,
        'overall_stats': stats
    }

def show_class_relationships():
    """Show the relationships between our classes."""
    
    print("\n🏗️ Class Architecture Overview")
    print("=" * 40)
    print("""
    JobMarketDataProcessor
    ├── Handles comprehensive data processing
    ├── Data quality assessment 
    └── Feature engineering
    
    SparkJobAnalyzer  
    ├── SQL-based analysis engine
    ├── Industry/skills/geographic analysis
    └── Replaces manual DataFrame operations
    
    SalaryVisualizer
    ├── Pandas-based visualization utilities
    ├── Backward compatibility with existing code
    └── Graceful fallbacks for missing data
    
    Integration Pattern:
    Data Loading → Processing → Analysis → Visualization → Export
    """)

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_class_usage()
    
    # Show architecture overview
    show_class_relationships()
    
    print("\n🎉 Class demonstration complete!")
    print("📚 See docs/class_architecture.md for complete UML diagram")
    print("🔧 Use these classes in your Quarto documents and notebooks")