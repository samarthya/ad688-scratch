#!/usr/bin/env python3
"""
Package Integration Test

This script tests the refactored package structure to ensure all modules
can be imported correctly and basic functionality works as expected.

Run this test after refactoring to verify the package is working properly.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_package_imports():
    """Test that all main package components can be imported."""
    print("Testing Package Imports...")
    
    try:
        # Test main package import
        import src
        print("   SUCCESS: Main package import: SUCCESS")
        
        # Test data module imports
        from src.data import JobMarketDataProcessor, SparkJobAnalyzer
        print("   SUCCESS: Data module imports: SUCCESS")
        
        # Test visualization module imports
        from src.visualization import QuartoChartExporter, SalaryDisparityChartConfig
        print("   SUCCESS: Visualization module imports: SUCCESS")
        
        # Test utilities module imports
        from src.utilities.get_stats import JobMarketStatistics
        print("   SUCCESS: Utilities module imports: SUCCESS")
        
        # Test config module imports
        import src.config.column_mapping
        print("   ✅ Config module imports: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_class_initialization():
    """Test that main classes can be initialized without errors."""
    print("\n🏗️ Testing Class Initialization...")
    
    try:
        # Test JobMarketDataProcessor
        from src.data.enhanced_processor import JobMarketDataProcessor
        processor = JobMarketDataProcessor("TestProcessor")
        print("   ✅ JobMarketDataProcessor initialization: SUCCESS")
        
        # Test QuartoChartExporter
        from src.visualization.quarto_charts import QuartoChartExporter
        chart_exporter = QuartoChartExporter(output_dir="test_figures")
        print("   ✅ QuartoChartExporter initialization: SUCCESS")
        
        # Test JobMarketStatistics
        from src.utilities.get_stats import JobMarketStatistics
        stats = JobMarketStatistics()
        print("   ✅ JobMarketStatistics initialization: SUCCESS")
        
        # Test SalaryDisparityChartConfig
        from src.visualization.chart_config import SalaryDisparityChartConfig
        layout = SalaryDisparityChartConfig.get_standard_layout()
        colors = SalaryDisparityChartConfig.get_salary_disparity_colors()
        print("   ✅ SalaryDisparityChartConfig methods: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Class initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_package_metadata():
    """Test that package metadata is properly defined."""
    print("\n📋 Testing Package Metadata...")
    
    try:
        import src
        
        # Check version
        if hasattr(src, '__version__'):
            print(f"   ✅ Package version: {src.__version__}")
        else:
            print("   ⚠️ Package version not defined")
        
        # Check author
        if hasattr(src, '__author__'):
            print(f"   ✅ Package author: {src.__author__}")
        else:
            print("   ⚠️ Package author not defined")
        
        # Check __all__ exports
        if hasattr(src, '__all__'):
            print(f"   ✅ Package exports: {len(src.__all__)} classes/functions")
            for export in src.__all__:
                print(f"      • {export}")
        else:
            print("   ⚠️ Package __all__ not defined")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Package metadata test failed: {e}")
        traceback.print_exc()
        return False

def test_documentation():
    """Test that modules have proper documentation."""
    print("\n📚 Testing Documentation...")
    
    try:
        modules_to_test = [
            'src.data.enhanced_processor',
            'src.visualization.chart_config', 
            'src.utilities.get_stats'
        ]
        
        for module_name in modules_to_test:
            module = __import__(module_name, fromlist=[''])
            if module.__doc__:
                doc_length = len(module.__doc__.strip())
                print(f"   ✅ {module_name}: {doc_length} chars of documentation")
            else:
                print(f"   ⚠️ {module_name}: No module documentation")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Documentation test failed: {e}")
        traceback.print_exc()
        return False

def test_demo_script():
    """Test that the demo script can be imported without errors."""
    print("\n🎬 Testing Demo Script...")
    
    try:
        from src.demo_class_usage import JobMarketAnalysisDemo
        demo = JobMarketAnalysisDemo()
        print("   ✅ Demo class import and initialization: SUCCESS")
        
        # Test that demo has required methods
        required_methods = [
            'setup_analysis_environment',
            'demonstrate_data_loading',
            'demonstrate_data_processing',
            'run_complete_demonstration'
        ]
        
        for method in required_methods:
            if hasattr(demo, method):
                print(f"   ✅ Demo method '{method}': PRESENT")
            else:
                print(f"   ❌ Demo method '{method}': MISSING")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Demo script test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all package tests and provide summary."""
    print("🚀 PACKAGE INTEGRATION TEST SUITE")
    print("=" * 50)
    print("Testing refactored package structure and functionality...")
    
    tests = [
        ("Package Imports", test_package_imports),
        ("Class Initialization", test_class_initialization),
        ("Package Metadata", test_package_metadata),
        ("Documentation", test_documentation),
        ("Demo Script", test_demo_script)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Package refactoring successful!")
        return True
    else:
        print("⚠️ Some tests failed. Review and fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)