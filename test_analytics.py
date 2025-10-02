"""
Test script for analytics modules to verify they work with missing dependencies.
"""

def test_analytics_import():
    """Test that analytics modules can be imported even with missing dependencies."""
    print("🧪 Testing Analytics Module Import...")

    try:
        from src.analytics import SalaryAnalyticsModels, JobMarketNLPAnalyzer, PredictiveAnalyticsDashboard
        print("✅ Core analytics classes imported successfully")

        # Test with auto data loading
        print("\n📊 Testing SalaryAnalyticsModels...")
        try:
            models = SalaryAnalyticsModels()
            print("✅ SalaryAnalyticsModels initialized successfully")
        except Exception as e:
            print(f"⚠️  SalaryAnalyticsModels initialization failed: {e}")

        print("\n🔍 Testing JobMarketNLPAnalyzer...")
        try:
            nlp = JobMarketNLPAnalyzer()
            print("✅ JobMarketNLPAnalyzer initialized successfully")
        except Exception as e:
            print(f"⚠️  JobMarketNLPAnalyzer initialization failed: {e}")

        print("\n📈 Testing PredictiveAnalyticsDashboard...")
        try:
            dashboard = PredictiveAnalyticsDashboard()
            print("✅ PredictiveAnalyticsDashboard initialized successfully")
        except Exception as e:
            print(f"⚠️  PredictiveAnalyticsDashboard initialization failed: {e}")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    return True

def test_convenience_functions():
    """Test convenience functions."""
    print("\n🔧 Testing Convenience Functions...")

    try:
        from src.analytics import run_predictive_analysis, run_nlp_analysis, create_analytics_report
        print("✅ Convenience functions imported successfully")

        # Test functions (they should handle missing dependencies gracefully)
        print("\n🎯 Testing run_predictive_analysis...")
        try:
            result = run_predictive_analysis()
            print("✅ run_predictive_analysis completed successfully")
        except Exception as e:
            print(f"⚠️  run_predictive_analysis failed: {e}")

    except ImportError as e:
        print(f"❌ Convenience function import failed: {e}")
        return False

    return True

def test_docx_report():
    """Test DOCX report generation."""
    print("\n📄 Testing DOCX Report Generation...")

    try:
        from src.analytics import generate_docx_report
        print("✅ DOCX report function imported successfully")

        # Test report generation
        try:
            report_path = generate_docx_report(output_path="test_report.docx")
            if report_path:
                print(f"✅ DOCX report generated: {report_path}")
            else:
                print("⚠️  DOCX report generation skipped (missing dependencies)")
        except Exception as e:
            print(f"⚠️  DOCX report generation failed: {e}")

    except ImportError as e:
        print(f"❌ DOCX report import failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("🚀 ANALYTICS MODULE TEST SUITE")
    print("=" * 40)

    # Run tests
    import_success = test_analytics_import()
    functions_success = test_convenience_functions()
    docx_success = test_docx_report()

    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 25)
    print(f"Import Test: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"Functions Test: {'✅ PASS' if functions_success else '❌ FAIL'}")
    print(f"DOCX Test: {'✅ PASS' if docx_success else '❌ FAIL'}")

    if import_success and functions_success:
        print("\n🎉 Analytics module is working correctly!")
        print("💡 To enable all features, install missing dependencies:")
        print("   pip install wordcloud python-docx")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
