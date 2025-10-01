#!/usr/bin/env python3
"""
Notebook Validation and Robustness Checker

This script analyzes all Jupyter notebooks in the project to identify potential
casting issues and data quality problems, then suggests or applies fixes.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotebookRobustnessChecker:
    """
    Analyzes Jupyter notebooks for potential data casting and quality issues.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.notebooks_dir = self.project_root / "notebooks"
        
        # Patterns that indicate potential casting issues
        self.risky_patterns = {
            'direct_cast': r'\.cast\([\'"](?:double|int|long|float)[\'"]?\)',
            'unsafe_filter': r'\.filter\([^)]*!=\s*[\'"][\'"][)]',
            'unsafe_isin': r'\.isin\([^)]*[\'"][\'"][^)]*\)',
            'no_null_check': r'\.cast\([^)]+\)(?!\s*\.filter\([^)]*isNotNull)',
            'missing_regex_validation': r'\.cast\([^)]+\)(?![^.]*\.rlike)',
            'bare_equality': r'==\s*[\'"][\'"]',
            'aggregation_without_try_catch': r'\.agg\(|\.groupBy\([^)]+\)\.count\(\)'
        }
        
        # Safe patterns to recommend
        self.safe_patterns = {
            'safe_cast_with_validation': 'when(col("column").rlike(r"^[0-9]+$"), col("column").cast("double")).otherwise(None)',
            'safe_filter': 'filter(col("column").isNotNull() & (length(col("column")) > 0))',
            'safe_null_exclusion': 'filter(col("column") != "") instead of isin(["", "null"])',
            'try_catch_aggregation': 'Use try/except blocks around aggregation operations'
        }
    
    def find_notebooks(self) -> List[Path]:
        """Find all Jupyter notebooks in the project."""
        notebooks = []
        if self.notebooks_dir.exists():
            notebooks.extend(self.notebooks_dir.glob("*.ipynb"))
        return notebooks
    
    def analyze_notebook_content(self, notebook_path: Path) -> Dict[str, Any]:
        """Analyze a single notebook for potential issues."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_content = json.load(f)
            
            analysis = {
                'path': str(notebook_path),
                'name': notebook_path.name,
                'total_cells': len(notebook_content.get('cells', [])),
                'code_cells': 0,
                'issues': [],
                'risk_score': 0,
                'recommendations': []
            }
            
            for cell_idx, cell in enumerate(notebook_content.get('cells', [])):
                if cell.get('cell_type') == 'code':
                    analysis['code_cells'] += 1
                    cell_source = ''.join(cell.get('source', []))
                    
                    # Check for risky patterns
                    cell_issues = self._analyze_cell_content(cell_source, cell_idx)
                    analysis['issues'].extend(cell_issues)
                    analysis['risk_score'] += len(cell_issues)
            
            # Generate recommendations based on issues found
            analysis['recommendations'] = self._generate_recommendations(analysis['issues'])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {notebook_path}: {e}")
            return {
                'path': str(notebook_path),
                'name': notebook_path.name,
                'error': str(e),
                'risk_score': 100  # Max risk for unreadable notebooks
            }
    
    def _analyze_cell_content(self, cell_source: str, cell_idx: int) -> List[Dict[str, Any]]:
        """Analyze individual cell content for issues."""
        issues = []
        
        for pattern_name, pattern in self.risky_patterns.items():
            matches = re.finditer(pattern, cell_source, re.IGNORECASE)
            for match in matches:
                issues.append({
                    'cell_index': cell_idx,
                    'issue_type': pattern_name,
                    'pattern': pattern,
                    'match': match.group(),
                    'line_start': cell_source[:match.start()].count('\n') + 1,
                    'severity': self._get_severity(pattern_name)
                })
        
        # Check for specific Spark/PySpark patterns
        if 'pyspark' in cell_source.lower() or 'from pyspark' in cell_source:
            issues.extend(self._check_spark_specific_issues(cell_source, cell_idx))
        
        return issues
    
    def _check_spark_specific_issues(self, cell_source: str, cell_idx: int) -> List[Dict[str, Any]]:
        """Check for Spark-specific casting and data quality issues."""
        spark_issues = []
        
        # Check for missing try_cast usage (if available)
        if '.cast(' in cell_source and 'try_cast' not in cell_source:
            spark_issues.append({
                'cell_index': cell_idx,
                'issue_type': 'missing_try_cast',
                'severity': 'medium',
                'description': 'Using cast() without try_cast() safety check'
            })
        
        # Check for DataFrame operations without error handling
        risky_operations = ['groupBy', 'agg', 'filter', 'join']
        for op in risky_operations:
            if f'.{op}(' in cell_source:
                # Check if it's in a try/except block
                if 'try:' not in cell_source or 'except' not in cell_source:
                    spark_issues.append({
                        'cell_index': cell_idx,
                        'issue_type': 'unprotected_operation',
                        'operation': op,
                        'severity': 'low',
                        'description': f'DataFrame.{op}() operation without error handling'
                    })
        
        return spark_issues
    
    def _get_severity(self, pattern_name: str) -> str:
        """Determine severity level for different issue types."""
        high_risk = ['direct_cast', 'unsafe_isin', 'no_null_check']
        medium_risk = ['unsafe_filter', 'missing_regex_validation']
        
        if pattern_name in high_risk:
            return 'high'
        elif pattern_name in medium_risk:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate specific recommendations based on found issues."""
        recommendations = []
        
        issue_types = set(issue.get('issue_type', '') for issue in issues)
        
        if 'direct_cast' in issue_types:
            recommendations.append(
                "Replace direct .cast() calls with safe casting using when() and rlike() validation"
            )
        
        if 'unsafe_isin' in issue_types:
            recommendations.append(
                "Replace .isin(['']) calls with length-based filtering to avoid casting issues"
            )
        
        if 'no_null_check' in issue_types:
            recommendations.append(
                "Add null checks before casting operations using .filter(col().isNotNull())"
            )
        
        if 'unprotected_operation' in issue_types:
            recommendations.append(
                "Wrap DataFrame operations in try/except blocks for robust error handling"
            )
        
        if not recommendations:
            recommendations.append("No major issues found - notebook appears robust")
        
        return recommendations
    
    def generate_robustness_report(self) -> Dict[str, Any]:
        """Generate a comprehensive robustness report for all notebooks."""
        notebooks = self.find_notebooks()
        
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'total_notebooks': len(notebooks),
            'notebooks_analyzed': 0,
            'notebooks_with_issues': 0,
            'total_issues': 0,
            'high_risk_notebooks': [],
            'medium_risk_notebooks': [],
            'low_risk_notebooks': [],
            'clean_notebooks': [],
            'detailed_analysis': []
        }
        
        for notebook_path in notebooks:
            analysis = self.analyze_notebook_content(notebook_path)
            report['detailed_analysis'].append(analysis)
            
            if 'error' not in analysis:
                report['notebooks_analyzed'] += 1
                report['total_issues'] += len(analysis['issues'])
                
                if analysis['issues']:
                    report['notebooks_with_issues'] += 1
                
                # Categorize by risk level
                risk_score = analysis['risk_score']
                if risk_score >= 10:
                    report['high_risk_notebooks'].append(notebook_path.name)
                elif risk_score >= 5:
                    report['medium_risk_notebooks'].append(notebook_path.name)
                elif risk_score > 0:
                    report['low_risk_notebooks'].append(notebook_path.name)
                else:
                    report['clean_notebooks'].append(notebook_path.name)
        
        return report
    
    def create_fix_suggestions(self, notebook_path: Path) -> List[Dict[str, Any]]:
        """Create specific fix suggestions for a notebook."""
        analysis = self.analyze_notebook_content(notebook_path)
        fixes = []
        
        for issue in analysis.get('issues', []):
            fix_suggestion = self._create_fix_for_issue(issue)
            if fix_suggestion:
                fixes.append(fix_suggestion)
        
        return fixes
    
    def _create_fix_for_issue(self, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create specific fix suggestion for an individual issue."""
        issue_type = issue.get('issue_type')
        
        if issue_type == 'direct_cast':
            return {
                'issue': issue,
                'fix_type': 'replace_code',
                'original_pattern': issue.get('match', ''),
                'suggested_replacement': 'when(col("column").rlike(r"^[0-9]+$"), col("column").cast("double")).otherwise(None)',
                'description': 'Replace direct casting with safe casting using validation'
            }
        
        elif issue_type == 'unsafe_isin':
            return {
                'issue': issue,
                'fix_type': 'replace_code',
                'original_pattern': issue.get('match', ''),
                'suggested_replacement': 'filter(col("column").isNotNull() & (length(col("column")) > 0))',
                'description': 'Replace unsafe isin() with length-based filtering'
            }
        
        elif issue_type == 'unprotected_operation':
            return {
                'issue': issue,
                'fix_type': 'wrap_in_try_catch',
                'description': 'Wrap operation in try/except block with fallback handling'
            }
        
        return None


def main():
    """Main function to run notebook robustness analysis."""
    # Get project root (assuming script is in src/utils)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    print(f"🔍 Analyzing notebooks in: {project_root}")
    
    checker = NotebookRobustnessChecker(str(project_root))
    report = checker.generate_robustness_report()
    
    # Print summary
    print(f"\n📊 NOTEBOOK ROBUSTNESS REPORT")
    print(f"=" * 50)
    print(f"Total notebooks found: {report['total_notebooks']}")
    print(f"Notebooks analyzed: {report['notebooks_analyzed']}")
    print(f"Notebooks with issues: {report['notebooks_with_issues']}")
    print(f"Total issues found: {report['total_issues']}")
    
    print(f"\n🚨 HIGH RISK NOTEBOOKS: {len(report['high_risk_notebooks'])}")
    for notebook in report['high_risk_notebooks']:
        print(f"  - {notebook}")
    
    print(f"\n⚠️  MEDIUM RISK NOTEBOOKS: {len(report['medium_risk_notebooks'])}")
    for notebook in report['medium_risk_notebooks']:
        print(f"  - {notebook}")
    
    print(f"\n✅ CLEAN NOTEBOOKS: {len(report['clean_notebooks'])}")
    for notebook in report['clean_notebooks']:
        print(f"  - {notebook}")
    
    # Detailed analysis for high-risk notebooks
    print(f"\n🔧 DETAILED RECOMMENDATIONS:")
    print(f"=" * 50)
    
    for analysis in report['detailed_analysis']:
        if analysis.get('risk_score', 0) >= 5:
            print(f"\n📓 {analysis['name']}:")
            print(f"   Risk Score: {analysis.get('risk_score', 0)}")
            print(f"   Issues: {len(analysis.get('issues', []))}")
            
            for rec in analysis.get('recommendations', []):
                print(f"   • {rec}")
    
    # Save detailed report
    report_path = project_root / 'notebook_robustness_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    return report


if __name__ == "__main__":
    import pandas as pd  # For timestamp
    main()