"""
Centralized Visualization Export Module - Salary Disparity Focus

This module provides standardized chart generation and export functionality
for consistent integration with Quarto documents. All charts focus on salary
disparity analysis and are optimized for readability.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from .chart_config import SalaryDisparityChartConfig

class QuartoChartExporter:
    """
    Salary Disparity-Focused Chart Generation for Quarto Integration.
    
    All charts emphasize salary disparities across experience, company size,
    education, and geographic factors. Charts are optimized for readability
    and professional presentation.
    """
    
    def __init__(self, output_dir="figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chart_registry = []
        
    def create_experience_salary_chart(self, data, title="Salary Disparity by Experience Level"):
        """Create salary disparity chart showing experience-based compensation gaps"""
        
        # Use the new standardized configuration
        fig = SalaryDisparityChartConfig.create_readable_bar_chart(
            data, 
            x_col='Experience Level', 
            y_col='Median Salary',
            title=title
        )
        
        # Add disparity-focused annotations
        if len(data) >= 2:
            min_salary = data['Median Salary'].min()
            max_salary = data['Median Salary'].max()
            disparity_ratio = max_salary / min_salary if min_salary > 0 else 0
            
            fig.add_annotation(
                text=f"<b>Experience Gap:</b> {disparity_ratio:.1f}x salary difference<br>Entry to Senior level",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98, xanchor='left', yanchor='top',
                font=dict(size=14, color='#2E2E2E'),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#8E44AD",
                borderwidth=2
            )
        
        return self._export_chart(fig, "experience_salary_disparity")
    
    def create_industry_salary_chart(self, data, title="Industry Salary Comparison"):
        """Create standardized industry salary comparison chart"""
        
        # Sort by median salary for better presentation
        data_sorted = data.sort_values('Median Salary', ascending=True)
        
        fig = px.bar(
            data_sorted,
            x='Median Salary',
            y='Industry', 
            orientation='h',
            title=title,
            labels={'Median Salary': 'Median Salary ($)'}
        )
        
        fig.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            height=600,
            showlegend=False
        )
        
        fig.update_layout(xaxis_tickformat="$,.0f")
        
        return self._export_chart(fig, "industry_salary_comparison")
    
    def create_location_salary_chart(self, data, title="Geographic Salary Analysis"):
        """Create standardized location salary analysis chart"""
        
        fig = px.scatter(
            data,
            x='Job Count',
            y='Median Salary', 
            size='Job Count',
            hover_name='Location',
            title=title,
            labels={
                'Job Count': 'Number of Job Postings',
                'Median Salary': 'Median Salary ($)'
            }
        )
        
        fig.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            showlegend=False
        )
        
        fig.update_layout(yaxis_tickformat="$,.0f")
        
        return self._export_chart(fig, "location_salary_analysis")
    
    def create_ai_premium_chart(self, data, title="AI Skills Salary Premium"):
        """Create AI skills premium analysis chart"""
        
        fig = px.bar(
            data,
            x='Industry',
            y='AI Premium',
            title=title,
            labels={'AI Premium': 'AI Premium (%)'}
        )
        
        fig.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            xaxis_tickangle=45,
            showlegend=False
        )
        
        fig.update_layout(yaxis_tickformat=".1f")
        
        return self._export_chart(fig, "ai_skills_premium")
    
    def create_correlation_heatmap(self, data, title="Feature Correlation Matrix"):
        """Create correlation heatmap using matplotlib for consistency"""
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        # Save matplotlib figure
        chart_name = "correlation_heatmap"
        png_path = self.output_dir / f"{chart_name}.png"
        svg_path = self.output_dir / f"{chart_name}.svg"
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        plt.close()
        
        chart_info = {
            'name': chart_name,
            'title': title,
            'type': 'matplotlib',
            'files': {
                'png': str(png_path),
                'svg': str(svg_path)
            }
        }
        
        self.chart_registry.append(chart_info)
        return chart_info
    
    def _export_chart(self, fig, chart_name):
        """Export plotly chart in multiple formats"""
        
        # Export as HTML (for interactive use)
        html_path = self.output_dir / f"{chart_name}.html"
        fig.write_html(html_path)
        
        # Export as PNG (for documents)
        png_path = self.output_dir / f"{chart_name}.png"
        fig.write_image(png_path, width=1200, height=800, scale=2)
        
        # Export as SVG (for scalable graphics)
        svg_path = self.output_dir / f"{chart_name}.svg"
        fig.write_image(svg_path, width=1200, height=800)
        
        chart_info = {
            'name': chart_name,
            'title': fig.layout.title.text,
            'type': 'plotly',
            'files': {
                'html': str(html_path),
                'png': str(png_path),
                'svg': str(svg_path)
            }
        }
        
        self.chart_registry.append(chart_info)
        return chart_info
    
    def export_chart_registry(self):
        """Export complete chart registry for Quarto reference"""
        
        registry_path = self.output_dir / "chart_registry.json"
        with open(registry_path, "w") as f:
            json.dump(self.chart_registry, f, indent=2)
        
        print(f"Chart registry exported to: {registry_path}")
        print(f"Total charts generated: {len(self.chart_registry)}")
        
        return registry_path
    
    def get_chart_paths(self, chart_name):
        """Get file paths for a specific chart"""
        
        for chart in self.chart_registry:
            if chart['name'] == chart_name:
                return chart['files']
        
        return None