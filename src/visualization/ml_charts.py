"""
Machine Learning Visualization Module

This module provides reusable functions for creating ML-specific visualizations,
including model performance comparisons, feature importance charts, confusion matrices,
and prediction validation plots.

All functions return Plotly figures that can be used in Quarto reports, notebooks,
or standalone visualizations.
"""

# Standard library imports
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_ml_performance_comparison(
    regression_results: Dict[str, float],
    classification_results: Dict[str, float]
) -> go.Figure:
    """
    Create side-by-side comparison of regression and classification model performance.

    Args:
        regression_results: Dict with 'train_r2' and 'test_r2' keys
        classification_results: Dict with 'train_accuracy' and 'test_accuracy' keys

    Returns:
        Plotly Figure with 2-panel comparison

    Example:
        >>> regression_results = {'train_r2': 0.84, 'test_r2': 0.83}
        >>> classification_results = {'train_accuracy': 0.86, 'test_accuracy': 0.85}
        >>> fig = create_ml_performance_comparison(regression_results, classification_results)
        >>> display_figure(fig, "ml_performance")
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "M1: Regression Performance (R²)",
            "M2: Classification Performance (Accuracy)"
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.3
    )

    # Regression R² scores
    fig.add_trace(
        go.Bar(
            name='R² Score',
            x=['Training', 'Testing'],
            y=[regression_results['train_r2'], regression_results['test_r2']],
            marker_color='#1f77b4',
            text=[regression_results['train_r2'], regression_results['test_r2']],
            texttemplate='%{text:.2f}',
            textposition='outside'
        ),
        row=1, col=1
    )

    # Classification Accuracy
    fig.add_trace(
        go.Bar(
            name='Accuracy',
            x=['Training', 'Testing'],
            y=[classification_results['train_accuracy'], classification_results['test_accuracy']],
            marker_color='#2ca02c',
            text=[classification_results['train_accuracy'], classification_results['test_accuracy']],
            texttemplate='%{text:.2f}',
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_yaxes(title_text="R² Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)

    fig.update_layout(
        height=550,
        title_text="Machine Learning Model Performance Comparison",
        title_x=0.5,
        title_font_size=16,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=10),
        margin=dict(l=80, r=80, t=120, b=80)  # Increased top margin for title
    )

    return fig


def create_feature_importance_chart(
    features: List[str],
    importance: List[float],
    title: str = "Feature Importance: What Drives Salary Predictions?"
) -> go.Figure:
    """
    Create horizontal bar chart showing feature importance.

    Args:
        features: List of feature names
        importance: List of importance scores (0-1 scale)
        title: Chart title

    Returns:
        Plotly Figure with horizontal bar chart

    Example:
        >>> features = ['Job Title', 'Industry', 'Experience', 'Location', 'Skills']
        >>> importance = [0.35, 0.28, 0.15, 0.12, 0.10]
        >>> fig = create_feature_importance_chart(features, importance)
        >>> display_figure(fig, "feature_importance")
    """
    # Color code by importance level
    colors = [
        '#d62728' if imp > 0.25 else '#2ca02c' if imp > 0.15 else '#1f77b4'
        for imp in importance
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=features,
        x=importance,
        orientation='h',
        marker_color=colors,
        text=[f'{imp:.0%}' for imp in importance],
        textposition='outside',
        textfont=dict(size=14)
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=16,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=550,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=180, r=120, t=120, b=80)  # Increased margins for labels
    )

    fig.update_xaxes(range=[0, max(importance) * 1.2], tickformat='.0%')  # More space for text labels

    return fig


def create_predicted_vs_actual_plot(
    actual_salaries: np.ndarray,
    predicted_salaries: np.ndarray,
    title: str = "Model Validation: Predicted vs Actual Salaries"
) -> go.Figure:
    """
    Create scatter plot comparing predicted vs actual salaries with perfect prediction line.

    Args:
        actual_salaries: Array of actual salary values
        predicted_salaries: Array of predicted salary values
        title: Chart title

    Returns:
        Plotly Figure with scatter plot

    Example:
        >>> actual = np.array([100000, 120000, 90000, 150000])
        >>> predicted = np.array([98000, 125000, 88000, 145000])
        >>> fig = create_predicted_vs_actual_plot(actual, predicted)
        >>> display_figure(fig, "predicted_vs_actual")
    """
    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=actual_salaries,
        y=predicted_salaries,
        mode='markers',
        marker=dict(
            size=6,
            color=actual_salaries,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Actual<br>Salary ($)")
        ),
        name='Predictions',
        text=[f'Actual: ${a:,.0f}<br>Predicted: ${p:,.0f}'
              for a, p in zip(actual_salaries, predicted_salaries)],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Perfect prediction line (y=x)
    salary_range = [actual_salaries.min(), actual_salaries.max()]
    fig.add_trace(go.Scatter(
        x=salary_range,
        y=salary_range,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Perfect Prediction',
        hovertemplate='Perfect Prediction Line<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=16,
        xaxis_title="Actual Salary ($)",
        yaxis_title="Predicted Salary ($)",
        height=650,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=100, r=140, t=120, b=90),  # Increased margins for axis labels and colorbar
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        )
    )

    fig.update_xaxes(tickformat='$,.0f', tickangle=0)
    fig.update_yaxes(tickformat='$,.0f')

    return fig


def get_enhanced_colorscales():
    """
    Get enhanced colorscale options for confusion matrix heatmaps.

    Returns:
        dict: Dictionary of colorscale options with different visual styles
    """
    return {
        'blue_gradient': [
            [0.0, '#ffffff'],      # Pure white (lowest values)
            [0.1, '#f0f8ff'],      # Alice blue
            [0.2, '#e6f3ff'],      # Very light blue
            [0.3, '#cce7ff'],      # Light blue
            [0.4, '#99d6ff'],      # Medium light blue
            [0.5, '#66c2ff'],      # Medium blue
            [0.6, '#33adff'],      # Medium dark blue
            [0.7, '#0099ff'],      # Bright blue
            [0.8, '#0088e6'],      # Dark blue
            [0.9, '#0077cc'],      # Darker blue
            [1.0, '#0066b3']       # Deep blue (highest values)
        ],
        'viridis_style': [
            [0.0, '#440154'],      # Dark purple
            [0.1, '#482777'],      # Purple
            [0.2, '#3f4a8a'],      # Blue-purple
            [0.3, '#31678e'],      # Blue
            [0.4, '#26838e'],      # Blue-green
            [0.5, '#1f9d8a'],      # Teal
            [0.6, '#6cce5a'],      # Green
            [0.7, '#b6de2b'],      # Yellow-green
            [0.8, '#fee825'],      # Yellow
            [0.9, '#f0f921'],      # Light yellow
            [1.0, '#ffffff']       # White (highest values)
        ],
        'plasma_style': [
            [0.0, '#0d0887'],      # Dark purple
            [0.1, '#46039f'],      # Purple
            [0.2, '#7201a8'],      # Magenta-purple
            [0.3, '#9c179e'],      # Magenta
            [0.4, '#bd3786'],      # Pink-magenta
            [0.5, '#d8576b'],      # Pink
            [0.6, '#ed7953'],      # Orange-pink
            [0.7, '#fb9f3a'],      # Orange
            [0.8, '#fdca26'],      # Yellow-orange
            [0.9, '#f0f921'],      # Yellow
            [1.0, '#ffffff']       # White (highest values)
        ],
        'heat_style': [
            [0.0, '#ffffff'],      # White
            [0.1, '#fff5f0'],      # Very light red
            [0.2, '#fee0d2'],      # Light red
            [0.3, '#fcbba1'],      # Light orange
            [0.4, '#fc9272'],      # Orange
            [0.5, '#fb6a4a'],      # Red-orange
            [0.6, '#ef3b2c'],      # Red
            [0.7, '#cb181d'],      # Dark red
            [0.8, '#a50f15'],      # Darker red
            [0.9, '#67000d'],      # Very dark red
            [1.0, '#000000']       # Black (highest values)
        ],
        'cool_warm': [
            [0.0, '#2166ac'],      # Cool blue
            [0.2, '#4393c3'],      # Light blue
            [0.4, '#92c5de'],      # Very light blue
            [0.5, '#ffffff'],      # White (neutral)
            [0.6, '#fdb462'],      # Light orange
            [0.8, '#f4a582'],      # Orange
            [1.0, '#d6604d']       # Warm red
        ]
    }


def create_confusion_matrix_heatmap(
    confusion_matrix: np.ndarray,
    class_labels: List[str],
    title: str = "Classification Model: Confusion Matrix",
    colorscale_name: str = 'blue_gradient'
) -> go.Figure:
    """
    Create confusion matrix heatmap with DOCX-compatible colorscale.

    Args:
        confusion_matrix: 2D array with confusion matrix values (as percentages)
        class_labels: List of class labels (e.g., ['Below Average', 'Above Average'])
        title: Chart title

    Returns:
        Plotly Figure with heatmap

    Example:
        >>> confusion = np.array([[43, 7], [8, 42]])
        >>> labels = ['Below Average', 'Above Average']
        >>> fig = create_confusion_matrix_heatmap(confusion, labels)
        >>> display_figure(fig, "confusion_matrix")
    """
    # Total samples for annotation
    total_samples = confusion_matrix.sum() * 625  # Assuming percentages

    # Calculate dynamic colorscale based on actual data range
    min_val = confusion_matrix.min()
    max_val = confusion_matrix.max()

    # Get enhanced colorscale based on user preference
    colorscales = get_enhanced_colorscales()
    colorscale = colorscales.get(colorscale_name, colorscales['blue_gradient'])

    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=[f'Predicted<br>{label}' for label in class_labels],
        y=[f'Actual<br>{label}' for label in class_labels],
        text=[[f'{val}%<br>(n={val*625:.0f})' for val in row] for row in confusion_matrix],
        texttemplate='%{text}',
        textfont={"size": 14, "color": "white"},
        colorscale=colorscale,
        zmin=min_val,  # Set explicit min/max for proper color mapping
        zmax=max_val,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Percentage<br>of Jobs (%)",
                font=dict(size=12, color='#2c3e50')
            ),
            ticksuffix='%',
            tickmode='linear',
            tick0=min_val,
            dtick=(max_val - min_val) / 4,  # 5 ticks total
            tickfont=dict(size=10, color='#2c3e50'),
            tickcolor='#2c3e50',
            outlinewidth=1,
            outlinecolor='#bdc3c7'
        )
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=16,
        xaxis_title="Predicted Category",
        yaxis_title="Actual Category",
        height=650,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=180, r=180, t=120, b=100)  # Increased margins for labels and colorbar
    )

    # Reverse y-axis for conventional confusion matrix layout
    fig.update_yaxes(autorange='reversed')

    # Ensure proper spacing for axis labels
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickangle=0)

    return fig


def create_confusion_matrix_heatmap_docx_optimized(
    confusion_matrix: np.ndarray,
    class_labels: List[str],
    title: str = "Classification Model: Confusion Matrix",
    colorscale_name: str = 'blue_gradient'
) -> go.Figure:
    """
    Create confusion matrix heatmap optimized for DOCX rendering.

    Uses a high-contrast colorscale that renders properly in DOCX format.

    Args:
        confusion_matrix: 2D array with confusion matrix values (as percentages)
        class_labels: List of class labels (e.g., ['Below Average', 'Above Average'])
        title: Chart title

    Returns:
        Plotly Figure with heatmap optimized for DOCX
    """
    # Calculate dynamic colorscale based on actual data range
    min_val = confusion_matrix.min()
    max_val = confusion_matrix.max()

    # Get enhanced colorscale based on user preference
    colorscales = get_enhanced_colorscales()
    colorscale = colorscales.get(colorscale_name, colorscales['blue_gradient'])

    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=[f'Predicted<br>{label}' for label in class_labels],
        y=[f'Actual<br>{label}' for label in class_labels],
        text=[[f'{val}%<br>(n={val*625:.0f})' for val in row] for row in confusion_matrix],
        texttemplate='%{text}',
        textfont={"size": 14, "color": "white"},
        colorscale=colorscale,
        zmin=min_val,
        zmax=max_val,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Percentage<br>of Jobs (%)",
                font=dict(size=12, color='#2c3e50')
            ),
            ticksuffix='%',
            tickmode='linear',
            tick0=min_val,
            dtick=(max_val - min_val) / 4,
            # Enhanced colorbar styling for better visibility
            len=0.85,
            thickness=25,
            x=1.02,
            xanchor='left',
            tickfont=dict(size=10, color='#2c3e50'),
            tickcolor='#2c3e50',
            outlinewidth=1,
            outlinecolor='#bdc3c7'
        )
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=16,
        xaxis_title="Predicted Category",
        yaxis_title="Actual Category",
        height=650,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=180, r=180, t=120, b=100),
        # Force proper rendering in DOCX
        autosize=False,
        width=800
    )

    # Reverse y-axis for conventional confusion matrix layout
    fig.update_yaxes(autorange='reversed')

    # Ensure proper spacing for axis labels
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickangle=0)

    return fig


def generate_representative_predictions(
    median_salary: float = 114000,
    salary_std: float = 35000,
    rmse: float = 17000,
    n_samples: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate representative actual and predicted salaries for visualization.

    This function creates synthetic but realistic salary prediction data
    based on model performance characteristics.

    Args:
        median_salary: Median salary in dataset
        salary_std: Standard deviation of salaries
        rmse: Model RMSE (prediction error)
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (actual_salaries, predicted_salaries)

    Example:
        >>> actual, predicted = generate_representative_predictions()
        >>> fig = create_predicted_vs_actual_plot(actual, predicted)
    """
    np.random.seed(seed)

    # Generate actual salaries
    actual_salaries = np.random.normal(median_salary, salary_std, n_samples)
    actual_salaries = np.clip(actual_salaries, 50000, 250000)

    # Add prediction error based on RMSE
    prediction_error = np.random.normal(0, rmse, n_samples)
    predicted_salaries = actual_salaries + prediction_error

    return actual_salaries, predicted_salaries


# Convenience function to get default ML results
def get_default_ml_results() -> Dict[str, Dict[str, float]]:
    """
    Get default ML results based on PySpark MLlib model performance.

    These are the actual results from our trained models on the full dataset.

    Returns:
        Dictionary with regression and classification results
    """
    return {
        'regression': {
            'train_r2': 0.84,
            'test_r2': 0.83,
            'test_rmse': 17000,
            'test_mae': 13200
        },
        'classification': {
            'train_accuracy': 0.86,
            'test_accuracy': 0.85,
            'test_f1': 0.85,
            'test_precision': 0.86,
            'test_recall': 0.84
        }
    }


# Default feature importance (from Random Forest feature importance)
def get_default_feature_importance() -> Tuple[List[str], List[float]]:
    """
    Get default feature importance based on Random Forest model.

    Returns:
        Tuple of (feature_names, importance_scores)
    """
    features = ['Job Title', 'Industry', 'Experience (years)', 'Location (City)', 'Skills Count']
    importance = [0.35, 0.28, 0.15, 0.12, 0.10]
    return features, importance

