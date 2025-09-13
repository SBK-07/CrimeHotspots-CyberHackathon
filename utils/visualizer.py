import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Union, Optional, Any
import io
import base64
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = "Predicted vs Actual Values") -> go.Figure:
    """
    Create scatter plot of predicted vs actual values.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    title : str, default="Predicted vs Actual Values"
        Title for the plot
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create a dataframe for plotting
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Create scatter plot
    fig = px.scatter(
        df, x='Actual', y='Predicted',
        title=title,
        labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'}
    )
    
    # Add perfect prediction line
    max_val = max(df['Actual'].max(), df['Predicted'].max())
    min_val = min(df['Actual'].min(), df['Predicted'].min())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Add annotation with metrics
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"RMSE: {rmse:.4f}<br>R²: {r2:.4f}",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig

def plot_feature_importance(model: Any, feature_names: List[str]) -> pd.Series:
    """
    Calculate and plot feature importance for a model.
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : List[str]
        List of feature names
        
    Returns:
    --------
    pd.Series
        Series with feature importances
    """
    # Try different methods to get feature importance
    importances = None
    
    # Method 1: Direct feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
    
    # Method 2: Coefficients (linear models)
    elif hasattr(model, 'coef_'):
        # For multi-output models, take the mean of coefficients
        if model.coef_.ndim > 1:
            importances = pd.Series(np.abs(model.coef_).mean(axis=0), index=feature_names)
        else:
            importances = pd.Series(np.abs(model.coef_), index=feature_names)
    
    # Method 3: Try to use permutation importance
    else:
        try:
            # This requires having the training data, which we don't have here
            # In a real implementation, this would need to be passed as an argument
            return None
        except:
            return None
    
    if importances is not None:
        # Sort importances
        importances = importances.sort_values(ascending=False)
    
    return importances

def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create visualizations for error analysis.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
        
    Returns:
    --------
    go.Figure
        Plotly figure with error analysis
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create subplots: residual plot, error distribution, QQ plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuals vs Predicted",
            "Error Distribution",
            "Residuals Over Data Points",
            "QQ Plot"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='Residuals'
        ),
        row=1, col=1
    )
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=min(y_pred),
        y0=0,
        x1=max(y_pred),
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
        row=1, col=1
    )
    
    # 2. Error Distribution
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='blue',
            opacity=0.7,
            name='Error Distribution'
        ),
        row=1, col=2
    )
    
    # 3. Residuals Over Data Points
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines+markers',
            marker=dict(size=6, color='blue', opacity=0.6),
            line=dict(width=1, color='blue'),
            name='Residuals'
        ),
        row=2, col=1
    )
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=len(residuals),
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
        row=2, col=1
    )
    
    # 4. QQ Plot
    from scipy.stats import probplot
    qq_data = probplot(residuals, dist="norm")
    theoretical_quantiles = qq_data[0][0]
    sample_quantiles = qq_data[0][1]
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='QQ Plot'
        ),
        row=2, col=2
    )
    
    # Add diagonal line for QQ plot
    max_val = max(abs(min(theoretical_quantiles)), abs(max(theoretical_quantiles)))
    fig.add_trace(
        go.Scatter(
            x=[-max_val, max_val],
            y=[-max_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Error Analysis",
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Data Points", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
    
    return fig

def plot_time_series_forecast(historical_dates: pd.Series, historical_values: pd.Series,
                           forecast_dates: pd.Series, forecast_values: pd.Series,
                           lower_ci: Optional[pd.Series] = None, upper_ci: Optional[pd.Series] = None,
                           title: str = "Time Series Forecast") -> go.Figure:
    """
    Create a time series forecast plot with optional confidence intervals.
    
    Parameters:
    -----------
    historical_dates : pd.Series
        Dates for historical data
    historical_values : pd.Series
        Values for historical data
    forecast_dates : pd.Series
        Dates for forecast period
    forecast_values : pd.Series
        Forecasted values
    lower_ci : pd.Series, optional
        Lower bound of confidence interval
    upper_ci : pd.Series, optional
        Upper bound of confidence interval
    title : str, default="Time Series Forecast"
        Title for the plot
        
    Returns:
    --------
    go.Figure
        Plotly figure with time series forecast
    """
    # Create the figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_dates,
            y=historical_values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Add confidence intervals if provided
    if lower_ci is not None and upper_ci is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=upper_ci,
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=lower_ci,
                mode='lines',
                name='Lower CI',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        width=900,
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_correlation_matrix(corr_matrix: pd.DataFrame, 
                          title: str = "Correlation Matrix") -> go.Figure:
    """
    Create a heatmap of the correlation matrix.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    title : str, default="Correlation Matrix"
        Title for the plot
        
    Returns:
    --------
    go.Figure
        Plotly figure with correlation heatmap
    """
    # Create the heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        width=900,
        height=700,
        template='plotly_white'
    )
    
    return fig

def plot_feature_effect(model: Any, feature: str, X: pd.DataFrame,
                      num_points: int = 100) -> go.Figure:
    """
    Create a partial dependence plot for a feature.
    
    Parameters:
    -----------
    model : object
        Trained model
    feature : str
        Feature name
    X : pd.DataFrame
        Feature matrix
    num_points : int, default=100
        Number of points to evaluate
        
    Returns:
    --------
    go.Figure
        Plotly figure with feature effect plot
    """
    # Check if feature exists in X
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in data")
    
    # Create a range of values for the feature
    feature_min = X[feature].min()
    feature_max = X[feature].max()
    feature_range = np.linspace(feature_min, feature_max, num_points)
    
    # Create a copy of X and vary the feature
    X_new = X.copy()
    predictions = []
    
    for value in feature_range:
        X_new[feature] = value
        try:
            # For models that take pandas dataframes
            pred = model.predict(X_new)
        except:
            # For models that require numpy arrays
            pred = model.predict(X_new.values)
        
        # Take the mean prediction
        predictions.append(np.mean(pred))
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=feature_range,
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(color='blue', width=3)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Effect of {feature} on Prediction",
        xaxis_title=feature,
        yaxis_title='Predicted Value',
        width=800,
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_model_comparison(model_results: pd.DataFrame, 
                         metric: str = 'Test R²') -> go.Figure:
    """
    Create a bar chart comparing model performance.
    
    Parameters:
    -----------
    model_results : pd.DataFrame
        DataFrame with model evaluation results
    metric : str, default='Test R²'
        Metric to use for comparison
        
    Returns:
    --------
    go.Figure
        Plotly figure with model comparison
    """
    # Check if metric exists
    if metric not in model_results.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    # Sort by the selected metric
    ascending = False if "R²" in metric or "Explained Variance" in metric else True
    sorted_results = model_results.sort_values(metric, ascending=ascending)
    
    # Create the plot
    fig = px.bar(
        sorted_results,
        x='Model',
        y=metric,
        title=f"Model Comparison by {metric}",
        text_auto='.3f'
    )
    
    # Add threshold line if it's R² or related
    if "R²" in metric:
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0,
            x1=len(sorted_results)-0.5,
            y1=0,
            line=dict(color="red", width=2, dash="dash")
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title=metric,
        width=800,
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_dashboard(models: Dict, X: pd.DataFrame, y: pd.Series, 
                   feature_names: List[str], target_name: str) -> Dict[str, go.Figure]:
    """
    Create a comprehensive dashboard of model evaluation visualizations.
    
    Parameters:
    -----------
    models : Dict
        Dictionary of trained models
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    feature_names : List[str]
        List of feature names
    target_name : str
        Name of the target variable
        
    Returns:
    --------
    Dict[str, go.Figure]
        Dictionary of plotly figures for the dashboard
    """
    dashboard = {}
    
    # Get the best model
    best_model_name = None
    best_r2 = -float('inf')
    
    for model_name, model in models.items():
        try:
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
        except:
            pass
    
    if best_model_name is None:
        return dashboard
    
    best_model = models[best_model_name]
    
    # 1. Predicted vs Actual for the best model
    y_pred = best_model.predict(X)
    dashboard['predicted_vs_actual'] = visualize_predictions(
        y, y_pred, 
        title=f"Predicted vs Actual Values ({best_model_name})"
    )
    
    # 2. Error analysis for the best model
    dashboard['error_analysis'] = plot_error_analysis(y, y_pred)
    
    # 3. Feature importance for the best model
    importances = plot_feature_importance(best_model, feature_names)
    if importances is not None:
        # Create feature importance plot
        dashboard['feature_importance'] = px.bar(
            x=importances.values,
            y=importances.index,
            orientation='h',
            title=f"Feature Importance for {best_model_name}",
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        dashboard['feature_importance'].update_layout(yaxis={'categoryorder': 'total ascending'})
    
    # 4. Model comparison
    model_results = pd.DataFrame(columns=['Model', 'Test R²', 'Test RMSE', 'Test MAE'])
    
    for model_name, model in models.items():
        try:
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            model_results = model_results.append({
                'Model': model_name,
                'Test R²': r2,
                'Test RMSE': rmse,
                'Test MAE': mae
            }, ignore_index=True)
        except:
            pass
    
    if not model_results.empty:
        dashboard['model_comparison'] = plot_model_comparison(model_results)
    
    # 5. Feature effect plots for top features
    if importances is not None:
        top_features = importances.head(3).index.tolist()
        
        for feature in top_features:
            try:
                dashboard[f'feature_effect_{feature}'] = plot_feature_effect(
                    best_model, feature, X
                )
            except:
                pass
    
    return dashboard