import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.prediction import identify_hotspots
import plotly.figure_factory as ff

def generate_crime_statistics(df):
    """
    Generate basic statistics from crime data
    
    Parameters:
    df (DataFrame): Crime data
    
    Returns:
    dict: Dictionary of crime statistics
    """
    stats = {
        "total_crimes": 0,
        "hotspot_count": 0,
        "most_common_crime": "N/A",
        "crime_by_type": {},
        "crime_by_time": {},
        "crime_by_day": {},
        "crime_by_month": {}
    }
    
    if df is None or df.empty:
        return stats
    
    # Total crimes
    stats["total_crimes"] = len(df)
    
    # Crime hotspots
    hotspots = identify_hotspots(df)
    stats["hotspot_count"] = len(hotspots)
    
    # Most common crime type
    if 'crime_type' in df.columns:
        crime_counts = df['crime_type'].value_counts()
        if len(crime_counts) > 0:
            stats["most_common_crime"] = crime_counts.index[0]
            stats["crime_by_type"] = crime_counts.to_dict()
    
    # Crime by time of day
    if 'hour' in df.columns:
        stats["crime_by_time"] = df.groupby('hour').size().to_dict()
    
    # Crime by day of week
    if 'day_of_week' in df.columns:
        stats["crime_by_day"] = df.groupby('day_of_week').size().to_dict()
    
    # Crime by month
    if 'month' in df.columns:
        stats["crime_by_month"] = df.groupby('month').size().to_dict()
    
    return stats

def create_crime_trend_chart(df, time_unit='month'):
    """
    Create a chart showing crime trends over time
    
    Parameters:
    df (DataFrame): Crime data
    time_unit (str): Time unit for aggregation ('day', 'week', 'month', 'year')
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    if df is None or df.empty or 'date' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for trend analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create a copy of the dataframe
    trend_df = df.copy()
    
    # Create date groups based on time unit
    if time_unit == 'day':
        trend_df['date_group'] = trend_df['date'].dt.date
    elif time_unit == 'week':
        trend_df['date_group'] = trend_df['date'].dt.to_period('W').dt.start_time
    elif time_unit == 'month':
        trend_df['date_group'] = trend_df['date'].dt.to_period('M').dt.start_time
    elif time_unit == 'year':
        trend_df['date_group'] = trend_df['date'].dt.year
    else:
        trend_df['date_group'] = trend_df['date'].dt.to_period('M').dt.start_time
    
    # Count crimes by date group
    counts = trend_df.groupby('date_group').size().reset_index(name='count')
    
    # Create plotly figure
    fig = px.line(
        counts, 
        x='date_group', 
        y='count',
        labels={'date_group': 'Date', 'count': 'Number of Crimes'},
        title=f'Crime Trend by {time_unit.capitalize()}'
    )
    
    # Add moving average if there are enough data points
    if len(counts) >= 3:
        counts['moving_avg'] = counts['count'].rolling(window=3, min_periods=1).mean()
        fig.add_scatter(
            x=counts['date_group'], 
            y=counts['moving_avg'],
            mode='lines',
            name='3-point Moving Average',
            line=dict(color='red', dash='dash')
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=time_unit.capitalize(),
        yaxis_title='Number of Crimes',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400
    )
    
    return fig

def create_crime_type_chart(df):
    """
    Create a chart showing distribution of crime types
    
    Parameters:
    df (DataFrame): Crime data
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    if df is None or df.empty or 'crime_type' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for crime type analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Count crimes by type
    crime_counts = df['crime_type'].value_counts().reset_index()
    crime_counts.columns = ['crime_type', 'count']
    
    # Sort by count (descending)
    crime_counts = crime_counts.sort_values('count', ascending=False)
    
    # Create plotly figure
    fig = px.bar(
        crime_counts,
        x='crime_type',
        y='count',
        color='crime_type',
        labels={'crime_type': 'Crime Type', 'count': 'Number of Incidents'},
        title='Distribution of Crime Types'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Crime Type',
        yaxis_title='Number of Incidents',
        showlegend=False,
        height=400
    )
    
    return fig

def create_time_heatmap(df):
    """
    Create a heatmap showing crime patterns by day of week and hour
    
    Parameters:
    df (DataFrame): Crime data
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    if df is None or df.empty or 'hour' not in df.columns or 'day_of_week' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for time pattern analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create a pivot table of crime counts by day and hour
    heatmap_data = pd.pivot_table(
        df,
        values='crime_type',
        index='day_of_week',
        columns='hour',
        aggfunc='count',
        fill_value=0
    )
    
    # Define day order (Monday first)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Reindex to ensure correct order
    if all(day in heatmap_data.index for day in day_order):
        heatmap_data = heatmap_data.reindex(day_order)
    
    # Create plotly figure
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Crime Count"),
        x=list(range(24)),
        y=heatmap_data.index,
        color_continuous_scale='Reds',
        title='Crime Patterns by Day and Hour'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=500
    )
    
    return fig

def create_location_frequency_chart(df, top_n=10):
    """
    Create a chart showing locations with highest crime frequency
    
    Parameters:
    df (DataFrame): Crime data
    top_n (int): Number of top locations to show
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    if df is None or df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No location data available for analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Round coordinates for grouping
    df_copy = df.copy()
    df_copy['location'] = df_copy.apply(
        lambda row: f"{row['latitude']:.4f}, {row['longitude']:.4f}",
        axis=1
    )
    
    # Count crimes by location
    location_counts = df_copy['location'].value_counts().reset_index()
    location_counts.columns = ['location', 'count']
    
    # Get top N locations
    top_locations = location_counts.head(top_n)
    
    # Create plotly figure
    fig = px.bar(
        top_locations,
        x='location',
        y='count',
        labels={'location': 'Location (Lat, Long)', 'count': 'Number of Crimes'},
        title=f'Top {top_n} Locations by Crime Frequency'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Location (Latitude, Longitude)',
        yaxis_title='Number of Crimes',
        height=400,
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_monthly_comparison_chart(df, year=None):
    """
    Create a chart comparing crime rates by month
    
    Parameters:
    df (DataFrame): Crime data
    year (int): Optional year for filtering
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    if df is None or df.empty or 'date' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for monthly comparison",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create a copy of the dataframe
    monthly_df = df.copy()
    
    # Extract year and month
    monthly_df['year'] = monthly_df['date'].dt.year
    monthly_df['month'] = monthly_df['date'].dt.month
    
    # Filter by year if specified
    if year:
        monthly_df = monthly_df[monthly_df['year'] == year]
    
    # Count crimes by month
    monthly_counts = monthly_df.groupby('month').size().reset_index(name='count')
    
    # Add month names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_counts['month_name'] = monthly_counts['month'].map(month_names)
    
    # Sort by month
    monthly_counts = monthly_counts.sort_values('month')
    
    # Create plotly figure
    fig = px.bar(
        monthly_counts,
        x='month_name',
        y='count',
        labels={'month_name': 'Month', 'count': 'Number of Crimes'},
        title=f'Crime Comparison by Month {year if year else ""}',
        color='count',
        color_continuous_scale='Reds'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Crimes',
        coloraxis_showscale=False,
        height=400
    )
    
    return fig

def crime_correlation_analysis(df):
    """
    Analyze correlations between different crime factors
    
    Parameters:
    df (DataFrame): Crime data
    
    Returns:
    plotly.graph_objects.Figure: Correlation heatmap
    """
    if df is None or df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for correlation analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create a copy of the dataframe
    corr_df = df.copy()
    
    # Select numerical columns for correlation
    num_cols = ['hour', 'day', 'month', 'year']
    
    # Filter columns that exist in the dataframe
    num_cols = [col for col in num_cols if col in corr_df.columns]
    
    # If no numerical columns are found, return empty figure
    if len(num_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient numerical data for correlation analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calculate correlation matrix
    corr_matrix = corr_df[num_cols].corr().round(2)
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu_r',
        title='Correlation Between Crime Factors'
    )
    
    # Update layout
    fig.update_layout(
        height=500
    )
    
    return fig
