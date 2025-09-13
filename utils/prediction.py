import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import random

def identify_hotspots(crime_data, threshold=0.8):
    """
    Identify crime hotspots based on density and frequency
    
    Parameters:
    df (DataFrame): Crime data
    threshold (float): Density threshold for hotspot identification
    
    Returns:
    DataFrame: Identified hotspots with coordinates and scores
    """
    if crime_data.empty:
        return pd.DataFrame(columns=['lat_rounded', 'lon_rounded', 'count', 'density_score'])
    
    # Round coordinates to create grid
    df = crime_data.copy()
    df['lat_rounded'] = df['latitude'].round(3)
    df['lon_rounded'] = df['longitude'].round(3)
    
    # Count crimes in each grid cell
    grid_counts = df.groupby(['lat_rounded', 'lon_rounded']).size().reset_index(name='count')
    
    # Extract coordinates for KDE
    coords = grid_counts[['lat_rounded', 'lon_rounded']].values
    
    # Apply KDE to estimate density
    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    kde.fit(np.radians(coords))
    
    # Get density scores
    density_scores = np.exp(kde.score_samples(np.radians(coords)))
    
    # Add density scores to grid
    grid_counts['density_score'] = density_scores
    
    # Normalize density scores to 0-1 range
    if len(density_scores) > 1:
        max_score = grid_counts['density_score'].max()
        min_score = grid_counts['density_score'].min()
        if max_score > min_score:
            grid_counts['density_score'] = (grid_counts['density_score'] - min_score) / (max_score - min_score)
    
    # Filter hotspots based on threshold
    hotspots = grid_counts[grid_counts['density_score'] >= threshold]
    
    # Add most common crime type for each hotspot
    if 'crime_type' in df.columns:
        def get_most_common_crime(lat, lon):
            area_crimes = df[(df['lat_rounded'] == lat) & (df['lon_rounded'] == lon)]
            if area_crimes.empty:
                return "Unknown"
            return area_crimes['crime_type'].value_counts().index[0]
        
        hotspots['most_common_crime'] = hotspots.apply(
            lambda row: get_most_common_crime(row['lat_rounded'], row['lon_rounded']), axis=1
        )
    
    return hotspots

def calculate_risk_scores(df, location=None, radius=None):
    """
    Calculate risk scores for areas based on crime data
    
    Parameters:
    df (DataFrame): Crime data
    location (tuple): Optional center location (lat, lon)
    radius (float): Optional radius in kilometers around location
    
    Returns:
    DataFrame: Areas with risk scores
    """
    if df.empty:
        return pd.DataFrame(columns=['lat_grid', 'lon_grid', 'count', 'risk_score'])
    
    # Create a copy of the dataframe
    crime_df = df.copy()
    
    # Filter by location if provided
    if location and radius:
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the distance between two points in kilometers"""
            from math import radians, cos, sin, asin, sqrt
            
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of earth in kilometers
            
            return c * r
        
        # Filter crimes within radius
        crime_df = crime_df[crime_df.apply(
            lambda row: haversine_distance(location[0], location[1], row['latitude'], row['longitude']) <= radius,
            axis=1
        )]
    
    # Create grid
    grid_size = 0.005  # approximately 500m
    crime_df['lat_grid'] = (crime_df['latitude'] / grid_size).round() * grid_size
    crime_df['lon_grid'] = (crime_df['longitude'] / grid_size).round() * grid_size
    
    # Count crimes in each grid cell
    grid_counts = crime_df.groupby(['lat_grid', 'lon_grid']).size().reset_index(name='count')
    
    # Calculate risk score components
    
    # 1. Crime frequency component (0-50 points)
    max_count = grid_counts['count'].max() if not grid_counts.empty else 1
    grid_counts['frequency_score'] = (grid_counts['count'] / max_count) * 50
    
    # 2. Crime severity component (0-30 points)
    if 'severity' in crime_df.columns:
        # Calculate average severity for each grid cell
        severity_by_grid = crime_df.groupby(['lat_grid', 'lon_grid'])['severity'].mean().reset_index()
        grid_counts = pd.merge(grid_counts, severity_by_grid, on=['lat_grid', 'lon_grid'], how='left')
        
        # Normalize severity to 0-30 range
        max_severity = 5  # Assuming severity is on a 1-5 scale
        grid_counts['severity_score'] = (grid_counts['severity'] / max_severity) * 30
    else:
        grid_counts['severity_score'] = 15  # Default middle value
    
    # 3. Recency component (0-20 points)
    if 'date' in crime_df.columns:
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(crime_df['date']):
            crime_df['date'] = pd.to_datetime(crime_df['date'])
        
        # Calculate the latest crime date for each grid cell
        latest_dates = crime_df.groupby(['lat_grid', 'lon_grid'])['date'].max().reset_index()
        grid_counts = pd.merge(grid_counts, latest_dates, on=['lat_grid', 'lon_grid'], how='left')
        
        # Calculate recency score (more recent = higher score)
        # Max score for crimes in the last 7 days
        today = datetime.now().date()
        grid_counts['days_ago'] = (today - grid_counts['date'].dt.date).dt.days
        grid_counts['recency_score'] = 20 * np.exp(-grid_counts['days_ago'] / 30)  # Exponential decay
    else:
        grid_counts['recency_score'] = 10  # Default middle value
    
    # Calculate total risk score (0-100)
    grid_counts['risk_score'] = grid_counts['frequency_score'] + grid_counts['severity_score'] + grid_counts['recency_score']
    
    # Round risk score to 1 decimal place
    grid_counts['risk_score'] = grid_counts['risk_score'].round(1)
    
    # Return the grid with risk scores
    result_columns = ['lat_grid', 'lon_grid', 'count', 'risk_score']
    if 'severity' in grid_counts.columns:
        result_columns.append('severity')
    if 'date' in grid_counts.columns:
        result_columns.append('days_ago')
    
    return grid_counts[result_columns]