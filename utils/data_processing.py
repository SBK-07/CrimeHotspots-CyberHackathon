import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

def load_data(file):
    """
    Load crime data from uploaded file (CSV or Excel)
    
    Parameters:
    file (UploadedFile): The uploaded file containing crime data
    
    Returns:
    DataFrame: Pandas DataFrame with loaded crime data
    """
    file_ext = file.name.split('.')[-1]
    
    if file_ext == 'csv':
        df = pd.read_csv(file)
    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    return df

def preprocess_data(df):
    """
    Preprocess the crime data for analysis and visualization
    
    Parameters:
    df (DataFrame): Raw crime data
    
    Returns:
    DataFrame: Preprocessed crime data
    """
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Check required columns
    required_columns = ['crime_type', 'date', 'latitude', 'longitude']
    
    # Try to automatically detect and rename columns if they don't match expected names
    column_mapping = {}
    for col in processed_df.columns:
        col_lower = col.lower()
        
        # Map date-related columns
        if any(date_keyword in col_lower for date_keyword in ['date', 'time', 'datetime']):
            column_mapping[col] = 'date'
        
        # Map latitude-related columns
        elif any(lat_keyword in col_lower for lat_keyword in ['lat', 'latitude']):
            column_mapping[col] = 'latitude'
        
        # Map longitude-related columns
        elif any(lon_keyword in col_lower for lon_keyword in ['lon', 'long', 'longitude']):
            column_mapping[col] = 'longitude'
        
        # Map crime type-related columns
        elif any(crime_keyword in col_lower for crime_keyword in ['crime', 'offense', 'type']):
            column_mapping[col] = 'crime_type'
    
    # Rename columns based on mapping
    if column_mapping:
        processed_df = processed_df.rename(columns=column_mapping)
    
    # Check if required columns exist after mapping
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}. Please make sure your dataset includes columns for crime type, date, latitude, and longitude.")
    
    # Convert date column to datetime if it's not already
    if 'date' in processed_df.columns:
        try:
            if not pd.api.types.is_datetime64_dtype(processed_df['date']):
                processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
            
            # Extract date components for analysis
            processed_df['year'] = processed_df['date'].dt.year
            processed_df['month'] = processed_df['date'].dt.month
            processed_df['day'] = processed_df['date'].dt.day
            processed_df['hour'] = processed_df['date'].dt.hour
            processed_df['day_of_week'] = processed_df['date'].dt.day_name()
        except Exception as e:
            st.warning(f"Could not process date column: {str(e)}")
    
    # Convert latitude and longitude to float
    try:
        processed_df['latitude'] = pd.to_numeric(processed_df['latitude'], errors='coerce')
        processed_df['longitude'] = pd.to_numeric(processed_df['longitude'], errors='coerce')
    except Exception as e:
        st.warning(f"Could not convert coordinates to numeric: {str(e)}")
    
    # Drop rows with missing coordinates
    processed_df = processed_df.dropna(subset=['latitude', 'longitude'])
    
    # Standardize crime types (convert to lowercase and strip spaces)
    if 'crime_type' in processed_df.columns:
        processed_df['crime_type'] = processed_df['crime_type'].str.lower().str.strip()
    
    # Add week and month for aggregation purposes
    try:
        processed_df['week'] = processed_df['date'].dt.isocalendar().week
        processed_df['month_name'] = processed_df['date'].dt.month_name()
    except:
        pass
    
    return processed_df

def get_crime_categories(df):
    """
    Extract unique crime categories from the dataset
    
    Parameters:
    df (DataFrame): Crime data
    
    Returns:
    list: List of unique crime categories
    """
    if df is None or 'crime_type' not in df.columns:
        return []
    
    return sorted(df['crime_type'].unique().tolist())

def filter_data(df, crime_types=None, start_date=None, end_date=None, location=None, radius=None):
    """
    Filter crime data based on various criteria
    
    Parameters:
    df (DataFrame): Crime data
    crime_types (list): List of crime types to include
    start_date (datetime): Start date for filtering
    end_date (datetime): End date for filtering
    location (tuple): (latitude, longitude) of the center location
    radius (float): Radius in kilometers for location filtering
    
    Returns:
    DataFrame: Filtered crime data
    """
    if df is None:
        return None
    
    filtered_df = df.copy()
    
    # Filter by crime type
    if crime_types and 'crime_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['crime_type'].isin(crime_types)]
    
    # Filter by date range
    if start_date and 'date' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['date'] >= start_date]
    
    if end_date and 'date' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['date'] <= end_date]
    
    # Filter by location radius
    if location and radius and 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        center_lat, center_lon = location
        
        # Haversine formula to calculate distance
        R = 6371  # Earth radius in kilometers
        
        # Convert degrees to radians
        lat_rad = np.radians(filtered_df['latitude'])
        lon_rad = np.radians(filtered_df['longitude'])
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        
        # Calculate differences
        dlon = lon_rad - center_lon_rad
        dlat = lat_rad - center_lat_rad
        
        # Apply Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(center_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        # Filter by distance
        filtered_df = filtered_df[distance <= radius]
    
    return filtered_df

def get_time_trends(df):
    """
    Generate time-based trends from crime data
    
    Parameters:
    df (DataFrame): Crime data
    
    Returns:
    dict: Dictionary containing various time-based trends
    """
    if df is None or df.empty or 'date' not in df.columns:
        return {
            'hourly': pd.DataFrame(),
            'daily': pd.DataFrame(),
            'monthly': pd.DataFrame(),
            'yearly': pd.DataFrame()
        }
    
    # Hourly trends
    hourly_trends = df.groupby('hour').size().reset_index(name='count')
    
    # Daily trends
    daily_trends = df.groupby('day_of_week').size().reset_index(name='count')
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_trends['day_order'] = daily_trends['day_of_week'].apply(lambda x: day_order.index(x) if x in day_order else -1)
    daily_trends = daily_trends.sort_values('day_order').drop('day_order', axis=1)
    
    # Monthly trends
    monthly_trends = df.groupby('month').size().reset_index(name='count')
    
    # Yearly trends
    yearly_trends = df.groupby('year').size().reset_index(name='count')
    
    return {
        'hourly': hourly_trends,
        'daily': daily_trends,
        'monthly': monthly_trends,
        'yearly': yearly_trends
    }
