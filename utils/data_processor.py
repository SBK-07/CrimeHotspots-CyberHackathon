import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

def preprocess_data(df):
    """
    Preprocess raw crime data for analysis and prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw crime data
        
    Returns:
    --------
    pandas.DataFrame
        Processed crime data
    """
    processed_df = df.copy()
    
    # Ensure required columns exist
    required_columns = ['crime_type', 'date', 'time', 'latitude', 'longitude']
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Convert date and time columns
    try:
        processed_df['date'] = pd.to_datetime(processed_df['date']).dt.date
    except:
        # Try different date formats
        formats = ['%d-%m-%Y', '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']
        date_parsed = False
        
        for date_format in formats:
            try:
                processed_df['date'] = pd.to_datetime(processed_df['date'], format=date_format).dt.date
                date_parsed = True
                break
            except:
                continue
        
        if not date_parsed:
            # If all standard formats fail, try a more flexible approach
            try:
                processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce').dt.date
                # Drop rows where date parsing failed
                processed_df = processed_df.dropna(subset=['date'])
            except:
                raise ValueError("Unable to parse date column. Please ensure dates are in a standard format.")
    
    # Convert time if needed
    if 'time' in processed_df.columns:
        try:
            processed_df['time'] = pd.to_datetime(processed_df['time']).dt.time
        except:
            # Try different time formats
            time_formats = ['%H:%M:%S', '%I:%M %p', '%H:%M', '%I:%M:%S %p']
            time_parsed = False
            
            for time_format in time_formats:
                try:
                    processed_df['time'] = pd.to_datetime(processed_df['time'], format=time_format).dt.time
                    time_parsed = True
                    break
                except:
                    continue
            
            if not time_parsed:
                # If all fails, try a more flexible approach or use a default
                try:
                    # Try flexible parsing with errors='coerce'
                    temp_times = pd.to_datetime(processed_df['time'], errors='coerce')
                    # Replace NaT with midnight
                    temp_times = temp_times.fillna(pd.Timestamp('00:00:00'))
                    processed_df['time'] = temp_times.dt.time
                except:
                    # Last resort: default time
                    processed_df['time'] = '00:00:00'
    
    # Extract time features
    processed_df['day_of_week'] = pd.to_datetime(processed_df['date']).dt.day_name()
    
    # Extract hour - handle possible errors
    try:
        if isinstance(processed_df['time'].iloc[0], str):
            # Convert string time to datetime first
            processed_df['hour'] = pd.to_datetime(processed_df['time'], format='%H:%M:%S', errors='coerce').dt.hour
        else:
            # Convert time objects to hour 
            processed_df['hour'] = pd.Series([t.hour if hasattr(t, 'hour') else 0 
                                            for t in processed_df['time']])
    except Exception as e:
        # Default to midnight if parsing fails
        processed_df['hour'] = 0
    
    # Ensure latitude and longitude are numeric
    # Handle formats like "11.7569 N", "79.7632 E", or coordinates with degree symbols (°)
    
    # Process latitude
    if isinstance(processed_df['latitude'].iloc[0], str):
        # First, clean up the coordinates by removing degree symbols and other non-numeric characters
        processed_df['latitude'] = processed_df['latitude'].apply(
            lambda x: x.replace('°', '').strip() if isinstance(x, str) else x
        )
        
        # Handle N/S directions
        if any(direction in str(processed_df['latitude'].iloc[0]) for direction in ['N', 'S']):
            # Extract the numeric part
            processed_df['latitude'] = processed_df['latitude'].apply(
                lambda x: float(x.split()[0]) if isinstance(x, str) and len(x.split()) > 0 else x
            )
            # Make southern latitudes negative
            processed_df['latitude'] = processed_df['latitude'].apply(
                lambda x: -float(x) if isinstance(x, str) and 'S' in x else float(x)
            )
    
    # Process longitude
    if isinstance(processed_df['longitude'].iloc[0], str):
        # First, clean up the coordinates by removing degree symbols and other non-numeric characters
        processed_df['longitude'] = processed_df['longitude'].apply(
            lambda x: x.replace('°', '').strip() if isinstance(x, str) else x
        )
        
        # Handle E/W directions
        if any(direction in str(processed_df['longitude'].iloc[0]) for direction in ['E', 'W']):
            # Extract the numeric part
            processed_df['longitude'] = processed_df['longitude'].apply(
                lambda x: float(x.split()[0]) if isinstance(x, str) and len(x.split()) > 0 else x
            )
            # Make western longitudes negative
            processed_df['longitude'] = processed_df['longitude'].apply(
                lambda x: -float(x) if isinstance(x, str) and 'W' in x else float(x)
            )
    
    # Convert to numeric for any remaining processing
    processed_df['latitude'] = pd.to_numeric(processed_df['latitude'], errors='coerce')
    processed_df['longitude'] = pd.to_numeric(processed_df['longitude'], errors='coerce')
    
    # Drop rows with missing coordinates
    processed_df = processed_df.dropna(subset=['latitude', 'longitude'])
    
    # Add severity if not present
    if 'severity' not in processed_df.columns:
        # Map common crime types to severity levels (1-5)
        severity_mapping = {
            'theft': 2,
            'burglary': 3,
            'robbery': 4,
            'assault': 4,
            'murder': 5,
            'homicide': 5,
            'rape': 5,
            'sexual_assault': 5,
            'kidnapping': 5,
            'arson': 4,
            'drug': 3,
            'vandalism': 2,
            'fraud': 2,
            'forgery': 2,
            'trespassing': 1,
            'vehicle_theft': 3
        }
        
        # Assign severity based on crime type (default to 1 if not in mapping)
        processed_df['severity'] = processed_df['crime_type'].map(
            lambda x: next((v for k, v in severity_mapping.items() if k in x.lower()), 1)
        )
    
    # Ensure crime_type is standardized
    processed_df['crime_type'] = processed_df['crime_type'].str.lower().str.replace(' ', '_')
    
    return processed_df

def filter_data(df, start_date=None, end_date=None, crime_type=None, min_severity=None, max_severity=None):
    """
    Filter crime data based on various criteria.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Crime data to filter
    start_date : datetime.date, optional
        Start date for filtering
    end_date : datetime.date, optional
        End date for filtering
    crime_type : str, optional
        Type of crime to filter for
    min_severity : int, optional
        Minimum severity level
    max_severity : int, optional
        Maximum severity level
        
    Returns:
    --------
    pandas.DataFrame
        Filtered crime data
    """
    filtered_df = df.copy()
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])  # Ensure datetime type

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df['date'] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df['date'] <= end_date]

    
    # Apply crime type filter if provided
    if crime_type is not None:
        filtered_df = filtered_df[filtered_df['crime_type'] == crime_type]
    
    return filtered_df

def aggregate_crime_data(df, eps=0.005, min_samples=2):
    """
    Aggregate crime data to identify clusters/hotspots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Crime data to aggregate
    eps : float, optional
        Maximum distance between points in a cluster (in degrees)
    min_samples : int, optional
        Minimum number of points to form a cluster
        
    Returns:
    --------
    pandas.DataFrame
        Aggregated crime data with cluster information
    """
    if df.empty:
        return pd.DataFrame(columns=['latitude', 'longitude', 'count', 'primary_crimes'])
    
    # Extract coordinates for clustering
    coords = df[['latitude', 'longitude']].values
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
    # Add cluster labels to dataframe
    df_clusters = df.copy()
    df_clusters['cluster'] = clustering.labels_
    
    # Filter out noise points (cluster = -1)
    df_clusters = df_clusters[df_clusters['cluster'] >= 0]
    
    if df_clusters.empty:
        return pd.DataFrame(columns=['latitude', 'longitude', 'count', 'primary_crimes'])
    
    # Aggregate by cluster
    cluster_stats = df_clusters.groupby('cluster').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'crime_type': lambda x: list(x),
    }).reset_index()
    
    # Count incidents in each cluster
    cluster_stats['count'] = cluster_stats['crime_type'].apply(len)
    
    # Identify primary crime types in each cluster
    def get_primary_crimes(crime_list, top_n=3):
        crime_counts = pd.Series(crime_list).value_counts()
        return crime_counts.nlargest(top_n).index.tolist()
    
    cluster_stats['primary_crimes'] = cluster_stats['crime_type'].apply(get_primary_crimes)
    
    return cluster_stats[['latitude', 'longitude', 'count', 'primary_crimes']]
