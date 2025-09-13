import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import random

def train_prediction_model(crime_data):
    """
    Train a prediction model for crime hotspots.
    
    Parameters:
    -----------
    crime_data : pandas.DataFrame
        Historical crime data
        
    Returns:
    --------
    tuple
        (trained_model, feature_list)
    """
    # Ensure we have the necessary time features
    df = crime_data.copy()
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract time features if not already present
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.day_name()
    
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
    
    if 'hour' not in df.columns and 'time' in df.columns:
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
    
    # Create location clusters for prediction (5 clusters should be sufficient for Cuddalore district)
    kmeans = KMeans(n_clusters=5, random_state=42)
    if not df.empty:
        df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    
    # Prepare categorical features for one-hot encoding
    categorical_features = ['day_of_week', 'month', 'location_cluster']
    numeric_features = ['hour'] if 'hour' in df.columns else []
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Select target variable (we'll predict crime count in each area)
    # Group data by location cluster and day/hour to get crime counts
    if not df.empty:
        crime_counts = df.groupby(['location_cluster', 'day_of_week', 'month', 'hour']).size().reset_index(name='crime_count')
        
        # Prepare features and target
        X = crime_counts[['location_cluster', 'day_of_week', 'month', 'hour']]
        y = crime_counts['crime_count']
        
        # Create and train model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model.fit(X, y)
        
        # Return trained model and necessary features
        return (model, {
            'kmeans': kmeans,
            'categorical_features': categorical_features,
            'numeric_features': numeric_features
        })
    else:
        # Return an untrained model if data is empty
        return (Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]), {
            'kmeans': kmeans,
            'categorical_features': categorical_features,
            'numeric_features': numeric_features
        })

def predict_future_crimes(crime_data, model=None, features=None, days=7):
    """
    Predict future crime hotspots based on historical data.
    
    Parameters:
    -----------
    crime_data : pandas.DataFrame
        Historical crime data
    model : sklearn.pipeline.Pipeline
        Trained prediction model
    features : dict
        Feature information from training
    days : int, optional
        Number of days to predict forward
        
    Returns:
    --------
    pandas.DataFrame
        Predicted crime incidents
    """
    print("\n=== PREDICTION DEBUGGING ===")
    print(f"Input data shape: {crime_data.shape}")
    if crime_data.empty:
        return pd.DataFrame(columns=['crime_type', 'date', 'latitude', 'longitude', 'severity'])
    
    # Extract necessary components
    kmeans = features['kmeans']
    
    # Get distribution of crime types and severities
    crime_type_dist = crime_data['crime_type'].value_counts(normalize=True).to_dict()
    
    if 'severity' in crime_data.columns:
        severity_dist = crime_data['severity'].value_counts(normalize=True).to_dict()
    else:
        # Default severity distribution
        severity_dist = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.15, 5: 0.05}
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Create date range for prediction
    start_date = datetime.now().date()
    date_range = [start_date + timedelta(days=i) for i in range(1, days + 1)]
    
    # Create hours range (consider all hours)
    hours_range = list(range(24))
    
    # Create empty list to store predictions
    predictions = []
    
    # Generate predictions for each day and hour
    for pred_date in date_range:
        day_of_week = pred_date.strftime('%A')
        month = pred_date.strftime('%B')  # Use month name instead of number
        
        for hour in hours_range:
            # Create a prediction input for each cluster
            for cluster_id in range(len(cluster_centers)):
                # Prepare input data with proper types
                pred_input = pd.DataFrame({
                    'location_cluster': [str(cluster_id)],  # Convert to string if model expects categorical
                    'day_of_week': [str(day_of_week)],     # Ensure string type
                    'month': [str(month)],                 # Ensure string type
                    'hour': [int(hour)]                    # Ensure integer type
                })
                
                try:
                    # Ensure column order matches training
                    pred_input = pred_input[features['feature_order']]
                    
                    # Predict crime count for this location and time
                    pred_count = max(0, round(float(model.predict(pred_input)[0])))  # Explicit float conversion
                    
                    # If crimes are predicted, generate individual crime records
                    if pred_count > 0:
                        # Get cluster center coordinates
                        center_lat, center_lon = cluster_centers[cluster_id]
                        
                        # Generate individual crime records
                        for _ in range(pred_count):
                            # Add some randomness to coordinates (within ~500m)
                            lat_variation = random.uniform(-0.004, 0.004)
                            lon_variation = random.uniform(-0.004, 0.004)
                            
                            latitude = float(center_lat + lat_variation)
                            longitude = float(center_lon + lon_variation)
                            
                            # Sample crime type and severity based on historical distribution
                            crime_type = random.choices(
                                list(crime_type_dist.keys()),
                                weights=list(crime_type_dist.values()),
                                k=1
                            )[0]
                            
                            severity = random.choices(
                                list(severity_dist.keys()),
                                weights=list(severity_dist.values()),
                                k=1
                            )[0]
                            
                            # Create prediction record
                            pred_record = {
                                'crime_type': str(crime_type),
                                'date': pred_date.strftime('%Y-%m-%d'),
                                'time': f"{hour:02d}:00:00",
                                'latitude': latitude,
                                'longitude': longitude,
                                'severity': int(severity),
                                'is_prediction': True
                            }
                            
                            predictions.append(pred_record)
                
                except Exception as e:
                    print(f"Prediction failed for cluster {cluster_id}, day {day_of_week}, hour {hour}")
                    print(f"Input data types: {pred_input.dtypes}")
                    print(f"Error: {str(e)}")
                    continue
    
    # Convert to DataFrame
    if predictions:
        pred_df = pd.DataFrame(predictions)
        # Ensure proper numeric types for coordinates
        pred_df['latitude'] = pd.to_numeric(pred_df['latitude'])
        pred_df['longitude'] = pd.to_numeric(pred_df['longitude'])
        return pred_df
    else:
        return pd.DataFrame(columns=['crime_type', 'date', 'latitude', 'longitude', 'severity'])
