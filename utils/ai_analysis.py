import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
from openai import OpenAI

# Initialize OpenAI client
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def analyze_crime_patterns(crime_data):
    """
    Use OpenAI to analyze crime patterns and provide insights.
    
    Parameters:
    -----------
    crime_data : pandas.DataFrame
        Crime data with various attributes
        
    Returns:
    --------
    dict
        Analysis results with insights and recommendations
    """
    client = get_openai_client()
    if client is None:
        return {
            "success": False,
            "error": "OpenAI API key is missing. Please provide an API key to enable AI analysis."
        }
    
    # Prepare data summary for analysis
    data_summary = {
        "total_crimes": len(crime_data),
        "date_range": f"{crime_data['date'].min()} to {crime_data['date'].max()}",
        "crime_types": dict(crime_data['crime_type'].value_counts().head(10)),
    }
    
    # Add time patterns if available
    if 'hour' in crime_data.columns:
        data_summary["hourly_distribution"] = dict(crime_data.groupby('hour').size())
    
    if 'day_of_week' in crime_data.columns:
        data_summary["daily_distribution"] = dict(crime_data.groupby('day_of_week').size())
    
    # Add location patterns
    if 'latitude' in crime_data.columns and 'longitude' in crime_data.columns:
        # Identify location clusters
        from sklearn.cluster import DBSCAN
        coords = crime_data[['latitude', 'longitude']].values
        clustering = DBSCAN(eps=0.01, min_samples=3).fit(coords)
        
        crime_data['cluster'] = clustering.labels_
        cluster_counts = crime_data[crime_data['cluster'] >= 0].groupby('cluster').size()
        
        top_clusters = cluster_counts.nlargest(5)
        cluster_locations = []
        
        for cluster_id in top_clusters.index:
            cluster_crimes = crime_data[crime_data['cluster'] == cluster_id]
            center_lat = cluster_crimes['latitude'].mean()
            center_lon = cluster_crimes['longitude'].mean()
            crime_types = dict(cluster_crimes['crime_type'].value_counts().head(3))
            
            cluster_locations.append({
                "cluster_id": int(cluster_id),
                "center": [float(center_lat), float(center_lon)],
                "crime_count": int(top_clusters[cluster_id]),
                "main_crime_types": crime_types
            })
        
        data_summary["hotspot_clusters"] = cluster_locations
    
    # Create prompt for OpenAI
    prompt = f"""
    As a criminal justice expert and data scientist, analyze the following crime data from Cuddalore district and provide insights:
    
    Crime Data Summary:
    {json.dumps(data_summary, indent=2)}
    
    Please provide:
    1. Key patterns observed in the data
    2. Potential underlying factors contributing to these patterns
    3. Recommended strategies for law enforcement
    4. Areas needing further investigation
    5. Predictions for future trends based on historical patterns
    
    Format your response as structured JSON with the following keys:
    - patterns (list of identified patterns)
    - factors (list of potential contributing factors)
    - strategies (list of recommended strategies)
    - investigation_areas (list of areas needing further investigation)
    - predictions (list of predictions for future trends)
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert crime analyst who provides objective, data-driven insights about crime patterns."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        # Parse the JSON response
        analysis_results = json.loads(response.choices[0].message.content)
        analysis_results["success"] = True
        
        return analysis_results
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_crime_report(crime_data, location=None, time_period=None):
    """
    Generate a comprehensive crime report for a specific area or time period.
    
    Parameters:
    -----------
    crime_data : pandas.DataFrame
        Crime data
    location : tuple, optional
        (latitude, longitude) of the center location
    time_period : tuple, optional
        (start_date, end_date) for the report period
        
    Returns:
    --------
    dict
        Report with structured sections
    """
    client = get_openai_client()
    if client is None:
        return {
            "success": False,
            "error": "OpenAI API key is missing. Please provide an API key to enable AI analysis."
        }
    
    # Filter data by location if provided
    filtered_data = crime_data.copy()
    
    if location and 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
        # Define a radius for location filtering (approximately 2km)
        lat, lon = location
        radius = 0.02  # Roughly 2km in degrees
        
        # Filter crimes within the radius
        filtered_data = filtered_data[
            (filtered_data['latitude'] >= lat - radius) & 
            (filtered_data['latitude'] <= lat + radius) &
            (filtered_data['longitude'] >= lon - radius) & 
            (filtered_data['longitude'] <= lon + radius)
        ]
    
    # Filter by time period if provided
    if time_period and 'date' in filtered_data.columns:
        start_date, end_date = time_period
        filtered_data = filtered_data[
            (filtered_data['date'] >= start_date) & 
            (filtered_data['date'] <= end_date)
        ]
    
    # Prepare data for the report
    data_summary = {
        "total_crimes": len(filtered_data),
        "crime_types": dict(filtered_data['crime_type'].value_counts()),
        "time_period": f"{filtered_data['date'].min()} to {filtered_data['date'].max()}" if 'date' in filtered_data.columns else "All time",
        "location": f"Around ({location[0]:.4f}, {location[1]:.4f})" if location else "All areas",
    }
    
    # Add severity information if available
    if 'severity' in filtered_data.columns:
        data_summary["severity_distribution"] = dict(filtered_data['severity'].value_counts())
        data_summary["average_severity"] = float(filtered_data['severity'].mean())
    
    # Add time trends if available
    if 'date' in filtered_data.columns:
        # Monthly trend
        filtered_data['month_year'] = pd.to_datetime(filtered_data['date']).dt.strftime('%Y-%m')
        monthly_trend = dict(filtered_data.groupby('month_year').size().tail(6))
        data_summary["monthly_trend"] = monthly_trend
    
    # Create prompt for OpenAI
    prompt = f"""
    As a crime analyst, generate a comprehensive crime report based on the following data:
    
    Crime Data Summary:
    {json.dumps(data_summary, indent=2)}
    
    Please structure your report with these sections:
    1. Executive Summary - brief overview of crime trends in the area/period
    2. Key Findings - detailed analysis of the most significant patterns
    3. High-Risk Areas & Times - identification of when and where crimes most frequently occur
    4. Crime Type Analysis - breakdown and analysis of different crime types
    5. Recommendations - actionable strategies for law enforcement
    
    Format your response as structured JSON with the following keys:
    - executive_summary (string)
    - key_findings (list of findings)
    - high_risk_areas_times (object with 'areas' and 'times' lists)
    - crime_type_analysis (object with crime types as keys and analysis as values)
    - recommendations (list of recommendation strings)
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert crime analyst who provides detailed reports based on crime data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse the JSON response
        report = json.loads(response.choices[0].message.content)
        report["success"] = True
        
        return report
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def analyze_specific_crime(crime_data, crime_type):
    """
    Analyze a specific crime type in depth.
    
    Parameters:
    -----------
    crime_data : pandas.DataFrame
        Crime data
    crime_type : str
        The specific crime type to analyze
        
    Returns:
    --------
    dict
        Analysis results specific to the crime type
    """
    client = get_openai_client()
    if client is None:
        return {
            "success": False,
            "error": "OpenAI API key is missing. Please provide an API key to enable AI analysis."
        }
    
    # Filter for the specific crime type
    filtered_data = crime_data[crime_data['crime_type'].str.lower() == crime_type.lower()]
    
    if filtered_data.empty:
        return {
            "success": False,
            "error": f"No data found for crime type: {crime_type}"
        }
    
    # Prepare data for analysis
    data_summary = {
        "crime_type": crime_type,
        "total_incidents": len(filtered_data),
        "time_range": f"{filtered_data['date'].min()} to {filtered_data['date'].max()}" if 'date' in filtered_data.columns else "All time",
    }
    
    # Time patterns
    if 'hour' in filtered_data.columns:
        hourly_counts = dict(filtered_data.groupby('hour').size())
        peak_hours = [int(hour) for hour, count in sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        data_summary["peak_hours"] = peak_hours
    
    if 'day_of_week' in filtered_data.columns:
        day_counts = dict(filtered_data.groupby('day_of_week').size())
        data_summary["day_distribution"] = day_counts
    
    # Location patterns
    if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
        # Identify main location clusters
        from sklearn.cluster import DBSCAN
        coords = filtered_data[['latitude', 'longitude']].values
        
        # Adjust eps based on the number of incidents
        eps = 0.02 if len(filtered_data) < 10 else 0.01
        min_samples = min(3, max(2, len(filtered_data) // 10))
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        filtered_data['cluster'] = clustering.labels_
        valid_clusters = filtered_data[filtered_data['cluster'] >= 0]
        
        if not valid_clusters.empty:
            cluster_counts = valid_clusters.groupby('cluster').size()
            top_clusters = cluster_counts.nlargest(3)
            
            hotspots = []
            for cluster_id in top_clusters.index:
                cluster_crimes = filtered_data[filtered_data['cluster'] == cluster_id]
                center_lat = cluster_crimes['latitude'].mean()
                center_lon = cluster_crimes['longitude'].mean()
                
                hotspots.append({
                    "center": [float(center_lat), float(center_lon)],
                    "crime_count": int(top_clusters[cluster_id]),
                    "percentage": float(top_clusters[cluster_id] / len(filtered_data) * 100)
                })
            
            data_summary["hotspots"] = hotspots
    
    # Create prompt for OpenAI
    prompt = f"""
    As a crime specialist focusing on {crime_type}, analyze the following data and provide insights:
    
    Crime Data Summary:
    {json.dumps(data_summary, indent=2)}
    
    Please provide:
    1. Patterns and characteristics specific to {crime_type} in this area
    2. Potential motivations and contributing factors
    3. Victim profiles and vulnerability factors
    4. Offender behavioral patterns
    5. Prevention strategies specific to this crime type
    6. Investigative recommendations
    
    Format your response as structured JSON with the following keys:
    - patterns (list of patterns and characteristics)
    - factors (list of potential motivations and contributing factors)
    - victim_profile (object with patterns about potential victims)
    - offender_behavior (list of behavioral patterns)
    - prevention (list of prevention strategies)
    - investigation (list of investigative recommendations)
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are an expert on {crime_type} criminal behavior and investigation."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        # Parse the JSON response
        analysis = json.loads(response.choices[0].message.content)
        analysis["success"] = True
        
        return analysis
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }