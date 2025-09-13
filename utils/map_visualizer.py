import folium
from folium.plugins import HeatMap, MarkerCluster
from branca.colormap import linear
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.neighbors import KernelDensity
import streamlit as st
# Cuddalore district coordinates (center point)
CUDDALORE_CENTER = [11.1489, 79.4494]



# Create function to create a hotspot map
def create_hotspot_map(df, center=(11.7500, 79.7500), zoom_start=10):
    """
    Create a map that highlights crime hotspots
    
    Parameters:
    df (DataFrame): Crime data with latitude and longitude
    center (tuple): Center coordinates for the map
    zoom_start (int): Initial zoom level
    
    Returns:
    folium.Map: Map object with hotspots highlighted
    """
    # Create base map
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    
    # If no data or missing required columns, return base map
    if df is None or df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return m
    
    # Create KDE-based hotspot identification
    # Extract coordinates
    coords = df[['latitude', 'longitude']].values
    
    # Fit KDE model
    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    kde.fit(np.radians(coords))
    
    # Create a grid of points
    lat_range = np.linspace(df['latitude'].min() - 0.05, df['latitude'].max() + 0.05, 100)
    lon_range = np.linspace(df['longitude'].min() - 0.05, df['longitude'].max() + 0.05, 100)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    
    # Reshape grid for KDE
    grid_points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
    
    # Get density scores
    density = np.exp(kde.score_samples(np.radians(grid_points)))
    
    # Reshape density for contour
    density = density.reshape(lat_grid.shape)
    
    # Determine hotspot threshold
    hotspot_threshold = np.percentile(density, 90)
    
    # Create hotspot points for visualization
    hotspot_points = []
    for i in range(len(lat_grid)):
        for j in range(len(lon_range)):
            if density[i, j] > hotspot_threshold:
                hotspot_points.append([lat_grid[i, j], lon_grid[i, j], density[i, j]])
    
    # Create a heatmap from hotspot points
    HeatMap(
        hotspot_points,
        radius=20,
        blur=15,
        gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
        max_val=max([p[2] for p in hotspot_points]) if hotspot_points else 1,
        min_opacity=0.5
    ).add_to(m)
    
    # Add marker cluster for individual crimes
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add individual crime markers
    for idx, row in df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
            
        # Create popup content
        popup_content = f"""
        <div style="font-family: sans-serif;">
            <h4>Crime Details</h4>
            <b>Type:</b> {row.get('crime_type', 'N/A')}<br>
            <b>Date:</b> {row.get('date', 'N/A')}<br>
            <b>Location:</b> {row.get('latitude', 'N/A')}, {row.get('longitude', 'N/A')}<br>
        </div>
        """
        
        # Determine marker color based on crime type
        crime_colors = {
            'murder': 'red',
            'robbery': 'darkred',
            'theft': 'orange',
            'assault': 'darkpurple',
            'burglary': 'darkblue'
        }
        
        crime_type = row.get('crime_type', '').lower()
        marker_color = crime_colors.get(crime_type, 'blue')
        
        # Create marker with popup
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=row.get('crime_type', 'Crime'),
            icon=folium.Icon(color=marker_color, icon='info-sign')
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

from branca.colormap import linear, StepColormap

def create_crime_map(crime_data, title="Crime Map", highlight_search=False, is_prediction=False):
    """
    Create an interactive crime map using Folium.
    
    Parameters:
    -----------
    crime_data : pandas.DataFrame
        Crime data with latitude, longitude, and other attributes
    title : str, optional
        Map title
    highlight_search : bool, optional
        Whether to highlight search results
    is_prediction : bool, optional
        Whether the data represents predictions
        
    Returns:
    --------
    folium.Map
        Interactive map with crime data
    """
    # Create base map centered on Cuddalore
    crime_map = folium.Map(
        location=CUDDALORE_CENTER,
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add title
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(crime_map)
    
    # Add Cuddalore district boundary if available
    from assets.map_data import cuddalore_geojson
    
    folium.GeoJson(
        cuddalore_geojson,
        name='Cuddalore District',
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.1
        }
    ).add_to(crime_map)
    
    # Create marker cluster for crime points
    marker_cluster = MarkerCluster(name="Crime Incidents").add_to(crime_map)
    
    # Set up color mapping based on severity or crime type
    if 'severity' in crime_data.columns:
        max_severity = crime_data['severity'].max()
        min_severity = crime_data['severity'].min()
        color_map = linear.RdYlGn_09.scale(min_severity, max_severity)
        
        # Sort thresholds to avoid errors
        sorted_thresholds = sorted(set(color_map.index))
        color_map = StepColormap(
            colors=color_map.colors,
            index=sorted_thresholds,
            vmin=min_severity,
            vmax=max_severity
        )
        
        # Add color legend
        color_map.caption = 'Crime Severity'
        color_map.add_to(crime_map)
    
    # Iterate through crime data and add markers
    for _, crime in crime_data.iterrows():
        # Determine marker color based on severity or prediction status
        if is_prediction:
            color = 'purple'  # Use purple for predictions
        elif highlight_search:
            color = 'red'  # Use red for search results
        elif 'severity' in crime_data.columns:
            # Color based on severity
            severity_colors = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red', 5: 'darkred'}
            color = severity_colors.get(int(crime['severity']), 'blue')
        else:
            color = 'blue'
        
        # Create popup content
        popup_html = f"""
        <div style="width: 200px">
            <b>Crime Type:</b> {crime['crime_type'].replace('_', ' ').title()}<br>
        """
        
        if 'date' in crime:
            popup_html += f"<b>Date:</b> {crime['date']}<br>"
        
        if 'time' in crime:
            popup_html += f"<b>Time:</b> {crime['time']}<br>"
        
        if 'severity' in crime:
            popup_html += f"<b>Severity:</b> {int(crime['severity'])}<br>"
        
        if 'description' in crime:
            popup_html += f"<b>Description:</b> {crime['description']}<br>"
        
        if is_prediction:
            popup_html += f"<b>Predicted</b><br>"
        
        popup_html += "</div>"
        
        # Create marker
        folium.Marker(
            location=[crime['latitude'], crime['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon='info-sign' if not is_prediction else 'exclamation-sign'),
            tooltip=f"{crime['crime_type'].replace('_', ' ').title()}"
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(crime_map)
    
    return crime_map




def create_heatmap(crime_data):
    """
    Create a heatmap that safely handles both string and numeric data

    Parameters:
    crime_data (pd.DataFrame): Crime data with latitude/longitude

    Returns:
    folium.Map: Heatmap visualization
    """
    # Create base map
    heat_map = folium.Map(
        location=CUDDALORE_CENTER,
        zoom_start=11,
        tiles='CartoDB positron'
    )

    # Validate input
    if crime_data is None or crime_data.empty:
        st.warning("No data provided for heatmap")
        return heat_map

    if not isinstance(crime_data, pd.DataFrame):
        st.warning("Invalid data format")
        return heat_map

    if 'latitude' not in crime_data.columns or 'longitude' not in crime_data.columns:
        st.warning("Missing required coordinate columns")
        return heat_map

    # Create clean working copy
    plot_data = crime_data.copy()

    # Convert to numeric with proper coercion
    plot_data['latitude'] = pd.to_numeric(plot_data['latitude'], errors='coerce')
    plot_data['longitude'] = pd.to_numeric(plot_data['longitude'], errors='coerce')
    plot_data['severity'] = pd.to_numeric(plot_data['severity'], errors='coerce')

    # Remove invalid coordinates
    plot_data.dropna(subset=['latitude', 'longitude', 'severity'], inplace=True)

    if plot_data.empty:
        st.warning("No valid coordinates after cleaning")
        return heat_map

    # Prepare heatmap data
    heat_data = plot_data[['latitude', 'longitude', 'severity']].values.tolist()

    # Create heatmap if we have valid data
    if heat_data:
        try:
            HeatMap(
                heat_data,
                radius=15,
                blur=10,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'},
                name='Crime Heatmap'
            ).add_to(heat_map)
        except Exception as e:
            st.error(f"Heatmap rendering failed: {str(e)}")
    else:
        st.warning("No valid data points available")

    return heat_map