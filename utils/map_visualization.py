import folium
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
from branca.colormap import linear
from assets.map_data import cuddalore_geojson

def display_crime_map(df, zoom_start=10, center=(11.7500, 79.7500), crime_filter=None):
    """
    Create and display an interactive map with crime data
    
    Parameters:
    df (DataFrame): Crime data with latitude and longitude columns
    zoom_start (int): Initial zoom level
    center (tuple): Center coordinates for the map
    crime_filter (str): Filter for specific crime type
    
    Returns:
    None: Displays the map directly using st.folium_static
    """
    # Create base map
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    
    # Add Cuddalore district boundary
    folium.GeoJson(
        data=cuddalore_geojson,
        name="Cuddalore District",
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 2
        }
    ).add_to(m)
    
    # If no data is provided, just return the base map with district boundary
    if df is None or df.empty:
        folium_static(m)
        return
    
    # Filter data if crime type is specified
    if crime_filter and 'crime_type' in df.columns:
        filtered_df = df[df['crime_type'] == crime_filter]
    else:
        filtered_df = df
    
    # Check if filtered data has required columns
    if 'latitude' not in filtered_df.columns or 'longitude' not in filtered_df.columns:
        st.error("Data does not contain required latitude and longitude columns")
        folium_static(m)
        return
    
    # Create data for heatmap
    heat_data = [[row['latitude'], row['longitude']] for index, row in filtered_df.iterrows() 
                if not pd.isna(row['latitude']) and not pd.isna(row['longitude'])]
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        radius=15,
        blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
    ).add_to(m)
    
    # Create a marker cluster group
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add individual crime markers
    for idx, row in filtered_df.iterrows():
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
    
    # Display the map
    folium_static(m)

def create_choropleth_map(df, geojson, location_column, value_column, center, zoom_start=10):
    """
    Create a choropleth map for visualizing crime density by region
    
    Parameters:
    df (DataFrame): Data containing location and value columns
    geojson (dict): GeoJSON data for map regions
    location_column (str): Column in df that matches GeoJSON properties
    value_column (str): Column with values for color mapping
    center (tuple): Center coordinates for the map
    zoom_start (int): Initial zoom level
    
    Returns:
    folium.Map: Map object for display
    """
    # Create base map
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    
    # Add choropleth layer
    choropleth = folium.Choropleth(
        geo_data=geojson,
        name='Crime Density',
        data=df,
        columns=[location_column, value_column],
        key_on=f'feature.properties.{location_column}',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Crime Count'
    ).add_to(m)
    
    # Add hover functionality
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=[location_column, value_column],
            aliases=['Region:', 'Crime Count:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    )
    
    # Add Cuddalore district boundary
    folium.GeoJson(
        data=cuddalore_geojson,
        name="Cuddalore District",
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 2
        }
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_hotspot_map(df, center, zoom_start=10):
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
    
    # Add Cuddalore district boundary
    folium.GeoJson(
        data=cuddalore_geojson,
        name="Cuddalore District",
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 2
        }
    ).add_to(m)
    
    # If no data or missing required columns, return base map
    if df is None or df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return m
    
    # Create KDE-based hotspot identification
    # This is a simplified approach - in practice, you might use more sophisticated spatial statistics
    from sklearn.neighbors import KernelDensity
    
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
    
    # Determine hotspot threshold (adjust as needed)
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
