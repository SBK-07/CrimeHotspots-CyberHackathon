import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
from utils.crime_predictor import predict_future_crimes, train_prediction_model
from utils.data_processor import aggregate_crime_data
from utils.map_visualizer import create_crime_map, create_heatmap
from utils.prediction import identify_hotspots, calculate_risk_scores

def show_predictions(crime_data):
    """
    Display the predictions page with crime forecasting
    
    Parameters:
    crime_data (DataFrame): The processed crime data
    """
    st.title("🔮 Crime Predictions")
    
    if crime_data is None:
        st.warning("Please upload a crime dataset to view predictions.")
        return
    
    # Create tabs for different prediction views
    tab1, tab2, tab3 = st.tabs(["Hotspot Analysis", "Future Crime Prediction", "Risk Assessment"])
    
    with tab1:
        st.subheader("Crime Hotspot Analysis")
        
        # Hotspot threshold slider
        hotspot_threshold = st.slider(
            "Hotspot density threshold:",
            min_value=0.5,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Higher values show fewer, more concentrated hotspots."
        )
        
        # Identify hotspots
        hotspots = identify_hotspots(crime_data, threshold=hotspot_threshold)
        
        # Display hotspot count
        st.metric("Number of Identified Hotspots", len(hotspots))
        
        # Create and display hotspot map
        st.subheader("Crime Hotspot Map")
        hotspot_map = create_hotspot_map(crime_data, center=(11.7500, 79.7500))
        folium_static(hotspot_map)
        
        # Display hotspot table
        if not hotspots.empty:
            st.subheader("Hotspot Details")
            
            # Format data for display
            display_hotspots = hotspots.copy()
            display_hotspots['density_score'] = display_hotspots['density_score'].map(lambda x: f"{x:.2f}")
            
            if 'most_common_crime' in display_hotspots.columns:
                cols_to_show = ['lat_rounded', 'lon_rounded', 'count', 'density_score', 'most_common_crime']
                col_names = {
                    'lat_rounded': 'Latitude',
                    'lon_rounded': 'Longitude',
                    'count': 'Crime Count',
                    'density_score': 'Density Score',
                    'most_common_crime': 'Most Common Crime'
                }
            else:
                cols_to_show = ['lat_rounded', 'lon_rounded', 'count', 'density_score']
                col_names = {
                    'lat_rounded': 'Latitude',
                    'lon_rounded': 'Longitude',
                    'count': 'Crime Count',
                    'density_score': 'Density Score'
                }
            
            st.dataframe(
                display_hotspots[cols_to_show].rename(columns=col_names),
                hide_index=True
            )
    
    with tab2:
        st.subheader("Future Crime Prediction")
        
        # Time period selector for prediction
        prediction_days = st.slider(
            "Number of days to predict:",
            min_value=7,
            max_value=60,
            value=30,
            step=1
        )
        
        # Generate predictions
        with st.spinner("Generating crime predictions..."):
            predictions = predict_future_crimes(crime_data, days=prediction_days)
        
        if predictions.empty:
            st.error("Could not generate predictions. Ensure your dataset has adequate historical data.")
        else:
            # Display prediction count
            st.metric("Predicted Crime Events", len(predictions))
            
            # Display predictions map
            st.subheader("Predicted Crime Locations")
            
            # Create base map
            m = folium.Map(location=(11.7500, 79.7500), zoom_start=10, tiles="OpenStreetMap")
            
            # Add crime prediction markers
            for idx, row in predictions.iterrows():
                # Create popup content
                popup_content = f"""
                <div style="font-family: sans-serif;">
                    <h4>Predicted Crime</h4>
                    <b>Type:</b> {row.get('predicted_crime_type', 'N/A')}<br>
                    <b>Date:</b> {row.get('date').strftime('%Y-%m-%d')}<br>
                    <b>Confidence:</b> {row.get('confidence', 0):.2f}<br>
                    <b>Location:</b> {row.get('latitude', 'N/A')}, {row.get('longitude', 'N/A')}<br>
                </div>
                """
                
                # Determine marker color based on confidence (higher confidence = more red)
                confidence = row.get('confidence', 0)
                marker_color = 'red' if confidence > 0.75 else 'orange' if confidence > 0.5 else 'blue'
                
                # Create marker with popup
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"{row.get('predicted_crime_type', 'Crime')} ({confidence:.2f})",
                    icon=folium.Icon(color=marker_color, icon='info-sign')
                ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            # Display prediction table
            st.subheader("Predicted Crime Details")
            
            # Format data for display
            display_predictions = predictions.copy()
            display_predictions['date'] = display_predictions['date'].dt.strftime('%Y-%m-%d')
            display_predictions['confidence'] = display_predictions['confidence'].map(lambda x: f"{x:.2f}")
            
            cols_to_show = ['date', 'predicted_crime_type', 'latitude', 'longitude', 'confidence']
            col_names = {
                'date': 'Date',
                'predicted_crime_type': 'Crime Type',
                'latitude': 'Latitude',
                'longitude': 'Longitude',
                'confidence': 'Confidence Score'
            }
            
            st.dataframe(
                display_predictions[cols_to_show].rename(columns=col_names),
                hide_index=True
            )
    
    with tab3:
        st.subheader("Area Risk Assessment")
        
        # Calculate risk scores
        risk_scores = calculate_risk_scores(crime_data)
        
        if risk_scores.empty:
            st.error("Could not calculate risk scores. Ensure your dataset has location data.")
        else:
            # Display risk map
            st.subheader("Crime Risk Map")
            
            # Create base map
            center = (11.7500, 79.7500)
            m = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap")
            
            # Add risk markers
            for idx, row in risk_scores.iterrows():
                # Determine marker color based on risk score
                risk = row['risk_score']
                marker_color = 'red' if risk > 75 else 'orange' if risk > 50 else 'yellow' if risk > 25 else 'green'
                
                # Create popup content
                popup_content = f"""
                <div style="font-family: sans-serif;">
                    <h4>Area Risk Assessment</h4>
                    <b>Risk Score:</b> {risk:.1f}/100<br>
                    <b>Crime Count:</b> {row['count']}<br>
                    <b>Location:</b> {row['lat_grid']}, {row['lon_grid']}<br>
                </div>
                """
                
                # Create circle marker with radius based on risk score
                folium.CircleMarker(
                    location=[row['lat_grid'], row['lon_grid']],
                    radius=max(5, min(15, risk/10)),  # Scale radius by risk (min 5, max 15)
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Risk Score: {risk:.1f}",
                    color=marker_color,
                    fill=True,
                    fill_color=marker_color,
                    fill_opacity=0.7
                ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            # Display risk score table
            st.subheader("Area Risk Scores")
            
            # Format data for display
            display_risk = risk_scores.copy()
            display_risk['risk_score'] = display_risk['risk_score'].map(lambda x: f"{x:.1f}")
            
            cols_to_show = ['lat_grid', 'lon_grid', 'count', 'risk_score']
            col_names = {
                'lat_grid': 'Latitude',
                'lon_grid': 'Longitude',
                'count': 'Crime Count',
                'risk_score': 'Risk Score (0-100)'
            }
            
            st.dataframe(
                display_risk[cols_to_show].rename(columns=col_names).sort_values('Risk Score (0-100)', ascending=False),
                hide_index=True
            )
            
            # Risk score distribution
            st.subheader("Risk Score Distribution")
            
            # Create histogram of risk scores
            fig = px.histogram(
                risk_scores,
                x='risk_score',
                nbins=20,
                labels={'risk_score': 'Risk Score', 'count': 'Number of Areas'},
                title='Distribution of Risk Scores Across Areas',
                color_discrete_sequence=['indianred']
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Risk Score (0-100)',
                yaxis_title='Number of Areas',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
