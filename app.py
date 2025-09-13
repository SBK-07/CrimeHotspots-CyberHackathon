import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import folium
from streamlit_folium import folium_static
import io

from utils.data_processor import preprocess_data, aggregate_crime_data, filter_data
from utils.map_visualizer import create_crime_map, create_heatmap
from utils.crime_predictor import train_prediction_model, predict_future_crimes

# Set page config
st.set_page_config(
    page_title="CrimeSpot:-",
    page_icon="🚨",
    layout="wide"
)
# Set background color
custom_css = """
<style>
/* Main background */
.stApp {
    background-color: #f5f5f5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #2c3e50 !important;
    color: white;
}
/* Sidebar text color */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Radio button labels */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }

/* Titles */
h1, h2, h3 {
    color: #2c3e50;
}

/* Buttons */
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 5px;
}

/* Change tab color */
[data-baseweb="tab"] {
    background-color: #ecf0f1;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# App title and description
st.title("🚨 CrimeSpot: Crime Analysis & Prediction Platform")
st.markdown("""
    ### Targeting Hotspots for Smarter Policing
    This platform analyzes historical crime data, identifies hotspots, and predicts future criminal activities to help law enforcement allocate resources effectively.
""")


# Initialize session state
if 'crime_data' not in st.session_state:
    st.session_state.crime_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = None
if 'prediction_features' not in st.session_state:
    st.session_state.prediction_features = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'current_search' not in st.session_state:
    st.session_state.current_search = None
    
# Automatically load dataset on startup if not already loaded
if st.session_state.crime_data is None:
    try:
        # Load data from attached assets
        file_path = 'attached_assets/Cuddalore_Crime_Database_Updated.csv'
        
        with st.spinner("Loading crime dataset..."):
            # Using latin-1 encoding to handle non-UTF-8 characters
            df = pd.read_csv(file_path, encoding='latin-1')
            processed_df = preprocess_data(df)
            st.session_state.crime_data = processed_df
            
            # Reset model state
            st.session_state.model_trained = False
            st.session_state.prediction_model = None
            
    except Exception as e:
        st.error(f"Error loading crime dataset: {str(e)}")
        st.info("There was an issue loading the crime dataset. Please check the file path and format.")

st.session_state.data_loaded=True
# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Crime Analysis", "Hotspot Visualization", "Crime Prediction"])

if page == "Dashboard":
    st.header("Crime Dashboard")
    st.info("Welcome to the CrimeSpot dashboard. This platform analyzes historical crime data to identify patterns, hotspots, and predict future criminal activities.")
    
    if st.session_state.crime_data is None:
        st.warning("Unable to load crime dataset. Please check if the file exists in the attached_assets folder.")
    else:
        # Data is loaded
        data = st.session_state.crime_data
        
        # Display data summary
        st.subheader("Crime Data Summary")
        st.write(f"Number of records: {len(data)}")
        st.write(f"Date range: {data['date'].min()} to {data['date'].max()}")
        st.write(f"Crime types: {', '.join(data['crime_type'].unique())}")
        
        # Quick overview charts
        st.subheader("Quick Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            # Crime type distribution
            crime_counts = data['crime_type'].value_counts().nlargest(10).reset_index()
            crime_counts.columns = ['Crime Type', 'Count']
            
            fig = px.bar(
                crime_counts, 
                x='Crime Type', 
                y='Count',
                color='Count',
                color_continuous_scale='Reds',
                title='Top 10 Crime Types'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Monthly trend
            if 'month' not in data.columns:
                data['month'] = pd.to_datetime(data['date']).dt.strftime('%b %Y')
                
            monthly_crimes = data.groupby('month').size().reset_index(name='count')
            monthly_crimes = monthly_crimes.tail(12)  # Last 12 months
            
            fig = px.line(
                monthly_crimes,
                x='month',
                y='count',
                markers=True,
                title='Crime Trend (Last 12 Months)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show hotspot map

    st.subheader("Report a Crime Incident")
    
    # Initialize session state for user submissions if not exists
    if 'user_submissions' not in st.session_state:
        st.session_state.user_submissions = []
    
    with st.form(key='crime_report_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            crime_type = st.selectbox(
                "Crime Type",
                options=["Theft", "Assault", "Burglary", "Vandalism", "Other"],
                index=0
            )
            date_reported = st.date_input("Date of Incident", value=datetime.now())
            location_desc = st.text_input("Location Description")
            
        with col2:
            time_reported = st.time_input("Time of Incident", value=datetime.now().time())

            additional_info = st.text_area("Additional Details")
        
        submitted = st.form_submit_button("Submit Crime Report")
        
        if submitted:
            # Create a new report dictionary
            new_report = {
                "crime_type": crime_type,
                "date": date_reported.strftime("%Y-%m-%d"),
                "time": time_reported.strftime("%H:%M:%S"),
                "location": location_desc,
                "details": additional_info,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to session state
            st.session_state.user_submissions.append(new_report)
            
            # Show success message
            st.success("Crime report submitted successfully! Thank you for your contribution.")
            
            # Show the submitted data
            with st.expander("View Your Submission"):
                st.json(new_report)
    
    # Display all user submissions (for demo purposes)
    if st.session_state.user_submissions:
        st.markdown("---")
        st.subheader("Your Submitted Reports")
        
        for i, report in enumerate(st.session_state.user_submissions, 1):
            with st.expander(f"Report #{i}: {report['crime_type']} at {report['location']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Type:** {report['crime_type']}")
                    st.markdown(f"**Date:** {report['date']}")
                    st.markdown(f"**Time:** {report['time']}")
                    st.markdown(f"**Location:** {report['location']}")
                    
                with col2:
                    
                    st.markdown(f"**Submitted on:** {report['submitted_at']}")
                    st.markdown(f"**Details:** {report['details']}")
                
                


elif page == "Crime Analysis":
    st.header("Crime Data Analysis")
    
    if st.session_state.crime_data is None:
        st.warning("Please go to the Dashboard to load the crime data.")
    else:
        data = st.session_state.crime_data
        
        # Filtering options
        st.subheader("Filter Data")
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range filter
            date_min = data['date'].min()
            date_max = data['date'].max()
            selected_date_range = st.date_input(
                "Select date range",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max
            )
        
        with col2:
            # Crime type filter
            crime_types = ['All'] + sorted(data['crime_type'].unique().tolist())
            selected_crime_type = st.selectbox("Select crime type", crime_types)
        
        start_date, end_date = selected_date_range
        # Apply filters
        filtered_data = filter_data(
            data, 
            start_date=start_date,
            end_date=end_date,
            crime_type=None if selected_crime_type == 'All' else selected_crime_type
        )
        
        # Display filtered data stats
        st.subheader("Filtered Data Statistics")
        st.write(f"Number of incidents: {len(filtered_data)}")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Crime by Type", "Crime by Time", "Crime by Location"])
        
        with tab1:
            # Crime by type analysis
            st.subheader("Crime Distribution by Type")
            crime_counts = filtered_data['crime_type'].value_counts().reset_index()
            crime_counts.columns = ['Crime Type', 'Count']
            
            fig = px.bar(
                crime_counts, 
                x='Crime Type', 
                y='Count',
                color='Count',
                color_continuous_scale='Reds',
                title='Distribution of Crimes by Type'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Crime by time analysis
            st.subheader("Crime Distribution by Time")
            
            # Add hour column if not exists
            if 'hour' not in filtered_data.columns:
                filtered_data['hour'] = pd.to_datetime(filtered_data['time']).dt.hour
            
            # Group by hour
            hourly_crimes = filtered_data.groupby('hour').size().reset_index(name='count')
            
            # Create 24-hour distribution chart
            fig = px.line(
                hourly_crimes, 
                x='hour', 
                y='count',
                markers=True,
                title='Crime Distribution by Hour of Day',
                labels={'hour': 'Hour of Day (24-hour format)', 'count': 'Number of Crimes'}
            )
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Group by day of week
            if 'day_of_week' not in filtered_data.columns:
                filtered_data['day_of_week'] = pd.to_datetime(filtered_data['date']).dt.day_name()
            
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_crimes = filtered_data['day_of_week'].value_counts().reindex(days_order).reset_index()
            daily_crimes.columns = ['Day of Week', 'Count']
            
            fig = px.bar(
                daily_crimes, 
                x='Day of Week', 
                y='Count',
                color='Count',
                color_continuous_scale='Reds',
                title='Crime Distribution by Day of Week'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            # Crime by location analysis
            st.subheader("Crime Distribution by Location")
            
            # Create a density heatmap
            fig = px.density_mapbox(
                filtered_data, 
                lat='latitude', 
                lon='longitude', 
                z='severity' if 'severity' in filtered_data.columns else None,
                radius=10,
                center=dict(lat=11.1489, lon=79.4494),
                zoom=10,
                mapbox_style="open-street-map",
                title='Crime Density Map'
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Hotspot Visualization":
    st.header("Crime Hotspot Visualization")

    if st.session_state.crime_data is None:
        st.warning("Please go to the Dashboard to load the crime data.")
    else:
        data = st.session_state.crime_data.copy()

        # Ensure correct data types
        data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.normalize()
        data['crime_type'] = data['crime_type'].astype(str).fillna("Unknown")
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data.dropna(subset=['latitude', 'longitude'], inplace=True)

        # Sidebar filtering options
        st.sidebar.subheader("Filter Hotspot Data")
        date_min = data['date'].min().date()
        date_max = data['date'].max().date()

        # Date range filter
        selected_date_range = st.sidebar.date_input(
            "Select date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max
        )

        # Convert selected dates to datetime
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])

        # Crime type filter
        crime_types = ['All'] + sorted(data['crime_type'].unique().tolist())
        selected_crime_type = st.sidebar.selectbox("Select crime type for hotspot", crime_types)

        # Apply filters
        filtered_data = filter_data(
            data,
            start_date=start_date,
            end_date=end_date,
            crime_type=None if selected_crime_type == 'All' else selected_crime_type
        )

        # Search functionality
        search_query = st.text_input("Search for specific crime types (comma-separated)")

        if search_query:
            search_terms = [term.strip().lower() for term in search_query.split(',')]
            data['crime_type'] = data['crime_type'].astype(str).fillna("Unknown")
            search_results = data[data['crime_type'].str.lower().isin(search_terms)]
            st.session_state.current_search = search_results

            if not search_results.empty:
                st.success(f"Found {len(search_results)} incidents matching your search.")
                search_results['severity'] = pd.to_numeric(search_results['severity'], errors='coerce')
                search_results.dropna(subset=['severity'], inplace=True)
                search_map = create_crime_map(search_results, "Cuddalore District - Search Results", highlight_search=True)
                folium_static(search_map, width=1000, height=600)
                st.subheader("Search Results")
                st.dataframe(search_results[['crime_type', 'date', 'latitude', 'longitude']])
            else:
                st.warning("No results found for your search.")
        else:
            if not filtered_data.empty:
                heatmap = create_heatmap(filtered_data)
                if heatmap:
                    folium_static(heatmap, width=1000, height=600)
                else:
                    st.warning("Could not generate heatmap - no valid data points")

elif page == "Crime Prediction":
    st.header("Crime Prediction Analysis")
    
    if st.session_state.crime_data is None:
        st.warning("Please go to the Dashboard to load the crime data.")
    else:
        data = st.session_state.crime_data
        
        # Train prediction model
        if not st.session_state.model_trained:
            st.info("Training prediction model...")
            with st.spinner("This may take a moment..."):
                model, features = train_prediction_model(data)
                st.session_state.prediction_model = model
                st.session_state.prediction_features = features
                st.session_state.model_trained = True
            st.success("Prediction model trained successfully!")
        
        # Prediction settings
        st.subheader("Prediction Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_days = st.slider("Prediction horizon (days)", 1, 30, 7)
        
        with col2:
            crime_types = ['All'] + sorted(data['crime_type'].unique().tolist())
            prediction_crime_type = st.selectbox("Predict specific crime type", crime_types)
        
        # Make predictions
        if st.button("Generate Predictions"):
            with st.spinner("Generating crime predictions..."):
                # Filter data for specific crime type if selected
                prediction_data = data if prediction_crime_type == 'All' else data[data['crime_type'] == prediction_crime_type]
                
                # Get predictions
                future_crimes = predict_future_crimes(
                    prediction_data,
                    st.session_state.prediction_model,
                    st.session_state.prediction_features,
                    days=prediction_days
                )
                
                # Create prediction map
                pred_map = create_crime_map(future_crimes, f"Predicted Crime Hotspots for Next {prediction_days} Days", is_prediction=True)
                st.subheader(f"Predicted Crime Hotspots for Next {prediction_days} Days")
                folium_static(pred_map, width=1000, height=600)
                
                # Display prediction statistics
                st.subheader("Prediction Statistics")
                
                # Number of predicted crimes
                st.metric("Predicted Incidents", len(future_crimes))
                
                # Top predicted crime areas
                top_areas = aggregate_crime_data(future_crimes)
                if not top_areas.empty:
                    st.subheader("Top Predicted Hotspots")
                    for i, area in enumerate(top_areas.nlargest(3, 'count').itertuples(), 1):
                        st.markdown(f"**Area {i}:** Near lat {area.latitude:.4f}, lon {area.longitude:.4f}")
                        st.markdown(f"Predicted incidents: {area.count}")
                        st.markdown(f"Main crime types: {', '.join(area.primary_crimes)}")
                        
                        # Add to alerts if high risk
                        if area.count >= 3:  # Threshold for high-risk areas
                            alert = {
                                'latitude': area.latitude,
                                'longitude': area.longitude,
                                'predicted_incidents': area.count,
                                'crime_types': area.primary_crimes,
                                'alert_date': datetime.now().strftime('%Y-%m-%d'),
                                'prediction_horizon': prediction_days
                            }
                            
                            # Avoid duplicate alerts
                            if alert not in st.session_state.alerts:
                                st.session_state.alerts.append(alert)
                
                # Download predictions
                pred_csv = future_crimes.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Data",
                    data=pred_csv,
                    file_name="crime_predictions.csv",
                    mime="text/csv"
                )

elif page == "AI Insights":
    from pages.ai_insights import show_ai_insights
    show_ai_insights(st.session_state.crime_data)

elif page == "Alerts Dashboard":
    st.header("Crime Alerts Dashboard")
    
    if not st.session_state.alerts:
        st.info("No alerts generated yet. Visit the Crime Prediction page to generate alerts for high-risk areas.")
    else:
        # Display alerts
        st.subheader(f"Active Alerts ({len(st.session_state.alerts)})")
        
        for i, alert in enumerate(st.session_state.alerts):
            with st.expander(f"Alert #{i+1}: High-risk area near ({alert['latitude']:.4f}, {alert['longitude']:.4f})"):
                st.markdown(f"**Alert Date:** {alert['alert_date']}")
                st.markdown(f"**Prediction Horizon:** {alert['prediction_horizon']} days")
                st.markdown(f"**Predicted Incidents:** {alert['predicted_incidents']}")
                st.markdown(f"**Predicted Crime Types:** {', '.join(alert['crime_types'])}")
                
                # Show alert location on mini-map
                alert_map = folium.Map(location=[alert['latitude'], alert['longitude']], zoom_start=15)
                folium.Marker(
                    [alert['latitude'], alert['longitude']],
                    popup=f"High-risk area: {alert['predicted_incidents']} predicted incidents",
                    icon=folium.Icon(color='red', icon='warning-sign')
                ).add_to(alert_map)
                
                # Add circle showing approximate area
                folium.Circle(
                    [alert['latitude'], alert['longitude']],
                    radius=500,  # 500m radius
                    color='red',
                    fill=True,
                    fill_opacity=0.2
                ).add_to(alert_map)
                
                folium_static(alert_map, width=700, height=300)
                
                # Action buttons (demonstrative)
                if st.button(f"Dismiss Alert #{i+1}"):
                    st.session_state.alerts.pop(i)
                    st.success("Alert dismissed!")
                    st.rerun()
        
        # Clear all alerts button
        if st.button("Clear All Alerts"):
            st.session_state.alerts = []
            st.success("All alerts cleared!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("🚨 **CrimeSpot** - Crime Analysis & Prediction Platform for Cuddalore District")
