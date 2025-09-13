import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils.analysis import (
    generate_crime_statistics, 
    create_crime_trend_chart,
    create_crime_type_chart,
    create_time_heatmap,
    create_location_frequency_chart,
    create_monthly_comparison_chart
)
from utils.map_visualization import display_crime_map

def show_dashboard(crime_data):
    """
    Display the dashboard page with crime analytics
    
    Parameters:
    crime_data (DataFrame): The processed crime data
    """
    st.title("📊 Crime Analytics Dashboard")
    
    if crime_data is None:
        st.warning("Please upload a crime dataset to view analytics.")
        return
    
    # Generate statistics
    stats = generate_crime_statistics(crime_data)
    
    # Show key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Crimes", stats["total_crimes"])
    
    with col2:
        st.metric("Crime Hotspots", stats["hotspot_count"])
    
    with col3:
        st.metric("Most Common Crime", stats["most_common_crime"])
    
    with col4:
        # Calculate crime rate trend
        if 'date' in crime_data.columns:
            recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            recent_crimes = crime_data[crime_data['date'] >= recent_cutoff].shape[0]
            older_cutoff = recent_cutoff - pd.Timedelta(days=30)
            older_crimes = crime_data[(crime_data['date'] >= older_cutoff) & 
                                    (crime_data['date'] < recent_cutoff)].shape[0]
            
            if older_crimes > 0:
                trend_pct = ((recent_crimes - older_crimes) / older_crimes) * 100
                st.metric("30-Day Trend", f"{recent_crimes} crimes", f"{trend_pct:.1f}%")
            else:
                st.metric("30-Day Count", recent_crimes)
        else:
            st.metric("Data Range", "N/A")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Crime Distribution", "Time Patterns"])
    
    with tab1:
        st.subheader("Crime Trends Over Time")
        
        # Time unit selector for trend chart
        time_unit = st.selectbox(
            "Select time unit for trend analysis:",
            options=["day", "week", "month", "year"],
            index=2  # Default to month
        )
        
        # Create and display trend chart
        trend_chart = create_crime_trend_chart(crime_data, time_unit)
        st.plotly_chart(trend_chart, use_container_width=True)
        
        # Monthly comparison
        st.subheader("Monthly Crime Comparison")
        
        # Year selector for monthly comparison
        if 'date' in crime_data.columns:
            years = sorted(crime_data['date'].dt.year.unique())
            if years:
                selected_year = st.selectbox("Select year:", years, index=len(years)-1)
                monthly_chart = create_monthly_comparison_chart(crime_data, selected_year)
                st.plotly_chart(monthly_chart, use_container_width=True)
        else:
            st.info("No date information available for monthly comparison.")
    
    with tab2:
        st.subheader("Crime Type Distribution")
        
        # Create and display crime type chart
        type_chart = create_crime_type_chart(crime_data)
        st.plotly_chart(type_chart, use_container_width=True)
        
        # Location frequency
        st.subheader("Top Crime Locations")
        
        # Number of top locations to show
        top_n = st.slider("Number of top locations to show:", 5, 20, 10)
        
        # Create and display location frequency chart
        location_chart = create_location_frequency_chart(crime_data, top_n)
        st.plotly_chart(location_chart, use_container_width=True)
    
    with tab3:
        st.subheader("Crime Patterns by Time")
        
        # Create and display time heatmap
        time_heatmap = create_time_heatmap(crime_data)
        st.plotly_chart(time_heatmap, use_container_width=True)
        
        # Day of week distribution
        if 'day_of_week' in crime_data.columns:
            st.subheader("Crime Distribution by Day of Week")
            
            # Count crimes by day
            day_counts = crime_data['day_of_week'].value_counts().reset_index()
            day_counts.columns = ['day', 'count']
            
            # Define day order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Sort days according to day_order
            day_counts['day_num'] = day_counts['day'].apply(lambda x: day_order.index(x) if x in day_order else -1)
            day_counts = day_counts.sort_values('day_num').drop('day_num', axis=1)
            
            # Create bar chart
            fig = px.bar(
                day_counts,
                x='day',
                y='count',
                color='count',
                labels={'day': 'Day of Week', 'count': 'Number of Crimes'},
                color_continuous_scale='Reds'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Day of Week',
                yaxis_title='Number of Crimes',
                coloraxis_showscale=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Crime map
    st.subheader("Crime Distribution Map")
    
    # Filter for map visualization
    map_crime_filter = st.selectbox(
        "Filter by crime type:",
        options=["All"] + sorted(crime_data['crime_type'].unique().tolist()),
        index=0
    )
    
    # Display crime map with selected filter
    if map_crime_filter == "All":
        display_crime_map(crime_data)
    else:
        filtered_map_data = crime_data[crime_data['crime_type'] == map_crime_filter]
        display_crime_map(filtered_map_data)
