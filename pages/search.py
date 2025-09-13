import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processing import filter_data, get_crime_categories
from utils.map_visualization import display_crime_map
from datetime import datetime, timedelta

def show_search(crime_data):
    """
    Display the search page for finding specific crime types and patterns
    
    Parameters:
    crime_data (DataFrame): The processed crime data
    """
    st.title("🔍 Crime Search")
    
    if crime_data is None:
        st.warning("Please upload a crime dataset to use the search functionality.")
        return
    
    # Create two columns for search filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Crime type selection
        crime_types = get_crime_categories(crime_data)
        selected_crime_types = st.multiselect(
            "Select crime type(s)",
            options=crime_types,
            default=None,
            help="Select one or more crime types to filter the data"
        )
    
    with col2:
        # Date range selection
        if 'date' in crime_data.columns:
            # Get min and max dates from data
            min_date = crime_data['date'].min().date()
            max_date = crime_data['date'].max().date()
            
            # Default to last 30 days if date range is large enough
            default_start = max(min_date, max_date - timedelta(days=30))
            
            start_date, end_date = st.date_input(
                "Select date range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Convert to datetime for filtering
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # End of the day
        else:
            start_datetime = None
            end_datetime = None
            st.info("No date information available for filtering.")
    
    # Location-based filtering
    st.subheader("Location Filter (Optional)")
    
    # Add a checkbox to enable location filtering
    use_location_filter = st.checkbox("Filter by location")
    
    if use_location_filter:
        # Create columns for location coordinates
        loc_col1, loc_col2, loc_col3 = st.columns(3)
        
        with loc_col1:
            latitude = st.number_input("Latitude", value=11.7500, format="%.4f")
        
        with loc_col2:
            longitude = st.number_input("Longitude", value=79.7500, format="%.4f")
        
        with loc_col3:
            radius = st.number_input("Radius (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
        
        location = (latitude, longitude)
    else:
        location = None
        radius = None
    
    # Apply filters
    filtered_data = filter_data(
        crime_data,
        crime_types=selected_crime_types,
        start_date=start_datetime,
        end_date=end_datetime,
        location=location,
        radius=radius
    )
    
    # Show filter summary and count
    if filtered_data is not None:
        st.metric("Filtered Crime Count", len(filtered_data))
        
        # Create filter description
        filter_parts = []
        
        if selected_crime_types:
            if len(selected_crime_types) == 1:
                filter_parts.append(f"Crime Type: {selected_crime_types[0]}")
            else:
                filter_parts.append(f"Crime Types: {', '.join(selected_crime_types)}")
        
        if start_datetime and end_datetime:
            filter_parts.append(f"Date Range: {start_date} to {end_date}")
        
        if location and radius:
            filter_parts.append(f"Location: {latitude:.4f}, {longitude:.4f} within {radius} km")
        
        if filter_parts:
            st.caption("Applied Filters: " + " | ".join(filter_parts))
    
    # Display map with filtered data
    st.subheader("Search Results Map")
    display_crime_map(filtered_data)
    
    # Show search results in tabular form
    st.subheader("Search Results Table")
    
    if filtered_data is None or filtered_data.empty:
        st.info("No results found matching your search criteria.")
    else:
        # Columns to display in the table
        display_columns = []
        
        # Add basic columns based on what's available
        if 'crime_type' in filtered_data.columns:
            display_columns.append('crime_type')
        
        if 'date' in filtered_data.columns:
            display_columns.append('date')
        
        if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
            display_columns.extend(['latitude', 'longitude'])
        
        # Add any other available columns that might be useful
        extra_columns = ['hour', 'day_of_week', 'year', 'month']
        for col in extra_columns:
            if col in filtered_data.columns:
                display_columns.append(col)
        
        # Prepare data for display
        display_data = filtered_data[display_columns].copy()
        
        # Rename columns for better readability
        column_names = {
            'crime_type': 'Crime Type',
            'date': 'Date & Time',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'hour': 'Hour',
            'day_of_week': 'Day of Week',
            'year': 'Year',
            'month': 'Month'
        }
        
        # Apply renaming only for columns that exist
        rename_dict = {k: v for k, v in column_names.items() if k in display_columns}
        display_data = display_data.rename(columns=rename_dict)
        
        # Display the table
        st.dataframe(display_data, hide_index=True)
        
        # Option to download filtered data
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="crime_search_results.csv",
            mime="text/csv"
        )
    
    # Show crime distribution by type (if filtered data exists)
    if filtered_data is not None and not filtered_data.empty and 'crime_type' in filtered_data.columns:
        st.subheader("Crime Distribution in Search Results")
        
        # Count by crime type
        crime_counts = filtered_data['crime_type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        
        # Create bar chart
        fig = px.bar(
            crime_counts,
            x='Crime Type',
            y='Count',
            color='Crime Type',
            title='Distribution of Crime Types in Search Results'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Crime Type',
            yaxis_title='Count',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show time distribution if date data is available
    if filtered_data is not None and not filtered_data.empty and 'hour' in filtered_data.columns:
        st.subheader("Time Distribution in Search Results")
        
        # Count by hour
        hour_counts = filtered_data.groupby('hour').size().reset_index()
        hour_counts.columns = ['Hour', 'Count']
        
        # Create line chart
        fig = px.line(
            hour_counts,
            x='Hour',
            y='Count',
            markers=True,
            title='Distribution of Crimes by Hour in Search Results'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Count',
            height=400,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
