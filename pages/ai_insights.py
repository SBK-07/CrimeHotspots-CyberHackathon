import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import plotly.express as px
from utils.ai_analysis import analyze_crime_patterns, generate_crime_report, analyze_specific_crime
from utils.data_processor import filter_data
from utils.map_visualizer import create_crime_map

def show_ai_insights(crime_data):
    """
    Display the AI Insights page with advanced crime analysis
    
    Parameters:
    crime_data (DataFrame): The processed crime data
    """
    st.title("🧠 AI-Powered Crime Insights")
    
    if crime_data is None:
        st.warning("Please upload crime data first.")
        return
    
    # Verify OpenAI API key
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        st.warning("⚠️ OpenAI API key is required for AI analysis features.")
        
        with st.expander("How to get an OpenAI API Key"):
            st.markdown("""
            1. Visit [OpenAI API](https://platform.openai.com/signup)
            2. Create an account or sign in
            3. Navigate to API keys section
            4. Create a new secret key
            5. Copy the key and paste it below
            """)
        
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set successfully! You can now use AI analysis features.")
            st.rerun()
        else:
            st.info("Please provide an API key to continue.")
            return
    
    # Create tabs for different AI analysis features
    tab1, tab2, tab3 = st.tabs(["Pattern Analysis", "Crime Reports", "Crime Type Analysis"])
    
    with tab1:
        st.subheader("Crime Pattern Analysis")
        st.info("This feature uses AI to analyze crime patterns and provide insights beyond statistical analysis.")
        
        # Filtering options
        st.subheader("Filter Data for Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range filter
            date_min = crime_data['date'].min()
            date_max = crime_data['date'].max()
            selected_date_range = st.date_input(
                "Select date range",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max
            )
        
        with col2:
            # Crime type filter
            crime_types = ['All'] + sorted(crime_data['crime_type'].unique().tolist())
            selected_crime_type = st.selectbox("Select crime type", crime_types)
        
        # Apply filters
        filtered_data = filter_data(
            crime_data, 
            start_date=selected_date_range[0] if len(selected_date_range) > 0 else date_min,
            end_date=selected_date_range[1] if len(selected_date_range) > 1 else date_max,
            crime_type=None if selected_crime_type == 'All' else selected_crime_type
        )
        
        st.write(f"Analyzing {len(filtered_data)} crime incidents")
        
        # Run AI analysis
        if st.button("Generate AI Analysis"):
            with st.spinner("Analyzing crime patterns with AI..."):
                analysis_results = analyze_crime_patterns(filtered_data)
                
                if not analysis_results.get("success", False):
                    st.error(f"Analysis failed: {analysis_results.get('error', 'Unknown error')}")
                else:
                    # Display patterns
                    st.subheader("Key Patterns Identified")
                    for i, pattern in enumerate(analysis_results.get("patterns", []), 1):
                        st.markdown(f"**{i}. {pattern}**")
                    
                    # Display contributing factors
                    st.subheader("Potential Contributing Factors")
                    for i, factor in enumerate(analysis_results.get("factors", []), 1):
                        st.markdown(f"**{i}.** {factor}")
                    
                    # Display recommendations
                    st.subheader("Recommended Strategies")
                    for i, strategy in enumerate(analysis_results.get("strategies", []), 1):
                        st.markdown(f"**{i}.** {strategy}")
                    
                    # Display investigation areas
                    st.subheader("Areas Needing Further Investigation")
                    for i, area in enumerate(analysis_results.get("investigation_areas", []), 1):
                        st.markdown(f"**{i}.** {area}")
                    
                    # Display predictions
                    st.subheader("Predictions for Future Trends")
                    for i, prediction in enumerate(analysis_results.get("predictions", []), 1):
                        st.markdown(f"**{i}.** {prediction}")
    
    with tab2:
        st.subheader("AI-Generated Crime Reports")
        st.info("Generate comprehensive crime reports for specific areas or time periods.")
        
        # Location selector
        st.subheader("Select Area for Report")
        
        # Create a map for location selection
        m = folium.Map(location=(11.7500, 79.7500), zoom_start=10)
        
        # Add cluster markers
        folium_static(m, width=700, height=400)
        
        # Manual coordinate input
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=11.7500, min_value=0.0, max_value=90.0, format="%.4f")
        with col2:
            longitude = st.number_input("Longitude", value=79.7500, min_value=0.0, max_value=180.0, format="%.4f")
        
        # Time period selector
        st.subheader("Select Time Period for Report")
        
        # Date range
        date_min = crime_data['date'].min()
        date_max = crime_data['date'].max()
        report_date_range = st.date_input(
            "Report date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max
        )
        
        # Generate report
        if st.button("Generate Crime Report"):
            with st.spinner("Generating comprehensive crime report..."):
                location = (latitude, longitude)
                time_period = (report_date_range[0], report_date_range[1]) if len(report_date_range) > 1 else None
                
                report = generate_crime_report(crime_data, location, time_period)
                
                if not report.get("success", False):
                    st.error(f"Report generation failed: {report.get('error', 'Unknown error')}")
                else:
                    # Display report
                    st.subheader("Crime Report")
                    
                    # Executive summary
                    st.markdown("### Executive Summary")
                    st.markdown(report.get("executive_summary", "No summary available"))
                    
                    # Key findings
                    st.markdown("### Key Findings")
                    for i, finding in enumerate(report.get("key_findings", []), 1):
                        st.markdown(f"**{i}.** {finding}")
                    
                    # High risk areas and times
                    st.markdown("### High-Risk Areas & Times")
                    high_risk = report.get("high_risk_areas_times", {})
                    
                    st.markdown("#### High-Risk Areas")
                    for area in high_risk.get("areas", []):
                        st.markdown(f"- {area}")
                    
                    st.markdown("#### High-Risk Times")
                    for time in high_risk.get("times", []):
                        st.markdown(f"- {time}")
                    
                    # Crime type analysis
                    st.markdown("### Crime Type Analysis")
                    crime_analysis = report.get("crime_type_analysis", {})
                    for crime_type, analysis in crime_analysis.items():
                        st.markdown(f"**{crime_type}**: {analysis}")
                    
                    # Recommendations
                    st.markdown("### Recommendations")
                    for i, recommendation in enumerate(report.get("recommendations", []), 1):
                        st.markdown(f"**{i}.** {recommendation}")
    
    with tab3:
        st.subheader("Specific Crime Type Analysis")
        st.info("Perform in-depth analysis of specific crime types.")
        
        # Crime type selector
        crime_types = sorted(crime_data['crime_type'].unique().tolist())
        selected_type = st.selectbox("Select crime type for analysis", crime_types)
        
        # Run specific crime analysis
        if st.button("Analyze Selected Crime"):
            with st.spinner(f"Analyzing {selected_type} incidents..."):
                crime_analysis = analyze_specific_crime(crime_data, selected_type)
                
                if not crime_analysis.get("success", False):
                    st.error(f"Analysis failed: {crime_analysis.get('error', 'Unknown error')}")
                else:
                    # Display crime type analysis
                    st.subheader(f"{selected_type.title()} Analysis")
                    
                    # Patterns
                    st.markdown("### Patterns and Characteristics")
                    for pattern in crime_analysis.get("patterns", []):
                        st.markdown(f"- {pattern}")
                    
                    # Factors
                    st.markdown("### Motivations and Contributing Factors")
                    for factor in crime_analysis.get("factors", []):
                        st.markdown(f"- {factor}")
                    
                    # Victim profile
                    st.markdown("### Victim Profile")
                    victim_profile = crime_analysis.get("victim_profile", {})
                    for key, value in victim_profile.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
                    
                    # Offender behavior
                    st.markdown("### Offender Behavioral Patterns")
                    for behavior in crime_analysis.get("offender_behavior", []):
                        st.markdown(f"- {behavior}")
                    
                    # Prevention
                    st.markdown("### Prevention Strategies")
                    for strategy in crime_analysis.get("prevention", []):
                        st.markdown(f"- {strategy}")
                    
                    # Investigation
                    st.markdown("### Investigative Recommendations")
                    for rec in crime_analysis.get("investigation", []):
                        st.markdown(f"- {rec}")