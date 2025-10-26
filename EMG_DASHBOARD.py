import pandas as pd
import altair as alt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(
    page_title="EMG RMS Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- CUSTOM CSS --------------------------- #
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: white;'>🧠 EMG Analytics</h2>
        <p style='color: white;'>Professional EMG RMS Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"], 
                                   help="Upload Excel file with sheets containing time and emg_rms_corrected_mV columns")
    
    st.markdown("---")
    
    st.subheader("⚙️ Analysis Settings")
    rolling_window = st.slider("Smoothing Window", min_value=1, max_value=100, value=20, 
                             help="Adjust the rolling window size for signal smoothing")
    
    show_raw_data = st.checkbox("Show Raw Data", value=False)
    enable_peak_detection = st.checkbox("Enable Peak Detection", value=True)
    
    st.markdown("---")
    
    st.subheader("📈 Chart Options")
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
    chart_height = st.slider("Chart Height", 400, 800, 500)

# --------------------------- MAIN HEADER --------------------------- #
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">🧠 EMG RMS Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Electromyography Signal Analysis Platform</div>', unsafe_allow_html=True)

# --------------------------- DATA PROCESSING --------------------------- #
if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        all_dfs = []
        sheet_stats = []

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Validate columns
            if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
                st.warning(f"⚠️ Sheet '{sheet_name}' skipped - missing required columns")
                continue

            # Ensure numeric and drop NaNs
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
            df = df.dropna(subset=['time', 'emg_rms_corrected_mV']).copy()

            # Reset time to start from zero
            df['time'] = df['time'] - df['time'].iloc[0]

            # Store raw data before smoothing
            df['emg_raw'] = df['emg_rms_corrected_mV']

            # Smooth EMG data
            df['emg_rms_corrected_mV'] = df['emg_rms_corrected_mV'].rolling(window=rolling_window, min_periods=1, center=True).mean()

            # Calculate statistics
            stats = {
                'sheet_name': sheet_name,
                'data_points': len(df),
                'duration': df['time'].max(),
                'mean_amplitude': df['emg_rms_corrected_mV'].mean(),
                'max_amplitude': df['emg_rms_corrected_mV'].max(),
                'min_amplitude': df['emg_rms_corrected_mV'].min(),
                'std_amplitude': df['emg_rms_corrected_mV'].std()
            }
            sheet_stats.append(stats)

            df['sheet'] = sheet_name
            all_dfs.append(df)

        if all_dfs:
            # --------------------------- OVERVIEW METRICS --------------------------- #
            st.markdown("## 📊 Dataset Overview")
            
            total_sheets = len(all_dfs)
            total_points = sum(stats['data_points'] for stats in sheet_stats)
            total_duration = max(stats['duration'] for stats in sheet_stats)
            avg_amplitude = np.mean([stats['mean_amplitude'] for stats in sheet_stats])

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📁 Sheets</h3>
                    <h2>{total_sheets}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Data Points</h3>
                    <h2>{total_points:,}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⏱️ Max Duration</h3>
                    <h2>{total_duration:.2f}s</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ Avg Amplitude</h3>
                    <h2>{avg_amplitude:.4f} mV</h2>
                </div>
                """, unsafe_allow_html=True)

            # --------------------------- DETAILED STATISTICS --------------------------- #
            st.markdown("## 📋 Sheet Statistics")
            stats_df = pd.DataFrame(sheet_stats)
            st.dataframe(stats_df.style.format({
                'duration': '{:.2f}',
                'mean_amplitude': '{:.4f}',
                'max_amplitude': '{:.4f}',
                'min_amplitude': '{:.4f}',
                'std_amplitude': '{:.4f}'
            }).background_gradient(subset=['mean_amplitude', 'max_amplitude'], cmap='Blues'), 
            use_container_width=True)

            # --------------------------- CREATE TABS --------------------------- #
            st.markdown("## 📈 Signal Visualization")
            tab_names = [f"{df['sheet'].iloc[0]} 📊" for df in all_dfs]
            tabs = st.tabs(tab_names)

            for tab, df, stats in zip(tabs, all_dfs, sheet_stats):
                with tab:
                    col1, col2 = st.columns([3, 1])
                    
                    with col2:
                        st.markdown("### 📊 Quick Stats")
                        st.metric("Data Points", f"{stats['data_points']:,}")
                        st.metric("Duration", f"{stats['duration']:.2f}s")
                        st.metric("Mean Amplitude", f"{stats['mean_amplitude']:.4f} mV")
                        st.metric("Max Amplitude", f"{stats['max_amplitude']:.4f} mV")
                        st.metric("Std Dev", f"{stats['std_amplitude']:.4f} mV")
                    
                    with col1:
                        # Create interactive plot with Plotly
                        fig = make_subplots(rows=1, cols=1)
                        
                        if show_raw_data:
                            fig.add_trace(go.Scatter(
                                x=df['time'], 
                                y=df['emg_raw'],
                                mode='lines',
                                name='Raw Signal',
                                line=dict(color='lightgray', width=1),
                                opacity=0.7
                            ))
                        
                        fig.add_trace(go.Scatter(
                            x=df['time'], 
                            y=df['emg_rms_corrected_mV'],
                            mode='lines',
                            name='Smoothed Signal',
                            line=dict(color='#1f77b4', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.1)'
                        ))
                        
                        if enable_peak_detection:
                            from scipy.signal import find_peaks
                            peaks, _ = find_peaks(df['emg_rms_corrected_mV'], height=stats['mean_amplitude'])
                            fig.add_trace(go.Scatter(
                                x=df['time'].iloc[peaks],
                                y=df['emg_rms_corrected_mV'].iloc[peaks],
                                mode='markers',
                                name='Detected Peaks',
                                marker=dict(color='red', size=8, symbol='circle')
                            ))
                        
                        fig.update_layout(
                            title=f"EMG RMS Signal - {df['sheet'].iloc[0]}",
                            xaxis_title="Time (s)",
                            yaxis_title="EMG RMS (mV)",
                            height=chart_height,
                            template=chart_theme,
                            showlegend=True,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # --------------------------- DATA TABLE --------------------------- #
                    with st.expander("📋 View Data Table"):
                        st.dataframe(df[['time', 'emg_rms_corrected_mV']].style.format({
                            'time': '{:.2f}',
                            'emg_rms_corrected_mV': '{:.6f}'
                        }), use_container_width=True)

            # --------------------------- COMPARISON VIEW --------------------------- #
            st.markdown("## 🔄 Signal Comparison")
            
            comparison_fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, (df, stats) in enumerate(zip(all_dfs, sheet_stats)):
                color = colors[i % len(colors)]
                comparison_fig.add_trace(go.Scatter(
                    x=df['time'], 
                    y=df['emg_rms_corrected_mV'],
                    mode='lines',
                    name=f"{df['sheet'].iloc[0]} (Max: {stats['max_amplitude']:.4f} mV)",
                    line=dict(color=color, width=2),
                    opacity=0.8
                ))
            
            comparison_fig.update_layout(
                title="All EMG RMS Signals Comparison",
                xaxis_title="Time (s)",
                yaxis_title="EMG RMS (mV)",
                height=500,
                template=chart_theme,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(comparison_fig, use_container_width=True)

            # --------------------------- EXPORT OPTIONS --------------------------- #
            st.markdown("## 💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download Processed Data (CSV)"):
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="processed_emg_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Download Statistics Report"):
                    stats_csv = pd.DataFrame(sheet_stats).to_csv(index=False)
                    st.download_button(
                        label="Download Stats Report",
                        data=stats_csv,
                        file_name="emg_statistics_report.csv",
                        mime="text/csv"
                    )

        else:
            st.error("❌ No valid sheets found in the uploaded file. Please check the column names.")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure the Excel file is not corrupted and contains the required columns.")

else:
    # --------------------------- LANDING PAGE --------------------------- #
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>Welcome to EMG RMS Analytics</h1>
        <p style='font-size: 1.2rem; margin-bottom: 2rem;'>Upload your Excel file to begin professional EMG signal analysis</p>
        <div style='font-size: 4rem; margin-bottom: 2rem;'>📊🧠⚡</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 📋 Required Excel Format:
        
        Each sheet should contain these columns:
        - **time**: Time values in seconds
        - **emg_rms_corrected_mV**: EMG RMS amplitude in millivolts
        
        ### 🚀 Features:
        - Multi-sheet analysis
        - Advanced signal smoothing
        - Peak detection
        - Statistical analysis
        - Interactive visualizations
        - Data export capabilities
        """)
    
    st.markdown("---")
    st.info("👈 Please upload an Excel file using the sidebar to visualize your EMG data.")

# --------------------------- FOOTER --------------------------- #
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>EMG RMS Analytics Dashboard • Professional Biomedical Signal Analysis</p>
</div>
""", unsafe_allow_html=True)