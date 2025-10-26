import pandas as pd
import altair as alt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import requests

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(
    page_title="EMG Muscle Analytics Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- CUSTOM CSS --------------------------- #
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .muscle-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 10px 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #96CEB4 0%, #FFEAA7 100%);
        border-radius: 15px 15px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
        color: #2d3436;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%) !important;
        color: white;
    }
    .animated-bg {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- MUSCLE IMAGES --------------------------- #
def get_muscle_image(muscle_name):
    """Return appropriate muscle image based on sheet name"""
    muscle_images = {
        'biceps': '💪', 'triceps': '🦵', 'quadriceps': '🦵', 'hamstring': '🦵',
        'deltoid': '👤', 'pectoral': '👤', 'abdominal': '👤', 'gluteus': '🍑',
        'gastrocnemius': '🦵', 'soleus': '🦵', 'trapezius': '👤'
    }
    
    for key, emoji in muscle_images.items():
        if key in muscle_name.lower():
            return emoji
    return '💪'  # default muscle emoji

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <div style='font-size: 3rem;'>💪</div>
        <h2 style='color: white;'>Muscle EMG Analytics</h2>
        <p style='color: white;'>Professional Muscle Activity Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"], 
                                   help="Upload Excel file with EMG data sheets")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Analysis Settings")
    rolling_window = st.slider("Smoothing Window", min_value=1, max_value=100, value=20)
    
    col1, col2 = st.columns(2)
    with col1:
        show_raw_data = st.checkbox("Show Raw Data", value=True)
        enable_peak_detection = st.checkbox("Peak Detection", value=True)
    with col2:
        show_muscle_info = st.checkbox("Muscle Info", value=True)
        normalize_data = st.checkbox("Normalize", value=False)
    
    st.markdown("---")
    
    st.markdown("### 🎨 Chart Options")
    chart_theme = st.selectbox("Theme", ["plotly", "plotly_white", "plotly_dark", "seaborn", "ggplot2"])
    chart_height = st.slider("Chart Height", 400, 800, 500)
    
    st.markdown("---")
    
    # Muscle selection for detailed view
    st.markdown("### 🔍 Focus Muscle")
    muscle_focus = st.selectbox("Select muscle for detailed analysis", 
                               ["Biceps", "Triceps", "Quadriceps", "Hamstring", "Deltoid", "All Muscles"])

# --------------------------- MAIN HEADER --------------------------- #
st.markdown("""
<div class="animated-bg">
    <div style='text-align: center; color: white;'>
        <h1 class="main-header">💪 EMG Muscle Analytics Dashboard</h1>
        <h3 style='color: white;'>Advanced Muscle Activity & Performance Analysis</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------- MUSCLE ANATOMY SECTION --------------------------- #
if show_muscle_info:
    st.markdown("## 🏃‍♂️ Muscle Anatomy Reference")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="muscle-card">
            <div style='font-size: 3rem;'>💪</div>
            <h4>Biceps Brachii</h4>
            <p>Primary elbow flexor</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="muscle-card" style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);">
            <div style='font-size: 3rem;'>🦵</div>
            <h4>Quadriceps</h4>
            <p>Knee extension</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="muscle-card" style="background: linear-gradient(135deg, #FFE66D 0%, #F9C74F 100%);">
            <div style='font-size: 3rem;'>👤</div>
            <h4>Deltoid</h4>
            <p>Shoulder movement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="muscle-card" style="background: linear-gradient(135deg, #96CEB4 0%, #6B8E23 100%);">
            <div style='font-size: 3rem;'>🍑</div>
            <h4>Gluteus</h4>
            <p>Hip extension</p>
        </div>
        """, unsafe_allow_html=True)

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
            df['emg_rms_corrected_mV'] = df['emg_rms_corrected_mV'].rolling(
                window=rolling_window, min_periods=1, center=True).mean()

            # Normalize if requested
            if normalize_data:
                df['emg_rms_corrected_mV'] = df['emg_rms_corrected_mV'] / df['emg_rms_corrected_mV'].max()

            # Calculate statistics
            stats = {
                'sheet_name': sheet_name,
                'muscle_emoji': get_muscle_image(sheet_name),
                'data_points': len(df),
                'duration': df['time'].max(),
                'mean_amplitude': df['emg_rms_corrected_mV'].mean(),
                'max_amplitude': df['emg_rms_corrected_mV'].max(),
                'min_amplitude': df['emg_rms_corrected_mV'].min(),
                'std_amplitude': df['emg_rms_corrected_mV'].std(),
                'muscle_activity': 'High' if df['emg_rms_corrected_mV'].mean() > 0.1 else 'Low'
            }
            sheet_stats.append(stats)

            df['sheet'] = sheet_name
            df['muscle_emoji'] = get_muscle_image(sheet_name)
            all_dfs.append(df)

        if all_dfs:
            # --------------------------- COLORFUL METRICS --------------------------- #
            st.markdown("## 📊 Muscle Activity Overview")
            
            total_sheets = len(all_dfs)
            total_points = sum(stats['data_points'] for stats in sheet_stats)
            total_duration = max(stats['duration'] for stats in sheet_stats)
            avg_amplitude = np.mean([stats['mean_amplitude'] for stats in sheet_stats])

            col1, col2, col3, col4 = st.columns(4)
            
            gradient_colors = [
                "linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%)",
                "linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%)",
                "linear-gradient(135deg, #45B7D1 0%, #96CEB4 100%)",
                "linear-gradient(135deg, #FFEAA7 0%, #F9C74F 100%)"
            ]
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background: {gradient_colors[0]};">
                    <h3>📁 Active Muscles</h3>
                    <h2>{total_sheets}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: {gradient_colors[1]};">
                    <h3>📈 Data Points</h3>
                    <h2>{total_points:,}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: {gradient_colors[2]};">
                    <h3>⏱️ Max Duration</h3>
                    <h2>{total_duration:.2f}s</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="background: {gradient_colors[3]};">
                    <h3>⚡ Avg Activity</h3>
                    <h2>{avg_amplitude:.4f} mV</h2>
                </div>
                """, unsafe_allow_html=True)

            # --------------------------- MUSCLE ACTIVITY GRID --------------------------- #
            st.markdown("## 🎯 Muscle Performance Grid")
            
            cols = st.columns(4)
            for idx, stats in enumerate(sheet_stats):
                with cols[idx % 4]:
                    activity_color = "#FF6B6B" if stats['muscle_activity'] == 'High' else "#4ECDC4"
                    st.markdown(f"""
                    <div style="background: {activity_color}; padding: 1rem; border-radius: 15px; color: white; text-align: center; margin: 0.5rem 0;">
                        <div style="font-size: 2rem;">{stats['muscle_emoji']}</div>
                        <h4>{stats['sheet_name']}</h4>
                        <p>Activity: {stats['muscle_activity']}</p>
                        <p>Max: {stats['max_amplitude']:.4f} mV</p>
                    </div>
                    """, unsafe_allow_html=True)

            # --------------------------- INTERACTIVE CHARTS --------------------------- #
            st.markdown("## 📈 Real-time Muscle Signals")
            
            # Create colorful tabs with muscle emojis
            tab_names = [f"{df['muscle_emoji'].iloc[0]} {df['sheet'].iloc[0]}" for df in all_dfs]
            tabs = st.tabs(tab_names)

            for tab, df, stats in zip(tabs, all_dfs, sheet_stats):
                with tab:
                    col1, col2 = st.columns([3, 1])
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; border-radius: 15px; color: white;">
                            <h3>📊 {stats['muscle_emoji']} Muscle Stats</h3>
                            <p>📏 Points: {stats['data_points']:,}</p>
                            <p>⏱️ Duration: {stats['duration']:.2f}s</p>
                            <p>📈 Mean: {stats['mean_amplitude']:.4f} mV</p>
                            <p>🔥 Max: {stats['max_amplitude']:.4f} mV</p>
                            <p>📊 Std: {stats['std_amplitude']:.4f} mV</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col1:
                        # Create colorful interactive plot
                        fig = go.Figure()
                        
                        if show_raw_data:
                            fig.add_trace(go.Scatter(
                                x=df['time'], y=df['emg_raw'],
                                mode='lines', name='Raw Signal',
                                line=dict(color='lightgray', width=1, dash='dot'),
                                opacity=0.6
                            ))
                        
                        # Color based on amplitude
                        color_scale = ['#4ECDC4', '#45B7D1', '#FFE66D', '#FF6B6B']
                        fig.add_trace(go.Scatter(
                            x=df['time'], y=df['emg_rms_corrected_mV'],
                            mode='lines', name='Processed Signal',
                            line=dict(color=color_scale[2], width=4),
                            fill='tozeroy',
                            fillcolor='rgba(255, 230, 109, 0.3)'
                        ))
                        
                        if enable_peak_detection:
                            from scipy.signal import find_peaks
                            peaks, _ = find_peaks(df['emg_rms_corrected_mV'], 
                                                height=stats['mean_amplitude'] * 1.5)
                            fig.add_trace(go.Scatter(
                                x=df['time'].iloc[peaks],
                                y=df['emg_rms_corrected_mV'].iloc[peaks],
                                mode='markers+text',
                                name='Muscle Peaks',
                                marker=dict(color='#FF6B6B', size=12, symbol='star'),
                                text=[f'Peak: {val:.3f}' for val in df['emg_rms_corrected_mV'].iloc[peaks]],
                                textposition="top center"
                            ))
                        
                        fig.update_layout(
                            title=f"{stats['muscle_emoji']} {df['sheet'].iloc[0]} - Muscle Activity",
                            xaxis_title="Time (s)",
                            yaxis_title="EMG RMS (mV)",
                            height=chart_height,
                            template=chart_theme,
                            showlegend=True,
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

            # --------------------------- MUSCLE COMPARISON --------------------------- #
            st.markdown("## 🔄 Cross-Muscle Comparison")
            
            comparison_fig = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', "#D2B41D", '#96CEB4', '#FFA5A5']
            
            for i, (df, stats) in enumerate(zip(all_dfs, sheet_stats)):
                color = colors[i % len(colors)]
                comparison_fig.add_trace(go.Scatter(
                    x=df['time'], y=df['emg_rms_corrected_mV'],
                    mode='lines', name=f"{stats['muscle_emoji']} {df['sheet'].iloc[0]}",
                    line=dict(color=color, width=3),
                    opacity=0.8
                ))
            
            comparison_fig.update_layout(
                title="🏃‍♂️ Multi-Muscle Activity Comparison",
                xaxis_title="Time (s)",
                yaxis_title="EMG RMS (mV)",
                height=500,
                template=chart_theme,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(comparison_fig, use_container_width=True)

            # --------------------------- EXPORT SECTION --------------------------- #
            st.markdown("## 💾 Export Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download All Data", use_container_width=True):
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="muscle_emg_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📊 Download Report", use_container_width=True):
                    stats_csv = pd.DataFrame(sheet_stats).to_csv(index=False)
                    st.download_button(
                        label="📊 Download Stats",
                        data=stats_csv,
                        file_name="muscle_performance_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        else:
            st.error("❌ No valid muscle data found. Please check your Excel file format.")

    except Exception as e:
        st.error(f"❌ Error processing muscle data: {str(e)}")

else:
    # --------------------------- VIBRANT LANDING PAGE --------------------------- #
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%); border-radius: 20px; color: white;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>Welcome to Muscle EMG Analytics! 🏃‍♂️</h1>
        <p style='font-size: 1.5rem; margin-bottom: 2rem;'>Upload your EMG data to unlock powerful muscle activity insights</p>
        <div style='font-size: 5rem; margin-bottom: 2rem;'>
            💪 🦵 👤 🍑 🏃‍♂️
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Features Include:
        
        <div style='background: linear-gradient(135deg, #FFE66D 0%, #F9C74F 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
        <h4>🎨 Visual Analytics</h4>
        <p>• Color-coded muscle activity</p>
        <p>• Interactive signal processing</p>
        <p>• Real-time peak detection</p>
        </div>
        
        <div style='background: linear-gradient(135deg, #96CEB4 0%, #6B8E23 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white;'>
        <h4>📊 Muscle Intelligence</h4>
        <p>• Cross-muscle comparison</p>
        <p>• Activity level classification</p>
        <p>• Performance metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 📋 Data Requirements:
        
        <div style='background: linear-gradient(135deg, #45B7D1 0%, #96CEB4 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white;'>
        <h4>📁 Excel Format</h4>
        <p>• Each sheet = One muscle</p>
        <p>• Required columns:</p>
        <p>  - <strong>time</strong> (seconds)</p>
        <p>  - <strong>emg_rms_corrected_mV</strong></p>
        </div>
        
        <div style='background: linear-gradient(135deg, #FFA5A5 0%, #FF6B6B 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white;'>
        <h4>🔬 Analysis Ready</h4>
        <p>• Automatic signal smoothing</p>
        <p>• Statistical summaries</p>
        <p>• Export capabilities</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------- COLORFUL FOOTER --------------------------- #
st.markdown("""
---
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
    <h3>💪 EMG Muscle Analytics Dashboard</h3>
    <p>Advanced Biomedical Signal Processing & Muscle Performance Analysis</p>
    <div style='font-size: 2rem; margin-top: 1rem;'>
        🏃‍♂️ 💪 🦵 👤 🍑 📊 🎯
    </div>
</div>
""", unsafe_allow_html=True)