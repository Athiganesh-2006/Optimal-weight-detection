import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
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

# --------------------------- MUSCLE IMAGES & ANIMATIONS --------------------------- #
def get_muscle_gif():
    """Return a base64 encoded muscle animation GIF"""
    # You can replace this with your own muscle animation GIF
    # For demo purposes, using a placeholder - in practice, you'd use a local file or URL
    muscle_gif_url = "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif"  # Muscle contraction GIF
    return muscle_gif_url

def load_muscle_image():
    """Load muscle anatomy image"""
    # Placeholder - replace with your actual image path or URL
    muscle_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Muscles_anterior_labeled.png/800px-Muscles_anterior_labeled.png"
    return muscle_img_url

# --------------------------- CUSTOM CSS FOR ANIMATIONS --------------------------- #
st.markdown("""
<style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .muscle-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .stats-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff4b4b;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.title("💪 Muscle EMG Analytics")
    
    # Muscle animation in sidebar
    muscle_gif = get_muscle_gif()
    st.markdown(f'<div class="pulse-animation"><img src="{muscle_gif}" width="100%"></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])
    st.markdown("---")
    
    st.subheader("Processing Settings")
    rolling_window = st.slider("Smoothing Window", min_value=1, max_value=100, value=20)
    show_raw_data = st.checkbox("Show Raw Data", value=True)
    normalize_data = st.checkbox("Normalize Data", value=False)
    
    st.markdown("---")
    st.subheader("💡 About EMG")
    st.info("Electromyography (EMG) measures muscle electrical activity during contraction and relaxation.")

# --------------------------- MAIN HEADER --------------------------- #
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💪 EMG Muscle Analytics Dashboard")
    st.markdown("Analyze muscle activity through EMG signal processing")
with col2:
    muscle_img = load_muscle_image()
    st.image(muscle_img, caption="Human Muscle Anatomy", use_column_width=True)

# --------------------------- MUSCLE INFO EXPANDER --------------------------- #
with st.expander("🧬 Muscle Physiology Information"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("EMG Basics")
        st.markdown("""
        - **EMG** = Electromyography
        - Measures electrical potential
        - Detects muscle activation
        - Quantifies muscle effort
        """)
    
    with col2:
        st.subheader("Signal Types")
        st.markdown("""
        - **Raw EMG**: Raw electrical signals
        - **RMS**: Root Mean Square
        - **Smoothed**: Moving average
        - **Normalized**: Scaled to max value
        """)
    
    with col3:
        st.subheader("Applications")
        st.markdown("""
        - Rehabilitation
        - Sports science
        - Neuromuscular research
        - Clinical diagnostics
        """)

# --------------------------- DATA PROCESSING --------------------------- #
if uploaded_file:
    # Add loading animation
    with st.spinner('🔄 Processing muscle data...'):
        xls = pd.ExcelFile(uploaded_file)
        all_dfs = []
        sheet_stats = []

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Validate columns
            if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
                st.warning(f"Sheet '{sheet_name}' skipped: required columns missing.")
                continue

            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
            df = df.dropna(subset=['time', 'emg_rms_corrected_mV']).copy()

            df['time'] = df['time'] - df['time'].iloc[0]  # reset time to zero
            df['emg_raw'] = df['emg_rms_corrected_mV']

            # Smoothing
            df['emg_rms_corrected_mV'] = df['emg_rms_corrected_mV'].rolling(
                window=rolling_window, min_periods=1, center=True).mean()

            # Normalize if requested
            if normalize_data:
                df['emg_rms_corrected_mV'] = df['emg_rms_corrected_mV'] / df['emg_rms_corrected_mV'].max()

            # Calculate stats
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
        # Success message with animation
        st.success(f"✅ Successfully processed {len(all_dfs)} muscle data sets!")
        
        # Muscle Activity Overview with enhanced visualization
        st.subheader("📊 Muscle Activity Overview")
        
        # Create columns for stats cards
        cols = st.columns(len(sheet_stats))
        for idx, (col, stats) in enumerate(zip(cols, sheet_stats)):
            with col:
                st.markdown(f"""
                <div class="muscle-card">
                    <h4>💪 {stats['sheet_name']}</h4>
                    <div class="stats-card">
                        <strong>Duration:</strong> {stats['duration']:.2f}s<br>
                        <strong>Mean Amp:</strong> {stats['mean_amplitude']:.3f} mV<br>
                        <strong>Max Amp:</strong> {stats['max_amplitude']:.3f} mV<br>
                        <strong>Data Points:</strong> {stats['data_points']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # EMG Signals Visualization
        st.subheader("📈 EMG Signals Analysis")
        
        for df, stats in zip(all_dfs, sheet_stats):
            # Create expandable section for each muscle
            with st.expander(f"🔬 Detailed Analysis: {stats['sheet_name']}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    
                    # Create more visually appealing plot
                    if show_raw_data:
                        ax.plot(df['time'], df['emg_raw'], color='lightgray', 
                               linestyle='--', alpha=0.7, label='Raw Signal')
                    
                    # Main processed signal with gradient fill
                    ax.plot(df['time'], df['emg_rms_corrected_mV'], 
                           color='#ff6b6b', linewidth=2.5, label='Processed EMG')
                    
                    # Fill under the curve for better visualization
                    ax.fill_between(df['time'], df['emg_rms_corrected_mV'], 
                                  alpha=0.3, color='#ff6b6b')
                    
                    ax.set_xlabel("Time (s)", fontsize=12)
                    ax.set_ylabel("EMG RMS Amplitude (mV)", fontsize=12)
                    ax.set_title(f"Muscle Activity: {stats['sheet_name']}", 
                               fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_facecolor('#f8f9fa')
                    fig.patch.set_facecolor('#f8f9fa')
                    
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("""
                    <div style="background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;">
                        <h4>📋 Signal Info</h4>
                        <strong>Muscle:</strong> {}<br>
                        <strong>Recording:</strong> {:.1f}s<br>
                        <strong>Peak:</strong> {:.3f} mV<br>
                        <strong>Avg:</strong> {:.3f} mV<br>
                        <strong>Variability:</strong> {:.3f}
                    </div>
                    """.format(
                        stats['sheet_name'],
                        stats['duration'],
                        stats['max_amplitude'],
                        stats['mean_amplitude'],
                        stats['std_amplitude']
                    ), unsafe_allow_html=True)

        # Enhanced Export Section
        st.subheader("💾 Export Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Data as CSV",
                data=csv,
                file_name="muscle_emg_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Generate summary report
            summary_data = {
                'Muscle': [s['sheet_name'] for s in sheet_stats],
                'Duration_s': [s['duration'] for s in sheet_stats],
                'Mean_Amplitude_mV': [s['mean_amplitude'] for s in sheet_stats],
                'Max_Amplitude_mV': [s['max_amplitude'] for s in sheet_stats],
                'Data_Points': [s['data_points'] for s in sheet_stats]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Summary Report",
                data=summary_csv,
                file_name="muscle_analysis_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Add muscle physiology insights
        st.markdown("---")
        st.subheader("🧠 Muscle Activity Insights")
        
        # Find the most active muscle
        if sheet_stats:
            most_active = max(sheet_stats, key=lambda x: x['max_amplitude'])
            avg_activity = np.mean([s['mean_amplitude'] for s in sheet_stats])
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Most Active Muscle:** {most_active['sheet_name']} "
                       f"(Peak: {most_active['max_amplitude']:.3f} mV)")
            
            with col2:
                st.info(f"**Average Muscle Activity:** {avg_activity:.3f} mV")

    else:
        st.error("""
        ❌ No valid muscle data found. 
        
        Please check that your Excel file contains sheets with:
        - 'time' column (numeric values)
        - 'emg_rms_corrected_mV' column (numeric values)
        """)
        
        # Show example format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Each sheet should contain:**
            ```
            time    emg_rms_corrected_mV
            0.0     0.012
            0.1     0.045
            0.2     0.078
            ...     ...
            ```
            """)

else:
    # Enhanced upload prompt with muscle animation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        ## 🚀 Get Started
        
        Upload an Excel file to analyze muscle EMG data. Each sheet should contain:
        
        - **time**: Time values in seconds
        - **emg_rms_corrected_mV**: EMG amplitude in millivolts
        
        ### 📁 Expected Format:
        Multiple sheets, each representing different muscle measurements or conditions.
        """)
    
    with col2:
        # Display muscle animation
        st.markdown("""
        <div style="text-align: center;">
            <h3>💪 Muscle Contraction</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(get_muscle_gif(), use_column_width=True)

# --------------------------- FOOTER --------------------------- #
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "💪 EMG Muscle Analytics Dashboard | Built for Sports Science & Rehabilitation Research"
    "</div>", 
    unsafe_allow_html=True
)