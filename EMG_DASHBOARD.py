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

# --------------------------- IMAGE LOADING HELPERS --------------------------- #
def load_image_from_url(url):
    """Safely load an image from a URL using requests."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.warning(f"⚠️ Could not load image from {url}: {e}")
        return None

# --------------------------- MUSCLE IMAGES & ANIMATIONS --------------------------- #
def get_muscle_gif():
    """Return a PIL Image of muscle animation GIF"""
    muscle_gif_url = "https://upload.wikimedia.org/wikipedia/commons/e/e0/Bicep_tricep.gif"
    return load_image_from_url(muscle_gif_url)

def load_muscle_image():
    """Load muscle anatomy image"""
    muscle_img_url = "https://www.ncbi.nlm.nih.gov/books/NBK519538/bin/article-18251-f2.jpg"
    return load_image_from_url(muscle_img_url)

def get_physiology_image():
    """Load sarcomere contraction diagram"""
    physiology_img_url = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Sarcomere_relaxed_contracted.PNG"
    return load_image_from_url(physiology_img_url)

# --------------------------- CUSTOM CSS FOR ANIMATIONS --------------------------- #
st.markdown("""
<style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
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
        color: #333; /* Darker text for readability */
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.title("💪 Muscle EMG Analytics")

    # Muscle animation in sidebar
    muscle_gif = get_muscle_gif()
    if muscle_gif:
        st.image(muscle_gif, caption="Antagonistic Pair: Biceps & Triceps")

    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])
    st.markdown("---")
    
    st.subheader("Processing Settings")
    rolling_window = st.slider("Smoothing Window", min_value=1, max_value=100, value=20)
    show_raw_data = st.checkbox("Show Raw Data", value=True)
    normalize_data = st.checkbox("Normalize Data (Show as % of Max)", value=False)
    
    st.markdown("---")
    st.subheader("💡 About This Dashboard")
    st.info("This tool analyzes Electromyography (EMG) data to quantify muscle activation. Upload an Excel file where each sheet represents a muscle or trial.")

# --------------------------- MAIN HEADER --------------------------- #
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💪 EMG Muscle Analytics Dashboard")
    st.markdown("Analyze muscle activity through EMG signal processing, designed for sports science and rehabilitation.")
with col2:
    muscle_img = load_muscle_image()
    if muscle_img:
        st.image(muscle_img, caption="Anatomy of the Biceps Brachii", use_column_width=True)

# --------------------------- MUSCLE INFO EXPANDER --------------------------- #
with st.expander("🧬 Muscle Physiology & Weightlifting: What EMG Shows", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Motor Unit Recruitment")
        st.markdown("""
        Your muscles are composed of **Motor Units (MUs)**, each consisting of a single motor neuron and the bundle of muscle fibers it controls.
        
        - **Henneman's Size Principle:** Your nervous system is incredibly efficient. It recruits MUs in a specific order:
            1.  **Small MUs (Type I, Slow-Twitch):** Recruited first for low-effort tasks like posture or walking. They are fatigue-resistant.
            2.  **Large MUs (Type II, Fast-Twitch):** Recruited last, only when high force is needed (e.g., lifting a heavy weight, sprinting). They are powerful but fatigue quickly.
        
        - **Why Weightlifting Works:** To lift heavy weights, your brain *must* recruit these large, high-threshold Type II motor units. This high level of recruitment creates the mechanical tension that signals the muscle to adapt and grow (hypertrophy).
        
        - **What Your EMG Graph Shows:** The amplitude (height) of the EMG signal represents the **sum of all active motor units** near the electrode. A higher peak on your graph means your brain is sending a stronger signal, recruiting *more* MUs and/or firing them *faster* to meet the high force demand.
        """)
    
    with col2:
        st.subheader("The Contractile Unit")
        physiology_img = get_physiology_image()
        if physiology_img:
            st.image(physiology_img, caption="A sarcomere: the basic contractile unit of a muscle fiber.")
        st.markdown("""
        - **Sarcomere:** This is the microscopic engine inside your muscle fibers.
        - **Actin & Myosin:** These are protein filaments. During a contraction, the 'Myosin' heads pull the 'Actin' filaments closer together, shortening the entire sarcomere.
        - **EMG Signal:** The electrical signal measured by EMG is the **"Go" command** (an *action potential*) that travels along the muscle fiber, telling the sarcomeres to contract.
        """)

# --------------------------- DATA PROCESSING --------------------------- #
if uploaded_file:
    with st.spinner('🔄 Processing muscle data...'):
        xls = pd.ExcelFile(uploaded_file)
        all_dfs = []
        sheet_stats = []
        max_global_duration = 0

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)

                if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
                    st.warning(f"Sheet '{sheet_name}' skipped: required 'time' or 'emg_rms_corrected_mV' columns missing.")
                    continue

                df['time'] = pd.to_numeric(df['time'], errors='coerce')
                df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
                df = df.dropna(subset=['time', 'emg_rms_corrected_mV']).copy()

                if df.empty:
                    st.warning(f"Sheet '{sheet_name}' skipped: no valid data after cleanup.")
                    continue
                
                df['time'] = df['time'] - df['time'].iloc[0]
                df['emg_raw'] = df['emg_rms_corrected_mV']
                df['emg_processed'] = df['emg_raw'].rolling(window=rolling_window, min_periods=1, center=True).mean()

                if normalize_data:
                    max_val = df['emg_processed'].max()
                    if max_val > 0:
                        df['emg_processed'] = df['emg_processed'] / max_val
                        df['emg_raw'] = df['emg_raw'] / max_val
                    else:
                        st.warning(f"Sheet '{sheet_name}' has max amplitude of 0. Cannot normalize.")

                stats = {
                    'sheet_name': sheet_name,
                    'data_points': len(df),
                    'duration': df['time'].max(),
                    'mean_amplitude': df['emg_processed'].mean(),
                    'max_amplitude': df['emg_processed'].max(),
                    'min_amplitude': df['emg_processed'].min(),
                    'std_amplitude': df['emg_processed'].std()
                }
                sheet_stats.append(stats)
                df['sheet'] = sheet_name
                all_dfs.append(df)

                if stats['duration'] > max_global_duration:
                    max_global_duration = stats['duration']

            except Exception as e:
                st.error(f"Error processing sheet '{sheet_name}': {e}")

    if all_dfs:
        st.success(f"✅ Successfully processed {len(all_dfs)} muscle data sets!")
        
        st.subheader("📊 Muscle Activity Overview")
        num_sheets = len(sheet_stats)
        cols = st.columns(num_sheets if num_sheets <= 4 else 4)
        
        for idx, stats in enumerate(sheet_stats):
            col = cols[idx % len(cols)]
            with col:
                unit = "% Max" if normalize_data else "mV"
                st.markdown(f"""
                <div class="muscle-card">
                    <h4>💪 {stats['sheet_name']}</h4>
                    <div class="stats-card">
                        <strong>Duration:</strong> {stats['duration']:.2f}s<br>
                        <strong>Mean Amp:</strong> {stats['mean_amplitude']:.3f} {unit}<br>
                        <strong>Max Amp:</strong> {stats['max_amplitude']:.3f} {unit}<br>
                        <strong>Data Points:</strong> {stats['data_points']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📈 EMG Signals Analysis")
        st.markdown(f"**Note:** All graphs are plotted on a consistent time scale (0 to {max_global_duration:.2f}s) for easy comparison.")
        unit_label = "EMG Amplitude (% Max)" if normalize_data else "EMG RMS Amplitude (mV)"

        for df, stats in zip(all_dfs, sheet_stats):
            with st.expander(f"🔬 Detailed Analysis: {stats['sheet_name']}", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    if show_raw_data:
                        ax.plot(df['time'], df['emg_raw'], color='lightgray', linestyle='--', alpha=0.7, label='Raw Signal')
                    ax.plot(df['time'], df['emg_processed'], color='#ff6b6b', linewidth=2.5, label=f'Processed (Window={rolling_window})')
                    ax.fill_between(df['time'], df['emg_processed'], alpha=0.3, color='#ff6b6b')
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel(unit_label)
                    ax.set_title(f"Muscle Activity: {stats['sheet_name']}")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(left=0, right=max_global_duration)
                    ax.legend()
                    ax.set_facecolor('#f8f9fa')
                    fig.patch.set_facecolor('#f8f9fa')
                    st.pyplot(fig)
                with col2:
                    unit = "%" if normalize_data else "mV"
                    st.markdown(f"""
                    <div style="background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;">
                        <h4>📋 Signal Info</h4>
                        <strong>Muscle:</strong> {stats['sheet_name']}<br>
                        <strong>Recording:</strong> {stats['duration']:.1f}s<br>
                        <strong>Peak:</strong> {stats['max_amplitude']:.3f} {unit}<br>
                        <strong>Avg:</strong> {stats['mean_amplitude']:.3f} {unit}<br>
                        <strong>Variability:</strong> {stats['std_amplitude']:.3f}
                    </div>
                    """, unsafe_allow_html=True)

        st.subheader("💾 Export Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.rename(columns={'emg_processed': f'emg_processed_{rolling_window}window'})
            csv = combined_df.to_csv(index=False)
            st.download_button("📥 Download All Processed Data (CSV)", csv, "muscle_emg_analysis_all.csv", "text/csv", use_container_width=True)
        with col2:
            summary_df = pd.DataFrame(sheet_stats)
            summary_csv = summary_df.to_csv(index=False)
            st.download_button("📊 Download Summary Statistics (CSV)", summary_csv, "muscle_analysis_summary.csv", "text/csv", use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 Key Performance Insights")
        if sheet_stats:
            most_active = max(sheet_stats, key=lambda x: x['max_amplitude'])
            avg_activity = np.mean([s['mean_amplitude'] for s in sheet_stats])
            unit = "% Max" if normalize_data else "mV"
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Most Active Muscle (Peak):** {most_active['sheet_name']} (Peak: {most_active['max_amplitude']:.3f} {unit})")
            with col2:
                st.info(f"**Average Muscle Activity (Mean):** {avg_activity:.3f} {unit}")
    else:
        st.error("❌ No valid muscle data found. Please check your Excel file format.")
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Each sheet should contain (at minimum):**
            ```
            time     emg_rms_corrected_mV
            0.0      0.012
            0.001    0.045
            0.002    0.078
            ...
            ```
            """)
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("""
        ## 🚀 Get Started
        
        Upload an Excel file (.xlsx) to analyze muscle EMG data.
        
        ### 📁 Expected Format:
        - Each sheet = one muscle or trial
        - Columns required:
            1. **time (s)**
            2. **emg_rms_corrected_mV (mV)**
        """)
    with col2:
        st.markdown("<div style='text-align: center;'><h3>💪 Muscle Contraction</h3></div>", unsafe_allow_html=True)
        muscle_gif = get_muscle_gif()
        if muscle_gif:
            st.image(muscle_gif, use_column_width=True, caption="Biceps (flexor) and Triceps (extensor) working as an antagonistic pair.")

# --------------------------- FOOTER --------------------------- #
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>💪 EMG weight  Analytics Dashboard | Built for Sports Science & Rehabilitation Research</div>", 
    unsafe_allow_html=True
)
