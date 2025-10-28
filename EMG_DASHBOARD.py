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

# --------------------------- IMAGE LOADING FUNCTION --------------------------- #
@st.cache_data(show_spinner=False)
def safe_load_image(url: str, fallback_path: str = None):
    """Safely load image from URL with fallback."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.warning(f"⚠️ Could not load image from {url}. ({e})")
        if fallback_path:
            try:
                return Image.open(fallback_path)
            except Exception as e2:
                st.error(f"⚠️ Fallback image also failed: {e2}")
        return None

# --------------------------- IMAGE SOURCES --------------------------- #
muscle_gif_url = "https://upload.wikimedia.org/wikipedia/commons/7/7e/Biceps_muscle_animation.gif"
anatomy_img_url = "https://upload.wikimedia.org/wikipedia/commons/3/32/Biceps_brachii.png"
sarcomere_img_url = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Sarcomere_relaxed_contracted.PNG"

# --------------------------- CUSTOM STYLING --------------------------- #
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
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.title("💪 Muscle EMG Analytics")

    # Load and display muscle GIF
    muscle_gif = safe_load_image(muscle_gif_url)
    if muscle_gif:
        st.image(muscle_gif, caption="Antagonistic Pair: Biceps & Triceps", use_column_width=True)

    # File uploader
    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])
    st.markdown("---")

    # Settings
    st.subheader("⚙️ Processing Settings")
    rolling_window = st.slider("Smoothing Window", min_value=1, max_value=100, value=20)
    show_raw_data = st.checkbox("Show Raw Data", value=True)
    normalize_data = st.checkbox("Normalize Data (Show as % of Max)", value=False)
    
    st.markdown("---")
    st.subheader("💡 About This Dashboard")
    st.info("Analyze Electromyography (EMG) data to quantify muscle activation. Each Excel sheet = one muscle or trial.")

# --------------------------- MAIN HEADER --------------------------- #
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💪 EMG Muscle Analytics Dashboard")
    st.markdown("Analyze muscle activity through EMG signal processing, designed for sports science and rehabilitation.")
with col2:
    anatomy_img = safe_load_image(anatomy_img_url)
    if anatomy_img:
        st.image(anatomy_img, caption="Anatomy of the Biceps Brachii", use_column_width=True)

# --------------------------- INFO SECTION --------------------------- #
with st.expander("🧬 Muscle Physiology & Weightlifting: What EMG Shows"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Motor Unit Recruitment")
        st.markdown("""
        Muscles are made up of **Motor Units (MUs)** — a single neuron and the muscle fibers it controls.

        - **Small MUs (Type I, Slow-Twitch):** Used for endurance and posture.
        - **Large MUs (Type II, Fast-Twitch):** Activated for strength and speed.
        - **Henneman’s Size Principle:** Your brain recruits small MUs first, then larger ones as force demand increases.

        **EMG amplitude** shows how many MUs are firing and how strongly.
        """)
    with col2:
        sarcomere_img = safe_load_image(sarcomere_img_url)
        if sarcomere_img:
            st.image(sarcomere_img, caption="Sarcomere: Basic contractile unit of a muscle fiber.")
        st.markdown("""
        - **Sarcomere:** The tiny engine of contraction.  
        - **Actin & Myosin:** Proteins that slide together to shorten the muscle.  
        - **EMG Signal:** The electrical activation that triggers these contractions.
        """)

# --------------------------- DATA PROCESSING --------------------------- #
if uploaded_file:
    with st.spinner('🔄 Processing EMG data...'):
        xls = pd.ExcelFile(uploaded_file)
        all_dfs, sheet_stats = [], []
        max_global_duration = 0

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
                    st.warning(f"⚠️ Sheet '{sheet_name}' missing required columns.")
                    continue

                df['time'] = pd.to_numeric(df['time'], errors='coerce')
                df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
                df = df.dropna(subset=['time', 'emg_rms_corrected_mV']).copy()
                if df.empty:
                    st.warning(f"⚠️ No valid data in '{sheet_name}'.")
                    continue

                df['time'] = df['time'] - df['time'].iloc[0]
                df['emg_raw'] = df['emg_rms_corrected_mV']
                df['emg_processed'] = df['emg_raw'].rolling(window=rolling_window, min_periods=1, center=True).mean()

                if normalize_data:
                    max_val = df['emg_processed'].max()
                    if max_val > 0:
                        df['emg_raw'] /= max_val
                        df['emg_processed'] /= max_val
                    else:
                        st.warning(f"⚠️ Sheet '{sheet_name}' has zero amplitude, skipped normalization.")

                stats = {
                    'sheet_name': sheet_name,
                    'data_points': len(df),
                    'duration': df['time'].max(),
                    'mean_amplitude': df['emg_processed'].mean(),
                    'max_amplitude': df['emg_processed'].max(),
                    'std_amplitude': df['emg_processed'].std()
                }

                sheet_stats.append(stats)
                df['sheet'] = sheet_name
                all_dfs.append(df)
                max_global_duration = max(max_global_duration, stats['duration'])

            except Exception as e:
                st.error(f"❌ Error in '{sheet_name}': {e}")

    # --------------------------- DISPLAY RESULTS --------------------------- #
    if all_dfs:
        st.success(f"✅ Processed {len(all_dfs)} muscle datasets.")

        st.subheader("📊 Muscle Activity Summary")
        cols = st.columns(min(4, len(sheet_stats)))
        for idx, stats in enumerate(sheet_stats):
            col = cols[idx % len(cols)]
            with col:
                unit = "% Max" if normalize_data else "mV"
                st.markdown(f"""
                <div class="muscle-card">
                    <h4>💪 {stats['sheet_name']}</h4>
                    <div class="stats-card">
                        <b>Duration:</b> {stats['duration']:.2f}s<br>
                        <b>Mean Amp:</b> {stats['mean_amplitude']:.3f} {unit}<br>
                        <b>Max Amp:</b> {stats['max_amplitude']:.3f} {unit}<br>
                        <b>Points:</b> {stats['data_points']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📈 EMG Signal Analysis")
        unit_label = "Amplitude (% Max)" if normalize_data else "EMG RMS (mV)"
        for df, stats in zip(all_dfs, sheet_stats):
            with st.expander(f"🔬 {stats['sheet_name']}", expanded=True):
                fig, ax = plt.subplots(figsize=(12, 4))
                if show_raw_data:
                    ax.plot(df['time'], df['emg_raw'], color='gray', linestyle='--', alpha=0.6, label='Raw')
                ax.plot(df['time'], df['emg_processed'], color='#ff6b6b', linewidth=2, label='Processed')
                ax.fill_between(df['time'], df['emg_processed'], alpha=0.3, color='#ff6b6b')
                ax.set_title(f"EMG Signal: {stats['sheet_name']}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(unit_label)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        # --------------------------- EXPORT RESULTS --------------------------- #
        combined_df = pd.concat(all_dfs, ignore_index=True)
        summary_df = pd.DataFrame(sheet_stats)
        st.subheader("💾 Export Results")
        st.download_button("📥 Download All Data (CSV)", combined_df.to_csv(index=False), "emg_all_data.csv", "text/csv")
        st.download_button("📊 Download Summary (CSV)", summary_df.to_csv(index=False), "emg_summary.csv", "text/csv")

else:
    st.info("""
    ### 🚀 Get Started
    Upload an Excel file (.xlsx) with:
    - Column 1: **time (s)**
    - Column 2: **emg_rms_corrected_mV (mV)**
    Each sheet = one muscle or trial.
    """)

    muscle_gif = safe_load_image(muscle_gif_url)
    if muscle_gif:
        st.image(muscle_gif, caption="Biceps and Triceps: Antagonistic Movement", use_column_width=True)

# --------------------------- FOOTER --------------------------- #
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>💪 EMG Muscle Analytics Dashboard | Built for Sports Science & Rehabilitation</div>", unsafe_allow_html=True)
