import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import requests

# ============================================================
# FUNCTION TO LOAD IMAGES SAFELY (works on Render & Local)
# ============================================================
def load_image_from_url(url, fallback_text="⚠️ Could not load image"):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.warning(f"{fallback_text}: {e}")
        return None

# ============================================================
# IMAGE FUNCTIONS
# ============================================================
def load_muscle_image():
    return load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/3/30/Biceps_brachii.png")

def get_muscle_gif():
    return load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/8/87/Biceps_curl_animation.gif")

def get_physiology_image():
    return load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/c/c1/Sarcomere_relaxed_contracted.PNG")

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="Muscle EMG Analytics", layout="wide", page_icon="💪")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 💪 Muscle EMG Analytics")

gif_img = get_muscle_gif()
if gif_img:
    st.sidebar.image(gif_img, use_container_width=True)

st.sidebar.header("📤 Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Drag and drop your Excel file (.xlsx)", type=["xlsx"], help="Upload EMG RMS data")

st.sidebar.markdown("---")
st.sidebar.info("Each sheet = one muscle or trial")

# ============================================================
# MAIN TITLE
# ============================================================
st.title("💪 EMG Muscle Analytics Dashboard")
st.markdown("Analyze muscle activity through EMG signal processing — ideal for sports science and rehabilitation.")

# ============================================================
# MUSCLE PHYSIOLOGY SECTION
# ============================================================
with st.expander("🧬 Muscle Physiology & Weightlifting: What EMG Shows", expanded=True):
    st.markdown("""
    **Electromyography (EMG)** measures the electrical activity produced by skeletal muscles.
    
    During **weightlifting**, EMG helps determine how much a muscle is activated as load increases.
    
    When EMG RMS increases with load, it means the muscle recruits more motor units.
    However, after a certain weight, the muscle reaches **saturation**—further load causes decreased EMG RMS due to fatigue.
    """)
    img = load_muscle_image()
    if img:
        st.image(img, caption="Anatomy of the Biceps Brachii", use_container_width=True)

# ============================================================
# GET STARTED SECTION
# ============================================================
st.subheader("🚀 Get Started")
st.markdown("""
Upload an Excel file (.xlsx) to analyze muscle EMG data.
""")

st.info("""
### 📁 Expected Format:
Each sheet = one muscle or trial  
Columns required:
1. `time (s)`  
2. `emg_rms_corrected_mV (mV)`  
3. `load (kg)`
""")

# ============================================================
# FILE PROCESSING AND ANALYSIS
# ============================================================
if uploaded_file is not None:
    try:
        # Read Excel file
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.success(f"✅ Loaded file successfully! Sheets found: {', '.join(sheet_names)}")

        for sheet in sheet_names:
            st.markdown(f"### 📊 Data from: {sheet}")
            df = pd.read_excel(xls, sheet_name=sheet)

            # Clean column names
            df.columns = df.columns.str.strip().str.lower()

            # Check required columns
            required_cols = ['time', 'emg_rms_corrected_mv', 'load']
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ Missing required columns in sheet '{sheet}'. Expected: {required_cols}")
                continue

            # Plot EMG RMS over time
            st.markdown("#### ⚡ EMG RMS vs Time")
            fig, ax = plt.subplots()
            ax.plot(df['time'], df['emg_rms_corrected_mv'], label="EMG RMS (mV)", linewidth=2)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("EMG RMS (mV)")
            ax.set_title(f"EMG RMS vs Time — {sheet}")
            ax.legend()
            st.pyplot(fig)

            # Average RMS vs Load (if multiple trials)
            st.markdown("#### 🏋️ EMG RMS vs Load")
            load_groups = df.groupby('load')['emg_rms_corrected_mv'].mean().reset_index()

            fig2, ax2 = plt.subplots()
            ax2.plot(load_groups['load'], load_groups['emg_rms_corrected_mv'], 'o-', linewidth=2, color='orange')
            ax2.set_xlabel("Load (kg)")
            ax2.set_ylabel("Average EMG RMS (mV)")
            ax2.set_title(f"Muscle Activation vs Load — {sheet}")
            ax2.grid(True)
            st.pyplot(fig2)

            # Detect saturation point (max RMS before drop)
            rms_values = load_groups['emg_rms_corrected_mv'].values
            loads = load_groups['load'].values
            if len(rms_values) > 2:
                peak_idx = np.argmax(rms_values)
                st.success(f"🏁 Optimal load (saturation point): **{loads[peak_idx]} kg** — Peak EMG RMS: {rms_values[peak_idx]:.2f} mV")
            else:
                st.warning("Not enough data points to determine saturation.")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

# ============================================================
# EXAMPLE VISUALS
# ============================================================
st.markdown("---")
st.subheader("📽️ Example Visuals")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💪 Muscle Contraction Animation")
    gif = get_muscle_gif()
    if gif:
        st.image(gif, caption="Biceps Curl Animation", use_container_width=True)

with col2:
    st.markdown("### ⚡ Physiology Diagram")
    physio = get_physiology_image()
    if physio:
        st.image(physio, caption="Sarcomere: Relaxed vs Contracted", use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Developed by Athi Ganesh R • Biomedical EMG Analytics • © 2025 Muscle EMG Dashboard")
