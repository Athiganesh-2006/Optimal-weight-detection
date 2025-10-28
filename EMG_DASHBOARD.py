import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(page_title="EMG & Health Dashboard", page_icon="💪", layout="wide")

# --------------------------- ASSETS --------------------------- #
# Developer-provided image path (available in the container)
HUMAN_IMG_PATH = '/mnt/data/709e3f80-2b8a-40c1-a770-6e9b9550efb3.png'

# --------------------------- STYLES --------------------------- #
st.markdown('''
<style>
/* general */
.card { background: white; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
.metric-card { border-radius: 10px; padding: 14px; color: #111; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }

/* right panel - dark BMI card */
.bmi-panel { background: #222; color: #fff; border-radius: 14px; padding: 18px; }
.bmi-mini { background: rgba(255,255,255,0.04); padding: 10px; border-radius: 8px; margin-bottom: 8px; }
.small-box { background: #fff; color: #111; padding: 8px 12px; border-radius: 8px; display:inline-block; }

/* chart container */
.chart-box { background: #fafafa; padding: 12px; border-radius: 10px; }

/* footer */
.footer { text-align:center; color: #777; padding:12px; }

/* responsive image */
.human-img { width: 220px; }

</style>
''', unsafe_allow_html=True)

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.title("💪 Muscle EMG Analytics")
    st.markdown("Upload EMG data, toggle settings and view quick stats.")

    uploaded_file = st.file_uploader("Upload Excel (.xlsx) where each sheet = one trial", type=["xlsx"])
    st.markdown("---")
    st.subheader("Processing")
    rolling_window = st.slider("Smoothing Window (samples)", 1, 200, 20)
    show_raw = st.checkbox("Show raw signal", True)
    normalize = st.checkbox("Normalize amplitude (0-1)", False)
    st.markdown("---")
    st.subheader("Quick Health")
    st.markdown("Track your BMI and body measurements on the right panel.")

# --------------------------- TOP HEADER --------------------------- #
col_l, col_r = st.columns([3,1])
with col_l:
    st.title("💪 EMG Muscle & Health Dashboard")
    st.markdown("A combined view of EMG analysis and simple health metrics for sports science & rehab.")
with col_r:
    st.markdown("")

# --------------------------- MAIN LAYOUT --------------------------- #
left, right = st.columns([3,1])

# LEFT: cards, activity chart, EMG analysis
with left:
    st.markdown("""
    <div class='card'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
      <div>
        <h3 style='margin:0'>Health Overview</h3>
        <div style='color:#666; font-size:13px;'>Live snapshot (simulated values)</div>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # three metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("<div class='metric-card card' style='border-left:6px solid #ff7a7a'>", unsafe_allow_html=True)
        st.subheader("Blood Sugar")
        st.metric(label="mg/dL", value="88", delta="-2")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card card' style='border-left:6px solid #6b8cff'>", unsafe_allow_html=True)
        st.subheader("Heart Rate")
        st.metric(label="bpm", value="72", delta="+3")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card card' style='border-left:6px solid #34d399'>", unsafe_allow_html=True)
        st.subheader("Blood Pressure")
        st.markdown("<b>102 / 72</b><div style='color:#777;font-size:12px'>Normal</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Activity growth chart (simulated)
    st.subheader("Activity Growth")
    days = pd.date_range(end=pd.Timestamp.today(), periods=30)
    activity = np.abs(np.random.normal(loc=50, scale=12, size=len(days))).cumsum() / 50
    fig1, ax1 = plt.subplots(figsize=(10,3))
    ax1.plot(days, activity, linewidth=2)
    ax1.fill_between(days, activity, alpha=0.2)
    ax1.set_ylabel('Activity (a.u.)')
    ax1.set_xlabel('Date')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    st.markdown("---")

    # EMG Section
    st.subheader("EMG Signal Analysis")
    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            dfs = []
            for s in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=s)
                if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
                    st.warning(f"Sheet '{s}' missing 'time' or 'emg_rms_corrected_mV'. Skipping")
                    continue
                df = df.dropna(subset=['time','emg_rms_corrected_mV']).copy()
                df['time'] = df['time'] - df['time'].iloc[0]
                df['processed'] = df['emg_rms_corrected_mV'].rolling(window=rolling_window, min_periods=1, center=True).mean()
                if normalize:
                    m = df['processed'].max()
                    if m>0: df[['emg_rms_corrected_mV','processed']] = df[['emg_rms_corrected_mV','processed']]/m
                dfs.append((s,df))

            for name, df in dfs:
                st.markdown(f"**{name}**")
                fig, ax = plt.subplots(figsize=(10,3))
                if show_raw:
                    ax.plot(df['time'], df['emg_rms_corrected_mV'], linestyle='--', alpha=0.6, label='Raw')
                ax.plot(df['time'], df['processed'], linewidth=2, label='Processed')
                ax.set_xlabel('Time (s)'); ax.set_ylabel('EMG RMS (mV)')
                ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig)

            if len(dfs)>0:
                combined = pd.concat([d for _,d in dfs], ignore_index=True)
                st.download_button("Download combined CSV", combined.to_csv(index=False), "emg_combined.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload an Excel file in the sidebar to analyze EMG data. Each sheet = one trial.")

# RIGHT: BMI + Body measurements (dark card)
with right:
    st.markdown("<div class='bmi-panel card'>", unsafe_allow_html=True)
    st.markdown("## BMI Calculator")

    # BMI inputs
    height_cm = st.slider('Height (cm)', 120, 210, 170)
    weight_kg = st.slider('Weight (kg)', 40, 140, 72)
    age = st.number_input('Age', min_value=12, max_value=100, value=28)
    sex = st.radio('Sex', ['Male', 'Female'])

    bmi = weight_kg / ((height_cm/100)**2)
    status = 'Underweight' if bmi<18.5 else ('Healthy' if bmi<25 else ('Overweight' if bmi<30 else 'Obese'))

    st.markdown(f"<div class='bmi-mini'><b>BMI</b> <span style='float:right'>{bmi:.1f} ({status})</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='display:flex; gap:8px; align-items:center;'>", unsafe_allow_html=True)
    # body image
    try:
        human = Image.open(HUMAN_IMG_PATH)
        st.image(human, width=200)
    except Exception:
        st.info('Human figure image missing in container.')
    st.markdown("</div>", unsafe_allow_html=True)

    # Body measurements boxes
    chest = st.number_input('Chest (in)', min_value=20.0, max_value=60.0, value=44.5, step=0.1)
    waist = st.number_input('Waist (in)', min_value=18.0, max_value=60.0, value=34.0, step=0.1)
    hip = st.number_input('Hip (in)', min_value=20.0, max_value=70.0, value=42.5, step=0.1)

    st.markdown('<div style="margin-top:10px"><span class="small-box">Chest: {}</span> <span class="small-box">Waist: {}</span> <span class="small-box">Hip: {}</span></div>'.format(chest, waist, hip), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown('---')
st.markdown("<div class='footer'>EMG & Health Dashboard — Built for Sports Science & Rehabilitation</div>", unsafe_allow_html=True)

# --- End of file ---
