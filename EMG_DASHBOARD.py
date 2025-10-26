import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(
    page_title="EMG Muscle Analytics Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.title("Muscle EMG Analytics")
    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])
    st.markdown("---")
    rolling_window = st.slider("Smoothing Window", min_value=1, max_value=100, value=20)
    show_raw_data = st.checkbox("Show Raw Data", value=True)
    normalize_data = st.checkbox("Normalize Data", value=False)

# --------------------------- MAIN HEADER --------------------------- #
st.title("💪 EMG Muscle Analytics Dashboard")

# --------------------------- DATA PROCESSING --------------------------- #
if uploaded_file:
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
        st.subheader("📊 Muscle Activity Overview")
        for stats in sheet_stats:
            st.markdown(f"**{stats['sheet_name']}**: Mean={stats['mean_amplitude']:.3f} mV, "
                        f"Max={stats['max_amplitude']:.3f} mV, Duration={stats['duration']:.2f}s")

        st.subheader("📈 EMG Signals")

        for df, stats in zip(all_dfs, sheet_stats):
            st.markdown(f"### {stats['sheet_name']}")
            fig, ax = plt.subplots(figsize=(12, 4))
            if show_raw_data:
                ax.plot(df['time'], df['emg_raw'], color='lightgray', linestyle='--', label='Raw')
            ax.plot(df['time'], df['emg_rms_corrected_mV'], color='blue', linewidth=2, label='Processed')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("EMG RMS (mV)")
            ax.set_title(f"{stats['sheet_name']} Muscle Activity")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        st.subheader("💾 Export Data")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        csv = combined_df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Data as CSV",
            data=csv,
            file_name="muscle_emg_analysis.csv",
            mime="text/csv"
        )

    else:
        st.error("❌ No valid muscle data found. Check your Excel file format.")
else:
    st.info("Upload an Excel file to start analysis. Each sheet should contain 'time' and 'emg_rms_corrected_mV' columns.")
