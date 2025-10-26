import pandas as pd
import altair as alt
import streamlit as st

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(
    page_title="EMG RMS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- SIDEBAR --------------------------- #
st.sidebar.header("Dashboard Options")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
rolling_window = st.sidebar.slider("Smoothing Window Size", min_value=1, max_value=50, value=5)

st.sidebar.markdown("""
This dashboard reads EMG RMS data from Excel sheets. Each sheet should contain:
- `time`
- `emg_rms_corrected_mV`
""")

# --------------------------- MAIN TITLE --------------------------- #
st.title("📊 EMG RMS Dashboard")
st.markdown("""
Visualize EMG RMS signals for multiple datasets.  
Each Excel sheet will appear in a separate tab with interactive plots.
""")

# --------------------------- DATA PROCESSING --------------------------- #
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_dfs = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Validate columns
        if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
            st.warning(f"Sheet '{sheet_name}' does not have required columns.")
            continue

        # Ensure numeric and drop NaNs
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
        df = df.dropna(subset=['time', 'emg_rms_corrected_mV'])

        # Reset time to start from zero
        df['time'] = df['time'] - df['time'].iloc[0]

        # Smooth EMG data
        df['emg_rms_corrected_mV'] = df['emg_rms_corrected_mV'].rolling(window=rolling_window, min_periods=1).mean()

        df['sheet'] = sheet_name
        all_dfs.append(df)

    if all_dfs:
        # Find max time for consistent x-axis
        max_time = max(df['time'].max() for df in all_dfs)

        # --------------------------- CREATE TABS --------------------------- #
        tabs = st.tabs([df['sheet'].iloc[0] for df in all_dfs])

        for tab, df in zip(tabs, all_dfs):
            with tab:
                st.subheader(f"Sheet: {df['sheet'].iloc[0]}")
                st.markdown(f"Data points: {len(df)} | Rolling window: {rolling_window}")

                # Altair chart
                chart = alt.Chart(df).mark_line(color='#1f77b4', strokeWidth=2).encode(
                    x=alt.X('time', title='Time (s)', scale=alt.Scale(domain=[0, max_time])),
                    y=alt.Y('emg_rms_corrected_mV', title='EMG RMS (mV)'),
                    tooltip=[alt.Tooltip('time', title='Time (s)'),
                             alt.Tooltip('emg_rms_corrected_mV', title='EMG RMS (mV)')]
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No valid sheets found in the uploaded file.")
else:
    st.info("Please upload an Excel file to visualize the data.")
