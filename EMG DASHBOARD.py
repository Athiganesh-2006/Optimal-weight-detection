import pandas as pd
import altair as alt
import streamlit as st

st.title("EMG RMS Dashboard")
st.markdown("""
Upload an Excel file with sheets containing:
- `time`
- `emg_rms_corrected_mV`

Each sheet will generate a separate graph.
""")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_dfs = []

    # Read all sheets
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Check required columns
        if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
            st.warning(f"Sheet '{sheet_name}' does not have required columns.")
            continue

        # Ensure numeric and drop NaNs
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
        df = df.dropna(subset=['time', 'emg_rms_corrected_mV'])

        # Reset time to start from zero
        df['time'] = df['time'] - df['time'].iloc[0]

        # Smooth EMG data using rolling mean
        df['emg_rms_corrected_mV'] = df['emg_rms_corrected_mV'].rolling(window=5, min_periods=1).mean()

        df['sheet'] = sheet_name
        all_dfs.append(df)

    if all_dfs:
        # Find max time for consistent x-axis
        max_time = max(df['time'].max() for df in all_dfs)

        for df in all_dfs:
            st.subheader(f"Sheet: {df['sheet'].iloc[0]}")

            chart = alt.Chart(df).mark_line(color='blue').encode(
                x=alt.X('time', title='Time (s)', scale=alt.Scale(domain=[0, max_time])),
                y=alt.Y('emg_rms_corrected_mV', title='EMG RMS (mV)'),
                tooltip=['time', 'emg_rms_corrected_mV']
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No valid sheets found in the uploaded file.")
else:
    st.info("Please upload an Excel file to visualize the data.")

.py
