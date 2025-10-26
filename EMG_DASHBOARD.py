import pandas as pd
import altair as alt
import streamlit as st
import time # For the animation

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="EMG RMS Dashboard with Animations",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ EMG RMS Dashboard with Interactive Elements & Animations")
st.markdown("""
Welcome to your advanced EMG RMS data visualization tool!
Upload an Excel file with sheets containing `time` and `emg_rms_corrected_mV` columns.
Each selected sheet will generate a separate interactive graph, and you can even try a fun animation!
""")

# Proactive image: An engaging image for the dashboard's welcome
st.image("https://images.unsplash.com/photo-1577006760662-310850024479?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NTgzODJ8MHwxfHNlYXJjaHwxfHxhbmF0b215JTIwYW5kJTIwZWxlY3Ryb215b2dyYXBoeXxlbnwwfHx8fDE3MTc1MTAwMTZ8MA&ixlib=rb-4.0.3&q=80&w=1080")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_sheet_names = xls.sheet_names

    # --- Sidebar for Sheet Selection ---
    st.sidebar.header("📊 Data Selection")
    selected_sheets = st.sidebar.multiselect(
        "Select Sheets to Visualize",
        options=all_sheet_names,
        default=all_sheet_names if all_sheet_names else [] # Select all by default
    )

    if not selected_sheets:
        st.warning("👈 Please select at least one sheet from the sidebar to visualize your data.")
    else:
        # --- Data Loading and Processing ---
        processed_dfs = []
        max_overall_time = 0.0

        for sheet_name in selected_sheets:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Check required columns
            if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
                st.error(f"🚫 **Sheet '{sheet_name}'** cannot be plotted.")
                st.warning(f"Required columns `time` and `emg_rms_corrected_mV` not found in '{sheet_name}'. Available columns: `{list(df.columns)}`.")
                continue

            # Ensure numeric and drop NaNs
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
            df = df.dropna(subset=['time', 'emg_rms_corrected_mV'])

            if df.empty:
                st.warning(f"Sheet '{sheet_name}' is empty or contains no valid data after cleaning.")
                continue

            # Reset time to start from zero
            df['time_normalized'] = df['time'] - df['time'].iloc[0]

            # Smooth EMG data using rolling mean
            df['emg_rms_corrected_mV_smoothed'] = df['emg_rms_corrected_mV'].rolling(window=5, min_periods=1).mean()

            df['sheet'] = sheet_name
            processed_dfs.append(df)

            # Update overall max time
            if not df.empty:
                current_max_time = df['time_normalized'].max()
                if current_max_time > max_overall_time:
                    max_overall_time = current_max_time

        # --- Visualization Section ---
        if processed_dfs:
            st.markdown("---")
            st.subheader("📈 EMG RMS Data Charts")
            st.info(f"🌐 All charts share a consistent X-axis from 0 to **{max_overall_time:.2f} s** for easy comparison.")

            # Arrange charts in two columns
            chart_cols = st.columns(2)
            col_idx = 0

            for df in processed_dfs:
                sheet_name = df['sheet'].iloc[0]
                
                with chart_cols[col_idx % 2]:
                    st.markdown(f"#### {sheet_name}")

                    # Create interactive chart with shared x-axis scale
                    chart = alt.Chart(df).mark_line(color='blue', opacity=0.8).encode(
                        x=alt.X('time_normalized', title='Time (s)', scale=alt.Scale(domain=[0, max_overall_time])),
                        y=alt.Y('emg_rms_corrected_mV_smoothed', title='EMG RMS (mV)'),
                        tooltip=[alt.Tooltip('time_normalized', format='.2f'), alt.Tooltip('emg_rms_corrected_mV_smoothed', format='.2f')]
                    ).properties(
                        title=f'EMG RMS for {sheet_name}'
                    ).interactive() # Make chart interactive (zoom, pan)

                    st.altair_chart(chart, use_container_width=True)

                    # Expander for raw data
                    with st.expander(f"View Raw Data for {sheet_name}"):
                        st.dataframe(df[['time', 'time_normalized', 'emg_rms_corrected_mV', 'emg_rms_corrected_mV_smoothed']], use_container_width=True)
                
                col_idx += 1
            
            st.markdown("---")

        else:
            st.warning("No valid sheets with required columns were selected or found in the uploaded file.")

else:
    st.info("⬆️ Please upload an Excel file to start your EMG RMS data analysis.")
    # Proactive image: An inviting image when no file is uploaded
    st.image("https://images.unsplash.com/photo-1542475734-d13a69642643?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NTgzODJ8MHwxfHNlYXJjaHwxfHxkYXRhJTIwYW5hbHlzaXMlMjBzdGFydHVwJTIwdXBsb2FkJTIwZmlsZXxlbnwwfHx8fDE3MTc1MTAwNTl8MA&ixlib=rb-4.0.3&q=80&w=1080")


# --- Fun Animation Exercise ---
st.sidebar.header("✨ Fun Exercise: EMG Animation!")
if st.sidebar.button("Start EMG Animation"):
    st.subheader("🚀 Real-time EMG Signal Simulation")
    st.write("Watch as a simulated EMG signal fluctuates over time!")

    # Create an empty chart placeholder
    chart_placeholder = st.empty()

    # Initialize data for animation
    animation_df = pd.DataFrame({'time': [], 'emg_value': []})
    current_time = 0.0
    
    # Proactive image: An animated GIF or a dynamic image representing an EMG signal
    st.image("https://images.unsplash.com/photo-1620397752628-98e3b3c3b0f2?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NTgzODJ8MHwxfHNlYXJjaHwxfHxlbWVnJTIwc2lnbmFsJTIwYW5pbWF0aW9uJTIwZ2lmJTIwaW1hZ2V8ZW58MHx8fHwxNzE3NTEwMTU2fDA&ixlib=rb-4.0.3&q=80&w=1080", caption="Simulated EMG Signal in action!")

    for i in range(200): # Animate for 200 frames
        new_emg_value = 5 + 2 * (0.5 * (i % 50) - 25) / 25 * (i % 50) + (i % 10) # Simulate a fluctuating EMG
        new_row = pd.DataFrame([{'time': current_time, 'emg_value': new_emg_value}])
        animation_df = pd.concat([animation_df, new_row], ignore_index=True)

        # Keep only the last 50 data points for a "scrolling" effect
        if len(animation_df) > 50:
            animation_df = animation_df.tail(50)

        # Create Altair chart for the animation
        animation_chart = alt.Chart(animation_df).mark_line(color='green').encode(
            x=alt.X('time', title='Time (s)', scale=alt.Scale(domain=[animation_df['time'].min(), animation_df['time'].max()])),
            y=alt.Y('emg_value', title='Simulated EMG (mV)', scale=alt.Scale(domain=[0, 20])), # Fixed Y-axis for stability
            tooltip=['time', 'emg_value']
        ).properties(
            title="Live EMG Signal"
        )
        
        chart_placeholder.altair_chart(animation_chart, use_container_width=True)

        current_time += 0.1 # Increment time
        time.sleep(0.05) # Control animation speed

    st.success("Animation complete!")


st.sidebar.markdown("""
---
### **Dashboard Info**
* **Target Columns:** `time`, `emg_rms_corrected_mV`
* **Time Normalization:** All 'time' axes start from 0 for consistency.
* **Smoothing:** A 5-point rolling mean is applied to `emg_rms_corrected_mV`.
* **Interactivity:** Charts support zoom and pan (drag to zoom, Shift+drag to pan).
""")