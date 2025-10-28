# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import requests
import base64

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(
    page_title="EMG Muscle Analytics Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- UTILS --------------------------- #
def load_image_from_url(url, timeout=8):
    """Load image from URL (safe call). Returns PIL.Image or None."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception:
        return None

def load_local_image(path):
    """Load local image from given path. Returns PIL.Image or None."""
    try:
        return Image.open(path)
    except Exception:
        return None

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# --------------------------- STYLES --------------------------- #
st.markdown(
    """
    <style>
    /* Layout helpers */
    .muscle-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px;
        border-radius: 12px;
        color: white;
        margin: 8px 0;
    }
    .stats-card {
        background: #f8fafc;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #ff4b4b;
        color: #111827;
        margin-top: 8px;
    }
    .info-box {
        background: #e8f4fd;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        color: #0f172a;
    }
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.title("💪 Muscle EMG Analytics")
    st.markdown("Upload your EMG Excel workbook (each sheet = one muscle/trial).")

    # Example local image (provided path in your container)
    sample_local_path = "/mnt/data/61e8ca10-f548-4227-adac-e057ff12d3d2.png"
    sample_image = load_local_image(sample_local_path)
    if sample_image:
        st.image(sample_image, caption="Dashboard UI Example", use_column_width=True)

    uploaded_file = st.file_uploader("📁 Upload Excel File (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.subheader("Processing Settings")
    rolling_window = st.slider("Smoothing Window (samples)", min_value=1, max_value=500, value=20, step=1)
    show_raw_data = st.checkbox("Show Raw Data", value=True)
    normalize_data = st.checkbox("Normalize Data (Show as % of Max)", value=False)
    resample_rate = st.number_input("Resample frequency (Hz, 0 = no resample)", min_value=0.0, value=0.0, step=1.0)
    st.markdown("---")
    st.subheader("ℹ️ Notes")
    st.info("Required columns per sheet: `time` (s) and `emg_rms_corrected_mV` (mV). Each sheet is treated as a separate muscle/trial.")

# --------------------------- HEADER --------------------------- #
col_left, col_right = st.columns([3, 1])
with col_left:
    st.title("💪 EMG Muscle Analytics Dashboard")
    st.markdown("Analyze muscle activity through EMG signal processing — useful for sports science and rehab.")
with col_right:
    # show anatomy here if available (prefer the local sample else try online)
    anatomy = sample_image or load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/6/6f/Muscular_system_front.svg")
    if anatomy:
        st.image(anatomy, caption="Anatomy Reference", use_column_width=True)

# --------------------------- INFO EXPANDER --------------------------- #
with st.expander("🧬 Muscle Physiology & EMG (Quick Primer)", expanded=False):
    st.markdown("""
    - *EMG amplitude* reflects the sum of electrical activity from motor units near the electrode.
    - *Smoothing* (rolling mean) reduces noise and helps show activation envelopes.
    - *Normalization* to max amplitude is useful to compare across muscles/sessions.
    - Ensure your sampling/time axis is correct; inconsistent time units produce misleading graphs.
    """)

# --------------------------- PROCESS UPLOADED DATA --------------------------- #
if uploaded_file:
    try:
        st.info("🔄 Reading workbook and processing sheets...")
        xls = pd.ExcelFile(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

    all_dfs = []
    sheet_stats = []
    max_global_duration = 0.0

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Check for required columns
            if 'time' not in df.columns or 'emg_rms_corrected_mV' not in df.columns:
                st.warning(f"Sheet '{sheet_name}' skipped: requires 'time' and 'emg_rms_corrected_mV' columns.")
                continue

            # enforce numeric
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['emg_rms_corrected_mV'] = pd.to_numeric(df['emg_rms_corrected_mV'], errors='coerce')
            df = df.dropna(subset=['time', 'emg_rms_corrected_mV']).copy()
            if df.empty:
                st.warning(f"Sheet '{sheet_name}' skipped: no valid rows after cleanup.")
                continue

            # Rebase time to zero
            df['time'] = df['time'] - df['time'].iloc[0]

            # Optional resample: if resample_rate > 0, perform simple linear interpolation to uniform spacing
            if resample_rate and resample_rate > 0:
                duration = df['time'].iloc[-1]
                new_times = np.arange(0, duration, 1.0 / resample_rate)
                df_interp = pd.DataFrame({'time': new_times})
                df_interp['emg_rms_corrected_mV'] = np.interp(new_times, df['time'].values, df['emg_rms_corrected_mV'].values)
                df = df_interp

            # store raw and processed columns
            df = df.reset_index(drop=True)
            df['emg_raw'] = df['emg_rms_corrected_mV']
            df['emg_processed'] = df['emg_raw'].rolling(window=rolling_window, min_periods=1, center=True).mean()

            if normalize_data:
                max_val = df['emg_processed'].max()
                if max_val > 0:
                    df['emg_processed'] = df['emg_processed'] / max_val * 100.0
                    df['emg_raw'] = df['emg_raw'] / max_val * 100.0
                else:
                    st.warning(f"Sheet '{sheet_name}' has max amplitude 0. Skipping normalization for that sheet.")

            stats = {
                'sheet_name': sheet_name,
                'data_points': len(df),
                'duration': float(df['time'].max()),
                'mean_amplitude': float(df['emg_processed'].mean()),
                'max_amplitude': float(df['emg_processed'].max()),
                'min_amplitude': float(df['emg_processed'].min()),
                'std_amplitude': float(df['emg_processed'].std()),
            }
            sheet_stats.append(stats)
            df['sheet'] = sheet_name
            all_dfs.append(df)

            if stats['duration'] > max_global_duration:
                max_global_duration = stats['duration']

        except Exception as e:
            st.error(f"Error processing sheet '{sheet_name}': {e}")

    if not all_dfs:
        st.error("❌ No valid muscle data found. Please check your Excel file format.")
    else:
        st.success(f"✅ Processed {len(all_dfs)} muscle dataset(s).")

        # ----- Overview cards ----- #
        st.subheader("📊 Muscle Activity Overview")
        num_sheets = len(sheet_stats)
        cols = st.columns(num_sheets if num_sheets <= 4 else 4)
        for idx, stats in enumerate(sheet_stats):
            col = cols[idx % len(cols)]
            with col:
                unit = "% Max" if normalize_data else "mV"
                st.markdown(
                    f"""
                    <div class="muscle-card">
                        <h4>💪 {stats['sheet_name']}</h4>
                        <div class="stats-card">
                            <strong>Duration:</strong> {stats['duration']:.2f}s<br>
                            <strong>Mean Amp:</strong> {stats['mean_amplitude']:.3f} {unit}<br>
                            <strong>Peak Amp:</strong> {stats['max_amplitude']:.3f} {unit}<br>
                            <strong>Data Points:</strong> {stats['data_points']:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----- Per-sheet analysis ----- #
        st.subheader("📈 EMG Signals Analysis")
        st.markdown(f"*All graphs use same time scale (0 to {max_global_duration:.2f}s) for comparability.*")
        unit_label = "EMG Amplitude (% Max)" if normalize_data else "EMG RMS Amplitude (mV)"

        for df, stats in zip(all_dfs, sheet_stats):
            with st.expander(f"🔬 Detailed Analysis — {stats['sheet_name']}", expanded=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig, ax = plt.subplots(figsize=(11, 4))
                    if show_raw_data:
                        ax.plot(df['time'], df['emg_raw'], linestyle='--', alpha=0.6, label='Raw Signal')
                    ax.plot(df['time'], df['emg_processed'], linewidth=2.2, label=f'Processed (win={rolling_window})')
                    ax.fill_between(df['time'], df['emg_processed'], alpha=0.25)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel(unit_label)
                    ax.set_title(f"Muscle Activity — {stats['sheet_name']}")
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, max_global_duration)
                    ax.set_ylim(bottom=0)
                    ax.legend()
                    fig.patch.set_facecolor('#ffffff')
                    st.pyplot(fig)
                with c2:
                    unit = "%" if normalize_data else "mV"
                    st.markdown(
                        f"""
                        <div class="info-box">
                            <h4>📋 Signal Info</h4>
                            <strong>Muscle:</strong> {stats['sheet_name']}<br>
                            <strong>Recording:</strong> {stats['duration']:.2f}s<br>
                            <strong>Peak:</strong> {stats['max_amplitude']:.3f} {unit}<br>
                            <strong>Mean:</strong> {stats['mean_amplitude']:.3f} {unit}<br>
                            <strong>Std Dev:</strong> {stats['std_amplitude']:.3f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ----- Export options ----- #
        st.subheader("💾 Export Analysis Results")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # rename processed with window info
        combined_df = combined_df.rename(columns={'emg_processed': f'emg_processed_{rolling_window}win'})
        csv_bytes = df_to_csv_bytes(combined_df)
        st.download_button(
            label="📥 Download All Processed Data (CSV)",
            data=csv_bytes,
            file_name="muscle_emg_analysis_all.csv",
            mime="text/csv",
            use_container_width=True,
        )

        summary_df = pd.DataFrame(sheet_stats)
        summary_csv = df_to_csv_bytes(summary_df)
        st.download_button(
            label="📊 Download Summary Statistics (CSV)",
            data=summary_csv,
            file_name="muscle_analysis_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ----- Insights ----- #
        st.markdown("---")
        st.subheader("🧠 Key Insights")
        most_active = max(sheet_stats, key=lambda x: x['max_amplitude'])
        avg_activity = np.mean([s['mean_amplitude'] for s in sheet_stats])
        unit = "% Max" if normalize_data else "mV"
        c1, c2 = st.columns(2)
        c1.info(f"*Most Active Muscle (Peak):* {most_active['sheet_name']} — Peak: {most_active['max_amplitude']:.3f} {unit}")
        c2.info(f"*Average Muscle Activity (Mean):* {avg_activity:.3f} {unit}")

else:
    # No file uploaded — show instructions and sample
    left, right = st.columns([2, 1])
    with left:
        st.info(
            """
            ## 🚀 Get Started
            Upload an Excel workbook (.xlsx) where each sheet is one muscle/trial.

            **Required columns (per sheet):**
            - `time` (in seconds)
            - `emg_rms_corrected_mV` (EMG RMS amplitude in mV)

            After upload, choose smoothing window, normalization, and optionally resample the signal for uniform sampling.
            """
        )
    with right:
        st.markdown("<div style='text-align:center'><h3>💪 Muscle Contraction</h3></div>", unsafe_allow_html=True)
        gif = load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/e/e0/Bicep_tricep.gif")
        if gif:
            st.image(gif, caption="Biceps & Triceps animation", use_column_width=True)

# --------------------------- FOOTER & CTA --------------------------- #
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray'>💪 EMG Analytics Dashboard — Built for sports science & rehab research</div>",
    unsafe_allow_html=True,
)

# --------------------------- HubSpot Promo (Affiliate) --------------------------- #
st.markdown("---")
st.markdown(
    """
    🚀 **Power Your Projects with HubSpot**  
    Streamline your team's workflow, meet deadlines, and collaborate seamlessly — all within HubSpot's all-in-one Project Management tools.  
    ✔️ Assign tasks  ✔️ Track progress  ✔️ Automate follow-ups  ✔️ Stay on budget  
    No more scattered tools — just results.  
    👉 Start managing smarter today: (*Affiliate*)https://go.try-hubspot.com/c/6231120/1197644/12893
    """
)

# End of app.py
