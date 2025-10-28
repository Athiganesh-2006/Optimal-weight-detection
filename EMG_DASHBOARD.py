"""
EMG_Muscle_Dashboard.py
Complete Streamlit app showing EMG physiological metrics + MPU6050 graphs.

Usage:
    streamlit run EMG_Muscle_Dashboard.py
For Render/hosting:
    streamlit run EMG_Muscle_Dashboard.py --server.port $PORT --server.address 0.0.0.0
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from PIL import Image
from io import BytesIO

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(page_title="EMG Muscle & Physiology Dashboard",
                   page_icon="💪",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------- HELPERS --------------------------- #
@st.cache_data
def safe_read_excel_or_csv(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return {"__single__": df}
    else:
        xls = pd.ExcelFile(uploaded_file)
        dfs = {}
        for s in xls.sheet_names:
            dfs[s] = pd.read_excel(xls, sheet_name=s)
        return dfs

@st.cache_data
def rolling_rms(signal, window_samples):
    # compute RMS using convolution for speed
    sq = np.square(signal)
    kern = np.ones(window_samples) / window_samples
    rms = np.sqrt(np.convolve(sq, kern, mode='valid'))
    return rms

def windowed_median_frequency(signal, fs, win_s=1.0, step_s=0.5, nperseg=256):
    """
    Compute median frequency in sliding windows using Welch's method.
    Returns times (center of windows) and median frequencies.
    """
    sig = np.asarray(signal)
    win = int(win_s * fs)
    step = int(step_s * fs)
    if win < 4:
        win = max(4, int(fs * 0.5))
    med_freqs = []
    times = []
    for start in range(0, max(1, len(sig) - win + 1), step):
        seg = sig[start:start + win]
        if len(seg) < 4:
            continue
        f, Pxx = welch(seg, fs=fs, nperseg=min(nperseg, len(seg)))
        cumsum = np.cumsum(Pxx)
        total = cumsum[-1]
        if total <= 0:
            med = 0.0
        else:
            idx = np.searchsorted(cumsum, total / 2.0)
            med = f[min(idx, len(f) - 1)]
        med_freqs.append(med)
        times.append((start + win / 2.0) / fs)
    return np.array(times), np.array(med_freqs)

def detect_contractions(rms_signal, time_axis, threshold):
    """
    Identify contraction bouts where rms_signal > threshold.
    Return list of (start_time, end_time, duration, peak) for each contraction.
    """
    above = rms_signal > threshold
    if len(above) == 0:
        return []
    contractions = []
    i = 0
    N = len(above)
    while i < N:
        if above[i]:
            start = i
            while i < N and above[i]:
                i += 1
            end = i - 1
            start_t = time_axis[start]
            end_t = time_axis[end]
            duration = end_t - start_t
            peak = np.max(rms_signal[start:end + 1])
            contractions.append({
                "start": float(start_t),
                "end": float(end_t),
                "duration": float(duration),
                "peak": float(peak)
            })
        else:
            i += 1
    return contractions

def estimate_force_from_rms(rms_value, max_rms, max_force):
    """
    Simple linear mapping: force = (rms / max_rms) * max_force
    If max_rms == 0 returns 0.
    """
    if max_rms <= 0:
        return 0.0
    return (rms_value / max_rms) * max_force

# --------------------------- LAYOUT: Sidebar --------------------------- #
st.sidebar.title("📤 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload EMG file (.xlsx or .csv). Multi-sheet supported.", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("Processing Settings")
fs_input = st.sidebar.number_input("Sampling frequency (Hz)", min_value=10.0, value=1000.0, step=1.0)
rms_window_ms = st.sidebar.slider("RMS window (ms)", min_value=10, max_value=500, value=200, step=10)
rms_window_samples = max(1, int(rms_window_ms * fs_input / 1000.0))
fft_win_s = st.sidebar.slider("FFT window (s) for median freq", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
fft_step_s = st.sidebar.slider("FFT step (s)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
contraction_threshold_pct = st.sidebar.slider("Contraction threshold (% of RMS max)", min_value=1, max_value=80, value=20)
max_force_calib = st.sidebar.number_input("Calibration: Max Force at max RMS (N)", min_value=1.0, value=200.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("MPU6050 (optional)")
mpu_present = st.sidebar.checkbox("Include MPU6050 plots if present in file", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for EMG physiology metrics • Upload file and select sheet(s)")

# --------------------------- Main header --------------------------- #
st.markdown("<h1 style='display:flex;align-items:center;'>💪 EMG Muscle & Physiology Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Analyze EMG muscle signals — RMS, fatigue (median freq), peaks, contraction durations, simple force estimate, and MPU6050 (accel/gyro) if available.")

# --------------------------- Load data --------------------------- #
if uploaded_file is None:
    st.info("Upload a `.csv` or `.xlsx` file. Each sheet can be one muscle/trial. Required columns: `time` (s) and an EMG column (e.g., `emg`, `emg_rms_corrected_mV`). Optional MPU columns: `accel_x`, `accel_y`, `accel_z`, `gyro_x`, `gyro_y`, `gyro_z`.")
    st.stop()

try:
    sheets = safe_read_excel_or_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# Present selection of sheets
sheet_names = list(sheets.keys())
selected_sheets = st.multiselect("Select sheet(s)/trials to analyze", sheet_names, default=sheet_names[:1])

if not selected_sheets:
    st.warning("Select at least one sheet to analyze.")
    st.stop()

# Prepare a combined summary list
summary_list = []
combined_dfs = []

# Layout: left (main) and right (controls/summary)
left_col, right_col = st.columns([3, 1])

with left_col:
    for sheet_name in selected_sheets:
        df = sheets[sheet_name].copy()
        st.subheader(f"🔬 Trial: {sheet_name}")

        # Standardize column names (lowercase)
        df.columns = [c.strip() for c in df.columns]
        cols_lower = {c: c.lower() for c in df.columns}
        df = df.rename(columns=cols_lower)  # rename to lowercase mapping

        # Detect time & emg columns heuristically
        time_col = None
        emg_col = None
        for cand in ['time', 't', 'timestamp']:
            if cand in df.columns:
                time_col = cand
                break
        for cand in ['emg', 'emg_rms_corrected_mV'.lower(), 'emg_rms', 'signal', 'emg_mv']:
            if cand in df.columns:
                emg_col = cand
                break
        # if not found, pick first numeric col as time and second numeric as emg
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col is None and len(numeric_cols) >= 1:
            time_col = numeric_cols[0]
        if emg_col is None and len(numeric_cols) >= 2:
            emg_col = numeric_cols[1]
        if time_col is None or emg_col is None:
            st.error(f"Could not find time & EMG columns in sheet '{sheet_name}'. Found numeric columns: {numeric_cols}")
            continue

        time = pd.to_numeric(df[time_col], errors='coerce').to_numpy(dtype=float)
        emg_raw = pd.to_numeric(df[emg_col], errors='coerce').to_numpy(dtype=float)

        # Ensure time monotonic starting at 0
        if np.isnan(time).all():
            st.error("Time column contains non-numeric values.")
            continue
        time = time - time[0]
        fs = float(fs_input)

        # Align lengths: compute RMS (valid mode reduces length)
        rms = rolling_rms(emg_raw, rms_window_samples)
        # time for rms is centered relative to original time array
        valid_len = len(rms)
        time_rms = time[:valid_len]

        # Peak activation
        peak_amp = float(np.max(rms)) if valid_len > 0 else 0.0
        mean_amp = float(np.mean(rms)) if valid_len > 0 else 0.0
        std_amp = float(np.std(rms)) if valid_len > 0 else 0.0

        # Median frequency (fatigue) sliding windows
        med_times, med_freqs = windowed_median_frequency(emg_raw, fs, win_s=fft_win_s, step_s=fft_step_s)

        # Fatigue index: slope (linear fit) of median freq over time (negative slope => fatigue)
        fat_idx = None
        if len(med_freqs) >= 3:
            # simple linear regression slope (Hz/s)
            A = np.vstack([med_times, np.ones_like(med_times)]).T
            m, c = np.linalg.lstsq(A, med_freqs, rcond=None)[0]
            fat_idx = float(m)  # slope in Hz/s
        else:
            fat_idx = 0.0

        # Contraction detection: threshold at X% of max RMS
        thresh = (contraction_threshold_pct / 100.0) * (np.max(rms) if np.max(rms) > 0 else 1.0)
        contractions = detect_contractions(rms, time_rms, thresh)

        # Force estimation: use user max_force_calib and map RMS to force linearly
        estimated_force = estimate_force_from_rms(peak_amp, np.max(rms) if np.max(rms) > 0 else 1.0, float(max_force_calib))

        # Save summary
        summary = {
            "sheet": sheet_name,
            "data_points": int(len(emg_raw)),
            "duration_s": float(time[-1]) if len(time) > 0 else 0.0,
            "mean_rms": mean_amp,
            "peak_rms": peak_amp,
            "rms_std": std_amp,
            "fatigue_slope_Hz_per_s": fat_idx,
            "n_contractions": len(contractions),
            "estimated_force_N_at_peak": estimated_force
        }
        summary_list.append(summary)

        # Prepare combined df for export (aligned by RMS)
        df_rms = pd.DataFrame({
            "time_rms": time_rms,
            f"{sheet_name}_rms": rms
        })
        combined_dfs.append(df_rms)

        # ---------- VISUALS ----------
        # Row: two columns - left big EMG/time, right small summary + contractions
        em_col, info_col = st.columns([3, 1])

        with em_col:
            # EMG raw + RMS
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(time, emg_raw, linestyle='--', alpha=0.5, label='EMG raw')
            ax.plot(time_rms, rms, linewidth=2, label=f'RMS ({rms_window_ms} ms)')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (a.u. or mV)")
            ax.set_title(f"EMG: {sheet_name}")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # Fatigue plot: median frequency over time
            if len(med_freqs) > 0:
                fig2, ax2 = plt.subplots(figsize=(10, 2.2))
                ax2.plot(med_times, med_freqs, marker='o', linestyle='-', label='Median Frequency (Hz)')
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Median Freq (Hz)")
                ax2.set_title("Median Frequency (sliding) — indicator of fatigue")
                ax2.grid(alpha=0.25)
                st.pyplot(fig2)

            # FFT of whole signal (for reference)
            if len(emg_raw) > 8:
                nfft = min(4096, len(emg_raw))
                f, Pxx = welch(emg_raw, fs=fs, nperseg=min(1024, nfft))
                fig3, ax3 = plt.subplots(figsize=(10, 2.2))
                ax3.semilogy(f, Pxx + 1e-12)
                ax3.set_xlim(0, min(500, fs / 2.0))
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("PSD")
                ax3.set_title("Power Spectrum (Welch)")
                ax3.grid(alpha=0.2)
                st.pyplot(fig3)

        with info_col:
            # Quick stat cards
            st.markdown("**Key metrics**")
            st.metric("Mean RMS", f"{mean_amp:.3f}")
            st.metric("Peak RMS", f"{peak_amp:.3f}")
            st.metric("Estimated Peak Force (N)", f"{estimated_force:.1f}")
            st.markdown("---")
            st.markdown("**Contractions detected**")
            if len(contractions) == 0:
                st.info("No contractions detected at current threshold.")
            else:
                # show a few contractions
                for i, c in enumerate(contractions[:8]):
                    st.write(f"{i+1}. {c['start']:.2f}s → {c['end']:.2f}s | dur {c['duration']:.2f}s | peak {c['peak']:.3f}")

            st.markdown("---")
            st.markdown(f"**Fatigue slope**: {fat_idx:.4f} Hz/s  (neg. → decreasing freq → fatigue)")
            st.caption("Fatigue slope is computed by linear fit on median frequency vs time (Welch).")

        st.markdown("""---""")

        # Add MPU6050 graphs if present and user wants them
        if mpu_present:
            accel_cols = [c for c in df.columns if any(x in c for x in ['accel', 'acc_x', 'accel_x'])]
            gyro_cols = [c for c in df.columns if any(x in c for x in ['gyro', 'gyro_x', 'gyroscope'])]
            has_accel = len(accel_cols) >= 3 or all(x in df.columns for x in ['accel_x', 'accel_y', 'accel_z'])
            has_gyro = len(gyro_cols) >= 3 or all(x in df.columns for x in ['gyro_x', 'gyro_y', 'gyro_z'])

            if has_accel:
                # try common names
                def getcol(df, names):
                    for n in names:
                        if n in df.columns:
                            return df[n].to_numpy(dtype=float)
                    return None
                ax_col = getcol(df, ['accel_x', 'acc_x', 'ax', 'accelx'])
                ay_col = getcol(df, ['accel_y', 'acc_y', 'ay', 'accely'])
                az_col = getcol(df, ['accel_z', 'acc_z', 'az', 'accelz'])
                # if None, try matching by pattern
                if ax_col is None:
                    # try pattern search
                    keys = [k for k in df.columns if 'acc' in k and ('x' in k or '_x' in k)]
                    if keys:
                        ax_col = df[keys[0]].to_numpy(dtype=float)
                if ay_col is None:
                    keys = [k for k in df.columns if 'acc' in k and ('y' in k or '_y' in k)]
                    if keys:
                        ay_col = df[keys[0]].to_numpy(dtype=float)
                if az_col is None:
                    keys = [k for k in df.columns if 'acc' in k and ('z' in k or '_z' in k)]
                    if keys:
                        az_col = df[keys[0]].to_numpy(dtype=float)

                if ax_col is not None and ay_col is not None and az_col is not None:
                    fig_a, ax_a = plt.subplots(figsize=(10, 2.2))
                    ax_a.plot(time[:len(ax_col)], ax_col, label='ax', alpha=0.9)
                    ax_a.plot(time[:len(ay_col)], ay_col, label='ay', alpha=0.9)
                    ax_a.plot(time[:len(az_col)], az_col, label='az', alpha=0.9)
                    ax_a.set_title("MPU6050 - Accelerometer")
                    ax_a.set_xlabel("Time (s)")
                    ax_a.legend()
                    st.pyplot(fig_a)
                    # roll/pitch from accelerometer (simple)
                    denom = np.sqrt(ay_col**2 + az_col**2) + 1e-12
                    roll = np.arctan2(ay_col, az_col) * 180.0 / np.pi
                    pitch = np.arctan2(-ax_col, denom) * 180.0 / np.pi
                    fig_ap, ax_ap = plt.subplots(figsize=(10, 2.2))
                    ax_ap.plot(time[:len(roll)], roll, label='roll (deg)')
                    ax_ap.plot(time[:len(pitch)], pitch, label='pitch (deg)')
                    ax_ap.set_title("Estimated roll & pitch from accel")
                    ax_ap.legend()
                    st.pyplot(fig_ap)
                else:
                    st.info("MPU accelerometer columns not found in expected names (accel_x, accel_y, accel_z).")

            if has_gyro:
                # similar process for gyro
                try:
                    gx = df[[c for c in df.columns if 'gyro' in c.lower() and ('x' in c.lower())]][df.columns[0]].to_numpy()
                except Exception:
                    gx = None
                # fallback names
                if gx is None:
                    keys = [k for k in df.columns if 'gyro_x' in k or 'gyr_x' in k]
                    if keys:
                        gx = df[keys[0]].to_numpy(dtype=float)
                # find other two
                gy = None
                gz = None
                keys_y = [k for k in df.columns if 'gyro_y' in k or 'gyr_y' in k]
                keys_z = [k for k in df.columns if 'gyro_z' in k or 'gyr_z' in k]
                if keys_y:
                    gy = df[keys_y[0]].to_numpy(dtype=float)
                if keys_z:
                    gz = df[keys_z[0]].to_numpy(dtype=float)
                if gx is not None and gy is not None and gz is not None:
                    fig_g, ax_g = plt.subplots(figsize=(10, 2.2))
                    ax_g.plot(time[:len(gx)], gx, label='gx')
                    ax_g.plot(time[:len(gy)], gy, label='gy')
                    ax_g.plot(time[:len(gz)], gz, label='gz')
                    ax_g.set_title("MPU6050 - Gyroscope")
                    ax_g.legend()
                    st.pyplot(fig_g)
                else:
                    st.info("MPU gyroscope columns not found in expected names (gyro_x, gyro_y, gyro_z).")

with right_col:
    st.markdown("## Summary & Export")
    summary_df = pd.DataFrame(summary_list)
    if not summary_df.empty:
        st.dataframe(summary_df.style.format({
            "duration_s": "{:.2f}",
            "mean_rms": "{:.3f}",
            "peak_rms": "{:.3f}",
            "rms_std": "{:.3f}",
            "fatigue_slope_Hz_per_s": "{:.4f}",
            "estimated_force_N_at_peak": "{:.1f}"
        }))
        csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary CSV", csv, "emg_summary.csv", "text/csv")

    # Combined RMS export
    if len(combined_dfs) > 0:
        merged = combined_dfs[0].copy()
        for dd in combined_dfs[1:]:
            merged = pd.merge_asof(merged.sort_values("time_rms"),
                                    dd.sort_values("time_rms"),
                                    on="time_rms", direction="nearest", tolerance=1.0/fs)
        st.download_button("📊 Download Combined RMS CSV", merged.to_csv(index=False).encode('utf-8'), "combined_rms.csv", "text/csv")

    st.markdown("---")
    st.markdown("### Force Estimation Calibration")
    st.markdown("Force = (RMS / RMS_max) * Max Force (N). Set Max Force (N) using the sidebar calibration input.")
    st.info("Tip: Increase contraction threshold to better pick discrete contractions. Use RMS window ~100-250 ms for human surface EMG.")

st.markdown("---")
st.caption("EMG Muscle & Physiology Dashboard — computes RMS, contractions, median frequency (fatigue), FFT, and basic MPU visualizations.")

# For hosting platforms that provide PORT variable (Render)
if "PORT" in os.environ:
    st.write(f"Running on port {os.environ.get('PORT')} (hosting platform assigned)")
